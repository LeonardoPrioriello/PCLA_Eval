"""Connect a PCLA agent to Bench2Drive.

B2D creates the simulated world and vehicle, collects sensor readings and scores
the drive. This bridge loads the selected PCLA agent, forwards those readings and
returns the agent's chosen commands. setup runs once; run_step runs repeatedly
while driving; destroy releases the agent at the end of the route.
"""

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import re
import sys

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as B2DProvider
from srunner.scenariomanager.timer import GameTime as B2DGameTime

def evaluation_identity(config):
    """Read the agent name and route details from the string supplied by B2D."""

    selector, route_name = config.split("+")
    match = re.match(r"RouteScenario_(\d+)_rep(\d+)_(Town[0-9A-Za-z]+)(?:_|$)", route_name)
    return SimpleNamespace(selector=selector, route_name=route_name,
                           route_index=int(match[1]), repetition=int(match[2]), town=match[3])


def convert_enum(value, target):
    """Translate named options such as LEFT or SENSORS to the class used by PCLA or B2D."""

    # LEFT has the same meaning in both projects, but each defines its own Python class for these options.
    if hasattr(value, 'name'):
        return target[value.name]
    return target(value)


def convert_plan(plan, road_option):
    """Keep every route point and translate its navigation instruction, such as LEFT."""

    # Each route point is a tuple of a waypoint and a navigation option.
    return [(waypoint, convert_enum(option, road_option)) for waypoint, option in plan]


def normalize_sensors(sensors):
    """Give B2D the radar field names it expects."""

    # Copy each sensor description so we do not change the agent's own list.
    # fov is the angle covered by a radar, B2D expects horizontal and vertical
    # angles separately; setdefault preserves values already given by the agent.
    result = [dict(spec) for spec in sensors]
    for spec in result:
        if spec['type'] == 'sensor.other.radar':
            spec.setdefault('horizontal_fov', spec.get('fov', 90))
            spec.setdefault('vertical_fov', spec.get('fov', 0.1))
    return result


_suppress_images = ContextVar('suppress_images', default=False)


def _image_writer(original, destination_index, destination_keyword, success):
    """Wrap an image-saving function so the optional disk files can be avoided."""

    def write(*args, **kwargs):
        if len(args) > destination_index:
            destination = args[destination_index]
        else:
            destination = kwargs.get(destination_keyword)

        is_file_path = isinstance(destination, (str, bytes, os.PathLike))
        if _suppress_images.get() and is_file_path:
            return success
        return original(*args, **kwargs)

    return write


@contextmanager
def image_output():
    """Temporarily suppress optional disk images while the PCLA agent runs."""

    if os.environ.get('PCLA_BENCH_SAVE_IMAGES') == '1':
        yield
        return
    import cv2
    from PIL import Image
    token = _suppress_images.set(True)
    try:
        with ExitStack() as stack:
            def replace_writer(owner, name, index, keyword, success):
                original = getattr(owner, name)
                stack.callback(setattr, owner, name, original)
                replacement = _image_writer(original, index, keyword, success)
                setattr(owner, name, replacement)

            replace_writer(Image.Image, 'save', 1, 'fp', None)
            replace_writer(cv2, 'imwrite', 0, 'filename', True)
            plotting = sys.modules.get('matplotlib.figure')
            if plotting is not None:
                replace_writer(plotting.Figure, 'savefig', 1, 'fname', None)
            yield
    finally:
        _suppress_images.reset(token)


def get_entry_point():
    """Return the class name B2D must instantiate when it loads this file."""

    return "PCLABench2DriveAgent"


@contextmanager
def _working_directory(path):
    """Use the PCLA directory temporarily so the agent can find its local files."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PCLABench2DriveAgent(AutonomousAgent):

    def __init__(self, carla_host, carla_port, debug=False):
        """Create the B2D-facing agent; the actual PCLA model is loaded later in setup."""

        super().__init__(carla_host, carla_port, debug)

        self._inner = None
        self._raw_plan = None
        self._identity = None
        self._pcla_root = None
        self._pcla_provider = None
        self._pcla_time = None
        self._road_option = None
        self._needs_vehicle = False

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Receive the full route before setup, in GPS and CARLA world coordinates."""

        self._raw_plan = (list(global_plan_gps), list(global_plan_world_coord))
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        if self._inner is not None:
            self._forward_plan()

    def setup(self, path_to_conf_file):
        """Load the requested PCLA model and give it the current driving route."""

        self._identity = evaluation_identity(path_to_conf_file)
        self._pcla_root = Path(os.environ['PCLA_ROOT']).resolve()
        routes = str(Path(os.environ['ROUTES']).resolve())

        self._prepare_environment()

        with image_output():
            self._load_inner(routes)

        if self._identity.selector in {"lav_fast", "lbc_lb", "lbc_nc"} and os.environ.get("WANDB_MODE") == "disabled":
            # These agents build videos even when Weights & Biases logging is disabled.
            self._inner.log_wandb = False

        # B2D supplies the weather for each route., so an agent-specific override would evaluate a different driving situation from the one requested.
        if self._identity.selector.startswith("tfv6_"):
            config = self._inner.config_closed_loop

            if config.random_weather or config.custom_weather is not None:
                raise ValueError('Disable TFv6 weather overrides: Bench2Drive sets the weather')

        self._forward_plan()
        self.track = convert_enum(self._inner.track, Track)

        # Some privileged agents also need the CARLA vehicle object.
        self._needs_vehicle = "vehicle" in inspect.signature(self._inner.run_step).parameters

        # Some agents ask for vehicle measurements when writing their logs.
        if not hasattr(self._inner, "get_metric_info"):
            self._inner.get_metric_info = self.get_metric_info

        print("Loaded " + self._identity.selector, flush=True)

    def _prepare_environment(self):
        """Expose the route ID and town through settings read by existing agent code."""

        identity = self._identity
        os.environ["IS_BENCH2DRIVE"] = "1"
        os.environ["BENCHMARK_ROUTE_ID"] = identity.route_name
        os.environ["TOWN"] = identity.town
        os.environ["REPETITION"] = str(identity.repetition)
        os.environ["ROUTE_INDEX"] = str(identity.route_index)

    def _load_inner(self, routes):
        """Find the agent file and weight configuration using PCLA, then load that agent."""

        root = self._pcla_root
        sys.path.insert(0, str(root))

        sys.path.insert(0, str(root / "pcla_agents/lmdrive/vision_encoder"))

        from pcla_functions.give_path import give_path
        from pcla_functions.clear_agent_modules import clear_agent_modules
        from leaderboard_codes.carla_data_provider import CarlaDataProvider
        from leaderboard_codes.local_planner import RoadOption
        from leaderboard_codes.timer import GameTime

        self._pcla_provider, self._pcla_time, self._road_option = CarlaDataProvider, GameTime, RoadOption
        self._sync_runtime()

        agent_path, config_path = give_path(self._identity.selector, str(root), routes)

        # These two loaders expect the route name after their config/weight paths.
        if self._identity.selector.startswith(("minddrive_", "orion_")):
            config_path += "+" + self._identity.route_name

        module_key = "b2d_pcla_" + self._identity.selector
        sys.path.insert(0, str(Path(agent_path).parent))
        clear_agent_modules(module_key)

        # Each agent lives in a different Python file. Load the file named by PCLA, then ask its get_entry_point() function for the agent class.
        spec = importlib.util.spec_from_file_location(module_key, agent_path)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module

        with _working_directory(root):
            spec.loader.exec_module(module)
            agent_class = getattr(module, module.get_entry_point())
            kwargs = {}

            if self._identity.selector.startswith("plant2_"):

                # Despite the parameter name, PlanT2 reads a route-name string.
                kwargs["route_index"] = self._identity.route_name

                kwargs["vehicle"] = None

            self._inner = agent_class(config_path, **kwargs)

    def _forward_plan(self):
        """Pass the stored route to the loaded PCLA agent using its navigation option class."""

        gps, world = self._raw_plan
        self._inner.set_global_plan(
            convert_plan(gps, self._road_option), convert_plan(world, self._road_option))

    def _sync_runtime(self):
        """Give PCLA the world state and simulation time currently used by B2D."""

        world, client = B2DProvider.get_world(), B2DProvider.get_client()

        provider = self._pcla_provider
        provider.set_client(client)

        if provider.get_world() is not world:
            provider.set_world(world)

        provider.set_traffic_manager_port(B2DProvider.get_traffic_manager_port())

        # Each dictionary stores motion or position for simulated objects.
        for name in ("_actor_velocity_map", "_actor_location_map", "_actor_transform_map"):
            setattr(provider, name, dict(getattr(B2DProvider, name)))

        # ego is the vehicle driven by the agent.
        ego = self.hero_actor
        velocity = ego.get_velocity()
        values = {
            "_actor_velocity_map": (velocity.x ** 2 + velocity.y ** 2) ** 0.5,
            "_actor_location_map": ego.get_location(),
            "_actor_transform_map": ego.get_transform(),
        }

        for name, value in values.items():
            cache = getattr(provider, name)
            key = ego

            for actor in cache:
                if actor.id == ego.id:
                    key = actor
                    break

            cache[key] = value

        # Synchronize the clocks: agents may use elapsed time to recognize prolonged stops or combine observations from successive driving steps.
        for name in ("_current_game_time", "_carla_time", "_last_frame", "_platform_timestamp", "_init"):
            setattr(self._pcla_time, name, getattr(B2DGameTime, name))

    def sensors(self):
        """Tell B2D which cameras, LiDARs and other sensors to attach to the vehicle."""

        return normalize_sensors(self._inner.sensors())

    def run_step(self, input_data, timestamp):
        """Give observations to PCLA and return its steering, throttle and brake command."""

        # Refresh the scene first, then forward exactly the readings B2D supplied.
        self._sync_runtime()
        kwargs = {}

        if self._needs_vehicle:
            kwargs['vehicle'] = self.hero_actor

        with _working_directory(self._pcla_root), image_output():
            return self._inner.run_step(input_data, timestamp, **kwargs)

    def destroy(self):
        """Close the PCLA agent while leaving removal of the simulated vehicle to B2D."""

        inner = self._inner
        self._inner = None

        if inner is not None:

            if self._identity is not None and self._identity.selector.startswith("plant2_"):
                planner = getattr(inner, "_local_planner", None)

                # PlanT2's planner normally destroys its stored vehicle when deleted, so detach B2D's vehicle first, since B2D will remove it.
                if self.hero_actor is not None and getattr(planner, "_vehicle", None) is self.hero_actor:
                    planner.reset_vehicle()

            with _working_directory(self._pcla_root), image_output():
                inner.destroy()
