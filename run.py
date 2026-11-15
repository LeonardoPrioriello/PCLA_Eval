"""Run PCLA agents through Bench2Drive and retry CARLA crashes."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
CARLA_CRASH = 'PCLA_BENCH_CARLA_SEGFAULT'

# The 40 agent names used in the study, as accepted by PCLA.
AGENTS = frozenset({
    "carl_carlv11", "carl_carl_0", "carl_plant_0", "carl_roach_0",
    "if_if", "lav_fast", "lav_lav", "lbc_lb", "lbc_nc",
    "lmdrive_llama", "lmdrive_llava", "lmdrive_vicuna",
    "minddrive_05b", "minddrive_3b", "neat_aim2ddepth", "neat_aim2dsem",
    "neat_aimbev", "neat_neat", "orion_base", "plant2_plant2_0",
    "simlingo_simlingo", "tfv3_gf", "tfv3_lf", "tfv3_ltf", "tfv3_tf",
    "tfv4_aim_0", "tfv4_l6_0", "tfv4_lav_0", "tfv4_wp_0",
    "tfv5_alltowns", "tfv5_notown13", "tfv6_4cameras", "tfv6_noradar",
    "tfv6_notown13", "tfv6_regnet", "tfv6_resnet", "tfv6_visiononly",
    "tt_tt", "wor_lb", "wor_nc",
})


def benchmark_track(selector):
    """Tell B2D whether the agent uses simulator information or sensor inputs."""

    return 'MAP' if selector.startswith(('carl_', 'plant2_')) else 'SENSORS'


def read_json(path):
    """Read a checkpoint or the crash counter; a new run may not have one yet."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    """Write the state safely: finish the new file before replacing the old one."""

    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def native_statistics(args, data, completed):
    """Load the existing checkpoint into B2D's own statistics manager.

    We use this manager for normal evaluation and also for adding the manual-skip record, if needed.
    """

    sys.path[:0] = [str(args.b2d / 'leaderboard'), str(args.b2d / 'scenario_runner')]
    from leaderboard.utils.statistics_manager import StatisticsManager
    manager = StatisticsManager(str(args.output / 'results.json'), str(args.output / 'live_results.txt'))
    if data:
        manager.add_file_records(str(args.output / 'results.json'))
        manager.save_sensors(data.get('sensors', []))
    manager.save_progress(completed, args.total)
    return manager


def save_native_progress(manager, completed, total, sensors_initialized=False):
    """Ask B2D to save progress, and calculate its overall scores at the end."""

    manager.save_progress(completed, total)
    if completed == total:
        manager.compute_global_statistics()
        manager.validate_and_write_statistics(sensors_initialized, False)
    else:
        manager.save_entry_status('Started')
        manager.write_statistics()


def watch_carla(server, stop):
    """Report a segmentation fault from the CARLA process started by B2D.

    B2D may still be waiting for a simulator reply after CARLA dies: this thread
    lets the parent process detect that death while the evaluation call is still waiting.
    """

    while not stop.is_set():
        code = server.poll()
        if code in (-signal.SIGSEGV, 128 + signal.SIGSEGV):
            print(CARLA_CRASH, flush=True)
            return
        if code is not None:
            return
        stop.wait(0.1)


def evaluate(args, index):
    """Run one route with B2D, appending directly to its existing checkpoint."""

    checkpoint = args.output / 'results.json'
    manager = native_statistics(args, read_json(checkpoint), index)
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from leaderboard.utils.route_indexer import RouteIndexer

    # Supply the same options that B2D's command-line entry point would create.
    options = SimpleNamespace(
        agent=str(ROOT / 'bridge.py'), agent_config=args.agent,
        routes=str(args.routes), routes_subset=','.join(args.route or []), repetitions=1,
        checkpoint=str(checkpoint), debug_checkpoint=str(args.output / 'live_results.txt'),
        debug=0, record='', host='127.0.0.1', port=args.port,
        traffic_manager_port=args.tm_port, traffic_manager_seed=0,
        gpu_rank=args.gpu, timeout=args.timeout, track=benchmark_track(args.agent),
    )
    routes = RouteIndexer(options.routes, 1, options.routes_subset)
    routes.index = index
    config = routes.get_next_config()

    # Select the first route without a saved record explicitly.
    manager.write_statistics()
    evaluator = LeaderboardEvaluator(options, manager)
    stop = threading.Event()
    monitor = threading.Thread(target=watch_carla, args=(evaluator.server, stop), daemon=True)
    monitor.start()
    try:
        evaluator._load_and_run_scenario(options, config)
    finally:
        stop.set()
        monitor.join()

        # Catch a server exit that happened just before the monitor was stopped.
        if evaluator.server.poll() in (-signal.SIGSEGV, 128 + signal.SIGSEGV):
            print(CARLA_CRASH, flush=True)
        records = manager._results.checkpoint.records

        # B2D sets the record's index when it registers a route outcome; while the route is still being prepared or driven, the index remains -1.
        has_outcome = len(records) > index and records[index].index == index
        completed = index + 1 if has_outcome else index
        save_native_progress(manager, completed, args.total, evaluator.sensors_initialized)
        del evaluator


def add_manual_skip(args, element, index, data):
    """Add the missing route after its third CARLA crash without an outcome."""

    manager = native_statistics(args, data, index)
    scenario = element.find('scenarios/scenario')
    scenario_name = scenario.get('name') if scenario is not None else None
    route_id = 'RouteScenario_{}_rep0'.format(element.get('id'))
    manager.create_route_data(route_id, scenario_name, None, 'manual_skip', element.get('town'), index)
    record = manager._results.checkpoint.records[index]
    record.index = index
    record.status = 'Failed - Manual skip (3 CARLA crashes without outcome)'

    record.meta.update(manual_skip=True, measurements='missing', carla_crashes=3)
    save_native_progress(manager, index + 1, args.total)


def run_attempt(args, index):
    """Start the child process, watch its log, and close only this attempt's processes."""

    import psutil

    command = [sys.executable, str(ROOT / 'run.py'), *sys.argv[1:], '--worker', str(index)]
    env = os.environ.copy()
    env.update({
        'PCLA_ROOT': str(args.pcla), 'CARLA_ROOT': str(args.carla),
        'LEADERBOARD_ROOT': str(args.b2d / 'leaderboard'),
        'SCENARIO_RUNNER_ROOT': str(args.b2d / 'scenario_runner'),
        'ROUTES': str(args.routes), 'SAVE_PATH': str(args.output / 'agent_output'),
        'PCLA_BENCH_SAVE_IMAGES': '1' if args.save_images else '0',
        'IS_BENCH2DRIVE': '1', 'DATAGEN': '0', 'NO_CARS': '0',
        'CHALLENGE_TRACK_CODENAME': benchmark_track(args.agent),
        'WANDB_MODE': 'disabled', 'PYTHONUNBUFFERED': '1', 'PYTHONNOUSERSITE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PYTHONPATH': os.pathsep.join(map(str, [ROOT, args.b2d / 'leaderboard',
            args.b2d / 'scenario_runner', args.carla / 'PythonAPI/carla'])),
    })

    # Pass absolute paths because B2D runs from its own directory.
    command += ['--pcla', str(args.pcla), '--b2d', str(args.b2d),
                '--carla', str(args.carla), '--output', str(args.output)]
    log = args.output / 'evaluator.log'
    crashed = False
    crash_time = None
    owned = set()
    with log.open('a', encoding='utf-8') as output, log.open(encoding='utf-8', errors='replace') as tail:
        tail.seek(0, os.SEEK_END)
        process = psutil.Popen(command, env=env, cwd=args.b2d, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True)
        recent = ''
        try:
            while True:
                code = process.poll()
                recent += tail.read()
                # Unreal also prints this signal message itself.
                crashed = crashed or CARLA_CRASH in recent or 'Signal 11 caught.' in recent
                recent = recent[-200:]
                if code is not None:
                    break
                try:
                    owned.update(process.children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                if crashed and crash_time is None:
                    crash_time = time.monotonic()
                if crash_time is not None and time.monotonic() - crash_time > 5:
                    # Allow B2D to save an outcome if it can.

                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
                    crash_time = None
                time.sleep(0.1)
        finally:
            try:
                owned.update(process.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            owned.add(process)
            for child in owned:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(list(owned), timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, remaining = psutil.wait_procs(alive, timeout=10)
            if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in remaining):
                raise RuntimeError('A process from this attempt is still stopping')
    return crashed


def completed_routes(data, routes):
    """Count saved outcomes; B2D leaves the route in progress out of its JSON."""

    records = data.get('_checkpoint', {}).get('records', [])
    expected = ['RouteScenario_{}_rep0'.format(route.get('id')) for route in routes]
    if [r['route_id'] for r in records] != expected[:len(records)]:
        raise ValueError('The checkpoint belongs to a different route selection')
    return len(records)


def supervise(args, routes):
    """Keep saved outcomes; retry a CARLA segfault with no outcome, including during setup."""

    checkpoint = args.output / 'results.json'
    state_path = args.output / 'run_state.json'
    selection = {'agent': args.agent, 'routes': [route.get('id') for route in routes]}
    state = read_json(state_path)
    if state and any(state.get(key) != value for key, value in selection.items()):
        raise ValueError('Use another output directory for a different agent or route selection')
    if not state and checkpoint.exists():
        raise ValueError('This checkpoint has no launcher state; use a new output directory')
    state = state or {**selection, 'crashes': {}}
    write_json(state_path, state)

    while True:
        data = read_json(checkpoint)
        index = completed_routes(data, routes)
        if index == len(routes):
            print('Results: ' + str(checkpoint))
            return
        route = routes[index]
        route_id = route.get('id')
        failures = state['crashes'].get(route_id, 0)
        if failures >= 3:
            add_manual_skip(args, route, index, data)
            print('Route ' + route_id + ': manual skip after three CARLA crashes')
            continue

        print('Route {}: attempt {}. Log: {}'.format(route_id, failures + 1, args.output / 'evaluator.log'), flush=True)
        crashed = run_attempt(args, index)
        try:
            after = read_json(checkpoint)
        except json.JSONDecodeError:
            if not crashed:
                raise

            # If the crash interrupted a checkpoint write, restore the complete checkpoint read before this attempt, preserving the earlier rssults.
            after = data
            write_json(checkpoint, data)
        saved = completed_routes(after, routes)
        if saved < index:
            if not crashed:
                raise RuntimeError('Earlier route outcomes disappeared from the checkpoint; see evaluator.log')

            # A confirmed crash must never erase the results from earlier routes.
            write_json(checkpoint, data)
            saved = index
        if saved > index:
            continue
        if not crashed:
            raise RuntimeError('No route outcome was saved. This was not a CARLA segfault; see evaluator.log')

        # The same route gets three attempts in total.
        state['crashes'][route_id] = failures + 1
        write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description='Run PCLA on Bench2Drive.')
    parser.add_argument('agent', nargs='?', choices=sorted(AGENTS))
    parser.add_argument('--list-agents', action='store_true')
    for name, folder in [('pcla', 'PCLA'), ('b2d', 'Bench2Drive'), ('carla', 'CARLA')]:
        parser.add_argument('--' + name, type=Path, default=os.environ.get(name.upper() + '_ROOT', ROOT / folder))
    parser.add_argument('--route', nargs='+', help='Route IDs; omit for all 220 routes')
    parser.add_argument('--output', type=Path, help='Output directory; repeat the command to resume')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list_agents:
        print('\n'.join(sorted(AGENTS)))
        return

    if not args.agent:
        parser.error('Specify an agent')

    for name in ('pcla', 'b2d', 'carla'):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    # B2D inserts this path into its CARLA launch command without shell quoting.
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(args.carla)):
        parser.error('Use a CARLA path without spaces or shell punctuation')

    args.routes = args.b2d / 'leaderboard/data/bench2drive220.xml'
    routes = ET.parse(args.routes).getroot().findall('route')

    if args.route:
        available = {route.get('id') for route in routes}

        if not set(args.route) <= available:
            parser.error('Unknown route IDs')

        routes = [route for route in routes if route.get('id') in args.route]

    args.total = len(routes)
    args.output = (args.output or ROOT / 'results' / args.agent).expanduser().resolve()
    (args.output / 'agent_output').mkdir(parents=True, exist_ok=True)

    if args.worker is not None:
        evaluate(args, args.worker)
        return

    # Only one launcher may update this checkpoint and its crash counter.
    import fcntl
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervise(args, routes)


if __name__ == '__main__':
    
    # A user interrupt stops the parent and its children without counting a crash.
    if '--worker' not in sys.argv:
        def interrupt(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. Repeat the same command to resume.', file=sys.stderr)
        sys.exit(130)
