"""Read the saved driving tests and calculate each agent's scores and infractions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Short names used in the analysis, followed by the names in Bench2Drive records.
EVENTS = {
    "blocked": "vehicle_blocked",
    "vehicle_collision": "collisions_vehicle",
    "pedestrian_collision": "collisions_pedestrian",
    "layout_collision": "collisions_layout",
    "red_light": "red_light",
    "stop": "stop_infraction",
    "failed_yield": "yield_emergency_vehicle_infractions",
}


def read_results(
    catalogue: pd.DataFrame, results_dir: Path, route_metadata: Path
) -> pd.DataFrame:
    """Join each agent's saved route records to the benchmark route descriptions."""

    routes = pd.read_csv(route_metadata, keep_default_na=False).to_dict("records")
    expected_order = [(route["route_id"], route["index"]) for route in routes]
    rows = []

    for agent in catalogue.itertuples(index=False):
        filename = results_dir / f"{agent.variant_id}.json"
        checkpoint = json.loads(filename.read_bytes())["_checkpoint"]
        records = checkpoint["records"]
        recorded_order = [(record["route_id"], record["index"]) for record in records]

        if recorded_order != expected_order:
            raise ValueError(
                f"{filename.name}: route IDs or order differ from routes.csv"
            )

        official_ds = checkpoint["global_record"]["scores_mean"]["score_composed"]

        for record, route in zip(records, routes):
            scores = record["scores"]
            infractions = record["infractions"]
            status = record["status"]
            completed = status in ("Completed", "Perfect")

            # Bench2Drive ignores speed-efficiency entries when checking success.
            strict_success = completed and not any(
                entries
                for name, entries in infractions.items()
                if name != "min_speed_infractions"
            )
            technical_failure = (
                "Manual skip" in status or status == "Failed - Agent crashed"
            )

            row = {
                "variant_id": agent.variant_id,
                "agent_family": agent.agent_family,
                "route_id": record["route_id"],
                "index": record["index"],
                "town": route["town"],
                "scenario": route["scenario"],
                "weather_id": route["weather_id"],
                "ds": scores["score_composed"],
                "rc": scores["score_route"],
                "ip": scores["score_penalty"],
                "official_ds": official_ds,
                "completed": int(completed),
                "strict_success": int(strict_success),
                "technical_record": int(technical_failure),
                # Length is in metres and Route Completion is a percentage.
                "progress_km": (
                    record["meta"]["route_length"] * scores["score_route"] / 100000
                ),
            }
            for name, recorded_name in EVENTS.items():
                row[f"{name}_count"] = len(infractions[recorded_name])
            rows.append(row)

    return pd.DataFrame(rows)


def summarize(routes: pd.DataFrame) -> pd.DataFrame:
    """Summarize the supplied routes, which may be the full suite or a subset."""
    
    summaries = []

    for agent_id, agent_routes in routes.groupby("variant_id"):
        route_count = len(agent_routes)
        progress_km = agent_routes["progress_km"].sum()
        completed = agent_routes["completed"].sum()
        successful = agent_routes["strict_success"].sum()

        summary = {
            "variant_id": agent_id,
            "agent_family": agent_routes["agent_family"].iloc[0],
            "n_routes": route_count,
            "ds": agent_routes["ds"].mean(),
            "official_ds": agent_routes["official_ds"].iloc[0],
            "rc_pct": agent_routes["rc"].mean(),
            "ip": agent_routes["ip"].mean(),
            "completed_pct": 100 * completed / route_count,
            "strict_success_pct": 100 * successful / route_count,
            "completed_routes": completed,
            "successful_routes": successful,
            "technical_records": agent_routes["technical_record"].sum(),
            "progress_km": progress_km,
        }

        for name in EVENTS:
            counts = agent_routes[f"{name}_count"]
            events = counts.sum()
            affected_routes = counts.gt(0).sum()
            summary[f"{name}_routes"] = affected_routes
            summary[f"{name}_events"] = events
            summary[f"{name}_route_pct"] = 100 * affected_routes / route_count
            summary[f"{name}_per_progress_km"] = (
                events / progress_km if progress_km > 0 else np.nan
            )

        summaries.append(summary)

    return pd.DataFrame(summaries)
