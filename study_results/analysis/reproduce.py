"""Calculate the statistical results from the catalogue and benchmark records."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import regressions as reg
from features import derive_features
from outcomes import read_results, summarize

HERE = Path(__file__).resolve().parent


def save_table(rows, output, name):
    """Save calculation results for later comparisons."""

    pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False, float_format="%.12g")


def fit_comparisons(data, routes):
    """Fit Driving Score and behaviour models using the same regression procedure."""

    models, coefficients, predictions = [], [], []

    def compare(
        model_id, stage, frame, codes, outcome="official_ds", population="all220"
    ):
        if reg.SENSOR.intersection(codes):
            frame = frame[frame[reg.PRIV].eq(0)]
        sample = frame.dropna(subset=[outcome, reg.CTX, *codes]).copy()
        model = reg.fit(sample, codes, outcome)
        baseline = reg.fit(sample, [reg.CTX], outcome)
        valid = model["valid_cv"]
        sign_stability = np.nan
        if len(codes) == 1 and reg.META[codes[0]]["kind"] != "categorical":
            code = codes[0]
            sign_stability = np.mean(
                [
                    np.sign(fold[code]) == np.sign(model["beta"][1])
                    for fold in model["foldbetas"].values()
                ]
            )

        safety_mae = np.nan
        if stage == "combined":
            safety = reg.fit(sample, [reg.CTX, "runtime_safety_or_recovery"], outcome)
            if safety["valid_cv"]:
                safety_mae = safety["lofo_mae"]

        # Both reference predictions use exactly the agents in this comparison.
        models.append(
            {
                "model": model_id,
                "stage": stage,
                "population": population,
                "outcome": outcome,
                "features": "|".join(codes),
                "agents": len(sample),
                "families": sample.agent_family.nunique(),
                "parameters": model["n_parameters"],
                "r_squared": model["r_squared"],
                "adjusted_r_squared": model["adjusted_r_squared"],
                "added_r_squared_vs_carla": model["r_squared"] - baseline["r_squared"],
                "valid_lofo": valid,
                "lofo_mae": model["lofo_mae"] if valid else np.nan,
                "lofo_r_squared": model["lofo_r_squared"] if valid else np.nan,
                "mean_only_mae": model["baseline_lofo_mae"],
                "carla_only_mae": baseline["lofo_mae"]
                if baseline["valid_cv"]
                else np.nan,
                "carla_and_safety_mae": safety_mae,
                "sign_stability": sign_stability,
                "unsupported_families": "|".join(model["unsupported"]),
            }
        )

        # The Driving Score analysis used a t critical value rounded to 4 decimals; behaviour comparisons used its full precision.
        lower = model["lo"] if outcome == "official_ds" else model["exact_lo"]
        upper = model["hi"] if outcome == "official_ds" else model["exact_hi"]
        for index, term in enumerate(model["names"]):
            feature = term.split("[", 1)[0]
            coefficients.append(
                {
                    "model": model_id,
                    "term": term,
                    "estimate": model["beta"][index],
                    "standard_error": model["se"][index],
                    "ci95_low": lower[index],
                    "ci95_high": upper[index],
                    "reference_category": model["level_config"]
                    .get(feature, {})
                    .get("reference", ""),
                    "feature_mean": model["means"].get(feature, np.nan),
                    "feature_sd": model["scales"].get(feature, np.nan),
                }
            )

        for index, (_, agent) in enumerate(sample.iterrows()):
            predictions.append(
                {
                    "model": model_id,
                    "agent": agent.variant_id,
                    "family": agent.agent_family,
                    "observed": agent[outcome],
                    "predicted": model["lofo_predictions"][index],
                    "mean_only_prediction": model["baseline_predictions"][index],
                    "carla_only_prediction": baseline["lofo_predictions"][index],
                    "estimable": agent.agent_family not in model["unsupported"],
                }
            )

    for code, spec in reg.META.items():
        compare(f"single:{code}", "single", data, [code])
        if spec["role"] == "Candidate":
            compare(f"adjusted:{code}", "adjusted", data, [reg.CTX, code])
    for name, codes in reg.COMB.items():
        compare(name, "combined", data, codes)

    # Repeat the selected behaviour comparisons on the recorded sensitivity samples.
    failed_routes = routes.loc[routes.technical_record.eq(1), "route_id"]
    common = summarize(routes[~routes.route_id.isin(failed_routes)])
    features = data[["variant_id", "agent_family", *reg.META]]
    populations = {
        "all220": data,
        "common151": features.merge(common, on=["variant_id", "agent_family"]),
        "all220_no_privileged": data[data[reg.PRIV].eq(0)],
        "without_TFV4": data[data.agent_family.ne("TFV4")],
    }

    questions = [
        ("runtime_safety_or_recovery", "blocked"),
        ("uses_lidar", "vehicle_collision"),
        ("explicit_traffic_rule_mechanism", "stop"),
        ("explicit_traffic_rule_mechanism", "red_light"),
        ("aux_traffic_rule_supervision", "stop"),
        ("aux_traffic_rule_supervision", "red_light"),
    ]

    for population, frame in populations.items():
        for code, event in questions:
            for measure in ["route_pct", "per_progress_km"]:
                outcome = f"{event}_{measure}"
                compare(
                    f"{population}:{code}:{outcome}",
                    "behaviour",
                    frame,
                    [reg.CTX, code],
                    outcome,
                    population,
                )

    weather_samples = {
        "reported46_all": data,
        "nonprivileged": data[data[reg.PRIV].eq(0)],
        "without_LAV": data[data.agent_family.ne("LAV")],
        "without_TFV4": data[data.agent_family.ne("TFV4")],
        "alternative_TFV6_52": data.assign(
            training_weather_count=np.where(
                data.agent_family.eq("TFV6"), 52, data.training_weather_count
            )
        ),
    }

    for population, frame in weather_samples.items():
        for outcome in ["blocked_route_pct", "vehicle_collision_route_pct"]:
            compare(
                f"weather:{population}:{outcome}",
                "weather",
                frame,
                [reg.CTX, "training_weather_count"],
                outcome,
                population,
            )

    return models, coefficients, predictions


def prediction_groups(data):
    """Compare feature presence within each recorded CARLA-version group."""

    rows = []
    features = [
        "environment_prediction_present",
        "auxiliary_used_at_runtime",
        "aux_bev_supervision",
    ]

    for feature in features:
        for version in [0, 1]:
            group = data[data[reg.CTX].eq(version)].copy()
            weights = reg.weights(group)
            present = group[feature].eq(1).to_numpy()
            for outcome in [
                "rc_pct",
                "blocked_route_pct",
                "vehicle_collision_route_pct",
            ]:
                values = group[outcome].to_numpy()
                absent_mean = np.average(values[~present], weights=weights[~present])
                present_mean = np.average(values[present], weights=weights[present])
                rows.append(
                    {
                        "feature": feature,
                        "carla_0915": version,
                        "outcome": outcome,
                        "agents_without": int((~present).sum()),
                        "agents_with": int(present.sum()),
                        "mean_without": absent_mean,
                        "mean_with": present_mean,
                        "with_minus_without": present_mean - absent_mean,
                    }
                )
    return rows


def variant_comparisons(data):
    """Calculate differences between every pair of agents in the same family."""

    rows = []
    for family, group in data.groupby("agent_family"):
        for (_, first), (_, second) in combinations(group.iterrows(), 2):
            changed = [
                code
                for code in reg.META
                if pd.notna(first[code])
                and pd.notna(second[code])
                and first[code] != second[code]
            ]
            for outcome in ["ds", "blocked_route_pct", "vehicle_collision_route_pct"]:
                rows.append(
                    {
                        "family": family,
                        "agent_a": first.variant_id,
                        "agent_b": second.variant_id,
                        "outcome": outcome,
                        "value_a": first[outcome],
                        "value_b": second[outcome],
                        "b_minus_a": second[outcome] - first[outcome],
                        "different_features": "|".join(changed),
                    }
                )
    return rows


def town13_comparisons(routes):
    """Resample paired routes for variants trained with and without Town13."""

    rows = []
    pairs = [
        ("TFV5", "tfv5_alltowns", "tfv5_notown13", 0),
        ("TFV6", "tfv6_resnet", "tfv6_notown13", 100),
    ]
    
    for family, reference, comparison, offset in pairs:
        first = routes[routes.variant_id.eq(reference)].set_index("route_id")
        second = routes[routes.variant_id.eq(comparison)].set_index("route_id")
        for scope in [1, 2]:
            include = first.town.eq("Town13") if scope == 1 else first.town.ne("Town13")
            ids = sorted(first.index[include])
            differences = (
                second.loc[ids, "ds"].to_numpy() - first.loc[ids, "ds"].to_numpy()
            )
            seed = 20260916 + offset + scope
            rng = np.random.default_rng(seed)
            draws = differences[rng.integers(0, len(ids), size=(20000, len(ids)))].mean(
                axis=1
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "family": family,
                    "reference": reference,
                    "comparison": comparison,
                    "scope": "Town13" if scope == 1 else "Other towns",
                    "routes": len(ids),
                    "reference_ds": first.loc[ids, "ds"].mean(),
                    "comparison_ds": second.loc[ids, "ds"].mean(),
                    "difference": differences.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "resamples": 20000,
                    "seed": seed,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE.parent / "calculated_results"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    catalogue = pd.read_excel(
        HERE.parent / "Agents Feature Catalogue.xlsx",
        sheet_name="Variant_Features",
        keep_default_na=False,
    )
    features = derive_features(catalogue)[["variant_id", "agent_family", *reg.META]]
    routes = read_results(
        catalogue, HERE.parent / "benchmark_results", HERE / "inputs/routes.csv"
    )
    data = features.merge(
        summarize(routes), on=["variant_id", "agent_family"], validate="one_to_one"
    )
    save_table(data, args.output, "agent_data")
    print("Agent features and driving measures calculated.", flush=True)

    models, coefficients, predictions = fit_comparisons(data, routes)
    save_table(models, args.output, "models")
    save_table(coefficients, args.output, "coefficients")
    save_table(predictions, args.output, "lofo_predictions")
    save_table(prediction_groups(data), args.output, "prediction_groups")
    save_table(variant_comparisons(data), args.output, "variant_comparisons")
    save_table(town13_comparisons(routes), args.output, "town13_comparisons")
    print(f"Finished. Numerical results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
