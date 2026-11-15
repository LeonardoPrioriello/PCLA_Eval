"""Family-weighted regressions and leave-one-family-out prediction checks."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

INPUTS = Path(__file__).resolve().parent / "inputs"
META = {
    item["code"]: item
    for item in json.loads(
        (INPUTS / "analysis_specs_47.json").read_text(encoding="utf-8")
    )
}
COMB = {
    item["id"]: item["features"]
    for item in json.loads(
        (INPUTS / "combined_models_18.json").read_text(encoding="utf-8")
    )
}

CTX = "source_carla_matches_b2d"
PRIV = "uses_privileged_runtime_information"

SENSOR = {
    "rgb_camera_count",
    "uses_lidar",
    "uses_radar",
    "uses_imu",
    "multi_camera_rgb",
    "front_focus_present",
    "surround_or_rear_camera_rgb",
    "visual_backbone_transformer_rgb",
}


def weights(data):
    """Divide each family's total weight of one among its included agents."""

    family_sizes = data.agent_family.value_counts()
    return np.array([1.0 / family_sizes[family] for family in data.agent_family])


def category_defs(data, codes):
    """Use the most common category by family weight as the reference."""

    sample_weights = weights(data)
    categories = {}
    for code in codes:
        if META[code]["kind"] != "categorical":
            continue
        values = data[code].astype(str).to_numpy()
        levels = sorted(set(values))
        reference = min(
            levels, key=lambda level: (-sample_weights[values == level].sum(), level)
        )
        categories[code] = {"levels": levels, "reference": reference}
    return categories


def matrix(training, data, codes, categories):
    """Encode predictors, using training agents alone for numerical scaling."""

    columns = [np.ones(len(data))]
    names = ["Intercept"]
    scales, means = {}, {}
    training_weights = weights(training)

    for code in codes:
        kind = META[code]["kind"]
        if kind == "categorical":

            # The reference category is represented by the intercept.
            for level in categories[code]["levels"]:
                if level != categories[code]["reference"]:
                    columns.append(
                        (data[code].astype(str).to_numpy() == level).astype(float)
                    )
                    names.append(f"{code}[{level}]")
        elif kind == "numeric":
            values = training[code].to_numpy(float)
            mean = np.average(values, weights=training_weights)
            deviation = np.sqrt(
                np.average((values - mean) ** 2, weights=training_weights)
            )
            # A constant training feature stays constant; no division by zero.
            scale = deviation if deviation > 1e-12 else 1.0
            columns.append((data[code].to_numpy(float) - mean) / scale)
            names.append(code)
            scales[code], means[code] = scale, mean
        else:
            columns.append(data[code].to_numpy(float))
            names.append(code)

    return np.column_stack(columns), names, scales, means


def estimate(design, observed, sample_weights, families):
    """Fit weighted least squares, with uncertainty clustered by family."""

    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_scores = observed * np.sqrt(sample_weights)
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_scores, rcond=None
    )
    predicted = design @ coefficients
    residuals = observed - predicted

    # Sum coefficient influences within each family before calculating variance.
    influences = (
        np.linalg.pinv(weighted_design) * (residuals * np.sqrt(sample_weights))[None, :]
    )
    family_names = sorted(set(families))
    family_influences = np.stack(
        [influences[:, families == family].sum(axis=1) for family in family_names]
    )
    family_count, agent_count = len(family_names), len(observed)
    correction = (family_count / (family_count - 1)) * (
        (agent_count - 1) / (agent_count - rank)
    )
    covariance = family_influences.T @ family_influences * correction
    standard_errors = np.sqrt(np.maximum(0, np.diag(covariance)))
    critical = float(student_t.ppf(0.975, family_count - 1))
    rounded_critical = round(critical, 4)

    squared_error = np.dot(sample_weights, residuals**2)
    mean_score = np.average(observed, weights=sample_weights)
    total_variation = np.dot(sample_weights, (observed - mean_score) ** 2)
    r_squared = 1 - squared_error / total_variation

    return {
        "beta": coefficients,
        "se": standard_errors,
        "lo": coefficients - rounded_critical * standard_errors,
        "hi": coefficients + rounded_critical * standard_errors,
        "exact_lo": coefficients - critical * standard_errors,
        "exact_hi": coefficients + critical * standard_errors,
        "r_squared": r_squared,
        "adjusted_r_squared": 1
        - (1 - r_squared) * (agent_count - 1) / (agent_count - rank),
        "rank": rank,
        "n_parameters": rank - 1,
    }


def crossvalidate(data, codes, target="official_ds", cfg=None):
    """Predict each family's scores after fitting on every other family."""

    categories = category_defs(data, codes) if cfg is None else cfg
    families = data.agent_family.to_numpy()
    observed = data[target].to_numpy(float)
    predictions = np.full(len(data), np.nan)
    mean_predictions = predictions.copy()
    unsupported = []
    fold_coefficients = {}

    for family in sorted(set(families)):
        held_out_mask = families == family
        training = data.loc[~held_out_mask]
        held_out = data.loc[held_out_mask]
        training_weights = weights(training)
        design, names, _, _ = matrix(training, training, codes, categories)
        held_out_design, _, _, _ = matrix(training, held_out, codes, categories)

        # A feature or category seen only in the excluded family cannot be learned.
        projection_error = np.max(
            np.abs(held_out_design - held_out_design @ np.linalg.pinv(design) @ design)
        )
        if projection_error > 1e-8:
            unsupported.append(str(family))

        coefficients = np.linalg.lstsq(
            design * np.sqrt(training_weights)[:, None],
            observed[~held_out_mask] * np.sqrt(training_weights),
            rcond=None,
        )[0]
        predictions[held_out_mask] = held_out_design @ coefficients
        mean_predictions[held_out_mask] = np.average(
            observed[~held_out_mask], weights=training_weights
        )
        fold_coefficients[str(family)] = dict(zip(names, coefficients))

    sample_weights = weights(data)
    errors = np.abs(observed - predictions)
    baseline_errors = np.abs(observed - mean_predictions)
    mean_score = np.average(observed, weights=sample_weights)
    total_variation = np.dot(sample_weights, (observed - mean_score) ** 2)

    return {
        "lofo_predictions": predictions,
        "baseline_predictions": mean_predictions,
        "unsupported": unsupported,
        "lofo_mae": np.average(errors, weights=sample_weights),
        "baseline_lofo_mae": np.average(baseline_errors, weights=sample_weights),
        "lofo_r_squared": 1
        - np.dot(sample_weights, (observed - predictions) ** 2) / total_variation,
        "foldbetas": fold_coefficients,
    }


def fit(data, codes, target="official_ds"):
    """Fit one comparison on agents with complete values, already selected by the caller."""
    
    categories = category_defs(data, codes)
    design, names, scales, means = matrix(data, data, codes, categories)
    estimates = estimate(
        design,
        data[target].to_numpy(float),
        weights(data),
        data.agent_family.to_numpy(),
    )
    validation = crossvalidate(data, codes, target, cfg=categories)
    full_rank = estimates["rank"] == len(names)

    return {
        **estimates,
        **validation,
        "names": names,
        "scales": scales,
        "means": means,
        "level_config": categories,
        "valid_cv": full_rank and not validation["unsupported"],
    }
