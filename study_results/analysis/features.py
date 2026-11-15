"""Prepare the analysis variables from the recorded agent characteristics."""

import numpy as np
import pandas as pd


def derive_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the catalogue fields and add the variables used in the analysis."""

    out = data.copy()

    binary_columns = [
        "uses_lidar",
        "uses_radar",
        "uses_imu",
        "uses_privileged_runtime_information",
        "bev_runtime_representation",
        "explicit_safety_layer",
        "stuck_recovery",
        "inference_ensemble",
        "uses_language_at_inference",
        "uses_non_ego_motion_supervision",
        "aux_semantic_supervision",
        "aux_depth_supervision",
        "aux_bev_supervision",
        "aux_object_supervision",
        "aux_traffic_rule_supervision",
    ]

    for column in binary_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["rgb_camera_count"] = pd.to_numeric(out["rgb_camera_count"], errors="coerce")
    out["training_town_count"] = pd.to_numeric(
        out["training_town_count"], errors="coerce"
    )

    out["training_weather_count"] = pd.to_numeric(
        out["training_weather_count"], errors="coerce"
    )

    out["source_carla_matches_b2d"] = (
        out["source_carla_version"].astype(str).eq("0.9.15").astype(int)
    )

    out["source_python_3_10"] = (
        out["source_python_version"].astype(str).str.startswith("3.10").astype(int)
    )

    out["leaderboard_api_v2"] = (
        out["leaderboard_api_version"].astype(str).eq("leaderboard_2").astype(int)
    )

    out["multi_camera_rgb"] = out["rgb_camera_count"].ge(3).astype(int)

    out["front_focus_present"] = (
        out["front_focus_input"]
        .astype(str)
        .isin({"dedicated_camera", "crop"})
        .astype(int)
    )

    out["surround_or_rear_camera_rgb"] = (
        out["camera_coverage"].isin({"surround", "front_wide_plus_rear"}).astype(float)
    )
    
    # Agents without cameras are excluded from this camera comparison.
    out.loc[out["camera_coverage"].eq("none"), "surround_or_rear_camera_rgb"] = np.nan

    target_point_values = {
        "command_plus_target",
        "target_point",
        "command_plus_multi_target",
        "route_segments_plus_target_point",
        "multi_target_points",
    }

    out["target_point_conditioning"] = (
        out["navigation_conditioning"].isin(target_point_values).astype(int)
    )

    transformer_fusion = {
        "transformer_sensor_fusion",
        "query_attention",
        "latent_bev_transformer",
        "object_token_transformer",
        "object_token_transformer_plus_sd_map",
        "vision_language_transformer",
        "vision_language_multimodal_fusion",
    }

    out["transformer_or_attention_representation"] = (
        out["fusion_type"].isin(transformer_fusion).astype(int)
    )

    out["visual_backbone_transformer_rgb"] = (
        out["visual_backbone_family"].eq("transformer_vit_family").astype(float)
    )
    out.loc[
        out["visual_backbone_family"].eq("not_applicable"),
        "visual_backbone_transformer_rgb",
    ] = np.nan

    out["environment_prediction_present"] = (
        out["environment_prediction_type"].astype(str).ne("none").astype(int)
    )

    out["temporal_reasoning_present"] = (
        out["temporal_reasoning"].astype(str).ne("none").astype(int)
    )

    waypoint_values = {
        "waypoints",
        "waypoints_plus_aux",
        "waypoint_offsets",
        "path_plus_target_speed",
        "path_plus_waypoints_plus_target_speed",
        "path_plus_speed_waypoints",
    }

    output_groups = {value: "waypoint_or_path" for value in waypoint_values}
    output_groups["direct_control"] = "direct_control"
    out["planning_output_class"] = (
        out["planning_output"].map(output_groups).fillna("trajectory_or_high_level")
    )

    out["predicts_target_speed"] = (
        out["planning_output"].astype(str).str.contains("target_speed", regex=False)
        | out["planning_output"]
        .astype(str)
        .eq("meta_actions_plus_path_speed_trajectories")
        | out["controller_type"].astype(str).eq("pid_target_speed")
    ).astype(int)

    out["decoder_refinement_present"] = (
        out["decoder_refinement_type"].astype(str).ne("none").astype(int)
    )

    augmented_pid_values = {
        "pid_target_speed",
        "pid_safety_controller",
        "pid_brake_override",
        "pid_red_light",
    }

    controller_groups = {
        "direct_postprocess": "direct_postprocess",
        "pid": "plain_PID",
    }
    out["controller_class"] = (
        out["controller_type"].map(controller_groups).fillna("augmented_PID")
    )

    out["controller_augmented_pid"] = (
        out["controller_type"].isin(augmented_pid_values).astype(int)
    )

    out["explicit_traffic_rule_mechanism"] = (
        out["traffic_rule_handling"].astype(str).ne("implicit").astype(int)
    )

    out["runtime_safety_or_recovery"] = (
        out["explicit_safety_layer"].eq(1) | out["stuck_recovery"].eq(1)
    ).astype(int)

    out["control_safeguard_count"] = (
        out["explicit_traffic_rule_mechanism"]
        + out["explicit_safety_layer"].fillna(0)
        + out["stuck_recovery"].fillna(0)
    )

    out["uses_rl_training"] = (
        out["training_paradigm"]
        .isin({"reinforcement_learning", "imitation_plus_online_rl"})
        .astype(int)
    )

    out["distillation_based_training"] = (
        out["training_paradigm"]
        .isin(
            {
                "privileged_distillation",
                "offline_q_distillation",
                "perception_planner_distillation",
            }
        )
        .astype(int)
    )

    paradigm_map = {
        "imitation_auxiliary": "imitation",
        "imitation": "imitation",
        "privileged_distillation": "distillation",
        "offline_q_distillation": "distillation",
        "perception_planner_distillation": "distillation",
        "reinforcement_learning": "RL",
        "imitation_plus_online_rl": "hybrid_RL",
    }

    out["training_paradigm_grouped"] = out["training_paradigm"].map(paradigm_map)

    out["multi_stage_training"] = (
        ~out["training_pipeline_structure"].isin(
            {"single_stage", "single_stage_multitask"}
        )
    ).astype(int)

    supervision_map = {
        "expert_policy_labels": "expert_policy",
        "simulator_ground_truth_labels": "simulator_labels_or_state",
        "privileged_simulator_state": "simulator_labels_or_state",
        "privileged_teacher_distillation": "distillation_or_value",
        "offline_value_targets": "distillation_or_value",
    }

    out["privileged_supervision_grouped"] = out["privileged_supervision_type"].map(
        supervision_map
    )

    aux_columns = [
        "aux_semantic_supervision",
        "aux_depth_supervision",
        "aux_bev_supervision",
        "aux_object_supervision",
        "aux_traffic_rule_supervision",
    ]

    out["aux_supervision_count"] = out[aux_columns].sum(
        axis=1, min_count=len(aux_columns)
    )

    out["auxiliary_used_at_runtime"] = (
        out["auxiliary_inference_role"]
        .isin(
            {
                "runtime_representation",
                "runtime_control_signal",
                "safety_controller_signal",
            }
        )
        .astype(int)
    )

    offline_numeric = pd.to_numeric(
        out["offline_training_sample_count"], errors="coerce"
    )

    out["log_offline_training_samples"] = np.where(
        offline_numeric > 0, np.log10(offline_numeric), np.nan
    )

    out["augmentation_present"] = (
        out["augmentation_strategy"].astype(str).ne("none").astype(int)
    )

    return out
