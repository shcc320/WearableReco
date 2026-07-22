from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

# Avoid unavailable physical-core probing by joblib on managed Windows hosts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) - 1)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_activity_dss.data import COMMON_ACTIVITIES, SENSOR_CONFIGS, download_datasets, make_synthetic_feature_datasets
from sports_activity_dss.evaluation import loso_model_benchmark, loso_sensor_ablation, summarize_loso, statistical_tests
from sports_activity_dss.extended import (
    TRANSFER_PROTOCOLS,
    best_five_class_subset,
    complete_subject_cohort,
    evaluate_transfer_grid,
    participant_activity_audit,
    prepare_extended_feature_datasets,
    select_feature_columns,
    transfer_protocol_summary,
    build_extended_mcda_decision,
)
from sports_activity_dss.models import build_models
from sports_activity_dss.mcda import rank_alternatives, rank_agreement, sensitivity_analysis
from sports_activity_dss.utils import ensure_directories, gpu_is_visible, load_yaml, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JASE extended validation experiments.")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "extended.yaml")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-five-class", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    return parser.parse_args()


def save_frames(directory: Path, frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(directory / name, index=False)


def select_range(
    datasets: dict[str, pd.DataFrame],
    model_summaries: dict[str, pd.DataFrame],
    preference: str,
) -> str:
    if preference in {"16g", "6g"}:
        return f"PAMAP2-{preference}"
    # Pre-specified auto rule: maximize complete-subject coverage, then windows;
    # never select based on predictive performance.
    candidates = []
    for name in ["PAMAP2-16g", "PAMAP2-6g"]:
        cohort = complete_subject_cohort(datasets[name], COMMON_ACTIVITIES)
        candidates.append((int(cohort["subject"].nunique()), len(cohort), name))
    return sorted(candidates, reverse=True)[0][2]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["models"]["random_seed"])
    set_global_seed(seed)

    out_root = ROOT / ("extended_smoke_outputs" if args.mode == "smoke" else "")
    if args.mode == "full":
        out_root = ROOT
    results = out_root / "extended_results"
    ensure_directories([results])

    device = args.device
    if device == "auto":
        device = "cuda" if gpu_is_visible() else "cpu"
    print(f"Resolved XGBoost device: {device}")

    if args.mode == "smoke":
        pamap, mhealth = make_synthetic_feature_datasets(seed)
        # Give the two synthetic PAMAP2 variants slightly different copies so
        # all range-selection and audit branches are exercised.
        datasets = {"PAMAP2-16g": pamap.copy(), "PAMAP2-6g": pamap.copy(), "MHEALTH": mhealth.copy()}
    else:
        download_datasets(ROOT / "data" / "raw")
        datasets = prepare_extended_feature_datasets(
            ROOT / "data" / "raw",
            ROOT / "data" / "processed",
            int(config["data"]["target_sampling_hz"]),
            int(config["data"]["window_samples"]),
            int(config["data"]["step_samples"]),
            force=args.force_features,
        )

    audit = participant_activity_audit(datasets)
    audit.to_csv(results / "participant_activity_audit.csv", index=False)

    six_class_frames = {
        name: complete_subject_cohort(frame, COMMON_ACTIVITIES)
        for name, frame in datasets.items()
    }
    for name, frame in six_class_frames.items():
        if frame["subject"].nunique() < 3:
            raise RuntimeError(f"{name} has fewer than three complete six-class subjects.")

    label_encoder = LabelEncoder().fit(COMMON_ACTIVITIES)
    model_specs = build_models(
        seed=seed,
        n_classes=len(COMMON_ACTIVITIES),
        n_jobs=int(config["models"]["n_jobs"]),
        device=device,
        random_forest_estimators=int(config["models"]["random_forest_estimators"]),
        boosting_estimators=int(config["models"]["boosting_estimators"]),
    )
    if args.mode == "smoke":
        model_specs = {name: model_specs[name] for name in [
            "Logistic regression", "Histogram gradient boosting"
        ]}

    range_summaries: dict[str, pd.DataFrame] = {}
    range_by_fold: dict[str, pd.DataFrame] = {}
    range_predictions: dict[str, pd.DataFrame] = {}
    range_rows = []
    for name in ["PAMAP2-16g", "PAMAP2-6g"]:
        frame = six_class_frames[name]
        columns = select_feature_columns(frame, "Full common configuration", "all")
        by_fold, predictions = loso_model_benchmark(frame, columns, model_specs, label_encoder)
        summary = summarize_loso(by_fold, "Model")
        range_summaries[name] = summary
        range_by_fold[name] = by_fold
        range_predictions[name] = predictions
        range_rows.append({
            "PAMAP2 range": name.replace("PAMAP2-", ""),
            "Complete subjects": int(frame["subject"].nunique()),
            "Windows": int(len(frame)),
            "Best model": str(summary.iloc[0]["Model"]),
            "Best model macro F1": float(summary.iloc[0]["Macro F1 mean"]),
        })
    range_summary = pd.DataFrame(range_rows)
    range_summary.to_csv(results / "pamap2_accelerometer_range_comparison.csv", index=False)

    selected_name = select_range(
        datasets,
        range_summaries,
        str(config["data"].get("preferred_pamap2_range", "auto")),
    )
    primary = six_class_frames[selected_name]
    external = six_class_frames["MHEALTH"]
    primary_summary = range_summaries[selected_name]
    primary_by_fold = range_by_fold[selected_name]
    primary_predictions = range_predictions[selected_name]
    primary_best_name = str(primary_summary.iloc[0]["Model"])
    primary_best_spec = model_specs[primary_best_name]
    print(f"Selected PAMAP2 variant by coverage rule: {selected_name}")

    # MHEALTH internal LOSO is the diagnostic check that target preprocessing is valid.
    external_columns = select_feature_columns(external, "Full common configuration", "all")
    mhealth_by_fold, mhealth_predictions = loso_model_benchmark(
        external, external_columns, model_specs, label_encoder
    )
    mhealth_summary = summarize_loso(mhealth_by_fold, "Model")
    mhealth_best_name = str(mhealth_summary.iloc[0]["Model"])
    mhealth_best_spec = model_specs[mhealth_best_name]

    active_sensor_configs = SENSOR_CONFIGS
    if args.mode == "smoke":
        active_sensor_configs = {name: SENSOR_CONFIGS[name] for name in [
            "Chest acceleration", "Full common configuration"
        ]}

    primary_sensor_by_fold, primary_sensor_predictions = loso_sensor_ablation(
        primary, active_sensor_configs, lambda f, c: select_feature_columns(f, c, "all"),
        primary_best_spec, label_encoder,
    )
    primary_sensor_summary = summarize_loso(primary_sensor_by_fold, "Configuration")
    mhealth_sensor_by_fold, mhealth_sensor_predictions = loso_sensor_ablation(
        external, active_sensor_configs, lambda f, c: select_feature_columns(f, c, "all"),
        mhealth_best_spec, label_encoder,
    )
    mhealth_sensor_summary = summarize_loso(mhealth_sensor_by_fold, "Configuration")

    protocol_names = list(config["transfer"]["protocols"] )
    if args.mode == "smoke":
        protocol_names = ["All features - raw", "Magnitude - CORAL"]
    configured_protocols = {name: TRANSFER_PROTOCOLS[name] for name in protocol_names}
    bootstrap_samples = args.bootstrap_samples or int(config["transfer"]["bootstrap_samples"])

    # Model-family transfer uses the full common configuration.
    forward_model_entities = {
        name: ("Full common configuration", spec) for name, spec in model_specs.items()
    }
    reverse_model_entities = forward_model_entities
    forward_models = evaluate_transfer_grid(
        primary, external, model_specs, label_encoder,
        f"{selected_name} -> MHEALTH", configured_protocols,
        forward_model_entities, seed, bootstrap_samples,
    )
    reverse_models = evaluate_transfer_grid(
        external, primary, model_specs, label_encoder,
        f"MHEALTH -> {selected_name}", configured_protocols,
        reverse_model_entities, seed + 1, bootstrap_samples,
    )

    # Sensor transfer uses the model selected independently inside each source dataset.
    forward_sensor_entities = {
        config_name: (config_name, primary_best_spec) for config_name in active_sensor_configs
    }
    reverse_sensor_entities = {
        config_name: (config_name, mhealth_best_spec) for config_name in active_sensor_configs
    }
    forward_sensors = evaluate_transfer_grid(
        primary, external, model_specs, label_encoder,
        f"{selected_name} -> MHEALTH", configured_protocols,
        forward_sensor_entities, seed + 2, bootstrap_samples,
    )
    reverse_sensors = evaluate_transfer_grid(
        external, primary, model_specs, label_encoder,
        f"MHEALTH -> {selected_name}", configured_protocols,
        reverse_sensor_entities, seed + 3, bootstrap_samples,
    )

    model_aggregate = pd.concat([forward_models.aggregate, reverse_models.aggregate], ignore_index=True)
    model_subject = pd.concat([forward_models.by_subject, reverse_models.by_subject], ignore_index=True)
    model_predictions = pd.concat([forward_models.predictions, reverse_models.predictions], ignore_index=True)
    model_cis = pd.concat([forward_models.confidence_intervals, reverse_models.confidence_intervals], ignore_index=True)
    sensor_aggregate = pd.concat([forward_sensors.aggregate, reverse_sensors.aggregate], ignore_index=True)
    sensor_subject = pd.concat([forward_sensors.by_subject, reverse_sensors.by_subject], ignore_index=True)
    sensor_predictions = pd.concat([forward_sensors.predictions, reverse_sensors.predictions], ignore_index=True)
    sensor_cis = pd.concat([forward_sensors.confidence_intervals, reverse_sensors.confidence_intervals], ignore_index=True)

    protocol_summary = transfer_protocol_summary(sensor_aggregate)

    # Save confusion matrices and class-wise diagnostics for the strongest
    # pre-specified sensor-transfer setting in each direction.
    best_setting_rows = []
    class_report_rows = []
    for direction in sensor_aggregate["Direction"].drop_duplicates():
        best = sensor_aggregate[sensor_aggregate["Direction"].eq(direction)].sort_values(
            "Macro F1", ascending=False
        ).iloc[0]
        selected = sensor_predictions[
            sensor_predictions["Direction"].eq(direction)
            & sensor_predictions["Protocol"].eq(best["Protocol"])
            & sensor_predictions["Entity"].eq(best["Entity"])
        ]
        y_true = label_encoder.transform(selected["True label"])
        y_pred = label_encoder.transform(selected["Predicted label"])
        matrix = confusion_matrix(
            y_true, y_pred, labels=np.arange(len(label_encoder.classes_)), normalize="true"
        )
        safe_direction = direction.lower().replace(" ", "_").replace(">", "to").replace("-", "_")
        pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
            results / f"best_transfer_confusion_{safe_direction}.csv"
        )
        report = classification_report(
            y_true, y_pred, labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )
        for activity in label_encoder.classes_:
            values = report[activity]
            class_report_rows.append({
                "Direction": direction, "Protocol": best["Protocol"],
                "Configuration": best["Entity"], "Activity": activity,
                "Precision": values["precision"], "Recall": values["recall"],
                "F1": values["f1-score"], "Support": int(values["support"]),
            })
        best_setting_rows.append({
            "Direction": direction, "Protocol": best["Protocol"],
            "Configuration": best["Entity"], "Model": best["Model"],
            "Macro F1": float(best["Macro F1"]),
        })
    best_transfer_settings = pd.DataFrame(best_setting_rows)
    best_transfer_class_report = pd.DataFrame(class_report_rows)

    mcda_weights = pd.Series(config["mcda"]["weights"], dtype=float)
    mcda_decisions = []
    mcda_rankings = []
    mcda_agreements = []
    mcda_sensitivity_samples = []
    mcda_sensitivity_summaries = []
    sensitivity_n = int(config["mcda"]["sensitivity_samples"] if args.mode == "full" else 100)
    for protocol_index, protocol in enumerate(configured_protocols):
        decision = build_extended_mcda_decision(
            primary_sensor_summary, mhealth_sensor_summary, sensor_aggregate, protocol
        )
        ranking = rank_alternatives(decision, mcda_weights)
        ranking["Protocol"] = protocol
        ranking = ranking.merge(
            decision[["Configuration", "Body locations"]], on="Configuration", how="left"
        )
        agreement = rank_agreement(ranking)
        agreement["Protocol"] = protocol
        sample_df, summary_df = sensitivity_analysis(
            decision, mcda_weights, sensitivity_n,
            float(config["mcda"]["concentrated_dirichlet_scale"]), seed + 20 + protocol_index,
        )
        sample_df["Protocol"] = protocol
        summary_df["Protocol"] = protocol
        mcda_decisions.append(decision)
        mcda_rankings.append(ranking)
        mcda_agreements.append(agreement)
        mcda_sensitivity_samples.append(sample_df)
        mcda_sensitivity_summaries.append(summary_df)
    mcda_decision = pd.concat(mcda_decisions, ignore_index=True)
    mcda_ranking = pd.concat(mcda_rankings, ignore_index=True)
    mcda_agreement = pd.concat(mcda_agreements, ignore_index=True)
    mcda_sensitivity_sample = pd.concat(mcda_sensitivity_samples, ignore_index=True)
    mcda_sensitivity_summary = pd.concat(mcda_sensitivity_summaries, ignore_index=True)

    # Five-class sensitivity is selected only by cohort coverage, not performance.
    five_activities, five_candidates = best_five_class_subset(datasets[selected_name])
    five_candidates.to_csv(results / "five_class_candidate_coverage.csv", index=False)
    five_outputs: dict[str, pd.DataFrame] = {}
    run_five = bool(config["sensitivity"].get("run_five_class", True)) and not args.skip_five_class
    if run_five:
        five_primary = complete_subject_cohort(datasets[selected_name], five_activities)
        five_external = complete_subject_cohort(datasets["MHEALTH"], five_activities)
        five_encoder = LabelEncoder().fit(five_activities)
        five_models = build_models(
            seed=seed,
            n_classes=len(five_activities),
            n_jobs=int(config["models"]["n_jobs"]),
            device=device,
            random_forest_estimators=int(config["models"]["random_forest_estimators"]),
            boosting_estimators=int(config["models"]["boosting_estimators"]),
        )
        five_columns = select_feature_columns(five_primary, "Full common configuration", "all")
        five_model_by_fold, five_model_predictions = loso_model_benchmark(
            five_primary, five_columns, five_models, five_encoder
        )
        five_model_summary = summarize_loso(five_model_by_fold, "Model")
        five_best_spec = five_models[str(five_model_summary.iloc[0]["Model"])]
        five_sensor_by_fold, five_sensor_predictions = loso_sensor_ablation(
            five_primary, SENSOR_CONFIGS, lambda f, c: select_feature_columns(f, c, "all"),
            five_best_spec, five_encoder,
        )
        five_sensor_summary = summarize_loso(five_sensor_by_fold, "Configuration")
        five_outputs = {
            "five_class_model_loso_by_fold.csv": five_model_by_fold,
            "five_class_model_loso_predictions.csv": five_model_predictions,
            "five_class_model_loso_summary.csv": five_model_summary,
            "five_class_sensor_loso_by_fold.csv": five_sensor_by_fold,
            "five_class_sensor_loso_predictions.csv": five_sensor_predictions,
            "five_class_sensor_loso_summary.csv": five_sensor_summary,
        }
        five_metadata = {
            "activities": five_activities,
            "primary_subjects": int(five_primary["subject"].nunique()),
            "external_subjects": int(five_external["subject"].nunique()),
            "primary_windows": int(len(five_primary)),
            "external_windows": int(len(five_external)),
        }
        write_json(results / "five_class_metadata.json", five_metadata)

    tests = []
    for family, frame, group in [
        (f"{selected_name} models", primary_by_fold, "Model"),
        ("MHEALTH models", mhealth_by_fold, "Model"),
        (f"{selected_name} sensors", primary_sensor_by_fold, "Configuration"),
        ("MHEALTH sensors", mhealth_sensor_by_fold, "Configuration"),
    ]:
        test = statistical_tests(frame, group)
        test["Family"] = family
        tests.append(test)

    outputs = {
        "pamap2_selected_model_loso_by_fold.csv": primary_by_fold,
        "pamap2_selected_model_loso_predictions.csv": primary_predictions,
        "pamap2_selected_model_loso_summary.csv": primary_summary,
        "mhealth_model_loso_by_fold.csv": mhealth_by_fold,
        "mhealth_model_loso_predictions.csv": mhealth_predictions,
        "mhealth_model_loso_summary.csv": mhealth_summary,
        "pamap2_selected_sensor_loso_by_fold.csv": primary_sensor_by_fold,
        "pamap2_selected_sensor_loso_predictions.csv": primary_sensor_predictions,
        "pamap2_selected_sensor_loso_summary.csv": primary_sensor_summary,
        "mhealth_sensor_loso_by_fold.csv": mhealth_sensor_by_fold,
        "mhealth_sensor_loso_predictions.csv": mhealth_sensor_predictions,
        "mhealth_sensor_loso_summary.csv": mhealth_sensor_summary,
        "bidirectional_model_transfer.csv": model_aggregate,
        "bidirectional_model_transfer_by_subject.csv": model_subject,
        "bidirectional_model_transfer_predictions.csv": model_predictions,
        "bidirectional_model_transfer_bootstrap_ci.csv": model_cis,
        "bidirectional_sensor_transfer.csv": sensor_aggregate,
        "bidirectional_sensor_transfer_by_subject.csv": sensor_subject,
        "bidirectional_sensor_transfer_predictions.csv": sensor_predictions,
        "bidirectional_sensor_transfer_bootstrap_ci.csv": sensor_cis,
        "transfer_protocol_summary.csv": protocol_summary,
        "best_transfer_settings.csv": best_transfer_settings,
        "best_transfer_class_report.csv": best_transfer_class_report,
        "extended_mcda_decision_matrix.csv": mcda_decision,
        "extended_mcda_ranking.csv": mcda_ranking,
        "extended_mcda_rank_agreement.csv": mcda_agreement,
        "extended_mcda_sensitivity_samples.csv": mcda_sensitivity_sample,
        "extended_mcda_sensitivity_summary.csv": mcda_sensitivity_summary,
        "extended_statistical_tests.csv": pd.concat(tests, ignore_index=True),
    }
    outputs.update(five_outputs)
    save_frames(results, outputs)

    for range_name in ["PAMAP2-16g", "PAMAP2-6g"]:
        safe = range_name.lower().replace("-", "_")
        range_by_fold[range_name].to_csv(results / f"{safe}_model_loso_by_fold.csv", index=False)
        range_summaries[range_name].to_csv(results / f"{safe}_model_loso_summary.csv", index=False)

    metadata = {
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "requested_device": args.device,
        "resolved_device": device,
        "selected_pamap2_variant": selected_name,
        "selection_rule": "maximum complete six-class subjects, then maximum windows",
        "primary_subjects": int(primary["subject"].nunique()),
        "mhealth_subjects": int(external["subject"].nunique()),
        "primary_windows": int(len(primary)),
        "mhealth_windows": int(len(external)),
        "primary_best_model": primary_best_name,
        "mhealth_best_model": mhealth_best_name,
        "transfer_protocols": list(configured_protocols),
        "bootstrap_samples": bootstrap_samples,
        "config": config,
    }
    write_json(results / "extended_run_metadata.json", metadata)

    print("Extended pipeline completed successfully.")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
