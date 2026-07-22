from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

# Configure joblib before importing libraries that may initialize its process
# backend.  Some managed Windows environments do not expose physical-core
# metadata.  A value below the logical-core count directs joblib to use that
# explicit limit instead of attempting unavailable physical-core detection.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) - 1)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_activity_dss.data import (
    COMMON_ACTIVITIES,
    SENSOR_CONFIGS,
    download_datasets,
    feature_columns_for_configuration,
    make_synthetic_feature_datasets,
    prepare_feature_datasets,
)
from sports_activity_dss.evaluation import (
    compute_feature_importance,
    confusion_for_predictions,
    external_model_transfer,
    external_sensor_transfer,
    fit_best_model_all,
    loso_model_benchmark,
    loso_sensor_ablation,
    statistical_tests,
    summarize_loso,
)
from sports_activity_dss.mcda import (
    build_decision_matrix,
    rank_agreement,
    rank_alternatives,
    sensitivity_analysis,
)
from sports_activity_dss.models import build_models
from sports_activity_dss.utils import ensure_directories, gpu_is_visible, load_yaml, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete JASE sports-activity experiment.")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-importance", action="store_true")
    parser.add_argument("--sensitivity-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["models"]["random_seed"])
    set_global_seed(seed)

    if args.mode == "smoke":
        output_root = ROOT / "smoke_outputs"
    else:
        output_root = ROOT
    results = output_root / "results"
    processed = output_root / "data" / "processed"
    ensure_directories([results, processed])

    device = args.device
    if device == "auto":
        device = "cuda" if gpu_is_visible() else "cpu"
    print(f"Execution device requested for XGBoost: {device}")

    if args.mode == "smoke":
        primary, external = make_synthetic_feature_datasets(seed)
    else:
        download_datasets(ROOT / "data" / "raw")
        primary, external = prepare_feature_datasets(
            ROOT / "data" / "raw",
            ROOT / "data" / "processed",
            int(config["data"]["target_sampling_hz"]),
            int(config["data"]["window_samples"]),
            int(config["data"]["step_samples"]),
            force=args.force_features,
        )

    # Keep an explicit common class order across both datasets.
    label_encoder = LabelEncoder()
    label_encoder.fit(COMMON_ACTIVITIES)
    full_columns = feature_columns_for_configuration(primary, "Full common configuration")
    common_columns = [c for c in full_columns if c in external.columns]
    if len(common_columns) != len(full_columns):
        missing = sorted(set(full_columns) - set(common_columns))
        raise RuntimeError(f"External dataset is missing common features: {missing[:10]}")

    model_specs = build_models(
        seed=seed,
        n_classes=len(label_encoder.classes_),
        n_jobs=int(config["models"]["n_jobs"]),
        device=device,
        random_forest_estimators=int(config["models"]["random_forest_estimators"]),
        boosting_estimators=int(config["models"]["boosting_estimators"]),
    )

    model_by_fold, model_predictions = loso_model_benchmark(
        primary, common_columns, model_specs, label_encoder
    )
    model_summary = summarize_loso(model_by_fold, "Model")
    best_model_name = str(model_summary.iloc[0]["Model"])
    best_model_spec = model_specs[best_model_name]
    print(f"Best primary model: {best_model_name}")

    sensor_by_fold, sensor_predictions = loso_sensor_ablation(
        primary, SENSOR_CONFIGS, feature_columns_for_configuration, best_model_spec, label_encoder
    )
    sensor_summary = summarize_loso(sensor_by_fold, "Configuration")

    ext_models = external_model_transfer(
        primary, external, common_columns, model_specs, label_encoder
    )
    ext_sensors = external_sensor_transfer(
        primary, external, SENSOR_CONFIGS, feature_columns_for_configuration, best_model_spec, label_encoder
    )

    trained = fit_best_model_all(primary, common_columns, best_model_spec, label_encoder)

    if args.skip_importance:
        importance = pd.DataFrame(columns=["Feature", "Importance mean", "Importance std", "Sensor group"])
    else:
        importance = compute_feature_importance(
            trained,
            primary,
            sample_size=int(config["importance"]["sample_size"]),
            repeats=int(config["importance"]["permutation_repeats"]),
            seed=seed,
        )
    group_importance = (
        importance.groupby("Sensor group", dropna=False)["Importance mean"].sum().sort_values(ascending=False).reset_index()
        if len(importance) else pd.DataFrame(columns=["Sensor group", "Importance mean"])
    )

    decision = build_decision_matrix(sensor_summary, ext_sensors)
    weights = pd.Series(config["mcda"]["weights"], dtype=float)
    ranking = rank_alternatives(decision, weights)
    agreement = rank_agreement(ranking)
    sensitivity_samples = args.sensitivity_samples or int(config["mcda"]["sensitivity_samples"])
    sensitivity_samples_df, sensitivity_summary = sensitivity_analysis(
        decision, weights, sensitivity_samples,
        float(config["mcda"]["concentrated_dirichlet_scale"]), seed,
    )

    best_sensor_name = str(sensor_summary.iloc[0]["Configuration"])
    confusion = confusion_for_predictions(
        sensor_predictions, label_encoder, "Configuration", best_sensor_name
    )

    model_tests = statistical_tests(model_by_fold, "Model")
    model_tests["Family"] = "Model"
    sensor_tests = statistical_tests(sensor_by_fold, "Configuration")
    sensor_tests["Family"] = "Sensor configuration"
    tests = pd.concat([model_tests, sensor_tests], ignore_index=True)

    # Save direct experiment results.
    outputs = {
        "model_loso_by_fold.csv": model_by_fold,
        "model_loso_predictions.csv": model_predictions,
        "model_loso_summary.csv": model_summary,
        "sensor_loso_by_fold.csv": sensor_by_fold,
        "sensor_loso_predictions.csv": sensor_predictions,
        "sensor_loso_summary.csv": sensor_summary,
        "external_model_transfer.csv": ext_models,
        "external_sensor_transfer.csv": ext_sensors,
        "feature_importance.csv": importance,
        "sensor_group_importance.csv": group_importance,
        "mcda_decision_matrix.csv": decision,
        "mcda_ranking.csv": ranking,
        "mcda_rank_agreement.csv": agreement,
        "sensitivity_samples.csv": sensitivity_samples_df,
        "sensitivity_summary.csv": sensitivity_summary,
        "statistical_tests.csv": tests,
    }
    for filename, frame in outputs.items():
        frame.to_csv(results / filename, index=False)
    np.savetxt(results / "best_sensor_confusion_matrix.csv", confusion, delimiter=",")

    metadata = {
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "requested_device": args.device,
        "resolved_device": device,
        "primary_dataset": "PAMAP2" if args.mode == "full" else "synthetic PAMAP2-like",
        "external_dataset": "MHEALTH" if args.mode == "full" else "synthetic MHEALTH-like",
        "primary_subjects": int(primary["subject"].nunique()),
        "external_subjects": int(external["subject"].nunique()),
        "primary_windows": int(len(primary)),
        "external_windows": int(len(external)),
        "feature_count_full": len(common_columns),
        "best_model": best_model_name,
        "best_sensor_configuration": best_sensor_name,
        "sensitivity_samples_per_scheme": sensitivity_samples,
        "config": config,
    }
    write_json(results / "run_metadata.json", metadata)

    print("Pipeline completed successfully.")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
