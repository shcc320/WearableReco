from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from .data import (
    COMMON_ACTIVITIES,
    COMMON_CHANNELS,
    MHEALTH_LABELS,
    SENSOR_CONFIGS,
    _signal_features,
    extract_window_features,
    feature_columns_for_configuration,
    load_mhealth_samples,
)
from .evaluation import metric_row
from .models import ModelSpec, fit_with_weights
from .utils import timed_predict_proba
from sklearn.utils.class_weight import compute_sample_weight


PAMAP2_LABELS = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
}

PAMAP2_ACCEL_COLUMNS = {
    # Official PAMAP2 column indices are zero-based here.
    # Both accelerometers are provided by each IMU. The previous pipeline used
    # the +/-16g channels; this extension also evaluates the +/-6g channels.
    "16g": {
        1: "activity_id",
        21: "chest_acc_x", 22: "chest_acc_y", 23: "chest_acc_z",
        38: "ankle_acc_x", 39: "ankle_acc_y", 40: "ankle_acc_z",
        44: "ankle_gyro_x", 45: "ankle_gyro_y", 46: "ankle_gyro_z",
    },
    "6g": {
        1: "activity_id",
        24: "chest_acc_x", 25: "chest_acc_y", 26: "chest_acc_z",
        41: "ankle_acc_x", 42: "ankle_acc_y", 43: "ankle_acc_z",
        44: "ankle_gyro_x", 45: "ankle_gyro_y", 46: "ankle_gyro_z",
    },
}

TRANSFER_PROTOCOLS = {
    "All features - raw": ("all", "raw"),
    "Magnitude only - raw": ("magnitude", "raw"),
    "Magnitude - domain z-score": ("magnitude", "domain_zscore"),
    "Magnitude - CORAL": ("magnitude", "coral"),
}

CHANNEL_COUNTS = {
    "Chest acceleration": 3,
    "Ankle acceleration": 3,
    "Ankle acceleration and gyroscope": 6,
    "Chest and ankle acceleration": 6,
    "Full common configuration": 9,
}
BODY_LOCATION_COUNTS = {
    "Chest acceleration": 1,
    "Ankle acceleration": 1,
    "Ankle acceleration and gyroscope": 1,
    "Chest and ankle acceleration": 2,
    "Full common configuration": 2,
}


def _subject_number(path: Path) -> int:
    match = re.search(r"subject(\d+)", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine subject id from {path}")
    return int(match.group(1))


def load_pamap2_samples_variant(
    raw_root: Path,
    target_hz: int = 50,
    accel_range: str = "6g",
) -> pd.DataFrame:
    if accel_range not in PAMAP2_ACCEL_COLUMNS:
        raise ValueError(f"Unknown PAMAP2 accelerometer range: {accel_range}")
    files = sorted((raw_root / "pamap2").glob("**/Protocol/subject*.dat"))
    if not files:
        raise FileNotFoundError("PAMAP2 protocol files were not found. Run the original downloader first.")

    selected = PAMAP2_ACCEL_COLUMNS[accel_range]
    frames: list[pd.DataFrame] = []
    for path in files:
        print(f"Reading PAMAP2 {accel_range} {path.name}")
        raw = pd.read_csv(path, sep=r"\s+", header=None, usecols=sorted(selected))
        raw = raw.rename(columns=selected)
        raw["segment_id"] = raw["activity_id"].ne(raw["activity_id"].shift()).cumsum()
        raw = raw[raw["activity_id"].isin(PAMAP2_LABELS)].copy()
        raw["activity"] = raw["activity_id"].map(PAMAP2_LABELS)
        raw["subject"] = f"P{_subject_number(path):03d}"

        cleaned_segments: list[pd.DataFrame] = []
        for _, segment in raw.groupby("segment_id", sort=False):
            segment = segment.copy()
            before = len(segment)
            segment[COMMON_CHANNELS] = segment[COMMON_CHANNELS].interpolate(
                method="linear", limit_direction="both"
            )
            segment = segment.dropna(subset=COMMON_CHANNELS)
            if target_hz == 50:
                segment = segment.iloc[::2].copy()
            if len(segment):
                cleaned_segments.append(segment)
        if cleaned_segments:
            subject_df = pd.concat(cleaned_segments, ignore_index=True)
            frames.append(subject_df[["subject", "segment_id", "activity", *COMMON_CHANNELS]])
    if not frames:
        raise RuntimeError(f"No usable PAMAP2 samples for accelerometer range {accel_range}.")
    return pd.concat(frames, ignore_index=True)


def prepare_extended_feature_datasets(
    raw_root: Path,
    processed_root: Path,
    target_hz: int,
    window_samples: int,
    step_samples: int,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    processed_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, pd.DataFrame] = {}
    for accel_range in ["16g", "6g"]:
        cache = processed_root / f"pamap2_{accel_range}_features_extended.pkl.gz"
        if force or not cache.exists():
            samples = load_pamap2_samples_variant(raw_root, target_hz, accel_range)
            features = extract_window_features(samples, target_hz, window_samples, step_samples)
            features.to_pickle(cache, compression="gzip")
        else:
            features = pd.read_pickle(cache, compression="gzip")
        output[f"PAMAP2-{accel_range}"] = features

    mhealth_cache = processed_root / "mhealth_features.pkl.gz"
    if force or not mhealth_cache.exists():
        samples = load_mhealth_samples(raw_root)
        features = extract_window_features(samples, target_hz, window_samples, step_samples)
        features.to_pickle(mhealth_cache, compression="gzip")
    else:
        features = pd.read_pickle(mhealth_cache, compression="gzip")
    output["MHEALTH"] = features
    return output


def participant_activity_audit(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required = set(COMMON_ACTIVITIES)
    for dataset, frame in datasets.items():
        counts = frame.groupby(["subject", "activity"]).size().unstack(fill_value=0)
        for activity in COMMON_ACTIVITIES:
            if activity not in counts:
                counts[activity] = 0
        counts = counts[COMMON_ACTIVITIES]
        for subject, row in counts.iterrows():
            present = {activity for activity in COMMON_ACTIVITIES if int(row[activity]) > 0}
            missing = sorted(required - present)
            record: dict[str, Any] = {
                "Dataset variant": dataset,
                "Subject": subject,
                "Complete six-class subject": len(missing) == 0,
                "Missing activities": "; ".join(missing),
                "Total windows": int(row.sum()),
            }
            for activity in COMMON_ACTIVITIES:
                record[f"Windows - {activity}"] = int(row[activity])
            rows.append(record)
    return pd.DataFrame(rows)


def complete_subject_cohort(frame: pd.DataFrame, activities: Iterable[str]) -> pd.DataFrame:
    activities = list(activities)
    activity_set = set(activities)
    coverage = frame.groupby("subject")["activity"].agg(lambda s: set(s.unique()))
    subjects = coverage[coverage.apply(lambda present: activity_set.issubset(present))].index
    return frame[frame["subject"].isin(subjects) & frame["activity"].isin(activities)].copy().reset_index(drop=True)


def best_five_class_subset(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    for dropped in COMMON_ACTIVITIES:
        activities = [a for a in COMMON_ACTIVITIES if a != dropped]
        cohort = complete_subject_cohort(frame, activities)
        candidates.append({
            "Dropped activity": dropped,
            "Activities": "; ".join(activities),
            "Complete subjects": int(cohort["subject"].nunique()),
            "Windows": int(len(cohort)),
        })
    table = pd.DataFrame(candidates).sort_values(
        ["Complete subjects", "Windows", "Dropped activity"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected = [a for a in COMMON_ACTIVITIES if a != str(table.iloc[0]["Dropped activity"])]
    return selected, table


def select_feature_columns(
    frame: pd.DataFrame,
    configuration: str,
    representation: str,
) -> list[str]:
    columns = feature_columns_for_configuration(frame, configuration)
    if representation == "all":
        return columns
    if representation == "magnitude":
        selected = [column for column in columns if "_mag__" in column]
        if not selected:
            raise RuntimeError(f"No magnitude features for {configuration}")
        return selected
    raise ValueError(f"Unknown representation: {representation}")


def _symmetric_matrix_power(matrix: np.ndarray, power: float, floor: float = 1e-9) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, floor)
    return (vectors * np.power(values, power)) @ vectors.T


def transform_domains(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
    transform: str,
    regularization: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xs = source[columns].to_numpy(dtype=float)
    xt = target[columns].to_numpy(dtype=float)
    if transform == "raw":
        return source[columns].copy(), target[columns].copy()

    if transform in {"domain_zscore", "coral"}:
        source_mean = xs.mean(axis=0)
        source_std = xs.std(axis=0, ddof=0)
        target_mean = xt.mean(axis=0)
        target_std = xt.std(axis=0, ddof=0)
        source_std = np.where(source_std > 1e-9, source_std, 1.0)
        target_std = np.where(target_std > 1e-9, target_std, 1.0)
        xs_z = (xs - source_mean) / source_std
        xt_z = (xt - target_mean) / target_std
        if transform == "domain_zscore":
            return pd.DataFrame(xs_z, columns=columns), pd.DataFrame(xt_z, columns=columns)

        # CORAL uses target feature statistics but never target labels.
        cs = np.cov(xs_z, rowvar=False) + regularization * np.eye(len(columns))
        ct = np.cov(xt_z, rowvar=False) + regularization * np.eye(len(columns))
        alignment = _symmetric_matrix_power(cs, -0.5) @ _symmetric_matrix_power(ct, 0.5)
        xs_coral = xs_z @ alignment
        return pd.DataFrame(xs_coral, columns=columns), pd.DataFrame(xt_z, columns=columns)
    raise ValueError(f"Unknown transform: {transform}")


def _fit_predict_transformed(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    estimator = copy.deepcopy(spec.estimator)
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    try:
        estimator = fit_with_weights(estimator, x_train, y_train, weights)
    except Exception:
        if spec.name == "XGBoost" and getattr(estimator, "get_params", lambda: {})().get("device") == "cuda":
            estimator.set_params(device="cpu")
            estimator = fit_with_weights(estimator, x_train, y_train, weights)
        else:
            raise
    probabilities = estimator.predict_proba(x_test)
    predictions = np.argmax(probabilities, axis=1)
    latency = timed_predict_proba(estimator, x_test)
    return predictions, probabilities, latency


@dataclass
class TransferOutput:
    aggregate: pd.DataFrame
    by_subject: pd.DataFrame
    predictions: pd.DataFrame
    confidence_intervals: pd.DataFrame


def bootstrap_subject_ci(values: np.ndarray, seed: int, samples: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_transfer_grid(
    source: pd.DataFrame,
    target: pd.DataFrame,
    model_specs: dict[str, ModelSpec],
    label_encoder: LabelEncoder,
    direction: str,
    protocols: dict[str, tuple[str, str]],
    entities: dict[str, tuple[str, ModelSpec]],
    seed: int,
    bootstrap_samples: int,
) -> TransferOutput:
    """Evaluate models or configurations across transfer protocols.

    entities maps display name -> (configuration, model spec). For model-family
    evaluation, configuration is Full common configuration and each model differs.
    For sensor evaluation, model spec is fixed and each configuration differs.
    """
    y_train = label_encoder.transform(source["activity"])
    y_test = label_encoder.transform(target["activity"])
    aggregate_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for protocol_name, (representation, transform) in protocols.items():
        for entity_name, (configuration, spec) in entities.items():
            columns = select_feature_columns(source, configuration, representation)
            missing = sorted(set(columns) - set(target.columns))
            if missing:
                raise RuntimeError(f"Target misses {len(missing)} features, e.g. {missing[:5]}")
            x_source, x_target = transform_domains(source, target, columns, transform)
            print(f"Transfer {direction}: protocol={protocol_name}; entity={entity_name}")
            pred, proba, latency = _fit_predict_transformed(spec, x_source, y_train, x_target)
            metrics = metric_row(y_test, pred, proba)
            aggregate_rows.append({
                "Direction": direction,
                "Protocol": protocol_name,
                "Entity": entity_name,
                "Configuration": configuration,
                "Model": spec.name,
                "Feature count": len(columns),
                **metrics,
                "Inference ms per 1000 windows": latency,
            })

            for target_subject in sorted(target["subject"].unique()):
                mask = target["subject"].eq(target_subject).to_numpy()
                subject_metrics = metric_row(y_test[mask], pred[mask], proba[mask])
                subject_rows.append({
                    "Direction": direction,
                    "Protocol": protocol_name,
                    "Entity": entity_name,
                    "Configuration": configuration,
                    "Model": spec.name,
                    "Target subject": target_subject,
                    **subject_metrics,
                })
            for row_index, subject, truth, predicted in zip(target.index, target["subject"], y_test, pred):
                prediction_rows.append({
                    "Direction": direction,
                    "Protocol": protocol_name,
                    "Entity": entity_name,
                    "Configuration": configuration,
                    "Model": spec.name,
                    "Row index": int(row_index),
                    "Target subject": subject,
                    "True label": label_encoder.inverse_transform([truth])[0],
                    "Predicted label": label_encoder.inverse_transform([predicted])[0],
                })

    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["Direction", "Protocol", "Macro F1"], ascending=[True, True, False]
    ).reset_index(drop=True)
    by_subject = pd.DataFrame(subject_rows)
    predictions = pd.DataFrame(prediction_rows)
    ci_rows: list[dict[str, Any]] = []
    for keys, group in by_subject.groupby(["Direction", "Protocol", "Entity", "Configuration", "Model"]):
        mean, low, high = bootstrap_subject_ci(group["Macro F1"].to_numpy(), seed, bootstrap_samples)
        ci_rows.append({
            "Direction": keys[0], "Protocol": keys[1], "Entity": keys[2],
            "Configuration": keys[3], "Model": keys[4],
            "Subject-mean macro F1": mean,
            "Bootstrap 95% CI low": low,
            "Bootstrap 95% CI high": high,
            "Target subjects": int(group["Target subject"].nunique()),
        })
    cis = pd.DataFrame(ci_rows)
    return TransferOutput(aggregate, by_subject, predictions, cis)


def normalized_confusion_from_predictions(
    predictions: pd.DataFrame,
    label_encoder: LabelEncoder,
    direction: str,
    protocol: str,
    entity: str,
) -> np.ndarray:
    selected = predictions[
        predictions["Direction"].eq(direction)
        & predictions["Protocol"].eq(protocol)
        & predictions["Entity"].eq(entity)
    ]
    y_true = label_encoder.transform(selected["True label"])
    y_pred = label_encoder.transform(selected["Predicted label"])
    return confusion_matrix(
        y_true, y_pred, labels=np.arange(len(label_encoder.classes_)), normalize="true"
    )


def transfer_protocol_summary(sensor_aggregate: pd.DataFrame) -> pd.DataFrame:
    return sensor_aggregate.groupby(["Direction", "Protocol"]).agg(
        Best_macro_F1=("Macro F1", "max"),
        Mean_macro_F1=("Macro F1", "mean"),
        Median_macro_F1=("Macro F1", "median"),
    ).reset_index()


def build_extended_mcda_decision(
    pamap_sensor_summary: pd.DataFrame,
    mhealth_sensor_summary: pd.DataFrame,
    sensor_transfer: pd.DataFrame,
    protocol: str,
) -> pd.DataFrame:
    pamap = pamap_sensor_summary.set_index("Configuration")
    mhealth = mhealth_sensor_summary.set_index("Configuration")
    transfer = sensor_transfer[sensor_transfer["Protocol"].eq(protocol)]
    directions = list(transfer["Direction"].drop_duplicates())
    rows: list[dict[str, Any]] = []
    common = [c for c in SENSOR_CONFIGS if c in pamap.index and c in mhealth.index]
    for configuration in common:
        directional = transfer[transfer["Entity"].eq(configuration)].set_index("Direction")
        transfer_values = [float(directional.loc[d, "Macro F1"]) for d in directions if d in directional.index]
        latencies = [
            float(pamap.loc[configuration, "Inference ms per 1000 windows mean"]),
            float(mhealth.loc[configuration, "Inference ms per 1000 windows mean"]),
        ]
        rows.append({
            "Configuration": configuration,
            "internal_mean_macro_f1": 0.5 * (
                float(pamap.loc[configuration, "Macro F1 mean"])
                + float(mhealth.loc[configuration, "Macro F1 mean"])
            ),
            "bidirectional_transfer_macro_f1": float(np.mean(transfer_values)),
            "stability_score": max(0.0, 1.0 - 0.5 * (
                float(pamap.loc[configuration, "Macro F1 std"])
                + float(mhealth.loc[configuration, "Macro F1 std"])
            )),
            "inference_efficiency": 1.0 / max(float(np.mean(latencies)), 1e-9),
            "body_location_score": 1.0 if BODY_LOCATION_COUNTS[configuration] == 1 else 0.0,
            "channel_score": 1.0 - (CHANNEL_COUNTS[configuration] - 3) / 6.0,
            "Channel count": CHANNEL_COUNTS[configuration],
            "Body locations": BODY_LOCATION_COUNTS[configuration],
            "Protocol": protocol,
        })
    return pd.DataFrame(rows)
