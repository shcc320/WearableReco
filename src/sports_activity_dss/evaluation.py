from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .models import ModelSpec, fit_with_weights
from .utils import timed_predict_proba


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    feature_columns: list[str]
    label_encoder: LabelEncoder


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.arange(probabilities.shape[1])
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Log loss": log_loss(y_true, probabilities, labels=labels),
    }


def _fit_predict(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, np.ndarray, float]:
    estimator = copy.deepcopy(spec.estimator)
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    try:
        estimator = fit_with_weights(estimator, x_train, y_train, weights)
    except Exception as exc:
        # CUDA-enabled XGBoost packages vary by environment. Retry on CPU without
        # changing the statistical protocol.
        if spec.name == "XGBoost" and getattr(estimator, "get_params", lambda: {})().get("device") == "cuda":
            estimator.set_params(device="cpu")
            estimator = fit_with_weights(estimator, x_train, y_train, weights)
        else:
            raise
    probabilities = estimator.predict_proba(x_test)
    predictions = np.argmax(probabilities, axis=1)
    latency_ms = timed_predict_proba(estimator, x_test)
    return estimator, predictions, probabilities, latency_ms


def loso_model_benchmark(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_specs: dict[str, ModelSpec],
    label_encoder: LabelEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    y_all = label_encoder.transform(frame["activity"])
    subjects = sorted(frame["subject"].unique())

    for held_out in subjects:
        test_mask = frame["subject"].eq(held_out).to_numpy()
        x_train = frame.loc[~test_mask, feature_columns]
        x_test = frame.loc[test_mask, feature_columns]
        y_train = y_all[~test_mask]
        y_test = y_all[test_mask]
        for model_name, spec in model_specs.items():
            print(f"LOSO model={model_name}, held-out={held_out}")
            _, y_pred, probabilities, latency = _fit_predict(spec, x_train, y_train, x_test)
            row = {"Model": model_name, "Held-out subject": held_out, **metric_row(y_test, y_pred, probabilities)}
            row["Inference ms per 1000 windows"] = latency
            rows.append(row)
            for idx, truth, pred in zip(frame.index[test_mask], y_test, y_pred):
                predictions.append({
                    "Model": model_name,
                    "Row index": int(idx),
                    "Subject": held_out,
                    "True label": label_encoder.inverse_transform([truth])[0],
                    "Predicted label": label_encoder.inverse_transform([pred])[0],
                })
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def summarize_loso(by_fold: pd.DataFrame, group_column: str) -> pd.DataFrame:
    metrics = [
        "Accuracy", "Balanced accuracy", "Macro precision", "Macro recall",
        "Macro F1", "Log loss", "Inference ms per 1000 windows",
    ]
    summary = by_fold.groupby(group_column)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        group_column if a == group_column else f"{a} {b}".strip()
        for a, b in summary.columns.to_flat_index()
    ]
    return summary.sort_values("Macro F1 mean", ascending=False).reset_index(drop=True)


def loso_sensor_ablation(
    frame: pd.DataFrame,
    configurations: dict[str, list[str]],
    feature_selector: Callable[[pd.DataFrame, str], list[str]],
    best_model_spec: ModelSpec,
    label_encoder: LabelEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    y_all = label_encoder.transform(frame["activity"])
    subjects = sorted(frame["subject"].unique())
    for configuration in configurations:
        columns = feature_selector(frame, configuration)
        for held_out in subjects:
            test_mask = frame["subject"].eq(held_out).to_numpy()
            print(f"LOSO configuration={configuration}, held-out={held_out}")
            _, y_pred, probabilities, latency = _fit_predict(
                best_model_spec,
                frame.loc[~test_mask, columns], y_all[~test_mask],
                frame.loc[test_mask, columns],
            )
            row = {
                "Configuration": configuration,
                "Held-out subject": held_out,
                "Feature count": len(columns),
                **metric_row(y_all[test_mask], y_pred, probabilities),
                "Inference ms per 1000 windows": latency,
            }
            rows.append(row)
            for idx, truth, pred in zip(frame.index[test_mask], y_all[test_mask], y_pred):
                predictions.append({
                    "Configuration": configuration,
                    "Row index": int(idx),
                    "Subject": held_out,
                    "True label": label_encoder.inverse_transform([truth])[0],
                    "Predicted label": label_encoder.inverse_transform([pred])[0],
                })
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def external_model_transfer(
    primary: pd.DataFrame,
    external: pd.DataFrame,
    feature_columns: list[str],
    model_specs: dict[str, ModelSpec],
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    y_train = label_encoder.transform(primary["activity"])
    y_test = label_encoder.transform(external["activity"])
    rows = []
    for model_name, spec in model_specs.items():
        print(f"External model transfer={model_name}")
        _, y_pred, probabilities, latency = _fit_predict(
            spec, primary[feature_columns], y_train, external[feature_columns]
        )
        rows.append({
            "Model": model_name,
            **metric_row(y_test, y_pred, probabilities),
            "Inference ms per 1000 windows": latency,
        })
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


def external_sensor_transfer(
    primary: pd.DataFrame,
    external: pd.DataFrame,
    configurations: dict[str, list[str]],
    feature_selector: Callable[[pd.DataFrame, str], list[str]],
    best_model_spec: ModelSpec,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    y_train = label_encoder.transform(primary["activity"])
    y_test = label_encoder.transform(external["activity"])
    rows = []
    for configuration in configurations:
        columns = feature_selector(primary, configuration)
        print(f"External sensor transfer={configuration}")
        _, y_pred, probabilities, latency = _fit_predict(
            best_model_spec, primary[columns], y_train, external[columns]
        )
        rows.append({
            "Configuration": configuration,
            "Feature count": len(columns),
            **metric_row(y_test, y_pred, probabilities),
            "Inference ms per 1000 windows": latency,
        })
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


def fit_best_model_all(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: ModelSpec,
    label_encoder: LabelEncoder,
) -> TrainedModel:
    y = label_encoder.transform(frame["activity"])
    weights = compute_sample_weight(class_weight="balanced", y=y)
    estimator = copy.deepcopy(spec.estimator)
    try:
        estimator = fit_with_weights(estimator, frame[feature_columns], y, weights)
    except Exception as exc:
        if spec.name == "XGBoost" and getattr(estimator, "get_params", lambda: {})().get("device") == "cuda":
            estimator.set_params(device="cpu")
            estimator = fit_with_weights(estimator, frame[feature_columns], y, weights)
        else:
            raise
    return TrainedModel(spec.name, estimator, feature_columns, label_encoder)


def compute_feature_importance(
    trained: TrainedModel,
    frame: pd.DataFrame,
    sample_size: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    sample = frame.sample(min(sample_size, len(frame)), random_state=seed)
    x = sample[trained.feature_columns]
    y = trained.label_encoder.transform(sample["activity"])

    # Permutation importance is model-agnostic and provides a reliable fallback.
    result = permutation_importance(
        trained.estimator,
        x,
        y,
        scoring=make_scorer(
            f1_score,
            labels=np.arange(len(trained.label_encoder.classes_)),
            average="macro",
            zero_division=0,
        ),
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "Feature": trained.feature_columns,
        "Importance mean": result.importances_mean,
        "Importance std": result.importances_std,
    }).sort_values("Importance mean", ascending=False)
    importance["Sensor group"] = importance["Feature"].str.extract(
        r"^(chest_acc|ankle_acc|ankle_gyro)", expand=False
    ).map({
        "chest_acc": "Chest acceleration",
        "ankle_acc": "Ankle acceleration",
        "ankle_gyro": "Ankle gyroscope",
    })
    return importance.reset_index(drop=True)


def confusion_for_predictions(
    predictions: pd.DataFrame,
    label_encoder: LabelEncoder,
    filter_column: str,
    filter_value: str,
) -> np.ndarray:
    selected = predictions[predictions[filter_column].eq(filter_value)]
    y_true = label_encoder.transform(selected["True label"])
    y_pred = label_encoder.transform(selected["Predicted label"])
    return confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)), normalize="true")


def statistical_tests(by_fold: pd.DataFrame, group_column: str) -> pd.DataFrame:
    pivot = by_fold.pivot(index="Held-out subject", columns=group_column, values="Macro F1")
    pivot = pivot.dropna(axis=0)
    rows: list[dict[str, Any]] = []
    if pivot.shape[1] >= 3 and pivot.shape[0] >= 3:
        statistic, p_value = friedmanchisquare(*[pivot[column] for column in pivot.columns])
        rows.append({
            "Test": "Friedman", "Comparison": group_column,
            "Statistic": statistic, "P value": p_value,
            "Adjusted alpha": np.nan, "Significant": p_value < 0.05,
        })
        comparisons = pivot.shape[1] * (pivot.shape[1] - 1) // 2
        alpha = 0.05 / comparisons
        columns = list(pivot.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1:]:
                try:
                    stat, p = wilcoxon(pivot[left], pivot[right], zero_method="zsplit")
                except ValueError:
                    stat, p = np.nan, 1.0
                rows.append({
                    "Test": "Wilcoxon", "Comparison": f"{left} vs {right}",
                    "Statistic": stat, "P value": p,
                    "Adjusted alpha": alpha, "Significant": p < alpha,
                })
    return pd.DataFrame(rows)
