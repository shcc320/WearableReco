from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _normalise_minmax(matrix: np.ndarray) -> np.ndarray:
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    span = np.where(maximum > minimum, maximum - minimum, 1.0)
    return (matrix - minimum) / span


def topsis(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = np.sqrt((matrix ** 2).sum(axis=0))
    denominator = np.where(denominator > 0, denominator, 1.0)
    weighted = (matrix / denominator) * weights
    ideal = weighted.max(axis=0)
    anti = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
    return d_neg / np.where(d_pos + d_neg > 0, d_pos + d_neg, 1.0)


def edas(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    average = matrix.mean(axis=0)
    average_safe = np.where(np.abs(average) > 1e-12, average, 1.0)
    pda = np.maximum(0.0, (matrix - average) / average_safe)
    nda = np.maximum(0.0, (average - matrix) / average_safe)
    sp = (pda * weights).sum(axis=1)
    sn = (nda * weights).sum(axis=1)
    nsp = sp / max(sp.max(), 1e-12)
    nsn = 1.0 - sn / max(sn.max(), 1e-12)
    return 0.5 * (nsp + nsn)


def mabac(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalised = _normalise_minmax(matrix)
    weighted = weights * (normalised + 1.0)
    border = np.prod(weighted, axis=0) ** (1.0 / matrix.shape[0])
    return (weighted - border).sum(axis=1)


def rank_scores(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(method="min", ascending=False).to_numpy(dtype=int)


def build_decision_matrix(
    sensor_summary: pd.DataFrame,
    external_sensor: pd.DataFrame,
) -> pd.DataFrame:
    external = external_sensor.set_index("Configuration")
    rows = []
    channel_counts = {
        "Chest acceleration": 3,
        "Ankle acceleration": 3,
        "Ankle acceleration and gyroscope": 6,
        "Chest and ankle acceleration": 6,
        "Full common configuration": 9,
    }
    max_channels = max(channel_counts.values())
    for _, row in sensor_summary.iterrows():
        configuration = row["Configuration"]
        latency = max(float(row["Inference ms per 1000 windows mean"]), 1e-6)
        rows.append({
            "Configuration": configuration,
            "within_macro_f1": float(row["Macro F1 mean"]),
            "external_macro_f1": float(external.loc[configuration, "Macro F1"]),
            "stability_score": max(0.0, 1.0 - float(row["Macro F1 std"])),
            "inference_efficiency": 1.0 / latency,
            "wearability_score": 1.0 - (channel_counts[configuration] - 3) / max(max_channels - 3, 1),
            "Channel count": channel_counts[configuration],
        })
    return pd.DataFrame(rows)


def rank_alternatives(decision: pd.DataFrame, weight_series: pd.Series) -> pd.DataFrame:
    criteria = list(weight_series.index)
    matrix = decision[criteria].to_numpy(dtype=float)
    weights = weight_series.to_numpy(dtype=float)
    weights = weights / weights.sum()
    scores = {
        "TOPSIS score": topsis(matrix, weights),
        "EDAS score": edas(matrix, weights),
        "MABAC score": mabac(matrix, weights),
    }
    result = decision[["Configuration", *criteria, "Channel count"]].copy()
    for name, values in scores.items():
        result[name] = values
        result[name.replace("score", "rank")] = rank_scores(values)
    rank_columns = ["TOPSIS rank", "EDAS rank", "MABAC rank"]
    result["Mean rank"] = result[rank_columns].mean(axis=1)
    result["Final rank"] = result["Mean rank"].rank(method="min", ascending=True).astype(int)
    return result.sort_values(["Final rank", "Mean rank"]).reset_index(drop=True)


def rank_agreement(ranking: pd.DataFrame) -> pd.DataFrame:
    methods = ["TOPSIS rank", "EDAS rank", "MABAC rank"]
    rows = []
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            rho, p = spearmanr(ranking[left], ranking[right])
            rows.append({"Method A": left, "Method B": right, "Spearman rho": rho, "P value": p})
    return pd.DataFrame(rows)


def sensitivity_analysis(
    decision: pd.DataFrame,
    base_weights: pd.Series,
    samples: int,
    concentrated_scale: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    criteria = list(base_weights.index)
    matrix = decision[criteria].to_numpy(dtype=float)
    names = decision["Configuration"].tolist()

    records = []
    for scheme, alpha in [
        ("Concentrated", base_weights.to_numpy() * concentrated_scale),
        ("Uniform", np.ones(len(criteria))),
    ]:
        sampled_weights = rng.dirichlet(alpha, size=samples)
        for sample_id, weights in enumerate(sampled_weights):
            method_ranks = np.column_stack([
                rank_scores(topsis(matrix, weights)),
                rank_scores(edas(matrix, weights)),
                rank_scores(mabac(matrix, weights)),
            ])
            aggregate_rank = method_ranks.mean(axis=1)
            final_rank = pd.Series(aggregate_rank).rank(method="min", ascending=True).to_numpy(dtype=int)
            for name, rank in zip(names, final_rank):
                records.append({
                    "Scheme": scheme,
                    "Sample": sample_id,
                    "Configuration": name,
                    "Final rank": int(rank),
                })
    samples_df = pd.DataFrame(records)
    summary = samples_df.groupby(["Scheme", "Configuration"])["Final rank"].agg(
        Median_rank="median",
        Mean_rank="mean",
        Top1_frequency=lambda s: float(np.mean(s == 1)),
    ).reset_index()
    return samples_df, summary
