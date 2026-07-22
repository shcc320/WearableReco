from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelSpec:
    name: str
    estimator: Any


def build_models(
    seed: int,
    n_classes: int,
    n_jobs: int,
    device: str,
    random_forest_estimators: int,
    boosting_estimators: int,
) -> dict[str, ModelSpec]:
    xgb_device = "cuda" if device == "cuda" else "cpu"
    models = {
        "Logistic regression": ModelSpec(
            "Logistic regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2500,
                    solver="lbfgs",
                    random_state=seed,
                )),
            ]),
        ),
        "Random forest": ModelSpec(
            "Random forest",
            RandomForestClassifier(
                n_estimators=random_forest_estimators,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            ),
        ),
        "Histogram gradient boosting": ModelSpec(
            "Histogram gradient boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=boosting_estimators,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=seed,
            ),
        ),
        "XGBoost": ModelSpec(
            "XGBoost",
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                n_estimators=boosting_estimators,
                learning_rate=0.055,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                tree_method="hist",
                device=xgb_device,
                random_state=seed,
                n_jobs=n_jobs,
            ),
        ),
        "LightGBM": ModelSpec(
            "LightGBM",
            LGBMClassifier(
                objective="multiclass",
                n_estimators=boosting_estimators,
                learning_rate=0.055,
                max_depth=-1,
                num_leaves=31,
                reg_lambda=0.1,
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            ),
        ),
    }
    return models


def fit_with_weights(estimator: Any, x: Any, y: Any, sample_weight: np.ndarray) -> Any:
    if isinstance(estimator, Pipeline):
        estimator.fit(x, y, model__sample_weight=sample_weight)
    else:
        estimator.fit(x, y, sample_weight=sample_weight)
    return estimator
