import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import time

from src.config import RANDOM_STATE, CV_FOLDS, RF_PARAMS, HGB_PARAMS
from preprocessing.preprocessor import build_preprocessor


def _get_model_step(pipeline: Pipeline):
    return pipeline.named_steps["model"]


def optimize_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series,
    n_iter: int = 30, cv: int = CV_FOLDS,
    year_col: str = "a\xf1o",
) -> tuple[Pipeline, dict, float]:
    base_pipeline = build_preprocessor(year_col=year_col)
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    full_pipeline = Pipeline([
        ("preprocessor", base_pipeline),
        ("model", model),
    ])

    param_distributions = {
        "model__" + k: v for k, v in RF_PARAMS.items()
    }

    search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    return search.best_estimator_, search.best_params_, elapsed


def optimize_hist_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series,
    n_iter: int = 30, cv: int = CV_FOLDS,
    year_col: str = "a\xf1o",
) -> tuple[Pipeline, dict, float]:
    base_pipeline = build_preprocessor(year_col=year_col)
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    full_pipeline = Pipeline([
        ("preprocessor", base_pipeline),
        ("model", model),
    ])

    param_distributions = {
        "model__" + k: v for k, v in HGB_PARAMS.items()
    }

    search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    return search.best_estimator_, search.best_params_, elapsed
