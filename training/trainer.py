import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import time

from src.config import TARGET, RANDOM_STATE, TEST_SIZE, MODELS_DIR
from preprocessing.preprocessor import build_preprocessor


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def make_pipeline(model, year_col: str = "a\xf1o") -> Pipeline:
    preprocessor = build_preprocessor(year_col=year_col)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series,
    year_col: str = "a\xf1o",
) -> tuple[Pipeline, float]:
    model = LinearRegression()
    pipeline = make_pipeline(model, year_col=year_col)
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    return pipeline, elapsed


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series,
    year_col: str = "a\xf1o",
    **kwargs
) -> tuple[Pipeline, float]:
    default_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    default_params.update(kwargs)
    model = RandomForestRegressor(**default_params)
    pipeline = make_pipeline(model, year_col=year_col)
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    return pipeline, elapsed


def train_hist_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series,
    year_col: str = "a\xf1o",
    **kwargs
) -> tuple[Pipeline, float]:
    default_params = {
        "max_iter": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_samples_leaf": 20,
        "random_state": RANDOM_STATE,
    }
    default_params.update(kwargs)
    model = HistGradientBoostingRegressor(**default_params)
    pipeline = make_pipeline(model, year_col=year_col)
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    return pipeline, elapsed


def save_model(pipeline: Pipeline, name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    return str(path)


def load_model(name: str) -> Pipeline:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
