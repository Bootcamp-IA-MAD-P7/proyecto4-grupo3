import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from typing import Callable

from src.config import CV_FOLDS, RANDOM_STATE, TARGET


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def compute_overfitting(train_r2: float, test_r2: float) -> float:
    return train_r2 - test_r2


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_log1p: bool = False,
) -> dict:
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    if use_log1p:
        y_true_train = np.expm1(y_train)
        y_true_test = np.expm1(y_test)
        y_pred_train = np.expm1(y_train_pred)
        y_pred_test = np.expm1(y_test_pred)
    else:
        y_true_train = y_train
        y_true_test = y_test
        y_pred_train = y_train_pred
        y_pred_test = y_test_pred

    train_metrics = compute_metrics(y_true_train, y_pred_train)
    test_metrics = compute_metrics(y_true_test, y_pred_test)
    overfitting = compute_overfitting(train_metrics["R2"], test_metrics["R2"])

    return {
        "train": train_metrics,
        "test": test_metrics,
        "overfitting": overfitting,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_true_train": y_true_train,
        "y_true_test": y_true_test,
    }


def cross_validate_model(
    pipeline_builder: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    target_is_log1p: bool = False,
) -> dict:
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    y_array = y.values

    for train_idx, val_idx in kfold.split(X):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y_array[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y_array[val_idx]

        pipeline = pipeline_builder()
        pipeline.fit(X_fold_train, y_fold_train)
        y_pred = pipeline.predict(X_fold_val)

        if target_is_log1p:
            y_fold_val_orig = np.expm1(y_fold_val)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_fold_val_orig = y_fold_val
            y_pred_orig = y_pred

        r2_scores.append(r2_score(y_fold_val_orig, y_pred_orig))
        mae_scores.append(mean_absolute_error(y_fold_val_orig, y_pred_orig))
        rmse_scores.append(np.sqrt(mean_squared_error(y_fold_val_orig, y_pred_orig)))

    return {
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "R2_scores": r2_scores,
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores)),
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores)),
    }


def compare_models(results: dict[str, dict]) -> str:
    best_model = None
    best_r2 = -float("inf")

    for name, result in results.items():
        test_r2 = result.get("test", {}).get("R2", result.get("R2_mean", -999))
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model = name

    return best_model
