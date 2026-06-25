import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
)
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna
import joblib
from pathlib import Path

from src.config import MODEL_PATH, FIGURES_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)
METRICS = ["MAE", "RMSE", "R2"]


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=10, min_samples_split=20,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1, reg_lambda=2, min_child_samples=5,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def cross_validate_kfold(model_fn, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2_scores, mae_scores, rmse_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        m = model_fn(X_fold_train, y_fold_train)
        y_pred = m.predict(X_fold_val)

        y_orig = np.expm1(y_fold_val)
        y_pred_orig = np.expm1(y_pred)

        r2_scores.append(r2_score(y_orig, y_pred_orig))
        mae_scores.append(mean_absolute_error(y_orig, y_pred_orig))
        rmse_scores.append(root_mean_squared_error(y_orig, y_pred_orig))

        logger.info("    Fold %s: R2=%.4f MAE=%.2f RMSE=%.2f",
                    fold, r2_scores[-1], mae_scores[-1], rmse_scores[-1])

    logger.info("  K-Fold CV (n=%s):", n_splits)
    logger.info("    R2:  media=%.4f +/- %.4f", np.mean(r2_scores), np.std(r2_scores))
    logger.info("    MAE: media=%.4f +/- %.4f", np.mean(mae_scores), np.std(mae_scores))
    logger.info("    RMSE: media=%.4f +/- %.4f", np.mean(rmse_scores), np.std(rmse_scores))

    return {"R2": r2_scores, "MAE": mae_scores, "RMSE": rmse_scores}


def evaluate_model(model, X_train, X_test, y_train, y_test, target_transformer, name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_orig = target_transformer.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test_orig = target_transformer.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_train_pred_orig = target_transformer.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
    y_test_pred_orig = target_transformer.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

    metrics = {
        "train": {
            "MAE": mean_absolute_error(y_train_orig, y_train_pred_orig),
            "RMSE": root_mean_squared_error(y_train_orig, y_train_pred_orig),
            "R2": r2_score(y_train_orig, y_train_pred_orig),
        },
        "test": {
            "MAE": mean_absolute_error(y_test_orig, y_test_pred_orig),
            "RMSE": root_mean_squared_error(y_test_orig, y_test_pred_orig),
            "R2": r2_score(y_test_orig, y_test_pred_orig),
        },
    }

    logger.info("")
    logger.info("=" * 50)
    logger.info("  %s", name)
    logger.info("=" * 50)
    logger.info("  %-10s %-15s %-15s", "Metrica", "Train", "Test")
    logger.info("  %s", "-" * 40)
    for m in METRICS:
        logger.info("  %-10s %-15.4f %-15.4f", m, metrics["train"][m], metrics["test"][m])

    overfitting = metrics["train"]["R2"] - metrics["test"]["R2"]
    logger.info("")
    logger.info("  Overfitting (R2 diff): %.4f", overfitting)

    return metrics, y_test_pred_orig, y_test_orig


def plot_predictions(y_test, y_pred, name: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=5, c="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Prediccion perfecta")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Valor real (ha)")
    ax.set_ylabel("Prediccion (ha)")
    ax.set_title(f"{name} - Prediccion vs Valor real")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"predictions_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_test, y_pred, name: str, output_dir: Path) -> None:
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=50, color="lightcoral", edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title(f"{name} - Histograma de residuos")
    axes[0].set_xlabel("Residuo (ha)")
    axes[0].set_ylabel("Frecuencia")

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=5, c="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title(f"{name} - Residuos vs Prediccion")
    axes[1].set_xlabel("Prediccion (ha)")
    axes[1].set_ylabel("Residuo (ha)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"residuals_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names: list, name: str, output_dir: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.info("  %s no tiene feature_importances_", name)
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(20), importances[indices][:20][::-1], align="center")
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names[i] for i in indices[:20][::-1]])
    ax.set_xlabel("Importancia")
    ax.set_title(f"{name} - Top 20 variables mas importantes")
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_importance_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def optimize_xgboost(X_train, y_train, target_transformer, n_trials=30):
    logger.info("--- Optimizacion de hiperparametros con Optuna ---")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_tr = model.predict(X_tr)
        y_pred_val = model.predict(X_val)
        y_tr_orig = target_transformer.inverse_transform(y_tr.reshape(-1, 1)).ravel()
        y_val_orig = target_transformer.inverse_transform(y_val.reshape(-1, 1)).ravel()
        y_pred_tr_orig = target_transformer.inverse_transform(y_pred_tr.reshape(-1, 1)).ravel()
        y_pred_val_orig = target_transformer.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()

        r2_tr = r2_score(y_tr_orig, y_pred_tr_orig)
        r2_val = r2_score(y_val_orig, y_pred_val_orig)
        overfitting = max(0, r2_tr - r2_val - 0.05)

        return r2_val - 2.0 * overfitting

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info("  Mejores hiperparametros encontrados:")
    for k, v in study.best_params.items():
        logger.info("    %s: %s", k, v)
    logger.info("  Mejor R2 en validacion: %.4f", study.best_value)

    return study.best_params


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modelo guardado en %s", path)


def run_training(
    X_train, X_test, y_train, y_test,
    preprocessor, target_transformer, feature_names,
):
    output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ENTRENAMIENTO DE MODELOS")
    logger.info("=" * 60)

    models = {}
    all_metrics = {}

    lr = train_linear_regression(X_train, y_train)
    lr_metrics, lr_y_pred, lr_y_test = evaluate_model(
        lr, X_train, X_test, y_train, y_test, target_transformer, "Linear Regression"
    )
    models["lr"] = lr
    all_metrics["Linear Regression"] = lr_metrics
    plot_predictions(lr_y_test, lr_y_pred, "Linear Regression", output_dir)
    plot_residuals(lr_y_test, lr_y_pred, "Linear Regression", output_dir)

    rf = train_random_forest(X_train, y_train)
    rf_metrics, rf_y_pred, rf_y_test = evaluate_model(
        rf, X_train, X_test, y_train, y_test, target_transformer, "Random Forest"
    )
    models["rf"] = rf
    all_metrics["Random Forest"] = rf_metrics
    plot_predictions(rf_y_test, rf_y_pred, "Random Forest", output_dir)
    plot_residuals(rf_y_test, rf_y_pred, "Random Forest", output_dir)
    plot_feature_importance(rf, feature_names, "Random Forest", output_dir)

    best_xgb_params = optimize_xgboost(X_train, y_train, target_transformer, n_trials=30)

    xgb_params = {**best_xgb_params, "early_stopping_rounds": 15, "random_state": RANDOM_STATE, "verbosity": 0}
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_metrics, xgb_y_pred, xgb_y_test = evaluate_model(
        xgb, X_train, X_test, y_train, y_test, target_transformer, "XGBoost (Optuna)"
    )
    models["xgb"] = xgb
    all_metrics["XGBoost (Optuna)"] = xgb_metrics
    plot_predictions(xgb_y_test, xgb_y_pred, "XGBoost_Optuna", output_dir)
    plot_residuals(xgb_y_test, xgb_y_pred, "XGBoost_Optuna", output_dir)
    plot_feature_importance(xgb, feature_names, "XGBoost_Optuna", output_dir)

    lgbm = train_lightgbm(X_train, y_train)
    lgbm_metrics, lgbm_y_pred, lgbm_y_test = evaluate_model(
        lgbm, X_train, X_test, y_train, y_test, target_transformer, "LightGBM"
    )
    models["lgbm"] = lgbm
    all_metrics["LightGBM"] = lgbm_metrics
    plot_predictions(lgbm_y_test, lgbm_y_pred, "LightGBM", output_dir)
    plot_residuals(lgbm_y_test, lgbm_y_pred, "LightGBM", output_dir)

    logger.info("  --- Validacion Cruzada (K-Fold, n=5) ---")
    cross_validate_kfold(
        lambda Xtr, Ytr: XGBRegressor(
            **best_xgb_params, random_state=RANDOM_STATE, verbosity=0,
        ).fit(Xtr, Ytr),
        np.vstack([X_train, X_test]),
        np.hstack([y_train, y_test]),
        n_splits=5,
    )

    logger.info("=" * 70)
    logger.info("  COMPARACION DE MODELOS (TEST)")
    logger.info("=" * 70)
    model_names = list(all_metrics.keys())
    header = f"  {'Metrica':<10}"
    for n in model_names:
        header += f" {n:<20}"
    logger.info(header)
    logger.info("  %s", "-" * (10 + 21 * len(model_names)))

    for m in METRICS:
        row = f"  {m:<10}"
        for n in model_names:
            row += f" {all_metrics[n]['test'][m]:<20.4f}"
        logger.info(row)

    logger.info("")
    logger.info("  Modelos que cumplen overfitting < 5%%:")
    valid_models = {}
    for n in model_names:
        overfit = all_metrics[n]["train"]["R2"] - all_metrics[n]["test"]["R2"]
        ok = "[OK]" if overfit < 0.05 else "[NO]"
        logger.info("    %-22s: over=%.4f R2=%.4f %s", n, overfit, all_metrics[n]["test"]["R2"], ok)
        if overfit < 0.05:
            valid_models[n] = all_metrics[n]["test"]["R2"]

    if valid_models:
        best_model_name = max(valid_models, key=valid_models.get)
        logger.info("")
        logger.info("  Mejor modelo (overfitting < 5%%): %s (R2=%.4f)", best_model_name, valid_models[best_model_name])
    else:
        best_model_name = max(model_names, key=lambda n: all_metrics[n]["test"]["R2"])
        logger.info("")
        logger.info("  Ningun modelo cumple overfitting < 5%%. Usando: %s", best_model_name)

    key_map = {"Linear Regression": "lr", "Random Forest": "rf",
               "XGBoost (Optuna)": "xgb", "LightGBM": "lgbm"}
    final_model = models[key_map[best_model_name]]
    save_model(final_model, MODEL_PATH)

    return final_model, all_metrics
