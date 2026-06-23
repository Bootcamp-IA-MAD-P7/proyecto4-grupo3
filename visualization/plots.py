import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import FIGURES_DIR, TARGET


def _save(name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_target_distribution(y: np.ndarray, log_transform: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data = np.log1p(y) if log_transform else y
    label = "log1p(superficie_quemada)" if log_transform else "superficie_quemada (ha)"

    axes[0].hist(data, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(np.mean(data), color="red", linestyle="--", label=f"Media: {np.mean(data):.2f}")
    axes[0].axvline(np.median(data), color="green", linestyle="--", label=f"Mediana: {np.median(data):.2f}")
    axes[0].set_xlabel(label)
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title(f"Distribución de {label}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[1].set_ylabel(label)
    axes[1].set_title(f"Boxplot de {label}")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    name = "dist_log1p.png" if log_transform else "dist_original.png"
    _save(name)


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric:
        numeric.remove(TARGET)
        numeric.insert(0, TARGET)
    corr = df[numeric].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Mapa de Correlaciones", fontsize=14)
    plt.tight_layout()
    _save("correlation_heatmap.png")


def plot_scatter_target_vs_features(
    X: pd.DataFrame, y: np.ndarray, top_n: int = 6
):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) > top_n:
        corrs = X[numeric].apply(lambda col: abs(col.corr(pd.Series(y, index=X.index))))
        numeric = corrs.sort_values(ascending=False).head(top_n).index.tolist()

    ncols = 3
    nrows = int(np.ceil(len(numeric) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric):
        axes[i].scatter(X[col], y, alpha=0.3, s=5, c="steelblue")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(TARGET)
        axes[i].set_title(f"{col} vs {TARGET}")
        axes[i].grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    _save("scatter_target_vs_features.png")


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, c="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal")
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Predicción")
    ax.set_title(f"Predicción vs Valor Real{ ' - ' + title if title else ''}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(f"pred_vs_actual_{title.replace(' ', '_') if title else 'all'}.png")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = ""):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residuo (Real - Predicción)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title(f"Histograma de Residuos{ ' - ' + title if title else ''}")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, c="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Residuo")
    axes[1].set_title("Residuos vs Predicción")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(f"residuals_{title.replace(' ', '_') if title else 'all'}.png")


def plot_feature_importance(
    pipeline, feature_names: list[str], title: str = "", top_n: int = 20
):
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_.flatten())
    else:
        return

    n_features = min(len(importances), len(feature_names), top_n)
    indices = np.argsort(importances)[-n_features:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(n_features), importances[indices], color="steelblue", alpha=0.8)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importancia")
    ax.set_title(f"Feature Importance{ ' - ' + title if title else ''}")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    _save(f"feature_importance_{title.replace(' ', '_') if title else 'all'}.png")

    return importances, indices


def plot_model_comparison(results: dict[str, dict]):
    names = list(results.keys())
    metrics_to_plot = {
        "MAE": "MAE (ha)",
        "RMSE": "RMSE (ha)",
        "R2": "R²",
    }

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot.items()):
        test_values = []
        train_values = []
        for name in names:
            result = results[name]
            test_metrics = result.get("test", {})
            train_metrics = result.get("train", {})
            if metric_key in test_metrics:
                test_values.append(test_metrics[metric_key])
                train_values.append(train_metrics.get(metric_key, 0))

        x = np.arange(len(names))
        width = 0.35
        axes[idx].bar(x - width / 2, train_values, width, label="Train", alpha=0.8)
        axes[idx].bar(x + width / 2, test_values, width, label="Test", alpha=0.8)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(names, rotation=45, ha="right")
        axes[idx].set_ylabel(metric_label)
        axes[idx].set_title(f"Comparación: {metric_label}")
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    _save("model_comparison.png")


def plot_overfitting_comparison(results: dict[str, dict]):
    names = list(results.keys())
    overfitting_values = []
    test_r2_values = []

    for name in names:
        result = results[name]
        overfitting_values.append(result.get("overfitting", 0))
        test_r2_values.append(result.get("test", {}).get("R2", 0))

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, test_r2_values, width, label="R² Test", alpha=0.8)
    bars2 = ax.bar(x + width / 2, overfitting_values, width, label="Overfitting (R² diff)", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de R² Test y Overfitting")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars1, test_r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, overfitting_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save("overfitting_comparison.png")


def plot_cv_results(cv_results: dict[str, dict]):
    names = list(cv_results.keys())
    r2_means = [cv_results[n]["R2_mean"] for n in names]
    r2_stds = [cv_results[n]["R2_std"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, r2_means, yerr=r2_stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("R² medio (CV)")
    ax.set_title("Validación Cruzada (KFold 5) - R² medio")
    ax.grid(alpha=0.3, axis="y")

    for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
        ax.text(i, mean + std + 0.01, f"{mean:.3f}±{std:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save("cv_results.png")
