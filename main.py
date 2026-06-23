#!/usr/bin/env python3
"""Pipeline completo de regresion para prediccion de superficie quemada.

Uso: python main.py
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.config import DATA_PROCESSED, TARGET, RANDOM_STATE
from preprocessing.analysis import load_dataset, generate_exploratory_report, print_report
from preprocessing.preprocessor import build_preprocessor, prepare_data, preprocess_dataframe
from training.trainer import (
    split_data,
    train_linear_regression,
    train_random_forest,
    train_hist_gradient_boosting,
    save_model,
)
from training.optimizer import optimize_random_forest, optimize_hist_gradient_boosting
from evaluation.metrics import (
    evaluate_model,
    cross_validate_model,
    compare_models,
)
from visualization.plots import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_scatter_target_vs_features,
    plot_prediction_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_model_comparison,
    plot_overfitting_comparison,
    plot_cv_results,
)
from visualization.reports import (
    generate_preprocessing_report,
    generate_final_report,
    save_report,
)

warnings.filterwarnings("ignore")


def detect_year_column(columns: list) -> str:
    for col in columns:
        col_clean = col.replace("\xf1", "n").lower()
        if "a" in col_clean and ("no" in col_clean or "os" in col_clean):
            return col
    return "a\xf1o"


def main():
    print("=" * 70)
    print("PIPELINE DE REGRESION - Prediccion de Superficie Quemada")
    print("=" * 70)

    # 1. Carga del dataset
    print("\n[1/10] Cargando dataset...")
    df = load_dataset()
    year_col = detect_year_column(df.columns)
    print(f"  Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"  Columna anyo detectada: '{year_col}'")

    # 2. Analisis exploratorio
    print("\n[2/10] Generando informe exploratorio...")
    report = generate_exploratory_report(df)
    print_report(report)
    report_md = generate_preprocessing_report(report)
    save_report(report_md, "informe_preprocesamiento")
    print("  Informe guardado en reports/informe_preprocesamiento.md")

    # 3. Convertir a pandas para sklearn
    print("\n[3/10] Preparando datos para modelado...")
    df_pd = preprocess_dataframe(df.to_pandas(), year_col=year_col)

    use_log1p = True
    df_pd[TARGET] = np.log1p(df_pd[TARGET])

    X, y = prepare_data(df_pd)
    print(f"  Features: {X.shape[1]}, Target: log1p(superficie_quemada)")

    # 4. Visualizaciones exploratorias
    print("\n[4/10] Generando visualizaciones exploratorias...")
    plot_target_distribution(df[TARGET].to_numpy(), log_transform=False)
    plot_target_distribution(np.expm1(y.values), log_transform=True)
    plot_correlation_heatmap(df_pd)
    plot_scatter_target_vs_features(X, np.expm1(y.values))
    print("  Visualizaciones guardadas en figures/")

    # 5. Train/Test split
    print("\n[5/10] Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 6. Entrenar modelos
    print("\n[6/10] Entrenando modelos...")

    model_results = {}
    cv_results = {}

    # --- LinearRegression
    print("\n  --- LinearRegression (baseline) ---")
    lr_pipeline, lr_time = train_linear_regression(X_train, y_train, year_col=year_col)
    lr_eval = evaluate_model(lr_pipeline, X_train, y_train, X_test, y_test,
                              use_log1p=use_log1p)
    model_results["LinearRegression"] = lr_eval
    print(f"    R2 test: {lr_eval['test']['R2']:.4f} | Overfitting: {lr_eval['overfitting']:.4f} | Tiempo: {lr_time:.2f}s")
    save_model(lr_pipeline, "linear_regression")

    def make_lr():
        p = build_preprocessor(year_col=year_col)
        return Pipeline([("preprocessor", p), ("model", LinearRegression())])

    lr_cv = cross_validate_model(make_lr, X_train, y_train, target_is_log1p=use_log1p)
    cv_results["LinearRegression"] = lr_cv
    print(f"    CV R2: {lr_cv['R2_mean']:.4f} +/- {lr_cv['R2_std']:.4f}")

    # --- RandomForest
    print("\n  --- RandomForestRegressor ---")
    rf_pipeline, rf_time = train_random_forest(X_train, y_train, year_col=year_col)
    rf_eval = evaluate_model(rf_pipeline, X_train, y_train, X_test, y_test,
                              use_log1p=use_log1p)
    model_results["RandomForest"] = rf_eval
    print(f"    R2 test: {rf_eval['test']['R2']:.4f} | Overfitting: {rf_eval['overfitting']:.4f} | Tiempo: {rf_time:.2f}s")
    save_model(rf_pipeline, "random_forest")

    def make_rf():
        p = build_preprocessor(year_col=year_col)
        return Pipeline([("preprocessor", p), ("model", RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1))])
    rf_cv = cross_validate_model(make_rf, X_train, y_train, target_is_log1p=use_log1p)
    cv_results["RandomForest"] = rf_cv
    print(f"    CV R2: {rf_cv['R2_mean']:.4f} +/- {rf_cv['R2_std']:.4f}")

    # --- HistGradientBoosting
    print("\n  --- HistGradientBoostingRegressor ---")
    hgb_pipeline, hgb_time = train_hist_gradient_boosting(X_train, y_train, year_col=year_col)
    hgb_eval = evaluate_model(hgb_pipeline, X_train, y_train, X_test, y_test,
                               use_log1p=use_log1p)
    model_results["HistGB"] = hgb_eval
    print(f"    R2 test: {hgb_eval['test']['R2']:.4f} | Overfitting: {hgb_eval['overfitting']:.4f} | Tiempo: {hgb_time:.2f}s")
    save_model(hgb_pipeline, "hist_gradient_boosting")

    def make_hgb():
        p = build_preprocessor(year_col=year_col)
        return Pipeline([("preprocessor", p), ("model", HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.1, min_samples_leaf=20,
            random_state=RANDOM_STATE))])
    hgb_cv = cross_validate_model(make_hgb, X_train, y_train, target_is_log1p=use_log1p)
    cv_results["HistGB"] = hgb_cv
    print(f"    CV R2: {hgb_cv['R2_mean']:.4f} +/- {hgb_cv['R2_std']:.4f}")

    # 7. Comparacion de modelos
    print("\n[7/10] Comparando modelos...")
    best_model_name = compare_models(model_results)
    print(f"  Mejor modelo (R2 test): {best_model_name}")

    plot_model_comparison(model_results)
    plot_overfitting_comparison(model_results)
    plot_cv_results(cv_results)

    # 8. Feature importance
    print("\n[8/10] Analizando importancia de features...")
    best_pipeline = {
        "LinearRegression": lr_pipeline,
        "RandomForest": rf_pipeline,
        "HistGB": hgb_pipeline,
    }[best_model_name]

    try:
        preprocessor = build_preprocessor(year_col=year_col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocessor.fit(X_train, y_train)
        cols_out = preprocessor.named_steps["preprocessor"].get_feature_names_out()
        plot_feature_importance(best_pipeline, list(cols_out), title=best_model_name)
        print("  Feature importance guardada en figures/")
    except Exception as e:
        print(f"  Nota: No se pudo generar feature importance: {e}")

    # 9. Diagnostic plots del mejor modelo
    print("\n[9/10] Generando graficos de diagnostico...")
    best_eval = model_results[best_model_name]
    plot_prediction_vs_actual(
        best_eval["y_true_test"], best_eval["y_pred_test"],
        title=best_model_name,
    )
    plot_residuals(
        best_eval["y_true_test"], best_eval["y_pred_test"],
        title=best_model_name,
    )
    print("  Graficos de diagnostico guardados en figures/")

    # 10. Optimizacion del mejor modelo
    print(f"\n[10/10] Optimizando {best_model_name} con RandomizedSearchCV...")
    optimization_results = {}
    try:
        if best_model_name == "RandomForest":
            opt_pipeline, best_params, opt_time = optimize_random_forest(
                X_train, y_train, n_iter=10, year_col=year_col
            )
            opt_eval = evaluate_model(
                opt_pipeline, X_train, y_train, X_test, y_test,
                use_log1p=use_log1p,
            )
            model_results["RandomForest_Optimized"] = opt_eval
            save_model(opt_pipeline, "random_forest_optimized")
            optimization_results = {
                "model_name": "RandomForest_Optimized",
                "best_params": best_params,
                "test_r2": opt_eval["test"]["R2"],
                "test_mae": opt_eval["test"]["MAE"],
                "test_rmse": opt_eval["test"]["RMSE"],
                "train_r2": opt_eval["train"]["R2"],
                "overfitting": opt_eval["overfitting"],
            }
        elif best_model_name == "HistGB":
            opt_pipeline, best_params, opt_time = optimize_hist_gradient_boosting(
                X_train, y_train, n_iter=10, year_col=year_col
            )
            opt_eval = evaluate_model(
                opt_pipeline, X_train, y_train, X_test, y_test,
                use_log1p=use_log1p,
            )
            model_results["HistGB_Optimized"] = opt_eval
            save_model(opt_pipeline, "hist_gb_optimized")
            optimization_results = {
                "model_name": "HistGB_Optimized",
                "best_params": best_params,
                "test_r2": opt_eval["test"]["R2"],
                "test_mae": opt_eval["test"]["MAE"],
                "test_rmse": opt_eval["test"]["RMSE"],
                "train_r2": opt_eval["train"]["R2"],
                "overfitting": opt_eval["overfitting"],
            }

        if optimization_results:
            print(f"  R2 test (optimizado): {optimization_results['test_r2']:.4f}")
            print(f"  Mejores parametros: {optimization_results['best_params']}")
            plot_prediction_vs_actual(
                opt_eval["y_true_test"], opt_eval["y_pred_test"],
                title=f"{best_model_name}_Optimized",
            )
            plot_residuals(
                opt_eval["y_true_test"], opt_eval["y_pred_test"],
                title=f"{best_model_name}_Optimized",
            )
    except Exception as e:
        print(f"  Nota: Error en optimizacion ({e}), continuando sin optimizar...")

    # Informe final
    final_report = generate_final_report(
        model_results, cv_results, best_model_name, optimization_results,
    )
    save_report(final_report, "informe_final_rendimiento")
    print("\n  Informe final guardado en reports/informe_final_rendimiento.md")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO")
    print("=" * 70)
    print(f"\nResumen:")
    print(f"  Mejor modelo: {best_model_name}")
    print(f"  R2 test: {model_results[best_model_name]['test']['R2']:.4f}")
    print(f"  MAE test: {model_results[best_model_name]['test']['MAE']:.2f} ha")
    print(f"  RMSE test: {model_results[best_model_name]['test']['RMSE']:.2f} ha")
    print(f"  Overfitting: {model_results[best_model_name]['overfitting']:.4f}")

    if optimization_results:
        opt_name = optimization_results["model_name"]
        print(f"\n  Modelo optimizado: {opt_name}")
        print(f"  R2 test (optimizado): {optimization_results['test_r2']:.4f}")
        print(f"  MAE test (optimizado): {optimization_results['test_mae']:.2f} ha")
        print(f"  RMSE test (optimizado): {optimization_results['test_rmse']:.2f} ha")

    print(f"\n  Resultados guardados en:")
    print(f"    - figures/ (graficos)")
    print(f"    - reports/ (informes)")
    print(f"    - models/ (modelos serializados)")


if __name__ == "__main__":
    main()
