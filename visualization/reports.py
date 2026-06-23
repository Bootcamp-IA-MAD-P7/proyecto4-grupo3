from pathlib import Path
from datetime import datetime
import json

from src.config import REPORTS_DIR


def generate_preprocessing_report(analysis_report: dict) -> str:
    lines = []
    lines.append("# Informe de Preprocesamiento y Análisis Exploratorio")
    lines.append(f"\nGenerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---")

    shape = analysis_report["shape"]
    lines.append(f"\n## 1. Dimensiones del Dataset")
    lines.append(f"- Filas: {shape['rows']:,}")
    lines.append(f"- Columnas: {shape['columns']}")

    lines.append("\n## 2. Variable Objetivo: superficie_quemada")
    ts = analysis_report["target_stats"]
    lines.append(f"- Media: {ts['mean']:.4f} ha")
    lines.append(f"- Desviación Estándar: {ts['std']:.4f} ha")
    lines.append(f"- Mínimo: {ts['min']:.4f} ha")
    lines.append(f"- Máximo: {ts['max']:.4f} ha")
    lines.append(f"- Skewness: {ts['skewness']:.4f}")
    lines.append(f"- Kurtosis: {ts['kurtosis']:.4f}")
    lines.append(f"- Ceros: {ts['zeros']} ({ts['zeros_pct']:.2f}%)")
    lines.append(f"- Positivos: {ts['positive']}")
    lines.append("\n**Decisión:** Aplicar transformación log1p por skewness extremo.")

    lines.append("\n## 3. Calidad de Datos")
    for issue in analysis_report["data_quality"]:
        lines.append(f"\n### [{issue['severity'].upper()}] {issue['type']}")
        lines.append(f"- {issue['detail']}")
        lines.append(f"- **Acción:** {issue['action']}")

    lines.append("\n## 4. Coherencia Meteorológica")
    for issue in analysis_report["weather_coherence"]:
        lines.append(f"\n### [{issue['severity'].upper()}] {issue['type']}")
        lines.append(f"- {issue['detail']}")
        lines.append(f"- **Acción:** {issue['action']}")

    lines.append("\n## 5. Columnas Eliminadas")
    lines.append("| Columna | Motivo |")
    lines.append("|---------|--------|")
    lines.append("| id | Identificador único sin valor predictivo |")
    lines.append("| fecha | Duplicada de fecha_incendio |")
    lines.append("| fecha_incendio | Fecha del incendio (usamos features derivadas: año, mes) |")
    lines.append("| month_name | Redundante con mes (numérico) |")
    lines.append("| trimestre | Redundante con mes (derivable) |")

    lines.append("\n## 6. Variables Categóricas")
    for col, card in sorted(analysis_report["cardinality"].items()):
        lines.append(f"- **{col}**: {card} valores únicos")
    lines.append("\n**Decisión:**")
    lines.append("- provincia (39): TargetEncoding")
    lines.append("- cc_aa (17): TargetEncoding")
    lines.append("- causa_incendio (6): OneHotEncoding")
    lines.append("- mes (12): Codificación cíclica (seno/coseno)")

    lines.append("\n## 7. Correlaciones con Variable Objetivo (Top 10)")
    for item in analysis_report["correlations"][:10]:
        lines.append(f"- {item['feature']}: {item['correlation']:.4f}")

    lines.append("\n## 8. Transformaciones Aplicadas")
    lines.append("- **Target**: log1p(superficie_quemada)")
    lines.append("- **Numéricas**: Imputación (mediana) + StandardScaler")
    lines.append("- **Geo**: latitud/longitud convertidas a float")
    lines.append("- **Categóricas baja cardinalidad**: OneHotEncoding")
    lines.append("- **Categóricas alta cardinalidad**: TargetEncoding")
    lines.append("- **Cíclicas**: mes → sin(mes), cos(mes)")

    report = "\n".join(lines)
    return report


def generate_final_report(
    model_results: dict, cv_results: dict,
    best_model_name: str, optimization_results: dict = None,
) -> str:
    lines = []
    lines.append("# Informe Final de Rendimiento")
    lines.append(f"\nGenerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---")

    lines.append("\n## 1. Comparación de Modelos")

    header = "| Modelo | R² Train | R² Test | MAE Test | RMSE Test | Overfitting |"
    sep = "|--------|----------|---------|----------|-----------|-------------|"
    lines.append(header)
    lines.append(sep)

    for name, result in model_results.items():
        train_r2 = result["train"]["R2"]
        test_r2 = result["test"]["R2"]
        mae = result["test"]["MAE"]
        rmse = result["test"]["RMSE"]
        overfit = result["overfitting"]
        lines.append(
            f"| {name} | {train_r2:.4f} | {test_r2:.4f} | "
            f"{mae:.2f} | {rmse:.2f} | {overfit:.4f} |"
        )

    lines.append(f"\n**Mejor modelo (por R² Test):** {best_model_name}")

    if cv_results:
        lines.append("\n## 2. Validación Cruzada (KFold 5)")
        cv_header = "| Modelo | R² medio | R² std | MAE medio | RMSE medio |"
        cv_sep = "|--------|----------|--------|-----------|------------|"
        lines.append(cv_header)
        lines.append(cv_sep)
        for name, result in cv_results.items():
            lines.append(
                f"| {name} | {result['R2_mean']:.4f} | {result['R2_std']:.4f} | "
                f"{result['MAE_mean']:.2f} | {result['RMSE_mean']:.2f} |"
            )

    if optimization_results:
        lines.append("\n## 3. Optimización de Hiperparámetros")
        lines.append(f"\n**Modelo optimizado:** {optimization_results.get('model_name', best_model_name)}")
        lines.append("\n**Mejores hiperparámetros:**")
        lines.append("```")
        for param, value in optimization_results.get("best_params", {}).items():
            lines.append(f"  {param}: {value}")
        lines.append("```")
        lines.append(f"\n**R² en test (modelo óptimo):** {optimization_results.get('test_r2', 'N/A'):.4f}")
        lines.append(f"**MAE en test (modelo óptimo):** {optimization_results.get('test_mae', 'N/A'):.2f}")
        lines.append(f"**RMSE en test (modelo óptimo):** {optimization_results.get('test_rmse', 'N/A'):.2f}")

    lines.append("\n## 4. Conclusiones")
    lines.append("- Las correlaciones entre features meteorológicos y superficie quemada son muy bajas (< 0.07).")
    lines.append("- La tarea de regresión es extremadamente ruidosa: el área quemada depende de muchos factores no capturados (tipo de vegetación, topografía, respuesta de extinción, etc.).")
    lines.append("- La transformación log1p es esencial debido a la distribución heavy-tailed de la variable objetivo.")
    lines.append("- Los modelos tree-based (RandomForest, HistGradientBoosting) superan significativamente a la regresión lineal.")
    lines.append("- Se recomienda incorporar features adicionales (tipo de vegetación, índice de sequía, velocidad de propagación) para mejorar el poder predictivo.")

    return "\n".join(lines)


def save_report(content: str, name: str) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)
