import polars as pl
import numpy as np
from src.config import (
    DATA_PROCESSED,
    TARGET,
    NUM_COLS,
    CAT_OHE,
    CAT_TARGET_ENCODE,
    GEO_COLS,
    CYCLIC_COLS,
    DROP_COLS,
)


def load_dataset() -> pl.DataFrame:
    df = pl.read_parquet(DATA_PROCESSED)
    return df


def get_year_column(df: pl.DataFrame) -> str:
    for col in df.columns:
        if "a" in col.lower() and ("o" in col.lower().replace("\xf1", "n")):
            return col
    return "a\xf1o"


def compute_nulls(df: pl.DataFrame) -> pl.DataFrame:
    total = df.height
    return (
        df.select([pl.col(c).is_null().sum().alias(c) for c in df.columns])
        .melt(value_name="null_count")
        .with_columns(
            [
                (pl.col("null_count") / total * 100).alias("null_pct"),
            ]
        )
    )


def compute_cardinality(df: pl.DataFrame) -> dict:
    return {
        c: df[c].n_unique()
        for c in df.columns
        if df[c].dtype in (pl.String, pl.Utf8, pl.Int64, pl.Int32)
    }


def compute_target_stats(df: pl.DataFrame) -> dict:
    y = df[TARGET]
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    return {
        "count": y.len(),
        "mean": y.mean(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "skewness": y.skew(),
        "kurtosis": y.kurtosis(),
        "zeros": (y == 0).sum(),
        "zeros_pct": (y == 0).sum() / y.len() * 100,
        "positive": (y > 0).sum(),
        "percentiles": {q: y.quantile(q) for q in quantiles},
    }


def compute_correlations(df: pl.DataFrame) -> list[dict]:
    numeric = [
        c
        for c, d in df.schema.items()
        if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8) and c != "id"
    ]
    results = []
    for col in numeric:
        corr = df.select(pl.corr(col, TARGET)).item()
        results.append({"feature": col, "correlation": corr})
    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


def compute_group_stats(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    return (
        df.group_by(group_col)
        .agg(
            [
                pl.len().alias("count"),
                pl.col(TARGET).mean().alias(f"mean_{TARGET}"),
                pl.col(TARGET).std().alias(f"std_{TARGET}"),
                pl.col(TARGET).median().alias(f"median_{TARGET}"),
            ]
        )
        .sort("count", descending=True)
    )


def check_data_quality(df: pl.DataFrame) -> list[dict]:
    issues = []

    if df["fecha"].equals(df["fecha_incendio"]):
        issues.append(
            {
                "type": "redundancy",
                "severity": "high",
                "detail": "fecha and fecha_incendio are identical (columnas duplicadas)",
                "action": "Eliminar fecha o fecha_incendio",
            }
        )

    if df["id"].n_unique() == df.height:
        issues.append(
            {
                "type": "identifier",
                "severity": "high",
                "detail": "id is a unique row identifier (no aporta valor predictivo)",
                "action": "Eliminar columna id",
            }
        )

    for col in ["latitud", "longitud"]:
        if df[col].dtype == pl.String:
            sample = df[col].head(5).to_list()
            issues.append(
                {
                    "type": "wrong_dtype",
                    "severity": "medium",
                    "detail": f"{col} es String, debería ser Float. Sample: {sample}",
                    "action": "Convertir a Float64",
                }
            )

    target = df[TARGET]
    skew = target.skew()
    if abs(skew) > 1:
        issues.append(
            {
                "type": "high_skew",
                "severity": "high",
                "detail": f"superficie_quemada tiene skewness={skew:.2f} (extremadamente asimétrica)",
                "action": "Aplicar transformación log1p",
            }
        )

    issues.append(
        {
            "type": "correlation_note",
            "severity": "info",
            "detail": "Todas las correlaciones con el target son muy bajas (<0.07). "
            "La superficie quemada puede ser difícil de predecir solo con datos meteorológicos.",
            "action": "Documentar limitación en el informe",
        }
    )

    return issues


def check_weather_coherence(df: pl.DataFrame) -> list[dict]:
    issues = []

    neg_precip = df.filter(pl.col("precipitacion") < 0).height
    if neg_precip > 0:
        issues.append(
            {
                "type": "negative_values",
                "severity": "high",
                "detail": f"{neg_precip} filas con precipitación negativa",
                "action": "Revisar calidad de datos de precipitación",
            }
        )

    return issues


def generate_exploratory_report(df: pl.DataFrame) -> dict:
    report = {
        "shape": {"rows": df.height, "columns": len(df.columns)},
        "columns": df.columns,
        "dtypes": {c: str(d) for c, d in df.schema.items()},
        "nulls": compute_nulls(df),
        "cardinality": compute_cardinality(df),
        "target_stats": compute_target_stats(df),
        "correlations": compute_correlations(df),
        "fires_by_provincia": compute_group_stats(df, "provincia"),
        "fires_by_ccaa": compute_group_stats(df, "cc_aa"),
        "fires_by_cause": compute_group_stats(df, "causa_incendio"),
        "data_quality": check_data_quality(df),
        "weather_coherence": check_weather_coherence(df),
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 70)
    print("INFORME EXPLORATORIO DEL DATASET")
    print("=" * 70)

    shape = report["shape"]
    print(f"\nDimensiones: {shape['rows']} filas, {shape['columns']} columnas")
    print(f"Columnas: {report['columns']}")
    print(f"\nTipos de datos:")
    for col, dtype in report["dtypes"].items():
        print(f"  {col}: {dtype}")

    nulls = report["nulls"]
    has_nulls = nulls.filter(pl.col("null_count") > 0)
    if has_nulls.height > 0:
        print(f"\nValores nulos:")
        for row in has_nulls.iter_rows():
            print(f"  {row[0]}: {row[1]} ({row[2]:.2f}%)")
    else:
        print(f"\nValores nulos: Ninguno (dataset completamente limpio)")

    print(f"\nCardinalidad de variables categóricas:")
    for col, card in sorted(report["cardinality"].items()):
        print(f"  {col}: {card} valores únicos")

    ts = report["target_stats"]
    print(f"\nEstadísticas de superficie_quemada:")
    print(f"  Media: {ts['mean']:.4f}")
    print(f"  Desv.Est.: {ts['std']:.4f}")
    print(f"  Mínimo: {ts['min']:.4f}")
    print(f"  Máximo: {ts['max']:.4f}")
    print(f"  Skewness: {ts['skewness']:.4f}")
    print(f"  Kurtosis: {ts['kurtosis']:.4f}")
    print(f"  Ceros: {ts['zeros']} ({ts['zeros_pct']:.2f}%)")
    print(f"  Positivos: {ts['positive']}")
    print(f"  Percentiles:")
    for q, v in ts["percentiles"].items():
        print(f"    p{q}: {v:.4f}")

    print(f"\nCorrelaciones con superficie_quemada (top 5):")
    for item in report["correlations"][:5]:
        print(f"  {item['feature']}: {item['correlation']:.4f}")

    print(f"\nIncendios por provincia (top 5):")
    for row in report["fires_by_provincia"].head(5).iter_rows():
        print(f"  {row[0]}: {row[1]} incendios, media {row[2]:.2f} ha")

    print(f"\nIncendios por CC.AA. (top 5):")
    for row in report["fires_by_ccaa"].head(5).iter_rows():
        print(f"  {row[0]}: {row[1]} incendios, media {row[2]:.2f} ha")

    print(f"\nSuperficie quemada media por causa:")
    for row in report["fires_by_cause"].iter_rows():
        print(f"  Causa {row[0]}: {row[1]} incendios, media {row[2]:.2f} ha")

    print(f"\nProblemas de calidad de datos:")
    for issue in report["data_quality"]:
        print(f"  [{issue['severity'].upper()}] {issue['detail']}")
        print(f"    -> Acción: {issue['action']}")

    print(f"\nCoherencia meteorológica:")
    for issue in report["weather_coherence"]:
        print(f"  [{issue['severity'].upper()}] {issue['detail']}")
        print(f"    -> Acción: {issue['action']}")

    print("=" * 70)

    return report
