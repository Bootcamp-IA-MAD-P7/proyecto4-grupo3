"""Merge AEMET weather data with fire incidents.

Pipeline:
  1. Load unified AEMET CSV → clean → group by (fecha, provincia) averaging weather
  2. Load processed fires (fires-all-prepro.parquet)
  3. Inner join on (fecha, provincia) → each fire gets that day's weather
  4. Save merged dataset

Usage:
  python scripts/merge_datasets.py
"""

import polars as pl
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
CLEANED = Path("data/cleaned")

AEMET_FILE = RAW / "aemet_all.csv"
FIRES_PATH = CLEANED / "fires_clean.parquet"
OUTPUT_PATH = PROCESSED / "fires_weather_merged.parquet"

COLUMNS_KEEP = [
    "temperatura_media",
    "precipitacion",
    "humedad_relativa_media",
    "velocidad_viento_media",
    "racha_maxima_viento",
    "sol",
    "dir",
]

provincia_to_ccaa = {
    "ALMERIA": "ANDALUCIA",
    "CADIZ": "ANDALUCIA",
    "CORDOBA": "ANDALUCIA",
    "GRANADA": "ANDALUCIA",
    "HUELVA": "ANDALUCIA",
    "JAEN": "ANDALUCIA",
    "MALAGA": "ANDALUCIA",
    "SEVILLA": "ANDALUCIA",
    "HUESCA": "ARAGON",
    "TERUEL": "ARAGON",
    "ZARAGOZA": "ARAGON",
    "ASTURIAS": "ASTURIAS",
    "BALEARES": "BALEARES",
    "ILLES BALEARS": "BALEARES",
    "LAS PALMAS": "CANARIAS",
    "STA. CRUZ DE TENERIFE": "CANARIAS",
    "SANTA CRUZ DE TENERIFE": "CANARIAS",
    "CANTABRIA": "CANTABRIA",
    "SANTANDER": "CANTABRIA",
    "ALBACETE": "CASTILLA-LA MANCHA",
    "CIUDAD REAL": "CASTILLA-LA MANCHA",
    "CUENCA": "CASTILLA-LA MANCHA",
    "GUADALAJARA": "CASTILLA-LA MANCHA",
    "TOLEDO": "CASTILLA-LA MANCHA",
    "AVILA": "CASTILLA Y LEON",
    "BURGOS": "CASTILLA Y LEON",
    "LEON": "CASTILLA Y LEON",
    "PALENCIA": "CASTILLA Y LEON",
    "SALAMANCA": "CASTILLA Y LEON",
    "SEGOVIA": "CASTILLA Y LEON",
    "SORIA": "CASTILLA Y LEON",
    "VALLADOLID": "CASTILLA Y LEON",
    "ZAMORA": "CASTILLA Y LEON",
    "BARCELONA": "CATALUNA",
    "GIRONA": "CATALUNA",
    "LLEIDA": "CATALUNA",
    "TARRAGONA": "CATALUNA",
    "BADAJOZ": "EXTREMADURA",
    "CACERES": "EXTREMADURA",
    "A CORUÑA": "GALICIA",
    "LUGO": "GALICIA",
    "OURENSE": "GALICIA",
    "PONTEVEDRA": "GALICIA",
    "LA RIOJA": "LA RIOJA",
    "MADRID": "MADRID",
    "MURCIA": "MURCIA",
    "NAVARRA": "NAVARRA",
    "ALAVA": "PAIS VASCO",
    "ARABA/ALAVA": "PAIS VASCO",
    "GUIPUZCOA": "PAIS VASCO",
    "GIPUZKOA": "PAIS VASCO",
    "BIZKAIA": "PAIS VASCO",
    "VIZCAYA": "PAIS VASCO",
    "ALICANTE": "VALENCIA",
    "CASTELLON": "VALENCIA",
    "VALENCIA": "VALENCIA",
    "CEUTA": "CEUTA",
    "MELILLA": "MELILLA",
}


def clean_aemet(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise and clean the unified AEMET DataFrame."""
    df = df.drop([
        "indicativo", "nombre", "horatmin", "horatmax", "horaHrMin",
        "horaracha", "horaPresMax", "horaPresMin",
        "horaHrMax", "presMax", "presMin",
    ])

    df = df.with_columns(
        pl.col("provincia")
        .str.to_uppercase()
        .replace_strict(provincia_to_ccaa, default=None)
        .alias("cc_aa")
    )

    df = df.rename({
        "tmed": "temperatura_media",
        "prec": "precipitacion",
        "hrMedia": "humedad_relativa_media",
        "velmedia": "velocidad_viento_media",
        "racha": "racha_maxima_viento",
        "tmin": "temperatura_minima",
        "tmax": "temperatura_maxima",
        "hrMax": "humedad_relativa_maxima",
        "hrMin": "humedad_relativa_minima",
    })

    df = df.filter(~pl.col("precipitacion").is_in(["Ip", "Acum"]))

    for col in [
        "temperatura_media", "temperatura_minima", "temperatura_maxima",
        "precipitacion", "velocidad_viento_media", "racha_maxima_viento",
        "sol",
    ]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).str.replace(",", ".").cast(pl.Float64)
            )

    for col in ["dir"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int32))

    df = df.with_columns(pl.col("fecha").str.to_date("%Y-%m-%d"))

    core = ["temperatura_media", "precipitacion"]
    df = df.drop_nulls(subset=core)

    return df


def average_per_day(df: pl.DataFrame) -> pl.DataFrame:
    """Average weather values per (fecha, provincia)."""
    weather_cols = [c for c in COLUMNS_KEEP if c in df.columns]
    return df.group_by(["fecha", "provincia"]).agg(
        pl.col("cc_aa").first(),
        pl.col("altitud").mean(),
        *[pl.col(c).mean().alias(c) for c in weather_cols],
    )


def load_fires(path: Path) -> pl.DataFrame:
    """Load processed fires, add fecha and ensure provincia is uppercase."""
    df = pl.read_parquet(path)
    df = df.with_columns(
        pl.col("fecha_incendio").alias("fecha"),
        pl.col("provincia").str.to_uppercase(),
    )
    return df


def main():
    print(f"Loading AEMET from {AEMET_FILE}...")
    df = pl.read_csv(
        AEMET_FILE,
        schema_overrides={"horaPresMin": pl.Utf8},
        null_values=["Varias"],
    )
    print(f"  Raw AEMET: {df.shape} rows")

    print("Cleaning AEMET...")
    df = clean_aemet(df)

    print("Averaging per day...")
    aemet_all = average_per_day(df)
    print(f"  AEMET cleaned: {aemet_all.shape} rows")

    print("Loading fires...")
    fires = load_fires(FIRES_PATH)
    print(f"  Fires: {fires.shape} rows")

    fires = fires.drop("cc_aa")

    print("Merging...")
    merged = aemet_all.join(
        fires,
        on=["fecha", "provincia"],
        how="inner",
    )
    print(f"  Merged: {merged.shape} rows")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
