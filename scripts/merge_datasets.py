"""Merge AEMET weather data with fire incidents.

Pipeline:
  1. Load each AEMET yearly CSV → clean → group by (fecha, provincia) averaging weather
  2. Concatenate all years
  3. Load processed fires (fires-all-prepro.parquet)
  4. Inner join on (fecha, provincia) → each fire gets that day's weather
  5. Save merged dataset

Usage:
  python scripts/merge_datasets.py
"""

import polars as pl
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

AEMET_PATTERN = "aemet_{}.csv"
FIRES_PATH = PROCESSED / "fires-all-prepro.parquet"
OUTPUT_PATH = PROCESSED / "fires_weather_merged.parquet"

COLUMNS_KEEP = [
    "temperatura_media", "precipitacion", "temperatura_minima",
    "temperatura_maxima", "humedad_relativa_media",
    "humedad_relativa_maxima", "humedad_relativa_minima",
    "velocidad_viento_media", "racha_maxima_viento",
]

provincia_to_ccaa = {
    "ALMERIA": "ANDALUCIA", "CADIZ": "ANDALUCIA",
    "CORDOBA": "ANDALUCIA", "GRANADA": "ANDALUCIA",
    "HUELVA": "ANDALUCIA", "JAEN": "ANDALUCIA",
    "MALAGA": "ANDALUCIA", "SEVILLA": "ANDALUCIA",
    "HUESCA": "ARAGON", "TERUEL": "ARAGON", "ZARAGOZA": "ARAGON",
    "ASTURIAS": "ASTURIAS",
    "BALEARES": "BALEARES", "ILLES BALEARS": "BALEARES",
    "LAS PALMAS": "CANARIAS",
    "STA. CRUZ DE TENERIFE": "CANARIAS",
    "SANTA CRUZ DE TENERIFE": "CANARIAS",
    "CANTABRIA": "CANTABRIA", "SANTANDER": "CANTABRIA",
    "ALBACETE": "CASTILLA-LA MANCHA",
    "CIUDAD REAL": "CASTILLA-LA MANCHA",
    "CUENCA": "CASTILLA-LA MANCHA",
    "GUADALAJARA": "CASTILLA-LA MANCHA",
    "TOLEDO": "CASTILLA-LA MANCHA",
    "AVILA": "CASTILLA Y LEON", "BURGOS": "CASTILLA Y LEON",
    "LEON": "CASTILLA Y LEON", "PALENCIA": "CASTILLA Y LEON",
    "SALAMANCA": "CASTILLA Y LEON", "SEGOVIA": "CASTILLA Y LEON",
    "SORIA": "CASTILLA Y LEON", "VALLADOLID": "CASTILLA Y LEON",
    "ZAMORA": "CASTILLA Y LEON",
    "BARCELONA": "CATALUNA", "GIRONA": "CATALUNA",
    "LLEIDA": "CATALUNA", "TARRAGONA": "CATALUNA",
    "BADAJOZ": "EXTREMADURA", "CACERES": "EXTREMADURA",
    "A CORUÑA": "GALICIA", "LUGO": "GALICIA",
    "OURENSE": "GALICIA", "PONTEVEDRA": "GALICIA",
    "LA RIOJA": "LA RIOJA", "MADRID": "MADRID",
    "MURCIA": "MURCIA", "NAVARRA": "NAVARRA",
    "ALAVA": "PAIS VASCO", "ARABA/ALAVA": "PAIS VASCO",
    "GUIPUZCOA": "PAIS VASCO", "GIPUZKOA": "PAIS VASCO",
    "BIZKAIA": "PAIS VASCO", "VIZCAYA": "PAIS VASCO",
    "ALICANTE": "VALENCIA", "CASTELLON": "VALENCIA",
    "VALENCIA": "VALENCIA", "CEUTA": "CEUTA", "MELILLA": "MELILLA",
}


def clean_aemet(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise and clean a single AEMET yearly DataFrame."""
    df = df.drop([
        "indicativo", "nombre", "horatmin", "horatmax", "horaHrMin",
        "dir", "horaracha", "horaPresMax", "horaPresMin", "sol",
        "horaHrMax", "presMax", "presMin",
    ])

    df = df.with_columns(
        pl.col("provincia")
        .str.to_uppercase()
        .replace_strict(provincia_to_ccaa, default=None)
        .alias("cc_aa")
    )

    # Rename
    df = df.rename({
        "tmed": "temperatura_media",
        "prec": "precipitacion",
        "tmin": "temperatura_minima",
        "tmax": "temperatura_maxima",
        "hrMedia": "humedad_relativa_media",
        "hrMax": "humedad_relativa_maxima",
        "hrMin": "humedad_relativa_minima",
        "velmedia": "velocidad_viento_media",
        "racha": "racha_maxima_viento",
    })

    # Filter out non-numeric markers in precipitation
    df = df.filter(~pl.col("precipitacion").is_in(["Ip", "Acum"]))

    # Convert comma decimals → float for weather cols
    for col in ["temperatura_media", "temperatura_minima",
                "temperatura_maxima", "precipitacion",
                "velocidad_viento_media", "racha_maxima_viento"]:
        df = df.with_columns(
            pl.col(col).str.replace(",", ".").cast(pl.Float64)
        )

    # Parse date
    df = df.with_columns(pl.col("fecha").str.to_date("%Y-%m-%d"))

    # Drop nulls in core weather columns
    core = ["temperatura_media", "precipitacion", "temperatura_minima",
            "temperatura_maxima"]
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
    years = range(2013, 2023)

    # 1. Process each AEMET file with map
    aemet_files = [RAW / AEMET_PATTERN.format(y) for y in years]

    def load_and_process(path):
        df = pl.read_csv(
            path,
            schema_overrides={"horaPresMin": pl.Utf8},
            null_values=["Varias"],
        )
        df = clean_aemet(df)
        return average_per_day(df)

    print("Processing AEMET files...")
    results = list(map(load_and_process, aemet_files))
    aemet_all = pl.concat(results)
    print(f"  AEMET combined: {aemet_all.shape} rows")

    # 2. Load fires
    print("Loading fires...")
    fires = load_fires(FIRES_PATH)
    print(f"  Fires: {fires.shape} rows")

    # 3. Drop fires' cc_aa (same as AEMET's)
    fires = fires.drop("cc_aa")

    # 4. Inner join on (fecha, provincia)
    print("Merging...")
    merged = aemet_all.join(
        fires,
        on=["fecha", "provincia"],
        how="inner",
    )
    print(f"  Merged: {merged.shape} rows")

    # 5. Save
    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
