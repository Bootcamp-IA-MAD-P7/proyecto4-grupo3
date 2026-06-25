import logging
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path

from src.config import (
    TARGET,
    FEATURES_NUMERIC,
    FEATURES_CATEGORICAL,
    FEATURES_TO_DROP,
    TEST_SIZE,
    RANDOM_STATE,
    PREPROCESSOR_PATH,
    TARGET_TRANSFORMER_PATH,
    TARGET_ENCODING_PATH,
    TARGET_CAP,
)

logger = logging.getLogger(__name__)
_impute_values = {}


class LogTransformer:
    def transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y)


def encode_dir_circular(df: pl.DataFrame) -> pl.DataFrame:
    direccion_rad = (pl.col("dir") * 10).cast(pl.Float64) * np.pi / 180.0
    df = df.with_columns(
        direccion_rad.sin().alias("dir_sin"),
        direccion_rad.cos().alias("dir_cos"),
    )
    return df


def prepare_features(df: pl.DataFrame, fit_imputer: bool = True) -> tuple:
    df = encode_dir_circular(df)

    for col in ["latitud", "longitud"]:
        if col in df.columns and df[col].dtype == pl.String:
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    df = df.drop([c for c in FEATURES_TO_DROP if c in df.columns], strict=False)

    global _impute_values
    if fit_imputer:
        _impute_values = {}
        for col in df.columns:
            if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and df[col].is_null().sum() > 0:
                _impute_values[col] = df[col].median()
                df = df.with_columns(pl.col(col).fill_null(_impute_values[col]))
                logger.info("  Imputado %s: %s nulos con mediana %.4f",
                            col, df[col].is_null().sum(), _impute_values[col])
    else:
        for col, val in _impute_values.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(val))

    df = df.with_columns([
        (pl.col("temperatura_media") * pl.col("humedad_relativa_media")).alias("temp_hum_interaction"),
        (pl.col("temperatura_media") * pl.col("velocidad_viento_media")).alias("temp_wind_interaction"),
        (pl.col("humedad_relativa_media") * pl.col("velocidad_viento_media")).alias("hum_wind_interaction"),
    ])

    y = df[TARGET]
    X = df.drop(TARGET)

    return X, y


def apply_target_encoding(X_train, X_test, y_train):
    X_train = X_train.copy()
    X_test = X_test.copy()
    alpha = 10

    X_train["_provincia_mes"] = X_train["provincia"].astype(str) + "_" + X_train["mes"].astype(str)
    X_test["_provincia_mes"] = X_test["provincia"].astype(str) + "_" + X_test["mes"].astype(str)
    X_train["_cc_aa_mes"] = X_train["cc_aa"].astype(str) + "_" + X_train["mes"].astype(str)
    X_test["_cc_aa_mes"] = X_test["cc_aa"].astype(str) + "_" + X_test["mes"].astype(str)

    mappings = {}
    for col, out_name in [
        ("provincia", "provincia_te"),
        ("_provincia_mes", "provincia_mes_te"),
        ("cc_aa", "cc_aa_te"),
        ("_cc_aa_mes", "cc_aa_mes_te"),
    ]:
        stats = pd.DataFrame({"_target": y_train, "_group": X_train[col]}).groupby("_group")["_target"].agg(["sum", "count"])
        global_mean = float(y_train.mean())
        stats["mean"] = (stats["sum"] + global_mean * alpha) / (stats["count"] + alpha)
        mapping = stats["mean"].to_dict()
        mappings[out_name] = {"mapping": mapping, "global_mean": global_mean}
        X_train[out_name] = X_train[col].map(mapping)
        X_test[out_name] = X_test[col].map(mapping).fillna(global_mean)

    X_train.drop(columns=["provincia", "cc_aa", "_provincia_mes", "_cc_aa_mes"], inplace=True)
    X_test.drop(columns=["provincia", "cc_aa", "_provincia_mes", "_cc_aa_mes"], inplace=True)

    logger.info("  provincia_te: %s valores unicos", X_train["provincia_te"].nunique())
    logger.info("  provincia_mes_te: %s valores unicos", X_train["provincia_mes_te"].nunique())
    logger.info("  cc_aa_te: %s valores unicos", X_train["cc_aa_te"].nunique())
    logger.info("  cc_aa_mes_te: %s valores unicos", X_train["cc_aa_mes_te"].nunique())

    return X_train, X_test, mappings


def split_data(X: pl.DataFrame, y: pl.Series):
    X_pd = X.to_pandas()
    y_np = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y_np, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train: %s muestras, Test: %s muestras", X_train.shape[0], X_test.shape[0])
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train) -> ColumnTransformer:
    ct = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CATEGORICAL),
    ])
    ct.fit(X_train)
    return ct


def run_preprocessing(data_path: Path):
    df = pl.read_parquet(data_path)
    logger.info("Dataset: %s", df.shape)

    logger.info("--- Capeando target a %s ha ---", TARGET_CAP)
    n_capped = (df[TARGET] > TARGET_CAP).sum()
    df = df.with_columns(pl.col(TARGET).clip(upper_bound=TARGET_CAP))
    logger.info("  Valores capeados: %s (%.1f%%)", n_capped, n_capped / len(df) * 100)

    logger.info("--- Preparando features ---")
    X, y = prepare_features(df)
    logger.info("  Features: %s, Target: %s", X.shape, y.shape)

    logger.info("--- Split train/test ---")
    X_train, X_test, y_train, y_test = split_data(X, y)

    logger.info("--- Target encoding de provincia ---")
    X_train, X_test, te_mappings = apply_target_encoding(X_train, X_test, y_train)

    logger.info("--- Aplicando log1p al target ---")
    target_transformer = LogTransformer()
    y_train_log = target_transformer.transform(y_train)
    y_test_log = target_transformer.transform(y_test)
    logger.info("  log1p shapes: y_train=%s, y_test=%s", y_train_log.shape, y_test_log.shape)

    logger.info("--- Construyendo preprocesador (ColumnTransformer) ---")
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info("  X_train: %s, X_test: %s", X_train_processed.shape, X_test_processed.shape)

    Path(PREPROCESSOR_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("  Preprocesador guardado en %s", PREPROCESSOR_PATH)

    joblib.dump(target_transformer, TARGET_TRANSFORMER_PATH)
    logger.info("  Target transformer guardado en %s", TARGET_TRANSFORMER_PATH)

    joblib.dump(te_mappings, TARGET_ENCODING_PATH)
    logger.info("  Target encoding mappings guardados en %s", TARGET_ENCODING_PATH)

    return (X_train_processed, X_test_processed, y_train_log, y_test_log,
            preprocessor, target_transformer, te_mappings)
