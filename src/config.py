from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

MERGED_DATA_PATH = PROCESSED_DIR / "fires_weather_merged.parquet"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
TARGET_TRANSFORMER_PATH = MODELS_DIR / "target_transformer.joblib"
TARGET_ENCODING_PATH = MODELS_DIR / "target_encoding.joblib"

TARGET = "superficie_quemada"

FEATURES_NUMERIC = [
    "altitud",
    "latitud",
    "longitud",
    "temperatura_media",
    "precipitacion",
    "humedad_relativa_media",
    "velocidad_viento_media",
    "racha_maxima_viento",
    "sol",
    "dir_sin",
    "dir_cos",
    "temp_hum_interaction",
    "temp_wind_interaction",
    "hum_wind_interaction",
    "provincia_te",
    "provincia_mes_te",
    "cc_aa_te",
    "cc_aa_mes_te",
]

FEATURES_CATEGORICAL = [
    "causa_incendio",
    "mes",
]

FEATURES_TO_DROP = [
    "id",
    "fecha",
    "fecha_incendio",
    "cc_aa_right",
    "month_name",
    "dir",
    "año",
]

TARGET_CAP = 10.0

RANDOM_STATE = 42
TEST_SIZE = 0.2

CAUSA_MAP = {
    1: "rayo",
    2: "quema_agricola",
    3: "quema_forestal",
    4: "intencionado",
    5: "negligencia",
    6: "otras",
}
