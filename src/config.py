from pathlib import Path

DATA_PROCESSED = Path("data/processed/fires_weather_merged.parquet")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")
DATA_PROCESSED_DIR = Path("data/processed")

TARGET = "superficie_quemada"

DROP_COLS = ["id", "fecha", "fecha_incendio", "month_name", "trimestre"]

GEO_COLS = ["latitud", "longitud"]

NUM_COLS = [
    "altitud",
    "temperatura_media",
    "precipitacion",
    "humedad_relativa_media",
    "velocidad_viento_media",
]

CAT_TARGET_ENCODE = ["provincia", "cc_aa"]

CAT_OHE = ["causa_incendio"]

CYCLIC_COLS = ["mes"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RF_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

HGB_PARAMS = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_iter": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "min_samples_leaf": [10, 20, 30, 50],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
}


def detect_year_column(columns: list[str]) -> str:
    for col in columns:
        col_lower = col.lower().replace("\xf1", "n")
        if "a" in col_lower and ("no" in col_lower or "os" in col_lower):
            return col
    return "a\xf1o"
