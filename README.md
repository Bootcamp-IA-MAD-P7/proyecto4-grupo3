# Wildfire Burned Area Prediction

A regression pipeline that estimates the burned area (hectares) of forest wildfires using historical weather data from the Spanish meteorological agency (AEMET) and fire incident records.

The pipeline downloads weather data via the AEMET OpenData API, merges it with cleaned fire incident records, engineers features, trains and evaluates four regression models, and selects the best one based on generalization performance.

---

## Technologies

- **Python 3.12** with `uv` for dependency management
- **Polars** for high-performance DataFrame operations (data loading, cleaning, grouping)
- **Scikit-learn** for preprocessing (StandardScaler, OneHotEncoder, ColumnTransformer) and baseline models (LinearRegression, RandomForestRegressor)
- **XGBoost** and **LightGBM** for gradient-boosted tree models
- **Optuna** for hyperparameter optimization (30 trials, overfitting penalty)
- **Joblib** for model serialization
- **Streamlit** for the interactive prediction web interface
- **Docker** for containerized deployment

---

## Project Structure

```
proyecto4-grupo3/
├── main.py                          Entry point (runs full pipeline)
├── app.py                           Streamlit prediction interface
├── Dockerfile                       Docker image definition
├── docker-compose.yml               Docker Compose configuration
├── pyproject.toml                   Project metadata and dependencies
├── scripts/
│   ├── fetch_aemet.py               Downloads AEMET weather data (per-year CSVs)
│   ├── unify_aemet.py               Concatenates yearly CSVs into aemet_all.csv
│   └── merge_datasets.py            Merges fire incidents with weather data
├── src/
│   ├── config.py                    Paths, feature lists, hyperparameters
│   ├── pipeline.py                  Pipeline orchestrator
│   ├── preprocess.py                Feature engineering and preprocessing
│   ├── train.py                     Model training, evaluation, and selection
│   └── predict.py                   Production inference helpers
├── data/
│   ├── raw/                         Raw AEMET CSVs and fire incidents CSV
│   ├── cleaned/                     Cleaned datasets (parquet)
│   └── processed/
│       └── fires_weather_merged.parquet  Final merged dataset (21 columns, 22300 rows)
├── models/                          Trained model artifacts (joblib)
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   ├── target_transformer.joblib
│   └── target_encoding.joblib
└── reports/figures/                 Evaluation plots (predictions, residuals, feature importance)
```

---

## Pipeline Overview

### Phase 0: Data Preparation (one-time setup)

1. **`scripts/fetch_aemet.py`** downloads daily weather data from the AEMET OpenData API in 14-day blocks for each year (default: 2013-2022). Saves one CSV per year to `data/raw/aemet_{year}.csv`. Requires `AEMET_API_KEY` in `.env`.
2. **`scripts/unify_aemet.py`** concatenates all yearly CSVs into `data/raw/aemet_all.csv`.
3. **`scripts/merge_datasets.py`** loads the unified AEMET data, cleans and standardizes column names, maps provinces to autonomous communities (`cc_aa`), averages weather values per day and province, then performs an inner join with cleaned fire incidents (`data/cleaned/fires_clean.parquet`) on `(fecha, provincia)`. The result is saved to `data/processed/fires_weather_merged.parquet`.

### Phase 1: Training Pipeline (`main.py`)

`main.py` runs the full training pipeline:

- **Features (21 columns in, 18 numeric + 18 one-hot = 36 out):**
  - Weather: temperature, precipitation, humidity, wind speed, wind gusts, sunshine hours, wind direction
  - Geographic: altitude, latitude, longitude, autonomous community
  - Temporal: month
  - Categorical: fire cause (6 types)
  - Engineered: wind direction sin/cos encoding, 3 interaction features (temp-humidity, temp-wind, humidity-wind), 4 target-encoded features (province, province-month, community, community-month)

- **Target:** `superficie_quemada` (burned area in hectares), capped at 10.0 ha and log-transformed with `log1p`.

- **Models trained:**
  1. Linear Regression (baseline)
  2. Random Forest (max_depth=8, n_estimators=200)
  3. XGBoost (hyperparameters optimized via Optuna, 30 trials)
  4. LightGBM (fixed hyperparameters)

- **Model selection:** Filters out models with over 5% train-test R2 gap, then selects the model with the highest test R2.

- **Outputs saved to `models/`:**
  - `best_model.joblib` - the selected model
  - `preprocessor.joblib` - fitted ColumnTransformer (StandardScaler + OneHotEncoder)
  - `target_transformer.joblib` - LogTransformer for log1p/expm1
  - `target_encoding.joblib` - target encoding mappings

- **Plots saved to `reports/figures/`:** prediction scatter plots, residual histograms, feature importance bar charts.

### Phase 2: Inference (Streamlit)

`app.py` loads the four model artifacts and provides a sidebar form with 14 input fields (province, fire cause, month, altitude, temperature, humidity, precipitation, sunshine, wind speed, wind gusts, wind direction, latitude, longitude). On prediction, it applies the same feature engineering pipeline used during training and displays the estimated burned area in hectares with a severity classification (Conato < 3 ha, Small < 10 ha, Medium < 50 ha, Large >= 50 ha).

---

## Installation

### Prerequisites

- Python 3.12 or later
- `uv` package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- (Optional) Docker

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd proyecto4-grupo3

# Install dependencies
uv sync

# Set up AEMET API key (required only for downloading weather data)
cp .env.example .env
# Edit .env and add your AEMET_API_KEY

# (Optional) Download weather data and prepare datasets
uv run python scripts/fetch_aemet.py
uv run python scripts/unify_aemet.py
uv run python scripts/merge_datasets.py

# Run the full training pipeline
uv run python main.py

# Launch the Streamlit interface
uv run streamlit run app.py
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker compose up --build

# The Streamlit app is available at http://localhost:8501
```

The `docker-compose.yml` mounts the local `models/` directory into the container, so trained artifacts persist across rebuilds.

---

## Docker Details

### Dockerfile

- Base image: `python:3.12-slim`
- `uv` is installed via pip
- `uv sync` installs all project dependencies from `pyproject.toml`
- Exposes port 8501 (Streamlit)
- Default command runs `streamlit run app.py` on `0.0.0.0`

### docker-compose.yml

- Maps host port 8501 to container port 8501
- Mounts `./models:/app/models` for persistent model artifacts

---

## Dataset Details

The final merged dataset (`fires_weather_merged.parquet`) contains 22,300 rows and 21 columns:

**Weather features (from AEMET):**
- `temperatura_media` - mean daily temperature (Celsius)
- `precipitacion` - daily precipitation (mm)
- `humedad_relativa_media` - mean relative humidity (%)
- `velocidad_viento_media` - mean wind speed (m/s)
- `racha_maxima_viento` - maximum wind gust (m/s)
- `sol` - sunshine hours (246 nulls imputed with median 9.425)
- `dir` - wind direction (0-36 scale, encoded as sin/cos)

**Geographic features:**
- `provincia` - province (39 unique values, target-encoded)
- `cc_aa` - autonomous community (17 values, derived from province)
- `altitud` - altitude (meters)

**Fire incident features:**
- `latitud` / `longitud` - fire coordinates
- `causa_incendio` - fire cause (6 categories: lightning, agricultural burn, forest burn, intentional, negligence, other)
- `mes` - month (1-12)
- `superficie_quemada` - target variable (burned area in hectares)

**Columns dropped during preprocessing:**
`id`, `fecha`, `fecha_incendio`, `año`, `trimestre`, `month_name`, `cc_aa_right`, `dir`

---

## Results

The model selection logic prioritizes generalization by filtering out models with train-test R2 gap exceeding 5%. Typical results show Linear Regression as the most stable model (lowest overfitting), while tree-based models achieve higher training performance but exhibit more overfitting.

Evaluation metrics are computed in the original hectare scale (after inverse log1p transformation):

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

---

## Configuration

All paths, feature lists, and hyperparameters are centralized in `src/config.py`. Key constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MERGED_DATA_PATH` | `data/processed/fires_weather_merged.parquet` | Input dataset |
| `MODEL_PATH` | `models/best_model.joblib` | Trained model output |
| `TARGET` | `superficie_quemada` | Target variable |
| `TARGET_CAP` | 10.0 ha | Target capping threshold |
| `TEST_SIZE` | 0.2 | Test split proportion |
| `RANDOM_STATE` | 42 | Global reproducibility seed |

---

## Quick Reference

```bash
# Train the full pipeline
uv run python main.py

# Launch the web interface
uv run streamlit run app.py

# Download weather data (requires AEMET_API_KEY)
uv run python scripts/fetch_aemet.py

# Unify yearly CSVs
uv run python scripts/unify_aemet.py

# Merge weather with fire incidents
uv run python scripts/merge_datasets.py

# Docker
docker compose up --build
```
