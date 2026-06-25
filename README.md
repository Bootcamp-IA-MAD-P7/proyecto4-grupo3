# Predicción de Superficie Quemada (Incendios Forestales)

Pipeline de regresión para estimar la superficie quemada en incendios forestales usando datos meteorológicos de AEMET.

---

## Estructura del proyecto

```
proyecto4-grupo3/
├── main.py                          ← PUNTO DE ENTRADA
├── app.py                           ← INTERFAZ STREAMLIT
├── scripts/
│   ├── fetch_aemet.py               ← Descarga datos AEMET
│   └── merge_datasets.py            ← Fusiona incendios + meteorología
├── src/
│   ├── config.py                    ← Constantes, rutas, hiperparámetros
│   ├── pipeline.py                  ← Orquestador (Fase 1 + Fase 2)
│   ├── preprocess.py                ← Preprocesamiento + feature engineering
│   ├── train.py                     ← Entrenamiento de 4 modelos
│   ├── predict.py                   ← Predicción en producción
│   └── eda.py                       ← Análisis exploratorio (no usado en pipeline)
├── data/processed/
│   └── fires_weather_merged.parquet ← DATASET FINAL (21 columnas, 22300 filas)
├── models/
│   ├── best_model.joblib            ← Modelo XGBoost guardado
│   ├── preprocessor.joblib          ← ColumnTransformer (StandardScaler + OHE)
│   ├── target_transformer.joblib    ← LogTransformer (log1p / expm1)
│   └── target_encoding.joblib       ← Mapas de target encoding
└── reports/figures/                 ← Gráficos (predicciones, residuos, importancia)
```

---

## Columnas del dataset (`fires_weather_merged.parquet`)

| Columna | Tipo | Se usa | Notas |
|---------|------|--------|-------|
| `provincia` | String | Sí → target encoding | 39 provincias |
| `cc_aa` | String | Sí → target encoding | 17 CCAA, se deriva de provincia |
| `altitud` | Float64 | Sí | 2122 valores únicos |
| `latitud` | String → Float64 | Sí | Proxy geográfico |
| `longitud` | String → Float64 | Sí | Proxy geográfico |
| `temperatura_media` | Float64 | Sí | Temperatura media diaria |
| `precipitacion` | Float64 | Sí | Precipitación acumulada |
| `humedad_relativa_media` | Float64 | Sí | Humedad relativa media |
| `velocidad_viento_media` | Float64 | Sí | Velocidad del viento |
| `racha_maxima_viento` | Float64 | Sí | Racha máxima |
| `sol` | Float64 | Sí | Horas de sol (246 nulos imputados) |
| `dir` | Float64 | Sí → dir_sin/dir_cos | Dirección del viento (0-36) |
| `causa_incendio` | Int64 | Sí → one-hot (6 categorías) | 1=rayo,…,6=otras |
| `mes` | Int8 | Sí → one-hot (12 categorías) | 1-12 |
| `superficie_quemada` | Float64 | **TARGET** | Variable a predecir (cappeada a 10 ha) |
| `id` | Int64 | No | Eliminada |
| `fecha` | Date | No | Eliminada |
| `fecha_incendio` | Date | No | Eliminada |
| `año` | Int32 | No | Eliminada |
| `trimestre` | Int8 | No | Eliminada |
| `month_name` | String | No | Eliminada |

---

## Features creadas (ingeniería)

| Feature | Creada en | Fórmula | Propósito |
|---------|-----------|---------|-----------|
| `dir_sin`, `dir_cos` | `prepare_features()` | `sin(dir×10×π/180)`, `cos(...)` | Codificación circular del viento (0°=360°) |
| `temp_hum_interaction` | `prepare_features()` | `temp_media × humedad_media` | Sinergia calor + sequedad |
| `temp_wind_interaction` | `prepare_features()` | `temp_media × viento_media` | Sinergia calor + viento |
| `hum_wind_interaction` | `prepare_features()` | `humedad_media × viento_media` | Sinergia sequedad + viento |
| `provincia_te` | `apply_target_encoding()` | Media de target por provincia (smoothing α=10) | Riesgo basal por provincia |
| `provincia_mes_te` | `apply_target_encoding()` | Media de target por (provincia, mes) | Estacionalidad regional |
| `cc_aa_te` | `apply_target_encoding()` | Media de target por CCAA | Riesgo basal por comunidad (más estable) |
| `cc_aa_mes_te` | `apply_target_encoding()` | Media de target por (CCAA, mes) | Estacionalidad por comunidad |

Total: **18 numéricas + 18 one-hot (6 causa + 12 mes) = 36 features** que entran al modelo.

---

## Flujo completo

### 0. Preparación de datos (manual)

**`scripts/fetch_aemet.py`** — descarga datos meteorológicos históricos de AEMET.

**`scripts/merge_datasets.py`** — cruza incendios + AEMET y genera `fires_weather_merged.parquet`.

### 1. Ejecución: `uv run python main.py`

```
main.py
  3-6   logging.basicConfig(level=INFO)     ← Configura logs limpios
  7     logging.getLogger("optuna").setLevel(WARNING)  ← Silencia Optuna
  9     from src.pipeline import main
 11-12  if __name__ == "__main__": main()
```

### 2. `src/pipeline.py` → Orquestador

```
pipeline.main()
├── FASE 1: preprocess.run_preprocessing(MERGED_DATA_PATH)
└── FASE 2: train.run_training(X_train, X_test, y_train, y_test, preprocessor, ...)
```

### 3. FASE 1: Preprocesamiento → `src/preprocess.py`

| Paso | Función | Línea | Qué hace |
|------|---------|-------|----------|
| Cargar datos | — | 152 | `pl.read_parquet(...)` |
| Capping target | — | 155-158 | `superficie_quemada.clip(upper=10.0)` |
| Feature engineering | `prepare_features()` | 160-162 | Crea interacciones, dir_sin/cos, imputa |
| Split train/test | `split_data()` | 164-165 | 80/20 → 17840/4460 |
| Target encoding | `apply_target_encoding()` | 167-168 | provincia_te, cc_aa_te, etc. |
| Log transform | `LogTransformer()` | 170-174 | `y = log1p(target)` |
| ColumnTransformer | `build_preprocessor()` | 176-181 | StandardScaler + OneHotEncoder |
| Guardar artefactos | `joblib.dump()` | 183-191 | 4 archivos en `models/` |

**`prepare_features()` (L60-92):**
- `encode_dir_circular()`: `dir` (escala 0-36) → `dir_sin`, `dir_cos`
- `latitud`, `longitud`: string → float64
- Elimina columnas de `FEATURES_TO_DROP` (`id`, `fecha`, `año`, `dir`, etc.)
- Imputa nulos con mediana (solo `sol` tiene 246 nulos)
- Crea 3 interacciones: `temp_hum`, `temp_wind`, `hum_wind`

**`apply_target_encoding()` (L95-128):**
- Para cada grupo: calcula media del target con smoothing α=10
- `(sum + global_mean × 10) / (count + 10)` — evita overfitting en grupos pequeños
- Elimina columnas originales (`provincia`, `cc_aa`)

**`build_preprocessor()` (L142-148):**
- `("num", StandardScaler(), FEATURES_NUMERIC)` → 18 columnas escaladas
- `("cat", OneHotEncoder(), FEATURES_CATEGORICAL)` → 18 columnas (6+12)

### 4. FASE 2: Entrenamiento → `src/train.py`

**`run_training()` (L233-339)** entrena 4 modelos:

| Modelo | Función | Líneas | Hiperparámetros clave |
|--------|---------|--------|-----------------------|
| Linear Regression | `train_linear_regression()` | 22-25 | Por defecto |
| Random Forest | `train_random_forest()` | 28-35 | max_depth=8, n_estimators=200, min_samples_leaf=10 |
| XGBoost (Optuna) | `optimize_xgboost()` + `XGBRegressor` | 177-224 + 266-278 | 30 trials, penaliza overfitting |
| LightGBM | `train_lightgbm()` | 38-46 | max_depth=5, n_estimators=300, lr=0.05 |

**`optimize_xgboost()` (L177-224):** Busca 7 parámetros con Optuna:
- `n_estimators` (100-400), `max_depth` (3-5), `learning_rate` (0.01-0.15)
- `subsample` (0.6-1.0), `colsample_bytree` (0.5-1.0)
- `reg_alpha` (0.5-5), `reg_lambda` (0.5-5), `min_child_weight` (3-10)
- Penaliza overfitting: objetivo = `r2_val - 2×max(0, r2_train-r2_val-0.05)`

**`evaluate_model()` (L78-113):**
- Predice train y test, deshace log1p con `expm1`
- Calcula MAE, RMSE, R², overfitting

**Gráficos guardados en `reports/figures/`:**
- `predictions_{modelo}.png` — scatter real vs predicción
- `residuals_{modelo}.png` — histograma + residuos vs predicción
- `feature_importance_{modelo}.png` — top 20 variables (RF y XGBoost)

**Selección del mejor modelo (L315-337):**
- Filtra modelos con overfitting (R²_train − R²_test) < 5%
- Entre esos, elige el de mayor R² en test
- Guarda con `joblib.dump()` en `models/best_model.joblib`

### 5. Predicción → `src/predict.py`

**`predict_single(input_data: dict)` (L103-114):**
1. Carga los 4 artefactos de `models/`
2. `prepare_input_for_prediction()` — replica el feature engineering del training
3. `preprocessor.transform()` — escala y one-hot
4. `model.predict()` — predicción en log-scale
5. `target_transformer.inverse_transform()` — expm1 → hectáreas

**`prepare_input_for_prediction()` (L67-100):**
- Deriva `cc_aa` desde `provincia` usando mapa PROVINCIA_TO_CCAA
- Convierte lat/long a float
- Calcula dir_sin/cos, interacciones
- Aplica target encoding con mappings guardados
- Elimina columnas originales

### 6. Interfaz → `app.py`

- `load_artifacts()` — carga los 4 modelos con cache
- 14 inputs en sidebar: provincia, causa, mes, altitud, latitud, longitud, temperatura_media, precipitacion, humedad, viento, racha, sol, dirección del viento
- Al predecir: feature engineering → preprocessor → modelo → muestra resultado en hectáreas con categoría (Conato < 3 ha, Pequeño < 10 ha, Mediano < 50 ha, Grande ≥ 50 ha)

---

## Configuración global → `src/config.py`

| Constante | Valor | Propósito |
|-----------|-------|-----------|
| `MERGED_DATA_PATH` | `data/processed/fires_weather_merged.parquet` | Dataset de entrada |
| `MODEL_PATH` | `models/best_model.joblib` | Modelo serializado |
| `PREPROCESSOR_PATH` | `models/preprocessor.joblib` | ColumnTransformer serializado |
| `TARGET_TRANSFORMER_PATH` | `models/target_transformer.joblib` | LogTransformer serializado |
| `TARGET_ENCODING_PATH` | `models/target_encoding.joblib` | Mapas de target encoding |
| `TARGET` | `superficie_quemada` | Variable objetivo |
| `FEATURES_NUMERIC` | 18 columnas | Escaladas con StandardScaler |
| `FEATURES_CATEGORICAL` | `["causa_incendio", "mes"]` | One-hot encode |
| `FEATURES_TO_DROP` | 7 columnas | Eliminadas en prepare_features |
| `TARGET_CAP` | 10.0 ha | Capping del target |
| `RANDOM_STATE` | 42 | Semilla global |
| `TEST_SIZE` | 0.2 | Proporción de test |

---

## Resumen visual del flujo

```
scripts/merge_datasets.py
        ↓
fires_weather_merged.parquet (21 cols, 22300 filas)
        ↓
main.py → pipeline.main()
        ↓
├─ FASE 1: preprocess.run_preprocessing()
│   ├─ prepare_features():      dir→dir_sin/cos, lat/long→float, imputa, +3 interacciones
│   ├─ split_data():            80/20 train/test (17840/4460)
│   ├─ apply_target_encoding(): provincia→provincia_te, cc_aa→cc_aa_te, +2 mes
│   ├─ LogTransformer():        y = log1p(target)
│   ├─ build_preprocessor():    ColumnTransformer(StandardScaler + OHE)
│   └─ joblib.dump():           models/{preprocessor,target_transformer,target_encoding}.joblib
│
└─ FASE 2: train.run_training()
    ├─ LinearRegression      → evaluate → plot → ¿overfitting < 5%?
    ├─ RandomForest          → evaluate → plot → ¿overfitting < 5%?
    ├─ optimize_xgboost()    → Optuna 30 trials → best params
    ├─ XGBoost(best_params)  → evaluate → plot → feature importance
    ├─ LightGBM              → evaluate → plot → ¿overfitting < 5%?
    ├─ cross_validate_kfold()→ K-Fold CV con XGBoost
    ├─ Comparación:          elige mejor R² con overfitting < 5%
    └─ joblib.dump(model)   → models/best_model.joblib

app.py → prepare_input_for_prediction() → preprocessor → model → resultado
```

---

## Comandos

```bash
uv run python main.py              # Entrena el pipeline completo
uv run streamlit run app.py        # Lanza la interfaz
```
