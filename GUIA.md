# GUÍA PASO A PASO DEL PROYECTO

## Qué estamos haciendo

Predecir cuántas hectáreas va a quemar un incendio forestal dadas las condiciones meteorológicas del día y la ubicación.

El target es `superficie_quemada` (hectáreas). Es un problema de **regresión numérica**.

---

## PASO 0: Los datos de entrada

### Archivo: `data/processed/fires_weather_merged.parquet`

Es el resultado de cruzar datos de incendios forestales con datos meteorológicos de AEMET.

Columnas disponibles (21):

**Identificación (no sirven para predecir):**
- `id` — identificador único de cada incendio
- `fecha` — cuándo ocurrió
- `fecha_incendio` — redundante
- `año` — redundante (ya tenemos mes)
- `month_name` — redundante (ya tenemos mes)
- `trimestre` — redundante (ya tenemos mes)

**Geográficas:**
- `provincia` — 39 valores (ej: "MADRID", "A CORUÑA")
- `cc_aa` — 17 comunidades autónomas
- `altitud` — altura sobre el nivel del mar
- `latitud` — coordenada norte-sur (guardada como string)
- `longitud` — coordenada este-oeste (guardada como string)

**Meteorológicas:**
- `temperatura_media` — temperatura media del día
- `precipitacion` — lluvia acumulada
- `humedad_relativa_media` — humedad media
- `velocidad_viento_media` — velocidad del viento
- `racha_maxima_viento` — racha más fuerte del día
- `sol` — horas de sol
- `dir` — dirección del viento (0=calma, 36=Norte, codificación AEMET)

**Causa:**
- `causa_incendio` — 1=rayo, 2=quema agrícola, 3=quema forestal, 4=intencionado, 5=negligencia, 6=otras

**Target:**
- `superficie_quemada` — lo que queremos predecir (hectáreas)

### El target: `superficie_quemada`

- Va de 0 a 14.953 hectáreas
- Está muy sesgado (skewness 38.7) — la mayoría son incendios pequeños y pocos muy grandes
- Se le aplica **capping** a 10 ha: todo lo que supere 10 hectáreas se trunca a 10
  - Esto afecta a 5475 filas (24.6% del dataset)
  - Se hace porque los valores extremos son imposibles de predecir con las variables disponibles
  - Después del capping, se aplica **logaritmo natural + 1** (`log1p`) para reducir la asimetría

---

## PASO 1: Ejecutar `uv run python main.py`

### Archivo: `main.py` (12 líneas)

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("optuna").setLevel(logging.WARNING)

from src.pipeline import main

if __name__ == "__main__":
    main()
```

**Qué hace:**
1. Configura logging para que se vean mensajes informativos en consola
2. Silencia los mensajes de Optuna (la biblioteca de optimización que prueba combinaciones de hiperparámetros, es muy verbosa)
3. Importa y ejecuta `pipeline.main()` — el orquestador

**Por qué existe main.py separado de pipeline.py:** Porque `main.py` es el punto de entrada que configura el entorno (logging). `pipeline.py` es la lógica pura del flujo. Separarlos permite probar `pipeline.main()` desde tests sin el logging.

---

## PASO 2: `pipeline.main()` — El orquestador

### Archivo: `src/pipeline.py` (42 líneas)

```python
from src.config import MERGED_DATA_PATH, FIGURES_DIR
from src.preprocess import run_preprocessing
from src.train import run_training

def main():
    # FASE 1: Preprocesamiento
    (X_train, X_test, y_train, y_test,
     preprocessor, target_transformer, te_mappings) = run_preprocessing(MERGED_DATA_PATH)

    # FASE 2: Entrenamiento
    feature_names = preprocessor.get_feature_names_out()
    run_training(X_train, X_test, y_train, y_test, preprocessor, target_transformer, feature_names)
```

**Flujo:**
1. Llama a `preprocess.run_preprocessing(MERGED_DATA_PATH)` → recibe los datos transformados
2. Obtiene los nombres de las 36 features finales del ColumnTransformer
3. Llama a `train.run_training(...)` con todo

**Lo que devuelve run_preprocessing (7 valores):**
- `X_train` (17840 × 36) — features de entrenamiento
- `X_test` (4460 × 36) — features de test
- `y_train` (17840,) — target de entrenamiento (en escala log1p)
- `y_test` (4460,) — target de test (en escala log1p)
- `preprocessor` — el ColumnTransformer ajustado
- `target_transformer` — el LogTransformer
- `te_mappings` — los mapas de target encoding

---

## PASO 3: FASE 1 — Preprocesamiento (`src/preprocess.py`)

### 3a. Cargar y capar el target

```python
df = pl.read_parquet(data_path)                    # 22300 filas, 21 columnas
n_capped = (df[TARGET] > TARGET_CAP).sum()          # 5475 valores > 10 ha (24.6%)
df = df.with_columns(pl.col(TARGET).clip(upper_bound=TARGET_CAP))  # se truncan a 10
```

**Por qué:** Los incendios extremadamente grandes (>10 ha) son outliers que el modelo no puede predecir bien con los datos meteorológicos. Al caparlos, el modelo se centra en predecir el rango donde hay más datos.

### 3b. `prepare_features()` — Ingeniería de características

#### 3b1. Codificación circular del viento

Archivo: `src/preprocess.py:51-57`

```python
def encode_dir_circular(df):
    direccion_rad = (pl.col("dir") * 10) * π / 180
    df = df.with_columns(
        direccion_rad.sin().alias("dir_sin"),
        direccion_rad.cos().alias("dir_cos"),
    )
```

La dirección del viento en AEMET viene como 0-36 (0=calma, 9=Este, 18=Sur, 27=Oeste, 36=Norte).

**Problema:** Si metemos el valor crudo (0-36), el modelo piensa que 0 y 36 están muy lejos, cuando en realidad los dos significan "Norte". Con seno y coseno se preserva la circularidad.

**Ejemplo:**
| dir | Sentido | dir_sin | dir_cos |
|-----|---------|---------|---------|
| 0   | Calma   | 0.0     | 1.0     |
| 9   | Este    | 1.0     | 0.0     |
| 18  | Sur     | 0.0     | -1.0    |
| 27  | Oeste   | -1.0    | 0.0     |
| 36  | Norte   | 0.0     | 1.0     |

Nota: 0 y 36 dan el mismo (0, 1) — correcto, ambos son Norte.

#### 3b2. Cast de latitud/longitud

```python
for col in ["latitud", "longitud"]:
    if col in df.columns and df[col].dtype == pl.String:
        df = df.with_columns(pl.col(col).cast(pl.Float64))
```

Los datos vienen como strings ("42.1635") y hay que convertirlos a número. El modelo necesita números.

#### 3b3. Eliminar columnas inútiles

```python
df = df.drop([c for c in FEATURES_TO_DROP if c in df.columns], strict=False)
```

Se eliminan: `id`, `fecha`, `fecha_incendio`, `cc_aa_right`, `month_name`, `dir` (reemplazada por dir_sin/dir_cos), `año`.

**Por qué se elimina cada una:**
- `id`: identificador único, zero predictivo
- `fecha`/`fecha_incendio`: ya tenemos mes y año; la fecha exacta no aporta
- `cc_aa_right`: columna sobrante del merge
- `month_name`: redundante con `mes` (numérico)
- `dir`: reemplazada por `dir_sin`/`dir_cos`
- `año`: correlación muy baja con el target, y no generaliza a años nuevos

#### 3b4. Imputación de nulos

```python
for col in df.columns:
    if col tiene nulos y es numérica:
        imputar con mediana
```

Solo `sol` tiene nulos (246 de 22300, ~1.1%). Se rellenan con la mediana (9.425 horas). La mediana se calcula en training y se guarda en memoria para usarla igual en test/predicción.

#### 3b5. Creación de interacciones

```python
df = df.with_columns([
    (pl.col("temperatura_media") * pl.col("humedad_relativa_media")).alias("temp_hum_interaction"),
    (pl.col("temperatura_media") * pl.col("velocidad_viento_media")).alias("temp_wind_interaction"),
    (pl.col("humedad_relativa_media") * pl.col("velocidad_viento_media")).alias("hum_wind_interaction"),
])
```

**Ejemplo práctico:** 35°C con 20% humedad = interacción = 700 (peligro alto). 20°C con 80% humedad = interacción = 1600. Suena a que 1600 > 700, pero en realidad 700 es más peligroso porque es calor extremo con aire seco.

Las interacciones capturan estas **sinergias** que las variables individuales no representan bien. Un árbol con una sola interacción puede separar "condiciones extremas" en un solo split, en lugar de necesitar dos splits anidados (temp > X AND hum < Y).

### 3c. `split_data()` — Separar train/test

```python
X_train, X_test, y_train, y_test = train_test_split(X_pd, y_np, test_size=0.2, random_state=42)
```

Resultado:
- **Train:** 17840 muestras (80%)
- **Test:** 4460 muestras (20%)

El random_state=42 asegura que siempre sea la misma partición, para que los resultados sean reproducibles.

### 3d. `apply_target_encoding()` — El truco más importante

```python
def apply_target_encoding(X_train, X_test, y_train):
    alpha = 10

    # Para cada grupo, calcular la media del target con smoothing
    for col, out_name in [
        ("provincia", "provincia_te"),
        ("_provincia_mes", "provincia_mes_te"),
        ("cc_aa", "cc_aa_te"),
        ("_cc_aa_mes", "cc_aa_mes_te"),
    ]:
        stats = grupo_por_columna
        global_mean = y_train.mean()
        stats["mean"] = (stats["sum"] + global_mean * alpha) / (stats["count"] + alpha)
        X_train[out_name] = X_train[col].map(mapping)
        X_test[out_name] = X_test[col].map(mapping).fillna(global_mean)
```

**¿Qué es target encoding?**

En lugar de usar "provincia" como categoría (39 columnas one-hot), la reemplazamos por UN SOLO número: la media histórica de superficie quemada para esa provincia.

**Ejemplo:**
- En Galicia los incendios queman mucho de media (mucha vegetación) → provincia_te ≈ valor alto
- En Madrid los incendios queman poco de media (menos vegetación) → provincia_te ≈ valor bajo

El modelo aprende: "si provincia_te es alto, este incendio parte con ventaja para ser grande".

**¿Qué es el smoothing (α=10)?**

Si una provincia tiene pocos incendios en los datos, su media podría ser ruidosa. El smoothing "tira" de la media hacia la media global:

```
valor_final = (suma_del_grupo + media_global × α) / (cuenta_del_grupo + α)
```

- Si hay muchas muestras: la media del grupo pesa más
- Si hay pocas muestras: se acerca a la media global (menos ruido)

**¿Por qué provincia Y cc_aa?**
- `provincia_te`: 39 grupos, más granular
- `cc_aa_te`: 17 grupos, más estable (más muestras por grupo)
- Cada una captura el riesgo regional a distinta escala
- Además se crean `provincia_mes_te` y `cc_aa_mes_te` para capturar la estacionalidad regional (ej: "Galicia en agosto es mucho más peligroso que Galicia en enero")

**Las columnas originales** (`provincia`, `cc_aa`) se eliminan después, porque ya han sido transformadas.

**Los mappings** se guardan (`target_encoding.joblib`) para poder aplicar la misma transformación en predicciones nuevas.

### 3e. `LogTransformer()` — Transformar el target

```python
class LogTransformer:
    def transform(self, y):
        return np.log1p(y)       # log(1 + y)

    def inverse_transform(self, y):
        return np.expm1(y)       # exp(y) - 1
```

**Por qué:** El target original (hectáreas quemadas) tiene una distribución muy asimétrica — muchos valores pequeños y pocos enormes. El logaritmo comprime la escala y hace que la distribución se parezca más a una normal. Los modelos de regresión funcionan mejor cuando el target tiene distribución simétrica.

**Ejemplo:**
| Valor real | log1p |
|------------|-------|
| 0.5 ha     | 0.405 |
| 2 ha       | 1.099 |
| 10 ha      | 2.398 |

Después de predecir, se aplica `expm1` para volver a hectáreas.

### 3f. `build_preprocessor()` — ColumnTransformer

```python
ct = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUMERIC),        # 18 columnas
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CATEGORICAL),  # 2 columnas
])
```

**Esto hace dos cosas:**

**Para las 18 columnas numéricas** (altitud, latitud, longitud, temperatura_media, precipitacion, humedad_media, viento, racha, sol, dir_sin, dir_cos, temp_hum_interaction, temp_wind_interaction, hum_wind_interaction, provincia_te, provincia_mes_te, cc_aa_te, cc_aa_mes_te):
- Aplica `StandardScaler`: resta la media y divide por la desviación típica
- Así todas las variables tienen media 0 y desviación 1
- Necesario porque las escalas son distintas (altitud: 0-2000m, temperatura: 0-40°C)

**Para las 2 columnas categóricas** (`causa_incendio`, `mes`):
- Aplica `OneHotEncoder`: crea columnas binarias (0/1) para cada categoría
- `causa_incendio` (6 valores) → 6 columnas
- `mes` (12 valores) → 12 columnas
- Total: 18 columnas one-hot

**Feature total: 18 + 18 = 36**

`handle_unknown="ignore"` permite que si en predicción aparece una categoría nueva, no se rompa (la ignora).

### 3g. Guardar artefactos

```python
joblib.dump(preprocessor, PREPROCESSOR_PATH)        # models/preprocessor.joblib
joblib.dump(target_transformer, TARGET_TRANSFORMER_PATH)  # models/target_transformer.joblib
joblib.dump(te_mappings, TARGET_ENCODING_PATH)       # models/target_encoding.joblib
```

**¿Qué se guarda y para qué?**

| Archivo | Contenido | Tamaño | Para qué sirve |
|---------|-----------|--------|----------------|
| `preprocessor.joblib` | ColumnTransformer ajustado (medias, desviaciones, categorías) | ~3.5 KB | Transformar inputs nuevos con la misma receta |
| `target_transformer.joblib` | LogTransformer (es solo una función) | 51 bytes | Deshacer el log1p en predicciones |
| `target_encoding.joblib` | Diccionario con medias por grupo | ~10 KB | Aplicar target encoding a inputs nuevos |

**Nota:** Se usa `joblib` en lugar de `pickle` porque es más eficiente con arrays grandes de numpy/sklearn.

---

## PASO 4: FASE 2 — Entrenamiento (`src/train.py`)

### 4a. `run_training()` — La función general

Recibe los datos ya transformados (X_train 17840×36, X_test 4460×36) y entrena 4 modelos.

### 4b. Linear Regression (`train_linear_regression`)

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

El modelo más simple. Asume relación lineal entre features y target. Sirve como **línea base** — si un modelo complejo no supera a este, algo va mal.

### 4c. Random Forest (`train_random_forest`)

```python
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
)
```

Un bosque de 200 árboles de decisión. Cada árbol hace splits en los datos, y el resultado final es el promedio de todos los árboles.

**Hiperparámetros:**
- `max_depth=8`: cada árbol tiene profundidad máxima 8 (evita que memorize)
- `min_samples_leaf=10`: cada hoja tiene al menos 10 muestras (evita overfitting)
- `n_estimators=200`: 200 árboles (suficientes para converger)

### 4d. XGBoost con Optuna (`optimize_xgboost` + entrenamiento)

**Fase 1: Optimización de hiperparámetros**

```python
def optimize_xgboost(X_train, y_train, target_transformer, n_trials=30):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        # Calcular R² en train y val (en escala original, deshaciendo log1p)
        r2_tr = r2_score(y_tr_orig, y_pred_tr_orig)
        r2_val = r2_score(y_val_orig, y_pred_val_orig)
        overfitting = max(0, r2_tr - r2_val - 0.05)
        
        return r2_val - 2.0 * overfitting  # Penaliza el overfitting
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params
```

Optuna prueba 30 combinaciones de 8 hiperparámetros:

| Parámetro | Rango | Efecto |
|-----------|-------|--------|
| `n_estimators` | 100-400 | Más árboles = mejor aprendizaje, pero más lento |
| `max_depth` | 3-5 | Más profundidad = más complejidad (riesgo overfitting) |
| `learning_rate` | 0.01-0.15 (log) | Más pequeño = cada árbol aporta menos, se necesitan más |
| `subsample` | 0.6-1.0 | Fracción de filas por árbol (regularización) |
| `colsample_bytree` | 0.5-1.0 | Fracción de columnas por árbol (regularización) |
| `reg_alpha` | 0.5-5 | Regularización L1 (castiga pesos grandes → features menos importantes a cero) |
| `reg_lambda` | 0.5-5 | Regularización L2 (castiga pesos grandes → los reduce) |
| `min_child_weight` | 3-10 | Mínimo de muestras por hoja (evita overfitting) |

**La función objetivo penaliza el overfitting:**
```
objetivo = r2_val - 2 × max(0, r2_train - r2_val - 0.05)
```

Si el R² en entrenamiento supera al de validación en más de 0.05, se penaliza el doble. Así Optuna busca modelos que generalicen bien.

**Fase 2: Entrenamiento final**

```python
xgb_params = {**best_xgb_params, "early_stopping_rounds": 15}
xgb = XGBRegressor(**xgb_params)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

Con los mejores parámetros encontrados, se entrena el modelo final usando todo el train y se evalúa en test. `early_stopping_rounds=15` detiene el entrenamiento si no mejora en 15 rondas.

### 4e. LightGBM (`train_lightgbm`)

```python
model = lgb.LGBMRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1, reg_lambda=2, min_child_samples=5,
)
```

Alternativa a XGBoost. Suele ser más rápido y a veces da mejores resultados. Se dejan hiperparámetros fijos (sin optimización) para ahorrar tiempo.

### 4f. `evaluate_model()` — Evaluación

```python
def evaluate_model(model, X_train, X_test, y_train, y_test, target_transformer, name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Deshacer log1p para tener métricas en hectáreas reales
    y_train_orig = expm1(y_train)
    y_test_orig = expm1(y_test)
    y_train_pred_orig = expm1(y_train_pred)
    y_test_pred_orig = expm1(y_test_pred)
    
    # Calcular métricas
    train_R2 = r2_score(y_train_orig, y_train_pred_orig)
    test_R2 = r2_score(y_test_orig, y_test_pred_orig)
    overfitting = train_R2 - test_R2
```

Métricas que se calculan:
- **MAE** (Error Absoluto Medio): en hectáreas. "De media, el modelo se equivoca en X hectáreas"
- **RMSE** (Raíz del Error Cuadrático Medio): en hectáreas. Penaliza más los errores grandes
- **R²** (Coeficiente de Determinación): entre -∞ y 1. "El modelo explica el X% de la varianza"
- **Overfitting**: diferencia entre R² de train y test. Si es > 0.05, el modelo no generaliza bien

### 4g. Gráficos

Todos se guardan en `reports/figures/`.

**`plot_predictions()`:** Scatter plot de valor real vs predicción. Idealmente los puntos están sobre la línea diagonal.

**`plot_residuals()`:** Histograma de errores (ideal: campana centrada en 0) + scatter de predicción vs error (ideal: nube aleatoria sin patrón).

**`plot_feature_importance()`:** Solo para RF y XGBoost (tienen `feature_importances_`). Muestra las 20 variables más importantes para el modelo.

### 4h. Validación Cruzada (K-Fold)

```python
cross_validate_kfold(
    lambda Xtr, Ytr: XGBRegressor(**best_xgb_params).fit(Xtr, Ytr),
    np.vstack([X_train, X_test]),    # Todos los datos
    np.hstack([y_train, y_test]),    # Todos los targets
    n_splits=5,
)
```

Divide los datos en 5 partes iguales. Entrena en 4 y prueba en 1, rotando 5 veces. Da una estimación más robusta del rendimiento real del modelo (reduce el riesgo de que la partición train/test tenga mala suerte).

### 4i. Comparación y selección del mejor modelo

```python
# Filtrar modelos con overfitting < 5%
for n in model_names:
    overfit = all_metrics[n]["train"]["R2"] - all_metrics[n]["test"]["R2"]
    if overfit < 0.05:
        valid_models[n] = all_metrics[n]["test"]["R2"]

# Elegir el de mayor R² en test entre los que cumplen
best_model_name = max(valid_models, key=valid_models.get)

# Guardar
save_model(final_model, MODEL_PATH)  # models/best_model.joblib
```

**La selección prioriza:**
1. Overfitting < 5% (obligatorio)
2. Mayor R² en test entre los que cumplen

Si ningún modelo cumple, se usa el de mayor R² aunque tenga overfitting.

---

## PASO 5: Predicción en producción (`src/predict.py`)

### 5a. `predict_single(input_data: dict)`

```python
def predict_single(input_data: dict) -> float:
    # 1. Cargar los 4 artefactos guardados
    model = joblib.load("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    target_transformer = joblib.load("models/target_transformer.joblib")
    te_mappings = joblib.load("models/target_encoding.joblib")
    
    # 2. Preparar el input (mismo feature engineering que en training)
    df = pd.DataFrame(input_data)
    df_prepared = prepare_input_for_prediction(df, te_mappings)
    
    # 3. Transformar, predecir, invertir
    X_processed = preprocessor.transform(df_prepared)     # Escalar + OHE
    y_pred_log = model.predict(X_processed)                # Predecir en log
    y_pred = expm1(y_pred_log)                             # Volver a hectáreas
    
    return y_pred[0]
```

**El paso crucial**: `prepare_input_for_prediction()` reproduce EXACTAMENTE el mismo feature engineering que se hizo en training:
1. Deriva `cc_aa` desde `provincia` (tabla provincia→CCAA hardcodeada)
2. Calcula interacciones (temp×hum, temp×wind, hum×wind)
3. Aplica target encoding usando los mappings guardados
4. Elimina columnas originales (provincia, cc_aa, dir)

Si este paso no replica exactamente el training, las columnas no coincidirán y el `preprocessor.transform()` fallará.

### 5b. `prepare_input_for_prediction()` — Mapa provincia → CCAA

```python
PROVINCIA_TO_CCAA = {
    "ALAVA": "PAIS VASCO",
    "ALBACETE": "CASTILLA-LA MANCHA",
    "A CORUÑA": "GALICIA",
    "MADRID": "MADRID",
    ...
}
```

52 entradas que mapean cada provincia a su comunidad autónoma. Se usa cuando el usuario proporciona provincia pero no CCAA.

---

## PASO 6: Interfaz Streamlit (`app.py`)

### 6a. Carga de artefactos

```python
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    target_transformer = joblib.load("models/target_transformer.joblib")
    te_mappings = joblib.load("models/target_encoding.joblib")
    return model, preprocessor, target_transformer, te_mappings
```

`@st.cache_resource` asegura que los modelos se cargan solo una vez y se reutilizan en todas las interacciones del usuario.

### 6b. Inputs del usuario (14 campos)

En la barra lateral:

**Ubicación:**
- Provincia (selectbox con 39 provincias)
- Altitud (número)
- Latitud (número, default Madrid)
- Longitud (número, default Madrid)

**Meteorología:**
- Temperatura media (°C)
- Precipitación (mm)
- Humedad relativa media (%)
- Velocidad del viento media (m/s)
- Racha máxima de viento (m/s)
- Horas de sol (0-24)
- Dirección del viento (slider 0-36)

**Contexto:**
- Causa del incendio (selectbox: rayo, quema agrícola, etc.)
- Mes (selectbox 1-12)

### 6c. Flujo de predicción en la app

```python
# 1. Calcular dir_sin/dir_cos desde el slider
direccion_rad = dir_viento * 10 * π / 180
dir_sin = sin(direccion_rad)
dir_cos = cos(direccion_rad)

# 2. Deriva CCAA desde provincia
cc_aa = PROVINCIA_TO_CCAA.get(provincia, "OTRAS")

# 3. Crear DataFrame con inputs
input_data = pd.DataFrame([{
    "altitud": ..., "latitud": ..., "longitud": ...,
    "temperatura_media": ..., "precipitacion": ...,
    "humedad_relativa_media": ..., "velocidad_viento_media": ...,
    "racha_maxima_viento": ..., "sol": ...,
    "dir_sin": ..., "dir_cos": ...,
    "provincia": ..., "cc_aa": ..., "causa_incendio": ..., "mes": ...,
}])

# 4. Feature engineering (replica training)
df = prepare_input_for_prediction(input_data, te_mappings)

# 5. Transformar y predecir
X_processed = preprocessor.transform(df)
y_pred_log = model.predict(X_processed)
y_pred = expm1(y_pred_log)  # Hectáreas reales

# 6. Mostrar resultado
st.metric(label="Superficie Quemada Estimada", value=f"{y_pred:.2f} ha")

# 7. Clasificar por gravedad
if y_pred < 3:     gravedad = "Conato"
elif y_pred < 10:  gravedad = "Pequeño"
elif y_pred < 50:  gravedad = "Mediano"
else:              gravedad = "Grande"
```

---

## MAPA COMPLETO: De principio a fin

```
┌─────────────────────────────────────────────────────────────────┐
│ DATOS CRUDOS                                                    │
│ scripts/fetch_aemet.py → CSVs meteorológicos                    │
│ scripts/merge_datasets.py → fires_weather_merged.parquet (21 cols, 22300 filas) │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ main.py (configura logging, silencia Optuna)                    │
│   → pipeline.main()                                             │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 1: preprocess.run_preprocessing()                          │
│                                                                  │
│ 1. Cargar parquet                                   (L152)      │
│ 2. Capping target a 10 ha                          (L155-158)   │
│ 3. prepare_features():                             (L160-162)   │
│    a. encode_dir_circular(): dir → dir_sin/cos     (L51-57)     │
│    b. lat/long string → float                      (L63-65)     │
│    c. Dropear columnas inútiles                    (L67)        │
│    d. Imputar nulos (sol)                          (L69-81)     │
│    e. Crear interacciones                          (L83-87)     │
│ 4. split_data(): 80/20 train/test                  (L164-165)   │
│ 5. apply_target_encoding():                       (L167-168)    │
│    a. provincia → provincia_te                    (L107)        │
│    b. provincia+mes → provincia_mes_te            (L108)        │
│    c. cc_aa → cc_aa_te                           (L109)        │
│    d. cc_aa+mes → cc_aa_mes_te                   (L110)        │
│    e. Eliminar provincia y cc_aa originales       (L120-121)    │
│ 6. LogTransformer: y = log1p(target)              (L170-174)    │
│ 7. build_preprocessor():                          (L176-181)    │
│    a. StandardScaler en 18 numéricas              (L144)        │
│    b. OneHotEncoder en 2 categóricas              (L145)        │
│ 8. joblib.dump() 3 artefactos                     (L183-191)    │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 2: train.run_training()                                    │
│                                                                  │
│ 1. LinearRegression()                              (L247-254)   │
│    → evaluate → plot_predictions → plot_residuals                │
│ 2. RandomForest(max_depth=8)                       (L256-264)   │
│    → evaluate → plot → feature_importance                        │
│ 3. optimize_xgboost(30 trials)                     (L266)       │
│    XGBoost(best_params)                            (L268-278)   │
│    → evaluate → plot → feature_importance                        │
│ 4. LightGBM                                        (L280-287)   │
│    → evaluate → plot                                              │
│ 5. cross_validate_kfold(5 folds)                   (L289-297)   │
│ 6. Comparación: tabla de métricas                  (L299-313)   │
│ 7. Selección: mejor R² con overfitting < 5%       (L315-332)    │
│ 8. joblib.dump(best_model) → models/best_model.joblib (L336-337)│
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ app.py (Streamlit)                                              │
│                                                                  │
│ 1. Cargar 4 artefactos de models/                  (L40-46)     │
│ 2. Mostrar 14 inputs en sidebar                    (L55-90)     │
│ 3. Al hacer clic en "Predecir":                    (L92-157)    │
│    a. Construir DataFrame con inputs                             │
│    b. prepare_input_for_prediction():                            │
│       - Derivar cc_aa desde provincia                           │
│       - Calcular interacciones                                   │
│       - Aplicar target encoding                                 │
│    c. preprocessor.transform()                                   │
│    d. model.predict()                                            │
│    e. expm1() → hectáreas                                        │
│    f. Mostrar métrica + gravedad                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## DÓNDE ESTÁ CADA COSA

| Qué quieres ver | Archivo | Línea(s) |
|-----------------|---------|----------|
| Punto de entrada | `main.py` | 11-12 |
| Orquestador del pipeline | `src/pipeline.py` | 10-38 |
| Configuración global | `src/config.py` | todas |
| Carga del dataset | `src/preprocess.py` | 152 |
| Capping del target | `src/preprocess.py` | 155-158 |
| Feature engineering (interacciones, dir, imputación) | `src/preprocess.py` | 60-92 |
| Target encoding | `src/preprocess.py` | 95-128 |
| Split train/test | `src/preprocess.py` | 131-139 |
| ColumnTransformer | `src/preprocess.py` | 142-148 |
| LogTransformer | `src/preprocess.py` | 29-34 |
| Guardar artefactos | `src/preprocess.py` | 183-191 |
| Linear Regression | `src/train.py` | 22-25 |
| Random Forest | `src/train.py` | 28-35 |
| LightGBM | `src/train.py` | 38-46 |
| Optuna optimization | `src/train.py` | 177-224 |
| XGBoost final | `src/train.py` | 266-278 |
| K-Fold CV | `src/train.py` | 49-75 |
| Evaluación (métricas) | `src/train.py` | 78-113 |
| Comparación/Selección | `src/train.py` | 299-337 |
| Gráficos | `src/train.py` | 116-174 |
| Modelos guardados | `models/` | todos |
| Feature engineering en producción | `src/predict.py` | 67-100 |
| predict_single() | `src/predict.py` | 103-114 |
| Mapa provincia → CCAA | `src/predict.py` | 35-55 |
| App Streamlit | `app.py` | todas |
| Inputs del usuario | `app.py` | 55-90 |
| Predicción en app | `app.py` | 92-157 |
