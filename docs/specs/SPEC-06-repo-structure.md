# SPEC-06: Estructura del Repositorio

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-06 |
| **Estado** | 🟢 Aprobado |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | — |
| **Bloqueado por** | SPEC-01, SPEC-02 |

## Contexto
Necesitamos una estructura de carpetas clara que facilite el trabajo colaborativo y cumpla con los requisitos del bootcamp.

## Decisión
Adoptar la siguiente estructura basada en proyectos de ciencia de datos en producción.

## Especificación Técnica

```
fire-risk-prediction/
├── 📁 .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── user-story.md
│   │   ├── bug-report.md
│   │   └── research-task.md
│   ├── workflows/
│   │   └── ci.yml              ← tests automáticos (nivel avanzado)
│   └── pull_request_template.md
│
├── 📁 data/
│   ├── raw/
│   │   ├── fires-all.csv
│   │   └── aemet/              ← datos meteorológicos brutos
│   ├── processed/
│   │   ├── train.parquet       ← dataset de entrenamiento
│   │   └── test.parquet
│   └── external/               ← datos de terceros si aplica
│
├── 📁 docs/
│   ├── specs/                  ← ← ← SPECS DEL PROYECTO
│   │   ├── SPEC-01-objective.md
│   │   ├── SPEC-02-scope.md
│   │   ├── SPEC-03-dod.md
│   │   ├── SPEC-04-data-architecture.md
│   │   ├── SPEC-05-metrics.md
│   │   ├── SPEC-06-repo-structure.md
│   │   ├── SPEC-07-sprints.md
│   │   └── SPEC-08-backlog.md
│   ├── decisions/              ← ADRs (Architecture Decision Records)
│   │   └── 001-granularidad-municipio.md
│   ├── features.md             ← documentación de features
│   └── data-pipeline.md        ← documentación del pipeline de datos
│
├── 📁 notebooks/
│   ├── 01_eda_fires.ipynb
│   ├── 02_eda_aemet.ipynb
│   ├── 03_merge_datasets.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling_baseline.ipynb
│   ├── 06_modeling_ensemble.ipynb
│   ├── 07_optuna.ipynb
│   └── 08_threshold_analysis.ipynb
│
├── 📁 src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_fires.py
│   │   ├── load_aemet.py
│   │   ├── build_dataset.py    ← crea train/test
│   │   └── merge.py            ← merge espaciotemporal
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal.py         ← features de fecha
│   │   ├── weather.py          ← features meteorológicas
│   │   └── historical.py       ← features históricas de incendios
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py         ← Logistic Regression, Random Forest
│   │   ├── ensemble.py         ← XGBoost, LightGBM
│   │   └── threshold.py        ← selección de umbral óptimo
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          ← cálculo de métricas
│       ├── calibration.py      ← curvas de calibración
│       └── plots.py            ← visualizaciones de evaluación
│
├── 📁 app/
│   ├── streamlit_app.py        ← aplicación principal
│   ├── alert_config.py         ← configuración de umbrales
│   └── utils.py                ← funciones auxiliares para la app
│
├── 📁 api/                     ← (nivel avanzado)
│   └── main.py                 ← FastAPI para predicciones
│
├── 📁 tests/                   ← (nivel avanzado)
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_metrics.py
│
├── 📁 docker/                  ← (nivel avanzado)
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📁 results/
│   ├── metrics.json            ← registro de métricas de modelos
│   ├── models/                 ← modelos serializados (.pkl, .joblib)
│   └── figures/                ← gráficos de evaluación
│
├── .gitignore
├── requirements.txt
├── environment.yml             ← (opcional, para conda)
└── README.md
```

## Reglas de Uso

| Carpeta | Regla |
|---------|-------|
| `data/raw/` | NUNCA se modifica manualmente. Solo lectura. |
| `data/processed/` | Generado por scripts, no commitear archivos grandes (usar .gitignore) |
| `notebooks/` | Cada notebook numerado y con nombre descriptivo. Output limpio al commitear. |
| `src/` | Código Python modular, importable. No código ad-hoc. |
| `app/` | Solo código de la aplicación Streamlit. No entrenamiento aquí. |
| `docs/specs/` | Las specs se versionan con el código. Cada PR puede referenciarlas. |

## Criterios de Aceptación
- [ ] Estructura creada en el repo de GitHub
- [ ] `.gitignore` configurado (excluye `data/`, `results/models/`, `.ipynb_checkpoints/`)
- [ ] `requirements.txt` inicial con dependencias básicas
- [ ] README.md con descripción del proyecto y cómo ejecutar

## Consecuencias
- Estructura clara reduce conflictos de merge
- Separación de notebooks (experimentación) y src (producción)
- Facilita la dockerización futura

## Notas
- Los notebooks deben ser ejecutables de arriba a abajo (reproducibilidad)
- Los scripts de `src/` deben tener docstrings y type hints
