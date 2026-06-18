# SPEC-02: Alcance y Niveles de Entrega

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-02 |
| **Estado** | 🟢 Aprobado |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | SPEC-06, SPEC-08 |
| **Bloqueado por** | SPEC-01 |

## Contexto
El proyecto tiene 2 semanas de plazo y 4 niveles de entrega definidos por el bootcamp.

## Decisión
Apuntar al nivel 🟡 Medio como mínimo garantizado, con avance hacia 🟠 Avanzado si el tiempo lo permite. El nivel 🔴 Experto queda fuera del alcance por limitación temporal.

## Especificación Técnica

### 🟢 Nivel Esencial (Obligatorio)
| Requisito | Estado |
|-----------|--------|
| Modelo de ML funcional (clasificación) | Pendiente |
| Análisis exploratorio de datos (EDA) con visualizaciones | Pendiente |
| Overfitting inferior al 5% | Pendiente |
| Aplicación productiva (Streamlit) | Pendiente |
| Informe de rendimiento con métricas (Precision, Recall, F1, AUC-ROC, AUC-PR) | Pendiente |

### 🟡 Nivel Medio (Objetivo)
| Requisito | Status |
|-----------|--------|
| Modelo ensemble (Random Forest, XGBoost, LightGBM) | Pendiente |
| Validación cruzada temporal (TimeSeriesSplit) | Pendiente |
| Optimización de hiperparámetros (Optuna) | Pendiente |
| Sistema de recogida de feedback | Pendiente |
| Pipeline de ingestión de datos nuevos | Pendiente |

### 🟠 Nivel Avanzado (Stretch Goal)
| Requisito | Status |
|-----------|--------|
| Dockerización | Pendiente |
| Base de datos (PostgreSQL/MongoDB) | Pendiente |
| Deploy en Render/AWS | Pendiente |
| Tests unitarios | Pendiente |

### 🔴 Nivel Experto (Fuera de alcance)
- MLOps, A/B testing, data drift, auto-reemplazo de modelos

## Criterios de Aceptación
- [ ] Todos los requisitos del nivel Esencial están completados
- [ ] Al menos 3 requisitos del nivel Medio están completados
- [ ] El nivel Avanzado se documenta como roadmap futuro

## Consecuencias
- El equipo debe priorizar el Esencial antes de avanzar al Medio
- La validación temporal es crítica (no usar train_test_split aleatorio)
- Optuna consume tiempo de computación; reservar recursos

## Notas
- Plazo: 2 semanas desde 2026-06-18
- Tecnologías: Jupyter, Python, sklearn, XGBoost, Optuna, Streamlit, Docker (opcional)
