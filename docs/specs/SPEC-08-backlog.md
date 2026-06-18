# SPEC-08: Backlog del Proyecto

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-08 |
| **Estado** | 🟡 Borrador (pendiente confirmación equipo) |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | — |
| **Bloqueado por** | SPEC-01, SPEC-02, SPEC-07 |

## Contexto
Backlog inicial para el Sprint 1, priorizado por dependencias y valor.

## Especificación Técnica

### Sprint 1 — Backlog

| ID | Título | Story Points | Prioridad | Bloqueado por | Asignado |
|----|--------|-------------|-----------|---------------|----------|
| US-01 | Inventario de datos AEMET: periodo, estaciones, variables, cobertura | 2 | P0 | — | Pendiente |
| US-02 | Merge espacial fires-AEMET: asignar estación a cada incendio | 3 | P0 | US-01 | Pendiente |
| US-03 | Crear dataset de entrenamiento (municipio + día → features + target) | 5 | P0 | US-02 | Pendiente |
| US-04 | Análisis de desbalanceo de clases | 2 | P0 | US-03 | Pendiente |
| US-05 | EDA conjunto: correlaciones clima-incendio, estacionalidad, mapa histórico | 3 | P1 | US-03 | Pendiente |
| US-06 | Modelo baseline: Logistic Regression con features meteorológicas | 3 | P1 | US-04 | Pendiente |
| US-07 | Documentar pipeline de datos en `docs/data-pipeline.md` | 2 | P2 | US-03 | Pendiente |

### Definición de User Stories

#### US-01: Inventario de datos AEMET
**Como** data engineer, **quiero** conocer el contenido completo del dataset AEMET **para que** pueda planificar el merge con fires-all.

**Criterios de aceptación:**
- [ ] Listado de variables meteorológicas disponibles
- [ ] Rango de fechas del dataset
- [ ] Número de estaciones y su cobertura geográfica
- [ ] Formato de los archivos (CSV, JSON, etc.)
- [ ] Documento resumen subido a `docs/aemet-inventory.md`

#### US-02: Merge espacial fires-AEMET
**Como** data engineer, **quiero** asignar datos meteorológicos a cada incendio **para que** pueda entrenar el modelo con features climáticas.

**Criterios de aceptación:**
- [ ] Cada incendio tiene asignada una estación AEMET (la más cercana)
- [ ] Documentación del método de asignación
- [ ] Análisis de pérdida de datos (% incendios sin estación cercana)
- [ ] Dataset intermedio guardado en `data/processed/fires_weather_merged.csv`

#### US-03: Dataset de entrenamiento
**Como** ML engineer, **quiero** un dataset estructurado por municipio y día **para que** pueda entrenar modelos de clasificación.

**Criterios de aceptación:**
- [ ] Dataset con una fila por municipio y día
- [ ] Features meteorológicas agregadas
- [ ] Variable target binaria (`incendio_ocurrido`)
- [ ] Features temporales (mes, estación)
- [ ] Features históricas (incendios previos)
- [ ] Guardado en `data/processed/train.parquet`

#### US-04: Análisis de desbalanceo
**Como** ML engineer, **quiero** cuantificar el desbalanceo de clases **para que** pueda elegir la estrategia adecuada.

**Criterios de aceptación:**
- [ ] % de días/municipios con incendio vs. sin incendio
- [ ] Visualización de la distribución
- [ ] Análisis de estrategias posibles (SMOTE, class_weight, etc.)
- [ ] Decisión documentada y justificada

#### US-05: EDA conjunto
**Como** equipo, **quiero** visualizar las relaciones entre clima e incendios **para que** podamos entender el problema antes de modelar.

**Criterios de aceptación:**
- [ ] Correlación entre variables meteorológicas y ocurrencia de incendio
- [ ] Distribución temporal (por mes, estación)
- [ ] Mapa de calor de incendios por provincia/municipio
- [ ] Boxplots de temperatura/viento en días con/sin incendio
- [ ] Figuras guardadas en `results/figures/`

#### US-06: Modelo baseline
**Como** ML engineer, **quiero** un modelo simple que prediga riesgo de incendio **para que** tenga una referencia para comparar modelos avanzados.

**Criterios de aceptación:**
- [ ] Logistic Regression entrenado y evaluado
- [ ] Métricas calculadas: Precision, Recall, F1, AUC-ROC, AUC-PR
- [ ] Comparación train/val con overfitting < 5%
- [ ] Modelo serializado en `results/models/baseline.pkl`
- [ ] Métricas registradas en `results/metrics.json`

#### US-07: Documentación del pipeline
**Como** equipo, **quiero** documentar cómo se construye el dataset **para que** el proceso sea reproducible.

**Criterios de aceptación:**
- [ ] Documento `docs/data-pipeline.md` con flujo completo
- [ ] Diagrama del pipeline (puede ser texto o imagen)
- [ ] Instrucciones para regenerar el dataset desde cero

## Criterios de Aceptación del Backlog
- [ ] Todas las US del Sprint 1 están creadas como Issues en GitHub
- [ ] Cada US tiene labels correctos (`sprint:1`, `priority:p0/p1/p2`, `spec:backlog`)
- [ ] Cada US tiene asignado un responsable
- [ ] El equipo ha validado la priorización en planning meeting

## Consecuencias
- Backlog visible y trackeable en GitHub
- Cada miembro del equipo sabe qué hacer y en qué orden

## Notas
- Las US se crearán como GitHub Issues cuando el equipo confirme
- Template de Issue disponible en `.github/ISSUE_TEMPLATE/user-story.md`
- Story points son estimaciones; se ajustarán en la retrospectiva del Sprint 1
