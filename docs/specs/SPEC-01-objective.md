# SPEC-01: Objetivo del Proyecto

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-01 |
| **Estado** | 🟢 Aprobado |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | SPEC-02, SPEC-03, SPEC-04, SPEC-05, SPEC-06, SPEC-07, SPEC-08 |
| **Bloqueado por** | — |

## Contexto
Proyecto grupal del bootcamp de IA. Dataset de incendios forestales en España (fires-all.csv) complementado con datos meteorológicos de AEMET (viento, lluvia, temperatura, humedad).

## Decisión
Construir un sistema de predicción de riesgo de incendio forestal que estime la probabilidad de que ocurra un incendio en una zona geográfica determinada en una fecha concreta, permitiendo la prevención y movilización de recursos antes del siniestro.

## Especificación Técnica
- **Tipo de problema:** Clasificación binaria + scoring de riesgo (0-1)
- **Variable objetivo:** `incendio_ocurrido` (1/0) por zona + fecha
- **Output del modelo:** Probabilidad de incendio (0-100%)
- **Uso previsto:** Mapa de calor de riesgo, alertas por umbral, dashboard de prevención

## Criterios de Aceptación
- [ ] El modelo predice probabilidad de incendio para una zona y fecha dadas
- [ ] El sistema es explicable (feature importance, SHAP values)
- [ ] La aplicación permite visualizar el riesgo en mapa o tabla

## Consecuencias
- Cambio de enfoque respecto a la idea inicial (regresión de superficie quemada → clasificación de riesgo)
- Mayor complejidad en el merge de datos espaciotemporales (fires + AEMET)
- Mayor impacto social del proyecto (prevención vs. análisis post-incendio)

## Notas
- Dataset fires-all: incendios históricos España (años 1968 en adelante)
- Dataset AEMET: datos meteorológicos (pendiente de recepción, 2026-06-19)
- Competencias evaluadas: análisis de datos, ML, visualización, trabajo en equipo
