# SPEC-07: Planificación de Sprints

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-07 |
| **Estado** | 🟡 Borrador |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | SPEC-08 |
| **Bloqueado por** | SPEC-02 |

## Contexto
El proyecto tiene 2 semanas (10 días laborables). Necesitamos distribuir el trabajo en sprints manejables.

## Decisión
3 sprints de duración variable, con buffer para imprevistos.

## Especificación Técnica

### Sprint 1: Fundamentos (Días 1-4)
**Objetivo:** Tener un dataset de entrenamiento funcional y un modelo baseline.

| Día | Focus |
|-----|-------|
| 1 | Recepción AEMET, merge fires-AEMET, EDA conjunto |
| 2 | Feature engineering, análisis de desbalanceo |
| 3 | Modelo baseline (Logistic Regression), métricas |
| 4 | Visualizaciones, documentación, demo interna |

**Entregable:** Notebook 05 ejecutable con métricas del baseline.

### Sprint 2: Modelado Avanzado (Días 5-8)
**Objetivo:** Mejorar el modelo con técnicas avanzadas y preparar la app.

| Día | Focus |
|-----|-------|
| 5 | Ensemble models (Random Forest, XGBoost) |
| 6 | Optuna para optimización de hiperparámetros |
| 7 | Validación cruzada temporal, análisis de umbral |
| 8 | App Streamlit con mapa de riesgo |

**Entregable:** App Streamlit funcional + modelo optimizado.

### Sprint 3: Productivización (Días 9-10)
**Objetivo:** Pulir, documentar y desplegar.

| Día | Focus |
|-----|-------|
| 9 | Docker (si se alcanza nivel avanzado), tests, documentación |
| 10 | Deploy, presentación, README final |

**Entregable:** Repositorio completo, app desplegada (o ejecutable local), presentación.

### Buffer
- 2 días de margen distribuidos entre sprints para imprevistos
- Si el merge AEMET falla, Sprint 1 se extiende y Sprint 2 se comprime

## Criterios de Aceptación
- [ ] Cada sprint tiene objetivo claro y entregable definido
- [ ] Daily standups registrados (aunque sea en GitHub Discussions)
- [ ] Retrospectiva al final de cada sprint

## Consecuencias
- Planificación realista con margen para problemas técnicos
- El merge AEMET es el punto crítico; si falla, todo el plan se resiente

## Notas
- Fechas concretas dependen de la fecha de inicio oficial del proyecto
- Recomendación: usar GitHub Projects (tablero Kanban) para visualizar el progreso
