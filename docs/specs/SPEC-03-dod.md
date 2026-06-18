# SPEC-03: Definition of Done (DoD)

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-03 |
| **Estado** | 🟢 Aprobado |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | Todas las US del backlog |
| **Bloqueado por** | SPEC-01, SPEC-02 |

## Contexto
Como equipo que nunca ha trabajado con Spec-Driven Development, necesitamos un DoD claro y compartido para todas las tareas.

## Decisión
Toda User Story o tarea técnica se considera "Done" cuando cumple los 5 criterios siguientes.

## Especificación Técnica

### Checklist DoD
- [ ] **1. Código revisado:** Al menos 1 compañero ha aprobado el PR en GitHub
- [ ] **2. Tests pasan:** Tests unitarios ejecutan sin errores (si aplica nivel Avanzado)
- [ ] **3. Documentación actualizada:** Notebook o markdown explica qué se hizo y por qué
- [ ] **4. Métricas registradas:** Resultados guardados en `results/metrics.json` o similar
- [ ] **5. Demo funcional:** El código ejecuta end-to-end sin errores manuales

### DoD por Tipo de Tarea

| Tipo | Criterios Adicionales |
|------|----------------------|
| **EDA** | Visualizaciones exportadas a `reports/figures/`, insights documentados |
| **Feature Engineering** | Features documentados en `docs/features.md`, correlaciones analizadas |
| **Modelo** | Métricas en train/val/test, overfitting < 5%, modelo serializado |
| **App** | Streamlit ejecutable con `streamlit run app/streamlit_app.py` |
| **Deploy** | URL funcional, Dockerfile builda sin errores |

## Criterios de Aceptación
- [ ] Todo el equipo ha leído y aceptado este DoD
- [ ] El DoD está visible en el README del repo
- [ ] Cada PR template incluye el checklist DoD

## Consecuencias
- Mayor calidad de código entregado
- Menos deuda técnica acumulada
- Revisión de código obligatoria (puede ralentizar en días de mucha carga)

## Notas
- Template de PR disponible en `.github/pull_request_template.md`
- En caso de urgencia, el Scrum Master puede aprobar excepciones documentadas
