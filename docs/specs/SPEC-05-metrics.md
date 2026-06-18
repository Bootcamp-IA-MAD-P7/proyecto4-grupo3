# SPEC-05: Métricas de Éxito

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-05 |
| **Estado** | 🟢 Aprobado |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | SPEC-08 (US-05, US-06) |
| **Bloqueado por** | SPEC-01, SPEC-02 |

## Contexto
En un problema de clasificación con clases desbalanceadas (incendios son eventos raros), las métricas clásicas como accuracy son engañosas.

## Decisión
Usar un panel de métricas orientadas a la detección de eventos raros, con énfasis en Recall (no perder incendios reales) manteniendo Precision razonable.

## Especificación Técnica

### Métricas Obligatorias

| Métrica | Fórmula/Definición | Umbral Mínimo | Objetivo | Por qué importa |
|---------|-------------------|---------------|----------|-----------------|
| **Precision** | TP / (TP + FP) | > 0.30 | > 0.50 | Evitar falsas alarmas (coste de despliegue) |
| **Recall** | TP / (TP + FN) | > 0.70 | > 0.85 | No perder incendios reales (coste humano) |
| **F1-Score** | 2 * (P * R) / (P + R) | > 0.45 | > 0.65 | Balance precisión/recall |
| **AUC-ROC** | Área bajo curva ROC | > 0.75 | > 0.90 | Capacidad discriminativa general |
| **AUC-PR** | Área bajo curva Precision-Recall | > 0.40 | > 0.60 | Crítica con clases desbalanceadas |

### Métricas Adicionales (Nivel Medio+)

| Métrica | Uso |
|---------|-----|
| **Calibration curve** | Verificar que probabilidades son confiables |
| **Brier Score** | Medir calidad de las probabilidades |
| **Matriz de confusión por umbral** | Analizar trade-offs a diferentes puntos de corte |
| **Feature Importance** | Explicabilidad del modelo |
| **SHAP values** | Explicabilidad por predicción individual |

### Estrategia ante Desbalanceo
- Clase positiva (incendio): ~1-5% del dataset estimado
- Técnicas a evaluar: `class_weight='balanced'`, SMOTE, undersampling, focal loss
- Umbral de decisión NO es 0.5; se optimiza con Precision-Recall curve

## Criterios de Aceptación
- [ ] Todas las métricas obligatorias calculadas y documentadas
- [ ] Comparativa de métricas entre modelos (baseline vs. ensemble)
- [ ] Análisis de umbral óptimo justificado
- [ ] Métricas en train/val/test con overfitting < 5%

## Consecuencias
- El modelo puede tener precision baja pero recall alto (aceptable para prevención)
- El umbral de alerta será configurable en la app Streamlit
- Las métricas se guardan en `results/metrics.json` para tracking

## Notas
- Template de registro de métricas:
```json
{
  "modelo": "xgboost_v1",
  "fecha_entrenamiento": "2026-06-25",
  "precision": 0.42,
  "recall": 0.78,
  "f1": 0.54,
  "auc_roc": 0.82,
  "auc_pr": 0.48,
  "umbral": 0.35,
  "overfitting_pct": 3.2
}
```
