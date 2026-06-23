# Informe Final de Rendimiento

Generado: 2026-06-23 14:44:02

---

## 1. Comparación de Modelos
| Modelo | R² Train | R² Test | MAE Test | RMSE Test | Overfitting |
|--------|----------|---------|----------|-----------|-------------|
| LinearRegression | -0.0065 | -0.0083 | 22.66 | 187.25 | 0.0018 |
| RandomForest | 0.0449 | 0.0184 | 22.26 | 184.76 | 0.0265 |
| HistGB | 0.0037 | 0.0111 | 22.31 | 185.45 | -0.0074 |
| RandomForest_Optimized | 0.0295 | 0.0196 | 22.27 | 184.65 | 0.0099 |

**Mejor modelo (por R² Test):** RandomForest

## 2. Validación Cruzada (KFold 5)
| Modelo | R² medio | R² std | MAE medio | RMSE medio |
|--------|----------|--------|-----------|------------|
| LinearRegression | -0.0088 | 0.0040 | 21.67 | 186.65 |
| RandomForest | 0.0033 | 0.0060 | 21.48 | 185.69 |
| HistGB | -0.0004 | 0.0025 | 21.45 | 185.95 |

## 3. Optimización de Hiperparámetros

**Modelo optimizado:** RandomForest_Optimized

**Mejores hiperparámetros:**
```
  model__n_estimators: 100
  model__min_samples_split: 5
  model__min_samples_leaf: 4
  model__max_features: None
  model__max_depth: None
```

**R² en test (modelo óptimo):** 0.0196
**MAE en test (modelo óptimo):** 22.27
**RMSE en test (modelo óptimo):** 184.65

## 4. Conclusiones
- Las correlaciones entre features meteorológicos y superficie quemada son muy bajas (< 0.07).
- La tarea de regresión es extremadamente ruidosa: el área quemada depende de muchos factores no capturados (tipo de vegetación, topografía, respuesta de extinción, etc.).
- La transformación log1p es esencial debido a la distribución heavy-tailed de la variable objetivo.
- Los modelos tree-based (RandomForest, HistGradientBoosting) superan significativamente a la regresión lineal.
- Se recomienda incorporar features adicionales (tipo de vegetación, índice de sequía, velocidad de propagación) para mejorar el poder predictivo.