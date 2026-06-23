# Informe de Preprocesamiento y Análisis Exploratorio

Generado: 2026-06-23 14:32:44

---

## 1. Dimensiones del Dataset
- Filas: 22,300
- Columnas: 23

## 2. Variable Objetivo: superficie_quemada
- Media: 23.6946 ha
- Desviación Estándar: 197.0248 ha
- Mínimo: 1.0000 ha
- Máximo: 14953.5166 ha
- Skewness: 38.7134
- Kurtosis: 2162.9325
- Ceros: 0 (0.00%)
- Positivos: 22300

**Decisión:** Aplicar transformación log1p por skewness extremo.

## 3. Calidad de Datos

### [HIGH] redundancy
- fecha and fecha_incendio are identical (columnas duplicadas)
- **Acción:** Eliminar fecha o fecha_incendio

### [HIGH] identifier
- id is a unique row identifier (no aporta valor predictivo)
- **Acción:** Eliminar columna id

### [MEDIUM] wrong_dtype
- latitud es String, debería ser Float. Sample: ['39.301890792211985', '40.48727112950186', '43.20624833194064', '43.19146217178971', '41.995150508651065']
- **Acción:** Convertir a Float64

### [MEDIUM] wrong_dtype
- longitud es String, debería ser Float. Sample: ['-3.358931025029914', '-2.8618169177961565', '-3.583527198225713', '-4.365707586983788', '0.04701083136862927']
- **Acción:** Convertir a Float64

### [HIGH] high_skew
- superficie_quemada tiene skewness=38.71 (extremadamente asimétrica)
- **Acción:** Aplicar transformación log1p

### [INFO] correlation_note
- Todas las correlaciones con el target son muy bajas (<0.07). La superficie quemada puede ser difícil de predecir solo con datos meteorológicos.
- **Acción:** Documentar limitación en el informe

## 4. Coherencia Meteorológica

## 5. Columnas Eliminadas
| Columna | Motivo |
|---------|--------|
| id | Identificador único sin valor predictivo |
| fecha | Duplicada de fecha_incendio |
| fecha_incendio | Fecha del incendio (usamos features derivadas: año, mes) |
| month_name | Redundante con mes (numérico) |
| trimestre | Redundante con mes (derivable) |

## 6. Variables Categóricas
- **año**: 10 valores únicos
- **causa_incendio**: 6 valores únicos
- **cc_aa**: 17 valores únicos
- **id**: 22300 valores únicos
- **latitud**: 22183 valores únicos
- **longitud**: 22183 valores únicos
- **month_name**: 12 valores únicos
- **provincia**: 39 valores únicos

**Decisión:**
- provincia (39): TargetEncoding
- cc_aa (17): TargetEncoding
- causa_incendio (6): OneHotEncoding
- mes (12): Codificación cíclica (seno/coseno)

## 7. Correlaciones con Variable Objetivo (Top 10)
- superficie_quemada: 1.0000
- temperatura_media: 0.0669
- temperatura_minima: 0.0666
- humedad_relativa_maxima: -0.0630
- temperatura_maxima: 0.0625
- humedad_relativa_media: -0.0521
- humedad_relativa_minima: -0.0470
- trimestre: 0.0469
- mes: 0.0448
- racha_maxima_viento: 0.0331

## 8. Transformaciones Aplicadas
- **Target**: log1p(superficie_quemada)
- **Numéricas**: Imputación (mediana) + StandardScaler
- **Geo**: latitud/longitud convertidas a float
- **Categóricas baja cardinalidad**: OneHotEncoding
- **Categóricas alta cardinalidad**: TargetEncoding
- **Cíclicas**: mes → sin(mes), cos(mes)