# Informe de Preprocesamiento y Análisis Exploratorio

Generado: 2026-06-25 13:45:30

---

## 1. Dimensiones del Dataset
- Filas: 22,300
- Columnas: 18

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
- latitud es String, debería ser Float. Sample: ['42.12175405338978', '43.30377824576887', '40.13240919673983', '41.48871624779493', '40.089361687892676']
- **Acción:** Convertir a Float64

### [MEDIUM] wrong_dtype
- longitud es String, debería ser Float. Sample: ['0.009903892163752984', '-5.121906218760324', '-3.594707235914124', '-0.5586123212861214', '-3.3915027692683215']
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
- humedad_relativa_media: -0.0521
- trimestre: 0.0469
- mes: 0.0448
- velocidad_viento_media: 0.0300
- altitud: -0.0191
- año: 0.0128
- precipitacion: -0.0103
- causa_incendio: -0.0061

## 8. Transformaciones Aplicadas
- **Target**: log1p(superficie_quemada)
- **Numéricas**: Imputación (mediana) + StandardScaler
- **Geo**: latitud/longitud convertidas a float
- **Categóricas baja cardinalidad**: OneHotEncoding
- **Categóricas alta cardinalidad**: TargetEncoding
- **Cíclicas**: mes → sin(mes), cos(mes)