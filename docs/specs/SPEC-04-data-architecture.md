# SPEC-04: Arquitectura de Datos

| Campo | Valor |
|-------|-------|
| **ID** | SPEC-04 |
| **Estado** | 🟡 Borrador (pendiente datos AEMET) |
| **Autor** | Equipo Bootcamp IA |
| **Fecha** | 2026-06-18 |
| **Bloquea** | SPEC-05, SPEC-08 (US-01 a US-06) |
| **Bloqueado por** | SPEC-01, recepción datos AEMET |

## Contexto
El proyecto integra dos fuentes de datos: incendios históricos (fires-all) y datos meteorológicos (AEMET). El desafío es crear un dataset de entrenamiento espaciotemporal.

## Decisión
Usar una granularidad de **municipio + día** como primera aproximación. Cada fila del dataset de entrenamiento representa un municipio en un día concreto, con features meteorológicas y la etiqueta de si hubo incendio.

## Especificación Técnica

### Fuentes de Datos
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   fires-all     │     │   AEMET          │     │   train.parquet │
│  (incendios)    │  +  │  (meteorología)  │  →  │   (municipio/   │
│                 │     │                  │     │    día)         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Estructura del Dataset de Entrenamiento
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `fecha` | datetime | Fecha del registro |
| `idprovincia` | int | Código de provincia |
| `idmunicipio` | int | Código de municipio |
| `mes` | int | Mes del año (1-12) |
| `estacion` | str | Estación del año |
| `temp_max` | float | Temperatura máxima (°C) |
| `temp_min` | float | Temperatura mínima (°C) |
| `precipitacion` | float | Precipitación (mm) |
| `velocidad_viento` | float | Velocidad del viento (km/h) |
| `direccion_viento` | str | Dirección del viento |
| `humedad_relativa` | float | Humedad relativa (%) |
| `presion_atmosferica` | float | Presión atmosférica (hPa) |
| `incendios_30d` | int | Incendios en el municipio últimos 30 días |
| `superficie_media_hist` | float | Superficie media quemada histórica en el municipio |
| `causa_predominante` | int | Causa más frecuente histórica en el municipio |
| **TARGET: `incendio_ocurrido`** | int (0/1) | ¿Hubo al menos un incendio? |

### Merge Espacial
- Asignar cada estación AEMET al municipio más cercano (distancia euclídea con lat/lng)
- Si un municipio tiene múltiples estaciones, usar la media
- Si un municipio no tiene estación cercana (< 50 km), descartar o interpolar

### Merge Temporal
- Datos AEMET diarios agregados (máximos, mínimos, sumas según variable)
- Periodo común: intersección de fechas entre fires-all y AEMET

## Criterios de Aceptación
- [ ] Dataset de entrenamiento generado sin errores
- [ ] Todos los municipios con incendio tienen datos meteorológicos asociados
- [ ] Documentación del proceso de merge en `docs/data-pipeline.md`
- [ ] Análisis de pérdida de datos por el merge (% de registros descartados)

## Consecuencias
- La calidad del modelo depende directamente de la calidad del merge
- Pérdida de datos es inevitable; documentar cuánto y por qué
- La granularidad municipio es un compromiso (más fino = más desbalanceo)

## Notas
- Pendiente: confirmar si AEMET tiene coordenadas de estaciones
- Pendiente: confirmar periodo temporal de datos AEMET
- Alternativa descartada: grid arbitrario (más complejo, se deja para v2)
