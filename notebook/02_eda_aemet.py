# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: proyecto4-grupo3 (3.12.13.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import altair as alt

plt.style.use('seaborn-v0_8-whitegrid')

# %% [markdown]
# ## 2. Data loading

# %%
df = pl.read_csv('../data/raw/aemet_2022.csv')

# %%
df.shape

# %%

df.sample(10)

# %%
df.describe()

# %%
df.schema

# %%
df.glimpse()

# %% [markdown]
# ## 3. Data overview

# %%
for col in df.columns:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(sort=True).head(10))

# %%
df.select(pl.all().n_unique())

# %%
df.null_count()

# %% [markdown]
# ## 4. Data cleaning

# %% [markdown]
# ### 4.1 Column cleanup'

# %%
df = df.drop(["indicativo", "nombre", "horatmin", "horatmax", "horaHrMin", "dir", "horaracha", "horaPresMax", "horaPresMin", "sol", "horaHrMax", "presMax", "presMin"])

# %% [markdown]
# ### 4.2 CCAA mapping

# %%
provincia_to_ccaa = {
    'ALMERIA': 'ANDALUCIA',
    'CADIZ': 'ANDALUCIA',
    'CORDOBA': 'ANDALUCIA',
    'GRANADA': 'ANDALUCIA',
    'HUELVA': 'ANDALUCIA',
    'JAEN': 'ANDALUCIA',
    'MALAGA': 'ANDALUCIA',
    'SEVILLA': 'ANDALUCIA',

    'HUESCA': 'ARAGON',
    'TERUEL': 'ARAGON',
    'ZARAGOZA': 'ARAGON',

    'ASTURIAS': 'ASTURIAS',

    'BALEARES': 'BALEARES',
    'ILLES BALEARS': 'BALEARES',

    'LAS PALMAS': 'CANARIAS',
    'STA. CRUZ DE TENERIFE': 'CANARIAS',
    'SANTA CRUZ DE TENERIFE': 'CANARIAS',

    'CANTABRIA': 'CANTABRIA',
    'SANTANDER': 'CANTABRIA',

    'ALBACETE': 'CASTILLA-LA MANCHA',
    'CIUDAD REAL': 'CASTILLA-LA MANCHA',
    'CUENCA': 'CASTILLA-LA MANCHA',
    'GUADALAJARA': 'CASTILLA-LA MANCHA',
    'TOLEDO': 'CASTILLA-LA MANCHA',

    'AVILA': 'CASTILLA Y LEON',
    'BURGOS': 'CASTILLA Y LEON',
    'LEON': 'CASTILLA Y LEON',
    'PALENCIA': 'CASTILLA Y LEON',
    'SALAMANCA': 'CASTILLA Y LEON',
    'SEGOVIA': 'CASTILLA Y LEON',
    'SORIA': 'CASTILLA Y LEON',
    'VALLADOLID': 'CASTILLA Y LEON',
    'ZAMORA': 'CASTILLA Y LEON',

    'BARCELONA': 'CATALUNA',
    'GIRONA': 'CATALUNA',
    'LLEIDA': 'CATALUNA',
    'TARRAGONA': 'CATALUNA',

    'BADAJOZ': 'EXTREMADURA',
    'CACERES': 'EXTREMADURA',

    'A CORUÑA': 'GALICIA',
    'LUGO': 'GALICIA',
    'OURENSE': 'GALICIA',
    'PONTEVEDRA': 'GALICIA',

    'LA RIOJA': 'LA RIOJA',

    'MADRID': 'MADRID',

    'MURCIA': 'MURCIA',

    'NAVARRA': 'NAVARRA',

    'ALAVA': 'PAIS VASCO',
    'ARABA/ALAVA': 'PAIS VASCO',
    'GUIPUZCOA': 'PAIS VASCO',
    'GIPUZKOA': 'PAIS VASCO',
    'BIZKAIA': 'PAIS VASCO',
    'VIZCAYA': 'PAIS VASCO',

    'ALICANTE': 'VALENCIA',
    'CASTELLON': 'VALENCIA',
    'VALENCIA': 'VALENCIA',

    'CEUTA': 'CEUTA',
    'MELILLA': 'MELILLA'
}

df = df.with_columns(
    pl.col("provincia")
    .replace_strict(provincia_to_ccaa, default=None)
    .alias("cc_aa")
)

df = df.select(
    "provincia",
    "cc_aa",
    pl.exclude(["provincia", "cc_aa"])
)


# %%
# %% [markdown]
# ### 4.3 Rename columns

# %%
df = df.rename({
    'tmed': 'temperatura_media',
    'prec': 'precipitacion',
    'tmin': 'temperatura_minima',
    'tmax': 'temperatura_maxima',
    'hrMedia': 'humedad_relativa_media',
    'hrMax': 'humedad_relativa_maxima',
    'hrMin': 'humedad_relativa_minima',
    'velmedia': 'velocidad_viento_media',
    'racha': 'racha_maxima_viento',
})

# %%
df = df.drop_nulls(subset=["temperatura_media", "precipitacion", "temperatura_minima", "temperatura_maxima", "humedad_relativa_media", "humedad_relativa_maxima", "humedad_relativa_minima"])

# %% [markdown]
# ### 4.4 Null removal

# %% [markdown]
# ### 4.5 Wind cleaning

# %%
df = df.with_columns(
    pl.col("velocidad_viento_media")
      .str.replace(",", ".")
      .cast(pl.Float64)
)

# %%
df = df.with_columns(
    pl.col("velocidad_viento_media")
      .fill_null(pl.col("velocidad_viento_media").median())
)
print(df["velocidad_viento_media"].median()) 

# %%
df = df.with_columns(
    pl.col("racha_maxima_viento")
      .str.replace(",", ".")
      .cast(pl.Float64)
)

# %%
df = df.with_columns(
    pl.col("racha_maxima_viento")
      .fill_null(pl.col("racha_maxima_viento").median())
)
print(df["racha_maxima_viento"].median()) 

# %% [markdown]
# ### 4.6 Temperature cleaning

# %%
df = df.with_columns(
    pl.col("temperatura_media", "temperatura_minima", "temperatura_maxima")
      .str.replace(",", ".")
      .cast(pl.Float64)
)

# %%
df = df.with_columns(
    pl.col("precipitacion")
)

# %% [markdown]
# ### 4.7 Precipitation cleaning

# %%
df.filter(pl.col('precipitacion') == 'Acum').height

# %%
df.filter(pl.col('precipitacion') == 'Ip').height

# %%
df = df.filter(pl.col('precipitacion') != 'Acum')

# %%
df = df.filter(pl.col('precipitacion') != 'Ip')

# %%
df = df.with_columns(
    pl.col("precipitacion")
      .str.replace(",", ".")
      .cast(pl.Float64)
)

# %% [markdown]
# ### 4.8 Date alignment

# %%
meses = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# %%
df = df.with_columns(
    pl.col("fecha").str.to_date("%Y-%m-%d").dt.month().cast(pl.Int64).replace_strict(meses).alias("mes")
).select(
    pl.col("fecha"), pl.col("mes"), pl.all().exclude("fecha", "mes")
)

# %%
df.sample(5)

# %% [markdown]
# ## 5. Data export

# %%
from pathlib import Path

Path("../data/cleaned").mkdir(parents=True, exist_ok=True)
df.write_parquet("../data/cleaned/aemet_clean.parquet")

# %% [markdown]
# ## 6. Bivariate analysis

# %%
import polars as pl
import altair as alt
import pyarrow as pa
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

alt.data_transformers.enable("vegafusion")
plt.style.use('seaborn-v0_8-whitegrid')

df = pl.read_parquet("../data/cleaned/aemet_clean.parquet")
df_pd = df.to_pandas()

# %%
prov_map = {
    'Illes Balears': 'ILLES BALEARS',
    'Asturias': 'ASTURIAS',
    'A Coruña': 'A CORUÑA',
    'Girona': 'GIRONA',
    'Las Palmas': 'LAS PALMAS',
    'Pontevedra': 'PONTEVEDRA',
    'Santa Cruz De Tenerife': 'SANTA CRUZ DE TENERIFE',
    'Cantabria': 'CANTABRIA',
    'Málaga': 'MALAGA',
    'Almería': 'ALMERIA',
    'Murcia': 'MURCIA',
    'Albacete': 'ALBACETE',
    'Ávila': 'AVILA',
    'Araba/Álava': 'ARABA/ALAVA',
    'Badajoz': 'BADAJOZ',
    'Alacant/Alicante': 'ALICANTE',
    'Ourense': 'OURENSE',
    'Barcelona': 'BARCELONA',
    'Burgos': 'BURGOS',
    'Cáceres': 'CACERES',
    'Cádiz': 'CADIZ',
    'Castelló/Castellón': 'CASTELLON',
    'Ciudad Real': 'CIUDAD REAL',
    'Jaén': 'JAEN',
    'Córdoba': 'CORDOBA',
    'Cuenca': 'CUENCA',
    'Granada': 'GRANADA',
    'Guadalajara': 'GUADALAJARA',
    'Gipuzkoa/Guipúzcoa': 'GIPUZKOA',
    'Huelva': 'HUELVA',
    'Huesca': 'HUESCA',
    'León': 'LEON',
    'Lleida': 'LLEIDA',
    'La Rioja': 'LA RIOJA',
    'Soria': 'SORIA',
    'Navarra': 'NAVARRA',
    'Ceuta': 'CEUTA',
    'Lugo': 'LUGO',
    'Madrid': 'MADRID',
    'Palencia': 'PALENCIA',
    'Salamanca': 'SALAMANCA',
    'Segovia': 'SEGOVIA',
    'Sevilla': 'SEVILLA',
    'Toledo': 'TOLEDO',
    'Tarragona': 'TARRAGONA',
    'Teruel': 'TERUEL',
    'València/Valencia': 'VALENCIA',
    'Valladolid': 'VALLADOLID',
    'Bizkaia/Vizcaya': 'BIZKAIA',
    'Zamora': 'ZAMORA',
    'Zaragoza': 'ZARAGOZA',
    'Melilla': 'MELILLA',
}

# %%
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


# %% [markdown]
# ### 6.1 Heatmap — Province x Month temperature
# **Type:** Heatmap | **Data:** Mean temperature per province per month
#
# Original: Mapa de calor — Temperatura media: provincia × mes
#

# %%
heatmap_data = (df_pd
    .groupby(["provincia", "mes"])["temperatura_media"]
    .mean()
    .reset_index()
    .rename(columns={"temperatura_media": "temp_mean"})
)

alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X("mes:N", title="Month", sort=list(month_order)),
    y=alt.Y("provincia:N", title="Province"),
    color=alt.Color("temp_mean:Q", title="Avg Temp (C)",
                    scale=alt.Scale(scheme="redyellowblue", reverse=True)),
    tooltip=["provincia", "mes", alt.Tooltip("temp_mean", format=".1f")]
).properties(
    title="Average Temperature: Province x Month",
    width=600,
    height=600
).interactive().show()

# %% [markdown]
# ### 6.2 Heatmap — Province x Month precipitation
# **Type:** Heatmap | **Data:** Mean precipitation per province per month
#
# Original: Mapa de calor — Precipitación media: provincia × mes
#

# %%
heatmap_precip_mean = (df
    .group_by(["provincia", "mes"])
    .agg(pl.col("precipitacion").mean().alias("precip_mean"))
    .with_columns(pl.col("mes").cast(pl.Enum(month_order)))
    .sort("mes")
)

alt.Chart(heatmap_precip_mean).mark_rect().encode(
    x=alt.X("mes:N", title="Month", sort=list(month_order)),
    y=alt.Y("provincia:N", title="Province"),
    color=alt.Color("precip_mean:Q", title="Avg Precipitation (mm)",
                    scale=alt.Scale(scheme="blues", reverse=True)),
    tooltip=["provincia", "mes", alt.Tooltip("precip_mean", format=".1f")]
).properties(
    title="Average Precipitation: Province x Month",
    width=600,
    height=600
).interactive().show()


# %% [markdown]
# ### 6.3 Heatmap — Province x Month wind speed
# **Type:** Heatmap | **Data:** Mean wind speed per province per month
#
# Original: Mapa de calor — Velocidad del viento media: provincia × mes
#

# %%
heatmap_viento_mean = (df
    .group_by(["provincia", "mes"])
    .agg(pl.col("velocidad_viento_media").mean().alias("viento_mean"))
    .with_columns(pl.col("mes").cast(pl.Enum(month_order)))
    .sort("mes")
)

alt.Chart(heatmap_viento_mean).mark_rect().encode(
    x=alt.X("mes:N", title="Month", sort=list(month_order)),
    y=alt.Y("provincia:N", title="Province"),
    color=alt.Color("viento_mean:Q", title="Avg Wind Speed (m/s)",
                    scale=alt.Scale(scheme="greens", reverse=True)),
    tooltip=["provincia", "mes", alt.Tooltip("viento_mean", format=".1f")]
).properties(
    title="Average Wind Speed: Province x Month",
    width=600,
    height=600
).interactive().show()


# %% [markdown]
# ### 6.4 Scatter — Temperature vs precipitation
# **Type:** Interactive scatter with province selector | **Data:** Temperature/precipitation relationship. Color = month, each point = a month's average.
#
# Original: Diagrama de dispersión — Temperatura vs Precipitación
#

# %%
scatter_precip = (df
    .group_by(["mes", "provincia"])
    .agg([
        pl.col("temperatura_media").mean().alias("temp_mean"),
        pl.col("precipitacion").mean().alias("precip_mean"),
    ])
    .with_columns(pl.col("mes").cast(pl.Enum(month_order)))
    .sort("mes")
)


# %%
scatter_precip = scatter_precip.with_columns([
    pl.col("provincia").cast(pl.String),
    pl.col("mes").cast(pl.String)
])

prov_list_precip = sorted(scatter_precip["provincia"].unique())

selector_precip = alt.selection_point(
    fields=["provincia"],
    bind=alt.binding_select(options=prov_list_precip, name="Select Province: "),
    value=[{"provincia": prov_list_precip[0]}]
)

alt.Chart(scatter_precip).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X("temp_mean:Q", title="Avg Temperature (C)"),
    y=alt.Y("precip_mean:Q", title="Avg Precipitation (mm)"),
    color=alt.Color("mes:N", title="Month", sort=list(month_order),
                    scale=alt.Scale(scheme="turbo")),
    tooltip=["mes", "provincia",
             alt.Tooltip("temp_mean", format=".1f"),
             alt.Tooltip("precip_mean", format=".2f")]
).add_params(
    selector_precip
).transform_filter(
    selector_precip
).properties(
    title="Temperature vs Precipitation by Province & Month",
    width=600,
    height=400
).interactive().show()


# %% [markdown]
# ### 6.5 Scatter — Temperature vs wind speed
# **Type:** Interactive scatter with province selector | **Data:** Temperature/wind relationship. Color = month.
#
# Original: Diagrama de dispersión — Temperatura vs Velocidad del viento
#

# %%
scatter_viento = (df
    .group_by(["mes", "provincia"])
    .agg([
        pl.col("temperatura_media").mean().alias("temp_mean"),
        pl.col("velocidad_viento_media").mean().alias("viento_mean"),
    ])
    .with_columns(pl.col("mes").cast(pl.Enum(month_order)))
    .sort("mes")
)


# %%
scatter_viento = scatter_viento.with_columns([
    pl.col("provincia").cast(pl.String),
    pl.col("mes").cast(pl.String)
])

prov_list_viento = sorted(scatter_viento["provincia"].unique())

selector_viento = alt.selection_point(
    fields=["provincia"],
    bind=alt.binding_select(options=prov_list_viento, name="Select Province: "),
    value=[{"provincia": prov_list_viento[0]}]
)

alt.Chart(scatter_viento).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X("temp_mean:Q", title="Avg Temperature (C)"),
    y=alt.Y("viento_mean:Q", title="Avg Wind Speed (m/s)"),
    color=alt.Color("mes:N", title="Month", sort=list(month_order),
                    scale=alt.Scale(scheme="turbo")),
    tooltip=["mes", "provincia",
             alt.Tooltip("temp_mean", format=".1f"),
             alt.Tooltip("viento_mean", format=".2f")]
).add_params(
    selector_viento
).transform_filter(
    selector_viento
).properties(
    title="Temperature vs Wind Speed by Province & Month",
    width=600,
    height=400
).interactive().show()


# %% [markdown]
# ### 6.6 Correlation heatmap
# **Type:** Correlation heatmap | **Data:** Correlation matrix: temperature, precipitation, humidity, wind speed
#
# Original: Mapa de calor — Correlaciones entre variables numéricas
#

# %%
num_cols = [
    "temperatura_media", "precipitacion", "temperatura_minima",
    "temperatura_maxima", "humedad_relativa_media", "humedad_relativa_maxima",
    "humedad_relativa_minima", "velocidad_viento_media", "racha_maxima_viento"
]

corr_df = df_pd[num_cols].corr().reset_index()
corr_df.columns = ["var1"] + num_cols
corr_melted = corr_df.melt(id_vars="var1", var_name="var2", value_name="corr")

alt.Chart(corr_melted).mark_rect().encode(
    x=alt.X("var1:N", title=""),
    y=alt.Y("var2:N", title=""),
    color=alt.Color("corr:Q", title="Correlation",
                scale=alt.Scale(scheme="rainbow", domain=[-1, 1])),
    tooltip=["var1", "var2", alt.Tooltip("corr", format=".2f")]
).properties(
    title="Correlation Heatmap (Numeric Variables)",
    width=500,
    height=500
).interactive().show()

# %% [markdown]
# ## 7. Geospatial analysis
# **Type:** Choropleth maps | **Data:** Temperature, precipitation, humidity, wind gust averages per province
#
# Original: Mapas coropléticos de España por provincia
#

# %%
spain = gpd.read_file("https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/spain-provinces.geojson")
spain = spain.to_crs(epsg=4326)

spain["provincia"] = spain["name"].map(prov_map)

df_agg = df_pd.groupby("provincia")["velocidad_viento_media"].mean().reset_index()
df_agg.columns = ["provincia", "viento_medio"]

map_df = spain.merge(df_agg, on="provincia")

canarias = map_df[map_df["provincia"].isin(["LAS PALMAS", "SANTA CRUZ DE TENERIFE"])]
peninsula = map_df[~map_df["provincia"].isin(["LAS PALMAS", "SANTA CRUZ DE TENERIFE"])]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
map_df.plot(
    column="viento_medio",
    ax=ax,
    legend=True,
    cmap = plt.cm.rainbow,
    legend_kwds={"label": "Average Wind Speed (km/h)"}
)

ax_inset = fig.add_axes([0.05, 0.05, 0.25, 0.2])
canarias.plot(column="viento_medio", ax=ax_inset, cmap=plt.cm.rainbow,
    vmin=map_df["viento_medio"].min(), vmax=map_df["viento_medio"].max())
ax_inset.set_axis_off()

ax.set_title("Average Wind Speed by Province", fontsize=14)
ax.set_xlim(-10, 5)
ax.set_ylim(35, 44)
ax.set_axis_off()
plt.show()

# %%
variables = {
    "temperatura_media": "Average Temperature (°C)",
    "precipitacion": "Total Precipitation (mm)",
    "humedad_relativa_media": "Average Relative Humidity (%)",
    "racha_maxima_viento": "Max Wind Gust (km/h)"
}

agg_funcs = {
    "temperatura_media": "mean",
    "precipitacion": "sum",
    "humedad_relativa_media": "mean",
    "racha_maxima_viento": "mean"
}

for var, label in variables.items():

    df_agg = df_pd.groupby("provincia")[var].agg(agg_funcs[var]).reset_index()
    df_agg.columns = ["provincia", "valor"]

    map_df = spain.merge(df_agg, on="provincia")

    canarias = map_df[map_df["provincia"].isin(["LAS PALMAS", "SANTA CRUZ DE TENERIFE"])]
    peninsula = map_df[~map_df["provincia"].isin(["LAS PALMAS", "SANTA CRUZ DE TENERIFE"])]

    cmap = LinearSegmentedColormap.from_list("custom", ["#FF477B", "#8858C8", "#C0F7F4", "#5ECFB8"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    map_df.plot(column="valor", ax=ax, legend=True, cmap=cmap,
        legend_kwds={"label": label})

    ax_inset = fig.add_axes([0.05, 0.05, 0.25, 0.2])
    canarias.plot(column="valor", ax=ax_inset, cmap=cmap,
        vmin=map_df["valor"].min(), vmax=map_df["valor"].max())
    ax_inset.set_axis_off()

    ax.set_title(f"{label} by Province", fontsize=14)
    ax.set_xlim(-10, 5)
    ax.set_ylim(35, 44)
    ax.set_axis_off()
    plt.show()
