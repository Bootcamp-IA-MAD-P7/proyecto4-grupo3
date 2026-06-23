# %%
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import altair as alt

df = pl.read_parquet("../data/processed/fires_weather_merged.parquet")
alt.data_transformers.enable("vegafusion")

# %%
df.shape

# %%

df.sample(10)

# %%
weather_vars = ["temperatura_media", "velocidad_viento_media", "humedad_relativa_media"]

plot_df = (
    df.select(weather_vars + ["incendio_ocurrido"])
    .with_columns(
        pl.when(pl.col("incendio_ocurrido") == 1)
        .then(pl.lit("Fire"))
        .otherwise(pl.lit("No Fire"))
        .alias("label")
    )
    .drop_nulls()
    .to_pandas()
)

# %%
plot_long = plot_df.melt(
    id_vars="label",
    value_vars=weather_vars,
    var_name="variable",
    value_name="value",
)

variable_labels = {
    "temperatura_media": "Avg Temperature (°C)",
    "velocidad_viento_media": "Avg Wind Speed (km/h)",
    "humedad_relativa_media": "Avg Humidity (%)",
}
plot_long["variable"] = plot_long["variable"].map(variable_labels)

# %%
chart = (
    alt.Chart(plot_long)
    .mark_boxplot(extent="min-max", size=40)
    .encode(
        x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=["Fire", "No Fire"],
                range=["#FF477B", "#68C8E8"],
            ),
            legend=None,
        ),
        column=alt.Column(
            "variable:N",
            title=None,
            header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
        ),
    )
    .properties(
        width=200, height=300, title="Weather Distribution: Fire vs No-Fire Days"
    )
    .configure_title(fontSize=15, anchor="middle")
    .configure_view(strokeWidth=0)
)

chart.show()

# %%
season_order = ["Spring", "Summer", "Autumn", "Winter"]
trimestre_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}

fires_only = df.filter(pl.col("incendio_ocurrido") == 1).drop_nulls(
    subset=["fecha_incendio"]
)

by_month = (
    fires_only.group_by("mes")
    .agg(pl.len().alias("count"))
    .sort("mes")
    .with_columns(pl.col("mes").cast(pl.Utf8).alias("mes_str"))
    .to_pandas()
)

by_season = (
    fires_only.with_columns(
        pl.col("trimestre")
        .cast(pl.Utf8)
        .replace({"1": "Winter", "2": "Spring", "3": "Summer", "4": "Autumn"})
        .alias("season")
    )
    .group_by("season")
    .agg(pl.len().alias("count"))
    .sort(pl.col("season").cast(pl.Enum(season_order)))
    .to_pandas()
)

month_chart = (
    alt.Chart(by_month)
    .mark_bar()
    .encode(
        x=alt.X(
            "mes_str:O",
            title="Month",
            sort=list(by_month["mes_str"]),
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("count:Q", title="Number of Fires"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="orangered"), legend=None),
        tooltip=["mes_str:O", "count:Q"],
    )
    .properties(width=400, height=280, title="Fires by Month")
)

season_chart = (
    alt.Chart(by_season)
    .mark_bar()
    .encode(
        x=alt.X(
            "season:O", title="Season", sort=season_order, axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y("count:Q", title="Number of Fires"),
        color=alt.Color(
            "season:O",
            scale=alt.Scale(
                domain=season_order, range=["#A8E0D0", "#FF477B", "#E898C0", "#68C8E8"]
            ),
            legend=None,
        ),
        tooltip=["season:O", "count:Q"],
    )
    .properties(width=280, height=280, title="Fires by Season")
)

(month_chart | season_chart)


# %%
import ipywidgets as widgets
from IPython.display import display

fires = (
    df.filter(pl.col("incendio_ocurrido") == 1)
    .with_columns(
        [
            pl.col("superficie_quemada").log1p().alias("log_superficie"),
            pl.col("causa_incendio")
            .cast(pl.Utf8)
            .replace(
                {
                    "1": "Rayo",
                    "2": "Negligencia",
                    "3": "Intencionado",
                    "4": "Desconocido",
                    "5": "Reproductores",
                    "6": "Escape",
                }
            )
            .alias("causa_label"),
        ]
    )
    .to_pandas()
)

causa_palette = {
    "Rayo": "#e63946",
    "Negligencia": "#f4a261",
    "Intencionado": "#2a9d8f",
    "Desconocido": "#457b9d",
    "Reproductores": "#8338ec",
    "Escape": "#6d6875",
}

weather_vars = {
    "Temperatura media (°C)": "temperatura_media",
    "Precipitación (mm)": "precipitacion",
    "Humedad relativa media (%)": "humedad_relativa_media",
    "Velocidad viento media (km/h)": "velocidad_viento_media",
    "Racha máxima viento (km/h)": "racha_maxima_viento",
}

all_ccaa = sorted(fires["cc_aa"].unique())

ccaa_widget = widgets.SelectMultiple(
    options=["Todas"] + all_ccaa,
    value=["Todas"],
    description="CCAA:",
    layout=widgets.Layout(height="160px", width="220px"),
)

weather_widget = widgets.Dropdown(
    options=list(weather_vars.keys()),
    value="Temperatura media (°C)",
    description="Variable:",
    layout=widgets.Layout(width="280px"),
)

log_widget = widgets.Checkbox(
    value=True,
    description="Eje Y en log(superficie)",
)


def plot(ccaa_sel, weather_label, use_log):
    col = weather_vars[weather_label]
    y_col = "log_superficie" if use_log else "superficie_quemada"
    y_label = "log(superficie quemada)" if use_log else "superficie quemada (ha)"

    if "Todas" in ccaa_sel or len(ccaa_sel) == 0:
        subset = fires
        title_suffix = "todas las CCAA"
    else:
        subset = fires[fires["cc_aa"].isin(ccaa_sel)]
        title_suffix = ", ".join(ccaa_sel)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{weather_label} vs superficie quemada — {title_suffix}", fontsize=13)

    for causa, group in subset.groupby("causa_label"):
        axes[0].scatter(
            group[col],
            group[y_col],
            label=causa,
            color=causa_palette.get(causa, "#999"),
            alpha=0.4,
            s=18,
        )
    axes[0].set_xlabel(weather_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Scatter por causa")
    axes[0].legend(title="Causa", fontsize=8, title_fontsize=9, markerscale=1.5)

    means = (
        subset.groupby("causa_label")[col]
        .mean()
        .reindex(list(causa_palette.keys()))
        .dropna()
    )
    colors = [causa_palette[c] for c in means.index]
    axes[1].bar(means.index, means.values, color=colors, edgecolor="white", width=0.6)
    axes[1].set_xlabel("Causa")
    axes[1].set_ylabel(f"Media {weather_label}")
    axes[1].set_title("Media de la variable meteorológica por causa")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


out = widgets.interactive_output(
    plot,
    {
        "ccaa_sel": ccaa_widget,
        "weather_label": weather_widget,
        "use_log": log_widget,
    },
)

display(
    widgets.HBox(
        [
            widgets.VBox([ccaa_widget, log_widget]),
            weather_widget,
        ]
    ),
    out,
)

# %%
