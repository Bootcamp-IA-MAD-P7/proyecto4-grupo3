"""Streamlit app for wildfire burned area prediction.

Usage:
  uv run streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DATA_PROCESSED, TARGET, MODELS_DIR
from preprocessing.preprocessor import preprocess_dataframe, prepare_data
from training.trainer import load_model

st.set_page_config(
    page_title="Prediccion de Superficie Quemada",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_data_and_model():
    df = pl.read_parquet(DATA_PROCESSED)
    df_pd = df.to_pandas()

    year_col = "a\u00f1o"
    df_pd[TARGET] = np.log1p(df_pd[TARGET])

    models = {}
    model_dir = Path(MODELS_DIR)
    for p in model_dir.glob("*.pkl"):
        name = p.stem
        models[name] = load_model(name)

    return df, df_pd, models, year_col


@st.cache_data
def compute_summary_stats(df_pd: pd.DataFrame):
    y_orig = np.expm1(df_pd[TARGET])
    return {
        "mean": y_orig.mean(),
        "median": y_orig.median(),
        "std": y_orig.std(),
        "min": y_orig.min(),
        "max": y_orig.max(),
        "q25": y_orig.quantile(0.25),
        "q75": y_orig.quantile(0.75),
    }


st.title("Prediccion de Superficie Quemada")
st.markdown(
    "Pipeline de regresion para estimar la superficie quemada en incendios forestales "
    "a partir de datos meteorologicos."
)

try:
    df_raw, df_pd, models, year_col = load_data_and_model()
    stats = compute_summary_stats(df_pd)
except Exception as e:
    st.error(f"Error al cargar datos/modelos: {e}")
    st.info("Ejecuta primero `uv run python main.py` para entrenar los modelos.")
    st.stop()

# ──────────── SIDEBAR ────────────
st.sidebar.header("Panel de Control")

model_names = list(models.keys())
default_model = "random_forest_optimized" if "random_forest_optimized" in model_names else model_names[0]
selected_model = st.sidebar.selectbox(
    "Seleccionar modelo", model_names, index=model_names.index(default_model)
)

st.sidebar.subheader("Resumen del Dataset")
st.sidebar.metric("Registros", f"{df_raw.shape[0]:,}")
st.sidebar.metric("Columnas", df_raw.shape[1])
st.sidebar.metric("Media superficie", f"{stats['mean']:.1f} ha")
st.sidebar.metric("Mediana superficie", f"{stats['median']:.1f} ha")

# ──────────── TABS ────────────
tab1, tab2, tab3 = st.tabs(["Exploracion", "Modelos y Metricas", "Prediccion Individual"])

with tab1:
    st.header("Analisis Exploratorio")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribucion de superficie_quemada")
        y_orig = np.expm1(df_pd[TARGET])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(y_orig, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(stats["mean"], color="red", ls="--", label=f"Media: {stats['mean']:.1f}")
        ax.axvline(stats["median"], color="green", ls="--", label=f"Mediana: {stats['median']:.1f}")
        ax.set_xlabel("Superficie Quemada (ha)")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Distribucion log1p")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(df_pd[TARGET], bins=80, color="coral", edgecolor="white", alpha=0.8)
        ax2.set_xlabel("log1p(Superficie Quemada)")
        ax2.set_ylabel("Frecuencia")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    st.subheader("Vista previa del dataset")
    st.dataframe(df_raw.head(100))

with tab2:
    st.header("Comparacion de Modelos")

    st.subheader("Resultados en Test")
    metrics_df = pd.DataFrame([
        {
            "Modelo": name.replace("_", " ").title(),
            "R2 Test": 0,
            "MAE (ha)": 0,
            "RMSE (ha)": 0,
        }
        for name in models.keys()
    ])

    # Show saved model files with metadata
    st.info(
        "Los modelos disponibles se entrenaron con "
        "log1p(superficie_quemada) como target. "
        "Las metricas visibles en el informe final estan en "
        "`reports/informe_final_rendimiento.md`."
    )

    with open("reports/informe_final_rendimiento.md", "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)

    st.subheader("Graficos de Diagnostico")
    from pathlib import Path
    figure_dir = Path("figures")
    png_files = sorted(figure_dir.glob("*.png"))

    cols = st.columns(3)
    for i, p in enumerate(png_files):
        with cols[i % 3]:
            st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)

with tab3:
    st.header("Prediccion Individual")
    st.markdown("Ingresa los datos meteorologicos para predecir la superficie quemada.")

    col1, col2, col3 = st.columns(3)

    with col1:
        provincia = st.selectbox(
            "Provincia",
            sorted(df_raw["provincia"].unique().to_list()),
        )
        causa = st.selectbox(
            "Causa del incendio",
            sorted(df_raw["causa_incendio"].unique().to_list()),
            format_func=lambda x: f"Causa {int(x)}",
        )
        altitud = st.number_input("Altitud (m)", value=500.0)

    with col2:
        temp_media = st.number_input("Temperatura media (C)", value=15.0)
        temp_min = st.number_input("Temperatura minima (C)", value=8.0)
        temp_max = st.number_input("Temperatura maxima (C)", value=22.0)
        precipitacion = st.number_input("Precipitacion (mm)", value=0.0)

    with col3:
        hum_media = st.number_input("Humedad relativa media (%)", value=65.0)
        hum_max = st.number_input("Humedad relativa maxima (%)", value=85.0)
        hum_min = st.number_input("Humedad relativa minima (%)", value=45.0)
        viento_media = st.number_input("Velocidad viento media (m/s)", value=10.0)
        racha_max = st.number_input("Racha maxima viento (m/s)", value=20.0)

    latitud = st.number_input("Latitud", value=40.0, format="%.6f")
    longitud = st.number_input("Longitud", value=-4.0, format="%.6f")

    if st.button("Predecir", type="primary"):
        input_data = pd.DataFrame([{
            "altitud": altitud,
            "temperatura_media": temp_media,
            "precipitacion": precipitacion,
            "temperatura_minima": temp_min,
            "temperatura_maxima": temp_max,
            "humedad_relativa_media": hum_media,
            "humedad_relativa_maxima": hum_max,
            "humedad_relativa_minima": hum_min,
            "velocidad_viento_media": viento_media,
            "racha_maxima_viento": racha_max,
            year_col: 2022,
            "latitud": latitud,
            "longitud": longitud,
            "provincia": provincia,
            "cc_aa": provincia,
            "causa_incendio": int(causa),
            "mes": 7,
        }])

        pipeline = models[selected_model]
        try:
            pred_log = pipeline.predict(input_data)[0]
            pred_ha = np.expm1(pred_log)

            st.success(f"**Superficie quemada estimada: {pred_ha:.2f} ha**")
            st.caption(f"(en escala logaritmica: {pred_log:.4f})")

            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.barh(["Prediccion"], [pred_ha], color="coral", alpha=0.8)
            ax3.set_xlabel("Superficie (ha)")
            ax3.grid(alpha=0.3, axis="x")
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Error al predecir: {e}")

st.markdown("---")
st.caption("proyecto4-grupo3 · Pipeline de regresion con Polars + scikit-learn")
