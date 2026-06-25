import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.config import (CAUSA_MAP, MODEL_PATH, PREPROCESSOR_PATH,
                         TARGET_TRANSFORMER_PATH, TARGET_ENCODING_PATH)
from src.predict import load_artifact, prepare_input_for_prediction, PROVINCIA_TO_CCAA

st.set_page_config(
    page_title="Prediccion de Incendios Forestales",
    page_icon="🔥",
    layout="centered",
)

st.title("Prediccion de Superficie Quemada")
st.markdown("Introduce las condiciones meteorologicas y caracteristicas del incendio para estimar la superficie quemada.")

provincias = [
    "A CORUÑA", "ALBACETE", "ALICANTE", "ASTURIAS", "BADAJOZ", "BALEARES",
    "BARCELONA", "BURGOS", "CANTABRIA", "CEUTA", "CIUDAD REAL", "CUENCA",
    "GIRONA", "GRANADA", "GUADALAJARA", "HUELVA", "HUESCA", "LA RIOJA",
    "LAS PALMAS", "LLEIDA", "LUGO", "MADRID", "MURCIA", "NAVARRA",
    "OURENSE", "PALENCIA", "PONTEVEDRA", "SALAMANCA", "SANTA CRUZ DE TENERIFE",
    "SEGOVIA", "SEVILLA", "SORIA", "TARRAGONA", "TERUEL", "TOLEDO",
    "VALENCIA", "VALLADOLID", "ZAMORA", "ZARAGOZA",
]
causas_labels = {v: k for k, v in CAUSA_MAP.items()}
meses_nombres = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

@st.cache_resource
def load_artifacts():
    model = load_artifact(MODEL_PATH)
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    target_transformer = load_artifact(TARGET_TRANSFORMER_PATH)
    te_mappings = load_artifact(TARGET_ENCODING_PATH)
    return model, preprocessor, target_transformer, te_mappings

try:
    model, preprocessor, target_transformer, te_mappings = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    model_loaded = False

st.sidebar.header("Parametros de entrada")

provincia = st.sidebar.selectbox("Provincia", options=provincias)
causa_label = st.sidebar.selectbox(
    "Causa del incendio",
    options=list(CAUSA_MAP.values()),
    format_func=lambda x: x.replace("_", " ").title(),
)
causa_incendio = causas_labels[causa_label]

mes = st.sidebar.selectbox("Mes", options=list(meses_nombres.keys()), format_func=lambda x: meses_nombres[x])

altitud = st.sidebar.number_input("Altitud (m)", value=600.0, step=100.0)

st.sidebar.header("Temperatura y Humedad")
temperatura_media = st.sidebar.number_input("Temperatura media (C)", value=20.0, step=1.0)
humedad_relativa_media = st.sidebar.number_input("Humedad relativa media (%)", value=60.0, step=5.0, min_value=0.0, max_value=100.0)

st.sidebar.header("Precipitacion y Sol")
precipitacion = st.sidebar.number_input("Precipitacion (mm)", value=0.0, step=1.0, min_value=0.0)
sol = st.sidebar.number_input("Horas de sol", value=9.4, step=1.0, min_value=0.0, max_value=24.0)

st.sidebar.header("Ubicacion")
latitud = st.sidebar.number_input("Latitud", value=40.4169, step=0.1, format="%.4f")
longitud = st.sidebar.number_input("Longitud", value=-3.7038, step=0.1, format="%.4f")

st.sidebar.header("Viento")
velocidad_viento_media = st.sidebar.number_input("Velocidad del viento media (m/s)", value=10.0, step=1.0, min_value=0.0)
racha_maxima_viento = st.sidebar.number_input("Racha maxima de viento (m/s)", value=20.0, step=1.0, min_value=0.0)
dir_viento = st.sidebar.slider("Direccion del viento (0=calma, 36=Norte)", min_value=0, max_value=36, value=18)

if st.sidebar.button("Predecir superficie quemada", type="primary", use_container_width=True):
    if not model_loaded:
        st.error("El modelo no esta cargado. Ejecuta primero 'python main.py' para entrenarlo.")
    else:
        direccion_rad = dir_viento * 10 * np.pi / 180.0
        dir_sin = np.sin(direccion_rad)
        dir_cos = np.cos(direccion_rad)

        cc_aa = PROVINCIA_TO_CCAA.get(provincia, "OTRAS")

        input_data = pd.DataFrame([{
            "altitud": altitud,
            "latitud": latitud,
            "longitud": longitud,
            "temperatura_media": temperatura_media,
            "precipitacion": precipitacion,
            "humedad_relativa_media": humedad_relativa_media,
            "velocidad_viento_media": velocidad_viento_media,
            "racha_maxima_viento": racha_maxima_viento,
            "sol": sol,
            "dir_sin": dir_sin,
            "dir_cos": dir_cos,
            "provincia": provincia,
            "cc_aa": cc_aa,
            "causa_incendio": causa_incendio,
            "mes": mes,
        }])

        df = prepare_input_for_prediction(input_data, te_mappings)
        X_processed = preprocessor.transform(df)
        y_pred_log = model.predict(X_processed)
        y_pred = target_transformer.inverse_transform(y_pred_log.reshape(-1, 1)).ravel()[0]

        col1, col2, col3 = st.columns(3)
        with col2:
            st.metric(
                label="Superficie Quemada Estimada",
                value=f"{y_pred:.2f} ha",
                delta=None,
            )

        if y_pred < 3:
            gravedad = "Conato"
        elif y_pred < 10:
            gravedad = "Pequeno"
        elif y_pred < 50:
            gravedad = "Mediano"
        else:
            gravedad = "Grande"

        st.info(
            f"**Gravedad estimada:** {gravedad}\n\n"
            f"**Prediccion:** {y_pred:.2f} hectareas\n\n"
            f"Datos del incendio:\n"
            f"- Provincia: {provincia}\n"
            f"- Causa: {causa_label.replace('_', ' ').title()}\n"
            f"- Mes: {meses_nombres[mes]}\n"
            f"- Temp. media: {temperatura_media}C / Humedad: {humedad_relativa_media}%\n"
            f"- Precipitacion: {precipitacion} mm / Sol: {sol}h\n"
            f"- Viento: {velocidad_viento_media} m/s (racha {racha_maxima_viento}), dir: {dir_viento*10}°"
        )

st.markdown("---")
