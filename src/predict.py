import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


def load_artifact(path: Path):
    logger.debug("Cargando artefacto: %s", path)
    return joblib.load(path)


PROVINCIA_TO_CCAA = {
    "ALAVA": "PAIS VASCO", "ALBACETE": "CASTILLA-LA MANCHA", "ALICANTE": "VALENCIA",
    "ALMERIA": "ANDALUCIA", "ASTURIAS": "ASTURIAS", "AVILA": "CASTILLA Y LEON",
    "BADAJOZ": "EXTREMADURA", "BALEARES": "BALEARES", "BARCELONA": "CATALUNA",
    "BURGOS": "CASTILLA Y LEON", "CACERES": "EXTREMADURA", "CADIZ": "ANDALUCIA",
    "CANTABRIA": "CANTABRIA", "CASTELLON": "VALENCIA", "CEUTA": "CEUTA",
    "CIUDAD REAL": "CASTILLA-LA MANCHA", "CORDOBA": "ANDALUCIA",
    "A CORUÑA": "GALICIA", "CUENCA": "CASTILLA-LA MANCHA",
    "GIRONA": "CATALUNA", "GRANADA": "ANDALUCIA", "GUADALAJARA": "CASTILLA-LA MANCHA",
    "GUIPUZCOA": "PAIS VASCO", "HUELVA": "ANDALUCIA", "HUESCA": "ARAGON",
    "JAEN": "ANDALUCIA", "LA RIOJA": "LA RIOJA", "LAS PALMAS": "CANARIAS",
    "LEON": "CASTILLA Y LEON", "LLEIDA": "CATALUNA", "LUGO": "GALICIA",
    "MADRID": "MADRID", "MALAGA": "ANDALUCIA", "MELILLA": "CEUTA",
    "MURCIA": "MURCIA", "NAVARRA": "NAVARRA", "OURENSE": "GALICIA",
    "PALENCIA": "CASTILLA Y LEON", "PONTEVEDRA": "GALICIA",
    "SALAMANCA": "CASTILLA Y LEON", "SANTA CRUZ DE TENERIFE": "CANARIAS",
    "SEGOVIA": "CASTILLA Y LEON", "SEVILLA": "ANDALUCIA", "SORIA": "CASTILLA Y LEON",
    "TARRAGONA": "CATALUNA", "TERUEL": "ARAGON", "TOLEDO": "CASTILLA-LA MANCHA",
    "VALENCIA": "VALENCIA", "VALLADOLID": "CASTILLA Y LEON",
    "VIZCAYA": "PAIS VASCO", "ZAMORA": "CASTILLA Y LEON", "ZARAGOZA": "ARAGON",
}


def _apply_te_column(df, col, out_name, te_mappings, global_mean_default=0):
    if out_name not in te_mappings:
        return
    mapping_info = te_mappings[out_name]
    mapping = mapping_info["mapping"]
    global_mean = mapping_info.get("global_mean", global_mean_default)
    df[out_name] = df[col].map(mapping).fillna(global_mean)


def prepare_input_for_prediction(input_df: pd.DataFrame, te_mappings: dict) -> pd.DataFrame:
    df = input_df.copy()

    if "cc_aa" not in df.columns and "provincia" in df.columns:
        df["cc_aa"] = df["provincia"].map(PROVINCIA_TO_CCAA).fillna("OTRAS")

    for c in ["latitud", "longitud"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(float)

    if "dir_sin" not in df.columns and "dir" in df.columns:
        direccion_rad = df["dir"].values * 10 * np.pi / 180.0
        df["dir_sin"] = np.sin(direccion_rad)
        df["dir_cos"] = np.cos(direccion_rad)

    df["temp_hum_interaction"] = df["temperatura_media"] * df["humedad_relativa_media"]
    df["temp_wind_interaction"] = df["temperatura_media"] * df["velocidad_viento_media"]
    df["hum_wind_interaction"] = df["humedad_relativa_media"] * df["velocidad_viento_media"]

    _apply_te_column(df, "provincia", "provincia_te", te_mappings)
    df["_provincia_mes"] = df["provincia"].astype(str) + "_" + df["mes"].astype(str)
    _apply_te_column(df, "_provincia_mes", "provincia_mes_te", te_mappings)
    df.drop(columns=["_provincia_mes"], inplace=True)

    _apply_te_column(df, "cc_aa", "cc_aa_te", te_mappings)
    df["_cc_aa_mes"] = df["cc_aa"].astype(str) + "_" + df["mes"].astype(str)
    _apply_te_column(df, "_cc_aa_mes", "cc_aa_mes_te", te_mappings)
    df.drop(columns=["_cc_aa_mes"], inplace=True)

    for drop_col in ["provincia", "cc_aa", "dir"]:
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True)

    return df
