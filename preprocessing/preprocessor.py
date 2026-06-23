import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import (
    DROP_COLS, NUM_COLS, CAT_TARGET_ENCODE, CAT_OHE,
    CYCLIC_COLS, GEO_COLS, TARGET, RANDOM_STATE,
)


def cyclic_encode(mes_series: np.ndarray) -> np.ndarray:
    mes = np.asarray(mes_series).astype(float).reshape(-1, 1)
    sin_feat = np.sin(2 * np.pi * mes / 12)
    cos_feat = np.cos(2 * np.pi * mes / 12)
    return np.column_stack([sin_feat, cos_feat])


def cyclic_feature_names(transformer, input_features):
    return ["mes_sin", "mes_cos"]


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        cols_present = [c for c in self.columns if c in X.columns]
        self.cols_to_drop_ = cols_present
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return [f for f in input_features if f not in self.cols_to_drop_]
        return []


class CastGeoToFloat(BaseEstimator, TransformerMixin):
    def __init__(self, geo_cols: list[str]):
        self.geo_cols = geo_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.geo_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return [f for f in input_features if not any(
                g in f for g in self.geo_cols)] + self.geo_cols
        return self.geo_cols


def preprocess_dataframe(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    df = df.copy()
    for col in GEO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    if year_col not in df.columns:
        df[year_col] = 0
    return df


def build_preprocessor(
    year_col: str = "a\xf1o",
    use_target_encoder: bool = True,
) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    geo_cols_actual = [c for c in GEO_COLS if c not in DROP_COLS]
    num_cols_final = NUM_COLS + [year_col] + geo_cols_actual
    num_cols_final = list(dict.fromkeys(num_cols_final))

    transformers = [
        ("num", numeric_pipeline, num_cols_final),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               min_frequency=10), CAT_OHE),
    ]

    if use_target_encoder:
        transformers.append(
            ("cat_target", TargetEncoder(target_type="continuous",
                                          random_state=RANDOM_STATE), CAT_TARGET_ENCODE)
        )
    else:
        transformers.append(
            ("cat_ohe_high", OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False, max_categories=20),
             CAT_TARGET_ENCODE)
        )

    transformers.append(
        ("cyclic", FunctionTransformer(cyclic_encode,
                                        feature_names_out=cyclic_feature_names),
         CYCLIC_COLS)
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    return pipeline


def prepare_data(df_pandas: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df_pandas.drop(columns=[TARGET], errors="ignore")
    y = df_pandas[TARGET]
    return X, y
