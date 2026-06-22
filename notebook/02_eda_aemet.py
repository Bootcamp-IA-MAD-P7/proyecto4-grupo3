import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    import altair as alt

    df = pl.read_csv('../data/raw/aemet_2022.csv')

    plt.style.use('seaborn-v0_8-whitegrid')
    return df, pl


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.sample(10)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.schema
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df):
    for col in df.columns:
        print(f"\n=== {col} ===")
        print(df[col].value_counts(sort=True).head(10))
    return


@app.cell
def _(df, pl):
    df.select(pl.all().n_unique())
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Total rows 308000 aprox.
        'fecha': 'fecha',
        'tmed': 'temperatura_media',
        'prec': 'precipitacion',
        'tmin': 'temperatura_minima',
        'tmax': 'temperatura_maxima',
        'hrMedia': 'humedad_relativa_media',
        'hrMax': 'humedad_relativa_maxima',
        'hrMin': 'humedad_relativa_minima',
        'velmedia': 'velocidad_viento_media',
        'racha': 'racha_maxima_viento',
        'sol': 'horas_sol'
    """)
    return


@app.cell
def _():
    df = df.drop(["indicativo", "nombre", "horatmin", "horatmax", "horaHrMin", "dir", "horaracha", "horaPresMax", "horaPresMin", "sol", "horaHrMax", "presMax", "presMin"])
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Added a Comunidad Autónoma (cc_aa) column for easier filtering later
    """)
    return


@app.cell
def _(pl):
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

    return (df,)


@app.cell
def _():
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
    return (df,)


@app.cell
def _():
    df = df.drop_nulls(subset=["temperatura_media", "precipitacion", "temperatura_minima", "temperatura_maxima", "humedad_relativa_media", "humedad_relativa_maxima", "humedad_relativa_minima"])
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **velocidad_viento_media** and **racha_maxima_viento** are strings so, before transforming null values into the median, we had to first replace , for . and cast the column type to float. Afterwards we could substitute those values with the median
    """)
    return


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("velocidad_viento_media")
          .str.replace(",", ".")
          .cast(pl.Float64)
    )
    return (df,)


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("velocidad_viento_media")
          .fill_null(pl.col("velocidad_viento_media").median())
    )
    print(df["velocidad_viento_media"].median()) 
    return (df,)


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("racha_maxima_viento")
          .str.replace(",", ".")
          .cast(pl.Float64)
    )
    return (df,)


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("racha_maxima_viento")
          .fill_null(pl.col("racha_maxima_viento").median())
    )
    print(df["racha_maxima_viento"].median()) 
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Converting the numeric columns froms str to float so we can use graphs later
    """)
    return


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("temperatura_media", "temperatura_minima", "temperatura_maxima")
          .str.replace(",", ".")
          .cast(pl.Float64)
    )
    return (df,)


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("precipitacion")
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Checking how many non numeric values has the **precipitacion** column, since there are only a few, filtering without those rows. Afterwards converting to float too
    """)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('precipitacion') == 'Acum').height
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('precipitacion') == 'Ip').height
    return


@app.cell
def _(pl):
    df = df.filter(pl.col('precipitacion') != 'Acum')
    df = df.filter(pl.col('precipitacion') != 'Ip')
    return (df,)


@app.cell
def _(pl):
    df = df.with_columns(
        pl.col("precipitacion")
          .str.replace(",", ".")
          .cast(pl.Float64)
    )
    return (df,)


@app.cell
def _():
    meses = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    return


@app.cell
def _(df):
    df.sample(5)
    return


@app.cell
def _(df):
    from pathlib import Path

    Path("../data/processed").mkdir(parents=True, exist_ok=True)
    df.write_parquet("../data/processed/aemet_clean.parquet")
    return


if __name__ == "__main__":
    app.run()
