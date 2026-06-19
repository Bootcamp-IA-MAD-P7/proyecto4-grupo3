import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AEMET_API_KEY")
if not API_KEY:
    raise ValueError("AEMET_API_KEY no encontrada en .env")

YEAR = 2022
START_DATE = datetime(YEAR, 1, 1)
END_DATE = datetime(YEAR, 12, 31)

OUTPUT_FILE = f"data/raw/aemet_{YEAR}.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

HEADERS = {"api_key": API_KEY}

BASE_URL = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/todasestaciones"

start_time = time.time()
all_data = []


def request_with_retry(url, max_retries=10):
    wait = 5
    for i in range(max_retries):
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            print(f"429 -> esperando {wait}s (reintento {i+1}/{max_retries})")
            time.sleep(wait)
            wait *= 2
            continue
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        time.sleep(wait)
    raise Exception("Fallo tras múltiples reintentos")


def fetch_data(start, end):
    url = BASE_URL.format(
        start.strftime("%Y-%m-%dT00:00:00UTC"),
        end.strftime("%Y-%m-%dT23:59:59UTC"),
    )
    meta = request_with_retry(url)
    if "datos" not in meta:
        raise Exception(f"API error: {meta.get('descripcion', str(meta))}")
    data_url = meta["datos"]

    wait = 3
    for i in range(6):
        resp = requests.get(data_url, timeout=180)
        if resp.status_code == 200 and resp.text.strip():
            try:
                data = resp.json()
                return pd.DataFrame(data)
            except Exception:
                pass
        print(f"  data URL vacio/error (intento {i+1}/6), esperando {wait}s...")
        time.sleep(wait)
        wait *= 2
    raise Exception("No se pudo obtener datos de la URL final")


total_blocks = 0
current = START_DATE

while current < END_DATE:
    total_blocks += 1
    next_date = min(current + timedelta(days=14), END_DATE)

    print(f"\nBloque {total_blocks}: {current.date()} -> {next_date.date()}")

    try:
        df = fetch_data(current, next_date)
        all_data.append(df)

        df_total = pd.concat(all_data, ignore_index=True)
        df_total.to_csv(OUTPUT_FILE, index=False)

        elapsed = time.time() - start_time
        print(f"OK -> filas totales: {len(df_total)} | {elapsed/60:.2f} min")

        time.sleep(1.5)

    except Exception as e:
        print(f"ERROR: {e}")
        raise

    current = next_date + timedelta(days=1)

df_final = pd.concat(all_data, ignore_index=True)
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\nDescarga completa. Total filas: {len(df_final)}")
print(f"Tiempo total: {(time.time()-start_time)/60:.2f} min")
print(f"Provincias: {sorted(df_final['provincia'].unique())}")