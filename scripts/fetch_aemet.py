import os
import sys
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AEMET_API_KEY")
if not API_KEY:
    raise ValueError("AEMET_API_KEY no encontrada en .env")

if len(sys.argv) > 1:
    YEARS = [int(sys.argv[1])]
else:
    YEARS = range(2013, 2023)

OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"api_key": API_KEY}

BASE_URL = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/todasestaciones"

total_start = time.time()


def request_with_retry(url, max_retries=10):
    wait = 5
    for i in range(max_retries):
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            print(f"  429 -> esperando {wait}s (reintento {i+1}/{max_retries})")
            time.sleep(wait)
            wait *= 2
            continue
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
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


for year in YEARS:
    print(f"\n{'='*60}")
    print(f"AÑO {year}")
    print(f"{'='*60}")

    year_start = time.time()
    all_data = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    output_file = os.path.join(OUTPUT_DIR, f"aemet_{year}.csv")

    total_blocks = 0
    current = start_date

    while current < end_date:
        total_blocks += 1
        next_date = min(current + timedelta(days=14), end_date)

        print(f"\nBloque {total_blocks}: {current.date()} -> {next_date.date()}")

        try:
            df = fetch_data(current, next_date)
            all_data.append(df)

            df_total = pd.concat(all_data, ignore_index=True)
            df_total.to_csv(output_file, index=False)

            elapsed = time.time() - year_start
            print(f"OK -> filas: {len(df_total)} | {elapsed/60:.2f} min año")

            time.sleep(1.5)

        except Exception as e:
            print(f"ERROR: {e}")
            raise

        current = next_date + timedelta(days=1)

    df_final = pd.concat(all_data, ignore_index=True)
    df_final.to_csv(output_file, index=False)

    year_elapsed = time.time() - year_start
    total_elapsed = time.time() - total_start
    print(f"\nAño {year} completado. Filas: {len(df_final)} | Tiempo: {year_elapsed/60:.2f} min")
    print(f"Tiempo total transcurrido: {total_elapsed/60:.2f} min")
    print(f"Provincias: {sorted(df_final['provincia'].unique())}")

print(f"\n{'='*60}")
print(f"DESCARGA COMPLETA (2013-2022)")
print(f"{'='*60}")
print(f"Tiempo total: {(time.time()-total_start)/60:.2f} min")
