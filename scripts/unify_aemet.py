"""Unify yearly AEMET CSV files into a single aemet_all.csv.

Usage:
  python scripts/unify_aemet.py
"""

import os
import pandas as pd

YEARS = range(2013, 2023)
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aemet_all.csv")

print("Unificando todos los años...")
all_years = []
for year in YEARS:
    path = os.path.join(OUTPUT_DIR, f"aemet_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_years.append(df)
        print(f"  {year}: {len(df)} filas")
    else:
        print(f"  {year}: archivo no encontrado, saltando")

if all_years:
    unified = pd.concat(all_years, ignore_index=True)
    unified.to_csv(OUTPUT_FILE, index=False)
    print(f"\nUnificado guardado: {OUTPUT_FILE} ({len(unified)} filas totales)")
else:
    print("No se encontraron archivos para unificar")
