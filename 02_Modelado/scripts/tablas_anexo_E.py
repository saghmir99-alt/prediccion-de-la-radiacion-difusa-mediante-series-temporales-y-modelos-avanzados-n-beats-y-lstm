"""Tablas E1 y E2 de la memoria: métricas completas por régimen y banda en validación y en el test.

Se calculan después del test, a partir de las predicciones ya guardadas; no se entrena ni se elige nada.
Uso, desde la raíz del repositorio: python 02_Modelado/scripts/tablas_anexo_E.py 02_Modelado/BASE_DIR
Escribe las dos tablas en BASE_DIR/resultados/diagnostico/T_test/anexo_E/. Las que recoge la memoria están en
anexo_E/, junto a este script.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DIAG = BASE / "resultados" / "diagnostico"


def metricas(y, p):
    e = p - y
    return {
        "MAE": np.abs(e).mean(),
        "RMSE": np.sqrt((e**2).mean()),
        "sesgo": e.mean(),
        "nMAE (%)": 100 * np.abs(e).mean() / y.mean(),
        "R2": 1 - (e**2).sum() / ((y - y.mean()) ** 2).sum(),
    }


def grupos(df):
    sol = df["regimen"] != "noche"
    transicion = df["regimen"] == "transicion"
    g = {
        "Horas con sol": sol,
        "Completamente diurno": df["regimen"] == "completamente_diurno",
        "Transición": transicion,
        "Amanecer": transicion & (df.index.hour < 12),
    }
    for prefijo, nombre in (("hombros", "Hombros (≤ 20°)"), ("intermedia", "Intermedia (20–40°)"), ("central", "Central (> 40°)")):
        g[nombre] = sol & df["banda"].astype(str).str.startswith(prefijo)
    return g


def tabla(df, modelos):
    filas = []
    for grupo, m in grupos(df).items():
        if m.sum() == 0:
            continue
        y = df.loc[m, "real"].to_numpy(float)
        for nombre, col in modelos.items():
            filas.append({"grupo": grupo, "horas": int(m.sum()), "DIF media": y.mean(), "modelo": nombre,
                          **metricas(y, df.loc[m, col].to_numpy(float))})
    return pd.DataFrame(filas)


# Validación (agosto y septiembre de 2024): modelos entrenados solo con entrenamiento
tabla_horaria = pd.read_pickle(DIAG / "faseB_radiografia_E1" / "tabla_horaria_E1.pkl")
tabla_horaria = tabla_horaria[tabla_horaria["conjunto"] == "validation"]
val = pd.read_pickle(DIAG / "nbeats" / "N6b_confirmacion" / "N6b_predicciones_validacion.pkl")
val = val.join(tabla_horaria[["regimen", "banda", "persistencia"]])
E1 = tabla(val, {"Persistencia simple": "persistencia", "PI1": "persistencia inteligente del último minuto",
                 "E1-30": "LSTM E1-30", "N-BEATS-1h": "N-BEATS-1h"})

# Test (octubre y noviembre de 2024): modelos principales, reentrenados con entrenamiento y validación
test = pd.read_pickle(DIAG / "T_test" / "T3_resultados" / "T3_predicciones_y_bandas.pkl")
test = test.join(pd.read_pickle(DIAG / "T_test" / "tabla_test.pkl")[["regimen", "banda"]])
E2 = tabla(test, {"Persistencia simple": "persistencia simple", "PI horaria": "persistencia inteligente horaria",
                  "PI1": "persistencia inteligente último minuto", "E1-30": "E1-30 train+val",
                  "N-BEATS-1h": "N-BEATS-1h train+val"})

SALIDA = DIAG / "T_test" / "anexo_E"
SALIDA.mkdir(parents=True, exist_ok=True)
E1.round(4).to_csv(SALIDA / "tabla_E1_validacion.csv", index=False)
E2.round(4).to_csv(SALIDA / "tabla_E2_test.csv", index=False)
pd.set_option("display.width", 160)
print(E1.round(2).to_string(index=False))
print()
print(E2.round(2).to_string(index=False))
print("\nGuardado en:", SALIDA)
