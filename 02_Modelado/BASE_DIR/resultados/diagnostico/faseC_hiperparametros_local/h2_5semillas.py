# =====================================================================
# H2+ · E1-30, INTENTO 23 E INTENTO 27 CON 5 SEMILLAS (solo TRAIN; validación NO se usa)
#   Mismo esquema que H2: meses fuera ene, mar, may, jul · purga 24 h · hasta 80 épocas (medición cada 5).
#   Semillas 42, 123, 2024, 7, 99 (las de 42 y 123 ya están en la caché y se reutilizan).
#   Cada configuración se evalúa en su mejor época con la media de las 5 semillas (igual que el modelo final).
#   Criterio para pasar a H3 (fijado ANTES de ejecutar):
#     1) mejora en OPERACIÓN con IC95 por debajo de 0
#     2) no empeora de forma significativa en OPERACIÓN ni en la BANDA CENTRAL
#     3) no empeora el AMANECER más de un 4 %
#   Uso: en la misma carpeta que h1_optuna.py →  python h2_5semillas.py   (reanudable)
# =====================================================================
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import h1_optuna as H

SEMILLAS = [42, 123, 2024, 7, 99]
CONFIGS = [("E1-30", H.E1_30),
           ("intento 23", {"lr": 0.00034311277026846887, "batch": 128, "hidden": 64, "capas": 1, "wd": 7.457180693872641e-06}),
           ("intento 27", {"lr": 0.0003144507111458922, "batch": 128, "hidden": 64, "capas": 2, "wd": 2.1412554070126444e-06,
                           "dropout_capas": 0.0})]


def evaluar(configs, semillas):
    D = H.datos()
    idx = np.where(np.isin(D["mes"], H.MESES_CRIBA))[0]
    sol = D["sol"][idx]; y = D["y"][idx]; T = D["T"].iloc[idx]
    hora = T.index.hour.to_numpy()
    amanecer = (T["regimen"].astype(str).to_numpy() == "transicion") & (hora < 12)
    central = T["banda"].astype(str).str.startswith("central").to_numpy()
    res = {}
    for nombre, p in configs:
        P = np.zeros((len(H.EPOCAS), len(idx)))
        por_semilla = {}
        for s in semillas:
            Ps = np.zeros_like(P)
            for m in H.MESES_CRIBA:
                Ps[:, D["mes"][idx] == m] = H.obtener_fold(p, m, s)
            por_semilla[s] = Ps
            P += Ps / len(semillas)
        mae = np.abs(P[:, sol] - y[sol]).mean(axis=1)
        suav = pd.Series(mae, index=H.EPOCAS).rolling(3, center=True, min_periods=1).mean()
        ep = int(suav.idxmin()); k = list(H.EPOCAS).index(ep)
        res[nombre] = dict(p=p, ep=ep, err=np.abs(P[k] - y),
                           mae_semilla={s: np.abs(por_semilla[s][k][sol] - y[sol]).mean() for s in semillas})
    e0 = res["E1-30"]["err"]; rng = np.random.default_rng(0); f_idx = D["fechas"][idx]
    filas = []
    for nombre, r in res.items():
        d = r["err"] - e0
        fila = {"configuración": nombre, "mejor época (5 semillas)": r["ep"], "MAE operación": r["err"][sol].mean()}
        fila.update({f"MAE s{s}": v for s, v in r["mae_semilla"].items()})
        if nombre != "E1-30":
            c, lo, hi = H._ic_sub(d, sol, f_idx, rng)
            cc, lc, hc = H._ic_sub(d, sol & central, f_idx, rng)
            ca, la, ha = H._ic_sub(d, sol & amanecer, f_idx, rng)
            pct_am = 100 * ca / e0[sol & amanecer].mean()
            mejora_sem = sum(r["mae_semilla"][s] < res["E1-30"]["mae_semilla"][s] for s in semillas)
            ok1, ok2, ok3 = hi < 0, (lo <= 0) and (lc <= 0), pct_am <= H.LIMITE_AMANECER_PCT
            fila.update({"cambio operación": c, "IC95 operación": f"[{lo:+.2f}, {hi:+.2f}]", "cambio (%)": 100 * c / e0[sol].mean(),
                         "cambio central": cc, "IC95 central": f"[{lc:+.2f}, {hc:+.2f}]",
                         "cambio amanecer (%)": pct_am, "IC95 amanecer (W/m²)": f"[{la:+.2f}, {ha:+.2f}]",
                         "semillas en que mejora": f"{mejora_sem}/{len(semillas)}",
                         "1 IC95<0": ok1, "2 sin empeorar op./central": ok2, "3 amanecer ≤4 %": ok3,
                         "veredicto": "PASA A H3" if (ok1 and ok2 and ok3) else "no pasa"})
        else:
            fila["veredicto"] = "referencia"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("configuración")


def main():
    H.SALIDA.mkdir(exist_ok=True); H.CACHE.mkdir(exist_ok=True)
    H.log("=" * 70); H.log("H2+ · 5 semillas para E1-30, intento 23 e intento 27")
    tareas = [(p, m, s) for _, p in CONFIGS for s in SEMILLAS for m in H.MESES_CRIBA
              if not (H.CACHE / f"{H.clave(p)}_{m}_s{s}.npy").exists()]
    H.log(f"{len(tareas)} entrenamientos pendientes (los demás ya están en la caché).")
    if tareas:
        with ProcessPoolExecutor(max_workers=H.N_TRABAJADORES) as ex:
            for k, (c, m, s) in enumerate(ex.map(H._tarea_h2, tareas), 1):
                H.log(f"   {k}/{len(tareas)} hecho ({c} · {m} · semilla {s})")
    R = evaluar(CONFIGS, SEMILLAS)
    R.to_csv(H.SALIDA / "H2plus_5semillas.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    lineas = ["H2+ · 5 semillas (solo train, 4 meses fuera), por parejas frente a E1-30", "=" * 70]
    for nombre, r in R.iterrows():
        if nombre == "E1-30":
            lineas.append(f"  E1-30: MAE {r['MAE operación']:.2f} W/m² (mejor época {int(r['mejor época (5 semillas)'])})")
            continue
        lineas.append(f"  {nombre}: MAE {r['MAE operación']:.2f} · cambio {r['cambio operación']:+.2f} W/m² ({r['cambio (%)']:+.1f} %) "
                      f"{r['IC95 operación']} · central {r['cambio central']:+.2f} {r['IC95 central']} · "
                      f"amanecer {r['cambio amanecer (%)']:+.1f} % · mejora en {r['semillas en que mejora']} semillas · "
                      f"época {int(r['mejor época (5 semillas)'])} → {r['veredicto']}")
    texto = "\n".join(lineas)
    (H.SALIDA / "INFORME_H2plus.txt").write_text(texto, encoding="utf-8")
    H.log("\n" + texto)
    H.log("FIN H2+.")


if __name__ == "__main__":
    main()