# =====================================================================
# H2e · ¿CUÁNTAS ÉPOCAS NECESITA EL INTENTO 27? (solo TRAIN; validación NO se usa)
#   El intento 27 tenía su mejor época en 80, justo el límite. Aquí se entrena hasta 150 épocas
#   (medición cada 5), con las 5 semillas y los mismos 4 meses fuera, en una caché aparte.
#   Regla (fijada ANTES de ejecutar): la época final del intento 27 para H3 = la mejor de la curva
#   suavizada de la media de 5 semillas, entre 5 y 150. Se informa también de su ventaja sobre E1-30
#   (E1-30 en su mejor época, de la caché de H2+).
#   Control: con la misma semilla, las épocas 5–80 deben coincidir con las ya guardadas en H2+.
#   Uso: en la carpeta de h1_optuna.py →  python h2e_epocas.py   (reanudable)
# =====================================================================
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import h1_optuna as H

SEMILLAS = [42, 123, 2024, 7, 99]
EPOCAS_LARGAS = 150
CACHE_LARGA = H.SALIDA / "cache_150_epocas"
P27 = {"lr": 0.0003144507111458922, "batch": 128, "hidden": 64, "capas": 2, "wd": 2.1412554070126444e-06, "dropout_capas": 0.0}


def _modo_largo():
    H.EPOCAS_MAX = EPOCAS_LARGAS
    H.CACHE = CACHE_LARGA


def tarea(args):
    _modo_largo()                       # en cada proceso trabajador
    p, m, s = args
    H.obtener_fold(p, m, s)
    return m, s


def main():
    H.SALIDA.mkdir(exist_ok=True); CACHE_LARGA.mkdir(exist_ok=True)
    cache_80 = H.CACHE
    H.log("=" * 70); H.log(f"H2e · intento 27 hasta {EPOCAS_LARGAS} épocas · 5 semillas")
    tareas = [(P27, m, s) for s in SEMILLAS for m in H.MESES_CRIBA
              if not (CACHE_LARGA / f"{H.clave(P27)}_{m}_s{s}.npy").exists()]
    H.log(f"{len(tareas)} entrenamientos pendientes.")
    if tareas:
        with ProcessPoolExecutor(max_workers=H.N_TRABAJADORES) as ex:
            for k, (m, s) in enumerate(ex.map(tarea, tareas), 1):
                H.log(f"   {k}/{len(tareas)} hecho ({m} · semilla {s})")

    D = H.datos()
    idx = np.where(np.isin(D["mes"], H.MESES_CRIBA))[0]
    sol = D["sol"][idx]; y = D["y"][idx]
    ep_l = np.arange(H.CADA, EPOCAS_LARGAS + 1, H.CADA); ep_c = np.arange(H.CADA, 80 + 1, H.CADA)

    # Control de reproducibilidad: épocas 5–80 frente a la caché de H2+
    dif_max = 0.0
    for s in SEMILLAS:
        for m in H.MESES_CRIBA:
            largo = np.load(CACHE_LARGA / f"{H.clave(P27)}_{m}_s{s}.npy")
            corto = np.load(cache_80 / f"{H.clave(P27)}_{m}_s{s}.npy")
            dif_max = max(dif_max, float(np.abs(largo[:len(ep_c)] - corto).max()))
    H.log(f"Control: diferencia máxima épocas 5–80 frente a H2+ = {dif_max:.4f} W/m² (≈0 = reproducible)")

    def media(cache, p, n_ep):
        P = np.zeros((n_ep, len(idx)))
        for s in SEMILLAS:
            for m in H.MESES_CRIBA:
                P[:, D["mes"][idx] == m] += np.load(cache / f"{H.clave(p)}_{m}_s{s}.npy") / len(SEMILLAS)
        return P

    P_l = media(CACHE_LARGA, P27, len(ep_l))
    mae = np.abs(P_l[:, sol] - y[sol]).mean(axis=1)
    suav = pd.Series(mae, index=ep_l).rolling(3, center=True, min_periods=1).mean()
    ep_best = int(suav.idxmin())
    P_0 = media(cache_80, H.E1_30, len(ep_c))
    mae0 = np.abs(P_0[:, sol] - y[sol]).mean(axis=1)
    suav0 = pd.Series(mae0, index=ep_c).rolling(3, center=True, min_periods=1).mean()
    ep0 = int(suav0.idxmin())
    e27 = np.abs(P_l[list(ep_l).index(ep_best)] - y); e0 = np.abs(P_0[list(ep_c).index(ep0)] - y)
    c, lo, hi = H._ic_sub(e27 - e0, sol, D["fechas"][idx], np.random.default_rng(0))

    tabla = pd.DataFrame({"MAE (5 semillas)": mae, "suavizado": suav.values}, index=ep_l)
    tabla.index.name = "época"; tabla.to_csv(H.SALIDA / "H2e_curva_intento27.csv")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ep_l, suav.values, color="tab:blue", lw=2, label="intento 27 (hasta 150 épocas)")
    ax.plot(ep_c, suav0.values, color="black", lw=2, label="E1-30 (hasta 80 épocas)")
    ax.axvline(80, color="gray", ls=":", label="límite anterior (80)")
    ax.scatter([ep_best], [suav.loc[ep_best]], color="tab:blue", s=60, zorder=3)
    ax.scatter([ep0], [suav0.loc[ep0]], color="black", s=60, zorder=3)
    ax.set_xlabel("época"); ax.set_ylabel("MAE meses no vistos [W/m²]"); ax.legend(fontsize=8)
    ax.set_title("¿Sigue mejorando el intento 27 después de la época 80?\n(el punto marca la mejor época)")
    plt.tight_layout(); plt.savefig(H.SALIDA / "H2e_epocas_intento27.png", dpi=120); plt.close()

    texto = "\n".join([
        f"H2e · intento 27 hasta {EPOCAS_LARGAS} épocas (5 semillas, train, 4 meses fuera)", "=" * 70,
        f"Mejor época: {ep_best} → MAE {suav.loc[ep_best]:.2f} W/m² (en la época 80: {suav.loc[80]:.2f})",
        f"E1-30 en su mejor época ({ep0}): MAE {suav0.loc[ep0]:.2f} W/m²",
        f"Cambio frente a E1-30 en la mejor época: {c:+.2f} W/m² [{lo:+.2f}, {hi:+.2f}]",
        f"ÉPOCAS PARA H3 (regla fijada): {ep_best}"])
    (H.SALIDA / "INFORME_H2e.txt").write_text(texto, encoding="utf-8")
    (H.SALIDA / "epocas_H3_intento27.txt").write_text(str(ep_best), encoding="utf-8")
    H.log("\n" + texto); H.log("FIN H2e.")


if __name__ == "__main__":
    main()