# =====================================================================
# H2+ · AJUSTE DE BONFERRONI (2 candidatos frente a E1-30) — solo lee la caché, no entrena nada
#   Con 2 comparaciones, cada intervalo pasa del 95 % al 97,5 % (percentiles 1,25 y 98,75 del bootstrap por días).
#   Mismos datos, semillas, épocas y bootstrap (2000 remuestras, semilla 0) que H2+.
#   Uso: en la carpeta de h1_optuna.py →  python h2plus_bonferroni.py
# =====================================================================
import numpy as np, pandas as pd
import h1_optuna as H
from h2_5semillas import CONFIGS, SEMILLAS

N_COMPARACIONES = 2
ALFA = 0.05 / N_COMPARACIONES


def ic(d, msk, fechas, rng, alfa):
    dias = fechas.normalize().to_numpy()[msk]; dd = d[msk]
    u, inv = np.unique(dias, return_inverse=True)
    g = [np.where(inv == i)[0] for i in range(len(u))]
    medias = [dd[np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])].mean() for _ in range(H.N_BOOT)]
    return float(dd.mean()), float(np.percentile(medias, 100 * alfa / 2)), float(np.percentile(medias, 100 * (1 - alfa / 2)))


def main():
    for f in [H.CACHE / f"{H.clave(p)}_{m}_s{s}.npy" for _, p in CONFIGS for s in SEMILLAS for m in H.MESES_CRIBA]:
        assert f.exists(), f"Falta {f.name}: ejecuta antes h2_5semillas.py"
    D = H.datos()
    idx = np.where(np.isin(D["mes"], H.MESES_CRIBA))[0]
    sol = D["sol"][idx]; y = D["y"][idx]; fechas = D["fechas"][idx]
    central = D["T"].iloc[idx]["banda"].astype(str).str.startswith("central").to_numpy()
    err = {}
    for nombre, p in CONFIGS:
        P = np.zeros((len(H.EPOCAS), len(idx)))
        for s in SEMILLAS:
            for m in H.MESES_CRIBA:
                P[:, D["mes"][idx] == m] += H.obtener_fold(p, m, s) / len(SEMILLAS)
        mae = np.abs(P[:, sol] - y[sol]).mean(axis=1)
        ep = int(pd.Series(mae, index=H.EPOCAS).rolling(3, center=True, min_periods=1).mean().idxmin())
        err[nombre] = np.abs(P[list(H.EPOCAS).index(ep)] - y)
    filas = []
    for nombre, _ in CONFIGS[1:]:
        d = err[nombre] - err["E1-30"]
        for zona, msk in [("operación", sol), ("banda central", sol & central)]:
            c, lo95, hi95 = ic(d, msk, fechas, np.random.default_rng(0), 0.05)
            _, loB, hiB = ic(d, msk, fechas, np.random.default_rng(0), ALFA)
            filas.append({"candidato": nombre, "zona": zona, "cambio [W/m²]": round(c, 2),
                          "IC95 (sin ajuste)": f"[{lo95:+.2f}, {hi95:+.2f}]",
                          f"IC{100*(1-ALFA):.1f} (Bonferroni)": f"[{loB:+.2f}, {hiB:+.2f}]",
                          "¿por debajo de 0 tras el ajuste?": "sí" if hiB < 0 else "no"})
    R = pd.DataFrame(filas)
    R.to_csv(H.SALIDA / "H2plus_bonferroni.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\nH2+ · Ajuste de Bonferroni por 2 comparaciones (5 semillas, train, 4 meses fuera)")
    print("=" * 100)
    print(R.to_string(index=False))
    print("\nGuardado en:", H.SALIDA / "H2plus_bonferroni.csv")


if __name__ == "__main__":
    main()