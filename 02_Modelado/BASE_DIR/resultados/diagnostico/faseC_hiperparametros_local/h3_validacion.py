# =====================================================================
# H3 · CONFIRMACIÓN EN VALIDACIÓN (ago–sep 2024) · UNA SOLA VEZ
#   E1-30 (lr 0,001 · 1 capa · 64 unidades · 30 épocas)  frente a  intento 27 (lr 0,00031 · 2 capas · 64 · N épocas de H2e)
#   Ambos entrenados con TODO train, 5 semillas (42, 123, 2024, 7, 99), en este PC. Predicción = media de las 5 semillas.
#   Criterio (fijado ANTES de ejecutar), horas con sol de validación, bootstrap por días (2000 remuestras):
#     1) mejora en OPERACIÓN con IC95 por debajo de 0
#     2) no empeora de forma significativa en OPERACIÓN ni en la BANDA CENTRAL
#     3) no empeora el AMANECER más de un 4 %
#   Complemento (no decide): test de Diebold–Mariano sobre las diferencias diarias de error (Newey–West + corrección HLN).
#   Una sola comparación → no hace falta Bonferroni.
#   Necesita: carpeta datos_validacion junto a este script · epocas_H3_intento27.txt (de H2e). Reanudable.
# =====================================================================
import os, random, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import h1_optuna as H

SEMILLAS = [42, 123, 2024, 7, 99]
DVAL = H.CARPETA / "datos_validacion"
DIR3 = H.SALIDA / "H3_validacion"
P27 = {"lr": 0.0003144507111458922, "batch": 128, "hidden": 64, "capas": 2, "wd": 2.1412554070126444e-06, "dropout_capas": 0.0}


def epocas_27():
    return int((H.SALIDA / "epocas_H3_intento27.txt").read_text().strip())


def modelos():
    return [("E1-30", H.E1_30, 30), ("intento 27", P27, epocas_27())]


_V = {}
def datos_val():
    if _V:
        return _V
    z = np.load(DVAL / "entradas_validacion.npz")
    Xb, y = z["X_val"].astype(np.float32), z["y_val"].astype(np.float32)
    fechas = pd.DatetimeIndex(pd.to_datetime(z["fechas_val"]))
    intra = pd.read_csv(DVAL / "variables_intrahorarias_val.csv", index_col=0, parse_dates=True).asfreq("h")
    f = fechas.to_numpy(dtype="datetime64[ns]")
    idx = f[:, None] - np.arange(24, 0, -1).astype("timedelta64[h]")[None, :]
    Xn = intra.reindex(pd.DatetimeIndex(idx.reshape(-1)))[H.CANALES_E1].to_numpy(dtype=np.float32).reshape(len(f), 24, len(H.CANALES_E1))
    for j, c in enumerate(H.CANALES_E1):
        hueco = np.isnan(Xn[:, :, j])
        Xn[:, :, j][hueco] = Xb[:, :, 0][hueco] if c == "DIF_ult15" else 0.0
    T = pd.read_csv(DVAL / "tabla_validacion.csv", index_col=0, parse_dates=True)
    assert len(T) == len(y) and (T.index == fechas).all(), "tabla_validacion.csv no cuadra"
    assert fechas.max() < pd.Timestamp("2024-10-01"), "Hay datos del test"
    _V.update(X=np.concatenate([Xb, Xn], axis=2).astype(np.float32), y=y, fechas=fechas, T=T)
    return _V


def entrenar_completo(args):
    """Entrena con TODO train (protocolo E1) y devuelve la predicción en validación. Con caché."""
    nombre, p, epocas, semilla = args
    ruta = DIR3 / f"{H.clave(p)}_ep{epocas}_s{semilla}.npy"
    if ruta.exists():
        return nombre, semilla
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(H.HILOS_POR_TRABAJADOR)
    D, V = H.datos(), datos_val()
    Xtr, ytr = D["X"], D["y"]
    sx = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2])); sy = StandardScaler().fit(ytr.reshape(-1, 1))
    e = lambda A: torch.from_numpy(sx.transform(A.reshape(-1, A.shape[2])).reshape(A.shape).astype(np.float32))
    ytr_e = sy.transform(ytr.reshape(-1, 1)).ravel().astype(np.float32)

    class Modelo(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=Xtr.shape[2], hidden_size=p["hidden"], num_layers=p["capas"], batch_first=True,
                                dropout=p.get("dropout_capas", 0.0) if p["capas"] > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(p["hidden"], 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            s, _ = self.lstm(x)
            return self.head(s[:, -1, :]).squeeze(-1)

    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True)
    modelo = Modelo()
    gen = torch.Generator(); gen.manual_seed(semilla)
    loader = DataLoader(TensorDataset(e(Xtr), torch.from_numpy(ytr_e)), batch_size=p["batch"], shuffle=True,
                        generator=gen, num_workers=0)
    opt = torch.optim.Adam(modelo.parameters(), lr=p["lr"], weight_decay=p["wd"])
    crit = nn.L1Loss()
    for _ in range(epocas):
        modelo.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(modelo(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            opt.step()
    modelo.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(modelo(e(V["X"])).numpy().reshape(-1, 1)).ravel().astype(np.float32)
    torch.save({"model_state_dict": modelo.state_dict(), "params": p, "epocas": epocas, "semilla": semilla},
               DIR3 / f"modelo_{H.clave(p)}_ep{epocas}_s{semilla}.pt")
    tmp = ruta.with_suffix(".tmp.npy"); np.save(tmp, pred); os.replace(tmp, ruta)
    return nombre, semilla


def diebold_mariano(d_horas, fechas, msk):
    """DM sobre la serie de diferencias medias DIARIAS (horas con sol), varianza Newey–West y corrección HLN."""
    from scipy import stats
    s = pd.Series(d_horas[msk], index=fechas[msk]).groupby(fechas[msk].normalize()).mean()
    d = s.to_numpy(); T = len(d); dbar = d.mean(); L = int(np.floor(T ** (1 / 3)))
    dc = d - dbar
    var = dc @ dc / T
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / T
    dm = dbar / np.sqrt(var / T)
    dm_hln = dm * np.sqrt((T - 1) / T)                      # Harvey–Leybourne–Newbold (h = 1)
    p = 2 * stats.t.sf(abs(dm_hln), df=T - 1)
    return dm_hln, p, T, L


def main():
    DIR3.mkdir(parents=True, exist_ok=True)
    H.log("=" * 70); H.log("H3 · confirmación en VALIDACIÓN (una sola vez)")
    MOD = modelos()
    for n, _, ep in MOD:
        H.log(f"   {n}: {ep} épocas")
    V = datos_val()
    H.log(f"Validación: {len(V['y'])} horas ({V['fechas'].min()} → {V['fechas'].max()})")
    tareas = [(n, p, ep, s) for n, p, ep in MOD for s in SEMILLAS
              if not (DIR3 / f"{H.clave(p)}_ep{ep}_s{s}.npy").exists()]
    H.log(f"{len(tareas)} entrenamientos con todo train pendientes.")
    if tareas:
        with ProcessPoolExecutor(max_workers=H.N_TRABAJADORES) as ex:
            for k, (n, s) in enumerate(ex.map(entrenar_completo, tareas), 1):
                H.log(f"   {k}/{len(tareas)} hecho ({n} · semilla {s})")

    y, T, fechas = V["y"], V["T"], V["fechas"]
    sol = (T["regimen"] != "noche").to_numpy()
    hora = fechas.hour.to_numpy()
    amanecer = sol & (T["regimen"].astype(str).to_numpy() == "transicion") & (hora < 12)
    central = sol & T["banda"].astype(str).str.startswith("central").to_numpy()
    pred, por_semilla = {}, {}
    for n, p, ep in MOD:
        P = np.array([np.load(DIR3 / f"{H.clave(p)}_ep{ep}_s{s}.npy") for s in SEMILLAS])
        pred[n] = P.mean(axis=0)
        por_semilla[n] = {s: np.abs(P[i][sol] - y[sol]).mean() for i, s in enumerate(SEMILLAS)}
    e0, e1 = np.abs(pred["E1-30"] - y), np.abs(pred["intento 27"] - y)
    d = e1 - e0
    rng = np.random.default_rng(0)
    c, lo, hi = H._ic_sub(d, sol, fechas, rng)
    cc, lc, hc = H._ic_sub(d, central, fechas, rng)
    ca, la, ha = H._ic_sub(d, amanecer, fechas, rng)
    pct_am = 100 * ca / e0[amanecer].mean()
    dm, p_dm, n_dias, L = diebold_mariano(d, fechas, sol)
    ok1, ok2, ok3 = hi < 0, (lo <= 0) and (lc <= 0), pct_am <= H.LIMITE_AMANECER_PCT
    mae_p = np.nanmean(np.abs(T["persistencia"].to_numpy()[sol] - y[sol]))

    # Desglose por grupos (informativo)
    filas = []
    grupos = {"TODAS las horas con sol": sol}
    for col in ["regimen", "banda", "cielo"]:
        for g in sorted(T[col].astype(str).unique()):
            if g != "noche":
                grupos[f"{col}: {g}"] = sol & (T[col].astype(str).to_numpy() == g)
    grupos["amanecer (transición, UTC < 12)"] = amanecer
    for g, msk in grupos.items():
        cg, lg, hg = H._ic_sub(d, msk, fechas, rng)
        filas.append({"grupo": g, "horas": int(msk.sum()), "MAE E1-30": e0[msk].mean(), "MAE intento 27": e1[msk].mean(),
                      "cambio [W/m²]": cg, "cambio (%)": 100 * cg / e0[msk].mean(), "IC95": f"[{lg:+.2f}, {hg:+.2f}]",
                      "lectura": "mejora" if hg < 0 else ("EMPEORA" if lg > 0 else "igual")})
    G = pd.DataFrame(filas).set_index("grupo"); G.round(3).to_csv(DIR3 / "H3_desglose.csv")
    S = pd.DataFrame(por_semilla); S.index.name = "semilla"; S.round(3).to_csv(DIR3 / "H3_por_semilla.csv")
    pd.DataFrame({"fecha": fechas, "real": y, "E1-30": pred["E1-30"], "intento 27": pred["intento 27"]}).to_csv(
        DIR3 / "H3_predicciones_validacion.csv", index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
    axs[0].bar([0, 1], [e0[sol].mean(), e1[sol].mean()], color=["tab:gray", "tab:green" if ok1 else "tab:blue"])
    for i, v in enumerate([e0[sol].mean(), e1[sol].mean()]):
        axs[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["E1-30", f"intento 27 ({MOD[1][2]} ép.)"])
    axs[0].set_ylabel("MAE validación, horas con sol [W/m²]"); axs[0].set_title("Error en validación (más bajo = mejor)")
    t = G.drop(index="TODAS las horas con sol")
    col = ["tab:green" if l == "mejora" else ("tab:red" if l == "EMPEORA" else "lightgray") for l in t["lectura"]]
    axs[1].barh(range(len(t)), t["cambio [W/m²]"], color=col); axs[1].axvline(0, color="k", lw=.8)
    axs[1].set_yticks(range(len(t))); axs[1].set_yticklabels([s.replace("nubes rotas", "nubosidad variable") for s in t.index], fontsize=8)
    axs[1].invert_yaxis(); axs[1].set_xlabel("cambio de MAE (intento 27 − E1-30) [W/m²]")
    axs[1].set_title("¿Dónde mejora o empeora?\n(verde = mejora real · gris = igual · rojo = empeora)")
    plt.tight_layout(); plt.savefig(DIR3 / "H3_validacion.png", dpi=120); plt.close()

    lineas = ["H3 · CONFIRMACIÓN EN VALIDACIÓN (ago–sep 2024, horas con sol, 5 semillas, una sola vez)", "=" * 78,
              f"E1-30:      MAE {e0[sol].mean():.2f} W/m²  (ventaja sobre persistencia {100*(1-e0[sol].mean()/mae_p):.1f} %)",
              f"intento 27: MAE {e1[sol].mean():.2f} W/m²  (ventaja sobre persistencia {100*(1-e1[sol].mean()/mae_p):.1f} %)",
              "",
              f"1) Operación:     {c:+.2f} W/m² ({100*c/e0[sol].mean():+.1f} %) IC95 [{lo:+.2f}, {hi:+.2f}] → {'cumple' if ok1 else 'NO cumple'}",
              f"2) Banda central: {cc:+.2f} W/m² IC95 [{lc:+.2f}, {hc:+.2f}] · sin empeorar op./central → {'cumple' if ok2 else 'NO cumple'}",
              f"3) Amanecer:      {pct_am:+.1f} % IC95 [{la:+.2f}, {ha:+.2f}] W/m² → {'cumple' if ok3 else 'NO cumple'} (límite +4 %)",
              f"Diebold–Mariano (diario, {n_dias} días, Newey–West L={L}, HLN): estadístico {dm:+.2f}, p = {p_dm:.4f}",
              f"Mejora por semilla: {sum(por_semilla['intento 27'][s] < por_semilla['E1-30'][s] for s in SEMILLAS)}/5",
              "",
              ("VEREDICTO: SE ACEPTA el intento 27 como nuevo modelo base." if (ok1 and ok2 and ok3)
               else "VEREDICTO: NO SE ACEPTA. E1-30 se mantiene como modelo base.")]
    texto = "\n".join(lineas)
    (DIR3 / "INFORME_H3.txt").write_text(texto, encoding="utf-8")
    H.log("\n" + texto); H.log(f"FIN H3. Resultados en {DIR3}")


if __name__ == "__main__":
    main()