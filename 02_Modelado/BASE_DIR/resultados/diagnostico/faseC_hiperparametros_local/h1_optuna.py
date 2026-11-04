# =====================================================================
# H1 + H2 · OPTIMIZACIÓN DE HIPERPARÁMETROS DE E1-30 CON OPTUNA (en el PC, CPU, en paralelo)
#
#   Uso (en Anaconda Prompt, con el entorno activado y dentro de la carpeta optuna_local):
#       python h1_optuna.py
#   Si se corta (reinicio, apagón…), se vuelve a lanzar el MISMO comando y sigue donde lo dejó.
#
#   H1 · Búsqueda: solo TRAIN · meses fuera ene, mar, may, jul · purga 24 h · semilla 42 · hasta 80 épocas
#        (medición cada 5; cada combinación se evalúa en su mejor época) · 30 intentos · TPE + MedianPruner.
#        El intento 0 es E1-30 (referencia). Se ejecutan N_TRABAJADORES intentos a la vez.
#   H2 · Contra la suerte: las 3 mejores combinaciones + E1-30, repetidas con la semilla 123.
#        Criterio (fijado antes), con 2 semillas y por parejas frente a E1-30 (bootstrap por días):
#          1) baja el MAE en OPERACIÓN ≥ 0,5 W/m² con IC95 < 0
#          2) no empeora de forma significativa en OPERACIÓN ni en la BANDA CENTRAL
#          3) no empeora el AMANECER más de un 4 %
#   Validación NO se usa aquí (H3 se hará aparte, una sola vez).
#   Resultados: carpeta "resultados_optuna" (tablas CSV, gráficas PNG e INFORME.txt).
# =====================================================================
import os, sys, json, time, hashlib, random, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ---------------------------- CONFIGURACIÓN ----------------------------
CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos"
SALIDA = CARPETA / "resultados_optuna"
CACHE = SALIDA / "cache_pliegues"
BD = SALIDA / "estudio_optuna.db"
LOG = SALIDA / "registro.txt"

N_INTENTOS = 30
N_TRABAJADORES = 4            # intentos a la vez (i5-12450H: 4 × 2 hilos = 8 de 12)
HILOS_POR_TRABAJADOR = 2
MESES_CRIBA = ["2024-01", "2024-03", "2024-05", "2024-07"]
EPOCAS_MAX, CADA = 80, 5
SEMILLA_BUSQUEDA, SEMILLA_H2 = 42, 123
TOP_H2 = 3
EFECTO_MINIMO, LIMITE_AMANECER_PCT, N_BOOT = 0.5, 4.0, 2000
E1_30 = {"lr": 1e-3, "batch": 128, "hidden": 64, "capas": 1, "wd": 1e-5}      # referencia (intento 0)
CANALES_E1 = ["DIF_ult15", "DIF_pend30", "DIF_acel60", "DIF_saltos", "SWD_cv"]


def log(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


# ---------------------------- DATOS (carga perezosa: cada proceso los lee una vez) ----------------------------
_D = {}
def datos():
    if _D:
        return _D
    z = np.load(DATOS / "entradas_train.npz")
    X_base, y = z["X_train"].astype(np.float32), z["y_train"].astype(np.float32)
    fechas = pd.DatetimeIndex(pd.to_datetime(z["fechas_train"]))
    intra = pd.read_csv(DATOS / "variables_intrahorarias.csv", index_col=0, parse_dates=True).asfreq("h")
    f = fechas.to_numpy(dtype="datetime64[ns]")
    idx = f[:, None] - np.arange(24, 0, -1).astype("timedelta64[h]")[None, :]
    Xn = intra.reindex(pd.DatetimeIndex(idx.reshape(-1)))[CANALES_E1].to_numpy(dtype=np.float32).reshape(len(f), 24, len(CANALES_E1))
    for j, c in enumerate(CANALES_E1):
        hueco = np.isnan(Xn[:, :, j])
        Xn[:, :, j][hueco] = X_base[:, :, 0][hueco] if c == "DIF_ult15" else 0.0
    X = np.concatenate([X_base, Xn], axis=2).astype(np.float32)
    T = pd.read_csv(DATOS / "tabla_train.csv", index_col=0, parse_dates=True)
    assert len(T) == len(y) and (T.index == fechas).all(), "tabla_train.csv no cuadra con entradas_train.npz"
    assert np.allclose(T["real"].to_numpy(), y, atol=0.01), "El valor real no coincide"
    mes = fechas.to_period("M").astype(str).to_numpy()
    mes = np.where(mes == "2023-12", "2024-01", mes)
    _D.update(X=X, y=y, fechas=fechas, T=T, mes=mes, sol=(T["regimen"] != "noche").to_numpy())
    return _D


def pliegue(m):
    D = datos()
    fuera = D["mes"] == m
    fin = D["fechas"][fuera].max()
    purga = (D["fechas"] > fin) & (D["fechas"] <= fin + pd.Timedelta("24h"))
    return ~fuera & ~purga, fuera


# ---------------------------- ENTRENAMIENTO DE UN PLIEGUE ----------------------------
def clave(p):
    norm = {k: (round(float(v), 10) if isinstance(v, float) else v) for k, v in sorted(p.items())}
    return hashlib.sha1(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:12]


def entrenar_fold(p, m, semilla):
    """Entrena con el protocolo de E1 y devuelve las predicciones del mes no visto cada CADA épocas (épocas × horas)."""
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(HILOS_POR_TRABAJADOR)
    D = datos(); entrena, fuera = pliegue(m)
    Xtr, ytr, Xte = D["X"][entrena], D["y"][entrena], D["X"][fuera]
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
    crit = nn.L1Loss(); Xte_t = e(Xte); preds = []
    for ep in range(1, EPOCAS_MAX + 1):
        modelo.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(modelo(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            opt.step()
        if ep % CADA == 0:
            modelo.eval()
            with torch.no_grad():
                preds.append(sy.inverse_transform(modelo(Xte_t).numpy().reshape(-1, 1)).ravel())
    return np.array(preds, dtype=np.float32)


def obtener_fold(p, m, semilla):
    """Con caché en disco: si ya se entrenó (aunque fuera en un intento cortado), se reutiliza."""
    ruta = CACHE / f"{clave(p)}_{m}_s{semilla}.npy"
    if ruta.exists():
        try:
            return np.load(ruta)
        except Exception:
            ruta.unlink()
    P = entrenar_fold(p, m, semilla)
    tmp = ruta.with_suffix(".tmp.npy"); np.save(tmp, P); os.replace(tmp, ruta)
    return P


EPOCAS = np.arange(CADA, EPOCAS_MAX + 1, CADA)
def curva(preds_por_mes, meses):
    """MAE (horas con sol) por época juntando los meses dados; devuelve (mae_por_epoca, suavizado, mejor_época)."""
    D = datos()
    num, den = np.zeros(len(EPOCAS)), 0
    for m in meses:
        msk = D["mes"] == m
        s = D["sol"][msk]
        num += np.abs(preds_por_mes[m][:, s] - D["y"][msk][s]).sum(axis=1); den += s.sum()
    mae = num / den
    suav = pd.Series(mae, index=EPOCAS).rolling(3, center=True, min_periods=1).mean()
    return mae, suav.values, int(suav.idxmin())


# ---------------------------- H1 · OPTUNA ----------------------------
def almacen():
    import optuna
    from optuna.storages import RDBStorage, RetryFailedTrialCallback
    return RDBStorage(f"sqlite:///{BD.as_posix()}", engine_kwargs={"connect_args": {"timeout": 120}},
                      heartbeat_interval=60, grace_period=300, failed_trial_callback=RetryFailedTrialCallback(max_retry=3))


def espacio(trial):
    p = {"lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
         "batch": trial.suggest_categorical("batch", [64, 128, 256]),
         "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
         "capas": trial.suggest_int("capas", 1, 2),
         "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True)}
    if p["capas"] == 2:
        p["dropout_capas"] = trial.suggest_categorical("dropout_capas", [0.0, 0.2])
    return p


def objetivo(trial):
    import optuna
    p = espacio(trial)
    preds, t0 = {}, time.time()
    for k, m in enumerate(MESES_CRIBA):
        preds[m] = obtener_fold(p, m, SEMILLA_BUSQUEDA)
        _, suav, ep = curva(preds, MESES_CRIBA[:k + 1])
        valor = float(suav.min())
        trial.report(valor, step=k)
        if trial.should_prune():
            log(f"intento {trial.number}: PODADO tras {k+1} meses (MAE {valor:.2f}) · {p}")
            raise optuna.TrialPruned()
    trial.set_user_attr("mejor_epoca", ep)
    log(f"intento {trial.number}: MAE {valor:.2f} W/m² (época {ep}) en {(time.time()-t0)/60:.1f} min · {p}")
    return valor


def trabajador(i):
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name="E1_30_hiperparametros", storage=almacen(),
                              sampler=optuna.samplers.TPESampler(seed=100 + i, n_startup_trials=8, multivariate=True,
                                                                 constant_liar=True),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.optimize(objetivo, callbacks=[optuna.study.MaxTrialsCallback(N_INTENTOS, states=(TrialState.COMPLETE, TrialState.PRUNED))])
    return i


# ---------------------------- H2 · CONTRA LA SUERTE ----------------------------
def _tarea_h2(args):
    p, m, s = args
    obtener_fold(p, m, s)
    return clave(p), m, s


def evaluar_h2(configs):
    """configs: lista de (nombre, params). Devuelve tabla con el criterio aplicado frente a E1-30 (2 semillas)."""
    D = datos()
    idx = np.where(np.isin(D["mes"], MESES_CRIBA))[0]
    sol = D["sol"][idx]; y = D["y"][idx]
    T = D["T"].iloc[idx]
    hora = T.index.hour.to_numpy()
    amanecer = (T["regimen"].astype(str).to_numpy() == "transicion") & (hora < 12)
    central = T["banda"].astype(str).str.startswith("central").to_numpy()
    res = {}
    for nombre, p in configs:
        P = np.zeros((len(EPOCAS), len(idx)))
        for s in (SEMILLA_BUSQUEDA, SEMILLA_H2):
            for m in MESES_CRIBA:
                P[:, D["mes"][idx] == m] += obtener_fold(p, m, s) / 2
        num = np.abs(P[:, sol] - y[sol]).mean(axis=1)
        suav = pd.Series(num, index=EPOCAS).rolling(3, center=True, min_periods=1).mean()
        ep = int(suav.idxmin())
        res[nombre] = dict(p=p, ep=ep, err=np.abs(P[list(EPOCAS).index(ep)] - y))
    e0 = res["E1-30"]["err"]
    rng = np.random.default_rng(0)
    # máscaras sobre 'idx' (todas las horas); el bootstrap usa días
    D_fechas_idx = D["fechas"][idx]
    filas = []
    for nombre, r in res.items():
        d = r["err"] - e0
        fila = {"configuración": nombre, "params": json.dumps(r["p"]), "mejor época (2 semillas)": r["ep"],
                "MAE operación": r["err"][sol].mean(), "MAE E1-30": e0[sol].mean()}
        if nombre != "E1-30":
            sub = lambda msk: _ic_sub(d, msk, D_fechas_idx, rng)
            c, lo, hi = sub(sol); cc, lc, hc = sub(sol & central); ca, la, ha = sub(sol & amanecer)
            pct_am = 100 * ca / e0[sol & amanecer].mean()
            ok1 = (hi < 0) and (-c >= EFECTO_MINIMO); ok2 = (lo <= 0) and (lc <= 0); ok3 = pct_am <= LIMITE_AMANECER_PCT
            fila.update({"cambio operación": c, "IC95 operación": f"[{lo:+.2f}, {hi:+.2f}]",
                         "cambio central": cc, "IC95 central": f"[{lc:+.2f}, {hc:+.2f}]",
                         "cambio amanecer (%)": pct_am, "IC95 amanecer (W/m²)": f"[{la:+.2f}, {ha:+.2f}]",
                         "1 mejora ≥0,5": ok1, "2 sin empeorar op./central": ok2, "3 amanecer ≤4 %": ok3,
                         "veredicto": "GANA" if (ok1 and ok2 and ok3) else "no gana"})
        else:
            fila.update({"veredicto": "referencia"})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("configuración")


def _ic_sub(d, msk, fechas_idx, rng):
    dias = fechas_idx.normalize().to_numpy()[msk]; dd = d[msk]
    u, inv = np.unique(dias, return_inverse=True)
    g = [np.where(inv == i)[0] for i in range(len(u))]
    medias = [dd[np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])].mean() for _ in range(N_BOOT)]
    return float(dd.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


# ---------------------------- INFORME ----------------------------
def informe(study, H2):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df.to_csv(SALIDA / "H1_intentos.csv", index=False)
    comp = df[df["state"] == "COMPLETE"].sort_values("number")
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8))
    axs[0].scatter(comp["number"], comp["value"], color="tab:blue", label="intento completo")
    axs[0].plot(comp["number"], comp["value"].cummin(), color="tab:green", lw=2, label="mejor hasta el momento")
    ref = comp.loc[comp["number"] == 0, "value"]
    if len(ref):
        axs[0].axhline(ref.iloc[0], color="k", ls=":", label="E1-30 (intento 0)")
    axs[0].set_xlabel("intento"); axs[0].set_ylabel("MAE meses no vistos [W/m²]"); axs[0].legend(fontsize=8)
    axs[0].set_title("H1 · Búsqueda de Optuna\n(puntos por debajo de la línea negra = mejor que E1-30)")
    try:
        import optuna
        imp = optuna.importance.get_param_importances(study)
        axs[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="tab:purple")
        axs[1].set_title("¿Qué hiperparámetro influye más en el error?")
        pd.Series(imp).to_csv(SALIDA / "H1_importancia_hiperparametros.csv")
    except Exception as ex:
        axs[1].text(0.1, 0.5, f"Importancia no disponible:\n{ex}", fontsize=8); axs[1].axis("off")
    plt.tight_layout(); plt.savefig(SALIDA / "H1_busqueda.png", dpi=120); plt.close()

    H2.to_csv(SALIDA / "H2_resumen.csv")
    otras = H2[H2["veredicto"] != "referencia"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    col = ["tab:green" if v == "GANA" else "lightgray" for v in otras["veredicto"]]
    ax.bar(range(len(otras)), otras["cambio operación"], color=col)
    ax.axhline(0, color="k", lw=.8); ax.axhline(-EFECTO_MINIMO, color="tab:green", ls="--", lw=1, label="mejora mínima")
    ax.set_xticks(range(len(otras))); ax.set_xticklabels(otras.index, fontsize=8)
    ax.set_ylabel("cambio de MAE frente a E1-30 [W/m²]"); ax.legend(fontsize=8)
    ax.set_title("H2 · Las mejores de Optuna con 2 semillas frente a E1-30\n(verde = cumple los 3 criterios)")
    plt.tight_layout(); plt.savefig(SALIDA / "H2_contra_la_suerte.png", dpi=120); plt.close()

    gan = otras[otras["veredicto"] == "GANA"].sort_values("cambio operación")
    n_pod = (df["state"] == "PRUNED").sum(); n_ok = (df["state"] == "COMPLETE").sum()
    lineas = ["INFORME H1 + H2 · Optimización de hiperparámetros de E1-30", "=" * 70,
              f"H1: {n_ok} intentos completos y {n_pod} podados. Datos: solo train, 4 meses fuera, semilla {SEMILLA_BUSQUEDA}.",
              f"Mejor intento de H1: n.º {int(comp.loc[comp['value'].idxmin(), 'number'])} con MAE {comp['value'].min():.2f} W/m² "
              f"(E1-30 en H1: {ref.iloc[0]:.2f})." if len(ref) else "",
              "", "H2 (semillas 42 y 123, por parejas frente a E1-30):"]
    for nombre, r in otras.iterrows():
        lineas.append(f"  {nombre}: cambio {r['cambio operación']:+.2f} W/m² {r['IC95 operación']} · central {r['cambio central']:+.2f} "
                      f"· amanecer {r['cambio amanecer (%)']:+.1f} % → {r['veredicto']}  · {r['params']}")
    lineas.append("")
    if len(gan):
        g = gan.iloc[0]
        lineas.append(f"VEREDICTO: GANA {g.name} ({g['params']}), mejor época {int(g['mejor época (2 semillas)'])}. "
                      "Siguiente: H3 (confirmación en validación con 5 semillas).")
    else:
        lineas.append("VEREDICTO: ninguna combinación cumple los 3 criterios. E1-30 se mantiene como modelo final; "
                      "no se toca validación. Siguiente: D1.")
    texto = "\n".join(lineas)
    (SALIDA / "INFORME.txt").write_text(texto, encoding="utf-8")
    log("\n" + texto)


# ---------------------------- PRINCIPAL ----------------------------
def main():
    import optuna
    from optuna.trial import TrialState
    SALIDA.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    log("=" * 70); log("Inicio / reanudación")
    D = datos()
    log(f"Datos: {D['X'].shape} (esperado (5149, 24, 9)) · meses de la criba {MESES_CRIBA}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="E1_30_hiperparametros", storage=almacen(), direction="minimize",
                                load_if_exists=True)
    if len(study.trials) == 0:
        study.enqueue_trial(E1_30)
        log("Intento 0 = E1-30 (referencia) puesto en cola.")
    hechos = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials)
    log(f"H1: {hechos}/{N_INTENTOS} intentos terminados. Lanzando {N_TRABAJADORES} trabajadores…")
    if hechos < N_INTENTOS:
        try:
            with ProcessPoolExecutor(max_workers=N_TRABAJADORES) as ex:
                list(ex.map(trabajador, range(N_TRABAJADORES)))
        except Exception as error:
            log(f"Aviso: un trabajador se detuvo ({error}). Se sigue con lo ya guardado.")
    study = optuna.load_study(study_name="E1_30_hiperparametros", storage=almacen())
    comp = [t for t in study.trials if t.state == TrialState.COMPLETE]
    log(f"H1 terminado: {len(comp)} completos.")

    # H2: 3 mejores combinaciones distintas (+ E1-30)
    ref_key = clave(E1_30); vistas, top = {ref_key}, []
    for t in sorted(comp, key=lambda t: t.value):
        p = dict(t.params)
        if clave(p) not in vistas:
            vistas.add(clave(p)); top.append((f"intento {t.number}", p))
        if len(top) == TOP_H2:
            break
    configs = [("E1-30", E1_30)] + top
    tareas = [(p, m, s) for _, p in configs for s in (SEMILLA_BUSQUEDA, SEMILLA_H2) for m in MESES_CRIBA
              if not (CACHE / f"{clave(p)}_{m}_s{s}.npy").exists()]
    log(f"H2: {len(configs)} configuraciones × 2 semillas · {len(tareas)} entrenamientos pendientes.")
    if tareas:
        with ProcessPoolExecutor(max_workers=N_TRABAJADORES) as ex:
            for k, (c, m, s) in enumerate(ex.map(_tarea_h2, tareas), 1):
                log(f"   H2 {k}/{len(tareas)} hecho ({c} · {m} · semilla {s})")
    H2 = evaluar_h2(configs)
    informe(study, H2)
    log(f"FIN. Resultados en: {SALIDA}")


if __name__ == "__main__":
    main()