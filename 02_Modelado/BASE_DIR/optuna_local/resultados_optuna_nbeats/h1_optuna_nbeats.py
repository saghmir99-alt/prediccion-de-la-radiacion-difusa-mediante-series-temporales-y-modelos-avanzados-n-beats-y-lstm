# =====================================================================
# H1 + H2 (N-BEATS) · OPTIMIZACIÓN DE HIPERPARÁMETROS DE LA N-BEATS-1h CON OPTUNA (en el PC, CPU, en paralelo)
#
#   Uso (en Anaconda Prompt, con el entorno "tfm" activado y dentro de la carpeta optuna_local):
#       python h1_optuna_nbeats.py
#   Necesita en optuna_local\datos: entrada_una_hora.npz y tabla_train.csv.
#   Si se corta (reinicio, apagón…), se vuelve a lanzar el MISMO comando y sigue donde lo dejó.
#
#   Mismo esquema que la búsqueda de la LSTM (h1_optuna.py), con el modelo y la entrada de N6/N6b:
#   H1 · Búsqueda: solo TRAIN · meses fuera ene, mar, may, jul · purga 24 h · semilla 42 · hasta 40 épocas
#        medidas en CADA época (adaptación fijada antes: la N-BEATS-1h converge en 5–15 épocas y la rejilla de 5 en 5
#        dejó su mejor época en el borde en N6) · cada combinación en su mejor época (media móvil de 3) ·
#        30 intentos · TPE + MedianPruner. El intento 0 es la N-BEATS-1h actual (referencia).
#   H2 · Contra la suerte: las 3 mejores combinaciones + la referencia, repetidas con la semilla 123.
#        Criterio (fijado antes; el MISMO que la LSTM), con 2 semillas y por parejas frente a la referencia:
#          1) baja el MAE en OPERACIÓN ≥ 0,5 W/m² con IC95 < 0
#          2) no empeora de forma significativa en OPERACIÓN ni en la BANDA CENTRAL
#          3) no empeora el AMANECER más de un 4 %
#        Si alguna gana → H3 (validación, una sola vez, 5 semillas; script aparte). No habrá fase intermedia
#        (en la LSTM hubo un H2+ con el criterio relajado, declarado como desviación; aquí no se repite).
#   Validación NO se usa aquí.
#   Resultados: carpeta "resultados_optuna_nbeats" (tablas CSV, gráficas PNG, registro.txt e INFORME.txt).
# =====================================================================
import os, sys, json, time, hashlib, random, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ---------------------------- CONFIGURACIÓN ----------------------------
CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos"
SALIDA = CARPETA / "resultados_optuna_nbeats"
CACHE = SALIDA / "cache_pliegues"
BD = SALIDA / "estudio_optuna_nbeats.db"
LOG = SALIDA / "registro.txt"
ESTUDIO = "NBEATS_1h_hiperparametros"

N_INTENTOS = 30
N_TRABAJADORES = 4            # intentos a la vez (i5-12450H: 4 × 2 hilos)
HILOS_POR_TRABAJADOR = 2
MESES_CRIBA = ["2024-01", "2024-03", "2024-05", "2024-07"]
EPOCAS_MAX, CADA = 40, 1
SEMILLA_BUSQUEDA, SEMILLA_H2 = 42, 123
TOP_H2 = 3
EFECTO_MINIMO, LIMITE_AMANECER_PCT, N_BOOT = 0.5, 4.0, 2000
REFERENCIA = {"lr": 1e-3, "batch": 128, "hidden": 128, "n_bloques": 6, "wd": 1e-5, "dropout": 0.0}   # N-BEATS-1h de N6/N6b


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
    z = np.load(DATOS / "entrada_una_hora.npz")
    SEQ, ESC = z["seq_train"].astype(np.float32), z["esc_train"].astype(np.float32)
    y = z["y_train"].astype(np.float32)
    fechas = pd.DatetimeIndex(pd.to_datetime(z["fechas_train"]))
    assert fechas.max() < pd.Timestamp("2024-08-01"), "Solo train"
    T = pd.read_csv(DATOS / "tabla_train.csv", index_col=0, parse_dates=True)
    assert len(T) == len(y) and (T.index == fechas).all(), "tabla_train.csv no cuadra con entrada_una_hora.npz"
    assert np.allclose(T["real"].to_numpy(), y, atol=0.01), "El valor real no coincide"
    assert np.isfinite(SEQ).all() and np.isfinite(ESC).all(), "Hay huecos en las entradas"
    mes = fechas.to_period("M").astype(str).to_numpy()
    mes = np.where(mes == "2023-12", "2024-01", mes)
    _D.update(SEQ=SEQ, ESC=ESC, PI1=ESC[:, 2].copy(), y=y, fechas=fechas, T=T, mes=mes, sol=(T["regimen"] != "noche").to_numpy())
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
    """N-BEATS-1h (la de N6) con los hiperparámetros p; predicciones del mes no visto en cada época (épocas × horas)."""
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(HILOS_POR_TRABAJADOR)
    D = datos(); entrena, fuera = pliegue(m)
    SEQ, ESC, PI1, y = D["SEQ"], D["ESC"], D["PI1"], D["y"]
    NB, NC, NE = SEQ.shape[1], SEQ.shape[2], ESC.shape[1]
    r = y - PI1
    s_seq = StandardScaler().fit(SEQ[entrena].reshape(-1, NC)); s_esc = StandardScaler().fit(ESC[entrena])
    s_y = StandardScaler().fit(r[entrena].reshape(-1, 1))
    tq = lambda Z: torch.from_numpy(s_seq.transform(Z.reshape(-1, NC)).reshape(Z.shape).astype(np.float32))
    te = lambda Z: torch.from_numpy(s_esc.transform(Z).astype(np.float32))

    class Bloque(nn.Module):
        def __init__(self, n_in, h, nb, dropout):
            super().__init__()
            capas = []
            for k in range(3):
                capas += [nn.Linear(n_in if k == 0 else h, h), nn.ReLU()]
                if dropout > 0:
                    capas.append(nn.Dropout(dropout))           # como REG en N2: tras cada capa oculta
            self.fc = nn.Sequential(*capas)
            self.backcast = nn.Linear(h, nb); self.forecast = nn.Linear(h, 1)
        def forward(self, x):
            h = self.fc(x)
            return self.backcast(h), self.forecast(h)

    class Modelo(nn.Module):
        def __init__(self):
            super().__init__()
            n_in = NB * NC + NE
            self.blocks = nn.ModuleList([Bloque(n_in, p["hidden"], NB, p["dropout"]) for _ in range(p["n_bloques"])])
        def forward(self, x_seq, x_esc):
            residuo = x_seq[:, :, 0]
            cov = torch.cat([x_seq[:, :, 1:].permute(0, 2, 1).reshape(x_seq.size(0), -1), x_esc], dim=1)
            total = torch.zeros((x_seq.size(0), 1), dtype=x_seq.dtype)
            for b in self.blocks:
                back, fore = b(torch.cat([residuo, cov], dim=1))
                residuo = residuo - back
                total = total + fore
            return total.squeeze(-1)

    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True)
    modelo = Modelo()
    gen = torch.Generator(); gen.manual_seed(semilla)
    r_e = torch.from_numpy(s_y.transform(r[entrena].reshape(-1, 1)).ravel().astype(np.float32))
    loader = DataLoader(TensorDataset(tq(SEQ[entrena]), te(ESC[entrena]), r_e), batch_size=p["batch"], shuffle=True,
                        generator=gen, num_workers=0)
    opt = torch.optim.Adam(modelo.parameters(), lr=p["lr"], weight_decay=p["wd"])
    crit = nn.L1Loss(); Xs_f, Xe_f = tq(SEQ[fuera]), te(ESC[fuera]); preds = []
    for ep in range(1, EPOCAS_MAX + 1):
        modelo.train()
        for xs, xe, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(modelo(xs, xe), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            opt.step()
        if ep % CADA == 0:
            modelo.eval()
            with torch.no_grad():
                corr = s_y.inverse_transform(modelo(Xs_f, Xe_f).numpy().reshape(-1, 1)).ravel()
            preds.append(np.clip(PI1[fuera] + corr, 0, None))
    P = np.array(preds, dtype=np.float32)
    assert np.isfinite(P).all(), f"Predicciones no finitas con {p} (mes {m}, semilla {semilla})"
    return P


def obtener_fold(p, m, semilla):
    """Con caché en disco: si ya se entrenó (aunque fuera en un intento cortado), se reutiliza."""
    ruta = CACHE / f"{clave(p)}_{m}_s{semilla}.npy"
    if ruta.exists():
        try:
            P = np.load(ruta)
            if P.shape != (len(EPOCAS), int((datos()["mes"] == m).sum())) or not np.isfinite(P).all():
                raise ValueError("caché con forma o contenido inesperado")
            return P
        except Exception:
            ruta.unlink()
    t0 = time.time()
    P = entrenar_fold(p, m, semilla)
    tmp = ruta.with_suffix(".tmp.npy"); np.save(tmp, P); os.replace(tmp, ruta)
    if p == REFERENCIA and m == MESES_CRIBA[0] and semilla == SEMILLA_BUSQUEDA:
        dur = time.time() - t0
        log(f"Tiempo medido: {dur:.0f} s por pliegue de {EPOCAS_MAX} épocas con la referencia → H1 tardará del orden de "
            f"{dur * 4 * N_INTENTOS / N_TRABAJADORES / 60 * 0.8:.0f}–{dur * 4 * N_INTENTOS / N_TRABAJADORES / 60 * 2.5:.0f} min "
            f"+ H2 ≈ {dur * 16 / N_TRABAJADORES / 60:.0f}–{dur * 16 / N_TRABAJADORES / 60 * 3:.0f} min "
            f"(depende del tamaño de las redes que pruebe y de la poda).")
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
    # Sin "heartbeat": con intentos largos (hasta ~45 min) marcaba como caídos intentos que seguían vivos y provocaba
    # "Cannot tell a COMPLETE trial". La reanudación ya la garantizan la caché de pliegues y la limpieza al arrancar.
    from optuna.storages import RDBStorage
    return RDBStorage(f"sqlite:///{BD.as_posix()}", engine_kwargs={"connect_args": {"timeout": 120}})


def espacio(trial):
    return {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch": trial.suggest_categorical("batch", [64, 128, 256]),
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "n_bloques": trial.suggest_int("n_bloques", 2, 8),
            "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.2])}


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
    study = optuna.load_study(study_name=ESTUDIO, storage=almacen(),
                              sampler=optuna.samplers.TPESampler(seed=100 + i, n_startup_trials=8, multivariate=True,
                                                                 constant_liar=True),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # Un intento cada vez. catch: si una combinación diverge (NaN), el intento queda como FAIL y el trabajador sigue.
    # Si Optuna da un error de registro (p. ej. "Cannot tell…"), se anota y el trabajador sigue con otro intento.
    while True:
        hechos = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.get_trials(deepcopy=False))
        if hechos >= N_INTENTOS:
            return i
        try:
            study.optimize(objetivo, n_trials=1, catch=(AssertionError, RuntimeError, ValueError))
        except Exception as ex:
            log(f"trabajador {i}: aviso de Optuna ({type(ex).__name__}: {ex}); se continúa con otro intento")


# ---------------------------- H2 · CONTRA LA SUERTE ----------------------------
def _tarea_h2(args):
    p, m, s = args
    obtener_fold(p, m, s)
    return clave(p), m, s


def _ic_sub(d, msk, fechas_idx, rng):
    dias = fechas_idx.normalize().to_numpy()[msk]; dd = d[msk]
    u, inv = np.unique(dias, return_inverse=True)
    g = [np.where(inv == i)[0] for i in range(len(u))]
    medias = [dd[np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])].mean() for _ in range(N_BOOT)]
    return float(dd.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def evaluar_h2(configs):
    """configs: lista de (nombre, params). Criterio frente a la referencia, con 2 semillas."""
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
    e0 = res["referencia"]["err"]
    rng = np.random.default_rng(0)
    fechas_idx = D["fechas"][idx]
    filas = []
    for nombre, r in res.items():
        d = r["err"] - e0
        fila = {"configuración": nombre, "params": json.dumps(r["p"]), "mejor época (2 semillas)": r["ep"],
                "MAE operación": r["err"][sol].mean(), "MAE referencia": e0[sol].mean()}
        if nombre != "referencia":
            sub = lambda msk: _ic_sub(d, msk, fechas_idx, rng)
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


# ---------------------------- INFORME ----------------------------
def informe(study, H2):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df.to_csv(SALIDA / "H1_intentos.csv", index=False)
    comp = df[df["state"] == "COMPLETE"].sort_values("number")
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8))
    axs[0].scatter(comp["number"], comp["value"], color="tab:orange", label="intento completo")
    axs[0].plot(comp["number"], comp["value"].cummin(), color="tab:green", lw=2, label="mejor hasta el momento")
    n_ref = [t.number for t in study.trials if t.value is not None and clave(dict(t.params)) == clave(REFERENCIA)]
    ref = comp.loc[comp["number"].isin(n_ref), "value"]
    if len(ref):
        axs[0].axhline(ref.iloc[0], color="k", ls=":", label="N-BEATS-1h actual (intento 0)")
    axs[0].set_xlabel("intento"); axs[0].set_ylabel("MAE meses no vistos [W/m²]"); axs[0].legend(fontsize=8)
    axs[0].set_title("H1 · Búsqueda de Optuna (N-BEATS-1h)\n(por debajo de la línea negra = mejor que la referencia)")
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
    ax.set_ylabel("cambio de MAE frente a la referencia [W/m²]"); ax.legend(fontsize=8)
    ax.set_title("H2 · Las mejores de Optuna con 2 semillas frente a la N-BEATS-1h actual\n(verde = cumple los 3 criterios)")
    plt.tight_layout(); plt.savefig(SALIDA / "H2_contra_la_suerte.png", dpi=120); plt.close()

    gan = otras[otras["veredicto"] == "GANA"].sort_values("cambio operación")
    n_pod = (df["state"] == "PRUNED").sum(); n_ok = (df["state"] == "COMPLETE").sum()
    lineas = ["INFORME H1 + H2 · Optimización de hiperparámetros de la N-BEATS-1h", "=" * 70,
              f"H1: {n_ok} intentos completos y {n_pod} podados. Datos: solo train, 4 meses fuera, semilla {SEMILLA_BUSQUEDA}, "
              f"épocas 1–{EPOCAS_MAX} medidas en cada época.",
              (f"Mejor intento de H1: n.º {int(comp.loc[comp['value'].idxmin(), 'number'])} con MAE {comp['value'].min():.2f} W/m² "
               f"(referencia en H1: {ref.iloc[0]:.2f}).") if len(ref) else "",
              "", "H2 (semillas 42 y 123, por parejas frente a la referencia):"]
    for nombre, r in otras.iterrows():
        lineas.append(f"  {nombre}: cambio {r['cambio operación']:+.2f} W/m² {r['IC95 operación']} · central {r['cambio central']:+.2f} "
                      f"· amanecer {r['cambio amanecer (%)']:+.1f} % → {r['veredicto']}  · época {int(r['mejor época (2 semillas)'])} · {r['params']}")
    rr = H2.loc["referencia"]
    lineas.append(f"  referencia: MAE {rr['MAE operación']:.2f} W/m², mejor época {int(rr['mejor época (2 semillas)'])}")
    lineas.append("")
    if len(gan):
        g = gan.iloc[0]
        lineas.append(f"VEREDICTO: GANA {g.name} ({g['params']}), mejor época {int(g['mejor época (2 semillas)'])}. "
                      "Siguiente: H3 (confirmación en validación con 5 semillas, una sola vez).")
    else:
        lineas.append("VEREDICTO: ninguna combinación cumple los 3 criterios. La N-BEATS-1h actual (N6b) se mantiene como "
                      "N-BEATS final; no se toca validación. Siguiente: N7 (bandas).")
    texto = "\n".join(lineas)
    (SALIDA / "INFORME.txt").write_text(texto, encoding="utf-8")
    log("\n" + texto)


# ---------------------------- PRINCIPAL ----------------------------
def main():
    import optuna
    from optuna.trial import TrialState
    SALIDA.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    log("=" * 70); log("Inicio / reanudación (N-BEATS-1h)")
    D = datos()
    log(f"Datos: secuencia {D['SEQ'].shape} (esperado (5149, 12, 5)) · escalares {D['ESC'].shape} · meses de la criba {MESES_CRIBA}")
    assert D["SEQ"].shape == (5149, 12, 5) and D["ESC"].shape == (5149, 4), "Forma de las entradas inesperada"
    crib = np.isin(D["mes"], MESES_CRIBA) & D["sol"]
    log(f"Comprobación: MAE de la persistencia inteligente del último minuto en los 4 meses de la criba = "
        f"{np.abs(D['PI1'][crib] - D['y'][crib]).mean():.2f} W/m² (esperado 29,24, el de L2/N6)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=ESTUDIO, storage=almacen(), direction="minimize", load_if_exists=True)
    if len(study.trials) == 0:
        study.enqueue_trial(REFERENCIA)
        log("Intento 0 = N-BEATS-1h actual (referencia) puesto en cola.")
    # Limpieza tras un corte: los intentos que quedaron "en curso" se marcan como fallidos (su trabajo está en la caché)
    for t in study.trials:
        if t.state == TrialState.RUNNING:
            try:
                study.tell(t.number, state=TrialState.FAIL); log(f"Intento {t.number} había quedado a medias: marcado como fallido.")
            except Exception as ex:
                log(f"No se pudo cerrar el intento {t.number}: {ex}")
    study = optuna.load_study(study_name=ESTUDIO, storage=almacen())
    ref_hecha = any(t.state == TrialState.COMPLETE and clave(dict(t.params)) == clave(REFERENCIA) for t in study.trials)
    ref_en_cola = any(t.state == TrialState.WAITING for t in study.trials)
    if not ref_hecha and not ref_en_cola and len(study.trials) > 0:
        study.enqueue_trial(REFERENCIA); log("La referencia no había terminado: se vuelve a poner en cola (reutiliza la caché).")
    hechos = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials)
    log(f"H1: {hechos}/{N_INTENTOS} intentos terminados. Lanzando {N_TRABAJADORES} trabajadores…")
    if hechos < N_INTENTOS:
        with ProcessPoolExecutor(max_workers=N_TRABAJADORES) as ex:
            list(ex.map(trabajador, range(N_TRABAJADORES)))
    study = optuna.load_study(study_name=ESTUDIO, storage=almacen())
    comp = [t for t in study.trials if t.state == TrialState.COMPLETE]
    log(f"H1 terminado: {len(comp)} completos.")

    ref_key = clave(REFERENCIA); vistas, top = {ref_key}, []
    for t in sorted(comp, key=lambda t: t.value):
        p = dict(t.params)
        if clave(p) not in vistas:
            vistas.add(clave(p)); top.append((f"intento {t.number}", p))
        if len(top) == TOP_H2:
            break
    configs = [("referencia", REFERENCIA)] + top
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
