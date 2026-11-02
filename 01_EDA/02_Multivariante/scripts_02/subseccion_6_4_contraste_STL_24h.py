"""6.4 Contraste original–STL. Entradas: ZIP 6.1 (24 retardos) y ZIP STL 4.2.4."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

deps = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pyarrow": "pyarrow",
}
missing = [v for k, v in deps.items() if importlib.util.find_spec(k) is None]
if missing:
    try:
        import google.colab
    except ImportError:
        raise RuntimeError("Instala: python -m pip install " + " ".join(missing)) from None
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.feature_selection import mutual_info_regression

Y = "DIF__media_h"
LAGS = list(range(1, 25))
DATA = "datos/pares_retardados_diurnos.parquet"


def member(z, suffix):
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError("No se identifica inequívocamente " + suffix)
    return names[0]


VARIABLES = [
    f"{f}__{s}"
    for f in ["SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]
    for s in ["media_h", "min_h", "max_h", "std_media_h"]
    if (f, s) != ("DIF", "media_h")
]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]

COMP = "componentes/01_componentes_STL_MSTL_train.parquet"
CONF = "configuracion/configuracion_4_2_4.json"
ALL = [Y] + VARIABLES


def kind(p):
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if any(n.endswith(COMP) for n in names) and any(n.endswith(CONF) for n in names):
                return "stl"
            if any(n.endswith(DATA) for n in names):
                m = json.loads(z.read(member(z, "README.json")))
                if str(m.get("version", "")).startswith("6.1") and m.get("retardos_h") == LAGS:
                    return "61"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        pass
    return None


def inputs(args):
    found = {}
    for k, p in [("61", args.zip_61), ("stl", args.zip_stl)]:
        if p:
            p = Path(p).expanduser().resolve()
            if kind(p) != k:
                raise ValueError(f"ZIP incorrecto para {k}: {p}")
            found[k] = p
    candidates = set()
    for folder in {Path.cwd(), Path("/content"), Path(__file__).resolve().parent}:
        if folder.is_dir():
            candidates.update(p.resolve() for p in folder.glob("*.zip"))
    for k in ["61", "stl"]:
        if k in found:
            continue
        options = sorted(p for p in candidates if kind(p) == k)
        if options:
            hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in options}
            if len(hashes) > 1:
                raise ValueError(
                    f"Varios ZIP {k} distintos. Indica --zip-61 y --zip-stl explícitamente."
                )
            found[k] = options[0]
    missing = set(["61", "stl"]) - set(found)
    if missing:
        try:
            from google.colab import files
        except ImportError:
            raise ValueError("Indica --zip-61 RUTA --zip-stl RUTA.") from None
        print(
            "Sube las entradas que faltan:",
            ", ".join("6.1 con 24 retardos" if k == "61" else "STL 4.2.4" for k in sorted(missing)),
        )
        uploaded = files.upload()
        for name in uploaded:
            p = Path(name).resolve()
            k = kind(p)
            if k not in missing or k in found:
                raise ValueError("Entrada duplicada o no requerida: " + name)
            found[k] = p
    if set(found) != set(["61", "stl"]):
        raise ValueError("Falta el ZIP 6.1 o el ZIP STL.")
    return found


def load(found):
    with zipfile.ZipFile(found["61"]) as z:
        train = pd.read_parquet(
            io.BytesIO(z.read(member(z, "checkpoint/checkpoint_train_horario.parquet")))
        )
        pairs = pd.read_parquet(io.BytesIO(z.read(member(z, DATA))))
    with zipfile.ZipFile(found["stl"]) as z:
        long = pd.read_parquet(io.BytesIO(z.read(member(z, COMP))))
        config = json.loads(z.read(member(z, CONF)))
    if config["contrato"]["periodo_stl_h"] != 24 or not config["contrato"]["robusto"]:
        raise ValueError("La configuración no corresponde al STL diario robusto de 4.2.4.")
    long["timestamp_utc"] = pd.to_datetime(long.timestamp_utc, utc=True)
    if long.duplicated(["timestamp_utc", "columna"]).any():
        raise ValueError("Duplicados en componentes.")
    if set(long.columna) != set(ALL):
        raise ValueError("Las componentes no contienen las 27 variables esperadas.")
    raw = long.pivot(index="timestamp_utc", columns="columna", values="valor_original").reindex(
        columns=ALL
    )
    res = long.pivot(index="timestamp_utc", columns="columna", values="stl_residuo").reindex(
        columns=ALL
    )
    flag = long.pivot(index="timestamp_utc", columns="columna", values="fue_interpolado").reindex(
        columns=ALL
    )
    if flag.isna().any().any() or not all(t == bool for t in flag.dtypes):
        raise ValueError("Marcas de interpolación inválidas.")
    if (
        not isinstance(train.index, pd.DatetimeIndex)
        or train.index.tz is None
        or train.index.has_duplicates
    ):
        raise ValueError("Índice de train inválido.")
    train = train.sort_index()
    train.index = train.index.tz_convert("UTC")
    if not train.index.equals(raw.index) or not train.index.equals(
        pd.date_range(train.index[0], train.index[-1], freq="h")
    ):
        raise ValueError("Train y STL deben compartir la rejilla horaria completa.")
    if not train.particion_observacion.eq("train").all():
        raise ValueError("Hay filas fuera de train.")
    if not np.allclose(
        raw.to_numpy(), train[ALL].to_numpy(dtype=float), rtol=0, atol=1e-9, equal_nan=True
    ):
        raise ValueError("Los originales del ZIP STL no coinciden con el checkpoint 6.1.")
    raw = raw.astype(float)
    res = res.astype(float)
    masked = flag | ~np.isfinite(raw)
    res = res.mask(masked)
    mask = pairs.muestra_principal_comun
    if mask.dtype != bool or mask.isna().any():
        raise ValueError("Máscara 6.1 inválida.")
    targets = pairs.index[mask].sort_values()
    if not train.loc[targets, "fraccion_minutos_dia"].eq(1).all():
        raise ValueError("Objetivos no diurnos.")
    good = np.isfinite(res[Y].reindex(targets).to_numpy())
    for h in LAGS:
        origins = targets - pd.Timedelta(hours=h)
        for frame in [raw, res]:
            good &= np.isfinite(frame.reindex(origins)[ALL].to_numpy()).all(axis=1)
    excluded = targets[~good]
    targets = targets[good]
    if len(targets) <= 10:
        raise ValueError("Muestra insuficiente.")
    audit = dict(
        n_objetivos_61=int(mask.sum()),
        n_objetivos_64=len(targets),
        n_excluidos_adicionales=len(excluded),
        excluidos_adicionales=list(map(str, excluded)),
        n_residuos_enmascarados=int(masked.sum().sum()),
        n_horas_train=len(train),
        originales_coinciden=True,
    )
    return raw, res, targets, config, audit


def measures(x, y):
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return dict(
            pearson=np.nan,
            spearman=np.nan,
            mi_k3_nats=np.nan,
            mi_k5_nats=np.nan,
            mi_k10_nats=np.nan,
        )
    out = dict(pearson=float(pearsonr(x, y).statistic), spearman=float(spearmanr(x, y).statistic))
    for k in [3, 5, 10]:
        out[f"mi_k{k}_nats"] = float(
            mutual_info_regression(
                x.reshape(-1, 1), y, discrete_features=False, n_neighbors=k, random_state=42
            )[0]
        )
    return out


def compare(raw, res, t):
    rows = []
    for i, v in enumerate(VARIABLES, 1):
        print(f"Contraste {i}/26: {v}", flush=True)
        for h in LAGS:
            row = dict(variable=v, retardo_h=h, n=len(t))
            for name, d in [("original", raw), ("residuo", res)]:
                vals = measures(
                    d[v].reindex(t - pd.Timedelta(hours=h)).to_numpy(), d[Y].reindex(t).to_numpy()
                )
                row.update({name + "_" + k: value for k, value in vals.items()})
            for metric in ["pearson", "spearman", "mi_k3_nats", "mi_k5_nats", "mi_k10_nats"]:
                row["delta_" + metric] = row["residuo_" + metric] - row["original_" + metric]
            for metric in ["pearson", "spearman"]:
                row["delta_abs_" + metric] = abs(row["residuo_" + metric]) - abs(
                    row["original_" + metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(res):
    acf = []
    months = []
    for v in ALL:
        x = res[v]
        scale = x.std(ddof=0)
        for h in LAGS:
            lag = x.shift(
                h
            )  # Sobre rejilla horaria completa, nunca sobre horas diurnas concatenadas.
            ok = x.notna() & lag.notna()
            acf.append(
                dict(
                    variable=v,
                    retardo_h=h,
                    n=int(ok.sum()),
                    autocorrelacion_pearson=x[ok].corr(lag[ok]),
                )
            )
        for month, g in x.groupby(x.index.strftime("%Y-%m")):
            n = g.count()
            start = pd.Timestamp(month + "-01", tz="UTC")
            months.append(
                dict(
                    variable=v,
                    mes=month,
                    n=int(n),
                    mes_parcial=len(g) < start.days_in_month * 24,
                    mediana=g.median(),
                    iqr=g.quantile(0.75) - g.quantile(0.25),
                    mediana_sobre_std_global=g.median() / scale if scale > 0 else np.nan,
                    iqr_sobre_std_global=(
                        (g.quantile(0.75) - g.quantile(0.25)) / scale if scale > 0 else np.nan
                    ),
                )
            )
    return pd.DataFrame(acf), pd.DataFrame(months)


def label(v):
    f, s = v.split("__")
    return (
        ("Presión" if f == "PoPoPoPo" else f)
        + " "
        + dict(media_h="media", min_h="mín", max_h="máx", std_media_h="std")[s]
    )


def panel(ax, data, variables, title, cmap, vmin, vmax):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#ccc")
    im = ax.imshow(data, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(variables)), [label(v) for v in variables], fontsize=8)
    ax.set_title(title, fontsize=11)
    return im


def plots(table, acf, months, folder):
    for num, metric in enumerate(["pearson", "spearman", "mi_k5_nats"], 1):
        arr = [
            table.pivot(index="variable", columns="retardo_h", values=prefix + metric)
            .reindex(index=VARIABLES, columns=LAGS)
            .to_numpy()
            for prefix in ["original_", "residuo_", "delta_"]
        ]
        mi = metric.startswith("mi")
        mx = max(0.01, float(np.nanmax(arr[:2]))) if mi else 1
        delta = max(0.01, float(np.nanmax(np.abs(arr[2]))))
        fig, axs = plt.subplots(1, 3, figsize=(21, 10), layout="constrained")
        for i, (a, data, title) in enumerate(
            zip(axs, arr, ["Original", "Residuo STL", "Residuo − original"])
        ):
            im = panel(
                a,
                data,
                VARIABLES,
                title,
                "RdBu_r" if i == 2 or not mi else "viridis",
                -delta if i == 2 else (0 if mi else -1),
                delta if i == 2 else mx,
            )
            a.set_xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
            a.set_xlabel("Retardo (horas)")
            fig.colorbar(im, ax=a, shrink=0.7, label="nats" if mi else "Coeficiente / diferencia")
        fig.suptitle(
            f"6.4 · {metric} · mismos {int(table.n.iloc[0])} objetivos diurnos\nDiferencia firmada: negativa no significa siempre atenuación",
            fontsize=14,
        )
        fig.savefig(folder / f"0{num}_contraste_{metric}.png", dpi=160)
        plt.close(fig)
    a = acf.pivot(index="variable", columns="retardo_h", values="autocorrelacion_pearson").reindex(
        index=ALL, columns=LAGS
    )
    fig, ax = plt.subplots(figsize=(13, 10), layout="constrained")
    im = panel(
        ax,
        a,
        ALL,
        "Autocorrelación residual · pares en la rejilla completa de train",
        "RdBu_r",
        -1,
        1,
    )
    ax.set_xticks(range(24), LAGS)
    ax.set_xlabel("Retardo (horas)")
    fig.colorbar(im, ax=ax, label="Pearson residual con su propio pasado")
    fig.savefig(folder / "04_autocorrelacion_residual.png", dpi=160)
    plt.close(fig)
    # Diciembre parcial se conserva en tabla, no se equipara visualmente a meses completos.
    full = months.loc[~months.mes_parcial]
    fig, axs = plt.subplots(1, 2, figsize=(14, 10), layout="constrained")
    for ax, col, title, cmap in zip(
        axs,
        ["mediana_sobre_std_global", "iqr_sobre_std_global"],
        ["Centro mensual / std global", "IQR mensual / std global"],
        ["RdBu_r", "viridis"],
    ):
        d = full.pivot(index="variable", columns="mes", values=col).reindex(index=ALL)
        lim = max(0.01, float(np.nanmax(np.abs(d.to_numpy()))))
        im = panel(ax, d, ALL, title, cmap, -lim if cmap == "RdBu_r" else 0, lim)
        ax.set_xticks(range(len(d.columns)), d.columns, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(
        "Diagnóstico mensual residual · horas completas de train\nCambios de centro y dispersión; no es una prueba de estacionariedad"
    )
    fig.savefig(folder / "05_diagnostico_mensual_residuos.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-61")
    parser.add_argument("--zip-stl")
    parser.add_argument("--salida", default=".")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    found = inputs(args)
    raw, res, t, config, audit = load(found)
    print("Validación:", audit, flush=True)
    table = compare(raw, res, t)
    acf, months = diagnostics(res)
    folder = Path(args.salida).resolve() / (
        "EDA_6_4_contraste_STL_24h_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    for s in ["tablas", "figuras", "datos"]:
        (folder / s).mkdir(parents=True, exist_ok=False)
    for data, name in [
        (table, "01_contraste_624_combinaciones"),
        (acf, "02_autocorrelacion_residual"),
        (months, "03_diagnostico_mensual"),
    ]:
        data.to_csv(folder / f"tablas/{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"timestamp_objetivo_utc": t}).to_csv(
        folder / "tablas/04_objetivos_comunes.csv", index=False
    )
    res.to_parquet(folder / "datos/residuos_STL_train_enmascarados.parquet")
    plots(table, acf, months, folder / "figuras")
    info = dict(
        version="6.4.1",
        **audit,
        variables=VARIABLES,
        retardos_h=LAGS,
        mi_k=[3, 5, 10],
        semilla=42,
        configuracion_STL_reutilizada=config,
        preblanqueamiento=False,
        versiones=dict(pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
        entradas={
            k: dict(nombre=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for k, p in found.items()
        },
    )
    (folder / "README.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "LEEME.md").write_text(
        """# 6.4 · Contraste tras ajuste STL

Se reutiliza STL robusto diario del ZIP 4.2.4, sin ajustar otra descomposición.
Los originales se contrastan con los guardados junto a las componentes. Se
enmascaran residuos de observaciones interpoladas/no finitas. La interpolación
diagnóstica pudo influir en componentes vecinas: el enmascaramiento no elimina
esa influencia del ajuste ya realizado. No se aplica MSTL ni preblanqueamiento.

## Comparación

Originales y residuos usan EXACTAMENTE los mismos objetivos diurnos en todos
los retardos 1–24 y variables. La respuesta ajustada es residuo_DIF(T) y la entrada
residuo_X(T−h). Se recalculan originales sobre esta intersección, no se restan
métricas procedentes de muestras diferentes. Las tablas incluyen todas las
624 combinaciones y MI k=3,5,10; las figuras muestran k=5 como referencia.

Las diferencias son residuo menos original. En Pearson/Spearman también se
exporta delta_abs: negativo indica menor magnitud de correlación. Una diferencia
firmada negativa puede significar una relación más negativa, no una atenuación.
MI puede aumentar o disminuir; no hay signo de dependencia ni escala máxima 1.
STL cambia la variable analizada; las diferencias no son porcentajes causales
de información eliminada. La sensibilidad a k no es un intervalo de confianza.

## Diagnósticos

La autocorrelación residual se calcula por pares finitos a 1–24 horas sobre
la rejilla horaria COMPLETA de train, incluida la noche. No se comprimen huecos.
No se añaden bandas clásicas ni se declara ruido blanco por coeficientes pequeños.
La tabla mensual incluye diciembre parcial; la figura muestra meses completos.
Mediana e IQR se dividen por la desviación global de cada residuo para comparar
cambios relativos dentro de cada variable. No es una prueba de estacionariedad
ni un estudio de la estabilidad mensual de las relaciones X–DIF.

## Límites

STL retira componentes estimadas, no controla completamente geometría solar,
estación o meteorología. Los residuos pueden conservar autocorrelación. La
descomposición es retrospectiva sobre todo train, incluye suavizado bilateral y
efectos de borde; estos residuos son diagnósticos, no entradas causales listas
para predecir fuera de muestra. No hay demostración de causalidad ni ganancia
predictiva incremental. El preblanqueamiento solo se considerará después para
una duda concreta. No se seleccionan variables ni retardos automáticamente.
""",
        encoding="utf-8",
    )
    out = folder.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder).as_posix())
    print("ZIP generado:", out)
    if not args.no_display:
        from IPython.display import Image, display

        for f in sorted((folder / "figuras").glob("*.png")):
            display(Image(filename=str(f)))
    if not args.no_download:
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(str(out))


if __name__ == "__main__":
    main()
