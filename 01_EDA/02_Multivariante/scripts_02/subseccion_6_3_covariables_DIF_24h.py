"""6.3 Relaciones retardadas X–DIF, 26 covariables y retardos 1–24. Entrada: ZIP 6.1 de 24 retardos."""

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


def valid(path):
    try:
        with zipfile.ZipFile(path) as z:
            member(z, DATA)
            meta = json.loads(z.read(member(z, "README.json")))
            return str(meta.get("version", "")).startswith("6.1") and meta.get("retardos_h") == LAGS
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def source(explicit):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not valid(p):
            raise ValueError("Selecciona el NUEVO ZIP de 6.1 con los 24 retardos.")
        return p
    candidates = set()
    for folder in {Path.cwd(), Path("/content"), Path(__file__).resolve().parent}:
        if folder.is_dir():
            candidates.update(p.resolve() for p in folder.glob("*.zip") if valid(p))
    if candidates:
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
        if len(hashes) > 1:
            raise ValueError(
                'Varios ZIP de 6.1 distintos: indica --zip "RUTA".\n'
                + "\n".join(map(str, candidates))
            )
        return sorted(candidates)[0]
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError("Indica --zip RUTA_AL_ZIP_DE_6_1.") from None
    print("Sube UN SOLO ZIP: resultados de 6.1 con los 24 retardos.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Selecciona un único ZIP de 6.1.")
    return source(next(iter(uploaded)))


VARIABLES = [
    f"{f}__{s}"
    for f in ["SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]
    for s in ["media_h", "min_h", "max_h", "std_media_h"]
    if (f, s) != ("DIF", "media_h")
]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]


def analyze(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read(member(z, "README.json")))
        df = pd.read_parquet(io.BytesIO(z.read(member(z, DATA))))
    if meta.get("retardos_h") != LAGS or set(meta.get("covariables", [])) != set(VARIABLES):
        raise ValueError("Se requiere el ZIP 6.1 con las 26 covariables y 24 retardos.")
    if (
        not isinstance(df.index, pd.DatetimeIndex)
        or df.index.tz is None
        or df.index.has_duplicates
        or df.index.hasnans
    ):
        raise ValueError("Índice temporal inválido.")
    mask = df["muestra_principal_comun"]
    if mask.dtype != bool or mask.isna().any():
        raise ValueError("Máscara común inválida.")
    sample = df.loc[mask].sort_index()
    if len(sample) != meta["n_muestra_principal_comun"] or len(sample) <= 10:
        raise ValueError("Muestra incompatible con 6.1 o insuficiente para k=10.")
    if not sample.fraccion_minutos_dia_objetivo.eq(1).all():
        raise ValueError("Se requieren objetivos completamente diurnos.")
    y = sample[Y].to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.unique(y).size < 2:
        raise ValueError("Objetivo no finito o constante.")
    for h in LAGS:
        origins = pd.DatetimeIndex(sample[f"timestamp_origen_{h:02d}h"])
        if not ((sample.index - origins) == pd.Timedelta(hours=h)).all():
            raise ValueError(f"Alineación incorrecta para retardo {h}.")
        if not sample[f"origen_train_observado_{h:02d}h"].all():
            raise ValueError("Hay orígenes ajenos a train.")
    rows = []
    for i, v in enumerate(VARIABLES, 1):
        print(f"Calculando {i}/26: {v}", flush=True)
        for h in LAGS:
            x = sample[f"{v}__lag_{h:02d}h"].to_numpy(dtype=float)
            if not np.isfinite(x).all():
                raise ValueError(f"{v}, lag {h}: valores no finitos en muestra común.")
            distinct = np.unique(x).size
            constant = distinct < 2
            row = dict(
                variable=v,
                familia=v.split("__")[0],
                retardo_h=h,
                n=len(x),
                pearson=np.nan if constant else float(pearsonr(x, y).statistic),
                spearman=np.nan if constant else float(spearmanr(x, y).statistic),
                n_ceros_origen=int((x == 0).sum()),
                n_distintos_origen=int(distinct),
                proporcion_modal_origen=float(pd.Series(x).value_counts().max() / len(x)),
                estado="constante: no estimado" if constant else "ok",
            )
            for k in [3, 5, 10]:
                row[f"mi_k{k}_nats"] = (
                    np.nan
                    if constant
                    else float(
                        mutual_info_regression(
                            x.reshape(-1, 1),
                            y,
                            discrete_features=False,
                            n_neighbors=k,
                            random_state=42,
                        )[0]
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows), meta, sample.index


def label(v):
    f, s = v.split("__")
    return (
        ("Presión" if f == "PoPoPoPo" else f)
        + " "
        + dict(media_h="media", min_h="mínimo", max_h="máximo", std_media_h="std")[s]
    )


def heatmap(table, metric, title, path):
    matrix = table.pivot(index="variable", columns="retardo_h", values=metric).reindex(
        index=VARIABLES, columns=LAGS
    )
    mi = metric.startswith("mi")
    fig, ax = plt.subplots(figsize=(14, 10), layout="constrained")
    cmap = plt.get_cmap("viridis" if mi else "RdBu_r").copy()
    cmap.set_bad("#dddddd")
    vmax = max(0.01, float(np.nanmax(matrix))) if mi else 1
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=0 if mi else -1, vmax=vmax)
    ax.set_xticks(np.arange(24), LAGS)
    ax.set_yticks(np.arange(26), [label(v) for v in VARIABLES])
    for i in range(1, 26):
        if VARIABLES[i].split("__")[0] != VARIABLES[i - 1].split("__")[0]:
            ax.axhline(i - 0.5, color="white", linewidth=1.6)
    ax.set_xlabel("Retardo h respecto a DIF(T), en horas · predictor X(T−h)")
    ax.set_title(
        f"6.3 · {title}\n26 covariables × 24 retardos · {int(table.n.iloc[0]):,} objetivos comunes".replace(
            ",", "."
        ),
        pad=14,
    )
    fig.colorbar(
        im,
        ax=ax,
        label="MI estimada (nats), k=5" if mi else "Coeficiente con signo (−1 a +1)",
        shrink=0.85,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip")
    parser.add_argument("--salida", default=".")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    path = source(args.zip)
    results, meta, index = analyze(path)
    folder = Path(args.salida).resolve() / (
        "EDA_6_3_covariables_DIF_24h_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    for sub in ["tablas", "figuras"]:
        (folder / sub).mkdir(parents=True, exist_ok=False)
    results.to_csv(
        folder / "tablas/01_metricas_26_covariables_24_retardos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for metric, title, name in [
        ("pearson", "Pearson · correlación cruzada descriptiva", "02_pearson"),
        ("spearman", "Spearman · asociación por rangos", "03_spearman"),
        ("mi_k5_nats", "Información mutua · dependencia marginal", "04_mi_k5"),
    ]:
        results.pivot(index="variable", columns="retardo_h", values=metric).reindex(
            index=VARIABLES, columns=LAGS
        ).to_csv(folder / f"tablas/{name}.csv", encoding="utf-8-sig")
        heatmap(results, metric, title, folder / f"figuras/{name}.png")
    pd.DataFrame({"timestamp_objetivo_utc": index}).to_csv(
        folder / "tablas/05_horas_objetivo_utilizadas.csv", index=False
    )
    doc = """# 6.3 · Relaciones retardadas entre covariables y DIF

Se comparan X(T−h) y DIF(T), para las 26 covariables y h=1,...,24 horas.
Todos los cálculos usan la misma máscara común de 6.1. Se incluyen las horas
disponibles de diciembre. Los objetivos son diurnos; sus orígenes pueden ser
nocturnos o de transición. No se recalculan retardos sobre filas diurnas comprimidas.

## Lectura

Los tres mapas tienen las mismas filas y columnas. Cada celda representa una
variable y un retardo. Pearson y Spearman utilizan una escala fija de −1 a +1;
MI utiliza su propia escala no negativa en nats. Un coeficiente negativo expresa
dirección inversa, no ausencia de asociación. MI no tiene signo ni máximo de 1.
El gris indica una medida no estimada para una variable constante.

La tabla 01 contiene las 624 combinaciones, sin seleccionar máximos ni aplicar
umbrales. Incluye MI para k=3,5,10, número de ceros, valores distintos y proporción
modal. k=5 es la referencia. Los otros k son sensibilidad, no intervalos de
confianza. La MI se estima como continua: las concentraciones de valores repetidos
requieren prudencia. Una MI positiva no demuestra por sí sola no linealidad y
un cero estimado no prueba independencia.

Las tablas 02–04 son versiones matriciales de las medidas principales. La tabla
05 identifica las horas objetivo. No se incluyen dispersiones: estos mapas
permiten localizar patrones, pero no establecer la forma exacta de las relaciones.

## Alcance

Pearson es una CCF descriptiva calculada como correlación de cada par alineado
en la muestra objetivo diurna. No es una CCF estacionaria convencional con
normalización global. No se aplican pruebas ni bandas automáticas de significación.
Las series originales conservan ciclos y tendencias; el ajuste temporal queda
para 6.4. Las diferencias entre retardos también coinciden con cambios en la
geometría solar de los orígenes, documentados en 6.1.

El horizonte previsto sigue siendo una hora: para T=t+1, h=1 usa X(t) y h=24 usa
X(t−23). Estos resultados no demuestran causalidad, capacidad predictiva fuera
de muestra ni información incremental respecto a DIF histórica. Los estadísticos
retardados de DIF pertenecen al historial del objetivo; su disponibilidad real
depende de que su intervalo haya terminado al emitir la predicción.

No se añade DIF media a las 26 covariables: su memoria ya está estudiada en 6.2.
Conservar el ZIP de 6.1 para los siguientes pasos; este resultado no duplica
su checkpoint. Las conclusiones deben apoyarse en patrones y tablas, sin
seleccionar automáticamente un retardo por el máximo entre 624 combinaciones.
"""
    (folder / "LEEME.md").write_text(doc, encoding="utf-8")
    info = dict(
        version="6.3.1",
        entrada=path.name,
        sha256_entrada=hashlib.sha256(path.read_bytes()).hexdigest(),
        n_objetivos=len(index),
        variables=VARIABLES,
        retardos_h=LAGS,
        n_combinaciones=len(results),
        mi_k_principal=5,
        mi_k_sensibilidad=[3, 10],
        semilla=42,
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        muestra="Máscara principal común de 6.1; sin filtrado adicional",
        ajuste_temporal=False,
        pruebas_significacion=False,
        referencia_mi="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html",
    )
    (folder / "README.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out = folder.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder).as_posix())
    print(
        f"Finalizado: {len(results)} combinaciones; {len(index)} objetivos comunes. Tres figuras."
    )
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
