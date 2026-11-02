"""6.2 Memoria temporal de DIF. Entrada única: ZIP generado en 6.1."""

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


def analyze(p):
    with zipfile.ZipFile(p) as z:
        df = pd.read_parquet(io.BytesIO(z.read(member(z, DATA))))
        meta = json.loads(z.read(member(z, "README.json")))
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or df.index.has_duplicates:
        raise ValueError("Índice temporal inválido.")
    if df.muestra_principal_comun.isna().any() or df.muestra_principal_comun.dtype != bool:
        raise ValueError("La máscara común debe ser booleana y no contener ausencias.")
    if meta.get("retardos_h") != LAGS:
        raise ValueError(
            "Este script requiere el NUEVO ZIP de 6.1 con retardos de 1 a 24 horas. Ejecuta primero subseccion_6_1_preparacion_retardos_24h.py."
        )
    sample = df.loc[df.muestra_principal_comun].sort_index()
    if len(sample) != meta["n_muestra_principal_comun"] or len(sample) <= 10:
        raise ValueError("Recuento de muestra incompatible con 6.1 o insuficiente para k=10.")
    if not sample.fraccion_minutos_dia_objetivo.eq(1).all():
        raise ValueError("Hay objetivos no completamente diurnos.")
    y = sample[Y].to_numpy(dtype=float)
    rows, quantiles = [], []
    for h in LAGS:
        col = f"{Y}__lag_{h:02d}h"
        origins = pd.DatetimeIndex(sample[f"timestamp_origen_{h:02d}h"])
        if not ((sample.index - origins) == pd.Timedelta(hours=h)).all():
            raise ValueError(f"Alineación incorrecta en h={h}.")
        if not sample[f"origen_train_observado_{h:02d}h"].all():
            raise ValueError("Orígenes ajenos a train.")
        x = sample[col].to_numpy(dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Valores no finitos dentro de la muestra común; revisar 6.1.")
        constant = np.unique(x).size < 2 or np.unique(y).size < 2
        row = dict(
            retardo_h=h,
            n=len(x),
            pearson=np.nan if constant else pearsonr(x, y).statistic,
            spearman=np.nan if constant else spearmanr(x, y).statistic,
            n_ceros_origen=int((x == 0).sum()),
            n_valores_distintos_origen=np.unique(x).size,
            proporcion_modal_origen=float(pd.Series(x).value_counts().max() / len(x)),
            estado="constante: asociación no estimada" if constant else "ok",
        )
        for k in [3, 5, 10]:
            row[f"mi_k{k}_nats"] = (
                np.nan
                if constant
                else float(
                    mutual_info_regression(
                        x.reshape(-1, 1), y, discrete_features=False, n_neighbors=k, random_state=42
                    )[0]
                )
            )
        regimes = sample[f"regimen_origen_{h:02d}h"].value_counts()
        for name in ["noche", "transicion", "dia_completo"]:
            row["n_origen_" + name] = int(regimes.get(name, 0))
        rows.append(row)
        bins = pd.qcut(pd.Series(x), q=8, duplicates="drop")
        temp = pd.DataFrame({"x": x, "y": y, "bin": bins})
        for interval, g in temp.groupby("bin", observed=True):
            quantiles.append(
                dict(
                    retardo_h=h,
                    intervalo=str(interval),
                    n=len(g),
                    x_min=g.x.min(),
                    x_max=g.x.max(),
                    x_mediana=g.x.median(),
                    dif_mediana=g.y.median(),
                    dif_p10=g.y.quantile(0.1),
                    dif_p90=g.y.quantile(0.9),
                )
            )
    return sample, pd.DataFrame(rows), pd.DataFrame(quantiles), meta


def plots(sample, metrics, q, folder):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    # Se representan todos los retardos horarios, de 1 a 24.
    for column, label in [("pearson", "Pearson"), ("spearman", "Spearman")]:
        ax[0].plot(LAGS, metrics[column], "o-", label=label)
    ax[0].axhline(0, color="gray", linewidth=0.8)
    ax[0].set_ylim(-1, 1)
    ax[0].set_ylabel("Coeficiente")
    ax[0].set_title("Asociación lineal y por rangos")
    for k in [3, 5, 10]:
        vals = metrics[f"mi_k{k}_nats"]
        ax[1].plot(
            LAGS,
            vals,
            "o-",
            label=f"k={k}" + (" · principal" if k == 5 else ""),
            linewidth=2.5 if k == 5 else 1,
            alpha=1 if k == 5 else 0.6,
        )
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel("Información mutua estimada (nats)")
    ax[1].set_title("Dependencia y sensibilidad a k")
    for a in ax:
        a.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 24])
        a.set_xticks(LAGS, minor=True)
        a.set_xlabel("Retardo respecto a DIF objetivo (horas)")
        a.grid(alpha=0.2)
        a.legend()
    fig.suptitle(
        f"6.2 · Memoria de DIF · {len(sample):,} objetivos diurnos comunes".replace(",", ".")
    )
    fig.savefig(folder / "01_memoria_por_retardo.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True, layout="constrained")
    cols = [Y] + [f"{Y}__lag_{h:02d}h" for h in [1, 6, 24]]
    lo = float(sample[cols].min().min())
    hi = float(sample[cols].max().max())
    pad = max((hi - lo) * 0.03, 1)
    lo -= pad
    hi += pad
    hexes = []
    for a, h in zip(axes, [1, 6, 24]):
        x = sample[f"{Y}__lag_{h:02d}h"]
        y = sample[Y]
        hb = a.hexbin(
            x, y, gridsize=35, extent=(lo, hi, lo, hi), mincnt=1, cmap="Blues", norm=LogNorm()
        )
        hexes.append(hb)
        sub = q.loc[q.retardo_h.eq(h)]
        a.plot(
            sub.x_mediana, sub.dif_mediana, "o-", color="#d46720", label="Medianas por cuantiles"
        )
        a.fill_between(sub.x_mediana, sub.dif_p10, sub.dif_p90, color="#d46720", alpha=0.13)
        a.plot(
            [lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="Igualdad DIF(T)=DIF(T−h)"
        )
        a.set_title(f"Retardo {h} h")
        a.set_xlabel(f"DIF(T−{h}) · W/m²")
        a.set_xlim(lo, hi)
        a.set_ylim(lo, hi)
        a.set_aspect("equal")
    vmax = max(float(h.get_array().max()) for h in hexes)
    for h in hexes:
        h.set_norm(LogNorm(vmin=1, vmax=max(2, vmax)))
    fig.colorbar(
        hexes[0], ax=axes, label="Observaciones por hexágono · escala logarítmica", shrink=0.7
    )
    axes[0].set_ylabel("DIF(T) · W/m²")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Forma y dispersión · mismos ejes y escala de color\nSombreado: P10–P90 de observaciones; no intervalo de confianza"
    )
    fig.savefig(folder / "02_forma_retardos_1_6_24.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip")
    parser.add_argument("--salida", default=".")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    p = source(args.zip)
    sample, m, q, meta = analyze(p)
    folder = Path(args.salida).resolve() / (
        "EDA_6_2_memoria_DIF_24h_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    for sub in ["tablas", "figuras", "datos"]:
        (folder / sub).mkdir(parents=True, exist_ok=False)
    m.to_csv(folder / "tablas/01_metricas_memoria_DIF.csv", index=False, encoding="utf-8-sig")
    q.to_csv(folder / "tablas/02_cuantiles_DIF_por_retardo.csv", index=False, encoding="utf-8-sig")
    sample[[Y] + [f"{Y}__lag_{h:02d}h" for h in LAGS]].to_parquet(
        folder / "datos/pares_DIF_analizados.parquet"
    )
    plots(sample, m, q, folder / "figuras")
    readme = dict(
        version="6.2.2",
        entrada=p.name,
        sha256_entrada=hashlib.sha256(p.read_bytes()).hexdigest(),
        n=len(sample),
        retardos_h=LAGS,
        semilla=42,
        mi_k_principal=5,
        mi_k_sensibilidad=[3, 10],
        sklearn_version=sklearn.__version__,
        muestra="muestra_principal_comun de 6.1, incluidos días de diciembre",
        convencion="DIF(T) frente a DIF(T-h); objetivo diurno, origen sin filtro solar adicional",
        ajuste_temporal=False,
        inferencia=False,
        advertencia="Dependencia descriptiva bruta condicionada al objetivo diurno. No es ACF estacionaria convencional, "
        "ni prueba de habilidad predictiva incremental. Ciclo solar, tendencias y empates permanecen.",
        entrada_6_3="Conservar ZIP de 6.1; este ZIP 6.2 solo contiene pares DIF.",
        referencia_mi="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html",
    )
    (folder / "README.json").write_text(
        json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "LEEME.md").write_text(
        """# 6.2 · Memoria temporal de DIF

Leer primero la tabla 01 y figura 01: mismo conjunto de objetivos en todos los retardos.
Pearson y Spearman indican dirección e intensidad lineal/monótona. MI no tiene signo
ni está limitada a 1; k=5 es la referencia y k=3,10 comprueban sensibilidad del estimador.
La sensibilidad a k no es un intervalo de confianza. MI positiva no prueba por sí sola
no linealidad; cero estimado tampoco prueba independencia. Los empates y ceros del
origen, registrados por retardo, aconsejan cautela con la estimación continua de MI.

Figura 02: retardos 1,6,24 elegidos previamente para representar última hora,
extremo de historia reciente y día anterior. Mismos ejes y colores; no se recortan
observaciones. Las medianas se agrupan en hasta ocho cuantiles de X, reducidos si hay
empates. P10–P90 mide dispersión, no incertidumbre de las medianas. La diagonal es
igualdad de valores, no un ajuste. Consultar tabla 02 para recuentos y extremos.

Los retardos se calculan en horas reales y las horas objetivo son diurnas. La
proporción de orígenes nocturnos cambia con h; no se controla aquí la geometría solar.
No se concluye causalidad, memoria meteorológica pura, optimalidad de retardos,
significación, estabilidad mensual ni mejora predictiva fuera de muestra.
El análisis temporalmente ajustado corresponde a 6.4. No hay bandas clásicas ±1.96/√N.
Para 6.3 se utilizará de nuevo el ZIP de 6.1, que contiene las 26 covariables.
""",
        encoding="utf-8",
    )
    output = folder.with_suffix(".zip")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder).as_posix())
    print(
        m[["retardo_h", "n", "pearson", "spearman", "mi_k5_nats"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print("ZIP generado:", output)
    if not args.no_display:
        from IPython.display import display, Image

        for f in sorted((folder / "figuras").glob("*.png")):
            display(Image(filename=str(f)))
    if not args.no_download:
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(str(output))


if __name__ == "__main__":
    main()
