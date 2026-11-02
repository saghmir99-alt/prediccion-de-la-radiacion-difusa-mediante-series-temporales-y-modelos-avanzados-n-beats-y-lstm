"""EDA contemporáneo X(t)–DIF(t) en horas COMPLETAMENTE diurnas.

Entrada: ZIP de 4.2.9 con el checkpoint horario enriquecido.
Salida: ZIP con el checkpoint sin alterar, tablas y figuras estáticas.

En Colab: ejecutar el archivo con %run subseccion_4_2_10_relaciones_dia_completo.py
En terminal: python subseccion_4_2_10_relaciones_dia_completo.py --zip RUTA.zip

No define bandas de elevación, no modifica el objetivo, no interpola datos
y no calcula asociaciones predictivas con DIF(t+1).
"""

from __future__ import annotations

import argparse
import io
import json
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

OBJETIVO = "DIF__media_h"
FRACCION = "fraccion_minutos_dia"
ESTADISTICOS = ("media_h", "min_h", "max_h", "std_media_h")
FAMILIAS = ("SWD", "DIR", "DIF", "LWD", "SWU", "LWU")
VARIABLES = [f"{familia}__{estadistico}" for familia in FAMILIAS for estadistico in ESTADISTICOS]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
COVARIABLES = [variable for variable in VARIABLES if variable != OBJETIVO]
K_VECINOS = (3, 5, 10)
N_BINS = 20
SEMILLA_MI = 42


def solicitar_zip(ruta: str | None) -> Path:
    if ruta:
        elegido = Path(ruta).expanduser().resolve()
    else:
        try:
            from google.colab import files
        except ImportError as exc:
            raise RuntimeError("Indica --zip RUTA.zip al ejecutar fuera de Colab.") from exc
        print("Sube un único ZIP de la subsección 4.2.9 (checkpoint enriquecido).")
        subidos = files.upload()
        zips = [nombre for nombre in subidos if nombre.lower().endswith(".zip")]
        if len(subidos) != 1 or len(zips) != 1:
            raise ValueError("Debes seleccionar exactamente un archivo .zip.")
        elegido = Path(zips[0]).resolve()
    if not elegido.is_file() or elegido.suffix.lower() != ".zip":
        raise FileNotFoundError(f"ZIP de entrada inexistente o inválido: {elegido}")
    return elegido


def cargar_checkpoint(
    ruta_zip: Path, miembro_explicito: str | None = None
) -> tuple[pd.DataFrame, str, bytes]:
    """Elige solo un parquet horario con geometría y las 27 variables."""
    with zipfile.ZipFile(ruta_zip) as zf:
        parquets = [
            info
            for info in zf.infolist()
            if not info.is_dir() and info.filename.endswith(".parquet")
        ]
        if miembro_explicito:
            parquets = [info for info in parquets if info.filename == miembro_explicito]
            if not parquets:
                raise ValueError(f"No está ese miembro en el ZIP: {miembro_explicito}")
        if not parquets:
            raise ValueError("El ZIP no contiene ningún checkpoint .parquet.")

        candidatos = []
        detalles = []
        for info in parquets:
            datos = zf.read(info)
            try:
                tabla = pd.read_parquet(io.BytesIO(datos))
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError("Falta un lector de parquet; instala pyarrow en Colab.") from exc
            except Exception as exc:
                detalles.append(f"{info.filename}: lectura fallida ({type(exc).__name__})")
                continue
            requeridas = {"particion_observacion", FRACCION, *VARIABLES}
            faltantes = sorted(requeridas - set(tabla.columns))
            if faltantes:
                detalles.append(f"{info.filename}: faltan {faltantes[:5]}")
                continue
            candidatos.append((info.filename, tabla, datos))

    if len(candidatos) != 1:
        disponibles = [nombre for nombre, _, _ in candidatos]
        raise ValueError(
            "Se necesita exactamente un parquet horario completo con fracción diurna. "
            f"Candidatos: {disponibles}; otros miembros: {detalles}. "
            "Si hay más de uno, usa --checkpoint-miembro con su nombre exacto dentro del ZIP."
        )
    return candidatos[0][1], candidatos[0][0], candidatos[0][2]


def preparar_train(tabla: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = tabla.loc[tabla["particion_observacion"].eq("train")].copy()
    if len(train) != 5208:
        raise ValueError(
            f"Se esperaban 5.208 horas de train del checkpoint 4.2.9; hay {len(train)}."
        )
    fraccion = pd.to_numeric(train[FRACCION], errors="coerce")
    if fraccion.isna().any() or (~fraccion.between(0, 1)).any():
        raise ValueError(
            "fraccion_minutos_dia debe ser finita y estar entre 0 y 1 para todo train."
        )
    noche = int(fraccion.eq(0).sum())
    transiciones = int(fraccion.between(0, 1, inclusive="neither").sum())
    completas = int(fraccion.eq(1).sum())
    if (noche, transiciones, completas) != (2191, 429, 2588):
        raise ValueError(
            "Los tamaños de los grupos difieren del diagnóstico 4.2.9: "
            f"noche={noche}, transición={transiciones}, completo={completas}. "
            "Revisa el ZIP y la definición del intervalo [t,t+60 min)."
        )
    return train, {
        "n_train": len(train),
        "n_noche": noche,
        "n_transicion": transiciones,
        "n_dia_completo": completas,
    }


def pares_finitos(subtabla: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(subtabla[variable], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(subtabla[OBJETIVO], errors="coerce").to_numpy(dtype=float)
    mascara = np.isfinite(x) & np.isfinite(y)
    return x[mascara], y[mascara]


def calcular_metricas(x: np.ndarray, y: np.ndarray) -> dict:
    """Mantiene los ceros, los extremos y los pares finitos originales."""
    n = len(x)
    valores = {
        "n_pares": n,
        "pearson": np.nan,
        "spearman": np.nan,
        **{f"MI_k{k}_nats": np.nan for k in K_VECINOS},
    }
    valores["n_X_cero"] = int(np.count_nonzero(x == 0))
    valores["n_X_valor_mas_repetido"] = int(pd.Series(x).value_counts().iloc[0]) if n else 0
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return valores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        valores["pearson"] = float(pearsonr(x, y).statistic)
        valores["spearman"] = float(spearmanr(x, y).statistic)
    for k in K_VECINOS:
        if n > k + 1:
            valores[f"MI_k{k}_nats"] = float(
                mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    discrete_features=False,
                    n_neighbors=k,
                    random_state=SEMILLA_MI,
                )[0]
            )
    return valores


def resumir_intervalos_cuantilicos(x: np.ndarray, y: np.ndarray, variable: str) -> pd.DataFrame:
    """Hasta veinte grupos por cuantiles; no separa artificialmente empates."""
    columnas = [
        "variable",
        "bin_cuantil_X",
        "n",
        "X_min",
        "X_mediana",
        "X_max",
        "posicion_acumulada_X",
        "DIF_p10",
        "DIF_p25",
        "DIF_mediana",
        "DIF_p75",
        "DIF_p90",
    ]
    if len(x) == 0:
        return pd.DataFrame(columns=columnas)
    tabla = pd.DataFrame({"X": x, "DIF": y})
    if tabla["X"].nunique() < 2:
        tabla["bin"] = 0
    else:
        tabla["bin"] = pd.qcut(tabla["X"], q=N_BINS, labels=False, duplicates="drop")
        if tabla["bin"].isna().any():
            raise ValueError(f"qcut produjo grupos ausentes en {variable}; inspecciona X.")
    filas = []
    for bin_id, grupo in tabla.groupby("bin", sort=True):
        filas.append(
            {
                "variable": variable,
                "bin_cuantil_X": int(bin_id) + 1,
                "n": len(grupo),
                "X_min": float(grupo["X"].min()),
                "X_mediana": float(grupo["X"].median()),
                "X_max": float(grupo["X"].max()),
                "DIF_p10": float(grupo["DIF"].quantile(0.10)),
                "DIF_p25": float(grupo["DIF"].quantile(0.25)),
                "DIF_mediana": float(grupo["DIF"].median()),
                "DIF_p75": float(grupo["DIF"].quantile(0.75)),
                "DIF_p90": float(grupo["DIF"].quantile(0.90)),
            }
        )
    resumen = pd.DataFrame(filas)
    resumen["posicion_acumulada_X"] = (resumen["n"].cumsum() - resumen["n"] / 2) / len(x)
    return resumen[columnas]


def unidades_x(variable: str) -> str:
    if variable.startswith("T2__"):
        return "°C"
    if variable.startswith("RH__"):
        return "%"
    if variable.startswith("PoPoPoPo__"):
        return "hPa"
    return "W/m²"


def figura_individual(
    x: np.ndarray, y: np.ndarray, bins: pd.DataFrame, variable: str, numero: int, destino: Path
) -> None:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:
        raise RuntimeError("Para las figuras LOWESS necesitas statsmodels instalado.") from exc

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"{numero:02d}. {variable} ↔ DIF · día completo · n={len(x):,}", fontsize=14)

    if len(x) > 0:
        hexagonal = ax_a.hexbin(x, y, gridsize=42, mincnt=1, cmap="viridis", norm=LogNorm())
        fig.colorbar(hexagonal, ax=ax_a, label="N.º de pares por hexágono (escala log)")
        if len(x) >= 15 and np.unique(x).size >= 5:
            ajustada = lowess(y, x, frac=0.25, it=1, return_sorted=True)
            if len(ajustada) > 1 and np.isfinite(ajustada).all():
                ax_a.plot(
                    ajustada[:, 0],
                    ajustada[:, 1],
                    color="#d1492e",
                    lw=2,
                    label="LOWESS (descriptiva)",
                )
                ax_a.legend(loc="best")
    ax_a.set(
        title="A · Distribución conjunta",
        xlabel=f"{variable} [{unidades_x(variable)}]",
        ylabel="DIF media horaria [W/m²]",
    )

    if not bins.empty:
        p = bins["posicion_acumulada_X"].to_numpy(dtype=float)
        ax_b.fill_between(
            p, bins["DIF_p10"], bins["DIF_p90"], color="#a9cff6", alpha=0.30, label="P10–P90"
        )
        ax_b.fill_between(
            p, bins["DIF_p25"], bins["DIF_p75"], color="#70addb", alpha=0.45, label="P25–P75"
        )
        ax_b.plot(
            p, bins["DIF_mediana"], marker="o", ms=3.5, color="#153c78", label="Mediana de DIF"
        )
        ax_b.legend(loc="best")
    ax_b.set(
        xlim=(0, 1),
        title=f"B · DIF según X ({len(bins)} intervalos efectivos)",
        xlabel="Posición acumulada de X",
        ylabel="DIF media horaria [W/m²]",
    )
    for ax in (ax_a, ax_b):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def figura_resumen(tabla: pd.DataFrame, destino: Path) -> None:
    orden = list(reversed(COVARIABLES))
    resumen = tabla.set_index("variable").loc[orden]
    pos = np.arange(len(resumen))
    fig, axes = plt.subplots(1, 3, figsize=(20, 14), sharey=True)
    for ax, clave, titulo in zip(
        axes,
        ("pearson", "spearman", "MI_k5_nats"),
        ("Pearson", "Spearman", "Información mutua (k=5, nats)"),
    ):
        ax.barh(
            pos - 0.19,
            resumen[f"{clave}_train"],
            height=0.36,
            color="#888888",
            label="Train completo",
        )
        ax.barh(
            pos + 0.19, resumen[f"{clave}_dia"], height=0.36, color="#1967a5", label="Día completo"
        )
        ax.set(title=titulo, yticks=pos, yticklabels=orden)
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
    fig.suptitle(
        "Asociaciones contemporáneas: train completo vs horas completamente diurnas", fontsize=16
    )
    fig.text(
        0.5,
        0.01,
        "Las muestras tienen distinto tamaño. MI es estimada, no normalizada: "
        "las diferencias no cuantifican el efecto causal de la noche.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(destino, dpi=155)
    plt.close(fig)


def ejecutar(ruta_zip: Path, miembro: str | None = None) -> Path:
    tabla, origen_miembro, datos_originales = cargar_checkpoint(ruta_zip, miembro)
    train, conteos = preparar_train(tabla)
    dia = train.loc[pd.to_numeric(train[FRACCION], errors="raise").eq(1)].copy()
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Instala statsmodels antes de generar las figuras LOWESS.") from exc
    print(f"Checkpoint: {origen_miembro}")
    print(
        f"Train={conteos['n_train']:,}; noche={conteos['n_noche']:,}; "
        f"transiciones={conteos['n_transicion']:,}; día completo={conteos['n_dia_completo']:,}."
    )

    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directorio = Path.cwd() / f"EDA_multivariante_F4_S4_210_dia_completo_{sello}"
    if directorio.exists():
        raise FileExistsError(f"No se sobrescribirá una salida existente: {directorio}")
    tablas_dir = directorio / "tablas"
    figuras_dir = directorio / "figuras"
    checkpoint_dir = directorio / "checkpoint"
    for carpeta in (tablas_dir, figuras_dir, checkpoint_dir):
        carpeta.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_horario_entrada_intacto.parquet").write_bytes(datos_originales)
    (directorio / "entrada_4_2_9_original.zip").write_bytes(ruta_zip.read_bytes())

    filas = []
    cuantiles = []
    for indice, variable in enumerate(COVARIABLES, start=1):
        x_train, y_train = pares_finitos(train, variable)
        x_dia, y_dia = pares_finitos(dia, variable)
        m_train = calcular_metricas(x_train, y_train)
        m_dia = calcular_metricas(x_dia, y_dia)
        filas.append(
            {
                "variable": variable,
                **{f"{k}_train": v for k, v in m_train.items()},
                **{f"{k}_dia": v for k, v in m_dia.items()},
            }
        )
        bins = resumir_intervalos_cuantilicos(x_dia, y_dia, variable)
        cuantiles.append(bins)
        figura_individual(
            x_dia,
            y_dia,
            bins,
            variable,
            indice,
            figuras_dir / f"{indice:02d}_{variable}_dia_completo.png",
        )
        print(f"[{indice:02d}/26] {variable} · n finitos día={len(x_dia):,}")

    resumen = pd.DataFrame(filas)
    resumen.to_csv(tablas_dir / "01_metricas_train_vs_dia_completo.csv", index=False)
    pd.concat(cuantiles, ignore_index=True).to_csv(
        tablas_dir / "02_mediana_DIF_cuantiles_X_dia_completo.csv", index=False
    )
    pd.DataFrame([conteos]).to_csv(tablas_dir / "03_recuento_filtro_geometrico.csv", index=False)
    figura_resumen(resumen, figuras_dir / "00_resumen_train_vs_dia_completo.png")
    (directorio / "README.json").write_text(
        json.dumps(
            {
                "subseccion": "4.2.10",
                "fuente_zip": ruta_zip.name,
                "miembro_original": origen_miembro,
                "checkpoint_inalterado": True,
                "zip_previo_inalterado": True,
                "filtro": "train & fraccion_minutos_dia == 1",
                "objetivo": OBJETIVO,
                "numero_covariables": len(COVARIABLES),
                "pares": "finitos de X y DIF en cada comparación",
                "cuantiles_maximos": N_BINS,
                "mi_k": K_VECINOS,
                "mi_random_state": SEMILLA_MI,
                "LOWESS": {"frac": 0.25, "it": 1},
                "horas": conteos,
                "advertencia": "Asociaciones contemporáneas descriptivas; no predicen DIF(t+1).",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    salida_zip = directorio.with_suffix(".zip")
    with zipfile.ZipFile(salida_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archivo in sorted(directorio.rglob("*")):
            if archivo.is_file():
                zf.write(archivo, arcname=archivo.relative_to(directorio))
    print(f"ZIP de salida: {salida_zip}")
    return salida_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", help="ZIP de 4.2.9 con checkpoint horario y fracción diurna")
    parser.add_argument(
        "--checkpoint-miembro", help="Nombre exacto del parquet si hay más de un candidato"
    )
    args = parser.parse_args()
    resultado = ejecutar(solicitar_zip(args.zip), args.checkpoint_miembro)
    try:
        from google.colab import files
    except ImportError:
        pass
    else:
        files.download(str(resultado))


if __name__ == "__main__":
    main()
