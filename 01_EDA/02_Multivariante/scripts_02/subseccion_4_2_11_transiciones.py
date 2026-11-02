"""EDA X(t)–DIF(t) en horas de transición solar; contraste con día completo 0–10°.

En Colab: %run subseccion_4_2_11_transiciones.py
La celda solicita el ZIP de salida de 4.2.10 y descarga un nuevo ZIP recargable.
En terminal: python subseccion_4_2_11_transiciones.py --zip ruta/4_2_10.zip
Las métricas de train y día completo se recuperan de 4.2.10, sin recalcularlas.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import warnings

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
ELEVACION = "elevacion_efectiva_h"
ESTADISTICOS = ("media_h", "min_h", "max_h", "std_media_h")
FAMILIAS = ("SWD", "DIR", "DIF", "LWD", "SWU", "LWU")
VARIABLES = [f"{familia}__{estadistico}" for familia in FAMILIAS for estadistico in ESTADISTICOS]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
COVARIABLES = [variable for variable in VARIABLES if variable != OBJETIVO]
K_VECINOS = (3, 5, 10)
SEMILLA_MI = 42
N_BINS = 8
FIGURAS_PRIORITARIAS = (
    "SWD__media_h",
    "SWD__max_h",
    "SWU__media_h",
    "DIR__media_h",
    "DIR__min_h",
    "DIR__std_media_h",
    "DIF__min_h",
    "LWU__std_media_h",
    "RH__media_h",
    "T2__media_h",
)


def solicitar_zip(ruta: str | None) -> Path:
    if ruta:
        elegido = Path(ruta).expanduser().resolve()
    else:
        try:
            from google.colab import files
        except ImportError as exc:
            raise RuntimeError("Fuera de Colab debes pasar --zip RUTA.zip") from exc
        print("Sube exactamente el ZIP generado por la subsección 4.2.10.")
        subidos = files.upload()
        zips = [nombre for nombre in subidos if nombre.lower().endswith(".zip")]
        if len(subidos) != 1 or len(zips) != 1:
            raise ValueError("Selecciona un solo archivo .zip de la subsección 4.2.10.")
        elegido = Path(zips[0]).resolve()
    if not elegido.is_file() or elegido.suffix.lower() != ".zip":
        raise FileNotFoundError(f"ZIP inexistente o no válido: {elegido}")
    return elegido


def cargar_4_2_10(
    ruta: Path, miembro_explicito: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, bytes, str]:
    with zipfile.ZipFile(ruta) as zf:
        miembros_parquet = [
            nombre
            for nombre in zf.namelist()
            if nombre.endswith("checkpoint_horario_entrada_intacto.parquet")
        ]
        if miembro_explicito:
            miembros_parquet = [
                nombre for nombre in miembros_parquet if nombre == miembro_explicito
            ]
        miembros_metricas = [
            nombre
            for nombre in zf.namelist()
            if nombre.endswith("01_metricas_train_vs_dia_completo.csv")
        ]
        if len(miembros_parquet) != 1 or len(miembros_metricas) != 1:
            raise ValueError(
                "El ZIP debe contener exactamente un checkpoint horario inalterado y "
                "la tabla 01_metricas_train_vs_dia_completo.csv de 4.2.10. "
                f"Checkpoints={miembros_parquet}; tablas={miembros_metricas}."
            )
        miembro = miembros_parquet[0]
        datos = zf.read(miembro)
        try:
            tabla = pd.read_parquet(io.BytesIO(datos))
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("En Colab instala pyarrow para leer el checkpoint parquet.") from exc
        anterior = pd.read_csv(io.BytesIO(zf.read(miembros_metricas[0])))
    requeridas = {"particion_observacion", FRACCION, ELEVACION, *VARIABLES}
    faltantes = sorted(requeridas - set(tabla.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en el checkpoint de 4.2.10: {faltantes}")
    columnas_anteriores = {
        "variable",
        "n_pares_train",
        "n_pares_dia",
        "pearson_train",
        "pearson_dia",
        "spearman_train",
        "spearman_dia",
        "MI_k3_nats_train",
        "MI_k3_nats_dia",
        "MI_k5_nats_train",
        "MI_k5_nats_dia",
        "MI_k10_nats_train",
        "MI_k10_nats_dia",
    }
    if columnas_anteriores - set(anterior.columns):
        raise ValueError("La tabla de métricas previas no tiene las columnas requeridas de 4.2.10.")
    if (
        len(anterior) != 26
        or anterior["variable"].duplicated().any()
        or set(anterior["variable"]) != set(COVARIABLES)
    ):
        raise ValueError("La tabla previa debe contener exactamente las mismas 26 covariables.")
    return tabla, anterior, datos, miembro


def preparar_grupos(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = tabla.loc[tabla["particion_observacion"].eq("train")].copy()
    if len(train) != 5208:
        raise ValueError(
            f"El entrenamiento de 4.2.9/4.2.10 debía tener 5.208 horas; hay {len(train)}."
        )
    fraccion = pd.to_numeric(train[FRACCION], errors="coerce")
    elevacion = pd.to_numeric(train[ELEVACION], errors="coerce")
    if fraccion.isna().any() or (~fraccion.between(0, 1)).any():
        raise ValueError("La fracción de minutos diurnos debe ser finita y estar entre 0 y 1.")
    if (fraccion.gt(0) & ~np.isfinite(elevacion)).any():
        raise ValueError("Falta la elevación efectiva de alguna hora con Sol.")
    noche = fraccion.eq(0)
    transicion = fraccion.gt(0) & fraccion.lt(1)
    dia_completo = fraccion.eq(1)
    dia_bajo = dia_completo & elevacion.gt(0) & elevacion.le(10)
    recuentos = {
        "n_train": len(train),
        "n_noche": int(noche.sum()),
        "n_transicion": int(transicion.sum()),
        "n_dia_completo": int(dia_completo.sum()),
        "n_dia_completo_0_10": int(dia_bajo.sum()),
        "n_transicion_fuera_0_10": int(
            (transicion & ~elevacion.between(0, 10, inclusive="right")).sum()
        ),
    }
    if (
        recuentos["n_noche"],
        recuentos["n_transicion"],
        recuentos["n_dia_completo"],
        recuentos["n_dia_completo_0_10"],
        recuentos["n_transicion_fuera_0_10"],
    ) != (2191, 429, 2588, 323, 0):
        raise ValueError(
            "El checkpoint no reproduce los recuentos documentados de 4.2.9. "
            f"Recuentos observados: {recuentos}. Revisar geometría y frontera de 10°."
        )
    return train, train.loc[transicion].copy(), train.loc[dia_bajo].copy(), recuentos


def validar_metricas_previas(anterior: pd.DataFrame) -> None:
    for fila in anterior.itertuples(index=False):
        es_lwd = fila.variable.startswith("LWD__")
        esperado_train = 5207 if es_lwd else 5208
        esperado_dia = 2587 if es_lwd else 2588
        if int(fila.n_pares_train) != esperado_train or int(fila.n_pares_dia) != esperado_dia:
            raise ValueError(
                f"Los pares anteriores de {fila.variable} no son los del ZIP de 4.2.10."
            )


def pares_finitos(grupo: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(grupo[variable], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(grupo[OBJETIVO], errors="coerce").to_numpy(dtype=float)
    valido = np.isfinite(x) & np.isfinite(y)
    return x[valido], y[valido]


def calcular_metricas(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    resultado = {
        "n_pares": n,
        "pearson": np.nan,
        "spearman": np.nan,
        **{f"MI_k{k}_nats": np.nan for k in K_VECINOS},
        "n_X_cero": int(np.count_nonzero(x == 0)),
        "n_X_valor_mas_repetido": int(pd.Series(x).value_counts().iloc[0]) if n else 0,
    }
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return resultado
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resultado["pearson"] = float(pearsonr(x, y).statistic)
        resultado["spearman"] = float(spearmanr(x, y).statistic)
    for k in K_VECINOS:
        if n > k + 1:
            resultado[f"MI_k{k}_nats"] = float(
                mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    discrete_features=False,
                    n_neighbors=k,
                    random_state=SEMILLA_MI,
                )[0]
            )
    return resultado


def cuantiles_por_x(x: np.ndarray, y: np.ndarray, variable: str, grupo: str) -> pd.DataFrame:
    columnas = [
        "variable",
        "grupo",
        "intervalo_efectivo",
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
    pares = pd.DataFrame({"X": x, "DIF": y})
    if pares["X"].nunique() < 2:
        pares["intervalo"] = 0
    else:
        pares["intervalo"] = pd.qcut(pares["X"], q=N_BINS, labels=False, duplicates="drop")
        if pares["intervalo"].isna().any():
            raise ValueError(f"qcut produjo intervalos inválidos: {variable}, {grupo}")
    filas = []
    for i, bloque in pares.groupby("intervalo", sort=True):
        filas.append(
            {
                "variable": variable,
                "grupo": grupo,
                "intervalo_efectivo": int(i) + 1,
                "n": len(bloque),
                "X_min": float(bloque["X"].min()),
                "X_mediana": float(bloque["X"].median()),
                "X_max": float(bloque["X"].max()),
                "DIF_p10": float(bloque["DIF"].quantile(0.10)),
                "DIF_p25": float(bloque["DIF"].quantile(0.25)),
                "DIF_mediana": float(bloque["DIF"].median()),
                "DIF_p75": float(bloque["DIF"].quantile(0.75)),
                "DIF_p90": float(bloque["DIF"].quantile(0.90)),
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


def graficar_hexbin(fig, ax, x: np.ndarray, y: np.ndarray, variable: str, titulo: str) -> None:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if len(x) > 0:
        hb = ax.hexbin(x, y, gridsize=29, mincnt=1, cmap="viridis", norm=LogNorm())
        fig.colorbar(hb, ax=ax, label="Pares por hexágono (log)")
        if len(x) >= 15 and np.unique(x).size >= 5:
            curva = lowess(y, x, frac=0.25, it=1, return_sorted=True)
            if len(curva) > 1 and np.isfinite(curva).all():
                ax.plot(
                    curva[:, 0], curva[:, 1], color="#d1492e", lw=2, label="LOWESS (orientativa)"
                )
                ax.legend(loc="best")
    ax.set(
        title=titulo,
        xlabel=f"{variable} [{unidades_x(variable)}]",
        ylabel="DIF media horaria [W/m²]",
    )
    ax.axhline(0, color="grey", lw=0.7)
    ax.grid(alpha=0.20)


def graficar_cuantiles(ax, tabla: pd.DataFrame, titulo: str) -> None:
    if not tabla.empty:
        p = tabla["posicion_acumulada_X"].to_numpy(dtype=float)
        ax.fill_between(
            p, tabla["DIF_p10"], tabla["DIF_p90"], color="#a9cff6", alpha=0.32, label="P10–P90"
        )
        ax.fill_between(
            p, tabla["DIF_p25"], tabla["DIF_p75"], color="#70addb", alpha=0.52, label="P25–P75"
        )
        ax.plot(p, tabla["DIF_mediana"], color="#153c78", marker="o", ms=5, label="Mediana de DIF")
        ax.legend(loc="best")
    ax.set(
        title=f"{titulo} · {len(tabla)} intervalos efectivos",
        xlim=(0, 1),
        xlabel="Posición acumulada de X (en este grupo)",
        ylabel="DIF media horaria [W/m²]",
    )
    ax.axhline(0, color="grey", lw=0.7)
    ax.grid(alpha=0.25)


def figura_individual(
    variable: str,
    x_trans: np.ndarray,
    y_trans: np.ndarray,
    x_dia: np.ndarray,
    y_dia: np.ndarray,
    bins_trans: pd.DataFrame,
    bins_dia: pd.DataFrame,
    destino: Path,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle(f"{variable} ↔ DIF · transición vs día completo 0–10°", fontsize=15)
    graficar_hexbin(
        fig, axs[0, 0], x_trans, y_trans, variable, f"A · Transición · n={len(x_trans):,}"
    )
    graficar_cuantiles(axs[0, 1], bins_trans, "B · Transición")
    graficar_hexbin(
        fig, axs[1, 0], x_dia, y_dia, variable, f"C · Día completo 0–10° · n={len(x_dia):,}"
    )
    graficar_cuantiles(axs[1, 1], bins_dia, "D · Día completo 0–10°")
    for datos, ejes in (
        ((x_trans, x_dia), (axs[0, 0], axs[1, 0])),
        ((y_trans, y_dia), (axs[0, 0], axs[1, 0])),
    ):
        valores = np.concatenate(datos)
        if len(valores):
            bajo, alto = float(np.min(valores)), float(np.max(valores))
            margen = max((alto - bajo) * 0.04, 0.5)
            for ax in ejes:
                if datos[0] is x_trans:
                    ax.set_xlim(bajo - margen, alto + margen)
                else:
                    ax.set_ylim(bajo - margen, alto + margen)
    modas = [
        pd.Series(v).value_counts().iloc[0] / len(v) if len(v) else 0 for v in (x_trans, x_dia)
    ]
    if max(modas) >= 0.25:
        fig.text(
            0.5,
            0.02,
            "X contiene muchos empates: observar tamaños reales de los "
            "intervalos y no atribuir forma exacta a LOWESS.",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(destino, dpi=145)
    plt.close(fig)


def figura_resumen(tabla: pd.DataFrame, destino: Path) -> None:
    orden = list(reversed(COVARIABLES))
    matriz = tabla.set_index("variable").loc[orden]
    posiciones = np.arange(len(orden))
    fig, axs = plt.subplots(1, 3, figsize=(20, 14), sharey=True)
    for ax, columna, titulo in zip(
        axs,
        ("pearson", "spearman", "MI_k5_nats"),
        ("Pearson", "Spearman", "Información mutua, k=5 (nats)"),
    ):
        ax.barh(
            posiciones - 0.19,
            matriz[f"{columna}_transicion"],
            height=0.35,
            color="#8b60a7",
            label="Transición",
        )
        ax.barh(
            posiciones + 0.19,
            matriz[f"{columna}_dia_0_10"],
            height=0.35,
            color="#1976ad",
            label="Día completo 0–10°",
        )
        ax.set(title=titulo, yticks=posiciones, yticklabels=orden)
        ax.axvline(0, color="black", lw=0.7)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
    fig.suptitle("Asociaciones contemporáneas por tipo de hora en la banda 0–10°", fontsize=16)
    fig.text(
        0.5,
        0.01,
        "429 vs 323 horas antes de máscaras finitas. MI depende también "
        "del tamaño y composición de cada muestra.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(destino, dpi=155)
    plt.close(fig)


def recuento_por_mes(train: pd.DataFrame) -> pd.DataFrame:
    if isinstance(train.index, pd.DatetimeIndex):
        fechas = train.index
    elif "timestamp_utc" in train.columns:
        fechas = pd.to_datetime(train["timestamp_utc"], utc=True, errors="coerce")
    else:
        raise ValueError(
            "No se encontró índice temporal ni timestamp_utc para recuentos mensuales."
        )
    if pd.isna(fechas).any():
        raise ValueError("Hay marcas horarias inválidas en el checkpoint.")
    f = pd.to_numeric(train[FRACCION], errors="raise").to_numpy(dtype=float)
    a = pd.to_numeric(train[ELEVACION], errors="coerce").to_numpy(dtype=float)
    grupo = np.where(
        (f > 0) & (f < 1),
        "transicion",
        np.where((f == 1) & (a > 0) & (a <= 10), "dia_0_10", "otro"),
    )
    if isinstance(fechas, pd.Series):
        meses = fechas.dt.strftime("%Y-%m").to_numpy()
    else:
        meses = fechas.strftime("%Y-%m")
    conteo = pd.crosstab(pd.Series(meses, name="mes"), pd.Series(grupo, name="grupo"))
    conteo = conteo.reindex(columns=["transicion", "dia_0_10"], fill_value=0)
    return conteo.reset_index()


def ejecutar(ruta: Path, miembro: str | None = None, todas_figuras: bool = False) -> Path:
    tabla, anterior, datos_checkpoint, nombre_miembro = cargar_4_2_10(ruta, miembro)
    train, transicion, dia_bajo, recuentos = preparar_grupos(tabla)
    validar_metricas_previas(anterior)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Para generar figuras LOWESS instala statsmodels en Colab.") from exc
    print(f"Checkpoint: {nombre_miembro}")
    print(
        f"Train={len(train):,}; transiciones={len(transicion):,}; "
        f"día completo 0–10°={len(dia_bajo):,}."
    )

    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    carpeta = Path.cwd() / f"EDA_multivariante_F4_S4_211_transiciones_{sello}"
    if carpeta.exists() or carpeta.with_suffix(".zip").exists():
        raise FileExistsError(f"No se sobrescribirá una salida anterior: {carpeta}")
    tablas_dir, figuras_dir, checkpoint_dir = (
        carpeta / "tablas",
        carpeta / "figuras",
        carpeta / "checkpoint",
    )
    for destino in (tablas_dir, figuras_dir, checkpoint_dir):
        destino.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_horario_entrada_intacto.parquet").write_bytes(datos_checkpoint)

    variables_a_graficar = set(COVARIABLES if todas_figuras else FIGURAS_PRIORITARIAS)
    filas_metricas = []
    filas_cuantiles = []
    anterior = anterior.set_index("variable").loc[COVARIABLES].reset_index()
    for numero, variable in enumerate(COVARIABLES, start=1):
        x_trans, y_trans = pares_finitos(transicion, variable)
        x_dia, y_dia = pares_finitos(dia_bajo, variable)
        resultado_trans = calcular_metricas(x_trans, y_trans)
        resultado_dia = calcular_metricas(x_dia, y_dia)
        filas_metricas.append(
            {
                "variable": variable,
                **{f"{clave}_transicion": valor for clave, valor in resultado_trans.items()},
                **{f"{clave}_dia_0_10": valor for clave, valor in resultado_dia.items()},
            }
        )
        bins_trans = cuantiles_por_x(x_trans, y_trans, variable, "transicion")
        bins_dia = cuantiles_por_x(x_dia, y_dia, variable, "dia_0_10")
        filas_cuantiles.extend([bins_trans, bins_dia])
        if variable in variables_a_graficar:
            figura_individual(
                variable,
                x_trans,
                y_trans,
                x_dia,
                y_dia,
                bins_trans,
                bins_dia,
                figuras_dir / f"{numero:02d}_{variable}_transicion_vs_dia_0_10.png",
            )
        print(
            f"[{numero:02d}/26] {variable}: transición n={len(x_trans)}, día 0–10° n={len(x_dia)}"
        )

    nuevas = pd.DataFrame(filas_metricas)
    tabla_comparativa = anterior.merge(nuevas, on="variable", validate="one_to_one", sort=False)
    tabla_comparativa.to_csv(
        tablas_dir / "01_metricas_train_dia_transicion_dia_0_10.csv", index=False
    )
    pd.concat(filas_cuantiles, ignore_index=True).to_csv(
        tablas_dir / "02_mediana_DIF_cuantiles_X_transicion_y_dia_0_10.csv", index=False
    )
    recuento_por_mes(train).to_csv(tablas_dir / "03_disponibilidad_mensual.csv", index=False)
    pd.DataFrame([recuentos]).to_csv(
        tablas_dir / "04_verificacion_grupos_geometricos.csv", index=False
    )
    figura_resumen(tabla_comparativa, figuras_dir / "00_resumen_transicion_vs_dia_0_10.png")
    (carpeta / "README.json").write_text(
        json.dumps(
            {
                "subseccion": "4.2.11",
                "fuente_zip_4_2_10": ruta.name,
                "miembro_checkpoint_4_2_10": nombre_miembro,
                "checkpoint_original_inalterado": True,
                "objetivo": OBJETIVO,
                "grupos": {
                    "transicion": "0 < fraccion_minutos_dia < 1",
                    "dia_0_10": "fraccion_minutos_dia == 1 & 0 < elevacion_efectiva_h <= 10",
                },
                "train_y_dia_previos": "Métricas idénticas a la tabla 01 de 4.2.10; no recalculadas",
                "n_covariables": len(COVARIABLES),
                "n_bins_maximo": N_BINS,
                "pares": "Máscara finita de X y DIF específica por variable y grupo",
                "mi_k": K_VECINOS,
                "mi_random_state": SEMILLA_MI,
                "lowess": {"frac": 0.25, "it": 1},
                "figuras_prioritarias": list(FIGURAS_PRIORITARIAS),
                "figuras_todas": todas_figuras,
                "recuentos": recuentos,
                "advertencia": "Asociaciones contemporáneas descriptivas; no predicen DIF(t+1). "
                "N=429 y N=323 son muestras diferentes; no atribuir las diferencias "
                "exclusivamente a la transición ni a minutos nocturnos.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    archivo_zip = carpeta.with_suffix(".zip")
    with zipfile.ZipFile(archivo_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archivo in sorted(carpeta.rglob("*")):
            if archivo.is_file():
                zf.write(archivo, arcname=archivo.relative_to(carpeta))
    print(f"ZIP recargable de salida: {archivo_zip}")
    return archivo_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", help="Ruta al ZIP producido en 4.2.10")
    parser.add_argument("--checkpoint-miembro", help="Miembro parquet exacto si hay ambigüedad")
    parser.add_argument(
        "--todas-figuras", action="store_true", help="Dibuja las 26 variables; por defecto 10"
    )
    args = parser.parse_args()
    salida = ejecutar(solicitar_zip(args.zip), args.checkpoint_miembro, args.todas_figuras)
    try:
        from google.colab import files
    except ImportError:
        pass
    else:
        files.download(str(salida))


if __name__ == "__main__":
    main()
