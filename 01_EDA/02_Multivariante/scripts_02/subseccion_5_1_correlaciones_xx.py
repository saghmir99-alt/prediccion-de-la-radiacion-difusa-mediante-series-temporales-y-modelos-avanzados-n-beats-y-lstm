"""5.1. Pearson y Spearman X–X: train completo frente a día completo.

Colab, después de subir este archivo:
    %run subseccion_5_1_correlaciones_xx.py

Si el ZIP de 4.2.12 ya está en /content, se utiliza automáticamente.
En otro caso, se solicita exactamente un ZIP. También se puede indicar:
    %run subseccion_5_1_correlaciones_xx.py --zip-checkpoint /content/mi_zip.zip

El ZIP de salida contiene tablas, figuras y una copia sin modificar del
checkpoint horario. Esta subsección solo calcula correlaciones X–X; no
selecciona variables, no calcula relaciones retardadas y no repite STL.
"""

from __future__ import annotations

import argparse
import io
import json
import warnings
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

VERSION = "5.1-v1-un-solo-zip"
PARTICION = "particion_observacion"
FRACCION = "fraccion_minutos_dia"
OBJETIVO = "DIF__media_h"
TIMESTAMP = "timestamp_utc"
FAMILIAS = ("SWD", "DIR", "DIF", "LWD", "SWU", "LWU")
ESTADISTICOS = ("media_h", "min_h", "max_h", "std_media_h")
VARIABLES = [f"{familia}__{estadistico}" for familia in FAMILIAS for estadistico in ESTADISTICOS]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
COVARIABLES = [nombre for nombre in VARIABLES if nombre != OBJETIVO]
CONTEXTOS = ("train", "dia_completo")
DERIVADAS_DIF = {"DIF__min_h", "DIF__max_h", "DIF__std_media_h"}
NOMBRE_CHECKPOINT = "checkpoint_horario_entrada_intacto.parquet"
PATRON_ZIP_212 = "*_212_regimenes_solares*.zip"


def miembros_checkpoint(ruta: Path) -> list[str]:
    with zipfile.ZipFile(ruta) as zf:
        return [
            nombre
            for nombre in zf.namelist()
            if Path(nombre).name == NOMBRE_CHECKPOINT and not nombre.endswith("/")
        ]


def validar_zip(ruta: str | Path) -> Path:
    ruta = Path(ruta).expanduser().resolve()
    if not ruta.is_file() or ruta.suffix.lower() != ".zip":
        raise FileNotFoundError(f"No existe un ZIP válido en {ruta}")
    miembros = miembros_checkpoint(ruta)
    if len(miembros) != 1:
        raise ValueError(
            f"Se esperaba un único {NOMBRE_CHECKPOINT} en {ruta.name}; "
            f"se encontraron {len(miembros)}. Revisa el ZIP de 4.2.12."
        )
    return ruta


def localizar_zip_212() -> Path | None:
    directorios = [Path.cwd()]
    colab = Path("/content")
    if colab.is_dir() and colab.resolve() != Path.cwd().resolve():
        directorios.append(colab)
    candidatos = {}
    for directorio in directorios:
        for ruta in directorio.glob(PATRON_ZIP_212):
            ruta = ruta.resolve()
            try:
                if len(miembros_checkpoint(ruta)) == 1:
                    candidatos[str(ruta)] = ruta
            except (zipfile.BadZipFile, OSError):
                continue
    if not candidatos:
        return None
    seleccion = max(candidatos.values(), key=lambda ruta: ruta.stat().st_mtime)
    print(f"ZIP de 4.2.12 seleccionado: {seleccion}")
    if len(candidatos) > 1:
        print("Había varias copias; si esta no es la deseada, usa --zip-checkpoint RUTA.")
    return seleccion


def seleccionar_entrada(entrada: str | None) -> Path:
    if entrada:
        return validar_zip(entrada)
    existente = localizar_zip_212()
    if existente is not None:
        return existente
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError(
            "Indica --zip-checkpoint /ruta/ZIP_4_2_12.zip; "
            "fuera de Colab no se abre el selector de archivos."
        ) from exc
    print("Selecciona solo el ZIP de 4.2.12 (regímenes solares).")
    subidos = files.upload()
    if len(subidos) != 1:
        raise ValueError("Selecciona exactamente un ZIP, el de 4.2.12.")
    nombre = next(iter(subidos))
    return validar_zip(nombre)


def leer_checkpoint(ruta_zip: Path) -> tuple[pd.DataFrame, bytes, str]:
    miembro = miembros_checkpoint(ruta_zip)[0]
    with zipfile.ZipFile(ruta_zip) as zf:
        datos = zf.read(miembro)
    try:
        tabla = pd.read_parquet(io.BytesIO(datos))
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Para leer el checkpoint Parquet instala pyarrow en Colab "
            "con `!pip -q install pyarrow` y vuelve a ejecutar este script."
        ) from exc
    requeridas = {PARTICION, FRACCION, *VARIABLES}
    if not isinstance(tabla.index, pd.DatetimeIndex):
        requeridas.add(TIMESTAMP)
    faltantes = sorted(requeridas - set(tabla.columns))
    if faltantes:
        raise ValueError(f"El checkpoint de 4.2.12 carece de columnas: {faltantes}")
    if tabla.columns.duplicated().any():
        raise ValueError("El checkpoint contiene nombres de columnas duplicados.")
    return tabla, datos, miembro


def preparar_muestras(tabla: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    train = tabla.loc[tabla[PARTICION].eq("train")].copy()
    if isinstance(train.index, pd.DatetimeIndex):
        fechas = pd.DatetimeIndex(train.index)
        fechas = fechas.tz_localize("UTC") if fechas.tz is None else fechas.tz_convert("UTC")
    else:
        fechas = pd.DatetimeIndex(pd.to_datetime(train[TIMESTAMP], utc=True, errors="coerce"))
    if fechas.isna().any() or fechas.duplicated().any():
        raise ValueError("Los timestamps UTC de train contienen nulos o duplicados.")
    fraccion = pd.to_numeric(train[FRACCION], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(fraccion).all() or not ((fraccion >= 0) & (fraccion <= 1)).all():
        raise ValueError("fraccion_minutos_dia contiene valores ausentes o fuera de [0, 1].")
    noche = int((fraccion == 0).sum())
    transicion = int(((fraccion > 0) & (fraccion < 1)).sum())
    dia = train.loc[fraccion == 1].copy()
    conteos = (len(train), noche, transicion, len(dia))
    if conteos != (5208, 2191, 429, 2588):
        raise ValueError(
            "El checkpoint no reproduce los grupos ya validados de 4.2.12. "
            "Se esperaban (train, noche, transición, día completo) = "
            f"(5208, 2191, 429, 2588) y se encontraron {conteos}."
        )
    info = {
        "n_train": len(train),
        "n_noche_completa": noche,
        "n_transicion": transicion,
        "n_dia_completo": len(dia),
        "inicio_train_utc": fechas.min().isoformat(),
        "fin_train_utc": fechas.max().isoformat(),
    }
    return {"train": train, "dia_completo": dia}, info


def datos_numericos(tabla: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        columna: pd.to_numeric(tabla[columna], errors="coerce").to_numpy(dtype=float)
        for columna in COVARIABLES
    }


def inventario_contexto(vectores: dict[str, np.ndarray], contexto: str) -> pd.DataFrame:
    filas = []
    for variable, valores in vectores.items():
        finitos = valores[np.isfinite(valores)]
        _, frecuencias = np.unique(finitos, return_counts=True)
        modal = int(frecuencias.max()) if frecuencias.size else 0
        filas.append(
            {
                "contexto": contexto,
                "variable": variable,
                "familia": variable.split("__")[0],
                "estadistico": variable.split("__")[1],
                "n_horas": len(valores),
                "n_finitos": len(finitos),
                "n_distintos": len(frecuencias),
                "n_valor_mas_repetido": modal,
                "proporcion_valor_mas_repetido": modal / len(finitos) if len(finitos) else np.nan,
                "derivada_DIF_misma_hora": variable in DERIVADAS_DIF,
            }
        )
    return pd.DataFrame(filas)


def calcular_matrices(
    tabla: pd.DataFrame, contexto: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    vectores = datos_numericos(tabla)
    n = len(COVARIABLES)
    matrices = {
        clave: pd.DataFrame(
            np.nan if clave != "n_pares" else 0,
            index=COVARIABLES,
            columns=COVARIABLES,
            dtype=float if clave != "n_pares" else int,
        )
        for clave in ("pearson", "spearman", "n_pares")
    }
    for variable in COVARIABLES:
        x = vectores[variable]
        x = x[np.isfinite(x)]
        matrices["n_pares"].loc[variable, variable] = len(x)
        if len(x) >= 3 and np.unique(x).size >= 2:
            matrices["pearson"].loc[variable, variable] = 1.0
            matrices["spearman"].loc[variable, variable] = 1.0

    filas = []
    for a, b in combinations(COVARIABLES, 2):
        x, y = vectores[a], vectores[b]
        finitos = np.isfinite(x) & np.isfinite(y)
        x, y = x[finitos], y[finitos]
        n_pares = len(x)
        coef_pearson = np.nan
        coef_spearman = np.nan
        estado = "ok"
        if n_pares < 3:
            estado = "menos_de_tres_pares"
        elif np.unique(x).size < 2 or np.unique(y).size < 2:
            estado = "variable_constante_en_pares"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                coef_pearson = float(pearsonr(x, y).statistic)
                coef_spearman = float(spearmanr(x, y).statistic)
            if not (np.isfinite(coef_pearson) and np.isfinite(coef_spearman)):
                estado = "coeficiente_no_finito"
        for var_fila, var_col in ((a, b), (b, a)):
            matrices["n_pares"].loc[var_fila, var_col] = n_pares
            matrices["pearson"].loc[var_fila, var_col] = coef_pearson
            matrices["spearman"].loc[var_fila, var_col] = coef_spearman
        filas.append(
            {
                "contexto": contexto,
                "variable_a": a,
                "variable_b": b,
                "misma_familia": a.split("__")[0] == b.split("__")[0],
                "alguna_derivada_DIF_misma_hora": a in DERIVADAS_DIF or b in DERIVADAS_DIF,
                "n_pares": n_pares,
                "pearson": coef_pearson,
                "spearman": coef_spearman,
                "estado": estado,
            }
        )
    return pd.DataFrame(filas), matrices, inventario_contexto(vectores, contexto)


def comparar_contextos(train: pd.DataFrame, dia: pd.DataFrame) -> pd.DataFrame:
    cols = ["variable_a", "variable_b", "n_pares", "pearson", "spearman", "estado"]
    auxiliares = ["misma_familia", "alguna_derivada_DIF_misma_hora"]
    comparacion = train[cols + auxiliares].merge(
        dia[cols],
        on=["variable_a", "variable_b"],
        suffixes=("_train", "_dia_completo"),
        validate="one_to_one",
    )
    if len(comparacion) != len(COVARIABLES) * (len(COVARIABLES) - 1) // 2:
        raise ValueError("La comparación no contiene exactamente 325 parejas X–X.")
    for metrica in ("pearson", "spearman"):
        comparacion[f"delta_{metrica}_dia_menos_train"] = (
            comparacion[f"{metrica}_dia_completo"] - comparacion[f"{metrica}_train"]
        )
        comparacion[f"delta_magnitud_{metrica}_dia_menos_train"] = (
            comparacion[f"{metrica}_dia_completo"].abs() - comparacion[f"{metrica}_train"].abs()
        )
    return comparacion


def etiquetas_variables() -> list[str]:
    nombre_estadistico = {
        "media_h": "media",
        "min_h": "mínimo",
        "max_h": "máximo",
        "std_media_h": "std",
    }
    return [
        f"{variable.split('__')[0]} {nombre_estadistico[variable.split('__')[1]]}"
        for variable in COVARIABLES
    ]


def dibujar_matrices(
    m_train: pd.DataFrame, m_dia: pd.DataFrame, metrica: str, salida: Path
) -> None:
    a = m_train.to_numpy(dtype=float)
    b = m_dia.to_numpy(dtype=float)
    diferencia = b - a
    validos = np.abs(diferencia[np.isfinite(diferencia)])
    limite_delta = max(0.05, float(validos.max()) if validos.size else 0.05)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    fig, ejes = plt.subplots(1, 3, figsize=(29, 12), constrained_layout=True)
    etiquetas = etiquetas_variables()
    for ax, datos, titulo, limite in zip(
        ejes,
        (a, b, diferencia),
        ("Train completo", "Día completo", "Día completo − train"),
        (1.0, 1.0, limite_delta),
    ):
        img = ax.imshow(
            np.ma.masked_invalid(datos),
            cmap=cmap,
            vmin=-limite,
            vmax=limite,
            interpolation="nearest",
        )
        ax.set_title(f"{metrica.capitalize()} · {titulo}", fontsize=14)
        ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=90, fontsize=7)
        ax.set_yticks(range(len(etiquetas)), etiquetas if ax is ejes[0] else [], fontsize=7)
        ax.tick_params(length=0)
        fig.colorbar(img, ax=ax, fraction=0.047, pad=0.03)
    fig.suptitle(
        f"Relaciones entre covariables ({metrica}) · 26 variables · "
        "train: 5.208 h; día completo: 2.588 h",
        fontsize=17,
    )
    fig.savefig(salida, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dibujar_comparacion_parejas(tabla: pd.DataFrame, salida: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    grupos = [
        ("Entre familias", tabla["misma_familia"].eq(False), "#4c78a8"),
        ("Dentro de familia", tabla["misma_familia"].eq(True), "#e45756"),
    ]
    for ax, metrica in zip(ejes, ("pearson", "spearman")):
        for nombre, mascara, color in grupos:
            ax.scatter(
                tabla.loc[mascara, f"{metrica}_train"],
                tabla.loc[mascara, f"{metrica}_dia_completo"],
                c=color,
                s=24,
                alpha=0.55,
                linewidths=0,
                label=nombre,
            )
        ax.plot([-1, 1], [-1, 1], color="black", lw=1, ls="--", label="Sin cambio")
        ax.set(
            xlim=(-1.03, 1.03),
            ylim=(-1.03, 1.03),
            xlabel=f"{metrica.capitalize()} en train completo",
            ylabel=f"{metrica.capitalize()} en día completo",
        )
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Comparación de las 325 parejas de covariables", fontsize=15)
    fig.savefig(salida, dpi=180)
    plt.close(fig)


def exportar(ruta_zip: Path) -> Path:
    checkpoint, bytes_checkpoint, miembro = leer_checkpoint(ruta_zip)
    muestras, info = preparar_muestras(checkpoint)
    resultados, matrices, inventarios = {}, {}, []
    for contexto in CONTEXTOS:
        parejas, mats, inventario = calcular_matrices(muestras[contexto], contexto)
        resultados[contexto] = parejas
        matrices[contexto] = mats
        inventarios.append(inventario)
        print(f"{contexto}: {len(muestras[contexto])} horas y {len(parejas)} parejas X–X")
    comparacion = comparar_contextos(resultados["train"], resultados["dia_completo"])

    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = Path.cwd() / f"EDA_multivariante_5_1_correlaciones_xx_{sello}"
    if destino.exists() or destino.with_suffix(".zip").exists():
        raise FileExistsError(f"Ya existe la salida {destino}; no se sobrescribe.")
    tablas, figuras, checkpoint_dir = (
        destino / "tablas",
        destino / "figuras",
        destino / "checkpoint",
    )
    for carpeta in (tablas, figuras, checkpoint_dir):
        carpeta.mkdir(parents=True, exist_ok=True)

    pd.concat(resultados.values(), ignore_index=True).to_csv(
        tablas / "01_correlaciones_parejas_train_y_dia.csv", index=False
    )
    comparacion.to_csv(tablas / "02_comparacion_parejas_dia_menos_train.csv", index=False)
    pd.concat(inventarios, ignore_index=True).to_csv(
        tablas / "03_inventario_valores_por_contexto.csv", index=False
    )
    for contexto in CONTEXTOS:
        for nombre, tabla in matrices[contexto].items():
            tabla.to_csv(tablas / f"matriz_{nombre}_{contexto}.csv", index_label="variable")
    for metrica in ("pearson", "spearman"):
        base = matrices["train"][metrica]
        completo = matrices["dia_completo"][metrica]
        (completo - base).to_csv(
            tablas / f"matriz_delta_{metrica}_dia_menos_train.csv", index_label="variable"
        )
        dibujar_matrices(base, completo, metrica, figuras / f"01_matrices_{metrica}.png")
    dibujar_comparacion_parejas(comparacion, figuras / "03_parejas_train_vs_dia_completo.png")
    (checkpoint_dir / NOMBRE_CHECKPOINT).write_bytes(bytes_checkpoint)

    metadata = {
        "subseccion": "5.1",
        "version_script": VERSION,
        "fuente_zip": ruta_zip.name,
        "miembro_checkpoint_entrada": miembro,
        "checkpoint_sin_modificar": True,
        "objetivo_excluido_de_matrices": OBJETIVO,
        "covariables": COVARIABLES,
        "n_covariables": len(COVARIABLES),
        "n_parejas_por_contexto": len(comparacion),
        "filtro_train": "particion_observacion == 'train'",
        "filtro_dia": "fraccion_minutos_dia == 1 dentro de train",
        "muestras": info,
        "pares": "Cada pareja se calcula con su intersección de valores finitos.",
        "metodos": "Correlaciones Pearson y Spearman; diferencia día completo menos train.",
        "avisos": [
            "El cambio de muestra excluye a la vez noches y transiciones.",
            "Las correlaciones entre covariables no demuestran redundancia predictiva.",
            "Las variables DIF__min_h, DIF__max_h y DIF__std_media_h proceden de la misma "
            "hora que el objetivo; revisar su disponibilidad temporal al modelar.",
            "La proporción del valor más repetido es solo un diagnóstico de empates.",
            "No se aplicaron imputación, pruebas de significación, STL ni retardos.",
        ],
    }
    (destino / "README.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    salida_zip = destino.with_suffix(".zip")
    with zipfile.ZipFile(salida_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archivo in sorted(destino.rglob("*")):
            if archivo.is_file():
                zf.write(archivo, arcname=archivo.relative_to(destino))
    print(f"Resultado guardado en {salida_zip}")
    return salida_zip


def main() -> None:
    print(f"Versión del script: {VERSION}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-checkpoint", help="Ruta al ZIP 4.2.12 con el checkpoint horario")
    argumentos = parser.parse_args()
    resultado = exportar(seleccionar_entrada(argumentos.zip_checkpoint))
    try:
        from google.colab import files
    except ImportError:
        return
    files.download(str(resultado))


if __name__ == "__main__":
    main()
