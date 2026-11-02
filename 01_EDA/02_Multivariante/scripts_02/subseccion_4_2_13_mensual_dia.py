"""Comprobación mensual X(t) ↔ DIF(t): train frente a día completo.

Colab:
    %run subseccion_4_2_13_mensual_dia.py

Terminal:
    python subseccion_4_2_13_mensual_dia.py \
        --zip-checkpoint RUTA_ZIP_4_2_12.zip

Calcula con el mismo procedimiento las métricas mensuales de train y de
fraccion_minutos_dia == 1 a partir del checkpoint de 4.2.12.
No repite STL, no separa bandas solares y no estudia DIF(t+1).
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

OBJETIVO = "DIF__media_h"
FRACCION = "fraccion_minutos_dia"
PARTICION = "particion_observacion"
TIEMPO = "timestamp_utc"
MESES = tuple(f"2024-{mes:02d}" for mes in range(1, 8))
K_MI = 5
SEMILLA_MI = 42
VERSION = "4.2.13-v2-un-solo-zip"

ESTADISTICOS = ("media_h", "min_h", "max_h", "std_media_h")
FAMILIAS = ("SWD", "DIR", "DIF", "LWD", "SWU", "LWU")
VARIABLES = [f"{familia}__{estadistico}" for familia in FAMILIAS for estadistico in ESTADISTICOS]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
COVARIABLES = [variable for variable in VARIABLES if variable != OBJETIVO]


def validar_zip(ruta: str | Path) -> Path:
    ruta = Path(ruta).expanduser().resolve()
    if not ruta.is_file() or ruta.suffix.lower() != ".zip":
        raise FileNotFoundError(f"ZIP inexistente o inválido: {ruta}")
    return ruta


def inspeccionar_zip(ruta: Path) -> dict:
    with zipfile.ZipFile(ruta) as zf:
        nombres = zf.namelist()
    return {
        "ruta": ruta,
        "checkpoint": [
            n for n in nombres if n.endswith("checkpoint_horario_entrada_intacto.parquet")
        ],
        "nombres": nombres,
    }


def localizar_checkpoint_existente() -> Path | None:
    """Localiza en el directorio actual un ZIP de 4.2.12 ya subido a Colab."""
    candidatos = []
    directorios = [Path.cwd()]
    carpeta_colab = Path("/content")
    if carpeta_colab.is_dir() and carpeta_colab.resolve() != Path.cwd().resolve():
        directorios.append(carpeta_colab)
    for directorio in directorios:
        for ruta in directorio.glob("*.zip"):
            if "213_mensual_dia" in ruta.name:
                continue
            try:
                inspeccion = inspeccionar_zip(ruta)
            except (zipfile.BadZipFile, OSError):
                continue
            if len(inspeccion["checkpoint"]) == 1:
                candidatos.append(ruta.resolve())
    if not candidatos:
        return None
    candidatos.sort(key=lambda ruta: ("_212_" in ruta.name, ruta.stat().st_mtime), reverse=True)
    elegido = candidatos[0]
    if len(candidatos) > 1:
        print(
            "Se encontraron varios ZIP con checkpoint. "
            f"Se utilizará el más reciente de 4.2.12: {elegido.name}"
        )
    else:
        print(f"ZIP de 4.2.12 localizado automáticamente: {elegido.name}")
    return elegido


def solicitar_entrada(zip_checkpoint: str | None) -> Path:
    if zip_checkpoint:
        ruta = validar_zip(zip_checkpoint)
        inspeccion = inspeccionar_zip(ruta)
        if len(inspeccion["checkpoint"]) != 1:
            raise ValueError(
                "El ZIP indicado debe contener un único "
                "checkpoint_horario_entrada_intacto.parquet."
            )
        return ruta
    existente = localizar_checkpoint_existente()
    if existente is not None:
        return existente
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("Fuera de Colab indica --zip-checkpoint ZIP_4_2_12.zip.") from exc
    print("Sube únicamente el ZIP generado en 4.2.12 (regímenes solares).")
    subidos = files.upload()
    rutas = [validar_zip(nombre) for nombre in subidos if nombre.lower().endswith(".zip")]
    if len(subidos) != 1 or len(rutas) != 1:
        raise ValueError("Debes seleccionar exactamente un archivo ZIP: el de 4.2.12.")
    inspecciones = [inspeccionar_zip(ruta) for ruta in rutas]
    con_checkpoint = [i for i in inspecciones if len(i["checkpoint"]) == 1]
    if len(con_checkpoint) != 1:
        muestra = inspecciones[0]["nombres"][:20]
        raise ValueError(
            "El ZIP no contiene un único checkpoint_horario_entrada_intacto.parquet. "
            f"Primeros miembros encontrados: {muestra}"
        )
    return con_checkpoint[0]["ruta"]


def leer_checkpoint(ruta_zip: Path) -> tuple[pd.DataFrame, bytes, str]:
    inspeccion = inspeccionar_zip(ruta_zip)
    miembros = inspeccion["checkpoint"]
    if len(miembros) != 1:
        raise ValueError(
            "El ZIP 4.2.12 debe contener un solo checkpoint_horario_entrada_intacto.parquet; "
            f"contiene {miembros}."
        )
    with zipfile.ZipFile(ruta_zip) as zf:
        datos = zf.read(miembros[0])
    try:
        tabla = pd.read_parquet(io.BytesIO(datos))
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Instala pyarrow en Colab para leer el checkpoint parquet.") from exc
    requeridas = {PARTICION, FRACCION, *VARIABLES}
    if not isinstance(tabla.index, pd.DatetimeIndex):
        requeridas.add(TIEMPO)
    faltantes = sorted(requeridas - set(tabla.columns))
    if faltantes:
        raise ValueError(f"El checkpoint carece de columnas necesarias: {faltantes}")
    return tabla, datos, miembros[0]


def preparar_train(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = tabla.loc[tabla[PARTICION].eq("train")].copy()
    if len(train) != 5208:
        raise ValueError(f"Se esperaban 5.208 horas de train; se encontraron {len(train)}.")
    if isinstance(train.index, pd.DatetimeIndex):
        fechas = pd.Series(train.index, index=train.index)
    else:
        fechas = pd.to_datetime(train[TIEMPO], utc=True, errors="coerce")
    if fechas.isna().any():
        raise ValueError("El checkpoint contiene timestamps inválidos.")
    if getattr(fechas.dt, "tz", None) is None:
        fechas = fechas.dt.tz_localize("UTC")
    else:
        fechas = fechas.dt.tz_convert("UTC")
    train["mes_analisis"] = fechas.dt.strftime("%Y-%m").to_numpy()
    fraccion = pd.to_numeric(train[FRACCION], errors="coerce")
    if fraccion.isna().any() or (~fraccion.between(0, 1)).any():
        raise ValueError("fraccion_minutos_dia debe ser finita y estar entre 0 y 1.")
    noche = fraccion.eq(0)
    transicion = fraccion.gt(0) & fraccion.lt(1)
    completa = fraccion.eq(1)
    if (int(noche.sum()), int(transicion.sum()), int(completa.sum())) != (2191, 429, 2588):
        raise ValueError(
            "El checkpoint no reproduce los grupos validados: "
            f"noche={int(noche.sum())}, transición={int(transicion.sum())}, "
            f"día completo={int(completa.sum())}."
        )
    dia = train.loc[completa].copy()
    disponibilidad = (
        train.groupby("mes_analisis", sort=True)
        .size()
        .rename("n_horas_train")
        .to_frame()
        .join(
            dia.groupby("mes_analisis", sort=True).size().rename("n_horas_dia_completo"), how="left"
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"mes_analisis": "mes"})
    )
    return train, dia, disponibilidad


def pares_finitos(tabla: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(tabla[variable], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(tabla[OBJETIVO], errors="coerce").to_numpy(dtype=float)
    finitos = np.isfinite(x) & np.isfinite(y)
    return x[finitos], y[finitos]


def calcular_metricas(x: np.ndarray, y: np.ndarray, sufijo: str) -> dict:
    resultado = {
        f"n_pares_{sufijo}": len(x),
        f"pearson_{sufijo}": np.nan,
        f"spearman_{sufijo}": np.nan,
        f"mi_k5_{sufijo}_nats": np.nan,
    }
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return resultado
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resultado[f"pearson_{sufijo}"] = float(pearsonr(x, y).statistic)
        resultado[f"spearman_{sufijo}"] = float(spearmanr(x, y).statistic)
    if len(x) > K_MI + 1:
        resultado[f"mi_k5_{sufijo}_nats"] = float(
            mutual_info_regression(
                x.reshape(-1, 1),
                y,
                discrete_features=False,
                n_neighbors=K_MI,
                random_state=SEMILLA_MI,
            )[0]
        )
    return resultado


def calcular_mensuales(train: pd.DataFrame, dia: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for variable in COVARIABLES:
        for mes in MESES:
            grupo_train = train.loc[train["mes_analisis"].eq(mes)]
            grupo_dia = dia.loc[dia["mes_analisis"].eq(mes)]
            x_train, y_train = pares_finitos(grupo_train, variable)
            x_dia, y_dia = pares_finitos(grupo_dia, variable)
            filas.append(
                {
                    "variable": variable,
                    "mes": mes,
                    "n_horas_train": len(grupo_train),
                    "n_horas_dia": len(grupo_dia),
                    **calcular_metricas(x_train, y_train, "train"),
                    **calcular_metricas(x_dia, y_dia, "dia"),
                }
            )
    resultado = pd.DataFrame(filas)
    if len(resultado) != len(COVARIABLES) * len(MESES):
        raise ValueError("No se generaron exactamente las 182 comparaciones mensuales.")
    resultado["delta_pearson_dia_menos_train"] = (
        resultado["pearson_dia"] - resultado["pearson_train"]
    )
    resultado["delta_spearman_dia_menos_train"] = (
        resultado["spearman_dia"] - resultado["spearman_train"]
    )
    resultado["delta_mi_k5_dia_menos_train_nats"] = (
        resultado["mi_k5_dia_nats"] - resultado["mi_k5_train_nats"]
    )
    resultado["cambio_signo_pearson"] = (
        np.sign(resultado["pearson_train"]) != np.sign(resultado["pearson_dia"])
    ) & resultado[["pearson_train", "pearson_dia"]].notna().all(axis=1)
    resultado["cambio_signo_spearman"] = (
        np.sign(resultado["spearman_train"]) != np.sign(resultado["spearman_dia"])
    ) & resultado[["spearman_train", "spearman_dia"]].notna().all(axis=1)
    orden_variable = {v: i for i, v in enumerate(COVARIABLES)}
    orden_mes = {m: i for i, m in enumerate(MESES)}
    resultado["_ov"] = resultado["variable"].map(orden_variable)
    resultado["_om"] = resultado["mes"].map(orden_mes)
    return resultado.sort_values(["_ov", "_om"]).drop(columns=["_ov", "_om"]).reset_index(drop=True)


def graficar_variable(variable: str, datos: pd.DataFrame, destino: Path) -> None:
    sub = datos.loc[datos["variable"].eq(variable)].set_index("mes").reindex(MESES)
    x = np.arange(len(MESES))
    etiquetas = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul"]
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    especificaciones = [
        ("pearson_train", "pearson_dia", "Pearson", (-1.02, 1.02)),
        ("spearman_train", "spearman_dia", "Spearman", (-1.02, 1.02)),
        ("mi_k5_train_nats", "mi_k5_dia_nats", "Información mutua k=5 [nats]", None),
    ]
    for ax, (col_train, col_dia, titulo, limites) in zip(axs, especificaciones):
        ax.plot(x, sub[col_train], marker="o", lw=2, color="#4c78a8", label="train completo")
        ax.plot(x, sub[col_dia], marker="o", lw=2, color="#e45756", label="día completo")
        ax.set_ylabel(titulo)
        ax.axhline(0, color="grey", lw=0.8)
        ax.grid(alpha=0.25)
        if limites is not None:
            ax.set_ylim(*limites)
        else:
            valores = sub[[col_train, col_dia]].to_numpy(dtype=float)
            finitos = valores[np.isfinite(valores)]
            maximo = float(finitos.max()) if finitos.size else 0.0
            ax.set_ylim(0, max(0.05, maximo * 1.12))
        ax.legend(loc="best")
    axs[-1].set_xticks(x, etiquetas)
    axs[-1].set_xlabel("Mes de 2024")
    pares = ", ".join(str(int(n)) if np.isfinite(n) else "NA" for n in sub["n_pares_dia"])
    fig.suptitle(
        f"{variable} ↔ DIF(t): estabilidad mensual\n" f"Pares diurnos Ene–Jul: {pares}", fontsize=14
    )
    fig.text(
        0.5,
        0.01,
        "Día completo: fraccion_minutos_dia = 1. La diferencia excluye noches y transiciones.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def matriz(datos: pd.DataFrame, columna: str) -> pd.DataFrame:
    return datos.pivot(index="variable", columns="mes", values=columna).reindex(
        index=COVARIABLES, columns=MESES
    )


def anotar_matriz(ax, tabla: pd.DataFrame, formato: str = ".2f") -> None:
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            valor = tabla.iat[i, j]
            if np.isfinite(valor):
                ax.text(j, i, format(valor, formato), ha="center", va="center", fontsize=6.2)


def graficar_resumen(
    datos: pd.DataFrame,
    columnas: tuple[str, str, str],
    titulos: tuple[str, str, str],
    titulo_general: str,
    destino: Path,
    diferencias: bool = False,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(20, 14), sharey=True)
    for ax, columna, titulo in zip(axs, columnas, titulos):
        tabla = matriz(datos, columna)
        valores = tabla.to_numpy(dtype=float)
        if diferencias:
            finitos = np.abs(valores[np.isfinite(valores)])
            limite = float(finitos.max()) if finitos.size else 1.0
            limite = max(limite, 1e-6)
            norma = TwoSlopeNorm(vmin=-limite, vcenter=0, vmax=limite)
            imagen = ax.imshow(
                np.ma.masked_invalid(valores), cmap="coolwarm", norm=norma, aspect="auto"
            )
        elif "mi_" in columna:
            imagen = ax.imshow(np.ma.masked_invalid(valores), cmap="viridis", vmin=0, aspect="auto")
        else:
            imagen = ax.imshow(
                np.ma.masked_invalid(valores), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
            )
        ax.set_title(titulo)
        ax.set_xticks(np.arange(len(MESES)), ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul"])
        ax.set_yticks(np.arange(len(COVARIABLES)), COVARIABLES)
        anotar_matriz(ax, tabla)
        fig.colorbar(imagen, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(titulo_general, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destino, dpi=165)
    plt.close(fig)


def ejecutar(zip_checkpoint: Path) -> Path:
    checkpoint, bytes_checkpoint, miembro_checkpoint = leer_checkpoint(zip_checkpoint)
    train, dia, disponibilidad = preparar_train(checkpoint)
    resultados = calcular_mensuales(train, dia)

    # Las horas mensuales deben coincidir con la disponibilidad del checkpoint.
    esperadas = disponibilidad.set_index("mes")["n_horas_train"]
    for mes in MESES:
        valores = resultados.loc[resultados["mes"].eq(mes), "n_horas_train"].unique()
        if len(valores) != 1 or int(valores[0]) != int(esperadas.loc[mes]):
            raise ValueError(f"n_horas_train no coincide con el checkpoint para {mes}.")

    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    carpeta = Path.cwd() / f"EDA_multivariante_F4_S4_213_mensual_dia_{sello}"
    if carpeta.exists() or carpeta.with_suffix(".zip").exists():
        raise FileExistsError(f"No se sobrescribirá una salida anterior: {carpeta}")
    tdir, fdir, pdir = carpeta / "tablas", carpeta / "figuras", carpeta / "checkpoint"
    detalle_dir = fdir / "01_por_variable"
    for destino in (tdir, fdir, pdir, detalle_dir):
        destino.mkdir(parents=True, exist_ok=False if destino == carpeta else True)

    resultados.to_csv(tdir / "01_metricas_mensuales_train_vs_dia_completo.csv", index=False)
    disponibilidad.to_csv(tdir / "02_disponibilidad_mensual.csv", index=False)
    (pdir / "checkpoint_horario_entrada_intacto.parquet").write_bytes(bytes_checkpoint)

    indice = []
    for numero, variable in enumerate(COVARIABLES, start=1):
        nombre = f"{numero:02d}_{variable}_train_vs_dia_mensual.png"
        graficar_variable(variable, resultados, detalle_dir / nombre)
        indice.append(
            {"numero": numero, "variable": variable, "figura": f"figuras/01_por_variable/{nombre}"}
        )
        print(f"[{numero:02d}/26] {variable}")
    pd.DataFrame(indice).to_csv(tdir / "03_indice_figuras.csv", index=False)

    graficar_resumen(
        resultados,
        ("pearson_dia", "spearman_dia", "mi_k5_dia_nats"),
        ("Pearson", "Spearman", "MI k=5 [nats]"),
        "Relaciones mensuales con DIF durante horas completamente diurnas",
        fdir / "00_metricas_mensuales_dia_completo.png",
    )
    graficar_resumen(
        resultados,
        (
            "delta_pearson_dia_menos_train",
            "delta_spearman_dia_menos_train",
            "delta_mi_k5_dia_menos_train_nats",
        ),
        ("Δ Pearson", "Δ Spearman", "Δ MI k=5 [nats]"),
        "Diferencia mensual: día completo − train",
        fdir / "00_diferencias_dia_menos_train.png",
        diferencias=True,
    )

    metadatos = {
        "subseccion": "4.2.13",
        "objetivo": "Comprobación mensual complementaria train frente a día completo",
        "zip_checkpoint": zip_checkpoint.name,
        "miembro_checkpoint": miembro_checkpoint,
        "checkpoint_original_inmutado": True,
        "metricas_train_y_dia_calculadas_con_el_mismo_codigo": True,
        "n_train": len(train),
        "n_dia_completo": len(dia),
        "filtro_dia": "fraccion_minutos_dia == 1",
        "meses_comparables": list(MESES),
        "diciembre": "Solo disponibilidad; excluido de métricas por ser parcial",
        "n_covariables": len(COVARIABLES),
        "objetivo_variable": OBJETIVO,
        "MI": {"k": K_MI, "random_state": SEMILLA_MI, "unidades": "nats"},
        "pares": "Máscara de pares finitos específica por variable y mes",
        "no_incluido": ["repetición de STL", "bandas de elevación", "relaciones X-X", "DIF(t+1)"],
        "advertencia": (
            "Día completo menos train excluye conjuntamente noches y transiciones; "
            "no estima un efecto causal ni capacidad predictiva."
        ),
    }
    (carpeta / "README.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    salida = carpeta.with_suffix(".zip")
    with zipfile.ZipFile(salida, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fichero in sorted(carpeta.rglob("*")):
            if fichero.is_file():
                zf.write(fichero, arcname=fichero.relative_to(carpeta))
    print(f"ZIP generado: {salida}")
    return salida


def main() -> None:
    print(f"Versión del script: {VERSION}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-checkpoint", help="ZIP 4.2.12 con checkpoint enriquecido")
    args = parser.parse_args()
    zip_checkpoint = solicitar_entrada(args.zip_checkpoint)
    salida = ejecutar(zip_checkpoint)
    try:
        from google.colab import files
    except ImportError:
        return
    files.download(str(salida))


if __name__ == "__main__":
    main()
