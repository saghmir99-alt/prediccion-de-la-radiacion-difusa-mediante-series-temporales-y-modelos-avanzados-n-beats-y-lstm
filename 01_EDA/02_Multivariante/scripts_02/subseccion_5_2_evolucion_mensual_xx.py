"""5.2. Evolución mensual Pearson/Spearman X–X: train frente a día completo.

Colab, después de subir este script y el ZIP de 5.1:
    %run subseccion_5_2_evolucion_mensual_xx.py

Si el ZIP ya está en /content se selecciona automáticamente. En caso contrario
se solicita EXACTAMENTE UN ZIP. También acepta directamente el ZIP de 4.2.12,
que contiene el mismo checkpoint horario; en ese caso reconstruye 5.1.

    %run subseccion_5_2_evolucion_mensual_xx.py --zip-entrada /content/mi_zip.zip

Fuera de Colab:
    python subseccion_5_2_evolucion_mensual_xx.py --zip-entrada mi_zip.zip

El archivo de salida contiene TODAS las parejas mensuales, criterios de
priorización, matrices visuales de todos los meses, gráficos individuales
de todas las parejas priorizadas y una copia intacta del checkpoint.
No hay imputación ni inferencia causal o selección automática de predictores.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
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

VERSION = "5.2-v1-una-entrada"
MESES = tuple(f"2024-{numero:02d}" for numero in range(1, 8))
ETIQUETAS_MESES = ("Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul")
ESTADISTICOS = ("media_h", "min_h", "max_h", "std_media_h")
FAMILIAS = ("SWD", "DIR", "DIF", "LWD", "SWU", "LWU")
TODAS = [f"{familia}__{estadistico}" for familia in FAMILIAS for estadistico in ESTADISTICOS]
TODAS += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
OBJETIVO = "DIF__media_h"
VARIABLES = [var for var in TODAS if var != OBJETIVO]
PAREJAS = list(combinations(VARIABLES, 2))
CONTEXTOS = ("train", "dia_completo")
METRICAS = ("pearson", "spearman")
CHECKPOINT = "checkpoint_horario_entrada_intacto.parquet"
BASE_51 = "02_comparacion_parejas_dia_menos_train.csv"
DERIVADAS_DIF = {"DIF__min_h", "DIF__max_h", "DIF__std_media_h"}

# Criterios descriptivos validados para decidir qué se ilustra en detalle.
UMBRAL_A = 0.90  # Asociación global en día completo, en al menos una medida.
UMBRAL_B = 0.30  # Cambio de magnitud global train -> día.
UMBRAL_C = 0.70  # Asociación mensual fuerte en un mismo coeficiente.
MIN_MESES_C = 4  # Al menos cuatro meses y sin oposición de magnitud >=0.30.
UMBRAL_SIGNO = 0.30  # Signo relevante, excluye cambios diminutos junto a cero.
MIN_MESES_D = 2  # Al menos dos meses con cada signo.
MIN_MESES_E = 2  # Dos meses con cambios de intensidad >=0.30 y misma dirección.

# Contrastes fijados antes del cálculo de 5.2; se dibujan aunque no activen A--E.
CONTRASTES_PREDEFINIDOS = (
    ("SWD__media_h", "SWD__std_media_h"),
    ("DIR__media_h", "DIR__min_h"),
    ("LWD__media_h", "LWD__std_media_h"),
    ("SWD__media_h", "RH__media_h"),
    ("DIR__media_h", "PoPoPoPo__media_h"),
)


def _miembros(ruta: Path) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(ruta) as zf:
        nombres = zf.namelist()
    checkpoint = [m for m in nombres if Path(m).name == CHECKPOINT and not m.endswith("/")]
    previo = [m for m in nombres if Path(m).name == BASE_51 and not m.endswith("/")]
    return checkpoint, previo


def validar_zip(ruta: str | Path) -> Path:
    ruta = Path(ruta).expanduser().resolve()
    if not ruta.is_file() or ruta.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Indica un ZIP existente: {ruta}")
    try:
        miembros, _ = _miembros(ruta)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"El archivo no es un ZIP legible: {ruta}") from exc
    if len(miembros) != 1:
        raise ValueError(
            f"{ruta.name} debe contener UN {CHECKPOINT}; se hallaron {len(miembros)}. "
            "Sube el ZIP de 5.1 o de 4.2.12."
        )
    return ruta


def obtener_zip(entrada: str | None) -> Path:
    if entrada:
        return validar_zip(entrada)
    directorios = [Path.cwd()]
    if Path("/content").is_dir() and Path("/content").resolve() != Path.cwd().resolve():
        directorios.append(Path("/content"))
    candidatos: dict[Path, tuple[int, float]] = {}
    for directorio in directorios:
        for ruta in directorio.glob("*.zip"):
            if "_5_2_evolucion_mensual_xx_" in ruta.name:
                continue
            try:
                miembros, previo = _miembros(ruta)
            except (zipfile.BadZipFile, OSError):
                continue
            if len(miembros) == 1:
                prioridad = 2 if previo else (1 if "_212_" in ruta.name else 0)
                candidatos[ruta.resolve()] = (prioridad, ruta.stat().st_mtime)
    if candidatos:
        elegido = max(candidatos, key=lambda p: candidatos[p])
        print(f"ZIP de entrada seleccionado: {elegido.name}")
        if len(candidatos) > 1:
            print("Si deseas otro ZIP, ejecuta el script con --zip-entrada /ruta/archivo.zip")
        return elegido
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("Indica --zip-entrada /ruta/al/ZIP_de_5_1_o_4_2_12.zip") from exc
    print("Sube un único ZIP: preferiblemente el de 5.1, o bien el de 4.2.12.")
    subidos = files.upload()
    if len(subidos) != 1:
        raise ValueError("Selecciona exactamente un ZIP; no se necesitan dos archivos.")
    return validar_zip(next(iter(subidos)))


def cargar_entrada(ruta: Path) -> tuple[pd.DataFrame, bytes, pd.DataFrame | None, str]:
    miembros, previo = _miembros(ruta)
    with zipfile.ZipFile(ruta) as zf:
        bytes_parquet = zf.read(miembros[0])
        base_51 = pd.read_csv(io.BytesIO(zf.read(previo[0]))) if len(previo) == 1 else None
    try:
        df = pd.read_parquet(io.BytesIO(bytes_parquet))
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Para leer el checkpoint instala pyarrow en Colab con "
            "`!pip -q install pyarrow` y vuelve a ejecutar."
        ) from exc
    requeridas = {*TODAS, "particion_observacion", "fraccion_minutos_dia"}
    if not isinstance(df.index, pd.DatetimeIndex):
        requeridas.add("timestamp_utc")
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes or df.columns.duplicated().any():
        raise ValueError(f"Checkpoint incompatible: faltan {faltantes} o hay nombres duplicados.")
    return df, bytes_parquet, base_51, miembros[0]


def preparar_datos(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    train = df.loc[df["particion_observacion"].eq("train")].copy()
    if isinstance(train.index, pd.DatetimeIndex):
        fechas = pd.DatetimeIndex(pd.to_datetime(train.index, utc=True, errors="coerce"))
    else:
        fechas = pd.DatetimeIndex(pd.to_datetime(train["timestamp_utc"], utc=True, errors="coerce"))
    if fechas.isna().any() or fechas.duplicated().any():
        raise ValueError("Las horas UTC del entrenamiento contienen nulos o duplicados.")
    train["mes_analisis"] = fechas.strftime("%Y-%m").to_numpy()
    fraccion = pd.to_numeric(train["fraccion_minutos_dia"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(fraccion).all() and ((fraccion >= 0) & (fraccion <= 1)).all()):
        raise ValueError("fraccion_minutos_dia debe ser finita y pertenecer a [0, 1].")
    grupos = (
        int((fraccion == 0).sum()),
        int(((fraccion > 0) & (fraccion < 1)).sum()),
        int((fraccion == 1).sum()),
    )
    if grupos != (2191, 429, 2588):
        raise ValueError(
            "El checkpoint no reproduce los grupos de 5.1: "
            f"noche/transición/día={grupos}, esperados=(2191, 429, 2588)."
        )
    dia = train.loc[fraccion == 1].copy()
    if (len(train), len(dia)) != (5208, 2588):
        raise ValueError(
            f"El checkpoint no coincide con 5.1: train={len(train)}, día={len(dia)}; "
            "se esperaban 5208 y 2588 horas."
        )
    disponibilidad = (
        train.groupby("mes_analisis")
        .size()
        .rename("n_train")
        .to_frame()
        .join(dia.groupby("mes_analisis").size().rename("n_dia_completo"), how="left")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"mes_analisis": "mes"})
    )
    dispo_analisis = disponibilidad.set_index("mes").reindex(MESES)
    if dispo_analisis.isna().any().any() or tuple(dispo_analisis.sum()) != (5112, 2560):
        raise ValueError("Las horas de enero--julio no reproducen 5112 train y 2560 día completo.")
    print("Horas enero--julio: train=5112, día completo=2560; diciembre excluido.")
    return {"train": train, "dia_completo": dia}, disponibilidad


def _numericos(frame: pd.DataFrame) -> pd.DataFrame:
    tabla = frame[VARIABLES].apply(pd.to_numeric, errors="coerce")
    return tabla.replace([np.inf, -np.inf], np.nan)


def matrices_y_pares(tabla: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    x = _numericos(tabla)
    matrices = {
        m: x.corr(method=m, min_periods=3).reindex(index=VARIABLES, columns=VARIABLES)
        for m in METRICAS
    }
    valido = x.notna().to_numpy(dtype=np.int32)
    n_pares = valido.T @ valido
    filas = []
    for i, j in combinations(range(len(VARIABLES)), 2):
        a, b = VARIABLES[i], VARIABLES[j]
        r, rho = (matrices[m].iat[i, j] for m in METRICAS)
        n = int(n_pares[i, j])
        estado = (
            "menos_de_tres_pares"
            if n < 3
            else "constante_o_indefinida" if not np.isfinite(r) or not np.isfinite(rho) else "ok"
        )
        filas.append(
            {
                "variable_a": a,
                "variable_b": b,
                "familia_a": a.split("__")[0],
                "familia_b": b.split("__")[0],
                "misma_familia": a.split("__")[0] == b.split("__")[0],
                "alguna_derivada_DIF_misma_hora": a in DERIVADAS_DIF or b in DERIVADAS_DIF,
                "n_pares": n,
                "pearson": r,
                "spearman": rho,
                "estado": estado,
            }
        )
    return matrices, pd.DataFrame(filas)


def recuperar_51(
    base: pd.DataFrame | None, muestras: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, str]:
    """Compara el CSV de 5.1 con el checkpoint; si falta, reproduce 5.1."""
    marcos = {}
    for contexto in CONTEXTOS:
        _, marcos[contexto] = matrices_y_pares(muestras[contexto])
    columnas = ["variable_a", "variable_b", "n_pares", "pearson", "spearman"]
    calculada = marcos["train"][columnas].merge(
        marcos["dia_completo"][columnas],
        on=["variable_a", "variable_b"],
        suffixes=("_train", "_dia_completo"),
        validate="one_to_one",
    )
    if base is None:
        return calculada, "recalculado desde el checkpoint (sin segundo ZIP)"
    requeridas = set(calculada.columns)
    if len(base) != len(PAREJAS) or not requeridas.issubset(base.columns):
        raise ValueError("La tabla de 5.1 dentro del ZIP es incompatible con las 325 parejas.")
    cotejo = calculada.merge(
        base[list(requeridas)],
        on=["variable_a", "variable_b"],
        suffixes=("_check", "_zip"),
        validate="one_to_one",
    )
    if len(cotejo) != len(PAREJAS):
        raise ValueError("Las parejas del CSV de 5.1 no coinciden con el checkpoint.")
    for nombre in (
        "n_pares_train",
        "n_pares_dia_completo",
        "pearson_train",
        "pearson_dia_completo",
        "spearman_train",
        "spearman_dia_completo",
    ):
        a, b = cotejo[f"{nombre}_check"], cotejo[f"{nombre}_zip"]
        if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError(f"CSV de 5.1 y checkpoint discrepan en {nombre}.")
    return (
        base.set_index(["variable_a", "variable_b"])
        .loc[pd.MultiIndex.from_tuples(PAREJAS)]
        .reset_index(),
        "CSV original de 5.1 comprobado frente al checkpoint",
    )


def calcular_meses(
    muestras: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    filas, inventarios, matrices = [], [], {}
    for mes in MESES:
        matrices[mes] = {}
        for contexto in CONTEXTOS:
            datos = muestras[contexto].loc[muestras[contexto]["mes_analisis"].eq(mes)]
            mats, parejas = matrices_y_pares(datos)
            matrices[mes][contexto] = mats
            parejas.insert(0, "mes", mes)
            parejas.insert(1, "contexto", contexto)
            parejas.insert(2, "n_horas_contexto", len(datos))
            filas.append(parejas)
            numericos = _numericos(datos)
            for var in VARIABLES:
                x = numericos[var].dropna()
                modal = int(x.value_counts(dropna=False).max()) if len(x) else 0
                inventarios.append(
                    {
                        "mes": mes,
                        "contexto": contexto,
                        "variable": var,
                        "n_horas": len(datos),
                        "n_finitos": len(x),
                        "n_distintos": int(x.nunique()),
                        "n_valor_mas_repetido": modal,
                        "proporcion_valor_mas_repetido": modal / len(x) if len(x) else np.nan,
                    }
                )
        print(
            f"{mes}: {len(muestras['train'].loc[muestras['train']['mes_analisis'].eq(mes)])} "
            "train y "
            f"{len(muestras['dia_completo'].loc[muestras['dia_completo']['mes_analisis'].eq(mes)])} "
            "día; 325 parejas en cada contexto"
        )
    todas = pd.concat(filas, ignore_index=True)
    cols = ["mes", "variable_a", "variable_b", "n_pares", "pearson", "spearman", "estado"]
    extra = ["misma_familia", "alguna_derivada_DIF_misma_hora"]
    comparacion = todas.loc[todas.contexto.eq("train"), cols + extra].merge(
        todas.loc[todas.contexto.eq("dia_completo"), cols],
        on=["mes", "variable_a", "variable_b"],
        suffixes=("_train", "_dia_completo"),
        validate="one_to_one",
    )
    for metrica in METRICAS:
        comparacion[f"delta_{metrica}_dia_menos_train"] = (
            comparacion[f"{metrica}_dia_completo"] - comparacion[f"{metrica}_train"]
        )
        comparacion[f"delta_magnitud_{metrica}_dia_menos_train"] = (
            comparacion[f"{metrica}_dia_completo"].abs() - comparacion[f"{metrica}_train"].abs()
        )
    if len(todas) != 4550 or len(comparacion) != 2275:
        raise ValueError("Recuento incorrecto: se esperaban 4550 registros y 2275 comparaciones.")
    return todas, comparacion, matrices, pd.DataFrame(inventarios)


def _criterio_c(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(
        ((x >= UMBRAL_C).sum() >= MIN_MESES_C and not (x <= -UMBRAL_SIGNO).any())
        or ((x <= -UMBRAL_C).sum() >= MIN_MESES_C and not (x >= UMBRAL_SIGNO).any())
    )


def _criterio_d(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(
        ((x >= UMBRAL_SIGNO).sum() >= MIN_MESES_D) and ((x <= -UMBRAL_SIGNO).sum() >= MIN_MESES_D)
    )


def _criterio_e(delta_magnitud: np.ndarray) -> bool:
    delta_magnitud = np.asarray(delta_magnitud, dtype=float)
    return bool(
        ((delta_magnitud >= UMBRAL_B).sum() >= MIN_MESES_E)
        or ((delta_magnitud <= -UMBRAL_B).sum() >= MIN_MESES_E)
    )


def marcar_parejas(base: pd.DataFrame, mensual: pd.DataFrame) -> pd.DataFrame:
    prev = base.set_index(["variable_a", "variable_b"])
    meses = mensual.set_index(["variable_a", "variable_b"]).sort_index()
    indices = {clave: k for k, clave in enumerate(PAREJAS, start=1)}
    controles = {frozenset(par) for par in CONTRASTES_PREDEFINIDOS}
    filas = []
    for a, b in PAREJAS:
        antes = prev.loc[(a, b)]
        sub = meses.loc[(a, b)].set_index("mes").loc[list(MESES)]
        A = any(
            np.isfinite(antes[f"{m}_dia_completo"]) and abs(antes[f"{m}_dia_completo"]) >= UMBRAL_A
            for m in METRICAS
        )
        B = any(
            np.isfinite(antes[f"{m}_dia_completo"])
            and np.isfinite(antes[f"{m}_train"])
            and abs(abs(antes[f"{m}_dia_completo"]) - abs(antes[f"{m}_train"])) >= UMBRAL_B
            for m in METRICAS
        )
        v = {m: sub[f"{m}_dia_completo"].to_numpy(dtype=float) for m in METRICAS}
        deltas = {
            m: sub[f"delta_magnitud_{m}_dia_menos_train"].to_numpy(dtype=float) for m in METRICAS
        }
        C = {m: _criterio_c(v[m]) for m in METRICAS}
        D = {m: _criterio_d(v[m]) for m in METRICAS}
        E = {m: _criterio_e(deltas[m]) for m in METRICAS}
        tiene = {"A": A, "B": B, "C": any(C.values()), "D": any(D.values()), "E": any(E.values())}
        fila = {
            "variable_a": a,
            "variable_b": b,
            "orden_pareja": indices[(a, b)],
            "familia_a": a.split("__")[0],
            "familia_b": b.split("__")[0],
            "misma_familia": a.split("__")[0] == b.split("__")[0],
            "alguna_derivada_DIF_misma_hora": a in DERIVADAS_DIF or b in DERIVADAS_DIF,
            "contraste_predefinido": frozenset((a, b)) in controles,
            "criterios_activados": ",".join(c for c, ok in tiene.items() if ok),
            "prioritaria": any(tiene.values()),
        }
        for letra, ok in tiene.items():
            fila[f"criterio_{letra}"] = ok
        for m in METRICAS:
            fila[f"criterio_C_{m}"] = C[m]
            fila[f"criterio_D_{m}"] = D[m]
            fila[f"criterio_E_{m}"] = E[m]
            for contexto in CONTEXTOS:
                valores = sub[f"{m}_{contexto}"].to_numpy(dtype=float)
                validos = valores[np.isfinite(valores)]
                fila[f"n_meses_validos_{m}_{contexto}"] = len(validos)
                fila[f"min_{m}_{contexto}"] = validos.min() if len(validos) else np.nan
                fila[f"max_{m}_{contexto}"] = validos.max() if len(validos) else np.nan
                fila[f"mediana_{m}_{contexto}"] = np.median(validos) if len(validos) else np.nan
                fila[f"n_meses_{m}_abs_ge_070_{contexto}"] = int((np.abs(valores) >= 0.70).sum())
            fila[f"n_meses_refuerzo_{m}_ge_030"] = int((deltas[m] >= UMBRAL_B).sum())
            fila[f"n_meses_atenuacion_{m}_ge_030"] = int((deltas[m] <= -UMBRAL_B).sum())
        fila["n_pares_min_train"] = int(sub.n_pares_train.min())
        fila["n_pares_min_dia_completo"] = int(sub.n_pares_dia_completo.min())
        fila["mostrar_figura_individual"] = fila["prioritaria"] or fila["contraste_predefinido"]
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_mensual(tabla: pd.DataFrame, disponibilidad: pd.DataFrame) -> pd.DataFrame:
    indice = disponibilidad.set_index("mes")
    filas = []
    for mes in MESES:
        sub = tabla.loc[tabla.mes.eq(mes)]
        fila = {
            "mes": mes,
            "n_horas_train": int(indice.loc[mes, "n_train"]),
            "n_horas_dia_completo": int(indice.loc[mes, "n_dia_completo"]),
            "n_parejas": len(sub),
        }
        for m in METRICAS:
            for contexto in CONTEXTOS:
                abs_r = sub[f"{m}_{contexto}"].abs()
                fila[f"n_parejas_abs_{m}_ge_090_{contexto}"] = int((abs_r >= 0.90).sum())
                fila[f"mediana_magnitud_{m}_{contexto}"] = float(abs_r.median())
            delta = sub[f"delta_magnitud_{m}_dia_menos_train"]
            fila[f"n_refuerzo_{m}_ge_030"] = int((delta >= 0.30).sum())
            fila[f"n_atenuacion_{m}_ge_030"] = int((delta <= -0.30).sum())
        filas.append(fila)
    return pd.DataFrame(filas)


def nombres_cortos() -> list[str]:
    abreviar = {"media_h": "media", "min_h": "mín.", "max_h": "máx.", "std_media_h": "std"}
    return [f"{v.split('__')[0]} {abreviar[v.split('__')[1]]}" for v in VARIABLES]


def etiqueta_par(a: str, b: str) -> str:
    dic = {v: nom for v, nom in zip(VARIABLES, nombres_cortos())}
    return f"{dic[a]} · {dic[b]}"


def paleta() -> matplotlib.colors.Colormap:
    c = plt.get_cmap("RdBu_r").copy()
    c.set_bad("#e5e5e5")
    return c


def dibujar_disponibilidad(disponibilidad: pd.DataFrame, destino: Path) -> None:
    d = disponibilidad.set_index("mes").loc[list(MESES)]
    x = np.arange(len(MESES))
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    ax.bar(x - 0.20, d.n_train, 0.37, label="Train", color="#4c78a8")
    ax.bar(x + 0.20, d.n_dia_completo, 0.37, label="Día completo", color="#e4a043")
    ax.set_xticks(x, ETIQUETAS_MESES)
    ax.set_ylabel("Horas disponibles")
    ax.set_title("Muestras mensuales utilizadas para X–X (enero–julio de 2024)")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(destino, dpi=165)
    plt.close(fig)


def dibujar_resumen_mes(tabla: pd.DataFrame, destino: Path) -> None:
    x = np.arange(len(MESES))
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for j, metrica in enumerate(METRICAS):
        ax = axs[0, j]
        ax.plot(x, tabla[f"n_parejas_abs_{metrica}_ge_090_train"], "o-", label="Train")
        ax.plot(
            x, tabla[f"n_parejas_abs_{metrica}_ge_090_dia_completo"], "o-", label="Día completo"
        )
        ax.set_title(f"{metrica.capitalize()}: parejas con |coeficiente| ≥ 0,90")
        ax.set_ylabel("N.º de parejas (de 325)")
        ax.legend()
        ax2 = axs[1, j]
        ax2.bar(
            x - 0.18,
            tabla[f"n_refuerzo_{metrica}_ge_030"],
            width=0.36,
            label="Se refuerzan ≥ 0,30",
            color="#b65150",
        )
        ax2.bar(
            x + 0.18,
            tabla[f"n_atenuacion_{metrica}_ge_030"],
            width=0.36,
            label="Se atenúan ≥ 0,30",
            color="#4c78a8",
        )
        ax2.set_title(f"{metrica.capitalize()}: cambio mensual de magnitud al filtrar")
        ax2.set_ylabel("N.º de parejas (de 325)")
        ax2.legend(fontsize=9)
        for ax_ in (ax, ax2):
            ax_.set_xticks(x, ETIQUETAS_MESES)
            ax_.grid(axis="y", alpha=0.2)
    fig.suptitle("Visión mensual de todas las parejas X–X; recuentos descriptivos", fontsize=15)
    fig.savefig(destino, dpi=155)
    plt.close(fig)


def _eje_matriz(
    ax,
    imagen: np.ndarray,
    titulo: str,
    etiquetas: list[str],
    cmap: matplotlib.colors.Colormap,
    limite: float,
    mostrar_y: bool,
):
    im = ax.imshow(
        np.ma.masked_invalid(imagen),
        cmap=cmap,
        vmin=-limite,
        vmax=limite,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(titulo, fontsize=12)
    ax.set_xticks(np.arange(len(etiquetas)), etiquetas, rotation=90, fontsize=6.5)
    ax.set_yticks(np.arange(len(etiquetas)), etiquetas if mostrar_y else [], fontsize=7)
    ax.tick_params(length=0)
    return im


def dibujar_matriz_mes(mes: str, matrices: dict, destino: Path) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(22, 17), layout="constrained")
    etiquetas = nombres_cortos()
    cmap = paleta()
    for i, metrica in enumerate(METRICAS):
        for j, contexto in enumerate(CONTEXTOS):
            datos = matrices[mes][contexto][metrica].to_numpy(dtype=float)
            im = _eje_matriz(
                ejes[i, j],
                datos,
                f"{metrica.capitalize()} · "
                f"{'Train' if contexto == 'train' else 'Día completo'}",
                etiquetas,
                cmap,
                1.0,
                mostrar_y=j == 0,
            )
    fig.colorbar(im, ax=ejes, shrink=0.55, pad=0.015, label="Correlación entre covariables")
    fig.suptitle(f"Matrices X–X · {mes} · escala fija [−1, +1] · 26 variables", fontsize=18)
    fig.savefig(destino, dpi=145)
    plt.close(fig)


def limite_cambios(matrices: dict) -> float:
    maximo = 0.0
    for mes in MESES:
        for metrica in METRICAS:
            tren = matrices[mes]["train"][metrica].abs()
            dia = matrices[mes]["dia_completo"][metrica].abs()
            delta = (dia - tren).to_numpy(dtype=float)
            for i in range(len(VARIABLES)):
                delta[i, i] = np.nan
            finitos = np.abs(delta[np.isfinite(delta)])
            if len(finitos):
                maximo = max(maximo, float(finitos.max()))
    return min(1.0, max(0.30, np.ceil(maximo * 10) / 10))


def dibujar_diferencia_mes(mes: str, matrices: dict, limite: float, destino: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(22, 10.8), layout="constrained")
    cmap = paleta()
    for ax, metrica in zip(ejes, METRICAS):
        train = matrices[mes]["train"][metrica].abs()
        dia = matrices[mes]["dia_completo"][metrica].abs()
        delta = (dia - train).to_numpy(dtype=float)
        np.fill_diagonal(delta, np.nan)
        im = _eje_matriz(
            ax,
            delta,
            f"Cambio en magnitud de {metrica}",
            nombres_cortos(),
            cmap,
            limite,
            mostrar_y=ax is ejes[0],
        )
    fig.colorbar(im, ax=ejes, shrink=0.65, pad=0.015, label="|Día completo| − |train|")
    fig.suptitle(
        f"Efecto del filtrado en las parejas X–X · {mes} · azul: menor intensidad; " "rojo: mayor",
        fontsize=15,
    )
    fig.savefig(destino, dpi=145)
    plt.close(fig)


def _raster_parejas(
    sub: pd.DataFrame, por_mes: pd.DataFrame, metrica: str, contexto: str
) -> np.ndarray:
    indexed = por_mes.set_index(["variable_a", "variable_b", "mes"]).sort_index()
    a = np.full((len(sub), len(MESES)), np.nan)
    for i, row in enumerate(sub.itertuples(index=False)):
        for j, mes in enumerate(MESES):
            a[i, j] = indexed.loc[(row.variable_a, row.variable_b, mes), f"{metrica}_{contexto}"]
    return a


def dibujar_calor_prioritarias(
    resumen: pd.DataFrame, mensual: pd.DataFrame, destino: Path, limite: float
) -> list[dict]:
    seleccion = resumen.loc[resumen.mostrar_figura_individual].copy()
    registros = []
    if seleccion.empty:
        return registros
    # Lotes de tamaño visual legible: ninguna pareja priorizada se elimina.
    lotes = [seleccion.iloc[inicio : inicio + 32] for inicio in range(0, len(seleccion), 32)]
    for k, lote in enumerate(lotes, start=1):
        nombres = [
            etiqueta_par(row.variable_a, row.variable_b)
            + (f"  [{row.criterios_activados}]" if row.criterios_activados else "  [contraste]")
            for row in lote.itertuples(index=False)
        ]
        altura = max(7, 0.34 * len(lote) + 2.7)
        for metrica in METRICAS:
            fig, ejes = plt.subplots(1, 2, figsize=(18, altura), layout="constrained")
            for j, contexto in enumerate(CONTEXTOS):
                val = _raster_parejas(lote, mensual, metrica, contexto)
                im = ejes[j].imshow(
                    np.ma.masked_invalid(val),
                    cmap=paleta(),
                    vmin=-1,
                    vmax=1,
                    aspect="auto",
                    interpolation="nearest",
                )
                ejes[j].set_title("Train" if contexto == "train" else "Día completo")
                ejes[j].set_xticks(range(7), ETIQUETAS_MESES)
                ejes[j].set_yticks(range(len(lote)), nombres if j == 0 else [])
                ejes[j].tick_params(axis="y", labelsize=8, length=0)
                ejes[j].tick_params(axis="x", length=0)
            fig.colorbar(im, ax=ejes, shrink=0.72, label=f"{metrica.capitalize()} X–X")
            fig.suptitle(
                f"Evolución mensual de {metrica}: parejas priorizadas " f"(grupo {k}/{len(lotes)})",
                fontsize=14,
            )
            f = destino / f"prioritarias_{metrica}_grupo_{k:02d}.png"
            fig.savefig(f, dpi=145)
            plt.close(fig)
            registros.append(
                {"archivo": str(f), "descripcion": "Heatmap mensual de parejas", "pareja": ""}
            )

        fig, ejes = plt.subplots(1, 2, figsize=(18, altura), layout="constrained")
        for j, metrica in enumerate(METRICAS):
            dia = _raster_parejas(lote, mensual, metrica, "dia_completo")
            train = _raster_parejas(lote, mensual, metrica, "train")
            im = ejes[j].imshow(
                np.ma.masked_invalid(np.abs(dia) - np.abs(train)),
                cmap=paleta(),
                vmin=-limite,
                vmax=limite,
                aspect="auto",
                interpolation="nearest",
            )
            ejes[j].set_title(f"Cambio de magnitud de {metrica}")
            ejes[j].set_xticks(range(7), ETIQUETAS_MESES)
            ejes[j].set_yticks(range(len(lote)), nombres if j == 0 else [])
            ejes[j].tick_params(axis="y", labelsize=8, length=0)
            ejes[j].tick_params(axis="x", length=0)
        fig.colorbar(im, ax=ejes, shrink=0.72, label="|Día completo| − |train|")
        fig.suptitle(
            f"Efecto mensual del filtrado · parejas priorizadas " f"(grupo {k}/{len(lotes)})",
            fontsize=14,
        )
        f = destino / f"prioritarias_cambio_magnitud_grupo_{k:02d}.png"
        fig.savefig(f, dpi=145)
        plt.close(fig)
        registros.append(
            {"archivo": str(f), "descripcion": "Cambio mensual de intensidad", "pareja": ""}
        )
    return registros


def dibujar_trayectorias(
    resumen: pd.DataFrame, mensual: pd.DataFrame, directorio: Path
) -> list[dict]:
    directorio.mkdir(exist_ok=True)
    registros = []
    index = mensual.set_index(["variable_a", "variable_b", "mes"]).sort_index()
    x = np.arange(len(MESES))
    for item in resumen.loc[resumen.mostrar_figura_individual].itertuples(index=False):
        sub = index.loc[(item.variable_a, item.variable_b)].loc[list(MESES)]
        fig, ejes = plt.subplots(2, 1, figsize=(10.8, 7), sharex=True, layout="constrained")
        for ax, metrica in zip(ejes, METRICAS):
            for contexto, color, estilo, etiqueta in (
                ("train", "#4c78a8", "-", "Train"),
                ("dia_completo", "#e4a043", "--", "Día completo"),
            ):
                ax.plot(
                    x,
                    sub[f"{metrica}_{contexto}"],
                    marker="o",
                    linestyle=estilo,
                    color=color,
                    linewidth=2,
                    markersize=5,
                    label=etiqueta,
                )
            ax.axhline(0, color="black", alpha=0.5, linewidth=0.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(metrica.capitalize())
            ax.set_yticks([-0.9, -0.5, 0, 0.5, 0.9])
            ax.grid(alpha=0.22)
            ax.legend(loc="lower left", ncol=2, fontsize=9)
        ejes[-1].set_xticks(x, ETIQUETAS_MESES)
        rango = f"n diurno por mes: {int(sub.n_pares_dia_completo.min())}–{int(sub.n_pares_dia_completo.max())}"
        marcas = item.criterios_activados if item.criterios_activados else "contraste predefinido"
        fig.suptitle(
            f"{etiqueta_par(item.variable_a, item.variable_b)} · criterios: {marcas}\n"
            f"{rango}; valores de enero–julio de 2024",
            fontsize=13,
        )
        file_name = (
            f"{item.orden_pareja:03d}_"
            + re.sub(r"[^A-Za-z0-9_]+", "_", item.variable_a + "_" + item.variable_b)
            + ".png"
        )
        salida = directorio / file_name
        fig.savefig(salida, dpi=155)
        plt.close(fig)
        registros.append(
            {
                "archivo": str(salida),
                "descripcion": "Evolución mensual por pareja",
                "pareja": f"{item.variable_a} | {item.variable_b}",
            }
        )
    return registros


def _crear_directorio() -> Path:
    nombre = "EDA_multivariante_5_2_evolucion_mensual_xx_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    raiz = Path.cwd() / nombre
    numero = 2
    while raiz.exists() or raiz.with_suffix(".zip").exists():
        raiz = Path.cwd() / f"{nombre}_{numero}"
        numero += 1
    (raiz / "tablas").mkdir(parents=True)
    (raiz / "figuras" / "matrices_mensuales").mkdir(parents=True)
    (raiz / "figuras" / "cambios_mensuales").mkdir(parents=True)
    (raiz / "figuras" / "parejas_prioritarias").mkdir(parents=True)
    return raiz


def exportar(ruta: Path) -> Path:
    df, checkpoint_bytes, base_csv, miembro = cargar_entrada(ruta)
    muestras, disponibilidad = preparar_datos(df)
    base, origen_base = recuperar_51(base_csv, muestras)
    print(f"5.1 utilizado: {origen_base}")
    bruto, mensual, mats, empates = calcular_meses(muestras)
    criterios = marcar_parejas(base, mensual)
    if len(criterios) != 325:
        raise ValueError("La tabla de criterios debe contener exactamente 325 parejas.")
    n_ab = int((criterios.criterio_A | criterios.criterio_B).sum())
    print(
        f"Criterios A/B desde 5.1: A={criterios.criterio_A.sum()}, "
        f"B={criterios.criterio_B.sum()}, unión={n_ab}"
    )
    print(
        "Criterios mensuales: "
        + ", ".join(
            f"{letra}={int(criterios[f'criterio_{letra}'].sum())}" for letra in ("C", "D", "E")
        )
    )
    print(f"Gráficos individuales: {criterios.mostrar_figura_individual.sum()} parejas")
    panorama = resumen_mensual(mensual, disponibilidad)
    raiz = _crear_directorio()
    tablas = raiz / "tablas"
    bruto.to_csv(
        tablas / "01_correlaciones_325_parejas_por_mes_y_contexto.csv",
        index=False,
        float_format="%.12g",
    )
    mensual.to_csv(
        tablas / "02_comparacion_mensual_dia_menos_train.csv", index=False, float_format="%.12g"
    )
    criterios.to_csv(
        tablas / "03_resumen_estabilidad_y_criterios_325_parejas.csv",
        index=False,
        float_format="%.12g",
    )
    disponibilidad.to_csv(tablas / "04_disponibilidad_mensual_incluye_diciembre.csv", index=False)
    empates.to_csv(
        tablas / "05_inventario_empates_mensuales_26_variables.csv",
        index=False,
        float_format="%.12g",
    )
    panorama.to_csv(
        tablas / "06_resumen_mensual_todas_las_parejas.csv", index=False, float_format="%.12g"
    )
    base.to_csv(tablas / "07_referencia_global_5_1.csv", index=False, float_format="%.12g")

    limite = limite_cambios(mats)
    figuras = raiz / "figuras"
    index_fig = []

    def registrar(archivo: Path, descripcion: str, pareja: str = "") -> None:
        index_fig.append(
            {
                "archivo": archivo.relative_to(raiz).as_posix(),
                "descripcion": descripcion,
                "pareja": pareja,
            }
        )

    f = figuras / "00_disponibilidad_mensual.png"
    dibujar_disponibilidad(disponibilidad, f)
    registrar(f, "Horas de train y día completo por mes")
    f = figuras / "00_resumen_325_parejas.png"
    dibujar_resumen_mes(panorama, f)
    registrar(f, "Recuentos de parejas de correlación alta y de cambios de magnitud")

    for mes in MESES:
        f = figuras / "matrices_mensuales" / f"{mes}_matrices_train_y_dia.png"
        dibujar_matriz_mes(mes, mats, f)
        registrar(f, "Matrices 26×26 de Pearson y Spearman en ambos contextos")
        f = figuras / "cambios_mensuales" / f"{mes}_cambio_de_magnitud.png"
        dibujar_diferencia_mes(mes, mats, limite, f)
        registrar(f, "Cambios absolutos de magnitud en las matrices X–X")
    for registro in dibujar_calor_prioritarias(
        criterios, mensual, figuras / "parejas_prioritarias", limite
    ):
        registrar(Path(registro["archivo"]), registro["descripcion"], registro["pareja"])
    for registro in dibujar_trayectorias(criterios, mensual, figuras / "trayectorias_individuales"):
        registrar(Path(registro["archivo"]), registro["descripcion"], registro["pareja"])
    pd.DataFrame(index_fig).to_csv(tablas / "08_indice_figuras.csv", index=False)

    metadata = {
        "subseccion": "5.2",
        "version_script": VERSION,
        "fuente_zip": ruta.name,
        "miembro_checkpoint_entrada": miembro,
        "checkpoint_sin_modificar": True,
        "sha256_checkpoint": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "origen_global_5_1": origen_base,
        "meses_incluidos": list(MESES),
        "diciembre": "excluido; solo 28 horas de día completo",
        "n_covariables": len(VARIABLES),
        "n_parejas": len(PAREJAS),
        "n_horas_train_enero_julio": 5112,
        "n_horas_dia_completo_enero_julio": 2560,
        "n_filas_metricas": len(bruto),
        "n_filas_comparacion": len(mensual),
        "n_A": int(criterios.criterio_A.sum()),
        "n_B": int(criterios.criterio_B.sum()),
        "n_union_AB": n_ab,
        "n_C": int(criterios.criterio_C.sum()),
        "n_D": int(criterios.criterio_D.sum()),
        "n_E": int(criterios.criterio_E.sum()),
        "n_parejas_priorizadas": int(criterios.prioritaria.sum()),
        "n_trayectorias_individuales": int(criterios.mostrar_figura_individual.sum()),
        "criterios": {
            "A": "max(abs(Pearson_dia_global), abs(Spearman_dia_global)) >= 0.90",
            "B": "alguna abs(abs(coef_dia_global)-abs(coef_train_global)) >= 0.30",
            "C": "mismo coeficiente en día >= +0.70 al menos 4 meses o <= -0.70 "
            "al menos 4 meses, sin sentido opuesto >= 0.30 en otros meses",
            "D": "mismo coeficiente en día >= +0.30 al menos 2 meses y <= -0.30 "
            "al menos 2 meses",
            "E": "cambio en magnitud día-train >= +0.30 al menos 2 meses o <= -0.30 "
            "al menos 2 meses, en la misma medida",
        },
        "avisos": [
            "Las 325 parejas se calculan y exportan independientemente de los criterios.",
            "Pearson y Spearman mensuales son asociaciones contemporáneas X-X.",
            "La diferencia de coeficientes firmados no equivale al cambio de su magnitud.",
            "El día completo excluye noches y horas de transición conjuntamente.",
            "No se hacen imputaciones ni pruebas de significación.",
            "Siete meses de un solo año no prueban una estacionalidad recurrente.",
            "Una alta correlación X-X no demuestra redundancia predictiva.",
            "DIF mínimo, máximo y std pertenecen a la misma magnitud y hora del objetivo.",
        ],
    }
    (raiz / "README.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    resultado = raiz.with_suffix(".zip")
    with zipfile.ZipFile(
        resultado, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as zf:
        for archivo in sorted(raiz.rglob("*")):
            if archivo.is_file():
                zf.write(archivo, archivo.relative_to(raiz).as_posix())
        zf.writestr(f"checkpoint/{CHECKPOINT}", checkpoint_bytes)
    print(f"Generado {resultado} ({len(index_fig)} figuras y 7 tablas + índice).")
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-entrada", help="Un único ZIP de 5.1 o 4.2.12")
    parser.add_argument("--sin-descarga", action="store_true", help="Evita la descarga en Colab")
    args = parser.parse_args()
    print(f"Script: {VERSION}")
    salida = exportar(obtener_zip(args.zip_entrada))
    if not args.sin_descarga:
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(str(salida))


if __name__ == "__main__":
    main()
