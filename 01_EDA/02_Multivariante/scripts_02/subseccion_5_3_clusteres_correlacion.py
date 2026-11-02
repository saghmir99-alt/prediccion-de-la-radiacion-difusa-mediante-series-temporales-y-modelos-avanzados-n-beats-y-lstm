#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM · Subsección 5.3 · Clústeres de correlación entre covariables.

Entrada: UN SOLO ZIP de la subsección 5.2. Reutiliza sus tablas de correlación
global (5.1) y mensual (5.2), calculadas exclusivamente con train.

En Google Colab:
    1. Sube este archivo .py a /content.
    2. Ejecuta: %run subseccion_5_3_clusteres_correlacion.py
    3. Si se solicita, selecciona SOLO el ZIP de 5.2.

Si el ZIP ya está en /content, se localizará sin solicitar otra subida.
También puedes especificarlo explícitamente:
    %run subseccion_5_3_clusteres_correlacion.py --zip /content/mi_5_2.zip

El programa entrega un ZIP con tres o cuatro figuras, tablas y un README.
El árbol es el resultado principal. Se muestran dos alturas de referencia
(|coeficiente| de 0,90 y 0,80), sin imponer un número de familias físicas.
"""

from __future__ import annotations

import argparse
import io
import json
from itertools import combinations
from pathlib import Path
import zipfile
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

REQUERIDOS = {
    "global": "tablas/07_referencia_global_5_1.csv",
    "mensual": "tablas/01_correlaciones_325_parejas_por_mes_y_contexto.csv",
    "disponibilidad": "tablas/04_disponibilidad_mensual_incluye_diciembre.csv",
}
ORDEN_FAMILIAS = {
    familia: i
    for i, familia in enumerate(("SWD", "DIR", "DIF", "LWD", "SWU", "LWU", "T2", "RH", "PoPoPoPo"))
}
ORDEN_ESTADISTICOS = {"media_h": 0, "min_h": 1, "max_h": 2, "std_media_h": 3}
DERIVADAS_DIF = {"DIF__min_h", "DIF__max_h", "DIF__std_media_h"}
METRICAS = ("pearson", "spearman")


def par(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def etiqueta(variable: str) -> str:
    familia, _, estadistico = variable.partition("__")
    nombre = {"media_h": "media", "min_h": "mín", "max_h": "máx", "std_media_h": "std"}.get(
        estadistico, estadistico
    )
    return f"{familia} {nombre}"


def miembro(zip_file: zipfile.ZipFile, sufijo: str) -> str | None:
    coincidencias = [n for n in zip_file.namelist() if n.endswith(sufijo)]
    return coincidencias[0] if len(coincidencias) == 1 else None


def zip_valido(path: Path) -> bool:
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return all(miembro(z, n) for n in REQUERIDOS.values())
    except (OSError, zipfile.BadZipFile):
        return False


def encontrar_zip(explicito: str | None) -> Path:
    if explicito:
        path = Path(explicito).expanduser().resolve()
        if not zip_valido(path):
            raise ValueError(
                "El archivo indicado no es el ZIP de 5.2: deben estar las tablas "
                "01_correlaciones_325_parejas_por_mes_y_contexto.csv y "
                "07_referencia_global_5_1.csv, entre otras."
            )
        return path
    directorios = [Path.cwd(), Path(__file__).resolve().parent, Path("/content")]
    candidatos: dict[Path, None] = {}
    for directorio in directorios:
        if directorio.is_dir():
            for path in directorio.glob("*.zip"):
                if zip_valido(path):
                    candidatos[path.resolve()] = None
    if candidatos:
        # Si existen varios, dar preferencia a ZIP cuyo nombre declare 5.2.
        return max(
            candidatos, key=lambda p: ("_5_2_" in p.name or "_52_" in p.name, p.stat().st_mtime)
        )
    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(
            "No se encontró el ZIP de 5.2. Ejecuta con --zip RUTA_AL_ZIP."
        ) from exc
    print("Selecciona UN SOLO archivo: el ZIP de resultados de la subsección 5.2.")
    subidos = files.upload()
    validos = [Path(nombre).resolve() for nombre in subidos if zip_valido(Path(nombre))]
    if len(validos) != 1:
        raise ValueError(
            "Se esperaba exactamente un ZIP válido de 5.2. "
            "Comprueba la selección o usa --zip RUTA_AL_ZIP."
        )
    return validos[0]


def cargar_entrada(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as z:
        tablas = {
            clave: pd.read_csv(io.BytesIO(z.read(miembro(z, nombre))))
            for clave, nombre in REQUERIDOS.items()
        }
    globales = tablas["global"].copy()
    mensuales = tablas["mensual"].copy()
    disponibilidad = tablas["disponibilidad"].copy()

    columnas_globales = {
        "variable_a",
        "variable_b",
        "pearson_dia_completo",
        "spearman_dia_completo",
        "pearson_train",
        "spearman_train",
        "n_pares_dia_completo",
        "n_pares_train",
    }
    columnas_mensuales = {
        "mes",
        "contexto",
        "variable_a",
        "variable_b",
        "n_pares",
        "pearson",
        "spearman",
    }
    if not columnas_globales.issubset(globales) or not columnas_mensuales.issubset(mensuales):
        raise ValueError("Faltan columnas necesarias en las tablas del ZIP de 5.2.")
    variables = set(globales.variable_a) | set(globales.variable_b)
    pares = [par(a, b) for a, b in zip(globales.variable_a, globales.variable_b)]
    if len(variables) != 26 or len(globales) != 325 or len(set(pares)) != 325:
        raise ValueError("La tabla global debe contener 26 variables y 325 parejas únicas.")
    if any(a == b for a, b in pares):
        raise ValueError("La tabla global contiene una pareja formada por la misma variable.")
    for metrica in METRICAS:
        for contexto in ("train", "dia_completo"):
            valores = pd.to_numeric(globales[f"{metrica}_{contexto}"], errors="coerce")
            if (~np.isfinite(valores) | (valores.abs() > 1.000001)).any():
                raise ValueError(f"Correlaciones inválidas en {metrica}_{contexto}.")
    if (globales.n_pares_dia_completo < 3).any():
        raise ValueError("Hay alguna pareja con menos de tres horas diurnas válidas.")
    dia = mensuales.loc[mensuales.contexto.eq("dia_completo")].copy()
    if len(dia) != 325 * 7 or dia.mes.nunique() != 7:
        raise ValueError(
            "La tabla mensual debe contener siete meses y 325 parejas diurnas por mes."
        )
    dia["clave_par"] = [par(a, b) for a, b in zip(dia.variable_a, dia.variable_b)]
    if dia.duplicated(["mes", "clave_par"]).any():
        raise ValueError("La tabla mensual contiene parejas repetidas dentro de un mes.")
    if set(dia.clave_par) != set(pares):
        raise ValueError("La tabla mensual y la global no contienen las mismas 325 parejas.")
    for metrica in METRICAS:
        valores = pd.to_numeric(dia[metrica], errors="coerce")
        if (~np.isfinite(valores) | (valores.abs() > 1.000001)).any():
            raise ValueError(f"Correlaciones mensuales inválidas para {metrica}.")
    if not {"mes", "n_dia_completo"}.issubset(disponibilidad):
        raise ValueError("Faltan mes o n_dia_completo en la disponibilidad mensual.")
    return globales, dia, disponibilidad


def variables_ordenadas(globales: pd.DataFrame) -> list[str]:
    def orden(v: str) -> tuple[int, int, str]:
        familia, _, estadistico = v.partition("__")
        return (ORDEN_FAMILIAS.get(familia, 99), ORDEN_ESTADISTICOS.get(estadistico, 99), v)

    return sorted(set(globales.variable_a) | set(globales.variable_b), key=orden)


def matrices_correlacion(globales: pd.DataFrame, variables: list[str]) -> dict[str, np.ndarray]:
    n = len(variables)
    indice = {v: i for i, v in enumerate(variables)}
    salida = {metrica: np.eye(n, dtype=float) for metrica in METRICAS}
    for fila in globales.itertuples(index=False):
        i, j = indice[fila.variable_a], indice[fila.variable_b]
        for metrica, matriz in salida.items():
            valor = float(getattr(fila, metrica + "_dia_completo"))
            matriz[i, j] = matriz[j, i] = valor
    return salida


def ajustar_arbol(coeficientes: np.ndarray, metrica: str) -> dict:
    distancia = np.clip(1.0 - np.abs(coeficientes), 0.0, 1.0)
    np.fill_diagonal(distancia, 0.0)
    # Enlace completo: un grupo compacto requiere relaciones fuertes entre
    # todas sus parejas. No usar Ward con distancias de correlación precomputadas.
    arbol = linkage(squareform(distancia, checks=True), method="complete", optimal_ordering=True)
    filas = []
    for k in range(2, min(16, len(coeficientes) - 2) + 1):
        etiquetas = cut_tree(arbol, n_clusters=k).ravel()
        tamanos = np.bincount(etiquetas)
        filas.append(
            {
                "metrica": metrica,
                "k": k,
                "silueta": float(silhouette_score(distancia, etiquetas, metric="precomputed")),
                "grupos_unitarios": int((tamanos == 1).sum()),
                "mayor_grupo": int(tamanos.max()),
            }
        )
    diagnosticos = pd.DataFrame(filas)
    # Los identificadores siguen el orden del árbol, no el de la familia física.
    hojas = dendrogram(arbol, no_plot=True)["leaves"]
    niveles = {}
    for distancia_corte, nombre in ((0.10, "090"), (0.20, "080")):
        enteros = fcluster(arbol, t=distancia_corte, criterion="distance")
        orden_ids = list(dict.fromkeys(int(enteros[i]) for i in hojas))
        prefijo = "S" if metrica == "spearman" else "P"
        ids = {g: f"{prefijo}{nombre}_{pos:02d}" for pos, g in enumerate(orden_ids, start=1)}
        niveles[nombre] = [ids[int(g)] for g in enteros]
    return {
        "metrica": metrica,
        "coeficientes": coeficientes,
        "distancia": distancia,
        "arbol": arbol,
        "diagnosticos": diagnosticos,
        "grupos_090": niveles["090"],
        "grupos_080": niveles["080"],
        "hojas": hojas,
    }


def lookup_mensual(mensual: pd.DataFrame) -> dict:
    resultado = {}
    for f in mensual.itertuples(index=False):
        resultado[(f.mes, f.clave_par)] = {
            "pearson": float(f.pearson),
            "spearman": float(f.spearman),
            "n_pares": int(f.n_pares),
        }
    return resultado


def tablas_grupos(
    globales: pd.DataFrame,
    mensual: pd.DataFrame,
    variables: list[str],
    ajustes: dict[str, dict],
    disponibilidad: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meses = sorted(mensual.mes.unique())
    indices = {v: i for i, v in enumerate(variables)}
    globales_por_par = {
        par(f.variable_a, f.variable_b): f for f in globales.itertuples(index=False)
    }
    mensuales_por_par = lookup_mensual(mensual)
    n_horas_mes = dict(zip(disponibilidad.mes, disponibilidad.n_dia_completo))
    filas_variables = []
    for v in variables:
        i = indices[v]
        filas_variables.append(
            {
                "variable": v,
                "familia": v.partition("__")[0],
                "derivada_de_DIF_misma_hora": v in DERIVADAS_DIF,
                "nucleo_090_spearman": ajustes["spearman"]["grupos_090"][i],
                "nucleo_090_pearson": ajustes["pearson"]["grupos_090"][i],
                "grupo_referencia_080_spearman": ajustes["spearman"]["grupos_080"][i],
                "grupo_referencia_080_pearson": ajustes["pearson"]["grupos_080"][i],
            }
        )
    filas_grupos, filas_meses, filas_pares = [], [], []
    parejas_ya_guardadas = set()
    for metrica, ajuste in ajustes.items():
        asignacion = ajuste["grupos_090"]
        for gid in list(dict.fromkeys(asignacion[i] for i in ajuste["hojas"])):
            miembros = [v for v, g in zip(variables, asignacion) if g == gid]
            pares_grupo = list(combinations(miembros, 2))
            coef = [
                getattr(globales_por_par[par(a, b)], metrica + "_dia_completo")
                for a, b in pares_grupo
            ]
            minimo_mensual = []
            for mes in meses:
                pearson = [mensuales_por_par[(mes, par(a, b))]["pearson"] for a, b in pares_grupo]
                spearman = [mensuales_por_par[(mes, par(a, b))]["spearman"] for a, b in pares_grupo]
                minimo = min(
                    (abs(v) for v in (pearson if metrica == "pearson" else spearman)),
                    default=np.nan,
                )
                minimo_mensual.append(minimo)
                if pares_grupo:
                    filas_meses.append(
                        {
                            "metrica_grupo": metrica,
                            "grupo": gid,
                            "mes": mes,
                            "n_variables": len(miembros),
                            "n_parejas": len(pares_grupo),
                            "n_horas_dia_completo": int(n_horas_mes[mes]),
                            "min_abs_pearson": min(abs(v) for v in pearson),
                            "mediana_abs_pearson": float(np.median(np.abs(pearson))),
                            "min_abs_spearman": min(abs(v) for v in spearman),
                            "mediana_abs_spearman": float(np.median(np.abs(spearman))),
                        }
                    )
            filas_grupos.append(
                {
                    "metrica_grupo": metrica,
                    "grupo": gid,
                    "n_variables": len(miembros),
                    "variables": " | ".join(miembros),
                    "familias": " | ".join(dict.fromkeys(v.partition("__")[0] for v in miembros)),
                    "n_parejas": len(pares_grupo),
                    "min_abs_correlacion_global": min(map(abs, coef)) if coef else np.nan,
                    "mediana_abs_correlacion_global": (
                        float(np.median(np.abs(coef))) if coef else np.nan
                    ),
                    "n_parejas_negativas": sum(v < 0 for v in coef),
                    "n_meses_con_todas_las_parejas_abs_ge_070": sum(
                        v >= 0.70 for v in minimo_mensual
                    ),
                    "contiene_descriptor_DIF_misma_hora": any(v in DERIVADAS_DIF for v in miembros),
                }
            )
            for a, b in pares_grupo:
                clave = par(a, b)
                if clave in parejas_ya_guardadas:
                    continue
                parejas_ya_guardadas.add(clave)
                f = globales_por_par[clave]
                valores_r = [mensuales_por_par[(mes, clave)]["pearson"] for mes in meses]
                valores_rho = [mensuales_por_par[(mes, clave)]["spearman"] for mes in meses]
                filas_pares.append(
                    {
                        "variable_a": a,
                        "variable_b": b,
                        "misma_familia": a.partition("__")[0] == b.partition("__")[0],
                        "nucleo_090_pearson": (
                            ajustes["pearson"]["grupos_090"][indices[a]]
                            if ajustes["pearson"]["grupos_090"][indices[a]]
                            == ajustes["pearson"]["grupos_090"][indices[b]]
                            else ""
                        ),
                        "nucleo_090_spearman": (
                            ajustes["spearman"]["grupos_090"][indices[a]]
                            if ajustes["spearman"]["grupos_090"][indices[a]]
                            == ajustes["spearman"]["grupos_090"][indices[b]]
                            else ""
                        ),
                        "pearson_dia_completo": f.pearson_dia_completo,
                        "spearman_dia_completo": f.spearman_dia_completo,
                        "n_meses_abs_pearson_ge_070": sum(abs(v) >= 0.70 for v in valores_r),
                        "n_meses_abs_spearman_ge_070": sum(abs(v) >= 0.70 for v in valores_rho),
                        "min_abs_pearson_mensual": min(map(abs, valores_r)),
                        "min_abs_spearman_mensual": min(map(abs, valores_rho)),
                    }
                )
    return (
        pd.DataFrame(filas_variables),
        pd.DataFrame(filas_grupos),
        pd.DataFrame(filas_meses),
        pd.DataFrame(filas_pares),
    )


def figura_arbol_matriz(ajuste: dict, variables: list[str], path: Path) -> None:
    metrica = ajuste["metrica"]
    titulo = "Spearman" if metrica == "spearman" else "Pearson"
    n = len(variables)
    fig = plt.figure(figsize=(19, 12.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.3], wspace=0.035)
    ax_arbol = fig.add_subplot(gs[0, 0])
    ax_matriz = fig.add_subplot(gs[0, 1])
    dibujo = dendrogram(
        ajuste["arbol"],
        orientation="left",
        labels=[etiqueta(v) for v in variables],
        color_threshold=0,
        above_threshold_color="#34495e",
        leaf_font_size=10,
        ax=ax_arbol,
    )
    orden = dibujo["leaves"]  # Primera hoja abajo en orientación izquierda.
    orden_x = list(reversed(orden))  # Diagonal desde arriba a la izquierda.
    ordenadas_x = [variables[i] for i in orden_x]
    ordenadas_y = [variables[i] for i in orden]
    c = ajuste["coeficientes"][np.ix_(orden, orden_x)]
    im = ax_matriz.imshow(
        c,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=(0, n, 0, 10 * n),
    )
    ax_matriz.set_xticks(
        np.arange(n) + 0.5, [etiqueta(v) for v in ordenadas_x], rotation=90, fontsize=8.5
    )
    ax_matriz.set_yticks(np.arange(n) * 10 + 5, [etiqueta(v) for v in ordenadas_y], fontsize=8.5)
    ax_matriz.yaxis.tick_right()
    ax_matriz.set_xlim(0, n)
    ax_matriz.set_ylim(0, 10 * n)
    ax_arbol.axvline(0.10, color="#8b6bb3", lw=1, ls="--", label="Referencia |coef.| = 0,90")
    ax_arbol.legend(fontsize=8, loc="upper left")
    ax_arbol.set_xlabel("Distancia = 1 − |correlación|")
    ax_arbol.set_ylabel("26 covariables")
    ax_matriz.set_title(f"Correlaciones {titulo} con signo; orden del árbol")
    fig.colorbar(im, ax=ax_matriz, shrink=0.62, pad=0.025, label=f"{titulo} (−1 a +1)")
    fig.suptitle(
        f"5.3 · Clústeres X–X durante día completo · {titulo}\n"
        "Enlace completo · árbol completo · 2.588 horas de train",
        fontsize=15,
    )
    fig.savefig(path, dpi=145)
    plt.close(fig)


def figura_particiones(ajustes: dict[str, dict], path: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.7), layout="constrained")
    for metrica, color in (("spearman", "#624785"), ("pearson", "#135995")):
        d = ajustes[metrica]["diagnosticos"]
        label = metrica.capitalize()
        ejes[0].plot(d.k, d.silueta, "-o", color=color, label=label)
        ejes[1].plot(d.k, d.grupos_unitarios, "-o", color=color, label=label)
    ejes[0].set_ylabel("Silueta media; mayor es mejor para esta distancia")
    ejes[1].set_ylabel("Número de grupos de una sola variable")
    for ax in ejes:
        ax.set_xlabel("Número de grupos K (exploratorio)")
        ax.set_xticks(ajustes["spearman"]["diagnosticos"].k)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        "Diagnóstico de particiones: silueta y fragmentación\n"
        "Una silueta creciente puede acompañar a grupos unitarios; no se fija K",
        fontsize=14,
    )
    fig.savefig(path, dpi=155)
    plt.close(fig)


def figura_estabilidad(grupos: pd.DataFrame, evolucion: pd.DataFrame, path: Path) -> bool:
    base = grupos.loc[(grupos.metrica_grupo == "spearman") & (grupos.n_variables > 1)].copy()
    if base.empty:
        return False
    ids = list(base.grupo)
    meses = sorted(evolucion.mes.unique())
    matriz = np.array(
        [
            [
                float(
                    evolucion.loc[
                        (evolucion.metrica_grupo == "spearman")
                        & (evolucion.grupo == grupo)
                        & (evolucion.mes == mes),
                        "min_abs_spearman",
                    ].iloc[0]
                )
                for mes in meses
            ]
            for grupo in ids
        ]
    )
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.65 * len(ids) + 2.7)), layout="constrained")
    im = ax.imshow(matriz, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(meses)), meses, rotation=25)
    ax.set_yticks(
        range(len(ids)),
        [f"{row.grupo} · {row.n_variables} variables" for row in base.itertuples(index=False)],
    )
    for i in range(len(ids)):
        for j in range(len(meses)):
            val = matriz[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val < 0.52 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mínimo |Spearman| entre parejas del grupo")
    ax.set_title(
        "Mes a mes: relación menos intensa dentro de cada núcleo Spearman (0,90)\n"
        "Las celdas indican la peor pareja, no la mediana ni un valor predictivo."
    )
    fig.savefig(path, dpi=155)
    plt.close(fig)
    return True


def directorio_libre() -> Path:
    base = "EDA_multivariante_F5_S53_clusteres_correlacion_" + (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    for i in range(1, 100):
        p = Path.cwd() / (base if i == 1 else f"{base}_{i}")
        if not p.exists() and not p.with_suffix(".zip").exists():
            p.mkdir()
            (p / "figuras").mkdir()
            (p / "tablas").mkdir()
            return p
    raise RuntimeError("No se pudo elegir un nombre libre para la salida.")


def crear_zip(carpeta: Path) -> Path:
    salida = carpeta.with_suffix(".zip")
    with zipfile.ZipFile(salida, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(carpeta.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(carpeta))
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clústeres X–X de 26 covariables; usa UN ZIP de 5.2."
    )
    parser.add_argument("--zip", dest="zip_entrada", help="Ruta al ZIP de resultados de 5.2.")
    parser.add_argument(
        "--no-download", action="store_true", help="No iniciar la descarga automática de Colab."
    )
    args = parser.parse_args()
    zip_entrada = encontrar_zip(args.zip_entrada)
    print("ZIP de entrada:", zip_entrada.name)
    globales, mensual, disponibilidad = cargar_entrada(zip_entrada)
    variables = variables_ordenadas(globales)
    matrices = matrices_correlacion(globales, variables)
    ajustes = {m: ajustar_arbol(matrices[m], m) for m in METRICAS}

    carpeta = directorio_libre()
    figuras, tablas = carpeta / "figuras", carpeta / "tablas"
    diagnosticos = pd.concat([ajustes[m]["diagnosticos"] for m in METRICAS], ignore_index=True)
    diagnosticos.to_csv(tablas / "01_diagnostico_particiones.csv", index=False)
    pertenencia, grupos, evolucion, parejas = tablas_grupos(
        globales, mensual, variables, ajustes, disponibilidad
    )
    pertenencia.to_csv(tablas / "02_pertenencia_26_variables.csv", index=False)
    grupos.to_csv(tablas / "03_resumen_nucleos_090.csv", index=False)
    evolucion.to_csv(tablas / "04_estabilidad_mensual_nucleos_090.csv", index=False)
    parejas.to_csv(tablas / "05_parejas_en_algun_nucleo_090.csv", index=False)

    figura_arbol_matriz(ajustes["spearman"], variables, figuras / "01_arbol_matriz_spearman.png")
    figura_arbol_matriz(ajustes["pearson"], variables, figuras / "02_arbol_matriz_pearson.png")
    figura_particiones(ajustes, figuras / "03_silueta_y_grupos_unitarios.png")
    hay_estabilidad = figura_estabilidad(
        grupos, evolucion, figuras / "04_estabilidad_mensual_grupos.png"
    )

    metadata = {
        "subseccion": "5.3",
        "zip_entrada": zip_entrada.name,
        "unica_entrada": True,
        "fuente_global": REQUERIDOS["global"],
        "fuente_mensual": REQUERIDOS["mensual"],
        "muestra": "train; fraccion_minutos_dia == 1",
        "n_variables": len(variables),
        "n_parejas": len(globales),
        "n_horas_dia_completo": int(globales.n_pares_dia_completo.max()),
        "n_meses": int(mensual.mes.nunique()),
        "distancia": "1 - abs(coeficiente)",
        "enlace": "complete",
        "rango_K_evaluado": [2, 16],
        "cortes_de_referencia_no_obligatorios": {
            "nucleos_muy_correlacionados": "|coeficiente| >= 0.90",
            "bloques_mas_amplios": "|coeficiente| >= 0.80",
        },
        "n_grupos_por_corte": {
            m: {nivel: len(set(ajustes[m]["grupos_" + nivel])) for nivel in ("090", "080")}
            for m in METRICAS
        },
        "figura_estabilidad_mensual": hay_estabilidad,
    }
    (carpeta / "README.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (carpeta / "LEEME.md").write_text(
        "# 5.3 · Clústeres X–X\n\n"
        "Entrada única: ZIP de 5.2. Se reutilizan las correlaciones de las "
        "325 parejas del conjunto de entrenamiento; no se usa validation o test.\n\n"
        "El árbol agrupa COLUMNAS, no horas. La distancia es 1 menos el valor "
        "absoluto de Pearson o Spearman. Correlaciones de signo opuesto pueden "
        "quedar en un grupo; la matriz conserva los signos.\n\n"
        "La tabla 01 evalúa K=2..16 y muestra silueta y número de grupos "
        "unitarios. Ningún máximo de silueta se declara óptimo: el árbol "
        "completo es el resultado principal. La tabla 02 indica pertenencias "
        "en dos niveles de referencia: distancia 0,10 (|coef.| de 0,90) para "
        "núcleos muy estrechos y 0,20 (|coef.| de 0,80) para bloques más amplios. "
        "Estos niveles no asignan un número verdadero de familias físicas. "
        "La línea violeta en el árbol señala únicamente el nivel 0,10.\n\n"
        "La figura 04 muestra, por cada mes, la relación menos intensa entre "
        "las parejas de cada núcleo de Spearman en el nivel 0,90. Solo incluye grupos "
        "de dos o más variables. Las correlaciones X–X y sus clústeres no "
        "demuestran independencia, redundancia predictiva o causalidad.\n\n"
        "DIF mínimo, máximo y desviación pertenecen a la misma magnitud y "
        "ventana horaria que DIF media. Su uso para estimar DIF media de la "
        "misma hora requiere revisar fuga de información. No se eliminan "
        "variables ni se ajusta STL en esta subsección.\n",
        encoding="utf-8",
    )
    salida = crear_zip(carpeta)
    print("\nResultado:", salida)
    print("Núcleos en |coef.| >= 0,90 (referencia, no número definitivo de familias):")
    for m in ("spearman", "pearson"):
        x = grupos.loc[(grupos.metrica_grupo == m) & (grupos.n_variables > 1)]
        print(
            f"  {m.capitalize()}: {len(x)} núcleos con al menos dos variables; "
            f"{int(x.n_variables.sum())} variables incluidas"
        )
    print(
        f"Figuras: {3 + int(hay_estabilidad)}. Tablas: 5. "
        "Todos los datos proceden exclusivamente del ZIP indicado."
    )

    if not args.no_download:
        try:
            from google.colab import files

            files.download(str(salida))
        except ImportError:
            pass
        except Exception as exc:
            print("La descarga automática no se completó; ZIP disponible en:", salida)
            print("Detalle:", exc)


if __name__ == "__main__":
    main()
