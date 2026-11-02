#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.3: clustering de variables por perfiles horarios estandarizados.

Colab: %run subseccion_5_3_perfiles_horarios.py
Entrada: UN ZIP de 5.2, con su checkpoint horario.
Salida: dos figuras, dos CSV y una guía de lectura, dentro de un ZIP.
"""

from __future__ import annotations

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


def dependencias():
    paquetes = ["numpy", "pandas", "scipy", "matplotlib", "pyarrow"]
    faltan = [p for p in paquetes if importlib.util.find_spec(p) is None]
    if not faltan:
        return
    try:
        import google.colab  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Instala las dependencias con: python -m pip install " + " ".join(faltan)
        ) from None
    print("Instalando las dependencias que faltan:", ", ".join(faltan))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltan])
    importlib.invalidate_caches()


dependencias()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

VARIABLES = [
    f"{familia}__{estadistico}"
    for familia in ["SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]
    for estadistico in ["media_h", "min_h", "max_h", "std_media_h"]
    if (familia, estadistico) != ("DIF", "media_h")
]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
CHECKPOINT = "checkpoint/checkpoint_horario_entrada_intacto.parquet"
MARCA_52 = "tablas/01_correlaciones_325_parejas_por_mes_y_contexto.csv"


def etiqueta(v):
    f, s = v.split("__", 1)
    f = "Presión" if f == "PoPoPoPo" else f
    s = {"media_h": "media", "min_h": "mín", "max_h": "máx", "std_media_h": "std"}[s]
    return f"{f} {s}"


def miembro(z, sufijo):
    encontrados = [n for n in z.namelist() if n.endswith(sufijo)]
    if len(encontrados) != 1:
        raise ValueError(f"El ZIP debe contener exactamente un archivo {sufijo}.")
    return encontrados[0]


def zip_valido(path):
    try:
        with zipfile.ZipFile(path) as z:
            miembro(z, CHECKPOINT)
            miembro(z, MARCA_52)
        return True
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def entrada(explicito):
    if explicito:
        p = Path(explicito).expanduser().resolve()
        if not zip_valido(p):
            raise ValueError("Selecciona el ZIP de 5.2 con su checkpoint horario.")
        return p
    candidatos = set()
    for carpeta in {Path.cwd(), Path(__file__).resolve().parent, Path("/content")}:
        if carpeta.is_dir():
            candidatos.update(p.resolve() for p in carpeta.glob("*.zip") if zip_valido(p))
    if candidatos:
        elegido = max(candidatos, key=lambda p: p.stat().st_mtime)
        print("ZIP localizado:", elegido.name)
        return elegido
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError("Indica el ZIP con --zip RUTA_AL_ZIP_DE_5_2.") from None
    print("Selecciona UN SOLO ZIP: los resultados del apartado 5.2.")
    subidos = files.upload()
    if len(subidos) != 1:
        raise ValueError("Sube únicamente el ZIP de 5.2.")
    return entrada(next(iter(subidos)))


def cargar(path):
    with zipfile.ZipFile(path) as z:
        contenido = z.read(miembro(z, CHECKPOINT))
    columnas = ["particion_observacion", "fraccion_minutos_dia", *VARIABLES]
    df = pd.read_parquet(io.BytesIO(contenido), columns=columnas)
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("Se requiere el índice horario DatetimeIndex del checkpoint.")
    if df.index.has_duplicates:
        raise ValueError("Hay horas duplicadas en el checkpoint; revisa la entrada.")
    train = df.loc[df.particion_observacion.eq("train")].sort_index()
    fraccion = pd.to_numeric(train.fraccion_minutos_dia, errors="raise")
    if fraccion.isna().any() or not fraccion.between(0, 1).all():
        raise ValueError("fraccion_minutos_dia debe tener valores entre 0 y 1 en train.")
    dia = train.loc[fraccion.eq(1), VARIABLES].apply(pd.to_numeric, errors="raise")
    finitos = np.isfinite(dia.to_numpy(dtype=float)).all(axis=1)
    x = dia.loc[finitos].astype(float)
    if len(x) < 3:
        raise ValueError("No hay suficientes horas diurnas completas para comparar.")
    media = x.mean()
    escala = x.std(ddof=0)
    constantes = escala.index[(escala <= 0) | ~np.isfinite(escala)].tolist()
    if constantes:
        raise ValueError("No se pueden estandarizar variables constantes: " + str(constantes))
    z = (x - media) / escala
    excluidas = [
        {
            "timestamp": str(t),
            "variables_no_finitas": [v for v in VARIABLES if not np.isfinite(dia.loc[t, v])],
        }
        for t in dia.index[~finitos]
    ]
    info = dict(
        n_train=len(train),
        n_dia_completo=len(dia),
        n_horas_utilizadas=len(x),
        n_horas_excluidas=len(dia) - len(x),
        exclusiones=excluidas,
        inicio_horas_utilizadas=str(x.index.min()),
        fin_horas_utilizadas=str(x.index.max()),
        zona_horaria=str(x.index.tz),
        sha256_checkpoint=hashlib.sha256(contenido).hexdigest(),
    )
    return z, media, escala, info


def analizar(z):
    # Cada fila que recibe pdist es UNA VARIABLE, con todas sus horas alineadas.
    # cityblock suma diferencias absolutas; dividimos por las horas para obtener la media.
    condensada = pdist(z.to_numpy().T, metric="cityblock") / len(z)
    distancias = squareform(condensada)
    arbol = linkage(condensada, method="complete", optimal_ordering=True)
    orden = dendrogram(arbol, no_plot=True)["leaves"][::-1]  # lectura de arriba abajo
    filas = [
        dict(
            variable_a=VARIABLES[i],
            variable_b=VARIABLES[j],
            distancia_media=float(distancias[i, j]),
        )
        for i in range(len(VARIABLES))
        for j in range(i + 1, len(VARIABLES))
    ]
    parejas = (
        pd.DataFrame(filas)
        .sort_values(["distancia_media", "variable_a", "variable_b"])
        .reset_index(drop=True)
    )
    return distancias, arbol, orden, parejas


def figuras(d, arbol, orden, carpeta, n):
    nombres = [etiqueta(v) for v in VARIABLES]
    fig, ax = plt.subplots(figsize=(12, 10))
    dendrogram(
        arbol,
        labels=nombres,
        orientation="right",
        ax=ax,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="#255c8a",
        link_color_func=lambda _: "#255c8a",
    )
    ax.set_xlabel("Distancia de unión · menor distancia = perfiles más parecidos", fontsize=11)
    ax.set_title(
        f"5.3 · Semejanza de perfiles horarios · 26 variables\n"
        f"{n:,} horas diurnas de train · distancia Manhattan media · enlace completo".replace(
            ",", "."
        ),
        fontsize=13,
        pad=18,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "Cada extremo es una variable. El árbol completo muestra uniones a distintos niveles.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    p1 = carpeta / "01_dendrograma_perfiles.png"
    fig.savefig(p1, dpi=170, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 11))
    imagen = ax.imshow(d[np.ix_(orden, orden)], cmap="Blues_r", vmin=0, vmax=d.max())
    labels = [nombres[i] for i in orden]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_title(
        "Distancias entre las 26 variables · orden del dendrograma\n"
        "Azul oscuro = perfiles más parecidos · claro = perfiles más diferentes",
        fontsize=13,
        pad=18,
    )
    fig.colorbar(
        imagen,
        ax=ax,
        fraction=0.045,
        pad=0.03,
        label="Diferencia absoluta media entre valores estandarizados",
    )
    fig.tight_layout()
    p2 = carpeta / "02_matriz_distancias.png"
    fig.savefig(p2, dpi=170, facecolor="white")
    plt.close(fig)
    return [p1, p2]


def resumen_vecinas(d, orden, media, escala):
    filas = []
    for pos, i in enumerate(orden, 1):
        dist = d[i].copy()
        dist[i] = np.inf
        minimo = dist.min()
        js = np.flatnonzero(np.isclose(dist, minimo, rtol=1e-10, atol=1e-12))
        filas.append(
            dict(
                orden_arbol=pos,
                variable=VARIABLES[i],
                vecina_mas_parecida=" | ".join(VARIABLES[j] for j in js),
                distancia_media=float(minimo),
                media_para_estandarizar=float(media.iloc[i]),
                desviacion_para_estandarizar=float(escala.iloc[i]),
            )
        )
    return pd.DataFrame(filas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", help="Ruta al único ZIP de entrada: apartado 5.2.")
    p.add_argument(
        "--no-download", action="store_true", help="Evitar descarga automática de Colab."
    )
    p.add_argument(
        "--no-display", action="store_true", help="Evitar mostrar figuras en el notebook."
    )
    args = p.parse_args()
    fuente = entrada(args.zip)
    z, media, escala, info = cargar(fuente)
    d, arbol, orden, parejas = analizar(z)
    vecinas = resumen_vecinas(d, orden, media, escala)
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    carpeta = Path.cwd() / f"EDA_5_3_perfiles_horarios_{sello}"
    carpeta.mkdir()
    (carpeta / "figuras").mkdir()
    (carpeta / "tablas").mkdir()
    vecinas.to_csv(carpeta / "tablas/01_vecina_mas_parecida.csv", index=False, encoding="utf-8-sig")
    parejas.to_csv(
        carpeta / "tablas/02_distancias_todas_las_parejas.csv", index=False, encoding="utf-8-sig"
    )
    png = figuras(d, arbol, orden, carpeta / "figuras", len(z))
    info.update(
        subseccion="5.3",
        enfoque="similitud de perfiles horarios",
        fuente_zip=fuente.name,
        variables=VARIABLES,
        estandarizacion="media y desviacion poblacional (ddof=0) de las mismas horas utilizadas",
        distancia="mean(abs(z_i(t)-z_j(t)))",
        enlace="complete",
        orden_arbol=[VARIABLES[i] for i in orden],
        uniones_arbol=arbol.tolist(),
        criterio_corte=None,
        n_figuras=2,
        n_tablas=2,
    )
    (carpeta / "README.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    texto = f"""# 5.3 · Cómo leer los resultados

## Muestra

- Horas de train: {info['n_train']}.
- Horas completamente diurnas: {info['n_dia_completo']}.
- Horas excluidas por algún valor no finito en las 26 variables: {info['n_horas_excluidas']}.
- Horas comunes utilizadas: **{len(z)}**. No se imputan valores.
- La muestra global conserva también las horas diurnas disponibles de diciembre.
- Las medias y desviaciones para estandarizar se calculan solo en esta muestra de train.

## Lectura de las figuras

**01 · Dendrograma.** Cada extremo representa una variable. Una unión cercana a cero
indica perfiles parecidos. Para unir dos bloques, el enlace completo utiliza la mayor
distancia entre sus parejas. Una unión tardía no representa necesariamente un grupo
cohesionado: el árbol termina uniendo todas las variables por construcción. La posición
arriba/abajo no es una jerarquía de importancia. No se fija un número de grupos.

**02 · Matriz.** El orden coincide con la lectura del árbol de arriba abajo. Azul oscuro
indica menor distancia. La diagonal es cero porque cada variable se compara consigo misma.
Los cuadrados oscuros a lo largo de la diagonal ayudan a localizar bloques parecidos.

## Tablas

**01 · Vecina más parecida:** una fila por variable, con su vecina de menor distancia.
Si hay empates se muestran todas. Las vecindades no tienen por qué ser recíprocas.
Esta tabla no asigna grupos; el árbol muestra la organización jerárquica.
Incluye la media y desviación utilizadas, para reproducir la estandarización.

**02 · Todas las parejas:** las 325 distancias exactas, ordenadas de menor a mayor.

## Qué significa la distancia

Primero cada serie se centra y se divide por su desviación. Después se promedian las
diferencias absolutas entre las dos series en las mismas horas. Por ejemplo, 0,20
significa una diferencia media de 0,20 unidades en la escala estandarizada. No significa
un 20 % ni una correlación de 0,80. No existe aquí un umbral universal de semejanza.

Las series con evolución opuesta pueden quedar lejos aunque estén muy relacionadas.
Las diferencias de nivel y de escala originales se eliminan al estandarizar.
Los ciclos solares y la evolución estacional siguen presentes. Este análisis mide
semejanza contemporánea de perfiles; no demuestra redundancia predictiva ni causalidad.
Los descriptores mínimo, máximo y std de DIF proceden de la misma hora que DIF media.
Su agrupación no justifica su disponibilidad como predictores del objetivo.

## Vecinas de las 26 variables

| Variable | Vecina más parecida | Distancia |
| --- | --- | ---: |
"""
    for f in vecinas.itertuples(index=False):
        vecina = ", ".join(etiqueta(v) for v in f.vecina_mas_parecida.split(" | "))
        texto += f"| {etiqueta(f.variable)} | {vecina} | {f.distancia_media:.3f} |\n"
    (carpeta / "LEEME.md").write_text(texto, encoding="utf-8")
    salida = carpeta.with_suffix(".zip")
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as archivo:
        for f in sorted(carpeta.rglob("*")):
            if f.is_file():
                archivo.write(f, f.relative_to(carpeta))
    print(f'\nTrain: {info["n_train"]} horas. Día completo: {info["n_dia_completo"]}.')
    print(f'Horas comunes utilizadas: {len(z)}. Excluidas: {info["n_horas_excluidas"]}.')
    print("Variables: 26. Parejas: 325. Figuras: 2. Tablas: 2.")
    print("Resultado:", salida)
    if not args.no_display:
        try:
            from IPython import get_ipython
            from IPython.display import Image, display

            if get_ipython() is not None:
                for imagen in png:
                    display(Image(filename=str(imagen), width=950))
                vista = vecinas[["variable", "vecina_mas_parecida", "distancia_media"]].copy()
                vista["variable"] = vista.variable.map(etiqueta)
                vista["vecina_mas_parecida"] = vista.vecina_mas_parecida.map(
                    lambda s: ", ".join(etiqueta(v) for v in s.split(" | "))
                )
                display(vista.round(3))
        except ImportError:
            pass
    if not args.no_download:
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(str(salida))


if __name__ == "__main__":
    main()
