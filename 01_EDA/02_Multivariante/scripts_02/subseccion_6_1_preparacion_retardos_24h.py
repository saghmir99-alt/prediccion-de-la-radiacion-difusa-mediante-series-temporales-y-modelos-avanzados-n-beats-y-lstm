#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.1: alineación temporal. Entrada: UN ZIP de 5.2. No calcula asociaciones.

Colab: %run subseccion_6_1_preparacion_retardos_24h.py
Local: python subseccion_6_1_preparacion_retardos_24h.py --zip resultados_5_2.zip
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
    faltan = [
        p
        for p in ["numpy", "pandas", "matplotlib", "pyarrow"]
        if importlib.util.find_spec(p) is None
    ]
    if not faltan:
        return
    try:
        import google.colab  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Instala las dependencias: python -m pip install " + " ".join(faltan)
        ) from None
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltan])
    importlib.invalidate_caches()


dependencias()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJETIVO = "DIF__media_h"
VARIABLES = [
    f"{f}__{s}"
    for f in ["SWD", "DIR", "DIF", "LWD", "SWU", "LWU"]
    for s in ["media_h", "min_h", "max_h", "std_media_h"]
]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
RETARDOS = list(range(1, 25))
CHECKPOINT = "checkpoint/checkpoint_horario_entrada_intacto.parquet"
MARCA_52 = "tablas/01_correlaciones_325_parejas_por_mes_y_contexto.csv"


def miembro(z, sufijo):
    nombres = [n for n in z.namelist() if n.endswith(sufijo)]
    if len(nombres) != 1:
        raise ValueError(f"Se requiere exactamente un archivo {sufijo} en el ZIP.")
    return nombres[0]


def zip_valido(p):
    try:
        with zipfile.ZipFile(p) as z:
            miembro(z, CHECKPOINT)
            miembro(z, MARCA_52)
        return True
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def entrada(ruta):
    if ruta:
        p = Path(ruta).expanduser().resolve()
        if not zip_valido(p):
            raise ValueError("La entrada debe ser el ZIP de 5.2 con su checkpoint horario.")
        return p
    candidatos = set()
    for carpeta in {Path.cwd(), Path(__file__).resolve().parent, Path("/content")}:
        if carpeta.is_dir():
            candidatos.update(p.resolve() for p in carpeta.glob("*.zip") if zip_valido(p))
    if len(candidatos) == 1:
        p = candidatos.pop()
        print("ZIP localizado:", p)
        return p
    if len(candidatos) > 1:
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidatos}
        if len(hashes) == 1:
            p = sorted(candidatos)[0]
            print("Copias idénticas encontradas; se utiliza:", p)
            return p
        raise ValueError(
            'Hay varios ZIP de 5.2 distintos. Indica uno con --zip "RUTA".\n'
            + "\n".join(map(str, sorted(candidatos)))
        )
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError("Indica --zip RUTA_AL_ZIP_DE_5_2.") from None
    print("Selecciona UN SOLO ZIP: los resultados de 5.2.")
    subidos = files.upload()
    if len(subidos) != 1:
        raise ValueError("Selecciona únicamente el ZIP de 5.2.")
    return entrada(next(iter(subidos)))


def cargar(p):
    with zipfile.ZipFile(p) as z:
        contenido = z.read(miembro(z, CHECKPOINT))
    df = pd.read_parquet(io.BytesIO(contenido))
    necesarias = ["particion_observacion", "fraccion_minutos_dia", *VARIABLES]
    faltan = sorted(set(necesarias) - set(df.columns))
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("Se requiere un DatetimeIndex válido.")
    if df.index.tz is None:
        raise ValueError("El índice debe documentar su zona horaria; no se asumirá UTC.")
    df.index = df.index.tz_convert("UTC")
    if df.index.has_duplicates or not df.index.equals(df.index.floor("h")):
        raise ValueError("El checkpoint contiene timestamps duplicados o fuera de la hora exacta.")
    train = df.loc[df.particion_observacion.eq("train")].sort_index().copy()
    if train.empty:
        raise ValueError("No hay filas etiquetadas train.")
    train[VARIABLES] = train[VARIABLES].apply(pd.to_numeric, errors="raise")
    frac = pd.to_numeric(train.fraccion_minutos_dia, errors="raise")
    if not frac.between(0, 1).all():
        raise ValueError("fraccion_minutos_dia debe estar definida entre 0 y 1 en train.")
    train["fraccion_minutos_dia"] = frac
    originales = train.index.copy()
    grid = pd.date_range(train.index.min(), train.index.max(), freq="h", name="timestamp_utc")
    train = train.reindex(grid)
    train["hora_train_observada"] = train.index.isin(originales)
    info = dict(
        sha256_checkpoint=hashlib.sha256(contenido).hexdigest(),
        n_train_observadas=len(originales),
        n_rejilla_train=len(grid),
        n_huecos_horarios=len(grid) - len(originales),
        inicio_train=str(grid.min()),
        fin_train=str(grid.max()),
    )
    return train, info


def preparar(train):
    t = train.index[train.hora_train_observada & train.fraccion_minutos_dia.eq(1)]
    if len(t) == 0:
        raise ValueError("No hay horas objetivo completamente diurnas.")
    y = train.loc[t, OBJETIVO].to_numpy(dtype=float)
    finito_y = np.isfinite(y)
    comun = finito_y.copy()
    datos = {OBJETIVO: y, "fraccion_minutos_dia_objetivo": np.ones(len(t))}
    razones = [[] if ok else ["DIF objetivo no finito"] for ok in finito_y]
    filas = []
    mapeo = []
    for h in RETARDOS:
        # Reindexar por TIMESTAMP real evita comprimir las noches o los huecos.
        origen = t - pd.Timedelta(hours=h)
        x = train.reindex(origen)
        presente = x.hora_train_observada.eq(True).to_numpy()
        valores = x[VARIABLES].to_numpy(dtype=float)
        finitos = np.isfinite(valores)
        comun &= presente & finitos.all(axis=1)
        sufijo = f"{h:02d}h"
        datos[f"timestamp_origen_{sufijo}"] = origen
        datos[f"origen_train_observado_{sufijo}"] = presente
        frac = x.fraccion_minutos_dia.to_numpy(dtype=float)
        datos[f"fraccion_minutos_dia_origen_{sufijo}"] = frac
        datos[f"regimen_origen_{sufijo}"] = np.select(
            [presente & (frac == 0), presente & (frac > 0) & (frac < 1), presente & (frac == 1)],
            ["noche", "transicion", "dia_completo"],
            default="no_disponible",
        )
        for i in np.flatnonzero(~presente):
            razones[i].append(f"lag {h}h: origen fuera de train o hueco horario")
        for i in np.flatnonzero(presente & ~finitos.all(axis=1)):
            malas = [v for v, ok in zip(VARIABLES, finitos[i]) if not ok]
            razones[i].append(f"lag {h}h: no finitos en " + ", ".join(malas))
        for j, v in enumerate(VARIABLES):
            col = f"{v}__lag_{sufijo}"
            datos[col] = valores[:, j]
            mapeo.append(dict(variable=v, retardo_h=h, columna=col))
            valido = finito_y & presente & finitos[:, j]
            filas.append(
                dict(
                    variable=v,
                    retardo_h=h,
                    n_objetivos_diurnos=len(t),
                    n_origenes_train_observados=int(presente.sum()),
                    n_pares_finitos_disponibles=int(valido.sum()),
                )
            )
    pares = pd.DataFrame(datos, index=t.rename("timestamp_objetivo_utc"))
    pares.insert(1, "muestra_principal_comun", comun)
    if comun.sum() < 3:
        raise ValueError(
            "La intersección común tiene menos de tres observaciones; revisar disponibilidad."
        )
    disponibilidad = pd.DataFrame(filas)
    disponibilidad["n_pares_muestra_comun"] = int(comun.sum())
    disponibilidad["n_disponibles_fuera_muestra_comun"] = (
        disponibilidad.n_pares_finitos_disponibles - int(comun.sum())
    )
    meses = sorted(set(train.index.strftime("%Y-%m")))
    mensuales = []
    for mes in meses:
        mascara = t.strftime("%Y-%m") == mes
        horas_train = int(
            train.loc[train.index.strftime("%Y-%m") == mes, "hora_train_observada"].sum()
        )
        mes_inicio = pd.Timestamp(mes + "-01", tz="UTC")
        esperadas = mes_inicio.days_in_month * 24
        mensuales.append(
            dict(
                mes=mes,
                n_horas_train=horas_train,
                n_horas_calendario_mes=esperadas,
                cobertura_train_mes=horas_train / esperadas,
                mes_train_incompleto=horas_train < esperadas,
                n_objetivos_diurnos=int(mascara.sum()),
                n_muestra_comun=int(comun[mascara].sum()),
                n_excluidos=int((~comun[mascara]).sum()),
            )
        )
    mensual = pd.DataFrame(mensuales)
    origenes = []
    for h in RETARDOS:
        r = pares.loc[comun, f"regimen_origen_{h:02d}h"].value_counts()
        origenes.append(
            dict(
                retardo_h=h,
                n_muestra_comun=int(comun.sum()),
                **{
                    f"n_{k}": int(r.get(k, 0))
                    for k in ["noche", "transicion", "dia_completo", "no_disponible"]
                },
            )
        )
    exclusiones = pd.DataFrame(
        [
            dict(timestamp_objetivo_utc=str(t[i]), motivos="; ".join(razones[i]))
            for i in np.flatnonzero(~comun)
        ],
        columns=["timestamp_objetivo_utc", "motivos"],
    )
    # Invariantes: todos los desfases conservan horas reales y no se usan datos externos.
    for h in RETARDOS:
        origen = pd.DatetimeIndex(pares[f"timestamp_origen_{h:02d}h"])
        assert ((t - origen) == pd.Timedelta(hours=h)).all()
        assert pares.loc[comun, f"origen_train_observado_{h:02d}h"].all()
    columnas_lag = [r["columna"] for r in mapeo]
    assert np.isfinite(pares.loc[comun, [OBJETIVO, *columnas_lag]].to_numpy(dtype=float)).all()
    return pares, disponibilidad, mensual, pd.DataFrame(origenes), exclusiones, mapeo


def figura(mensual, origenes, path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4))
    ax = axes[0]
    pos = np.arange(len(mensual))
    a = ax.bar(
        pos - 0.19, mensual.n_objetivos_diurnos, 0.38, label="Objetivos diurnos", color="#8fb4cc"
    )
    b = ax.bar(pos + 0.19, mensual.n_muestra_comun, 0.38, label="Muestra común", color="#245b78")
    for bars in (a, b):
        ax.bar_label(bars, fontsize=8, padding=3)
    labels = [r.mes + ("*" if r.mes_train_incompleto else "") for r in mensual.itertuples()]
    ax.set_xticks(pos, labels, rotation=45, ha="right")
    ax.set_ylabel("Número de horas objetivo")
    ax.set_ylim(0, max(mensual.n_objetivos_diurnos) * 1.25)
    ax.set_title("Disponibilidad mensual")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax = axes[1]
    bottom = np.zeros(len(origenes))
    pos = np.arange(len(origenes))
    for col, label, color in [
        ("n_dia_completo", "Día completo", "#e9ba49"),
        ("n_transicion", "Transición", "#c87d53"),
        ("n_noche", "Noche", "#34557d"),
    ]:
        v = origenes[col].to_numpy()
        ax.bar(pos, v, bottom=bottom, label=label, color=color)
        bottom += v
    ax.set_xticks(pos, [str(h) for h in origenes.retardo_h])
    ax.set_xlabel("Antigüedad del predictor respecto al objetivo (horas)")
    ax.set_ylabel("Número de orígenes en la muestra común")
    ax.set_ylim(0, max(bottom) * 1.26)
    ax.set_title("¿De qué horas proceden los predictores?")
    ax.legend(fontsize=9, ncol=3, loc="upper center")
    n = int(origenes.n_muestra_comun.iloc[0])
    fig.suptitle(
        f"6.1 · Preparación temporal · {n:,} objetivos comunes".replace(",", "."), fontsize=15
    )
    fig.text(
        0.5,
        0.012,
        "* Mes con cobertura train incompleta. Todos los objetivos son diurnos; "
        "sus predictores pueden proceder de noche o transición.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_entrada")
    parser.add_argument("--salida", default=".", help="Carpeta donde crear resultados")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    p = entrada(args.zip_entrada)
    train, info = cargar(p)
    pares, disponibilidad, mensual, origenes, exclusiones, mapeo = preparar(train)
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    carpeta = Path(args.salida).expanduser().resolve() / f"EDA_6_1_preparacion_retardos_24h_{sello}"
    for sub in ["datos", "checkpoint", "tablas", "figuras"]:
        (carpeta / sub).mkdir(parents=True, exist_ok=False)
    pares.to_parquet(carpeta / "datos/pares_retardados_diurnos.parquet")
    train.to_parquet(carpeta / "checkpoint/checkpoint_train_horario.parquet")
    for tabla, nombre in [
        (disponibilidad, "01_disponibilidad_pares"),
        (mensual, "02_disponibilidad_mensual"),
        (origenes, "03_origen_geometrico_retardos"),
        (exclusiones, "04_objetivos_excluidos_muestra_comun"),
    ]:
        tabla.to_csv(carpeta / f"tablas/{nombre}.csv", index=False, encoding="utf-8-sig")
    png = carpeta / "figuras/01_disponibilidad_y_origen.png"
    figura(mensual, origenes, png)
    info.update(
        version="6.1.2",
        entrada=p.name,
        sha256_zip_entrada=hashlib.sha256(p.read_bytes()).hexdigest(),
        objetivo=OBJETIVO,
        variables_retardadas=VARIABLES,
        covariables=[v for v in VARIABLES if v != OBJETIVO],
        retardos_h=RETARDOS,
        mapa_columnas_retardadas=mapeo,
        n_objetivos_diurnos=len(pares),
        n_muestra_principal_comun=int(pares.muestra_principal_comun.sum()),
        n_excluidos_muestra_comun=len(exclusiones),
        zona_horaria="UTC",
        imputacion=False,
        estandarizacion=False,
        convencion="DIF(T) frente a X(T-h); h es antigüedad antes del objetivo.",
        horizonte_prediccion_h=1,
        tiempo_emision="No fijado por el timestamp: verificar si etiqueta inicio o fin del agregado. "
        "Los predictores deben estar completos y disponibles al emitir.",
        filtro_objetivo="Hora original de train con fraccion_minutos_dia == 1.",
        muestra_principal="Objetivo y 27 variables en los 24 retardos finitos; "
        "todos los orígenes pertenecen a train observado.",
        pares="datos/pares_retardados_diurnos.parquet",
        checkpoint="checkpoint/checkpoint_train_horario.parquet",
        inferencia="No se han calculado correlaciones, MI, STL ni pruebas estadísticas.",
    )
    (carpeta / "README.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    texto = f"""# 6.1 · Preparación de relaciones retardadas

Entrada única: `{p.name}`. Tiempo UTC. Solo train.

- Horas observadas de train: {info['n_train_observadas']}.
- Huecos insertados en la rejilla, sin imputación: {info['n_huecos_horarios']}.
- Objetivos completamente diurnos: {len(pares)}.
- Objetivos de la muestra común: {info['n_muestra_principal_comun']}.
- Exclusiones de la muestra común: {len(exclusiones)}; consultar tabla 04.

## Convención y uso

Cada fila tiene un objetivo DIF(T) y columnas X(T-h), para h=1,2,...,24 horas.
El horizonte previsto es una hora: si t es la última hora agregada disponible, T=t+1;
h=1 usa X(t) y h=24 usa X(t-23). La etiqueta de inicio o fin de intervalo y el
tiempo real de disponibilidad deben respetarse al implantar la predicción.

El Parquet de pares conserva TODOS los objetivos diurnos. Para las comparaciones
principales de 6.2 y 6.3, seleccionar `muestra_principal_comun == True`.
Los 27 predictores son las 26 covariables anteriores y DIF media retardada.
`DIF__media_h` sin sufijo lag es exclusivamente la respuesta.
Las columnas geométricas y de timestamp son de auditoría, no nuevos predictores.

El desplazamiento se construye con timestamps de la cronología horaria completa,
antes de seleccionar los objetivos diurnos. No se unen artificialmente tardes
y mañanas. No se aprovechan filas de pre_segmento, validación o test.
Se admiten predictores de noche y transición: eran observables antes del objetivo.
El filtro diurno describe el objetivo, no elimina el ciclo solar.

La tabla 01 diferencia disponibilidad por pareja y uso en la muestra común.
La tabla 02 mantiene diciembre e indica su cobertura parcial; esa descripción
no justifica tratarlo como un mes completo en futuras comparaciones mensuales.
La tabla 03 clasifica los orígenes de los predictores sobre la muestra común.
La figura contiene dos paneles de disponibilidad, sin métricas de asociación.

Para análisis posteriores se necesita únicamente ESTE ZIP. Incluye el checkpoint
horario de train, manteniendo las columnas originales y añadiendo la marca de hora
observada. No es el checkpoint original intacto: se ha limitado a train y, si
existían huecos, se han representado mediante filas ausentes. No hay imputación.
Los futuros diagnósticos sobre residuos/filtros tendrán que recalcular su soporte
válido y comparar series originales y ajustadas en las mismas horas.
"""
    (carpeta / "LEEME.md").write_text(texto, encoding="utf-8")
    destino = carpeta.with_suffix(".zip")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(carpeta.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(carpeta).as_posix())
    print("\nPreparación finalizada. No se han calculado asociaciones.")
    print(
        f'Objetivos diurnos: {len(pares)} | Muestra común: {info["n_muestra_principal_comun"]}'
        f" | Excluidos: {len(exclusiones)}"
    )
    print(
        mensual[["mes", "n_objetivos_diurnos", "n_muestra_comun", "n_excluidos"]].to_string(
            index=False
        )
    )
    print("\nZIP generado:", destino)
    if not args.no_display:
        try:
            from IPython.display import Image, display

            display(Image(filename=str(png)))
        except ImportError:
            print("Figura guardada:", png)
    if not args.no_download:
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(str(destino))


if __name__ == "__main__":
    main()
