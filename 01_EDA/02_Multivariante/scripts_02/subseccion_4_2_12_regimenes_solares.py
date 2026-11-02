"""EDA contemporáneo X(t) ↔ DIF(t) en seis bandas de elevación, solo día completo.

Colab: %run subseccion_4_2_12_regimenes_solares.py
El programa pide el ZIP de 4.2.11 y descarga un ZIP con el checkpoint intacto.
Terminal: python subseccion_4_2_12_regimenes_solares.py --zip ruta/4_2_11.zip
Alternativa rápida: --solo-tablas (mantiene gráficos resumen, omite 26 figuras).
No rehace STL, no altera observaciones y no estudia DIF(t+1).
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
ELEVACION = "elevacion_efectiva_h"
ESTADISTICOS = ("media_h", "min_h", "max_h", "std_media_h")
FAMILIAS = ("SWD", "DIR", "DIF", "LWD", "SWU", "LWU")
VARIABLES = [f"{familia}__{estadistico}" for familia in FAMILIAS for estadistico in ESTADISTICOS]
VARIABLES += ["T2__media_h", "RH__media_h", "PoPoPoPo__media_h"]
COVARIABLES = [v for v in VARIABLES if v != OBJETIVO]
K_VECINOS = (3, 5, 10)
SEMILLA_MI = 42
MAX_BINS_X = 8
BANDAS = ("0_10", "10_20", "20_30", "30_40", "40_50", "mayor_50")
ETIQUETAS = {
    "0_10": "0–10°",
    "10_20": "10–20°",
    "20_30": "20–30°",
    "30_40": "30–40°",
    "40_50": "40–50°",
    "mayor_50": ">50°",
}
HORAS_ESPERADAS = {
    "0_10": 323,
    "10_20": 618,
    "20_30": 486,
    "30_40": 397,
    "40_50": 343,
    "mayor_50": 421,
}


def solicitar_zip(ruta: str | None) -> Path:
    if ruta:
        elegido = Path(ruta).expanduser().resolve()
    else:
        try:
            from google.colab import files
        except ImportError as exc:
            raise RuntimeError("Fuera de Colab pasa --zip RUTA.zip") from exc
        print("Selecciona exactamente el ZIP de salida de la subsección 4.2.11.")
        seleccion = files.upload()
        zips = [nombre for nombre in seleccion if nombre.lower().endswith(".zip")]
        if len(seleccion) != 1 or len(zips) != 1:
            raise ValueError("Debe subirse un solo ZIP de la subsección 4.2.11.")
        elegido = Path(zips[0]).resolve()
    if not elegido.is_file() or elegido.suffix.lower() != ".zip":
        raise FileNotFoundError(f"ZIP no encontrado: {elegido}")
    return elegido


def cargar_checkpoint_y_contexto(
    ruta_zip: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, bytes, bytes, str]:
    with zipfile.ZipFile(ruta_zip) as zf:
        miembros = zf.namelist()
        parquets = [m for m in miembros if m.endswith("checkpoint_horario_entrada_intacto.parquet")]
        anteriores = [
            m for m in miembros if m.endswith("01_metricas_train_dia_transicion_dia_0_10.csv")
        ]
        if len(parquets) != 1 or len(anteriores) != 1:
            raise ValueError(
                "El ZIP de 4.2.11 debe contener un checkpoint horario intacto "
                "y la tabla 01_metricas_train_dia_transicion_dia_0_10.csv. "
                f"Checkpoint: {parquets}; métricas: {anteriores}."
            )
        nombre_checkpoint = parquets[0]
        checkpoint_bytes = zf.read(nombre_checkpoint)
        contexto_bytes = zf.read(anteriores[0])
    try:
        tabla = pd.read_parquet(io.BytesIO(checkpoint_bytes))
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Instala pyarrow en Colab para leer el checkpoint parquet.") from exc
    contexto = pd.read_csv(io.BytesIO(contexto_bytes))
    requeridas = {"particion_observacion", FRACCION, ELEVACION, *VARIABLES}
    faltantes = sorted(requeridas - set(tabla.columns))
    if faltantes:
        raise ValueError(f"El checkpoint no contiene las columnas necesarias: {faltantes}")
    if (
        len(contexto) != 26
        or contexto["variable"].duplicated().any()
        or set(contexto["variable"]) != set(COVARIABLES)
    ):
        raise ValueError(
            "Las métricas de contexto anteriores no contienen las mismas 26 variables."
        )
    columnas = {
        "variable",
        "n_pares_train",
        "n_pares_dia",
        "n_pares_transicion",
        "n_pares_dia_0_10",
        "pearson_dia_0_10",
        "spearman_dia_0_10",
    }
    for g in ("train", "dia", "transicion", "dia_0_10"):
        columnas.update({f"pearson_{g}", f"spearman_{g}", f"MI_k5_nats_{g}"})
    faltantes_contexto = sorted(columnas - set(contexto.columns))
    if faltantes_contexto:
        raise ValueError(f"La tabla de 4.2.11 carece de {faltantes_contexto}")
    return tabla, contexto, checkpoint_bytes, contexto_bytes, nombre_checkpoint


def preparar_bandas(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = tabla.loc[tabla["particion_observacion"].eq("train")].copy()
    if len(train) != 5208:
        raise ValueError(f"Se esperaban 5.208 horas de train; el checkpoint tiene {len(train)}.")
    fraccion = pd.to_numeric(train[FRACCION], errors="coerce")
    elevacion = pd.to_numeric(train[ELEVACION], errors="coerce")
    if fraccion.isna().any() or (~fraccion.between(0, 1)).any():
        raise ValueError("fraccion_minutos_dia debe ser finita y estar entre 0 y 1.")
    noche = fraccion.eq(0)
    transicion = fraccion.gt(0) & fraccion.lt(1)
    completas = fraccion.eq(1)
    if (completas & (~np.isfinite(elevacion) | elevacion.le(0))).any():
        raise ValueError("Una hora completa diurna carece de elevación efectiva positiva finita.")
    dia = train.loc[completas].copy()
    dia["banda_4_2_12"] = pd.cut(
        pd.to_numeric(dia[ELEVACION], errors="raise"),
        bins=[0, 10, 20, 30, 40, 50, np.inf],
        labels=list(BANDAS),
        right=True,
    )
    if dia["banda_4_2_12"].isna().any():
        raise ValueError("Algunas horas completas diurnas no pertenecen a ninguna banda.")
    observados = dia["banda_4_2_12"].value_counts(sort=False).to_dict()
    observados = {nombre: int(observados.get(nombre, 0)) for nombre in BANDAS}
    verificacion = {
        "n_train": len(train),
        "n_noche": int(noche.sum()),
        "n_transicion": int(transicion.sum()),
        "n_dia_completo": len(dia),
        "recuento_bandas": observados,
    }
    if (verificacion["n_noche"], verificacion["n_transicion"], len(dia)) != (2191, 429, 2588):
        raise ValueError(
            f"El checkpoint no reproduce noche/transición/día de 4.2.11: {verificacion}"
        )
    if observados != HORAS_ESPERADAS:
        raise ValueError(
            "Los límites de elevación no reproducen los recuentos de 4.2.9. "
            f"Observados: {observados}. Esperados: {HORAS_ESPERADAS}. "
            "Revisar el checkpoint y las fronteras antes de interpretar resultados."
        )
    return train, dia, verificacion


def pares_finitos(subtabla: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(subtabla[variable], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(subtabla[OBJETIVO], errors="coerce").to_numpy(dtype=float)
    validos = np.isfinite(x) & np.isfinite(y)
    return x[validos], y[validos]


def calcular_metricas(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    salida = {
        "n_pares": n,
        "pearson": np.nan,
        "spearman": np.nan,
        **{f"MI_k{k}_nats": np.nan for k in K_VECINOS},
        "n_X_cero": int(np.count_nonzero(x == 0)),
        "n_X_valor_mas_repetido": int(pd.Series(x).value_counts().iloc[0]) if n else 0,
    }
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return salida
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        salida["pearson"] = float(pearsonr(x, y).statistic)
        salida["spearman"] = float(spearmanr(x, y).statistic)
    for k in K_VECINOS:
        if n > k + 1:
            salida[f"MI_k{k}_nats"] = float(
                mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    discrete_features=False,
                    n_neighbors=k,
                    random_state=SEMILLA_MI,
                )[0]
            )
    return salida


def cuantiles_por_x(x: np.ndarray, y: np.ndarray, variable: str, banda: str) -> pd.DataFrame:
    columnas = [
        "variable",
        "banda",
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
    tabla = pd.DataFrame({"X": x, "DIF": y})
    if tabla["X"].nunique() < 2:
        tabla["bin"] = 0
    else:
        tabla["bin"] = pd.qcut(tabla["X"], q=MAX_BINS_X, labels=False, duplicates="drop")
        if tabla["bin"].isna().any():
            raise ValueError(f"Intervalos cuantílicos inválidos en {variable}/{banda}")
    filas = []
    for bin_id, grupo in tabla.groupby("bin", sort=True):
        filas.append(
            {
                "variable": variable,
                "banda": banda,
                "intervalo_efectivo": int(bin_id) + 1,
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


def graficar_variable(variable: str, datos: dict, resumenes: dict, destino: Path) -> None:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:
        raise RuntimeError("Para las figuras LOWESS instala statsmodels en Colab.") from exc
    fig, axs = plt.subplots(6, 2, figsize=(16, 22))
    fig.suptitle(f"{variable} ↔ DIF(t) por elevación · solo horas completas diurnas", fontsize=17)
    for fila, banda in enumerate(BANDAS):
        ax_a, ax_b = axs[fila]
        x, y = datos[banda]
        bins = resumenes[banda]
        if len(x):
            hexa = ax_a.hexbin(x, y, gridsize=30, mincnt=1, cmap="viridis", norm=LogNorm())
            fig.colorbar(hexa, ax=ax_a, fraction=0.045, label="N.º de pares (log)")
            n_moda = int(pd.Series(x).value_counts().iloc[0])
            if len(x) >= 15 and np.unique(x).size >= 5 and n_moda / len(x) < 0.5:
                curva = lowess(y, x, frac=0.25, it=1, return_sorted=True)
                if len(curva) > 1 and np.isfinite(curva).all():
                    ax_a.plot(
                        curva[:, 0], curva[:, 1], lw=2, color="#d1492e", label="LOWESS descriptiva"
                    )
                    ax_a.legend(loc="best")
            elif n_moda / len(x) >= 0.5:
                ax_a.text(
                    0.03,
                    0.95,
                    "LOWESS omitida: numerosos empates",
                    transform=ax_a.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
        ax_a.set(
            title=f"{ETIQUETAS[banda]} · hexbin · n={len(x)}",
            xlabel=f"{variable} [{unidades_x(variable)}]",
            ylabel="DIF media horaria [W/m²]",
        )
        ax_a.axhline(0, color="grey", lw=0.7)
        ax_a.grid(alpha=0.2)

        if not bins.empty:
            pos = bins["posicion_acumulada_X"].to_numpy(dtype=float)
            p10, p25, p75, p90 = (
                bins[col].to_numpy(dtype=float)
                for col in ("DIF_p10", "DIF_p25", "DIF_p75", "DIF_p90")
            )
            if len(bins) > 1:
                ax_b.fill_between(pos, p10, p90, color="#a9cff6", alpha=0.34, label="P10–P90")
                ax_b.fill_between(pos, p25, p75, color="#70addb", alpha=0.52, label="P25–P75")
            else:
                ax_b.vlines(pos, p10, p90, color="#a9cff6", linewidth=11, label="P10–P90")
                ax_b.vlines(pos, p25, p75, color="#70addb", linewidth=5, label="P25–P75")
            ax_b.plot(
                pos, bins["DIF_mediana"], marker="o", ms=4, color="#153c78", label="Mediana de DIF"
            )
            ax_b.legend(loc="best", fontsize=8)
        ax_b.set(
            title=f"{ETIQUETAS[banda]} · DIF por X · {len(bins)} intervalos reales",
            xlim=(0, 1),
            xlabel="Posición acumulada de X en esta banda",
            ylabel="DIF media horaria [W/m²]",
        )
        ax_b.axhline(0, color="grey", lw=0.7)
        ax_b.grid(alpha=0.25)
    fig.text(
        0.5,
        0.01,
        "Cada banda usa sus propias escalas de X y DIF. Comprobar "
        "ejes y tamaño de los intervalos antes de comparar formas.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(destino, dpi=145)
    plt.close(fig)


def graficar_matriz_metricas(m: pd.DataFrame, destino: Path) -> None:
    orden = list(reversed(COVARIABLES))
    fig, axs = plt.subplots(1, 3, figsize=(22, 15), sharey=True)
    for ax, clave, titulo, paleta in zip(
        axs,
        ("pearson", "spearman", "MI_k5_nats"),
        ("Pearson", "Spearman", "Información mutua k=5 (nats)"),
        ("coolwarm", "coolwarm", "viridis"),
    ):
        tabla = m.pivot(index="variable", columns="banda", values=clave).reindex(
            index=orden, columns=BANDAS
        )
        array = np.ma.masked_invalid(tabla.to_numpy(dtype=float))
        opciones = {"vmin": -1, "vmax": 1} if clave != "MI_k5_nats" else {"vmin": 0}
        im = ax.imshow(array, cmap=paleta, aspect="auto", **opciones)
        ax.set(
            title=titulo,
            xticks=np.arange(len(BANDAS)),
            xticklabels=[ETIQUETAS[b] for b in BANDAS],
            yticks=np.arange(len(orden)),
            yticklabels=orden,
        )
        ax.tick_params(axis="x", labelrotation=40)
        for i in range(len(orden)):
            for j in range(len(BANDAS)):
                if np.isfinite(tabla.iat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{tabla.iat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        color=(
                            "white"
                            if abs(tabla.iat[i, j]) > 0.6 and clave != "MI_k5_nats"
                            else "black"
                        ),
                    )
        fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    fig.suptitle("26 covariables frente a DIF(t) · seis bandas completamente diurnas", fontsize=16)
    fig.text(
        0.5,
        0.02,
        "Son muestras de tamaño y meses diferentes; comparar MI de forma "
        "descriptiva, sin atribuir cambios causalmente a la elevación.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def disponibilidad_mensual(dia: pd.DataFrame) -> pd.DataFrame:
    if isinstance(dia.index, pd.DatetimeIndex):
        fechas = dia.index.strftime("%Y-%m")
    elif "timestamp_utc" in dia.columns:
        stamp = pd.to_datetime(dia["timestamp_utc"], utc=True, errors="coerce")
        if stamp.isna().any():
            raise ValueError("El checkpoint contiene timestamps inválidos.")
        fechas = stamp.dt.strftime("%Y-%m").to_numpy()
    else:
        raise ValueError("Falta un índice temporal o timestamp_utc para la cobertura mensual.")
    disponibilidad = pd.crosstab(
        pd.Series(fechas, name="mes"),
        pd.Series(dia["banda_4_2_12"].astype(str).to_numpy(), name="banda"),
    )
    return disponibilidad.reindex(columns=BANDAS, fill_value=0).reset_index()


def graficar_disponibilidad(t: pd.DataFrame, destino: Path) -> None:
    matriz = t[list(BANDAS)].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matriz, cmap="Blues", aspect="auto")
    ax.set(
        title="Horas completamente diurnas por mes y elevación",
        xticks=np.arange(6),
        xticklabels=[ETIQUETAS[b] for b in BANDAS],
        yticks=np.arange(len(t)),
        yticklabels=t["mes"],
        xlabel="Banda",
        ylabel="Mes",
    )
    for i in range(len(t)):
        for j in range(6):
            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matriz[i, j] >= matriz.max() * 0.63 else "#303030",
            )
    fig.colorbar(im, ax=ax, label="Número de horas")
    fig.tight_layout()
    fig.savefig(destino, dpi=155)
    plt.close(fig)


def comparar_low_con_contexto(m: pd.DataFrame, contexto: pd.DataFrame) -> None:
    antigua = contexto.set_index("variable").loc[COVARIABLES]
    actual = m.loc[m["banda"].eq("0_10")].set_index("variable").loc[COVARIABLES]
    if not (
        actual["n_pares"].to_numpy(dtype=int) == antigua["n_pares_dia_0_10"].to_numpy(dtype=int)
    ).all():
        raise ValueError("Los pares de 0–10° no coinciden con los documentados en 4.2.11.")
    for clave in ("pearson", "spearman"):
        if not np.allclose(
            actual[clave], antigua[f"{clave}_dia_0_10"], atol=1e-7, rtol=0, equal_nan=True
        ):
            raise ValueError(f"{clave} de 0–10° no coincide con 4.2.11: revisar intervalos.")


def ejecutar(ruta_zip: Path, solo_tablas: bool = False) -> Path:
    tabla, contexto, checkpoint_bytes, contexto_bytes, miembro = cargar_checkpoint_y_contexto(
        ruta_zip
    )
    train, dia, conteos = preparar_bandas(tabla)
    print(f"Checkpoint: {miembro}; train={len(train)}, día completo={len(dia)}")
    print(f"Horas por banda: {conteos['recuento_bandas']}")

    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directorio = Path.cwd() / f"EDA_multivariante_F4_S4_212_regimenes_solares_{sello}"
    if directorio.exists() or directorio.with_suffix(".zip").exists():
        raise FileExistsError(f"No se sobrescribirá una salida anterior: {directorio}")
    tdir, fdir, pdir = (directorio / "tablas", directorio / "figuras", directorio / "checkpoint")
    for carpeta in (tdir, fdir, pdir):
        carpeta.mkdir(parents=True)
    (pdir / "checkpoint_horario_entrada_intacto.parquet").write_bytes(checkpoint_bytes)
    (tdir / "06_contexto_4_2_11_intacto.csv").write_bytes(contexto_bytes)

    datos_por_banda = {b: dia.loc[dia["banda_4_2_12"].eq(b)] for b in BANDAS}
    filas_metricas, filas_cuantiles = [], []
    for numero, variable in enumerate(COVARIABLES, start=1):
        figura_datos, figura_resumenes = {}, {}
        for banda in BANDAS:
            x, y = pares_finitos(datos_por_banda[banda], variable)
            metrica = calcular_metricas(x, y)
            filas_metricas.append(
                {
                    "variable": variable,
                    "banda": banda,
                    "n_horas_banda": len(datos_por_banda[banda]),
                    **metrica,
                }
            )
            grupos = cuantiles_por_x(x, y, variable, banda)
            filas_cuantiles.append(grupos)
            figura_datos[banda] = (x, y)
            figura_resumenes[banda] = grupos
        if not solo_tablas:
            graficar_variable(
                variable,
                figura_datos,
                figura_resumenes,
                fdir / f"{numero:02d}_{variable}_seis_bandas.png",
            )
        print(f"[{numero:02d}/26] {variable} · seis bandas completadas")

    metricas = pd.DataFrame(filas_metricas)
    comparar_low_con_contexto(metricas, contexto)
    metricas.to_csv(tdir / "01_metricas_26_variables_seis_bandas.csv", index=False)
    contextoid = contexto.set_index("variable").loc[COVARIABLES]
    amplia = contextoid[
        [
            "pearson_train",
            "spearman_train",
            "MI_k5_nats_train",
            "pearson_dia",
            "spearman_dia",
            "MI_k5_nats_dia",
            "pearson_transicion",
            "spearman_transicion",
            "MI_k5_nats_transicion",
        ]
    ].copy()
    for banda in BANDAS:
        subtabla = metricas.loc[metricas["banda"].eq(banda)].set_index("variable").loc[COVARIABLES]
        for clave in ("n_pares", "pearson", "spearman", "MI_k5_nats"):
            amplia[f"{clave}_{banda}"] = subtabla[clave]
    amplia.reset_index().to_csv(tdir / "02_resumen_contexto_vs_seis_bandas.csv", index=False)
    pd.concat(filas_cuantiles, ignore_index=True).to_csv(
        tdir / "03_mediana_DIF_por_cuantiles_X_en_bandas.csv", index=False
    )
    meses = disponibilidad_mensual(dia)
    if meses[list(BANDAS)].sum().to_dict() != HORAS_ESPERADAS:
        raise ValueError("El recuento mensual de bandas no coincide con el recuento total.")
    meses.to_csv(tdir / "04_disponibilidad_mensual_bandas.csv", index=False)
    pd.DataFrame(
        [
            {
                **{k: v for k, v in conteos.items() if k != "recuento_bandas"},
                **{f"n_{banda}": conteos["recuento_bandas"][banda] for banda in BANDAS},
            }
        ]
    ).to_csv(tdir / "05_verificacion_geometrica.csv", index=False)
    graficar_matriz_metricas(metricas, fdir / "00_matriz_26_variables_seis_bandas.png")
    graficar_disponibilidad(meses, fdir / "00_disponibilidad_mensual.png")

    (directorio / "README.json").write_text(
        json.dumps(
            {
                "subseccion": "4.2.12",
                "ZIP_entrada_4_2_11": ruta_zip.name,
                "checkpoint_original_inmutado": True,
                "metricas_previas_4_2_11_inmutadas": True,
                "filtro": "particion_observacion == train & fraccion_minutos_dia == 1",
                "elevacion": ELEVACION,
                "fronteras": [0, 10, 20, 30, 40, 50, ">50"],
                "pertenencia": "(0,10], (10,20], (20,30], (30,40], (40,50], (50,+inf)",
                "bandas": list(BANDAS),
                "n_bandas": conteos["recuento_bandas"],
                "n_covariables": 26,
                "n_bins_cuantiles_maximo": MAX_BINS_X,
                "estimador_MI": {
                    "k": K_VECINOS,
                    "random_state": SEMILLA_MI,
                    "discrete_features": False,
                },
                "pares": "Solo pares finitos X y DIF, máscara distinta por variable y banda",
                "figuras_individuales": not solo_tablas,
                "LOWESS": {"frac": 0.25, "it": 1, "omitida_si_X_moda_50pct_o_mas": True},
                "advertencia": "No seleccionar variables para DIF(t+1) con relaciones X(t)–DIF(t). "
                "La elevación, el momento del día y el mes no quedan aislados "
                "por completo; descriptores de DIF en t son parte del objetivo.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    salida_zip = directorio.with_suffix(".zip")
    with zipfile.ZipFile(salida_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fichero in sorted(directorio.rglob("*")):
            if fichero.is_file():
                zf.write(fichero, arcname=fichero.relative_to(directorio))
    print(f"ZIP recargable: {salida_zip}")
    return salida_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", help="Ruta al ZIP de 4.2.11 fuera de Colab")
    parser.add_argument(
        "--solo-tablas",
        action="store_true",
        help="Calcula todas las métricas y figuras resumen, sin 26 figuras individuales",
    )
    args = parser.parse_args()
    salida = ejecutar(solicitar_zip(args.zip), solo_tablas=args.solo_tablas)
    try:
        from google.colab import files
    except ImportError:
        pass
    else:
        files.download(str(salida))


if __name__ == "__main__":
    main()
