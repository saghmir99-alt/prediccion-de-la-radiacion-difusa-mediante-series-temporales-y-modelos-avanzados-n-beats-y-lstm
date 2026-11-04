# BASE_DIR · Resultados del modelado

Esta carpeta es la copia de la carpeta de trabajo de Google Drive «03. Modelado». Los cuadernos `03_modelado.ipynb`,
`04_modelado_combinaciones.ipynb`, `05_linea_LSTM.ipynb` y `06_linea_NBEATS_y_test.ipynb` la llaman `BASE_DIR` y leen y
escriben en ella. Reúne lo que generaron esos cuadernos y los scripts del ordenador personal: matrices de entrada, modelos
entrenados, predicciones, tablas, figuras, informes y registros de ejecución.

Las tablas y los veredictos que recoge la memoria están también en las salidas de los cuadernos. Esta carpeta conserva los
ficheros de los que proceden, de modo que cada cifra puede rastrearse hasta su origen y los cuadernos pueden volver a
ejecutarse. Se mantiene la estructura original de Drive, que es la que esperan los cuadernos.

**Tamaño:** 1.929 ficheros, unos 278 MB.

## Contenido

| Ruta | Contenido | Origen |
|---|---|---|
| Raíz (`*.zip`, `*.csv`, `*.parquet`) | 15 ZIP con los resultados de las referencias, Ridge, HGB y las primeras N-BEATS y LSTM, y 7 tablas de la síntesis de modelos | `03_modelado.ipynb`, partes 1 a 4 |
| `documentacion/` | Módulos `entorno_tfm.py` y `gestor_ejecuciones.py`; configuración del entorno; punteros a la última ejecución de cada etapa (`ULTIMA_*.json`, `ULTIMO_*.json`); plan del cribado de variables y decisión sobre el bloque LWD; punto de continuidad del modelado (`checkpoint_continuidad_modelado_*`), e inventarios (ficheros, esquema del parquet, bloques de variables y ausencias de LWD) | `03_modelado.ipynb`, partes 5 y 6 |
| `datos/meteorologia_cuadratica/` | Variables meteorológicas horarias preparadas para el cribado de bloques | `03_modelado.ipynb`, parte 6 |
| `comparativas/` | Comparaciones de los bloques SWD y LWD (`comparacion_externa_*`) | `03_modelado.ipynb`, parte 6 |
| `resultados/ejecuciones/` | 29 ejecuciones del cribado de bloques de variables (preparación, selección interna de épocas y reentrenamiento), de las combinaciones con los estadísticos de la DIF y de la recuperación de modelos. `reentrenamiento_difstats_20261006_182213/` contiene LSTM-DIFstats y la primera N-BEATS, de las que parten los cuadernos 05 y 06 | `03_modelado.ipynb`, parte 6, y `04_modelado_combinaciones.ipynb` |
| `resultados/comparativas/` | Comparaciones globales y relativas, rankings de modelos, reconstrucción de las referencias clásicas y recuperación de modelos pendientes | `03_modelado.ipynb`, parte 6, y `04_modelado_combinaciones.ipynb`, parte 1 |
| `resultados/diagnostico/` | Líneas LSTM y N-BEATS, búsqueda de hiperparámetros de la LSTM y evaluación en el test (detalle más abajo) | `05_linea_LSTM.ipynb`, `06_linea_NBEATS_y_test.ipynb` y ordenador personal |
| `optuna_local/` | Datos exportados para el ordenador personal (`datos/`, `datos_validacion/`) y búsqueda de hiperparámetros de la N-BEATS-1h (`resultados_optuna_nbeats/`) | `05_linea_LSTM.ipynb`, parte 8, y ordenador personal |

## Ficheros que respaldan la memoria

| Memoria | Ficheros |
|---|---|
| Apartado 5.5 y anexo D: búsqueda de hiperparámetros de la LSTM | `resultados/diagnostico/faseC_hiperparametros_local/`: scripts `h1_optuna.py`, `h2_5semillas.py`, `h2plus_bonferroni.py`, `h2e_epocas.py` y `h3_validacion.py`. En `resultados_optuna/`, sus informes: `INFORME.txt` y `H2_resumen.csv` (H1 y H2), `INFORME_H2plus.txt` y `H2plus_bonferroni.csv` (cinco semillas y Bonferroni), `INFORME_H2e.txt` (épocas del intento 27), `H3_validacion/INFORME_H3.txt` (confirmación en validación) y `H1_importancia_hiperparametros.csv` (importancias de la Tabla D1) |
| Apartado 5.5 y anexo D: búsqueda de hiperparámetros de la N-BEATS-1h | `optuna_local/resultados_optuna_nbeats/`: `h1_optuna_nbeats.py`, `INFORME.txt`, `H1_intentos.csv`, `H2_resumen.csv` y `H1_importancia_hiperparametros.csv`. La confirmación en validación del intento 19 está en `resultados/diagnostico/nbeats/H3_confirmacion_optuna/` |
| Apartado 4.9: MAE de E1-30 en el ordenador personal | `resultados/diagnostico/faseC_hiperparametros_local/resultados_optuna/H3_validacion/INFORME_H3.txt` |
| Capítulo 6: evaluación en el test | `resultados/diagnostico/T_test/T3_resultados/`: métricas, comparaciones, desgloses, veredicto y `T3_registro_ejecuciones.txt`, el registro de la evaluación única |
| Apartado 6.4: análisis exploratorio posterior al test | `resultados/diagnostico/T_test/T4_exploratorio/`, con el coste computacional en `T4_F_coste.csv` |
| Anexo E | `02_Modelado/scripts/tablas_anexo_E.py`, que lee `resultados/diagnostico/faseB_radiografia_E1/tabla_horaria_E1.pkl`, `resultados/diagnostico/nbeats/N6b_confirmacion/N6b_predicciones_validacion.pkl`, `resultados/diagnostico/T_test/T3_resultados/T3_predicciones_y_bandas.pkl` y `resultados/diagnostico/T_test/tabla_test.pkl` |

## Detalle de `resultados/diagnostico/`

Cada carpeta corresponde a una o varias fases de la Tabla A1 de la memoria. La correspondencia completa entre fases,
código y apartados está en `docs/fases.md`.

| Carpeta | Fases (Tabla A1) | Cuaderno | Memoria |
|---|---|---|---|
| `diagnostico_LSTM_DIFstats_20261007_150138` | 2 · diagnóstico del modelo de partida | 05, parte 1 | 5.3.2 |
| `fase0_semillas`, `fase1_intrahoraria` | 3 · semillas, E1 y E2 | 05, parte 2 | 5.3.1 y 5.3.3 |
| `fase2_kd`, `fase2_historia`, `fase3_perdida`, `fase4_ventana` | 4 · E3 a E7 | 05, parte 3 | 5.3.4 y Tabla B1 |
| `faseA_rampas`, `faseA_anticipabilidad` | 5 · rampas y fase A | 05, parte 4 | 5.3.2, 5.3.4 y anexo B |
| `faseB_radiografia_E1` | 6 · fase B | 05, parte 5 | 5.3.3 y anexo E |
| `faseC_regularizacion`, `faseC_normalizacion` | 7 a 9 · C5a a C5c y C6 | 05, parte 6 | 5.3.3 y Tabla B1 |
| `depuracion_E1_30` | 10 · P1 a P2b | 05, parte 7 | 5.3.4 y Tabla 13 |
| `faseC_hiperparametros_local` | 11 y 12 · H1 a H3 de la LSTM | Ordenador personal | 5.5 y anexo D |
| `D1_bandas` | 13 · bandas de incertidumbre | 05, parte 9 | Fuera del alcance de la memoria |
| `R1_radiografia_error`, `R2_cielo_anterior`, `faseC_cielo` | 14 y 15 · R1, R2 y C7 | 05, parte 10 | 5.3.2, 5.3.4 y Tabla B1 |
| `R3_patron_error`, `R4_dentro_de_la_hora`, `R5_dif_intrahoraria` | 16 a 18 | 05, parte 11 | 5.3.4 y 5.3.5 |
| `faseC_ultimo_minuto`, `R6_radiografia_V3` | 19 a 21 · C8, C8b, R6 y R6b | 05, parte 12 | 5.3.5, 5.6.1, anexo B.2 y Tabla 13 |
| `R7_epocas_superpuestas`, `H0_horizontes`, `L_entrada_una_hora` | 24 a 28 · R7, H0 y L0 a L3 | 05, parte 13 | 5.3.5 |
| `diagnostico_NBEATS_DIFstats_20261008_212809` | 22 · primera N-BEATS (complementario) | 06, partes 1 y 2 | No se utiliza en la memoria |
| `nbeats/P1_sobreaprendizaje_y_robustez`, `nbeats/P2_validacion` | 22 · P1 y P2 de la N-BEATS | 06, parte 3 | 5.4.1, Tablas 8 y C1 |
| `nbeats/N0_sesgo`, `nbeats/N1_escala_cielo_despejado`, `nbeats/N2_log_y_regularizacion`, `nbeats/N2b_replica_REG` | 29 a 32 | 06, parte 4 | 5.4.2 y Tabla C1 |
| `nbeats/N3_radiografia` | 33 | 06, parte 5 | 5.4.3 y Tabla C1 |
| `nbeats/N6_entrada_una_hora`, `nbeats/N6b_confirmacion` | 34 y 35 · N-BEATS-1h | 06, parte 6 | 5.4.4, 5.6 y 6.1; Tablas 7, 8 y C1; anexo E |
| `nbeats/H3_confirmacion_optuna` | 37 · H3 de la N-BEATS | 06, parte 7 | 5.5 y anexo D |
| `nbeats/N7_bandas` | 38 · bandas de incertidumbre | 06, parte 8 | Fuera del alcance de la memoria |
| `T_test` | 39 a 41 · T0 a T4 | 06, parte 9 | 4.5, capítulo 6, Tabla 13 y anexo E |

## Cómo volver a ejecutar los cuadernos

1. Copiar esta carpeta a Google Drive.
2. Copiar dentro el fichero `datos/radiation_data_completo.parquet` de la raíz del repositorio.
3. En cada cuaderno, apuntar `BASE_DIR` a esa carpeta de Drive.
4. Para los scripts del ordenador personal, seguir las indicaciones de `docs/fases.md`.

## Convenciones de nombres

- Las carpetas de `resultados/diagnostico/` llevan el código de fase de la Tabla A1: por ejemplo, `faseC_regularizacion`
  agrupa C5a a C5c, y `nbeats/N6b_confirmacion` corresponde a N6b.
- La fecha y la hora que aparecen en el nombre de una carpeta o de un ZIP son las de su ejecución.
- En los modelos y las predicciones, `s42`, `s123`, `s2024`, `s7` y `s99` indican la semilla, y `epN`, la época.

## Qué no se incluye

De la carpeta original de Drive se han excluido:

- el parquet minutal, que ya está en `datos/` de la raíz del repositorio;
- las salidas del análisis exploratorio propio del autor (`resultados/diagnostico/EDA_DIF_P90*`), fase 23 de la Tabla A1,
  cuyo código no forma parte del repositorio;
- una ejecución anterior de P1 de la N-BEATS que no usa ningún cuaderno;
- un cuaderno de trabajo sustituido por la parte 1 de `06_linea_NBEATS_y_test.ipynb`, notas personales y dos figuras
  sueltas que no genera ningún cuaderno.

Las reglas de exclusión están en el `.gitignore` de la raíz del repositorio.
