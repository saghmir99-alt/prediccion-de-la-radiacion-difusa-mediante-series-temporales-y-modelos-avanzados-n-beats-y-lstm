# Fases del trabajo, código y apartados de la memoria

La memoria sigue un orden lógico. Este documento da el orden real en que se ejecutaron las fases (Tabla A1 de la
memoria), los datos con los que se decidió en cada una, dónde está su código y en qué apartados de la memoria se tratan.
Dentro de cada cuaderno, la nota del principio de cada parte indica a qué apartados y fases de la Tabla A1 corresponde.

## Estructura

```
01_EDA/                         análisis exploratorio (cuadernos 00 a 02 y scripts del cuaderno 02)
02_Modelado/                    cuadernos de modelado 03 a 06
02_Modelado/BASE_DIR/           copia de la carpeta de Google Drive «03. Modelado», que los cuadernos 03 a 06 llaman BASE_DIR
02_Modelado/scripts/            tablas del anexo E, calculadas con las predicciones guardadas
configuracion/parametros.yaml   semillas, particiones, criterios y configuración de los modelos
datos/                          datos de la BSRN (ver docs/datos.md)
docs/                           este documento, los datos y el protocolo del test
```

## Orden de ejecución

1. `01_EDA/00_carga_y_analisis_univariante.ipynb` une los 13 ficheros mensuales en `radiation_data_completo.parquet`
   (ver `docs/datos.md`).
2. `01_EDA/01_analisis_exploratorio_DIF.ipynb` y `01_EDA/02_Multivariante/02_analisis_exploratorio_multivariante.ipynb`
   (con `scripts_02/`): análisis exploratorio del apartado 5.1.
3. `02_Modelado/03_modelado.ipynb`, `04_modelado_combinaciones.ipynb`, `05_linea_LSTM.ipynb` y
   `06_linea_NBEATS_y_test.ipynb`, en ese orden. Las partes 1 a 4 del cuaderno 03 trabajan con ficheros subidos a la
   sesión y descargan un ZIP por etapa; desde la parte 5, los cuadernos leen y escriben en `BASE_DIR`. Para volver a
   ejecutarlos se copia `02_Modelado/BASE_DIR/` a Google Drive, se pone dentro `datos/radiation_data_completo.parquet` y se
   apunta `BASE_DIR` a esa copia.
4. La búsqueda de hiperparámetros (fases 11, 12 y 36) se ejecutó en el ordenador personal con los scripts que se
   describen más abajo, intercalada con los cuadernos 05 y 06 en el orden de la tabla.

Los errores del test (octubre y noviembre de 2024) se calculan una sola vez, en la subsección 9.5 del cuaderno 06, con
el protocolo de `docs/protocolo_test.md`; las subsecciones 9.2 a 9.4 preparan las entradas y las predicciones sin calcular
ningún error. La celda de la 9.5 lleva un registro de ejecuciones y se detiene si se vuelve a lanzar sin declarar el
motivo.

## Fases

| N.º | Fase | Datos usados para decidir | Decisión o resultado | Código | Memoria |
|---|---|---|---|---|---|
| 1 | Preparación, persistencias, variantes iniciales y LSTM-DIFstats | División interna de entrenamiento; validación | LSTM-DIFstats como base | `03_modelado.ipynb` y `04_modelado_combinaciones.ipynb` | 4.2–4.6, 5.2 y 5.3.1; Tablas 6 y 11 |
| 2 | Diagnóstico inicial (retraso, rampas, memoria) | Validación | Hipótesis de variables intrahorarias | `05_linea_LSTM.ipynb`, parte 1 | 5.3.2 |
| 3 | Fase 0 (semillas) y fase 1 (E1, E2) | Validación | E1 como nueva base | `05_linea_LSTM.ipynb`, parte 2 | 5.3.1 y 5.3.3 |
| 4 | E3–E7 (fracción difusa, historia diaria, pérdidas, ventana) | Validación | Todas rechazadas; regla de daño fijada tras E6 | `05_linea_LSTM.ipynb`, parte 3 | 5.3.4 y Tabla B1 |
| 5 | Rampas (umbral con entrenamiento) y fase A con familias de variables | Entrenamiento (validación cruzada por meses) y validación | Ninguna variable anticipa las rampas | `05_linea_LSTM.ipynb`, parte 4 | 5.3.2, 5.3.4 y anexo B |
| 6 | Fase B (presupuesto de error, control con persistencia, predicciones fuera de muestra) | Validación y entrenamiento fuera de muestra | Sobreajuste: regularizar o parar antes | `05_linea_LSTM.ipynb`, parte 5 | 5.3.3 |
| 7 | C5a–C5b (curvas de aprendizaje y criba; efecto mínimo de 0,5 W/m²) | Solo entrenamiento | Época 30 | `05_linea_LSTM.ipynb`, parte 6 | 5.3.3 y Tabla B1 |
| 8 | C5c (confirmación) | Validación, una vez | E1-30, modelo de referencia | `05_linea_LSTM.ipynb`, parte 6 | 5.3.3 y Tabla B1 |
| 9 | C6 (normalización física) | Solo entrenamiento | Descartada | `05_linea_LSTM.ipynb`, parte 6 | 5.3.3 y Tabla B1 |
| 10 | P1–P2b (fuera de muestra y permutación; límite del 4 % en el amanecer) | Solo entrenamiento | Se mantienen las 24 h | `05_linea_LSTM.ipynb`, parte 7 | 5.3.4 y Tabla 13 |
| 11 | H1–H2 de la LSTM (Optuna, cinco semillas, Bonferroni, épocas) | Solo entrenamiento | Intento 27 a confirmación | Scripts del PC en `BASE_DIR/resultados/diagnostico/faseC_hiperparametros_local/`: `h1_optuna.py`, `h2_5semillas.py`, `h2plus_bonferroni.py` y `h2e_epocas.py` (datos exportados en la parte 8 de `05_linea_LSTM.ipynb`) | 5.5 y anexo D |
| 12 | H3 de la LSTM (confirmación) | Validación, una vez | Intento 27 rechazado | Script del PC `h3_validacion.py`, en la misma carpeta | 5.5 y anexo D |
| 13 | D1 (bandas de incertidumbre, fuera del alcance de esta memoria) | Entrenamiento fuera de muestra y validación | Bandas aceptadas en validación; no cambian los modelos | `05_linea_LSTM.ipynb`, parte 9 | Fuera del alcance de la memoria |
| 14 | R1–R2 (radiografía del error, cielo previo) | Entrenamiento fuera de muestra y validación | Señal pequeña tras cielo cubierto | `05_linea_LSTM.ipynb`, parte 10 | 5.3.2, 5.3.4 y Tabla B1 |
| 15 | C7 (cielo previo como entrada) | Solo entrenamiento | Descartado | `05_linea_LSTM.ipynb`, parte 10 | 5.3.2, 5.3.4 y Tabla B1 |
| 16 | R3 (69 variables frente al error) | Entrenamiento fuera de muestra y validación | Sin patrón | `05_linea_LSTM.ipynb`, parte 11 | 5.3.4 y 5.3.5 |
| 17 | R4 (tipos de fallo grande) | Validación (descriptivo) | Los fallos grandes empiezan al inicio de la hora objetivo | `05_linea_LSTM.ipynb`, parte 11 | 5.3.4 y 5.3.5 |
| 18 | R5 (DIF intrahoraria y último minuto) | Entrenamiento fuera de muestra y validación | El último minuto aporta: se prueba C8 | `05_linea_LSTM.ipynb`, parte 11 | 5.3.4 y 5.3.5 |
| 19 | C8 (criba de V3) | Solo entrenamiento | V3 pasa a validación (época 50) | `05_linea_LSTM.ipynb`, parte 12 | 5.3.5, 5.6.1, anexo B.2 y Tabla 13 |
| 20 | C8b (confirmación de V3-50) | Validación, una vez | V3-50 confirmada; su adopción se decide tras R6–R6b | `05_linea_LSTM.ipynb`, parte 12 | 5.3.5, 5.6.1, anexo B.2 y Tabla 13 |
| 21 | R6–R6b (radiografía de V3 y explicación) | Validación | Hipótesis y explicación no confirmadas | `05_linea_LSTM.ipynb`, parte 12 | 5.3.5, 5.6.1, anexo B.2 y Tabla 13 |
| 22 | Primera N-BEATS, P1 y P2 de la N-BEATS | Entrenamiento y validación | N-BEATS-DIFstats, época 15 | `06_linea_NBEATS_y_test.ipynb`, partes 1 a 3 (las partes 1 y 2 son complementarias) | 5.4.1, Tabla 8 y Tabla C1 |
| 23 | Análisis exploratorio del autor (minutal y cruce de horas) | Solo entrenamiento | Hipótesis: últimos minutos y variables externas | No incluido: análisis exploratorio propio del autor, sin criterios previos; sus resultados se trataron como hipótesis y se contrastaron después (R7 y H0) | Anexo B.3 |
| 24 | R7 (épocas superpuestas, Spearman parcial) | Solo entrenamiento (validación, descriptiva) | Señal sobre el cambio, no sobre el error de E1-30 | `05_linea_LSTM.ipynb`, parte 13 | 5.3.5 |
| 25 | H0 (horizontes de 60, 30 y 15 min) | Solo entrenamiento | La estación no aporta en ningún horizonte | `05_linea_LSTM.ipynb`, parte 13 | 5.3.5 |
| 26 | L0–L1 (entrada de una hora; HGB) | Entrenamiento; validación una vez como referencia | El HGB iguala a E1-30 | `05_linea_LSTM.ipynb`, parte 13 | 5.3.5 |
| 27 | L2 (criba de la LSTM de una hora, A y B) | Solo entrenamiento | A pasa; B no aporta sobre A | `05_linea_LSTM.ipynb`, parte 13 | 5.3.5 |
| 28 | L3 (confirmación de A) | Validación, una vez | No se confirma | `05_linea_LSTM.ipynb`, parte 13 | 5.3.5 |
| 29 | N0 (origen del sesgo de la N-BEATS) | Entrenamiento fuera de muestra y validación | Defecto del modelo, no del periodo | `06_linea_NBEATS_y_test.ipynb`, parte 4 | 5.4.2 y Tabla C1 |
| 30 | N1 (escala de cielo despejado de t+1) | Solo entrenamiento | No pasa | `06_linea_NBEATS_y_test.ipynb`, parte 4 | 5.4.2 y Tabla C1 |
| 31 | N2 (escala logarítmica y dropout) | Solo entrenamiento | No pasa; el dropout, cerca del umbral | `06_linea_NBEATS_y_test.ipynb`, parte 4 | 5.4.2 y Tabla C1 |
| 32 | N2b (réplica del dropout en febrero, abril y junio) | Solo entrenamiento | No se confirma | `06_linea_NBEATS_y_test.ipynb`, parte 4 | 5.4.2 y Tabla C1 |
| 33 | N3 (radiografía comparada con la LSTM) | Entrenamiento fuera de muestra y validación | Probar la entrada de una hora | `06_linea_NBEATS_y_test.ipynb`, parte 5 | 5.4.3 y Tabla C1 |
| 34 | N6 (criba de la entrada de una hora) | Solo entrenamiento | Pasa a validación (época 5) | `06_linea_NBEATS_y_test.ipynb`, parte 6 | 5.4.4, 5.6 y 6.1; Tablas 7, 8 y C1 |
| 35 | N6b (confirmación) | Validación, una vez | N-BEATS-1h, N-BEATS final | `06_linea_NBEATS_y_test.ipynb`, parte 6 | 5.4.4, 5.6 y 6.1; Tablas 7, 8 y C1 |
| 36 | H1–H2 de la N-BEATS (Optuna) | Solo entrenamiento | Intento 19 a confirmación | Script del PC `BASE_DIR/optuna_local/resultados_optuna_nbeats/h1_optuna_nbeats.py` | 5.5 y anexo D |
| 37 | H3 de la N-BEATS (confirmación) | Validación, una vez | No se confirma | `06_linea_NBEATS_y_test.ipynb`, parte 7 | 5.5 y anexo D |
| 38 | N7 (bandas de incertidumbre de la N-BEATS, fuera del alcance de esta memoria) | Entrenamiento fuera de muestra y validación | Bandas casi idénticas a las de la LSTM; no cambian los modelos | `06_linea_NBEATS_y_test.ipynb`, parte 8 | Fuera del alcance de la memoria |
| 39 | Protocolo del test y T0–T2 (inventario, entradas y predicciones, sin calcular errores) | Entrenamiento y validación | Modelos principales y secundarios cerrados | `06_linea_NBEATS_y_test.ipynb`, subsecciones 9.1 a 9.4 y `docs/protocolo_test.md` | 4.5, capítulo 6, Tabla 13 y anexo E |
| 40 | T3 (evaluación única del test) | Test, una vez | N-BEATS-1h con menos error que E1-30 | `06_linea_NBEATS_y_test.ipynb`, subsección 9.5 | 4.5, capítulo 6, Tabla 13 y anexo E |
| 41 | T4 (análisis exploratorio posterior al test) | Todo el periodo (descriptivo) | No cambia ningún veredicto | `06_linea_NBEATS_y_test.ipynb`, subsección 9.6 | 4.5, capítulo 6, Tabla 13 y anexo E |

Las fases 1 a 6 eligieron con la validación; desde la 7, las decisiones se tomaron con entrenamiento y la validación solo
confirmó. Las fases 13 y 38 (bandas de incertidumbre) se ejecutaron, pero quedan fuera del alcance de la memoria.

## Scripts del ordenador personal

Están dentro de `02_Modelado/BASE_DIR/`, junto a los datos que leen y a sus resultados. Se ejecutan con
`python <script>` en un entorno con `requirements-pc.txt`; si se interrumpen, se vuelven a lanzar y continúan donde se
quedaron.

- **LSTM** (`resultados/diagnostico/faseC_hiperparametros_local/`): `h1_optuna.py` (H1 y H2), `h2_5semillas.py` (fase
  con cinco semillas), `h2plus_bonferroni.py` (ajuste de Bonferroni), `h2e_epocas.py` (épocas del intento 27) y
  `h3_validacion.py` (H3). Se ejecutan desde esa carpeta, que contiene `datos/` y `datos_validacion/` (exportados en la
  parte 8 de `05_linea_LSTM.ipynb`); los resultados y los informes quedan en `resultados_optuna/`.
- **N-BEATS-1h** (`optuna_local/resultados_optuna_nbeats/`): `h1_optuna_nbeats.py` (H1 y H2), con sus resultados y su
  informe. Para volver a ejecutarlo, se copia el script a `optuna_local/` y se pone en `optuna_local/datos/`, junto a
  `tabla_train.csv`, el fichero `entrada_una_hora.npz`, que genera la parte 13 de `05_linea_LSTM.ipynb` en
  `resultados/diagnostico/L_entrada_una_hora/`. La confirmación en validación (H3) está en la parte 7 de
  `06_linea_NBEATS_y_test.ipynb`.

Los dos scripts de H1 fijan 30 intentos. Como cuatro trabajadores evalúan intentos en paralelo, los que estaban en marcha
al llegar a 30 también terminan; por eso los informes registran 32, que son los que recoge la memoria.
