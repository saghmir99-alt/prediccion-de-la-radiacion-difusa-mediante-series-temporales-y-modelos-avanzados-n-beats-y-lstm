# BASE_DIR

Copia de la carpeta persistente de Google Drive «03. Modelado», que los cuadernos 03 a 06 llaman `BASE_DIR`. Conserva la
estructura original, que es la que esperan los cuadernos:

- `documentacion/`: `entorno_tfm.py` y `gestor_ejecuciones.py`, que crea el cuaderno 03 y usan los siguientes, e
  inventarios de las ejecuciones.
- `resultados/ejecuciones/` y `resultados/comparativas/`: ejecuciones y comparaciones de los cuadernos 03 y 04. Los
  cuadernos 05 y 06 toman de aquí el modelo de partida LSTM-DIFstats.
- `resultados/diagnostico/`: tablas, figuras, predicciones, modelos y registros de las líneas LSTM y N-BEATS y del test
  (`nbeats/` y `T_test/`), y la carpeta de la búsqueda de hiperparámetros de la LSTM en el ordenador personal
  (`faseC_hiperparametros_local/`).
- `optuna_local/`: datos exportados para el ordenador personal y búsqueda de hiperparámetros de la N-BEATS-1h
  (`resultados_optuna_nbeats/`).
- `datos/` y `comparativas/`: variables meteorológicas preparadas y comparaciones de la parte 6 del cuaderno 03.
- En la raíz están los ZIP que descargaron las partes 1 a 4 del cuaderno 03 y varias tablas de resultados sueltas.

No se incluyen el parquet minutal, que ya está en `datos/` del repositorio; las salidas del análisis exploratorio propio
del autor (`resultados/diagnostico/EDA_DIF_P90*`, fase 23 de la Tabla A1, cuyo código no forma parte del repositorio); una
ejecución anterior de P1 de la N-BEATS que no usa ningún cuaderno; un cuaderno de trabajo sustituido por la parte 1 de
`06_linea_NBEATS_y_test.ipynb`, y notas personales.

Para volver a ejecutar los cuadernos, se copia esta carpeta a Google Drive, se pone dentro
`datos/radiation_data_completo.parquet` y se apunta `BASE_DIR` a esa copia. El orden de ejecución y la correspondencia
con la memoria están en `docs/fases.md`.
