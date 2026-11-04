# BASE_DIR

Copia de la carpeta persistente de Google Drive «03. Modelado», que los cuadernos 03 a 06 llaman `BASE_DIR`. Conserva la
estructura original, que es la que esperan los cuadernos:

- `documentacion/`: `entorno_tfm.py` y `gestor_ejecuciones.py`, que crea el cuaderno 03 y usan los siguientes, e
  inventarios de las ejecuciones.
- `resultados/diagnostico/`: tablas, figuras, predicciones y registros de las líneas LSTM y N-BEATS y del test
  (`nbeats/` y `T_test/`), y la carpeta de la búsqueda de hiperparámetros de la LSTM en el ordenador personal
  (`faseC_hiperparametros_local/`).
- `optuna_local/`: datos exportados para el ordenador personal y búsqueda de hiperparámetros de la N-BEATS-1h
  (`resultados_optuna_nbeats/`).
- En la raíz están, además, la copia del parquet minutal que leen los cuadernos (`radiation_data_completo.parquet`) y
  los ZIP que descargaron las partes 1 a 4 del cuaderno 03.

Para volver a ejecutar los cuadernos, se copia esta carpeta a Google Drive y se apunta `BASE_DIR` a esa copia. El orden
de ejecución y la correspondencia con la memoria están en `docs/fases.md`.
