# Predicción de la radiación difusa mediante series temporales y modelos avanzados N-BEATS y LSTM

**Trabajo Fin de Máster** · Máster Universitario en Big Data y Ciencia de Datos · Universidad Internacional de Valencia (VIU) ·
Curso 2025–2026

**Autor:** Sáber Aghmir Aghmir · **Director:** Gonzalo Surribas Sayago

Este repositorio reúne el código, los cuadernos, los resultados y la documentación del Trabajo Fin de Máster. La memoria
expone el planteamiento, la metodología y la discusión; aquí está lo necesario para comprobar cada cifra que presenta y para
reproducir el trabajo.

## El proyecto

La radiación solar difusa es la parte de la irradiancia que llega al suelo después de dispersarse en la atmósfera y en las
nubes. Predecirla a corto plazo es relevante para estimar la producción de los módulos fotovoltaicos inclinados y para
integrar la energía solar en la red eléctrica, sobre todo en climas nubosos.

El trabajo compara dos arquitecturas de redes neuronales profundas, una **LSTM** y una **N-BEATS**, para predecir con una
hora de antelación la irradiancia difusa horizontal (DIF) media horaria en la estación de **Cabauw (Países Bajos)** de la
Baseline Surface Radiation Network (BSRN), con datos minutales de noviembre de 2023 a noviembre de 2024.

Las dos redes compiten en igualdad de condiciones: mismos datos, particiones, métricas, referencias y procedimiento de
selección. Una vez consolidado el protocolo, las decisiones se toman con validación cruzada en entrenamiento, la
validación solo las confirma y el test se evalúa una sola vez.

| Conjunto | Periodo (hora objetivo) | Horas | Uso |
|---|---|---|---|
| Entrenamiento | 30-12-2023 a 31-07-2024 | 5.149 (3.027 con sol) | Ajuste; selección de épocas, variables e hiperparámetros mediante validación cruzada |
| Validación | 01-08-2024 a 30-09-2024 | 1.464 (884 con sol) | Confirmación, una sola vez, de cada decisión tomada con entrenamiento |
| Test | 01-10-2024 a 30-11-2024 | 1.464 (646 con sol) | Evaluación única, con un protocolo escrito antes de abrirlo |

Los modelos se evalúan en las horas con sol y por régimen solar. Se comparan con cinco persistencias, entre ellas una
basada en el último minuto disponible, y con modelos Ridge y de gradient boosting (HGB). Las diferencias se contrastan
con bootstrap por días y con la prueba de Diebold–Mariano.

## Resultados principales

| Modelo final | Entrada | MAE en validación | MAE en test |
|---|---|---|---|
| LSTM E1-30 | Ventana de 24 horas con nueve canales horarios | 23,25 W/m² | 14,77 W/m² |
| N-BEATS-1h | Última hora en bloques de 5 minutos; corrige la persistencia del último minuto | 23,24 W/m² | 12,93 W/m² |

- En validación, las dos configuraciones finales tienen un error casi idéntico, un 16 % menor que la persistencia del
  último minuto.
- En el test, la N-BEATS-1h obtiene 1,84 W/m² menos de error que la LSTM E1-30 (IC95 [−2,72; −0,94]). El resultado se
  refiere a esas dos configuraciones, que usan entradas distintas, y no permite afirmar que una arquitectura sea superior
  a la otra.
- Las demás variables de la estación y la búsqueda de hiperparámetros no aportan mejoras confirmadas, y no se justifica
  combinar los modelos. La receta de entrenamiento de la N-BEATS-1h es unas diez veces más rápida.
- Los resultados se limitan a una estación, un horizonte de una hora y un test de dos meses de otoño.

## Estructura del repositorio

```
.
├── README.md
├── requirements-colab.txt          dependencias del entorno de Google Colab
├── requirements-pc.txt             dependencias del ordenador personal
├── configuracion/
│   └── parametros.yaml             semillas, particiones, criterios y configuración de los modelos
├── datos/
│   ├── originales/                 13 ficheros mensuales de la BSRN (Cabauw, 2023-11 a 2024-11)
│   └── radiation_data_completo.parquet
├── docs/
│   ├── fases.md                    orden real de las fases, código de cada una y apartados de la memoria
│   ├── datos.md                    origen, descarga, formato y cita de los datos
│   └── protocolo_test.md           protocolo del test, fijado antes de abrirlo
├── 01_EDA/                         análisis exploratorio
│   ├── 00_carga_y_analisis_univariante.ipynb
│   ├── 01_analisis_exploratorio_DIF.ipynb
│   └── 02_Multivariante/
│       ├── 02_analisis_exploratorio_multivariante.ipynb
│       ├── scripts_02/             scripts de las subsecciones 4.2.10 a 6.4 del cuaderno 02
│       └── resultados_02/          resultados de cada etapa del cuaderno 02
└── 02_Modelado/                    modelado, líneas LSTM y N-BEATS y test
    ├── 03_modelado.ipynb
    ├── 04_modelado_combinaciones.ipynb
    ├── 05_linea_LSTM.ipynb
    ├── 06_linea_NBEATS_y_test.ipynb
    ├── scripts/                    cálculo de las tablas del anexo E
    └── BASE_DIR/                   resultados de los cuadernos 03 a 06 y scripts del ordenador personal
```

## Cuadernos y memoria

| Cuaderno | Contenido | Memoria |
|---|---|---|
| `00_carga_y_analisis_univariante` | Unión de los 13 ficheros mensuales, integridad temporal y análisis univariante | 4.2 |
| `01_analisis_exploratorio_DIF` | Calidad, distribución y estructura temporal de la DIF | 5.1 |
| `02_analisis_exploratorio_multivariante` | Tabla horaria, particiones, geometría solar y relaciones con las demás variables | Capítulos 4 y 5 |
| `03_modelado` | Conjunto supervisado, persistencias, Ridge, HGB, primeras redes y cribado de bloques de variables | 4.2 a 4.6, 5.2 y 5.3.1 |
| `04_modelado_combinaciones` | Combinaciones de los estadísticos de la DIF con otros bloques de variables | 5.3.1 |
| `05_linea_LSTM` | Diagnóstico y mejora de la LSTM hasta E1-30, variante V3-50 y LSTM de una hora | 5.3, 5.6.1 y anexo B |
| `06_linea_NBEATS_y_test` | Línea N-BEATS hasta la N-BEATS-1h y evaluación única en el test | 5.4 a 5.6, capítulo 6 y anexos C y E |
| Scripts del ordenador personal | Búsqueda de hiperparámetros con Optuna y confirmación del candidato de la LSTM | 5.5 y anexo D |

Los cuadernos conservan las salidas de su ejecución original, con las tablas de resultados y los veredictos de cada fase.
En los cuadernos 01 a 06, las notas de cada parte indican en qué apartados de la memoria se utiliza; las partes que no se
utilizan están marcadas como complementarias. La correspondencia completa entre las fases de la Tabla A1 de la memoria, su código y sus
apartados está en [`docs/fases.md`](docs/fases.md), y el contenido de `BASE_DIR` se describe en su propio
[`README`](02_Modelado/BASE_DIR/README.md).

## Reproducción

**Entornos** (siempre sin GPU):

- **Google Colab** (CPU; Python 3.13.16, PyTorch 2.11.0): preparación de datos, análisis exploratorio, entrenamiento y
  validación de las dos redes, y test. Dependencias en `requirements-colab.txt`.
- **Ordenador personal** (Windows; Python 3.11, torch 2.14.1, Optuna 5.0.0): búsqueda de hiperparámetros de las dos redes
  y confirmación en validación del candidato de la LSTM. Dependencias en `requirements-pc.txt`.

**Pasos:**

1. Ejecutar `01_EDA/00_carga_y_analisis_univariante.ipynb` con los 13 ficheros de `datos/originales/`, que genera
   `radiation_data_completo.parquet`.
2. Ejecutar los cuadernos `01` y `02`. El `02` usa los scripts de `scripts_02/`.
3. Copiar `02_Modelado/BASE_DIR/` a Google Drive, poner dentro `datos/radiation_data_completo.parquet` y apuntar `BASE_DIR`
   a esa carpeta en los cuadernos `03` a `06`, que se ejecutan en ese orden.
4. Ejecutar los scripts del ordenador personal desde sus carpetas, como se indica en `docs/fases.md`.
5. Recalcular las tablas del anexo E con `python 02_Modelado/scripts/tablas_anexo_E.py 02_Modelado/BASE_DIR`.

Las confirmaciones, los modelos finales y el test usan cinco semillas (42, 123, 2024, 7 y 99), y la predicción es la media
de las cinco; las cribas en entrenamiento usan una o dos. Los algoritmos de PyTorch están fijados como deterministas. Las semillas, las particiones, los criterios de decisión y la
configuración de cada modelo están reunidos en `configuracion/parametros.yaml`.

## Datos

Los datos proceden de la estación de Cabauw de la BSRN, publicados en PANGAEA por Knap (2022). El repositorio incluye una
copia de los 13 ficheros mensuales usados. Su uso está sujeto a las
[condiciones de la BSRN](https://bsrn.awi.de/data/conditions-of-data-release). El origen, la descarga desde la fuente
original, el formato y la cita se describen en [`docs/datos.md`](docs/datos.md).

## Cómo citar

> Aghmir Aghmir, S. (2026). *Predicción de la radiación difusa mediante series temporales y modelos avanzados N-BEATS y
> LSTM* [Trabajo Fin de Máster, Universidad Internacional de Valencia].
> https://github.com/saghmir99-alt/prediccion-de-la-radiacion-difusa-mediante-series-temporales-y-modelos-avanzados-n-beats-y-lstm

Datos:

> Knap, W. (2022). *Basic and other measurements of radiation at station Cabauw (2005-02 et seq)* [serie de conjuntos de
> datos]. Koninklijk Nederlands Meteorologisch Instituut, PANGAEA. https://doi.org/10.1594/PANGAEA.940531
