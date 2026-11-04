# Datos originales

## Fuente

- **Red:** Baseline Surface Radiation Network (BSRN), red de referencia de la Organización Meteorológica Mundial para la
  medida de la radiación en superficie (Driemel et al., 2018).
- **Estación:** Cabauw (código CAB), Países Bajos (51,968° N; 4,928° E), operada por el Real Instituto Meteorológico de los
  Países Bajos (KNMI).
- **Serie de PANGAEA:** Knap, W. (2022). *Basic and other measurements of radiation at station Cabauw (2005-02 et seq)*
  [serie de conjuntos de datos]. Koninklijk Nederlands Meteorologisch Instituut, PANGAEA.
  https://doi.org/10.1594/PANGAEA.940531
- **Periodo usado:** 13 conjuntos mensuales, de noviembre de 2023 a noviembre de 2024, con resolución de un minuto.

## Copia incluida en el repositorio

- `datos/originales/`: los 13 ficheros mensuales tal como se descargaron de PANGAEA.
- `datos/radiation_data_completo.parquet`: los 13 ficheros unidos por `01_EDA/00_carga_y_analisis_univariante.ipynb`. Los
  cuadernos 03 a 06 leen una copia del mismo fichero desde `BASE_DIR` (`02_Modelado/BASE_DIR/`).

Son datos de la BSRN. Su uso está sujeto a las condiciones de la BSRN
(https://bsrn.awi.de/data/conditions-of-data-release) y exige citar el conjunto de datos como se indica al final. Los
datos originales se obtienen directamente de PANGAEA, como se explica a continuación.

## Descarga desde la fuente original

1. Abrir la serie de PANGAEA (enlace anterior) y localizar los conjuntos mensuales de Cabauw de 2023-11 a 2024-11.
2. Descargar cada uno en formato de texto separado por tabuladores (`.tab`). PANGAEA puede pedir un registro gratuito y la
   aceptación de la política de datos de la BSRN.
3. Guardar los 13 ficheros con estos nombres, que son los que usó el trabajo:

```
CAB_radiation_2023-11.tab   CAB_radiation_2024-04.tab   CAB_radiation_2024-09.tab
CAB_radiation_2023-12.tab   CAB_radiation_2024-05.tab   CAB_radiation_2024-10.tab
CAB_radiation_2024-01.tab   CAB_radiation_2024-06.tab   CAB_radiation_2024-11.tab
CAB_radiation_2024-02.tab   CAB_radiation_2024-07.tab
CAB_radiation_2024-03.tab   CAB_radiation_2024-08.tab
```

4. Ejecutar `01_EDA/00_carga_y_analisis_univariante.ipynb`, que une los ficheros en `radiation_data_completo.parquet`, y
   colocar ese fichero en `BASE_DIR` para los cuadernos 03 a 06.

## Formato de `radiation_data_completo.parquet`

570.240 registros minutales y 31 columnas, sin marcas de tiempo duplicadas ni minutos omitidos en el índice temporal:

| Columnas | Contenido |
|---|---|
| `Date/Time` | marca de tiempo del minuto (UTC) |
| `Height [m]` | altura de medida (2 m, constante; no se usa) |
| `SWD`, `DIR`, `DIF`, `LWD`, `SWU`, `LWU` | para cada magnitud, media del minuto (`[W/m**2]`) y su desviación típica, mínimo y máximo dentro del minuto (`std dev`, `min`, `max`) |
| `T2 [°C]`, `RH [%]`, `PoPoPoPo [hPa]` | temperatura del aire, humedad relativa y presión |
| `source_file`, `year_month` | fichero de origen y mes |

SWD es la global, DIR la directa normal, DIF la difusa (variable objetivo), SWU la reflejada, y LWD y LWU la onda larga
descendente y ascendente.

## Calidad y tratamiento

- El 0,61 % de las celdas está vacío; las ausencias se concentran en DIR y DIF (2,16 % y 2,04 % de los minutos).
- El hueco principal va del 20 al 27 de diciembre de 2023 (11.077 minutos). No se interpola: la serie horaria de trabajo
  empieza el 28 de diciembre de 2023.
- Los valores nocturnos ligeramente negativos de la DIF (*offset* térmico de los piranómetros) se conservan sin recortar.
- Cada hora se etiqueta por su inicio y su valor es la media de los minutos válidos del intervalo [t, t + 60 min); una
  media horaria solo es válida con al menos 30 minutos válidos.

## Cita

- Knap, W. (2022). *Basic and other measurements of radiation at station Cabauw (2005-02 et seq)* [serie de conjuntos de
  datos]. Koninklijk Nederlands Meteorologisch Instituut, PANGAEA. https://doi.org/10.1594/PANGAEA.940531
- Driemel, A., Augustine, J., Behrens, K., Colle, S., Cox, C., Cuevas-Agulló, E., … König-Langlo, G. (2018). Baseline
  Surface Radiation Network (BSRN): Structure and data description (1992–2017). *Earth System Science Data, 10*(3),
  1491–1501. https://doi.org/10.5194/essd-10-1491-2018
