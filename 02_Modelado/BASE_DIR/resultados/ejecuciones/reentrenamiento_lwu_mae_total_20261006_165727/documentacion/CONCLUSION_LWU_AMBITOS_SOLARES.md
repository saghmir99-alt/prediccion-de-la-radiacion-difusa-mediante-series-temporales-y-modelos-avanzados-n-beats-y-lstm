# Conclusión sobre LWU por ámbito solar

## Resultado principal

La evaluación externa muestra que las métricas globales de los modelos LWU
están fuertemente favorecidas por la inclusión de las horas nocturnas.

### LSTM-LWU

- MAE global: 16.8410 W/m².
- MAE en operación: 27.3957 W/m².
- MAE nocturno: 0.7541 W/m².
- RMSE global: 33.4230 W/m².
- RMSE en operación: 43.0043 W/m².
- RMSE nocturno: 1.0045 W/m².
- La noche aporta únicamente el 1.77 % del error absoluto global.
- La operación aporta el 98.23 %.

### N-BEATS-LWU

- MAE global: 20.1954 W/m².
- MAE en operación: 32.7147 W/m².
- MAE nocturno: 1.1141 W/m².
- RMSE global: 37.9299 W/m².
- RMSE en operación: 48.7920 W/m².
- RMSE nocturno: 1.7197 W/m².
- La noche aporta únicamente el 2.19 % del error absoluto global.
- La operación aporta el 97.81 %.

## Lectura por régimen

El mayor error se concentra en las horas completamente diurnas:

- LSTM-LWU: MAE de 30.7613 W/m².
- N-BEATS-LWU: MAE de 36.2907 W/m².

En transición los errores son menores:

- LSTM-LWU: MAE de 5.7595 W/m².
- N-BEATS-LWU: MAE de 9.7264 W/m².

La noche presenta errores absolutos muy reducidos y un sesgo próximo a cero,
especialmente en LSTM. El R² nocturno no debe utilizarse para valorar estos
modelos, ya que la variabilidad del target durante la noche es extremadamente
pequeña y su media es próxima a cero.

## Conclusión metodológica

La mejora aparente de las métricas globales de LWU se explica en gran medida
por la facilidad de las horas nocturnas. Cuando se excluye la noche y se evalúa
únicamente el periodo de operación, los errores aumentan notablemente.

Por tanto, las métricas globales no son suficientes para valorar la capacidad
predictiva útil de los modelos. En la comparación final deberá darse prioridad
al rendimiento durante operación y, dentro de ella, al régimen completamente
diurno.

Estos resultados todavía no demuestran que LWU aporte una mejora nocturna
respecto a SWD, LWD, DIR u otros bloques. Esa afirmación requerirá comparar los
mismos regímenes entre modelos y queda reservada para el análisis detallado de
los candidatos finales.

El conjunto test no se ha utilizado.
