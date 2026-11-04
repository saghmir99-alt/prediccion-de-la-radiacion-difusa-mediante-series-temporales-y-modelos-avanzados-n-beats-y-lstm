# Punto de reanudación del modelado

Checkpoint creado: 20261006_125123

## Estado

Se han completado los modelos DIF, elevación histórica, geometría objetivo,
SWD, LWD, DIR y las combinaciones SWD+DIR, SWD+FD y SWD+DIR+FD con
N-BEATS y LSTM.

El mejor MAE externo observado es LSTM+SWD:

- MAE: 15.867523 W/m²
- nMAE: 18.6166 %
- RMSE: 32.383934 W/m²
- R²: 0.910285

LSTM+SWD+DIR+FD queda prácticamente empatado en MAE y mejora ligeramente
RMSE y R².

LSTM+DIF+elevación histórica+GeomT presenta el mejor RMSE y R².

El conjunto test no se ha utilizado.

## Próxima reanudación

1. Conectar un runtime nuevo.
2. Montar Google Drive.
3. Ejecutar la celda «RESTAURAR CHECKPOINT».
4. Verificar el mensaje `ENTORNO RESTAURADO CORRECTAMENTE`.
5. Continuar sin repetir entrenamientos.

## Pendiente

- Consolidación definitiva del inventario completo.
- LWU todavía no se ha entrenado.
- Comparación por regímenes reservada para los candidatos finales.