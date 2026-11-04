# Plan de cribado de bloques y optimización de modelos

Fecha de creación: 2026-10-06T11:31:14.791072+02:00

## Objetivo

Seleccionar de forma reproducible los bloques de variables históricas que
aportan información a N-BEATS y LSTM para la predicción horaria de irradiancia
difusa, manteniendo reservado el conjunto de test hasta la comparación final.

## Estado de partida

Se dispone de resultados comparables para:

- N-BEATS con DIF y elevación histórica.
- LSTM con DIF y elevación histórica.
- N-BEATS con DIF, elevación y bloque SWD.
- LSTM con DIF, elevación y bloque SWD.

El bloque SWD completo mejora LSTM en validation, especialmente en el régimen
operacional, las horas completamente diurnas y Q5. El mismo bloque perjudica
N-BEATS en su arquitectura actual. Esto demuestra que la utilidad de una
variable puede depender de la arquitectura.

## Principio metodológico

La selección de variables se realizará antes de la optimización arquitectónica.

Durante el cribado se mantendrán congeladas las arquitecturas actuales y se
utilizarán las mismas particiones, semillas, escalado, optimizador y criterio
de selección.

La época se seleccionará mediante el MAE total de validación interna. Las
métricas operacionales, diurnas, de transición, Q5, dirección y sesgos de
rampa se conservarán como diagnósticos secundarios.

## Etapa 1: aportación marginal de cada bloque

Cada bloque se añadirá individualmente a la base común:

DIF histórica + elevación histórica + bloque candidato

Cada bloque se probará por separado en:

- N-BEATS.
- LSTM.

En esta etapa solo se utilizarán subtrain y validación interna. No se
reentrenará con todo train ni se consultará validation externa después de
cada bloque.

Bloques que deben inventariarse y verificarse:

- SWD.
- LWD.
- LWU.
- DIR.
- SWU.
- Albedo.
- Humedad relativa.
- Presión.
- Otros bloques justificados por el EDA y disponibles en los datos.

## Etapa 2: selección reducida y combinaciones

Una vez finalizado el cribado individual, cada arquitectura conservará como
máximo dos o tres bloques candidatos. Se estudiarán pocas combinaciones
acumulativas y previamente definidas.

No se realizará una búsqueda combinatoria exhaustiva.

## Etapa 3: robustez interna

Las configuraciones internas finalistas se confirmarán con varias semillas.
Las decisiones no se basarán en una única ejecución si las diferencias entre
variantes son pequeñas.

## Etapa 4: optimización arquitectónica limitada

Con las entradas ya fijadas se evaluará un presupuesto pequeño y equivalente
de configuraciones arquitectónicas para N-BEATS y LSTM.

El objetivo no se formulará como óptimo global, sino como mejor configuración
encontrada bajo un protocolo y presupuesto de búsqueda definidos.

## Etapa 5: evaluación externa

Se seleccionará una configuración final de cada familia. Cada modelo se
reinicializará y reentrenará desde cero con todo train durante el número de
épocas seleccionado internamente.

Validation externa se utilizará únicamente para comparar configuraciones ya
fijadas, no para continuar ajustando hiperparámetros.

## Etapa 6: test final

Después de congelar HGB, N-BEATS y LSTM, el conjunto de test se abrirá una
sola vez para la comparación final.

## Comparaciones que se conservarán

1. Comparación controlada N-BEATS frente a LSTM con DIF + elevación.
2. Comparación controlada N-BEATS frente a LSTM con DIF + elevación + SWD.
3. Comparación entre la mejor variante final de cada arquitectura.
4. Comparación final con HGB.
5. Coste computacional y complejidad de cada finalista.

## Calendario de referencia

- 6–9 de octubre: cribado interno de bloques.
- 9–10 de octubre: combinaciones reducidas y robustez.
- 10–11 de octubre: optimización arquitectónica limitada.
- 11–12 de octubre: reentrenamiento y validation externa.
- 12–13 de octubre: congelación y test final.
- Hasta el 18 de octubre: redacción, revisión y anexos.

## Restricciones

- No utilizar test durante el desarrollo.
- No calcular escaladores fuera de train.
- No modificar ni sobrescribir resultados anteriores.
- No inferir que un bloque ayuda a una arquitectura porque ayude a otra.
- Registrar configuraciones, semillas, épocas, predicciones, métricas,
  tiempos, parámetros, checkpoints y manifiestos.
