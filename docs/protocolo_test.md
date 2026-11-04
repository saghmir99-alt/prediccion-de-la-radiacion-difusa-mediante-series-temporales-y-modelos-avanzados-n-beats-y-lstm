## Protocolo del test (fijado antes de abrir los datos de octubre y noviembre de 2024)

Este apartado se escribe y se guarda **antes** de evaluar el test. Su objetivo es que la evaluación final no pueda ajustarse a la vista de los resultados.

### 1. Qué se evalúa

| Papel | Modelo | Entrenamiento | Hiperparámetros y época |
|---|---|---|---|
| **Principal** | LSTM E1-30 | train + validación (30-12-2023 → 30-09-2024) | los de E1-30; **30 épocas** |
| **Principal** | N-BEATS-1h | train + validación | los de N6b; **5 épocas** |
| Secundario | LSTM E1-30 | solo train (el validado en C5c) | igual |
| Secundario | N-BEATS-1h | solo train (el validado en N6b) | igual |

- **Igual en todos los modelos:** 5 semillas (42, 123, 2024, 7, 99), y la predicción es la media de las cinco.
- **Fijado de antemano:** en el reentrenamiento con train + validación no se elige nada nuevo. La arquitectura, los hiperparámetros, las épocas y las semillas son los ya validados. Los escaladores se ajustan con train + validación.
- **Lo que no se puede hacer:** el reentrenado no puede validarse de nuevo, porque ya no queda ningún dato limpio. El test es su única evaluación.
- **Para qué sirven los modelos secundarios:** son los que se validaron. Muestran si añadir agosto y septiembre ayudó y protegen frente a un problema inesperado del reentrenamiento. **Las conclusiones se basan en los principales**, y no se elegirá entre principal y secundario a la vista del test.
- **Modelos que no van al test:** V3-50 queda fuera por decisión del autor. Superó la validación (−0,78 W/m²), pero su mecanismo no se pudo explicar (R6, R6b). Se declara como desviación.

### 2. Datos

- **Test:** del 01-10-2024 al 30-11-2024, 1464 horas. Se evalúa solo en las horas con sol (régimen distinto de noche).
- **Construcción de las entradas:** con el mismo código que en train y validación:
  - los 4 canales base y los 5 intrahorarios de E1 para la LSTM;
  - la entrada de una hora de L0 para la N-BEATS.
- **Control de las variables intrahorarias:** se recalculan con la misma función y deben reproducir los valores ya guardados en el periodo común.
- **Preparar no es evaluar:** la preparación de los datos no calcula ningún error.

### 3. Métricas

- MAE (principal), RMSE, sesgo (ŷ − y), nMAE y R².
- Mejora S frente a tres referencias: persistencia simple, persistencia inteligente horaria y persistencia inteligente del último minuto.

**Aviso de interpretación, escrito antes:** octubre y noviembre no tienen banda central (en Cabauw el sol no pasa de ~35° a principios de octubre ni de ~16–24° en noviembre) y la DIF es más baja y varía menos. Por eso el MAE absoluto y el R² no son comparables con los de validación. La lectura principal será la mejora S frente a las persistencias, el nMAE y la comparación por parejas entre modelos.

### 4. Comparación entre arquitecturas

- **Comparación principal (una sola):** N-BEATS-1h − LSTM E1-30, ambos reentrenados con train + validación.
  - Se compara la diferencia de errores absolutos hora a hora.
  - El IC95 se obtiene con bootstrap por días (2000 remuestras, semilla 0).
  - Se añade el Diebold–Mariano diario (Newey–West con L = ⌊n^(1/3)⌋ y corrección HLN).
- **Lectura:**
  - IC entero por debajo de 0: la N-BEATS es mejor.
  - IC entero por encima de 0: la LSTM es mejor.
  - IC que cruza el 0: no se distinguen (lo que no equivale a que sean equivalentes).
- **Comparaciones secundarias (informativas):**
  - cada modelo reentrenado frente a su versión de solo train;
  - la misma comparación entre arquitecturas con los modelos de solo train.

### 5. Desgloses (descriptivos, no deciden)

- Régimen (transición / completamente diurno) y amanecer.
- Bandas de elevación con su número de horas. La central quedará vacía o casi vacía.
- Tipo de cielo de la hora T, usado solo para describir.
- Rampas con el umbral de train (±97,78 W/m²).

### 6. Bandas de incertidumbre (se calculan; su inclusión en la memoria queda a decisión del autor)

- **Método:** el mismo que D1/N7, al 80 % y al 90 %.
- **Calibración:** el modelo de riesgo y el cuantil conformal se recalculan con los errores fuera de muestra de train (P1 y N7) más los errores de validación de los modelos de solo train, que también son errores sobre datos no vistos.
- **Qué se informa:** cobertura global y por grupo, ancho medio y Winkler frente a la banda fija, con los mismos cuatro criterios.

### 7. Reglas de ejecución

- El test se evalúa **una sola vez**, con una única celda que calcula todo lo anterior.
- Si después aparece un error de código, se corrige solo ese error, se vuelve a ejecutar y se declara en la memoria. Nunca se cambian métricas, grupos, modelos ni criterios.
- Todo lo que no figura en este protocolo y se mire después se presentará como análisis exploratorio posterior al test.
