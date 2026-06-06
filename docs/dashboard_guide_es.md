# Guia del Dashboard — Estudiante

## Inicio de Sesion

Tu instructor proporcionara el email y contrasena de tu equipo. Ingresa a la URL del dashboard e inicia sesion. Llegaras a la pagina **Upload Predictions**.

Si olvidaste tu contrasena, contacta a tu instructor para restablecerla.

---

## Subir Predicciones

### Envios de Competencia

1. Ve a **Upload Predictions** → pestana **Competition (Test Set)**.
2. Selecciona la ronda para la cual estas enviando (solo se muestran las rondas abiertas).
3. Sube un CSV con exactamente dos columnas: `zpid` y `predicted_price` (5,038 filas que coincidan con el test set). Los precios deben estar en dolares, no transformados con logaritmo.
4. El dashboard valida tu archivo: columnas correctas, todos los zpids presentes, sin valores faltantes ni negativos.
5. Haz clic en **Submit Predictions**.

Puedes volver a subir para la misma ronda tantas veces como quieras mientras la ronda este abierta. Una vez que el organizador bloquea una ronda, los envios se cierran.

Si solo envias para la ronda 1, tus predicciones de la ronda 1 se usan automaticamente (forward-fill) para cualquier ronda posterior que no hayas enviado.

### Envios de Practica (OOF)

1. Ve a **Upload Predictions** → pestana **Practice (Train Set OOF)**.
2. Sube predicciones out-of-fold sobre el **training set** (11,840 filas, mismo formato `zpid, predicted_price`).
3. Dale una etiqueta a cada envio (ej. "lgbm_baseline", "multimodal_v2") para poder distinguirlos.

Los envios de practica se usan en la pagina Practice Simulation (ver abajo). No afectan la competencia.

---

## Resultados de la Competencia

Los resultados aparecen despues de que el organizador ejecuta una simulacion y libera la ronda. Si una ronda ha sido evaluada pero no liberada, veras un mensaje indicando que los resultados estan pendientes.

### Leaderboard

Una tabla resumen que muestra todos los equipos, con la fila de tu equipo resaltada. Puedes filtrar por ronda o ver todas las rondas combinadas. Las columnas son:


| Metrica                    | Que significa                                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Mean ROI (%)**           | Retorno de inversion promedio en todos los escenarios simulados. Positivo = tu modelo gana dinero en promedio. Esta es la metrica principal de ranking.                               |
| **Median ROI (%)**         | El escenario del medio — la mitad de tus resultados son mejores, la otra mitad peores. Menos sensible a valores extremos que la media.                                               |
| **Std ROI (%)**            | Cuanto varian tus retornos de escenario a escenario. Menor es mas consistente. Dos equipos con el mismo Mean ROI pero diferente Std tienen perfiles de riesgo muy distintos.         |
| **VaR 5% (%)**             | Value at Risk — el ROI en el percentil 5. "En el 95% de los escenarios, obtienes al menos este resultado." Una medida de tu peor resultado realista.                                |
| **CVaR 5% (%)**            | Conditional Value at Risk — el ROI promedio en tu peor 5% de escenarios. Que tan mal se ponen las cosas cuando salen mal. Siempre peor (menor) que VaR.                             |
| **Prob Positive (%)**      | El porcentaje de escenarios donde ganaste dinero (ROI > 0). Un modelo con 80% de prob positive genera ganancias 4 de cada 5 veces.                                                  |
| **Mean Properties Bought** | Cuantas propiedades compra tu modelo en promedio por escenario (de 250 disponibles). Un modelo que compra muy pocas es demasiado conservador; uno que compra demasiadas puede estar sobreprediciendo. |
| **Hit Rate (%)**           | De las propiedades que compraste, que fraccion fue rentable (valor real > costo pagado). Alto hit rate = tu modelo es bueno eligiendo ganadoras.                                     |
| **Sharpe**                 | Mean ROI dividido por Std ROI. Mide el retorno ajustado por riesgo — cuanto retorno obtienes por unidad de volatilidad. Mayor es mejor. Un Sharpe superior a 1.0 es fuerte.         |


### ROI Distribution

- Box plot comparando las distribuciones de ROI de todos los equipos.
- Histograma superpuesto — selecciona equipos especificos para comparar sus distribuciones de retorno.

### Drill-Down (5 Rondas Representativas)

El sistema selecciona 5 rondas de simulacion representativas en los percentiles P0 (tu peor), P25, P50 (mediana), P75 y P100 (tu mejor). Son fijas — ves las mismas 5 cada vez que inicias sesion.

Para cada una de estas 5 rondas, ves una tabla con las 250 propiedades que aparecieron en el mercado:


| Que ves                     | Cuando                                                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------- |
| Your prediction             | Siempre                                                                                        |
| Decision reason             | Siempre — "Bid", "Skip: below threshold", o "Skip: no budget"                                 |
| Asking price                | Solo si hiciste una oferta por la propiedad                                                    |
| Auction details             | Si ofertaste y la propiedad fue disputada — todas las ofertas con nombres de equipo, ganador, precio pagado |
| True value and profit       | Solo si ganaste la propiedad (la compraste)                                                    |


Las propiedades que omitiste muestran tu prediccion y la razon por la que la omitiste, pero no el asking price ni el valor real.

### Property Map

Un mapa interactivo muestra las 250 propiedades de la ronda de drill-down seleccionada, coloreadas por resultado:

- **Gris** — Skip (no ofertaste)
- **Amarillo oscuro** — Auction Lost (ofertaste pero perdiste)
- **Verde oscuro** — Profit (compraste y ganaste dinero)
- **Rojo oscuro** — Loss (compraste y perdiste dinero)

Haz clic en un marcador para ver las fotos del listado de la propiedad debajo del mapa.

---

## Practice Simulation

1. Ve a **Practice (OOF)**.
2. Selecciona 2 o mas de tus conjuntos de predicciones de practica subidos.
3. Opcionalmente ajusta los parametros de simulacion (transaction cost, market noise, etc.).
4. Haz clic en **Run Practice Simulation**.

La simulacion se ejecuta sobre el **training set** (donde los precios reales son conocidos), asi que tienes visibilidad completa: los valores reales se muestran para todas las propiedades, no solo las que compraste. Usa esto para entender como funciona la simulacion y como se comparan diferentes modelos antes de enviar a la competencia.

Los resultados se muestran en el mismo formato que los resultados de la competencia: tabla resumen, box plot y drill-down con detalles completos de propiedades.

---

## My Submissions

Un historial de todos tus envios — competencia y practica — con marcas de tiempo y estado (active o replaced).

---

## Consejos

- **Empieza con practica** — sube dos predicciones OOF diferentes y ejecuta una simulacion de practica para entender la mecanica antes de tu primer envio de competencia.
- **Cada ronda de competencia usa propiedades diferentes** — no puedes mejorar memorizando el feedback de rondas anteriores. Enfocate en mejorar tu modelo en general.
- **Forward-fill trabaja a tu favor** — si estas satisfecho con tu modelo actual, no necesitas re-enviar cada ronda.
- **La dinamica de subastas importa** — si multiples equipos predicen valores similares, competiran en subastas. Sobreprediccir gana subastas pero paga de mas. La precision gana.
