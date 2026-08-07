# Guión Presentación — Equipo SAM
## 15 minutos · ~140 palabras/minuto · 14 slides · Martin y Juan mitad cada uno

---

## SLIDE 1 — Título · JUAN · 45s (~105 palabras)

**En slide:** Logo SAM, título, subtítulo.

> "Buenas tardes. Somos el equipo SAM — Martin Ceriotti y Juan Ignacio Cacchione.
> Durante este cuatrimestre participamos en una competencia de inversión inmobiliaria simulada en Miami.
> El objetivo era predecir el precio de venta de propiedades residenciales mejor que los equipos competidores.
> En la práctica, pasamos por 9 rondas de modelos, cometimos errores que nos costaron plata virtual, y aprendimos algo que no esperábamos aprender.
> Les vamos a contar el recorrido."

---

## SLIDE 2 — El Juego / Mecánica · MARTIN · 1:30 (~200 palabras)

**En slide:** Diagrama de la simulación, 11,840 train / 5,038 test, Vickrey, ROI.

> "Antes de arrancar con los modelos, vale entender cómo funciona la competencia, porque cambia completamente cómo hay que pensar el problema.
>
> El organizador tiene una base de datos de miles de ventas reales de propiedades en Miami. Nos da dos partes: el train set con 11,840 casas donde conocemos el precio real, y el test set con 5,038 casas donde el precio está oculto. Eso es lo que predecimos.
>
> Una vez que subimos nuestras predicciones, el sistema corre 1,000 simulaciones de mercado. En cada simulación, aparece una propiedad con un precio de venta aleatorio basado en el valor real. Nosotros compramos si nuestra predicción supera ese precio en un margen mínimo del 8%.
>
> Si compramos, entramos a una subasta Vickrey: gana el equipo que más ofertó, pero paga lo que ofertó el segundo. Nuestra oferta siempre es nuestra predicción multiplicada por 0.85.
>
> Al final de las 1,000 simulaciones se mide el ROI promedio. Eso determina el ranking."

---

## SLIDE 3 — Métricas · MARTIN · 45s (~105 palabras)

**En slide:** MAPE vs ROI — la tensión central.

> "La competencia tiene dos métricas que vale entender porque no siempre van de la mano.
>
> El MAPE es el error promedio de predicción: un 26% significa que erramos $26 por cada $100 de precio real.
>
> El ROI es cuánta ganancia generamos en la simulación. Es la métrica que determina el ganador.
>
> El aprendizaje central del proyecto, que vamos a ver reflejado en varias rondas, es que bajar el MAPE no garantiza subir el ROI. Son cosas distintas."

---

## SLIDE 4 — Ronda 1: LightGBM · MARTIN · 1:45 (~245 palabras)

**En slide:** 📚 train/test, 🌳×100, CV 5-fold visual con EXAMEN rotando, 26.5% MAPE.

> "Nuestra primera ronda fue un baseline sólido con LightGBM.
>
> LightGBM es un modelo de boosting por gradiente. La idea: entrenamos 100 árboles de decisión en secuencia, donde cada árbol aprende de los errores del anterior. El árbol 1 hace una predicción, se equivoca. El árbol 2 corrige esos errores. Y así 100 veces. La predicción final es la suma.
>
> Para evaluar usamos CV 5-Fold. Tomamos las 11,840 casas y las dividimos en 5 grupos. En cada vuelta, entrenamos con 4 grupos y examinamos con el quinto. Rotamos el grupo de examen 5 veces.
>
> ¿Por qué? Si evaluáramos con las mismas casas con las que entrenamos, el resultado sería optimista — el modelo memorizó. El CV 5-Fold nos da una medida honesta del error real.
>
> Promedio de los 5 errores: MAPE de 26.5%. Por cada $100 de precio real, nos equivocamos en promedio $26. Es nuestro punto de partida."

---

## SLIDE 5 — Ronda 2a: Ratios Financieros · JUAN · 1:00 (~140 palabras)

**En slide:** 4 cards con fórmula y pregunta que responde cada ratio.

> "En la Ronda 2 decidimos darle más información al modelo creando variables nuevas a partir de los datos que ya teníamos.
>
> Las primeras son ratios financieros. El impuesto por metro cuadrado: si una casa paga mucho de impuesto relativo a su tamaño, es cara de mantener. La relación entre precio de lista y valuación municipal: si el vendedor pide mucho menos de lo que el municipio dice que vale, puede ser señal de apuro.
>
> El precio mediano del barrio, que calculamos dentro de cada fold para no contaminar el modelo. Y el HOA por metro cuadrado, que indica si las expensas son proporcionales al tamaño.
>
> Cada uno de estos ratios le da al modelo una perspectiva que antes no tenía."

---

## SLIDE 6 — Ronda 2b: Scores Compuestos + Segmentación · JUAN · 1:00 (~140 palabras)

**En slide:** Score de Lujo (pool+water+garage+HOA), Score Venta Forzada (embargo+bajada+cambios), 3 modelos separados.

> "El segundo tipo de feature nuevo son scores compuestos.
>
> El score de lujo suma señales de alta categoría: si tiene pileta, frente al agua, garage, y HOA mayor a 200 dólares. Puede llegar a 4. El score de venta forzada suma señales de apuro del vendedor: si está en proceso de embargo, si bajó el precio de lista, si cambió el precio más de dos veces.
>
> Además dividimos el problema: en vez de un modelo para todas las casas, entrenamos tres separados según tipo de propiedad: casas individuales, condominios, y el resto.
>
> El MAPE mejoró. El ROI empeoró. No supimos qué causó qué, porque lo probamos todo junto. Esa es la primera lección."

---

## SLIDE 7 — Ronda 3: Distressed Sales · JUAN · 2:00 (~280 palabras)

**En slide:** Ejemplo $700K → $140K, 5 señales de alerta, taxAssessedValue >> precio real.

> "En la Ronda 3, analizando el dashboard, encontramos el problema central que va a marcar el resto del proyecto.
>
> El modelo predecía $700,000 por propiedades que en realidad se habían vendido en $140,000. ¿Cómo puede pasar eso?
>
> Algunas propiedades se venden en situaciones de urgencia: un divorcio, una herencia, una deuda que hay que liquidar ya. El dueño acepta lo que sea. Se llaman distressed sales, ventas forzadas.
>
> El problema: antes de la venta, el municipio había valuado esa misma propiedad en $700,000. Esa valuación fiscal figura en los datos. El modelo la usa como referencia y predice en consecuencia. Pero el precio real fue $140,000 porque el vendedor no tenía opción.
>
> Desarrollamos cinco señales para detectar estas propiedades. La primera: si el municipio valúa mucho más que casas similares del mismo barrio. La segunda: si la valuación es un outlier estadístico. La tercera: si el vendedor nunca publicó precio de lista. La cuarta: si el precio publicado es sospechosamente bajo respecto al mercado del barrio. La quinta viene de las fotos, la vemos en la siguiente slide.
>
> Estas señales se convirtieron en features del modelo y son la base del fix quirúrgico de la Ronda 8."

---

## SLIDE 8 — Ronda 4a: Quantile Regression · JUAN · 1:00 (~140 palabras)

**En slide:** Campana de Gauss, $430K (p35) vs $500K (p50), oferta más baja = más margen.

> "En la Ronda 4 incorporamos dos mejoras importantes. Esta es la primera.
>
> Quantile Regression con alpha 0.35. Normalmente el modelo predice el precio promedio esperado — el centro de la campana. Nosotros lo configuramos para predecir el percentil 35, el valor conservador.
>
> Traducido: si el precio promedio esperado de una casa es $500,000, nuestro modelo predice $430,000. No porque creamos que vale menos, sino porque queremos comprar con más margen de seguridad.
>
> Nuestra oferta en subasta es siempre predicción por 0.85. Si predecimos $430,000, ofertamos $365,000. Si ganamos esa subasta, compramos a un precio bajo y generamos más ROI.
>
> En la campana: la zona verde a la izquierda del percentil 35 es donde nos posicionamos."

---

## SLIDE 9 — Ronda 4b: CLIP Embeddings · JUAN · 1:00 (~140 palabras)

**En slide:** Foto → CLIP → 512 → PCA → 32 features, clip_distress_score, +92% ROI.

> "La segunda mejora de la Ronda 4 fue incorporar las fotos de las propiedades como datos.
>
> CLIP es una red neuronal que convierte cada foto en un vector de 512 números que describen lo que ve. Usamos PCA para comprimir esos 512 a 32 features manejables, que sumamos a los datos tabulares.
>
> Creamos un feature específico: el clip_distress_score, que mide deterioro visual en la foto. Paredes dañadas, jardín descuidado, signos de abandono, score alto. Casa impecable, score bajo.
>
> Este score no redujo el MAPE, pero mejoró las decisiones de compra — eso confirma que error de predicción y calidad de decisión son cosas distintas.
>
> Resultado de la Ronda 4: ROI del 49.7%, una mejora del 92% respecto a la ronda anterior. El mayor salto del proyecto."

---

## SLIDE 10 — Ronda 8: Fix Quirúrgico · MARTIN · 1:45 (~245 palabras)

**En slide:** 6 propiedades identificadas, predicción $700-900K vs real $140-200K, fix = true_value × 0.92, ROI ~32%.

> "Después de varias iteraciones, en la Ronda 8 llegamos al modelo campeón con lo que llamamos el fix quirúrgico.
>
> Usando el drill-down del dashboard, que mostraba el precio real de propiedades que habíamos comprado en rondas anteriores, identificamos 6 propiedades específicas donde el modelo fallaba sistemáticamente. En todos los casos predecíamos entre $700,000 y $900,000 por casas que habían valido $140,000 a $200,000. Errores de cuatro y cinco veces el precio real.
>
> La solución fue quirúrgica: para esas 6 propiedades, ignoramos el modelo completamente y fijamos la predicción manualmente en el precio real por 0.92.
>
> ¿Por qué funciona? Dos escenarios. Primero, si el precio pedido es cercano al valor real, nuestra predicción de $184,000 no supera el umbral de compra del 8%, no compramos — evitamos la trampa. Segundo, si el vendedor tiene mucho apuro y pide menos, sí compramos, pero a un precio realmente bajo, generando ROI positivo.
>
> En ambos casos el resultado es mejor que antes. Este modelo dio un ROI medio de aproximadamente 32%, con Sharpe Ratio de 2.7."

---

## SLIDE 11 — Ronda 9: LLM · MARTIN · 30s (~70 palabras)

**En slide:** Qwen 1.5B, texto → keywords, ROI sin mejora.

> "En la Ronda 9 probamos usar un modelo de lenguaje local — Qwen, con 1,500 millones de parámetros — para extraer información de las descripciones textuales: palabras clave como 'renovado', 'vista al mar', 'urgente'. El experimento no mejoró el ROI. Lo dejamos documentado como un camino explorado."

---

## SLIDE 12 — Kelly + Scale 0.83 · MARTIN · 1:00 (~140 palabras)

**En slide:** Fórmula Kelly f* = p/L − q/W, factor 0.83, calibración independiente de ambos.

> "Dos calibraciones finales que mejoraron el modelo campeón.
>
> El Criterio de Kelly es una fórmula de teoría de la información: dado el porcentaje de victorias esperado y el retorno cuando se gana versus cuando se pierde, dice cuánto apostar. Aplicado a nuestro modelo nos dio un factor óptimo de 0.83 — nuestras predicciones estaban sesgadas hacia arriba en ese porcentaje.
>
> Multiplicamos todas las predicciones por 0.83. Esto corrigió el sesgo sin reentrenar.
>
> Lo que nos dio confianza: Juan y yo calculamos el factor Kelly de forma independiente, sin coordinarnos, y nos dio prácticamente el mismo valor — 0.828 y 0.832. Cuando dos personas llegan al mismo resultado por caminos distintos, eso es una buena señal."

---

## SLIDE 13 — Resultados · JUAN · 45s (~105 palabras)

**En slide:** ROI ~32%, rango 27-36%, Sharpe 2.7, Hit Rate 81%.

> "El modelo campeón combina quantile regression, embeddings CLIP, el fix quirúrgico en las 6 propiedades, y el factor de escala Kelly de 0.83.
>
> Los resultados en práctica: ROI medio de aproximadamente 32%, con rango entre 27% y 36% dependiendo de cuántos equipos compiten en cada simulación. Sharpe Ratio de 2.7 — retorno alto y consistente. Hit Rate del 81%.
>
> Kelly coincidió entre los dos al cuarto decimal, lo cual para nosotros fue la señal de que el proceso estaba bien hecho."

---

## SLIDE 14 — Lección / Cierre · JUAN · 45s (~105 palabras)

**En slide:** Meme Ricky Fort, frase de cierre.

> "Lo que nos llevamos de este proyecto va más allá de LightGBM o CLIP.
>
> El error de predicción no es lo mismo que tomar una buena decisión. Tuvimos rondas donde el MAPE mejoró y el ROI empeoró, y rondas donde una señal no redujo el error pero cambió completamente la calidad de las compras.
>
> El algoritmo era la parte fácil. Lo difícil fue entender el juego: cómo funciona una subasta Vickrey, qué es una venta distressed, cuándo no comprar.
>
> Gracias."

---

## Resumen de tiempos y división

| # | Slide | Quién | Tiempo |
|---|---|---|---|
| 1 | Título | **Juan** | 45s |
| 2 | El Juego / Mecánica | **Martin** | 1:30 |
| 3 | Métricas (MAPE vs ROI) | **Martin** | 45s |
| 4 | Ronda 1 — LightGBM + CV | **Martin** | 1:45 |
| 5 | Ronda 2a — Ratios Financieros | **Juan** | 1:00 |
| 6 | Ronda 2b — Scores + Segmentación | **Juan** | 1:00 |
| 7 | Ronda 3 — Distressed Sales | **Juan** | 2:00 |
| 8 | Ronda 4a — Quantile Regression | **Juan** | 1:00 |
| 9 | Ronda 4b — CLIP Embeddings | **Juan** | 1:00 |
| 10 | Ronda 8 — Fix Quirúrgico | **Martin** | 1:45 |
| 11 | Ronda 9 — LLM | **Martin** | 30s |
| 12 | Kelly + Scale 0.83 | **Martin** | 1:00 |
| 13 | Resultados | **Juan** | 45s |
| 14 | Lección / Cierre | **Juan** | 45s |
| | **Total** | | **~15 min** |

**Martin: slides 2,3,4,10,11,12 → ~6:15**
**Juan: slides 1,5,6,7,8,9,13,14 → ~8:15**

> *Nota: Juan puede pasarle a Martin el slide de CLIP (9) si el tiempo queda desbalanceado en ensayo.*

---

## Clave visual — qué ver en cada slide para recordar el speech

| Slide | Ancla visual | Qué dispara |
|---|---|---|
| 1 | Logo SAM + Maiameee! | "somos el equipo SAM..." |
| 2 | 11,840 / 5,038 | "train set donde sabemos el precio..." |
| 3 | MAPE ≠ ROI (dos flechas) | "no siempre van de la mano..." |
| 4 | 🌳×100 → EXAMEN rotando | "cada árbol aprende del error anterior..." |
| 5 | 4 cards con fórmulas | "impuesto por m², lista/valuación..." |
| 6 | Score lujo +1+1+1+1 | "suma señales: pileta, agua, garage..." |
| 7 | $700K → $140K | "algunas se venden en situaciones de urgencia..." |
| 8 | Campana, zona verde p35 | "predecimos el percentil 35..." |
| 9 | Foto → 512 → 32 + +92% | "CLIP convierte la foto en 512 números..." |
| 10 | 6 propiedades, × 0.92 | "identificamos 6 donde fallábamos..." |
| 11 | Qwen 1.5B | "probamos un LLM local..." |
| 12 | f* = p/L − q/W, factor 0.83 | "Kelly dice cuánto apostar dado..." |
| 13 | 32% ROI, Sharpe 2.7 | "el modelo campeón combina..." |
| 14 | Meme + frase final | "lo que nos llevamos va más allá..." |
