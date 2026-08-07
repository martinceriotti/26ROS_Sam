# Guión Presentación v2 — Equipo SAM
## 15 minutos · 14 slides · 7 slides cada uno · ~7:15 por persona

> **v2 corresponde a presentacion_sam.html (14 slides)**
> v1 era la versión de 12 slides

---

## Distribución

| # | Slide | **Quién** | Tiempo |
|---|---|---|---|
| 1 | Título | **Juan** | 45s |
| 2 | El Juego / Mecánica | **Martin** | 1:15 |
| 3 | Métricas (MAPE vs ROI) | **Martin** | 45s |
| 4 | Ronda 1 — LightGBM + CV | **Martin** | 1:15 |
| 5 | Ronda 2a — Ratios Financieros | **Juan** | 1:00 |
| 6 | Ronda 2b — Scores + Segmentación | **Juan** | 1:00 |
| 7 | Ronda 3 — Distressed Sales | **Juan** | 2:15 |
| 8 | Ronda 4a — Quantile Regression | **Juan** | 1:00 |
| 9 | Ronda 4b — CLIP Embeddings | **Martin** | 1:00 |
| 10 | Ronda 8 — Fix Quirúrgico | **Martin** | 1:30 |
| 11 | Ronda 9 — LLM | **Martin** | 30s |
| 12 | Kelly + Scale 0.83 | **Martin** | 1:00 |
| 13 | Resultados | **Juan** | 45s |
| 14 | Lección / Cierre | **Juan** | 30s |

**Juan: slides 1,5,6,7,8,13,14 → ~7:15**
**Martin: slides 2,3,4,9,10,11,12 → ~7:15**

---

## SLIDE 1 — Título · JUAN · 45s

**Ancla visual:** Logo SAM + Maiameee!

> "Buenas tardes. Somos el equipo SAM — Martin Ceriotti y Juan Ignacio Cacchione.
> Durante este cuatrimestre participamos en una competencia de inversión inmobiliaria simulada en Miami.
> El objetivo era predecir el precio de venta de propiedades residenciales mejor que los equipos competidores.
> En la práctica, pasamos por 9 rondas de modelos, cometimos errores que nos costaron plata virtual, y aprendimos algo que no esperábamos aprender.
> Les vamos a contar el recorrido."

---

## SLIDE 2 — El Juego / Mecánica · MARTIN · 1:15

**Ancla visual:** 11,840 / 5,038 · Vickrey · 1,000 simulaciones · ROI

> "Antes de arrancar con los modelos, vale entender bien cómo funciona la competencia, porque cambia cómo hay que pensar el problema.
>
> El organizador tiene una base de datos de ventas reales de propiedades en Miami. Nos da dos partes: el train set con 11,840 casas donde conocemos el precio real, y el test set con 5,038 casas donde el precio está oculto. Eso es lo que predecimos.
>
> Una vez que subimos nuestras predicciones, el sistema corre 1,000 simulaciones de mercado. Nosotros compramos si nuestra predicción supera el precio pedido en al menos un 8%.
>
> Si compramos, entramos a una subasta Vickrey: gana el equipo que más ofertó, pero paga lo que ofertó el segundo. Nuestra oferta es siempre predicción por 0.85.
>
> Al final de las 1,000 simulaciones se mide el ROI promedio. Eso determina el ranking."

---

## SLIDE 3 — Métricas · MARTIN · 45s

**Ancla visual:** MAPE ↓ ≠ ROI ↑

> "La competencia tiene dos métricas que no siempre van de la mano.
>
> El MAPE es el error promedio de predicción: un 26% significa que erramos $26 por cada $100 de precio real.
>
> El ROI es cuánta ganancia generamos en la simulación. Es la métrica que determina el ganador.
>
> El aprendizaje central: bajar el MAPE no garantiza subir el ROI. Lo vamos a ver repetido en varias rondas."

---

## SLIDE 4 — Ronda 1: LightGBM · MARTIN · 1:15

**Ancla visual:** 🌳×100 → CV 5-fold visual → 26.5% MAPE

> "Nuestra primera ronda fue un baseline con LightGBM. La idea: 100 árboles de decisión en secuencia, donde cada árbol aprende de los errores del anterior. La predicción final es la suma de los 100.
>
> Para evaluarlo usamos CV 5-Fold. Tomamos las 11,840 casas y las dividimos en 5 grupos. En cada vuelta entrenamos con 4 grupos y examinamos con el quinto, rotando. Así obtenemos una medida honesta del error real — si evaluáramos con las mismas casas con las que entrenamos, el resultado sería demasiado optimista.
>
> Resultado: MAPE de 26.5%. $26 de error por cada $100. Es nuestro punto de partida."

---

## SLIDE 5 — Ronda 2a: Ratios Financieros · JUAN · 1:00

**Ancla visual:** 4 cards con fórmula arriba y pregunta abajo

> "En la Ronda 2 le dimos más información al modelo creando variables nuevas a partir de los datos que ya teníamos.
>
> Las primeras son ratios financieros. El impuesto por metro cuadrado pregunta si la casa es cara de mantener para su tamaño. La relación entre precio de lista y valuación municipal pregunta si el vendedor pide mucho menos de lo que el municipio dice que vale — señal de apuro.
>
> El precio mediano del barrio lo calculamos dentro de cada fold para no contaminar el modelo. El HOA por metro cuadrado indica si las expensas son proporcionales al tamaño.
>
> Cada ratio le da al modelo una perspectiva nueva."

---

## SLIDE 6 — Ronda 2b: Scores Compuestos + Segmentación · JUAN · 1:00

**Ancla visual:** Score lujo: +1 +1 +1 +1 · Score forzada: +1 +1 +1 · 3 modelos separados

> "El segundo tipo de feature son scores compuestos.
>
> El score de lujo suma: pileta más uno, frente al agua más uno, garage más uno, HOA mayor a 200 dólares más uno. Llega hasta 4.
>
> El score de venta forzada suma señales de apuro: embargo, bajada de precio, más de dos cambios de precio. Llega hasta 3.
>
> Además dividimos el problema: en vez de un modelo para todas las casas, entrenamos tres separados por tipo — casas individuales, condominios, y el resto.
>
> El MAPE mejoró. El ROI empeoró. No supimos qué causó qué, porque lo probamos todo junto. Primera lección."

---

## SLIDE 7 — Ronda 3: Distressed Sales · JUAN · 2:15

**Ancla visual:** $700K → $140K · taxAssessedValue >> precio real · 5 señales

> "En la Ronda 3, analizando el dashboard, encontramos el problema central del proyecto.
>
> El modelo predecía $700,000 por propiedades que en realidad se habían vendido en $140,000. ¿Cómo?
>
> Algunas propiedades se venden en situaciones de urgencia: un divorcio, una herencia complicada, una deuda que hay que liquidar ya. El dueño acepta lo que sea. Se llaman ventas forzadas o distressed sales.
>
> El problema: antes de esa venta, el municipio había valuado la propiedad en $700,000. Esa valuación fiscal está en los datos. El modelo la usa como referencia principal y predice en consecuencia. Pero el precio real fue $140,000 porque el vendedor no tenía opción.
>
> Desarrollamos cinco señales para detectar estas propiedades. Primera: si el municipio valúa mucho más que casas similares del mismo barrio. Segunda: si la valuación es un outlier estadístico en términos absolutos. Tercera: si el vendedor nunca publicó precio de lista. Cuarta: si el precio publicado es sospechosamente bajo respecto al barrio. Quinta: deterioro visual en la foto — eso lo vemos en la siguiente slide.
>
> Estas cinco señales se volvieron features del modelo y son la base del fix quirúrgico que viene."

---

## SLIDE 8 — Ronda 4a: Quantile Regression · JUAN · 1:00

**Ancla visual:** Campana de Gauss · zona verde izquierda · $430K (p35) vs $500K (p50)

> "En la Ronda 4, dos mejoras importantes. Esta es la primera.
>
> Quantile Regression con alpha 0.35. El modelo normal predice el precio promedio — el centro de la campana. Nosotros lo configuramos para predecir el percentil 35, el valor conservador.
>
> Si el precio promedio esperado es $500,000, nuestro modelo predice $430,000. No porque creamos que vale menos — sino para comprar con más margen.
>
> Nuestra oferta es siempre predicción por 0.85. Si predecimos $430,000, ofertamos $365,000. Si ganamos, compramos barato y generamos más ROI.
>
> En la campana: la zona verde a la izquierda del percentil 35 es donde nos posicionamos."

---

## SLIDE 9 — Ronda 4b: CLIP Embeddings · MARTIN · 1:00

**Ancla visual:** Foto → 512 → PCA → 32 · clip_distress_score · +92% ROI

> "La segunda mejora de la Ronda 4: incorporamos las fotos de las propiedades como datos.
>
> CLIP es una red neuronal que convierte cada foto en 512 números que describen lo que ve. Usamos PCA para comprimir esos 512 a 32 features manejables, que sumamos a los datos tabulares.
>
> Creamos el clip_distress_score: mide deterioro visual en la foto. Casa deteriorada, score alto. Casa impecable, score bajo.
>
> Este score no redujo el MAPE, pero mejoró las decisiones de compra. Eso confirma que error de predicción y calidad de decisión son cosas distintas.
>
> Resultado de la Ronda 4: ROI de 49.7%, mejora del 92%. El mayor salto del proyecto."

---

## SLIDE 10 — Ronda 8: Fix Quirúrgico · MARTIN · 1:30

**Ancla visual:** 6 propiedades identificadas · predicción $700-900K vs real $140-200K · × 0.92 · ROI ~32%

> "Después de varias iteraciones, en la Ronda 8 llegamos al modelo campeón con lo que llamamos el fix quirúrgico.
>
> Usando el drill-down del dashboard identificamos 6 propiedades donde el modelo fallaba sistemáticamente. Predecíamos entre $700,000 y $900,000 por casas que habían valido $140,000 a $200,000. Errores de cuatro y cinco veces el precio real.
>
> La solución fue quirúrgica: para esas 6 propiedades ignoramos el modelo y fijamos la predicción en el precio real por 0.92.
>
> Dos escenarios: primero, si el precio pedido es cercano al valor real, nuestra predicción de $184,000 no supera el umbral del 8%, no compramos, evitamos la trampa. Segundo, si el vendedor tiene mucho apuro y pide menos, compramos a precio realmente bajo, ROI positivo.
>
> Resultado: ROI medio de 32%, Sharpe Ratio de 2.7."

---

## SLIDE 11 — Ronda 9: LLM · MARTIN · 30s

**Ancla visual:** Qwen 1.5B · texto → keywords · sin mejora de ROI

> "En la Ronda 9 probamos usar un modelo de lenguaje local — Qwen, con 1,500 millones de parámetros — para extraer información de las descripciones textuales: palabras clave como renovado, vista al mar, urgente. El experimento no mejoró el ROI. Lo dejamos documentado."

---

## SLIDE 12 — Kelly + Scale 0.83 · MARTIN · 1:00

**Ancla visual:** Fórmula f* = p/L − q/W · factor 0.83 · coincidencia 0.828 / 0.832

> "Dos calibraciones finales.
>
> El Criterio de Kelly es una fórmula de teoría de la información: dado el porcentaje de victorias esperado y el retorno en caso de ganar versus perder, dice cuánto apostar. Aplicado a nuestro modelo nos dio un factor de 0.83 — nuestras predicciones estaban sesgadas hacia arriba en ese porcentaje.
>
> Multiplicamos todas las predicciones por 0.83. Corregimos el sesgo sin reentrenar.
>
> Lo que nos dio confianza: Juan y yo calculamos el factor Kelly de forma independiente y nos dio prácticamente el mismo valor — 0.828 y 0.832. Cuando dos personas llegan al mismo resultado por caminos distintos, es buena señal."

---

## SLIDE 13 — Resultados · JUAN · 45s

**Ancla visual:** 32% ROI · Sharpe 2.7 · Hit Rate 81%

> "El modelo campeón combina quantile regression, embeddings CLIP, el fix quirúrgico en las 6 propiedades, y el factor de escala Kelly de 0.83.
>
> Resultados en práctica: ROI medio de aproximadamente 32%, rango entre 27% y 36% según la competencia. Sharpe Ratio de 2.7 — retorno alto y consistente. Hit Rate del 81%.
>
> Kelly coincidió al cuarto decimal entre los dos. Para nosotros esa fue la señal de que el proceso estaba bien hecho."

---

## SLIDE 14 — Lección / Cierre · JUAN · 30s

**Ancla visual:** Meme Maiameee! + frase final

> "Lo que nos llevamos: el error de predicción no es lo mismo que tomar una buena decisión.
>
> Tuvimos rondas donde el MAPE mejoró y el ROI empeoró, y rondas donde una señal no redujo el error pero cambió completamente la calidad de las compras.
>
> El algoritmo era la parte fácil. Lo difícil fue entender el juego.
>
> Gracias."

---

## Clave visual — anclas para hablar sin machete

| Slide | Qué ves en pantalla | Primera frase que dispara |
|---|---|---|
| 1 | Logo SAM + Maiameee! | "Somos el equipo SAM..." |
| 2 | 11,840 / 5,038 + subasta | "El organizador nos da dos partes..." |
| 3 | MAPE ≠ ROI | "No siempre van de la mano..." |
| 4 | 🌳×100 + EXAMEN rotando | "100 árboles en secuencia..." |
| 5 | 4 fórmulas con pregunta | "El impuesto por metro cuadrado..." |
| 6 | +1+1+1+1 / +1+1+1 | "El score de lujo suma..." |
| 7 | $700K → $140K rojo | "El modelo predecía $700,000..." |
| 8 | Campana, zona verde p35 | "El modelo normal predice el promedio..." |
| 9 | Foto → 512 → 32 + +92% | "CLIP convierte cada foto en 512 números..." |
| 10 | 6 zpids + × 0.92 | "Identificamos 6 propiedades donde fallábamos..." |
| 11 | Qwen 1.5B | "Probamos un LLM local..." |
| 12 | f* = p/L − q/W + 0.83 | "El Criterio de Kelly nos da un factor..." |
| 13 | 32% ROI, Sharpe 2.7 | "El modelo campeón combina..." |
| 14 | Meme + frase | "Lo que nos llevamos..." |
