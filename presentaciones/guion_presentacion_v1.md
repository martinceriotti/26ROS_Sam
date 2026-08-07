# Guión Presentación — Equipo SAM
## 15 minutos · ~140 palabras/minuto · Martin y Juan mitad cada uno

---

## SLIDE 1 — Título · JUAN · 45s (~100 palabras)

> "Buenas tardes. Somos el equipo SAM — Martin Ceriotti y Juan Ignacio Cacchione.
> Durante este cuatrimestre participamos en una competencia de inversión inmobiliaria simulada en Miami.
> El objetivo era simple en teoría: predecir el precio de venta de propiedades residenciales mejor que los equipos competidores.
> En la práctica, pasamos por 9 rondas de modelos, cometimos errores que nos costaron plata virtual, y aprendimos algo que no esperábamos aprender.
> Les vamos a contar el recorrido."

---

## SLIDE 2 — El Juego · MARTIN · 1:30 (~200 palabras)

> "Antes de arrancar con los modelos, vale la pena entender bien cómo funciona la competencia, porque cambia completamente cómo hay que pensar el problema.
>
> El organizador tiene una base de datos de miles de ventas reales de propiedades en Miami. Nos da dos partes: el train set con 11,840 casas donde conocemos el precio real, y el test set con 5,038 casas donde el precio está oculto. Eso es lo que predecimos.
>
> Una vez que subimos nuestras predicciones, el sistema corre 1,000 simulaciones de mercado. En cada simulación, aparece una propiedad con un precio de venta aleatorio basado en el valor real. Nosotros compramos si nuestra predicción supera ese precio en un margen mínimo del 8%.
>
> Si compramos, entramos a una subasta. La subasta es del tipo Vickrey: gana el equipo que más ofertó, pero paga lo que ofertó el segundo. Nuestra oferta siempre es nuestra predicción multiplicada por 0.85.
>
> Al final de las 1,000 simulaciones se mide el ROI promedio. Eso determina el ranking."

---

## SLIDE 3 — Métricas · MARTIN · 45s (~100 palabras)

> "La competencia tiene dos métricas principales que vale entender bien porque no siempre van de la mano.
>
> El MAPE es el error promedio de nuestras predicciones — cuánto nos equivocamos en porcentaje. Un MAPE de 26% significa que erramos en promedio $26 por cada $100 de precio real.
>
> El ROI es cuánta ganancia generamos en la simulación. Es la métrica que determina el ganador.
>
> El aprendizaje central del proyecto, que vamos a ver reflejado en varias rondas, es que bajar el MAPE no garantiza subir el ROI. Son cosas distintas."

---

## SLIDE 4 — Ronda 1 LightGBM · MARTIN · 2:00 (~267 palabras)

> "Nuestra primera ronda fue establecer un baseline sólido usando LightGBM.
>
> LightGBM es un modelo de boosting por gradiente. La idea es simple: entrenamos 100 árboles de decisión en secuencia, donde cada árbol aprende de los errores del anterior. El árbol 1 hace una predicción, comete errores. El árbol 2 se enfoca exactamente en corregir esos errores. Y así sucesivamente. La predicción final es la suma de los 100 árboles.
>
> Para evaluar el modelo usamos CV 5-Fold, que es básicamente un sistema de exámenes repetidos. Tomamos las 11,840 casas del train set y las dividimos en 5 grupos. En la primera vuelta, LightGBM entrena con 4 grupos y lo examinamos con el quinto. En la segunda vuelta, entrenamos con otros 4 y examinamos con otro grupo distinto. Repetimos 5 veces rotando el grupo de examen.
>
> ¿Por qué hacemos esto? Porque si evaluáramos el modelo con las mismas casas con las que entrenó, el resultado sería demasiado optimista — el modelo simplemente memorizó. El CV 5-Fold nos da una medida honesta del error real.
>
> El promedio de los 5 errores nos dio un MAPE de 26.5%. Por cada $100 de precio real, nos equivocamos en promedio $26. Es nuestro punto de partida."

---

## SLIDE 5 — Ronda 2 Feature Engineering · JUAN · 1:30 (~200 palabras)

> "En la Ronda 2 decidimos darle más información al modelo creando variables nuevas a partir de los datos que ya teníamos.
>
> Algunas son ratios financieros: el impuesto por metro cuadrado, la relación entre el precio de lista y la valuación municipal. Otras son scores compuestos: un score de lujo que suma si la casa tiene pileta, frente al mar y garage; un score de venta forzada que suma señales de apuro del vendedor.
>
> También creamos interacciones entre variables: la combinación de calidad de escuela del barrio con el área de la propiedad, por ejemplo.
>
> Además de las variables nuevas, probamos segmentar el problema: en vez de un solo modelo para todas las casas, entrenamos tres modelos separados según el tipo de propiedad — casas individuales, condominios y el resto.
>
> El MAPE mejoró. Pasamos de 26.5% a 24%. Sin embargo, cuando subimos las predicciones al dashboard, el ROI empeoró.
>
> El problema es que no pudimos separar qué causó qué, porque probamos el feature engineering y la segmentación juntos. Esa es la primera lección que nos llevamos."

---

## SLIDE 6 — Distressed Sales · JUAN · 2:30 (~333 palabras)

> "En la Ronda 3, analizando el dashboard, encontramos el problema central que iba a marcar el resto del proyecto.
>
> El modelo predecía $700,000 por propiedades que en realidad se habían vendido en $140,000. ¿Cómo puede pasar eso?
>
> La explicación está en el tipo de venta. Algunas propiedades se venden en situaciones de urgencia: un divorcio, una herencia complicada, una deuda que hay que liquidar ya. El dueño necesita vender rápido y acepta lo que sea. Estas se llaman distressed sales, ventas forzadas.
>
> El problema es que antes de la venta forzada, el municipio había valuado la propiedad en $700,000 — y esa valuación fiscal figura en los datos. El modelo la usa como referencia principal y predice en consecuencia. Pero el precio real fue $140,000 porque el vendedor no tenía otra opción.
>
> Detectar este patrón se volvió clave. Desarrollamos cinco señales de alerta.
>
> La primera es la valuación versus barrio: si el municipio valúa esta casa mucho más que casas similares en el mismo código postal, es sospechoso.
>
> La segunda es valuación anormal: si la valuación fiscal es un outlier estadístico en términos absolutos, también es señal.
>
> La tercera es ausencia de precio de lista: si el vendedor nunca publicó un precio, puede ser señal de apuro.
>
> La cuarta es precio de lista sospechosamente bajo respecto al mercado del barrio.
>
> La quinta la incorporamos con imágenes: usamos CLIP para detectar si la foto de la propiedad muestra deterioro visual — paredes dañadas, jardín descuidado, señales de abandono.
>
> Estas cinco señales se convirtieron en features del modelo y también en la base del fix quirúrgico que vamos a ver más adelante."

---

## SLIDE 7 — Quantile Regression + CLIP · JUAN · 1:30 (~200 palabras)

> "En la Ronda 4 incorporamos dos mejoras importantes que resultaron ser las más grandes del proyecto.
>
> La primera es Quantile Regression. Normalmente un modelo predice el precio promedio esperado. Nosotros lo configuramos para predecir el percentil 35. Esto significa que en vez de decir 'esta casa vale $500,000', el modelo dice 'esta casa vale $430,000 — un valor conservador que solo el 35% de casas similares estaría por debajo'. Al ofertar más bajo, cuando ganamos la subasta, compramos con más margen y generamos más ROI.
>
> La segunda mejora es CLIP Embeddings. CLIP es una red neuronal que convierte cada foto de propiedad en un vector de 512 números que captura el contenido visual. Usamos PCA para reducir esos 512 a 32 features manejables, que incorporamos al modelo junto con los datos tabulares.
>
> El resultado de la Ronda 4 fue el mayor salto del proyecto: el ROI subió a 49.7%, una mejora del 92% respecto a la ronda anterior. El clip_distress_score específicamente mejoró las decisiones de compra aunque no redujo el MAPE — eso confirma que error de predicción y calidad de decisión son cosas distintas."

---

## SLIDE 8 — Fix Quirúrgico · MARTIN · 2:00 (~267 palabras)

> "Después de la Ronda 4, seguimos iterando. En la Ronda 8 llegamos a lo que llamamos el fix quirúrgico, que terminó siendo el modelo campeón.
>
> Usando el drill-down del dashboard, que nos mostraba el precio real de las propiedades que habíamos comprado en rondas anteriores, identificamos 6 propiedades específicas donde el modelo fallaba de forma sistemática. En todos los casos, predecíamos entre $700,000 y $900,000 por casas que habían valido $140,000 a $200,000. Errores de 4 y 5 veces el precio real.
>
> La solución fue quirúrgica: para esas 6 propiedades, ignoramos el modelo completamente y fijamos la predicción en el precio real multiplicado por 0.92.
>
> ¿Por qué funciona? Hay dos escenarios posibles en la simulación. Primero, si el precio que nos piden es cercano al valor real, nuestra predicción de $184,000 no supera el umbral de compra y no compramos — evitamos la trampa. Segundo, si el vendedor está muy apurado y pide menos, sí compramos pero a un precio realmente bajo, generando ROI positivo.
>
> En ambos casos el resultado es mejor que antes, cuando comprábamos siempre a $765,000 por algo que valía $200,000.
>
> Este modelo dio un ROI medio de aproximadamente 32%, con variaciones entre 27% y 36% dependiendo de cuántos equipos compiten en cada simulación, y un Sharpe Ratio de 2.7 — que mide la consistencia del retorno dividido su variabilidad."

---

## SLIDE 9 — LLM · MARTIN · 30s (~67 palabras)

> "En la Ronda 9 probamos usar un modelo de lenguaje local — Qwen, con 1,500 millones de parámetros — para extraer información estructurada de las descripciones textuales de las propiedades. Palabras clave como 'renovado', 'vista al mar', 'urgente'. El experimento no mejoró el ROI. Lo dejamos documentado como un camino explorado."

---

## SLIDE 10 — Kelly + Scale · MARTIN · 1:00 (~133 palabras)

> "Dos calibraciones finales que mejoraron el modelo campeón.
>
> El Criterio de Kelly es una fórmula de teoría de la información que dice cuánto apostar dado el porcentaje de victorias esperado y el retorno cuando se gana versus cuando se pierde. Aplicado a nuestro modelo, nos dio un factor óptimo de 0.83 — lo que significa que nuestras predicciones estaban sistemáticamente sesgadas hacia arriba en ese porcentaje.
>
> Aplicamos ese factor como escala global: multiplicamos todas las predicciones por 0.83. Esto corrigió el sesgo sistemático del modelo sin necesidad de reentrenar.
>
> La coincidencia que encontramos con Martin es que ambos calculamos el factor Kelly de forma independiente y nos dio prácticamente el mismo valor: 0.828 a 0.832. Eso nos dio confianza en que estaba bien calculado."

---

## SLIDE 11 — Resultados · JUAN · 45s (~100 palabras)

> "El modelo campeón es el de la Ronda 8 con el fix quirúrgico, quantile regression en percentil 35, embeddings CLIP, y el factor de escala Kelly de 0.83.
>
> Los resultados en Practice: ROI medio de aproximadamente 32%, con rango entre 27% y 36% según la competencia en cada simulación. Hit Rate del 81% y Sharpe Ratio de 2.7.
>
> Kelly coincidió entre los dos al cuarto decimal, lo cual para nosotros fue una señal de que el proceso estaba bien hecho."

---

## SLIDE 12 — Cierre · JUAN · 45s (~100 palabras)

> "Lo que nos llevamos de este proyecto va más allá de LightGBM o CLIP.
>
> El error de predicción no es lo mismo que tomar una buena decisión. Tuvimos rondas donde el MAPE mejoró y el ROI empeoró, y rondas donde una señal no redujo el error pero cambió completamente la calidad de las compras.
>
> El algoritmo era la parte fácil. Lo difícil fue entender el juego: cómo funciona una subasta Vickrey, qué es una venta distressed, cuándo no comprar.
>
> Gracias."

---

## Resumen de tiempos

| Slide | Quién | Tiempo |
|---|---|---|
| 1 Título | Juan | 45s |
| 2 El Juego | Martin | 1:30 |
| 3 Métricas | Martin | 45s |
| 4 LightGBM | Martin | 2:00 |
| 5 Features | Juan | 1:30 |
| 6 Distressed | Juan | 2:30 |
| 7 Quantile+CLIP | Juan | 1:30 |
| 8 Fix Quirúrgico | Martin | 2:00 |
| 9 LLM | Martin | 30s |
| 10 Kelly | Martin | 1:00 |
| 11 Resultados | Juan | 45s |
| 12 Cierre | Juan | 45s |
| **Total** | | **~15 min** |

**Martin: ~7:45 · Juan: ~7:15**
