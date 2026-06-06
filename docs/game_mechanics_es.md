# Competencia de Inversion — Mecanica del Juego

## El Negocio

El mercado inmobiliario de Miami es volatil, rapido y lleno de asimetria de informacion. Vendedores en apuros — enfrentando ejecuciones hipotecarias, reubicaciones, divorcios o liquidaciones de herencia — rutinariamente listan propiedades por debajo de su valor real de mercado porque necesitan efectivo rapidamente. Para una firma de inversion con la ventaja analitica correcta, esto crea una oportunidad repetible: identificar propiedades genuinamente subvaloradas, adquirirlas y venderlas a su valor real de mercado.

Esto no es hipotetico. Firmas como Opendoor, Offerpad y docenas de flippers regionales operan exactamente con este modelo. La economia es ajustada — los costos de transaccion comen los margenes, la competencia de otros compradores sube los precios, y una sola mala compra puede eliminar la ganancia de varias buenas. Las firmas que sobreviven son las que tienen los mejores modelos de precios.

En esta competencia, tu eres esa firma.

---

## Las Decisiones Detras del Negocio

Si realmente estuvieras dirigiendo una firma de inversion inmobiliaria, tu equipo enfrentaria una cascada de decisiones interconectadas en cada operacion:

**Valuacion** — Cuanto vale realmente esta propiedad? Toda tu estrategia depende de este numero. Equivocarte por un 10% y un gran negocio se convierte en un pozo de dinero. Cada otra decision posterior — si ofertar, cuanto ofrecer, cuanto capital arriesgar — depende de tu confianza en esta estimacion.

**Lectura de mercado** — Como interpretas el precio de venta? Esta el vendedor realmente en apuros o solo esta probando el mercado? Cuanto ruido hay entre el precio de venta y el valor real? Una propiedad listada en $400K podria valer $500K (una ganga) o $350K (una trampa). Tu modelo necesita notar la diferencia.

**Seleccion de operaciones** — Que propiedades vale la pena perseguir? No puedes comprar todo — el capital es finito, y cada dolar atado en un negocio mediocre es un dolar no disponible para uno excelente. Necesitas un umbral: cuanta ventaja percibida justifica desplegar capital?

**Estrategia de oferta** — Cuando compites contra otros compradores, que tan agresivamente ofertas? Oferta muy alto y ganas cada subasta pero pagas de mas. Oferta muy bajo y pierdes cada negocio disputado ante alguien mas audaz. La oferta optima depende de tu confianza en tu valuacion y tu lectura de la competencia.

**Construccion de portafolio** — Cuanto capital concentras en una sola propiedad? Un fondo de $5M que pone $2M en una casa esta tomando un riesgo idiosincratico masivo. La diversificacion protege contra errores individuales de valuacion pero limita el upside en tus mejores picks.

**Despliegue de capital** — Cuanto de tu fondo realmente inviertes vs. mantienes en reserva? Ser demasiado agresivo significa quedarte sin capital antes de que aparezcan los mejores negocios. Ser demasiado conservador significa que tu dinero esta inactivo sin generar nada mientras las oportunidades pasan.

Cada una de estas decisiones interactua con las otras, y en el mundo real, inversores experimentados desarrollan intuicion para todas ellas a lo largo de anos de practica.

---

## Tu Trabajo en la Competencia

Vamos a simplificar.

En esta competencia, tu equipo es responsable de **una sola cosa**: producir el modelo de valuacion de propiedades mas preciso que puedas. Envias un CSV con tu valor de mercado predicho para cada propiedad en el test set. Ese es tu unico entregable.

Todas las otras decisiones — seleccion de operaciones, estrategia de oferta, dimensionamiento de portafolio, gestion de capital — son manejadas por el motor de simulacion usando reglas fijas y transparentes que son iguales para todos los equipos. La simulacion toma tus predicciones y juega el juego de inversion en tu nombre, tomando decisiones racionales basadas en como tus valores predichos se comparan con los precios de venta del mercado.

**Por que funciona esta simplificacion**: en el negocio real, la calidad de cada decision posterior depende fundamentalmente de la calidad de la valuacion inicial. Una firma con estimaciones de precio perfectas puede optimizar trivialmente cualquier otro parametro. Una firma con estimaciones terribles perdera dinero sin importar cuan inteligente sea su estrategia de oferta. Al fijar la capa de estrategia y dejar que los equipos compitan puramente en precision de prediccion, aislamos la variable que mas importa — y la que la ciencia de datos realmente puede mejorar.

**Predeciras todas las propiedades del test**: como no sabes de antemano que propiedades estaran en el pool de tu ronda, necesitas proporcionar una estimacion para cada propiedad del test. El motor de simulacion luego ejecutara mil escenarios de mercado diferentes sobre el pool de tu ronda para construir una imagen robusta del rendimiento real de inversion de tu modelo.

---

## Lo Que Esta en Juego: La Calidad del Modelo Impulsa los Retornos

La simulacion traduce la precision de prediccion directamente en retornos de inversion. Esto es lo que puedes esperar con los parametros por defecto de la competencia:

La tabla a continuacion muestra resultados reales de simulacion de modelos construidos con este dataset, usando parametros por defecto. No son hipoteticos — provienen de ejecutar 1,000 simulaciones Monte Carlo.


| Model                               | MAPE | Mean ROI | Interpretacion                                              |
| ----------------------------------- | ---- | -------- | ----------------------------------------------------------- |
| Passive (buy nothing)               | —    | -4.0%    | El piso — lo que pasa si te quedas con efectivo             |
| Image only                          | ~50% | -19%     | Predicciones demasiado ruidosas para ganar dinero           |
| Text only                           | ~43% | -10%     | Algo de senal, pero lejos de ser suficiente                 |
| Tabular (minimal)                   | ~29% | -4%      | Aproximadamente lo mismo que no hacer nada                  |
| Tabular (extended)                  | ~29% | -3%      | Todavia perdiendo, pero acercandose al punto de equilibrio  |
| Multimodal (tabular + image + text) | ~28% | +3%      | Rentable — cada modalidad agrega valor real                 |


La brecha entre ~29% MAPE y ~28% MAPE parece pequena en un leaderboard, pero en la simulacion es la diferencia entre perder dinero consistentemente y generar ganancias 4 de cada 5 veces. Cada fraccion de porcentaje importa cuando los costos de transaccion y la competencia comprimen los margenes.

---

## Mecanica de la Competencia en Detalle

El resto de este documento explica exactamente como funciona la simulacion. No necesitas memorizar esto para competir — tu unico trabajo sigue siendo "hacer mejores predicciones" — pero entender la mecanica te ayuda a interpretar tus resultados y razonar sobre por que ciertos modelos superan a otros.

### Vista General de la Simulacion

La competencia se evalua via **simulacion Monte Carlo**: el juego de inversion se ejecuta 1,000 veces con diferentes condiciones aleatorias de mercado cada vez. Tus predicciones son fijas en todas las simulaciones — solo cambian la seleccion de propiedades y el ruido de mercado. Esto produce una **distribucion** de resultados, no un solo numero, revelando cuan robusto es tu modelo.

Cada ronda de competencia usa su propio pool de ~1,260 propiedades (el test set se divide en 4 pools que no se solapan). Dentro de una ronda, la simulacion se ejecuta 1,000 veces. Cada simulacion consiste en **4 rondas internas de inversion**. En cada ronda interna, se seleccionan 250 propiedades del pool (sin reemplazo), dando ~1,000 propiedades por simulacion de ~1,260 disponibles. El capital se reinicia a $5M en cada ronda interna.

### Paso a Paso: Que Sucede en Cada Ronda

**Step 1 — Property selection** `[SIMULATION]`

El motor selecciona aleatoriamente 250 propiedades del pool restante para esta ronda.

**Step 2 — Asking prices** `[SIMULATION]`

Cada propiedad recibe un precio de venta sintetico:

```
asking_price = true_value × (1 + Normal(market_bias, market_noise))
```

Por defecto: `market_bias = -0.07` (los vendedores subvaloran un 7% en promedio), `market_noise = 0.35` (alta varianza — algunas son grandes ofertas, otras son trampas).

**Step 3 — Buy/skip decisions** `[SIMULATION, usando tus predicciones]`

Las propiedades se presentan secuencialmente. Para cada propiedad, la simulacion verifica si tu equipo quiere comprar. Dos condiciones deben cumplirse:

1. **Signal check** — tu modelo ve suficiente ventaja:
  ```
   prediction > asking_price × (1 + buy_threshold)
  ```
   El umbral por defecto es 8%. Si el asking price es $400K, tu modelo debe predecir al menos $432K para activar una compra.
2. **Affordability check** — puedes pagarla:
  ```
   asking_price × (1 + transaction_cost) ≤ remaining_capital × max_position_pct
  ```
   No puedes gastar mas del 25% de tu capital restante de la ronda en una propiedad.

**No hay look-ahead**: la simulacion no puede saltarse un negocio mediocre para ahorrar capital para uno mejor despues. Esto es realista — en un mercado real no sabes que viene despues.

Si ambas verificaciones pasan, la oferta sellada de tu equipo se calcula: `prediction × bid_fraction` (por defecto 0.85).

**Step 4 — Resolution** `[SIMULATION]`

Cada propiedad se resuelve segun cuantos equipos la quieren:

- **Nadie la quiere** → el portafolio complemento del mercado compra al asking price
- **Un equipo la quiere** → compra sin competencia al asking price + costos de transaccion
- **Multiples equipos la quieren** → subasta sellada de segundo precio (ver abajo)

**Step 5 — P&L** `[SIMULATION]`

Para cada propiedad ganada:

```
cost = purchase_price × (1 + transaction_cost)
profit = true_value - cost
```

Al final de cada ronda, el capital inactivo se penaliza:

```
penalty = idle_capital × (opportunity_cost_rate / n_rounds)
```

### Subastas

Cuando 2+ equipos quieren la misma propiedad, una **subasta sellada de segundo precio (Vickrey)** decide el resultado:

1. La oferta de cada equipo ya fue calculada: `prediction × bid_fraction`
2. **El mayor postor gana**
3. El ganador paga: `max(asking_price, second_highest_bid)` mas costos de transaccion
4. Los empates se resuelven con un coin flip aleatorio

Por que segundo precio? Es strategy-proof — tu oferta optima es proporcional a tu valuacion real, sin incentivo para bluffear. Nunca pagas tu propia oferta, solo la segunda mas alta. Equipos que sobrepredicen ganan mas subastas pero sistematicamente pagan de mas. Equipos que subpredicen pierden competencias pero evitan pagar de mas. **La precision gana.**

### Calculo de ROI

**Equipos:**

```
net = total_profit - total_opportunity_cost
total_capital = starting_capital × n_rounds
ROI = net / total_capital × 100
```

El ROI se mide contra el capital total disponible (no solo el capital desplegado). Esto evita que un equipo que casi no compra nada muestre un porcentaje de retorno enganosamente alto sobre su pequena inversion.

**Market complement** (sin restriccion de capital):

```
ROI = total_profit / total_invested × 100
```

### Condicion de Victoria

Cada simulacion produce un ROI para cada equipo. El ganador es el equipo con el ROI mas alto en esa simulacion. A traves de 1,000 simulaciones, tu **win rate** es la fraccion donde quedaste en primer lugar.

---

## Referencia de Parametros

Todos los parametros son configurables en el dashboard. Los valores por defecto estan calibrados para que un modelo tabular basico aproximadamente quede en equilibrio, y cada mejora de modalidad produzca ganancias visibles pero realistas.

### Competition Structure


| Parameter        | Default    | Range     | Proposito                                                 |
| ---------------- | ---------- | --------- | --------------------------------------------------------- |
| Rounds           | 4          | 1–8       | Rondas de inversion independientes por simulacion         |
| Properties/round | 250        | 10–500    | Propiedades ofrecidas cada ronda                          |
| Simulations      | 1,000      | 100–5,000 | Ejecuciones Monte Carlo para robustez estadistica         |
| Starting capital | $5,000,000 | $1M–$50M  | Capital fresco cada ronda                                 |


### Economics


| Parameter        | Default | Range     | Proposito                                                      |
| ---------------- | ------- | --------- | -------------------------------------------------------------- |
| Transaction cost | 2%      | 1–10%     | Costos de cierre ida y vuelta en cada compra                   |
| Buy threshold    | 8%      | 0–20%     | Minimo upside percibido para activar una compra                |
| Opportunity cost | 4%      | 0–8%      | Penalizacion anual sobre capital inactivo                      |
| Max position     | 25%     | 10–50%    | Maximo capital por propiedad individual (diversificacion)      |
| Bid fraction     | 0.85    | 0.70–1.00 | Oferta = prediction × esta fraccion en subastas               |


### Market


| Parameter       | Default | Range        | Proposito                                                    |
| --------------- | ------- | ------------ | ------------------------------------------------------------ |
| Seller discount | -7%     | -15% to +10% | Sesgo sistematico de precios (negativo = vendedores en apuros) |
| Market noise    | 35%     | 5–50%        | Variacion aleatoria en precios de venta                      |


---

## Tu Flujo de Trabajo

1. **Construye un modelo** que prediga el precio de venta para cada propiedad del test
2. **Exporta un CSV** con columnas `zpid, predicted_price` (5,038 filas)
3. **Inicia sesion** en el dashboard de la competencia y sube tu CSV para la ronda actual
4. **Revisa los resultados** una vez que el organizador ejecute la simulacion y libere los resultados — distribucion de ROI, leaderboard, drill-down en escenarios representativos, mapa de propiedades
5. **Itera** — mejora tu modelo, vuelve a enviar para la siguiente ronda

Cada ronda de competencia se evalua sobre un pool separado de ~1,260 propiedades (el test set se divide en pools que no se solapan). Esto significa que no puedes sobreajustar al feedback de rondas anteriores — las propiedades cambian en cada ronda.
