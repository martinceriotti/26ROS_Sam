"""
Contenido estructurado del proyecto — Equipo SAM
MDM Austral 2026 — Labo 2: Property Investment Competition

Fuente unica de verdad para el PDF de desarrollo y el PPT de resumen,
para no transcribir los mismos numeros dos veces en archivos separados.
"""

PROJECT_TITLE = "Property Investment Competition"
TEAM = "Equipo SAM"
COURSE = "MDM Austral 2026 · Labo 2"
FINAL_MODEL = "round8_distress_fix_scale083.csv"

# ─── Paleta Austral (extraida de reports/UAustral-TextMiningClase01.pptx) ─────
NAVY   = "#1C3087"   # wordmark AUSTRAL
ORANGE = "#C68136"   # linea bajo "Posgrados"
PINK   = "#DB5B7D"   # triangulo decorativo 1
TEAL   = "#1C9C82"   # triangulo decorativo 2 / fondo cierre
INDIGO = "#4B5995"   # triangulo decorativo 3
GRAY   = "#5B5B5B"   # texto secundario
DARK   = "#222222"
GREEN_OK  = "#1C9C82"
RED_BAD   = "#C0392B"

MECHANICS = [
    "Cada propiedad tiene un precio de oferta (asking price) generado al azar "
    "con una distribucion normal (gaussiana), media -7% y desvio 35%: "
    "asking = valor_real x (1 + Normal(-7%, 35%)). En criollo: el asking suele "
    "salir un poco por debajo del valor real, pero con mucha variacion — a "
    "veces sale muy bajo, a veces muy alto.",
    "Compramos solo si nuestra prediccion supera el asking price en un 8% "
    "(pred > asking x 1.08).",
    "Si compramos, ofertamos pred x 0.85 en una subasta Vickrey (gana el mayor "
    "postor pero paga el precio del segundo postor). Ese x0.85 es una regla "
    "fija de la competencia, no algo que ajustamos nosotros — lo que si "
    "calibramos es la prediccion que entra a esa formula (alpha, escala).",
    "Se corren 1,000 simulaciones Monte Carlo, 4 rondas por simulacion, "
    "capital de $5M por ronda.",
    "Gana el equipo con mayor Mean ROI promedio (metrica principal). "
    "El Win Rate es secundario: no decide el ganador.",
]

ROUNDS = [
    dict(
        n="Ronda 1", date="2026-06-09", name="Baseline LightGBM",
        summary="Primer modelo: LightGBM con todos los features tabulares disponibles y "
                "validacion cruzada 5-fold.",
        result="wMAPE ~26% · Practice: Mean ROI 23.27%, Sharpe 1.15",
        verdict="ok", lesson=None,
    ),
    dict(
        n="Ronda 2", date="2026-06-09", name="Segmentacion + Feature Engineering",
        summary="Dos tecnicas distintas, probadas juntas en la misma ronda.\n\n"
                "(1) Segmentacion: en vez de un solo modelo, se entrenaron 3 modelos "
                "separados, uno por tipo de propiedad (SINGLE_FAMILY / CONDO / RESTO) — "
                "no es una columna nueva, es una decision de arquitectura.\n\n"
                "(2) Feature engineering: columnas nuevas calculadas a partir de las "
                "existentes — ratios financieros (ej. tax_per_sqft = impuesto / "
                "superficie), scores compuestos (ej. luxury_score = suma de pileta + "
                "vista al agua + garage + HOA alto) e interacciones (ej. school_x_area "
                "= rating de escuela x superficie).",
        result="Practice: Mean ROI 20.63%, Sharpe 1.06 — PEOR que Ronda 1",
        verdict="bad",
        lesson="Mas feature engineering no garantiza mejor resultado. Hay que medir "
               "cada cambio en Practice antes de asumir que mejora. Como las dos "
               "tecnicas se probaron juntas, no sabemos cual de las dos causo la caida.",
    ),
    dict(
        n="Ronda 3", date="2026-06-11", name="Deteccion de \"distressed sales ocultas\"",
        summary="Encontramos 16 propiedades con perdidas catastroficas: el impuesto "
                "(taxAssessedValue) era 1.3x a 6.8x el precio real de venta, sin ninguna "
                "bandera de foreclosure que las delatara. El modelo las sobre-predecia "
                "sistematicamente. Se agregaron features de ZIP relativas "
                "(tax_vs_zip_ratio, is_tax_outlier) y un techo (cap) al percentil 95 "
                "de precio por ZIP.",
        result="OOF MAPE 27.0% · Sharpe 1.074 vs 1.049 de Ronda 2",
        verdict="ok", lesson=None,
    ),
    dict(
        n="Ronda 3-4", date="2026-06-27", name="Embeddings de imagenes (CLIP)",
        summary="Se proceso la foto principal de cada propiedad con CLIP (red neuronal "
                "de vision-lenguaje) y se redujo a 32 componentes principales (PCA). "
                "Tambien se calculo un score visual de deterioro (clip_distress_score) "
                "de forma zero-shot, sin entrenar nada especifico.",
        result="wMAPE 22.4% · Practice: Mean ROI 49.7% (+92% vs ronda anterior)",
        verdict="great",
        lesson="El score visual de distress no mejoro el error promedio global, pero "
               "mejoro muchisimo la seleccion de que propiedades comprar en la subasta. "
               "Una señal puede no bajar el MAPE y aun asi ser muy valiosa para decidir.",
    ),
    dict(
        n="Ronda 4", date="2026-06-26", name="Regresion cuantil (percentil 35, no la media)",
        summary="Detectamos que 2 propiedades causaron el 52% de todas las perdidas "
                "acumuladas (el modelo las habia sobre-predicho 5.7x y 8.8x el valor real). "
                "El error es asimetrico en este juego: sobreestimar destruye capital "
                "(compramos caro), subestimar solo cuesta la oportunidad (no compramos). "
                "Por eso se cambio el modelo para predecir el percentil 35 de la "
                "distribucion de precio en vez de la media.",
        result="Practice: Mean ROI 44.47%, Sharpe 2.05, Prob. positivo 99.2%",
        verdict="great", lesson=None,
    ),
    dict(
        n="Simulador propio", date="2026-06-26", name="Monte Carlo local calibrado con datos reales",
        summary="Se construyo un simulador local que reproduce exactamente la mecanica "
                "del dashboard, calibrado con 946 ofertas reales observadas de 6 equipos "
                "rivales en una ronda de competencia ya jugada. Sirve para probar ideas "
                "sin gastar rondas reales ni subir al dashboard cada vez.",
        result="En la calibracion inicial, nuestro equipo ganaba el 34.3% de las simulaciones",
        verdict="ok", lesson=None,
    ),
    dict(
        n="Ronda 5", date="2026-06-27", name="Mas conservador, penalidad dura a foreclosure",
        summary="Se valido que alpha=0.35 (Ronda 4) era el punto correcto: bajarlo a 0.30 "
                "hundia el win rate de 34.3% a 12.4%. En una subasta Vickrey, predecir "
                "demasiado bajo simplemente regala las propiedades buenas a los "
                "competidores. Tambien se descarto last_listing_price como señal de "
                "distress: 22% del train tiene ese ratio bajo por precios historicos "
                "viejos, no por problemas reales de la propiedad.",
        result="Confirma alpha=0.35 como calibracion base",
        verdict="ok", lesson=None,
    ),
    dict(
        n="Ronda 6", date="2026-06-27", name="Experimento: prediccion central + descuento quirurgico",
        summary="Se probo alpha=0.50 (prediccion central, no conservadora) buscando "
                "comprar mas volumen, compensando con un descuento solo a las "
                "propiedades con peor score visual de distress (percentil 90).",
        result="Mas volumen de compra, pero mayor tasa de errores costosos",
        verdict="mixed", lesson=None,
    ),
    dict(
        n="Ronda 7", date="2026-07-03", name="Punto intermedio: alpha=0.42",
        summary="En una ronda real, un equipo competidor gano comprando mucho mas "
                "volumen (15.4 props/ronda) que nosotros (9.7 props/ronda) aunque con "
                "menor precision por compra. Se probo alpha=0.42 como punto intermedio "
                "entre conservador y agresivo.",
        result="Buscando 12-13 props/ronda con hit rate > 65%",
        verdict="mixed", lesson=None,
    ),
    dict(
        n="Ronda 8", date="2026-07-03", name="Fix quirurgico final (mejor modelo)",
        summary="Se uso el resultado REAL de subastas ya jugadas (exports de la "
                "competencia) para identificar con certeza 6 propiedades donde el modelo "
                "se equivoco por hasta 4x el valor real. Se forzo su prediccion a "
                "valor_real x 0.92. Se agrego ademas un detector automatico para el "
                "train set: cualquier propiedad con ratio prediccion/verdadero > 2.5x.",
        result="Se convirtio en el mejor modelo del proyecto hasta la calibracion final",
        verdict="great", lesson=None,
    ),
    dict(
        n="Ronda 9", date="2026-07-03 a 2026-07-18",
        name="Intento con texto (LLM local)",
        summary="Se extrajeron ~18 features semanticas (condicion, vista, penthouse, "
                "amenities, señales de distress) de las descripciones de las propiedades "
                "usando un modelo de lenguaje corriendo LOCAL y gratis (Ollama, Qwen2.5 "
                "1.5B). Cobertura real: 48% de las propiedades tenian descripcion "
                "procesable.\n\n"
                "Primer intento (relleno con 0/1 por defecto donde faltaba dato): "
                "EMPEORO el modelo — Mean ROI bajo de 29.1% a 26.3%. Una comparacion "
                "controlada confirmo que la caida era 100% atribuible a esas 18 columnas "
                "(todo lo demas era identico a Ronda 8).\n\n"
                "Diagnostico: el valor de relleno (0) colisionaba con una categoria real "
                "del LLM (\"unknown\" tambien es 0), mezclando \"no tengo dato\" con "
                "\"el LLM realmente no supo\". Se corrigio dejando NaN real donde no "
                "habia descripcion. La brecha se redujo mucho (de -2.8pp a -0.5pp) pero "
                "nunca supero a Ronda 8.\n\n"
                "Tambien se probo detectar distressed en el test set usando proxies "
                "disponibles (fotos, tag de foreclosure, features de texto). Ningun "
                "proxy tuvo poder predictivo mejor que el azar (precision ~1.6%, "
                "igual a la tasa base).",
        result="Pausado: no aporta con la calidad/cobertura de extraccion actual",
        verdict="bad",
        lesson="Mas señal no es necesariamente buena señal si es ruidosa o parcial. "
               "Medir con una comparacion controlada (cambiar una sola cosa a la vez) "
               "evita atribuir una mejora o caida a la causa equivocada.",
    ),
    dict(
        n="Calibracion final", date="2026-07-18", name="Barrido de escala sobre Ronda 8",
        summary="Con Ronda 8 como mejor modelo, se probo multiplicar todas sus "
                "predicciones por un factor de escala (arriba y abajo de 1.0) para ver "
                "el efecto en el ROI. Primero se valido con el simulador local "
                "(calibrado con datos reales de subastas pasadas): se encontro una "
                "meseta optima entre 0.80 y 0.85, con el Mean ROI subiendo de 8.26% a "
                "~10.1-10.3% (+24% relativo). Se confirmo despues en el dashboard real "
                "(Practice) con 3 corridas independientes, con distintos campos "
                "competitivos: la escala 0.83 le gano de forma consistente y nunca "
                "revertida a Ronda 8 sin escalar, por un margen de 2x a 4x en Mean ROI.",
        result="Mejora real, grande y reproducible — validada en 3 corridas independientes",
        verdict="great", lesson=None,
    ),
]

SCALE_SWEEP_LOCAL = [
    # (escala, Mean ROI %, Sharpe)
    (0.55, 6.70, 1.892), (0.60, 7.95, 1.971), (0.65, 8.59, 2.128),
    (0.68, 9.11, 2.168), (0.70, 9.20, 2.218), (0.72, 9.46, 2.341),
    (0.75, 9.80, 2.424), (0.78, 9.96, 2.426), (0.80, 10.11, 2.515),
    (0.83, 9.95, 2.575), (0.85, 10.33, 2.506), (0.87, 10.08, 2.471),
    (0.90, 9.99, 2.539), (0.92, 9.51, 2.409), (0.94, 9.21, 2.420),
    (0.97, 8.85, 2.360), (1.00, 8.26, 2.271),
]

SCALE_PRACTICE_RUNS = [
    # (corrida, n_modelos_compitiendo, ROI scale083, ROI scale085, ROI round8_baseline)
    ("#1", 3, 76.55, 31.43, 27.02),
    ("#2", 7, 61.21, 30.08, 15.48),
    ("#3", 9, 46.85, 29.06, 10.99),
]

ROUND9_COMPARISON = [
    # (variante, Mean ROI, Sharpe, CVaR5)
    ("Round 8 (sin texto)", 29.08, 1.612, 0.78),
    ("Round 9 relleno dummy (0/1)", 26.26, 1.449, -1.79),
    ("Round 9 relleno NaN (corregido)", 27.10, 1.512, 0.17),
]

DISTRESS_PROXY_TEST = [
    # (proxy, precision %, recall %)
    ("tag_foreclosure", 0.0, 0.0),
    ("num_price_changes > 2", 1.0, 12.2),
    ("clip_distress_score > 0.7", 2.0, 4.6),
    ("llm_is_fixer", 3.4, 17.8),
    ("raw_listing_to_tax < 0.5", 1.6, 22.3),
]
DISTRESS_PROXY_BASELINE = 1.66  # % tasa base de propiedades realmente distressed

META_LESSONS = [
    ("El error promedio no es lo que gana la competencia",
     "En Ronda 4 encontramos que 2 propiedades causaron el 52% de todas las "
     "perdidas. Bajar el MAPE global importa menos que eliminar esos pocos "
     "errores extremos que destruyen capital."),
    ("Mas features no siempre ayudan",
     "Ronda 2 (mas feature engineering) y Ronda 9 (features de texto por LLM) "
     "empeoraron el modelo. Hay que medir cada cambio, no asumir que suma."),
    ("Calibrar cuanto arriesgar importa tanto como el modelo en si",
     "El ajuste final de escala (multiplicar las predicciones por 0.83) genero "
     "una mejora de ROI comparable a la de agregar imagenes — y costo una "
     "tarde de trabajo, no semanas."),
    ("Validar siempre en Practice antes de gastar una ronda real",
     "Cada modelo nuevo se subio primero como OOF al tab Practice del "
     "dashboard. Solo se uso una ronda real cuando el Practice confirmaba "
     "una mejora."),
    ("Desconfiar de una sola corrida",
     "La simulacion de Practice varia segun cuantos modelos compiten a la vez "
     "en la misma corrida. Antes de decidir, se repitieron las comparaciones "
     "clave 2 o 3 veces para confirmar que la diferencia era real y no ruido "
     "del muestreo Monte Carlo."),
    ("Comparar siempre contra el campeon vigente, no contra una version vieja",
     "Kelly y Edge parecian ganar por mucho contra versiones sin escalar — pero "
     "comparados de forma justa contra scale083 (el campeon real), quedaron "
     "empatados o por debajo. Confirmar que un desafiante le gana al mejor "
     "modelo actual, no a un rival mas facil, evita una falsa sensacion de mejora."),
]

KELLY_RESULTS = [
    # (segmento, Hit Rate %, W ganancia %, L perdida %, Kelly f*, escala sugerida)
    ("SF",    75.0, 30.8, 19.3, 3.079, 0.828),
    ("CONDO", 70.5, 30.4, 16.6, 3.282, 0.832),
    ("REST",  72.7, 30.0, 17.8, 3.171, 0.830),
]
KELLY_FLAT_ROI = 9.95      # Mean ROI local, escala plana 0.83 (misma base cruda)
KELLY_SEGMENT_ROI = 10.15  # Mean ROI local, escala Kelly por segmento

EDGE_RESULTS = [
    # (bucket, Hit Rate %, EV por subasta como % del bid, escala sugerida)
    ("SF-bajo", 78.6, 1.55, 0.806), ("SF-medio", 85.9, 1.88, 0.847), ("SF-alto", 89.9, 2.04, 0.867),
    ("CONDO-bajo", 78.8, 1.50, 0.800), ("CONDO-medio", 86.0, 1.69, 0.823), ("CONDO-alto", 91.3, 1.86, 0.845),
    ("REST-bajo", 74.2, 1.47, 0.796), ("REST-medio", 86.4, 1.69, 0.823), ("REST-alto", 93.0, 2.10, 0.876),
]
EDGE_FLAT_ROI = 9.95     # Mean ROI local, escala plana 0.83
EDGE_BUCKET_ROI = 10.22  # Mean ROI local, escala por bucket fino (segmento x tercil de precio)

# Validacion final en el dashboard real de Practice: Kelly y Edge comparados
# contra el campeon vigente (scale083), no contra versiones sin escalar.
PRACTICE_FINAL_VALIDATION = [
    # (modelo, Mean ROI %, Sharpe, Hit Rate %, Props/sim)
    ("round8_distress_fix_scale083 (campeon)", 32.72, 2.732, 81.0, 13.8),
    ("kelly_segment", 32.57, 2.702, 82.1, 15.7),
    ("edge_bucket", 31.01, 2.800, 85.4, 9.3),
]

STUDY_QA = [
    ("Que mide el wMAPE y en que se diferencia del Mean ROI?",
     "wMAPE mide el error de prediccion de precio (cuanto se equivoca el modelo, "
     "en %, ponderado por precio real). Mean ROI mide la ganancia real de "
     "invertir siguiendo esas predicciones en la simulacion de subasta. Son "
     "independientes: un modelo puede tener buen wMAPE y mal ROI, o al reves "
     "(paso en Ronda 3-4 y en Ronda 9)."),
    ("Como se genera el asking price de cada propiedad en la simulacion?",
     "asking = valor_real x (1 + shock), con shock de una distribucion normal, "
     "media -7% y desvio 35%. En promedio el asking esta un poco por debajo del "
     "valor real, pero con mucha variacion."),
    ("Que es una subasta Vickrey y por que la usa la competencia?",
     "Es una subasta de segundo precio: gana el que mas ofrece, pero paga lo "
     "que ofrecio el segundo. Reduce el incentivo a subestimar cuanto vale algo "
     "para uno."),
    ("Cuantas propiedades tiene el train set y cuantas el test set?",
     "Train: 11,840 propiedades con precio real conocido. Test: 5,038 sin "
     "precio, sobre las que se generan las predicciones de la competencia real."),
    ("Que metrica define al ganador: Mean ROI o Win Rate?",
     "Mean ROI (promedio de ganancia en las 1,000 simulaciones). Win Rate es "
     "secundario, no decide el ganador."),
    ("Por que el 0.85 de la oferta no es algo que calibraron ustedes?",
     "Es una regla fija de la mecanica de la competencia (oferta = prediccion "
     "x 0.85), dada por el curso. Lo que si calibraron fue la prediccion que "
     "entra a esa formula: el alpha de la regresion cuantil y la escala final "
     "(0.83)."),
    ("Que diferencia hay entre segmentacion y feature engineering en Ronda 2?",
     "Segmentacion = entrenar 3 modelos separados por tipo de propiedad "
     "(arquitectura, no una columna). Feature engineering = columnas nuevas "
     "(ratios, scores compuestos, interacciones). Se probaron juntas en la "
     "misma ronda, asi que no se puede aislar cual de las dos causo la caida."),
    ("Que son los embeddings CLIP y para que sirvio el PCA?",
     "CLIP es una red neuronal ya entrenada que convierte cada foto en 512 "
     "numeros que resumen su contenido visual. PCA comprime esos 512 numeros "
     "a 32, quedandose con lo mas relevante, antes de dárselo al modelo."),
    ("Por que el fix de Ronda 8 corrige train y test de forma distinta?",
     "En test solo se conocen los valores reales de 6 propiedades puntuales "
     "(por resultados de subastas ya jugadas), asi que se corrigen a mano. En "
     "train se conoce el valor real de las 11,840 propiedades, asi que se "
     "arma una regla automatica (ratio prediccion/verdadero > 2.5x)."),
    ("Diferencia entre las 6 de test y la deteccion automatica en train?",
     "Las 6 de test se identificaron con certeza porque ya se jugo esa subasta "
     "y se vio el resultado real. La regla de train es estadistica (umbral de "
     "ratio), aplicada a gran escala sin mirar caso por caso."),
    ("Por que el clip_distress_score mejoro el ROI sin bajar el MAPE?",
     "El MAPE es un promedio sobre todas las propiedades; el score ayuda a "
     "evitar comprar especificamente las peores (visualmente deterioradas), "
     "pocas pero muy costosas si se compran mal. Cambia la decision de compra "
     "mas de lo que cambia la prediccion promedio."),
    ("Por que sobreestimar es mas costoso que subestimar?",
     "Si se sobreestima, se puede ganar la subasta pagando de mas -> perdida "
     "real de capital. Si se subestima, simplemente no se compra esa "
     "propiedad -> se pierde la oportunidad, pero no plata. Por eso conviene "
     "sesgar la prediccion hacia abajo (percentil 35)."),
    ("En el grafico local 0.85 gana — por que igual eligieron 0.83?",
     "Ese grafico es una simulacion aproximada casera (datos de una ronda "
     "vieja de competidores). La decision real se tomo subiendo ambas escalas "
     "al dashboard real de Practice: ahi 0.83 le gano a 0.85 en las 3 corridas "
     "independientes que se hicieron."),
    ("Que asume Kelly clasico que no se cumple en esta subasta?",
     "Que perder = perder toda la apuesta. Aca, si se pierde la subasta, no se "
     "pierde nada (el capital queda intacto) — solo se pierde si se gana la "
     "subasta pagando de mas. Se adapto la formula a esa condicion."),
    ("Por que los proxies para detectar distressed no funcionaron?",
     "Se midio precision y recall de foreclosure, fotos, tags y texto contra "
     "los casos reales — ninguno supero la tasa base (~1.66%, o sea, adivinar "
     "al azar). Las perdidas catastroficas parecen ser idiosincraticas, no un "
     "patron detectable en los datos disponibles."),
    ("Por que comparar Kelly/Edge contra round8 sin escalar era enganoso?",
     "Round8 sin escalar es la peor version conocida del modelo. Cualquier "
     "variante nueva le gana facil a esa base debil. La comparacion justa es "
     "contra el campeon real (scale083), y ahi la ventaja de Kelly y Edge se "
     "achico o desaparecio."),
    ("Por que Edge perdio en la prueba real si localmente mejoraba mas?",
     "Edge se volvio mas selectivo (compraba menos propiedades, ~9 vs ~14-17 "
     "del campeon) buscando mayor Hit Rate. Eso le dio mejor precision pero "
     "menos volumen, y en el juego real el volumen importa tanto como acertar."),
    ("ROI alto y variable vs ROI un poco mas bajo y estable — cual conviene?",
     "Segun las reglas, gana el mayor Mean ROI promedio, asi que en el margen "
     "el ROI mas alto gana aunque sea mas variable. Pero el curso tambien "
     "valora el Std bajo como objetivo secundario — por eso Kelly (mejor "
     "Sharpe) es un candidato razonable si se prioriza consistencia."),
    ("Por que no alcanza con correr Practice una sola vez?",
     "La simulacion varia segun cuantos modelos compiten a la vez y tiene "
     "ruido de muestreo Monte Carlo. Se repitieron las comparaciones clave "
     "2-3 veces con distintos campos competitivos antes de decidir."),
    ("Que retomarian con un mes mas de trabajo?",
     "Probablemente texto/LLM, pero cambiando el modelo de extraccion (uno "
     "mas grande que qwen2.5:1.5b) y mejorando la cobertura (hoy 48%) antes "
     "de reintentarlo — el problema no era el enfoque sino la calidad/"
     "cobertura de los datos de entrada."),
]

STUDY_MC = [
    ("Que significa wMAPE?",
     ["Win/Mean Auction Profit Estimate",
      "Una medida de riesgo del portafolio",
      "Weighted Mean Absolute Percentage Error — error de prediccion de precio",
      "El porcentaje de subastas ganadas"], 2),
    ("Si compramos y pagamos de mas por una propiedad...",
     ["No pasa nada, se ajusta automaticamente",
      "Se cancela la compra",
      "Ganamos el Win Rate igual",
      "Perdemos capital real"], 3),
    ("Cual es la metrica principal que define al ganador de la competencia?",
     ["Mean ROI", "wMAPE", "Win Rate", "Sharpe"], 0),
    ("El multiplicador de la oferta (prediccion x 0.85)...",
     ["Lo calibramos nosotros en la Ronda 7",
      "Es una regla fija de la competencia",
      "Cambia segun el segmento",
      "Se ajusto junto con la escala 0.83"], 1),
    ("En la Ronda 2, segmentacion se refiere a...",
     ["Una columna nueva que indica el tipo de propiedad",
      "Dividir el dataset en train/test",
      "Un score compuesto de lujo",
      "Entrenar 3 modelos separados por tipo de propiedad"], 3),
    ("Que hace PCA con los embeddings de CLIP?",
     ["Los comprime de 512 a 32 numeros, conservando lo mas relevante",
      "Los entrena desde cero para propiedades",
      "Calcula el score de distress",
      "Elimina las fotos de mala calidad"], 0),
    ("Por que se uso el percentil 35 en vez de la media?",
     ["Porque es mas facil de calcular",
      "Porque el dataset tiene 35% de datos faltantes",
      "Porque sobreestimar es mas costoso que subestimar en este juego",
      "Porque asi lo pide la competencia"], 2),
    ("Las 6 propiedades del fix quirurgico en test se identificaron...",
     ["Con un modelo de deteccion automatica",
      "Mirando resultados reales de subastas ya jugadas",
      "Al azar",
      "Usando el clip_distress_score"], 1),
    ("Al probar proxies (fotos, tags, texto) para detectar distressed en test...",
     ["Ninguno predijo mejor que el azar",
      "Funcionaron mejor que el modelo principal",
      "Solo el tag de foreclosure funciono bien",
      "Funcionaron, pero solo en CONDO"], 0),
    ("Kelly clasico asume que perder significa...",
     ["Pagar de mas en la subasta",
      "No participar en la subasta",
      "Bajar el Hit Rate",
      "Perder toda la apuesta"], 3),
    ("En el barrido de escala, el grafico local mostraba que...",
     ["0.83 era claramente el mejor punto",
      "0.85 tenia un pico un poco mas alto que 0.83",
      "1.00 (sin escalar) era el mejor",
      "No habia diferencia entre escalas"], 1),
    ("Por que se eligio 0.83 en vez de 0.85 pese al grafico local?",
     ["Porque en 3 corridas reales del dashboard, 0.83 le gano a 0.85",
      "Por practicidad, es un numero redondo",
      "Porque 0.85 no se pudo probar",
      "Porque el profesor lo indico"], 0),
    ("Edge, comparado con Kelly, encontro diferencias reales al cortar por...",
     ["Segmento solamente",
      "Fecha de la venta",
      "Segmento x tercil de precio dentro del segmento",
      "Cantidad de fotos"], 2),
    ("Por que Edge perdio en la prueba real pese a mejorar mas en la simulacion local?",
     ["Por un error en el codigo",
      "Porque no se probo nunca en Practice",
      "Porque el Hit Rate bajo",
      "Porque compro muchas menos propiedades (menos volumen)"], 3),
    ("Cual fue el modelo final elegido para la presentacion?",
     ["oof_round9_text_llm_llm5704", "round8_distress_fix_scale083",
      "round8_kelly_segment", "round8_edge_bucket"], 1),
]

FINAL_RECOMMENDATION = (
    "Modelo elegido para la presentacion final: round8_distress_fix_scale083.csv\n\n"
    "Es el modelo de Ronda 8 (tabular + imagenes CLIP + fix quirurgico de "
    "propiedades distressed) con todas las predicciones escaladas x0.83. Se "
    "probaron dos refinamientos mas finos para tratar de superarlo (Kelly por "
    "segmento y Edge por segmento x precio, secciones 6 y 7): ninguno le gano "
    "de forma clara en el dashboard real de Practice. La escala 0.83 queda "
    "confirmada como la mejor opcion disponible."
)
