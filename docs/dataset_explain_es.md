# Descripcion del Dataset

## Vista General

El dataset contiene ventas de propiedades residenciales en Miami / Sur de Florida. Tu objetivo es predecir el **precio de venta** de cada propiedad usando features estructurados, fotos del listado y descripciones de texto.

La division train/test esta predefinida. Entrenas con `train_processed.csv` (con precios conocidos) y predices sobre `test_processed.csv` (precios ocultos).

---

## Archivos

```
participant/
├── data/
│   ├── tabular/
│   │   ├── train_processed.csv    (11,840 filas — features + target)
│   │   └── test_processed.csv     (5,038 filas — solo features, sin target)
│   ├── images/
│   │   └── props/                 (398K fotos de propiedades, JPG/PNG)
│   ├── train_photo_metadata.csv   (151K filas — mapea zpid → archivos de imagen)
│   └── test_photo_metadata.csv    (64K filas)
├── submissions/
│   ├── template.csv               (zpid + predicted_price = 0)
│   └── purerandom_01..04.csv      (ejemplo: predicciones aleatorias)
├── scripts/                       (scripts de inicio de ejemplo)
└── docs/                          (este archivo + reglas del juego)
```

---

## Tabular Features

### Identificador

| Column | Descripcion |
|--------|-------------|
| `zpid` | ID unico de la propiedad. Usalo para unir datos tabulares con metadata de fotos y para enviar predicciones. |

### Target (solo train)

| Column | Descripcion |
|--------|-------------|
| `lastSoldPrice_hpi_adjusted` | Precio de venta ajustado por el Housing Price Index (dolares). Esto es lo que predices. |
| `log_price` | `log1p(lastSoldPrice_hpi_adjusted)` — se recomienda entrenar en escala logaritmica. Convierte de vuelta con `np.expm1()`. |

### Core Property Features

| Column | Type | Descripcion |
|--------|------|-------------|
| `bedrooms` | int | Numero de habitaciones |
| `bathrooms` | float | Numero de banos (incluye medio-banos como 0.5) |
| `livingArea` | float | Area habitable en pies cuadrados |
| `yearBuilt` | float | Ano de construccion de la propiedad |
| `latitude` | float | Latitud de la propiedad |
| `longitude` | float | Longitud de la propiedad |
| `lotAreaValue` | float | Tamano del lote en pies cuadrados |
| `photoCount` | int | Numero de fotos del listado |
| `homeType` | category | SINGLE_FAMILY, CONDO, TOWNHOUSE, MULTI_FAMILY, etc. |
| `zipcode` | category | Codigo postal de 5 digitos |

### Tax & Financial

| Column | Type | Descripcion |
|--------|------|-------------|
| `taxAssessedValue` | float | Valor catastral del condado |
| `propertyTaxRate` | float | Tasa efectiva de impuesto a la propiedad |
| `latest_tax_value` | float | Evaluacion fiscal mas reciente |
| `latest_tax_paid` | float | Pago de impuestos mas reciente |
| `num_tax_records` | int | Numero de registros fiscales historicos |
| `last_listing_price` | float | Precio de listado mas reciente |

### Transaction History

| Column | Type | Descripcion |
|--------|------|-------------|
| `num_sales` | int | Numero de ventas previas |
| `num_price_changes` | int | Numero de cambios de precio en el listado |

### Schools

| Column | Type | Descripcion |
|--------|------|-------------|
| `avg_school_rating` | float | Calificacion promedio de escuelas cercanas (1-10) |
| `max_school_rating` | float | Mejor calificacion de escuela cercana |
| `num_nearby_schools` | int | Cantidad de escuelas cercanas |
| `min_school_distance` | float | Distancia a la escuela mas cercana (millas) |

### Property Attributes

| Column | Type | Descripcion |
|--------|------|-------------|
| `has_hoa` | bool | Tiene asociacion de propietarios |
| `hoa_fee_monthly` | float | Cuota mensual de HOA (0 si no tiene) |
| `has_pool` | bool | La propiedad tiene piscina |
| `has_garage` | bool | La propiedad tiene garage |
| `has_waterfront` | bool | Propiedad frente al agua |

### Listing Tags

| Column | Type | Descripcion |
|--------|------|-------------|
| `tag_price_cut` | bool | Listado con reduccion de precio |
| `tag_new_construction` | bool | Construccion nueva |
| `tag_foreclosure` | bool | Ejecucion hipotecaria o propiedad bancaria |

### Derived Features (pre-calculados)

| Column | Type | Descripcion |
|--------|------|-------------|
| `property_age` | float | 2024 - yearBuilt |
| `bath_to_bed_ratio` | float | bathrooms / bedrooms |
| `log_living_area` | float | log1p(livingArea) |
| `log_lot_area` | float | log1p(lotAreaValue) |
| `zip_3digit` | category | Primeros 3 digitos del ZIP (agrupacion regional) |

### Text Features (pre-calculados desde `description`)

| Column | Type | Descripcion |
|--------|------|-------------|
| `description` | text | Descripcion completa del listado (texto libre). ~52% son plantillas genericas. |
| `desc_length` | int | Conteo de caracteres |
| `desc_word_count` | int | Conteo de palabras |
| `desc_is_boilerplate` | bool | Coincide con patron generico "This X square foot" |
| `desc_mentions_renovated` | bool | La descripcion menciona renovacion |
| `desc_mentions_pool` | bool | La descripcion menciona piscina |
| `desc_mentions_view` | bool | La descripcion menciona vista |
| `desc_mentions_new` | bool | La descripcion menciona "new" |

---

## Photo Metadata

Cada fila en `train_photo_metadata.csv` / `test_photo_metadata.csv` mapea una propiedad a una de sus fotos del listado.

| Column | Descripcion |
|--------|-------------|
| `zpid` | ID de propiedad (clave de union con datos tabulares) |
| `filename` | Nombre del archivo de imagen (ej. `1009018_007.jpg`) |
| `image_index` | Orden de la foto para esta propiedad (0 = foto principal) |
| `image_path` | Ruta relativa desde participant/ al archivo de imagen |
| `address`, `city`, `state` | Ubicacion de la propiedad |
| `bedrooms`, `bathrooms`, `homeType`, `yearBuilt` | Features basicos (duplicados de tabular por conveniencia) |

**Consejos:**
- Las propiedades tienen 1-30+ fotos. Empieza con `image_index == 0` (foto principal).
- Enfoques multi-imagen: extrae embeddings por imagen, luego mean-pool por propiedad.
- Las imagenes son las fotos reales del listado — exteriores, interiores, aereas, planos.

---

## Formato de Envio

Tu CSV de envio debe tener exactamente dos columnas:

```csv
zpid,predicted_price
1009018,450000.00
1001436,280000.00
...
```

- `zpid`: debe coincidir con el test set (5,038 filas)
- `predicted_price`: tu precio de venta predicho en dolares (no transformado con logaritmo)

Sube tu CSV a traves del dashboard de la competencia. Selecciona la ronda para la cual estas enviando — el dashboard valida tu archivo al subirlo.

---

## Consejos

1. **Entrena en log-price**: `log1p(price)` estabiliza el aprendizaje en el amplio rango de precios ($50K-$2M). Convierte de vuelta con `np.expm1()` para envios.

2. **Empieza simple**: Un LightGBM basico con 10 features tabulares da ~29% MAPE. Agrega features iterativamente.

3. **Las imagenes agregan senal**: Embeddings de CNN pre-entrenadas (ResNet, CLIP) capturan calidad visual, vecindario y tipo de propiedad. No entrenes una CNN desde cero — extrae features congelados.

4. **El texto es ruidoso**: ~52% de las descripciones son plantillas genericas. TF-IDF + SVD es un buen baseline. Embeddings de transformers (DistilBERT) pueden ayudar pero son mas dificiles de hacer bien.

5. **Ensambla con cuidado**: Combinar modalidades funciona mejor cuando cada una agrega senal unica. Usa predicciones out-of-fold (OOF) para stacking correcto.

6. **Cuidado con el leakage**: `taxAssessedValue` y `last_listing_price` correlacionan fuertemente con el precio de venta. Usarlos esta permitido pero puedes aprender menos.

7. **Mejora por ronda**: Envias predicciones por ronda via el dashboard. Usa las primeras rondas para probar modelos simples, mejora para rondas posteriores. La competencia hace forward-fill: si solo envias la ronda 1, se usa para todas las rondas posteriores tambien. Cada ronda evalua sobre un pool diferente de propiedades, asi que no puedes sobreajustar al feedback previo.
