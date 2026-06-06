# MDM Austral 2026 — Labo 2: Property Investment Competition
# Equipo: Aragorn

## El proyecto

Competencia de predicción de precios de propiedades residenciales en Miami/Sur de Florida.
El objetivo es predecir el precio de venta de cada propiedad en el test set.
Las predicciones se evalúan mediante una simulación Monte Carlo de inversión inmobiliaria
donde el modelo más preciso genera mayor ROI.

## Estructura del repositorio

```
participant/
├── data/
│   ├── tabular/
│   │   ├── train_processed.csv    (11,840 filas — features + target)
│   │   └── test_processed.csv     (5,038 filas — sin target)
│   ├── images/props/              (~199,000 fotos JPG de propiedades)
│   ├── train_photo_metadata.csv
│   └── test_photo_metadata.csv
├── notebooks/
│   └── eda.ipynb                  — EDA completo (12 secciones)
├── models/
│   ├── baseline_lgbm.py           — Ronda 1: LightGBM baseline con CV 5-fold
│   └── round2_segments.py         — Ronda 2: Feature engineering + modelos por segmento
├── scripts/                       — Scripts de ejemplo provistos por el curso
│   ├── 01_lgbm_basic.py
│   ├── 02_image_embeddings.py
│   ├── 03_text_embeddings.py
│   └── 04_ensemble.py
├── submissions/                   — CSVs para subir al dashboard
│   ├── template.csv
│   ├── baseline_lgbm.csv          — Predicciones test (generado por baseline_lgbm.py)
│   ├── oof_baseline_lgbm.csv      — OOF train (subir al tab Practice del dashboard)
│   ├── round2_segments.csv        — Predicciones test (generado por round2_segments.py)
│   └── oof_round2_segments.csv    — OOF train (subir al tab Practice del dashboard)
└── docs/                          — Documentación del curso (dataset, mecánica, dashboard)
```

## Dataset — lo más importante

- **Target:** `lastSoldPrice_hpi_adjusted` — precio de venta ajustado por HPI
- **Entrenar en:** `log_price = log1p(precio)`, convertir de vuelta con `np.expm1()`
- **Rango de precios:** $51K – $1.99M, media $559K
- **Tipos de propiedad:** SINGLE_FAMILY (41%), CONDO (40%), resto menor
- **Missing grave:** `lotAreaValue` (45%), `last_listing_price` (33%)
- **Features leaky** (correlación alta pero derivan del precio): `taxAssessedValue`, `latest_tax_value`, `latest_tax_paid` — se pueden usar, pero no enseñan drivers reales

## Mecánica de la simulación (cómo se evalúa)

```
asking_price = true_value × (1 + Normal(-0.07, 0.35))
Compramos si: predicción > asking_price × 1.08
Nuestra oferta en subasta: predicción × 0.85
Subasta Vickrey: gana el mayor postor, paga el segundo precio
```

- 1,000 simulaciones Monte Carlo por ronda de competencia
- 4 rondas internas por simulación, capital $5M por ronda
- **Ganador:** equipo con mayor win rate (% de simulaciones en primer lugar)
- **Métrica a optimizar:** ROI medio alto + std del ROI bajo (predicciones consistentes)

## Convenciones de código

- Correr todos los scripts desde `participant/` como directorio de trabajo
- CV siempre es 5-fold con `random_state=42`
- Cada script genera dos CSVs: uno para test (`submissions/nombre.csv`) y uno OOF para práctica (`submissions/oof_nombre.csv`)
- Los features de ZIP (basados en el target) se calculan **dentro del fold** para evitar leakage
- Imputación de missing: mediana por `homeType`, fallback a mediana global

## Ventaja táctica — tab Practice del dashboard

El dashboard tiene dos tabs al subir predicciones:
- **Competition (Test Set):** usa rondas reales de la competencia (limitadas)
- **Practice (Train Set OOF):** acepta predicciones OOF del train set y corre la simulación sin gastar rondas reales

**Protocolo:** por cada modelo nuevo, subir primero el OOF al tab Practice, ver ROI medio y std,
y solo si mejora al modelo anterior subir el test CSV a la ronda real.

El drill-down de resultados muestra `True Value` y bids de competidores.
Las predicciones de competidores se pueden inferir como `bid / 0.85`.

## Roadmap de modelos

| Ronda | Archivo | Estado | Descripción |
|---|---|---|---|
| 1 | `models/baseline_lgbm.py` | Listo | LightGBM con todos los features tabulares, CV 5-fold |
| 2 | `models/round2_segments.py` | Listo | + Feature engineering + modelos separados SF/CONDO/REST |
| 3 | — | Pendiente | + Embeddings de imágenes (CLIP o ResNet, foto principal) |
| 4 | — | Pendiente | Ensemble completo + calibración |

## Features de Ronda 2 (nuevos respecto al baseline)

**Ratios financieros:** `tax_per_sqft`, `listing_to_tax_ratio`, `tax_to_area`, `hoa_to_area`

**Scores compuestos:**
- `luxury_score` = pool + waterfront + garage + (HOA > $200)
- `distress_score` = foreclosure + price_cut + (num_price_changes > 2)

**Interacciones:** `school_x_area`, `waterfront_x_lat`, `age_x_school`, `beds_x_baths`

**Agregaciones por ZIP** (calculadas dentro del fold):
`zip_median_log_price`, `zip_std_log_price`, `zip_count`, `zip_median_area`, `zip_price_per_sqft_median`

## Próximos pasos

1. Obtener acceso al dashboard y subir OOF de baseline y round2 al tab Practice
2. Comparar ROI medio y std entre ambos modelos
3. Analizar el drill-down: calibración por segmento, oportunidades perdidas
4. Implementar embeddings de imágenes (Ronda 3)
