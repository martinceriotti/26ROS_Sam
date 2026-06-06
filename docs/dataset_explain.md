# Dataset Description

## Overview

The dataset contains residential property sales in Miami / South Florida.  Your goal is to predict the **sale price** of each property using structured features, listing photos, and text descriptions.

The train/test split is pre-defined.  You train on `train_processed.csv` (with known prices) and predict on `test_processed.csv` (prices withheld).

---

## Files

```
participant/
├── data/
│   ├── tabular/
│   │   ├── train_processed.csv    (11,840 rows — features + target)
│   │   └── test_processed.csv     (5,038 rows — features only, no target)
│   ├── images/
│   │   └── props/                 (398K property photos, JPG/PNG)
│   ├── train_photo_metadata.csv   (151K rows — maps zpid → image files)
│   └── test_photo_metadata.csv    (64K rows)
├── submissions/
│   ├── template.csv               (zpid + predicted_price = 0)
│   └── purerandom_01..04.csv      (example: random predictions)
├── scripts/                       (sample starter scripts)
└── docs/                          (this file + game rules)
```

---

## Tabular Features

### Identifier

| Column | Description |
|--------|-------------|
| `zpid` | Unique property ID. Use this to join tabular data with photo metadata and to submit predictions. |

### Target (train only)

| Column | Description |
|--------|-------------|
| `lastSoldPrice_hpi_adjusted` | Sale price adjusted for the Housing Price Index (dollars). This is what you predict. |
| `log_price` | `log1p(lastSoldPrice_hpi_adjusted)` — training on log-scale is recommended. Convert back with `np.expm1()`. |

### Core Property Features

| Column | Type | Description |
|--------|------|-------------|
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms (includes half-baths as 0.5) |
| `livingArea` | float | Living area in square feet |
| `yearBuilt` | float | Year the property was built |
| `latitude` | float | Property latitude |
| `longitude` | float | Property longitude |
| `lotAreaValue` | float | Lot size in square feet |
| `photoCount` | int | Number of listing photos |
| `homeType` | category | SINGLE_FAMILY, CONDO, TOWNHOUSE, MULTI_FAMILY, etc. |
| `zipcode` | category | 5-digit ZIP code |

### Tax & Financial

| Column | Type | Description |
|--------|------|-------------|
| `taxAssessedValue` | float | County tax-assessed value |
| `propertyTaxRate` | float | Effective property tax rate |
| `latest_tax_value` | float | Most recent tax assessment |
| `latest_tax_paid` | float | Most recent tax payment |
| `num_tax_records` | int | Number of historical tax records |
| `last_listing_price` | float | Most recent listing price |

### Transaction History

| Column | Type | Description |
|--------|------|-------------|
| `num_sales` | int | Number of prior sales |
| `num_price_changes` | int | Number of listing price changes |

### Schools

| Column | Type | Description |
|--------|------|-------------|
| `avg_school_rating` | float | Average rating of nearby schools (1-10) |
| `max_school_rating` | float | Best nearby school rating |
| `num_nearby_schools` | int | Count of nearby schools |
| `min_school_distance` | float | Distance to nearest school (miles) |

### Property Attributes

| Column | Type | Description |
|--------|------|-------------|
| `has_hoa` | bool | Has a homeowners association |
| `hoa_fee_monthly` | float | Monthly HOA fee (0 if none) |
| `has_pool` | bool | Property has a pool |
| `has_garage` | bool | Property has a garage |
| `has_waterfront` | bool | Waterfront property |

### Listing Tags

| Column | Type | Description |
|--------|------|-------------|
| `tag_price_cut` | bool | Listed with a price cut |
| `tag_new_construction` | bool | New construction |
| `tag_foreclosure` | bool | Foreclosure or bank-owned |

### Derived Features (pre-computed)

| Column | Type | Description |
|--------|------|-------------|
| `property_age` | float | 2024 - yearBuilt |
| `bath_to_bed_ratio` | float | bathrooms / bedrooms |
| `log_living_area` | float | log1p(livingArea) |
| `log_lot_area` | float | log1p(lotAreaValue) |
| `zip_3digit` | category | First 3 digits of ZIP (regional grouping) |

### Text Features (pre-computed from `description`)

| Column | Type | Description |
|--------|------|-------------|
| `description` | text | Full listing description (free text). ~52% are boilerplate. |
| `desc_length` | int | Character count |
| `desc_word_count` | int | Word count |
| `desc_is_boilerplate` | bool | Matches generic "This X square foot" pattern |
| `desc_mentions_renovated` | bool | Description mentions renovation |
| `desc_mentions_pool` | bool | Description mentions pool |
| `desc_mentions_view` | bool | Description mentions view |
| `desc_mentions_new` | bool | Description mentions "new" |

---

## Photo Metadata

Each row in `train_photo_metadata.csv` / `test_photo_metadata.csv` maps a property to one of its listing photos.

| Column | Description |
|--------|-------------|
| `zpid` | Property ID (join key to tabular data) |
| `filename` | Image filename (e.g., `1009018_007.jpg`) |
| `image_index` | Photo order for this property (0 = primary photo) |
| `image_path` | Relative path from participant/ to the image file |
| `address`, `city`, `state` | Property location |
| `bedrooms`, `bathrooms`, `homeType`, `yearBuilt` | Basic features (duplicated from tabular for convenience) |

**Tips:**
- Properties have 1-30+ photos.  Start with `image_index == 0` (primary photo).
- Multi-image approaches: extract embeddings per image, then mean-pool per property.
- Images are the actual listing photos — exteriors, interiors, aerials, floor plans.

---

## Submission Format

Your submission CSV must have exactly two columns:

```csv
zpid,predicted_price
1009018,450000.00
1001436,280000.00
...
```

- `zpid`: must match the test set (5,038 rows)
- `predicted_price`: your predicted sale price in dollars (not log-transformed)

Upload your CSV through the competition dashboard. Select the round you are submitting for — the dashboard validates your file on upload.

---

## Tips

1. **Train on log-price**: `log1p(price)` stabilizes learning across the wide price range ($50K-$2M). Convert back with `np.expm1()` for submissions.

2. **Start simple**: A basic LightGBM on 10 tabular features gives ~29% MAPE. Add features iteratively.

3. **Images add signal**: Pre-trained CNN embeddings (ResNet, CLIP) capture visual quality, neighborhood, and property type. Don't train a CNN from scratch — extract frozen features.

4. **Text is noisy**: ~52% of descriptions are boilerplate. TF-IDF + SVD is a good baseline. Transformer embeddings (DistilBERT) can help but are harder to get right.

5. **Ensemble carefully**: Combining modalities works best when each adds unique signal. Use out-of-fold (OOF) predictions for proper stacking.

6. **Watch for leakage**: `taxAssessedValue` and `last_listing_price` correlate strongly with sale price. Using them is allowed but may teach you less.

7. **Per-round improvement**: You submit predictions per round via the dashboard. Use early rounds to test simple models, improve for later rounds. The competition forward-fills: if you only submit round 1, it's used for all subsequent rounds too. Each round evaluates on a different property pool, so you cannot overfit to previous feedback.
