"""
Ronda 2 — Feature engineering + modelos por segmento.

Mejoras sobre baseline_lgbm.py:
1. Feature engineering:
   - Agregaciones por ZIP (mediana precio, precio/ft²) — calculadas dentro del fold para evitar leakage
   - Ratios: tax_per_sqft, listing_to_tax_ratio, tax_to_area
   - Scores compuestos: luxury_score, distress_score
   - Interacciones: school × area, waterfront × latitude

2. Modelos por segmento:
   - SINGLE_FAMILY  (~4,900 propiedades)
   - CONDO          (~4,700 propiedades)
   - REST           (resto: townhouse, apartment, multi-family, etc.)
   Cada segmento entrena su propio LightGBM con los features más relevantes para él.

Run desde participant/:
    python models/round2_segments.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Cargar datos ──────────────────────────────────────────────────────────────
train = pd.read_csv('data/tabular/train_processed.csv')
test  = pd.read_csv('data/tabular/test_processed.csv')

TARGET = 'log_price'

SEGMENT_COL = 'segment'

def assign_segment(df):
    df = df.copy()
    df[SEGMENT_COL] = 'REST'
    df.loc[df['homeType'] == 'SINGLE_FAMILY', SEGMENT_COL] = 'SF'
    df.loc[df['homeType'] == 'CONDO',         SEGMENT_COL] = 'CONDO'
    return df

train = assign_segment(train)
test  = assign_segment(test)

# ── Preprocesamiento base (igual que baseline) ────────────────────────────────
CAT_FEATURES = ['homeType', 'zipcode']

def preprocess(df):
    df = df.copy()
    for col in ['bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
                'lotAreaValue', 'taxAssessedValue', 'latest_tax_value',
                'latest_tax_paid', 'property_age', 'log_living_area', 'log_lot_area']:
        if col in df.columns and df[col].isnull().any():
            medians = df.groupby('homeType')[col].transform('median')
            df[col] = df[col].fillna(medians).fillna(df[col].median())

    if 'last_listing_price' in df.columns:
        mask = df['last_listing_price'].isnull()
        df.loc[mask, 'last_listing_price'] = df.loc[mask, 'taxAssessedValue'] * 1.1

    df['bath_to_bed_ratio'] = df['bath_to_bed_ratio'].fillna(0)

    for col in ['latitude', 'longitude']:
        df[col] = df[col].fillna(df[col].median())

    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    return df


train = preprocess(train)
test  = preprocess(test)

# ── Feature engineering estático (sin usar el target → sin leakage) ───────────
def add_static_features(df):
    df = df.copy()

    # Ratios financieros
    df['tax_per_sqft']         = df['taxAssessedValue'] / (df['livingArea'] + 1)
    df['listing_to_tax_ratio'] = df['last_listing_price'] / (df['taxAssessedValue'] + 1)
    df['tax_to_area']          = df['latest_tax_value'] / (df['livingArea'] + 1)
    df['hoa_to_area']          = df['hoa_fee_monthly'] / (df['livingArea'] + 1)

    # Scores compuestos
    df['luxury_score']  = (df['has_pool'] + df['has_waterfront'] +
                           df['has_garage'] + (df['hoa_fee_monthly'] > 200).astype(int))
    df['distress_score'] = (df['tag_foreclosure'] + df['tag_price_cut'] +
                            (df['num_price_changes'] > 2).astype(int))

    # Interacciones
    df['school_x_area']     = df['avg_school_rating'] * df['log_living_area']
    df['waterfront_x_lat']  = df['has_waterfront'] * df['latitude']
    df['age_x_school']      = df['property_age'] * df['avg_school_rating']
    df['beds_x_baths']      = df['bedrooms'] * df['bathrooms']

    # Habitaciones totales
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    return df


train = add_static_features(train)
test  = add_static_features(test)

# ── Feature engineering con el target (debe calcularse dentro del fold) ───────
ZIP_AGG_FEATURES = ['zip_median_log_price', 'zip_price_per_sqft_median',
                    'zip_std_log_price', 'zip_count', 'zip_median_area']

def add_zip_features(train_fold, apply_df):
    """
    Calcula estadísticas por ZIP a partir de train_fold y las aplica a apply_df.
    Siempre recibe train_fold para evitar leakage del target.
    """
    agg = train_fold.groupby('zipcode').agg(
        zip_median_log_price  = ('log_price', 'median'),
        zip_std_log_price     = ('log_price', 'std'),
        zip_count             = ('zpid', 'count'),
        zip_median_area       = ('livingArea', 'median'),
    ).reset_index()

    # Precio/ft² aproximado usando tax como proxy (no usa log_price del apply_df)
    train_fold = train_fold.copy()
    train_fold['_price_per_sqft'] = train_fold['taxAssessedValue'] / (train_fold['livingArea'] + 1)
    agg2 = train_fold.groupby('zipcode')['_price_per_sqft'].median().reset_index()
    agg2.columns = ['zipcode', 'zip_price_per_sqft_median']

    agg = agg.merge(agg2, on='zipcode', how='left')

    result = apply_df.merge(agg, on='zipcode', how='left')

    # Fallback global para ZIPs que no aparecen en train_fold (test con ZIPs raros)
    for col in ZIP_AGG_FEATURES:
        global_val = agg[col].median()
        result[col] = result[col].fillna(global_val)

    return result

# ── Definición de features por segmento ──────────────────────────────────────
FEATURES_BASE = [
    'bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
    'latitude', 'longitude', 'lotAreaValue', 'photoCount',
    'homeType', 'zipcode',
    'taxAssessedValue', 'propertyTaxRate', 'latest_tax_value',
    'latest_tax_paid', 'num_tax_records',
    'last_listing_price', 'num_sales', 'num_price_changes',
    'avg_school_rating', 'max_school_rating', 'num_nearby_schools', 'min_school_distance',
    'has_hoa', 'hoa_fee_monthly', 'has_pool', 'has_garage', 'has_waterfront',
    'tag_price_cut', 'tag_new_construction', 'tag_foreclosure',
    'property_age', 'bath_to_bed_ratio', 'log_living_area', 'log_lot_area', 'zip_3digit',
    'desc_length', 'desc_word_count', 'desc_is_boilerplate',
    'desc_mentions_renovated', 'desc_mentions_pool', 'desc_mentions_view', 'desc_mentions_new',
    # Nuevos features estáticos
    'tax_per_sqft', 'listing_to_tax_ratio', 'tax_to_area', 'hoa_to_area',
    'luxury_score', 'distress_score',
    'school_x_area', 'waterfront_x_lat', 'age_x_school', 'beds_x_baths', 'total_rooms',
]

FEATURES_WITH_ZIP = FEATURES_BASE + ZIP_AGG_FEATURES

# Features específicos a excluir por segmento (poco relevantes o constantes)
EXCLUDE_SF    = ['hoa_fee_monthly', 'hoa_to_area']  # mayoría no tiene HOA
EXCLUDE_CONDO = ['lotAreaValue', 'log_lot_area']      # condos no tienen lote

def get_segment_features(segment):
    features = [f for f in FEATURES_WITH_ZIP if f in train.columns or f in ZIP_AGG_FEATURES]
    if segment == 'SF':
        return [f for f in features if f not in EXCLUDE_SF]
    if segment == 'CONDO':
        return [f for f in features if f not in EXCLUDE_CONDO]
    return features

# ── Parámetros LightGBM ───────────────────────────────────────────────────────
def get_params(segment):
    base = dict(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    # REST tiene menos datos → menos hojas para no sobreajustar
    if segment == 'REST':
        base['num_leaves'] = 63
        base['min_child_samples'] = 30
    return base

# ── Cross-validation ──────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

print('Entrenando modelos por segmento con CV 5-fold...\n')
print(f'  Segmentos: SF={( train.segment=="SF").sum()}  '
      f'CONDO={(train.segment=="CONDO").sum()}  '
      f'REST={(train.segment=="REST").sum()}\n')

segment_fold_mapes = {'SF': [], 'CONDO': [], 'REST': []}

for fold, (tr_idx, val_idx) in enumerate(kf.split(train), 1):
    tr_fold  = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()

    # Agregar features ZIP calculados desde el fold de entrenamiento
    tr_fold  = add_zip_features(tr_fold, tr_fold)
    val_fold = add_zip_features(tr_fold.drop(columns=ZIP_AGG_FEATURES, errors='ignore'),
                                val_fold)
    test_zip = add_zip_features(tr_fold.drop(columns=ZIP_AGG_FEATURES, errors='ignore'),
                                test.copy())

    fold_test_preds = np.zeros(len(test))
    fold_counts     = np.zeros(len(test))  # para promediar si un test cae en varios segmentos

    for seg in ['SF', 'CONDO', 'REST']:
        seg_features = get_segment_features(seg)
        # Filtrar solo features que existen en el df
        seg_features = [f for f in seg_features if f in tr_fold.columns]

        tr_seg  = tr_fold[tr_fold[SEGMENT_COL] == seg]
        val_seg = val_fold[val_fold[SEGMENT_COL] == seg]
        te_seg  = test_zip[test_zip[SEGMENT_COL] == seg]

        if len(tr_seg) == 0:
            continue

        model = lgb.LGBMRegressor(**get_params(seg))
        model.fit(
            tr_seg[seg_features], tr_seg[TARGET],
            eval_set=[(val_seg[seg_features], val_seg[TARGET])],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
            categorical_feature=[c for c in CAT_FEATURES if c in seg_features],
        )

        val_pred = model.predict(val_seg[seg_features])
        oof_preds[val_seg.index] = val_pred

        val_price  = np.expm1(val_seg[TARGET].values)
        pred_price = np.expm1(val_pred)
        seg_mape   = np.mean(np.abs((val_price - pred_price) / val_price)) * 100
        segment_fold_mapes[seg].append(seg_mape)

        # Predicciones test
        te_idx_in_test = te_seg.index
        fold_test_preds[te_idx_in_test] = model.predict(te_seg[seg_features])
        fold_counts[te_idx_in_test] = 1

    test_preds += fold_test_preds / kf.n_splits

    overall_mape = np.mean([np.mean(v) for v in segment_fold_mapes.values() if v])
    print(f'  Fold {fold}:  '
          f'SF={segment_fold_mapes["SF"][-1]:.2f}%  '
          f'CONDO={segment_fold_mapes["CONDO"][-1]:.2f}%  '
          f'REST={segment_fold_mapes["REST"][-1]:.2f}%')

# ── Métricas OOF ─────────────────────────────────────────────────────────────
oof_price      = np.expm1(train[TARGET])
oof_pred_price = np.expm1(oof_preds)
oof_mape = np.mean(np.abs((oof_price - oof_pred_price) / oof_price)) * 100
oof_mae  = mean_absolute_error(oof_price, oof_pred_price)

print(f'\n{"─"*45}')
print(f'OOF MAPE global:  {oof_mape:.2f}%')
print(f'OOF MAE global:   ${oof_mae:,.0f}')
print(f'{"─"*45}')

print('\nMAPE promedio por segmento:')
for seg, mapes in segment_fold_mapes.items():
    n = (train[SEGMENT_COL] == seg).sum()
    print(f'  {seg:8s} ({n:,} props): {np.mean(mapes):.2f}%')

# ── Submission (test set) ─────────────────────────────────────────────────────
submission = pd.DataFrame({
    'zpid': test['zpid'],
    'predicted_price': np.expm1(test_preds),
})
output_path = 'submissions/round2_segments.csv'
submission.to_csv(output_path, index=False)
print(f'\nGuardado: {output_path}  ({len(submission)} filas)')
print(f'Precio predicho: min=${submission.predicted_price.min():,.0f}  '
      f'mediana=${submission.predicted_price.median():,.0f}  '
      f'max=${submission.predicted_price.max():,.0f}')

# ── OOF para Practice submission ──────────────────────────────────────────────
# Subir al dashboard en: Practice (Train Set OOF) → label "round2_segments"
oof_submission = pd.DataFrame({
    'zpid': train['zpid'],
    'predicted_price': np.expm1(oof_preds),
})
oof_path = 'submissions/oof_round2_segments.csv'
oof_submission.to_csv(oof_path, index=False)
print(f'OOF guardado:  {oof_path}  ({len(oof_submission)} filas)  ← subir al tab Practice')
