"""
Round 5 — Mas conservador + penalidades duras anti-distressed

Diagnostico Round 2 (competencia real):
  - SAM compra 12.2 props/sim, hit rate 55.3%, ROI 6.50% (#2)
  - Gimli compra 9.7 props/sim, hit rate 70.9%, ROI 19.29% (#1)
  - Las 2 peores compras (ratio 4.4x y 3.75x) destruyen el 38% de todas las perdidas
  - Propiedades con ratio pred/true > 2x = 64.5% de las perdidas

Cambios respecto a Round 4:
  1. Alpha: 0.35 (mismo que R4) — bajar alpha entrega propiedades buenas a competidores
     (simulador: R4 win rate 34.3%, alpha=0.30 → 12.4%, alpha=0.25 → 4.8%)
  2. Cap ZIP: 1.2x → 1.0x (cap mas apretado — 106 props capeadas vs 72 en R4)
  3. Penalidad: tag_foreclosure=1 → pred × 0.60 (solo 11 props en train)
     NOTA: last_listing_price NO sirve como señal de distress — 22% del train tiene
     listing/tax < 0.10 por ser precios historicos, sin relacion con distress real

Objetivo: bajar a 9-10 props/sim con hit rate > 65%, imitando perfil ganador de Gimli.

Run desde participant/:
    python models/round5_conservative.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

ALPHA = 0.35   # Igual que Round 4 — bajar alpha entrega propiedades buenas a competidores

# Penalidades post-processing
# last_listing_price es un precio historico irrelevante (no sirve como señal de distress).
# Solo penalizamos foreclosures explicitos (11 props en train).
PENALTY_FORECLOSURE = 0.60   # tag_foreclosure = 1

LOG_CAP_BUFFER = np.log(1.0)   # sin buffer extra sobre P95 del ZIP

# -- Cargar datos --------------------------------------------------------------
train = pd.read_csv('data/tabular/train_processed.csv')
test  = pd.read_csv('data/tabular/test_processed.csv')

TARGET      = 'log_price'
SEGMENT_COL = 'segment'

def assign_segment(df):
    df = df.copy()
    df[SEGMENT_COL] = 'REST'
    df.loc[df['homeType'] == 'SINGLE_FAMILY', SEGMENT_COL] = 'SF'
    df.loc[df['homeType'] == 'CONDO',         SEGMENT_COL] = 'CONDO'
    return df

train = assign_segment(train)
test  = assign_segment(test)

# -- Preprocesamiento ----------------------------------------------------------
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
        df['listing_is_missing'] = df['last_listing_price'].isnull().astype(int)
        df['raw_listing_to_tax'] = df['last_listing_price'] / (df['taxAssessedValue'] + 1)
        df['is_listing_distress'] = (
            (~df['last_listing_price'].isnull()) &
            (df['raw_listing_to_tax'] < 0.5)
        ).astype(int)
        df['listing_distress_cap'] = np.where(
            df['is_listing_distress'] == 1,
            np.log1p(df['last_listing_price'] * 1.2),
            np.inf,
        )
        listing_median_by_type = df.groupby('homeType')['last_listing_price'].transform('median')
        df['last_listing_price'] = df['last_listing_price'].fillna(
            listing_median_by_type
        ).fillna(df['last_listing_price'].median())

    df['bath_to_bed_ratio'] = df['bath_to_bed_ratio'].fillna(0)
    for col in ['latitude', 'longitude']:
        df[col] = df[col].fillna(df[col].median())
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')
    return df

train = preprocess(train)
test  = preprocess(test)

for col in CAT_FEATURES:
    all_cats = train[col].cat.categories.union(test[col].cat.categories)
    train[col] = train[col].cat.set_categories(all_cats)
    test[col]  = test[col].cat.set_categories(all_cats)

# -- Feature engineering -------------------------------------------------------
all_data   = pd.concat([train, test])
photo_med_map = all_data.groupby('homeType')['photoCount'].median()

def add_static_features(df):
    df = df.copy()
    df['tax_per_sqft']         = df['taxAssessedValue'] / (df['livingArea'] + 1)
    df['listing_to_tax_ratio'] = df['last_listing_price'] / (df['taxAssessedValue'] + 1)
    df['tax_to_area']          = df['latest_tax_value'] / (df['livingArea'] + 1)
    df['hoa_to_area']          = df['hoa_fee_monthly'] / (df['livingArea'] + 1)

    df['luxury_score']   = (df['has_pool'] + df['has_waterfront'] +
                            df['has_garage'] + (df['hoa_fee_monthly'] > 200).astype(int))
    df['distress_score'] = (df['tag_foreclosure'] + df['tag_price_cut'] +
                            (df['num_price_changes'] > 2).astype(int))

    df['school_x_area']    = df['avg_school_rating'] * df['log_living_area']
    df['waterfront_x_lat'] = df['has_waterfront'] * df['latitude']
    df['age_x_school']     = df['property_age'] * df['avg_school_rating']
    df['beds_x_baths']     = df['bedrooms'] * df['bathrooms']
    df['total_rooms']      = df['bedrooms'] + df['bathrooms']
    df['photo_signal']     = df['photoCount'] / df['homeType'].map(photo_med_map).fillna(1)
    df['area_per_room']    = df['livingArea'] / (df['bedrooms'] + df['bathrooms'] + 1)
    if 'desc_mentions_renovated' in df.columns:
        df['renovation_x_age'] = df['desc_mentions_renovated'] * df['property_age']
    else:
        df['renovation_x_age'] = 0
    return df

train = add_static_features(train)
test  = add_static_features(test)

# -- Features ZIP (dentro del fold) -------------------------------------------
ZIP_AGG_FEATURES = [
    'zip_median_log_price', 'zip_price_per_sqft_median',
    'zip_std_log_price', 'zip_count', 'zip_median_area',
    'zip_median_taxAssessed', 'zip_median_tax_per_sqft',
    'zip_median_listing', 'zip_p95_log_price',
]

def add_zip_features(train_fold, apply_df):
    train_fold = train_fold.copy()
    train_fold['_price_per_sqft'] = train_fold['taxAssessedValue'] / (train_fold['livingArea'] + 1)
    agg = train_fold.groupby('zipcode').agg(
        zip_median_log_price      = ('log_price', 'median'),
        zip_std_log_price         = ('log_price', 'std'),
        zip_count                 = ('zpid', 'count'),
        zip_median_area           = ('livingArea', 'median'),
        zip_price_per_sqft_median = ('_price_per_sqft', 'median'),
        zip_median_taxAssessed    = ('taxAssessedValue', 'median'),
        zip_median_tax_per_sqft   = ('tax_per_sqft', 'median'),
        zip_median_listing        = ('last_listing_price', 'median'),
        zip_p95_log_price         = ('log_price', lambda x: np.percentile(x, 95)),
    ).reset_index()
    result = apply_df.merge(agg, on='zipcode', how='left')
    for col in ZIP_AGG_FEATURES:
        result[col] = result[col].fillna(agg[col].median())
    return result

def add_zip_relative_features(df):
    df = df.copy()
    df['tax_vs_zip_ratio'] = df['taxAssessedValue'] / (df['zip_median_taxAssessed'] + 1)
    df['log_tax_vs_zip']   = np.log1p(df['tax_vs_zip_ratio'])
    df['is_tax_outlier']   = (df['tax_vs_zip_ratio'] > 2.0).astype(int)
    df['tax_sqft_vs_zip']  = df['tax_per_sqft'] / (df['zip_median_tax_per_sqft'] + 1)
    df['listing_vs_zip']   = df['last_listing_price'] / (df['zip_median_listing'] + 1)
    return df

ZIP_RELATIVE_FEATURES = [
    'tax_vs_zip_ratio', 'log_tax_vs_zip', 'is_tax_outlier',
    'tax_sqft_vs_zip', 'listing_vs_zip',
]

FEATURES_BASE = [
    'bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
    'latitude', 'longitude', 'lotAreaValue', 'photoCount',
    'homeType', 'zipcode',
    'taxAssessedValue', 'propertyTaxRate', 'latest_tax_value',
    'latest_tax_paid', 'num_tax_records',
    'last_listing_price', 'listing_is_missing',
    'raw_listing_to_tax', 'is_listing_distress',
    'num_sales', 'num_price_changes',
    'avg_school_rating', 'max_school_rating', 'num_nearby_schools', 'min_school_distance',
    'has_hoa', 'hoa_fee_monthly', 'has_pool', 'has_garage', 'has_waterfront',
    'tag_price_cut', 'tag_new_construction', 'tag_foreclosure',
    'property_age', 'bath_to_bed_ratio', 'log_living_area', 'log_lot_area', 'zip_3digit',
    'desc_length', 'desc_word_count', 'desc_is_boilerplate',
    'desc_mentions_renovated', 'desc_mentions_pool', 'desc_mentions_view', 'desc_mentions_new',
    'tax_per_sqft', 'listing_to_tax_ratio', 'tax_to_area', 'hoa_to_area',
    'luxury_score', 'distress_score',
    'school_x_area', 'waterfront_x_lat', 'age_x_school', 'beds_x_baths', 'total_rooms',
    'photo_signal', 'area_per_room', 'renovation_x_age',
]

FEATURES_WITH_ZIP = FEATURES_BASE + ZIP_AGG_FEATURES + ZIP_RELATIVE_FEATURES
EXCLUDE_SF    = ['hoa_fee_monthly', 'hoa_to_area']
EXCLUDE_CONDO = ['lotAreaValue', 'log_lot_area']

def get_segment_features(segment):
    features = [f for f in FEATURES_WITH_ZIP
                if f in train.columns or f in ZIP_AGG_FEATURES or f in ZIP_RELATIVE_FEATURES]
    if segment == 'SF':
        return [f for f in features if f not in EXCLUDE_SF]
    if segment == 'CONDO':
        return [f for f in features if f not in EXCLUDE_CONDO]
    return features

# -- Parametros LightGBM -------------------------------------------------------
def get_params(segment):
    base = dict(
        objective='quantile',
        alpha=ALPHA,
        metric='quantile',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        verbosity=-1,
    )
    if segment == 'REST':
        base['num_leaves'] = 63
        base['min_child_samples'] = 30
    return base

# -- Cross-validation 5-fold ---------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(train))
oof_caps   = np.zeros(len(train))
test_preds = np.zeros(len(test))
test_caps  = np.zeros(len(test))

print(f'Entrenando Round 5 — Quantile alpha={ALPHA} + penalidades distress\n')
print(f'  SF={(train.segment=="SF").sum()}  '
      f'CONDO={(train.segment=="CONDO").sum()}  '
      f'REST={(train.segment=="REST").sum()}\n')

segment_fold_mapes = {'SF': [], 'CONDO': [], 'REST': []}

for fold, (tr_idx, val_idx) in enumerate(kf.split(train), 1):
    tr_fold  = train.iloc[tr_idx].copy()
    val_fold = train.iloc[val_idx].copy()

    tr_fold['_orig_idx']  = tr_fold.index.values
    val_fold['_orig_idx'] = val_fold.index.values

    tr_fold  = add_zip_features(tr_fold, tr_fold)
    val_fold = add_zip_features(tr_fold.drop(columns=ZIP_AGG_FEATURES, errors='ignore'), val_fold)
    test_zip = add_zip_features(tr_fold.drop(columns=ZIP_AGG_FEATURES, errors='ignore'), test.copy())

    oof_caps[val_fold['_orig_idx'].values.astype(int)] = val_fold['zip_p95_log_price'].values

    tr_fold  = add_zip_relative_features(tr_fold)
    val_fold = add_zip_relative_features(val_fold)
    test_zip = add_zip_relative_features(test_zip)

    fold_test_preds = np.zeros(len(test))
    fold_test_caps  = np.zeros(len(test))

    for seg in ['SF', 'CONDO', 'REST']:
        seg_features = get_segment_features(seg)
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
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
            categorical_feature=[c for c in CAT_FEATURES if c in seg_features],
        )

        val_pred = model.predict(val_seg[seg_features])
        oof_preds[val_seg['_orig_idx'].values.astype(int)] = val_pred

        val_price  = np.expm1(val_seg[TARGET].values)
        pred_price = np.expm1(val_pred)
        seg_mape   = np.mean(np.abs((val_price - pred_price) / val_price)) * 100
        segment_fold_mapes[seg].append(seg_mape)

        fold_test_preds[te_seg.index] = model.predict(te_seg[seg_features])
        fold_test_caps[te_seg.index]  = te_seg['zip_p95_log_price'].values

    test_preds += fold_test_preds / kf.n_splits
    test_caps  += fold_test_caps  / kf.n_splits

    print(f'  Fold {fold}:  '
          f'SF={segment_fold_mapes["SF"][-1]:.2f}%  '
          f'CONDO={segment_fold_mapes["CONDO"][-1]:.2f}%  '
          f'REST={segment_fold_mapes["REST"][-1]:.2f}%')

# -- Post-processing: cap ZIP --------------------------------------------------
oof_preds_raw  = oof_preds.copy()
test_preds_raw = test_preds.copy()

oof_preds  = np.minimum(oof_preds,  oof_caps  + LOG_CAP_BUFFER)
test_preds = np.minimum(test_preds, test_caps + LOG_CAP_BUFFER)

n_oof_capped  = (oof_preds  < oof_preds_raw).sum()
n_test_capped = (test_preds < test_preds_raw).sum()
print(f'\nCap ZIP (x1.0 P95): {n_oof_capped} OOF | {n_test_capped} test')

# -- Post-processing: penalidades duras anti-distressed -----------------------
# Convertir a escala $ para aplicar multiplicadores
oof_prices_pp  = np.expm1(oof_preds)
test_prices_pp = np.expm1(test_preds)

def apply_distress_penalties(prices, df, label=''):
    prices = prices.copy()

    # Solo penalizamos foreclosures explicitamente marcados
    mask_fc = df['tag_foreclosure'].values == 1
    prices[mask_fc] *= PENALTY_FORECLOSURE

    print(f'{label} penalizadas: foreclosure={mask_fc.sum()} | '
          f'reduccion={PENALTY_FORECLOSURE:.0%}')
    return prices

oof_prices_pp  = apply_distress_penalties(oof_prices_pp,  train, 'OOF ')
test_prices_pp = apply_distress_penalties(test_prices_pp, test,  'Test')

# -- Metricas OOF (sobre predicciones SIN penalidades para medir precision) ---
oof_price     = np.expm1(train[TARGET])
oof_pred_pre  = np.expm1(oof_preds)   # antes de penalidades
oof_pred_post = oof_prices_pp          # despues de penalidades

oof_wmape_pre  = np.sum(np.abs(oof_price - oof_pred_pre))  / np.sum(oof_price) * 100
oof_wmape_post = np.sum(np.abs(oof_price - oof_pred_post)) / np.sum(oof_price) * 100
oof_bias_pre   = np.mean((oof_pred_pre  - oof_price) / oof_price) * 100
oof_bias_post  = np.mean((oof_pred_post - oof_price) / oof_price) * 100

print(f'\n{"-"*55}')
print(f'  wMAPE antes de penalidades: {oof_wmape_pre:.2f}%   sesgo: {oof_bias_pre:+.2f}%')
print(f'  wMAPE con penalidades:      {oof_wmape_post:.2f}%   sesgo: {oof_bias_post:+.2f}%')
print(f'  (Round 4 referencia:          21.81%   sesgo: +2.43%)')
print(f'{"-"*55}')

print('\nMAPE por segmento:')
for seg, mapes in segment_fold_mapes.items():
    n = (train[SEGMENT_COL] == seg).sum()
    print(f'  {seg:8s} ({n:,} props): {np.mean(mapes):.2f}%')

# Efecto sobre foreclosures del train
fc_mask = train['tag_foreclosure'].values == 1
if fc_mask.sum() > 0:
    true_fc     = oof_price[fc_mask]
    pred_pre_fc = oof_pred_pre[fc_mask]
    pred_post   = oof_pred_post[fc_mask]
    bias_pre    = np.mean((pred_pre_fc - true_fc) / true_fc) * 100
    bias_post2  = np.mean((pred_post   - true_fc) / true_fc) * 100
    print(f'\nEfecto sobre foreclosures ({fc_mask.sum()} props):')
    print(f'  Sesgo pre-penalidad:   {bias_pre:+.1f}%')
    print(f'  Sesgo post-penalidad:  {bias_post2:+.1f}%')

# -- Submissions ---------------------------------------------------------------
submission = pd.DataFrame({
    'zpid':            test['zpid'],
    'predicted_price': test_prices_pp,
})
output_path = 'submissions/round5_conservative.csv'
submission.to_csv(output_path, index=False)
print(f'\nGuardado: {output_path}  ({len(submission)} filas)')
print(f'  min=${submission.predicted_price.min():,.0f}  '
      f'mediana=${submission.predicted_price.median():,.0f}  '
      f'max=${submission.predicted_price.max():,.0f}')

oof_submission = pd.DataFrame({
    'zpid':            train['zpid'],
    'predicted_price': oof_prices_pp,
})
oof_path = 'submissions/oof_round5_conservative.csv'
oof_submission.to_csv(oof_path, index=False)
print(f'OOF guardado: {oof_path}  ({len(oof_submission)} filas)  <- subir al tab Practice')
print('\nRecordatorio: subir OOF al Practice ANTES de usar una ronda real.')
