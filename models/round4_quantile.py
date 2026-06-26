"""
Round 4 — Modelo creativo: Quantile Regression (P35)

Insight del análisis de los exports del dashboard:
  - 2 propiedades causaron el 52% de todas las pérdidas de SAM
  - Ambas tenían ratio pred/true de 5.7x y 8.8x (distressed sales ocultas)
  - La simulación usa nuestra predicción para DOS cosas:
      a) Decidir si comprar  → pred > asking × 1.08
      b) Calcular nuestro bid → pred × 0.85
  - El error es asimétrico: sobreestimar destruye capital, subestimar solo hace
    que no compremos (costo de oportunidad, no pérdida real)

Estrategia: predecir el percentil 35 del precio (quantile regression α=0.35)
  - La loss cuantil penaliza overestimates más que underestimates
  - Reduce las predicciones extremas en propiedades atípicas
  - wMAPE subirá ~1-2% pero el ROI debería mejorar significativamente
  - Incorpora features no-leaky del EDA: photo_signal, area_per_room, renovation_x_age

Run desde participant/:
    python models/round4_quantile.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

ALPHA = 0.35   # percentil objetivo — conservador para evitar overestimates catastróficos

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

# -- Feature engineering estático ---------------------------------------------
# Calculamos mediana de photoCount por homeType sobre todo el dataset (no es target-based)
photo_median_by_type = pd.concat([train, test]).groupby('homeType')['photoCount'].transform('median')
all_data = pd.concat([train, test])
photo_med_map = all_data.groupby('homeType')['photoCount'].median()

def add_static_features(df):
    df = df.copy()

    # Features de Round 2
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

    # Nuevos features no-leaky del EDA (Round 4)
    # photo_signal: cuántas fotos tiene vs la mediana de su tipo
    df['photo_signal'] = df['photoCount'] / df['homeType'].map(photo_med_map).fillna(1)

    # area_per_room: espacio promedio por cuarto
    df['area_per_room'] = df['livingArea'] / (df['bedrooms'] + df['bathrooms'] + 1)

    # renovation_x_age: captura premium de propiedades viejas renovadas
    if 'desc_mentions_renovated' in df.columns:
        df['renovation_x_age'] = df['desc_mentions_renovated'] * df['property_age']
    else:
        df['renovation_x_age'] = 0

    return df

train = add_static_features(train)
test  = add_static_features(test)

# -- Features ZIP (dentro del fold para evitar leakage) -----------------------
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
    # Nuevos Round 4
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

# -- Parámetros LightGBM — Quantile --------------------------------------------
def get_params(segment):
    base = dict(
        objective='quantile',
        alpha=ALPHA,
        metric='quantile',
        n_estimators=1500,        # más árboles porque quantile loss es más ruidosa
        learning_rate=0.03,       # más lento para mejor convergencia
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,           # regularización un poco mayor
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

print(f'Entrenando Quantile Regression alpha={ALPHA} por segmento - CV 5-fold...\n')
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

# -- Post-processing cap -------------------------------------------------------
# Cap más ajustado que round3 (×1.2 vs ×1.5) porque el modelo ya es conservador
LOG_CAP_BUFFER = np.log(1.2)

oof_preds_raw  = oof_preds.copy()
test_preds_raw = test_preds.copy()

oof_preds  = np.minimum(oof_preds,  oof_caps  + LOG_CAP_BUFFER)
test_preds = np.minimum(test_preds, test_caps + LOG_CAP_BUFFER)

n_oof_capped  = (oof_preds  < oof_preds_raw).sum()
n_test_capped = (test_preds < test_preds_raw).sum()

print(f'\nPost-processing cap ZIP (×1.2): {n_oof_capped} OOF | {n_test_capped} test')

# -- Métricas OOF -------------------------------------------------------------
oof_price      = np.expm1(train[TARGET])
oof_pred_price = np.expm1(oof_preds)

oof_mape  = np.mean(np.abs((oof_price - oof_pred_price) / oof_price)) * 100
oof_wmape = np.sum(np.abs(oof_price - oof_pred_price)) / np.sum(oof_price) * 100
oof_mae   = mean_absolute_error(oof_price, oof_pred_price)
oof_bias  = np.mean((oof_pred_price - oof_price) / oof_price) * 100

print(f'\n{"-"*50}')
print(f'OOF MAPE:   {oof_mape:.2f}%  (Round3 fue ~27%)')
print(f'OOF wMAPE:  {oof_wmape:.2f}%  (Round3 fue ~21.7%)')
print(f'OOF MAE:    ${oof_mae:,.0f}')
print(f'Sesgo medio: {oof_bias:+.2f}%  (Round3 fue +7.3% — buscamos cerca de 0%)')
print(f'{"-"*50}')

print('\nMAPE promedio por segmento:')
for seg, mapes in segment_fold_mapes.items():
    n = (train[SEGMENT_COL] == seg).sum()
    print(f'  {seg:8s} ({n:,} props): {np.mean(mapes):.2f}%')

# Chequeo sobre las propiedades catastróficas del export
bad_zpids_export = [1006940, 1006321, 1008936, 1016851, 1001884]
bad_mask = train['zpid'].isin(bad_zpids_export)
if bad_mask.sum() > 0:
    print(f'\nPropiedades catastróficas del export ({bad_mask.sum()} en train):')
    for zpid, true_p, pred_p in zip(
        train.loc[bad_mask, 'zpid'],
        np.expm1(train.loc[bad_mask, TARGET]),
        oof_pred_price[bad_mask],
    ):
        err = (pred_p - true_p) / true_p * 100
        print(f'  zpid {int(zpid)}: true=${true_p:,.0f}  pred=${pred_p:,.0f}  err={err:+.0f}%')

# Distribución del sesgo por cuantil de precio
print('\nSesgo por rango de precio:')
price_q = pd.qcut(oof_price, q=4, labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])
for label in ['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']:
    mask = price_q == label
    bias = np.mean((oof_pred_price[mask] - oof_price[mask]) / oof_price[mask]) * 100
    print(f'  {label:12s}: sesgo {bias:+.1f}%')

# -- Submissions ---------------------------------------------------------------
submission = pd.DataFrame({
    'zpid':            test['zpid'],
    'predicted_price': np.expm1(test_preds),
})
output_path = 'submissions/round4_quantile.csv'
submission.to_csv(output_path, index=False)
print(f'\nGuardado: {output_path}  ({len(submission)} filas)')
print(f'Precio predicho: min=${submission.predicted_price.min():,.0f}  '
      f'mediana=${submission.predicted_price.median():,.0f}  '
      f'max=${submission.predicted_price.max():,.0f}')

oof_submission = pd.DataFrame({
    'zpid':            train['zpid'],
    'predicted_price': np.expm1(oof_preds),
})
oof_path = 'submissions/oof_round4_quantile.csv'
oof_submission.to_csv(oof_path, index=False)
print(f'OOF guardado:  {oof_path}  ({len(oof_submission)} filas)  <- subir al tab Practice')
