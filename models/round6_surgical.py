"""
Round 6 — Prediccion central (a=0.50) + descuento quirurgico por distress visual

Cambios respecto a Round 3 (mejor modelo anterior, 49.7% ROI en Practice):

  1. ALPHA 0.35 -> 0.50: prediccion central en vez del P35 conservador.
     Con a=0.35 eramos el ofertante mas bajo en 27/62 subastas (rank 4.35/5.7).
     La subasta Vickrey de segundo precio no penaliza ofertar mas alto en
     propiedades confiables: ganas la subasta y pagas el precio del segundo.

  2. Descuento quirurgico por distress visual:
     - Solo aplicamos descuento donde clip_distress_score > P90 del train
     - Factor configurable SURGICAL_DISCOUNT (default 0.82 = -18%)
     - Afecta ~10% de propiedades (las visualmente deterioradas)
     - El 90% restante va sin descuento (prediccion central limpia)

  3. Sin penalidad de foreclosure: solo 11 casos en 11,840 props de train.
     El descuento quirurgico por distress visual reemplaza esa logica.

Todo lo demas identico a round3_images.py:
  - 32 PCA CLIP + clip_distress_score (embeddings de foto principal)
  - Segmentacion SF/CONDO/REST
  - ZIP features dentro del fold (sin leakage)
  - Cap ZIP P95

Requiere haber corrido primero:
    python models/extract_embeddings.py

Run desde participant/:
    python models/round6_surgical.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

ALPHA             = 0.50   # prediccion central (era 0.35)
SURGICAL_DISCOUNT = 0.82   # factor para el top-10% de distress visual (-18%)
N_PCA             = 32
LOG_CAP_BUFFER    = np.log(1.0)   # cap ZIP P95 sin margen adicional

# -- Cargar datos ----------------------------------------------------------------
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

# -- Cargar embeddings CLIP ------------------------------------------------------
print("Cargando embeddings CLIP...")
data_emb   = np.load('data/embeddings_clip512.npz')
emb_zpids  = data_emb['zpids'].astype(int)
emb_matrix = data_emb['embeddings']    # (N_emb, 512)

zpid_to_row = {z: i for i, z in enumerate(emb_zpids)}

distress_df      = pd.read_csv('data/distress_scores.csv')
zpid_to_distress = distress_df.set_index('zpid')['clip_distress_score'].to_dict()

print(f"  {emb_matrix.shape[0]:,} embeddings cargados")

def get_embedding_matrix(df):
    out = np.zeros((len(df), emb_matrix.shape[1]), dtype=np.float32)
    for i, zpid in enumerate(df['zpid'].values.astype(int)):
        row = zpid_to_row.get(zpid)
        if row is not None:
            out[i] = emb_matrix[row]
    return out

# PCA: fit solo en train, transform ambos (no target-dependent)
print(f"Aplicando PCA 512 -> {N_PCA}...")
train_emb = get_embedding_matrix(train)
test_emb  = get_embedding_matrix(test)

pca = PCA(n_components=N_PCA, random_state=42)
train_pca = pca.fit_transform(train_emb)
test_pca  = pca.transform(test_emb)
print(f"  Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}")

IMG_PC_COLS = [f'img_pc_{i:02d}' for i in range(N_PCA)]

for i, col in enumerate(IMG_PC_COLS):
    train[col] = train_pca[:, i]
    test[col]  = test_pca[:, i]

train['clip_distress_score'] = train['zpid'].map(zpid_to_distress).fillna(0.5)
test['clip_distress_score']  = test['zpid'].map(zpid_to_distress).fillna(0.5)

IMAGE_FEATURES = IMG_PC_COLS + ['clip_distress_score']

# -- Preprocesamiento ------------------------------------------------------------
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

# -- Feature engineering ---------------------------------------------------------
all_data      = pd.concat([train, test])
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

# -- Features ZIP (dentro del fold) ----------------------------------------------
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

# -- Features --------------------------------------------------------------------
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
] + IMAGE_FEATURES

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

# -- Parametros LightGBM ---------------------------------------------------------
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

# -- Cross-validation 5-fold -----------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(train))
oof_caps   = np.zeros(len(train))
test_preds = np.zeros(len(test))
test_caps  = np.zeros(len(test))

print(f'\nEntrenando Round 6 (tabular + imagenes) — Quantile alpha={ALPHA}')
print(f'Descuento quirurgico: {SURGICAL_DISCOUNT:.0%} sobre clip_distress > P90')
print(f'Features totales: {len(FEATURES_WITH_ZIP)}')
print(f'SF={(train.segment=="SF").sum()}  '
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

        if fold == 1 and seg == 'SF':
            fi = pd.Series(model.feature_importances_, index=seg_features)
            img_fi = fi[[c for c in IMAGE_FEATURES if c in seg_features]].sort_values(ascending=False)
            print(f"  [Fold 1 SF] Importancia features imagen (top 5):")
            for fname, fval in img_fi.head(5).items():
                print(f"    {fname}: {fval}")

    test_preds += fold_test_preds / kf.n_splits
    test_caps  += fold_test_caps  / kf.n_splits

    print(f'  Fold {fold}:  '
          f'SF={segment_fold_mapes["SF"][-1]:.2f}%  '
          f'CONDO={segment_fold_mapes["CONDO"][-1]:.2f}%  '
          f'REST={segment_fold_mapes["REST"][-1]:.2f}%')

# -- Post-processing: cap ZIP ----------------------------------------------------
oof_preds  = np.minimum(oof_preds,  oof_caps  + LOG_CAP_BUFFER)
test_preds = np.minimum(test_preds, test_caps + LOG_CAP_BUFFER)

n_oof_capped  = (oof_preds  < np.minimum(oof_preds,  oof_caps  + LOG_CAP_BUFFER + 1)).sum()
print(f'\nCap ZIP (P95): aplicado')

# Convertir a espacio de precios
oof_prices_pp  = np.expm1(oof_preds)
test_prices_pp = np.expm1(test_preds)

# -- Post-processing: descuento quirurgico ---------------------------------------
# Solo el top-10% mas distressed visualmente recibe descuento
distress_threshold = np.percentile(train['clip_distress_score'].values, 90)

oof_distress  = train['clip_distress_score'].values
test_distress = test['clip_distress_score'].values

risky_oof  = oof_distress  > distress_threshold
risky_test = test_distress > distress_threshold

oof_prices_pp[risky_oof]   *= SURGICAL_DISCOUNT
test_prices_pp[risky_test] *= SURGICAL_DISCOUNT

print(f'\nDescuento quirurgico (clip_distress > {distress_threshold:.3f}):')
print(f'  OOF:  {risky_oof.sum():4d} props ({risky_oof.mean()*100:.1f}%) '
      f'-> precio reducido {(1-SURGICAL_DISCOUNT)*100:.0f}%')
print(f'  Test: {risky_test.sum():4d} props ({risky_test.mean()*100:.1f}%) '
      f'-> precio reducido {(1-SURGICAL_DISCOUNT)*100:.0f}%')

# -- Metricas OOF ----------------------------------------------------------------
oof_price     = np.expm1(train[TARGET])
oof_wmape     = np.sum(np.abs(oof_price - oof_prices_pp)) / np.sum(oof_price) * 100
oof_bias      = np.mean((oof_prices_pp - oof_price) / oof_price) * 100

print(f'\n{"-"*55}')
print(f'  wMAPE OOF: {oof_wmape:.2f}%   sesgo: {oof_bias:+.2f}%')
print(f'  (Round 3 referencia: 22.39%   sesgo: ~+2%)')
print(f'  (Round 5 tabular:    21.91%   sesgo: ~+2%)')
print(f'{"-"*55}')

print('\nMAPE por segmento:')
for seg, mapes in segment_fold_mapes.items():
    n = (train[SEGMENT_COL] == seg).sum()
    print(f'  {seg:8s} ({n:,} props): {np.mean(mapes):.2f}%')

# -- Submissions -----------------------------------------------------------------
submission = pd.DataFrame({
    'zpid':            test['zpid'],
    'predicted_price': test_prices_pp,
})
output_path = 'submissions/round6_surgical.csv'
submission.to_csv(output_path, index=False)
print(f'\nGuardado: {output_path}  ({len(submission):,} filas)')
print(f'  min=${submission.predicted_price.min():,.0f}  '
      f'mediana=${submission.predicted_price.median():,.0f}  '
      f'max=${submission.predicted_price.max():,.0f}')

oof_submission = pd.DataFrame({
    'zpid':            train['zpid'],
    'predicted_price': oof_prices_pp,
})
oof_path = 'submissions/oof_round6_surgical.csv'
oof_submission.to_csv(oof_path, index=False)
print(f'OOF guardado: {oof_path}  ({len(oof_submission):,} filas)  <- subir al Practice')

# -- Distribucion del descuento quirurgico ---------------------------------------
print('\nDistribucion clip_distress_score (train):')
for p in [50, 75, 90, 95, 99]:
    val = np.percentile(train['clip_distress_score'].values, p)
    print(f'  P{p:2d}: {val:.4f}')
print(f'  Umbral usado (P90): {distress_threshold:.4f}')
print(f'  Props con descuento: {risky_oof.sum()} de {len(train):,} train')
print(f'  Comparacion mediana precio:')
print(f'    Con descuento:    ${np.median(oof_prices_pp[risky_oof]):,.0f}')
print(f'    Sin descuento:    ${np.median(oof_prices_pp[~risky_oof]):,.0f}')
print(f'    True (distress):  ${np.median(oof_price.values[risky_oof]):,.0f}')
print(f'    True (normal):    ${np.median(oof_price.values[~risky_oof]):,.0f}')

print('\nRecordatorio: subir OOF al Practice ANTES de usar una ronda real.')
print(f'Protocolo: oof_round6_surgical.csv -> Practice tab -> comparar vs 49.7% de Round 3')
