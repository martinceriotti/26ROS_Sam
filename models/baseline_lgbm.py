"""
Baseline LightGBM con cross-validation y más features que el script de ejemplo.

Mejoras sobre 01_lgbm_basic.py:
- Más features (tax, schools, booleans, derived)
- Imputación de missing values
- Cross-validation 5-fold para evaluar MAPE real (sin data leakage)
- Guarda el modelo entrenado en models/

Run desde participant/:
    python models/baseline_lgbm.py
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

# ── Features ──────────────────────────────────────────────────────────────────
# Agrupados por categoría para facilitar experimentos

FEATURES_CORE = [
    'bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
    'latitude', 'longitude', 'lotAreaValue', 'photoCount',
    'homeType', 'zipcode',
]

FEATURES_TAX = [
    'taxAssessedValue', 'propertyTaxRate',
    'latest_tax_value', 'latest_tax_paid', 'num_tax_records',
]

FEATURES_FINANCIAL = [
    'last_listing_price', 'num_sales', 'num_price_changes',
]

FEATURES_SCHOOLS = [
    'avg_school_rating', 'max_school_rating',
    'num_nearby_schools', 'min_school_distance',
]

FEATURES_ATTRIBUTES = [
    'has_hoa', 'hoa_fee_monthly',
    'has_pool', 'has_garage', 'has_waterfront',
]

FEATURES_TAGS = [
    'tag_price_cut', 'tag_new_construction', 'tag_foreclosure',
]

FEATURES_DERIVED = [
    'property_age', 'bath_to_bed_ratio',
    'log_living_area', 'log_lot_area', 'zip_3digit',
]

FEATURES_TEXT = [
    'desc_length', 'desc_word_count', 'desc_is_boilerplate',
    'desc_mentions_renovated', 'desc_mentions_pool',
    'desc_mentions_view', 'desc_mentions_new',
]

FEATURES = (
    FEATURES_CORE + FEATURES_TAX + FEATURES_FINANCIAL +
    FEATURES_SCHOOLS + FEATURES_ATTRIBUTES + FEATURES_TAGS +
    FEATURES_DERIVED + FEATURES_TEXT
)

CAT_FEATURES = ['homeType', 'zipcode']

# ── Preprocesamiento ──────────────────────────────────────────────────────────

def preprocess(df):
    df = df.copy()

    # Imputar con mediana por homeType (más inteligente que mediana global)
    for col in ['bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
                'lotAreaValue', 'taxAssessedValue', 'latest_tax_value',
                'latest_tax_paid', 'property_age', 'log_living_area', 'log_lot_area']:
        if col in df.columns and df[col].isnull().any():
            medians = df.groupby('homeType')[col].transform('median')
            df[col] = df[col].fillna(medians).fillna(df[col].median())

    # last_listing_price: muchos missing → imputar con taxAssessedValue * 1.1
    if 'last_listing_price' in df.columns:
        mask = df['last_listing_price'].isnull()
        df.loc[mask, 'last_listing_price'] = df.loc[mask, 'taxAssessedValue'] * 1.1

    # bath_to_bed_ratio: missing cuando bedrooms=0
    if 'bath_to_bed_ratio' in df.columns:
        df['bath_to_bed_ratio'] = df['bath_to_bed_ratio'].fillna(0)

    # Lat/lon: muy pocos missing → imputar con mediana global
    for col in ['latitude', 'longitude']:
        df[col] = df[col].fillna(df[col].median())

    # Categorías
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')

    return df


train = preprocess(train)
test  = preprocess(test)

# ── Cross-validation ──────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)

print('Entrenando con cross-validation 5-fold...\n')

for fold, (tr_idx, val_idx) in enumerate(kf.split(train), 1):
    X_tr, y_tr = train.iloc[tr_idx][FEATURES], train.iloc[tr_idx][TARGET]
    X_val, y_val = train.iloc[val_idx][FEATURES], train.iloc[val_idx][TARGET]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        categorical_feature=CAT_FEATURES,
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    val_price = np.expm1(y_val)
    pred_price = np.expm1(val_pred)
    mape = np.mean(np.abs((val_price - pred_price) / val_price)) * 100

    test_preds += model.predict(test[FEATURES]) / kf.n_splits
    print(f'  Fold {fold}: MAPE = {mape:.2f}%  |  best iter = {model.best_iteration_}')

# ── Métricas OOF (fuera de muestra, sin leakage) ─────────────────────────────
oof_price = np.expm1(train[TARGET])
oof_pred_price = np.expm1(oof_preds)
oof_mape = np.mean(np.abs((oof_price - oof_pred_price) / oof_price)) * 100
oof_mae  = mean_absolute_error(oof_price, oof_pred_price)

print(f'\nOOF MAPE: {oof_mape:.2f}%')
print(f'OOF MAE:  ${oof_mae:,.0f}')

# ── Importancia de features ───────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES)
print('\nTop 15 features más importantes:')
print(importances.sort_values(ascending=False).head(15).to_string())

# ── Submission (test set) ─────────────────────────────────────────────────────
submission = pd.DataFrame({
    'zpid': test['zpid'],
    'predicted_price': np.expm1(test_preds),
})
output_path = 'submissions/baseline_lgbm.csv'
submission.to_csv(output_path, index=False)
print(f'\nGuardado: {output_path}  ({len(submission)} filas)')
print(f'Precio predicho: min=${submission.predicted_price.min():,.0f}  '
      f'mediana=${submission.predicted_price.median():,.0f}  '
      f'max=${submission.predicted_price.max():,.0f}')

# ── OOF para Practice submission ──────────────────────────────────────────────
# Subir al dashboard en: Practice (Train Set OOF) → label "baseline_lgbm"
oof_submission = pd.DataFrame({
    'zpid': train['zpid'],
    'predicted_price': np.expm1(oof_preds),
})
oof_path = 'submissions/oof_baseline_lgbm.csv'
oof_submission.to_csv(oof_path, index=False)
print(f'OOF guardado:  {oof_path}  ({len(oof_submission)} filas)  ← subir al tab Practice')
