"""
Round 3 — Solo imágenes (sin datos tabulares).

Usa embeddings CLIP pre-computados (data/embeddings_clip512.npz) + PCA → LightGBM.
Objetivo didáctico: cuantificar cuánto predicen las imágenes solas, sin datos de precio,
impuestos ni ubicación.

Requiere haber corrido primero:
    python models/extract_embeddings.py

Run desde participant/:
    python models/round3_images_only.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

N_PCA = 64

# ── Cargar datos tabulares (solo para zpid, target y homeType) ────────────────
train_tab = pd.read_csv('data/tabular/train_processed.csv')
test_tab  = pd.read_csv('data/tabular/test_processed.csv')

TARGET = 'log_price'

# ── Cargar embeddings ─────────────────────────────────────────────────────────
print("Cargando embeddings CLIP...")
data_emb   = np.load('data/embeddings_clip512.npz')
emb_zpids  = data_emb['zpids'].astype(int)
emb_matrix = data_emb['embeddings']            # (N_emb, 512)

zpid_to_row = {z: i for i, z in enumerate(emb_zpids)}

distress_df     = pd.read_csv('data/distress_scores.csv')
zpid_to_distress = distress_df.set_index('zpid')['clip_distress_score'].to_dict()

print(f"Embeddings cargados: {emb_matrix.shape[0]:,} propiedades, {emb_matrix.shape[1]} dims")

# ── Construir matrices de embeddings ──────────────────────────────────────────
def get_embedding_matrix(df):
    out = np.zeros((len(df), emb_matrix.shape[1]), dtype=np.float32)
    for i, zpid in enumerate(df['zpid'].values.astype(int)):
        row = zpid_to_row.get(zpid)
        if row is not None:
            out[i] = emb_matrix[row]
    return out

train_emb = get_embedding_matrix(train_tab)
test_emb  = get_embedding_matrix(test_tab)

missing_train = sum(1 for z in train_tab['zpid'].values.astype(int) if z not in zpid_to_row)
missing_test  = sum(1 for z in test_tab['zpid'].values.astype(int)  if z not in zpid_to_row)
print(f"Sin embedding: {missing_train} train | {missing_test} test")

# ── PCA: fit en train, transform ambos ────────────────────────────────────────
print(f"\nAplicando PCA: 512 -> {N_PCA} componentes...")
pca = PCA(n_components=N_PCA, random_state=42)
train_pca = pca.fit_transform(train_emb)
test_pca  = pca.transform(test_emb)
explained = pca.explained_variance_ratio_.sum()
print(f"Varianza explicada: {explained:.1%}")

# ── Construir DataFrames de features ─────────────────────────────────────────
IMG_PC_COLS = [f'img_pc_{i:02d}' for i in range(N_PCA)]

train = pd.DataFrame(train_pca, columns=IMG_PC_COLS)
test  = pd.DataFrame(test_pca,  columns=IMG_PC_COLS)

train['clip_distress_score'] = train_tab['zpid'].map(zpid_to_distress).fillna(0.5).values
test['clip_distress_score']  = test_tab['zpid'].map(zpid_to_distress).fillna(0.5).values

train[TARGET] = train_tab[TARGET].values

FEATURES = IMG_PC_COLS + ['clip_distress_score']

# ── Parámetros LightGBM ───────────────────────────────────────────────────────
PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
)

# ── Cross-validation 5-fold ───────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))

print(f"\nEntrenando modelo solo-imágenes con CV 5-fold ({N_PCA} PC + distress)...\n")

fold_mapes = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(train), 1):
    tr_fold  = train.iloc[tr_idx]
    val_fold = train.iloc[val_idx]

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        tr_fold[FEATURES], tr_fold[TARGET],
        eval_set=[(val_fold[FEATURES], val_fold[TARGET])],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(val_fold[FEATURES])
    oof_preds[val_idx] = val_pred

    val_price  = np.expm1(val_fold[TARGET].values)
    pred_price = np.expm1(val_pred)
    mape = np.mean(np.abs((val_price - pred_price) / val_price)) * 100
    fold_mapes.append(mape)

    test_preds += model.predict(test[FEATURES]) / kf.n_splits
    print(f"  Fold {fold}: MAPE={mape:.2f}%")

# ── Métricas OOF ─────────────────────────────────────────────────────────────
oof_price      = np.expm1(train[TARGET])
oof_pred_price = np.expm1(oof_preds)
oof_mape = np.mean(np.abs((oof_price - oof_pred_price) / oof_price)) * 100
oof_mae  = mean_absolute_error(oof_price, oof_pred_price)
oof_wmape = np.sum(np.abs(oof_price - oof_pred_price)) / np.sum(oof_price) * 100

print(f'\n{"-"*50}')
print(f'OOF MAPE:  {oof_mape:.2f}%')
print(f'OOF wMAPE: {oof_wmape:.2f}%')
print(f'OOF MAE:   ${oof_mae:,.0f}')
print(f'(Referencia tabular Round 5: ~21-22% wMAPE)')
print(f'{"-"*50}')

# ── Submissions ───────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    'zpid': test_tab['zpid'],
    'predicted_price': np.expm1(test_preds),
})
output_path = 'submissions/round3_images_only.csv'
submission.to_csv(output_path, index=False)
print(f'\nGuardado: {output_path}  ({len(submission):,} filas)')
print(f'  min=${submission.predicted_price.min():,.0f}  '
      f'mediana=${submission.predicted_price.median():,.0f}  '
      f'max=${submission.predicted_price.max():,.0f}')

oof_submission = pd.DataFrame({
    'zpid': train_tab['zpid'],
    'predicted_price': np.expm1(oof_preds),
})
oof_path = 'submissions/oof_round3_images_only.csv'
oof_submission.to_csv(oof_path, index=False)
print(f'OOF guardado: {oof_path}  ({len(oof_submission):,} filas)  <- subir al tab Practice')
