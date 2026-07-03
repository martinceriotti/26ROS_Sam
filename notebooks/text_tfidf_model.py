# %% [markdown]
# # 6. Text Modeling - TF-IDF + LightGBM
#
# **Objetivo**: predecir `log_price` usando *solo* el campo `description`.
#
# Flujo:
# 1. Limpieza y preprocesamiento del texto
# 2. Vectorización TF-IDF (bag-of-words ponderado)
# 3. Entrenamiento LightGBM con cross-validation 5-fold
# 4. Métricas: MAE, RMSE y R2
# 5. Palabras más influyentes en el precio
# 6. Predicciones para el test set -> submission.csv

# %% - Imports
import re
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')

# Forzar UTF-8 en consola Windows (necesario para simbolos como +/-, R2, ->)
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Estilo de plots consistente con el proyecto
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'axes.titlesize': 13, 'axes.labelsize': 11})

# %% - Carga de datos
# ROOT apunta al directorio participant/ independientemente de desde dónde se corra
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # notebooks/../ = participant/

train = pd.read_csv(ROOT / 'data/tabular/train_processed.csv')
test  = pd.read_csv(ROOT / 'data/tabular/test_processed.csv')

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Target log_price - media: {train['log_price'].mean():.3f}  "
      f"std: {train['log_price'].std():.3f}  "
      f"rango: [{train['log_price'].min():.2f}, {train['log_price'].max():.2f}]")
print(f"\nDescripciones boilerplate en train: "
      f"{train['desc_is_boilerplate'].sum():,} ({train['desc_is_boilerplate'].mean():.1%})")

# %% [markdown]
# ## 6.1 Preprocesamiento de texto
#
# Los pasos son:
# - Convertir a minúsculas
# - Eliminar caracteres especiales y números aislados
# - Eliminar espacios extra
#
# **Importante**: el 52% de las descripciones son boilerplate generadas por Zillow
# ("This X sqft single family home has..."). Contienen casi cero información más allá
# de lo que ya está en las columnas tabulares. Las tratamos igual - el modelo aprenderá
# a ignorarlas o a darles peso bajo.

# %% - Limpieza de texto
def clean_text(text: str) -> str:
    """Normalización básica: minúsculas + limpieza de ruido tipográfico."""
    if not isinstance(text, str) or text.strip() == '':
        return 'sin descripcion'
    text = text.lower()
    # Quitar URLs, emails
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', '', text)
    # Quitar números puros (direcciones, zip codes) pero no palabras con números
    text = re.sub(r'\b\d{4,}\b', '', text)
    # Quitar caracteres especiales excepto apóstrofe
    text = re.sub(r"[^a-z\s']", ' ', text)
    # Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train['desc_clean'] = train['description'].apply(clean_text)
test['desc_clean']  = test['description'].apply(clean_text)

# Vista rápida del resultado
print("Ejemplo boilerplate (antes):", train.loc[train['desc_is_boilerplate']==1, 'description'].iloc[0][:100])
print("Ejemplo boilerplate (después):", train.loc[train['desc_is_boilerplate']==1, 'desc_clean'].iloc[0][:100])
print()
print("Ejemplo real (antes):", train.loc[train['desc_is_boilerplate']==0, 'description'].iloc[0][:100])
print("Ejemplo real (después):", train.loc[train['desc_is_boilerplate']==0, 'desc_clean'].iloc[0][:100])

# %% - Distribución de longitud por tipo
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (label, mask) in zip(axes, [
    ('Boilerplate', train['desc_is_boilerplate'] == 1),
    ('Contenido real', train['desc_is_boilerplate'] == 0),
]):
    lengths = train.loc[mask, 'desc_clean'].str.split().str.len()
    ax.hist(lengths, bins=40, color='steelblue' if 'real' in label else 'coral', edgecolor='white', alpha=0.8)
    ax.set_title(f'{label} (n={mask.sum():,})')
    ax.set_xlabel('Palabras en descripción (post-limpieza)')
    ax.set_ylabel('Propiedades')
    ax.axvline(lengths.median(), color='black', ls='--', lw=1, label=f'mediana={lengths.median():.0f}')
    ax.legend()

plt.suptitle('Distribución de longitud de descripción', fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6.2 TF-IDF Vectorización
#
# **TF-IDF** (Term Frequency–Inverse Document Frequency) convierte texto en una
# matriz numérica donde cada columna es una palabra (o bi-grama) y el valor refleja
# cuán *característica* es esa palabra para ese documento relativo al corpus.
#
# Parámetros elegidos:
# - `ngram_range=(1,2)`: palabras sueltas Y bi-gramas ("ocean view", "fixer upper")
# - `max_features=800`: vocabulario limitado - las descripciones son cortas
# - `min_df=5`: una palabra debe aparecer en al menos 5 propiedades para incluirse
# - `sublinear_tf=True`: log(TF+1) suaviza el efecto de palabras muy repetidas
# - `max_df=0.85`: ignorar palabras que aparecen en más del 85% del corpus (demasiado genéricas)

# %% - Crear matriz TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # unigramas + bigramas
    max_features=800,       # top 800 términos más informativos
    min_df=5,               # mínimo 5 documentos
    max_df=0.85,            # ignorar términos casi universales
    sublinear_tf=True,      # log(tf+1) en lugar de tf crudo
    strip_accents='ascii',
    token_pattern=r"(?u)\b[a-z][a-z']+\b",  # solo palabras de >=2 letras
)

# Ajustar SOLO en train (nunca en test - para evitar data leakage)
X_train_tfidf = vectorizer.fit_transform(train['desc_clean'])
X_test_tfidf  = vectorizer.transform(test['desc_clean'])

vocab = vectorizer.get_feature_names_out()
print(f"Vocabulario: {len(vocab):,} términos")
print(f"Matriz train: {X_train_tfidf.shape}  (sparsa, densidad: {X_train_tfidf.nnz / X_train_tfidf.shape[0] / X_train_tfidf.shape[1]:.3%})")
print(f"Matriz test:  {X_test_tfidf.shape}")
print(f"\nMuestra del vocabulario: {list(vocab[:15])}")
print(f"                         {list(vocab[-15:])}")

# %% - Agregar feature flag: ¿descripcion es boilerplate?
# El modelo puede aprender que el texto no aportará info en estos casos
flag_train = csr_matrix(train[['desc_is_boilerplate']].values.astype(float))
flag_test  = csr_matrix(test[['desc_is_boilerplate']].values.astype(float))

X_train = hstack([X_train_tfidf, flag_train])
X_test  = hstack([X_test_tfidf,  flag_test])

feature_names = list(vocab) + ['es_boilerplate']
y_train = train['log_price'].values

print(f"\nMatrix final train: {X_train.shape}")

# %% [markdown]
# ## 6.3 Entrenamiento - LightGBM con CV 5-fold
#
# LightGBM maneja matrices sparse nativamente - ideal para TF-IDF.
# Usamos 5-fold cross-validation (mismo esquema que el resto de modelos del proyecto)
# para obtener predicciones OOF (out-of-fold) sin data leakage.

# %% - Parámetros LightGBM para texto sparse
params = dict(
    objective='regression',
    metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,          # más bajo que tabular: features sparse son más ruidosas
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.3,   # samplear pocas columnas - clave para features sparse
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

# %% - Cross-validation 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds     = np.zeros(len(train))
test_preds    = np.zeros(len(test))
feature_imps  = np.zeros(len(feature_names))
fold_metrics  = []

print("Entrenando TF-IDF + LightGBM - 5-fold CV\n")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    # Acumular predicciones test
    test_preds += model.predict(X_test) / kf.n_splits

    # Acumular feature importance
    feature_imps += model.feature_importances_ / kf.n_splits

    # Métricas del fold (en escala log_price)
    mae  = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2   = r2_score(y_val, val_pred)
    fold_metrics.append({'fold': fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
                         'best_iter': model.best_iteration_})

    print(f"  Fold {fold}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  "
          f"iters={model.best_iteration_}")

# %% [markdown]
# ## 6.4 Métricas OOF globales
#
# Las métricas OOF (out-of-fold) son una estimación *sin bias* del error real del modelo,
# ya que cada predicción fue hecha por un modelo que NO vio ese dato en el entrenamiento.

# %% - Métricas OOF globales
oof_mae  = mean_absolute_error(y_train, oof_preds)
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
oof_r2   = r2_score(y_train, oof_preds)

# Convertir a escala de precio real para interpretabilidad
real_prices    = np.expm1(y_train)
pred_prices    = np.expm1(oof_preds)
oof_wmape      = np.sum(np.abs(real_prices - pred_prices)) / np.sum(real_prices) * 100
oof_mae_dollar = mean_absolute_error(real_prices, pred_prices)

metrics_df = pd.DataFrame(fold_metrics).set_index('fold')
print("=" * 55)
print(f"  Resultados OOF - TF-IDF + LightGBM (texto solo)")
print("=" * 55)
print(f"  MAE  (log_price):  {oof_mae:.4f}  +/-{metrics_df['MAE'].std():.4f}")
print(f"  RMSE (log_price):  {oof_rmse:.4f}  +/-{metrics_df['RMSE'].std():.4f}")
print(f"  R2   (log_price):  {oof_r2:.4f}  +/-{metrics_df['R2'].std():.4f}")
print(f"")
print(f"  wMAPE (precio $):  {oof_wmape:.2f}%")
print(f"  MAE   (precio $):  ${oof_mae_dollar:,.0f}")
print("=" * 55)
print(f"\n  Referencia - modelo tabular completo: wMAPE ~21%")
print(f"  El texto solo captura una fracción del valor - esperado.")

# %% - Tabla por fold
print("\nMétricas por fold:")
print(metrics_df.to_string())

# %% - Plot: predicciones OOF vs valores reales
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter OOF
ax = axes[0]
ax.scatter(y_train, oof_preds, alpha=0.15, s=4, color='steelblue', rasterized=True)
mn, mx = y_train.min(), y_train.max()
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Predicción perfecta')
ax.set_xlabel('log_price real')
ax.set_ylabel('log_price predicho (OOF)')
ax.set_title(f'OOF: Real vs Predicho  (R2={oof_r2:.3f})', fontweight='bold')
ax.legend()

# Distribución de errores
ax = axes[1]
residuals = oof_preds - y_train
ax.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', ls='--', lw=1.5)
ax.axvline(residuals.mean(), color='orange', ls='-', lw=1.5,
           label=f'Sesgo = {residuals.mean():+.4f}')
ax.set_xlabel('Residuo (predicho - real)')
ax.set_ylabel('Propiedades')
ax.set_title('Distribución de errores OOF', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6.5 Palabras más relevantes para el precio
#
# LightGBM registra cuántas veces usó cada feature en sus splits ("split importance").
# Las palabras con mayor importancia son las más *informativas* para discriminar precios.
# Separamos las que elevan el precio (coeficiente de correlación positivo con log_price)
# de las que lo reducen.

# %% - Feature importance: top palabras
n_top = 25

imp_df = pd.DataFrame({
    'term':       feature_names,
    'importance': feature_imps,
}).sort_values('importance', ascending=False)

# Calcular si la palabra se asocia a precio alto o bajo
# (correlación entre presencia de la palabra en el documento y log_price)
X_dense_sample = X_train_tfidf[:, [i for i, f in enumerate(vocab)
                                    if f in imp_df.head(60)['term'].values]].toarray()
top_vocab_idx  = {f: i for i, f in enumerate(vocab)}

corrs = {}
for term in imp_df.head(60)['term'].values:
    if term in top_vocab_idx:
        col = X_train_tfidf[:, top_vocab_idx[term]].toarray().ravel()
        if col.std() > 0:
            corrs[term] = np.corrcoef(col, y_train)[0, 1]
        else:
            corrs[term] = 0

imp_df['corr_log_price'] = imp_df['term'].map(corrs).fillna(0)
imp_df['direction']      = imp_df['corr_log_price'].apply(
    lambda c: '[+] precio alto' if c > 0.01 else ('[-] precio bajo' if c < -0.01 else 'neutro')
)

top_imp = imp_df.head(n_top).copy()

# ── Plot de importancia ──
fig, ax = plt.subplots(figsize=(10, 8))

colors = top_imp['direction'].map({
    '[+] precio alto': '#2ecc71',
    '[-] precio bajo': '#e74c3c',
    'neutro':        '#95a5a6',
})

bars = ax.barh(
    top_imp['term'][::-1],
    top_imp['importance'][::-1],
    color=colors[::-1],
    edgecolor='white',
    height=0.7,
)

ax.set_xlabel('Importancia (splits acumulados, promedio 5 folds)', labelpad=10)
ax.set_title(f'Top {n_top} palabras más influyentes en el precio\n'
             f'TF-IDF + LightGBM (vocabulario: {len(vocab):,} términos)',
             fontweight='bold', pad=14)
ax.tick_params(axis='y', labelsize=10)

# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='[+] Asociada a precio ALTO'),
    Patch(facecolor='#e74c3c', label='[-] Asociada a precio BAJO'),
    Patch(facecolor='#95a5a6', label='[~] Efecto neutro / mixto'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()

# %% - Top palabras positivas y negativas por correlación
print("=" * 50)
print("  TOP 15 PALABRAS --> PRECIO ALTO")
print("=" * 50)
top_pos = imp_df[imp_df['corr_log_price'] > 0].nlargest(15, 'corr_log_price')
for _, r in top_pos.iterrows():
    print(f"  {r['term']:<25}  corr={r['corr_log_price']:+.3f}  imp={r['importance']:.0f}")

print()
print("=" * 50)
print("  TOP 15 PALABRAS --> PRECIO BAJO")
print("=" * 50)
top_neg = imp_df[imp_df['corr_log_price'] < 0].nsmallest(15, 'corr_log_price')
for _, r in top_neg.iterrows():
    print(f"  {r['term']:<25}  corr={r['corr_log_price']:+.3f}  imp={r['importance']:.0f}")

# %% - Heatmap de correlación: top bigramas vs precio
bigrams_only = imp_df[imp_df['term'].str.contains(' ')].head(20)

fig, ax = plt.subplots(figsize=(10, 5))
colors_bg = ['#2ecc71' if c > 0 else '#e74c3c' for c in bigrams_only['corr_log_price']]
bars = ax.barh(
    bigrams_only['term'][::-1],
    bigrams_only['corr_log_price'][::-1],
    color=colors_bg[::-1],
    edgecolor='white',
    height=0.65,
)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Correlación con log_price')
ax.set_title('Top bi-gramas - correlación con precio de venta', fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6.6 Análisis por segmento: ¿el texto aporta más en SF o CONDO?

# %% - R2 por tipo de propiedad
train['oof_pred_text'] = oof_preds
train['residuo_texto'] = oof_preds - train['log_price']

seg_metrics = []
for ht in train['homeType'].value_counts().head(4).index:
    sub = train[train['homeType'] == ht]
    r2  = r2_score(sub['log_price'], sub['oof_pred_text'])
    mae = mean_absolute_error(sub['log_price'], sub['oof_pred_text'])
    seg_metrics.append({'homeType': ht, 'n': len(sub), 'R2': r2, 'MAE (log)': mae})

seg_df = pd.DataFrame(seg_metrics)
print("R2 y MAE por tipo de propiedad (modelo SOLO texto):")
print(seg_df.to_string(index=False))

# El texto aporta más donde las descripciones son más ricas (CONDOs con amenities)

# %% [markdown]
# ## 7. Submission
#
# Generamos las predicciones para el test set.
# El modelo de texto solo **no debe usarse directamente** como submission final -
# su wMAPE es mayor al del modelo tabular completo.
#
# **Uso recomendado**:
# - Como análisis de qué información aporta el texto
# - Como componente de un ensemble: `0.15 x pred_texto + 0.85 x pred_tabular`

# %% - Predicciones en escala de precio (expm1 invierte el log1p del target)
test_prices_text = np.expm1(test_preds)

print(f"Predicciones test (precio $):")
print(f"  min:    ${test_prices_text.min():,.0f}")
print(f"  mediana:${np.median(test_prices_text):,.0f}")
print(f"  media:  ${test_prices_text.mean():,.0f}")
print(f"  max:    ${test_prices_text.max():,.0f}")
print(f"\nReferencia train - mediana real: ${np.median(np.expm1(y_train)):,.0f}")

# %% - Exportar submission (texto solo)
submission_text = pd.DataFrame({
    'zpid':            test['zpid'],
    'predicted_price': test_prices_text,
})
submission_text.to_csv(ROOT / 'submissions/text_only_tfidf.csv', index=False)
print(f"\nGuardado: submissions/text_only_tfidf.csv  ({len(submission_text):,} filas)")

# %% - Exportar OOF (para tab Practice del dashboard)
oof_prices = np.expm1(oof_preds)
submission_oof = pd.DataFrame({
    'zpid':            train['zpid'],
    'predicted_price': oof_prices,
})
submission_oof.to_csv(ROOT / 'submissions/oof_text_only_tfidf.csv', index=False)
print(f"Guardado: submissions/oof_text_only_tfidf.csv  ({len(submission_oof):,} filas)")

# %% - Ensemble simple con modelo tabular existente (si existe)
import os
tabular_path = ROOT / 'submissions/round3_images.csv'

if os.path.exists(tabular_path):
    tabular_preds = pd.read_csv(tabular_path)

    # Merge por zpid para alinear correctamente
    merged = test[['zpid']].merge(tabular_preds, on='zpid', how='left')
    merged = merged.rename(columns={'predicted_price': 'pred_tabular'})
    merged['pred_texto']  = test_prices_text

    # Ensemble: 15% texto, 85% tabular
    # El peso bajo al texto refleja que el modelo tabular es mucho mejor en solitario
    W_TEXT    = 0.15
    W_TABULAR = 1 - W_TEXT

    merged['pred_ensemble'] = W_TABULAR * merged['pred_tabular'] + W_TEXT * merged['pred_texto']

    submission_ensemble = merged[['zpid', 'pred_ensemble']].rename(
        columns={'pred_ensemble': 'predicted_price'}
    )
    submission_ensemble.to_csv(ROOT / 'submissions/ensemble_tabular_text.csv', index=False)
    print(f"Ensemble guardado: submissions/ensemble_tabular_text.csv")
    print(f"  Pesos: {W_TEXT:.0%} texto + {W_TABULAR:.0%} tabular")
    print(f"  Mediana pred ensemble: ${submission_ensemble['predicted_price'].median():,.0f}")
    print(f"  Mediana pred tabular:  ${merged['pred_tabular'].median():,.0f}")
else:
    print(f"(No se encontró {tabular_path} - solo se exportó el modelo texto)")

# %% - Resumen final del notebook
print()
print("=" * 60)
print("  RESUMEN - Text Modeling (TF-IDF + LightGBM)")
print("=" * 60)
print(f"  Vocabulario TF-IDF:   {len(vocab):,} términos (uni + bi-gramas)")
print(f"  Propiedades train:    {len(train):,}")
print(f"  Propiedades test:     {len(test):,}")
print()
print(f"  OOF wMAPE (texto):    {oof_wmape:.2f}%")
print(f"  OOF R2    (texto):    {oof_r2:.4f}")
print(f"  MAE       (texto):    ${oof_mae_dollar:,.0f}")
print()
print(f"  Referencia tabular:   ~21% wMAPE")
print()
print("  Palabras clave POSITIVAS (precio alto):")
for _, r in imp_df[imp_df['corr_log_price'] > 0.05].nlargest(5, 'importance').iterrows():
    print(f"    + {r['term']}")
print()
print("  Palabras clave NEGATIVAS (precio bajo):")
for _, r in imp_df[imp_df['corr_log_price'] < -0.05].nsmallest(5, 'importance').iterrows():
    print(f"    - {r['term']}")
print("=" * 60)
