"""
03 — Text embedding model.

Extracts features from property listing descriptions using TF-IDF,
reduces with SVD, then feeds into LightGBM.

Run from the participant/ directory:
    python scripts/03_text_embeddings.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load data ────────────────────────────────────────────────────────────
train = pd.read_csv("data/tabular/train_processed.csv")
test = pd.read_csv("data/tabular/test_processed.csv")

TARGET = "log_price"

# Fill missing descriptions
train["description"] = train["description"].fillna("")
test["description"] = test["description"].fillna("")

print(f"Train: {len(train)} rows, Test: {len(test)} rows")
print(f"Non-empty descriptions: train={(train['description'] != '').sum()}, test={(test['description'] != '').sum()}")

# ── TF-IDF + SVD ────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5,
)
train_tfidf = tfidf.fit_transform(train["description"])
test_tfidf = tfidf.transform(test["description"])

N_COMPONENTS = 50
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

print(f"TF-IDF: {train_tfidf.shape[1]} features -> SVD {N_COMPONENTS} dims")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

# ── Train LightGBM on text features ─────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    random_state=42, verbosity=-1,
)
model.fit(train_svd, train[TARGET])

train_pred = model.predict(train_svd)
train_price = train["lastSoldPrice_hpi_adjusted"]
pred_price = np.expm1(train_pred)
print(f"\nText-only Train R² (log): {r2_score(train[TARGET], train_pred):.4f}")
print(f"Text-only Train MAE ($):  ${mean_absolute_error(train_price, pred_price):,.0f}")

# ── Predict & save ──────────────────────────────────────────────────────
test_pred_log = model.predict(test_svd)

submission = pd.DataFrame({
    "zpid": test["zpid"],
    "predicted_price": np.expm1(test_pred_log),
})
submission.to_csv("submissions/my_team_03.csv", index=False)
print(f"Saved submissions/my_team_03.csv ({len(submission)} rows)")
