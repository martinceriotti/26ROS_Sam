"""
01 — Basic LightGBM tabular model.

Trains a LightGBM regressor on minimal tabular features with default
hyperparameters.  No tuning, no feature engineering — just the bare minimum
to produce a valid submission CSV.

Run from the participant/ directory:
    python scripts/01_lgbm_basic.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load data ────────────────────────────────────────────────────────────
train = pd.read_csv("data/tabular/train_processed.csv")
test = pd.read_csv("data/tabular/test_processed.csv")

TARGET = "log_price"  # log1p(lastSoldPrice_hpi_adjusted)

# Minimal feature set — only the most obvious columns
FEATURES = [
    "bedrooms", "bathrooms", "livingArea", "yearBuilt",
    "latitude", "longitude", "lotAreaValue", "photoCount",
    "homeType", "zipcode",
]

CAT_FEATURES = ["homeType", "zipcode"]
for col in CAT_FEATURES:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# ── Train ────────────────────────────────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    verbosity=-1,
)
model.fit(train[FEATURES], train[TARGET], categorical_feature=CAT_FEATURES)

# ── Evaluate (on train — you don't have test targets) ───────────────────
train_pred = model.predict(train[FEATURES])
train_price = train["lastSoldPrice_hpi_adjusted"]
pred_price = np.expm1(train_pred)
print(f"Train R² (log):  {r2_score(train[TARGET], train_pred):.4f}")
print(f"Train MAE ($):   ${mean_absolute_error(train_price, pred_price):,.0f}")
print(f"Train MAPE (%):  {np.mean(np.abs((train_price - pred_price) / train_price)) * 100:.1f}%")

# ── Predict & save submission ────────────────────────────────────────────
test_pred_log = model.predict(test[FEATURES])

submission = pd.DataFrame({
    "zpid": test["zpid"],
    "predicted_price": np.expm1(test_pred_log),
})
submission.to_csv("submissions/my_team_01.csv", index=False)
print(f"\nSaved submissions/my_team_01.csv ({len(submission)} rows)")
