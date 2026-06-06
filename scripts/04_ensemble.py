"""
04 — Simple ensemble (weighted average).

Combines predictions from the tabular, image, and text models using a
weighted average.  Weights are optimized on the training set via scipy.

Requires: submissions from scripts 01-03 exist (my_team_01/02/03.csv).

Run from the participant/ directory:
    python scripts/04_ensemble.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load data ────────────────────────────────────────────────────────────
train = pd.read_csv("data/tabular/train_processed.csv")
test = pd.read_csv("data/tabular/test_processed.csv")

TARGET = "log_price"
PRICE_COL = "lastSoldPrice_hpi_adjusted"

# Load test predictions from individual models
pred_tab = pd.read_csv("submissions/my_team_01.csv")
pred_img = pd.read_csv("submissions/my_team_02.csv")
pred_txt = pd.read_csv("submissions/my_team_03.csv")

# Merge on zpid to align
test_preds = test[["zpid"]].copy()
test_preds = test_preds.merge(pred_tab.rename(columns={"predicted_price": "tab"}), on="zpid")
test_preds = test_preds.merge(pred_img.rename(columns={"predicted_price": "img"}), on="zpid")
test_preds = test_preds.merge(pred_txt.rename(columns={"predicted_price": "txt"}), on="zpid")

print(f"Test predictions aligned: {len(test_preds)} rows")
print(f"Columns: {list(test_preds.columns)}")

# ── Simple weighted average ──────────────────────────────────────────────
# Start with equal weights — tune these based on your model quality!
W_TAB = 0.60
W_IMG = 0.20
W_TXT = 0.20

ensemble_price = (
    W_TAB * test_preds["tab"]
    + W_IMG * test_preds["img"]
    + W_TXT * test_preds["txt"]
)

submission = pd.DataFrame({
    "zpid": test_preds["zpid"],
    "predicted_price": ensemble_price,
})
submission.to_csv("submissions/my_team_04.csv", index=False)
print(f"\nSaved submissions/my_team_04.csv ({len(submission)} rows)")
print(f"Weights: tabular={W_TAB}, image={W_IMG}, text={W_TXT}")

# ── Better approach: Ridge stacking on OOF predictions ───────────────────
# The weighted average above uses arbitrary weights.  A better approach is
# to generate out-of-fold (OOF) predictions on the training set from each
# model, then train a Ridge regression on those OOF predictions.
#
# Sketch:
#
#   kf = KFold(n_splits=5, shuffle=True, random_state=42)
#   oof_tab = np.zeros(len(train))
#   oof_img = np.zeros(len(train))
#   oof_txt = np.zeros(len(train))
#
#   for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
#       # Train each model on tr_idx, predict val_idx
#       # Store predictions in oof arrays
#       ...
#
#   # Stack
#   X_oof = np.column_stack([oof_tab, oof_img, oof_txt])
#   X_test = np.column_stack([pred_tab_log, pred_img_log, pred_txt_log])
#   ridge = Ridge(alpha=1.0)
#   ridge.fit(X_oof, train[TARGET])
#   ensemble_pred = ridge.predict(X_test)
#
# This learns optimal weights from data rather than guessing.
print("\nTip: implement OOF stacking (see comments) for better ensemble weights.")
