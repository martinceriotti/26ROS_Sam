"""
02 — Image embedding model.

Extracts visual features from property photos using a pre-trained ResNet50
(frozen — no fine-tuning), reduces dimensionality with PCA, then feeds the
embeddings into LightGBM.

Requires: torch, torchvision, Pillow

Run from the participant/ directory:
    python scripts/02_image_embeddings.py
"""

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load metadata (first image per property only) ────────────────────────
train_meta = pd.read_csv("data/train_photo_metadata.csv")
test_meta = pd.read_csv("data/test_photo_metadata.csv")

# Keep only the first image per zpid (image_index == 0)
train_meta = train_meta[train_meta["image_index"] == 0].copy()
test_meta = test_meta[test_meta["image_index"] == 0].copy()

train_tab = pd.read_csv("data/tabular/train_processed.csv")
test_tab = pd.read_csv("data/tabular/test_processed.csv")

TARGET = "log_price"

# Merge to get zpid + image_path + target
train_img = train_tab[["zpid", TARGET]].merge(train_meta[["zpid", "image_path"]], on="zpid", how="inner")
test_img = test_tab[["zpid"]].merge(test_meta[["zpid", "image_path"]], on="zpid", how="inner")
print(f"Train images: {len(train_img)}, Test images: {len(test_img)}")


# ── Image dataset ────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PropertyImageDataset(Dataset):
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img)
        except Exception:
            return torch.zeros(3, 224, 224)


# ── Extract embeddings ───────────────────────────────────────────────────
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Remove classification head -> 2048-dim feature vector
backbone = torch.nn.Sequential(*list(backbone.children())[:-1], torch.nn.Flatten())
backbone = backbone.to(DEVICE).eval()


def extract_embeddings(image_paths, batch_size=64):
    ds = PropertyImageDataset(image_paths)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    all_embs = []
    with torch.no_grad():
        for batch in dl:
            embs = backbone(batch.to(DEVICE))
            all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


print("Extracting train embeddings...")
train_embs = extract_embeddings(train_img["image_path"].tolist())
print(f"  shape: {train_embs.shape}")

print("Extracting test embeddings...")
test_embs = extract_embeddings(test_img["image_path"].tolist())
print(f"  shape: {test_embs.shape}")

# ── PCA reduction ────────────────────────────────────────────────────────
N_COMPONENTS = 64
pca = PCA(n_components=N_COMPONENTS, random_state=42)
train_pca = pca.fit_transform(train_embs)
test_pca = pca.transform(test_embs)
print(f"PCA: {train_embs.shape[1]} -> {N_COMPONENTS} dims, explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# ── Train LightGBM on image features ────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    random_state=42, verbosity=-1,
)
model.fit(train_pca, train_img[TARGET].values)

train_pred = model.predict(train_pca)
train_price = train_tab.set_index("zpid").loc[train_img["zpid"].values, "lastSoldPrice_hpi_adjusted"].values
pred_price = np.expm1(train_pred)
print(f"\nImage-only Train R² (log): {r2_score(train_img[TARGET], train_pred):.4f}")
print(f"Image-only Train MAE ($):  ${mean_absolute_error(train_price, pred_price):,.0f}")

# ── Predict & save ──────────────────────────────────────────────────────
test_pred_log = model.predict(test_pca)

# Some test zpids may not have images — fill with median prediction
all_test = test_tab[["zpid"]].copy()
img_preds = pd.DataFrame({"zpid": test_img["zpid"].values, "predicted_price": np.expm1(test_pred_log)})
submission = all_test.merge(img_preds, on="zpid", how="left")
submission["predicted_price"].fillna(submission["predicted_price"].median(), inplace=True)

submission.to_csv("submissions/my_team_02.csv", index=False)
print(f"Saved submissions/my_team_02.csv ({len(submission)} rows)")
