"""
Extrae embeddings CLIP promediando hasta N_PHOTOS fotos por propiedad.

Mejora sobre extract_embeddings.py (1 foto): el embedding promedio de 5 fotos
captura exterior, living, cocina, dormitorio y baño — representación mucho más robusta.

Guarda:
  data/embeddings_clip512_multi.npz  — zpids (N,) + embeddings (N x 512) promediados y L2-normalizados
  data/clip_scores_multi.csv         — 5 scores zero-shot calculados del embedding promedio

Tiempo estimado: ~3-4 horas en CPU (hasta 87,000 imagenes).

Run desde participant/:
    python models/extract_embeddings_multi.py
"""

import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_ID   = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
N_PHOTOS   = 5   # maximo de fotos por propiedad

SCORE_TEXTS = {
    'clip_distress_score': (
        "a well-maintained, clean, move-in ready residential property",
        "a distressed, damaged, or abandoned property in poor condition",
    ),
    'clip_luxury_score': (
        "a basic affordable property with standard finishes",
        "a luxury high-end property with premium finishes, elegant design",
    ),
    'clip_renovation_score': (
        "an older property with dated finishes needing renovation",
        "a recently renovated property with modern kitchen and updated bathrooms",
    ),
    'clip_view_score': (
        "a property with no view, surrounded by other buildings",
        "a property with beautiful water view, ocean view, or waterfront",
    ),
    'clip_pool_score': (
        "a property without a swimming pool",
        "a residential property with a swimming pool in the backyard",
    ),
}

# ── Cargar metadata ───────────────────────────────────────────────────────────
print("Cargando metadata de fotos...")
train_meta = pd.read_csv('data/train_photo_metadata.csv')
test_meta  = pd.read_csv('data/test_photo_metadata.csv')
meta_all   = pd.concat([train_meta, test_meta], ignore_index=True)

# Tomar hasta N_PHOTOS fotos por propiedad (las de menor image_index primero)
meta_top = (
    meta_all
    .sort_values('image_index')
    .groupby('zpid', as_index=False)
    .head(N_PHOTOS)
)

all_zpids    = meta_top['zpid'].values.astype(int)
all_paths    = meta_top['image_path'].values
unique_zpids = meta_top['zpid'].unique().astype(int)
N_props      = len(unique_zpids)
N_images     = len(meta_top)

photos_per_prop = meta_top.groupby('zpid').size()
print(f"Propiedades: {N_props:,}")
print(f"Imagenes a procesar: {N_images:,} (promedio {N_images/N_props:.1f} fotos/prop, max {N_PHOTOS})")
print(f"Distribucion fotos por prop: {photos_per_prop.value_counts().sort_index().to_dict()}")

# ── Cargar modelo CLIP ────────────────────────────────────────────────────────
print(f"\nCargando {MODEL_ID}...")
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model     = CLIPModel.from_pretrained(MODEL_ID)
model.eval()

# Pre-computar texto para scores
text_features_dict = {}
for score_name, (text_neg, text_pos) in SCORE_TEXTS.items():
    text_inputs = processor(text=[text_neg, text_pos], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_enc = model.text_model(input_ids=text_inputs['input_ids'],
                                    attention_mask=text_inputs['attention_mask'])
        tf = model.text_projection(text_enc.pooler_output)
        text_features_dict[score_name] = F.normalize(tf, dim=-1)

logit_scale = model.logit_scale.exp()
print(f"Modelo listo. Procesando {N_images:,} imagenes en batches de {BATCH_SIZE}...\n")

# ── Extraer embeddings por imagen ─────────────────────────────────────────────
# Guardamos embedding de cada imagen individualmente, luego promediamos por zpid
all_img_embeddings = np.zeros((N_images, 512), dtype=np.float32)

t0 = time.time()
for batch_start in range(0, N_images, BATCH_SIZE):
    batch_end   = min(batch_start + BATCH_SIZE, N_images)
    batch_paths = all_paths[batch_start:batch_end]

    pil_images = []
    for p in batch_paths:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception:
            pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        vis_enc      = model.vision_model(pixel_values=inputs['pixel_values'])
        img_features = model.visual_projection(vis_enc.pooler_output)
        img_features = F.normalize(img_features, dim=-1)

    all_img_embeddings[batch_start:batch_end] = img_features.numpy()

    if batch_end % 500 < BATCH_SIZE or batch_end == N_images:
        elapsed = time.time() - t0
        eta     = elapsed / batch_end * (N_images - batch_end) if batch_end < N_images else 0
        pct     = batch_end / N_images * 100
        print(f"  {batch_end:>6,}/{N_images:,} ({pct:.1f}%)  "
              f"{elapsed/60:.1f} min transcurridos"
              f"{f'  ~{eta/60:.1f} min restantes' if eta > 0 else '  listo!'}")

# ── Promediar embeddings por propiedad ────────────────────────────────────────
print("\nPromediando embeddings por propiedad...")
prop_embeddings = np.zeros((N_props, 512), dtype=np.float32)
zpid_to_idx     = {z: i for i, z in enumerate(unique_zpids)}

# Acumular
counts = np.zeros(N_props, dtype=np.int32)
for img_i, zpid in enumerate(all_zpids):
    prop_i = zpid_to_idx[zpid]
    prop_embeddings[prop_i] += all_img_embeddings[img_i]
    counts[prop_i] += 1

# Dividir y re-normalizar (el promedio de vectores unitarios no es unitario)
for i in range(N_props):
    if counts[i] > 0:
        prop_embeddings[i] /= counts[i]

# Re-normalizar a norma 1
norms = np.linalg.norm(prop_embeddings, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-8)
prop_embeddings /= norms

print(f"  Promedio de {counts.mean():.1f} fotos por propiedad")
print(f"  Normas L2 (muestra): {np.linalg.norm(prop_embeddings[:5], axis=1).round(4)}")

# ── Calcular scores zero-shot ─────────────────────────────────────────────────
emb_tensor = torch.from_numpy(prop_embeddings)
scores_results = {'zpid': unique_zpids}

print("\nCalculando scores zero-shot...")
for score_name, text_features in text_features_dict.items():
    logits = logit_scale * (emb_tensor @ text_features.T)
    probs  = logits.softmax(dim=-1)
    vals   = probs[:, 1].detach().numpy()
    scores_results[score_name] = vals
    print(f"  {score_name}: mediana={np.median(vals):.3f}  >0.5: {(vals>0.5).sum():,} ({(vals>0.5).mean()*100:.1f}%)")

# ── Guardar ───────────────────────────────────────────────────────────────────
total_time = time.time() - t0
print(f"\nTiempo total: {total_time/60:.1f} minutos")

np.savez_compressed(
    'data/embeddings_clip512_multi.npz',
    zpids=unique_zpids,
    embeddings=prop_embeddings,
    n_photos=counts,
)
print(f"Guardado: data/embeddings_clip512_multi.npz  (shape: {prop_embeddings.shape})")

scores_df = pd.DataFrame(scores_results)
scores_df.to_csv('data/clip_scores_multi.csv', index=False)
print(f"Guardado: data/clip_scores_multi.csv  ({len(scores_df):,} filas)")

# ── Verificacion ──────────────────────────────────────────────────────────────
print(f"\n--- Verificacion ---")
print(f"Normas L2 (primeras 10): {np.linalg.norm(prop_embeddings[:10], axis=1).round(4)}")
print(f"Fotos procesadas: {counts.sum():,} imagenes en {N_props:,} propiedades")
print(f"Propiedades con 1 foto:  {(counts==1).sum()}")
print(f"Propiedades con 2-4 fotos: {((counts>=2)&(counts<5)).sum()}")
print(f"Propiedades con 5 fotos: {(counts==5).sum()}")
