"""
Extrae embeddings CLIP de la foto principal de cada propiedad.

Guarda:
  data/embeddings_clip512.npz   — zpids (N,) + embeddings (N x 512), L2-normalizados
  data/distress_scores.csv      — zpid, clip_distress_score (0=bueno, 1=deteriorado)

Distress score: probabilidad CLIP zero-shot de que la propiedad se vea deteriorada/abandonada.
Tiempo estimado: 30-45 min en CPU.

Run desde participant/:
    python models/extract_embeddings.py
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

TEXT_POSITIVE = "a well-maintained, clean, move-in ready residential property"
TEXT_NEGATIVE = "a distressed, damaged, or abandoned property in poor condition"

# ── Cargar metadata de fotos ──────────────────────────────────────────────────
print("Cargando metadata de fotos...")
train_meta = pd.read_csv('data/train_photo_metadata.csv')
test_meta  = pd.read_csv('data/test_photo_metadata.csv')
meta_all   = pd.concat([train_meta, test_meta], ignore_index=True)

# Una foto por propiedad: preferir image_index==0; fallback al menor disponible
meta_one = (
    meta_all
    .sort_values('image_index')
    .groupby('zpid', as_index=False)
    .first()
)
zpids     = meta_one['zpid'].values.astype(int)
img_paths = meta_one['image_path'].values
N = len(zpids)
print(f"Propiedades a procesar: {N:,}")

# ── Cargar modelo CLIP ────────────────────────────────────────────────────────
print(f"\nCargando {MODEL_ID}...")
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model     = CLIPModel.from_pretrained(MODEL_ID)
model.eval()

# Pre-computar features de texto (solo una vez)
text_inputs = processor(text=[TEXT_POSITIVE, TEXT_NEGATIVE],
                        return_tensors="pt", padding=True)
with torch.no_grad():
    text_enc    = model.text_model(input_ids=text_inputs['input_ids'],
                                   attention_mask=text_inputs['attention_mask'])
    text_features = model.text_projection(text_enc.pooler_output)  # (2, 512)
    text_features = F.normalize(text_features, dim=-1)
    logit_scale   = model.logit_scale.exp()                         # escalar

print(f"Modelo cargado. Procesando {N:,} imágenes en batches de {BATCH_SIZE}...\n")

# ── Loop de extracción ────────────────────────────────────────────────────────
all_embeddings = np.zeros((N, 512), dtype=np.float32)
all_distress   = np.zeros(N, dtype=np.float32)

t0 = time.time()
for batch_start in range(0, N, BATCH_SIZE):
    batch_end   = min(batch_start + BATCH_SIZE, N)
    batch_paths = img_paths[batch_start:batch_end]

    pil_images = []
    for p in batch_paths:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception:
            pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    inputs = processor(images=pil_images, return_tensors="pt", padding=True)

    with torch.no_grad():
        vis_enc      = model.vision_model(pixel_values=inputs['pixel_values'])
        img_features = model.visual_projection(vis_enc.pooler_output)  # (B, 512)
        img_features = F.normalize(img_features, dim=-1)

    all_embeddings[batch_start:batch_end] = img_features.numpy()

    # Distress score: softmax sobre [positivo, negativo], tomar prob(negativo)
    logits = logit_scale * (img_features @ text_features.T)   # (B, 2)
    probs  = logits.softmax(dim=-1)
    all_distress[batch_start:batch_end] = probs[:, 1].numpy()

    # Progreso cada ~200 propiedades
    if batch_end % 200 < BATCH_SIZE or batch_end == N:
        elapsed = time.time() - t0
        eta     = elapsed / batch_end * (N - batch_end) if batch_end < N else 0
        print(f"  {batch_end:>6,}/{N:,}  ({elapsed/60:.1f} min transcurridos"
              f"{f', ~{eta/60:.1f} min restantes' if eta > 0 else ', listo'})")

# ── Guardar resultados ────────────────────────────────────────────────────────
total_time = time.time() - t0
print(f"\nTiempo total: {total_time/60:.1f} minutos")

np.savez_compressed(
    'data/embeddings_clip512.npz',
    zpids=zpids,
    embeddings=all_embeddings,
)
print(f"Guardado: data/embeddings_clip512.npz  (shape: {all_embeddings.shape})")

distress_df = pd.DataFrame({'zpid': zpids, 'clip_distress_score': all_distress})
distress_df.to_csv('data/distress_scores.csv', index=False)
print(f"Guardado: data/distress_scores.csv  ({len(distress_df):,} filas)")

# ── Verificación ──────────────────────────────────────────────────────────────
print(f"\n--- Verificación ---")
norms = np.linalg.norm(all_embeddings[:20], axis=1)
print(f"Normas L2 (primeras 20, deben ser ~1.0): {norms.round(4)}")
print(f"Distress score: min={all_distress.min():.3f}  "
      f"mediana={np.median(all_distress):.3f}  "
      f"max={all_distress.max():.3f}")
print(f"Propiedades con distress > 0.5: "
      f"{(all_distress > 0.5).sum():,} ({(all_distress > 0.5).mean()*100:.1f}%)")
