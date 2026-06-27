"""
Extrae scores zero-shot adicionales de CLIP usando los embeddings ya guardados.
No procesa imágenes nuevamente — usa data/embeddings_clip512.npz directamente.

Scores generados (todos en rango [0,1]):
  clip_distress_score   — propiedad deteriorada/abandonada
  clip_luxury_score     — propiedad de lujo con terminaciones premium
  clip_renovation_score — propiedad recientemente renovada
  clip_view_score       — propiedad con vista al agua/oceano
  clip_pool_score       — propiedad con pileta/piscina

Guarda: data/clip_scores.csv

Run desde participant/:
    python models/extract_clip_scores.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"

# Pares de texto para cada score (positivo=1, negativo=0)
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

# ── Cargar embeddings guardados ───────────────────────────────────────────────
print("Cargando embeddings guardados...")
data_emb   = np.load('data/embeddings_clip512.npz')
zpids      = data_emb['zpids'].astype(int)
embeddings = data_emb['embeddings']   # (N, 512) — ya L2-normalizados
N = len(zpids)
print(f"  {N:,} propiedades, {embeddings.shape[1]} dims")

emb_tensor = torch.from_numpy(embeddings)  # (N, 512)

# ── Cargar modelo CLIP (solo para encodear texto) ─────────────────────────────
print(f"\nCargando {MODEL_ID} (solo para texto)...")
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model     = CLIPModel.from_pretrained(MODEL_ID)
model.eval()

logit_scale = model.logit_scale.exp()

# ── Calcular cada score ───────────────────────────────────────────────────────
results = {'zpid': zpids}

for score_name, (text_neg, text_pos) in SCORE_TEXTS.items():
    print(f"Calculando {score_name}...")

    text_inputs = processor(text=[text_neg, text_pos],
                            return_tensors="pt", padding=True)
    with torch.no_grad():
        text_enc      = model.text_model(input_ids=text_inputs['input_ids'],
                                         attention_mask=text_inputs['attention_mask'])
        text_features = model.text_projection(text_enc.pooler_output)  # (2, 512)
        text_features = F.normalize(text_features, dim=-1)

    # Similitud: (N, 2) — [neg_sim, pos_sim]
    logits = logit_scale * (emb_tensor @ text_features.T)
    probs  = logits.softmax(dim=-1)

    # Score = prob(texto positivo)
    results[score_name] = probs[:, 1].detach().numpy()

    vals = results[score_name]
    print(f"  min={vals.min():.3f}  mediana={np.median(vals):.3f}  "
          f"max={vals.max():.3f}  >0.5: {(vals>0.5).sum():,} ({(vals>0.5).mean()*100:.1f}%)")

# ── Guardar ───────────────────────────────────────────────────────────────────
scores_df = pd.DataFrame(results)
scores_df.to_csv('data/clip_scores.csv', index=False)
print(f"\nGuardado: data/clip_scores.csv  ({len(scores_df):,} filas, {len(SCORE_TEXTS)+1} columnas)")
print(scores_df.describe().round(3))
