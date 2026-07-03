"""
extract_text_features_ollama.py — Extraccion de features via modelo local (Ollama)
====================================================================================
Alternativa GRATUITA a la version con API de Anthropic.
Usa Qwen2.5:1.5b corriendo localmente a traves de Ollama.

Setup previo (una sola vez):
    1. Instalar Ollama: winget install Ollama.Ollama
    2. Bajar el modelo: ollama pull qwen2.5:1.5b
    3. Ollama arranca automaticamente como servicio en localhost:11434

Velocidad estimada en este hardware (Ryzen 7, 16 cores):
    ~30-40 tokens/seg -> ~8-12 segundos por propiedad
    100 propiedades: ~15-20 minutos
    8,099 propiedades completas: ~20-27 horas (modo FULL)

Modos de ejecucion:
    DEMO_SIZE = 200    -> modo clase/demo (~40 minutos)
    DEMO_SIZE = None   -> procesa las 8,099 (corre de noche)

Checkpoint automatico: puede interrumpirse y reanudar.

Output: data/llm_description_features.csv (misma estructura que la version API)

Run desde participant/:
    python models/extract_text_features_ollama.py
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL      = 'http://localhost:11434/api/generate'
MODEL_NAME      = 'qwen2.5:1.5b'   # cambiar a 'qwen2.5:3b' para mejor calidad
DEMO_SIZE       = 200              # None = procesar todo
MAX_DESC_CHARS  = 800              # descripcion mas corta = mas rapido
CHECKPOINT_PATH = Path('data/llm_descriptions_checkpoint.json')
OUTPUT_PATH     = Path('data/llm_description_features.csv')
CURRENT_YEAR    = 2026

# ── Prompt del sistema ────────────────────────────────────────────────────────
# Prompt conciso y directo: modelos pequeños responden mejor a instrucciones cortas
PROMPT_TEMPLATE = """Extract features from this real estate listing. Reply ONLY with JSON, no explanation.

JSON format:
{{"condition":"excellent|good|fair|poor|unknown","views":[],"floor_level":null,"is_penthouse":false,"is_corner_unit":false,"amenities":[],"distress":[],"renovation_year":null,"tier":"luxury|standard|budget"}}

Rules:
- views: ocean, bay, city, golf, intracoastal, garden
- amenities: concierge, valet, elevator, smart_home, boat_dock, guest_house
- distress: as_is, fixer, short_sale, foreclosure, cash_only
- floor_level: number or null
- renovation_year: 4-digit year or null

Listing:
{description}

JSON:"""


# ── Verificar que Ollama esta corriendo ───────────────────────────────────────
def check_ollama():
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            print(f"Ollama OK  |  Modelos disponibles: {models}")
            if not any(MODEL_NAME.split(':')[0] in m for m in models):
                print(f"\nATENCION: modelo '{MODEL_NAME}' no encontrado.")
                print(f"  Ejecutar: ollama pull {MODEL_NAME}")
                sys.exit(1)
            return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama no esta corriendo.")
        print("  Si ya lo instalaste, ejecutar: ollama serve")
        print("  O simplemente abrir la app Ollama.")
        sys.exit(1)


# ── Llamada al modelo local ───────────────────────────────────────────────────
def call_ollama(description: str) -> str:
    """Llama al modelo y devuelve el texto de respuesta."""
    prompt = PROMPT_TEMPLATE.format(description=description[:MAX_DESC_CHARS])
    payload = {
        'model': MODEL_NAME,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.0,    # determinista: misma entrada = misma salida
            'num_predict': 200,    # maximo tokens de salida
            'top_p': 1.0,
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get('response', '')
    except Exception as e:
        return f'ERROR: {e}'


def extract_json(raw: str) -> dict:
    """Extrae el JSON de la respuesta del modelo (que puede tener texto extra)."""
    # Buscar el primer bloque JSON completo
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Segundo intento: el modelo a veces agrega texto antes/despues
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def process_one(zpid: int, description: str) -> dict:
    """Procesa una descripcion y devuelve el dict con features."""
    raw = call_ollama(description)

    if raw.startswith('ERROR:'):
        return {'zpid': int(zpid), '_ok': False, '_error': raw[:80], '_raw': ''}

    data = extract_json(raw)

    if not data:
        return {'zpid': int(zpid), '_ok': False, '_error': 'json_parse_failed', '_raw': raw[:100]}

    data['zpid'] = int(zpid)
    data['_ok']  = True
    return data


# ── Aplanar JSON a features escalares ────────────────────────────────────────
CONDITION_MAP = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}
TIER_MAP      = {'luxury': 2, 'standard': 1, 'budget': 0}

def _parse_floor(val) -> float:
    if not val:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r'\d+', str(val))
    return float(m.group()) if m else np.nan

def flatten(record: dict) -> dict:
    views     = record.get('views')     or []
    amenities = record.get('amenities') or []
    distress  = record.get('distress')  or []
    reno_year = record.get('renovation_year')

    valid_reno = (
        reno_year is not None
        and isinstance(reno_year, (int, float))
        and 1950 < int(reno_year) <= CURRENT_YEAR
    )

    return {
        'zpid':              record['zpid'],
        'llm_condition':     CONDITION_MAP.get(record.get('condition', 'unknown'), 0),
        'llm_tier':          TIER_MAP.get(record.get('tier', 'standard'), 1),
        'llm_has_view':      int(len(views) > 0),
        'llm_has_ocean_view':int('ocean' in views),
        'llm_has_bay_view':  int('bay' in views or 'intracoastal' in views),
        'llm_has_city_view': int('city' in views),
        'llm_has_golf_view': int('golf' in views),
        'llm_floor_level':   _parse_floor(record.get('floor_level')),
        'llm_is_penthouse':  int(bool(record.get('is_penthouse'))),
        'llm_is_corner':     int(bool(record.get('is_corner_unit'))),
        'llm_has_concierge': int('concierge' in amenities),
        'llm_has_elevator':  int('elevator' in amenities or 'valet' in amenities),
        'llm_has_boat_dock': int('boat_dock' in amenities),
        'llm_has_guest_house':int('guest_house' in amenities),
        'llm_amenities_count':len(amenities),
        'llm_is_fixer':      int('fixer' in distress or 'as_is' in distress),
        'llm_distress_count':len(distress),
        'llm_renovation_age':float(CURRENT_YEAR - int(reno_year)) if valid_reno else np.nan,
        'llm_ok':            int(record.get('_ok', False)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    check_ollama()

    # Cargar datos
    train = pd.read_csv('data/tabular/train_processed.csv')
    test  = pd.read_csv('data/tabular/test_processed.csv')
    all_df = pd.concat([
        train[['zpid', 'description', 'desc_is_boilerplate']],
        test[['zpid', 'description', 'desc_is_boilerplate']],
    ], ignore_index=True)

    # Solo descripciones con contenido real (boilerplate no aporta)
    to_process = all_df[all_df['desc_is_boilerplate'] == 0].copy()

    if DEMO_SIZE is not None:
        to_process = to_process.sample(n=min(DEMO_SIZE, len(to_process)), random_state=42)
        print(f"MODO DEMO: procesando {len(to_process)} descripciones de {len(all_df):,} totales")
    else:
        print(f"MODO COMPLETO: procesando {len(to_process):,} descripciones")

    # Cargar checkpoint
    checkpoint: dict = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
        print(f"Checkpoint: {len(checkpoint):,} ya procesadas")

    already   = {str(z) for z in checkpoint}
    remaining = to_process[~to_process['zpid'].astype(str).isin(already)]
    print(f"Pendientes: {len(remaining):,}")

    if len(remaining) == 0:
        print("Todo procesado. Generando CSV final...")
    else:
        # Estimacion de tiempo
        est_sec = len(remaining) * 10  # ~10 seg/prop en CPU
        print(f"Tiempo estimado: {est_sec/60:.0f} minutos ({est_sec/3600:.1f} horas)\n")

        t0 = time.time()

        for i, (_, row) in enumerate(remaining.iterrows(), 1):
            result = process_one(row['zpid'], row['description'])
            checkpoint[str(row['zpid'])] = result

            # Guardar checkpoint cada 10 propiedades
            if i % 10 == 0:
                CHECKPOINT_PATH.write_text(json.dumps(checkpoint))

            # Progress
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining) - i) / rate / 60 if rate > 0 else 0

            status = 'OK' if result.get('_ok') else 'ERR'
            print(f"  [{i:4d}/{len(remaining)}] zpid={row['zpid']}  {status}  "
                  f"| {rate:.1f} prop/min  ETA {eta_min:.0f} min", end='\r')

        # Guardar checkpoint final
        CHECKPOINT_PATH.write_text(json.dumps(checkpoint))
        print(f"\n\nExtraccion completa en {(time.time()-t0)/60:.1f} min")

    # Analizar calidad de extraccion
    ok_count  = sum(1 for v in checkpoint.values() if isinstance(v, dict) and v.get('_ok'))
    err_count = len(checkpoint) - ok_count
    print(f"\n--- Calidad de extraccion ---")
    print(f"  OK:    {ok_count:,}")
    print(f"  Error: {err_count:,}")

    if err_count > 0 and err_count <= 10:
        print("\n  Errores:")
        for z, v in checkpoint.items():
            if not v.get('_ok'):
                print(f"    zpid {z}: {v.get('_error', '?')} | raw: {v.get('_raw', '')[:60]}")

    # Construir CSV final
    print("\nGenerando features finales...")
    records = [
        flatten(v) for v in checkpoint.values()
        if isinstance(v, dict) and v.get('_ok') and 'zpid' in v
    ]

    if not records:
        print("Sin resultados OK todavia. Correr de nuevo cuando el modelo este disponible.")
        return

    llm_df    = pd.DataFrame(records)
    all_zpids = all_df[['zpid']].copy()
    merged    = all_zpids.merge(llm_df, on='zpid', how='left')

    # Boilerplate y no-procesadas: valores por defecto
    zero_cols = [c for c in merged.columns
                 if c.startswith('llm_has_') or c.startswith('llm_is_')
                 or c in ('llm_amenities_count', 'llm_distress_count', 'llm_ok')]
    for col in zero_cols:
        merged[col] = merged[col].fillna(0).astype(int)
    merged['llm_condition'] = merged['llm_condition'].fillna(0).astype(int)
    merged['llm_tier']      = merged['llm_tier'].fillna(1).astype(int)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Guardado: {OUTPUT_PATH}  ({len(merged):,} filas)")

    print(f"\n--- Distribucion de features extraidos ---")
    print(f"  Tier luxury:          {(merged['llm_tier']==2).sum():,}")
    print(f"  Tier budget:          {(merged['llm_tier']==0).sum():,}")
    print(f"  Condicion excellent:  {(merged['llm_condition']==4).sum():,}")
    print(f"  Condicion poor:       {(merged['llm_condition']==1).sum():,}")
    print(f"  Penthouse:            {merged['llm_is_penthouse'].sum():,}")
    print(f"  Con vista:            {merged['llm_has_view'].sum():,}")
    print(f"  Fixer/as-is:          {merged['llm_is_fixer'].sum():,}")
    print(f"  Con concierge:        {merged['llm_has_concierge'].sum():,}")
    print(f"  Con elevator:         {merged['llm_has_elevator'].sum():,}")
    print(f"  Con reno year:        {merged['llm_renovation_age'].notna().sum():,}")
    print(f"  Con floor level:      {merged['llm_floor_level'].notna().sum():,}")

    # Mostrar ejemplos de extraccion exitosa
    ok_sample = llm_df[llm_df['llm_ok'] == 1].head(5)
    if len(ok_sample) > 0:
        print("\n--- Ejemplos de extraccion ---")
        cols_show = ['zpid', 'llm_condition', 'llm_tier', 'llm_is_penthouse',
                     'llm_has_ocean_view', 'llm_is_fixer', 'llm_floor_level']
        print(ok_sample[cols_show].to_string(index=False))

    if DEMO_SIZE is not None:
        print(f"\n[DEMO] Para correr las {len(to_process):,} propiedades completas:")
        print("  Cambiar DEMO_SIZE = None en el script y correr de noche")

    print("\nSiguiente paso: python models/round9_text_llm.py")


if __name__ == '__main__':
    main()
