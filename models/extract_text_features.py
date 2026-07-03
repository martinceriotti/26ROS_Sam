"""
extract_text_features.py — Extraccion de features LLM desde descripciones
=========================================================================
Usa Claude Haiku 4.5 para extraer campos estructurados de las ~8,100
propiedades con descripciones NO-boilerplate (48% del dataset).

Costo estimado: ~$3-4 USD  |  Tiempo: ~10-15 min (10 workers async)
Checkpoint automatico cada 100 propiedades — puede interrumpirse y reanudar.

Output: data/llm_description_features.csv  (16,878 filas, una por propiedad)

Run desde participant/:
    python models/extract_text_features.py
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic, RateLimitError

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path('data/llm_descriptions_checkpoint.json')
OUTPUT_PATH     = Path('data/llm_description_features.csv')
MAX_DESC_CHARS  = 1200   # cubre el 90th percentile de longitud
CONCURRENCY     = 10     # workers simultaneos
BATCH_SIZE      = 100    # guardar checkpoint cada N propiedades
CURRENT_YEAR    = 2026

SYSTEM_PROMPT = """Extract structured features from this real estate listing description. Return ONLY valid JSON, nothing else.

Schema (use exactly these keys):
{
  "condition": "excellent|good|fair|poor|unknown",
  "views": [],
  "floor_level": null,
  "is_penthouse": false,
  "is_corner_unit": false,
  "amenities": [],
  "distress": [],
  "renovation_year": null,
  "tier": "luxury|standard|budget"
}

Rules:
- views: only from [ocean, bay, city, golf, intracoastal, garden]
- amenities: only from [concierge, valet, elevator, smart_home, boat_dock, guest_house, gym, spa, wine_cellar]
- distress: only from [as_is, fixer, short_sale, foreclosure, cash_only, motivated_seller]
- floor_level: integer if mentioned, null otherwise
- renovation_year: 4-digit year if renovation/remodel year mentioned, null otherwise
- tier: luxury if high-end/premium language; budget if distressed/investor/needs work; else standard"""


# ── LLM Call ──────────────────────────────────────────────────────────────────
async def extract_features(
    client: AsyncAnthropic,
    zpid: int,
    description: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        for attempt in range(3):
            try:
                msg = await client.messages.create(
                    model='claude-haiku-4-5-20251001',
                    max_tokens=220,
                    system=SYSTEM_PROMPT,
                    messages=[{'role': 'user', 'content': description[:MAX_DESC_CHARS]}],
                )
                raw = msg.content[0].text.strip()
                # Extraer JSON aunque haya texto extra antes/despues
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    data['zpid'] = int(zpid)
                    data['_ok'] = True
                    return data
                return {'zpid': int(zpid), '_ok': False, '_error': 'no_json_found'}
            except RateLimitError:
                await asyncio.sleep(5 * (attempt + 1))
            except json.JSONDecodeError:
                if attempt == 2:
                    return {'zpid': int(zpid), '_ok': False, '_error': 'json_parse_error'}
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == 2:
                    return {'zpid': int(zpid), '_ok': False, '_error': str(e)[:80]}
                await asyncio.sleep(1)
        return {'zpid': int(zpid), '_ok': False, '_error': 'max_retries'}


# ── Feature Flattening ────────────────────────────────────────────────────────
CONDITION_MAP = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}
TIER_MAP      = {'luxury': 2, 'standard': 1, 'budget': 0}

def flatten(record: dict) -> dict:
    """Convierte el JSON del LLM en features escalares para LightGBM."""
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
        'zpid': record['zpid'],
        # Condicion y tier
        'llm_condition':       CONDITION_MAP.get(record.get('condition', 'unknown'), 0),
        'llm_tier':            TIER_MAP.get(record.get('tier', 'standard'), 1),
        # Vistas
        'llm_has_view':        int(len(views) > 0),
        'llm_has_ocean_view':  int('ocean' in views),
        'llm_has_bay_view':    int('bay' in views or 'intracoastal' in views),
        'llm_has_city_view':   int('city' in views),
        'llm_has_golf_view':   int('golf' in views),
        # Tipo de unidad
        'llm_floor_level':     float(record['floor_level']) if record.get('floor_level') else np.nan,
        'llm_is_penthouse':    int(bool(record.get('is_penthouse'))),
        'llm_is_corner':       int(bool(record.get('is_corner_unit'))),
        # Amenities
        'llm_has_concierge':   int('concierge' in amenities),
        'llm_has_elevator':    int('elevator' in amenities or 'valet' in amenities),
        'llm_has_boat_dock':   int('boat_dock' in amenities),
        'llm_has_guest_house': int('guest_house' in amenities),
        'llm_amenities_count': len(amenities),
        # Distress
        'llm_is_fixer':        int('fixer' in distress or 'as_is' in distress),
        'llm_distress_count':  len(distress),
        # Renovacion
        'llm_renovation_age':  float(CURRENT_YEAR - int(reno_year)) if valid_reno else np.nan,
        # Meta
        'llm_ok':              int(record.get('_ok', False)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def run():
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('ERROR: ANTHROPIC_API_KEY no configurada.')
        print('  Ejecutar: set ANTHROPIC_API_KEY=sk-ant-...')
        sys.exit(1)

    # Cargar datos
    train = pd.read_csv('data/tabular/train_processed.csv')
    test  = pd.read_csv('data/tabular/test_processed.csv')
    all_df = pd.concat([
        train[['zpid', 'description', 'desc_is_boilerplate']],
        test[['zpid', 'description', 'desc_is_boilerplate']],
    ], ignore_index=True)

    to_process = all_df[all_df['desc_is_boilerplate'] == 0].copy()
    print(f'Total propiedades a procesar: {len(to_process):,}  '
          f'(boilerplate omitidas: {(all_df.desc_is_boilerplate == 1).sum():,})')

    # Cargar checkpoint existente
    checkpoint: dict = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
        print(f'Checkpoint: {len(checkpoint):,} ya procesadas')

    already    = {str(z) for z in checkpoint}
    remaining  = to_process[~to_process['zpid'].astype(str).isin(already)]
    print(f'Pendientes: {len(remaining):,}')

    if len(remaining) > 0:
        client = AsyncAnthropic()
        sem    = asyncio.Semaphore(CONCURRENCY)
        rows   = list(remaining.itertuples(index=False))

        t0         = time.time()
        done_count = 0
        error_count = 0

        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch   = rows[batch_start:batch_start + BATCH_SIZE]
            tasks   = [extract_features(client, r.zpid, r.description, sem) for r in batch]
            results = await asyncio.gather(*tasks)

            ok_in_batch = 0
            for r in results:
                if r:
                    checkpoint[str(r['zpid'])] = r
                    if r.get('_ok'):
                        ok_in_batch += 1
                    else:
                        error_count += 1

            CHECKPOINT_PATH.write_text(json.dumps(checkpoint))
            done_count += len(batch)

            elapsed = time.time() - t0
            rate    = done_count / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining) - done_count) / rate / 60 if rate > 0 else 0
            print(f'  {done_count:,}/{len(remaining):,} ({done_count/len(remaining)*100:.1f}%)'
                  f'  |  {rate:.1f} prop/s  |  ETA {eta_min:.1f} min'
                  f'  |  ok={ok_in_batch}/{len(batch)}  errores acum={error_count}')

        print(f'\nExtraccion completa en {(time.time()-t0)/60:.1f} min')

    # ── Construir CSV final ────────────────────────────────────────────────────
    print('\nGenerando features finales...')

    records = [
        flatten(v)
        for v in checkpoint.values()
        if isinstance(v, dict) and 'zpid' in v
    ]
    llm_df = pd.DataFrame(records)

    # Merge con todas las propiedades (las boilerplate quedan con NaN/0)
    all_zpids = all_df[['zpid']].copy()
    merged    = all_zpids.merge(llm_df, on='zpid', how='left')

    # Boilerplate: features binarios → 0, tier → 1 (standard), condition → 0 (unknown)
    zero_cols = [c for c in merged.columns
                 if c.startswith('llm_has_') or c.startswith('llm_is_')
                 or c in ('llm_amenities_count', 'llm_distress_count', 'llm_ok')]
    for col in zero_cols:
        merged[col] = merged[col].fillna(0).astype(int)
    merged['llm_condition'] = merged['llm_condition'].fillna(0).astype(int)
    merged['llm_tier']      = merged['llm_tier'].fillna(1).astype(int)
    # llm_floor_level, llm_renovation_age → quedan NaN (LightGBM los maneja como missing)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f'Guardado: {OUTPUT_PATH}  ({len(merged):,} filas, {len(merged.columns)} columnas)')

    # Stats de cobertura
    ok = int(merged['llm_ok'].sum())
    print(f'\n--- Cobertura de extraccion ---')
    print(f'  Procesadas OK:        {ok:,} / {len(to_process):,} ({ok/len(to_process)*100:.1f}%)')
    print(f'  Penthouse detectados: {merged["llm_is_penthouse"].sum()}')
    print(f'  Con vista:            {merged["llm_has_view"].sum()}')
    print(f'  Fixer/as-is:          {merged["llm_is_fixer"].sum()}')
    print(f'  Tier luxury:          {(merged["llm_tier"] == 2).sum()}')
    print(f'  Tier budget:          {(merged["llm_tier"] == 0).sum()}')
    print(f'  Con año renovacion:   {merged["llm_renovation_age"].notna().sum()}')
    print(f'  Con floor level:      {merged["llm_floor_level"].notna().sum()}')
    print(f'\nSiguiente paso: python models/round9_text_llm.py')


if __name__ == '__main__':
    asyncio.run(run())
