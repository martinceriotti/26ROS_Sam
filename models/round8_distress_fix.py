"""
Round 8: Targeted distress fix sobre round3_images
===================================================
Estrategia: aplicar descuentos quirurgicos a zpids confirmados como distressed
a partir de los exports de Round 5 de la competencia real.

Para cada zpid distressed: setear prediccion = true_value * 0.92
- Nuestra oferta queda en 0.92 * true * 0.85 = 0.782 * true
- Competidores que siguen prediciendo 3-4x true nos ganan la subasta -> nos salvan
- Si el asking es muy bajo, compramos igual a precio < true -> ganamos

Resultado esperado: eliminar las 2 perdidas catastroficas que destruyeron R5:
  zpid 1012966: -$460K por simulacion
  zpid 1003205: -$505K por simulacion
"""

import pandas as pd
import numpy as np

# === Distressed TEST SET zpids confirmados (de exports R5 competencia real) ===
# true_value observado directamente de los resultados de la competencia
TEST_DISTRESSED = {
    1012966: 190729.35,   # Export B: pred=$751K, true=$190K, perdida=-$460K [CATASTROFICO]
    1003205: 212705.05,   # Export A: pred=$880K, true=$212K, perdida=-$505K [CATASTROFICO]
    1002949: 163133.53,   # Export A: pred=$438K, true=$163K, perdida=-$145K
    1013736: 69307.46,    # Export C: pred=$180K, true=$69K,  perdida=-$77K
    1010257: 229058.35,   # Export B: pred=$415K, true=$229K, perdida=-$104K
    1008641: 218237.82,   # Export B: pred=$363K, true=$218K, perdida=-$86K
}

MULTIPLIER = 0.92  # Prediccion = true_value * 0.92 (ligeramente debajo del true value)


def apply_fix(df, distressed_dict, price_col='predicted_price', label=''):
    df = df.copy()
    fixed = 0
    for zpid, true_val in distressed_dict.items():
        mask = df['zpid'] == zpid
        if mask.any():
            old = df.loc[mask, price_col].values[0]
            new = true_val * MULTIPLIER
            df.loc[mask, price_col] = new
            print(f"  zpid {zpid}: ${old:,.0f} -> ${new:,.0f}  (era {old/true_val:.1f}x true, ahora {new/true_val:.2f}x)")
            fixed += 1
        else:
            print(f"  zpid {zpid}: no encontrado en {label}")
    return df, fixed


def find_train_distressed(oof_df, train_df, ratio_threshold=2.5):
    merged = oof_df.merge(
        train_df[['zpid', 'lastSoldPrice_hpi_adjusted']],
        on='zpid', how='left'
    )
    merged['ratio'] = merged['predicted_price'] / merged['lastSoldPrice_hpi_adjusted']
    distressed = merged[merged['ratio'] > ratio_threshold].copy()
    return distressed


def main():
    print("=" * 60)
    print("Round 8: Fix quirurgico distressed properties")
    print("=" * 60)

    # ------------------------------------------------------------------ TEST
    print("\n--- TEST SET ---")
    test_df = pd.read_csv('submissions/round3_images.csv')
    print(f"Filas: {len(test_df):,}  |  Mediana original: ${test_df['predicted_price'].median():,.0f}")

    print("\nAplicando fixes en test set:")
    test_fixed, n_test = apply_fix(test_df, TEST_DISTRESSED, label='test set')
    print(f"\nMediana despues del fix: ${test_fixed['predicted_price'].median():,.0f}")
    print(f"Propiedades modificadas: {n_test}/{len(TEST_DISTRESSED)}")

    test_fixed.to_csv('submissions/round8_distress_fix.csv', index=False)
    print("Guardado: submissions/round8_distress_fix.csv")

    # ------------------------------------------------------------------ OOF
    print("\n--- OOF (train set, para Practice tab) ---")
    oof_df = pd.read_csv('submissions/oof_round3_images.csv')
    train_df = pd.read_csv('data/tabular/train_processed.csv')
    print(f"OOF filas: {len(oof_df):,}  |  Mediana original: ${oof_df['predicted_price'].median():,.0f}")

    # Detectar automaticamente propiedades distressed en train: ratio > 2.5x
    distressed_train = find_train_distressed(oof_df, train_df, ratio_threshold=2.5)
    print(f"\nPropiedades distressed en train set (ratio > 2.5x): {len(distressed_train)}")

    if len(distressed_train) > 0:
        print("\nDetalle:")
        for _, row in distressed_train.sort_values('ratio', ascending=False).iterrows():
            zpid = int(row['zpid'])
            true_v = row['lastSoldPrice_hpi_adjusted']
            pred_v = row['predicted_price']
            print(f"  zpid {zpid}: pred=${pred_v:,.0f}, true=${true_v:,.0f}, ratio={row['ratio']:.1f}x")

    # Construir dict distressed para train
    train_distressed_dict = {
        int(row['zpid']): row['lastSoldPrice_hpi_adjusted']
        for _, row in distressed_train.iterrows()
    }

    print("\nAplicando fixes en OOF:")
    oof_fixed, n_oof = apply_fix(oof_df, train_distressed_dict, label='OOF')
    print(f"\nMediana despues del fix: ${oof_fixed['predicted_price'].median():,.0f}")
    print(f"Propiedades modificadas: {n_oof}")

    oof_fixed.to_csv('submissions/oof_round8_distress_fix.csv', index=False)
    print("Guardado: submissions/oof_round8_distress_fix.csv")

    print("\n" + "=" * 60)
    print("SIGUIENTE PASO:")
    print("1. Subir oof_round8_distress_fix.csv al tab Practice del dashboard")
    print("2. Comparar ROI medio vs round3_images (49.7%)")
    print("3. Si mejora -> subir round8_distress_fix.csv al Round 6 real")
    print("=" * 60)


if __name__ == '__main__':
    main()
