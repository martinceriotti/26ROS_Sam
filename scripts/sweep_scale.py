"""
Barrido de factores de escala sobre predicciones OOF.

Para cada escala corre 1,000 simulaciones Monte Carlo con los mismos
parametros del simulador de competencia, y reporta una tabla comparativa.

Uso:
    python scripts/sweep_scale.py [oof_file] [export_dir]

Ejemplo:
    python scripts/sweep_scale.py oof_round6_surgical.csv resultado_6

Si no se especifica, usa oof_round6_surgical.csv + resultado_6.
Los factores de escala se definen en SCALES (ver abajo).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# -- Importar funciones del simulador ------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from simulate_competition import (
    parse_all_exports,
    fit_competitor_model,
    run_monte_carlo,
    CAPITAL_PER_ROUND,
    N_ROUNDS,
    N_SIMS,
)

# -- Configuracion ---------------------------------------------------------------
SCALES = [0.97, 1.00, 1.03, 1.05, 1.08, 1.10]   # factores a barrer

# -- Args ------------------------------------------------------------------------
base         = Path(".")
oof_filename = sys.argv[1] if len(sys.argv) > 1 else "oof_round6_surgical.csv"
export_dir   = sys.argv[2] if len(sys.argv) > 2 else "resultado_6"

model_label  = oof_filename.replace("oof_", "").replace(".csv", "")

print(f"\nBarrido de escalas — {model_label}")
print(f"Escalas a probar: {SCALES}")
print(f"Exports: {export_dir}")
print("="*70)

# -- Cargar datos ----------------------------------------------------------------
oof   = pd.read_csv(base / "submissions" / oof_filename)
oof   = oof.rename(columns={"predicted_price": "our_pred"})

train = pd.read_csv(base / "data" / "tabular" / "train_processed.csv",
                    usecols=["zpid", "lastSoldPrice_hpi_adjusted"])
train = train.rename(columns={"lastSoldPrice_hpi_adjusted": "true_value"})

merged = oof.merge(train, on="zpid", how="inner")
merged = merged[(merged["true_value"] > 0) & (merged["our_pred"] > 0)].dropna()
print(f"\n[1] OOF: {len(merged):,} propiedades con true value")

our_preds_base = merged["our_pred"].values
true_values    = merged["true_value"].values

# -- Calibrar competidores (una sola vez) ----------------------------------------
export_path = str(base / "Salidas" / export_dir)
ratio_df, n_sam_bids = parse_all_exports(export_path)
print(f"[2] Exports '{export_dir}': {len(ratio_df):,} bids  |  SAM bids: {n_sam_bids}")
print(f"    Equipos: {sorted(ratio_df['team'].unique())}")

# Exports recientes (resultado_5, resultado_6) ya tienen ratios vs R5 -> sin ajuste
if export_dir in ("resultado_6", "resultado_5", "resultados_5"):
    ratio_df["ratio_adjusted"] = ratio_df["ratio"]
else:
    # Intentar ajuste de escala vs round3_distress_fix
    try:
        r3_test  = pd.read_csv(base / "submissions" / "round3_distress_fix.csv")
        cur_test = pd.read_csv(base / "submissions" / oof_filename.replace("oof_", ""))
        r3_test  = r3_test.rename(columns={"predicted_price": "r3"})
        cur_test = cur_test.rename(columns={"predicted_price": "cur"})
        r_merge  = r3_test.merge(cur_test, on="zpid")
        scale_ref = (r_merge["cur"] / r_merge["r3"]).median()
    except Exception:
        scale_ref = 1.0
    ratio_df["ratio_adjusted"] = ratio_df["ratio"] / scale_ref
    print(f"    Factor escala vs R3: {scale_ref:.4f}")

competitor_params = {}
for team in sorted(ratio_df["team"].unique()):
    ratios = ratio_df[ratio_df["team"] == team]["ratio_adjusted"].values
    if len(ratios) < 3:
        continue
    competitor_params[team] = fit_competitor_model(ratios, n_sam_bids)

print(f"[3] Competidores calibrados: {list(competitor_params.keys())}")

# -- Barrido de escalas ---------------------------------------------------------
print(f"\n[4] Corriendo {N_SIMS:,} sims x {len(SCALES)} escalas...\n")

results = []
total_invested = N_ROUNDS * CAPITAL_PER_ROUND

for scale in SCALES:
    our_preds_scaled = our_preds_base * scale

    roi_by_team, props_bought, profitable = run_monte_carlo(
        our_preds_scaled,
        true_values,
        competitor_params,
        n_sims=N_SIMS,
        rng_seed=42,
    )

    # Calcular win rate
    teams      = list(roi_by_team.keys())
    roi_matrix = np.array([roi_by_team[t] for t in teams])
    win_rates  = {}
    for i, t in enumerate(teams):
        win_rates[t] = (roi_matrix.argmax(axis=0) == i).sum() / N_SIMS * 100

    sam_roi  = roi_by_team['SAM']
    sam_prop = props_bought['SAM']
    sam_hit  = (profitable['SAM'] / np.maximum(props_bought['SAM'], 1)).mean() * 100

    results.append({
        'Escala':         scale,
        'ROI Medio (%)':  round(sam_roi.mean(),   2),
        'ROI Mediana (%)':round(np.median(sam_roi), 2),
        'Std ROI (%)':    round(sam_roi.std(),    2),
        'Sharpe':         round(sam_roi.mean() / sam_roi.std() if sam_roi.std() > 0 else 0, 3),
        'Win Rate (%)':   round(win_rates['SAM'], 2),
        'VaR 5% (%)':     round(np.percentile(sam_roi, 5), 2),
        'Prob+ (%)':      round((sam_roi > 0).mean() * 100, 2),
        'Props/sim':      round(sam_prop.mean(), 1),
        'Hit Rate (%)':   round(sam_hit, 1),
    })

    print(f"  Escala x{scale:.2f} -> ROI={sam_roi.mean():.1f}%  "
          f"WinRate={win_rates['SAM']:.1f}%  "
          f"Props={sam_prop.mean():.1f}  "
          f"VaR5={np.percentile(sam_roi,5):.1f}%")

# -- Tabla final ----------------------------------------------------------------
print("\n" + "="*70)
print("TABLA COMPARATIVA DE ESCALAS - EQUIPO SAM")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Destacar mejor escala por cada metrica
best_roi     = results_df.loc[results_df['ROI Medio (%)'].idxmax(), 'Escala']
best_winrate = results_df.loc[results_df['Win Rate (%)'].idxmax(), 'Escala']
best_sharpe  = results_df.loc[results_df['Sharpe'].idxmax(), 'Escala']

print(f"\nMejor por ROI medio:   x{best_roi:.2f}")
print(f"Mejor por Win Rate:    x{best_winrate:.2f}")
print(f"Mejor por Sharpe:      x{best_sharpe:.2f}")

# Guardar
Path("reports").mkdir(exist_ok=True)
out_csv = f"reports/sweep_scale_{model_label}.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nTabla guardada en: {out_csv}")
