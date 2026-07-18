"""
Kelly Criterion aplicado a la estrategia de bidding — Equipo SAM
=================================================================
El profesor sugirio usar el criterio de Kelly para decidir cuanto "arriesgar".

Kelly clasico asume que perder = perder toda la apuesta. Ese NO es nuestro caso:
si nos gana la subasta, no perdemos nada (el capital queda intacto). Lo que
realmente puede pasar es GANAR la subasta y pagar de mas (perdida acotada al
sobreprecio, no a todo el capital). Por eso adaptamos Kelly a las dos
magnitudes reales del juego, condicionado a que ganamos la subasta:

    p = Hit Rate (prob. de que la compra sea rentable, dado que compramos)
    q = 1 - p
    W = ganancia media como fraccion del costo, cuando es rentable
    L = perdida media como fraccion del costo, cuando NO es rentable

    f* = p/L - q/W      (Kelly generalizado para ganancias/perdidas asimetricas)

f* > 0 y grande  -> el segmento tiene edge real, conviene ser MAS agresivo
f* <= 0          -> sin edge (o edge negativo) -> ser MAS conservador

Usamos esto para pasar de una escala PLANA (0.83 para todas las propiedades,
la que ya validamos en Practice) a una escala VARIABLE por segmento
(SINGLE_FAMILY / CONDO / RESTO), mas agresiva donde el Kelly de ese segmento
es alto y mas conservadora donde es bajo.

Run desde participant/:
    python models/kelly_experiment.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from simulate_competition import (
    parse_all_exports, fit_competitor_model,
    CAPITAL_PER_ROUND, N_ROUNDS, N_SIMS, N_PROPS_PER_ROUND,
    ASK_MU, ASK_SIGMA, BID_MULT, BUY_THRESHOLD,
)

SEGMENTS = ['SF', 'CONDO', 'REST']
BASELINE_SCALE = 0.83


def assign_segment(home_type):
    if home_type == 'SINGLE_FAMILY':
        return 'SF'
    if home_type == 'CONDO':
        return 'CONDO'
    return 'REST'


def run_instrumented(our_preds, true_values, segments, competitor_params,
                      n_sims=N_SIMS, rng_seed=42):
    """
    Igual mecanica que simulate_competition.run_monte_carlo, pero ademas de el
    ROI agregado, acumula por segmento: cuantas veces ganamos la subasta,
    cuantas de esas fueron rentables, y la magnitud de la ganancia/perdida
    como fraccion del costo pagado. Eso es lo que necesita Kelly.
    """
    rng = np.random.default_rng(rng_seed)
    n_props = len(our_preds)

    # Los buckets se derivan de lo que venga en `segments` (no asumimos que
    # son los 3 segmentos de siempre) — asi la misma funcion sirve tanto para
    # Kelly por segmento como para el EV a grano mas fino (ver edge_experiment.py).
    stats = {b: dict(n_bid=0, n_win=0, n_win_profit=0, profit_sum_ev=0.0,
                      profit_frac_sum=0.0, loss_frac_sum=0.0)
             for b in np.unique(segments)}

    sam_roi = np.zeros(n_sims)
    total_invested = N_ROUNDS * CAPITAL_PER_ROUND

    for sim in range(n_sims):
        sam_profit_total = 0.0
        for _ in range(N_ROUNDS):
            capital_sam = float(CAPITAL_PER_ROUND)
            capital_comp = {t: float(CAPITAL_PER_ROUND) for t in competitor_params}

            idx = rng.choice(n_props, size=N_PROPS_PER_ROUND, replace=False)
            for i in idx:
                true_val = true_values[i]
                our_pred = our_preds[i]
                seg = segments[i]

                shock = rng.normal(ASK_MU, ASK_SIGMA)
                asking = max(true_val * (1 + shock), 10_000)

                bids = {}
                if our_pred > asking * BUY_THRESHOLD and capital_sam > 0:
                    bids['SAM'] = our_pred * BID_MULT
                    stats[seg]['n_bid'] += 1

                if 'SAM' in bids:
                    for team, params in competitor_params.items():
                        if capital_comp[team] <= 0:
                            continue
                        if rng.random() > params['p_cond']:
                            continue
                        ratio = float(rng.lognormal(params['log_mean'], params['log_std']))
                        comp_pred = our_pred * ratio
                        if comp_pred > asking * BUY_THRESHOLD:
                            bids[team] = comp_pred * BID_MULT

                if not bids:
                    continue

                sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
                winner = sorted_bids[0][0]
                cost = sorted_bids[1][1] if len(sorted_bids) >= 2 else sorted_bids[0][1]

                if winner == 'SAM':
                    if cost > capital_sam:
                        continue
                    capital_sam -= cost
                    profit = true_val - cost
                    sam_profit_total += profit
                    s = stats[seg]
                    s['n_win'] += 1
                    s['profit_sum_ev'] += profit
                    if profit > 0:
                        s['n_win_profit'] += 1
                        s['profit_frac_sum'] += profit / cost
                    else:
                        s['loss_frac_sum'] += (-profit) / cost
                else:
                    if cost > capital_comp.get(winner, 0):
                        continue
                    capital_comp[winner] -= cost

        sam_roi[sim] = sam_profit_total / total_invested * 100

    return sam_roi, stats


def kelly_from_stats(s):
    """f* = p/L - q/W  (Kelly generalizado, ganancia/perdida asimetrica)."""
    n_win = s['n_win']
    if n_win == 0:
        return dict(p=np.nan, W=np.nan, L=np.nan, f=np.nan)
    p = s['n_win_profit'] / n_win
    q = 1 - p
    n_loss = n_win - s['n_win_profit']
    W = s['profit_frac_sum'] / s['n_win_profit'] if s['n_win_profit'] > 0 else np.nan
    L = s['loss_frac_sum'] / n_loss if n_loss > 0 else 1e-6  # casi sin perdidas -> L chico
    if L <= 0 or np.isnan(W) or W <= 0:
        f = 0.0
    else:
        f = p / L - q / W
    return dict(p=p, W=W, L=L, f=f)


def main():
    base = Path(".")
    train = pd.read_csv(base / "data/tabular/train_processed.csv",
                        usecols=["zpid", "homeType", "lastSoldPrice_hpi_adjusted"])
    train = train.rename(columns={"lastSoldPrice_hpi_adjusted": "true_value"})
    train["segment"] = train["homeType"].map(assign_segment)

    # Usamos las predicciones CRUDAS (escala 1.0, sin ningun ajuste previo) como
    # base para todo: asi el Kelly refleja el edge "natural" del modelo, y la
    # comparacion final contra la escala plana 0.83 es limpia (misma base,
    # sin doble-escalado).
    oof = pd.read_csv(base / "submissions/oof_round8_distress_fix.csv")
    oof = oof.rename(columns={"predicted_price": "our_pred"})

    merged = train.merge(oof, on="zpid", how="inner")
    merged = merged[(merged.true_value > 0) & (merged.our_pred > 0)].dropna()
    print(f"[1] OOF + true value (escala 1.0, cruda): {len(merged):,} propiedades")
    print(f"    SF={sum(merged.segment=='SF')}  CONDO={sum(merged.segment=='CONDO')}  "
          f"REST={sum(merged.segment=='REST')}")

    ratio_df, n_sam_bids = parse_all_exports(str(base / "Salidas/resultado_6"))
    print(f"[2] Exports resultado_6: {len(ratio_df):,} bids  |  SAM bids: {n_sam_bids}")

    competitor_params = {}
    for team in sorted(ratio_df["team"].unique()):
        ratios = ratio_df[ratio_df["team"] == team]["ratio"].values
        if len(ratios) >= 3:
            competitor_params[team] = fit_competitor_model(ratios, n_sam_bids)

    raw_preds = merged["our_pred"].values
    true_values = merged["true_value"].values
    segments = merged["segment"].values

    print(f"\n[3] Corriendo simulacion instrumentada ({N_SIMS:,} sims x {N_ROUNDS} rondas) "
          f"sobre predicciones crudas (para medir el edge real por segmento)...")
    raw_roi, stats = run_instrumented(raw_preds, true_values, segments, competitor_params)
    print(f"    Prediccion cruda (escala 1.0) Mean ROI local: {raw_roi.mean():.2f}%")

    baseline_preds = raw_preds * BASELINE_SCALE
    baseline_roi, _ = run_instrumented(baseline_preds, true_values, segments, competitor_params)
    print(f"    Baseline (escala {BASELINE_SCALE} plana) Mean ROI local: {baseline_roi.mean():.2f}%")

    print(f"\n[4] Kelly por segmento (condicionado a ganar la subasta):")
    print(f"    {'Segmento':<8} {'n_win':>7} {'Hit Rate':>10} {'W (gan.)':>10} "
          f"{'L (perd.)':>10} {'Kelly f*':>10}")
    kelly = {}
    for seg in SEGMENTS:
        k = kelly_from_stats(stats[seg])
        kelly[seg] = k
        print(f"    {seg:<8} {stats[seg]['n_win']:>7} {k['p']*100:>9.1f}% "
              f"{k['W']*100:>9.1f}% {k['L']*100:>9.1f}% {k['f']:>10.3f}")

    # ── Traducir Kelly relativo a una escala por segmento ──────────────────────
    # Los valores crudos de f* (Kelly sin apalancamiento acotado) dan numeros
    # grandes (~4-5): tipico de Kelly cuando las perdidas observadas son chicas
    # y poco frecuentes en la muestra — no hay que tomarlos como una fraccion
    # literal de apuesta. Los usamos solo como ranking RELATIVO entre segmentos,
    # y aplicamos un ajuste modesto alrededor de la escala plana ya validada
    # (0.83), en vez de extrapolar a una zona [0.70, 1.00] que el barrido de
    # escala nunca demostro que fuera mejor.
    n_win_by_seg = {seg: stats[seg]['n_win'] for seg in SEGMENTS}
    f_mean = sum(kelly[seg]['f'] * n_win_by_seg[seg] for seg in SEGMENTS) / sum(n_win_by_seg.values())
    ADJUST_RANGE = 0.05   # cuanto puede moverse la escala de un segmento respecto al baseline
    SCALE_LO, SCALE_HI = 0.75, 0.90  # banda ya explorada y validada por el barrido de escala

    scale_by_segment = {}
    for seg in SEGMENTS:
        rel = (kelly[seg]['f'] - f_mean) / f_mean if f_mean != 0 else 0
        rel = np.clip(rel, -1, 1)
        scale = BASELINE_SCALE + ADJUST_RANGE * rel
        scale_by_segment[seg] = round(float(np.clip(scale, SCALE_LO, SCALE_HI)), 3)

    print(f"\n[5] Escala sugerida por segmento (ajuste modesto alrededor de {BASELINE_SCALE}, "
          f"banda [{SCALE_LO}, {SCALE_HI}]):")
    for seg in SEGMENTS:
        print(f"    {seg:<8} f*={kelly[seg]['f']:.3f} (rel. al promedio ponderado {f_mean:.3f})  "
              f"->  escala {scale_by_segment[seg]}")

    # ── Aplicar y comparar contra la escala plana 0.83 ─────────────────────────
    kelly_preds = np.array([raw_preds[i] * scale_by_segment[segments[i]]
                            for i in range(len(raw_preds))])

    print(f"\n[6] Comparando escala plana (0.83) vs escala Kelly por segmento...")
    kelly_roi, _ = run_instrumented(kelly_preds, true_values, segments, competitor_params)
    print(f"    Escala plana 0.83   -> Mean ROI local: {baseline_roi.mean():.2f}%  "
          f"(Std {baseline_roi.std():.2f})")
    print(f"    Escala Kelly/segm.  -> Mean ROI local: {kelly_roi.mean():.2f}%  "
          f"(Std {kelly_roi.std():.2f})")

    # ── Generar submissions (test + OOF) con la escala Kelly ───────────────────
    test = pd.read_csv(base / "data/tabular/test_processed.csv", usecols=["zpid", "homeType"])
    test["segment"] = test["homeType"].map(assign_segment)
    test_preds_df = pd.read_csv(base / "submissions/round8_distress_fix.csv")  # escala 1.0, cruda
    test_merged = test.merge(test_preds_df, on="zpid")
    test_merged["predicted_price"] = [
        test_merged["predicted_price"].values[j] * scale_by_segment[test_merged["segment"].values[j]]
        for j in range(len(test_merged))
    ]
    out_test = test_merged[["zpid", "predicted_price"]]
    out_test.to_csv(base / "submissions/round8_kelly_segment.csv", index=False)

    oof_out = merged[["zpid"]].copy()
    oof_out["predicted_price"] = kelly_preds
    oof_out.to_csv(base / "submissions/oof_round8_kelly_segment.csv", index=False)

    print(f"\nGuardado: submissions/round8_kelly_segment.csv")
    print(f"Guardado: submissions/oof_round8_kelly_segment.csv  <- subir al tab Practice")


if __name__ == "__main__":
    main()
