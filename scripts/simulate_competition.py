"""
Simulador local de competencia con modelos de equipos adversarios.

Inputs:
  - submissions/oof_round4_quantile.csv   (nuestras predicciones OOF en train set)
  - data/tabular/train_processed.csv      (true values del train set)
  - Salidas/primerround/*.csv             (exports de ronda real con Auction Bids)

Salida:
  - Tabla comparativa de ROI por equipo
  - reports/simulator_results.png
  - reports/competitor_ratios.png

Mecánica replicada exactamente:
  asking = true_value × (1 + Normal(-0.07, 0.35))
  compramos si pred > asking × 1.08
  bid = pred × 0.85
  Vickrey: gana el mayor postor, paga el segundo precio
  4 rondas por simulación, capital $5M por ronda (se resetea)
  1,000 simulaciones Monte Carlo

Modelado de competidores:
  De los exports de ronda real inferimos: competitor_pred = bid / 0.85
  Ratio = competitor_pred / our_round3_pred por propiedad
  Ajuste round3 → round4 usando la diferencia en escala de predicciones
  En simulación: competitor_pred_i = our_round4_oof_i × ratio_sampled_from_lognormal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import glob
from pathlib import Path

# ─── Parámetros de simulación ─────────────────────────────────────────────────
CAPITAL_PER_ROUND = 5_000_000
N_ROUNDS          = 4
N_SIMS            = 1_000
N_PROPS_PER_ROUND = 250
ASK_MU            = -0.07
ASK_SIGMA         = 0.35
BID_MULT          = 0.85
BUY_THRESHOLD     = 1.08

# ─── Colores del equipo ────────────────────────────────────────────────────────
TEAM_COLORS = {
    'SAM':     '#C8102E',
    'Legolas': '#FFD700',
    'Merry':   '#228B22',
    'Aragorn': '#4169E1',
    'Pippin':  '#9370DB',
    'Frodo':   '#FF8C00',
    'Gimli':   '#8B4513',
}

# ─── 1. Parsear bids de competitors de exports de ronda real ──────────────────

def parse_all_exports(export_dir: str) -> tuple:
    """
    Lee todos los exports de competencia real.

    Retorna:
      ratio_df  - DataFrame con ratios competitor/our_pred para cuando bidan
      n_sam_bids - int, total de bids de SAM en los exports (denominador para p_cond)
    """
    pattern = str(Path(export_dir) / "*.csv")
    files = glob.glob(pattern)

    records     = []
    n_sam_bids  = 0  # cuántas veces SAM bid en total

    for fpath in files:
        df = pd.read_csv(fpath)
        if 'Auction Bids' not in df.columns:
            continue

        # Filas donde SAM bid (no Skip)
        sam_bid_mask = df['Decision'] == 'Bid'
        n_sam_bids  += sam_bid_mask.sum()

        # Filas con bids de otros equipos (solo donde SAM también bid)
        mask = (sam_bid_mask &
                df['Auction Bids'].notna() &
                (df['Auction Bids'] != ''))
        auctions = df[mask].copy()

        for _, row in auctions.iterrows():
            our_pred = row.get('My Prediction', np.nan)
            if pd.isna(our_pred) or our_pred < 1000:
                continue

            bids_str = str(row['Auction Bids'])
            matches = re.findall(r'(\w+): \$([0-9,]+)', bids_str)
            for team, bid_str in matches:
                if team == 'You':
                    continue
                bid = float(bid_str.replace(',', ''))
                comp_pred = bid / BID_MULT
                ratio = comp_pred / our_pred
                if 0.05 < ratio < 20:  # filtrar outliers extremos
                    records.append({
                        'zpid': row['zpid'],
                        'team': team,
                        'our_pred_r3': our_pred,
                        'comp_pred': comp_pred,
                        'ratio': ratio,
                    })

    return pd.DataFrame(records), int(n_sam_bids)


def fit_competitor_model(ratios: np.ndarray, n_sam_bids: int) -> dict:
    """
    Ajusta modelo de competidor:
      - p_cond: probabilidad de bid DADO que SAM también bid
      - lognormal sobre los ratios cuando bid

    Nota: solo observamos competidores cuando SAM bid (sesgo de selección).
    Modelamos P(comp bids | SAM bids) directamente.
    """
    log_r = np.log(ratios)
    return {
        'log_mean': log_r.mean(),
        'log_std': max(log_r.std(), 0.05),   # mínimo std numérico
        'median': np.median(ratios),
        'p_cond': min(len(ratios) / max(n_sam_bids, 1), 1.0),
        'n': len(ratios),
        'n_sam': n_sam_bids,
    }


# ─── 2. Simulación Monte Carlo ─────────────────────────────────────────────────

def run_monte_carlo(
    our_preds: np.ndarray,
    true_values: np.ndarray,
    competitor_params: dict,
    n_sims: int = N_SIMS,
    rng_seed: int = 42,
) -> dict:
    """
    Corre n_sims simulaciones Monte Carlo.
    Retorna dict {team: array(n_sims,) de ROI porcentual}.
    """
    rng = np.random.default_rng(rng_seed)
    teams = ['SAM'] + list(competitor_params.keys())

    roi_by_team = {t: np.zeros(n_sims) for t in teams}
    props_bought = {t: np.zeros(n_sims) for t in teams}
    profitable    = {t: np.zeros(n_sims) for t in teams}

    n_props = len(our_preds)
    total_invested = N_ROUNDS * CAPITAL_PER_ROUND

    for sim in range(n_sims):
        total_profit = {t: 0.0 for t in teams}
        total_bought = {t: 0     for t in teams}
        total_profit_trades = {t: 0 for t in teams}

        for _ in range(N_ROUNDS):
            capital = {t: float(CAPITAL_PER_ROUND) for t in teams}

            # Muestra aleatoria de propiedades para esta ronda
            idx = rng.choice(n_props, size=N_PROPS_PER_ROUND, replace=False)

            for i in idx:
                true_val  = true_values[i]
                our_pred  = our_preds[i]

                # Precio pedido estocástico
                shock  = rng.normal(ASK_MU, ASK_SIGMA)
                asking = true_val * (1 + shock)
                asking = max(asking, 10_000)

                # Decisiones y bids por equipo
                bids = {}

                # SAM
                if our_pred > asking * BUY_THRESHOLD and capital['SAM'] > 0:
                    bids['SAM'] = our_pred * BID_MULT

                # Competidores — modelo condicional: P(bid | SAM bid) × ratio lognormal
                # Solo relevante cuando SAM bid (sino no hay auctions que importe)
                if 'SAM' in bids:  # SAM participa → simulamos si competidores también
                    for team, params in competitor_params.items():
                        if capital[team] <= 0:
                            continue
                        # ¿El competidor decide participar esta vez?
                        if rng.random() > params['p_cond']:
                            continue
                        # Si participa, ¿cuánto predice?
                        ratio = float(rng.lognormal(
                            mean=params['log_mean'],
                            sigma=params['log_std'],
                        ))
                        comp_pred = our_pred * ratio
                        if comp_pred > asking * BUY_THRESHOLD:
                            bids[team] = comp_pred * BID_MULT

                if not bids:
                    continue

                # Subasta Vickrey: ganador = mayor bid, paga el segundo precio
                sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
                winner = sorted_bids[0][0]
                cost   = sorted_bids[1][1] if len(sorted_bids) >= 2 else sorted_bids[0][1]

                if cost > capital[winner]:
                    continue  # sin presupuesto

                capital[winner] -= cost

                profit = true_val - cost
                total_profit[winner] += profit
                total_bought[winner] += 1
                if profit > 0:
                    total_profit_trades[winner] += 1

        # Calcular ROI de esta simulación
        for t in teams:
            roi_by_team[t][sim]  = total_profit[t] / total_invested * 100
            props_bought[t][sim] = total_bought[t]
            profitable[t][sim]   = total_profit_trades[t]

    return roi_by_team, props_bought, profitable


# ─── 3. Análisis y visualización ──────────────────────────────────────────────

def print_summary(roi_by_team, props_bought, profitable, competitor_params):
    print("\n" + "="*80)
    print("  SIMULADOR DE COMPETENCIA — EQUIPO SAM  (1,000 simulaciones)")
    print("="*80)

    # Ordenar por win rate (% simulaciones en 1er puesto)
    teams = list(roi_by_team.keys())

    win_rates = {}
    for t in teams:
        # Contar cuántas sims ganó este equipo
        roi_matrix = np.array([roi_by_team[t2] for t2 in teams])  # (n_teams, n_sims)
        wins = (roi_matrix.argmax(axis=0) == teams.index(t)).sum()
        win_rates[t] = wins / N_SIMS * 100

    print(f"\n{'Equipo':<12} {'ROI Medio':>10} {'ROI Std':>9} {'Sharpe':>8} "
          f"{'Win Rate':>10} {'Props/sim':>10} {'Hit Rate':>10}")
    print("-"*73)

    sorted_teams = sorted(teams, key=lambda t: win_rates[t], reverse=True)
    for t in sorted_teams:
        roi    = roi_by_team[t]
        sharpe = roi.mean() / roi.std() if roi.std() > 0 else 0
        bought = props_bought[t].mean()
        hit    = (profitable[t] / np.maximum(props_bought[t], 1)).mean() * 100
        marker = " <-- NOSOTROS" if t == 'SAM' else ""
        print(f"{t:<12} {roi.mean():>9.1f}% {roi.std():>8.1f}% "
              f"{sharpe:>8.2f} {win_rates[t]:>9.1f}% "
              f"{bought:>9.1f}  {hit:>9.1f}%{marker}")

    print("\n  Datos de calibracion de competidores:")
    print(f"  {'Equipo':<10} {'N obs':>8} {'p_cond':>10} {'Ratio med':>11} {'log_std':>10}")
    print("  " + "-"*53)
    for team, p in competitor_params.items():
        print(f"  {team:<10} {p['n']:>8} {p['p_cond']:>10.1%} {p['median']:>11.3f} {p['log_std']:>10.3f}")
    print()


def plot_results(roi_by_team, win_rates, output_path: str):
    teams_sorted = sorted(roi_by_team.keys(), key=lambda t: np.mean(roi_by_team[t]), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Simulador de Competencia — Equipo SAM vs. Adversarios", fontsize=14, fontweight='bold')

    # Panel 1: distribución de ROI por equipo (boxplot)
    ax = axes[0]
    data = [roi_by_team[t] for t in teams_sorted]
    colors = [TEAM_COLORS.get(t, '#888888') for t in teams_sorted]
    bp = ax.boxplot(data, tick_labels=teams_sorted, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2},
                    flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_ylabel("ROI (%)")
    ax.set_title("Distribución de ROI por equipo (1,000 sims)")
    ax.tick_params(axis='x', rotation=30)

    # Panel 2: win rate (% veces en primer lugar)
    ax2 = axes[1]
    wr_sorted = [win_rates[t] for t in teams_sorted]
    bar_colors = [TEAM_COLORS.get(t, '#888888') for t in teams_sorted]
    bars = ax2.bar(teams_sorted, wr_sorted, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, team in zip(bars, teams_sorted):
        if team == 'SAM':
            bar.set_edgecolor('black')
            bar.set_linewidth(2.5)
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("% simulaciones en 1er lugar (Win Rate)")
    ax2.tick_params(axis='x', rotation=30)
    ax2.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Gráfico guardado en: {output_path}")
    plt.close()


def plot_ratio_distributions(ratio_df: pd.DataFrame, output_path: str):
    teams = ratio_df['team'].unique()
    n = len(teams)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, team in enumerate(sorted(teams)):
        ax = axes[i]
        ratios = ratio_df[ratio_df['team'] == team]['ratio'].values
        color = TEAM_COLORS.get(team, '#888888')
        ax.hist(ratios, bins=25, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(np.median(ratios), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mediana: {np.median(ratios):.2f}x')
        ax.axvline(1.0, color='red', linestyle=':', linewidth=1, label='Paridad con SAM R3')
        ax.set_title(f"{team}  (n={len(ratios)})")
        ax.set_xlabel("comp_pred / sam_pred_round3")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribución de ratios de predicción: competidores vs SAM Ronda 3", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Ratios guardados en: {output_path}")
    plt.close()


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    base = Path(".")

    # Aceptar OOF alternativo via argumento: python simulate_competition.py oof_round5_conservative.csv
    oof_filename = sys.argv[1] if len(sys.argv) > 1 else "oof_round4_quantile.csv"
    oof_path = base / "submissions" / oof_filename
    oof = pd.read_csv(oof_path).rename(columns={'predicted_price': 'our_pred'})
    model_label = oof_filename.replace("oof_", "").replace(".csv", "")
    print(f"[1] OOF cargado ({model_label}): {len(oof):,} propiedades")

    # 2. Cargar true values del train set
    train = pd.read_csv(base / "data" / "tabular" / "train_processed.csv",
                        usecols=['zpid', 'lastSoldPrice_hpi_adjusted'])
    train = train.rename(columns={'lastSoldPrice_hpi_adjusted': 'true_value'})
    print(f"[2] Train cargado: {len(train):,} propiedades")

    # 3. Merge OOF + true values
    merged = oof.merge(train, on='zpid', how='inner')
    merged = merged[(merged['true_value'] > 0) & (merged['our_pred'] > 0)].dropna()
    print(f"[3] Merge OK: {len(merged):,} propiedades con true value + predicción")

    our_preds   = merged['our_pred'].values
    true_values = merged['true_value'].values

    # 4. Parsear exports de ronda real para calibrar competidores
    ratio_df, n_sam_bids = parse_all_exports(str(base / "Salidas" / "primerround"))
    print(f"\n[4] Exports de competencia parseados: {len(ratio_df):,} observaciones de bids")
    print(f"    Total bids de SAM en exports: {n_sam_bids}")
    print(f"    Equipos detectados: {sorted(ratio_df['team'].unique())}")

    # Calcular factor de escala: el modelo actual vs R3 (baseline de calibracion)
    # Los ratios de competidores fueron medidos usando nuestras predicciones R3
    # En simulacion usamos predicciones del modelo actual → ajustar ratios
    try:
        r3_test  = pd.read_csv(base / "submissions" / "round3_distress_fix.csv")
        cur_test = pd.read_csv(base / "submissions" / oof_filename.replace("oof_", ""))
        r3_test  = r3_test.rename(columns={'predicted_price': 'r3'})
        cur_test = cur_test.rename(columns={'predicted_price': 'cur'})
        r_merge  = r3_test.merge(cur_test, on='zpid')
        scale_cur_r3 = (r_merge['cur'] / r_merge['r3']).median()
        print(f"\n[4b] Factor de escala {model_label}/R3: {scale_cur_r3:.4f}")
    except Exception:
        try:
            r3_test = pd.read_csv(base / "submissions" / "round3_distress_fix.csv")
            r4_test = pd.read_csv(base / "submissions" / "round4_quantile.csv")
            r3_test = r3_test.rename(columns={'predicted_price': 'r3'})
            r4_test = r4_test.rename(columns={'predicted_price': 'r4'})
            r_merge = r3_test.merge(r4_test, on='zpid')
            scale_cur_r3 = (r_merge['r4'] / r_merge['r3']).median()
            print(f"\n[4b] Factor de escala R4/R3 (fallback): {scale_cur_r3:.4f}")
        except Exception:
            scale_cur_r3 = 1.0
            print("\n[4b] No se pudo calcular factor de escala, usando 1.0")

    # Ajustar ratios: competitor_pred_real = our_r3 × ratio = our_cur × (ratio / scale_cur_r3)
    ratio_df['ratio_adjusted'] = ratio_df['ratio'] / scale_cur_r3

    # Filtrar ratios extremos (Legolas y Merry en distressed sales)
    # Usamos percentil 5-95 para ser robustos
    ratio_df_clean = ratio_df.copy()
    # Mantener outliers — capturar comportamiento real de Legolas/Merry

    # 5. Ajustar modelo lognormal por equipo (con p_cond)
    competitor_params = {}
    for team in sorted(ratio_df_clean['team'].unique()):
        ratios = ratio_df_clean[ratio_df_clean['team'] == team]['ratio_adjusted'].values
        if len(ratios) < 3:
            continue
        params = fit_competitor_model(ratios, n_sam_bids)
        competitor_params[team] = params
        print(f"    {team:<12} n={params['n']:>4}  p_cond={params['p_cond']:.2%}  "
              f"mediana={params['median']:.3f}x  log_std={params['log_std']:.3f}")

    # Graficamos distribución de ratios (diagnóstico)
    Path("reports").mkdir(exist_ok=True)
    plot_ratio_distributions(ratio_df, "reports/competitor_ratios.png")

    # 6. Correr simulación Monte Carlo
    print(f"\n[5] Corriendo {N_SIMS:,} simulaciones Monte Carlo "
          f"({N_ROUNDS} rondas × {N_PROPS_PER_ROUND} props/ronda)...")

    roi_by_team, props_bought, profitable = run_monte_carlo(
        our_preds, true_values, competitor_params, n_sims=N_SIMS
    )

    # 7. Calcular win rates
    teams = list(roi_by_team.keys())
    roi_matrix = np.array([roi_by_team[t] for t in teams])  # (n_teams, n_sims)
    win_rates = {}
    for i, t in enumerate(teams):
        win_rates[t] = (roi_matrix.argmax(axis=0) == i).sum() / N_SIMS * 100

    # 8. Mostrar resultados
    print_summary(roi_by_team, props_bought, profitable, competitor_params)

    # 9. Graficar
    plot_results(roi_by_team, win_rates, "reports/simulator_results.png")

    # 10. Exportar tabla de resultados
    rows = []
    for t in sorted(teams, key=lambda x: win_rates[x], reverse=True):
        roi = roi_by_team[t]
        rows.append({
            'Equipo': t,
            'ROI Medio (%)': round(roi.mean(), 2),
            'ROI Mediana (%)': round(np.median(roi), 2),
            'Std ROI (%)': round(roi.std(), 2),
            'VaR 5% (%)': round(np.percentile(roi, 5), 2),
            'Prob Positive (%)': round((roi > 0).mean() * 100, 2),
            'Win Rate (%)': round(win_rates[t], 2),
            'Props/sim (media)': round(props_bought[t].mean(), 2),
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv("reports/simulator_results.csv", index=False)
    print(f"  Tabla guardada en: reports/simulator_results.csv")


if __name__ == "__main__":
    main()
