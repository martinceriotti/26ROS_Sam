"""Genera el PDF de resumen de la sesion 2026-06-11."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np, textwrap

pdf_path = "docs/sesion_2026-06-11_resumen.pdf"

def wrap(text, width=100):
    return "\n".join(textwrap.wrap(text, width))

AZUL    = "#1a3a5c"
VERDE   = "#2e7d32"
ROJO    = "#c62828"
GRIS    = "#f5f5f5"
NARANJA = "#e65100"

def add_header(fig, titulo, subtitulo=""):
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0.88, 1, 0.12])
    ax.set_facecolor(AZUL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.03, 0.65, titulo, color='white', fontsize=16, fontweight='bold', va='center')
    ax.text(0.03, 0.25, subtitulo, color='#add8e6', fontsize=9, va='center')
    ax.text(0.97, 0.5, "Equipo Aragorn  |  MDM Austral 2026  |  Labo 2",
            color='#add8e6', fontsize=8, va='center', ha='right')

with PdfPages(pdf_path) as pdf:

    # ── Pag 1: Portada ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(AZUL)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_facecolor(AZUL)
    ax.text(0.5, 0.74, "MDM Austral 2026", color='#add8e6', fontsize=14,
            ha='center', style='italic')
    ax.text(0.5, 0.64, "Labo 2 — Property Investment Competition",
            color='white', fontsize=20, fontweight='bold', ha='center')
    ax.text(0.5, 0.54, "Resumen de Sesion — 11 de Junio de 2026",
            color='#add8e6', fontsize=14, ha='center')
    ax.text(0.5, 0.38, "Equipo: Aragorn",
            color='white', fontsize=22, fontweight='bold', ha='center')
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, 0.20), 0.70, 0.10,
        boxstyle="round,pad=0.01", facecolor='#2c5f8a', edgecolor='white', linewidth=1))
    ax.text(0.5, 0.25,
            "Analisis drill-down  |  Diagnostico distressed  |  Modelo Round 3",
            color='white', fontsize=10, ha='center', va='center')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Pag 2: Agenda ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Agenda de la Sesion", "Que se hizo hoy")
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.80]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    items = [
        ("1", VERDE,  "Pull del repositorio",
         "Se bajo la ultima version de master. Llegaron: round2_segments.py actualizado, "
         "generate_report.py,\nreporte PDF del equipo y salidas HTML de corridas anteriores."),
        ("2", AZUL,   "Analisis drill-down del dashboard (Practice OOF)",
         "Se descargaron y analizaron los exports completos (P0-P100) de 2 submissions: "
         "'unlabeled' y\n'oof_round2_segments'. Total: 10 archivos x 250 propiedades cada uno."),
        ("3", ROJO,   "Diagnostico: causa raiz de perdidas catastroficas",
         "Se identificaron 16 propiedades donde el modelo pierde >$50K por subasta. "
         "Causa raiz:\n'distressed sales ocultas' donde taxAssessedValue >> precio real de venta."),
        ("4", NARANJA,"Implementacion Round 3 -- round3_distress_fix.py",
         "Nuevo modelo con features de deteccion de distress, mejor imputacion de "
         "last_listing_price,\ny cap de post-processing basado en P95 del ZIP."),
        ("5", VERDE,  "Evaluacion comparativa Round2 vs Round3",
         "Se subio el OOF de Round3 al dashboard Practice. Analisis completo por percentil. "
         "Resultado:\nmejora en Sharpe (+2.4%), Median ROI (+6.1%) y Std ROI (-1.7%)."),
    ]

    y = 0.92
    for num, color, titulo, desc in items:
        circ = plt.Circle((0.03, y - 0.01), 0.025, color=color, zorder=3)
        ax.add_patch(circ)
        ax.text(0.03, y - 0.01, num, color='white', fontsize=10,
                fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(0.08, y, titulo, fontsize=11, fontweight='bold', color=color, va='top')
        ax.text(0.08, y - 0.045, desc, fontsize=9, color='#333333', va='top', linespacing=1.5)
        ax.axhline(y - 0.14, xmin=0.06, xmax=0.98, color='#dddddd', linewidth=0.5)
        y -= 0.175

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Pag 3: Comparacion Round2 vs Round3 ──────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Comparacion Round2 vs Round3",
               "Profit por percentil y metricas de simulacion Monte Carlo (1,000 runs)")

    # tabla profit
    ax1 = fig.add_axes([0.03, 0.54, 0.56, 0.30])
    ax1.axis('off')
    r2_vals = [-1959710, 375294, 905757, 1576667, 6432518, 7330526]
    r3_vals = [-1734600, 364721, 940783, 1582323, 5876007, 7029235]
    labels  = ['P0', 'P25', 'P50', 'P75', 'P100', 'TOTAL']
    tdata = []
    for p, v2, v3 in zip(labels, r2_vals, r3_vals):
        d = v3 - v2
        tdata.append([p, f"${v2/1e6:.2f}M", f"${v3/1e6:.2f}M",
                      ("+" if d >= 0 else "") + f"${d/1e3:.0f}K"])
    t = ax1.table(cellText=tdata, colLabels=['Percentil','Round2','Round3','Delta'],
                  cellLoc='center', loc='center')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor(AZUL); cell.set_text_props(color='white', fontweight='bold')
        elif r == len(labels):
            cell.set_facecolor('#e3f2fd'); cell.set_text_props(fontweight='bold')
        else:
            d_val = r3_vals[r-1] - r2_vals[r-1]
            if c == 3:
                cell.set_facecolor('#e8f5e9' if d_val >= 0 else '#ffebee')
            else:
                cell.set_facecolor(GRIS if r % 2 == 0 else 'white')
        cell.set_edgecolor('#cccccc')
    ax1.set_title('Profit por percentil (muestra de 250 propiedades)', fontsize=10,
                  fontweight='bold', color=AZUL, pad=8)

    # barras
    ax2 = fig.add_axes([0.63, 0.54, 0.34, 0.30])
    x = np.arange(5); w = 0.35
    ax2.bar(x - w/2, [v/1e6 for v in r2_vals[:5]], w, label='Round2',
            color='#1565c0', alpha=0.8)
    ax2.bar(x + w/2, [v/1e6 for v in r3_vals[:5]], w, label='Round3',
            color='#43a047', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(['P0','P25','P50','P75','P100'], fontsize=8)
    ax2.set_ylabel('Profit ($M)', fontsize=8); ax2.legend(fontsize=8)
    ax2.set_title('Profit por percentil', fontsize=9, fontweight='bold', color=AZUL)
    ax2.tick_params(labelsize=8)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # tabla metricas
    ax3 = fig.add_axes([0.03, 0.18, 0.94, 0.30])
    ax3.axis('off')
    mdata = [
        ['Mean ROI (%)',       '20.88%', '21.01%', '+0.13 pp', 'Round3'],
        ['Median ROI (%)',     '17.72%', '18.81%', '+1.09 pp', 'Round3'],
        ['Std ROI (%)',        '19.90%', '19.56%', '-0.34 pp', 'Round3'],
        ['Sharpe ratio',       '1.049',  '1.074',  '+2.4%',    'Round3'],
        ['wMAPE (%)',          '21.77%', '21.66%', '-0.11 pp', 'Round3'],
        ['Prob Positive (%)',  '88.55%', '87.75%', '-0.80 pp', 'Round2'],
    ]
    t2 = ax3.table(cellText=mdata,
                   colLabels=['Metrica','Round2','Round3','Delta','Ganador'],
                   cellLoc='center', loc='center')
    t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1, 1.5)
    for (r, c), cell in t2.get_celld().items():
        if r == 0:
            cell.set_facecolor(AZUL); cell.set_text_props(color='white', fontweight='bold')
        elif c == 4 and r > 0:
            cell.set_facecolor('#e8f5e9' if mdata[r-1][4] == 'Round3' else '#fff8e1')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor(GRIS if r % 2 == 0 else 'white')
        cell.set_edgecolor('#cccccc')
    ax3.set_title('Metricas de simulacion Monte Carlo (1,000 runs)', fontsize=10,
                  fontweight='bold', color=AZUL, pad=8)

    ax4 = fig.add_axes([0.03, 0.03, 0.94, 0.11]); ax4.axis('off')
    ax4.text(0, 0.7, "Conclusion:", fontsize=10, fontweight='bold', color=AZUL)
    ax4.text(0, 0.15,
             "Round3 mejora en todas las metricas de riesgo-retorno (Sharpe, Median ROI, Std ROI). "
             "El profit total de la muestra es levemente\nmenor por menor performance en P100 "
             "(modelo mas conservador en ZIPs de lujo). La diferencia es marginal.",
             fontsize=9, color='#333')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Pag 4: Diagnostico distressed ─────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Diagnostico: Distressed Sales Ocultas",
               "Causa raiz de las perdidas catastroficas")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.80]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.axhline(0.93, color=ROJO, linewidth=1.5, alpha=0.4)
    ax.text(0, 0.935, "El problema", fontsize=11, fontweight='bold', color=ROJO)
    ax.text(0, 0.88,
            "16 propiedades con perdidas >$50K por subasta. Patron comun: taxAssessedValue entre "
            "1.3x y 6.8x el precio real de venta,\ncon tag_foreclosure=0. "
            "Son short sales, ventas de estate o divorcios. El modelo interpreta el tax alto como "
            "senal de lujo.", fontsize=9, va='top', color='#333')

    ax.axhline(0.75, color=ROJO, linewidth=1.5, alpha=0.4)
    ax.text(0, 0.755, "Los 6 peores casos (de los 16 identificados)", fontsize=11,
            fontweight='bold', color=ROJO)
    casos = [
        ["1004535", "$116K", "$1,260K", "+985%", "-$873K", "CONDO ZIP 33483 (Palm Beach)"],
        ["1008845", "$197K", "$1,298K", "+558%", "-$702K", "SF ZIP 33138 (Miami) nueva const."],
        ["1014533", "$150K", "$821K",   "+449%", "-$560K", "SF ZIP 33128 6bed/4bath"],
        ["1000512", "$276K", "$1,051K", "+281%", "-$553K", "CONDO ZIP 33432"],
        ["1014399", "$85K",  "$523K",   "+518%", "-$364K", "SF ZIP 33162 HOA $382"],
        ["1014759", "$376K", "$826K",   "+119%", "-$287K", "CONDO ZIP 33483"],
    ]
    t = ax.table(cellText=casos,
                 colLabels=['zpid','Precio Real','Prediccion','Error','Perdida','Notas'],
                 bbox=[0, 0.48, 1, 0.24], cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.4)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor(ROJO); cell.set_text_props(color='white', fontweight='bold')
        elif c in (3, 4):
            cell.set_facecolor('#ffebee')
        else:
            cell.set_facecolor(GRIS if r % 2 == 0 else 'white')
        cell.set_edgecolor('#cccccc')

    ax.axhline(0.44, color=NARANJA, linewidth=1.5, alpha=0.4)
    ax.text(0, 0.445, "Por que no se detectan con las features actuales",
            fontsize=11, fontweight='bold', color=NARANJA)
    ax.text(0, 0.39,
            "distress_score (round2) = foreclosure + price_cut + (num_price_changes > 2). "
            "Las 16 propiedades tienen tag_foreclosure=0 y tag_price_cut=0.\n"
            "No hay flags de mercado que las identifiquen. Solo el ratio "
            "taxAssessedValue / precio_real lo revela, pero ese ratio usa el target.",
            fontsize=9, va='top', color='#333')

    ax.axhline(0.27, color=AZUL, linewidth=1.5, alpha=0.4)
    ax.text(0, 0.275, "Ratio tax/precio real en el train set", fontsize=11,
            fontweight='bold', color=AZUL)
    rdata = [['<1.0 (normal)', '$459K-523K', '99% del train'],
             ['1.0-1.5',       '$246K',       '0.6%'],
             ['1.5-2.0',       '$157K',       '0.3%'],
             ['2.0-3.0',       '$150K',       '0.3%'],
             ['>3.0',          '$88K',        '0.1%']]
    t2 = ax.table(cellText=rdata,
                  colLabels=['Ratio tax/precio real', 'Precio mediano de venta', '% del train'],
                  bbox=[0, 0.04, 0.70, 0.20], cellLoc='center')
    t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 1.4)
    for (r, c), cell in t2.get_celld().items():
        if r == 0:
            cell.set_facecolor(AZUL); cell.set_text_props(color='white', fontweight='bold')
        else:
            intensity = min(r - 1, 4)
            red = min(255, 220 + intensity * 8)
            green = max(220 - intensity * 40, 100)
            cell.set_facecolor(f'#{red:02x}{green:02x}{green:02x}')
        cell.set_edgecolor('#cccccc')

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Pag 5: Cambios tecnicos Round3 ────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Round 3 — Cambios Tecnicos", "models/round3_distress_fix.py")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.80]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    cambios = [
        (VERDE,   "Fix 1: Imputacion de last_listing_price",
         "Antes: NaN -> taxAssessedValue x 1.1 (amplificaba el problema en propiedades distressed)\n"
         "Ahora: NaN -> mediana por homeType + flag listing_is_missing. "
         "Mas conservador para propiedades con tax anomalo."),
        (VERDE,   "Fix 2: Nuevos features ZIP calculados dentro del fold (sin leakage)",
         "zip_median_taxAssessed, zip_median_tax_per_sqft, zip_median_listing, zip_p95_log_price.\n"
         "Base para los features relativos que detectan anomalias de tax por ZIP."),
        (VERDE,   "Fix 3: Features de deteccion de distress relativo al ZIP",
         "tax_vs_zip_ratio = taxAssessedValue / zip_median_taxAssessed\n"
         "log_tax_vs_zip, is_tax_outlier (ratio > 2.0), tax_sqft_vs_zip, "
         "listing_vs_zip, raw_listing_to_tax, is_listing_distress"),
        (NARANJA, "Fix 4: Post-processing cap (red de seguridad)",
         "Prediccion <= zip_p95_log_price + log(1.5). Capo 3 propiedades OOF.\n"
         "Efectividad limitada en ZIPs de lujo donde el P95 es muy alto."),
        (ROJO,    "Limitacion conocida",
         "Los casos mas extremos (tax 5-7x el precio real) persisten. Solo 0.1% del train.\n"
         "MAPE en 16 propiedades distressed: 257%. Requiere enfoque diferente (round4/ensemble)."),
    ]

    color_map = {VERDE: '#e8f5e9', NARANJA: '#fff3e0', ROJO: '#ffebee', AZUL: '#e3f2fd'}
    y = 0.92
    for color, titulo, desc in cambios:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.115), 1.0, 0.125,
            boxstyle="round,pad=0.01",
            facecolor=color_map.get(color, '#f5f5f5'),
            edgecolor=color, linewidth=1.2, alpha=0.85))
        ax.text(0.01, y - 0.005, titulo, fontsize=10, fontweight='bold', color=color, va='top')
        ax.text(0.01, y - 0.038, desc, fontsize=8.5, color='#333', va='top', linespacing=1.5)
        y -= 0.172

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Pag 6: Proximos pasos ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, "Proximos Pasos", "Roadmap hacia la victoria")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.80]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    pasos = [
        ("Inmediato", NARANJA, [
            "Decidir si subir round3 al test real (Sharpe y Median ROI mejoran, diferencia marginal).",
            "Monitorear resultados de competidores en el dashboard Competition Results.",
        ]),
        ("Round 4 -- Embeddings de imagenes (mayor impacto esperado)", AZUL, [
            "Implementar CLIP o ResNet sobre la foto principal de cada propiedad.",
            "Agregar embeddings como features al LightGBM (base: scripts/02_image_embeddings.py).",
            "Este es el salto de calidad mas grande disponible en el roadmap.",
        ]),
        ("Round 4 -- Ensemble completo", VERDE, [
            "Combinar round3 (tabular) + modelo de imagenes con pesos por segmento.",
            "Calibrar predictions con los errores observados en el drill-down.",
            "Ver scripts/04_ensemble.py como punto de partida.",
        ]),
        ("Mejora tactica de bidding", ROJO, [
            "Inferir bids de competidores (bid / 0.85) del drill-down.",
            "Analizar si conviene elevar el umbral de oferta (>1.08x asking) para ser mas selectivos.",
            "El problema principal no es el umbral sino los overestimates extremos.",
        ]),
    ]

    y = 0.91
    for prioridad, color, items_list in pasos:
        h = 0.065 + len(items_list) * 0.042
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.01, y - h), 1.02, h + 0.02,
            boxstyle="round,pad=0.01", facecolor=GRIS, edgecolor=color, linewidth=1.5))
        ax.text(0.00, y - 0.005, prioridad, fontsize=11, fontweight='bold', color=color, va='top')
        for i, item in enumerate(items_list):
            ax.text(0.02, y - 0.042 - i * 0.042, f"  {item}", fontsize=9,
                    color='#333', va='top')
        y -= (h + 0.025)

    ax.text(0.5, 0.035,
            "Estado del roadmap:  Ronda 1: LISTO  |  Ronda 2: LISTO  |  "
            "Ronda 3: LISTO  |  Ronda 4 (imagenes): PENDIENTE",
            fontsize=10, ha='center', color=AZUL, fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

print(f"PDF generado: {pdf_path}")
