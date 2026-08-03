"""
Genera el PDF de desarrollo del proyecto — Equipo SAM
MDM Austral 2026 - Labo 2: Property Investment Competition

Reporte completo desde el inicio del proyecto: que hicimos, que mejoro,
que probamos y no funciono, y la estrategia final. Pensado para compartir
con los profesores. Version compacta: rondas agrupadas de a 3 por pagina.

Run desde participant/:
    python scripts/generate_final_pdf.py
"""

import sys
import textwrap
from pathlib import Path
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import report_content as C

PAGE_W, PAGE_H = 8.5, 11.0
VERDICT_COLOR = {
    "great": C.GREEN_OK, "ok": C.INDIGO, "mixed": C.ORANGE, "bad": C.RED_BAD,
}
VERDICT_LABEL = {
    "great": "MEJORA GRANDE", "ok": "OK", "mixed": "RESULTADO MIXTO",
    "bad": "NO FUNCIONO / EMPEORO",
}

# ── Geometria del encabezado (con margen real arriba de la pagina) ────────────
HEADER_TOP = 0.98
HEADER_H = 0.06
HEADER_BOTTOM = HEADER_TOP - HEADER_H
CONTENT_START = HEADER_BOTTOM - 0.035
CONTENT_START_WITH_PILL = HEADER_BOTTOM - 0.085


class PageWriter:
    """Escribe texto en una figura matplotlib con avance automatico de Y."""

    def __init__(self, ax, y_start=CONTENT_START, x_margin=0.07):
        self.ax = ax
        self.y = y_start
        self.x = x_margin
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def skip(self, dy=0.02):
        self.y -= dy

    def write(self, text, fontsize=11, color=C.DARK, bold=False, italic=False,
              indent=0, dy_after=0.032, x=None, wrap=None, ha='left'):
        xi = (x if x is not None else self.x) + indent * 0.03
        style = 'italic' if italic else 'normal'
        lines = textwrap.wrap(text, wrap) if wrap else [text]
        for line in lines:
            self.ax.text(xi, self.y, line, transform=self.ax.transAxes,
                         fontsize=fontsize, fontweight='bold' if bold else 'normal',
                         style=style, color=color, va='top', ha=ha)
            self.y -= dy_after * 0.62
        self.y -= dy_after * 0.38

    def bullet(self, text, fontsize=10, color=C.DARK, indent=1, dy_after=0.030, wrap=88):
        xi = self.x + indent * 0.025
        lines = textwrap.wrap(text, wrap)
        for i, line in enumerate(lines):
            prefix = "▸  " if i == 0 else "    "
            self.ax.text(xi, self.y, prefix + line, transform=self.ax.transAxes,
                         fontsize=fontsize, color=color, va='top')
            self.y -= dy_after
        self.y -= 0.006

    def header_bar(self, page_label, title):
        self.ax.add_patch(FancyBboxPatch((0, HEADER_BOTTOM), 1, HEADER_H,
                           transform=self.ax.transAxes, boxstyle="square,pad=0",
                           linewidth=0, facecolor=C.NAVY, clip_on=False))
        self.ax.text(0.07, HEADER_TOP - 0.013, page_label, transform=self.ax.transAxes,
                     fontsize=9, color="white", va='top', alpha=0.85)
        self.ax.text(0.07, HEADER_TOP - 0.032, title, transform=self.ax.transAxes,
                     fontsize=15, color="white", va='top', fontweight='bold')
        self.ax.add_patch(plt.Rectangle((0, HEADER_BOTTOM), 1, 0.005, transform=self.ax.transAxes,
                          facecolor=C.ORANGE, clip_on=False))
        self.y = CONTENT_START

    def footer(self, page_num):
        self.ax.text(0.5, 0.015, f"{C.TEAM} · {C.COURSE} · p. {page_num}",
                     transform=self.ax.transAxes, fontsize=8, color=C.GRAY,
                     va='bottom', ha='center')

    def result_box(self, text, color=C.TEAL, fontsize=10.5):
        self.y -= 0.006
        y0 = self.y
        h = 0.05
        self.ax.add_patch(FancyBboxPatch((self.x, y0 - h), 1 - 2 * self.x, h,
                           transform=self.ax.transAxes, boxstyle="round,pad=0.006",
                           linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.10))
        self.ax.text(self.x + 0.015, y0 - h / 2, text, transform=self.ax.transAxes,
                     fontsize=fontsize, color=color, va='center', fontweight='bold')
        self.y = y0 - h - 0.028

    def lesson_box(self, text, wrap=82):
        lines = textwrap.wrap("LECCION: " + text, wrap)
        y0 = self.y
        h = 0.026 * len(lines) + 0.022
        self.ax.add_patch(FancyBboxPatch((self.x, y0 - h), 1 - 2 * self.x, h,
                           transform=self.ax.transAxes, boxstyle="round,pad=0.008",
                           linewidth=1, edgecolor=C.ORANGE, facecolor="#FDF3E9"))
        yy = y0 - 0.016
        for i, line in enumerate(lines):
            self.ax.text(self.x + 0.015, yy, line, transform=self.ax.transAxes,
                         fontsize=9.5, color="#7A4A15", va='top',
                         fontweight='bold' if i == 0 else 'normal')
            yy -= 0.026
        self.y = y0 - h - 0.03

    def verdict_pill(self, verdict, x_right=0.93):
        color = VERDICT_COLOR[verdict]
        self.ax.text(x_right, HEADER_BOTTOM - 0.018, VERDICT_LABEL[verdict],
                     transform=self.ax.transAxes, fontsize=8.5, color=color, va='top',
                     ha='right', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.3))
        self.y = CONTENT_START_WITH_PILL


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


def cover_page(pdf):
    fig, ax = new_page()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=C.NAVY))
    ax.add_patch(plt.Rectangle((0, 0.62), 1, 0.008, facecolor=C.ORANGE))
    ax.text(0.07, 0.74, "UNIVERSIDAD AUSTRAL", fontsize=15, color="white",
            va='top', alpha=0.85, fontweight='bold')
    ax.text(0.07, 0.68, C.COURSE, fontsize=13, color=C.ORANGE, va='top')

    ax.text(0.07, 0.52, C.PROJECT_TITLE, fontsize=25, color="white",
            va='top', fontweight='bold')
    ax.text(0.07, 0.42, "Desarrollo del proyecto, estrategia y resultados",
            fontsize=15, color="#D8DEEF", va='top')

    ax.text(0.07, 0.28, C.TEAM, fontsize=16, color=C.ORANGE, va='top', fontweight='bold')
    ax.text(0.07, 0.235, f"Modelo final: {C.FINAL_MODEL}", fontsize=10.5,
            color="#B9C2DE", va='top', family='monospace')

    ax.text(0.07, 0.155, "Contenido: mecanica del juego, los datos, cronologia ronda a\n"
            "ronda, Ronda 9 en detalle, calibracion de escala, criterio de Kelly,\n"
            "aprendizajes y recomendacion final.", fontsize=9, color="#8E98BE", va='top')

    ax.text(0.07, 0.06, date.today().isoformat(), fontsize=10, color="#8E98BE", va='top')

    for dx, dy, s, col in [
        (0.80, 0.90, 0.16, C.PINK), (0.90, 0.80, 0.10, C.TEAL), (0.72, 0.82, 0.07, C.INDIGO)
    ]:
        tri = plt.Polygon([[dx, dy], [dx + s, dy], [dx, dy - s]], closed=True,
                          facecolor=col, alpha=0.9, transform=ax.transAxes)
        ax.add_patch(tri)

    pdf.savefig(fig); plt.close(fig)


def mechanics_page(pdf, page_num):
    fig, ax = new_page()
    w = PageWriter(ax)
    w.header_bar("SECCION 1", "Objetivo y mecanica de la competencia")
    w.write(
        "Es una competencia de inversion inmobiliaria simulada. Predecimos el precio "
        "de venta de propiedades en Miami / Sur de Florida, y esa prediccion se usa "
        "para decidir, en una simulacion, si compramos cada propiedad y cuanto ofertamos.",
        fontsize=11, wrap=86, dy_after=0.034,
    )
    w.skip(0.01)
    w.write("Como funciona cada simulacion", fontsize=12.5, bold=True, color=C.NAVY)
    for m in C.MECHANICS:
        w.bullet(m, fontsize=10.3, dy_after=0.034, wrap=84)
    w.skip(0.02)
    w.write("Que significa esto en la practica", fontsize=12.5, bold=True, color=C.NAVY)
    w.write(
        "Si predecimos MUY POR ENCIMA del valor real, compramos propiedades caras y "
        "perdemos plata (perdida real). Si predecimos por debajo, simplemente no "
        "compramos esa propiedad (perdemos la oportunidad, pero no perdemos capital). "
        "Este desbalance — sobreestimar es mucho mas costoso que subestimar — explica "
        "varias de las decisiones de diseno del proyecto (ver Ronda 4, mas adelante).",
        fontsize=10.5, wrap=88, color=C.GRAY, dy_after=0.030,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def data_page(pdf, page_num):
    fig, ax = new_page()
    w = PageWriter(ax)
    w.header_bar("SECCION 2", "Los datos con los que trabajamos")
    w.write("Dos conjuntos de propiedades", fontsize=12.5, bold=True, color=C.NAVY)
    w.bullet("Train: 11,840 propiedades con precio de venta real conocido — con esto "
             "entrenamos los modelos.", dy_after=0.036, wrap=84)
    w.bullet("Test: 5,038 propiedades sin precio — sobre estas generamos las "
             "predicciones que se usan en la competencia real.", dy_after=0.036, wrap=84)
    w.skip(0.02)
    w.write("Que sabemos de cada propiedad", fontsize=12.5, bold=True, color=C.NAVY)
    w.bullet("Datos estructurados: tamano, habitaciones, banos, antiguedad, ubicacion, "
             "impuestos, escuelas cercanas, HOA, historial de precio.", dy_after=0.036, wrap=84)
    w.bullet("Fotos de la propiedad (usadas en Rondas 3 a 8 via embeddings CLIP).",
             dy_after=0.036, wrap=84)
    w.bullet("Descripcion de texto (usada en el experimento de Ronda 9).",
             dy_after=0.036, wrap=84)
    w.skip(0.02)
    w.write("Dificultades del dataset", fontsize=12.5, bold=True, color=C.NAVY)
    w.bullet("Faltantes importantes: lotAreaValue (45% de las filas), "
             "last_listing_price (33%).", dy_after=0.036, wrap=84)
    w.bullet("Features \"filtradas\" (leaky): taxAssessedValue y variables de impuestos "
             "correlacionan mucho con el precio pero no explican por que vale eso — "
             "se usan con cuidado, no como unica fuente de señal.", dy_after=0.036, wrap=84)
    w.bullet("Un numero chico de propiedades (menos del 2%) son ventas atipicas "
             "(remates, ventas familiares forzadas, etc.) con precio real muy por "
             "debajo de lo esperado — estas pocas propiedades terminaron siendo el "
             "factor mas importante de todo el proyecto (ver Rondas 3, 4 y 8).",
             dy_after=0.036, wrap=84)
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def round_card(ax, top, height, r):
    """Tarjeta compacta de una ronda (2-3 por pagina). Resumen condensado a
    lo esencial; el detalle completo de Ronda 9 y Calibracion vive en sus
    propias paginas 'en detalle'."""
    color = VERDICT_COLOR[r['verdict']]
    bottom = top - height
    ax.add_patch(FancyBboxPatch((0.05, bottom), 0.90, height, boxstyle="round,pad=0.004",
                 transform=ax.transAxes, linewidth=1.1, edgecolor=color, facecolor="white"))
    ax.text(0.07, top - 0.020, f"{r['n']} · {r['date']}", fontsize=7.8, color=C.GRAY,
            va='top', transform=ax.transAxes)
    ax.text(0.07, top - 0.046, r['name'], fontsize=12, fontweight='bold', color=C.NAVY,
            va='top', transform=ax.transAxes)
    ax.text(0.93, top - 0.020, VERDICT_LABEL[r['verdict']], fontsize=7, color=color,
            ha='right', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=color, lw=1.0))

    # Texto completo, sin cortar a mitad de frase — las rondas con resumen muy
    # largo (mas de 2 parrafos, ej. Ronda 9) ya tienen su propia pagina "en
    # detalle" mas adelante, asi que en la tarjeta compacta solo mostramos el
    # primer parrafo con una referencia a esa seccion.
    paras = r['summary'].split("\n\n")
    note = None
    if len(paras) > 3:
        paras, note = paras[:1], "(ver el detalle completo en la seccion siguiente)"

    y = top - 0.082
    for para in paras:
        for line in textwrap.wrap(para, 112):
            ax.text(0.07, y, line, fontsize=8.2, color=C.DARK, va='top', transform=ax.transAxes)
            y -= 0.0185
        y -= 0.008
    if note:
        ax.text(0.07, y, note, fontsize=7.6, color=C.GRAY, style='italic',
                va='top', transform=ax.transAxes)
        y -= 0.022

    y -= 0.010
    ax.text(0.07, y, r['result'], fontsize=8.4, fontweight='bold',
            color=color, va='top', transform=ax.transAxes)
    y -= 0.028

    if r['lesson']:
        for line in textwrap.wrap("Leccion: " + r['lesson'], 118):
            ax.text(0.07, y, line, fontsize=7.4, color="#7A4A15", style='italic',
                    va='top', transform=ax.transAxes)
            y -= 0.0165


def rounds_page(pdf, page_num, rounds_subset, page_label):
    fig, ax = new_page()
    w = PageWriter(ax)
    w.header_bar("SECCION 3 · CRONOLOGIA", page_label)
    n = len(rounds_subset)
    gap = 0.022
    top_area, bottom_area = CONTENT_START, 0.045
    card_h = (top_area - bottom_area - gap * (n - 1)) / n
    y = top_area
    for r in rounds_subset:
        round_card(ax, y, card_h, r)
        y -= card_h + gap
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def round9_detail_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PageWriter(ax)
    w.header_bar("SECCION 4 · EN DETALLE", "Ronda 9 — por que el texto no ayudo")

    chart_ax = fig.add_axes([0.10, 0.56, 0.80, 0.28])
    labels = [x[0] for x in C.ROUND9_COMPARISON]
    values = [x[1] for x in C.ROUND9_COMPARISON]
    colors = [C.GREEN_OK, C.RED_BAD, C.ORANGE]
    bars = chart_ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6)
    chart_ax.set_ylabel("Mean ROI (%)", fontsize=10)
    chart_ax.set_title("Mismo modelo, unico cambio: agregar features de texto (LLM)",
                       fontsize=11, color=C.NAVY, fontweight='bold')
    chart_ax.tick_params(axis='x', labelsize=8.3)
    for b, v in zip(bars, values):
        chart_ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha='center',
                      fontsize=9.5, fontweight='bold')
    chart_ax.spines[['top', 'right']].set_visible(False)
    chart_ax.axhline(0, color='black', linewidth=0.6)

    w.y = 0.50
    w.write("Prueba de proxies para detectar \"distressed\" en el test set", fontsize=12,
            bold=True, color=C.NAVY, dy_after=0.034)
    w.write(
        f"Ninguno predijo mejor que el azar (tasa base: {C.DISTRESS_PROXY_BASELINE:.2f}%):",
        fontsize=10, wrap=88, color=C.GRAY, dy_after=0.026,
    )
    for name, prec, rec in C.DISTRESS_PROXY_TEST:
        bar_w = min(prec / 5.0, 1.0) * 0.42
        y = w.y
        w.ax.text(0.10, y, name, fontsize=9.3, va='top', family='monospace')
        w.ax.add_patch(plt.Rectangle((0.42, y - 0.020), bar_w, 0.015,
                       transform=w.ax.transAxes, facecolor=C.INDIGO, alpha=0.8))
        w.ax.text(0.42 + bar_w + 0.012, y, f"precision {prec:.1f}%  (recall {rec:.1f}%)",
                  fontsize=8.8, va='top', color=C.GRAY)
        w.y -= 0.030
    w.skip(0.012)
    w.write(
        "Conclusion: las perdidas catastroficas parecen ser idiosincraticas "
        "(circunstancias del vendedor), no un patron detectable en fotos, tags o "
        "descripciones. Se descarto la correccion automatica basada en proxies.",
        fontsize=10, wrap=88, color=C.DARK, dy_after=0.028,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def scale_sweep_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PageWriter(ax)
    w.header_bar("SECCION 5 · EN DETALLE", "La calibracion final: barrido de escala")

    chart_ax = fig.add_axes([0.11, 0.55, 0.78, 0.30])
    scales = [x[0] for x in C.SCALE_SWEEP_LOCAL]
    rois = [x[1] for x in C.SCALE_SWEEP_LOCAL]
    chart_ax.plot(scales, rois, '-o', color=C.NAVY, markersize=4, label='Mean ROI (%)')
    chart_ax.axvline(0.83, color=C.ORANGE, linestyle='--', linewidth=1.3, label='Escala elegida (0.83)')
    chart_ax.axvline(1.00, color=C.GRAY, linestyle=':', linewidth=1.1, label='Escala original (1.00)')
    chart_ax.set_xlabel("Factor de escala aplicado a la prediccion", fontsize=9.5)
    chart_ax.set_ylabel("Mean ROI (%) — simulacion local", fontsize=9.5)
    chart_ax.set_title("Barrido de escala sobre Ronda 8 (datos reales de subastas pasadas)",
                       fontsize=10.5, color=C.NAVY, fontweight='bold')
    chart_ax.legend(fontsize=8.5, loc='lower center')
    chart_ax.spines[['top', 'right']].set_visible(False)
    chart_ax.grid(alpha=0.25)

    w.y = 0.475
    w.write(
        "Ojo: en este grafico local, la escala 0.85 muestra un pico un poco mas "
        "alto que 0.83 — pero esta simulacion casera es aproximada, no la fuente "
        "de verdad (ver por que mas abajo).",
        fontsize=9.3, wrap=90, color=C.GRAY, italic=True, dy_after=0.024,
    )
    w.write(
        "Meseta optima entre escala 0.80 y 0.85: el Mean ROI local sube de 8.26% "
        "(sin escalar) a ~10.1-10.3% (+24% relativo). Validado en el dashboard real "
        "con 3 corridas independientes, cada una con un campo competitivo distinto:",
        fontsize=10, wrap=88, color=C.DARK, dy_after=0.028,
    )
    w.skip(0.008)
    headers = ["Corrida", "N° modelos", "Escala 0.83", "Escala 0.85", "Ronda 8 (sin escalar)"]
    col_x = [0.10, 0.28, 0.46, 0.64, 0.80]
    y0 = w.y
    for cx, h in zip(col_x, headers):
        w.ax.text(cx, y0, h, fontsize=9, fontweight='bold', color=C.NAVY, va='top')
    y0 -= 0.028
    for run, n, r083, r085, r8 in C.SCALE_PRACTICE_RUNS:
        vals = [run, str(n), f"{r083:.1f}%", f"{r085:.1f}%", f"{r8:.1f}%"]
        for cx, v in zip(col_x, vals):
            w.ax.text(cx, y0, v, fontsize=9.3, va='top', family='monospace')
        y0 -= 0.026
    w.y = y0 - 0.014
    w.write(
        "El orden nunca se invirtio: escala 0.83 > escala 0.85 > Ronda 8 sin escalar, "
        "en las 3 corridas. Por eso se adopto la escala 0.83 con confianza.",
        fontsize=10, wrap=88, color=C.DARK, dy_after=0.028,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def kelly_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PageWriter(ax)
    w.header_bar("SECCION 6 · EN DETALLE", "Criterio de Kelly aplicado a la subasta")

    w.write(
        "En criollo: pensa 3 mesas de casino, una por segmento (SF, CONDO, RESTO). "
        "En cada mesa sabemos, por el historial, que tan seguido ganamos (Hit Rate) "
        "y cuanto ganamos o perdemos cuando pasa. El criterio de Kelly usa esos dos "
        "datos para decidir en que mesa conviene apostar mas fuerte y en cual mas "
        "flojo — en vez de apostar siempre lo mismo en las tres.",
        fontsize=10.3, wrap=88, color=C.DARK, dy_after=0.026,
    )
    w.write(
        "Ojo con un detalle: Kelly clasico asume que perder = perder toda la "
        "apuesta. Ese no es nuestro caso — si nos gana la subasta, no perdemos "
        "nada (el capital queda intacto). Lo adaptamos a lo que realmente puede "
        "pasar: ganar la subasta y pagar de mas, condicionado a que ganamos "
        "(formula: f* = p/L − q/W, con p=Hit Rate, W=ganancia media y L=perdida "
        "media, ambas como fraccion del costo).",
        fontsize=9.3, wrap=92, color=C.GRAY, dy_after=0.030,
    )

    y0 = w.y
    headers = ["Segmento", "Hit Rate", "Ganancia (W)", "Perdida (L)", "Kelly f*", "Escala"]
    col_x = [0.10, 0.26, 0.40, 0.55, 0.68, 0.82]
    for cx, h in zip(col_x, headers):
        w.ax.text(cx, y0, h, fontsize=9, fontweight='bold', color=C.NAVY, va='top')
    y0 -= 0.028
    for seg, p, wgan, l, f, scale in C.KELLY_RESULTS:
        vals = [seg, f"{p:.1f}%", f"{wgan:.1f}%", f"{l:.1f}%", f"{f:.2f}", f"{scale}"]
        for cx, v in zip(col_x, vals):
            w.ax.text(cx, y0, v, fontsize=9.3, va='top', family='monospace')
        y0 -= 0.028
    w.y = y0 - 0.020

    w.write("Resultado honesto", fontsize=12, bold=True, color=C.NAVY, dy_after=0.030)
    w.write(
        f"Las 3 mesas salieron casi iguales (Hit Rate 70-75% en las tres) — no hay "
        f"una claramente mejor que las otras. Por eso Kelly sugirio una escala casi "
        f"identica en los 3 segmentos, muy cerca de la plana ya validada (0.83): el "
        f"uso real que le dimos fue de verificacion, no de descubrimiento. Aun asi, "
        f"escalar por segmento en vez de plano mejoro el Mean ROI local de "
        f"{C.KELLY_FLAT_ROI:.2f}% a {C.KELLY_SEGMENT_ROI:.2f}%.",
        fontsize=10.3, wrap=88, color=C.DARK, dy_after=0.028,
    )
    w.result_box(
        f"Mean ROI local: escala plana 0.83 = {C.KELLY_FLAT_ROI:.2f}%  →  "
        f"escala Kelly por segmento = {C.KELLY_SEGMENT_ROI:.2f}%",
        color=C.TEAL,
    )
    m = C.PRACTICE_FINAL_VALIDATION
    w.lesson_box(
        f"Validado en el dashboard real de Practice contra el campeon vigente "
        f"(scale083): kelly_segment quedo practicamente empatado "
        f"({m[1][1]:.2f}% vs {m[0][1]:.2f}% Mean ROI, -{m[0][1]-m[1][1]:.2f}pp). "
        f"No lo supera, pero confirma que la escala 0.83 ya es una eleccion solida."
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def edge_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PageWriter(ax)
    w.header_bar("SECCION 7 · EN DETALLE", "Edge (valor esperado) por segmento x precio")

    w.write(
        "En criollo: Kelly (Seccion 6) separo en 3 mesas por segmento y no "
        "encontro diferencias reales entre ellas. Edge va un paso mas — parte "
        "cada una de esas 3 mesas en otras 3 segun que tan cara predice el "
        "modelo que es la propiedad DENTRO de su propio segmento (barata / "
        "media / cara). Quedan 9 mesas en total.",
        fontsize=10, wrap=90, color=C.DARK, dy_after=0.024,
    )
    w.write(
        "Formalmente es lo mismo de siempre: si pudieramos repetir la misma "
        "subasta muchas veces, cuanto ganariamos en promedio? "
        "EV = P(ganar) x (valor_real − costo pagado) — el costo solo se paga si "
        "ganamos.",
        fontsize=9.3, wrap=92, color=C.GRAY, dy_after=0.028,
    )
    w.skip(0.004)

    y0 = w.y
    headers = ["Bucket", "Hit Rate", "EV (% del bid)", "Escala sugerida"]
    col_x = [0.10, 0.34, 0.55, 0.78]
    for cx, h in zip(col_x, headers):
        w.ax.text(cx, y0, h, fontsize=9, fontweight='bold', color=C.NAVY, va='top')
    y0 -= 0.024
    for bucket, hit, ev, scale in C.EDGE_RESULTS:
        vals = [bucket, f"{hit:.1f}%", f"{ev:.2f}%", f"{scale}"]
        for cx, v in zip(col_x, vals):
            w.ax.text(cx, y0, v, fontsize=8.8, va='top', family='monospace')
        y0 -= 0.0225
    w.y = y0 - 0.018

    w.write("Resultado honesto", fontsize=12, bold=True, color=C.NAVY, dy_after=0.028)
    w.write(
        f"A diferencia de las 3 mesas de Kelly, aca SI hay un patron consistente: "
        f"en las 3 mesas 'caras' (Hit Rate ~90-93%) acertamos mucho mas seguido que "
        f"en las 3 mesas 'baratas' (Hit Rate ~74-79%) — se repite igual en los 3 "
        f"segmentos, no es casualidad. Usamos escalas mas agresivas en las mesas "
        f"caras y mas conservadoras en las baratas, lo que mejoro el Mean ROI local "
        f"de {C.EDGE_FLAT_ROI:.2f}% a {C.EDGE_BUCKET_ROI:.2f}% — mas que Kelly solo.",
        fontsize=10, wrap=90, color=C.DARK, dy_after=0.026,
    )
    w.result_box(
        f"Mean ROI local: escala plana 0.83 = {C.EDGE_FLAT_ROI:.2f}%  →  "
        f"escala por bucket fino = {C.EDGE_BUCKET_ROI:.2f}%",
        color=C.TEAL,
    )
    m = C.PRACTICE_FINAL_VALIDATION
    w.lesson_box(
        f"En el dashboard real, comparado contra el campeon vigente, edge_bucket "
        f"quedo por debajo ({m[2][1]:.2f}% vs {m[0][1]:.2f}% Mean ROI, "
        f"-{m[0][1]-m[2][1]:.2f}pp) pese al mejor Hit Rate (85.4%) y Sharpe (2.80) — "
        f"su selectividad (9 props/sim vs 14) le costo volumen. La heterogeneidad "
        f"que encontramos es real, pero no alcanza para superar la escala plana."
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def lessons_and_final_page(pdf, page_num):
    fig, ax = new_page()
    w = PageWriter(ax)
    w.header_bar("SECCION 8", "Aprendizajes clave y recomendacion final")

    for i, (title, body) in enumerate(C.META_LESSONS, start=1):
        w.write(f"{i}. {title}", fontsize=10.8, bold=True, color=C.NAVY, dy_after=0.024)
        w.write(body, fontsize=9, wrap=98, color=C.GRAY, dy_after=0.020)
        w.skip(0.008)

    w.skip(0.006)
    w.ax.add_patch(FancyBboxPatch((0.06, w.y - 0.19), 0.88, 0.19, boxstyle="round,pad=0.010",
                   transform=w.ax.transAxes, linewidth=1.4, edgecolor=C.ORANGE, facecolor="#FAFBFF"))
    y0 = w.y - 0.022
    w.ax.text(0.09, y0, "MODELO ELEGIDO PARA LA PRESENTACION FINAL", fontsize=9.5, color=C.GRAY,
             va='top', fontweight='bold', transform=w.ax.transAxes)
    y0 -= 0.032
    w.ax.text(0.09, y0, C.FINAL_MODEL, fontsize=14, color=C.NAVY, va='top',
             fontweight='bold', family='monospace', transform=w.ax.transAxes)
    y0 -= 0.040
    lines = textwrap.wrap(C.FINAL_RECOMMENDATION.split("\n\n")[1], 96)
    for line in lines:
        w.ax.text(0.09, y0, line, fontsize=9.3, color=C.DARK, va='top', transform=w.ax.transAxes)
        y0 -= 0.024

    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def main():
    out_dir = Path(__file__).parent.parent / "reports"
    out_path = out_dir / "desarrollo_proyecto_SAM.pdf"

    # Rondas agrupadas de a 2 por pagina (12 rondas -> 6 paginas) — con 3 por
    # pagina el resumen se cortaba a mitad de frase en las rondas mas largas.
    R = C.ROUNDS
    groups = [
        (R[0:2], "Rondas 1 y 2 — primeros pasos"),
        (R[2:4], "Ronda 3 y 3-4 — distressed ocultas e imagenes"),
        (R[4:6], "Ronda 4 y el simulador propio"),
        (R[6:8], "Rondas 5 y 6 — calibrando la agresividad"),
        (R[8:10], "Ronda 7 y Ronda 8 — el mejor modelo"),
        (R[10:12], "Ronda 9 y la calibracion final"),
    ]

    with PdfPages(out_path) as pdf:
        page = 1
        cover_page(pdf)
        mechanics_page(pdf, page); page += 1
        data_page(pdf, page); page += 1
        for subset, label in groups:
            rounds_page(pdf, page, subset, label); page += 1
        round9_detail_page(pdf, page); page += 1
        scale_sweep_page(pdf, page); page += 1
        kelly_page(pdf, page); page += 1
        edge_page(pdf, page); page += 1
        lessons_and_final_page(pdf, page)

    print(f"PDF generado: {out_path}  ({page} paginas + portada = {page + 1} totales)")


if __name__ == "__main__":
    main()
