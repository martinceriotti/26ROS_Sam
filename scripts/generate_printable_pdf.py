"""
Genera la version IMPRIMIBLE (blanco y negro, sin fondos) del PDF de
desarrollo — Equipo SAM. Mismo contenido que desarrollo_proyecto_SAM.pdf,
pensado para estudiar/imprimir sin gastar tinta de color: fondo blanco,
texto negro, cajas con borde fino en vez de relleno de color.

Run desde participant/:
    python scripts/generate_printable_pdf.py
"""

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
import report_content as C

PAGE_W, PAGE_H = 8.5, 11.0
BLACK = "#000000"
GRAY = "#555555"
LIGHT_GRAY = "#888888"

VERDICT_TAG = {
    "great": "[FUNCIONO]", "ok": "[OK]", "mixed": "[MIXTO]", "bad": "[NO FUNCIONO]",
}

HEADER_TOP = 0.97
CONTENT_START = HEADER_TOP - 0.10
CONTENT_START_WITH_TAG = HEADER_TOP - 0.135


class PW:
    """PageWriter monocromo: mismo patron que el PDF a color, sin relleno."""

    def __init__(self, ax, y_start=CONTENT_START, x_margin=0.07):
        self.ax = ax
        self.y = y_start
        self.x = x_margin
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def skip(self, dy=0.02):
        self.y -= dy

    def write(self, text, fontsize=11, bold=False, italic=False, indent=0,
              dy_after=0.032, x=None, wrap=None, color=BLACK):
        xi = (x if x is not None else self.x) + indent * 0.03
        style = 'italic' if italic else 'normal'
        lines = textwrap.wrap(text, wrap) if wrap else [text]
        for line in lines:
            self.ax.text(xi, self.y, line, transform=self.ax.transAxes, fontsize=fontsize,
                        fontweight='bold' if bold else 'normal', style=style, color=color, va='top')
            self.y -= dy_after * 0.62
        self.y -= dy_after * 0.38

    def bullet(self, text, fontsize=10, indent=1, dy_after=0.030, wrap=88):
        xi = self.x + indent * 0.025
        for i, line in enumerate(textwrap.wrap(text, wrap)):
            prefix = "-  " if i == 0 else "   "
            self.ax.text(xi, self.y, prefix + line, transform=self.ax.transAxes,
                        fontsize=fontsize, color=BLACK, va='top')
            self.y -= dy_after
        self.y -= 0.006

    def header(self, page_label, title):
        self.ax.text(0.07, HEADER_TOP, page_label, transform=self.ax.transAxes,
                     fontsize=9, color=GRAY, va='top')
        self.ax.text(0.07, HEADER_TOP - 0.024, title, transform=self.ax.transAxes,
                     fontsize=15, color=BLACK, va='top', fontweight='bold')
        self.ax.add_patch(plt.Rectangle((0, HEADER_TOP - 0.075), 1, 0.0015,
                          transform=self.ax.transAxes, facecolor=BLACK, clip_on=False))
        self.y = CONTENT_START

    def footer(self, page_num):
        self.ax.text(0.5, 0.015, f"{C.TEAM} · {C.COURSE} · p. {page_num} (imprimible)",
                     transform=self.ax.transAxes, fontsize=8, color=GRAY, va='bottom', ha='center')

    def box(self, text, label="", dashed=False, fontsize=10.5):
        self.y -= 0.006
        y0 = self.y
        lines = textwrap.wrap(text, 92)
        h = 0.026 * max(len(lines), 1) + 0.024 + (0.020 if label else 0)
        self.ax.add_patch(FancyBboxPatch((self.x, y0 - h), 1 - 2 * self.x, h,
                          transform=self.ax.transAxes, boxstyle="round,pad=0.008",
                          linewidth=1.0, edgecolor=BLACK, facecolor="white",
                          linestyle='dashed' if dashed else 'solid'))
        yy = y0 - 0.016
        if label:
            self.ax.text(self.x + 0.015, yy, label, transform=self.ax.transAxes,
                        fontsize=9, color=BLACK, va='top', fontweight='bold')
            yy -= 0.024
        for line in lines:
            self.ax.text(self.x + 0.015, yy, line, transform=self.ax.transAxes,
                        fontsize=fontsize, color=BLACK, va='top')
            yy -= 0.026
        self.y = y0 - h - 0.026

    def tag(self, verdict, x_right=0.93):
        self.ax.text(x_right, HEADER_TOP - 0.010, VERDICT_TAG[verdict], transform=self.ax.transAxes,
                     fontsize=9, color=BLACK, va='top', ha='right', fontweight='bold')
        self.y = CONTENT_START_WITH_TAG


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


def cover_page(pdf):
    fig, ax = new_page()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0.90), 1, 0.003, facecolor=BLACK))
    ax.text(0.07, 0.85, "UNIVERSIDAD AUSTRAL", fontsize=13, color=BLACK, va='top', fontweight='bold')
    ax.text(0.07, 0.81, C.COURSE, fontsize=12, color=GRAY, va='top')
    ax.text(0.07, 0.68, C.PROJECT_TITLE, fontsize=24, color=BLACK, va='top', fontweight='bold')
    ax.text(0.07, 0.60, "Version imprimible para estudiar — sin colores ni fondos",
            fontsize=12, color=GRAY, va='top', style='italic')
    ax.text(0.07, 0.50, C.TEAM, fontsize=14, color=BLACK, va='top', fontweight='bold')
    ax.text(0.07, 0.465, f"Modelo final: {C.FINAL_MODEL}", fontsize=10, color=GRAY,
            va='top', family='monospace')
    ax.add_patch(plt.Rectangle((0, 0.10), 1, 0.003, facecolor=BLACK))
    pdf.savefig(fig); plt.close(fig)


def mechanics_page(pdf, page_num):
    fig, ax = new_page()
    w = PW(ax)
    w.header("SECCION 1", "Objetivo y mecanica de la competencia")
    w.write(
        "Es una competencia de inversion inmobiliaria simulada. Predecimos el precio "
        "de venta de propiedades en Miami / Sur de Florida, y esa prediccion se usa "
        "para decidir, en una simulacion, si compramos cada propiedad y cuanto ofertamos.",
        fontsize=11, wrap=86, dy_after=0.034,
    )
    w.skip(0.01)
    w.write("Como funciona cada simulacion", fontsize=12.5, bold=True)
    for m in C.MECHANICS:
        w.bullet(m, fontsize=10.3, dy_after=0.034, wrap=84)
    w.skip(0.02)
    w.write("Que significa esto en la practica", fontsize=12.5, bold=True)
    w.write(
        "Si predecimos MUY POR ENCIMA del valor real, compramos propiedades caras y "
        "perdemos plata (perdida real). Si predecimos por debajo, simplemente no "
        "compramos esa propiedad (perdemos la oportunidad, pero no perdemos capital). "
        "Este desbalance explica varias decisiones de diseno del proyecto (Ronda 4).",
        fontsize=10.5, wrap=88, color=GRAY, dy_after=0.030,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def data_page(pdf, page_num):
    fig, ax = new_page()
    w = PW(ax)
    w.header("SECCION 2", "Los datos con los que trabajamos")
    w.write("Dos conjuntos de propiedades", fontsize=12.5, bold=True)
    w.bullet("Train: 11,840 propiedades con precio de venta real conocido — con esto "
             "entrenamos los modelos.", dy_after=0.036, wrap=84)
    w.bullet("Test: 5,038 propiedades sin precio — sobre estas generamos las "
             "predicciones que se usan en la competencia real.", dy_after=0.036, wrap=84)
    w.skip(0.02)
    w.write("Que sabemos de cada propiedad", fontsize=12.5, bold=True)
    w.bullet("Datos estructurados: tamano, habitaciones, banos, antiguedad, ubicacion, "
             "impuestos, escuelas cercanas, HOA, historial de precio.", dy_after=0.036, wrap=84)
    w.bullet("Fotos de la propiedad (usadas en Rondas 3 a 8 via embeddings CLIP).",
             dy_after=0.036, wrap=84)
    w.bullet("Descripcion de texto (usada en el experimento de Ronda 9).", dy_after=0.036, wrap=84)
    w.skip(0.02)
    w.write("Dificultades del dataset", fontsize=12.5, bold=True)
    w.bullet("Faltantes importantes: lotAreaValue (45% de las filas), "
             "last_listing_price (33%).", dy_after=0.036, wrap=84)
    w.bullet("Features \"filtradas\" (leaky): taxAssessedValue y variables de impuestos "
             "se usan con cuidado, no como unica fuente de señal.", dy_after=0.036, wrap=84)
    w.bullet("Menos del 2% de las propiedades son ventas atipicas — y terminaron "
             "siendo el factor mas importante de todo el proyecto (Rondas 3, 4 y 8).",
             dy_after=0.036, wrap=84)
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def round_card(ax, top, height, r):
    bottom = top - height
    ax.add_patch(FancyBboxPatch((0.05, bottom), 0.90, height, boxstyle="round,pad=0.004",
                 transform=ax.transAxes, linewidth=0.9, edgecolor=BLACK, facecolor="white"))
    ax.text(0.07, top - 0.020, f"{r['n']} · {r['date']}", fontsize=7.8, color=GRAY,
            va='top', transform=ax.transAxes)
    ax.text(0.07, top - 0.046, r['name'], fontsize=12, fontweight='bold', color=BLACK,
            va='top', transform=ax.transAxes)
    ax.text(0.93, top - 0.020, VERDICT_TAG[r['verdict']], fontsize=8, color=BLACK,
            ha='right', va='top', fontweight='bold', transform=ax.transAxes)

    paras = r['summary'].split("\n\n")
    note = None
    if len(paras) > 3:
        paras, note = paras[:1], "(ver el detalle completo en la seccion siguiente)"

    y = top - 0.082
    for para in paras:
        for line in textwrap.wrap(para, 112):
            ax.text(0.07, y, line, fontsize=8.2, color=BLACK, va='top', transform=ax.transAxes)
            y -= 0.0185
        y -= 0.008
    if note:
        ax.text(0.07, y, note, fontsize=7.6, color=GRAY, style='italic',
                va='top', transform=ax.transAxes)
        y -= 0.022

    y -= 0.010
    ax.text(0.07, y, r['result'], fontsize=8.4, fontweight='bold', color=BLACK,
            va='top', transform=ax.transAxes)
    y -= 0.028
    if r['lesson']:
        for line in textwrap.wrap("Leccion: " + r['lesson'], 118):
            ax.text(0.07, y, line, fontsize=7.4, color=GRAY, style='italic',
                    va='top', transform=ax.transAxes)
            y -= 0.0165


def rounds_page(pdf, page_num, rounds_subset, page_label):
    fig, ax = new_page()
    w = PW(ax)
    w.header("SECCION 3 · CRONOLOGIA", page_label)
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
    w = PW(ax)
    w.header("SECCION 4 · EN DETALLE", "Ronda 9 — por que el texto no ayudo")

    chart_ax = fig.add_axes([0.10, 0.56, 0.80, 0.28])
    labels = [x[0] for x in C.ROUND9_COMPARISON]
    values = [x[1] for x in C.ROUND9_COMPARISON]
    grays = ["#333333", "#999999", "#666666"]
    bars = chart_ax.bar(labels, values, color=grays, edgecolor='black', linewidth=0.8)
    chart_ax.set_ylabel("Mean ROI (%)", fontsize=10)
    chart_ax.set_title("Mismo modelo, unico cambio: agregar features de texto (LLM)",
                       fontsize=11, color=BLACK, fontweight='bold')
    chart_ax.tick_params(axis='x', labelsize=8.3)
    for b, v in zip(bars, values):
        chart_ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha='center',
                      fontsize=9.5, fontweight='bold')
    chart_ax.spines[['top', 'right']].set_visible(False)
    chart_ax.axhline(0, color='black', linewidth=0.6)

    w.y = 0.50
    w.write("Prueba de proxies para detectar \"distressed\" en el test set", fontsize=12,
            bold=True, dy_after=0.034)
    w.write(f"Ninguno predijo mejor que el azar (tasa base: {C.DISTRESS_PROXY_BASELINE:.2f}%):",
            fontsize=10, wrap=88, color=GRAY, dy_after=0.026)
    for name, prec, rec in C.DISTRESS_PROXY_TEST:
        bar_w = min(prec / 5.0, 1.0) * 0.42
        y = w.y
        w.ax.text(0.10, y, name, fontsize=9.3, va='top', family='monospace')
        w.ax.add_patch(plt.Rectangle((0.42, y - 0.020), bar_w, 0.015,
                       transform=w.ax.transAxes, facecolor="#999999", edgecolor="black", linewidth=0.5))
        w.ax.text(0.42 + bar_w + 0.012, y, f"precision {prec:.1f}%  (recall {rec:.1f}%)",
                  fontsize=8.8, va='top', color=GRAY)
        w.y -= 0.030
    w.skip(0.012)
    w.write(
        "Conclusion: las perdidas catastroficas parecen ser idiosincraticas "
        "(circunstancias del vendedor), no un patron detectable en fotos, tags o "
        "descripciones. Se descarto la correccion automatica basada en proxies.",
        fontsize=10, wrap=88, dy_after=0.028,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def scale_sweep_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PW(ax)
    w.header("SECCION 5 · EN DETALLE", "La calibracion final: barrido de escala")

    chart_ax = fig.add_axes([0.11, 0.55, 0.78, 0.30])
    scales = [x[0] for x in C.SCALE_SWEEP_LOCAL]
    rois = [x[1] for x in C.SCALE_SWEEP_LOCAL]
    chart_ax.plot(scales, rois, '-o', color=BLACK, markersize=4, label='Mean ROI (%)')
    chart_ax.axvline(0.83, color=BLACK, linestyle='--', linewidth=1.3, label='Escala elegida (0.83)')
    chart_ax.axvline(1.00, color=GRAY, linestyle=':', linewidth=1.1, label='Escala original (1.00)')
    chart_ax.set_xlabel("Factor de escala aplicado a la prediccion", fontsize=9.5)
    chart_ax.set_ylabel("Mean ROI (%) — simulacion local", fontsize=9.5)
    chart_ax.set_title("Barrido de escala sobre Ronda 8 (datos reales de subastas pasadas)",
                       fontsize=10.5, color=BLACK, fontweight='bold')
    chart_ax.legend(fontsize=8.5, loc='lower center')
    chart_ax.spines[['top', 'right']].set_visible(False)
    chart_ax.grid(alpha=0.25)

    w.y = 0.475
    w.write(
        "Ojo: en este grafico local, la escala 0.85 muestra un pico un poco mas "
        "alto que 0.83 — pero esta simulacion casera es aproximada, no la fuente "
        "de verdad (ver por que mas abajo).",
        fontsize=9.3, wrap=90, color=GRAY, italic=True, dy_after=0.024,
    )
    w.write(
        "Meseta optima entre escala 0.80 y 0.85: el Mean ROI local sube de 8.26% "
        "(sin escalar) a ~10.1-10.3% (+24% relativo). Validado en el dashboard real "
        "con 3 corridas independientes, cada una con un campo competitivo distinto:",
        fontsize=10, wrap=88, dy_after=0.028,
    )
    w.skip(0.008)
    headers = ["Corrida", "N° modelos", "Escala 0.83", "Escala 0.85", "Ronda 8 (sin escalar)"]
    col_x = [0.10, 0.28, 0.46, 0.64, 0.80]
    y0 = w.y
    for cx, h in zip(col_x, headers):
        w.ax.text(cx, y0, h, fontsize=9, fontweight='bold', color=BLACK, va='top')
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
        fontsize=10, wrap=88, dy_after=0.028,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def kelly_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PW(ax)
    w.header("SECCION 6 · EN DETALLE", "Criterio de Kelly aplicado a la subasta")

    w.write(
        "En criollo: pensa 3 mesas de casino, una por segmento (SF, CONDO, RESTO). "
        "En cada mesa sabemos, por el historial, que tan seguido ganamos (Hit Rate) "
        "y cuanto ganamos o perdemos cuando pasa. El criterio de Kelly usa esos dos "
        "datos para decidir en que mesa conviene apostar mas fuerte y en cual mas "
        "flojo — en vez de apostar siempre lo mismo en las tres.",
        fontsize=10.3, wrap=88, dy_after=0.026,
    )
    w.write(
        "Ojo con un detalle: Kelly clasico asume que perder = perder toda la "
        "apuesta. Ese no es nuestro caso — si nos gana la subasta, no perdemos "
        "nada (el capital queda intacto). Lo adaptamos a lo que realmente puede "
        "pasar: ganar la subasta y pagar de mas, condicionado a que ganamos "
        "(formula: f* = p/L − q/W, con p=Hit Rate, W=ganancia media y L=perdida "
        "media, ambas como fraccion del costo).",
        fontsize=9.3, wrap=92, color=GRAY, dy_after=0.030,
    )

    y0 = w.y
    headers = ["Segmento", "Hit Rate", "Ganancia (W)", "Perdida (L)", "Kelly f*", "Escala"]
    col_x = [0.10, 0.26, 0.40, 0.55, 0.68, 0.82]
    for cx, h in zip(col_x, headers):
        w.ax.text(cx, y0, h, fontsize=9, fontweight='bold', color=BLACK, va='top')
    y0 -= 0.028
    for seg, p, wgan, l, f, scale in C.KELLY_RESULTS:
        vals = [seg, f"{p:.1f}%", f"{wgan:.1f}%", f"{l:.1f}%", f"{f:.2f}", f"{scale}"]
        for cx, v in zip(col_x, vals):
            w.ax.text(cx, y0, v, fontsize=9.3, va='top', family='monospace')
        y0 -= 0.028
    w.y = y0 - 0.020

    w.write("Resultado honesto", fontsize=12, bold=True, dy_after=0.030)
    w.write(
        f"Las 3 mesas salieron casi iguales (Hit Rate 70-75% en las tres) — no hay "
        f"una claramente mejor que las otras. Por eso Kelly sugirio una escala casi "
        f"identica en los 3 segmentos, muy cerca de la plana ya validada (0.83): el "
        f"uso real que le dimos fue de verificacion, no de descubrimiento. Aun asi, "
        f"escalar por segmento en vez de plano mejoro el Mean ROI local de "
        f"{C.KELLY_FLAT_ROI:.2f}% a {C.KELLY_SEGMENT_ROI:.2f}%.",
        fontsize=10.3, wrap=88, dy_after=0.028,
    )
    w.box(
        f"Mean ROI local: escala plana 0.83 = {C.KELLY_FLAT_ROI:.2f}%  ->  "
        f"escala Kelly por segmento = {C.KELLY_SEGMENT_ROI:.2f}%",
        label="RESULTADO",
    )
    m = C.PRACTICE_FINAL_VALIDATION
    w.box(
        f"Validado en el dashboard real de Practice contra el campeon vigente "
        f"(scale083): kelly_segment quedo practicamente empatado "
        f"({m[1][1]:.2f}% vs {m[0][1]:.2f}% Mean ROI, -{m[0][1]-m[1][1]:.2f}pp). "
        f"No lo supera, pero confirma que la escala 0.83 ya es una eleccion solida.",
        label="LECCION", dashed=True,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def edge_page(pdf, page_num):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    w = PW(ax)
    w.header("SECCION 7 · EN DETALLE", "Edge (valor esperado) por segmento x precio")

    w.write(
        "En criollo: Kelly (Seccion 6) separo en 3 mesas por segmento y no "
        "encontro diferencias reales entre ellas. Edge va un paso mas — parte "
        "cada una de esas 3 mesas en otras 3 segun que tan cara predice el "
        "modelo que es la propiedad DENTRO de su propio segmento (barata / "
        "media / cara). Quedan 9 mesas en total.",
        fontsize=10, wrap=90, dy_after=0.024,
    )
    w.write(
        "Formalmente es lo mismo de siempre: si pudieramos repetir la misma "
        "subasta muchas veces, cuanto ganariamos en promedio? "
        "EV = P(ganar) x (valor_real − costo pagado) — el costo solo se paga si "
        "ganamos.",
        fontsize=9.3, wrap=92, color=GRAY, dy_after=0.028,
    )
    w.skip(0.004)

    y0 = w.y
    headers = ["Bucket", "Hit Rate", "EV (% del bid)", "Escala sugerida"]
    col_x = [0.10, 0.34, 0.55, 0.78]
    for cx, h in zip(col_x, headers):
        w.ax.text(cx, y0, h, fontsize=9, fontweight='bold', color=BLACK, va='top')
    y0 -= 0.024
    for bucket, hit, ev, scale in C.EDGE_RESULTS:
        vals = [bucket, f"{hit:.1f}%", f"{ev:.2f}%", f"{scale}"]
        for cx, v in zip(col_x, vals):
            w.ax.text(cx, y0, v, fontsize=8.8, va='top', family='monospace')
        y0 -= 0.0225
    w.y = y0 - 0.018

    w.write("Resultado honesto", fontsize=12, bold=True, dy_after=0.028)
    w.write(
        f"A diferencia de las 3 mesas de Kelly, aca SI hay un patron consistente: "
        f"en las 3 mesas 'caras' (Hit Rate ~90-93%) acertamos mucho mas seguido que "
        f"en las 3 mesas 'baratas' (Hit Rate ~74-79%) — se repite igual en los 3 "
        f"segmentos, no es casualidad. Usamos escalas mas agresivas en las mesas "
        f"caras y mas conservadoras en las baratas, lo que mejoro el Mean ROI local "
        f"de {C.EDGE_FLAT_ROI:.2f}% a {C.EDGE_BUCKET_ROI:.2f}% — mas que Kelly solo.",
        fontsize=10, wrap=90, dy_after=0.026,
    )
    w.box(
        f"Mean ROI local: escala plana 0.83 = {C.EDGE_FLAT_ROI:.2f}%  ->  "
        f"escala por bucket fino = {C.EDGE_BUCKET_ROI:.2f}%",
        label="RESULTADO",
    )
    m = C.PRACTICE_FINAL_VALIDATION
    w.box(
        f"En el dashboard real, comparado contra el campeon vigente, edge_bucket "
        f"quedo por debajo ({m[2][1]:.2f}% vs {m[0][1]:.2f}% Mean ROI, "
        f"-{m[0][1]-m[2][1]:.2f}pp) pese al mejor Hit Rate (85.4%) y Sharpe (2.80) — "
        f"su selectividad (9 props/sim vs 14) le costo volumen. La heterogeneidad "
        f"que encontramos es real, pero no alcanza para superar la escala plana.",
        label="LECCION", dashed=True,
    )
    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def lessons_and_final_page(pdf, page_num):
    fig, ax = new_page()
    w = PW(ax)
    w.header("SECCION 8", "Aprendizajes clave y recomendacion final")

    for i, (title, body) in enumerate(C.META_LESSONS, start=1):
        w.write(f"{i}. {title}", fontsize=10.8, bold=True, dy_after=0.024)
        w.write(body, fontsize=9, wrap=98, color=GRAY, dy_after=0.020)
        w.skip(0.008)

    w.skip(0.006)
    w.ax.add_patch(FancyBboxPatch((0.06, w.y - 0.19), 0.88, 0.19, boxstyle="round,pad=0.010",
                   transform=w.ax.transAxes, linewidth=1.2, edgecolor=BLACK, facecolor="white"))
    y0 = w.y - 0.022
    w.ax.text(0.09, y0, "MODELO ELEGIDO PARA LA PRESENTACION FINAL", fontsize=9.5, color=GRAY,
             va='top', fontweight='bold', transform=w.ax.transAxes)
    y0 -= 0.032
    w.ax.text(0.09, y0, C.FINAL_MODEL, fontsize=14, color=BLACK, va='top',
             fontweight='bold', family='monospace', transform=w.ax.transAxes)
    y0 -= 0.040
    for line in textwrap.wrap(C.FINAL_RECOMMENDATION.split("\n\n")[1], 96):
        w.ax.text(0.09, y0, line, fontsize=9.3, color=BLACK, va='top', transform=w.ax.transAxes)
        y0 -= 0.024

    w.footer(page_num)
    pdf.savefig(fig); plt.close(fig)


def main():
    out_dir = Path(__file__).parent.parent / "reports"
    out_path = out_dir / "desarrollo_proyecto_SAM_imprimible.pdf"

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

    print(f"PDF imprimible generado: {out_path}  ({page} paginas + portada = {page + 1} totales)")


if __name__ == "__main__":
    main()
