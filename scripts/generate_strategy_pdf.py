"""
Genera PDF de estrategia y tecnología — Equipo SAM
MDM Austral 2026 - Labo 2: Property Investment Competition

Run desde participant/:
    python scripts/generate_strategy_pdf.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Paleta California Republic ───────────────────────────────────────────────
C_RED   = '#C8102E'
C_BLUE  = '#003087'
C_WHITE = '#FFFFFF'
C_LIGHT = '#F0F4F8'
C_DARK  = '#1A1A2E'
C_GRAY  = '#6B7280'
C_GREEN = '#16A34A'
C_GOLD  = '#D97706'

PAGE_W = 8.5
PAGE_H = 11.0


# ─── Clase helper para escribir texto con tracking de posición ────────────────
class PageWriter:
    """Escribe texto en una figura matplotlib con avance automático de Y."""

    def __init__(self, ax, y_start=0.75, x_margin=0.05):
        self.ax = ax
        self.y = y_start
        self.x = x_margin
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def skip(self, dy=0.018):
        self.y -= dy

    def write(self, text, fontsize=10.5, color=C_DARK, bold=False,
              indent=0, dy_after=0.032, x=None):
        xi = (x if x is not None else self.x) + indent * 0.03
        self.ax.text(xi, self.y, text,
                     transform=self.ax.transAxes,
                     fontsize=fontsize, fontweight='bold' if bold else 'normal',
                     color=color, va='top')
        self.y -= dy_after

    def bullet(self, text, fontsize=10, color=C_DARK, indent=1, dy_after=0.036):
        xi = self.x + indent * 0.025
        self.ax.text(xi, self.y, "•  " + text,
                     transform=self.ax.transAxes,
                     fontsize=fontsize, color=color, va='top')
        self.y -= dy_after

    def section(self, text, dy_before=0.01, dy_after=0.038):
        self.y -= dy_before
        self.ax.text(self.x, self.y, text,
                     transform=self.ax.transAxes,
                     fontsize=12, fontweight='bold', color=C_BLUE, va='top')
        self.y -= 0.018
        self.ax.plot([self.x, 0.95], [self.y, self.y],
                     transform=self.ax.transAxes,
                     color=C_BLUE, linewidth=1.0, alpha=0.4)
        self.y -= dy_after

    def kv_row(self, key, value, key_w=0.26, fontsize=10,
               key_color=C_BLUE, val_color=C_DARK, dy_after=0.034,
               bg=None):
        """Una fila clave: valor con fondo opcional."""
        if bg:
            rect = FancyBboxPatch((self.x - 0.01, self.y - 0.027), 0.92, 0.030,
                                   boxstyle="square,pad=0",
                                   linewidth=0, facecolor=bg)
            self.ax.add_patch(rect)
        self.ax.text(self.x, self.y, key,
                     transform=self.ax.transAxes,
                     fontsize=fontsize, fontweight='bold',
                     color=key_color, va='top')
        self.ax.text(self.x + key_w, self.y, value,
                     transform=self.ax.transAxes,
                     fontsize=fontsize, color=val_color, va='top')
        self.y -= dy_after

    def box_start(self, color=C_BLUE, height=0.08, label=None):
        """Dibuja una caja de color con label opcional. Retorna la rect."""
        rect = FancyBboxPatch((self.x - 0.01, self.y - height), 0.92, height + 0.005,
                               boxstyle="round,pad=0.008",
                               linewidth=1.5, edgecolor=color,
                               facecolor=color + '15')
        self.ax.add_patch(rect)
        if label:
            self.ax.text(self.x + 0.01, self.y - 0.005, label,
                         transform=self.ax.transAxes,
                         fontsize=10.5, fontweight='bold', color=color, va='top')
            self.y -= 0.032


def draw_header(ax, title, subtitle, page_n, total):
    """
    Header rojo fijo en la parte superior.
    Diseñado para axes de altura 0.16 (16% del figure).
    Coordenadas en transAxes (0-1 dentro del axes).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Barra roja — cubre todo el axes
    rect = FancyBboxPatch((0, 0), 1, 1,
                           boxstyle="square,pad=0", linewidth=0,
                           facecolor=C_RED, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)

    # Título grande — centrado verticalmente en la mitad superior
    ax.text(0.04, 0.78, title,
            transform=ax.transAxes,
            fontsize=18, fontweight='bold', color=C_WHITE, va='center')

    # Subtítulo — mitad inferior de la barra
    ax.text(0.04, 0.40, subtitle,
            transform=ax.transAxes,
            fontsize=9.5, color='#FFD0D0', va='center')

    # Número de página — arriba a la derecha
    ax.text(0.96, 0.78, f"{page_n} / {total}",
            transform=ax.transAxes,
            fontsize=10, color='#FFD0D0', va='center', ha='right')

    # Línea azul en el borde inferior
    ax.axhline(y=0.08, color=C_BLUE, linewidth=2.5, xmin=0, xmax=1)


# ─── PÁGINA 1: Portada ────────────────────────────────────────────────────────
def page_portada(pdf):
    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(C_WHITE)

    # Bloque rojo superior
    rect = FancyBboxPatch((0, 0.70), 1, 0.30,
                           boxstyle="square,pad=0", linewidth=0, facecolor=C_RED)
    ax.add_patch(rect)
    ax.text(0.5, 0.885, "Equipo SAM",
            ha='center', fontsize=34, fontweight='bold', color=C_WHITE)
    ax.text(0.5, 0.820, "MDM Austral 2026  —  Labo 2",
            ha='center', fontsize=14, color='#FFD0D0')
    ax.text(0.5, 0.765, "Property Investment Competition · Miami / Sur de Florida",
            ha='center', fontsize=11, color='#FFAAAA')

    ax.plot([0.06, 0.94], [0.700, 0.700], color=C_BLUE, linewidth=4)

    ax.text(0.5, 0.645, "Estrategia, Algoritmos y Tecnologia",
            ha='center', fontsize=22, fontweight='bold', color=C_BLUE)
    ax.text(0.5, 0.600, "Documento tecnico de prediccion de precios residenciales",
            ha='center', fontsize=12, color=C_GRAY)

    # Caja de metricas
    rect2 = FancyBboxPatch((0.06, 0.30), 0.88, 0.24,
                            boxstyle="round,pad=0.01",
                            linewidth=1.5, edgecolor=C_BLUE, facecolor=C_LIGHT)
    ax.add_patch(rect2)
    ax.text(0.5, 0.515, "Resultados en simulacion local  (1,000 corridas Monte Carlo)",
            ha='center', fontsize=10, color=C_GRAY, fontstyle='italic')

    metricas = [
        ("34.3%", "Win Rate\n1er lugar"),
        ("7.1%",  "ROI Medio\ncompetencia real"),
        ("44.5%", "ROI Practice\nsin rivales reales"),
        ("98.4%", "Prob. Resultado\nPositivo"),
    ]
    for i, (val, lbl) in enumerate(metricas):
        xc = 0.13 + i * 0.215
        ax.text(xc, 0.465, val, ha='center',
                fontsize=18, fontweight='bold', color=C_RED)
        ax.text(xc, 0.385, lbl, ha='center',
                fontsize=8.5, color=C_DARK)

    ax.text(0.5, 0.18, "Competencia de prediccion de precios residenciales",
            ha='center', fontsize=11, color=C_DARK)
    ax.text(0.5, 0.13, "Dataset: 11,840 propiedades de entrenamiento  ·  5,038 propiedades de test",
            ha='center', fontsize=10, color=C_GRAY)
    ax.text(0.5, 0.08, "Precio objetivo: lastSoldPrice_hpi_adjusted  ·  Rango: $51K – $1.99M  ·  Media: $559K",
            ha='center', fontsize=10, color=C_GRAY)
    ax.text(0.5, 0.03, "26 de junio de 2026",
            ha='center', fontsize=10, color=C_GRAY)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── PÁGINA 2: Mecánica de la competencia ────────────────────────────────────
def page_mecanica(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(C_WHITE)
    ax_h = fig.add_axes([0, 0.84, 1, 0.16])
    draw_header(ax_h, "La Competencia",
                "Mecanica de la simulacion y objetivo de optimizacion", 2, 7)
    ax = fig.add_axes([0, 0, 1, 0.84])
    w = PageWriter(ax, y_start=0.97)

    w.section("Que es la competencia")
    w.bullet("Predecimos el precio de venta de 5,038 propiedades en Miami y Sur de Florida.")
    w.bullet("El objetivo NO es minimizar el error de prediccion: es maximizar el ROI de inversion.")
    w.bullet("Las predicciones se evaluan con 1,000 simulaciones Monte Carlo con capital real.")
    w.skip(0.01)

    w.section("Mecanica exacta de la subasta (Vickrey auction)")

    pasos = [
        "1.  El sistema genera un precio pedido aleatorio:",
        "         asking = precio_real × ( 1 + Normal(−7%, 35%) )",
        "2.  Compramos si nuestra prediccion supera el precio pedido por un margen:",
        "         compramos  si  prediccion > asking × 1.08",
        "3.  Hacemos una oferta proporcional a nuestra prediccion:",
        "         oferta (bid) = prediccion × 0.85",
        "4.  Subasta Vickrey: gana el equipo con la oferta mas alta,",
        "         pero paga el SEGUNDO precio mas alto (no el propio bid).",
        "5.  Capital: $5,000,000 por ronda  ·  4 rondas por simulacion  ·  1,000 simulaciones.",
    ]
    for p in pasos:
        bold = p[0].isdigit()
        color = C_BLUE if bold else C_DARK
        w.write(p, fontsize=10, color=color, bold=bold,
                indent=0 if bold else 1, dy_after=0.028)
    w.skip(0.01)

    w.section("El problema central: el error es asimetrico")

    # Caja roja — sobreestimar
    w.write("SOBREESTIMAR (peligroso):", fontsize=10.5, color=C_RED,
            bold=True, dy_after=0.026)
    w.write("Predecimos $600K para una propiedad que vale $100K.",
            fontsize=10, indent=1, dy_after=0.026)
    w.write("Compramos, pagamos ~$500K, recuperamos $100K  →  perdemos $400K de capital.",
            fontsize=10, color=C_RED, indent=1, dy_after=0.034)

    # Caja verde — subestimar
    w.write("SUBESTIMAR (aceptable):", fontsize=10.5, color=C_GREEN,
            bold=True, dy_after=0.026)
    w.write("Predecimos $300K para una propiedad que vale $500K.",
            fontsize=10, indent=1, dy_after=0.026)
    w.write("No compramos  →  perdemos la oportunidad, pero NO perdemos capital.",
            fontsize=10, color=C_GREEN, indent=1, dy_after=0.034)

    w.section("Metrica de evaluacion")
    w.write("Ganador = equipo con mayor Win Rate (% de simulaciones en 1er lugar).",
            fontsize=10.5, dy_after=0.028)
    w.write("No alcanza con tener el ROI promedio mas alto: hay que ganarlo mas veces.",
            fontsize=10.5, color=C_GRAY, dy_after=0.028)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── PÁGINA 3: Estrategia ────────────────────────────────────────────────────
def page_estrategia(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(C_WHITE)
    ax_h = fig.add_axes([0, 0.84, 1, 0.16])
    draw_header(ax_h, "Estrategia",
                "Como pensamos ganar la competencia", 3, 7)
    ax = fig.add_axes([0, 0, 1, 0.84])
    w = PageWriter(ax, y_start=0.97)

    w.write('"No queremos ser los que mas compran. Queremos ser los que mejor compran."',
            fontsize=11, color=C_BLUE, dy_after=0.040)

    w.section("PASO 1 — Modelo Conservador (activo)")
    pasos1 = [
        "Usamos quantile regression con alpha = 0.35.",
        "Un modelo normal predice el PROMEDIO. El nuestro predice intencionalmente MAS BAJO.",
        "Analogia: le preguntamos a 10 expertos cuanto vale una casa. 3 dicen $300K,",
        "  4 dicen $500K, 3 dicen $700K. El promedio da $500K. Nosotros usamos ~$350K.",
        "Efecto: solo compramos cuando el precio pedido es MUY bajo — los deals mas obvios.",
        "En esta competencia, comprar mal destruye capital. No comprar solo es una oportunidad perdida.",
        "Resultado: wMAPE sube ~1%, pero ROI pasa de 21% a 44.5% en Practice.",
    ]
    for p in pasos1:
        indent = 2 if p.startswith("  ") else 1
        w.bullet(p.strip(), indent=indent, dy_after=0.030)
    w.skip(0.01)

    w.section("PASO 2 — Diferenciacion por Imagenes (pendiente segun resultados)")
    pasos2 = [
        "Frodo (rival principal, 29.3% win rate) tiene predicciones casi identicas a SAM (ratio 1.02x).",
        "Para diferenciarnos: incorporar embeddings de imagenes con ResNet50 o CLIP.",
        "Las fotos de propiedades contienen informacion que los datos tabulares no capturan.",
        "Solo se implementa si Paso 1 no es suficiente para ganar la ronda real.",
    ]
    for p in pasos2:
        w.bullet(p, dy_after=0.032)
    w.skip(0.01)

    w.section("Por que funciona contra los competidores")

    rivales = [
        ("Legolas y Merry", C_RED,
         "Bids 2.6x el precio real en distressed sales.",
         "Resultado: pierden -14% de ROI. Se destruyen solos sin que hagamos nada."),
        ("Aragorn y Pippin", C_GOLD,
         "Mismo modelo (bids exactamente iguales en cada propiedad).",
         "Compran mucho (45 props/sim) con hit rate bajo (52%). Win rate: 18% combinado."),
        ("Frodo", C_BLUE,
         "Nuestro rival principal. Ratio 1.02x nuestro modelo.",
         "Participa en el 85% de las oportunidades de SAM. Win rate: 29.3%."),
        ("Gimli", C_GREEN,
         "Conservador similar a SAM pero compra menos props.",
         "Tercero proyectado con 18.6% de win rate."),
    ]

    for nombre, color, linea1, linea2 in rivales:
        w.write(nombre + ":", fontsize=10.5, bold=True, color=color, dy_after=0.026)
        w.write(linea1, fontsize=10, indent=1, dy_after=0.024)
        w.write(linea2, fontsize=10, color=C_GRAY, indent=1, dy_after=0.034)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── PÁGINA 4: Algoritmos ────────────────────────────────────────────────────
def page_algoritmos(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(C_WHITE)
    ax_h = fig.add_axes([0, 0.84, 1, 0.16])
    draw_header(ax_h, "Algoritmos",
                "LightGBM + Quantile Regression por segmento", 4, 7)
    ax = fig.add_axes([0, 0, 1, 0.84])
    w = PageWriter(ax, y_start=0.97)

    w.section("LightGBM — Gradient Boosting sobre arboles de decision")
    lgbm_lines = [
        "Entrenamos un conjunto de miles de arboles de decision en forma secuencial (boosting).",
        "Cada arbol corrige los errores del arbol anterior.",
        "Ventajas: maneja features mixtos (numericos + categoricos), robusto con valores faltantes,",
        "  muy rapido de entrenar, soporta funciones de loss personalizadas (quantile).",
        "Hiperparametros clave: 1,500 arboles, learning_rate = 0.03, num_leaves = 127,",
        "  regularizacion L1 = 0.1, L2 = 1.5, submuestra de datos 80% y features 80%.",
    ]
    for l in lgbm_lines:
        w.write(l, fontsize=10, dy_after=0.028, indent=0 if not l.startswith("  ") else 1)
    w.skip(0.01)

    w.section("Quantile Regression (alpha = 0.35)")
    w.write("La funcion de loss penaliza sobreestimar mas que subestimar:",
            fontsize=10, dy_after=0.026)
    w.write("Loss = 0.35 × max(y − pred, 0)  +  0.65 × max(pred − y, 0)",
            fontsize=11, color=C_RED, bold=True, dy_after=0.026)
    w.write("Sobreestimar (pred > y):  penalidad = 0.65 por cada unidad de error.",
            fontsize=10, color=C_DARK, dy_after=0.022)
    w.write("Subestimar   (pred < y):  penalidad = 0.35 por cada unidad de error.",
            fontsize=10, color=C_DARK, dy_after=0.022)
    w.write("Resultado: el modelo aprende a predecir por debajo del promedio para",
            fontsize=10, color=C_BLUE, dy_after=0.020)
    w.write("  minimizar las penalidades altas de sobreestimar.",
            fontsize=10, color=C_BLUE, dy_after=0.034)

    w.section("Modelos por segmento de propiedad")
    segs = [
        ("SINGLE_FAMILY  (41%)",
         "Casas unifamiliares. Drivers: superficie del lote, calidad de escuelas, barrio."),
        ("CONDO          (40%)",
         "Departamentos. Drivers: HOA fees, amenities, piso, proximidad al agua."),
        ("REST           (19%)",
         "Townhouse, multifamily, otros. Modelo mas generalista con mas regularizacion."),
    ]
    for seg, desc in segs:
        w.write(seg, fontsize=10.5, bold=True, color=C_BLUE, dy_after=0.022)
        w.write(desc, fontsize=10, color=C_DARK, indent=1, dy_after=0.032)
    w.skip(0.005)

    w.section("Cross-validation 5-Fold — como prevenimos overfitting")
    cv_lines = [
        "Dividimos el dataset en 5 partes iguales.",
        "Entrenamos en 4 partes, predecimos en la 5ta. Repetimos 5 veces.",
        "Los features de ZIP (basados en precios) se calculan DENTRO de cada fold: sin data leakage.",
        "Las predicciones Out-of-Fold (OOF) se suben al tab Practice del dashboard para validar",
        "  el modelo antes de gastar una ronda real de competencia.",
    ]
    for l in cv_lines:
        w.write(l, fontsize=10, dy_after=0.028, indent=0 if not l.startswith("  ") else 1)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── PÁGINA 5: Feature Engineering ──────────────────────────────────────────
def page_features(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(C_WHITE)
    ax_h = fig.add_axes([0, 0.84, 1, 0.16])
    draw_header(ax_h, "Feature Engineering",
                "Las variables mas importantes del modelo", 5, 7)
    ax = fig.add_axes([0, 0, 1, 0.84])
    w = PageWriter(ax, y_start=0.97)

    w.section("Features nuevas creadas para Round 4")
    nuevas = [
        ("photo_signal", "SHAP +0.152",
         "Cuantas fotos tiene el anuncio comparado con casas similares del mismo tipo.",
         "Anuncios con muchas mas fotos de lo normal suelen ser propiedades mas caras."),
        ("area_per_room", "SHAP +0.079",
         "Metros cuadrados de living area dividido por (dormitorios + banos + 1).",
         "No es lo mismo 80m2 con 2 cuartos que 80m2 con 6."),
        ("renovation_x_age", "SHAP +0.075",
         "Antiguedad de la propiedad multiplicada por mencion de renovacion en la descripcion.",
         "Una casa antigua renovada vale mas que una antigua sin tocar."),
    ]
    for feat, imp, linea1, linea2 in nuevas:
        w.write(feat + "   —   " + imp, fontsize=11, bold=True,
                color=C_RED, dy_after=0.024)
        w.write(linea1, fontsize=10, indent=1, dy_after=0.022)
        w.write(linea2, fontsize=10, color=C_GRAY, indent=1, dy_after=0.032)

    w.section("Otras features clave del dataset")
    otras = [
        ("taxAssessedValue",      "Valuacion fiscal. Alta correlacion con precio real."),
        ("livingArea",            "Superficie habitable. Driver principal del precio."),
        ("zip_median_log_price",  "Precio mediano por ZIP. Captura calidad del barrio."),
        ("tax_per_sqft",          "Impuesto / superficie. Proxy de calidad de zona."),
        ("luxury_score",          "Score compuesto: piscina + frente al agua + garage + HOA alto."),
        ("distress_score",        "Score de alerta: foreclosure + recorte de precio + muchos cambios."),
        ("school_x_area",         "Calidad de escuelas x superficie. Interaccion importante en SF."),
        ("listing_to_tax_ratio",  "Precio de lista / valuacion fiscal. Mediana observada: 1.81x."),
    ]
    for feat, desc in otras:
        w.write(feat, fontsize=10, bold=True, color=C_BLUE, dy_after=0.018)
        w.write(desc, fontsize=10, color=C_DARK, indent=1, dy_after=0.030)

    w.section("El caso critico: distressed sales")
    w.write("En el primer round, 2 propiedades causaron el 52% de todas las perdidas de SAM:",
            fontsize=10.5, dy_after=0.030)
    casos = [
        ("zpid 1006940:", "Prediccion $1.44M  vs  Precio real $252K  →  Perdida: -$905K  (ratio 5.7x)"),
        ("zpid 1006321:", "Prediccion $663K   vs  Precio real $76K   →  Perdida: -$404K  (ratio 8.8x)"),
    ]
    for lbl, val in casos:
        w.write(lbl, fontsize=10.5, bold=True, color=C_RED, dy_after=0.022)
        w.write(val, fontsize=10, color=C_DARK, indent=1, dy_after=0.030)
    w.write("El distress_score y la quantile regression son la respuesta directa a este problema.",
            fontsize=10.5, color=C_BLUE, dy_after=0.020)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── PÁGINA 6: Resultados ────────────────────────────────────────────────────
def page_resultados(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(C_WHITE)
    ax_h = fig.add_axes([0, 0.84, 1, 0.16])
    draw_header(ax_h, "Resultados",
                "Evolucion de modelos y simulacion de competencia", 6, 7)

    # ── Tabla evolución ──
    ax_t = fig.add_axes([0.04, 0.60, 0.92, 0.26])
    ax_t.axis('off')
    ax_t.text(0.0, 1.02, "Evolucion de modelos — dashboard Practice (OOF sobre train set)",
              transform=ax_t.transAxes, fontsize=11, fontweight='bold',
              color=C_BLUE, va='bottom')

    cols  = ["Ronda", "Modelo", "ROI Medio", "Sharpe", "Prob. Positivo", "wMAPE"]
    rows  = [
        ["1", "Baseline LightGBM",  "  —",    "  —",   "  —",     "21.51%"],
        ["2", "Segmentos + FE",     "20.88%",  "1.049",  "88.6%",   "21.77%"],
        ["3", "Distress Fix",       "21.01%",  "1.074",  "87.8%",   "21.66%"],
        ["4", "Quantile alpha=0.35","44.47%",  "2.054",  "99.2%",   "21.81%"],
    ]
    col_x = [0.00, 0.10, 0.46, 0.60, 0.72, 0.87]
    row_h = 0.185

    for r, row in enumerate([cols] + rows):
        is_h = (r == 0)
        is_best = (r == 4)
        bg = C_BLUE if is_h else ('#FFF0F0' if is_best else (C_LIGHT if r % 2 == 0 else C_WHITE))
        rect = mpatches.Rectangle((col_x[0], 1.0 - (r + 1) * row_h), 1.0, row_h,
                                    facecolor=bg,
                                    edgecolor='#D1D5DB', linewidth=0.5,
                                    transform=ax_t.transAxes)
        ax_t.add_patch(rect)
        for c, (val, x) in enumerate(zip(row, col_x)):
            ax_t.text(x + 0.01, 1.0 - (r + 0.5) * row_h, val,
                      transform=ax_t.transAxes,
                      fontsize=9.5 if r > 0 else 9.0,
                      fontweight='bold' if (is_h or is_best) else 'normal',
                      color=C_WHITE if is_h else (C_RED if (is_best and c >= 2) else C_DARK),
                      va='center')

    # ── Gráficos ──
    ax_wr = fig.add_axes([0.04, 0.30, 0.43, 0.27])
    ax_roi = fig.add_axes([0.54, 0.30, 0.43, 0.27])

    teams  = ['SAM','Frodo','Gimli','Pippin','Aragorn','Legolas','Merry']
    wrates = [34.3, 29.3, 18.6, 10.2, 7.6, 0.0, 0.0]
    rois   = [7.13, 6.03, 4.83, 1.98, 1.75, -14.65, -13.09]
    colors = [C_RED,'#4169E1','#16A34A','#9370DB','#6B7280','#D97706','#228B22']

    ax_wr.barh(teams[::-1], wrates[::-1], color=colors[::-1], alpha=0.85, edgecolor='white')
    ax_wr.set_xlabel("Win Rate (%)", fontsize=9)
    ax_wr.set_title("Win Rate — 1,000 simulaciones", fontsize=9.5, fontweight='bold', color=C_BLUE)
    ax_wr.tick_params(labelsize=9)
    for i, (t, v) in enumerate(zip(teams[::-1], wrates[::-1])):
        if v > 0:
            ax_wr.text(v + 0.3, i, f"{v:.1f}%", va='center', fontsize=8.5, fontweight='bold')

    bars = ax_roi.barh(teams[::-1], rois[::-1], color=colors[::-1], alpha=0.85, edgecolor='white')
    ax_roi.axvline(0, color='black', linewidth=1)
    ax_roi.set_xlabel("ROI Medio (%)", fontsize=9)
    ax_roi.set_title("ROI Medio — con rivales reales", fontsize=9.5, fontweight='bold', color=C_BLUE)
    ax_roi.tick_params(labelsize=9)

    # ── Conclusiones ──
    ax_c = fig.add_axes([0.04, 0.03, 0.92, 0.24])
    ax_c.axis('off')
    cw = PageWriter(ax_c, y_start=0.98, x_margin=0.01)
    cw.section("Conclusiones del simulador")
    concls = [
        "SAM lidera con 34.3% de win rate. Frodo es el unico rival real con 29.3%.",
        "El ROI baja de 44.5% (Practice) a 7.1% en la simulacion con rivales reales.",
        "Legolas y Merry pierden -14% de ROI. Sus bids en distressed sales los destruyen.",
        "Calibracion del simulador: 20.2 props/sim predichas vs 18.86 reales (diferencia 7%).",
    ]
    for c_txt in concls:
        cw.bullet(c_txt, fontsize=10, dy_after=0.14)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── PÁGINA 7: Tecnología ────────────────────────────────────────────────────
def page_tecnologia(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor(C_WHITE)
    ax_h = fig.add_axes([0, 0.84, 1, 0.16])
    draw_header(ax_h, "Tecnologia",
                "Stack completo del pipeline de modelado", 7, 7)
    ax = fig.add_axes([0, 0, 1, 0.84])
    w = PageWriter(ax, y_start=0.97)

    tech_sections = [
        ("Lenguaje y entorno", C_BLUE, [
            ("Python 3.11",        "Lenguaje principal de todo el pipeline de modelado."),
            ("Jupyter Notebook",   "Analisis exploratorio (EDA) y visualizaciones interactivas."),
            ("Git / GitHub",       "Control de versiones. Repositorio: martinceriotti/26ROS_Sam."),
        ]),
        ("Machine Learning", C_RED, [
            ("LightGBM 4.x",       "Modelo principal. Gradient Boosting optimizado para velocidad."),
            ("scikit-learn",       "Cross-validation 5-Fold, metricas de evaluacion, preprocessing."),
            ("NumPy / SciPy",      "Operaciones numericas y distribuciones estadisticas (Monte Carlo)."),
        ]),
        ("Datos y procesamiento", C_GREEN, [
            ("pandas",             "Manipulacion de datasets (11,840 train + 5,038 test propiedades)."),
            ("SHAP",               "Interpretabilidad del modelo. Importancia de features por propiedad."),
            ("Pillow / torchvision","(Round 5 pendiente) Procesamiento de ~199,000 fotos JPG."),
        ]),
        ("Visualizacion y reportes", C_GOLD, [
            ("matplotlib",         "Graficos del simulador, distribuciones y reporte PDF (este documento)."),
            ("Power BI Desktop",   "Dashboard interactivo con tema California Republic personalizado."),
            ("python-pptx",        "Presentacion de clase generada 100% programaticamente."),
        ]),
        ("Simulador de competencia", '#7C3AED', [
            ("simulate_competition.py", "Monte Carlo 1,000 sims x 4 rondas. Modela 6 equipos rivales."),
            ("Modelo de rivales",       "946 bids reales parseados. Probabilidad condicional + lognormal."),
            ("Calibracion automatica",  "Factor escala R3 to R4 calculado del test set. Error de calibracion: 7%."),
        ]),
    ]

    for sec_name, color, items in tech_sections:
        # Titulo de sección con barra de color
        w.ax.add_patch(FancyBboxPatch((w.x - 0.01, w.y - 0.027), 0.92, 0.028,
                                       boxstyle="square,pad=0",
                                       linewidth=0, facecolor=color))
        w.ax.text(w.x + 0.01, w.y - 0.003, sec_name.upper(),
                  transform=w.ax.transAxes,
                  fontsize=9.5, fontweight='bold', color=C_WHITE, va='top')
        w.y -= 0.034

        for tech, desc in items:
            w.write(tech, fontsize=10, bold=True, color=color, dy_after=0.018)
            w.write(desc, fontsize=10, color=C_DARK, indent=1, dy_after=0.028)
        w.skip(0.010)

    # Footer
    w.ax.add_patch(FancyBboxPatch((0.03, 0.01), 0.92, 0.055,
                                   boxstyle="round,pad=0.01",
                                   linewidth=1, edgecolor=C_BLUE + '40',
                                   facecolor=C_LIGHT))
    w.ax.text(0.5, 0.048,
              "Pipeline: baseline  →  segmentos  →  distress fix  →  quantile regression  →  (imagenes)",
              ha='center', va='center', transform=w.ax.transAxes,
              fontsize=9, color=C_GRAY)
    w.ax.text(0.5, 0.022,
              "Cada modelo se testea primero en Practice (OOF) antes de usar una ronda real.",
              ha='center', va='center', transform=w.ax.transAxes,
              fontsize=9, color=C_GRAY, fontstyle='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    Path("reports").mkdir(exist_ok=True)
    output = "reports/estrategia_SAM.pdf"

    with PdfPages(output) as pdf:
        page_portada(pdf)
        page_mecanica(pdf)
        page_estrategia(pdf)
        page_algoritmos(pdf)
        page_features(pdf)
        page_resultados(pdf)
        page_tecnologia(pdf)

        d = pdf.infodict()
        d['Title']   = 'Estrategia y Tecnologia — Equipo SAM'
        d['Author']  = 'Equipo SAM — MDM Austral 2026'
        d['Subject'] = 'Labo 2: Property Investment Competition'
        d['Keywords']= 'LightGBM, quantile regression, Monte Carlo, real estate'

    print(f"PDF generado: {output}")


if __name__ == "__main__":
    main()
