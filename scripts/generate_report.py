"""
Genera el reporte PDF de progreso del equipo Aragorn.
Run desde la raiz del proyecto: python scripts/generate_report.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import date

OUT = "docs/reporte_progreso_aragorn.pdf"

class PDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 50)
        self.rect(0, 0, 210, 18, 'F')
        self.set_font("helvetica", "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(0, 10, "MDM Austral 2026  |  Labo II: Property Investment Competition  |  Equipo Aragorn",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Pagina {self.page_no()}  |  Generado {date.today().strftime('%d/%m/%Y')}", align="C")

    def section_title(self, text):
        self.set_fill_color(45, 90, 160)
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 12)
        self.cell(0, 9, f"  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, text):
        self.set_font("helvetica", "B", 10)
        self.set_text_color(45, 90, 160)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("helvetica", "", 9)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text, indent=8):
        self.set_font("helvetica", "", 9)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5, f"- {text}")

    def kv(self, key, value, color=(0, 0, 0)):
        self.set_font("helvetica", "B", 9)
        self.set_x(self.l_margin + 8)
        self.cell(52, 5, key)
        self.set_font("helvetica", "", 9)
        self.set_text_color(*color)
        self.cell(0, 5, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        self.set_fill_color(220, 230, 245)
        self.set_font("helvetica", "B", 8)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("helvetica", "", 8)
        for i, row in enumerate(rows):
            self.set_fill_color(248, 248, 255)
            for val, w in zip(row, col_widths):
                self.cell(w, 6, str(val), border=1, fill=(i % 2 == 0), align="C")
            self.ln()
        self.ln(3)


pdf = PDF()
pdf.set_margins(15, 22, 15)
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ── TITULO ─────────────────────────────────────────────────────────────────────
pdf.set_font("helvetica", "B", 22)
pdf.set_text_color(30, 30, 50)
pdf.ln(8)
pdf.cell(0, 14, "Reporte de Progreso", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.set_font("helvetica", "", 13)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "Equipo Aragorn  |  Competencia de Inversion Inmobiliaria Miami",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.set_font("helvetica", "I", 9)
pdf.cell(0, 6, f"Fecha: {date.today().strftime('%d/%m/%Y')}",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.set_text_color(0, 0, 0)
pdf.ln(6)

# Recuadro objetivo
pdf.set_fill_color(240, 245, 255)
pdf.set_draw_color(45, 90, 160)
pdf.set_line_width(0.5)
box_y = pdf.get_y()
pdf.rect(15, box_y, 180, 28, 'DF')
pdf.set_xy(20, box_y + 4)
pdf.set_font("helvetica", "B", 10)
pdf.set_text_color(30, 30, 50)
pdf.cell(0, 6, "Objetivo de la competencia", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("helvetica", "", 9)
pdf.set_text_color(50, 50, 50)
pdf.set_x(20)
pdf.multi_cell(170, 5,
    "Predecir el precio de venta de propiedades residenciales en Miami / Sur de Florida. "
    "Las predicciones se evaluan mediante una simulacion Monte Carlo de inversion inmobiliaria "
    "donde el modelo mas preciso genera mayor ROI.")
pdf.set_text_color(0, 0, 0)
pdf.ln(10)

# ── SECCION 1: ENTORNO ─────────────────────────────────────────────────────────
pdf.section_title("1. Estado del entorno")
pdf.kv("Python:", "3.14.5")
pdf.kv("LightGBM:", "4.6.0")
pdf.kv("pandas:", "3.0.3")
pdf.kv("numpy:", "2.4.6")
pdf.ln(2)
pdf.sub_title("Dataset disponible")
pdf.bullet("train_processed.csv  -- 11,840 filas (features + target)")
pdf.bullet("test_processed.csv   --  5,038 filas (sin target)")
pdf.bullet("train_photo_metadata.csv  +  test_photo_metadata.csv")
pdf.bullet("~199,000 fotos JPG de propiedades (data/images/props/)")
pdf.ln(3)

# ── SECCION 2: MODELOS ─────────────────────────────────────────────────────────
pdf.section_title("2. Modelos entrenados")

pdf.sub_title("Ronda 1 -- Baseline LightGBM  (models/baseline_lgbm.py)")
pdf.body(
    "LightGBM con cross-validation 5-fold (random_state=42). Entrena sobre el espacio "
    "completo de features tabulares sin segmentacion. Incluye imputacion de missing values "
    "por mediana dentro de homeType y early stopping en 50 rondas."
)
pdf.kv("OOF MAPE:", "26.51%", color=(180, 60, 0))
pdf.kv("OOF MAE:", "$119,372")
pdf.kv("Test submission:", "submissions/baseline_lgbm.csv  (5,038 filas)")
pdf.kv("OOF Practice:", "submissions/oof_baseline_lgbm.csv  (11,840 filas)")
pdf.ln(4)

pdf.sub_title("Ronda 2 -- Segmented LightGBM  (models/round2_segments.py)")
pdf.body(
    "Extiende el baseline con feature engineering avanzado y modelos separados por segmento "
    "de propiedad (SINGLE_FAMILY / CONDO / REST). Los features de ZIP (basados en el target) "
    "se calculan dentro de cada fold para evitar leakage."
)
pdf.body("Features nuevos incorporados:")
pdf.bullet("Ratios financieros: tax_per_sqft, listing_to_tax_ratio, tax_to_area, hoa_to_area")
pdf.bullet("Scores compuestos: luxury_score (pool+waterfront+garage+HOA), distress_score (foreclosure+price_cut)")
pdf.bullet("Interacciones: school_x_area, waterfront_x_lat, age_x_school, beds_x_baths")
pdf.bullet("Agregaciones ZIP dentro del fold: zip_median_log_price, zip_std_log_price, zip_count, zip_median_area, zip_price_per_sqft_median")
pdf.ln(2)
pdf.kv("OOF MAPE global:", "27.06%", color=(180, 60, 0))
pdf.kv("OOF MAE global:", "$121,714")
pdf.kv("MAPE por segmento:", "SF=28.61%  |  CONDO=26.10%  |  REST=25.68%")
pdf.kv("Test submission:", "submissions/round2_segments.csv  (5,038 filas)")
pdf.kv("OOF Practice:", "submissions/oof_round2_segments.csv  (11,840 filas)")
pdf.ln(3)

# ── SECCION 3: RESULTADOS DASHBOARD ───────────────────────────────────────────
pdf.section_title("3. Resultados Practice Simulation (dashboard)")
pdf.body(
    "Ambos modelos fueron subidos al tab Practice del dashboard. La simulacion corre "
    "1,000 iteraciones Monte Carlo con 4 rondas internas de 250 propiedades cada una "
    "(4,000 observaciones round-level en total)."
)

headers = ["Modelo", "wMAPE (%)", "Mean ROI (%)", "Median ROI (%)", "Std ROI (%)", "Sharpe", "Hit Rate (%)"]
rows = [
    ["baseline (unlabeled)", "21.35", "23.27", "20.91", "20.24", "1.150", "68.62"],
    ["round2_segments",      "21.77", "20.63", "18.16", "19.42", "1.063", "66.99"],
]
pdf.table(headers, rows, col_widths=[42, 22, 24, 26, 22, 20, 24])

pdf.sub_title("Interpretacion de resultados")
pdf.bullet("El baseline supera al round2 en ROI (+2.6pp) y Sharpe (1.15 vs 1.06).")
pdf.bullet("El round2 tiene ligeramente menor Std ROI, pero su media es inferior.")
pdf.bullet("El feature engineering de Ronda 2 no mejoro la simulacion -- posible sobreajuste en REST (solo 2,217 props).")
pdf.bullet("La metrica clave es ROI de simulacion, no MAPE: un modelo puede tener mejor MAPE y peor ROI si sus errores estan mal calibrados para la estrategia de compra.")
pdf.ln(3)

# ── SECCION 4: BUGS CORREGIDOS ────────────────────────────────────────────────
pdf.section_title("4. Bugs encontrados y corregidos")

pdf.sub_title("Bug 1 -- Categorias desalineadas en prediccion por segmento")
pdf.body(
    "Al filtrar el test set por segmento (SF/CONDO/REST), las columnas categoricas "
    "(homeType, zipcode) tenian niveles distintos a los del fold de entrenamiento, "
    "causando ValueError en LightGBM al predecir."
)
pdf.body(
    "Fix: alinear categorias de train y test una vez tras el preprocesamiento usando "
    ".cat.set_categories() con la union de ambos sets."
)
pdf.ln(2)

pdf.sub_title("Bug 2 -- Indices OOF incorrectos tras merge de ZIP features")
pdf.body(
    "La funcion add_zip_features() realiza un merge que resetea el indice del DataFrame. "
    "Al guardar predicciones OOF con val_seg.index, se escribia en posiciones erroneas, "
    "produciendo un OOF MAPE global de 98.57% (vs ~27% real por fold)."
)
pdf.body(
    "Fix: guardar los indices originales en columna _orig_idx antes del merge y usarlos "
    "para indexar oof_preds[]."
)
pdf.ln(3)

# ── SECCION 5: ROADMAP ────────────────────────────────────────────────────────
pdf.section_title("5. Roadmap de modelos")

headers2 = ["Ronda", "Archivo", "Estado", "Descripcion"]
rows2 = [
    ["1", "baseline_lgbm.py",   "Listo",    "LightGBM tabular completo, CV 5-fold"],
    ["2", "round2_segments.py", "Listo",    "Feature eng. + modelos por segmento SF/CONDO/REST"],
    ["3", "--",                 "Pendiente","Embeddings de imagenes (CLIP o ResNet, foto principal)"],
    ["4", "--",                 "Pendiente","Ensemble completo + calibracion de predicciones"],
]
pdf.table(headers2, rows2, col_widths=[18, 50, 26, 96])

pdf.sub_title("Proximos pasos prioritarios")
pdf.bullet("Analizar drill-down del dashboard: identificar propiedades donde el modelo falla sistematicamente.")
pdf.bullet("Implementar embeddings de imagenes con CLIP (base en scripts/02_image_embeddings.py).")
pdf.bullet("Evaluar calibracion: el modelo subestima precios altos (max predicho $1.56M vs rango real hasta $1.99M).")
pdf.bullet("Investigar por que round2 pierde vs baseline: posible sobreajuste en segmento REST.")
pdf.ln(3)

# ── SECCION 6: MECANICA ───────────────────────────────────────────────────────
pdf.section_title("6. Mecanica de evaluacion (referencia)")
pdf.body("La simulacion Monte Carlo replica el mercado inmobiliario con los siguientes parametros:")
pdf.bullet("asking_price = true_value x (1 + Normal(-0.07, 0.35))")
pdf.bullet("Compramos si: prediccion > asking_price x 1.08")
pdf.bullet("Nuestra oferta en subasta: prediccion x 0.85")
pdf.bullet("Subasta Vickrey: gana el mayor postor, paga el segundo precio")
pdf.ln(2)
pdf.body(
    "Capital inicial: $5M por ronda. 4 rondas internas por simulacion. "
    "1,000 simulaciones Monte Carlo. Ganador: equipo con mayor win rate (% de simulaciones en 1er lugar)."
)

pdf.output(OUT)
print(f"PDF generado: {OUT}")
