from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

from .utils import fmt_usd

def build_pdf(inputs: Dict, res: Dict, best_species: str, conf_pct: float,
              prices_no_energy: Dict[str, float], energy_costs: Dict[str, float], prices_with_energy: Dict[str, float],
              fig_png_bytes: Optional[bytes]) -> bytes:
    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    margin = 0.75 * inch

    def text_line(x, y, s, size=10):
        c.setFont("Helvetica", size); c.drawString(x, y, str(s))

    def maybe_pagebreak(y, needed=100):
        return (y - needed) < margin

    def draw_image(png_bytes, x, y, w_in=3.6, h_in=3.6):
        try:
            c.drawImage(ImageReader(BytesIO(png_bytes)), x, y, width=w_in*inch, height=h_in*inch, preserveAspectRatio=True, anchor='sw')
            return True
        except Exception:
            return False

    # Header
    y = H - margin
    c.setTitle("Industry Standard Gem Valuer Report")
    text_line(margin, y, "Industry Standard Gem Valuer — Report", 14); y -= 18
    text_line(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 9); y -= 14

    # Inputs
    text_line(margin, y, "Inputs:", 12); y -= 12
    for label, val in inputs.items():
        text_line(margin, y, f"• {label}: {val}", 10); y -= 12

    # Physical
    y -= 2
    text_line(margin, y, "Physical Properties:", 12); y -= 12
    text_line(margin, y, f"• Density used: {res['density_cm3']:.6f} g/cm³", 10); y -= 12
    text_line(margin, y, f"• Mass: {res['mass_g']:.6f} g ({res['mass_ct']:.6f} ct)", 10); y -= 12
    text_line(margin, y, f"• Volume: {res['volume_mm3']:.6f} mm³", 10); y -= 14

    # Identification & Value (six prices)
    text_line(margin, y, "Identification & Value:", 12); y -= 12
    text_line(margin, y, f"• Best species estimate: {best_species}", 10); y -= 12
    text_line(margin, y, f"• Confidence (top-5 normalized): {conf_pct:.2f}%", 10); y -= 12

    def row(label, dct):
        return f"{label}: " + ", ".join([f"{r}: {fmt_usd(v)}" for r, v in dct.items()])

    text_line(margin, y, row("Price (no energy)", prices_no_energy), 10); y -= 12
    text_line(margin, y, row("Energy cost", energy_costs), 10); y -= 12
    text_line(margin, y, row("Price + energy", prices_with_energy), 10); y -= 14

    # 3D Snapshot
    if fig_png_bytes:
        if maybe_pagebreak(y, needed=int(3.8*inch)):
            c.showPage(); y = H - margin
        text_line(margin, y, "3D Snapshot:", 12); y -= 12
        ok = draw_image(fig_png_bytes, margin, y - 3.6*inch)
        if ok:
            y -= (3.6*inch + 12)
        else:
            text_line(margin, y, "(Snapshot unavailable.)", 10); y -= 14

    c.showPage()
    c.save()
    return buf.getvalue()
