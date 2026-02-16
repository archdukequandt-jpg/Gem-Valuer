# Industry Standard Gem Valuer v25_mod — 3D (Streamlit App)
# Last updated: 2025-11-15
# Programmer: Ryan Childs, 23, Boston College
# AI Assistance: ChatGPT 5 Thinking
#
# v25_mod Highlights
# ------------------
# - Wide UI (full-width) with sticky cart panel and larger canvas.
# - Bottom-of-page "drop-down" (expander) menus summarizing all major calculations.
# - Hydrostatic SG workflow (sidebar) feeds identification and density (g/cm³).
# - Energy equivalents section: kWh, gasoline (gal), natural gas (MMBtu) with live/manual price controls.
# - Rich PDF export per item with 3D snapshot and "what-if" valuations for other Top-5 candidates.
# - Everything from v23 kept: retail/wholesale presets, live region editors, DB-backed pricing, metals,
#   NaN-free DB browser, BytesIO/ImageReader PDF fix, etc.
#
# UI polish in this version:
# - Top navigation bar with quick links (GIA, shapes, encyclopedia, wiki) + contact info.
# - Clear sectioning: Identification, Database, Resources.
# - Subtle CSS to keep right “Order” panel sticky and page wide.
# - Header tagline retained as requested.

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from io import BytesIO
from datetime import datetime
import math
import numpy as np
import pandas as pd

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    st = None
    plt = None

# Plotly optional
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

# Report/PDF optional
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    REPORTLAB = True
except Exception:
    REPORTLAB = False

# Spot prices optional
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False


# ----------------- Constants -----------------
CARAT_TO_GRAM = 0.2
MM3_TO_CM3 = 1.0 / 1000.0
TROY_OUNCE_TO_GRAM = 31.1034768

# Geometric volume factors (k) ~ V ≈ k * L * W * H
SHAPE_FACTORS: Dict[str, float] = {
    "Diamond": 0.60, "Round": 0.60, "Oval": 0.60, "Princess Cut": 0.68,
    "Emerald Cut": 0.72, "Antique Cushion": 0.65, "Marquise": 0.58,
    "Pear": 0.59, "Trillant": 0.55, "Square": 0.70, "Octagon": 0.70,
    "Baguette": 0.70, "Tapered Baguette": 0.66, "Heart Shape": 0.60,
    "Briolette": 0.50, "Cabochon": 0.52,
}

DEFAULT_DEPTH_RATIO_BY_CUT: Dict[str, float] = {
    "Diamond": 0.61, "Round": 0.61, "Oval": 0.60, "Princess Cut": 0.68, "Emerald Cut": 0.65,
    "Antique Cushion": 0.64, "Marquise": 0.60, "Pear": 0.62, "Trillant": 0.55,
    "Square": 0.68, "Octagon": 0.65, "Baguette": 0.60, "Tapered Baguette": 0.58,
    "Heart Shape": 0.62, "Briolette": 0.55, "Cabochon": 0.52,
}

# ----------------- Compact species catalog -----------------
CATALOG_DENSITY_RANGES: Dict[str, Tuple[float, float]] = {
    "Diamond — Natural": (3.50, 3.54),
    "Diamond — Lab": (3.50, 3.54),
    "Simulant — Moissanite": (3.20, 3.23),
    "Simulant — CZ/YAG/GGG/Glass": (2.35, 6.10),
    "Corundum (Ruby/Sapphire)": (3.95, 4.10),
    "Spinel": (3.58, 3.64),
    "Beryl (Emerald/Aquamarine/Morganite)": (2.67, 2.80),
    "Quartz & Chalcedony": (2.58, 2.70),
    "Garnet (group)": (3.50, 4.30),
    "Tourmaline": (3.00, 3.30),
    "Topaz": (3.49, 3.60),
    "Zircon": (3.90, 4.70),
    "Jade (Jadeite/Nephrite)": (2.90, 3.40),
    "Peridot (Olivine)": (3.27, 3.37),
    "Chrysoberyl (incl. Alexandrite)": (3.68, 3.78),
    "Diopside": (3.22, 3.38),
    "Sphene (Titanite)": (3.50, 3.60),
    "Apatite/Iolite/Fluorite": (2.58, 3.25),
    "Opal": (1.98, 2.25),
    "Opaque Ornamentals (Lapis/Turquoise/Sodalite/Malachite/Rhodochrosite/Hematite/Obsidian)": (2.10, 5.30),
}

# Regional price factors (live-editable defaults live in session_state)
REGION_FACTORS_DEFAULT = {"North America": 1.00, "Western Europe": 1.08, "Asia": 0.97}
REGION_OVERRIDES_DEFAULT: Dict[str, Dict[str, float]] = {
    "Diamond — Lab": {"North America": 0.92, "Western Europe": 0.95, "Asia": 0.85},
    "Simulant — CZ/YAG/GGG/Glass": {"North America": 0.85, "Western Europe": 0.90, "Asia": 0.80},
    "Jade (Jadeite/Nephrite)": {"North America": 0.95, "Western Europe": 1.00, "Asia": 1.40},
    "Opal": {"North America": 1.15, "Western Europe": 1.05, "Asia": 0.95},
    "Corundum (Ruby/Sapphire)": {"North America": 1.00, "Western Europe": 1.05, "Asia": 1.10},
}

# Rarity index
RARITY_INDEX: Dict[str, float] = {
    "Diamond — Natural": 0.85, "Diamond — Lab": 0.35,
    "Simulant — Moissanite": 0.10, "Simulant — CZ/YAG/GGG/Glass": 0.05,
    "Corundum (Ruby/Sapphire)": 0.70, "Spinel": 0.60,
    "Beryl (Emerald/Aquamarine/Morganite)": 0.50, "Quartz & Chalcedony": 0.05,
    "Garnet (group)": 0.35, "Tourmaline": 0.45, "Topaz": 0.20, "Zircon": 0.40,
    "Jade (Jadeite/Nephrite)": 0.70, "Peridot (Olivine)": 0.25, "Chrysoberyl (incl. Alexandrite)": 0.85,
    "Diopside": 0.18, "Sphene (Titanite)": 0.60, "Apatite/Iolite/Fluorite": 0.15,
    "Opal": 0.35, "Opaque Ornamentals (Lapis/Turquoise/Sodalite/Malachite/Rhodochrosite/Hematite/Obsidian)": 0.10,
}

# --------- GEM DATABASE ----------
def _gem_db_rows() -> List[Dict]:
    rows = []
    def add(species, color, beta, qty_ct, nat_lo=None, nat_hi=None, lab_lo=None, lab_hi=None):
        rows.append({
            "species": species, "color": color, "beta": float(beta),
            "quantity_ct": float(qty_ct),
            "natural_low": None if nat_lo is None else float(nat_lo),
            "natural_high": None if nat_hi is None else float(nat_hi),
            "lab_low": None if lab_lo is None else float(lab_lo),
            "lab_high": None if lab_hi is None else float(lab_hi),
        })
    # (DB content abridged for brevity, same structure preserved)
    add("Diamond — Natural", "colorless", 0.35, 5e10, 4000, 9000)
    add("Diamond — Natural", "yellow",    0.35, 1e9,  3000, 6000)
    add("Diamond — Natural", "brown",     0.35, 1e10, 800, 3000)
    add("Diamond — Natural", "pink",      0.35, 3e6,  20000, 200000)
    add("Diamond — Natural", "blue",      0.35, 1e6,  30000, 250000)
    add("Diamond — Lab", "colorless", 0.25, 1e12, None, None, 800, 3000)
    add("Diamond — Lab", "yellow",    0.25, 5e11, None, None, 600, 2000)
    add("Diamond — Lab", "pink",      0.25, 5e10, None, None, 1500, 4000)
    add("Diamond — Lab", "blue",      0.25, 5e10, None, None, 2200, 5200)
    add("Corundum (Ruby/Sapphire)", "red",   0.30, 3e8,  2000, 10000, 60, 200)
    add("Corundum (Ruby/Sapphire)", "blue",  0.30, 1e9,   600, 6000,  40, 160)
    add("Corundum (Ruby/Sapphire)", "pink",  0.30, 7e8,   400, 3000,  30, 120)
    add("Corundum (Ruby/Sapphire)", "yellow",0.30, 9e8,   250, 1500,  25, 100)
    add("Corundum (Ruby/Sapphire)", "padparadscha",0.30, 2e7, 3000, 20000, 80, 200)
    add("Spinel", "red", 0.28, 5e8,  600, 4000)
    add("Spinel", "blue",0.28, 3e8,  400, 2500)
    add("Spinel", "pink",0.28, 6e8,  200, 1500)
    add("Beryl (Emerald/Aquamarine/Morganite)", "green", 0.25, 2e8, 1500, 8000, 70, 200)
    add("Beryl (Emerald/Aquamarine/Morganite)", "blue",  0.25, 1e9,   60, 600)
    add("Beryl (Emerald/Aquamarine/Morganite)", "pink",  0.25, 1.2e9,  80, 400)
    add("Quartz & Chalcedony", "purple", 0.10, 1e12, 4, 15)
    add("Quartz & Chalcedony", "yellow", 0.10, 8e11, 3, 12)
    add("Quartz & Chalcedony", "pink",   0.10, 1.5e12, 2, 8)
    add("Quartz & Chalcedony", "white/opaque", 0.10, 2e12, 1, 5)
    add("Garnet (group)", "red",    0.24, 1e11, 20, 300)
    add("Garnet (group)", "green",  0.27, 8e8,  300, 1000)
    add("Garnet (group)", "orange", 0.26, 3e9,  120, 400)
    add("Tourmaline", "green",   0.22, 8e9,  120, 500)
    add("Tourmaline", "blue",    0.22, 6e8,  400, 1500)
    add("Tourmaline", "pink",    0.22, 1e10, 120, 500)
    add("Tourmaline", "bi-color",0.22, 5e8,  200, 800)
    add("Topaz", "colorless", 0.18, 2e11, 6, 20)
    add("Topaz", "blue",      0.18, 3e11, 8, 30)
    add("Topaz", "imperial",  0.22, 2e8,  150, 800)
    add("Zircon", "blue",     0.22, 1e9,  80, 400)
    add("Zircon", "white",    0.22, 5e9,  50, 200)
    add("Zircon", "red",      0.22, 2e8,  120, 600)
    add("Jade (Jadeite/Nephrite)", "green", 0.18, 5e10, 200, 10000)
    add("Jade (Jadeite/Nephrite)", "lavender", 0.18, 3e9, 80, 3000)
    add("Jade (Jadeite/Nephrite)", "white", 0.18, 8e10, 40, 600)
    add("Peridot (Olivine)", "green", 0.25, 6e10, 20, 150)
    add("Chrysoberyl (incl. Alexandrite)", "yellow", 0.30, 5e9, 200, 1500, 100, 300)
    add("Chrysoberyl (incl. Alexandrite)", "alexandrite (chg)", 0.40, 3e7, 3000, 15000, 150, 600)
    add("Diopside", "green", 0.22, 4e10, 20, 150)
    add("Sphene (Titanite)", "green", 0.30, 5e8, 200, 800)
    add("Apatite/Iolite/Fluorite", "neon blue (apatite)", 0.18, 5e9, 20, 100)
    add("Apatite/Iolite/Fluorite", "violet (iolite)",     0.18, 8e9, 15, 80)
    add("Apatite/Iolite/Fluorite", "green (fluorite)",    0.12, 2e10, 5, 40)
    add("Opal", "black",   0.18, 2e8, 400, 2500, 2, 10)
    add("Opal", "crystal", 0.18, 1e9,  80, 600,   1, 6)
    opa = "Opaque Ornamentals (Lapis/Turquoise/Sodalite/Malachite/Rhodochrosite/Hematite/Obsidian)"
    add(opa, "lapis blue", 0.15, 2e10, 2, 30)
    add(opa, "turquoise blue-green", 0.15, 1e10, 5, 60)
    add(opa, "malachite green", 0.15, 8e9, 5, 50)
    add(opa, "hematite/obsidian", 0.15, 5e10, 1, 20)
    return rows

GEM_DB = pd.DataFrame(_gem_db_rows())

# Fallback species mid $/ct (ensures no NaN pricing)
PRICING_FALLBACKS: Dict[str, Dict[str, Optional[float]]] = {
    "Diamond — Natural": {"natural": 6500.0, "lab": None},
    "Diamond — Lab": {"natural": None, "lab": 1900.0},
    "Simulant — Moissanite": {"natural": None, "lab": 500.0},
    "Simulant — CZ/YAG/GGG/Glass": {"natural": None, "lab": 50.0},
    "Corundum (Ruby/Sapphire)": {"natural": 5100.0, "lab": 110.0},
    "Spinel": {"natural": 1400.0, "lab": 10.0},
    "Beryl (Emerald/Aquamarine/Morganite)": {"natural": 1800.0, "lab": 140.0},
    "Quartz & Chalcedony": {"natural": 6.0, "lab": 3.0},
    "Garnet (group)": {"natural": 300.0, "lab": None},
    "Tourmaline": {"natural": 450.0, "lab": None},
    "Topaz": {"natural": 18.0, "lab": None},
    "Zircon": {"natural": 180.0, "lab": None},
    "Jade (Jadeite/Nephrite)": {"natural": 2400.0, "lab": None},
    "Peridot (Olivine)": {"natural": 85.0, "lab": None},
    "Chrysoberyl (incl. Alexandrite)": {"natural": 4400.0, "lab": 260.0},
    "Diopside": {"natural": 85.0, "lab": None},
    "Sphene (Titanite)": {"natural": 420.0, "lab": None},
    "Apatite/Iolite/Fluorite": {"natural": 45.0, "lab": None},
    "Opal": {"natural": 320.0, "lab": 6.0},
    "Opaque Ornamentals (Lapis/Turquoise/Sodalite/Malachite/Rhodochrosite/Hematite/Obsidian)": {"natural": 22.0, "lab": None},
}

# ----------------- Derive models from DB -----------------
def _mid(lo, hi):
    if lo is None or hi is None: return None
    return 0.5 * (float(lo) + float(hi))

def derive_models_from_db(df: pd.DataFrame):
    species_nat_mid: Dict[str, Optional[float]] = {}
    species_lab_mid: Dict[str, Optional[float]] = {}
    size_exponent: Dict[str, float] = {}

    for species, g in df.groupby("species"):
        nat_mids = [_mid(r["natural_low"], r["natural_high"]) for _, r in g.iterrows() if pd.notna(r["natural_low"])]
        lab_mids = [_mid(r["lab_low"], r["lab_high"]) for _, r in g.iterrows() if pd.notna(r["lab_low"])]
        nat_mids = [x for x in nat_mids if x is not None]
        lab_mids = [x for x in lab_mids if x is not None]

        nat_avg = float(np.mean(nat_mids)) if nat_mids else PRICING_FALLBACKS.get(species, {}).get("natural")
        lab_avg = float(np.mean(lab_mids)) if lab_mids else PRICING_FALLBACKS.get(species, {}).get("lab")
        species_nat_mid[species] = nat_avg
        species_lab_mid[species] = lab_avg

        beta_vals = [float(b) for b in g["beta"].tolist() if pd.notna(b)]
        size_exponent[species] = float(np.mean(beta_vals)) if beta_vals else 0.20

    for s in CATALOG_DENSITY_RANGES.keys():
        species_nat_mid.setdefault(s, PRICING_FALLBACKS.get(s, {}).get("natural"))
        species_lab_mid.setdefault(s, PRICING_FALLBACKS.get(s, {}).get("lab"))
        size_exponent.setdefault(s, 0.20)

    price_table = {s: {"natural": species_nat_mid[s], "lab": species_lab_mid[s]} for s in species_nat_mid.keys()}

    # Color multipliers from DB
    color_mult: Dict[str, Dict[str, float]] = {}
    for species, g in df.groupby("species"):
        ref_nat = species_nat_mid.get(species)
        ref_lab = species_lab_mid.get(species)
        rarity = RARITY_INDEX.get(species, 0.25)
        if ref_nat is None and ref_lab is None:
            ref = 1.0
        elif ref_nat is None:
            ref = ref_lab
        elif ref_lab is None:
            ref = ref_nat
        else:
            ref = rarity * ref_nat + (1 - rarity) * ref_lab
        if ref is None or ref <= 0:
            ref = 1.0
        cmap = {}
        for _, r in g.iterrows():
            mid_nat = _mid(r["natural_low"], r["natural_high"])
            mid_lab = _mid(r["lab_low"], r["lab_high"])
            if (mid_nat is None and mid_lab is None):
                m = 1.0
            elif mid_nat is None:
                m = mid_lab / ref
            elif mid_lab is None:
                m = mid_nat / ref
            else:
                m = (rarity * mid_nat + (1 - rarity) * mid_lab) / ref
            m = float(max(0.1, min(20.0, m)))
            cmap[str(r["color"]).lower()] = m
        color_mult[species] = cmap

    return price_table, color_mult, size_exponent

PRICE_TABLE, COLOR_PRICE_MULTIPLIERS, SIZE_EXPONENT = derive_models_from_db(GEM_DB)
DEFAULT_BETA = 0.20

# Transparency hints
TRANSPARENCY_HINTS: Dict[str, List[str]] = {
    "Diamond — Natural": ["clear"],
    "Diamond — Lab": ["clear"],
    "Simulant — Moissanite": ["clear"],
    "Simulant — CZ/YAG/GGG/Glass": ["clear", "semi-translucent"],
    "Corundum (Ruby/Sapphire)": ["clear", "semi-translucent"],
    "Spinel": ["clear", "semi-translucent"],
    "Beryl (Emerald/Aquamarine/Morganite)": ["clear", "semi-translucent"],
    "Quartz & Chalcedony": ["clear", "semi-translucent", "opaque"],
    "Garnet (group)": ["clear", "semi-translucent"],
    "Tourmaline": ["clear", "semi-translucent"],
    "Topaz": ["clear"],
    "Zircon": ["clear"],
    "Jade (Jadeite/Nephrite)": ["semi-translucent", "opaque"],
    "Peridot (Olivine)": ["clear"],
    "Chrysoberyl (incl. Alexandrite)": ["clear"],
    "Diopside": ["clear"],
    "Sphene (Titanite)": ["clear"],
    "Apatite/Iolite/Fluorite": ["clear", "semi-translucent"],
    "Opal": ["semi-translucent", "opaque"],
    "Opaque Ornamentals (Lapis/Turquoise/Sodalite/Malachite/Rhodochrosite/Hematite/Obsidian)": ["opaque"],
}

# ----------------- Utilities -----------------
def round6(x: float) -> float:
    try:
        return float(np.round(float(x), 6))
    except Exception:
        return x

# ----------------- Structures -----------------
@dataclass
class StoneInput:
    cut: str
    weight_value: float
    weight_unit: str
    diameter_mm: Optional[float] = None
    depth_pct: Optional[float] = None
    crown_angle_deg: Optional[float] = None
    pavilion_angle_deg: Optional[float] = None
    table_pct: Optional[float] = None
    culet_pct: Optional[float] = None
    girdle_pct: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    color: Optional[str] = None
    transparency: Optional[str] = None

# ----------------- Core math & geometry -----------------
def mass_grams(value: float, unit: str) -> float:
    u = unit.lower()
    if u in ("carat", "ct", "carats"):
        return value * CARAT_TO_GRAM
    if u in ("gram", "g", "grams"):
        return value
    raise ValueError("weight unit must be 'carat' or 'gram'")

def grams_to_carats(g: float) -> float:
    return g / CARAT_TO_GRAM

def _normalize_cut(cut: str) -> str:
    for key in SHAPE_FACTORS.keys():
        if key.lower() == cut.lower():
            return key
    raise ValueError(f"Unsupported cut: {cut}")

def volume_mm3_from_dims(cut: str, L: float, W: float, H: float) -> float:
    k = SHAPE_FACTORS[_normalize_cut(cut)]
    return max(0.0, k * L * W * H)

# Angle-based RB profile
def profile_round_brilliant_angles(R: float, H_target: float, table_pct: float, culet_pct: float,
                                   girdle_pct: float, crown_angle_deg: float, pavilion_angle_deg: float,
                                   n_samples: int = 220):
    alpha = math.radians(max(10.0, min(50.0, crown_angle_deg)))
    theta = math.radians(max(35.0, min(45.0, pavilion_angle_deg)))
    r_table = max(0.0, min(1.0, table_pct)) * R
    r_culet = max(0.0, min(1.0, culet_pct)) * R
    z_crown0 = (R - r_table) * math.tan(alpha)
    z_pav0 = (R - r_culet) * math.tan(theta)
    g0 = max(0.0, min(0.2, girdle_pct)) * (z_crown0 + z_pav0)
    H0 = z_crown0 + g0 + z_pav0
    s = H_target / max(H0, 1e-6)
    z_crown = z_crown0 * s
    z_girdle = g0 * s
    z = np.linspace(0.0, H_target, n_samples)
    r = np.empty_like(z)
    tan_alpha_eff = math.tan(alpha) * s
    tan_theta_eff = math.tan(theta) * s
    z1 = z_crown
    z2 = z_crown + z_girdle
    for i, zi in enumerate(z):
        if zi <= z1:
            r[i] = r_table + zi / max(tan_alpha_eff, 1e-9)
        elif zi <= z2:
            r[i] = R
        else:
            r[i] = R - (zi - z2) / max(tan_theta_eff, 1e-9)
    return z, r

def revolve_mesh(z: np.ndarray, r: np.ndarray, L: float, W: float, n_theta: int = 180):
    z = z.reshape(-1); r = r.reshape(-1)
    theta = np.linspace(0.0, 2.0*np.pi, n_theta)
    Z, TH = np.meshgrid(z, theta, indexing='ij')
    Rgrid = np.repeat(r[:, None], n_theta, axis=1)
    Xc = Rgrid * np.cos(TH); Yc = Rgrid * np.sin(TH)
    Rmax = max(float(np.max(r)), 1e-6)
    sx = (L/2.0) / Rmax; sy = (W/2.0) / Rmax
    return Xc * sx, Yc * sy, Z

def superellipse_outline(a: float, b: float, n: float, samples: int = 120):
    t = np.linspace(0.0, 2.0*np.pi, samples)
    cos_t, sin_t = np.cos(t), np.sin(t)
    x = np.sign(cos_t) * (np.abs(cos_t) ** (2.0/max(n, 1e-3))) * a
    y = np.sign(sin_t) * (np.abs(sin_t) ** (2.0/max(n, 1e-3))) * b
    return x, y

def loft_superellipse(L: float, W: float, H: float, crown_frac: float, table_frac: float, culet_frac: float,
                      n_plan: float = 0.8, layers: int = 40, samples: int = 120):
    zs = np.linspace(0.0, H, layers)
    z_crown = crown_frac * H
    z_girdle = z_crown + 0.05 * H
    Rmax_a, Rmax_b = L/2.0, W/2.0
    table_a = table_frac * Rmax_a; table_b = table_frac * Rmax_b
    culet_a = culet_frac * Rmax_a; culet_b = culet_frac * Rmax_b
    Xs, Ys, Zs = [], [], []
    for z in zs:
        if z <= z_crown:
            t = z / max(z_crown, 1e-6)
            a = table_a + (Rmax_a - table_a) * (t**0.7)
            b = table_b + (Rmax_b - table_b) * (t**0.7)
        elif z <= z_girdle:
            a, b = Rmax_a, Rmax_b
        else:
            t = (z - z_girdle) / max(H - z_girdle, 1e-6)
            a = Rmax_a - (Rmax_a - culet_a) * (t**0.7)
            b = Rmax_b - (Rmax_b - culet_b) * (t**0.7)
        x, y = superellipse_outline(a, b, n_plan, samples=samples)
        Xs.append(x); Ys.append(y); Zs.append(np.full_like(x, z, dtype=float))
    return np.array(Xs, dtype=float), np.array(Ys, dtype=float), np.array(Zs, dtype=float)

def add_round_facet_guides(ax3d, X, Y, Z, star_len=0.5, lower_girdle=0.75):
    try:
        n_cols = X.shape[1]
        for j in np.linspace(0, n_cols-1, 8, dtype=int):
            ax3d.plot3D(X[:, j], Y[:, j], Z[:, j], color="k", linewidth=0.3, alpha=0.35)
        i_star = max(1, int(star_len * (X.shape[0]-1)))
        i_lg = max(1, int(lower_girdle * (X.shape[0]-1)))
        ax3d.plot3D(X[i_star, :], Y[i_star, :], Z[i_star, :], color="k", linewidth=0.3, alpha=0.35)
        ax3d.plot3D(X[i_lg, :], Y[i_lg, :], Z[i_lg, :], color="k", linewidth=0.3, alpha=0.35)
    except Exception:
        pass

def render_3d(cut: str, L: float, W: float, H: float,
              table_pct: float, culet_pct: float, girdle_pct: float,
              crown_angle_deg: float, pavilion_angle_deg: float,
              renderer: str = "matplotlib", color_rgb: Tuple[int, int, int] = (200, 200, 200)):
    cut_key = _normalize_cut(cut)
    if cut_key in ("Diamond", "Round"):
        R = max(L, W) / 2.0
        z, r = profile_round_brilliant_angles(R, H, table_pct, culet_pct, girdle_pct,
                                              crown_angle_deg, pavilion_angle_deg)
        X, Y, Z = revolve_mesh(z, r, L, W, n_theta=180)
    elif cut_key in ("Oval", "Antique Cushion"):
        X, Y, Z = loft_superellipse(L, W, H, crown_frac=0.40, table_frac=table_pct, culet_frac=culet_pct, n_plan=1.6)
    elif cut_key == "Marquise":
        X, Y, Z = loft_superellipse(L, W, H, crown_frac=0.40, table_frac=table_pct, culet_frac=culet_pct, n_plan=0.7)
    elif cut_key == "Pear":
        X, Y, Z = loft_superellipse(L, W, H, crown_frac=0.42, table_frac=table_pct, culet_frac=culet_pct, n_plan=0.9)
    elif cut_key == "Trillant":
        X, Y, Z = loft_superellipse(L, W, H, crown_frac=0.40, table_frac=table_pct, culet_frac=culet_pct, n_plan=0.55)
    elif cut_key in ("Princess Cut", "Emerald Cut", "Square", "Octagon", "Baguette", "Tapered Baguette"):
        X, Y, Z = loft_superellipse(L, W, H, crown_frac=0.38, table_frac=table_pct, culet_frac=culet_pct, n_plan=4.0)
    else:
        X, Y, Z = loft_superellipse(L, W, H, crown_frac=0.40, table_frac=table_pct, culet_frac=culet_pct, n_plan=1.5)

    if renderer == "plotly" and HAVE_PLOTLY:
        fig = go.Figure()
        col = f'rgb{color_rgb}'
        fig.add_surface(x=X, y=Y, z=Z, showscale=False, opacity=1.0, colorscale=[[0, col], [1, col]])
        fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
        return ("plotly", fig, X, Y, Z)
    else:
        fig = plt.figure(figsize=(6.6, 6.6))  # slightly larger
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.95,
                        color=np.array(color_rgb)/255.0)
        if cut_key in ("Diamond", "Round"):
            add_round_facet_guides(ax, X, Y, Z, star_len=0.50, lower_girdle=0.75)
        ax.set_box_aspect((max(L,1e-6), max(W,1e-6), max(H,1e-6)))
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.view_init(elev=20, azim=45)
        return ("mpl", fig, X, Y, Z)

# ---- Pricing & ranking ----
def _species_ref_mid(species: str) -> float:
    per_ct_nat = PRICE_TABLE.get(species, {}).get("natural")
    per_ct_lab = PRICE_TABLE.get(species, {}).get("lab")
    rarity = RARITY_INDEX.get(species, 0.25)
    if per_ct_nat is None and per_ct_lab is None:
        fb = PRICING_FALLBACKS.get(species, {})
        per_ct_nat = fb.get("natural"); per_ct_lab = fb.get("lab")
    if per_ct_nat is None and per_ct_lab is None:
        return 1.0
    if per_ct_nat is None:
        return float(per_ct_lab)
    if per_ct_lab is None:
        return float(per_ct_nat)
    return float(rarity * per_ct_nat + (1.0 - rarity) * per_ct_lab)

def _color_multiplier(species: str, color: Optional[str]) -> float:
    if not color:
        return 1.0
    cm = COLOR_PRICE_MULTIPLIERS.get(species, {})
    mult = cm.get(str(color).lower())
    return float(mult) if (mult is not None and not (isinstance(mult, float) and np.isnan(mult))) else 1.0

def _transparency_match_score(species: str, t: Optional[str]) -> float:
    if not t:
        return 1.0
    prefs = TRANSPARENCY_HINTS.get(species, ["clear", "semi-translucent", "opaque"])
    return 1.0 if t in prefs else 0.6

def color_match_score(_stone_name: str, _selected_color: Optional[str]) -> float:
    return 1.0

def density_match_score(density_cm3: float, rng: Tuple[float, float]) -> float:
    lo, hi = rng
    if lo <= density_cm3 <= hi:
        return 1.0
    width = max(hi - lo, 1e-6)
    d = (lo - density_cm3) if density_cm3 < lo else (density_cm3 - hi)
    return max(0.0, 1.0 - (d / (width * 1.5)))

def _diamond_origin_adjust(name: str, pref: Optional[str]) -> float:
    if pref is None:
        return 1.0
    if pref == "natural":
        if name == "Diamond — Natural": return 1.15
        if name == "Diamond — Lab":     return 0.0
    if pref == "lab":
        if name == "Diamond — Lab":     return 1.15
        if name == "Diamond — Natural": return 0.0
    return 1.0

def raw_candidate_score(stone_name: str, density_cm3: float, selected_color: Optional[str], transparency: Optional[str]) -> float:
    rng = CATALOG_DENSITY_RANGES[stone_name]
    ds = density_match_score(density_cm3, rng)
    cs = color_match_score(stone_name, selected_color)
    ts = _transparency_match_score(stone_name, transparency)
    return max(0.0, min(1.0, 0.70 * ds + 0.10 * cs + 0.20 * ts))

def rank_candidate_gems(density_cm3: float, selected_color: Optional[str], transparency: Optional[str],
                        diamond_pref: Optional[str], top_n: int = 5):
    rows = []
    for name, rng in CATALOG_DENSITY_RANGES.items():
        within = (rng[0] <= density_cm3 <= rng[1])
        dist = 0.0 if within else (rng[0] - density_cm3 if density_cm3 < rng[0] else density_cm3 - rng[1])
        score = raw_candidate_score(name, density_cm3, selected_color, transparency)
        score *= _diamond_origin_adjust(name, diamond_pref)
        rows.append([name, rng, dist, within, score])
    rows.sort(key=lambda x: (x[3] is False, x[2], -x[4]))
    top = rows[:top_n]
    scores = np.array([max(0.0, r[4]) for r in top], dtype=float)
    if scores.sum() <= 1e-12:
        scores[:] = 1.0
    norm = scores / scores.sum()
    for i in range(len(top)):
        top[i][4] = float(norm[i])
    return top, rows

def _size_premium(species: str, carats: float) -> float:
    beta = SIZE_EXPONENT.get(species, DEFAULT_BETA)
    if beta is None or (isinstance(beta, float) and np.isnan(beta)):
        beta = DEFAULT_BETA
    return max(0.75, carats ** float(beta))

def _supply_scarcity_mult(species: str, color: Optional[str]) -> float:
    if not color:
        return 1.0
    g = GEM_DB[(GEM_DB["species"] == species) & (GEM_DB["color"].str.lower() == str(color).lower())]
    if g.empty:
        return 1.0
    qty_color = float(g["quantity_ct"].iloc[0])
    species_qty = GEM_DB[GEM_DB["species"] == species]["quantity_ct"]
    if len(species_qty) == 0:
        return 1.0
    ref_qty = float(np.nanmedian(species_qty))
    if qty_color <= 0 or ref_qty <= 0:
        return 1.0
    gamma = 0.08
    mult = (ref_qty / qty_color) ** gamma
    return float(max(0.85, min(1.25, mult)))

def _base_price_blended(species: str, color: Optional[str]) -> float:
    ref = _species_ref_mid(species)
    return float(ref) * _color_multiplier(species, color)

def _apply_transparency_price(species: str, transparency: Optional[str], price: float) -> float:
    ts = _transparency_match_score(species, transparency)
    mult = 0.85 + 0.15 * ts
    return float(price) * mult

def _apply_supply_price(species: str, color: Optional[str], price: float) -> float:
    return float(price) * _supply_scarcity_mult(species, color)

def _market_preset_factor() -> float:
    return float(st.session_state.get("market_preset_factor", 1.0))

def single_price_estimate(carats: float, color: Optional[str], transparency: Optional[str],
                          top_candidates: List[Tuple[str, Tuple[float,float], float, bool, float]]) -> float:
    if carats <= 0:
        return 0.0
    total = 0.0
    for row in top_candidates:
        species = row[0]; w = float(row[4])
        per_ct = _base_price_blended(species, color)
        Pi = _size_premium(species, carats) * per_ct * carats
        Pi = _apply_transparency_price(species, transparency, Pi)
        Pi = _apply_supply_price(species, color, Pi)
        total += w * Pi
    return float(total) * _market_preset_factor()

def price_for_species(species: str, carats: float, color: Optional[str], transparency: Optional[str]) -> float:
    if carats <= 0:
        return 0.0
    per_ct = _base_price_blended(species, color)
    price = _size_premium(species, carats) * per_ct * carats
    price = _apply_transparency_price(species, transparency, price)
    price = _apply_supply_price(species, color, price)
    return float(price) * _market_preset_factor()


# ----------------- Regional multipliers — LIVE EDIT + PRESETS -----------------
def _init_region_state():
    if "base_region_factors" not in st.session_state:
        st.session_state["base_region_factors"] = REGION_FACTORS_DEFAULT.copy()
    if "region_overrides_df" not in st.session_state:
        cols = ["Species", "North America", "Western Europe", "Asia"]
        data = []
        for s in sorted(CATALOG_DENSITY_RANGES.keys()):
            row = {"Species": s, "North America": np.nan, "Western Europe": np.nan, "Asia": np.nan}
            if s in REGION_OVERRIDES_DEFAULT:
                for r, v in REGION_OVERRIDES_DEFAULT[s].items():
                    row[r] = float(v)
            data.append(row)
        st.session_state["region_overrides_df"] = pd.DataFrame(data, columns=cols)
    if "market_preset_factor" not in st.session_state:
        st.session_state["market_preset_factor"] = 1.0

def _sidebar_market_preset():
    with st.sidebar.expander("Market preset (Retail vs Wholesale)", expanded=True):
        preset = st.radio("Preset", ["Retail (MSRP)", "Wholesale"], index=0, horizontal=True)
        factor_map = {"Retail (MSRP)": 1.00, "Wholesale": 0.70}
        st.session_state["market_preset_factor"] = factor_map[preset]
        st.caption(f"Applied factor: ×{st.session_state['market_preset_factor']:.2f} to all prices.")

def _sidebar_region_editor():
    _init_region_state()
    _sidebar_market_preset()
    with st.sidebar.expander("Regional multipliers (live edit)", expanded=False):
        b = st.session_state["base_region_factors"]
        st.caption("Base multipliers (applied to all species unless overridden below).")
        b["North America"] = st.number_input("North America base", value=float(b["North America"]), step=0.01, key="base_na")
        b["Western Europe"] = st.number_input("Western Europe base", value=float(b["Western Europe"]), step=0.01, key="base_we")
        b["Asia"] = st.number_input("Asia base", value=float(b["Asia"]), step=0.01, key="base_asia")
        st.session_state["base_region_factors"] = b

        st.caption("Optional species overrides (blank = inherit base). Double-click cells to edit.")
        edited = st.data_editor(
            st.session_state["region_overrides_df"],
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Species": st.column_config.TextColumn(disabled=True),
                "North America": st.column_config.NumberColumn(format="%.2f"),
                "Western Europe": st.column_config.NumberColumn(format="%.2f"),
                "Asia": st.column_config.NumberColumn(format="%.2f"),
            },
            key="region_editor"
        )
        st.session_state["region_overrides_df"] = edited

        cols = st.columns(2)
        if cols[0].button("Reset region factors to defaults"):
            st.session_state["base_region_factors"] = REGION_FACTORS_DEFAULT.copy()
            del st.session_state["region_overrides_df"]
            _init_region_state()
            st.experimental_rerun()
        cols[1].markdown(
            "<div style='font-size:0.9em;color:#666'>Tip: Multipliers scale final USD values after all adjustments.</div>",
            unsafe_allow_html=True
        )

def _region_factor_for_species(species: str, region: str) -> float:
    _init_region_state()
    base = st.session_state["base_region_factors"].get(region, 1.0)
    df = st.session_state["region_overrides_df"]
    row = df[df["Species"] == species]
    if not row.empty:
        val = row.iloc[0][region]
        if pd.notna(val):
            return float(val)
    return float(base)

def regionalize(price_usd: float, species: str) -> Dict[str, float]:
    regions = ["North America", "Western Europe", "Asia"]
    return {r: float(price_usd) * _region_factor_for_species(species, r) for r in regions}


# ----------------- Precious Metals Spot -----------------
def _fetch_spot_yf(ticker: str) -> Optional[float]:
    try:
        if not HAVE_YF: return None
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is None or hist.empty: return None
        price = float(hist["Close"].dropna().iloc[-1])
        return price
    except Exception:
        return None

def get_spot_prices() -> Dict[str, Dict[str, float]]:
    sp = st.session_state.get("spot_manual", {"gold_oz": 2000.0, "silver_oz": 25.0, "platinum_oz": 900.0})
    if st.session_state.get("use_manual_spot", False):
        g_oz, s_oz, p_oz = float(sp["gold_oz"]), float(sp["silver_oz"]), float(sp["platinum_oz"])
    else:
        g_oz = _fetch_spot_yf("GC=F") or sp["gold_oz"]
        s_oz = _fetch_spot_yf("SI=F") or sp["silver_oz"]
        p_oz = _fetch_spot_yf("PL=F") or sp["platinum_oz"]
    return {
        "gold":     {"per_oz": g_oz, "per_g": g_oz / TROY_OUNCE_TO_GRAM},
        "silver":   {"per_oz": s_oz, "per_g": s_oz / TROY_OUNCE_TO_GRAM},
        "platinum": {"per_oz": p_oz, "per_g": p_oz / TROY_OUNCE_TO_GRAM},
    }

# ----------------- Energy Prices (manual/live) -----------------
def get_energy_prices() -> Dict[str, float]:
    defaults = {"electricity_kwh": 0.20, "gasoline_gal": 3.85, "natgas_mmbtu": 12.00}
    user = st.session_state.get("energy_manual", defaults.copy())
    if st.session_state.get("use_manual_energy", True):
        return {k: float(user.get(k, defaults[k])) for k in defaults}
    px = defaults.copy()
    try:
        if HAVE_YF:
            ng = _fetch_spot_yf("NG=F")
            if ng: px["natgas_mmbtu"] = float(ng)
            cl = _fetch_spot_yf("CL=F")
            if cl:
                per_gal = (float(cl) / 42.0) * 1.55
                px["gasoline_gal"] = max(per_gal, px["gasoline_gal"])
    except Exception:
        pass
    return px

# ----------------- Order Builder (with sticky UI) -----------------
def _init_order_state():
    if "order_items" not in st.session_state:
        st.session_state["order_items"] = []
    if "order_seq" not in st.session_state:
        st.session_state["order_seq"] = 1

def _add_to_order(item: Dict):
    _init_order_state()
    item = dict(item)
    item["id"] = int(st.session_state["order_seq"])
    st.session_state["order_seq"] += 1
    st.session_state["order_items"].append(item)

def _remove_from_order(item_id: int):
    _init_order_state()
    st.session_state["order_items"] = [it for it in st.session_state["order_items"] if it["id"] != item_id]

def _order_totals():
    _init_order_state()
    init_sum = 0.0
    final_sum = 0.0
    for it in st.session_state["order_items"]:
        initial = float(it.get("estimate_usd", 0.0))
        override = it.get("override_usd", None)
        final_val = float(override) if (override is not None and override >= 0) else initial
        init_sum += initial
        final_sum += final_val
    return init_sum, final_sum

# ------------- UI polish (CSS + Top Navbar) -------------
def _inject_global_css():
    st.markdown("""
    <style>
    /* Full-width feel */
    section.main > div.block-container {max-width: 1500px; padding-left: 2rem; padding-right: 2rem;}

    /* Top bar */
    .rc-topbar {
        position: relative;
        top: -0.5rem;
        display: flex;
        gap: 1rem;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(250,250,252,0.9), rgba(245,245,249,0.9));
        border: 1px solid rgba(0,0,0,0.05);
    }
    .rc-title {
        font-weight: 700;
        font-size: 1.05rem;
    }
    .rc-tagline {
        font-size: 0.9rem;
        color: #5b5b66;
        margin-top: 0.15rem;
    }
    .rc-byline {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    .rc-links {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        justify-content: flex-end;
    }
    .rc-chip {
        padding: 0.32rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.08);
        background: white;
        font-size: 0.85rem;
        text-decoration: none !important;
    }
    .rc-chip:hover {
        background: #f6f7fb;
    }

    /* Right column sticky panel */
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"]{
        position: sticky;
        top: 1rem;
    }
    @media (max-width: 1200px){
        div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"]{
            position: static;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def _top_navbar():
    st.markdown("""
    <div class="rc-topbar">
      <div>
        <div class="rc-title">Industry Standard Gem Valuer v25</div>
        <div class="rc-tagline">Wide UI • Sticky order • 3D geometry • Hydrostatic SG • density & regional factors • retail/wholesale presets • Energy & Metals equivalents • Rich PDF.</div>
        <div class="rc-byline">Created By: Ryan Childs (<a href="mailto:ryanchilds10@gmail.com">ryanchilds10@gmail.com</a> • <a href="mailto:archdukequandt@gmail.com">archdukequandt@gmail.com</a>)</div>
      </div>
      <div class="rc-links">
        <a class="rc-chip" href="https://www.juwelo.com/media/wysiwyg/en/lexicon/classic-shapes-cuts.jpg" target="_blank">Classic shapes</a>
        <a class="rc-chip" href="https://4cs.gia.edu/en-us/diamond-cut/" target="_blank">GIA Cut Grading</a>
        <a class="rc-chip" href="https://www.gia.edu/gem-encyclopedia" target="_blank">GIA Encyclopedia</a>
        <a class="rc-chip" href="https://en.wikipedia.org/wiki/Cut_(gems)" target="_blank">Gem cuts (wiki)</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

def _render_order_panel():
    _init_order_state()
    st.markdown("### Order — Stones & Estimates")
    if not st.session_state["order_items"]:
        st.info("No stones added yet. Use **Add to Order (right)** in the main section.")
        return

    init_sum, final_sum = _order_totals()
    st.metric("Order total (final)", f"${final_sum:,.2f}", delta=f"Initial total: ${init_sum:,.2f}")

    for it in st.session_state["order_items"]:
        with st.expander(f"#{it['id']} • {it['species']} • {it['weight_ct']:.2f} ct • {it['selected_region']}", expanded=False):
            colA, colB = st.columns([2,1])
            with colA:
                st.write(f"**Cut:** {it['cut']}")
                st.write(f"**Color / Transparency:** {it['color']} / {it['transparency']}")
                st.write(f"**Confidence:** {it['confidence']*100:.2f}%")
                st.write(f"**Density used:** {it.get('density_cm3', float('nan')):.6f} g/cm³")
                st.write(f"**Initial estimate ({it['selected_region']}):** ${it['estimate_usd']:,.2f}")
            with colB:
                new_region = st.selectbox(
                    "Region", ["North America", "Western Europe", "Asia"],
                    index=["North America","Western Europe","Asia"].index(it["selected_region"]),
                    key=f"item_region_{it['id']}"
                )
                if new_region != it["selected_region"]:
                    it["selected_region"] = new_region
                    it["estimate_usd"] = float(it["region_prices"].get(new_region, it["estimate_usd"]))
                override = st.number_input("Override price (USD)", min_value=0.0,
                                           value=float(it.get("override_usd", 0.0) or 0.0),
                                           step=1.0, format="%.2f",
                                           key=f"item_override_{it['id']}")
                it["override_usd"] = float(override) if override > 0 else None
                final_val = float(it["override_usd"]) if it["override_usd"] is not None else float(it["estimate_usd"])
                st.write(f"**Final price:** ${final_val:,.2f}")
                if st.button("Remove", key=f"remove_{it['id']}"):
                    _remove_from_order(it["id"])
                    st.experimental_rerun()

    init_sum, final_sum = _order_totals()
    st.divider()
    st.write(f"**Final total:** ${final_sum:,.2f}")
    st.caption(f"Initial total (sum of initial estimates): ${init_sum:,.2f}")

    rows = []
    for it in st.session_state["order_items"]:
        rows.append({
            "id": it["id"],
            "species": it["species"],
            "cut": it["cut"],
            "color": it["color"],
            "transparency": it["transparency"],
            "weight_ct": it["weight_ct"],
            "region": it["selected_region"],
            "initial_estimate_usd": it["estimate_usd"],
            "override_usd": it.get("override_usd", None),
            "final_usd": it.get("override_usd", it["estimate_usd"]),
            "confidence": it["confidence"],
        })
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download order (CSV)", data=csv_bytes, file_name="gem_order.csv", mime="text/csv")
    if st.button("Clear order"):
        st.session_state["order_items"] = []
        st.experimental_rerun()


# ----------------- UI helpers -----------------
def _reset_all_params():
    keep_keys = {"region_overrides_df", "base_region_factors", "market_preset_factor",
                 "spot_manual", "use_manual_spot", "order_items", "order_seq",
                 "energy_manual", "use_manual_energy"}
    for k in list(st.session_state.keys()):
        if k not in keep_keys:
            del st.session_state[k]
    st.experimental_rerun()

def _cuts_reference_links() -> Dict[str, str]:
    return {
        "Classic shapes (image sheet)": "https://www.juwelo.com/media/wysiwyg/en/lexicon/classic-shapes-cuts.jpg",
        "GIA: Diamond Cut Grading": "https://4cs.gia.edu/en-us/diamond-cut/",
        "GIA: Gem Encyclopedia": "https://www.gia.edu/gem-encyclopedia",
        "Gemstone cuts overview": "https://en.wikipedia.org/wiki/Cut_(gems)",
    }

def _resources_tables():
    left, right = st.columns(2)
    with left:
        st.markdown("#### Resources — Gemology")
        st.markdown("- [GIA: Diamond Cut Grading](https://4cs.gia.edu/en-us/diamond-cut/)")
        st.markdown("- [GIA: Gem Encyclopedia](https://www.gia.edu/gem-encyclopedia)")
        st.markdown("- [Classic shapes (image sheet)](https://www.juwelo.com/media/wysiwyg/en/lexicon/classic-shapes-cuts.jpg)")
        st.markdown("- [Gemstone cuts overview (Wikipedia)](https://en.wikipedia.org/wiki/Cut_(gems))")
    with right:
        st.markdown("#### Contact")
        st.markdown("""
**Ryan Childs**  
[ryanchilds10@gmail.com](mailto:ryanchilds10@gmail.com) • [archdukequandt@gmail.com](mailto:archdukequandt@gmail.com)
""")

def _st_title():
    # must be called before any other st.* call to set layout properly
    st.set_page_config(layout="wide", page_title="Industry Standard Gem Valuer v25", page_icon="💎")
    _inject_global_css()
    _top_navbar()
    st.button("Reset all parameters", on_click=_reset_all_params, help="Clear inputs and return to defaults")

def _st_inputs():
    cuts = list(SHAPE_FACTORS.keys())

    st.sidebar.header("Inputs")
    renderer = st.sidebar.selectbox("3D renderer", ["Matplotlib (robust)", "Plotly (interactive if available)"], index=0)
    renderer = "plotly" if (renderer.startswith("Plotly") and HAVE_PLOTLY) else "matplotlib"

    # Regional multipliers & presets
    _sidebar_region_editor()

    # Precious metals spot (live/manual)
    with st.sidebar.expander("Precious metal spot prices (USD)"):
        use_manual = st.checkbox("Use manual spot prices", value=st.session_state.get("use_manual_spot", False))
        st.session_state["use_manual_spot"] = use_manual
        if use_manual:
            sm = st.session_state.get("spot_manual", {"gold_oz": 2000.0, "silver_oz": 25.0, "platinum_oz": 900.0})
            sm["gold_oz"] = st.number_input("Gold $/oz", value=float(sm["gold_oz"]), step=1.0)
            sm["silver_oz"] = st.number_input("Silver $/oz", value=float(sm["silver_oz"]), step=0.1, format="%.3f")
            sm["platinum_oz"] = st.number_input("Platinum $/oz", value=float(sm["platinum_oz"]), step=1.0)
            st.session_state["spot_manual"] = sm
        else:
            st.caption("Uses Yahoo Finance benchmarks (GC=F/SI=F/PL=F). Click Calculate to refresh.")

    # Energy prices (manual/live)
    with st.sidebar.expander("Energy prices (USD) — for equivalents"):
        use_m = st.checkbox("Use manual energy prices", value=st.session_state.get("use_manual_energy", True))
        st.session_state["use_manual_energy"] = use_m
        em = st.session_state.get("energy_manual", {"electricity_kwh": 0.20, "gasoline_gal": 3.85, "natgas_mmbtu": 12.00})
        em["electricity_kwh"] = st.number_input("Electricity $/kWh", value=float(em["electricity_kwh"]), step=0.01, format="%.3f")
        em["gasoline_gal"] = st.number_input("Gasoline $/gal", value=float(em["gasoline_gal"]), step=0.01, format="%.3f")
        em["natgas_mmbtu"] = st.number_input("Natural Gas $/MMBtu", value=float(em["natgas_mmbtu"]), step=0.10, format="%.3f")
        st.session_state["energy_manual"] = em
        if not use_m:
            st.caption("Attempts to fetch proxies (NG=F and CL=F). Electricity remains manual.")

    # Stone type override
    species_options = ["Auto (let program decide)"] + list(CATALOG_DENSITY_RANGES.keys())
    override_species = st.sidebar.selectbox("Stone Type Override (optional)", species_options, index=0)
    override_species = None if override_species.startswith("Auto") else override_species

    # Diamond origin toggle
    diamond_pref_raw = st.sidebar.selectbox("Diamond origin", ["Auto", "Natural", "Lab"], index=0)
    diamond_pref = None if diamond_pref_raw == "Auto" else diamond_pref_raw.lower()

    # Color
    available_colors = sorted({str(c).lower() for c in GEM_DB["color"].unique()})
    color_choice = st.sidebar.selectbox("Stone color (helps pricing)", ["Auto/Unknown"] + available_colors, index=0)
    color_norm = None if color_choice.lower().startswith("auto") else color_choice

    # Transparency
    transparency = st.sidebar.selectbox("Transparency", ["Auto/Unknown", "clear", "semi-translucent", "opaque"], index=0)
    tr_norm = None if transparency.lower().startswith("auto") else transparency

    # Weight
    weight_unit = st.sidebar.selectbox("Weight unit", ["carat", "gram"], index=0)
    weight_value = st.sidebar.number_input(f"Weight ({'ct' if weight_unit=='carat' else 'g'})", min_value=0.0, value=1.000000, step=0.000001, format="%.6f")

    # Shape selection
    cut = st.sidebar.selectbox("Shape / Cut", cuts, index=0)

    # Dimensions
    st.sidebar.markdown("### Dimensions")
    if cut in ("Diamond", "Round"):
        diameter = st.sidebar.number_input("Diameter (mm)", min_value=0.0, value=6.500000, step=0.000001, format="%.6f")
        L = W = diameter
        H_input = st.sidebar.number_input("Height / Total Depth (mm) — optional", min_value=0.0, value=0.000000, step=0.000001, format="%.6f")
    else:
        L = st.sidebar.number_input("Length L (mm)", min_value=0.0, value=7.000000, step=0.000001, format="%.6f")
        W = st.sidebar.number_input("Width W (mm)", min_value=0.0, value=5.000000, step=0.000001, format="%.6f")
        H_input = st.sidebar.number_input("Height / Total Depth H (mm) — optional", min_value=0.0, value=0.000000, step=0.000001, format="%.6f")
        diameter = None

    # Proportions / Angles (RB tuned)
    st.sidebar.markdown("### 3D Proportions (RB tuned)")
    depth_pct = st.sidebar.slider("Total Depth % of L/diameter", 40, 85, 61, 1) / 100.0
    table_pct = st.sidebar.slider("Table % of half-width", 55, 57, 56, 1) / 100.0
    crown_angle_deg = st.sidebar.slider("Crown angle (degrees)", 34.0, 35.0, 34.5, 0.1)
    pavilion_angle_deg = st.sidebar.slider("Pavilion angle (degrees)", 40.7, 41.0, 40.8, 0.1)
    culet_pct = st.sidebar.slider("Culet % of half-width", 0, 8, 1, 1) / 100.0
    girdle_pct = st.sidebar.slider("Girdle thickness % of depth", 0, 10, 4, 1) / 100.0

    # Hydrostatic SG (supports ct/g)
    with st.sidebar.expander("Hydrostatic SG helper (optional)"):
        sg_unit = st.radio("Input units", ["carat (ct)", "gram (g)"], index=0, horizontal=True)
        use_sg = st.checkbox("Use hydrostatic SG to identify (override density)", value=False)
        w_air = st.number_input("Weight in air", min_value=0.0, value=0.000000, step=0.000001, format="%.6f")
        w_water = st.number_input("Weight in water", min_value=0.0, value=0.000000, step=0.000001, format="%.6f")
        apply_temp_corr = st.checkbox("Apply water temperature correction", value=False)
        water_temp_c = st.number_input("Water temperature (°C)", value=20.0, step=0.5) if apply_temp_corr else 20.0

        sg_value = None
        if use_sg and w_air > 0 and 0 < w_water < w_air:
            if sg_unit.startswith("carat"):
                w_air_g = w_air * CARAT_TO_GRAM
                w_water_g = w_water * CARAT_TO_GRAM
            else:
                w_air_g, w_water_g = w_air, w_water
            rho_water = 0.9982 - 0.0003*(water_temp_c - 20.0) if apply_temp_corr else 0.9982
            sg_value = (w_air_g / (w_air_g - w_water_g)) * (rho_water / 0.9982)
            st.info(f"Computed SG = {round6(sg_value):.6f} (≈ g/cm³)")

    params = dict(
        renderer=renderer, diameter=diameter, H_override=H_input if H_input > 0 else None,
        depth_pct=depth_pct, table_pct=table_pct, crown_angle_deg=crown_angle_deg,
        pavilion_angle_deg=pavilion_angle_deg, culet_pct=culet_pct, girdle_pct=girdle_pct,
        diamond_pref=diamond_pref
    )
    dims = dict(L=L, W=W)
    return color_norm, tr_norm, params, weight_value, weight_unit, dims, cut, sg_value, override_species

def compute_density_result(cut: str, weight_value: float, weight_unit: str, dims: Dict[str, float], params: Dict[str, float]):
    L, W = dims["L"], dims["W"]
    if cut in ("Diamond", "Round") and params.get("diameter"):
        L = W = params["diameter"]
    H = params["H_override"] if params.get("H_override") else params["depth_pct"] * max(L, W)
    m_g = (weight_value * CARAT_TO_GRAM) if weight_unit.lower().startswith("carat") else weight_value
    m_ct = m_g / CARAT_TO_GRAM
    if cut in ("Diamond", "Round") and L > 0:
        depth_ratio = params["depth_pct"] if params["H_override"] is None else (H / L if L > 0 else 0.61)
        H_eff = H if H > 0 else depth_ratio * L
        V_mm3 = max(0.0, SHAPE_FACTORS["Diamond"] * L * W * H_eff)
    else:
        H_eff = H if H > 0 else DEFAULT_DEPTH_RATIO_BY_CUT.get(cut, 0.60) * max(L, W)
        V_mm3 = max(0.0, SHAPE_FACTORS.get(cut, 0.60) * L * W * H_eff)
    V_cm3 = V_mm3 * MM3_TO_CM3
    density_cm3 = (m_g / V_cm3) if V_cm3 > 0 else float("inf")
    return {
        "L": round6(L), "W": round6(W), "H": round6(H),
        "mass_g": round6(m_g), "mass_ct": round6(m_ct),
        "volume_mm3": round6(V_mm3), "density_cm3": round6(density_cm3)
    }

def db_view_table(region: str, carats: float, species_filter: Optional[List[str]] = None, color_filter: Optional[str] = None) -> pd.DataFrame:
    df = GEM_DB.copy()
    if species_filter:
        df = df[df["species"].isin(species_filter)]
    if color_filter:
        df = df[df["color"].str.lower() == color_filter.lower()]
    out = []
    mf = _market_preset_factor()
    for _, r in df.iterrows():
        s = str(r["species"]); col = str(r["color"]); beta = float(r["beta"])
        rarity = RARITY_INDEX.get(s, 0.25)
        mid_nat = _mid(r["natural_low"], r["natural_high"])
        mid_lab = _mid(r["lab_low"], r["lab_high"])
        if mid_nat is None and mid_lab is None:
            blended_per_ct1 = _species_ref_mid(s) * (COLOR_PRICE_MULTIPLIERS.get(s, {}).get(col.lower(), 1.0))
        elif mid_nat is None:
            blended_per_ct1 = mid_lab
        elif mid_lab is None:
            blended_per_ct1 = mid_nat
        else:
            blended_per_ct1 = rarity * mid_nat + (1 - rarity) * mid_lab
        blended_per_ct1 = float(blended_per_ct1 if blended_per_ct1 is not None else _species_ref_mid(s))
        if blended_per_ct1 <= 0 or np.isnan(blended_per_ct1):
            blended_per_ct1 = _species_ref_mid(s)

        per_ct_size = max(0.75, carats ** beta) * blended_per_ct1
        scarcity = _supply_scarcity_mult(s, col)
        per_ct_size *= scarcity
        reg_mult = _region_factor_for_species(s, region)
        per_ct_region = float(per_ct_size * reg_mult * mf)
        total_value = float(per_ct_region * carats)

        out.append({
            "Species": s,
            "Color": col,
            "β (size exponent)": round6(beta),
            "Global Qty (ct)": float(r["quantity_ct"]),
            f"Per-ct @ {carats:.2f} ct in {region} (USD)": per_ct_region,
            f"Total @ {carats:.2f} ct in {region} (USD)": total_value
        })
    tbl = pd.DataFrame(out)
    for c in tbl.columns:
        if "USD" in c or c in ["β (size exponent)", "Global Qty (ct)"]:
            tbl[c] = pd.to_numeric(tbl[c], errors="coerce").fillna(0.0)
    return tbl

# ----------------- Helpers for PDF order details -----------------
def _order_item_snapshot_png(it: Dict) -> Optional[bytes]:
    try:
        color_map = {
            "colorless": (220, 230, 240), "white": (230, 230, 230), "yellow": (240, 220, 120),
            "pink": (245, 190, 210), "red": (210, 60, 60), "blue": (110, 160, 240),
            "green": (90, 180, 120), "purple": (160, 120, 220), "brown": (150, 110, 70),
            "black": (40, 40, 40), "orange": (240, 150, 80), "bi-color": (180, 180, 220),
            "white/opaque": (230, 230, 230)
        }
        rgb = color_map.get(str(it.get("color","")).lower(), (200, 200, 200))
        kind, fig3d, *_ = render_3d(
            it["cut"],
            float(it["geom"]["L"]), float(it["geom"]["W"]), float(it["geom"]["H"]),
            it["render"]["table_pct"], it["render"]["culet_pct"], it["render"]["girdle_pct"],
            it["render"]["crown_angle_deg"], it["render"]["pavilion_angle_deg"],
            renderer="matplotlib",
            color_rgb=rgb
        )
        buf_png = BytesIO()
        fig3d.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig3d)
        png = buf_png.getvalue()
        buf_png.close()
        return png
    except Exception:
        return None

def _top5_alt_values_for_item(it: Dict) -> List[Dict]:
    rows = []
    names = it.get("top5_names", []) or []
    confs = it.get("top5_confs", []) or []
    if not names:
        names = [it["species"]]
        confs = [it.get("confidence", 1.0)]
    for idx, name in enumerate(names):
        if idx == 0:
            continue
        price_na = price_for_species(name, it["weight_ct"], it["color"], it["transparency"])
        reg_prices = regionalize(price_na, name)
        rows.append({
            "Rank": idx+1,
            "Species": name,
            "Confidence%": float(confs[idx]) * 100.0 if idx < len(confs) else 0.0,
            "NA (USD)": float(reg_prices.get("North America", price_na)),
            f"{it['selected_region']} (USD)": float(reg_prices.get(it["selected_region"], price_na))
        })
    return rows[:4]

# ----------------- PDF Builder -----------------
def _build_pdf(inputs: Dict, res: Dict, est_species: str, est_conf_pct: float,
               regional: Dict[str, float], db_table: pd.DataFrame,
               fig_png_bytes: Optional[bytes], include_full_db: bool, top_n_rows: int,
               metals: Dict[str, Dict[str, float]], grams_equiv: Dict[str, float],
               include_order_details: bool, order_items: List[Dict]) -> bytes:
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

    # -------- Cover / current stone summary --------
    y = H - margin
    c.setTitle("Industry Standard Gem Valuer Report")
    text_line(margin, y, "Industry Standard Gem Valuer — Report", 14); y -= 18
    text_line(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 9); y -= 14

    text_line(margin, y, "Inputs:", 12); y -= 12
    for k in ["cut","weight","dimensions","color","transparency","renderer","preset","diamond"]:
        if k in inputs:
            text_line(margin, y, f"• {k.capitalize()}: {inputs[k]}", 10); y -= 12

    y -= 2
    text_line(margin, y, "Physical Properties:", 12); y -= 12
    text_line(margin, y, f"• Density used: {res['density_cm3']:.6f} g/cm³", 10); y -= 12
    text_line(margin, y, f"• Mass: {res['mass_g']:.6f} g ({res['mass_ct']:.6f} ct)", 10); y -= 12
    text_line(margin, y, f"• Volume: {res['volume_mm3']:.6f} mm³", 10); y -= 14

    text_line(margin, y, "Identification & Value:", 12); y -= 12
    text_line(margin, y, f"• Best species estimate: {est_species}", 10); y -= 12
    text_line(margin, y, f"• Confidence (top-5 normalized): {est_conf_pct:.2f}%", 10); y -= 12
    text_line(margin, y, "• Estimated value by region:", 10); y -= 12
    for r, v in regional.items():
        text_line(margin+16, y, f"{r}: ${float(v):,.2f}", 10); y -= 12

    y -= 4
    text_line(margin, y, "Metal grams-equivalent of North America estimate:", 12); y -= 12
    for metal in ["gold","silver","platinum"]:
        text_line(margin+16, y, f"{metal.capitalize()}: {grams_equiv[metal]:.6f} g (spot ${metals[metal]['per_g']:.2f}/g)", 10); y -= 12

    # 3D Snapshot
    if fig_png_bytes:
        if maybe_pagebreak(y, needed=int(3.8*inch)):
            c.showPage(); y = H - margin
        text_line(margin, y, "3D Snapshot:", 12); y -= 12
        ok = draw_image(fig_png_bytes, margin, y - 3.6*inch)
        if ok:
            y -= (3.6*inch + 12)
        else:
            text_line(margin, y, "(Snapshot unavailable — renderer was Plotly or image capture failed.)", 10); y -= 14

    # Database section
    if maybe_pagebreak(y, needed=140): c.showPage(); y = H - margin
    text_line(margin, y, "Database Section:", 12); y -= 14
    if db_table is None or db_table.empty:
        text_line(margin, y, "(No rows for current filters.)", 10); y -= 12
    else:
        sort_cols = [c for c in db_table.columns if "Total @" in c] or [db_table.columns[-1]]
        show = db_table.sort_values(sort_cols[0], ascending=False)
        if not include_full_db:
            show = show.head(max(1, int(top_n_rows)))
            text_line(margin, y, f"(Top {len(show)} by total value)", 10); y -= 12
        else:
            text_line(margin, y, f"(Full table: {len(show)} rows)", 10); y -= 12
        for _, r in show.iterrows():
            val = float(r[sort_cols[0]])
            line = f"- {r['Species']} | {r['Color']} | {sort_cols[0]}: ${val:,.2f}"
            if maybe_pagebreak(y, needed=20):
                c.showPage(); y = H - margin
            text_line(margin, y, line, 9); y -= 11

    # -------- Detailed section for each order item --------
    if include_order_details and order_items:
        for it in order_items:
            c.showPage(); y = H - margin
            text_line(margin, y, f"Order Item #{it['id']} — {it['species']} ({it['weight_ct']:.2f} ct)", 13); y -= 16
            text_line(margin, y, f"Cut: {it['cut']} • Color: {it['color']} • Transparency: {it['transparency']}", 10); y -= 12
            text_line(margin, y, f"Density used: {float(it.get('density_cm3', 0.0)):.6f} g/cm³", 10); y -= 12
            text_line(margin, y, f"Confidence (best species): {it['confidence']*100:.2f}%", 10); y -= 12
            text_line(margin, y, f"Selected region: {it['selected_region']}", 10); y -= 12
            text_line(margin, y, f"Initial estimate: ${it['estimate_usd']:,.2f}"
                                 f" • Final: ${float(it.get('override_usd', it['estimate_usd'])):,.2f}", 10); y -= 14

            png = _order_item_snapshot_png(it)
            if png:
                if maybe_pagebreak(y, needed=int(3.8*inch)):
                    c.showPage(); y = H - margin
                text_line(margin, y, "Rendering:", 12); y -= 12
                _ = draw_image(png, margin, y - 3.6*inch)
                y -= (3.6*inch + 8)

            rows = _top5_alt_values_for_item(it)
            text_line(margin, y, "What-if: other Top-5 species valuations", 12); y -= 12
            if not rows:
                text_line(margin, y, "(No alternate candidates recorded.)", 10); y -= 12
            else:
                for r in rows:
                    if maybe_pagebreak(y, needed=30):
                        c.showPage(); y = H - margin
                    text_line(margin, y, f"- Rank {r['Rank']}: {r['Species']}  "
                                         f"(Conf {r['Confidence%']:.2f}%)", 10); y -= 12
                    reg_key = f"{it['selected_region']} (USD)"
                    text_line(margin + 18, y, f"North America: ${r['NA (USD)']:,.2f}   {it['selected_region']}: ${r[reg_key]:,.2f}", 10)
                    y -= 12

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


def main():
    if st is None:
        print("This module can be imported for CLI use, but the UI requires Streamlit.")
        return

    # ---------- Header / top links ----------
    _st_title()
    _init_order_state()

    # Quick “Resources” row near the top (non-intrusive)
    with st.expander("Resources & Contacts", expanded=False):
        _resources_tables()

    color_choice, transparency, params, weight_value, weight_unit, dims, cut, sg_value, override_species = _st_inputs()

    # ------------- Identification -------------
    st.markdown("---")
    st.header("Identification")

    # Common Gem Cuts — reference links
    with st.expander("Common Gem Cuts — reference links"):
        links = _cuts_reference_links()
        for k, v in links.items():
            st.markdown(f"- [{k}]({v})")

    # Compute density/mass/volume
    res = compute_density_result(cut, weight_value, weight_unit, dims, params)
    density_cm3 = sg_value if (sg_value and sg_value > 0) else res["density_cm3"]
    density_mm3 = density_cm3 / 1000.0

    st.subheader("Physical Properties")
    st.metric("Density used (g/cm³)", f"{round6(density_cm3):0.6f}")
    st.caption(f"Equivalent density in g/mm³: {round6(density_mm3):.6f}")
    st.write(
        f"**Mass:** {res['mass_g']:0.6f} g  \n"
        f"**Mass:** {res['mass_ct']:0.6f} ct  \n"
        f"**Volume:** {res['volume_mm3']:0.6f} mm³"
    )
    if sg_value:
        st.info("Identification uses **hydrostatic SG override** (see bottom for formula details).")

    # 3D color map
    color_map = {
        "colorless": (220, 230, 240), "white": (230, 230, 230), "yellow": (240, 220, 120),
        "pink": (245, 190, 210), "red": (210, 60, 60), "blue": (110, 160, 240),
        "green": (90, 180, 120), "purple": (160, 120, 220), "brown": (150, 110, 70),
        "black": (40, 40, 40), "orange": (240, 150, 80), "bi-color": (180, 180, 220),
        "white/opaque": (230, 230, 230)
    }
    rgb = color_map.get((color_choice or "").lower(), (200, 200, 200))

    # 3D preview
    st.subheader("3D Preview (facet guides tuned to industry RB parameters)")
    kind, fig3d, X, Y, Z = render_3d(
        cut, res["L"], res["W"], res["H"],
        table_pct=params["table_pct"], culet_pct=params["culet_pct"], girdle_pct=params["girdle_pct"],
        crown_angle_deg=params["crown_angle_deg"], pavilion_angle_deg=params["pavilion_angle_deg"],
        renderer=params["renderer"], color_rgb=rgb
    )
    fig_png_bytes = None
    if kind == "plotly":
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("Tip: Switch renderer to Matplotlib to embed a 3D snapshot in the PDF.")
    else:
        st.pyplot(fig3d, use_container_width=True)
        try:
            buf_png = BytesIO()
            fig3d.savefig(buf_png, format="png", dpi=200, bbox_inches="tight")
            fig_png_bytes = buf_png.getvalue()
            buf_png.close()
        except Exception:
            fig_png_bytes = None

    # Candidate ranking (top-5 normalized)
    topN, _full = rank_candidate_gems(density_cm3, color_choice, transparency, params.get("diamond_pref"), top_n=5)
    st.subheader("Top 5 Candidates (confidences sum to 100%)")
    def row_for(name, rng, dist, within, conf_norm):
        per_ct_nat = PRICE_TABLE.get(name, {}).get("natural")
        per_ct_lab = PRICE_TABLE.get(name, {}).get("lab")
        if per_ct_nat is None and per_ct_lab is None:
            fb = PRICING_FALLBACKS.get(name, {})
            per_ct_nat = fb.get("natural"); per_ct_lab = fb.get("lab")
        return {"Stone": name,
                "Typical Density (g/cm³)": f"{rng[0]:.2f}–{rng[1]:.2f}",
                "Match": "Within range" if within else f"Δ={dist:.6f}",
                "Confidence": f"{conf_norm*100:.2f}%",
                "Avg $/ct nat": "—" if per_ct_nat is None else f"${float(per_ct_nat):,.2f}",
                "Avg $/ct lab": "—" if per_ct_lab is None else f"${float(per_ct_lab):,.2f}"}
    st.table(pd.DataFrame([row_for(*row) for row in topN]))

    # Final single estimates + regions
    st.subheader("Value (Single Best Estimate + Regions)")
    if override_species:
        est_species = override_species
        est_conf = 1.0
        st.info(f"Species override applied: **{est_species}** (100% by user).")
        est_price = price_for_species(est_species, res["mass_ct"], color_choice, transparency)
    else:
        best = topN[0] if topN else None
        if best:
            est_species = best[0]
            est_conf = best[-1]  # normalized
        else:
            est_species = "Unknown"
            est_conf = 0.0
        est_price = single_price_estimate(res["mass_ct"], color_choice, transparency, topN)

    st.write(f"**Stone type (best estimate):** {est_species}")
    st.write(f"**Confidence (top-5 normalized):** {est_conf*100:.2f}%")

    regional = regionalize(est_price, est_species)
    st.markdown("**Estimated value by region**")
    st.table(pd.DataFrame([{"Region": k, "Estimated value (USD)": f"${float(v):,.2f}"} for k, v in regional.items()]))

    # Metals & Energy equivalents
    metals = get_spot_prices()
    na_value = float(regional.get("North America", est_price))
    grams_equiv = {
        "gold":     (na_value / max(1e-9, metals["gold"]["per_g"])),
        "silver":   (na_value / max(1e-9, metals["silver"]["per_g"])),
        "platinum": (na_value / max(1e-9, metals["platinum"]["per_g"])),
    }
    st.markdown("**Metals grams-equivalent (North America est.)**")
    st.write(
        f"- Gold: **{round6(grams_equiv['gold']):.6f} g**  (spot ${metals['gold']['per_g']:.2f}/g)\n\n"
        f"- Silver: **{round6(grams_equiv['silver']):.6f} g**  (spot ${metals['silver']['per_g']:.2f}/g)\n\n"
        f"- Platinum: **{round6(grams_equiv['platinum']):.6f} g**  (spot ${metals['platinum']['per_g']:.2f}/g)"
    )

    st.markdown("**Energy equivalents (North America est.)**")
    epx = get_energy_prices()
    kwh = na_value / max(1e-9, epx["electricity_kwh"])
    gal = na_value / max(1e-9, epx["gasoline_gal"])
    mmbtu = na_value / max(1e-9, epx["natgas_mmbtu"])
    st.write(
        f"- Electricity: **{round6(kwh):.3f} kWh**  (price ${epx['electricity_kwh']:.2f}/kWh)\n\n"
        f"- Gasoline: **{round6(gal):.3f} gal**  (price ${epx['gasoline_gal']:.2f}/gal)\n\n"
        f"- Natural Gas: **{round6(mmbtu):.3f} MMBtu**  (price ${epx['natgas_mmbtu']:.2f}/MMBtu)"
    )
    st.caption("Adjust energy prices in the sidebar. These equivalents transform the NA USD estimate into commodity quantities.")

    # ----------------- Add to Order + Right Panel Layout (sticky) -----------------
    st.markdown("---")
    st.header("Order")
    colL, colR = st.columns([2, 1], gap="large")
    with colR:
        _render_order_panel()

    with colL:
        st.subheader("Add to Order")
        selected_region = st.selectbox("Region for order item", ["North America", "Western Europe", "Asia"], index=0, key="order_region_pick")
        initial_estimate = float(regional.get(selected_region, est_price))
        st.write(f"**Selected region estimate:** ${initial_estimate:,.2f}")

        # Capture Top-5 names & confs for "what-if" in PDF
        top5_names = [r[0] for r in topN] if topN else [est_species]
        top5_confs = [float(r[4]) for r in topN] if topN else [1.0]

        if st.button("Add this stone to Order (right panel)"):
            _add_to_order({
                "species": est_species,
                "confidence": float(est_conf),
                "top5_names": top5_names,
                "top5_confs": top5_confs,
                "region_prices": {k: float(v) for k, v in regional.items()},
                "selected_region": selected_region,
                "estimate_usd": float(initial_estimate),
                "override_usd": None,
                "weight_ct": float(res["mass_ct"]),
                "color": str(color_choice or "unknown"),
                "transparency": str(transparency or "unknown"),
                "cut": cut,
                "density_cm3": float(density_cm3),
                "geom": {"L": float(res["L"]), "W": float(res["W"]), "H": float(res["H"])},
                "render": {
                    "table_pct": params["table_pct"], "culet_pct": params["culet_pct"], "girdle_pct": params["girdle_pct"],
                    "crown_angle_deg": params["crown_angle_deg"], "pavilion_angle_deg": params["pavilion_angle_deg"]
                }
            })
            st.success("Stone added to the order.")

    # -------- Database browser --------
    st.markdown("---")
    st.header("Database")
    with st.expander("Open gemstone database & regional color pricing", expanded=True):
        region = st.selectbox("Region", ["North America", "Western Europe", "Asia"], index=0, key="db_region")
        size_ct = st.number_input("Price size (carats)", min_value=0.01, value=1.000000, step=0.000001, format="%.6f", key="db_size")
        species_sel = st.multiselect("Filter species (optional)", sorted(list(GEM_DB["species"].unique())), key="db_species")
        color_sel = st.selectbox("Filter color (optional)", ["(all)"] + sorted(list({c for c in GEM_DB["color"].unique()})), index=0, key="db_color")
        color_filter = None if color_sel == "(all)" else color_sel

        tbl = db_view_table(region=region, carats=size_ct, species_filter=species_sel or None, color_filter=color_filter)

        fmt_tbl = tbl.copy()
        def fmt_qty(x):
            try:
                x = float(x)
                if x >= 1e9: return f"{x/1e9:.2f} B ct"
                if x >= 1e6: return f"{x/1e6:.2f} M ct"
                if x >= 1e3: return f"{x/1e3:.2f} K ct"
                return f"{x:.0f} ct"
            except Exception:
                return str(x)
        if "Global Qty (ct)" in fmt_tbl.columns:
            fmt_tbl["Global Qty (ct)"] = fmt_tbl["Global Qty (ct)"].apply(fmt_qty)
        money_cols = [c for c in fmt_tbl.columns if "USD" in c]
        for ccol in money_cols:
            fmt_tbl[ccol] = fmt_tbl[ccol].apply(lambda v: f"${float(v):,.2f}")

        st.dataframe(fmt_tbl, use_container_width=True)
        csv_bytes = tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Download database slice (CSV)", data=csv_bytes, file_name="gem_database_slice.csv", mime="text/csv")

    # -------- Resources (bottom quick section) --------
    st.markdown("---")
    st.header("Resources")
    _resources_tables()

    # -------- PDF Export (now includes per-item details) --------
    st.markdown("---")
    st.header("Report / Export")
    if not REPORTLAB:
        st.warning("PDF export requires the 'reportlab' package. Install with:  \n`pip install reportlab`")
    cols = st.columns(4)
    with cols[0]:
        include_full_db = st.checkbox("Include entire DB in PDF", value=False)
    with cols[1]:
        top_n_rows = st.number_input("If not entire DB, Top-N rows", min_value=5, value=20, step=5)
    with cols[2]:
        include_order_details = st.checkbox("Include order items (each with 3D & what-ifs)", value=True)
    with cols[3]:
        gen_pdf = st.button("Generate PDF report")

    if gen_pdf and REPORTLAB:
        region = st.session_state.get("db_region", "North America")
        size_ct = st.session_state.get("db_size", 1.00)
        species_sel = st.session_state.get("db_species", [])
        color_sel = st.session_state.get("db_color", "(all)")
        color_filter = None if color_sel == "(all)" else color_sel
        db_table = db_view_table(region=region, carats=size_ct, species_filter=species_sel or None, color_filter=color_filter)

        inputs_summary = {
            "cut": cut,
            "weight": f"{weight_value:.6f} {weight_unit} ({res['mass_ct']:.6f} ct)",
            "dimensions": f"L={res['L']:.6f} mm  W={res['W']:.6f} mm  H={res['H']:.6f} mm",
            "color": (color_choice or "unknown"),
            "transparency": (transparency or "unknown"),
            "renderer": params["renderer"],
            "preset": "Wholesale" if st.session_state.get("market_preset_factor", 1.0) < 0.9 else "Retail",
            "diamond": "Auto" if params.get("diamond_pref") is None else ("Natural" if params["diamond_pref"]=="natural" else "Lab"),
        }
        res_for_pdf = dict(res); res_for_pdf["density_cm3"] = round6(density_cm3)

        metals = get_spot_prices()
        na_value = float(regional.get("North America", est_price))
        grams_equiv_pdf = {k: round6(na_value / max(1e-9, v["per_g"])) for k, v in metals.items()}

        pdf_bytes = _build_pdf(
            inputs=inputs_summary,
            res=res_for_pdf,
            est_species=est_species,
            est_conf_pct=est_conf*100.0,
            regional=regional,
            db_table=db_table,
            fig_png_bytes=fig_png_bytes,
            include_full_db=bool(include_full_db),
            top_n_rows=int(top_n_rows),
            metals=metals,
            grams_equiv=grams_equiv_pdf,
            include_order_details=bool(include_order_details),
            order_items=st.session_state.get("order_items", [])
        )
        st.download_button("Download PDF report", data=pdf_bytes, file_name="gem_report.pdf", mime="application/pdf")

    # -------- Bottom reference drop-down menus (calculation summaries) --------
    st.markdown("---")
    st.header("Reference & Calculation Details")
    with st.expander("Hydrostatic SG (Specific Gravity) — formula & notes", expanded=False):
        st.markdown(r"""
**Formula:**  
\(
\mathrm{SG} \approx \frac{W_\text{air}}{W_\text{air}-W_\text{water}} \times \frac{\rho_\text{water}(T)}{0.9982}
\)  
where \(\rho_\text{water}(20^\circ\text{C}) \approx 0.9982\ \mathrm{g/cm^3}\).  
You can toggle temperature correction in the sidebar.
        """)
        st.markdown("- Units: input in carats or grams (app converts to grams).")
        st.markdown("- SG is numerically close to density in g/cm³ for solids in water displacement.")
    with st.expander("Density & volume approximation", expanded=False):
        st.markdown(r"""
**Volume model:** \(V \approx k \cdot L \cdot W \cdot H\), with cut-dependent \(k\).  
Round uses an angle-tuned profile (table, crown, pavilion, culet, girdle) and numeric revolution.
        """)
        st.json(SHAPE_FACTORS)
    with st.expander("Pricing model", expanded=False):
        st.markdown("""
- DB stores per-color ranges and size exponent β; species baseline blends nat/lab by rarity.
- Adjustments: size premium (ct^β), transparency conformity, color scarcity, regional multipliers,
  Retail/Wholesale preset, diamond-origin bias in identification.
        """)
    with st.expander("Energy & commodity equivalents", expanded=False):
        st.markdown("""
Energy equivalents convert the **North America USD estimate** into commodity quantities.
You can adjust energy prices in the sidebar to align with current market conditions.
- Electricity (kWh), Gasoline (gal), Natural Gas (MMBtu)
- Metals equivalents computed from live/manual spot prices (gold/silver/platinum).
        """)
    with st.expander("Gem cut references (links)", expanded=False):
        for k, v in _cuts_reference_links().items():
            st.markdown(f"- [{k}]({v})")


if __name__ == "__main__":
    main()
