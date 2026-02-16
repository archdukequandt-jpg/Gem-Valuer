from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .constants import RARITY_INDEX, PRICING_FALLBACKS

def _mid(lo, hi):
    if lo is None or hi is None: return None
    return 0.5 * (float(lo) + float(hi))

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
    # Database identical to previous version (compact but adequate)
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
    # Ensure all known species exist
    for s in df["species"].unique():
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
        ref = 1.0 if (ref is None or ref <= 0) else ref
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
