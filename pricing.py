from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .constants import (
    CARAT_TO_GRAM, MM3_TO_CM3, SHAPE_FACTORS, DEFAULT_DEPTH_RATIO_BY_CUT,
    CATALOG_DENSITY_RANGES, RARITY_INDEX, TRANSPARENCY_HINTS, REGION_FACTORS_DEFAULT,
)
from .data import GEM_DB, PRICE_TABLE, COLOR_PRICE_MULTIPLIERS, SIZE_EXPONENT, DEFAULT_BETA
from .utils import round6
from .energy import energy_costs_by_region, get_kwh_per_ct_for_species

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

def compute_density_result(cut: str, weight_value: float, weight_unit: str, L: float, W: float, H_input: Optional[float], depth_pct: float):
    if cut in ("Diamond", "Round"):
        H = H_input if H_input and H_input > 0 else depth_pct * max(L, W)
        k = SHAPE_FACTORS["Diamond"]
    else:
        H = H_input if H_input and H_input > 0 else DEFAULT_DEPTH_RATIO_BY_CUT.get(cut, 0.60) * max(L, W)
        k = SHAPE_FACTORS.get(cut, 0.60)
    m_g = mass_grams(weight_value, weight_unit)
    m_ct = grams_to_carats(m_g)
    V_mm3 = max(0.0, k * L * W * H)
    V_cm3 = V_mm3 * MM3_TO_CM3
    density_cm3 = (m_g / V_cm3) if V_cm3 > 0 else float("inf")
    return {"L": round6(L), "W": round6(W), "H": round6(H), "mass_g": round6(m_g), "mass_ct": round6(m_ct),
            "volume_mm3": round6(V_mm3), "density_cm3": round6(density_cm3)}

def _species_ref_mid(species: str) -> float:
    per_ct_nat = PRICE_TABLE.get(species, {}).get("natural")
    per_ct_lab = PRICE_TABLE.get(species, {}).get("lab")
    rarity = RARITY_INDEX.get(species, 0.25)
    if per_ct_nat is None and per_ct_lab is None:
        return 1.0
    if per_ct_nat is None:
        return float(per_ct_lab)
    if per_ct_lab is None:
        return float(per_ct_nat)
    return float(rarity * per_ct_nat + (1.0 - rarity) * per_ct_lab)

def _color_multiplier(species: str, color: Optional[str]) -> float:
    if not color: return 1.0
    cm = COLOR_PRICE_MULTIPLIERS.get(species, {})
    mult = cm.get(str(color).lower())
    return float(mult) if (mult is not None and not (isinstance(mult, float) and np.isnan(mult))) else 1.0

def _transparency_match_score(species: str, t: Optional[str]) -> float:
    if not t: return 1.0
    prefs = TRANSPARENCY_HINTS.get(species, ["clear", "semi-translucent", "opaque"])
    return 1.0 if t in prefs else 0.6

def color_match_score(_stone_name: str, _selected_color: Optional[str]) -> float:
    return 1.0

def density_match_score(density_cm3: float, rng: Tuple[float, float]) -> float:
    lo, hi = rng
    if lo <= density_cm3 <= hi:
        return 1.0
    width = max(hi - lo, 1e-6)
    d = (lo - density_cm3) if density_cm3 < lo else density_cm3 - hi
    return max(0.0, 1.0 - (d / (width * 1.5)))

def _diamond_origin_adjust(name: str, pref: Optional[str]) -> float:
    if pref is None: return 1.0
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
    if not color: return 1.0
    g = GEM_DB[(GEM_DB["species"] == species) & (GEM_DB["color"].str.lower() == str(color).lower())]
    if g.empty: return 1.0
    qty_color = float(g["quantity_ct"].iloc[0])
    species_qty = GEM_DB[GEM_DB["species"] == species]["quantity_ct"]
    if len(species_qty) == 0: return 1.0
    ref_qty = float(np.nanmedian(species_qty))
    if qty_color <= 0 or ref_qty <= 0: return 1.0
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

def single_price_estimate(carats: float, color: Optional[str], transparency: Optional[str],
                          top_candidates: List[Tuple[str, Tuple[float,float], float, bool, float]]) -> float:
    if carats <= 0: return 0.0
    total = 0.0
    for row in top_candidates:
        species = row[0]; w = float(row[4])
        per_ct = _base_price_blended(species, color)
        Pi = _size_premium(species, carats) * per_ct * carats
        Pi = _apply_transparency_price(species, transparency, Pi)
        Pi = _apply_supply_price(species, color, Pi)
        total += w * Pi
    return float(total)

def _region_factor_for_species(st, species: str, region: str) -> float:
    b = st.session_state.get("base_region_factors", REGION_FACTORS_DEFAULT.copy())
    overrides = st.session_state.get("region_overrides_df", None)
    base = float(b.get(region, 1.0))
    if overrides is not None and not overrides.empty:
        row = overrides[overrides["Species"] == species]
        if not row.empty:
            val = row.iloc[0][region]
            try:
                if pd.notna(val): return float(val)
            except Exception:
                pass
    return base

def regionalize_prices(st, base_price_usd: float, species: str) -> Dict[str, float]:
    regions = ["North America", "Western Europe", "Asia"]
    return {r: float(base_price_usd) * _region_factor_for_species(st, species, r) for r in regions}

def compute_six_prices(st, species: str, carats: float, color: Optional[str], transparency: Optional[str],
                       top_candidates: List[Tuple[str, Tuple[float,float], float, bool, float]]):
    # Base (no region): weighted price of top candidates
    base = single_price_estimate(carats, color, transparency, top_candidates)
    no_energy = regionalize_prices(st, base, species)  # 3 prices (no energy)
    energy = energy_costs_by_region(st, species, carats)  # 3 energy costs
    with_energy = {r: no_energy.get(r, 0.0) + energy.get(r, 0.0) for r in no_energy.keys()}
    return no_energy, energy, with_energy
