from typing import Dict, Tuple

# Units / conversions
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

# Default depth ratio if no height provided
DEFAULT_DEPTH_RATIO_BY_CUT: Dict[str, float] = {
    "Diamond": 0.61, "Round": 0.61, "Oval": 0.60, "Princess Cut": 0.68, "Emerald Cut": 0.65,
    "Antique Cushion": 0.64, "Marquise": 0.60, "Pear": 0.62, "Trillant": 0.55,
    "Square": 0.68, "Octagon": 0.65, "Baguette": 0.60, "Tapered Baguette": 0.58,
    "Heart Shape": 0.62, "Briolette": 0.55, "Cabochon": 0.52,
}

# Compact species catalog — density ranges (g/cm³)
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

# Regional price factors (defaults editable in UI)
REGION_FACTORS_DEFAULT = {"North America": 1.00, "Western Europe": 1.08, "Asia": 0.97}
REGION_OVERRIDES_DEFAULT = {
    "Diamond — Lab": {"North America": 0.92, "Western Europe": 0.95, "Asia": 0.85},
    "Simulant — CZ/YAG/GGG/Glass": {"North America": 0.85, "Western Europe": 0.90, "Asia": 0.80},
    "Jade (Jadeite/Nephrite)": {"North America": 0.95, "Western Europe": 1.00, "Asia": 1.40},
    "Opal": {"North America": 1.15, "Western Europe": 1.05, "Asia": 0.95},
    "Corundum (Ruby/Sapphire)": {"North America": 1.00, "Western Europe": 1.05, "Asia": 1.10},
}

# Rarity index for blended pricing
RARITY_INDEX = {
    "Diamond — Natural": 0.85, "Diamond — Lab": 0.35,
    "Simulant — Moissanite": 0.10, "Simulant — CZ/YAG/GGG/Glass": 0.05,
    "Corundum (Ruby/Sapphire)": 0.70, "Spinel": 0.60,
    "Beryl (Emerald/Aquamarine/Morganite)": 0.50, "Quartz & Chalcedony": 0.05,
    "Garnet (group)": 0.35, "Tourmaline": 0.45, "Topaz": 0.20, "Zircon": 0.40,
    "Jade (Jadeite/Nephrite)": 0.70, "Peridot (Olivine)": 0.25, "Chrysoberyl (incl. Alexandrite)": 0.85,
    "Diopside": 0.18, "Sphene (Titanite)": 0.60, "Apatite/Iolite/Fluorite": 0.15,
    "Opal": 0.35, "Opaque Ornamentals (Lapis/Turquoise/Sodalite/Malachite/Rhodochrosite/Hematite/Obsidian)": 0.10,
}

# Energy defaults
ENERGY_PRICE_USD_PER_KWH_DEFAULT = {
    "North America": 0.0853,
    "Western Europe": 0.18,
    "Asia": 0.10,
}

# Synthetic energy library (kWh per carat) — new canonical dictionary for "most likely" estimate
SYNTH_ENERGY_LIBRARY = {
    "Diamond — Lab": {"kwh_per_ct": 215.0, "method": "CVD/HPHT", "note": "midpoint literature", "confidence": 0.7},
    "Simulant — Moissanite": {"kwh_per_ct": 150.0, "method": "PVT growth", "note": "estimate", "confidence": 0.6},
    "Simulant — CZ/YAG/GGG/Glass": {"kwh_per_ct": 25.0, "method": "Skull/CCIM melt", "note": "estimate", "confidence": 0.6},
    "Corundum (Ruby/Sapphire)": {"kwh_per_ct": 30.0, "method": "Verneuil/Kyropoulos", "note": "estimate", "confidence": 0.6},
    "Spinel": {"kwh_per_ct": 30.0, "method": "Flame fusion", "note": "estimate", "confidence": 0.6},
    "Chrysoberyl (incl. Alexandrite)": {"kwh_per_ct": 80.0, "method": "Hydrothermal/flux", "note": "estimate", "confidence": 0.5},
    "Quartz & Chalcedony": {"kwh_per_ct": 12.0, "method": "Hydrothermal", "note": "estimate", "confidence": 0.6},
    "Opal": {"kwh_per_ct": 8.0, "method": "Polymer/precip.", "note": "estimate", "confidence": 0.4},
    # Map "Diamond — Natural" to lab diamond energy when asking hypothetical synthetic equivalent
    "Diamond — Natural": {"kwh_per_ct": 215.0, "method": "CVD/HPHT (equivalent)", "note": "equiv to lab", "confidence": 0.5},
}

# Transparency hints
TRANSPARENCY_HINTS = {
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

# Pricing fallbacks to avoid NaNs
PRICING_FALLBACKS = {
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
