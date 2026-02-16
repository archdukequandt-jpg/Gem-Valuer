from typing import Dict, Optional
import numpy as np
import pandas as pd
from .constants import ENERGY_PRICE_USD_PER_KWH_DEFAULT, SYNTH_ENERGY_LIBRARY

def init_energy_state(st):
    if "energy_prices_usd_per_kwh" not in st.session_state:
        st.session_state["energy_prices_usd_per_kwh"] = ENERGY_PRICE_USD_PER_KWH_DEFAULT.copy()
    if "energy_kwh_per_ct_overrides" not in st.session_state:
        rows = []
        for species, row in SYNTH_ENERGY_LIBRARY.items():
            rows.append({"Species": species, "kWh per ct (artificial)": float(row["kwh_per_ct"])})
        st.session_state["energy_kwh_per_ct_overrides"] = pd.DataFrame(rows, columns=["Species", "kWh per ct (artificial)"])

def get_energy_prices(st) -> Dict[str, float]:
    return st.session_state.get("energy_prices_usd_per_kwh", ENERGY_PRICE_USD_PER_KWH_DEFAULT)

def get_kwh_per_ct_for_species(st, species: str) -> Optional[float]:
    df = st.session_state.get("energy_kwh_per_ct_overrides", None)
    if df is None or df.empty:
        return float(SYNTH_ENERGY_LIBRARY.get(species, {}).get("kwh_per_ct")) if species in SYNTH_ENERGY_LIBRARY else None
    row = df[df["Species"] == species]
    if not row.empty and pd.notna(row.iloc[0]["kWh per ct (artificial)"]):
        return float(row.iloc[0]["kWh per ct (artificial)"])
    # If natural diamond, map to lab diamond equivalent
    if species == "Diamond — Natural":
        return float(SYNTH_ENERGY_LIBRARY.get("Diamond — Lab", {}).get("kwh_per_ct", np.nan))
    return None

def energy_costs_by_region(st, species: str, carats: float) -> Dict[str, float]:
    kwh_ct = get_kwh_per_ct_for_species(st, species)
    if kwh_ct is None or carats <= 0:
        return {}
    prices = get_energy_prices(st)
    return {r: float(kwh_ct * carats * float(p)) for r, p in prices.items()}
