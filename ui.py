from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

from .constants import REGION_FACTORS_DEFAULT, REGION_OVERRIDES_DEFAULT, CATALOG_DENSITY_RANGES
from .utils import fmt_usd
from .energy import init_energy_state, get_energy_prices
from .metals import get_spot_prices

def init_region_state():
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

def title():
    st.title("Industry Standard Gem Valuer v25 (Modular)")
    st.caption("3D geometry • DB pricing • density • regional editors • energy six-price breakdown • sticky order")

def sidebar_market_and_inputs():
    # Preset
    with st.sidebar.expander("Market preset (Retail vs Wholesale)", expanded=True):
        preset = st.radio("Preset", ["Retail (MSRP)", "Wholesale"], index=0, horizontal=True)
        st.session_state["market_preset_factor"] = {"Retail (MSRP)": 1.00, "Wholesale": 0.70}[preset]
        st.caption(f"Applied factor ×{st.session_state['market_preset_factor']:.2f} to prices.")

    # Region editor
    init_region_state()
    with st.sidebar.expander("Regional multipliers (live edit)", expanded=False):
        b = st.session_state["base_region_factors"]
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

    # Energy editor
    init_energy_state(st)
    with st.sidebar.expander("Energy: electricity prices & per‑carat kWh (artificial)", expanded=True):
        ep = st.session_state["energy_prices_usd_per_kwh"].copy()
        ep["North America" ] = st.number_input("Electricity price North America ($/kWh)",  value=float(ep["North America"]),  step=0.001, format="%.4f")
        ep["Western Europe"] = st.number_input("Electricity price Western Europe ($/kWh)", value=float(ep["Western Europe"]), step=0.001, format="%.4f")
        ep["Asia"          ] = st.number_input("Electricity price Asia ($/kWh)",           value=float(ep["Asia"]),           step=0.001, format="%.4f")
        st.session_state["energy_prices_usd_per_kwh"] = ep

        edited = st.data_editor(
            st.session_state["energy_kwh_per_ct_overrides"],
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Species": st.column_config.TextColumn(disabled=True),
                "kWh per ct (artificial)": st.column_config.NumberColumn(format="%.2f"),
            },
            key="energy_kwh_per_ct_editor",
        )
        st.session_state["energy_kwh_per_ct_overrides"] = edited

    # Metals
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
            st.caption("Will fetch via Yahoo Finance when available.")
