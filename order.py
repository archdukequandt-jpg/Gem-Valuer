from typing import Dict, List, Tuple
import pandas as pd

def init_order_state(st):
    if "order_items" not in st.session_state:
        st.session_state["order_items"] = []
    if "order_seq" not in st.session_state:
        st.session_state["order_seq"] = 1

def add_to_order(st, item: Dict):
    init_order_state(st)
    item = dict(item)
    item["id"] = int(st.session_state["order_seq"])
    st.session_state["order_seq"] += 1
    st.session_state["order_items"].append(item)

def remove_from_order(st, item_id: int):
    init_order_state(st)
    st.session_state["order_items"] = [it for it in st.session_state["order_items"] if it["id"] != item_id]

def order_totals(st):
    init_order_state(st)
    init_sum = 0.0
    final_sum = 0.0
    for it in st.session_state["order_items"]:
        initial = float(it.get("estimate_usd", 0.0))
        override = it.get("override_usd", None)
        final_val = float(override) if (override is not None and override >= 0) else initial
        init_sum += initial
        final_sum += final_val
    return init_sum, final_sum

def export_order_csv(st) -> bytes:
    rows = []
    for it in st.session_state.get("order_items", []):
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
            # Six-price breakdown for CSV convenience
            "price_no_energy_NA": it.get("prices_no_energy", {}).get("North America", None),
            "price_no_energy_WE": it.get("prices_no_energy", {}).get("Western Europe", None),
            "price_no_energy_Asia": it.get("prices_no_energy", {}).get("Asia", None),
            "energy_cost_NA": it.get("energy_costs", {}).get("North America", None),
            "energy_cost_WE": it.get("energy_costs", {}).get("Western Europe", None),
            "energy_cost_Asia": it.get("energy_costs", {}).get("Asia", None),
            "price_with_energy_NA": it.get("prices_with_energy", {}).get("North America", None),
            "price_with_energy_WE": it.get("prices_with_energy", {}).get("Western Europe", None),
            "price_with_energy_Asia": it.get("prices_with_energy", {}).get("Asia", None),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")
