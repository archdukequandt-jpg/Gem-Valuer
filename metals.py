from typing import Dict, Optional
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

from .constants import TROY_OUNCE_TO_GRAM

def _fetch_spot_yf(ticker: str) -> Optional[float]:
    try:
        if not HAVE_YF: return None
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is None or hist.empty: return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None

def get_spot_prices(st) -> Dict[str, Dict[str, float]]:
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
