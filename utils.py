import numpy as np

def round6(x: float) -> float:
    try:
        return float(np.round(float(x), 6))
    except Exception:
        return x

def fmt_usd(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)
