from typing import Tuple
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .constants import SHAPE_FACTORS, DEFAULT_DEPTH_RATIO_BY_CUT

def _normalize_cut(cut: str) -> str:
    for key in SHAPE_FACTORS.keys():
        if key.lower() == cut.lower():
            return key
    raise ValueError(f"Unsupported cut: {cut}")

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

def render_3d(cut: str, L: float, W: float, H: float,
              table_pct: float, culet_pct: float, girdle_pct: float,
              crown_angle_deg: float, pavilion_angle_deg: float,
              color_rgb=(200,200,200)):
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
    # Always return a Matplotlib figure for snapshotting
    fig = plt.figure(figsize=(5.8, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.95,
                    color=(color_rgb[0]/255.0, color_rgb[1]/255.0, color_rgb[2]/255.0))
    ax.set_box_aspect((max(L,1e-6), max(W,1e-6), max(H,1e-6)))
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(elev=20, azim=45)
    return fig
