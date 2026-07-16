import os, math, threading, warnings
import numpy as np
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


#    50,000 evenly spread test directions (a Fibonacci
#    lattice). cam_basis() turns yaw/pitch/roll into the camera's three axes.
# ------------------------------------------------------------------

MC_SAMPLES = 50000

def _fibonacci_sphere(n):
    phi = (1 + np.sqrt(5)) / 2
    pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(1 - z * z)
        t = 2 * np.pi * i / phi
        pts[i] = [r * np.cos(t), r * np.sin(t), z]
    return pts

UNIT_DIRS = _fibonacci_sphere(MC_SAMPLES)

# Precompute elevation angles for polar zone analysis
_ELEV_DEG = np.degrees(np.arcsin(np.clip(UNIT_DIRS[:, 2], -1, 1)))


def cam_basis(yaw_deg, pitch_deg, roll_deg=0.0):
    yr, pr = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(yr)*np.cos(pr), np.sin(yr)*np.cos(pr), np.sin(pr)])
    up_w = np.array([0, 0, 1.0])
    if abs(np.dot(fwd, up_w)) > 0.99:
        up_w = np.array([0, 1, 0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    # Apply roll: rotate right and up around fwd axis
    if roll_deg != 0.0:
        rr = np.radians(roll_deg)
        cr, sr = np.cos(rr), np.sin(rr)
        right_new = cr * right + sr * up
        up_new = -sr * right + cr * up
        right, up = right_new, up_new
    return fwd, right, up


def make_ring(n, lat_deg, offset_deg=0.0):
    if n == 0:
        return []
    sp = 360.0 / n
    return [(offset_deg + i * sp, lat_deg) for i in range(n)]


def parse_config(config_str):
    return [int(x.strip()) for x in config_str.split('-')]


def build_orientations_with_radii(ring_counts, latitudes, staggers,
                                   ring_radii, pitch_offsets=None):
    """Build camera orientations with per-ring radius and optional pitch offsets.
    Returns list of (yaw, pitch, radius_mm) tuples."""
    oris = []
    if pitch_offsets is None:
        pitch_offsets = [0.0] * len(ring_counts)
    for i, (n, lat, stg, R) in enumerate(
            zip(ring_counts, latitudes, staggers, ring_radii)):
        poff = pitch_offsets[i] if i < len(pitch_offsets) else 0.0
        actual_pitch = lat + poff
        if n == 1 and abs(lat) > 85:
            oris.append((0.0, actual_pitch, R))
        else:
            for yaw, pitch in make_ring(n, actual_pitch, stg):
                oris.append((yaw, pitch, R))
    return oris


def auto_layout(config_str, fov_h=80, fov_v=65):
    counts = parse_config(config_str)
    n_rings = len(counts)
    is_pole_bot = (counts[0] == 1)
    is_pole_top = (counts[-1] == 1)
    lats = []
    if n_rings == 1:
        lats = [0.0]
    elif n_rings == 2:
        if is_pole_bot:
            lats = [-90.0, 0.0]
        elif is_pole_top:
            lats = [0.0, 90.0]
        else:
            lats = [-30.0, 30.0]
    else:
        inner_start = 1 if is_pole_bot else 0
        inner_end = n_rings - 1 if is_pole_top else n_rings
        n_inner = inner_end - inner_start
        for i in range(n_rings):
            if i == 0 and is_pole_bot:
                lats.append(-90.0)
            elif i == n_rings - 1 and is_pole_top:
                lats.append(90.0)
            else:
                idx = i - inner_start
                if n_inner == 1:
                    lats.append(0.0)
                else:
                    lat = -90 + 180 * (idx + 1) / (n_inner + 1)
                    lats.append(round(lat, 1))

    staggers = [0.0] * n_rings
    for i in range(1, n_rings):
        if counts[i] > 1 and counts[i - 1] > 1:
            staggers[i] = 360.0 / (2 * counts[i])

    return counts, lats, staggers

#    A test direction is "seen" if it lands inside a camera's rectangular
#    field of view. Cameras sit on the body surface (radius R), so their
#    position offset is included. cc[i] = number of cameras seeing direction i.
# ------------------------------------------------------------------

def compute_coverage_nonuniform(oris_with_R, fov_h, fov_v, eval_dist_mm):
    """Coverage computation supporting per-camera radius and optional roll.
    oris_with_R: list of (yaw, pitch, radius_mm) or (yaw, pitch, radius_mm, roll_deg)"""
    hth = np.tan(np.radians(fov_h / 2))
    htv = np.tan(np.radians(fov_v / 2))
    M = len(UNIT_DIRS)
    cc = np.zeros(M, dtype=np.int32)
    cam_vis = []  # per-camera visibility mask

    for ori in oris_with_R:
        if len(ori) == 4:
            yaw, pitch, R, roll = ori
        else:
            yaw, pitch, R = ori
            roll = 0.0
        fwd, right, up = cam_basis(yaw, pitch, roll)
        cam_pos = fwd * R
        eval_pts = UNIT_DIRS * (R + eval_dist_mm)
        to_pt = eval_pts - cam_pos
        pf = to_pt @ fwd
        inf = pf > 0
        pfs = np.where(inf, pf, 1.0)
        ah = np.abs((to_pt @ right) / pfs)
        av = np.abs((to_pt @ up) / pfs)
        vis = inf & (ah < hth) & (av < htv)
        cc += vis.astype(np.int32)
        cam_vis.append(vis)

    cov_pct = np.sum(cc > 0) / M * 100
    stereo_pct = np.sum(cc >= 2) / M * 100
    triple_pct = np.sum(cc >= 3) / M * 100
    return cov_pct, stereo_pct, triple_pct, cc, cam_vis

# ------------------------------------------------------------------
#    Fibonacci: golden-angle spiral formula (no optimization).
#    Thomson: cameras repel like charges; minimize total energy sum(1/d).
#    Tammes: maximize the minimum angle between any two cameras.
#    The ring scaffold (method 4) comes from auto_layout() in section 2.
# ------------------------------------------------------------------

def _xyz_to_yaw_pitch(x, y, z):
    """Convert unit-sphere point to (yaw_deg, pitch_deg) facing outward."""
    pitch = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    yaw = np.degrees(np.arctan2(y, x)) % 360
    return yaw, pitch


def _sph_to_xyz(theta, phi):
    """Spherical coords (theta=polar from +z, phi=azimuthal) to unit xyz."""
    st = np.sin(theta)
    return st * np.cos(phi), st * np.sin(phi), np.cos(theta)


def _pack_angles(thetas, phis):
    """Flatten N spherical coords into 1-D array for optimizer."""
    return np.concatenate([thetas, phis])


def _unpack_angles(x, n):
    """Unpack 1-D array into (thetas, phis)."""
    return x[:n], x[n:]


#  1. Fibonacci Spiral Placement

def fibonacci_spiral_orientations(n, R=250.0):
    """Place N cameras on Fibonacci spiral, each pointing radially outward.
    Returns list of (yaw_deg, pitch_deg, R_mm)."""
    golden = (1 + np.sqrt(5)) / 2
    oris = []
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r_xy = np.sqrt(max(0, 1 - z * z))
        t = 2 * np.pi * i / golden
        x, y = r_xy * np.cos(t), r_xy * np.sin(t)
        yaw, pitch = _xyz_to_yaw_pitch(x, y, z)
        oris.append((yaw, pitch, R))
    return oris


#  2. Thomson's Problem ‚Äî Minimize Coulomb Energy

def _thomson_energy_grad(x, n):
    """Coulomb energy E = Œ£_{i<j} 1/|r·µ¢‚àír‚±º| and gradient w.r.t. (theta, phi)."""
    thetas, phis = _unpack_angles(x, n)
    st = np.sin(thetas)
    ct = np.cos(thetas)
    sp = np.sin(phis)
    cp = np.cos(phis)

    # Cartesian positions
    X = st * cp
    Y = st * sp
    Z = ct

    energy = 0.0
    grad_theta = np.zeros(n)
    grad_phi = np.zeros(n)

    for i in range(n):
        dx = X[i] - X
        dy = Y[i] - Y
        dz = Z[i] - Z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        dist[i] = 1e30  # avoid self

        inv_d = 1.0 / dist
        inv_d3 = inv_d ** 3
        energy += np.sum(inv_d[i+1:])  # only i<j

        # Partial derivatives of position w.r.t. theta_i, phi_i
        dxi_dt = ct[i] * cp[i]
        dyi_dt = ct[i] * sp[i]
        dzi_dt = -st[i]

        dxi_dp = -st[i] * sp[i]
        dyi_dp = st[i] * cp[i]
        dzi_dp = 0.0

        # dE/d(theta_i) = -Œ£_{j‚â†i} (dx¬∑dxi_dt + dy¬∑dyi_dt + dz¬∑dzi_dt) / dist^3
        grad_theta[i] = -np.sum((dx * dxi_dt + dy * dyi_dt + dz * dzi_dt) * inv_d3)
        grad_phi[i] = -np.sum((dx * dxi_dp + dy * dyi_dp + dz * dzi_dp) * inv_d3)

    grad = np.concatenate([grad_theta, grad_phi])
    return energy, grad


def _thomson_single_start(args):
    """Run one Thomson optimization from a single starting point.
    args: (n, R, max_iter, seed_idx)
    Returns (orientations, energy)."""
    n, R, max_iter, seed_idx = args
    np.random.seed(12345 + seed_idx * 7919)  # deterministic per start ‚Äî reproducible runs

    if seed_idx == 0:
        # Start from Fibonacci spiral
        fib_oris = fibonacci_spiral_orientations(n, R)
        thetas0 = []
        phis0 = []
        for ori in fib_oris:
            yr, pr = np.radians(ori[0]), np.radians(ori[1])
            theta = np.pi / 2 - pr
            phi = yr
            thetas0.append(theta)
            phis0.append(phi)
    else:
        # Random start on the unit sphere
        thetas0 = []
        phis0 = []
        for _ in range(n):
            z = np.random.uniform(-1, 1)
            phi_r = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.clip(z, -1, 1))
            thetas0.append(theta)
            phis0.append(phi_r)

    x0 = _pack_angles(np.array(thetas0), np.array(phis0))

    result = minimize(
        _thomson_energy_grad, x0,
        args=(n,),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
    )

    thetas, phis = _unpack_angles(result.x, n)
    oris = []
    for theta, phi in zip(thetas, phis):
        x, y, z_val = _sph_to_xyz(theta, phi)
        yaw, pitch = _xyz_to_yaw_pitch(x, y, z_val)
        oris.append((yaw, pitch, R))

    return oris, result.fun


def thomson_solve(n, R=250.0, max_iter=5000, n_starts=100):
    """Solve Thomson's Problem for N points on a sphere of radius R.
    Uses parallel multi-start optimization with n_starts initial configs.
    Returns (orientations, energy).
    orientations: [(yaw, pitch, R), ...]"""
    args_list = [(n, R, max_iter, i) for i in range(n_starts)]
    max_workers = min(n_starts, os.cpu_count() or 4)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_thomson_single_start, args_list))
    except Exception:
        # Fall back to sequential if multiprocessing fails
        results = [_thomson_single_start(a) for a in args_list]

    # Select the result with the lowest energy
    best_oris, best_energy = min(results, key=lambda r: r[1])
    return best_oris, best_energy


#  3. Tammes Problem ‚Äî Maximize Minimum Distance

def _tammes_smooth_objective(x, n, k=50.0):
    """Smooth approximation using log-sum-exp of negative distances.
    Differentiable surrogate for min-distance maximization."""
    thetas, phis = _unpack_angles(x, n)
    st = np.sin(thetas)
    ct = np.cos(thetas)
    sp = np.sin(phis)
    cp = np.cos(phis)

    X = st * cp
    Y = st * sp
    Z = ct

    # Collect all pairwise distances
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            dz = Z[i] - Z[j]
            d = np.sqrt(dx*dx + dy*dy + dz*dz + 1e-16)
            dists.append(d)

    dists = np.array(dists)
    # Smooth min approximation: -1/k * log(Œ£ exp(-k*d))
    # Minimizing this maximizes the minimum distance
    log_sum = -1.0/k * np.log(np.sum(np.exp(-k * dists)) + 1e-30)
    return -log_sum  # we minimize, so negate to get "minimize neg min dist"


def _tammes_single_start(args):
    """Run one Tammes optimization with progressive sharpening from a single start.
    args: (n, R, max_iter, seed_idx)
    Returns (orientations, min_distance_on_unit_sphere)."""
    n, R, max_iter, seed_idx = args
    np.random.seed(12345 + seed_idx * 7919)  # deterministic per start ‚Äî reproducible runs

    if seed_idx == 0:
        # Start from Fibonacci spiral
        fib_oris = fibonacci_spiral_orientations(n, R)
        thetas0 = []
        phis0 = []
        for ori in fib_oris:
            yr, pr = np.radians(ori[0]), np.radians(ori[1])
            theta = np.pi / 2 - pr
            phi = yr
            thetas0.append(theta)
            phis0.append(phi)
    else:
        # Random start on the unit sphere
        thetas0 = []
        phis0 = []
        for _ in range(n):
            z = np.random.uniform(-1, 1)
            phi_r = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.clip(z, -1, 1))
            thetas0.append(theta)
            phis0.append(phi_r)

    x0 = _pack_angles(np.array(thetas0), np.array(phis0))

    # Progressive sharpening ‚Äî start smooth, increase k for better min approx
    for k_val in [20, 50, 100, 200]:
        result = minimize(
            _tammes_smooth_objective, x0,
            args=(n, k_val),
            method='L-BFGS-B',
            options={'maxiter': max_iter // 4, 'ftol': 1e-14}
        )
        x0 = result.x  # warm-start next sharpening level

    # Evaluate true min distance
    thetas, phis = _unpack_angles(result.x, n)
    st = np.sin(thetas)
    X = st * np.cos(phis)
    Y = st * np.sin(phis)
    Z = np.cos(thetas)

    min_d = 1e30
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
            if d < min_d:
                min_d = d

    oris = []
    for theta, phi in zip(thetas, phis):
        x, y, z_val = _sph_to_xyz(theta, phi)
        yaw, pitch = _xyz_to_yaw_pitch(x, y, z_val)
        oris.append((yaw, pitch, R))

    return oris, min_d


def tammes_solve(n, R=250.0, max_iter=5000, n_starts=100):
    """Solve Tammes Problem for N points via parallel multi-start optimization.
    Returns (orientations, min_distance_on_unit_sphere)."""
    args_list = [(n, R, max_iter, i) for i in range(n_starts)]
    max_workers = min(n_starts, os.cpu_count() or 4)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_tammes_single_start, args_list))
    except Exception:
        # Fall back to sequential if multiprocessing fails
        results = [_tammes_single_start(a) for a in args_list]

    # Select the result with the largest min distance
    best_oris, best_min_dist = max(results, key=lambda r: r[1])
    return best_oris, best_min_dist


#    All 16 cameras' yaw, pitch and roll are packed into one 48-number
#    vector x = [theta1..16 | phi1..16 | roll1..16]. L-BFGS-B follows the
#    gradient of the coverage score (partial derivatives approximated by
#    finite differences) and moves every camera at once.

def _fast_coverage(thetas, phis, R, fov_h, fov_v, eval_dist, rolls=None, dirs=None):
    """Fully vectorized coverage computation using matrix ops.
    No Python loops in hot path (except basis setup). All N cameras evaluated simultaneously.
    rolls: per-camera roll in radians (optional, default 0).
    dirs: direction set to test (defaults to the full UNIT_DIRS); pass a subsample to make
          the optimizer's gradient evals cheap. Returns (coverage%, stereo%, blind, cc)."""
    if dirs is None:
        dirs = UNIT_DIRS
    n = len(thetas)
    hth = np.tan(np.radians(fov_h / 2))
    htv = np.tan(np.radians(fov_v / 2))

    # Camera directions on unit sphere ‚Üí (n, 3)
    st = np.sin(thetas)
    ct = np.cos(thetas)
    sp = np.sin(phis)
    cp = np.cos(phis)

    # Forward vectors (n, 3) ‚Äî camera pointing direction
    # theta = polar angle from +z, phi = azimuthal
    # Convert to yaw/pitch style: fwd = [cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]
    # where pitch = pi/2 - theta, yaw = phi
    pitch = np.pi / 2 - thetas  # (n,)
    yaw = phis  # (n,)
    cp2 = np.cos(pitch)
    sp2 = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    fwd = np.column_stack([cy * cp2, sy * cp2, sp2])  # (n, 3)
    cam_pos = fwd * R  # (n, 3) ‚Äî camera positions on sphere

    # For each camera, compute right and up vectors
    up_world = np.array([0.0, 0.0, 1.0])
    # Handle pole cameras where fwd ‚âà ¬±z
    dots = np.abs(fwd @ up_world)
    alt_up = np.array([0.0, 1.0, 0.0])

    # Right = fwd √ó up_world (or alt_up for poles)
    rights = np.zeros((n, 3))
    ups = np.zeros((n, 3))
    for i in range(n):
        uw = alt_up if dots[i] > 0.99 else up_world
        r = np.cross(fwd[i], uw)
        r /= np.linalg.norm(r) + 1e-30
        u = np.cross(r, fwd[i])
        u /= np.linalg.norm(u) + 1e-30
        # Apply roll if provided
        if rolls is not None and rolls[i] != 0.0:
            cr = np.cos(rolls[i])
            sr = np.sin(rolls[i])
            r_new = cr * r + sr * u
            u_new = -sr * r + cr * u
            r, u = r_new, u_new
        rights[i] = r
        ups[i] = u

    # Eval points: (M, 3) ‚Äî directions on sphere at eval distance
    eval_pts = dirs * (R + eval_dist)  # (M, 3)
    M = len(dirs)

    # FULLY VECTORIZED: compute visibility for ALL cameras at once
    # to_pt[cam, sample, :] = eval_pts[sample] - cam_pos[cam]  ‚Üí (n, M, 3)
    # But (n, M, 3) with n=16, M=50000 is only 2.4M floats = 19MB ‚Äî fits easily
    to_pt = eval_pts[np.newaxis, :, :] - cam_pos[:, np.newaxis, :]  # (n, M, 3)

    # Project onto camera axes using einsum
    pf = np.einsum('nmi,ni->nm', to_pt, fwd)      # (n, M) ‚Äî forward projection
    pr = np.einsum('nmi,ni->nm', to_pt, rights)    # (n, M) ‚Äî right projection
    pu = np.einsum('nmi,ni->nm', to_pt, ups)       # (n, M) ‚Äî up projection

    # Visibility: forward > 0 AND within FOV cone
    in_front = pf > 0
    pf_safe = np.where(in_front, pf, 1.0)
    ah = np.abs(pr / pf_safe)
    av = np.abs(pu / pf_safe)
    vis = in_front & (ah < hth) & (av < htv)  # (n, M) boolean

    # Count coverage
    cc = vis.astype(np.int32).sum(axis=0)  # (M,) ‚Äî how many cameras see each point
    
    cov_pct = np.sum(cc > 0) / M * 100
    stereo_pct = np.sum(cc >= 2) / M * 100
    blind_count = np.sum(cc == 0)

    return cov_pct, stereo_pct, blind_count, cc


def _fast_objective(x, n, R, fov_h, fov_v, eval_dist, blind_weight=0.5, dirs=None):
    """Objective function: maximize coverage, penalize blind spots aggressively.
    Supports both 2N (theta+phi) and 3N (theta+phi+roll) variable packing.
    blind_weight controls how much to penalize blind spots vs reward coverage.
    dirs: optional subsampled direction set (for cheap gradient evals)."""
    has_roll = len(x) == 3 * n
    thetas = x[:n]
    phis = x[n:2*n]
    rolls = x[2*n:3*n] if has_roll else None
    cov, stereo, blind, cc = _fast_coverage(thetas, phis, R, fov_h, fov_v, eval_dist, rolls, dirs)

    # Score: coverage + stereo bonus
    score = cov + 0.3 * stereo

    # AGGRESSIVE blind spot penalty (fixes problem #1 ‚Äî weak gradient)
    # Penalize exponentially: each blind spot matters MORE as we get closer to full coverage
    blind_frac = blind / len(cc)
    penalty = blind_weight * (blind_frac * 100) ** 1.5  # superlinear penalty

    # Uniformity bonus: penalize cameras that are too close together
    st = np.sin(thetas)
    X = st * np.cos(phis)
    Y = st * np.sin(phis)
    Z = np.cos(thetas)
    pts = np.column_stack([X, Y, Z])
    dists = np.sqrt(((pts[:, np.newaxis] - pts[np.newaxis, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, 999)
    min_nn = dists.min(axis=1)
    # Gentle spacing penalty: only discourage *near-coincident* cameras (< ~16¬∞), so the
    # optimizer is free to CLUSTER where coverage needs it. Coverage ‚â† uniformity ‚Äî the old
    # 29¬∞ threshold blocked exactly the clustering that closes rectangular-FOV seam gaps.
    uniformity_penalty = np.sum(np.maximum(0, 0.28 - min_nn)) * 1.5

    return -(score - penalty - uniformity_penalty)
    

BG='white'; BG2='#f0f0f0'; FG='black'; ACCENT='#d5d5d5'
HL='#333333'; GN='#00897b'; YL='#b8860b'; BL='#1565c0'; DM='#666666'
RCOLS=['#e94560','#ffd93d','#00d4aa','#4fc3f7','#bb86fc','#ff7043','#26c6da']

HM_COLORS=['#1a1a2e','#e94560','#ff7043','#ffd93d','#00d4aa','#4fc3f7']
HM_CMAP=ListedColormap(HM_COLORS)
HM_BOUNDS=[0,0.5,1.5,2.5,3.5,4.5,10]
HM_NORM=BoundaryNorm(HM_BOUNDS,HM_CMAP.N)

# Pre-generate 3000 uniform sphere points (Fibonacci lattice) for dot sphere
_N_DOTS=10000
_golden=(1+np.sqrt(5))/2
_i_dots=np.arange(_N_DOTS)
_dot_theta=2*np.pi*_i_dots/_golden
_dot_phi=np.arccos(1-2*(_i_dots+0.5)/_N_DOTS)
DOT_SPHERE=np.column_stack([np.sin(_dot_phi)*np.cos(_dot_theta),
                             np.sin(_dot_phi)*np.sin(_dot_theta),
                             np.cos(_dot_phi)])


def build_tab6(app, parent_frame):
    bar=tk.Frame(parent_frame,bg=BG2,pady=6,padx=10); bar.pack(fill='x')
    tk.Label(bar,text="8-WAY COMPARISON",bg=BG2,fg=HL,
             font=('Helvetica',13,'bold')).pack(side='left',padx=(0,16))
    tk.Label(bar,text="Eval Dist:",bg=BG2,fg=FG,font=('Helvetica',11)).pack(side='left')
    app.cmp_dist_var=tk.StringVar(value='2000')
    for d in ['1500','2000']:
        tk.Radiobutton(bar,text=f"{d}mm",variable=app.cmp_dist_var,value=d,
                       bg=BG2,fg=GN,selectcolor=ACCENT,activebackground=BG2,
                       font=('Helvetica',10,'bold')).pack(side='left',padx=4)
    # color scheme: detailed overlap counts, or the blue/green/black report style
    tk.Label(bar,text="  Colors:",bg=BG2,fg=FG,font=('Helvetica',11)).pack(side='left')
    app.cmp_scheme=tk.StringVar(value='overlap')
    def _reskin():
        if app._cmp_configs and getattr(app,'_cmp_meta',None):
            R,fh,fv,eval_d,el=app._cmp_meta
            draw_comparison(app,app._cmp_configs,R,fh,fv,eval_d,el)
    for txt,val in [("Overlap","overlap"),("Blue/Green/Black","bgb")]:
        tk.Radiobutton(bar,text=txt,variable=app.cmp_scheme,value=val,command=_reskin,
                       bg=BG2,fg=BL,selectcolor=ACCENT,activebackground=BG2,
                       font=('Helvetica',10,'bold')).pack(side='left',padx=4)
    tk.Button(bar,text="RUN 8-WAY",command=lambda:run_comparison(app),
              bg=HL,fg='black',font=('Helvetica',12,'bold'),relief='flat',
              padx=14,cursor='hand2').pack(side='left',padx=16)
    tk.Label(bar,text="(click any heatmap to inspect)",bg=BG2,fg=DM,
             font=('Helvetica',9,'italic')).pack(side='left',padx=8)
    app.cmp_status=tk.StringVar(value="Ready")
    tk.Label(bar,textvariable=app.cmp_status,bg=BG2,fg=YL,
             font=('Menlo',10,'italic')).pack(side='right')
    app.fig_cmp=Figure(figsize=(18,9),facecolor=BG)
    app.c_cmp=FigureCanvasTkAgg(app.fig_cmp,master=parent_frame)
    app.c_cmp.get_tk_widget().pack(fill='both',expand=True)
    app._cmp_configs=[]; app._cmp_axes=[]; app._cmp_click_cid=None


def run_comparison(app):
    try:
        R=float(app.base_r.get())
        fh=float(app.fov_h.get()); fv=float(app.fov_v.get())
        eval_d=int(app.cmp_dist_var.get())
    except:
        app.cmp_status.set("ERROR: bad input"); return
    app.cmp_status.set("Running 8 configs..."); app.root.update()

    def work():
        import time
        try:
            t0=time.time(); configs=[]
            def st(m): app.root.after(0,lambda:app.cmp_status.set(m))

            st("[1/8] Ring 1-4-6-4-1...")
            c,l,s=auto_layout('1-4-6-4-1',fh,fv)
            ro=build_orientations_with_radii(c,l,s,[R]*len(c))
            _,_,_,cc,_=compute_coverage_nonuniform(ro,fh,fv,eval_d)
            configs.append(('1-4-6-4-1\n(raw)',cc,np.sum(cc>0)/len(cc)*100,ro))

            st("[2/8] Fibonacci...")
            fo=fibonacci_spiral_orientations(16,R)
            _,_,_,cc,_=compute_coverage_nonuniform(fo,fh,fv,eval_d)
            configs.append(('Fibonacci\n(raw)',cc,np.sum(cc>0)/len(cc)*100,fo))

            st("[3/8] Thomson...")
            # these raw configs are only SEEDS for the roll optimizer, so a handful of
            # restarts is plenty ‚Äî 100√ó was wasted work.
            to,_=thomson_solve(16,R,max_iter=1500,n_starts=10)
            _,_,_,cc,_=compute_coverage_nonuniform(to,fh,fv,eval_d)
            configs.append(('Thomson\n(raw)',cc,np.sum(cc>0)/len(cc)*100,to))

            st("[4/8] Tammes...")
            ta,_=tammes_solve(16,R,max_iter=1500,n_starts=12)
            _,_,_,cc,_=compute_coverage_nonuniform(ta,fh,fv,eval_d)
            configs.append(('Tammes\n(raw)',cc,np.sum(cc>0)/len(cc)*100,ta))

            for i,(nm,raw) in enumerate([('1-4-6-4-1',ro),('Fibonacci',fo),('Thomson',to),('Tammes',ta)]):
                st(f"[{5+i}/8] Optimizing {nm} (multi-start roll)...")
                op = _multistart_roll_opt(raw, fh, fv, eval_d, R)
                _,_,_,cc,_=compute_coverage_nonuniform(op,fh,fv,eval_d)
                configs.append((f'{nm}\n(optimized)',cc,np.sum(cc>0)/len(cc)*100,op))

            el=time.time()-t0
            app.root.after(0,lambda:draw_comparison(app,configs,R,fh,fv,eval_d,el))
        except Exception as e:
            import traceback; traceback.print_exc()
            app.root.after(0,lambda:app.cmp_status.set(f"ERROR: {e}"))
    threading.Thread(target=work,daemon=True).start()


def _lbfgs_worker(args):
    """Refine one full (theta,phi,roll) vector with L-BFGS-B (module-level for pickling).
    Uses a SUBSAMPLED direction set so each finite-difference gradient eval is ~6√ó cheaper;
    the caller re-scores winners at full resolution."""
    x0, n, R, fh, fv, eval_d, max_iter = args
    from scipy.optimize import minimize as sci_minimize
    import numpy as np
    sub = UNIT_DIRS[::6]   # ~8.3k dirs for the gradient search (full set is 50k)
    bounds = ([(0.01, np.pi - 0.01)] * n +
              [(None, None)] * n +
              [(-np.pi/2, np.pi/2)] * n)
    result = sci_minimize(_fast_objective, np.asarray(x0), args=(n, R, fh, fv, eval_d, 0.5, sub),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-11, 'eps': 5e-5})
    return (result.x.copy(), result.fun)


def _parmap(fn, args_list):
    """Parallel map with a sequential fallback (processes for CPU-bound L-BFGS-B).
    Workers are capped so many 50k-point coverage tensors don't exhaust memory."""
    try:
        workers = min(len(args_list), max(2, (os.cpu_count() or 4) // 2), 6)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, args_list))
    except Exception:
        return [fn(a) for a in args_list]


def _multistart_roll_opt(init_oris, fh, fv, eval_d, R,
                         coarse_iters=250, refine_iters=4000,
                         n_explore=24, n_refine=4, seed=1234):
    """Two-stage roll-aware optimizer (faster AND higher quality than flat multi-start):
      Stage A ‚Äî EXPLORE: many diverse roll seeds with a cheap, shallow L-BFGS-B to rank basins.
      Stage B ‚Äî REFINE:  deep L-BFGS-B on only the best few basins.
    ~20√ó fewer total iterations than 100 seeds √ó 5000, and refinement is concentrated where
    it pays off. Reproducible (fixed seed)."""
    n = len(init_oris)
    thetas_base = np.array([np.pi/2 - np.radians(o[1]) for o in init_oris])
    phis_base = np.array([np.radians(o[0]) for o in init_oris])

    rng = np.random.default_rng(seed)
    rolls = [np.zeros(n),
             np.full(n, np.pi/8), np.full(n, -np.pi/8),
             np.full(n, np.pi/4), np.full(n, -np.pi/4),
             np.array([(-1)**i * np.pi/8 for i in range(n)]),
             np.array([(-1)**i * np.pi/4 for i in range(n)])]
    while len(rolls) < n_explore:
        rolls.append(rng.uniform(-np.pi/3, np.pi/3, n))
    rolls = rolls[:n_explore]

    # STAGE A ‚Äî explore (shallow)
    explore = [(np.concatenate([thetas_base, phis_base, rs]), n, R, fh, fv, eval_d, coarse_iters)
               for rs in rolls]
    ranked = sorted(_parmap(_lbfgs_worker, explore), key=lambda t: t[1])   # lower obj = better

    # STAGE B ‚Äî refine (deep) the best basins
    refine = [(ranked[k][0], n, R, fh, fv, eval_d, refine_iters)
              for k in range(min(n_refine, len(ranked)))]
    best_x = min(_parmap(_lbfgs_worker, refine), key=lambda t: t[1])[0]

    # Convert to orientations
    thetas = best_x[:n]
    phis = best_x[n:2*n]
    rolls = best_x[2*n:3*n]
    oris = []
    for theta, phi, roll_r in zip(thetas, phis, rolls):
        st = np.sin(theta)
        x, y, z = st * np.cos(phi), st * np.sin(phi), np.cos(theta)
        pitch = np.degrees(np.arcsin(np.clip(z, -1, 1)))
        yaw = np.degrees(np.arctan2(y, x)) % 360
        oris.append((yaw, pitch, R, np.degrees(roll_r)))
    return oris


def _blind_stats(cc):
    """(max equator blind arc in deg, largest connected blind region in % of sphere)."""
    M=len(cc); blind_idx=np.where(cc==0)[0]
    if len(blind_idx)==0:
        return 0.0, 0.0
    # max equator gap: longest contiguous blind arc in the |lat|<2 deg band
    band=(np.abs(UNIT_DIRS[:,2])<0.035)&(cc==0)
    gap=0.0
    if band.any():
        lons=np.sort(np.degrees(np.arctan2(UNIT_DIRS[band,1],UNIT_DIRS[band,0])))
        splits=np.where(np.diff(lons)>3.0)[0]
        runs=np.split(lons,splits+1)
        if len(runs)>1 and (lons[0]+360-lons[-1])<=3.0:      # circular wrap
            runs[0]=np.concatenate([runs[-1]-360,runs[0]]); runs=runs[:-1]
        gap=max(float(r[-1]-r[0])+0.9 for r in runs)          # +lattice spacing
    # largest connected blind region via radius graph
    big=len(blind_idx)/M*100
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        pts=UNIT_DIRS[blind_idx]
        pairs=np.array(list(cKDTree(pts).query_pairs(0.03)))
        if len(pairs):
            n=len(pts)
            g=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
            _,lab=connected_components(g,directed=False)
            big=np.bincount(lab).max()/M*100
        else:
            big=1.0/M*100
    except Exception:
        pass
    return gap, big


def draw_comparison(app, configs, R, fh, fv, eval_d, elapsed):
    app.fig_cmp.clear(); app._cmp_configs=configs; app._cmp_axes=[]
    app._cmp_meta=(R,fh,fv,eval_d,elapsed)
    lon=np.arctan2(UNIT_DIRS[:,1],UNIT_DIRS[:,0])
    lat=np.arcsin(np.clip(UNIT_DIRS[:,2],-1,1))
    best_cov=max(c[2] for c in configs)
    bgb=getattr(app,'cmp_scheme',None) and app.cmp_scheme.get()=='bgb'

    for idx,(label,cc,cov,oris) in enumerate(configs):
        ax=app.fig_cmp.add_subplot(2,4,idx+1,projection='mollweide',facecolor='white')
        app._cmp_axes.append(ax)
        M=len(cc); blind=int(np.sum(cc==0)); bp=blind/M*100
        if bgb:
            col=np.empty((M,3))
            col[cc==0]=(0.05,0.05,0.05)          # black \u2014 blindspot
            col[cc==1]=(0.11,0.40,0.75)          # blue  \u2014 1 camera
            col[cc>=2]=(0.18,0.49,0.20)          # green \u2014 2+ cameras
            ax.scatter(lon,lat,c=col,s=0.8,edgecolors='none')
            gap,big=_blind_stats(cc)
            ax.set_title(f"{label}\nCoverage {cov:.1f}%  Blind {bp:.1f}%\n"
                         f"Max equator gap {gap:.1f}\u00b0  Largest blind region {big:.1f}%",
                         color=FG,fontsize=8,fontweight='bold',pad=6)
        else:
            ax.scatter(lon,lat,c=np.clip(cc,0,5),cmap=HM_CMAP,norm=HM_NORM,s=0.8,alpha=0.85,edgecolors='none')
            for ori in oris:
                cx=np.radians(ori[0]); cx=cx if cx<=np.pi else cx-2*np.pi
                ax.plot(cx,np.radians(ori[1]),'D',color='black',markersize=3,
                        markeredgecolor='black',markeredgewidth=0.4,zorder=10)
            ib=abs(cov-best_cov)<0.01
            tc=GN if ib else (YL if cov>=99 else ('#e94560' if cov<95 else FG))
            ax.set_title(f"{label}\n{cov:.2f}% cov | {bp:.1f}% blind",color=tc,fontsize=9,fontweight='bold',pad=6)
            if ib: ax.text(0.5,1.30,'best',transform=ax.transAxes,fontsize=8,color=GN,
                           fontweight='bold',ha='center',va='bottom')
        ax.grid(True,alpha=0.15,color='gray',linewidth=0.3); ax.tick_params(colors=DM,labelsize=5)

    if bgb:
        app.fig_cmp.text(0.5,0.045,
            'Blue: observed by one camera.   Green: observed by two or more cameras.   Black: blindspot.',
            ha='center',fontsize=11,color=FG)
    else:
        import matplotlib.cm as mcm
        cbar_ax=app.fig_cmp.add_axes([0.15,0.04,0.7,0.015])
        sm=mcm.ScalarMappable(cmap=HM_CMAP,norm=HM_NORM); sm.set_array([])
        cb=app.fig_cmp.colorbar(sm,cax=cbar_ax,orientation='horizontal')
        cb.set_ticks([0.25,1,2,3,4,7]); cb.set_ticklabels(['Blind','1 cam','2 cam','3 cam','4 cam','5+'])
        cb.ax.tick_params(labelsize=8,colors=FG)
        cb.set_label(f'Camera Overlap @ {eval_d/1000:.1f}m (R={R:.0f}mm) \u2014 Click any to inspect',color=FG,fontsize=10)
    app.fig_cmp.subplots_adjust(left=0.03,right=0.97,top=0.90,bottom=0.10,wspace=0.15,hspace=0.55)
    app.fig_cmp.suptitle(f'8-Way Comparison \u2014 16 Cameras @ R={R:.0f}mm, Eval={eval_d/1000:.1f}m',
                         color=HL,fontsize=13,fontweight='bold',y=0.98)
    if app._cmp_click_cid is not None: app.c_cmp.mpl_disconnect(app._cmp_click_cid)
    app._cmp_click_cid=app.c_cmp.mpl_connect('button_press_event',
        lambda ev:_on_click(app,ev,R,fh,fv,eval_d))
    app.c_cmp.draw()
    app.cmp_status.set(f"Done \u2014 {elapsed:.1f}s  (click any to inspect)")


def _on_click(app, ev, R, fh, fv, eval_d):
    if ev.inaxes is None: return
    for i,ax in enumerate(app._cmp_axes):
        if ev.inaxes==ax:
            _open_detail(app,app._cmp_configs[i],R,fh,fv,eval_d); return


def _open_percam(app, oris, fh, fv, label, eval_d):
    """Grid of Mollweide maps ‚Äî one per camera ‚Äî each showing that camera's FOV footprint
    on the viewing sphere (F450-report style)."""
    win=tk.Toplevel(app.root); win.title(f'Per-camera coverage: {label}')
    win.configure(bg=BG); win.geometry('1450x840')
    n=len(oris); cols=4; rows=math.ceil(n/cols)
    fig=Figure(figsize=(3.4*cols,2.35*rows),facecolor=BG)
    hth=math.tan(math.radians(fh/2)); htv=math.tan(math.radians(fv/2))
    def _v(pts,o):
        f,rt,up=cam_basis(o[0],o[1],o[3] if len(o)==4 else 0.0)
        pf=pts@f; inf=pf>0; pfs=np.where(inf,pf,1.0)
        return inf&(np.abs(pts@rt/pfs)<hth)&(np.abs(pts@up/pfs)<htv)
    # header numbers via the SAME metric the 8-way tiles use, so they match exactly
    _,_,_,cc,_=compute_coverage_nonuniform(oris,fh,fv,eval_d)
    cov=(cc>0).mean()*100; stereo=(cc>=2).mean()*100; blind=(cc==0).mean()*100
    fig.suptitle(f"Per-camera coverage ‚Äî {label}\n"
                 f"Coverage {cov:.1f}%   ¬∑   Stereo (2+) {stereo:.1f}%   ¬∑   Blind {blind:.2f}%",
                 color=FG,fontsize=14,fontweight='bold')
    dots=DOT_SPHERE[::2]                                   # ~5k directions for the footprint scatter
    lon=np.arctan2(dots[:,1],dots[:,0]); lat=np.arcsin(np.clip(dots[:,2],-1,1))
    for i,o in enumerate(oris):
        ax=fig.add_subplot(rows,cols,i+1,projection='mollweide')
        vis=_v(dots,o)
        ax.scatter(lon[~vis],lat[~vis],c='#e9e9e9',s=1.5,edgecolors='none',zorder=1)
        ax.scatter(lon[vis],lat[vis],c=RCOLS[i%len(RCOLS)],s=2.5,edgecolors='none',zorder=2)
        p=o[1]; rr=o[3] if len(o)==4 else 0.0
        role='top pole' if p>80 else 'bottom pole' if p<-80 else f'lat {p:.0f}¬∞'
        ax.set_title(f'Camera {i+1}: {role}\nyaw {o[0]:.0f}¬∞  pitch {p:.0f}¬∞  roll {rr:.0f}¬∞',
                     color=FG,fontsize=8.5,fontweight='bold')
        ax.grid(True,alpha=0.2,color='gray',linewidth=0.3)
        ax.set_xticklabels([]); ax.set_yticklabels([])
    fig.tight_layout(rect=[0,0,1,0.93])
    c=FigureCanvasTkAgg(fig,master=win); c.get_tk_widget().pack(fill='both',expand=True); c.draw()


def _open_detail(app, cfg, R, fh, fv, eval_d):
    label,cc,cov,oris=cfg
    cl=label.replace('\n',' ')
    win=tk.Toplevel(app.root); win.title(f"Detail: {cl}")
    win.configure(bg=BG); win.geometry("1500x900")

    # Header
    hdr=tk.Frame(win,bg=BG2,pady=6,padx=10); hdr.pack(fill='x')
    M=len(cc); blind=int(np.sum(cc==0))
    stereo=np.sum(cc>=2)/M*100; triple=np.sum(cc>=3)/M*100
    tk.Label(hdr,text=cl,bg=BG2,fg=HL,font=('Helvetica',14,'bold')).pack(side='left',padx=(0,16))
    tk.Button(hdr,text='Per-camera coverage',command=lambda:_open_percam(app,oris,fh,fv,cl,eval_d),
              fg='#0d47a1',font=('Helvetica',11,'bold'),relief='solid',bd=1,padx=8,
              cursor='hand2').pack(side='left',padx=(0,16))
    tk.Label(hdr,text=f"Cov:{cov:.2f}% Stereo:{stereo:.1f}% 3+:{triple:.1f}% Blind:{blind}",
             bg=BG2,fg=GN,font=('Menlo',11,'bold')).pack(side='left')
    tk.Label(hdr,text=f"R={R:.0f}mm FOV={fh:.0f}\u00b0\u00d7{fv:.0f}\u00b0 Eval={eval_d}mm",
             bg=BG2,fg=DM,font=('Menlo',10)).pack(side='right')

    # Toggle bar
    tog=tk.Frame(win,bg=BG2,pady=3,padx=6); tog.pack(fill='x')
    tk.Label(tog,text="SHOW:",bg=BG2,fg=DM,font=('Helvetica',9,'bold')).pack(side='left')
    show_pyr=tk.BooleanVar(value=False); show_blind=tk.BooleanVar(value=True)
    show_lbl=tk.BooleanVar(value=True); show_grid=tk.BooleanVar(value=True)
    show_dims=tk.BooleanVar(value=True)

    # Main content: figure on top, camera table on bottom
    fig=Figure(figsize=(16,8),facecolor=BG)
    canvas=FigureCanvasTkAgg(fig,master=win)
    canvas.get_tk_widget().pack(fill='both',expand=True)

    # Camera info table at bottom
    tbl_frame=tk.Frame(win,bg=BG,pady=4,padx=6); tbl_frame.pack(fill='x')
    _build_cam_table(tbl_frame, oris, R)

    state={'drag':None,'zoom':1.0}

    def redraw():
        fig.clear()
        ax1=fig.add_subplot(121,projection='3d',facecolor='white')
        _pw=set(i for i,pv in enumerate(pyr_vars) if pv.get())
        _draw_blueprint(ax1,oris,cc,R,fh,fv,eval_d,
                       show_pyr.get(),show_blind.get(),show_lbl.get(),show_grid.get(),
                       state['zoom'],show_dims.get(),_pw)
        ax1.set_title(f'3D Camera Structure\n{cl}',color=FG,fontsize=10,fontweight='bold')

        ax2=fig.add_subplot(122,projection='3d',facecolor='white')
        _draw_dot_sphere(ax2,cc,eval_d,state['zoom'],oris,R,show_dims.get(),fh,fv)
        ax2.set_title(f'3D Coverage Sphere\n{cov:.2f}% coverage',color=FG,fontsize=10,fontweight='bold')

        fig.subplots_adjust(left=0.0,right=1.0,top=0.90,bottom=0.02,wspace=0.0)
        fig.suptitle(f'{cl} \u2014 Detailed Inspection',color=HL,fontsize=13,fontweight='bold',y=0.97)
        state['ax1']=ax1; state['ax2']=ax2
        canvas.draw()

    def on_toggle():
        redraw()
    for txt,var in [("Pyramids",show_pyr),("Blind Spots",show_blind),
                    ("Labels",show_lbl),("Grid",show_grid),("Dimensions",show_dims)]:
        tk.Checkbutton(tog,text=txt,variable=var,fg=FG,bg=BG2,selectcolor=ACCENT,
                       activebackground=BG2,font=('Helvetica',9),
                       command=on_toggle).pack(side='left',padx=4)
    # per-camera pyramid toggles (multi-select) ‚Äî gated by the Pyramids master checkbox
    tk.Label(tog,text='  pyramids:',bg=BG2,fg=DM,font=('Helvetica',9)).pack(side='left')
    pyr_vars=[tk.BooleanVar(value=True) for _ in oris]
    def _pyr_all(v):
        for pv in pyr_vars: pv.set(v)
        on_toggle()
    tk.Button(tog,text='all',command=lambda:_pyr_all(True),bg=BG2,fg=FG,
              font=('Helvetica',8),relief='flat',padx=3).pack(side='left')
    tk.Button(tog,text='none',command=lambda:_pyr_all(False),bg=BG2,fg=FG,
              font=('Helvetica',8),relief='flat',padx=3).pack(side='left')
    for i in range(len(oris)):
        tk.Checkbutton(tog,text=f'{i+1}',variable=pyr_vars[i],fg=FG,bg=BG2,
                       selectcolor=ACCENT,activebackground=BG2,font=('Menlo',8),
                       indicatoron=0,padx=2,pady=0,relief='flat',borderwidth=1,
                       command=on_toggle).pack(side='left',padx=1)

    def press(ev):
        if ev.button==1 and ev.inaxes in (state.get('ax1'),state.get('ax2')):
            state['drag']=(ev.inaxes,ev.x,ev.y,ev.inaxes.elev,ev.inaxes.azim)
    def release(ev):
        state['drag']=None
    def drag(ev):
        d=state.get('drag')
        if d is None or ev.button!=1: return
        a,x0,y0,e0,az0=d
        a.view_init(elev=e0+(ev.y-y0)*0.3,azim=az0-(ev.x-x0)*0.3)   # unclamped ‚Üí full rotation
        canvas.draw_idle()
    def scroll(ev):
        a=ev.inaxes
        if a not in (state.get('ax1'),state.get('ax2')): return
        state['zoom']=max(0.2,min(30,state['zoom']*(1.1 if ev.button=='up' else 1/1.1)))
        # each sphere has its own base extent ‚Äî blueprint uses eval_d*1.05, the dot
        # sphere is drawn at sphere_r*1.20 = eval_d*1.14, so zoom from the right scale
        base=eval_d*1.14 if a is state.get('ax2') else eval_d*1.05
        lim=base/state['zoom']
        a.set_xlim(-lim,lim); a.set_ylim(-lim,lim); a.set_zlim(-lim,lim)
        a.set_box_aspect([1,1,1])
        canvas.draw_idle()

    canvas.mpl_connect('button_press_event',press)
    canvas.mpl_connect('button_release_event',release)
    canvas.mpl_connect('motion_notify_event',drag)
    canvas.mpl_connect('scroll_event',scroll)
    redraw()


def _build_cam_table(parent, oris, R):
    """Build a compact camera info table with yaw/pitch/roll and inter-camera angles."""
    n=len(oris)
    # Compute unit direction vectors for angular separation
    dirs=[]
    for o in oris:
        yaw,pitch=o[0],o[1]; roll=o[3] if len(o)==4 else 0.0
        yr,pr=np.radians(yaw),np.radians(pitch)
        dirs.append([np.cos(yr)*np.cos(pr),np.sin(yr)*np.cos(pr),np.sin(pr)])
    dirs=np.array(dirs)

    # Header
    hdr=tk.Frame(parent,bg=BG2); hdr.pack(fill='x')
    cols=['Cam','Yaw¬∞','Pitch¬∞','Roll¬∞','R mm','Nearest','Sep¬∞']
    for j,c in enumerate(cols):
        w=6 if j>0 else 4
        tk.Label(hdr,text=c,bg=BG2,fg=YL,font=('Menlo',9,'bold'),width=w,
                 anchor='center').pack(side='left',padx=2)
    tk.Label(hdr,text='‚îÇ',bg=BG2,fg=DM,font=('Menlo',9)).pack(side='left')
    # Mini angular matrix header
    for i in range(min(n,16)):
        tk.Label(hdr,text=f'C{i+1}',bg=BG2,fg=DM,font=('Menlo',7),width=4,
                 anchor='center').pack(side='left')

    # Angle matrix
    ang_mat=np.degrees(np.arccos(np.clip(dirs @ dirs.T,-1,1)))
    np.fill_diagonal(ang_mat,999)

    # Rows
    for i,o in enumerate(oris):
        row=tk.Frame(parent,bg=BG if i%2==0 else BG2); row.pack(fill='x')
        yaw,pitch=o[0],o[1]; roll=o[3] if len(o)==4 else 0.0
        Ri=o[2]
        nearest_idx=int(np.argmin(ang_mat[i]))
        nearest_ang=ang_mat[i,nearest_idx]
        rc=RCOLS[i%len(RCOLS)]
        vals=[f'C{i+1}',f'{yaw:.1f}',f'{pitch:.1f}',f'{roll:.1f}',f'{Ri:.0f}',
              f'C{nearest_idx+1}',f'{nearest_ang:.1f}']
        bg_r=BG if i%2==0 else BG2
        for j,v in enumerate(vals):
            fc=rc if j==0 else (GN if j<5 else YL)
            w=6 if j>0 else 4
            tk.Label(row,text=v,bg=bg_r,fg=fc,font=('Menlo',8),width=w,
                     anchor='center').pack(side='left',padx=2)
        tk.Label(row,text='‚îÇ',bg=bg_r,fg=DM,font=('Menlo',8)).pack(side='left')
        # Angular separation to each other camera
        for j in range(min(n,16)):
            a=ang_mat[i,j] if i!=j else 0
            tc=HL if a<25 and i!=j else (GN if a>40 else FG)
            tk.Label(row,text=f'{a:.0f}' if i!=j else '‚Äî',bg=bg_r,fg=tc,
                     font=('Menlo',7),width=4,anchor='center').pack(side='left')


def _draw_blueprint(ax, oris, cc, R, fh, fv, eval_d, pyramids, blind, labels, grid, zoom, dims=False, pyr_which=None):
    """Tab 1 blueprint: wireframe sphere, cameras, FOV pyramids, blind spots, grid cage."""
    ax.disable_mouse_rotation()
    fov_h=np.radians(fh); fov_v=np.radians(fv)

    # Wireframe body sphere
    u=np.linspace(0,2*np.pi,24); v=np.linspace(0,np.pi,16)
    sx=R*np.outer(np.cos(u),np.sin(v))
    sy=R*np.outer(np.sin(u),np.sin(v))
    sz=R*np.outer(np.ones_like(u),np.cos(v))
    ax.plot_wireframe(sx,sy,sz,color='#5a6b7a',alpha=0.3,linewidth=0.4)

    # Equator + meridians
    circ=np.linspace(0,2*np.pi,60)
    ax.plot(R*np.cos(circ),R*np.sin(circ),np.zeros_like(circ),color='gray',linewidth=0.8,alpha=0.5)
    ax.plot(R*np.cos(circ),np.zeros_like(circ),R*np.sin(circ),color='gray',linewidth=0.8,alpha=0.5)
    ax.plot(np.zeros_like(circ),R*np.cos(circ),R*np.sin(circ),color='gray',linewidth=0.8,alpha=0.5)

    # Reference circles at eval distance
    t=np.linspace(0,2*np.pi,60)
    for _ in range(1):
        ax.plot(eval_d*np.cos(t),eval_d*np.sin(t),np.zeros_like(t),color='silver',alpha=0.35,linewidth=0.6)
        ax.plot(eval_d*np.cos(t),np.zeros_like(t),eval_d*np.sin(t),color='silver',alpha=0.35,linewidth=0.6)
        ax.plot(np.zeros_like(t),eval_d*np.cos(t),eval_d*np.sin(t),color='silver',alpha=0.35,linewidth=0.6)

    cam_color=(0.2,0.7,0.3)

    # Cameras
    for ci,ori in enumerate(oris):
        yaw,pitch,Ri = ori[0],ori[1],ori[2]
        roll = ori[3] if len(ori)==4 else 0.0
        fwd,right,up=cam_basis(yaw,pitch,roll)
        pos=fwd*Ri
        ax.scatter(*pos,c=[cam_color],s=45,zorder=5,edgecolors='#37474f',linewidths=0.6,depthshade=False)

        if labels:
            ld=fwd*R*0.4
            lbl=f'C{ci+1}'
            if abs(roll)>1: lbl+=f'\\n{roll:.0f}¬∞'
            ax.text(pos[0]+ld[0],pos[1]+ld[1],pos[2]+ld[2],lbl,fontsize=5,
                    color='#37474f',fontweight='bold',ha='center',va='center',zorder=6)

        if pyramids and (pyr_which is None or ci in pyr_which):
            fl=eval_d
            hw=fl*math.tan(fov_h/2); vw=fl*math.tan(fov_v/2)
            fc=pos+fwd*fl
            c0=fc+right*hw+up*vw; c1=fc-right*hw+up*vw
            c2=fc-right*hw-up*vw; c3=fc+right*hw-up*vw
            faces=[[pos,c0,c1],[pos,c1,c2],[pos,c2,c3],[pos,c3,c0],[c0,c1,c2,c3]]
            poly=Poly3DCollection(faces,alpha=0.2,facecolor=cam_color,
                                   edgecolor=(*cam_color,0.4),linewidth=0.5)
            ax.add_collection3d(poly)
            for corner in [c0,c1,c2,c3]:
                ax.plot([pos[0],corner[0]],[pos[1],corner[1]],[pos[2],corner[2]],
                       color=cam_color,alpha=0.3,linewidth=0.6)

    # Blind spots
    if blind and cc is not None:
        bm=cc==0
        if bm.sum()>0:
            bd=UNIT_DIRS[bm]
            if len(bd)>200:
                bd=bd[np.random.choice(len(bd),200,replace=False)]
            for d in bd:
                tip=d*eval_d*1.5
                ax.plot([0,tip[0]],[0,tip[1]],[0,tip[2]],color='crimson',linewidth=1.2,alpha=0.7)

    # Dimension overlay: connect cameras (nearest neighbours) with angle + length labels,
    # plus radial spokes from the centre ‚Äî shows the structure's symmetry.
    if dims and len(oris)>=2:
        cpos=[]; cdir=[]
        for ori in oris:
            f,_,_=cam_basis(ori[0],ori[1],ori[3] if len(ori)==4 else 0.0)
            cdir.append(f); cpos.append(f*ori[2])
        cpos=np.array(cpos); cdir=np.array(cdir); nC=len(cpos)
        for i in range(nC):   # radial spokes centre ‚Üí camera
            ax.plot([0,cpos[i,0]],[0,cpos[i,1]],[0,cpos[i,2]],color=BL,lw=0.5,alpha=0.3,zorder=2)
        ax.text(cpos[0,0]*0.5,cpos[0,1]*0.5,cpos[0,2]*0.5,f'R{R:.0f}mm',
                fontsize=5,color=BL,ha='center',va='center',zorder=6)
        ang=np.degrees(np.arccos(np.clip(cdir@cdir.T,-1,1))); np.fill_diagonal(ang,999.0)
        drawn=set()
        for i in range(nC):   # each camera ‚Üí its 3 nearest neighbours
            for j in np.argsort(ang[i])[:3]:
                key=tuple(sorted((int(i),int(j))))
                if key in drawn or ang[i,j]>=999: continue
                drawn.add(key)
                p,q=cpos[i],cpos[j]; chord=float(np.linalg.norm(p-q)); m=(p+q)/2
                ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],color=YL,lw=0.8,alpha=0.6,zorder=4)
                ax.text(m[0],m[1],m[2],f'{ang[i,j]:.0f}¬∞  {chord:.0f}mm',
                        fontsize=4.5,color=YL,ha='center',va='center',zorder=6)

    # Grid cage
    L=eval_d*1.0
    if grid:
        for s1 in [-L,L]:
            for s2 in [-L,L]:
                ax.plot([s1,s1],[s2,s2],[-L,L],color='darkgray',linewidth=0.5,alpha=0.35)
                ax.plot([s1,s1],[-L,L],[s2,s2],color='darkgray',linewidth=0.5,alpha=0.35)
                ax.plot([-L,L],[s1,s1],[s2,s2],color='darkgray',linewidth=0.5,alpha=0.35)
        gv=np.arange(-L,L+1,L*0.25)
        for g in gv:
            ax.plot([g,g],[-L,L],[-L,-L],color='lightgray',linewidth=0.2,alpha=0.3)
            ax.plot([-L,L],[g,g],[-L,-L],color='lightgray',linewidth=0.2,alpha=0.3)

    lim=eval_d*1.05/zoom
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1, 1, 1]); ax.set_aspect('equal')   # true sphere, not an ellipsoid
    ax.set_axis_off()
    ax.view_init(elev=25,azim=-60)


def _draw_dot_sphere(ax, cc, eval_d, zoom, oris=None, R=250, show_dims=False, fh=80, fv=65):
    """Solid dot sphere with camera positions, dimension lines, and angular separations."""
    ax.disable_mouse_rotation()
    sphere_r=eval_d*0.95

    # Coverage per dot computed DIRECTLY from the cameras (memory-safe ‚Üí allows many dots)
    dots=DOT_SPHERE
    if oris:
        hth=math.tan(math.radians(fh/2)); htv=math.tan(math.radians(fv/2))
        dot_cc=np.zeros(len(dots))
        for o in oris:
            f,rt,up=cam_basis(o[0],o[1],o[3] if len(o)==4 else 0.0)
            pf=dots@f; inf=pf>0; pfs=np.where(inf,pf,1.0)
            dot_cc+=inf&(np.abs(dots@rt/pfs)<hth)&(np.abs(dots@up/pfs)<htv)
    else:
        dot_cc=np.ones(len(dots))
    pts=dots*sphere_r
    cc_clip=np.clip(dot_cc,0,5)

    cmap={0:(0.10,0.10,0.18), 1:(0.91,0.27,0.37), 2:(1.00,0.44,0.26),
          3:(1.00,0.85,0.24), 4:(0.00,0.83,0.67), 5:(0.31,0.76,0.97)}
    colors=np.array([cmap.get(int(min(c,5)),(0.31,0.76,0.97)) for c in cc_clip])

    ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=30,alpha=0.95,
               depthshade=False,edgecolors='none',zorder=1)

    # Draw cameras on the sphere
    if oris is not None and len(oris)>0:
        n=len(oris)
        cam_dirs=[]
        for o in oris:
            yr,pr=np.radians(o[0]),np.radians(o[1])
            cam_dirs.append([np.cos(yr)*np.cos(pr),np.sin(yr)*np.cos(pr),np.sin(pr)])
        cam_dirs=np.array(cam_dirs)
        cam_pts=cam_dirs*sphere_r  # camera positions on eval sphere surface

        # Camera dots ‚Äî large white diamonds
        ax.scatter(cam_pts[:,0],cam_pts[:,1],cam_pts[:,2],
                   c='#263238',s=80,marker='D',edgecolors='black',
                   linewidths=0.8,zorder=10,depthshade=False)

        # Labels with yaw/pitch/roll for each camera
        for i,o in enumerate(oris):
            yaw,pitch=o[0],o[1]; roll=o[3] if len(o)==4 else 0.0
            d=cam_dirs[i]
            # Offset label outward
            lp=d*sphere_r*1.12
            rc=RCOLS[i%len(RCOLS)]
            lbl=f'C{i+1}\nY{yaw:.0f} P{pitch:.0f}'
            if abs(roll)>1: lbl+=f' R{roll:.0f}'
            ax.text(lp[0],lp[1],lp[2],lbl,fontsize=5.5,color=rc,
                    fontweight='bold',ha='center',va='center',zorder=11)

        if show_dims:
            # Radial lines from center to each camera
            for i in range(n):
                cp=cam_pts[i]
                ax.plot([0,cp[0]],[0,cp[1]],[0,cp[2]],
                        color='gray',alpha=0.4,linewidth=0.6,zorder=2)
                # R distance label at midpoint
                mp=cam_dirs[i]*sphere_r*0.5
                ax.text(mp[0],mp[1],mp[2],f'{R:.0f}mm',fontsize=4,
                        color='gray',ha='center',va='center',alpha=0.7)

            # Arcs between nearest-neighbor pairs with angular separation
            ang_mat=np.degrees(np.arccos(np.clip(cam_dirs @ cam_dirs.T,-1,1)))
            np.fill_diagonal(ang_mat,999)
            drawn=set()
            for i in range(n):
                j=int(np.argmin(ang_mat[i]))
                pair=tuple(sorted([i,j]))
                if pair in drawn: continue
                drawn.add(pair)
                # Draw arc between cameras i and j on the sphere surface
                d1,d2=cam_dirs[i],cam_dirs[j]
                sep=ang_mat[i,j]
                arc_pts_n=max(10,int(sep))
                ts=np.linspace(0,1,arc_pts_n)
                arc=np.array([d1*(1-t)+d2*t for t in ts])
                norms=np.linalg.norm(arc,axis=1,keepdims=True)+1e-30
                arc=arc/norms*sphere_r*1.02  # slightly outside sphere
                ax.plot(arc[:,0],arc[:,1],arc[:,2],
                        color=YL,linewidth=1.2,alpha=0.7,zorder=5)
                # Angle label at midpoint of arc
                mid=arc[len(arc)//2]
                mo=mid/np.linalg.norm(mid)*sphere_r*1.08
                ax.text(mo[0],mo[1],mo[2],f'{sep:.1f}\u00b0',fontsize=6,
                        color=YL,fontweight='bold',ha='center',va='center',zorder=11)

    # Origin marker
    ax.scatter([0],[0],[0],c='black',s=30,marker='+',zorder=3,depthshade=False)

    lim=sphere_r*1.20/zoom
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1, 1, 1]); ax.set_aspect('equal')   # true sphere, not an ellipsoid
    ax.set_axis_off()
    ax.view_init(elev=20,azim=-50)
    
class App:
    """Minimal shell: FOV and body-radius inputs, then the comparison view."""
    def __init__(self, root):
        self.root = root
        root.title('8-Way Camera Placement Comparison')
        root.configure(bg=BG)
        root.geometry('1540x920')
        bar = tk.Frame(root, bg=BG2, pady=6, padx=10)
        bar.pack(fill='x')
        for label, attr, val in [('FOV H\u00b0:', 'fov_h', '80'),
                                 ('FOV V\u00b0:', 'fov_v', '65'),
                                 ('Base R mm:', 'base_r', '300')]:
            tk.Label(bar, text=label, bg=BG2, fg=FG, font=('Helvetica', 11)).pack(side='left')
            var = tk.StringVar(value=val)
            setattr(self, attr, var)
            tk.Entry(bar, textvariable=var, width=5, font=('Courier', 12), bg='white',
                     fg=BL, relief='solid', bd=1).pack(side='left', padx=(2, 12))
        body = tk.Frame(root, bg=BG)
        body.pack(fill='both', expand=True)
        build_tab6(self, body)


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
     
