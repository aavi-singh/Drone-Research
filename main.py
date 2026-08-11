 """Multi-Camera Drone Coverage Analyzer ‚Äî single file.

16 cameras on a spherical body, with a propeller model (solid rotors + arms),
mechanical clearance checking, and the 12-way comparison GUI.

Run:  python3 camera_coverage_analyzer.py
"""
import matplotlib
matplotlib.use('TkAgg')
import multiprocessing, warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GEOMETRY AND COVERAGE CORE
# ============================================================================

import numpy as np
import warnings
warnings.filterwarnings('ignore')

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


def camera_pos_3d(yaw, pitch, R):
    yr, pr = np.radians(yaw), np.radians(pitch)
    return np.array([
        R * np.cos(yr) * np.cos(pr),
        R * np.sin(yr) * np.cos(pr),
        R * np.sin(pr)
    ])


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


def compute_polar_zones(cc, zones=None):
    """Break coverage into polar zones.
    Returns dict of zone_name -> (coverage%, stereo%, total_points)"""
    if zones is None:
        zones = [
            ('Top Pole (60-90¬∞)', 60, 90),
            ('Upper Mid (30-60¬∞)', 30, 60),
            ('Equator (-30 to 30¬∞)', -30, 30),
            ('Lower Mid (-60 to -30¬∞)', -60, -30),
            ('Bottom Pole (-90 to -60¬∞)', -90, -60),
        ]
    results = {}
    for name, lo, hi in zones:
        mask = (_ELEV_DEG >= lo) & (_ELEV_DEG < hi)
        n_pts = np.sum(mask)
        if n_pts == 0:
            results[name] = (0.0, 0.0, 0)
            continue
        zone_cc = cc[mask]
        cov = np.sum(zone_cc > 0) / n_pts * 100
        stereo = np.sum(zone_cc >= 2) / n_pts * 100
        results[name] = (cov, stereo, int(n_pts))
    return results


def compute_cross_coverage(cam_vis):
    """Compute pairwise overlap matrix between cameras.
    Returns NxN matrix where entry [i,j] = % of camera i's visible area
    also visible to camera j."""
    n = len(cam_vis)
    matrix = np.zeros((n, n))
    for i in range(n):
        vi = cam_vis[i]
        ni = np.sum(vi)
        if ni == 0:
            continue
        for j in range(n):
            if i == j:
                matrix[i, j] = 100.0
                continue
            overlap = np.sum(vi & cam_vis[j])
            matrix[i, j] = overlap / ni * 100
    return matrix


def find_worst_gap(cc):
    uncovered = UNIT_DIRS[cc == 0]
    if len(uncovered) == 0:
        return None, 0
    center = uncovered.mean(axis=0)
    norm = np.linalg.norm(center)
    if norm < 1e-9:
        center = uncovered[0]
    else:
        center /= norm
    az = np.degrees(np.arctan2(center[1], center[0])) % 360
    el = np.degrees(np.arcsin(np.clip(center[2], -1, 1)))
    return (az, el), len(uncovered) / len(UNIT_DIRS) * 100


def angular_separations(oris_with_R):
    dirs = []
    for o in oris_with_R:
        yr, pr = np.radians(o[0]), np.radians(o[1])
        dirs.append([np.cos(yr)*np.cos(pr), np.sin(yr)*np.cos(pr), np.sin(pr)])
    dirs = np.array(dirs)
    n = len(dirs)
    angs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dot = np.clip(np.dot(dirs[i], dirs[j]), -1, 1)
            angs[i, j] = np.degrees(np.arccos(dot))
    return angs


def nearest_neighbor_angles(oris_with_R):
    angs = angular_separations(oris_with_R)
    n = len(oris_with_R)
    nn = []
    for i in range(n):
        row = angs[i].copy()
        row[i] = 999
        nn.append(row.min())
    return nn


def full_analysis(config_str, fov_h=80, fov_v=65, base_R=250,
                  ring_radii=None, pitch_offsets=None,
                  eval_distances=None):
    """Full analysis with non-uniform radii support."""
    counts, lats, staggers = auto_layout(config_str, fov_h, fov_v)

    if ring_radii is None:
        ring_radii = [base_R] * len(counts)
    else:
        # Pad if needed
        while len(ring_radii) < len(counts):
            ring_radii.append(base_R)

    if pitch_offsets is None:
        pitch_offsets = [0.0] * len(counts)
    else:
        while len(pitch_offsets) < len(counts):
            pitch_offsets.append(0.0)

    oris = build_orientations_with_radii(counts, lats, staggers,
                                          ring_radii, pitch_offsets)
    total_cams = len(oris)

    results = {
        'config': config_str, 'total_cams': total_cams,
        'fov_h': fov_h, 'fov_v': fov_v, 'base_R': base_R,
        'counts': counts, 'latitudes': lats, 'staggers': staggers,
        'ring_radii': ring_radii, 'pitch_offsets': pitch_offsets,
        'orientations': oris,
    }

    # Camera positions
    # Camera positions
    positions = [camera_pos_3d(o[0], o[1], o[2]) for o in oris]
    results['positions_xyz'] = [(p[0], p[1], p[2]) for p in positions]

    # Nearest neighbor
    nn = nearest_neighbor_angles(oris)
    results['nn_angles'] = nn
    results['min_nn_angle'] = min(nn) if nn else 0
    results['max_nn_angle'] = max(nn) if nn else 0
    results['avg_nn_angle'] = np.mean(nn) if nn else 0

    # Physical distances
    if len(positions) > 1:
        pos_arr = np.array(positions)
        min_d = 1e9
        for i in range(len(pos_arr)):
            for j in range(i+1, len(pos_arr)):
                d = np.linalg.norm(pos_arr[i] - pos_arr[j])
                if d < min_d:
                    min_d = d
        results['min_cam_dist_mm'] = min_d
    else:
        results['min_cam_dist_mm'] = 0

    # Solid angle budget
    fov_diag = np.degrees(2 * np.arctan(np.sqrt(
        np.tan(np.radians(fov_h/2))**2 + np.tan(np.radians(fov_v/2))**2)))
    results['fov_diagonal'] = fov_diag
    sa_per_cam = 4 * np.arcsin(
        np.sin(np.radians(fov_h/2)) * np.sin(np.radians(fov_v/2)))
    results['solid_angle_per_cam_sr'] = sa_per_cam
    results['total_solid_angle_sr'] = sa_per_cam * total_cams
    results['sphere_solid_angle_sr'] = 4 * np.pi
    results['raw_coverage_ratio'] = (sa_per_cam * total_cams) / (4*np.pi) * 100

    # Coverage sweep
    if eval_distances is None:
        eval_distances = [500, 1000, 1500, 2000, 3000, 5000]
    sweep = []
    for d in eval_distances:
        c, s, t, cc, cam_vis = compute_coverage_nonuniform(oris, fov_h, fov_v, d)
        gap_dir, gap_pct = find_worst_gap(cc)
        polar = compute_polar_zones(cc)
        cross = compute_cross_coverage(cam_vis)
        sweep.append({
            'dist_mm': d, 'coverage': c, 'stereo': s, 'triple': t,
            'gap_dir': gap_dir, 'gap_pct': gap_pct,
            'polar_zones': polar, 'cross_matrix': cross,
            'cc': cc, 'cam_vis': cam_vis,
        })
    results['sweep'] = sweep

    # Detailed analysis at 1.5m and 2.0m
    for focus_d in [1500, 2000]:
        c, s, t, cc, cam_vis = compute_coverage_nonuniform(oris, fov_h, fov_v, focus_d)
        gap_dir, gap_pct = find_worst_gap(cc)
        polar = compute_polar_zones(cc)
        cross = compute_cross_coverage(cam_vis)
        key = f'focus_{focus_d}'
        results[key] = {
            'coverage': c, 'stereo': s, 'triple': t,
            'gap_dir': gap_dir, 'gap_pct': gap_pct,
            'polar_zones': polar, 'cross_matrix': cross,
            'per_cam_coverage': [np.sum(v)/len(v)*100 for v in cam_vis],
        }

    # Store cc at 2m for 3D blind spot visualization
    results['cc_2000'] = results['focus_2000'].get('cc_2000', None)
    # Recompute quickly for direct access
    _, _, _, cc_viz, _ = compute_coverage_nonuniform(oris, fov_h, fov_v, 2000)
    results['cc_2000'] = cc_viz

    return results


# ============================================================================
# PLACEMENT SOLVERS (Fibonacci / Thomson / Tammes)
# ============================================================================

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  Helpers: spherical ‚Üî Cartesian
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

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


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  1. Fibonacci Spiral Placement
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

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


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  2. Thomson's Problem ‚Äî Minimize Coulomb Energy
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

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


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  3. Tammes Problem ‚Äî Maximize Minimum Distance
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

def _tammes_objective(x, n):
    """Negative minimum pairwise distance (we minimize this)."""
    thetas, phis = _unpack_angles(x, n)
    st = np.sin(thetas)
    ct = np.cos(thetas)
    sp = np.sin(phis)
    cp = np.cos(phis)

    X = st * cp
    Y = st * sp
    Z = ct

    min_dist = 1e30
    for i in range(n):
        for j in range(i+1, n):
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            dz = Z[i] - Z[j]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            if d < min_dist:
                min_dist = d

    return -min_dist  # minimize negative = maximize distance


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


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  4. Coverage-Maximizing Optimizer (with roll DOF)
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

def optimize_coverage(init_oris, fov_h=80, fov_v=65, eval_dist=2000,
                      R=250.0, max_iter=2000, callback=None):
    """Optimize camera orientations + roll to maximize coverage.
    Uses multi-start roll seeding to escape roll=0 local minima.
    init_oris: initial [(yaw, pitch, R), ...] or [(yaw, pitch, R, roll), ...]
    Returns (optimized_oris_with_roll, best_score, history).
    Output orientations are 4-tuples: (yaw, pitch, R, roll_deg)."""
    n = len(init_oris)

    # Convert to base optimization variables (theta, phi)
    thetas_base = []
    phis_base = []
    rolls_base = []
    for ori in init_oris:
        if len(ori) == 4:
            yaw, pitch, _, roll = ori
        else:
            yaw, pitch, _ = ori
            roll = 0.0
        yr, pr = np.radians(yaw), np.radians(pitch)
        theta = np.pi/2 - pr
        phi = yr
        thetas_base.append(theta)
        phis_base.append(phi)
        rolls_base.append(np.radians(roll))

    thetas_arr = np.array(thetas_base)
    phis_arr = np.array(phis_base)
    rolls_arr = np.array(rolls_base)

    # Bounds: theta [0.01, pi-0.01], phi unbounded, roll [-pi/2, pi/2]
    bounds = ([(0.01, np.pi - 0.01)] * n +
              [(None, None)] * n +
              [(-np.pi/2, np.pi/2)] * n)

    all_history = []

    def objective(x):
        thetas = x[:n]
        phis = x[n:2*n]
        rolls = x[2*n:3*n]
        oris = []
        for theta, phi, roll_r in zip(thetas, phis, rolls):
            xx, yy, zz = _sph_to_xyz(theta, phi)
            yaw, pitch = _xyz_to_yaw_pitch(xx, yy, zz)
            oris.append((yaw, pitch, R, np.degrees(roll_r)))

        cov, stereo, triple, cc, _ = compute_coverage_nonuniform(
            oris, fov_h, fov_v, eval_dist)

        score = cov + 0.3 * stereo + 0.1 * triple
        blind_n = np.sum(cc == 0)
        penalty = blind_n * 0.01

        all_history.append(score)
        return -(score - penalty)

    def _run_one(roll_init):
        x0 = np.concatenate([thetas_arr.copy(), phis_arr.copy(), roll_init])
        result = minimize(
            objective, x0,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'eps': 1e-4}
        )
        return result.x, -result.fun

    # Multi-start: original rolls + 3 random roll seeds
    np.random.seed(24680)  # deterministic ‚Äî reproducible runs
    roll_seeds = [
        rolls_arr.copy(),                                     # original (or zero)
        np.random.uniform(-np.pi/4, np.pi/4, n),             # random ¬±45¬∞
        np.random.uniform(-np.pi/4, np.pi/4, n),             # random ¬±45¬∞
        np.array([(-1)**i * np.pi/8 for i in range(n)]),      # alternating ¬±22.5¬∞
    ]

    best_x = None
    best_score = -1e9
    for seed in roll_seeds:
        x_opt, score = _run_one(seed)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Extract best result
    thetas = best_x[:n]
    phis = best_x[n:2*n]
    rolls = best_x[2*n:3*n]
    oris = []
    for theta, phi, roll_r in zip(thetas, phis, rolls):
        x, y, z = _sph_to_xyz(theta, phi)
        yaw, pitch = _xyz_to_yaw_pitch(x, y, z)
        oris.append((yaw, pitch, R, np.degrees(roll_r)))

    return oris, best_score, all_history


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  5. Results Bridge ‚Äî orientations ‚Üí full analysis dict
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

def orientations_to_results(oris, fov_h=80, fov_v=65, base_R=250,
                            eval_distances=None, label='optimizer'):
    """Convert raw orientations to a full results dict compatible with GUI.
    Same format as full_analysis() output."""
    if eval_distances is None:
        eval_distances = [500, 1000, 1500, 2000, 3000, 5000]

    total_cams = len(oris)

    # Infer ring structure (all cameras = 1 ring for optimizer output)
    counts = [total_cams]
    latitudes = [0.0]
    staggers = [0.0]
    ring_radii = [oris[0][2]] if oris else [base_R]
    pitch_offsets = [0.0]

    results = {
        'config': f'{label}-{total_cams}', 'total_cams': total_cams,
        'fov_h': fov_h, 'fov_v': fov_v, 'base_R': base_R,
        'counts': counts, 'latitudes': latitudes, 'staggers': staggers,
        'ring_radii': ring_radii * total_cams,  # per-cam radii
        'pitch_offsets': pitch_offsets,
        'orientations': oris,
    }

    # Camera positions (extract yaw, pitch, R regardless of tuple length)
    positions = [camera_pos_3d(o[0], o[1], o[2]) for o in oris]
    results['positions_xyz'] = [(p[0], p[1], p[2]) for p in positions]

    # Nearest neighbor
    nn = nearest_neighbor_angles(oris)
    results['nn_angles'] = nn
    results['min_nn_angle'] = min(nn) if nn else 0
    results['max_nn_angle'] = max(nn) if nn else 0
    results['avg_nn_angle'] = np.mean(nn) if nn else 0

    # Physical distances
    if len(positions) > 1:
        pos_arr = np.array(positions)
        min_d = 1e9
        for i in range(len(pos_arr)):
            for j in range(i+1, len(pos_arr)):
                d = np.linalg.norm(pos_arr[i] - pos_arr[j])
                if d < min_d:
                    min_d = d
        results['min_cam_dist_mm'] = min_d
    else:
        results['min_cam_dist_mm'] = 0

    # Solid angle budget
    fov_diag = np.degrees(2 * np.arctan(np.sqrt(
        np.tan(np.radians(fov_h/2))**2 + np.tan(np.radians(fov_v/2))**2)))
    results['fov_diagonal'] = fov_diag
    sa_per_cam = 4 * np.arcsin(
        np.sin(np.radians(fov_h/2)) * np.sin(np.radians(fov_v/2)))
    results['solid_angle_per_cam_sr'] = sa_per_cam
    results['total_solid_angle_sr'] = sa_per_cam * total_cams
    results['sphere_solid_angle_sr'] = 4 * np.pi
    results['raw_coverage_ratio'] = (sa_per_cam * total_cams) / (4*np.pi) * 100

    # Coverage sweep
    sweep = []
    for d in eval_distances:
        c, s, t, cc, cam_vis = compute_coverage_nonuniform(oris, fov_h, fov_v, d)
        gap_dir, gap_pct = find_worst_gap(cc)
        polar = compute_polar_zones(cc)
        cross = compute_cross_coverage(cam_vis)
        sweep.append({
            'dist_mm': d, 'coverage': c, 'stereo': s, 'triple': t,
            'gap_dir': gap_dir, 'gap_pct': gap_pct,
            'polar_zones': polar, 'cross_matrix': cross,
            'cc': cc, 'cam_vis': cam_vis,
        })
    results['sweep'] = sweep

    # Detailed analysis at 1.5m and 2.0m
    for focus_d in [1500, 2000]:
        c, s, t, cc, cam_vis = compute_coverage_nonuniform(oris, fov_h, fov_v, focus_d)
        gap_dir, gap_pct = find_worst_gap(cc)
        polar = compute_polar_zones(cc)
        cross = compute_cross_coverage(cam_vis)
        key = f'focus_{focus_d}'
        results[key] = {
            'coverage': c, 'stereo': s, 'triple': t,
            'gap_dir': gap_dir, 'gap_pct': gap_pct,
            'polar_zones': polar, 'cross_matrix': cross,
            'per_cam_coverage': [np.sum(v)/len(v)*100 for v in cam_vis],
        }

    # cc at 2m for 3D blind spot visualization
    _, _, _, cc_viz, _ = compute_coverage_nonuniform(oris, fov_h, fov_v, 2000)
    results['cc_2000'] = cc_viz

    return results


# ============================================================================
# FAST COVERAGE OBJECTIVE (used by the roll optimizer)
# ============================================================================

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  Vectorized Coverage Engine (the core speedup)
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

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


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  Global Optimizer: Basin-Hopping + Simulated Annealing Hybrid
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

def optimize_global(n=16, R=250.0, fov_h=80, fov_v=65, eval_dist=2000,
                    max_iter=50000, n_restarts=20, method='basin_hop',
                    callback=None):
    """Global optimizer that escapes local minima. Now optimizes 3N DOF (theta+phi+roll).
    Methods: 'basin_hop', 'anneal', 'differential', 'multi_local'
    Returns (orientations_4tuple, best_score, history)."""
    
    history = []
    best_score = -1e9
    best_x = None
    
    def obj(x):
        return _fast_objective(x, n, R, fov_h, fov_v, eval_dist)
    
    # Generate diverse starting points ‚Äî 3N: [thetas | phis | rolls]
    def _make_start(kind):
        if kind == 'fibonacci':
            golden = (1 + np.sqrt(5)) / 2
            thetas = np.array([np.arccos(np.clip(1 - (2*i+1)/n, -1, 1)) for i in range(n)])
            phis = np.array([2 * np.pi * i / golden for i in range(n)])
        elif kind == 'ring_14641':
            # Manually construct 1-4-6-4-1
            pitches = [-90, -45,-45,-45,-45, 0,0,0,0,0,0, 45,45,45,45, 90]
            yaws = [0, 0,90,180,270, 30,90,150,210,270,330, 45,135,225,315, 0]
            thetas = np.array([np.pi/2 - np.radians(p) for p in pitches])
            phis = np.array([np.radians(y) for y in yaws])
        elif kind == 'random':
            z = np.random.uniform(-1, 1, n)
            thetas = np.arccos(z)
            phis = np.random.uniform(0, 2*np.pi, n)
        elif kind == 'perturbed_ring':
            pitches = [-90, -45,-45,-45,-45, 0,0,0,0,0,0, 45,45,45,45, 90]
            yaws = [0, 0,90,180,270, 30,90,150,210,270,330, 45,135,225,315, 0]
            thetas = np.array([np.pi/2 - np.radians(p) for p in pitches]) + np.random.normal(0, 0.15, n)
            phis = np.array([np.radians(y) for y in yaws]) + np.random.normal(0, 0.15, n)
        else:
            z = np.random.uniform(-1, 1, n)
            thetas = np.arccos(z)
            phis = np.random.uniform(0, 2*np.pi, n)
        rolls = np.zeros(n)  # Start with zero roll
        return np.concatenate([thetas, phis, rolls])

    # Bounds for 3N: theta [0.01, pi-0.01], phi [0, 2pi], roll [-pi/2, pi/2]
    bounds_3n = ([(0.01, np.pi - 0.01)] * n +
                 [(0, 2*np.pi)] * n +
                 [(-np.pi/2, np.pi/2)] * n)

    if method == 'multi_local':
        # Multi-start L-BFGS-B with diverse starts + massive iterations
        starts = ['ring_14641', 'fibonacci'] + ['perturbed_ring'] * 6 + ['random'] * (n_restarts - 8)
        for i, kind in enumerate(starts):
            x0 = _make_start(kind)
            result = minimize(obj, x0, method='L-BFGS-B',
                            bounds=bounds_3n,
                            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'eps': 5e-5})
            score = -result.fun
            history.append((kind, score))
            if callback:
                callback(i+1, len(starts), kind, score)
            if score > best_score:
                best_score = score
                best_x = result.x.copy()
                
    elif method == 'anneal':
        # Simulated annealing ‚Äî true global optimizer
        x0 = _make_start('ring_14641')
        result = dual_annealing(obj, bounds_3n, x0=x0,
                                maxiter=max_iter, seed=42,
                                initial_temp=5230, restart_temp_ratio=2e-5,
                                visit=2.62, accept=-5.0)
        best_x = result.x
        best_score = -result.fun
        history.append(('anneal', best_score))
        
    elif method == 'differential':
        # Differential evolution ‚Äî population-based global optimizer
        result = differential_evolution(obj, bounds_3n, maxiter=max_iter,
                                       popsize=30, tol=1e-12, seed=42,
                                       mutation=(0.5, 1.5), recombination=0.9,
                                       polish=True)
        best_x = result.x
        best_score = -result.fun
        history.append(('differential', best_score))
        
    elif method == 'basin_hop':
        # Basin-hopping: local optimization + random perturbations
        x0 = _make_start('ring_14641')
        
        class StepTaker:
            def __init__(self, stepsize=0.15):
                self.stepsize = stepsize
            def __call__(self, x):
                x += np.random.uniform(-self.stepsize, self.stepsize, x.shape)
                x[:n] = np.clip(x[:n], 0.01, np.pi - 0.01)  # theta bounds
                x[2*n:] = np.clip(x[2*n:], -np.pi/2, np.pi/2)  # roll bounds
                return x
        
        from scipy.optimize import basinhopping
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds_3n,
            'options': {'maxiter': max_iter // n_restarts, 'ftol': 1e-15, 'eps': 5e-5}
        }
        result = basinhopping(obj, x0, niter=n_restarts,
                             minimizer_kwargs=minimizer_kwargs,
                             take_step=StepTaker(0.15), seed=42)
        best_x = result.x
        best_score = -result.fun
        history.append(('basin_hop', best_score))

    # Convert best solution to orientations (4-tuples with roll)
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
    
    return oris, best_score, history


# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê
#  Convenience: optimized versions of existing solvers
# ‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê‚ïê

def optimize_coverage_v2(init_oris, fov_h=80, fov_v=65, eval_dist=2000,
                         R=250.0, max_iter=50000):
    """Drop-in replacement for optimize_coverage with vectorized engine + roll DOF.
    Output orientations are 4-tuples: (yaw, pitch, R, roll_deg)."""
    n = len(init_oris)
    thetas0 = np.array([np.pi/2 - np.radians(o[1]) for o in init_oris])
    phis0 = np.array([np.radians(o[0]) for o in init_oris])
    rolls0 = np.array([np.radians(o[3]) if len(o) == 4 else 0.0 for o in init_oris])
    x0 = np.concatenate([thetas0, phis0, rolls0])

    bounds = ([(0.01, np.pi - 0.01)] * n +
              [(None, None)] * n +
              [(-np.pi/2, np.pi/2)] * n)

    def obj(x):
        return _fast_objective(x, n, R, fov_h, fov_v, eval_dist)

    result = minimize(obj, x0, method='L-BFGS-B',
                     bounds=bounds,
                     options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'eps': 5e-5})

    thetas = result.x[:n]
    phis = result.x[n:2*n]
    rolls = result.x[2*n:3*n]
    oris = []
    for theta, phi, roll_r in zip(thetas, phis, rolls):
        st = np.sin(theta)
        x, y, z = st * np.cos(phi), st * np.sin(phi), np.cos(theta)
        pitch = np.degrees(np.arcsin(np.clip(z, -1, 1)))
        yaw = np.degrees(np.arctan2(y, x)) % 360
        oris.append((yaw, pitch, R, np.degrees(roll_r)))
    return oris, -result.fun, []


# ============================================================================
# PROPELLER MODEL, OPTIMIZERS AND THE 12-WAY COMPARISON TAB
# ============================================================================

import tkinter as tk
import os, threading, math, numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    tk.Label(bar,text="12-WAY COMPARISON",bg=BG2,fg=HL,
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
    # propeller count: 0 = none; discs are placed at the best azimuth per config,
    # radius scaled 120*sqrt(4/P) so total lift stays constant (takes effect on RUN)
    tk.Label(bar,text="  Props:",bg=BG2,fg=FG,font=('Helvetica',11)).pack(side='left')
    app.prop_var=tk.StringVar(value='0')
    _pm=tk.OptionMenu(bar,app.prop_var,'0','1','2','3','4','5','6')
    _pm.config(bg=BG2,fg=FG,font=('Helvetica',10,'bold'),activebackground=ACCENT,
               highlightthickness=0,relief='flat')
    _pm.pack(side='left',padx=2)
    # rotor-plane height above the equator: 0 = equatorial mount (props dead in the
    # equator cameras' view), +250 = slung-ball sweet spot, +400 = overhead
    tk.Label(bar,text="  Rotor h:",bg=BG2,fg=FG,font=('Helvetica',11)).pack(side='left')
    app.prop_h_var=tk.StringVar(value='300')
    _hm=tk.OptionMenu(bar,app.prop_h_var,'0','150','250','300','350','400')
    _hm.config(bg=BG2,fg=FG,font=('Helvetica',10,'bold'),activebackground=ACCENT,
               highlightthickness=0,relief='flat')
    _hm.pack(side='left',padx=2)
    tk.Label(bar,text="mm",bg=BG2,fg=DM,font=('Helvetica',9)).pack(side='left')
    # boom length: body surface -> rotor axis (the arm you would actually cut)
    tk.Label(bar,text="  Arm:",bg=BG2,fg=FG,font=('Helvetica',11)).pack(side='left')
    app.arm_var=tk.StringVar(value='130')
    _am=tk.OptionMenu(bar,app.arm_var,'130','175','240','300','400')
    _am.config(bg=BG2,fg=FG,font=('Helvetica',10,'bold'),activebackground=ACCENT,
               highlightthickness=0,relief='flat')
    _am.pack(side='left',padx=2)
    tk.Label(bar,text="mm",bg=BG2,fg=DM,font=('Helvetica',9)).pack(side='left')
    # quad rotor radius (mm); other counts hold total disc area: r_P = r4*sqrt(4/P)
    tk.Label(bar,text="  Prop r:",bg=BG2,fg=FG,font=('Helvetica',11)).pack(side='left')
    app.prop_r_var=tk.StringVar(value=str(int(PROP_R4)))
    tk.Entry(bar,textvariable=app.prop_r_var,width=5,bg='white',fg=FG,
             font=('Menlo',11),relief='solid',borderwidth=1).pack(side='left')
    tk.Label(bar,text="mm",bg=BG2,fg=DM,font=('Helvetica',9)).pack(side='left')
    tk.Button(bar,text="RUN 12-WAY",command=lambda:run_comparison(app),
              bg=HL,fg='black',font=('Helvetica',12,'bold'),relief='flat',
              padx=14,cursor='hand2').pack(side='left',padx=16)
    tk.Label(bar,text="(click any heatmap to inspect)",bg=BG2,fg=DM,
             font=('Helvetica',9,'italic')).pack(side='left',padx=8)
    app.cmp_status=tk.StringVar(value="Ready")
    tk.Label(bar,textvariable=app.cmp_status,bg=BG2,fg=YL,
             font=('Menlo',10,'italic')).pack(side='right')
    app.fig_cmp=Figure(figsize=(18,12),facecolor=BG)
    app.c_cmp=FigureCanvasTkAgg(app.fig_cmp,master=parent_frame)
    app.c_cmp.get_tk_widget().pack(fill='both',expand=True)
    # live readout for whatever the cursor is over on a map
    app.cmp_hover=tk.StringVar(value='Hover any map region to see what it is worth.')
    tk.Label(parent_frame,textvariable=app.cmp_hover,bg=BG,fg=HL,anchor='w',
             font=('Menlo',10)).pack(fill='x',side='bottom',padx=10,pady=(0,4))
    app._cmp_configs=[]; app._cmp_axes=[]; app._cmp_click_cid=None; app._cmp_hover_cid=None


def _cyl_block(cp, v, ax, ay, r, zlo, zhi):
    """Boolean mask over rays cp + t*v (t in (0,1)] that pass through a SOLID
    vertical cylinder: radius r about the axis (ax,ay), spanning z in [zlo,zhi].
    A real rotor is a volume, not an infinitely thin disc, so a near-horizontal
    ray grazing the rotor at its own height is correctly counted as blocked."""
    vx,vy,vz=v[:,0],v[:,1],v[:,2]
    dx,dy=cp[0]-ax,cp[1]-ay
    a=vx*vx+vy*vy
    b=2.0*(dx*vx+dy*vy)
    c=dx*dx+dy*dy-r*r
    disc=b*b-4.0*a*c
    a_safe=np.where(a<1e-12,1.0,a)
    sq=np.sqrt(np.maximum(disc,0.0))
    th1=(-b-sq)/(2.0*a_safe); th2=(-b+sq)/(2.0*a_safe)   # ray inside cylinder wall
    vert=a<1e-12                                          # straight-up/down rays
    th1=np.where(vert,0.0,th1); th2=np.where(vert,1.0,th2)
    horiz=np.where(vert,c<0.0,disc>0.0)
    vz_safe=np.where(vz==0.0,1e-9,vz)
    tz1=(zlo-cp[2])/vz_safe; tz2=(zhi-cp[2])/vz_safe     # ray within the z-band
    tzlo=np.minimum(tz1,tz2); tzhi=np.maximum(tz1,tz2)
    flat=np.abs(vz)<1e-9; inband=(cp[2]>=zlo)&(cp[2]<=zhi)
    tzlo=np.where(flat,np.where(inband,0.0,1.0),tzlo)
    tzhi=np.where(flat,np.where(inband,1.0,0.0),tzhi)
    lo=np.maximum(np.maximum(th1,tzlo),1e-3)             # intersect the two intervals
    hi=np.minimum(np.minimum(th2,tzhi),1.0)             # ...within (0,1)
    return horiz&(hi>lo)


def _seg_block(cp, v, q0, q1, rad):
    """Boolean mask over rays cp + s*v (s in (0,1]) that pass within `rad` of the
    line segment q0->q1 ‚Äî i.e. hit the solid capsule (boom) of that radius.
    Vectorised closest-distance between the ray segment and the arm segment."""
    d1=v                                   # (N,3) ray direction
    d2=q1-q0                               # (3,)  arm direction
    r0=cp-q0                               # (3,)
    a=np.einsum('ij,ij->i',d1,d1)          # (N,)
    e=float(d2@d2); f=float(d2@r0)
    b=d1@d2; c=d1@r0                       # (N,)
    denom=a*e-b*b
    s=np.where(np.abs(denom)<1e-9,0.0,(b*f-c*e)/np.where(np.abs(denom)<1e-9,1.0,denom))
    s=np.clip(s,0.0,1.0)
    t=np.clip((b*s+f)/e,0.0,1.0)           # closest param on the arm, clamped
    s=np.clip((b*t-c)/np.where(a<1e-12,1.0,a),0.0,1.0)   # re-solve ray param for that t
    pr=cp+s[:,None]*d1; pa=q0+t[:,None]*d2
    dist2=np.sum((pr-pa)**2,axis=1)
    return (dist2<rad*rad)&(s>1e-3)&(s<1.0)


def _boom_ends(ax, ay, h, R):
    """Where a boom starts and ends. It leaves the BODY SURFACE and runs straight to
    the rotor axis, so it stays attached at any rotor height. (Anchoring it instead at
    the body's half-width sqrt(R^2-h^2) breaks once h > R: that goes to zero and the
    arm starts on the z-axis, floating above the sphere, joined to nothing.)"""
    q1=np.array([ax,ay,h])
    n=float(np.linalg.norm(q1))
    if n<1e-6: return np.array([0.0,0.0,R]),np.array([0.0,0.0,h])
    return q1*(R/n),q1                       # surface point along the rotor's direction


def _prop_block(cp, v, tips, r, h, R):
    """Combined occlusion for one camera: the solid-cylinder rotors AND the solid
    booms carrying them. Each boom runs from the body surface out to its rotor axis
    (a central mast for the single-rotor case)."""
    ht=PROP_T/2.0
    blk=np.zeros(len(v),bool)
    for ax,ay in tips:
        blk|=_cyl_block(cp,v,ax,ay,r,h-ht,h+ht)          # rotor
        q0,q1=_boom_ends(ax,ay,h,R)                       # strut to the body surface
        blk|=_seg_block(cp,v,q0,q1,ARM_RAD)
    return blk


def build_ring_scaffold(counts, R, lats=None, stag=None):
    """Ring scaffold from a partition like [1,4,6,4,1]: rings at evenly spaced
    latitudes pole-to-pole, cameras even in longitude within a ring, alternate
    rings staggered half a spacing (antiprism), each pointing radially outward."""
    k=len(counts); oris=[]
    if lats is None: lats=[90.0-180.0*i/(k-1) for i in range(k)]
    if stag is None: stag=[(180.0/counts[i] if i%2==1 else 0.0) for i in range(k)]
    for i,n in enumerate(counts):
        for j in range(n):
            oris.append((j*360.0/n+stag[i], lats[i], R, 0.0))
    return oris


def _clearance(tips, r, h, oris, R):
    """Minimum MECHANICAL clearance (mm) of a rotor/boom layout ‚Äî negative means
    parts intersect. Without this the scorer happily returns physically impossible
    geometry (rotors buried inside the airframe, or a boom passing through a
    camera) as "optimal", because nothing occludes what is inside the body.
    Checks: rotor-vs-body, rotor-vs-rotor, rotor-vs-camera, boom-vs-camera."""
    ht=PROP_T/2.0
    worst=1e9
    # rotor disc vs the body sphere (evaluated where the body is widest in the band)
    zc=0.0 if abs(h)<=ht else abs(h)-ht
    body_r=math.sqrt(max(R*R-zc*zc,0.0))
    for ax,ay in tips:
        d=math.hypot(ax,ay)
        if d>1e-6:
            worst=min(worst,(d-r)-body_r)
    # rotor vs rotor (adjacent tips on the same circle)
    if len(tips)>1:
        d0=math.hypot(*tips[0])
        if d0>1e-6:
            worst=min(worst,2.0*d0*math.sin(math.pi/len(tips))-2.0*r)
    # rotors and booms vs each camera housing
    for o in oris:
        f,_,_=cam_basis(o[0],o[1],o[3] if len(o)==4 else 0.0)
        cp=f*o[2]
        for ax,ay in tips:
            dh=math.hypot(cp[0]-ax,cp[1]-ay)-r      # radial gap to the rotor wall
            dv=abs(cp[2]-h)-ht                      # vertical gap to the rotor band
            if dh<0 and dv<0:
                gap=max(dh,dv)                      # inside the cylinder
            else:
                gap=math.hypot(max(dh,0.0),max(dv,0.0))
            worst=min(worst,gap-CAM_RAD)
            q0,q1=_boom_ends(ax,ay,h,R)             # boom segment, body surface -> rotor
            w=q1-q0; L2=float(w@w)
            if L2>1e-9:
                t=float(np.clip(((cp-q0)@w)/L2,0.0,1.0))
                worst=min(worst,float(np.linalg.norm(cp-(q0+t*w)))-ARM_RAD-CAM_RAD)
    return worst


def _prop_occluded(oris, cam_vis, eval_d, P, prop_h=None):
    """Apply P solid-cylinder rotors plus their booms to the per-camera visibility
    and pick the best placement. Disc radius scales as 120*sqrt(4/P) so the total
    swept area (lift capacity) is the same for every P; each rotor has real
    vertical thickness PROP_T (prop + hub + motor bell) and sits on a solid boom
    of radius ARM_RAD. P>=2: rotors on evenly spaced arms at ARM_R, rotor plane at
    height prop_h, azimuth swept. P=1: one centred rotor on a mast, height swept.

    Placements that violate mechanical clearance (CLEAR_MIN) are REJECTED, and
    candidates are ranked on coverage with STEREO as the tie-break ‚Äî several
    azimuths tie exactly on coverage while differing by >1pt of stereo, so a
    coverage-only ranking silently threw stereo away.
    Returns (cc, tips, r, h, clearance_mm); clearance < CLEAR_MIN means no legal
    placement existed and the least-illegal one is returned."""
    if prop_h is None: prop_h=PROP_H
    r=PROP_R4*np.sqrt(4.0/P)   # equal total disc area for every rotor count
    Rb=oris[0][2] if len(oris[0])>2 else 300.0
    def build(stride):
        """Per-camera ray geometry, PREFILTERED to rays that could possibly hit a rotor
        at ANY arm azimuth: the ray must cross the rotor z-band and, while inside it,
        have a horizontal radius within the rotor annulus. Everything else is provably
        unblockable and is dropped. The filter does not depend on azimuth, so it is
        built once and reused for the whole sweep ‚Äî this is the speed-up."""
        D=UNIT_DIRS[::stride]
        # the solid is rotor + BOOM, and the boom runs inward to the body surface, so
        # the radial window must start there ‚Äî not at the rotor's inner rim, which
        # silently dropped rays that hit the arm (measured 1.2pt of missed occlusion).
        zh=max(PROP_T/2.0,ARM_RAD)
        zlo,zhi=prop_h-zh,prop_h+zh
        rbody=math.sqrt(max(Rb*Rb-prop_h*prop_h,0.0))
        rin=max(min(ARM_R-r,rbody-ARM_RAD),0.0); rout=ARM_R+r
        g=[]; base=np.zeros(len(D))
        for o,vis in zip(oris,cam_vis):
            f,_,_=cam_basis(o[0],o[1],o[3] if len(o)==4 else 0.0)
            cp=f*o[2]; v=D*(o[2]+eval_d)-cp
            vb=np.asarray(vis,bool)[::stride]; base+=vb
            vz=np.where(v[:,2]==0,1e-9,v[:,2])
            t1=(zlo-cp[2])/vz; t2=(zhi-cp[2])/vz
            tlo=np.maximum(np.minimum(t1,t2),1e-3); thi=np.minimum(np.maximum(t1,t2),1.0)
            ok=thi>tlo
            a2=v[:,0]**2+v[:,1]**2                    # rho(t) is convex, so its extremes
            bx=cp[0]*v[:,0]+cp[1]*v[:,1]              # sit at the endpoints or the vertex
            ts=np.where(a2>1e-12,-bx/np.where(a2>1e-12,a2,1.0),0.0)
            rho=lambda t:np.hypot(cp[0]+t*v[:,0],cp[1]+t*v[:,1])
            tl=np.where(ok,tlo,0.0); th=np.where(ok,thi,0.0)
            rl,rh=rho(tl),rho(th); rv=rho(np.clip(ts,tl,th))
            cand=ok&(np.maximum(rl,rh)>=rin)&(np.minimum(np.minimum(rl,rh),rv)<=rout)
            idx=np.where(cand)[0]
            g.append((cp,v[idx],vb[idx],o[2],idx))
        return (g,base),len(D)
    def occ(gb,n,tips,h):
        g,base=gb
        cnt=base.copy()
        for cp,v,vb,R,idx in g:
            if len(idx)==0: continue
            cnt[idx]-=vb&_prop_block(cp,v,tips,r,h,R)
        return cnt
    cands=[]
    if P==1:
        for h in (350.0,400.0,450.0,500.0):
            cands.append(([(0.0,0.0)],h))
    else:
        per=360.0/P
        for a in np.arange(0.0,per,5.0):
            cands.append(([(ARM_R*np.cos(np.radians(a+per*k)),
                            ARM_R*np.sin(np.radians(a+per*k))) for k in range(P)],prop_h))
    # SHORTLIST the azimuths on a 1-in-4 subsample, then re-score the best few at FULL
    # resolution and pick among those. Shortlisting alone picked a different winner
    # among near-tied azimuths and shifted the answer by ~0.05pt; rescoring removes it.
    gs,ns=build(4)
    legal=[]; fallback=None; fb_key=None
    for tips,h in cands:
        clr=_clearance(tips,r,h,oris,Rb)
        if clr>=CLEAR_MIN:
            cs=occ(gs,ns,tips,h)
            legal.append(((int((cs>0).sum()),int((cs>=2).sum())),tips,h,clr))
        if fb_key is None or clr>fb_key:             # least-illegal, if nothing is legal
            fb_key=clr; fallback=(tips,h,clr)
    gf,nf=build(1)
    if not legal:
        tips,h,clr=fallback
        return occ(gf,nf,tips,h),tips,r,h,clr
    legal.sort(key=lambda x:x[0],reverse=True)
    best=None; bkey=None
    for _,tips,h,clr in legal[:4]:
        cf=occ(gf,nf,tips,h)
        key=(int((cf>0).sum()),int((cf>=2).sum()))   # coverage first, stereo breaks ties
        if bkey is None or key>bkey: bkey=key; best=(cf,tips,r,h,clr)
    return best


PROP_R4=228.0  # QUAD rotor radius (mm) = an 18-inch propeller. NOT a guess: a 600mm
               # sensor sphere with 16 cameras, frame, compute and battery is ~5.3kg;
               # at 2:1 thrust-to-weight that is 26 N per rotor, and at a typical
               # 150 N/m^2 disc loading each rotor needs ~235mm of radius. Other rotor
               # counts hold the same total disc area: r_P = PROP_R4*sqrt(4/P).
               # (The earlier 120mm was a 9-inch placeholder ‚Äî far too small to fly this.)
ARM_LEN=130.0  # BOOM length (mm): body surface -> rotor axis ‚Äî the arm you would cut.
               # A realistic 228mm rotor on a 130mm boom reaches inward to r=202mm,
               # inside the 300mm body, so it CANNOT sit at the equator; the rotor
               # plane has to rise to |h| >= ~256mm where the body has narrowed.
ARM_R=430.0    # propeller arm radius (mm) ‚Äî rotor axis distance from centre
               # (= body radius + ARM_LEN; reset from the GUI's Arm selector)
PROP_H=250.0   # default rotor-plane height (mm) above the equator. +250 is the
               # "slung-ball" sweet spot: equator cams look under the rotors, top
               # cams still look up past their inner edge -> ~0 occlusion any ring.
               # Overridden live by the GUI's Rotor-h selector (0 = equator mount).
PROP_T=50.0    # rotor vertical thickness (mm): 18-inch blade sweep + hub + motor bell
ARM_RAD=15.0   # boom (arm) tube radius (mm)
CAM_RAD=25.0   # camera housing radius (mm) ‚Äî cameras are NOT dimensionless points
CLEAR_MIN=10.0 # minimum mechanical clearance (mm) a placement must keep


def run_comparison(app):
    global ARM_R,PROP_R4        # declared up front: both are read below as defaults
    try:
        R=float(app.base_r.get())
        fh=float(app.fov_h.get()); fv=float(app.fov_v.get())
        eval_d=int(app.cmp_dist_var.get())
        P=int(app.prop_var.get()) if hasattr(app,'prop_var') else 0
        PH=float(app.prop_h_var.get()) if hasattr(app,'prop_h_var') else 250.0
        AL=float(app.arm_var.get()) if hasattr(app,'arm_var') else ARM_LEN
        PR=float(app.prop_r_var.get()) if hasattr(app,'prop_r_var') else PROP_R4
    except:
        app.cmp_status.set("ERROR: bad input"); return
    ARM_R=R+AL          # boom length is what the user picks; rotors sit that far out
    PROP_R4=PR          # quad rotor radius; other counts scale as sqrt(4/P)
    app.cmp_status.set("Running 12 configs..."); app.root.update()

    def work():
        import time
        try:
            t0=time.time()
            def st(m): app.root.after(0,lambda:app.cmp_status.set(m))

            # The camera geometry depends ONLY on (fov, eval distance, body radius) ‚Äî
            # never on any propeller setting. It is also ~99% of the runtime, so cache
            # it: changing Props / Rotor h / Arm / Prop r then re-runs in a second or
            # two instead of minutes.
            key=(fh,fv,eval_d,R)
            cached=_GEOM_CACHE.get(key)
            if cached is None:
                base=[]
                st("[1/6] Ring 1-4-6-4-1...")
                c,l,s=auto_layout('1-4-6-4-1',fh,fv)
                ro=build_orientations_with_radii(c,l,s,[R]*len(c))
                st("[2/6] Fibonacci...")
                fo=fibonacci_spiral_orientations(16,R)
                # raw configs are only SEEDS for the roll optimizer, so a handful of
                # restarts is plenty ‚Äî 100x was wasted work.
                st("[3/6] Thomson...")
                to,_=thomson_solve(16,R,max_iter=1500,n_starts=10)
                st("[4/6] Tammes...")
                ta,_=tammes_solve(16,R,max_iter=1500,n_starts=12)
                st("[5/6] Ring scaffolds (6-ring)...")
                r6a=build_ring_scaffold([1,4,3,3,4,1],R)
                r6b=build_ring_scaffold([1,3,4,4,3,1],R)
                raws=[('1-4-6-4-1',ro),('Fibonacci',fo),('Thomson',to),
                      ('Tammes',ta),('1-4-3-3-4-1',r6a),('1-3-4-4-3-1',r6b)]
                for nm,o in raws: base.append((f'{nm}\n(raw)',o))
                for i,(nm,raw) in enumerate(raws):
                    st(f"[6/6] Optimizing {nm} ({i+1}/6)...")
                    base.append((f'{nm}\n(optimized)',_multistart_roll_opt(raw,fh,fv,eval_d,R)))
                cached=[]
                for lbl,o in base:
                    _,_,_,cc,cv=compute_coverage_nonuniform(o,fh,fv,eval_d)
                    cached.append((lbl,o,cc,cv))
                _GEOM_CACHE[key]=cached
                st("geometry cached ‚Äî propeller changes now re-run instantly")

            st(f"Placing propellers...") if P>0 else None
            # interleave so each configuration's raw tile sits directly beside its
            # optimized one ‚Äî before/after is the comparison you actually make, and
            # in cache order (all six raw, then all six optimized) the pair landed
            # in different rows.
            n=len(cached)//2
            order=[j for i in range(n) for j in (i,i+n)]
            configs=[]
            for k in order:
                lbl,o,cc,cv=cached[k]
                props=None; ccp=None
                if P>0:
                    ccp,tips,r,h,clr=_prop_occluded(o,cv,eval_d,P,prop_h=PH)
                    props=(tips,r,h,clr)
                configs.append((lbl,cc,np.sum(cc>0)/len(cc)*100,o,props,ccp))

            el=time.time()-t0
            app.root.after(0,lambda:draw_comparison(app,configs,R,fh,fv,eval_d,el))
        except Exception as e:
            import traceback; traceback.print_exc()
            app.root.after(0,lambda:app.cmp_status.set(f"ERROR: {e}"))
    threading.Thread(target=work,daemon=True).start()


_GEOM_CACHE={}


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


def _prop_dirs(props, eval_d, R):
    """Which of the 50000 map directions the propeller hardware occupies, seen from
    the drone's centre ‚Äî i.e. where to paint the rotors on the coverage map.
    (Directional silhouette from the origin; each individual camera sits 300mm off
    centre and so sees the rotors at slightly different bearings.)"""
    if not props: return None
    tips,r,h=props[0],props[1],props[2]
    return _prop_block(np.zeros(3),UNIT_DIRS*(R+eval_d),tips,r,h,R)


def _mask_outline(mask, k=7):
    """Boundary of a direction mask: mask points having at least one neighbour
    outside it. Used to draw the propeller as an OUTLINE so the coverage colour
    underneath stays visible ‚Äî a filled overlay wrongly reads as 'the propeller
    consumed this sky' when in fact the cameras still see straight past it."""
    out=np.zeros_like(mask)
    idx=np.where(mask)[0]
    if len(idx)==0: return out
    try:
        _,nb=_dir_tree().query(UNIT_DIRS[idx],k=k)
        out[idx[~mask[nb].all(axis=1)]]=True
    except Exception:
        out[idx]=True
    return out


_DIR_TREE=None
def _dir_tree():
    """KD-tree over the 50000 map directions, built once, for hover lookups."""
    global _DIR_TREE
    if _DIR_TREE is None:
        from scipy.spatial import cKDTree
        _DIR_TREE=cKDTree(UNIT_DIRS)
    return _DIR_TREE


def _region_size(mask, idx):
    """Size (% of sphere) of the connected patch of `mask` containing point idx ‚Äî
    how much sky this one contiguous region is actually worth."""
    sel=np.where(mask)[0]
    if len(sel)==0 or not mask[idx]: return 0.0
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        pairs=np.array(list(cKDTree(UNIT_DIRS[sel]).query_pairs(0.03)))
        if not len(pairs): return 1.0/len(mask)*100
        n=len(sel)
        g=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
        _,lab=connected_components(g,directed=False)
        return float((lab==lab[int(np.searchsorted(sel,idx))]).sum())/len(mask)*100
    except Exception:
        return float(mask.sum())/len(mask)*100


def _hover_report(oris, cc, props, idx, fh, fv, eval_d, R):
    """What the direction under the cursor is worth: which cameras reach it, what the
    propellers take away there, and how big the contiguous region is."""
    d=UNIT_DIRS[idx]; pt=d*(R+eval_d)
    hth=math.tan(math.radians(fh/2)); htv=math.tan(math.radians(fv/2))
    sees=[]; blocked=[]
    for i,o in enumerate(oris):
        f,rt,up=cam_basis(o[0],o[1],o[3] if len(o)==4 else 0.0)
        cp=f*o[2]; v=pt-cp; pf=float(v@f)
        if pf<=0: continue
        if abs(float(v@rt)/pf)<hth and abs(float(v@up)/pf)<htv:
            hit=bool(_prop_block(cp,v.reshape(1,3),props[0],props[1],props[2],o[2])[0]) if props else False
            (blocked if hit else sees).append(i+1)
    lon=math.degrees(math.atan2(d[1],d[0])); lat=math.degrees(math.asin(max(-1.0,min(1.0,d[2]))))
    onprop=bool(_prop_block(np.zeros(3),pt.reshape(1,3),props[0],props[1],props[2],R)[0]) if props else False
    if cc[idx]==0:   state,mask='BLIND ‚Äî no camera reaches this',(cc==0)
    elif cc[idx]==1: state,mask='covered by 1 camera',(cc==1)
    else:            state,mask=f'stereo ‚Äî {int(cc[idx])} cameras',(cc>=2)
    if onprop: state=('RED: propeller sits over a blind spot' if cc[idx]==0
                      else f'ORANGE: propeller here ({state})')
    txt=f"lon {lon:+.0f}¬∞  lat {lat:+.0f}¬∞   |   {state}   |   this region = {_region_size(mask,idx):.2f}% of sphere"
    if sees: txt+=(f"   |   still seen by cameras {sees}" if blocked else f"   |   seen by cameras {sees}")
    if blocked:
        txt+=(f"   |   propeller BLOCKS camera(s) {blocked}"
              +("  ->  this direction is LOST" if not sees
                else f"  ->  {len(sees)+len(blocked)} cameras drop to {len(sees)}"))
    return txt


def draw_comparison(app, configs, R, fh, fv, eval_d, elapsed):
    app.fig_cmp.clear(); app._cmp_configs=configs; app._cmp_axes=[]
    app._cmp_meta=(R,fh,fv,eval_d,elapsed)
    lon=np.arctan2(UNIT_DIRS[:,1],UNIT_DIRS[:,0])
    lat=np.arcsin(np.clip(UNIT_DIRS[:,2],-1,1))
    best_cov=max(c[2] for c in configs)
    bgb=getattr(app,'cmp_scheme',None) and app.cmp_scheme.get()=='bgb'

    for idx,cfg in enumerate(configs):
        label,cc,cov,oris=cfg[:4]
        _pm=_prop_dirs(cfg[4],eval_d,R) if len(cfg)>4 and cfg[4] else None
        _ccp=cfg[5] if len(cfg)>5 else None    # with-propeller counts (None when Props=0)
        # what the propellers actually cost: covered by cameras, blind once rotors exist
        _lost=((cc>0)&(_ccp==0)) if _ccp is not None else np.zeros(len(cc),bool)
        _covp=float((_ccp>0).mean()*100) if _ccp is not None else None
        _step=float((_ccp>=2).mean()*100) if _ccp is not None else None
        _ste=float((cc>=2).mean()*100)
        ax=app.fig_cmp.add_subplot(3,4,idx+1,projection='mollweide',facecolor='white')
        app._cmp_axes.append(ax)
        M=len(cc); blind=int(np.sum(cc==0)); bp=blind/M*100
        if bgb:
            col=np.empty((M,3))
            col[cc==0]=(0.05,0.05,0.05)          # black \u2014 blindspot
            col[cc==1]=(0.11,0.40,0.75)          # blue  \u2014 1 camera
            col[cc>=2]=(0.18,0.49,0.20)          # green \u2014 2+ cameras
            ax.scatter(lon,lat,c=col,s=0.8,edgecolors='none')
            if _pm is not None:
                # RED marks what the propellers actually COST (a direction the cameras
                # cover that goes blind once the rotors are there) \u2014 drawn where the
                # loss really is, per camera. ORANGE is the rotor body that costs
                # nothing. The red area equals the gap between the two title numbers.
                _oo=_mask_outline(_pm&~_lost)      # outline only: don't hide coverage
                ax.scatter(lon[_oo],lat[_oo],c='#ff8c00',s=2.2,
                           edgecolors='none',zorder=6)
                if _lost.any():
                    ax.scatter(lon[_lost],lat[_lost],c='#e60000',s=1.8,
                               edgecolors='none',zorder=7)
            gap,big=_blind_stats(cc)
            # 2dp on coverage so the printed subtraction is exact: at 1dp the rounded
            # numbers did not reproduce the rounded loss (99.9 - 99.3 read as 0.6
            # beside a printed -0.5) and looked like an error.
            _t2=f"Cameras  {cov:.2f}% cov \u00b7 {_ste:.1f}% stereo \u00b7 {bp:.2f}% blind"
            _t3=(("\nWith props  no coverage lost" if cov-_covp<0.005 else
                  f"\nWith props  {_covp:.2f}% cov  (\u2212{cov-_covp:.2f})")
                 +f" \u00b7 {_step:.1f}% stereo  ({_step-_ste:+.1f})"
                 if _covp is not None else f"\nLargest blind region {big:.1f}%")
            ax.set_title(f"{label}\n{_t2}{_t3}",color=FG,fontsize=8,fontweight='bold',pad=6)
        else:
            ax.scatter(lon,lat,c=np.clip(cc,0,5),cmap=HM_CMAP,norm=HM_NORM,s=0.8,alpha=0.85,edgecolors='none')
            if _pm is not None:
                _oo=_mask_outline(_pm&~_lost)      # outline only: don't hide coverage
                ax.scatter(lon[_oo],lat[_oo],c='#ff8c00',s=2.2,
                           edgecolors='none',zorder=6)
                if _lost.any():
                    ax.scatter(lon[_lost],lat[_lost],c='#e60000',s=1.8,
                               edgecolors='none',zorder=7)
            for ori in oris:
                cx=np.radians(ori[0]); cx=cx if cx<=np.pi else cx-2*np.pi
                ax.plot(cx,np.radians(ori[1]),'D',color='black',markersize=3,
                        markeredgecolor='black',markeredgewidth=0.4,zorder=10)
            ib=abs(cov-best_cov)<0.01
            tc=GN if ib else (YL if cov>=99 else ('#e94560' if cov<95 else FG))
            _ptx=(("\nwith props: no coverage lost" if cov-_covp<0.005 else
                   f"\nwith props {_covp:.2f}% cov  (‚àí{cov-_covp:.2f})")
                  +f" | {_step:.1f}% stereo  ({_step-_ste:+.1f})"
                  if _covp is not None else "")
            ax.set_title(f"{label}\n{cov:.2f}% cov | {_ste:.1f}% stereo | {bp:.2f}% blind{_ptx}",
                         color=tc,fontsize=8,fontweight='bold',pad=6)
            if ib: ax.text(0.5,1.30,'best',transform=ax.transAxes,fontsize=8,color=GN,
                           fontweight='bold',ha='center',va='bottom')
        ax.grid(True,alpha=0.15,color='gray',linewidth=0.3); ax.tick_params(colors=DM,labelsize=5)

    if bgb:
        app.fig_cmp.text(0.5,0.045,
            'Blue: observed by one camera.   Green: observed by two or more cameras.   '
            'Black: blindspot (cameras only).   Orange outline: propeller body '
            '(coverage inside is unaffected).   '
            'Red: coverage LOST to the propellers (each camera is blocked along its OWN\n'
            'line of sight, so red need not sit under the orange).',
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
    _pr=next((c[4] for c in configs if len(c)>4 and c[4]),None)
    _pclr=(f', clearance {_pr[3]:.0f}mm' if (_pr and len(_pr)>3 and _pr[3]>=CLEAR_MIN)
           else (f', *** CLEARANCE VIOLATION {_pr[3]:.0f}mm ***' if (_pr and len(_pr)>3) else ''))
    _ptxt=(f',  {len(_pr[0])} props r={_pr[1]:.0f}mm, arm {ARM_R-R:.0f}mm @ h={_pr[2]:+.0f}mm{_pclr}' if _pr else '')
    app.fig_cmp.suptitle(f'12-Way Comparison \u2014 16 Cameras @ R={R:.0f}mm, Eval={eval_d/1000:.1f}m{_ptxt}',
                         color=HL,fontsize=13,fontweight='bold',y=0.98)
    if app._cmp_click_cid is not None: app.c_cmp.mpl_disconnect(app._cmp_click_cid)
    app._cmp_click_cid=app.c_cmp.mpl_connect('button_press_event',
        lambda ev:_on_click(app,ev,R,fh,fv,eval_d))
    if getattr(app,'_cmp_hover_cid',None) is not None: app.c_cmp.mpl_disconnect(app._cmp_hover_cid)
    app._cmp_hover_cid=app.c_cmp.mpl_connect('motion_notify_event',
        lambda ev:_on_hover(app,ev,R,fh,fv,eval_d))
    app.c_cmp.draw()
    app.cmp_status.set(f"Done \u2014 {elapsed:.1f}s  (click any to inspect)")


def _on_hover(app, ev, R, fh, fv, eval_d):
    """Report what the map region under the cursor is worth. Mollweide axes give
    data coords straight in (lon,lat) radians, so the direction is a KD-tree lookup."""
    if ev.inaxes is None or ev.xdata is None or ev.ydata is None:
        return
    try:
        i=app._cmp_axes.index(ev.inaxes)
    except ValueError:
        return
    key=(i,round(float(ev.xdata),3),round(float(ev.ydata),3))
    if getattr(app,'_hover_key',None)==key: return       # same spot ‚Äî skip the work
    app._hover_key=key
    lo,la=float(ev.xdata),float(ev.ydata)
    d=np.array([math.cos(la)*math.cos(lo),math.cos(la)*math.sin(lo),math.sin(la)])
    dist,idx=_dir_tree().query(d)
    if dist>0.06: return                                  # cursor is off the map body
    cfg=app._cmp_configs[i]
    label,cc,cov,oris=cfg[:4]; props=cfg[4] if len(cfg)>4 else None
    try:
        txt=_hover_report(oris,cc,props,int(idx),fh,fv,eval_d,R)
        app.cmp_hover.set(f"{label.replace(chr(10),' ')}:   {txt}")
    except Exception as e:
        app.cmp_hover.set(f"hover error: {e}")


def _on_click(app, ev, R, fh, fv, eval_d):
    if ev.inaxes is None: return
    for i,ax in enumerate(app._cmp_axes):
        if ev.inaxes==ax:
            _open_detail(app,app._cmp_configs[i],R,fh,fv,eval_d); return


def _open_percam(app, oris, fh, fv, label, eval_d, cc_known=None):
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
    if cc_known is not None:
        cc=cc_known
    else:
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
    label,cc,cov,oris=cfg[:4]
    props=cfg[4] if len(cfg)>4 else None
    cl=label.replace('\n',' ')
    win=tk.Toplevel(app.root); win.title(f"Detail: {cl}")
    win.configure(bg=BG); win.geometry("1500x900")

    # Header
    hdr=tk.Frame(win,bg=BG2,pady=6,padx=10); hdr.pack(fill='x')
    M=len(cc); blind=int(np.sum(cc==0))
    stereo=np.sum(cc>=2)/M*100; triple=np.sum(cc>=3)/M*100
    tk.Label(hdr,text=cl,bg=BG2,fg=HL,font=('Helvetica',14,'bold')).pack(side='left',padx=(0,16))
    tk.Button(hdr,text='‚ñ¶ Per-camera coverage',command=lambda:_open_percam(app,oris,fh,fv,cl,eval_d,cc),
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

    state={'drag':None,'zoom1':1.0,'zoom2':1.0}

    def redraw():
        fig.clear()
        ax1=fig.add_subplot(121,projection='3d',facecolor='white')
        _pw=set(i for i,pv in enumerate(pyr_vars) if pv.get())
        _draw_blueprint(ax1,oris,cc,R,fh,fv,eval_d,
                       show_pyr.get(),show_blind.get(),show_lbl.get(),show_grid.get(),
                       state['zoom1'],show_dims.get(),_pw,props)
        ax1.set_title(f'3D Camera Structure\n{cl}',color=FG,fontsize=10,fontweight='bold')

        ax2=fig.add_subplot(122,projection='3d',facecolor='white')
        _draw_dot_sphere(ax2,cc,eval_d,state['zoom2'],oris,R,show_dims.get(),fh,fv,props)
        ax2.set_title(f'3D Coverage Sphere\n{cov:.2f}% coverage',color=FG,fontsize=10,fontweight='bold')

        fig.subplots_adjust(left=0.0,right=1.0,top=0.90,bottom=0.02,wspace=0.0)
        fig.suptitle(f'{cl} \u2014 Detailed Inspection',color=HL,fontsize=13,fontweight='bold',y=0.97)
        state['ax1']=ax1; state['ax2']=ax2
        canvas.draw()

    # ---- zoom controls (right side): the structure sits inside the 2m coverage
    # sphere, so the body needs zooming in on. Each panel zooms independently and
    # goes through redraw() so it works even where trackpad scroll doesn't fire.
    _ZMIN,_ZMAX=0.25,40.0
    def _fit_body_zoom():           # zoom that frames body + arms + rotors
        return max(1.0,(eval_d*1.05)/(R*2.4))
    def set_zoom1(factor=None,fit=False,reset=False):
        if reset: state['zoom1']=1.0
        elif fit: state['zoom1']=_fit_body_zoom()
        else: state['zoom1']=min(_ZMAX,max(_ZMIN,state['zoom1']*factor))
        redraw()
    zf=tk.Frame(tog,bg=BG2); zf.pack(side='right')
    tk.Label(zf,text='structure zoom:',bg=BG2,fg=DM,font=('Helvetica',9)).pack(side='left',padx=(0,2))
    for _t,_cmd in [('‚àí',lambda:set_zoom1(1/1.4)),('Fit body',lambda:set_zoom1(fit=True)),
                    ('+',lambda:set_zoom1(1.4)),('Reset',lambda:set_zoom1(reset=True))]:
        tk.Button(zf,text=_t,command=_cmd,bg=BG2,fg=FG,font=('Helvetica',9,'bold'),
                  relief='flat',borderwidth=1,padx=6,pady=0,highlightbackground=BG2).pack(side='left',padx=1)

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
        f=1.1 if ev.button=='up' else 1/1.1
        # each panel has its own base extent and its own zoom level; blueprint
        # frames eval_d*1.05, the dot sphere sits at sphere_r*1.20 = eval_d*1.14
        if a is state.get('ax1'):
            state['zoom1']=min(_ZMAX,max(_ZMIN,state['zoom1']*f)); base=eval_d*1.05; z=state['zoom1']
        elif a is state.get('ax2'):
            state['zoom2']=min(_ZMAX,max(_ZMIN,state['zoom2']*f)); base=eval_d*1.14; z=state['zoom2']
        else:
            return
        lim=base/z
        a.set_xlim(-lim,lim); a.set_ylim(-lim,lim); a.set_zlim(-lim,lim)
        a.set_box_aspect([1,1,1]); a.set_aspect('equal')
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


def _draw_blueprint(ax, oris, cc, R, fh, fv, eval_d, pyramids, blind, labels, grid, zoom, dims=False, pyr_which=None, props=None):
    """Tab 1 blueprint: wireframe sphere, cameras, FOV pyramids, blind spots, grid cage."""
    ax.disable_mouse_rotation()
    fov_h=np.radians(fh); fov_v=np.radians(fv)

    # Solid-cylinder rotors (orange) on solid booms: top + bottom rings show the
    # rotor thickness; a thick line from the body surface out to each rotor axis
    # is the boom (a central mast for the single-rotor case)
    if props:
        _tips,_pr,_ph=props[0],props[1],props[2]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        _tc=np.linspace(0,2*np.pi,48); _cx=_pr*np.cos(_tc); _cy=_pr*np.sin(_tc)
        for _ax,_ay in _tips:
            # filled solid disc (shaded) so the rotor reads as opaque, not a hoop
            _disc=[list(zip(_ax+_cx,_ay+_cy,np.full_like(_tc,_ph)))]
            ax.add_collection3d(Poly3DCollection(_disc,facecolor='darkorange',
                                                 edgecolor='none',alpha=0.4))
            for _z in (_ph-PROP_T/2.0,_ph+PROP_T/2.0):
                ax.plot(_ax+_cx,_ay+_cy,np.full_like(_tc,_z),
                        color='darkorange',linewidth=1.3,alpha=0.85)
            for _k in range(0,48,12):   # short vertical struts to read as a cylinder
                ax.plot([_ax+_cx[_k]]*2,[_ay+_cy[_k]]*2,[_ph-PROP_T/2.0,_ph+PROP_T/2.0],
                        color='darkorange',linewidth=0.8,alpha=0.7)
            _q0,_q1=_boom_ends(_ax,_ay,_ph,R)   # strut anchored on the body surface
            ax.plot([_q0[0],_q1[0]],[_q0[1],_q1[1]],[_q0[2],_q1[2]],
                    color='#8B4500',linewidth=3.0,alpha=0.9,solid_capstyle='round')

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


def _draw_dot_sphere(ax, cc, eval_d, zoom, oris=None, R=250, show_dims=False, fh=80, fv=65, props=None):
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
            vis=inf&(np.abs(dots@rt/pfs)<hth)&(np.abs(dots@up/pfs)<htv)
            if props:
                _tips,_pr,_ph=props[0],props[1],props[2]
                _cr=o[2] if len(o)>2 else R
                cp=f*_cr
                v=dots*sphere_r-cp
                vis=vis&~_prop_block(cp,v,_tips,_pr,_ph,_cr)
            dot_cc+=vis
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


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

import multiprocessing
import tkinter as tk


class App:
    """Minimal host for the comparison tab: it only needs the three geometry vars."""
    pass


def main():
    root = tk.Tk()
    root.title('Camera Arrangement Comparison')
    root.geometry('1700x1000')
    root.configure(bg=BG)

    app = App()
    app.root = root

    # --- geometry bar (the inputs the comparison tab reads) ---
    top = tk.Frame(root, bg=BG2, pady=6, padx=10)
    top.pack(fill='x')
    tk.Label(top, text='CAMERA COMPARISON', bg=BG2, fg=HL,
             font=('Helvetica', 13, 'bold')).pack(side='left', padx=(0, 20))

    app.fov_h = tk.StringVar(value='80')
    app.fov_v = tk.StringVar(value='65')
    app.base_r = tk.StringVar(value='300')
    for label, var, width in [('FOV H¬∞:', app.fov_h, 5),
                              ('FOV V¬∞:', app.fov_v, 5),
                              ('Base R mm:', app.base_r, 6)]:
        tk.Label(top, text=label, bg=BG2, fg=FG,
                 font=('Helvetica', 11)).pack(side='left', padx=(8, 2))
        tk.Entry(top, textvariable=var, width=width, bg='white', fg=FG,
                 font=('Menlo', 11), relief='solid', borderwidth=1).pack(side='left')

    tk.Label(top, text='  (16 cameras ‚Äî press RUN to build all twelve configurations)',
             bg=BG2, fg=DM, font=('Helvetica', 9, 'italic')).pack(side='left', padx=10)

    # --- the comparison tab itself ---
    tab = tk.Frame(root, bg=BG)
    tab.pack(fill='both', expand=True)
    build_tab6(app, tab)

    root.mainloop()


# ============================================================================
# ENTRY POINT
# The __main__ guard is REQUIRED: the optimizer uses a process pool, and macOS
# spawns workers by re-importing this module. Without the guard every worker
# would rebuild the Tk window.
# ============================================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
 
