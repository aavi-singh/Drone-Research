import tkinter as tk
from tkinter import ttk, font as tkfont
import numpy as np
import math
import threading
import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HUB_RADIUS_MM   = 100
ARM_LENGTH_MM   = 150
FOV_H_DEG       = 55
FOV_V_DEG       = 70
FOV_H           = np.radians(FOV_H_DEG)
FOV_V           = np.radians(FOV_V_DEG)
MC_SAMPLES      = 50000
SOLID_ANGLE_CAM = 4 * np.arcsin(np.sin(FOV_H/2) * np.sin(FOV_V/2))
FULL_SPHERE_SR  = 4 * np.pi

COLORS = {
    'bg': 'white', 'panel': 'whitesmoke', 'border': 'lightgray',
    'accent': 'dodgerblue', 'accent2': 'orangered', 'green': 'forestgreen',
    'red': 'crimson', 'text': 'black', 'dim': 'gray',
    'optimizer': 'mediumpurple', 'gold': 'darkgoldenrod',
}

def generate_sphere_points(n=MC_SAMPLES):
    phi = (1 + np.sqrt(5)) / 2
    pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / phi
        pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return pts

def get_camera_vectors(yaw_deg, pitch_deg):
    yr, pr = np.radians(yaw_deg), np.radians(pitch_deg)
    forward = np.array([np.cos(yr)*np.cos(pr), np.sin(yr)*np.cos(pr), np.sin(pr)])
    up_world = np.array([0, 0, 1.0])
    if abs(np.dot(forward, up_world)) > 0.99:
        up_world = np.array([0, 1, 0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up

def camera_surface_point(fwd, hub_radius, hub_half_h):
    """Place camera on the exterior body face most aligned with its pointing direction.
    
    For a rectangular body (±hub_radius on X/Y, ±hub_half_h on Z):
    - Determines which of the 6 box faces the camera should mount on
      (the face whose outward normal best aligns with the camera forward)
    - Places the camera at that face, offset from center based on the
      remaining direction components so it sits where you'd physically mount it
    """
    d = fwd / np.linalg.norm(fwd)
    
    # The 6 face normals and their limits
    faces = [
        (np.array([ 1, 0, 0]), hub_radius),   # +X face
        (np.array([-1, 0, 0]), hub_radius),   # -X face
        (np.array([ 0, 1, 0]), hub_radius),   # +Y face
        (np.array([ 0,-1, 0]), hub_radius),   # -Y face
        (np.array([ 0, 0, 1]), hub_half_h),   # +Z (top) face
        (np.array([ 0, 0,-1]), hub_half_h),   # -Z (bottom) face
    ]
    
    # Pick the face whose normal is most aligned with camera direction
    best_dot = -1
    best_face = None
    for normal, dist in faces:
        alignment = np.dot(d, normal)
        if alignment > best_dot:
            best_dot = alignment
            best_face = (normal, dist)
    
    normal, dist = best_face
    
    # Camera sits on this face at distance 'dist' along the normal
    # Plus an offset on the face plane based on remaining direction components
    # Find which axes are "on the face" (perpendicular to normal)
    pos = normal * dist
    
    # Project remaining direction onto the face to determine position on face
    face_tangent = d - np.dot(d, normal) * normal  # component parallel to face
    ft_len = np.linalg.norm(face_tangent)
    
    if ft_len > 1e-9:
        face_tangent /= ft_len
        # Determine face extent in the tangent direction
        # Clamp to face boundaries
        limits = np.array([hub_radius, hub_radius, hub_half_h])
        # How far we can go in tangent direction while staying on face
        max_offset = 0
        for ax in range(3):
            if abs(normal[ax]) > 0.5:
                continue  # this is the face normal axis, skip
            if abs(face_tangent[ax]) > 1e-9:
                max_off = limits[ax] / abs(face_tangent[ax])
                if max_offset == 0 or max_off < max_offset:
                    max_offset = max_off
        
        # Push camera toward the edge based on the pitch/direction — use 80% of max
        offset_amount = max_offset * 0.8 * ft_len
        pos = pos + face_tangent * offset_amount
        
        # Clamp to face boundaries
        pos[0] = np.clip(pos[0], -hub_radius, hub_radius)
        pos[1] = np.clip(pos[1], -hub_radius, hub_radius)
        pos[2] = np.clip(pos[2], -hub_half_h, hub_half_h)
    
    return pos

def evaluate_coverage(orientations, num_arms=4, prop_radius=70, eval_radius=300, pts=None,
                      hub_radius=None, hub_height=None, arm_length=None):
    if hub_radius is None:
        hub_radius = HUB_RADIUS_MM
    if hub_height is None:
        hub_height = 15
    if arm_length is None:
        arm_length = ARM_LENGTH_MM

    if pts is None:
        pts_unit = generate_sphere_points(MC_SAMPLES)
    else:
        pts_unit = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    M = len(pts_unit)
    cov_counts = np.zeros(M, dtype=int)

    for yaw_deg, pitch_deg in orientations:
        fwd, right, up = get_camera_vectors(yaw_deg, pitch_deg)
        cam_pos = camera_surface_point(fwd, hub_radius, hub_height / 2)
        # Compute vector from camera position to each sphere point
        pts_world = pts_unit * eval_radius
        to_pts = pts_world - cam_pos  # (M, 3)
        dists = np.linalg.norm(to_pts, axis=1, keepdims=True)
        to_pts_norm = to_pts / np.maximum(dists, 1e-9)
        # Check if each point falls within the camera's FOV
        proj_fwd = to_pts_norm @ fwd
        proj_right = to_pts_norm @ right
        proj_up = to_pts_norm @ up
        in_front = proj_fwd > 0
        ang_h = np.abs(np.arctan2(proj_right, proj_fwd))
        ang_v = np.abs(np.arctan2(proj_up, proj_fwd))
        in_fov = in_front & (ang_h < FOV_H / 2) & (ang_v < FOV_V / 2)
        cov_counts += in_fov.astype(int)

    if num_arms > 0 and prop_radius > 0:
        for arm in range(num_arms):
            angle = 2 * np.pi * arm / num_arms
            arm_dir = np.array([np.cos(angle), np.sin(angle), 0])
            arm_center = arm_dir * (hub_radius + arm_length)
            for pt_idx in range(M):
                if cov_counts[pt_idx] == 0:
                    continue
                pt_dir = pts_unit[pt_idx]
                pt_world = pt_dir * eval_radius
                to_pt = pt_world - arm_center
                up_comp = abs(to_pt[2])
                horiz = np.linalg.norm(to_pt[:2])
                if up_comp < hub_height and horiz < prop_radius * 0.3:
                    cov_counts[pt_idx] = max(0, cov_counts[pt_idx] - 1)

    lats = np.degrees(np.arcsin(np.clip(pts_unit[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(pts_unit[:, 1], pts_unit[:, 0]))

    total_cov = np.sum(cov_counts > 0) / M * 100
    stereo_cov = np.sum(cov_counts >= 2) / M * 100
    blind_n = int(np.sum(cov_counts == 0))
    avg_cams = np.mean(cov_counts[cov_counts > 0]) if np.any(cov_counts > 0) else 0

    eq = np.abs(lats) <= 30
    eq_cov = np.sum(cov_counts[eq] > 0) / eq.sum() * 100 if eq.sum() > 0 else 0

    return {
        'cov_counts': cov_counts,
        'pts_unit': pts_unit,
        'lats': lats,
        'lons': lons,
        'total_coverage': total_cov,
        'stereo_coverage': stereo_cov,
        'blind_spots': blind_n,
        'avg_cameras': avg_cams,
        'equator_coverage': eq_cov,
    }

class DroneVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Research")
        self.root.configure(bg=COLORS['bg'])
        self.root.geometry("1440x920")

        self.total_cameras = tk.IntVar(value=16)
        self.num_arms = tk.IntVar(value=4)
        self.prop_radius = tk.IntVar(value=70)
        self.eval_radius = tk.IntVar(value=2000)
        self.orientations = []
        self.analysis = None
        self._computing = False

        self.hub_radius = tk.IntVar(value=HUB_RADIUS_MM)
        self.hub_height = tk.IntVar(value=50)
        self.arm_length = tk.IntVar(value=ARM_LENGTH_MM)


        self.show_body = tk.BooleanVar(value=True)
        self.show_arms = tk.BooleanVar(value=True)
        self.show_propellers = tk.BooleanVar(value=True)
        self.show_blind = tk.BooleanVar(value=True)
        self.show_dims = tk.BooleanVar(value=True)
        self.cam_visible = []

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background=COLORS['bg'])
        style.configure('Dark.TLabel', background=COLORS['bg'], foreground=COLORS['text'],
                        font=('Inter', 10))
        style.configure('Title.TLabel', background=COLORS['bg'], foreground=COLORS['text'],
                        font=('Inter', 22, 'bold'))

        self._build_ui()
        self.root.after(200, self._compute)

    def _build_ui(self):
        main = ttk.Frame(self.root, style='Dark.TFrame')
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        header = ttk.Frame(main, style='Dark.TFrame')
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Drone Research", style='Title.TLabel').pack()

        ctrl_frame = tk.Frame(main, bg=COLORS['panel'], highlightbackground=COLORS['border'],
                              highlightthickness=1, padx=12, pady=8)
        ctrl_frame.pack(fill=tk.X, pady=(0, 8))

        sliders_row = tk.Frame(ctrl_frame, bg=COLORS['panel'])
        sliders_row.pack(fill=tk.X, pady=(4, 0))

        self._make_slider(sliders_row, "Total Cameras", self.total_cameras, 1, 40, 0)
        self._make_slider(sliders_row, "Arms", self.num_arms, 0, 8, 1)
        self._make_slider(sliders_row, "Prop Radius (mm)", self.prop_radius, 0, 150, 2)

        body_row = tk.Frame(ctrl_frame, bg=COLORS['panel'])
        body_row.pack(fill=tk.X, pady=(6, 0))
        for label, var in [("Body Radius (mm)", self.hub_radius),
                           ("Body Height (mm)", self.hub_height),
                           ("Arm Length (mm)", self.arm_length)]:
            f = tk.Frame(body_row, bg=COLORS['panel'])
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            tk.Label(f, text=label, font=('Inter', 8), fg=COLORS['dim'],
                     bg=COLORS['panel']).pack(anchor='w')
            e = tk.Entry(f, textvariable=var, font=('Menlo', 10), width=6,
                        bg=COLORS['bg'], fg=COLORS['text'], relief='flat',
                        highlightbackground=COLORS['border'], highlightthickness=1)
            e.pack(fill=tk.X)
            var.trace_add('write', lambda *args: self._schedule_redraw())

        btn_frame = tk.Frame(ctrl_frame, bg=COLORS['panel'])
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        self.compute_btn = tk.Button(btn_frame, text="COMPUTE", font=('Inter', 10, 'bold'),
                                     fg='white', bg=COLORS['optimizer'], activebackground='mediumpurple',
                                     relief='flat', padx=16, pady=3, command=self._compute)
        self.compute_btn.pack(side=tk.RIGHT)

        self.status_label = tk.Label(btn_frame, text="Ready", font=('Inter', 9),
                                     fg=COLORS['dim'], bg=COLORS['panel'])
        self.status_label.pack(side=tk.RIGHT, padx=12)

        self.metrics_frame = tk.Frame(main, bg=COLORS['bg'])
        self.metrics_frame.pack(fill=tk.X, pady=(0, 6))

        content = tk.Frame(main, bg=COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        viewer_frame = tk.Frame(content, bg=COLORS['panel'], highlightbackground=COLORS['border'],
                                highlightthickness=1)
        viewer_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 6))

        self.fig_3d = Figure(figsize=(8, 5.5), facecolor=COLORS['bg'])
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d', facecolor=COLORS['bg'])
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=viewer_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        side_frame = tk.Frame(content, bg=COLORS['panel'], highlightbackground=COLORS['border'],
                              highlightthickness=1, padx=10, pady=8)
        side_frame.grid(row=0, column=1, sticky='nsew')

        tk.Label(side_frame, text="VISIBILITY", font=('Inter', 9, 'bold'),
                 fg=COLORS['dim'], bg=COLORS['panel']).pack(anchor='w')
        toggle_row = tk.Frame(side_frame, bg=COLORS['panel'])
        toggle_row.pack(fill=tk.X, pady=(2, 6))
        for text, var in [("Body", self.show_body), ("Arms", self.show_arms),
                          ("Props", self.show_propellers), ("Blind", self.show_blind),
                          ("Dims", self.show_dims)]:
            cb = tk.Checkbutton(toggle_row, text=text, variable=var,
                               fg=COLORS['text'], bg=COLORS['panel'], selectcolor=COLORS['border'],
                               activebackground=COLORS['panel'], font=('Inter', 8),
                               command=self._redraw_3d)
            cb.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(side_frame, text="CAMERAS", font=('Inter', 9, 'bold'),
                 fg=COLORS['dim'], bg=COLORS['panel']).pack(anchor='w', pady=(4, 0))
        self.cam_checkbox_frame = tk.Frame(side_frame, bg=COLORS['panel'])
        self.cam_checkbox_frame.pack(fill=tk.X, pady=(2, 6))

    def _make_slider(self, parent, label, var, min_v, max_v, col):
        frame = tk.Frame(parent, bg=COLORS['panel'])
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        top = tk.Frame(frame, bg=COLORS['panel'])
        top.pack(fill=tk.X)
        tk.Label(top, text=label, font=('Inter', 9), fg=COLORS['dim'],
                 bg=COLORS['panel']).pack(side=tk.LEFT)
        val_label = tk.Label(top, text=str(var.get()), font=('Menlo', 10, 'bold'),
                             fg=COLORS['accent'], bg=COLORS['panel'])
        val_label.pack(side=tk.RIGHT)

        slider = tk.Scale(frame, from_=min_v, to=max_v, variable=var, orient='horizontal',
                          bg=COLORS['panel'], fg=COLORS['text'], troughcolor=COLORS['border'],
                          highlightthickness=0, sliderrelief='flat', length=180,
                          showvalue=False, font=('Inter', 8),
                          command=lambda v, vl=val_label: vl.configure(text=str(int(float(v)))))
        slider.pack(fill=tk.X)

    def _compute(self):
        if self._computing:
            return
        self._computing = True
        self._compute_start = time.time()
        self.compute_btn.configure(state='disabled', text='Computing...')
        self.status_label.configure(text='Running Greedy Set Cover...')
        self.root.update()

        thread = threading.Thread(target=self._compute_worker, daemon=True)
        thread.start()
        self.root.after(200, lambda: self._check_thread(thread))

    def _check_thread(self, thread):
        if thread.is_alive():
            elapsed = time.time() - self._compute_start
            self.status_label.configure(text=f'Optimizing... {elapsed:.0f}s')
            self.root.after(500, lambda: self._check_thread(thread))
        else:
            self._update_views()
            self._computing = False
            self.compute_btn.configure(state='normal', text='COMPUTE')

    def _compute_worker(self):
        t0 = time.time()
        n = self.total_cameras.get()
        n_arms = self.num_arms.get()
        p_radius = self.prop_radius.get()
        e_radius = self.eval_radius.get()
        hub_r = self.hub_radius.get()
        hub_h = self.hub_height.get() // 2
        arm_l = self.arm_length.get()

        pts = generate_sphere_points(MC_SAMPLES)
        M = len(pts)
        half_tan_h = np.tan(FOV_H / 2)
        half_tan_v = np.tan(FOV_V / 2)
        pts_world = pts * e_radius  # precompute world positions once

        def _vis(yaw_deg, pitch_deg):
            fwd, right, up = get_camera_vectors(yaw_deg, pitch_deg)
            cam_pos = camera_surface_point(fwd, hub_r, hub_h)
            to_pts = pts_world - cam_pos
            # Project onto camera axes (no need to normalize — use ratios)
            pf = to_pts @ fwd
            in_front = pf > 0
            pf_safe = np.where(in_front, pf, 1.0)
            ah = np.abs((to_pts @ right) / pf_safe)
            av = np.abs((to_pts @ up) / pf_safe)
            return in_front & (ah < half_tan_h) & (av < half_tan_v)

        yaw_grid = np.arange(-180, 180, 5.0)
        pitch_grid = np.arange(-90, 91, 5.0)
        cands = [(float(y), float(p)) for y in yaw_grid for p in pitch_grid]
        K = len(cands)

        cand_vis = np.zeros((K, M), dtype=bool)
        for ki, (y, p) in enumerate(cands):
            cand_vis[ki] = _vis(y, p)

        uncovered = np.ones(M, dtype=bool)
        greedy_idx = []
        used = set()
        for _ in range(n):
            if np.any(uncovered):
                marginal = np.sum(cand_vis[:, uncovered], axis=1)
            else:
                marginal = np.sum(cand_vis, axis=1)
            for si in used:
                marginal[si] = 0
            best_k = int(np.argmax(marginal))
            greedy_idx.append(best_k)
            used.add(best_k)
            uncovered &= ~cand_vis[best_k]

        best_ori = [cands[k] for k in greedy_idx]

        vis_cache = [_vis(y, p) for y, p in best_ori]
        cov_counts = np.zeros(M, dtype=int)
        for v in vis_cache:
            cov_counts += v.astype(int)
        best_cov = np.sum(cov_counts > 0) / M * 100

        rng = np.random.default_rng(42)
        cur_ori = list(best_ori)
        cur_cov = best_cov
        T = 12.0
        n_sa = 100000
        alpha_sa = (0.005 / 12.0) ** (1.0 / n_sa)

        for it in range(n_sa):
            ci = rng.integers(n)
            oy, op = cur_ori[ci]
            scale = max(0.3, T * 0.5)
            ny = oy + rng.normal(0, scale)
            npv = float(np.clip(op + rng.normal(0, scale), -90, 90))
            if ny > 180: ny -= 360
            if ny < -180: ny += 360

            old_vis = vis_cache[ci]
            new_vis = _vis(ny, npv)

            cov_counts -= old_vis.astype(int)
            cov_counts += new_vis.astype(int)
            nc = np.sum(cov_counts > 0) / M * 100

            delta = nc - cur_cov
            if delta > 0 or rng.random() < math.exp(delta / max(T, 0.001)):
                cur_cov = nc
                vis_cache[ci] = new_vis
                cur_ori[ci] = (ny, npv)
                if nc > best_cov:
                    best_ori = list(cur_ori)
                    best_cov = nc
            else:
                cov_counts -= new_vis.astype(int)
                cov_counts += old_vis.astype(int)
            T *= alpha_sa

        self.orientations = best_ori
        self._method_label = 'Greedy Set Cover + SA'

        self.analysis = evaluate_coverage(
            self.orientations, n_arms, p_radius, e_radius,
            hub_radius=hub_r, hub_height=hub_h,
            arm_length=arm_l)

        self._compute_time = time.time() - t0

    def _update_views(self):
        a = self.analysis
        if a is None:
            return

        for w in self.cam_checkbox_frame.winfo_children():
            w.destroy()
        self.cam_visible = []
        n = len(self.orientations)
        cols = 4
        for ci in range(n):
            var = tk.BooleanVar(value=True)
            self.cam_visible.append(var)
            cb = tk.Checkbutton(self.cam_checkbox_frame, text=f"C{ci+1}", variable=var,
                               fg=COLORS['text'], bg=COLORS['panel'], selectcolor=COLORS['border'],
                               activebackground=COLORS['panel'], font=('Menlo', 8),
                               command=self._redraw_3d)
            cb.grid(row=ci // cols, column=ci % cols, sticky='w', padx=1)

        self._update_metrics(a)
        self._update_3d(a)

        self.status_label.configure(
            text=f"{self._method_label} — {self._compute_time*1000:.0f}ms")

    def _redraw_3d(self):
        if self.analysis is not None:
            self._update_3d(self.analysis)

    def _make_metric_card(self, parent, title, value, color, subtitle=""):
        card = tk.Frame(parent, bg=COLORS['panel'], highlightbackground=COLORS['border'],
                        highlightthickness=1, padx=10, pady=6)
        card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        tk.Label(card, text=title.upper(), font=('Inter', 7, 'bold'),
                 fg=COLORS['dim'], bg=COLORS['panel']).pack()
        tk.Label(card, text=str(value), font=('Menlo', 15, 'bold'),
                 fg=color, bg=COLORS['panel']).pack()
        if subtitle:
            tk.Label(card, text=subtitle, font=('Inter', 7),
                     fg=COLORS['dim'], bg=COLORS['panel']).pack()

    def _update_metrics(self, a):
        for w in self.metrics_frame.winfo_children():
            w.destroy()

        blind_color = COLORS['green'] if a['blind_spots'] == 0 else COLORS['red']
        cov_color = COLORS['green'] if a['total_coverage'] > 98 else COLORS['gold'] if a['total_coverage'] > 90 else COLORS['red']

        self._make_metric_card(self.metrics_frame, "Coverage",
                               f"{a['total_coverage']:.2f}%", cov_color, "of full sphere")
        self._make_metric_card(self.metrics_frame, "Blind Spots",
                               f"{a['blind_spots']}", blind_color,
                               "ZERO" if a['blind_spots'] == 0 else "remaining gaps")
        self._make_metric_card(self.metrics_frame, "Stereo Overlap",
                               f"{a['stereo_coverage']:.1f}%", COLORS['accent'], "2+ cameras")
        self._make_metric_card(self.metrics_frame, "Cameras",
                               f"{len(self.orientations)}", COLORS['optimizer'],
                               self._method_label)

    def _schedule_redraw(self):
        """Debounced redraw — waits 300ms after last change before redrawing."""
        if hasattr(self, '_redraw_timer') and self._redraw_timer is not None:
            self.root.after_cancel(self._redraw_timer)
        self._redraw_timer = self.root.after(300, self._safe_redraw)

    def _safe_redraw(self):
        """Redraw with error handling for partial input in entry fields."""
        try:
            self._redraw_3d()
        except (tk.TclError, ValueError):
            pass  # Ignore errors from empty/partial entry fields

    def _update_3d(self, a):
        try:
            saved_elev = self.ax_3d.elev
            saved_azim = self.ax_3d.azim
        except:
            saved_elev, saved_azim = 25, -60

        if not hasattr(self, '_zoom'):
            self._zoom = 1.0

        # Disconnect old event handlers to prevent stacking
        if hasattr(self, '_cids'):
            for cid in self._cids:
                self.canvas_3d.mpl_disconnect(cid)
        self._cids = []

        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d', facecolor=COLORS['bg'])
        self.ax_3d = ax

        ax.disable_mouse_rotation()

        self._drag_start = None
        def _on_press(event):
            if event.inaxes == self.ax_3d and event.button == 1:
                self._drag_start = (event.x, event.y, self.ax_3d.elev, self.ax_3d.azim)
        def _on_release(event):
            self._drag_start = None
        def _on_drag(event):
            if self._drag_start is None or event.button != 1:
                return
            x0, y0, elev0, azim0 = self._drag_start
            dx = event.x - x0
            dy = event.y - y0
            sensitivity = 0.3
            new_azim = azim0 - dx * sensitivity
            new_elev = max(-80, min(80, elev0 + dy * sensitivity))
            self.ax_3d.view_init(elev=new_elev, azim=new_azim)
            self.canvas_3d.draw_idle()

        self._cids.append(self.canvas_3d.mpl_connect('button_press_event', _on_press))
        self._cids.append(self.canvas_3d.mpl_connect('button_release_event', _on_release))
        self._cids.append(self.canvas_3d.mpl_connect('motion_notify_event', _on_drag))

        eval_r = self.eval_radius.get()
        num_arms = self.num_arms.get()
        prop_r = self.prop_radius.get()
        hub_r = self.hub_radius.get()
        hub_h = self.hub_height.get() / 2
        arm_l = self.arm_length.get()

        t = np.linspace(0, 2 * np.pi, 60)
        for axis in ['xy', 'xz', 'yz']:
            if axis == 'xy':
                ax.plot(eval_r*np.cos(t), eval_r*np.sin(t), np.zeros_like(t),
                       color='silver', alpha=0.35, linewidth=0.6)
            elif axis == 'xz':
                ax.plot(eval_r*np.cos(t), np.zeros_like(t), eval_r*np.sin(t),
                       color='silver', alpha=0.35, linewidth=0.6)
            else:
                ax.plot(np.zeros_like(t), eval_r*np.cos(t), eval_r*np.sin(t),
                       color='silver', alpha=0.35, linewidth=0.6)

        cc = a['cov_counts']
        pts = a['pts_unit']
        blind_mask = cc == 0
        if blind_mask.sum() > 0 and self.show_blind.get():
            blind_dirs = pts[blind_mask]
            for d in blind_dirs:
                tip = d * eval_r * 1.5
                ax.plot([0, tip[0]], [0, tip[1]], [0, tip[2]],
                       color='crimson', linewidth=1.2, alpha=0.7)

        if self.show_body.get():
            bc = 'dimgray'
            ec = 'gray'
            r, h = hub_r, hub_h
            corners_top = [[-r, -r, h], [r, -r, h], [r, r, h], [-r, r, h]]
            corners_bot = [[-r, -r, -h], [r, -r, -h], [r, r, -h], [-r, r, -h]]
            ax.add_collection3d(Poly3DCollection([corners_top], alpha=0.3, facecolor=bc, edgecolor=ec, linewidth=0.5))
            ax.add_collection3d(Poly3DCollection([corners_bot], alpha=0.3, facecolor=bc, edgecolor=ec, linewidth=0.5))
            for i in range(4):
                j = (i + 1) % 4
                side = [corners_top[i], corners_top[j], corners_bot[j], corners_bot[i]]
                ax.add_collection3d(Poly3DCollection([side], alpha=0.2, facecolor=bc, edgecolor=ec, linewidth=0.3))

        if num_arms > 0 and self.show_arms.get():
            for arm in range(num_arms):
                angle = 2 * np.pi * arm / num_arms
                # Arm starts at body edge, extends outward
                sx = hub_r * np.cos(angle)  # start at body edge
                sy = hub_r * np.sin(angle)
                ex = (hub_r + arm_l) * np.cos(angle)  # motor at body edge + arm length
                ey = (hub_r + arm_l) * np.sin(angle)
                ax.plot([sx, ex], [sy, ey], [0, 0], color='gray', linewidth=2, alpha=0.6)
                if prop_r > 0 and self.show_propellers.get():
                    pc = np.linspace(0, 2*np.pi, 12)
                    ax.plot(ex + prop_r*np.cos(pc), ey + prop_r*np.sin(pc),
                            np.zeros(12), color='orangered', linewidth=1, alpha=0.5)

        cam_color = (0.2, 0.7, 0.3)

        def _surface_point(direction):
            return camera_surface_point(direction, hub_r, hub_h)

        for ci, (yaw, pitch) in enumerate(self.orientations):
            fwd, right, up = get_camera_vectors(yaw, pitch)
            cam_pos = _surface_point(fwd)

            # Camera dot — always visible
            ax.scatter(*cam_pos, c=[cam_color], s=45, zorder=5,
                      edgecolors='darkslategray', linewidths=0.6, depthshade=False)
            # Camera label
            label_dir = fwd * hub_r * 0.4
            ax.text(cam_pos[0]+label_dir[0], cam_pos[1]+label_dir[1], cam_pos[2]+label_dir[2],
                    f'C{ci+1}', fontsize=5, color='darkslategray', fontweight='bold',
                    ha='center', va='center', zorder=6)

            # FOV pyramid — only when camera checkbox is checked
            show_pyramid = ci >= len(self.cam_visible) or self.cam_visible[ci].get()
            if show_pyramid:
                fl = eval_r
                hw = fl * math.tan(FOV_H / 2)
                vw = fl * math.tan(FOV_V / 2)
                fc = cam_pos + fwd * fl
                c0 = fc + right*hw + up*vw
                c1 = fc - right*hw + up*vw
                c2 = fc - right*hw - up*vw
                c3 = fc + right*hw - up*vw

                faces = [
                    [cam_pos, c0, c1], [cam_pos, c1, c2],
                    [cam_pos, c2, c3], [cam_pos, c3, c0],
                    [c0, c1, c2, c3],
                ]
                poly = Poly3DCollection(faces, alpha=0.2, facecolor=cam_color,
                                         edgecolor=(*cam_color, 0.4), linewidth=0.5)
                ax.add_collection3d(poly)

                for corner in [c0, c1, c2, c3]:
                    ax.plot([cam_pos[0], corner[0]], [cam_pos[1], corner[1]], [cam_pos[2], corner[2]],
                           color=cam_color, alpha=0.3, linewidth=0.6)

        # ─── DIMENSION ANNOTATIONS ───
        if self.show_dims.get():
            dim_color = '#1a73e8'
            dim_font = 6
            dim_lw = 1.0
            dim_alpha = 0.85
            offset = hub_r * 0.35  # offset for dimension lines

            def _dim_line(p1, p2, label, side_offset=None, font_size=None):
                """Draw a measurement line between two 3D points with a label."""
                fs = font_size or dim_font
                p1 = np.array(p1, dtype=float)
                p2 = np.array(p2, dtype=float)
                # Line
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=dim_color, linewidth=dim_lw, alpha=dim_alpha,
                        linestyle='--')
                # End ticks (small perpendicular marks)
                for pt in [p1, p2]:
                    ax.scatter(*pt, c=[dim_color], s=12, zorder=6,
                              marker='|', linewidths=0.8)
                # Label at midpoint
                mid = (p1 + p2) / 2
                if side_offset is not None:
                    mid = mid + np.array(side_offset, dtype=float)
                ax.text(mid[0], mid[1], mid[2], label,
                        fontsize=fs, color=dim_color, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor=dim_color, alpha=0.85, linewidth=0.5))

            # --- Body width (front edge, top face) ---
            _dim_line([-hub_r, -hub_r, hub_h + offset],
                      [hub_r, -hub_r, hub_h + offset],
                      f'{int(hub_r*2)}mm',
                      side_offset=[0, -offset*0.5, offset*0.3])
            # Tie-lines from edge to dimension line
            ax.plot([-hub_r, -hub_r], [-hub_r, -hub_r], [hub_h, hub_h + offset],
                    color=dim_color, linewidth=0.5, alpha=0.5)
            ax.plot([hub_r, hub_r], [-hub_r, -hub_r], [hub_h, hub_h + offset],
                    color=dim_color, linewidth=0.5, alpha=0.5)

            # --- Body height (front-right edge) ---
            _dim_line([hub_r + offset, -hub_r, -hub_h],
                      [hub_r + offset, -hub_r, hub_h],
                      f'{int(hub_h*2)}mm',
                      side_offset=[offset*0.6, 0, 0])
            ax.plot([hub_r, hub_r + offset], [-hub_r, -hub_r], [hub_h, hub_h],
                    color=dim_color, linewidth=0.5, alpha=0.5)
            ax.plot([hub_r, hub_r + offset], [-hub_r, -hub_r], [-hub_h, -hub_h],
                    color=dim_color, linewidth=0.5, alpha=0.5)

            # --- Arm lengths ---
            if num_arms > 0:
                # Label first arm only to avoid clutter
                angle0 = 0
                sx0 = hub_r * np.cos(angle0)
                sy0 = hub_r * np.sin(angle0)
                ex0 = (hub_r + arm_l) * np.cos(angle0)
                ey0 = (hub_r + arm_l) * np.sin(angle0)
                _dim_line([sx0, sy0, -offset * 0.8],
                          [ex0, ey0, -offset * 0.8],
                          f'Arm {int(arm_l)}mm',
                          side_offset=[0, -offset*0.8, -offset*0.3])

            # --- Propeller diameter (first prop) ---
            if num_arms > 0 and prop_r > 0:
                angle0 = 0
                cx = (hub_r + arm_l) * np.cos(angle0)
                cy = (hub_r + arm_l) * np.sin(angle0)
                _dim_line([cx - prop_r, cy, offset * 0.6],
                          [cx + prop_r, cy, offset * 0.6],
                          f'Prop ⌀{int(prop_r*2)}mm',
                          side_offset=[0, -offset*0.5, offset*0.4])

            # --- Camera-to-camera distances (nearest neighbor for each) ---
            all_cams = []
            for ci2, (yaw2, pitch2) in enumerate(self.orientations):
                fwd2, _, _ = get_camera_vectors(yaw2, pitch2)
                pos2 = _surface_point(fwd2)
                all_cams.append((ci2, pos2))

            if len(all_cams) >= 2:
                # Find nearest neighbor for each camera, draw unique pairs
                drawn_pairs = set()
                for i_idx, (ci_a, pos_a) in enumerate(all_cams):
                    best_dist = float('inf')
                    best_j = -1
                    for j_idx, (ci_b, pos_b) in enumerate(all_cams):
                        if i_idx == j_idx:
                            continue
                        d = np.linalg.norm(pos_a - pos_b)
                        if d < best_dist:
                            best_dist = d
                            best_j = j_idx
                    if best_j >= 0:
                        pair = tuple(sorted([i_idx, best_j]))
                        if pair not in drawn_pairs:
                            drawn_pairs.add(pair)
                            pos_b = all_cams[best_j][1]
                            ci_b = all_cams[best_j][0]
                            mid = (pos_a + pos_b) / 2
                            # Offset label outward from center
                            outward = mid / (np.linalg.norm(mid) + 1e-9) * offset * 0.6
                            ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
                                    [pos_a[2], pos_b[2]],
                                    color='darkorange', linewidth=0.8, alpha=0.6,
                                    linestyle=':')
                            ax.text(mid[0]+outward[0], mid[1]+outward[1], mid[2]+outward[2],
                                    f'{best_dist:.0f}mm',
                                    fontsize=5, color='darkorange', fontweight='bold',
                                    ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                              edgecolor='darkorange', alpha=0.8, linewidth=0.4))

        L = eval_r * 1.0

        for s1 in [-L, L]:
            for s2 in [-L, L]:
                ax.plot([s1, s1], [s2, s2], [-L, L], color='darkgray', linewidth=0.5, alpha=0.35)
                ax.plot([s1, s1], [-L, L], [s2, s2], color='darkgray', linewidth=0.5, alpha=0.35)
                ax.plot([-L, L], [s1, s1], [s2, s2], color='darkgray', linewidth=0.5, alpha=0.35)

        grid_vals = np.arange(-L, L + 1, L * 0.25)
        for g in grid_vals:
            ax.plot([g, g], [-L, L], [-L, -L], color='lightgray', linewidth=0.2, alpha=0.3)
            ax.plot([-L, L], [g, g], [-L, -L], color='lightgray', linewidth=0.2, alpha=0.3)
            ax.plot([g, g], [-L, -L], [-L, L], color='lightgray', linewidth=0.2, alpha=0.3)
            ax.plot([-L, L], [-L, -L], [g, g], color='lightgray', linewidth=0.2, alpha=0.3)
            ax.plot([-L, -L], [g, g], [-L, L], color='lightgray', linewidth=0.2, alpha=0.3)
            ax.plot([-L, -L], [-L, L], [g, g], color='lightgray', linewidth=0.2, alpha=0.3)

        for v in np.arange(-L, L + 1, L * 0.5):
            iv = int(v)
            ax.text(v, -L*1.1, -L, f'{iv}', fontsize=5, color='dimgray', ha='center')
            ax.text(-L*1.1, v, -L, f'{iv}', fontsize=5, color='dimgray', ha='center')
            ax.text(-L*1.1, -L, v, f'{iv}', fontsize=5, color='dimgray', ha='center')

        ax.text(0, -L*1.22, -L, 'X', fontsize=7, color='dimgray', ha='center', fontweight='bold')
        ax.text(-L*1.22, 0, -L, 'Y', fontsize=7, color='dimgray', ha='center', fontweight='bold')
        ax.text(-L*1.22, -L, 0, 'Z', fontsize=7, color='dimgray', ha='center', fontweight='bold')

        lim = eval_r * 1.05 / self._zoom
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_axis_off()
        ax.view_init(elev=saved_elev, azim=saved_azim)

        def on_scroll(event):
            if event.button == 'up':
                self._zoom = min(30.0, self._zoom * 1.1)
            else:
                self._zoom = max(0.2, self._zoom / 1.1)
            lim = eval_r * 1.05 / self._zoom
            self.ax_3d.set_xlim(-lim, lim); self.ax_3d.set_ylim(-lim, lim); self.ax_3d.set_zlim(-lim, lim)
            self.canvas_3d.draw_idle()

        def on_key(event):
            if event.key in ('+', '='):
                self._zoom = min(30.0, self._zoom * 1.15)
            elif event.key in ('-', '_'):
                self._zoom = max(0.2, self._zoom / 1.15)
            else:
                return
            lim = eval_r * 1.05 / self._zoom
            self.ax_3d.set_xlim(-lim, lim); self.ax_3d.set_ylim(-lim, lim); self.ax_3d.set_zlim(-lim, lim)
            self.canvas_3d.draw_idle()

        self._cids.append(self.canvas_3d.mpl_connect('scroll_event', on_scroll))
        self._cids.append(self.canvas_3d.mpl_connect('key_press_event', on_key))

        self.fig_3d.tight_layout(pad=0.5)
        self.canvas_3d.draw()

if __name__ == '__main__':
    print("Running...")
    root = tk.Tk()
    app = DroneVisionGUI(root)
    root.mainloop()
