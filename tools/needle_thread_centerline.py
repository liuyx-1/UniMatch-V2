"""Geometric centerline extraction for needle + thread segmentation masks.

Two-class binary masks in (often fragmented) come in. Pipeline:

  1. Per-class cleanup: drop tiny components, close small intra-fragment gaps.
  2. Per-component centerline: skeletonize -> 8-connected graph -> longest
     path via double-BFS (graph diameter). Returned as an ordered polyline.
  3. Spline smoothing: parametric B-spline over (x, y).
  4. Fragment merging by trajectory continuity. Same-class fragments are
     glued if their endpoints are close AND the outgoing tangents are
     anti-parallel (consistent direction of travel). Iterative greedy.
  5. Needle <-> thread junction: closest endpoint of the needle centerline
     to ANY point on the thread centerline (not just thread endpoints --
     the thread end can be occluded, so the visible-thread point nearest
     to the needle is the actual junction). That needle endpoint is the
     needle TAIL; the other endpoint is the needle HEAD.
  6. Equidistant sampling: 10 arc-length-equidistant points along the
     oriented needle centerline, head -> tail.

Public entry point::

    from tools.needle_thread_centerline import extract_needle_thread_geometry
    out = extract_needle_thread_geometry(needle_mask, thread_mask)

Output dict keys::

    needle_centerline    : (M, 2) float32, ordered (x, y), HEAD -> TAIL
    thread_centerline    : (N, 2) float32, ordered (x, y), JUNCTION -> free end
    needle_head          : (x, y) float32
    needle_tail          : (x, y) float32  (== junction point on needle side)
    junction_thread_pt   : (x, y) float32  (closest thread point to tail)
    junction_distance    : float pixel gap between needle_tail and thread
    needle_sample_points : (10, 2) float32, head -> tail, equidistant
    needle_fragments     : list of (K, 2) -- residual unmerged needle curves
    thread_fragments     : list of (K, 2) -- residual unmerged thread curves

Any field may be None if its mask is empty or geometric checks fail.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import remove_small_objects, skeletonize


# ----------------------------------------------------------------------
# 1. mask cleanup
# ----------------------------------------------------------------------
def _binarize(mask) -> np.ndarray:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m > 0
    return m.astype(np.uint8)


def _cleanup_mask(mask, min_area: int = 30, close_radius: int = 1,
                  bridge_radius: int = 0) -> np.ndarray:
    """Drop tiny CCs, close intra-fragment gaps, optionally bridge
    inter-fragment gaps.

    Args:
        close_radius : small morphological CLOSE (fills 1-2 px holes inside
                       a fragment without merging separate fragments).
        bridge_radius: larger CLOSE with rectangular kernel applied AFTER
                       small-area removal. Use this when mask predictions
                       are broken into multiple pieces along the same
                       structure (typical for thread). A value of 5-10
                       usually bridges gaps up to 2*bridge_radius pixels.
    """
    m = _binarize(mask)
    if m is None or m.sum() == 0:
        return np.zeros_like(m) if m is not None else None
    m = remove_small_objects(m.astype(bool), min_size=min_area).astype(np.uint8)
    if close_radius > 0 and m.sum() > 0:
        k = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    if bridge_radius > 0 and m.sum() > 0:
        k = 2 * bridge_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


# ----------------------------------------------------------------------
# 2. skeleton -> graph -> longest path (per component)
# ----------------------------------------------------------------------
def _neighbors8(y: int, x: int, H: int, W: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx


def _skel_adjacency(skel: np.ndarray) -> dict:
    H, W = skel.shape
    ys, xs = np.where(skel)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    adj: dict = {p: [] for p in pixels}
    for (y, x) in pixels:
        for ny, nx in _neighbors8(y, x, H, W):
            if (ny, nx) in pixels:
                adj[(y, x)].append((ny, nx))
    return adj


def _prune_spurs(adj: dict, min_branch_len: int = 5) -> dict:
    """Iteratively trim leaf branches shorter than min_branch_len pixels.

    Helps when skeletonize produces small Y-spurs from rough mask edges.
    """
    adj = {k: list(v) for k, v in adj.items()}
    changed = True
    while changed:
        changed = False
        leaves = [n for n, nbrs in adj.items() if len(nbrs) == 1]
        for leaf in leaves:
            if leaf not in adj:
                continue
            # walk inward up to min_branch_len; if we hit a junction within
            # that many steps, drop the walked nodes.
            path = [leaf]
            prev = None
            cur = leaf
            for _ in range(min_branch_len):
                nbrs = [n for n in adj.get(cur, []) if n != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
            # if the next step lands on a junction (>= 3 neighbors), prune
            nbrs = [n for n in adj.get(cur, []) if n != prev]
            if len(nbrs) >= 2:  # cur is a junction; trim path[:-1] (keep junction)
                for p in path[:-1]:
                    for q in list(adj.get(p, [])):
                        adj.get(q, []).remove(p) if p in adj.get(q, []) else None
                    adj.pop(p, None)
                changed = True
    return adj


def _bfs_farthest(adj: dict, start) -> Tuple[Tuple[int, int], dict]:
    parent = {start: None}
    dq = deque([start])
    far = start
    while dq:
        u = dq.popleft()
        far = u
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                dq.append(v)
    return far, parent


def _diameter_path(adj: dict) -> List[Tuple[int, int]]:
    if not adj:
        return []
    start = next(iter(adj))
    a, _ = _bfs_farthest(adj, start)
    b, par = _bfs_farthest(adj, a)
    path: List[Tuple[int, int]] = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = par[cur]
    return path  # b -> a


def _pca_axis_polyline(comp_mask: np.ndarray, n: int = 20) -> Optional[np.ndarray]:
    """Project mask pixels onto their PCA major axis, return polyline.

    Useful for tiny / blob-like fragments where skeletonize is unstable
    (the needle in close-up frames often looks blob-shaped).
    """
    ys, xs = np.where(comp_mask > 0)
    if len(xs) < 5:
        return None
    P = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = P.mean(axis=0)
    Pc = P - mean
    cov = np.cov(Pc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # major
    proj = Pc @ axis
    t_lo, t_hi = float(proj.min()), float(proj.max())
    if t_hi - t_lo < 2.0:
        return None
    ts = np.linspace(t_lo, t_hi, n)
    pts = mean[None, :] + ts[:, None] * axis[None, :]
    return pts.astype(np.float32)


def _component_centerline(comp_mask: np.ndarray,
                          prune_len: int = 5,
                          pca_fallback_area: int = 0
                          ) -> Optional[np.ndarray]:
    """Skeleton -> graph -> diameter path; returns ordered (x, y) array.

    If pca_fallback_area > 0 and the component has fewer than that many
    pixels, skip skeletonization and use PCA major-axis projection instead
    (more stable for tiny blob-like fragments).
    """
    area = int((comp_mask > 0).sum())
    if pca_fallback_area > 0 and area < pca_fallback_area:
        pts = _pca_axis_polyline(comp_mask)
        if pts is not None:
            return pts
    skel = skeletonize(comp_mask > 0)
    if skel.sum() < 2:
        # last-ditch PCA if skel failed entirely
        return _pca_axis_polyline(comp_mask)
    adj = _skel_adjacency(skel.astype(np.uint8))
    adj = _prune_spurs(adj, min_branch_len=prune_len)
    if not adj:
        return None
    # split into connected sub-graphs (pruning may disconnect things)
    seen: set = set()
    best: List[Tuple[int, int]] = []
    for node in list(adj.keys()):
        if node in seen:
            continue
        # BFS-collect this sub-graph
        sub: dict = {}
        dq = deque([node])
        while dq:
            u = dq.popleft()
            if u in seen:
                continue
            seen.add(u)
            sub[u] = adj[u]
            for v in adj[u]:
                if v not in seen:
                    dq.append(v)
        path = _diameter_path(sub)
        if len(path) > len(best):
            best = path
    if len(best) < 2:
        return None
    pts = np.array([(x, y) for (y, x) in best], dtype=np.float32)
    return pts


# ----------------------------------------------------------------------
# 3. smoothing
# ----------------------------------------------------------------------
def _dedup(pts: np.ndarray, tol: float = 0.5) -> np.ndarray:
    if len(pts) < 2:
        return pts
    diff = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], diff > tol])
    return pts[keep]


def _smooth_curve(pts: np.ndarray,
                  n_out: int = 200,
                  s_factor: float = 2.0) -> np.ndarray:
    """Parametric B-spline smoothing. Falls back to raw points if spline fails."""
    pts = _dedup(pts)
    if len(pts) < 4:
        return pts
    try:
        s = s_factor * len(pts)
        k = min(3, len(pts) - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, k=k)
        u_new = np.linspace(0.0, 1.0, n_out)
        xs, ys = splev(u_new, tck)
        return np.stack([xs, ys], axis=1).astype(np.float32)
    except Exception:
        return pts


# ----------------------------------------------------------------------
# 4. fragment merging
# ----------------------------------------------------------------------
def _endpoint_tangent(curve: np.ndarray, end: str, k: int = 5) -> np.ndarray:
    """Outward-pointing unit tangent at the chosen endpoint.

    ``end='start'``: tangent points OUT of the curve at index 0.
    ``end='end'``  : tangent points OUT of the curve at index -1.
    """
    if len(curve) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    k = min(k, len(curve) - 1)
    if end == 'start':
        p_in, p_out = curve[k], curve[0]
    else:
        p_in, p_out = curve[-1 - k], curve[-1]
    v = p_out - p_in
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _curvature_magnitude(curve: np.ndarray, end: str, k: int = 8) -> float:
    """Approx curvature near an endpoint: angle change over arc length."""
    if len(curve) < 3:
        return 0.0
    if end == 'start':
        sub = curve[:min(k, len(curve))]
    else:
        sub = curve[-min(k, len(curve)):]
    if len(sub) < 3:
        return 0.0
    d1 = sub[1] - sub[0]
    d2 = sub[-1] - sub[-2]
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(d1, d2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos))) / max(1.0, len(sub))


def _merge_score(curve_a: np.ndarray, curve_b: np.ndarray,
                 dist_thresh: float, angle_thresh_deg: float,
                 curv_diff_thresh: float = 25.0
                 ) -> Tuple[float, Optional[str], Optional[str]]:
    """Score the best way to glue curve_b onto curve_a.

    Returns (score, a_end, b_end). +inf if no pair satisfies thresholds.
    Lower score = better merge.
    """
    a_endpts = [('start', curve_a[0]), ('end', curve_a[-1])]
    b_endpts = [('start', curve_b[0]), ('end', curve_b[-1])]
    best = (np.inf, None, None)
    for a_end, a_pt in a_endpts:
        for b_end, b_pt in b_endpts:
            dist = float(np.linalg.norm(a_pt - b_pt))
            if dist > dist_thresh:
                continue
            ta = _endpoint_tangent(curve_a, a_end)
            tb = _endpoint_tangent(curve_b, b_end)
            # if we join a@a_end to b@b_end, the curve direction should be
            # continuous: outward tangent of a should ~= INVERSE of outward
            # tangent of b (since b's "outward" at the joining end points
            # away from the join).
            cos = float(np.clip(np.dot(ta, -tb), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos)))
            if angle > angle_thresh_deg:
                continue
            curv_a = _curvature_magnitude(curve_a, a_end)
            curv_b = _curvature_magnitude(curve_b, b_end)
            if abs(curv_a - curv_b) > curv_diff_thresh:
                continue
            score = dist + 2.0 * angle + 0.5 * abs(curv_a - curv_b)
            if score < best[0]:
                best = (score, a_end, b_end)
    return best


def _join_curves(curve_a: np.ndarray, curve_b: np.ndarray,
                 a_end: str, b_end: str) -> np.ndarray:
    """Glue b onto a so the result is one continuous polyline."""
    a = curve_a if a_end == 'end' else curve_a[::-1]
    b = curve_b if b_end == 'start' else curve_b[::-1]
    return np.concatenate([a, b], axis=0).astype(np.float32)


def _merge_fragments(curves: Sequence[np.ndarray],
                     dist_thresh: float,
                     angle_thresh_deg: float,
                     curv_diff_thresh: float = 25.0
                     ) -> List[np.ndarray]:
    """Greedily merge geometrically continuous fragments."""
    curves = [c for c in curves if c is not None and len(c) >= 2]
    while len(curves) > 1:
        best = (np.inf, -1, -1, None, None)
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                score, ae, be = _merge_score(
                    curves[i], curves[j],
                    dist_thresh, angle_thresh_deg, curv_diff_thresh)
                if score < best[0]:
                    best = (score, i, j, ae, be)
        if not np.isfinite(best[0]):
            break
        _, i, j, ae, be = best
        merged = _join_curves(curves[i], curves[j], ae, be)
        new_list = [c for k, c in enumerate(curves) if k != i and k != j]
        new_list.append(merged)
        curves = new_list
    curves.sort(key=lambda c: -len(c))
    return curves


# ----------------------------------------------------------------------
# 5. needle <-> thread junction
# ----------------------------------------------------------------------
def _nearest_point_to_curve(point: np.ndarray, curve: np.ndarray
                            ) -> Tuple[float, int]:
    d = np.linalg.norm(curve - point[None, :], axis=1)
    i = int(np.argmin(d))
    return float(d[i]), i


# ----------------------------------------------------------------------
# 6. equidistant sampling
# ----------------------------------------------------------------------
def _equidistant_sample(curve: np.ndarray, n: int = 10) -> np.ndarray:
    if len(curve) < 2:
        return np.tile(curve[:1], (n, 1))
    diffs = np.diff(curve, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.tile(curve[:1], (n, 1))
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    j = 0
    for k, t in enumerate(targets):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        denom = cum[j + 1] - cum[j]
        if denom < 1e-6:
            out[k] = curve[j]
        else:
            r = (t - cum[j]) / denom
            out[k] = curve[j] * (1.0 - r) + curve[j + 1] * r
    return out


# ----------------------------------------------------------------------
# helpers: per-mask component centerlines
# ----------------------------------------------------------------------
def _curve_to_curve_min_dist(c1: np.ndarray, c2: np.ndarray) -> float:
    """Min point-to-point distance between two polylines (greedy O(N1*N2),
    fine for typical N<=200)."""
    if c1 is None or c2 is None or len(c1) == 0 or len(c2) == 0:
        return float('inf')
    # sample every k points to keep cost bounded
    k1 = max(1, len(c1) // 50)
    k2 = max(1, len(c2) // 50)
    a = c1[::k1]; b = c2[::k2]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min())


def _reject_bad_components(curves: List[np.ndarray],
                            class_mask: np.ndarray,
                            max_area: int,
                            min_aspect: float,
                            reject_log: list,
                            class_name: str) -> List[np.ndarray]:
    """Drop curves whose owning connected component fails area / aspect
    geometry checks (heuristic false-positive filter for instrument body
    or blob-shaped mis-segmentations).
    """
    if max_area <= 0 and min_aspect <= 0.0:
        return curves
    if class_mask is None or class_mask.sum() == 0:
        return curves
    n_lab, lab = cv2.connectedComponents(class_mask.astype(np.uint8))
    # build a per-pixel CC lookup for fast assignment of curves
    kept = []
    for c in curves:
        # take the curve midpoint pixel, look up its CC label
        if len(c) == 0:
            continue
        mid = c[len(c) // 2]
        x, y = int(round(mid[0])), int(round(mid[1]))
        x = max(0, min(class_mask.shape[1] - 1, x))
        y = max(0, min(class_mask.shape[0] - 1, y))
        cc_id = int(lab[y, x])
        if cc_id == 0:
            # midpoint fell on background; sample a few more points to find owner
            cc_id = 0
            for p in c[::max(1, len(c) // 5)]:
                xi = max(0, min(class_mask.shape[1] - 1, int(round(p[0]))))
                yi = max(0, min(class_mask.shape[0] - 1, int(round(p[1]))))
                if lab[yi, xi] > 0:
                    cc_id = int(lab[yi, xi])
                    break
        if cc_id == 0:
            kept.append(c)
            continue
        comp_mask = (lab == cc_id).astype(np.uint8)
        area, aspect, bbox, centroid = _component_geom_stats(comp_mask)
        if max_area > 0 and area > max_area:
            reject_log.append({'class': class_name, 'reason': 'area>max',
                                'area': area, 'limit': int(max_area)})
            continue
        if min_aspect > 0 and aspect < min_aspect:
            reject_log.append({'class': class_name, 'reason': 'aspect<min',
                                'aspect': aspect, 'limit': float(min_aspect)})
            continue
        kept.append(c)
    return kept


def _component_geom_stats(comp_mask: np.ndarray):
    """Return (area, aspect_ratio, bbox(x,y,w,h), centroid(x,y))."""
    ys, xs = np.where(comp_mask > 0)
    if len(xs) < 3:
        return 0, 1.0, (0, 0, 0, 0), (0.0, 0.0)
    area = int(len(xs))
    P = np.stack([xs, ys], axis=1).astype(np.float32)
    Pc = P - P.mean(axis=0)
    cov = np.cov(Pc.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    # ratio of major / minor PCA axes ≈ length / width
    aspect = float(np.sqrt(eigvals[-1] / eigvals[0]))
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return area, aspect, (x0, y0, x1 - x0 + 1, y1 - y0 + 1), \
           (float(xs.mean()), float(ys.mean()))


def _all_centerlines(mask: np.ndarray,
                     prune_len: int,
                     smooth_n: int,
                     smooth_s: float,
                     pca_fallback_area: int = 0,
                     max_area: int = 0,
                     min_aspect: float = 0.0,
                     reject_log: list = None,
                     class_name: str = '') -> List[np.ndarray]:
    if mask is None or mask.sum() == 0:
        return []
    n_lab, lab = cv2.connectedComponents(mask.astype(np.uint8))
    curves: List[np.ndarray] = []
    for k in range(1, n_lab):
        comp = (lab == k).astype(np.uint8)
        pts = _component_centerline(comp, prune_len=prune_len,
                                     pca_fallback_area=pca_fallback_area)
        if pts is None:
            continue
        if len(pts) >= 4:
            curves.append(_smooth_curve(pts, n_out=smooth_n, s_factor=smooth_s))
        else:
            curves.append(pts)
    return curves


# ----------------------------------------------------------------------
# top-level entry
# ----------------------------------------------------------------------
def extract_needle_thread_geometry(
    needle_mask,
    thread_mask,
    *,
    min_area: int = 30,
    close_radius: int = 1,
    needle_bridge_radius: int = 0,
    thread_bridge_radius: int = 0,
    needle_pca_fallback_area: int = 0,
    thread_pca_fallback_area: int = 0,
    prune_len: int = 5,
    smooth_n: int = 200,
    smooth_s: float = 2.0,
    needle_dist_thresh: float = 60.0,
    needle_angle_thresh_deg: float = 50.0,
    needle_curv_diff_thresh: float = 30.0,
    thread_dist_thresh: float = 120.0,
    thread_angle_thresh_deg: float = 40.0,
    thread_curv_diff_thresh: float = 25.0,
    n_sample: int = 10,
    needle_template: str = 'auto',
    needle_template_pad_deg: float = 3.0,
    needle_template_min_inlier_ratio: float = 0.35,
    needle_template_inlier_thresh: float = 3.0,
    # ── stronger rigid-arc constraints (post-fit validation) ───────────
    needle_template_radius_min: float = 0.0,    # px; reject fits with r < this
    needle_template_radius_max: float = 0.0,    # px; reject fits with r > this
    needle_template_arc_min_deg: float = 0.0,   # reject fits covering < this arc span
    needle_template_arc_max_deg: float = 0.0,   # reject fits covering > this (rules out full circles)
    # ── post-segmentation false-positive filters ───────────────────────
    needle_max_area: int = 0,            # drop needle CCs larger than this (instrument body false-positive)
    needle_min_aspect: float = 0.0,      # drop needle CCs with PCA major/minor < this (blob-shape false-positive)
    needle_must_be_near_thread: bool = False,
    needle_near_thread_dist: float = 80.0,  # px gap above which a needle candidate is rejected (when thread is present)
    thread_max_area: int = 0,
    thread_min_aspect: float = 0.0,
) -> dict:
    """Run the full geometric pipeline.

    Args:
        needle_mask, thread_mask: HxW binary (bool or uint8).
        min_area: drop CCs with fewer than this many pixels.
        close_radius: morphological close kernel half-size (0 disables).
        prune_len: trim skeleton spurs shorter than this many pixels.
        smooth_n: number of resampled points after spline smoothing.
        smooth_s: spline smoothing factor (higher = smoother).
        needle_*: fragment-merge tolerances for the needle (stricter, since
                  the needle arc is short and a wrong merge is catastrophic).
        thread_*: fragment-merge tolerances for the thread (looser, since
                  the thread is long, often broken into many pieces).
        n_sample: number of equidistant samples along the needle, head->tail.
    """
    out = dict(
        needle_centerline=None, thread_centerline=None,
        needle_head=None, needle_tail=None,
        junction_thread_pt=None, junction_distance=None,
        needle_sample_points=None,
        needle_fragments=[], thread_fragments=[],
        needle_template_fit=None,
    )

    nm = _cleanup_mask(needle_mask, min_area=min_area,
                        close_radius=close_radius,
                        bridge_radius=needle_bridge_radius)
    tm = _cleanup_mask(thread_mask, min_area=min_area,
                        close_radius=close_radius,
                        bridge_radius=thread_bridge_radius)

    # ---- optional rigid-arc template fit on the needle ----
    # This is a SECOND, geometric-prior-driven centerline candidate. It
    # fits a single circle (or ellipse) to ALL needle mask pixels at once,
    # which is essential when the mask is fragmented (multiple connected
    # components). If the fit is reliable, we replace the spline-derived
    # needle centerline with the arc-sampled curve.
    template_curve: Optional[np.ndarray] = None
    template_fit: Optional[dict] = None
    if needle_template in ('auto', 'circle', 'ellipse', 'force'):
        try:
            from tools.needle_arc_template import fit_needle_arc, sample_template_centerline
            template_fit = fit_needle_arc(
                nm,
                inlier_thresh=needle_template_inlier_thresh,
                min_inlier_ratio=needle_template_min_inlier_ratio,
            )
            if template_fit is not None:
                # post-fit rigid validation: reject implausible arc params
                _r = float(template_fit.get('radius', 0.0))
                _span = float(template_fit.get('arc_span_deg', 0.0))
                _reject = None
                if needle_template_radius_min > 0 and _r < needle_template_radius_min:
                    _reject = f'r={_r:.1f} < min={needle_template_radius_min:.1f}'
                elif needle_template_radius_max > 0 and _r > needle_template_radius_max:
                    _reject = f'r={_r:.1f} > max={needle_template_radius_max:.1f}'
                elif needle_template_arc_min_deg > 0 and _span < needle_template_arc_min_deg:
                    _reject = f'arc={_span:.1f} < min={needle_template_arc_min_deg:.1f}'
                elif needle_template_arc_max_deg > 0 and _span > needle_template_arc_max_deg:
                    _reject = f'arc={_span:.1f} > max={needle_template_arc_max_deg:.1f}'
                if _reject is not None:
                    template_fit['rejected_reason'] = _reject
                    template_fit = None
                    template_curve = None
                else:
                    template_curve = sample_template_centerline(
                        template_fit, n=smooth_n,
                        pad_deg=needle_template_pad_deg)
        except Exception as e:  # never crash the spline pipeline
            template_fit = None
            template_curve = None
            print(f'[needle_template] fit failed: {e}')
    out['needle_template_fit'] = template_fit

    reject_log = []
    needle_curves = _all_centerlines(nm, prune_len, smooth_n, smooth_s,
                                      pca_fallback_area=needle_pca_fallback_area,
                                      reject_log=reject_log, class_name='needle')
    thread_curves = _all_centerlines(tm, prune_len, smooth_n, smooth_s,
                                      pca_fallback_area=thread_pca_fallback_area,
                                      reject_log=reject_log, class_name='thread')

    # ── per-class geometric false-positive filter (BEFORE fragment merge) ──
    needle_curves = _reject_bad_components(
        needle_curves, nm,
        max_area=needle_max_area,
        min_aspect=needle_min_aspect,
        reject_log=reject_log, class_name='needle')
    thread_curves = _reject_bad_components(
        thread_curves, tm,
        max_area=thread_max_area,
        min_aspect=thread_min_aspect,
        reject_log=reject_log, class_name='thread')

    needle_curves = _merge_fragments(
        needle_curves,
        dist_thresh=needle_dist_thresh,
        angle_thresh_deg=needle_angle_thresh_deg,
        curv_diff_thresh=needle_curv_diff_thresh)
    thread_curves = _merge_fragments(
        thread_curves,
        dist_thresh=thread_dist_thresh,
        angle_thresh_deg=thread_angle_thresh_deg,
        curv_diff_thresh=thread_curv_diff_thresh)

    # ── thread-proximity filter on needle (after thread curves are merged) ──
    if needle_must_be_near_thread and thread_curves:
        kept = []
        for nc in needle_curves:
            d_min = _curve_to_curve_min_dist(nc, thread_curves[0])
            if d_min <= float(needle_near_thread_dist):
                kept.append(nc)
            else:
                reject_log.append({'class': 'needle', 'reason': 'far_from_thread',
                                    'dist': float(d_min)})
        needle_curves = kept

    out['_reject_log'] = reject_log

    if not needle_curves and template_curve is None:
        return out

    # ---- choose primary needle centerline ----
    # auto: prefer template if its arc supports >= 60% of spline length and
    #       fit residual is small. force: always use template if available.
    use_template = False
    if needle_template == 'force' and template_curve is not None:
        use_template = True
    elif needle_template in ('auto', 'circle', 'ellipse') \
            and template_curve is not None:
        if not needle_curves:
            use_template = True
        else:
            # Both candidates exist: prefer arc when the arc inlier ratio is
            # high (mask points actually lie on a single arc) OR when the
            # spline path is short relative to the arc (sign of fragmentation
            # that the spline couldn't bridge).
            spline_len = float(np.sum(np.linalg.norm(
                np.diff(needle_curves[0], axis=0), axis=1)))
            arc_len = float(np.sum(np.linalg.norm(
                np.diff(template_curve, axis=0), axis=1)))
            ir = template_fit.get('inlier_ratio', 0.0) \
                if template_fit is not None else 0.0
            if ir >= 0.45 and arc_len > 0.8 * spline_len:
                use_template = True
            elif ir >= 0.6:
                use_template = True

    if use_template:
        needle_curve = template_curve
        out['needle_fragments'] = [c.astype(np.float32) for c in needle_curves]
    else:
        needle_curve = needle_curves[0] if needle_curves else template_curve
        out['needle_fragments'] = [c.astype(np.float32) for c in needle_curves[1:]] \
            if needle_curves else []
    out['thread_fragments'] = [c.astype(np.float32) for c in thread_curves[1:]] \
        if len(thread_curves) > 1 else []

    # ---- needle <-> thread coupling ----
    if thread_curves:
        thread_curve = thread_curves[0]
        # For each needle endpoint, find the nearest point on the entire
        # thread polyline (not just thread endpoints -- the thread end may
        # be occluded near the needle).
        n_ends = [needle_curve[0], needle_curve[-1]]
        cands = []
        for i, p in enumerate(n_ends):
            d, idx = _nearest_point_to_curve(p, thread_curve)
            cands.append((d, i, idx))
        cands.sort()
        d_best, tail_idx, thread_pt_idx = cands[0]
        # orient needle so it runs head -> tail
        if tail_idx == 0:
            needle_curve = needle_curve[::-1]
        # orient thread so it runs junction -> free end
        if thread_pt_idx > len(thread_curve) / 2:
            thread_curve = thread_curve[::-1]
        out['needle_centerline'] = needle_curve
        out['thread_centerline'] = thread_curve
        out['needle_head'] = needle_curve[0].astype(np.float32)
        out['needle_tail'] = needle_curve[-1].astype(np.float32)
        out['junction_distance'] = float(d_best)
        # recompute thread point now that thread may be reversed
        _, idx2 = _nearest_point_to_curve(out['needle_tail'], thread_curve)
        out['junction_thread_pt'] = thread_curve[idx2].astype(np.float32)
    else:
        # No thread visible: keep needle as-is. Head/tail are still both
        # endpoints; downstream caller can choose with another heuristic
        # (e.g. tip sharpness, motion prior).
        out['needle_centerline'] = needle_curve
        out['needle_head'] = needle_curve[0].astype(np.float32)
        out['needle_tail'] = needle_curve[-1].astype(np.float32)

    # ---- equidistant samples along the (oriented) needle ----
    out['needle_sample_points'] = _equidistant_sample(
        out['needle_centerline'], n=n_sample)

    return out


# ----------------------------------------------------------------------
# convenience: pull needle / thread masks out of a multi-class seg map
# ----------------------------------------------------------------------
def extract_from_label_map(label_map: np.ndarray,
                           needle_id: int = 1,
                           thread_id: int = 2,
                           **kwargs) -> dict:
    """Helper: split a HxW int label map by class id then run the pipeline."""
    nm = (label_map == needle_id).astype(np.uint8)
    tm = (label_map == thread_id).astype(np.uint8)
    return extract_needle_thread_geometry(nm, tm, **kwargs)


# ----------------------------------------------------------------------
# visualization helper (optional)
# ----------------------------------------------------------------------
def draw_result(image: np.ndarray, result: dict,
                needle_color=(0, 255, 0),
                thread_color=(255, 128, 0),
                fragment_color=(128, 128, 128),
                head_color=(0, 0, 255),
                tail_color=(0, 255, 255),
                sample_color=(255, 255, 255),
                thickness: int = 2,
                sample_radius: int = 4) -> np.ndarray:
    """Render centerlines + key points on an image (BGR or grayscale).

    Useful when iterating on thresholds. NOT used by the algorithm.
    """
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def _poly(curve, color, th):
        if curve is None or len(curve) < 2:
            return
        pts = curve.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), color, th)

    for c in result.get('thread_fragments', []):
        _poly(c, fragment_color, max(1, thickness - 1))
    for c in result.get('needle_fragments', []):
        _poly(c, fragment_color, max(1, thickness - 1))
    _poly(result.get('thread_centerline'), thread_color, thickness)
    _poly(result.get('needle_centerline'), needle_color, thickness)

    if result.get('needle_head') is not None:
        hp = result['needle_head'].astype(int)
        cv2.circle(vis, tuple(hp), sample_radius + 3, head_color, -1)
        cv2.circle(vis, tuple(hp), sample_radius + 5, (255, 255, 255), 1)
        cv2.putText(vis, 'HEAD (tip)', (int(hp[0]) + 10, int(hp[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 2, cv2.LINE_AA)
    if result.get('needle_tail') is not None:
        tp = result['needle_tail'].astype(int)
        cv2.circle(vis, tuple(tp), sample_radius + 3, tail_color, -1)
        cv2.circle(vis, tuple(tp), sample_radius + 5, (255, 255, 255), 1)
        cv2.putText(vis, 'TAIL (to thread)', (int(tp[0]) + 10, int(tp[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tail_color, 2, cv2.LINE_AA)
    if result.get('junction_thread_pt') is not None and \
       result.get('needle_tail') is not None:
        a = tuple(result['needle_tail'].astype(int))
        b = tuple(result['junction_thread_pt'].astype(int))
        cv2.line(vis, a, b, tail_color, 1, cv2.LINE_AA)

    sp = result.get('needle_sample_points')
    if sp is not None:
        for i, p in enumerate(sp):
            pt = (int(p[0]), int(p[1]))
            # white outer ring (= visible against any background)
            cv2.circle(vis, pt, sample_radius + 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, pt, sample_radius, sample_color, -1)
            # bigger numeric label, with black halo for legibility
            label = str(i + 1)
            org = (pt[0] + 7, pt[1] - 6)
            cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        sample_color, 1, cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------------
# CLI: run on a label-map PNG and save a visualization
# ----------------------------------------------------------------------
def _cli() -> int:
    import argparse, os
    ap = argparse.ArgumentParser(
        description='Extract needle/thread centerlines from a seg label PNG.')
    ap.add_argument('--label', required=True,
                    help='HxW PNG with needle/thread class ids')
    ap.add_argument('--image', default=None,
                    help='optional RGB image for visualization')
    ap.add_argument('--needle-id', type=int, default=1)
    ap.add_argument('--thread-id', type=int, default=2)
    ap.add_argument('--out', default='centerline_vis.png')
    ap.add_argument('--min-area', type=int, default=30)
    ap.add_argument('--n-sample', type=int, default=10)
    args = ap.parse_args()

    lab = cv2.imread(args.label, cv2.IMREAD_UNCHANGED)
    if lab is None:
        print(f'failed to read label: {args.label}')
        return 1
    if lab.ndim == 3:
        lab = lab[..., 0]
    res = extract_from_label_map(lab, needle_id=args.needle_id,
                                  thread_id=args.thread_id,
                                  min_area=args.min_area,
                                  n_sample=args.n_sample)
    print('junction_distance :', res.get('junction_distance'))
    print('needle_head       :', res.get('needle_head'))
    print('needle_tail       :', res.get('needle_tail'))
    sp = res.get('needle_sample_points')
    if sp is not None:
        print('needle_sample_points:')
        for i, p in enumerate(sp):
            print(f'  {i + 1:2d}: ({p[0]:.1f}, {p[1]:.1f})')

    if args.image is not None and os.path.isfile(args.image):
        img = cv2.imread(args.image)
    else:
        img = np.zeros((lab.shape[0], lab.shape[1], 3), dtype=np.uint8)
        img[lab == args.needle_id] = (0, 100, 0)
        img[lab == args.thread_id] = (100, 50, 0)
    vis = draw_result(img, res)
    cv2.imwrite(args.out, vis)
    print(f'[ok] wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
