"""Rigid-arc template fitting for surgical needles.

Surgical needles are arcs of a known geometric family (1/4, 3/8, 1/2
circle). A segmentation network can break the needle into multiple
disjoint mask fragments (occluding instrument, specular highlights,
thresholding noise). Skeleton + spline on those fragments can never
recover the true arc -- it just connects fragment endpoints with the
shortest path it can find, often degenerating to a near-straight line.

This module solves that by fitting a *parametric template* to ALL needle
mask pixels at once (so fragments contribute together), then
re-rasterising the centerline from the template. Because the template
has only 3-5 parameters (circle: cx, cy, r; ellipse: + axes + rotation),
it is dominated by the bulk of the inliers and ignores the gaps.

Pipeline::

    pixels = nonzero(needle_mask)        # union of all fragments
    fit    = RANSAC circle fit -> refine via algebraic LSQ on inliers
    if fit residual still too large:
        fit = OpenCV ellipse fit on inlier hull
    arc_extent = angular span on the fitted curve that has pixel support
                 (gap-based method handling +/- pi wrap)
    centerline = arc sampled with n_out points
    head/tail  = endpoints of the arc (orientation decided downstream
                 by needle <-> thread coupling)

Public entry points::

    fit = fit_needle_arc(needle_mask)
    if fit['ok']:
        centerline = sample_arc(fit, n=200)

Or via the convenience wrapper that mirrors the spline extractor::

    out = extract_needle_arc_centerline(needle_mask, n_out=200)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Circle from three points (algebraic)
# ----------------------------------------------------------------------
def _circle_from_3pts(p1, p2, p3) -> Optional[Tuple[float, float, float]]:
    """Return (cx, cy, r) for the circle through three 2D points.
    Returns None if the points are colinear."""
    ax, ay = p1
    bx, by = p2
    cx_, cy_ = p3
    d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy_)
          + (bx * bx + by * by) * (cy_ - ay)
          + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx)
          + (bx * bx + by * by) * (ax - cx_)
          + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    r = float(np.hypot(ax - ux, ay - uy))
    return float(ux), float(uy), r


# ----------------------------------------------------------------------
# Algebraic least-squares circle fit (Taubin / Kasa form)
# ----------------------------------------------------------------------
def _algebraic_circle_fit(pts: np.ndarray) -> Tuple[float, float, float]:
    """Kasa fit: minimize sum( (x-cx)^2 + (y-cy)^2 - r^2 )^2 in closed form.
    Bias toward larger circles, but good enough after RANSAC seeded it."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx * cx + cy * cy, 1e-9)))
    return float(cx), float(cy), r


# ----------------------------------------------------------------------
# RANSAC circle fit
# ----------------------------------------------------------------------
def fit_circle_ransac(
    pixels: np.ndarray,
    *,
    max_iter: int = 300,
    dist_thresh: float = 3.0,
    min_inliers_ratio: float = 0.35,
    r_min: float = 8.0,
    r_max: float = 1e4,
    seed: int = 0,
) -> Optional[dict]:
    """Fit a circle (cx, cy, r) to a 2D pixel cloud via RANSAC.

    Args:
        pixels   : (N, 2) array of (x, y) pixels (all needle mask pixels,
                   not just skeleton -- larger N is more robust).
        dist_thresh : an inlier lies within this many pixels of the circle.
        min_inliers_ratio : reject if the best model has fewer inliers.
        r_min/r_max : sanity bounds on radius (in pixels).

    Returns dict with keys cx, cy, r, inliers (bool [N]), inlier_ratio,
    residual_mean (pixels). None if no acceptable fit was found.
    """
    if pixels is None or len(pixels) < 3:
        return None
    pixels = np.asarray(pixels, dtype=np.float32)
    N = len(pixels)
    rng = np.random.default_rng(seed)

    best_n = 0
    best_inliers = None
    best_circle = None
    for _ in range(max_iter):
        idx = rng.choice(N, 3, replace=False)
        circ = _circle_from_3pts(pixels[idx[0]], pixels[idx[1]], pixels[idx[2]])
        if circ is None:
            continue
        cx, cy, r = circ
        if not (r_min < r < r_max):
            continue
        d = np.abs(np.hypot(pixels[:, 0] - cx, pixels[:, 1] - cy) - r)
        inliers = d < dist_thresh
        n_in = int(inliers.sum())
        if n_in > best_n:
            best_n = n_in
            best_inliers = inliers
            best_circle = (cx, cy, r)

    if best_circle is None:
        return None
    if best_n / N < min_inliers_ratio:
        return None

    # ---- refine via algebraic LSQ on inliers ----
    inlier_pts = pixels[best_inliers]
    cx, cy, r = _algebraic_circle_fit(inlier_pts)
    if not (r_min < r < r_max):
        cx, cy, r = best_circle  # fall back to RANSAC seed

    # final residual / inlier set after refinement
    d_final = np.abs(np.hypot(pixels[:, 0] - cx, pixels[:, 1] - cy) - r)
    inliers_final = d_final < dist_thresh
    return dict(
        cx=float(cx), cy=float(cy), r=float(r),
        inliers=inliers_final,
        inlier_ratio=float(inliers_final.mean()),
        residual_mean=float(d_final[inliers_final].mean()
                             if inliers_final.any() else d_final.mean()),
        model='circle',
    )


# ----------------------------------------------------------------------
# Arc-extent: figure out the angular span of an arc by gap analysis
# ----------------------------------------------------------------------
def arc_extent(pixels_xy: np.ndarray, cx: float, cy: float
               ) -> Tuple[float, float]:
    """Find (theta_start, theta_end) of an arc on the circle (cx, cy)
    that is *covered* by the projected pixels.

    Strategy: project pixels to angles in [0, 2pi). Sort them. The
    "arc" we want is everything EXCEPT the largest empty angular gap.
    This handles +/- pi wrap-around automatically and is robust to
    fragmentation (small gaps between mask pieces stay inside the arc).
    """
    theta = np.arctan2(pixels_xy[:, 1] - cy, pixels_xy[:, 0] - cx) % (2 * np.pi)
    theta_s = np.sort(theta)
    # gaps between consecutive sorted angles + wrap-around gap
    diffs = np.diff(theta_s)
    wrap = 2 * np.pi - (theta_s[-1] - theta_s[0])
    diffs = np.concatenate([diffs, [wrap]])
    g = int(np.argmax(diffs))
    if g == len(theta_s) - 1:
        # gap is the wrap-around; arc = [theta_s[0], theta_s[-1]]
        return float(theta_s[0]), float(theta_s[-1])
    # gap is between theta_s[g] and theta_s[g+1]; arc wraps THROUGH the wrap-around
    # arc start = theta_s[g+1], end = theta_s[g] + 2*pi (or equivalently
    # traverse forward modulo 2pi).
    return float(theta_s[g + 1]), float(theta_s[g] + 2 * np.pi)


def sample_arc(fit: dict, n: int = 200, pad_deg: float = 0.0) -> np.ndarray:
    """Resample the fitted arc with n equidistant arc-length points.

    Returns (n, 2) float32 array of (x, y) along the arc.
    pad_deg slightly extends the arc on both ends (sometimes useful to
    cover masked-out tip when the segmentation under-segments it).
    """
    cx, cy, r = fit['cx'], fit['cy'], fit['r']
    a, b = fit['theta_start'], fit['theta_end']
    pad = np.deg2rad(pad_deg)
    a -= pad
    b += pad
    ts = np.linspace(a, b, n)
    xs = cx + r * np.cos(ts)
    ys = cy + r * np.sin(ts)
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ----------------------------------------------------------------------
# Ellipse fallback (for perspective-distorted near-circular arcs)
# ----------------------------------------------------------------------
def fit_ellipse_fallback(pixels: np.ndarray) -> Optional[dict]:
    """Use cv2.fitEllipse on the convex hull of pixels. Useful when a
    circle fit gives high residual because the needle is viewed
    obliquely (the projected arc becomes an ellipse arc)."""
    if pixels is None or len(pixels) < 5:
        return None
    pts = pixels.astype(np.float32).reshape(-1, 1, 2)
    try:
        (ex, ey), (MA, ma), ang_deg = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    return dict(
        cx=float(ex), cy=float(ey),
        a=float(MA / 2.0), b=float(ma / 2.0),  # semi-axes
        angle_deg=float(ang_deg),
        model='ellipse',
    )


def _sample_ellipse_arc(fit: dict, theta_s: float, theta_e: float,
                        n: int = 200) -> np.ndarray:
    """Parametric ellipse: pixel(t) = center + R(ang) * (a cos t, b sin t)."""
    cx, cy = fit['cx'], fit['cy']
    a, b = fit['a'], fit['b']
    ang = np.deg2rad(fit['angle_deg'])
    cosA, sinA = np.cos(ang), np.sin(ang)
    ts = np.linspace(theta_s, theta_e, n)
    x_local = a * np.cos(ts)
    y_local = b * np.sin(ts)
    xs = cx + cosA * x_local - sinA * y_local
    ys = cy + sinA * x_local + cosA * y_local
    return np.stack([xs, ys], axis=1).astype(np.float32)


def ellipse_arc_extent(pixels: np.ndarray, fit: dict) -> Tuple[float, float]:
    """Project pixels into ellipse-local frame, compute angular extent
    (eccentric anomaly) with the same gap method."""
    cx, cy = fit['cx'], fit['cy']
    a, b = fit['a'], fit['b']
    ang = np.deg2rad(fit['angle_deg'])
    cosA, sinA = np.cos(-ang), np.sin(-ang)
    dx = pixels[:, 0] - cx
    dy = pixels[:, 1] - cy
    x_local = cosA * dx - sinA * dy
    y_local = sinA * dx + cosA * dy
    # eccentric anomaly: divide each axis by its semi-axis
    theta = np.arctan2(y_local / max(b, 1e-6),
                        x_local / max(a, 1e-6)) % (2 * np.pi)
    theta_s = np.sort(theta)
    diffs = np.diff(theta_s)
    wrap = 2 * np.pi - (theta_s[-1] - theta_s[0])
    diffs = np.concatenate([diffs, [wrap]])
    g = int(np.argmax(diffs))
    if g == len(theta_s) - 1:
        return float(theta_s[0]), float(theta_s[-1])
    return float(theta_s[g + 1]), float(theta_s[g] + 2 * np.pi)


# ----------------------------------------------------------------------
# Top-level: fit template + return centerline
# ----------------------------------------------------------------------
def fit_needle_arc(
    needle_mask: np.ndarray,
    *,
    use_skeleton_pixels: bool = False,
    min_pixels: int = 30,
    ransac_iter: int = 300,
    inlier_thresh: float = 3.0,
    min_inlier_ratio: float = 0.35,
    r_min: float = 8.0,
    r_max: Optional[float] = None,
    ellipse_residual_thresh: float = 4.0,
) -> Optional[dict]:
    """Fit a circle (or ellipse fallback) to all needle mask pixels.

    Args:
        needle_mask : HxW binary mask of the needle class (fragments OK).
        use_skeleton_pixels : if True, fit on the skeleton instead of all
                              mask pixels. Usually False is better -- the
                              full mask carries more support, especially
                              for fragmented predictions.
        min_pixels  : require this many pixels to attempt a fit.
        r_max       : sanity bound on radius (defaults to 2 * max(H, W)).

    Returns a fit dict on success:
        {model='circle', cx, cy, r,
         theta_start, theta_end, inlier_ratio, residual_mean, pixels}
        OR
        {model='ellipse', cx, cy, a, b, angle_deg,
         theta_start, theta_end, inlier_ratio, residual_mean, pixels}
    Returns None if the mask is too small or no acceptable fit was found.
    """
    if needle_mask is None or needle_mask.sum() < min_pixels:
        return None
    H, W = needle_mask.shape
    if r_max is None:
        r_max = 2.0 * max(H, W)

    if use_skeleton_pixels:
        from skimage.morphology import skeletonize
        sk = skeletonize(needle_mask > 0)
        ys, xs = np.where(sk)
    else:
        ys, xs = np.where(needle_mask > 0)
    if len(xs) < min_pixels:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # ---- try circle fit ----
    circ = fit_circle_ransac(
        pts, max_iter=ransac_iter, dist_thresh=inlier_thresh,
        min_inliers_ratio=min_inlier_ratio, r_min=r_min, r_max=r_max,
    )

    if circ is not None and circ['residual_mean'] < ellipse_residual_thresh:
        # success with circle
        inlier_pts = pts[circ['inliers']]
        if len(inlier_pts) < 3:
            return None
        a0, a1 = arc_extent(inlier_pts, circ['cx'], circ['cy'])
        out = dict(circ)
        out['theta_start'] = a0
        out['theta_end'] = a1
        out['pixels'] = pts
        return out

    # ---- ellipse fallback (perspective-distorted needle) ----
    if circ is not None:
        inlier_pts = pts[circ['inliers']]
    else:
        inlier_pts = pts
    ell = fit_ellipse_fallback(inlier_pts)
    if ell is None:
        return None
    a0, a1 = ellipse_arc_extent(inlier_pts, ell)
    ell['theta_start'] = a0
    ell['theta_end'] = a1
    # residual on inliers (approximate -- dist to ellipse is hard;
    # use radial residual in normalized frame instead)
    ell['inlier_ratio'] = (float(len(inlier_pts)) / max(1, len(pts)))
    ell['residual_mean'] = float('nan')
    ell['pixels'] = pts
    return ell


def sample_template_centerline(fit: dict, n: int = 200,
                                pad_deg: float = 0.0) -> np.ndarray:
    """Sample n equidistant points along the fitted template arc."""
    a = fit['theta_start']
    b = fit['theta_end']
    pad = np.deg2rad(pad_deg)
    a -= pad
    b += pad
    if fit['model'] == 'circle':
        cx, cy, r = fit['cx'], fit['cy'], fit['r']
        ts = np.linspace(a, b, n)
        xs = cx + r * np.cos(ts)
        ys = cy + r * np.sin(ts)
        return np.stack([xs, ys], axis=1).astype(np.float32)
    elif fit['model'] == 'ellipse':
        return _sample_ellipse_arc(fit, a, b, n=n)
    raise ValueError(f"unknown model {fit['model']!r}")


# ----------------------------------------------------------------------
# Drop-in replacement that mirrors the spline extractor's contract
# ----------------------------------------------------------------------
def extract_needle_arc_centerline(
    needle_mask: np.ndarray,
    n_out: int = 200,
    **fit_kwargs,
) -> Optional[dict]:
    """Run the arc template fit and return both the fit metadata and
    the sampled centerline. Returns None if no fit was found.
    """
    fit = fit_needle_arc(needle_mask, **fit_kwargs)
    if fit is None:
        return None
    curve = sample_template_centerline(fit, n=n_out)
    return dict(fit=fit, centerline=curve)
