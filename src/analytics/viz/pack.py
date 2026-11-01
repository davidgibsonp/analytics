"""Layout geometry: greedy circle packing, spreading angles or positions apart, and the segment and
box tests label placers use (boxes are (x0, y0, x1, y1) in pixels)."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def pack(radii, pad: float) -> np.ndarray:
    """Centres (n x 2) for circles of these radii, `pad` apart, centred on their bounding box.
    Greedy, largest first: each circle goes to the free tangent spot nearest the origin."""
    order = np.argsort(-np.asarray(radii), kind="stable")
    r = np.asarray(radii, float)
    c = np.zeros((len(r), 2))
    done = []
    for k in order:
        if done:
            cand = [
                c[i] + (r[i] + r[k] + pad) * np.array([np.cos(a), np.sin(a)])
                for i in done
                for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)
            ]
            for i, j in combinations(done, 2):
                d, di, dj = np.linalg.norm(c[j] - c[i]), r[i] + r[k] + pad, r[j] + r[k] + pad
                if 0 < d <= di + dj and d >= abs(di - dj):
                    x = (d * d + di * di - dj * dj) / (2 * d)
                    h = np.sqrt(max(di * di - x * x, 0))
                    u = (c[j] - c[i]) / d
                    cand += [c[i] + x * u + sgn * h * np.array([-u[1], u[0]]) for sgn in (1, -1)]
            cand = np.array(cand)
            ok = np.all(
                np.linalg.norm(cand[:, None] - c[done][None], axis=2) >= r[done] + r[k] + pad - 1e-6, axis=1
            )
            c[k] = cand[ok][np.argmin(np.linalg.norm(cand[ok], axis=1))]
        done.append(k)
    return c - ((c - r[:, None]).min(0) + (c + r[:, None]).max(0)) / 2


def spread(phi: np.ndarray, half: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Push angles (radians, in place) apart until cyclic neighbours are at least half_a + half_b apart."""
    for _ in range(max_iter):
        moved = False
        o = np.argsort(phi)
        for a, b in zip(o, np.roll(o, -1), strict=True):
            gap = (phi[b] - phi[a]) % (2 * np.pi)
            if len(o) > 1 and gap < half[a] + half[b] - 1e-9:
                d = (half[a] + half[b] - gap) / 2
                phi[a], phi[b], moved = phi[a] - d, phi[b] + d, True
        if not moved:
            return phi
    return phi


def dodge(x, half, gap: float = 0.0) -> np.ndarray:
    """1D spread: positions near x whose neighbours are at least half_i + half_j + gap apart. Items
    that would touch form a block, centred on the mean of their wanted positions."""
    x, half = np.asarray(x, float), np.asarray(half, float)

    def offsets(idx):
        return np.r_[0, np.cumsum([half[a] + half[b] + gap for a, b in zip(idx, idx[1:], strict=False)])]

    blocks = []  # [indices, start]
    for i in np.argsort(x, kind="stable"):
        blocks.append([[i], x[i]])
        while True:
            idx = blocks[-1][0]
            blocks[-1][1] = np.mean(x[idx] - offsets(idx))
            if len(blocks) < 2:
                break
            pidx, pstart = blocks[-2]
            if pstart + offsets(pidx)[-1] + half[pidx[-1]] + half[idx[0]] + gap <= blocks[-1][1]:
                break
            blocks[-2][0] = pidx + idx
            blocks.pop()
    pos = np.empty(len(x))
    for idx, start in blocks:
        pos[idx] = start + offsets(idx)
    return pos


def crosses(a, b, c, d) -> bool:
    """Do segments ab and cd cross properly (touching or sharing an end point does not count)?"""

    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def cuts(box, a, b) -> bool:
    """Does segment ab cross an edge of box (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(crosses(a, b, p, q) for p, q in zip(corners, corners[1:] + corners[:1], strict=True))


def segment_in_box(box, a, b) -> bool:
    """Does segment ab touch box (x0, y0, x1, y1): cross an edge or have an end point inside?"""
    x0, y0, x1, y1 = box
    inside = any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (a, b))
    return inside or cuts(box, a, b)


def overlap(a, b) -> float:
    """Overlap area of two boxes (x0, y0, x1, y1); 0 when they are apart."""
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)


def nearest_on_box(p, box) -> np.ndarray:
    """The point of box (x0, y0, x1, y1) nearest to p (p itself when inside)."""
    return np.array([np.clip(p[0], box[0], box[2]), np.clip(p[1], box[1], box[3])])
