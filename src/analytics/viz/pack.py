"""Layout geometry: greedy circle packing and spreading angles apart."""

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
