#!/usr/bin/env python3
"""Analytical / vectorized placers on synthetic netlists.
Methods:
  random      - uniform random baseline
  quadratic   - from-scratch Laplacian (clique/star) analytical placer, sparse solve
  fd_allpairs - batched all-pairs repulsion + net-centroid attraction  (O(n^2))
  fd_grid     - net-centroid attraction + bin-density gradient spreading (O(n + bins))
All methods followed by the same row legalizer; HPWL reported pre- and post-legalization.
"""
import time, json, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.ndimage import gaussian_filter


# ------------------------------------------------------------ netlist
def gen_netlist(n, seed, util=0.6):
    """Cells have a hidden 2D 'true' location; nets connect spatial neighbours (Rent-like locality),
    so a good placement exists. Returns dict with nets (list of arrays), pads, region size."""
    rng = np.random.default_rng(seed)
    side = np.sqrt(n / util)  # region side in cell-widths
    true = rng.random((n, 2)) * side
    nets = []
    from scipy.spatial import cKDTree
    tree = cKDTree(true)
    for i in range(n):
        k = rng.choice([2, 2, 2, 3, 3, 4, 5])
        _, idx = tree.query(true[i], k=min(k + 3, n))
        pins = np.unique(np.concatenate([[i], rng.choice(idx[1:], size=min(k - 1, len(idx) - 1), replace=False)]))
        nets.append(pins)
    # pads: fixed terminals on the boundary tied to nearest cells
    npad = max(8, int(2.5 * np.sqrt(n)))
    pad_xy = []
    for t in range(npad):
        e = t % 4; u = rng.random() * side
        pad_xy.append([(0, u), (side, u), (u, 0), (u, side)][e])
    pad_xy = np.array(pad_xy, float)
    for t, p in enumerate(pad_xy):
        d = np.linalg.norm(true - p, axis=1)
        near = np.argsort(d)[:2]
        nets.append(np.array([n + t, *near]))  # pad index offset n
    return {"n": n, "nets": nets, "pads": pad_xy, "side": side, "true": true}


def hpwl(P, nl):
    allp = np.vstack([P, nl["pads"]])
    tot = 0.0
    for pins in nl["nets"]:
        xy = allp[pins]
        tot += np.ptp(xy[:, 0]) + np.ptp(xy[:, 1])
    return tot


class Incidence:
    """Sparse net-cell incidence for vectorized centroid math (pads folded in as constants)."""
    def __init__(self, nl):
        n = nl["n"]; rows, cols = [], []
        self.padc = np.zeros((len(nl["nets"]), 2)); self.padn = np.zeros(len(nl["nets"]))
        for j, pins in enumerate(nl["nets"]):
            for p in pins:
                if p < n: rows.append(j); cols.append(p)
                else: self.padc[j] += nl["pads"][p - n]; self.padn[j] += 1
        self.A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nl["nets"]), n))
        self.deg = np.asarray(self.A.sum(1)).ravel() + self.padn
        self.At = self.A.T.tocsr()
        self.cell_nets = np.asarray(self.At.sum(1)).ravel()

    def centroids(self, P):
        return (self.A @ P + self.padc) / self.deg[:, None]

    def pull(self, P):
        """per-cell mean of incident net centroids"""
        C = self.centroids(P)
        return (self.At @ C) / np.maximum(self.cell_nets, 1)[:, None]


# ------------------------------------------------------------ placers
def place_random(nl, seed=0):
    return np.random.default_rng(seed).random((nl["n"], 2)) * nl["side"]


def place_quadratic(nl):
    n = nl["n"]; side = nl["side"]
    rows, cols, vals = [], [], []
    bx = np.zeros(n); by = np.zeros(n)
    extra = 0; entries = []
    # star model for all nets: auxiliary node per net (keeps matrix sparse & exact for quadratic WL)
    m = len(nl["nets"]); N = n + m
    Lr, Lc, Lv = [], [], []
    b = np.zeros((N, 2)); diag = np.zeros(N)
    for j, pins in enumerate(nl["nets"]):
        w = 1.0 / max(len(pins) - 1, 1) * 2
        s = n + j
        for p in pins:
            if p < n:
                Lr += [s, p]; Lc += [p, s]; Lv += [-w, -w]; diag[s] += w; diag[p] += w
            else:
                b[s] += w * nl["pads"][p - n]; diag[s] += w
    eps = 1e-3  # weak anchor to region centre: keeps system non-singular for pad-disconnected components
    diag[:n] += eps; b[:n] += eps * side / 2
    L = sp.csr_matrix((Lv, (Lr, Lc)), shape=(N, N)) + sp.diags(diag)
    sol = np.column_stack([spl.spsolve(L.tocsc(), b[:, 0]), spl.spsolve(L.tocsc(), b[:, 1])])
    P = sol[:n]
    # quadratic placement clumps -> spread by rank-preserving uniform remap per axis
    for ax in (0, 1):
        order = np.argsort(P[:, ax]); r = np.empty(n); r[order] = np.arange(n)
        P[:, ax] = (r + 0.5) / n * side * 0.0 + P[:, ax]  # keep raw; spreading done in blend below
    # blend with rank-uniform spreading (partial) so legalizer has a workable density
    Q = P.copy()
    for ax in (0, 1):
        order = np.argsort(P[:, ax]); r = np.empty(n); r[order] = np.arange(n)
        Q[:, ax] = 0.5 * P[:, ax] + 0.5 * ((r + 0.5) / n * side)
    return Q


def _clip(P, side):
    np.clip(P, 0, side, out=P); return P


def place_fd_allpairs(nl, iters=150, seed=0):
    n = nl["n"]; side = nl["side"]; inc = Incidence(nl)
    P = place_random(nl, seed) * 0.4 + side * 0.3
    for it in range(iters):
        t = 1 - it / iters
        pull = inc.pull(P) - P
        diff = P[:, None, :] - P[None, :, :]
        d2 = (diff ** 2).sum(-1) + 0.25
        rep = (diff / d2[..., None] ** 1.0).sum(1)  # ~1/r repulsion, batched all pairs
        P = _clip(P + 0.6 * pull + (0.9 * t + 0.15) * rep * (side / np.sqrt(n)) * 0.6, side)
    return P


def place_fd_grid(nl, iters=150, seed=0, bins=None):
    n = nl["n"]; side = nl["side"]; inc = Incidence(nl)
    B = bins or max(8, int(np.sqrt(n / 2)))
    P = place_random(nl, seed) * 0.4 + side * 0.3
    cell = side / B; target = n / (B * B)
    for it in range(iters):
        t = 1 - it / iters
        pull = inc.pull(P) - P
        ij = np.clip((P / cell).astype(int), 0, B - 1)
        dens = np.zeros((B, B)); np.add.at(dens, (ij[:, 0], ij[:, 1]), 1.0)
        dens = gaussian_filter(dens, 1.0, mode="nearest")
        over = np.maximum(dens - target, 0)
        gx, gy = np.gradient(gaussian_filter(over, 0.8, mode="nearest"))
        push = -np.column_stack([gx[ij[:, 0], ij[:, 1]], gy[ij[:, 0], ij[:, 1]]])
        P = _clip(P + 0.6 * pull + (1.2 * t + 0.4) * push * cell / (target + 1e-9) * 1.0, side)
    return P


# ------------------------------------------------------------ legalizer
def legalize(P, nl):
    """Row-based Tetris legalizer: unit-width cells, rows of height 1, no overlap."""
    n = nl["n"]; side = nl["side"]
    nrows = int(np.floor(side)); width = side
    nxt = np.zeros(nrows)
    order = np.argsort(P[:, 0])
    out = np.zeros_like(P)
    for i in order:
        x, y = P[i]
        r0 = int(np.clip(round(y - 0.5), 0, nrows - 1))
        best = None
        for dr in range(0, nrows):
            for r in {r0 - dr, r0 + dr}:
                if 0 <= r < nrows:
                    px = max(nxt[r], min(x, width - 1.0))
                    if px + 1.0 <= width + 1e-9:
                        cost = abs(px - x) + abs(r + 0.5 - y)
                        if best is None or cost < best[0]: best = (cost, r, px)
            if best is not None and dr > best[0]: break
        if best is None:  # overflow fallback: least-filled row
            r = int(np.argmin(nxt)); best = (0, r, nxt[r])
        _, r, px = best
        out[i] = (px + 0.5, r + 0.5); nxt[r] = px + 1.0
    return out


def overlap_count(P):
    # cells legalized to integer-grid rows: count pairs overlapping in same row
    rows = np.floor(P[:, 1]).astype(int); bad = 0
    for r in np.unique(rows):
        xs = np.sort(P[rows == r, 0]); bad += int(np.sum(np.diff(xs) < 1 - 1e-6))
    return bad


METHODS = {"random": place_random, "quadratic": place_quadratic,
           "fd_allpairs": place_fd_allpairs, "fd_grid": place_fd_grid}


def run_bench(sizes=(50, 100, 200, 400, 800, 1600), seeds=(0, 1, 2), allpairs_max=1600):
    rows = []
    for n in sizes:
        for s in seeds:
            nl = gen_netlist(n, 100 + s)
            ref = hpwl(nl["true"], nl)
            for name, fn in METHODS.items():
                if name == "fd_allpairs" and n > allpairs_max: continue
                t0 = time.perf_counter(); P = fn(nl); dt = time.perf_counter() - t0
                H = hpwl(P, nl); L = legalize(P, nl); HL = hpwl(L, nl)
                rows.append({"n": n, "seed": s, "method": name, "hpwl_global": H, "hpwl_legal": HL,
                             "overlaps": overlap_count(L), "time_s": dt, "hpwl_hidden_ref": ref})
            print(f"n={n} seed={s} done", flush=True)
    return rows


if __name__ == "__main__":
    rows = run_bench()
    json.dump(rows, open("bench.json", "w"), indent=1)
