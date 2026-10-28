#!/usr/bin/env python3
"""PyTorch port of the two force-directed placers. Runs on CUDA when available, else CPU (same code path).
Validated against the NumPy reference in placer.py (see bench_torch.py)."""
import time, json, sys, numpy as np, torch
import torch.nn.functional as F
from placer import gen_netlist, hpwl, legalize, overlap_count, place_fd_grid, place_fd_allpairs, place_random

def dev(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inc:
    def __init__(s, nl, d):
        n = nl["n"]; ci, ni = [], []; padc = np.zeros((len(nl["nets"]), 2)); padn = np.zeros(len(nl["nets"]))
        for j, pins in enumerate(nl["nets"]):
            for p in pins:
                if p < n: ci.append(p); ni.append(j)
                else: padc[j] += nl["pads"][p - n]; padn[j] += 1
        s.ci = torch.tensor(ci, device=d); s.ni = torch.tensor(ni, device=d); s.m = len(nl["nets"]); s.n = n
        s.padc = torch.tensor(padc, dtype=torch.float32, device=d)
        s.deg = torch.zeros(s.m, device=d).index_add_(0, s.ni, torch.ones(len(ci), device=d)) + torch.tensor(padn, dtype=torch.float32, device=d)
        s.cn = torch.zeros(n, device=d).index_add_(0, s.ci, torch.ones(len(ci), device=d)).clamp(min=1)
    def pull(s, P):
        C = torch.zeros(s.m, 2, device=P.device).index_add_(0, s.ni, P[s.ci]); C = (C + s.padc) / s.deg[:, None]
        out = torch.zeros(s.n, 2, device=P.device).index_add_(0, s.ci, C[s.ni]); return out / s.cn[:, None]

def gauss1d(sigma, d):
    r = int(4 * sigma + 0.5); x = torch.arange(-r, r + 1, dtype=torch.float32, device=d); k = torch.exp(-0.5 * (x / sigma) ** 2); return k / k.sum()
def gfilter(img, sigma):
    k = gauss1d(sigma, img.device); r = len(k) // 2; x = img[None, None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1)); return x[0, 0]

def fd_grid_t(nl, iters=150, seed=0, d=None):
    d = d or dev(); n = nl["n"]; side = nl["side"]; inc = Inc(nl, d); B = max(8, int(np.sqrt(n / 2))); cell = side / B; target = n / (B * B)
    P = torch.tensor(place_random(nl, seed) * 0.4 + side * 0.3, dtype=torch.float32, device=d)
    for it in range(iters):
        t = 1 - it / iters; pull = inc.pull(P) - P
        ij = (P / cell).long().clamp(0, B - 1); dens = torch.zeros(B * B, device=d).index_add_(0, ij[:, 0] * B + ij[:, 1], torch.ones(n, device=d)).view(B, B)
        over = (gfilter(dens, 1.0) - target).clamp(min=0); gx, gy = torch.gradient(gfilter(over, 0.8))
        push = -torch.stack([gx[ij[:, 0], ij[:, 1]], gy[ij[:, 0], ij[:, 1]]], 1)
        P = (P + 0.6 * pull + (1.2 * t + 0.4) * push * cell / (target + 1e-9)).clamp(0, side)
    return P.cpu().numpy().astype(float)

def fd_allpairs_t(nl, iters=150, seed=0, d=None):
    d = d or dev(); n = nl["n"]; side = nl["side"]; inc = Inc(nl, d)
    P = torch.tensor(place_random(nl, seed) * 0.4 + side * 0.3, dtype=torch.float32, device=d)
    for it in range(iters):
        t = 1 - it / iters; pull = inc.pull(P) - P; diff = P[:, None, :] - P[None, :, :]
        d2 = (diff ** 2).sum(-1) + 0.25; rep = (diff / d2[..., None]).sum(1)
        P = (P + 0.6 * pull + (0.9 * t + 0.15) * rep * (side / np.sqrt(n)) * 0.6).clamp(0, side)
    return P.cpu().numpy().astype(float)
