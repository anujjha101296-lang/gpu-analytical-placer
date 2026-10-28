# Vectorized analytical placers
`placer.py`: synthetic locality-aware netlists (50-1600 cells, boundary pads), four placers, one shared row legalizer, HPWL.
Methods: random; quadratic (clique/star Laplacian, sparse solve, partial rank-spreading); fd_allpairs (batched O(n^2) repulsion + net-centroid attraction); fd_grid (net-centroid attraction + bin-density gradient, O(n + bins)).

Mean over 3 seeds, legalized HPWL improvement vs random / runtime:
| cells | quadratic | fd_allpairs | fd_grid | fd_grid time | fd_allpairs time |
|---|---|---|---|---|---|
| 200 | 77.5% | 52.8% | 78.0% | 0.03 s | 0.32 s |
| 800 | 87.4% | 52.4% | 87.3% | 0.04 s | 5.29 s |
| 1600 | 90.8% | 52.9% | 91.6% | 0.07 s | 24.6 s |
All methods: 0 overlaps after legalization.

Limits: CPU/NumPy only. No CUDA/PyTorch port (torch could not be installed in the build sandbox). The quadratic placer is a from-scratch baseline, not a tuned industrial one. Synthetic netlists, not ISPD benchmarks.

## PyTorch / CUDA port (UNTESTED - needs a machine with working PyTorch)
`placer_torch.py` ports fd_grid and fd_allpairs to PyTorch (CUDA if available, else CPU). `bench_torch.py` compares it with the NumPy reference (legalized HPWL, overlaps, runtime) and writes bench_torch.json. Syntax-checked only; no results yet, and the GPU path has never been executed.
