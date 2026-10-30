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

Limits: the benchmark data is from synthetic netlists, not ISPD benchmarks. The quadratic placer is a from-scratch baseline, not a tuned industrial placer. CUDA hardware execution remains unverified in CI.

## PyTorch / CUDA port
`placer_torch.py` ports fd_grid and fd_allpairs to PyTorch and selects CUDA when available, otherwise CPU. The CI CPU parity test compares both methods with NumPy on 32- and 64-cell cases across two seeds: legalized HPWL matched to within 0.01%, and both implementations produced zero overlaps. `bench_torch.py` runs larger CPU/GPU comparisons and writes `bench_torch.json`. CUDA performance and numerical behavior still need validation on a machine with an NVIDIA GPU. Install PyTorch separately; it is intentionally optional in `requirements.txt`.
