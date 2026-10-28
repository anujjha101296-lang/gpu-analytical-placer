import json, time, numpy as np, torch
from placer_torch import *
print("device:", dev())
rows = []
for n in (200, 800, 1600):
    for s in (0, 1, 2):
        nl = gen_netlist(n, 100 + s); r = {"n": n, "seed": s}
        for name, fn in (("np_grid", place_fd_grid), ("torch_grid", fd_grid_t), ("np_allpairs", place_fd_allpairs), ("torch_allpairs", fd_allpairs_t)):
            if name.endswith("allpairs") and n > 800: continue
            t = time.perf_counter(); P = fn(nl); dt = time.perf_counter() - t
            L = legalize(P, nl); r[name] = {"hpwl_legal": hpwl(L, nl), "overlaps": overlap_count(L), "t": dt}
        rows.append(r); print(n, s, {k: (round(v["hpwl_legal"]), round(v["t"], 2)) for k, v in r.items() if isinstance(v, dict)}, flush=True)
json.dump({"device": str(dev()), "rows": rows}, open("bench_torch.json", "w"), indent=1)
