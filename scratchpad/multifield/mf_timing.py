import time, warnings
warnings.filterwarnings("ignore")
from uuls import cases, fields
t = time.time()
c = cases.load_case("sim_tiling")
groups = fields.split_fields(c.pointings(), 1.0)
print(len(groups), [len(g) for g in groups])
fs = fields.build_fields(c, None)
print("built", time.time() - t)
for f in fs[:3]:
    print(f.name, f.geom.data_shape, f.gwmap.prob.sum(), f.gwmap.mask.sum())
print(sum(f.gwmap.cdf.nbytes for f in fs) / 1e6, "MB")
