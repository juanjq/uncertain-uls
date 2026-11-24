import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from uuls import fields, paths, pointings, simdata, lambda_stat
inputs = paths.RESULTS / "0_tools" / "03_simulated_data"
point = pointings.read(inputs / "pointings.ecsv")
sim = json.loads((inputs / "simulation.json").read_text())
dl3 = simdata.simulate(point, nsb=sim["nsb"], seed=sim["seed"])
four = fields.build_fields(point, dl3, {"background": "fov"}, skymap=paths.GW_SKYMAPS / "S240615dg.fits", link_deg=0.3)
ds = four[0].reduced.stacked
e = ds.exposure.data
img = e.sum(axis=0)
sep = ds.counts.geom.to_image().get_coord().skycoord.separation(four[0].center).deg
safe = ds.mask_safe.data.any(axis=0)
for r in (0.5, 1.0, 1.2, 1.4, 2.0, 2.4):
    ring = (np.abs(sep - r) < 0.05)
    print(r, "median summed expo", np.median(img[ring]), "max per-bin over e", np.median(e.max(axis=0)[ring]), "safe frac", safe[ring].mean())
print("peak", img.max(), "exposure>0 frac in safe", (img[safe] > 0).mean())
print("energy_true centers", ds.exposure.geom.axes["energy_true"].center)
import astropy.units as u
iy = img.shape[0]//2
print(np.array2string(e[:, iy, ::4] / e.max(), precision=3, max_line_width=250))
