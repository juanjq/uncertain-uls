import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from uuls import fields, paths, pointings, simdata, realisations as R, gwmap as gwm
inputs = paths.RESULTS / "0_tools" / "03_simulated_data"
point = pointings.read(inputs / "pointings.ecsv")
sim = json.loads((inputs / "simulation.json").read_text())
dl3 = simdata.simulate(point, nsb=sim["nsb"], seed=sim["seed"])
sky = paths.GW_SKYMAPS / "S240615dg.fits"
t = time.time()
one = fields.build_fields(point, dl3, {"background": "fov"}, skymap=sky, link_deg=1.0, verbose=True)
print("one field", time.time() - t, one[0].geom.data_shape, one[0].gwmap.mask.sum()); t = time.time()
four = fields.build_fields(point, dl3, {"background": "fov"}, skymap=sky, link_deg=0.3, verbose=True)
print("four fields", time.time() - t); t = time.time()
inj = fields.PWLSkyInjector2D(gwm.read_healpix(sky))
print("injector", time.time() - t); t = time.time()
a = R.simulate_fields(one, inj, np.zeros(50), seed=1)
print("50 realisations one field", time.time() - t); t = time.time()
b = R.simulate_fields(four, inj, np.zeros(50), seed=1)
print("50 realisations four fields", time.time() - t)
print(np.percentile(a["lambda"], [50, 90]), np.percentile(b["lambda"], [50, 90]))
