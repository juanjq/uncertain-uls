import sys, importlib.util
sys.path.insert(0, "tests"); import toy
import numpy as np
spec = importlib.util.spec_from_file_location("uuls.realisations_before", "scratchpad/multifield/realisations_before.py")
old = importlib.util.module_from_spec(spec); sys.modules[spec.name] = old; spec.loader.exec_module(old)
from uuls import realisations as R
red = toy.toy_reduced(); gw = toy.toy_gwmap(red.geom)
s = np.r_[np.zeros(3), np.full(3, 5e-12)]
for new_inj, old_inj in [(R.PWLInjector2D(gw, jitter=False), old.PWLInjector2D(gw)),
                         (R.PWLInjector3D(gw, jitter=False), old.PWLInjector3D(gw))]:
    a = old.simulate(red, gw, old_inj, s * (1 if isinstance(old_inj, old.PWLInjector2D) else 1e56), seed=7)
    b = R.simulate(red, gw, new_inj, s * (1 if isinstance(old_inj, old.PWLInjector2D) else 1e56), seed=7)
    print(all(np.array_equal(a[k], b[k]) for k in a), sorted(set(b) - set(a)))
