import time, warnings
warnings.filterwarnings("ignore")
from uuls import cases
t = time.time()
c = cases.load_case("sim_tiling")
tab = c.pointings()
print(len(tab), "tiles", tab["pgw"].sum(), time.time() - t)
print(tab)
