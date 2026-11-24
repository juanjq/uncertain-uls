import sys; sys.path.insert(0, "tests")
import numpy as np, healpy as hp
from astropy.coordinates import SkyCoord
import test_fields as T
from uuls import fields
sky = T.blob_healpix()
d = fields.sample_healpix(sky, 20000, np.random.default_rng(0))
c = SkyCoord(d["ra"], d["dec"], unit="deg")
sep = c.separation(T.BLOB).deg
print("frac<0.5", (sep < 0.5).mean(), 1 - np.exp(-0.5), "mean ra dec", d["ra"].mean(), d["dec"].mean())
pc = hp.pix2ang(256, d["pix"], nest=True, lonlat=True)
print("pix centre vs draw sep max", SkyCoord(*pc, unit="deg").separation(c).deg.max())
import toy
g = toy.toy_geom(center=T.CENTRES[0]).to_image()
print(g.width, g.center_skydir)
print("frac in field0", g.contains(c).mean())
ra, dec = hp.pix2ang(256, np.arange(sky.prob.size), nest=True, lonlat=True)
pcs = SkyCoord(ra, dec, unit="deg")
print("prob in field0", sky.prob[g.contains(pcs)].sum())
sp = pcs.separation(T.BLOB).deg
print("pix prob sep<0.5", sky.prob[sp < 0.5].sum())
lev, ip = __import__("astropy_healpix").uniq_to_level_ipix(sky.uniq[:5]); print(lev, ip, sky.nside[:3])
inside_pix = g.contains(pcs)
print("pix ra range inside", ra[inside_pix & (sp<3)].min(), ra[inside_pix & (sp<3)].max())
print("draw ra range inside", d["ra"][g.contains(c)].min(), d["ra"][g.contains(c)].max())
