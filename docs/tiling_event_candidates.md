# Tiling event candidates (`sim_tiling`)

Produced by `scripts/select_tiling_event.py` (2026-11-23). Rerun it to regenerate the table;
per-release skymap statistics are cached under `cache/gwtc_skymap_stats/`, the skymap
tarballs under `cache/gwtc_skymaps/`.

## Inputs and rules

- Events: confident events in the GWOSC event API, catalogs `GWTC-2.1-confident`,
  `GWTC-3-confident`, `GWTC-4.1` (O4a, supersedes 4.0) and `GWTC-5.0` (O4b part 1).
  Events are matched to skymaps by GPS time, because GWOSC drops the time suffix for some
  names (GW150914, GW190521, ...).
- Skymaps: the PE skymap tarballs of the Zenodo PE releases (GWTC-2.1: 6513631, GWTC-3:
  8177023, GWTC-4.1: 20275769, GWTC-5.0: 20348005). One map per event: `Mixed` when present,
  otherwise `IMRPhenomXPHM_SpinTaylor` (GWTC-5.0 has no mixed map). 281 events have a skymap.
- LST field: a disc of radius 2.5 deg, tilepy's LST default `FOV`, so 19.6 deg2. The number
  of fields is A90 / 19.6 deg2, which ignores overlap and is a lower bound on what a tiling
  needs. Candidates have 10 to 50 fields (A90 from 196 to 982 deg2): 80 events. The ranking
  keeps this nominal field. The `sim_tiling` case later uses 2.0 deg (12.6 deg2) to match the
  simulst IRFs, which gives about 38 fields for the chosen event, still inside the range.
- Visibility: P(dec > -25 deg) inside the 90% region, and the largest fraction of the 90%
  region's probability at zenith angle < 60 deg from LST-1 at one instant (best sidereal time).
- Score: area_fit x visible fraction, with area_fit = exp(-0.5 (ln(n / 22.4) / ln(50 / 22.4))^2),
  so 1 at the geometric middle of the 10 to 50 range and 0.61 at its edges.
- theta_jn: every PE release used here (GWTC-2.1/3 `mixed_cosmo.h5`, GWTC-4.1/5.0
  `combined_PEDataRelease.hdf5`) carries `theta_jn` in its posterior samples, so this
  criterion does not separate the candidates. This holds for the release formats; it has not
  been checked file by file, since no PE file was downloaded.
- NS content is not required (decision 4 in `docs/plan.md`).

## Ranked table (top 15)

| rank | event | superevent | release | skymap analysis | A90 (deg2) | LST fields | P(dec > -25 deg) in 90% | best P(z < 60 deg) in 90% | peak (ra, dec) | distance (Mpc) | score | PE Zenodo record, file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GW240923_204006 | S240923ct | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 438 | 22.3 | 1.00 | 1.00 | (239.5, +22.0) | 4011 ± 1131 | 1.00 | 20348005 `IGWN-GWTC5p0-29ebe06b7_25-GW240923_204006-combined_PEDataRelease.hdf5` |
| 2 | GW190915_235702 | S190915ak | GWTC-2.1 | Mixed | 449 | 22.9 | 1.00 | 1.00 | (195.0, +37.4) | 1758 ± 418 | 1.00 | 6513631 `IGWN-GWTC2p1-v2-GW190915_235702_PEDataRelease_mixed_cosmo.h5` |
| 3 | GW241230_233618 | S241230ev | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 450 | 22.9 | 1.00 | 1.00 | (48.5, +10.0) | 2906 ± 1067 | 1.00 | 20348006 `IGWN-GWTC5p0-29ebe06b7_25-GW241230_233618-combined_PEDataRelease.hdf5` |
| 4 | GW241130_034908 | S241130n | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 453 | 23.1 | 1.00 | 1.00 | (231.0, +83.2) | 1971 ± 577 | 1.00 | 20348005 `IGWN-GWTC5p0-29ebe06b7_25-GW241130_034908-combined_PEDataRelease.hdf5` |
| 5 | GW190521_074359 | S190521r | GWTC-2.1 | Mixed | 467 | 23.8 | 0.99 | 1.00 | (281.1, +19.2) | 1099 ± 346 | 0.99 | 6513631 `IGWN-GWTC2p1-v2-GW190521_074359_PEDataRelease_mixed_cosmo.h5` |
| 6 | GW241111_111552 | S241111bn | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 486 | 24.7 | 1.00 | 1.00 | (63.7, +40.2) | 1520 ± 418 | 0.99 | 20348005 `IGWN-GWTC5p0-29ebe06b7_25-GW241111_111552-combined_PEDataRelease.hdf5` |
| 7 | **GW191216_213338** | **S191216ap** | GWTC-3 | Mixed | 480 | 24.5 | 1.00 | 1.00 | (317.3, +32.5) | 339 ± 78 | 0.99 | 8177023 `IGWN-GWTC3p0-v2-GW191216_213338_PEDataRelease_mixed_cosmo.h5` |
| 8 | GW170608_020116 | (G288732, O2) | GWTC-2.1 | Mixed | 379 | 19.3 | 1.00 | 1.00 | (123.0, +28.9) | 340 ± 76 | 0.98 | 6513631 `IGWN-GWTC2p1-v2-GW170608_020116_PEDataRelease_mixed_cosmo.h5` |
| 9 | GW190924_021846 | S190924h | GWTC-2.1 | Mixed | 377 | 19.2 | 1.00 | 1.00 | (133.9, +5.2) | 546 ± 136 | 0.98 | 6513631 `IGWN-GWTC2p1-v2-GW190924_021846_PEDataRelease_mixed_cosmo.h5` |
| 10 | GW240825_055146 | S240825ar | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 512 | 26.1 | 1.00 | 1.00 | (256.1, +64.5) | 1248 ± 327 | 0.98 | 20348005 `IGWN-GWTC5p0-29ebe06b7_25-GW240825_055146-combined_PEDataRelease.hdf5` |
| 11 | GW250109_010541 | S250109f | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 517 | 26.4 | 1.00 | 1.00 | (52.3, +32.9) | 3390 ± 1017 | 0.98 | 20348006 `IGWN-GWTC5p0-29ebe06b7_25-GW250109_010541-combined_PEDataRelease.hdf5` |
| 12 | GW250101_011205 | S250101k | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 532 | 27.1 | 1.00 | 1.00 | (63.1, +23.2) | 3225 ± 1002 | 0.97 | 20348006 `IGWN-GWTC5p0-29ebe06b7_25-GW250101_011205-combined_PEDataRelease.hdf5` |
| 13 | GW231206_233901 | S231206cc | GWTC-4.1 | Mixed | 355 | 18.1 | 1.00 | 1.00 | (1.0, +8.7) | 1463 ± 255 | 0.97 | 20275769 `IGWN-GWTC4p1-18965dda8_5-GW231206_233901-combined_PEDataRelease.hdf5` |
| 14 | GW200225_060421 | S200225q | GWTC-3 | Mixed | 509 | 25.9 | 1.00 | 0.94 | (110.8, +54.1) | 1144 ± 317 | 0.93 | 8177023 `IGWN-GWTC3p0-v2-GW200225_060421_PEDataRelease_mixed_cosmo.h5` |
| 15 | GW240920_073424 | S240920bz | GWTC-5.0 | IMRPhenomXPHM_SpinTaylor | 419 | 21.4 | 0.94 | 0.92 | (15.1, +58.1) | 1457 ± 357 | 0.91 | 20348005 `IGWN-GWTC5p0-29ebe06b7_25-GW240920_073424-combined_PEDataRelease.hdf5` |

Distances are the skymap header's DISTMEAN ± DISTSTD. Skymaps copied to `data/gw/skymaps/`
(about 0.7 MB each): `S191216ap.fits`, `GW170608_020116.fits` and `S190924h.fits`. No PE file
was downloaded.

## Recommendation

Use **S191216ap (GW191216_213338)** for `sim_tiling`. The top of the table is a near tie
(score 0.97 to 1.00), so area and visibility cannot pick one event. Among the tied events it
is the nearest one with an O3 superevent: 339 ± 78 Mpc, against 1 to 4 Gpc for the
higher-ranked events. Distance is what the model-based studies (2 to 4) depend on, because
a gamma-ray counterpart at a few hundred Mpc gives limits that actually cut into the
(E_iso, θ_core) plane. Its 90% region (480 deg2, about 24 nominal fields) is an arc of
about 85 deg running from dec +60 to dec -25. All of it lies north of dec -25, and from La Palma all of it can be at
zenith < 60 deg at the same moment. So the tiling needs many fields and their order matters, and the
whole region can still be reached in one night. The case itself uses a smaller field than the
ranking: `configs/cases/sim_tiling.yaml` sets a 2.0 deg radius and zenith < 55 deg, which follow the
extent of the simulst IRFs. With 20 min runs and up to 20 runs, tilepy then schedules 19 tiles
holding 0.746 of the probability. With the 2.5 deg tilepy default and zenith < 60 deg it was 16
tiles holding 0.80. The event also had real follow-up (an IceCube neutrino candidate and a HAWC
follow-up search, both reported in GCN circulars), which gives the mock case an external reference. Its PE is in the GWTC-3 release
(Zenodo 8177023, `IGWN-GWTC3p0-v2-GW191216_213338_PEDataRelease_mixed_cosmo.h5`, 500 MB,
label `C01:Mixed`). That is a different file layout from the GWTC-5.0 `combined` files used
for S240615dg, so `uuls.gwpe.standardize` must handle the GWTC-3 layout. GW170608_020116
(340 Mpc, 19 fields, O2, no S-id) is the alternative if a GWTC-5.0-style release is not
needed. S190924h (546 Mpc, 19 fields, peak near dec +5) is the second alternative.
