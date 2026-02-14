# Z/XY Geometry Scaling Audit — Progress Log

> Started: 2026-02-14
> Goal: Exhaustive audit of z-direction and xy-direction geometry scaling

---

## Pre-Audit Summary

**Recent fixes verified by Explore agent:**
- 7 files correctly use pixel_row_size for z-direction
- CTGeometry constructor correctly uses detector_row_size for z
- NB05 has per-scanner water calibration z_cm

**Gaps identified:**
- polychromatic.jl, driver.jl, photon_counting.jl: volume_fov threading not fully verified
- scatter.jl, crosstalk.jl, heel_effect.jl, PCCT files: not checked for pixel_size/pixel_row_size
- SIRT/CGLS/Hybrid IR: not verified for z-direction handling
- NB01-04: z_cm consistency not checked

---

## Audit Log

### 2026-02-14: GEO-001 — IN PROGRESS
**Agent:** Exhaustive grep of pixel_size/pixel_row_size/pixel_col_size in src/
**Plan:** Check every occurrence, classify direction (xy/z), give verdict (CORRECT/WRONG)

Files to check (from grep results):
1. src/projection/siddon.jl — KNOWN CORRECT
2. src/source/flat_filter.jl — KNOWN CORRECT
3. src/source/focal_spot.jl — KNOWN CORRECT
4. src/source/bowtie_filter.jl — KNOWN CORRECT
5. src/detector/detector_efficiency.jl — KNOWN CORRECT
6. src/reconstruction/core/backprojection.jl — KNOWN CORRECT
7. src/reconstruction/core/filtering.jl — KNOWN CORRECT
8. src/metrics/psf.jl — NEEDS CHECK
9. src/metrics/nps.jl — NEEDS CHECK
10. src/metrics/mtf.jl — NEEDS CHECK
11. src/reconstruction/mbir/mbir.jl — NEEDS CHECK
12. src/reconstruction/fbp/fdk.jl — NEEDS CHECK
13. src/detector/detector_noise.jl — NEEDS CHECK
14. src/detector/photon_counting.jl — NEEDS CHECK
15. src/detector/pcct/detector_response.jl — NEEDS CHECK
16. src/detector/pcct/charge_transport.jl — NEEDS CHECK
17. src/api/driver.jl — NEEDS CHECK
18. src/api/workspace.jl — NEEDS CHECK
19. src/scanners/scanners.jl — NEEDS CHECK
20. src/geometry/scanner.jl — NEEDS CHECK
