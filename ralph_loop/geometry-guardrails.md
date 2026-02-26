# Geometry Scaling Audit Guardrails

> Rules and key locations for the z/xy geometry audit.

---

## The Core Rule

**CHECK EVERY occurrence. Do NOT assume any file is correct without reading the code.**

For each geometry variable found:
1. What direction is it used for? (z/rows/v or xy/cols/u)
2. Is the correct variable used? (pixel_row_size for z, pixel_size for xy)
3. Are units consistent? (cm in CTGeometry, mm in Scanner)

---

## Variable Mapping

| Variable | Domain | Direction | Units |
|----------|--------|-----------|-------|
| `pixel_size` | CTGeometry | xy (columns) | cm |
| `pixel_row_size` | CTGeometry | z (rows) | cm |
| `detector_col_size` | Scanner | xy (columns) | mm at isocenter |
| `detector_row_size` | Scanner | z (rows) | mm at isocenter |
| `pixel_size_mm` | PhotonCountingDetector | varies | mm |
| `pixel_pitch` | PCCT code | varies | mm |
| `fov[1]`, `fov[2]` | CTGeometry | xy | cm |
| `fov[3]` | CTGeometry | z | cm |

**WARNING:** `pixel_size_mm` in PhotonCountingDetector may be ambiguous — check if it's row, col, or both.

---

## Key File Locations

### Already Verified (7 files — pixel_row_size fix)

| File | Status | What was fixed |
|------|--------|----------------|
| `src/projection/siddon.jl` | CORRECT | v_offset uses pixel_row_size |
| `src/reconstruction/core/backprojection.jl` | CORRECT | pixel_row_mag for z |
| `src/reconstruction/core/filtering.jl` | CORRECT | v uses pixel_row_size |
| `src/source/flat_filter.jl` | CORRECT | v_offset uses pixel_row_size_det |
| `src/source/bowtie_filter.jl` | CORRECT | v_offset uses pixel_row_size_det |
| `src/source/focal_spot.jl` | CORRECT | blur_length uses pixel_row_size_det |
| `src/detector/detector_efficiency.jl` | CORRECT | v_offset uses pixel_row_size_det |

### Need Verification

| File | Why |
|------|-----|
| `src/detector/scatter.jl` | May use detector geometry for scatter kernel |
| `src/detector/crosstalk.jl` | Pixel neighbor operations |
| `src/detector/detector_lag.jl` | May reference geometry |
| `src/source/heel_effect.jl` | Anode-cathode axis geometry |
| `src/detector/photon_counting.jl` | PCCT detector pixel geometry |
| `src/detector/pcct/*.jl` | Charge cloud, fluorescence, pileup |
| `src/reconstruction/ir/sirt.jl` | Forward projection in iterative loop |
| `src/reconstruction/ir/cgls.jl` | Forward projection in iterative loop |
| `src/reconstruction/hybrid_ir/hybrid_ir.jl` | PWLS forward projection |
| `src/reconstruction/mbir/mbir.jl` | Model-based forward projection |
| `src/reconstruction/fbp/helical_recon.jl` | Spiral z-geometry |
| `src/projection/polychromatic.jl` | volume_fov threading |
| `src/api/driver.jl` | volume_fov in all simulate! paths |
| `src/api/workspace.jl` | Workspace geometry setup |

---

## Common Mistakes to Watch For

1. **Using `pixel_size` for z-direction** — The most common error. Should be `pixel_row_size`.

2. **Using `geom.pixel_size * magnification` for both u and v** — Should use `pixel_row_size * magnification` for v.

3. **Hardcoded square pixel assumption** — Code that uses `pixel_size` for both directions assumes square pixels. GE Revolution has 0.6mm col × 0.625mm row.

4. **Missing volume_fov in forward projection calls** — Calls to siddon_forward_project! during simulation should pass `volume_fov=phantom.fov`. Calls during reconstruction should NOT.

5. **detector_col_size used for z-direction** — Scanner struct has separate row/col sizes.

6. **pixel_size_mm ambiguity in PCCT** — The PhotonCountingDetector may have a single `pixel_size_mm` field that assumes square pixels. For non-square PCCT detectors (future), this would break.

---

## Absolute Rules

1. **NEVER** skip a grep result — check every occurrence
2. **NEVER** make code changes during AUDIT stories
3. **NEVER** go more than 10 minutes without a checkpoint commit
4. **NEVER** exit with zero commits
5. **ALWAYS** document exact file:line for every finding
6. **ALWAYS** specify which direction (z/xy) each variable is used for
7. **ALWAYS** check context — a `pixel_size` in a comment is not a bug
8. **ALWAYS** verify variable assignments, not just field accesses

---

## Working Directory

`/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`
