# XCAT PCCT Artifact Guardrails

> Read this BEFORE doing any work. Key code locations and rules for the PCCT cupping investigation.

---

## The Core Rule

**READ code, TRACE data flow, COMPARE EICT vs PCCT paths. Do NOT guess.**

Every diagnostic finding must include:
1. Exact file path and line number
2. Exact numeric value or formula
3. Comparison between EICT and PCCT behavior
4. Verdict: does this explain PCCT-specific cupping?

---

## Key File Locations

### PCCT Signal Chain (Most Relevant)

| File | Key Function | Line | Purpose |
|------|-------------|------|---------|
| `src/api/driver.jl` | `simulate!(ws::PCCTWorkspace, ...)` | ~545 | PCCT simulation hot path |
| `src/api/driver.jl` | `_combine_pcct_bins()` | 467 | Combine energy bins → single sinogram |
| `src/api/driver.jl` | `simulate!(ws::EICTWorkspace, ...)` | ~672 | EICT simulation (for comparison) |
| `src/api/driver.jl` | `build_physics_config()` | ~1298 | Physics config construction |
| `src/detector/photon_counting.jl` | `pcct_forward_project()` | — | Energy-resolved projection with DRM |
| `src/detector/photon_counting.jl` | `apply_pcct_noise!()` | — | PCCT-specific noise model |
| `src/correction/beam_hardening_correction.jl` | `bhc_water_default()` | 205 | BHC coefficients [0, 1.05, -0.02, 0.001] |
| `src/correction/beam_hardening_correction.jl` | `apply_bhc!()` | ~465 | Apply polynomial correction |

### EICT Signal Chain (For Comparison)

| File | Key Function | Purpose |
|------|-------------|---------|
| `src/projection/polychromatic.jl` | `_forward_project_poly!()` | Beer-Lambert spectral integration |
| `src/detector/physics_pipeline.jl` | `apply_physics_effects!()` | 13 physics effects pipeline |
| `src/detector/detector_noise.jl` | `sim_detect()` | EICT noise application |
| `src/detector/detector_noise.jl` | `compute_detector_I0()` | I0 from geometry |

### Reconstruction

| File | Key Function | Purpose |
|------|-------------|---------|
| `src/reconstruction/fbp/fdk.jl` | `fdk_reconstruct()` | FDK entry point |
| `src/reconstruction/core/filtering.jl` | `filter_sinogram!()` | Ramp/standard filter |
| `src/reconstruction/core/backprojection.jl` | `voxel_backproject!()` | Weighted backprojection |

### Workspace

| File | Key Struct | Purpose |
|------|-----------|---------|
| `src/api/workspace.jl` | `PCCTWorkspace` | PCCT workspace (line ~30) — has `combined` field |
| `src/api/workspace.jl` | `EICTWorkspace` | EICT workspace — has `sinogram` field |
| `src/api/workspace.jl` | `create_workspace()` | PCCT workspace factory |
| `src/api/workspace.jl` | `create_eict_workspace()` | EICT workspace factory |

### Notebook

| File | Purpose |
|------|---------|
| `verification/notebooks/05_xcat_full.jl` | THE notebook with the artifact |

---

## Critical Architecture Insights

### PCCT Path (simulate! → PCCTWorkspace)

```
simulate!(ws::PCCTWorkspace, phantom, scanner, protocol, sim_opts, recon_opts)
  │
  ├── pcct_forward_project(phantom.mask, geom, pcct_detector; ...)
  │     │ For each energy bin:
  │     │   Forward project → μ_E volumes → Siddon → sinograms
  │     │   Apply DRM (charge sharing, K-fluorescence, pileup)
  │     │   Normalize: sino_bin = -log(N_bin / I0_bin)
  │     └── Returns EnergyResolvedSinogram with per-bin sinograms
  │
  ├── _combine_pcct_bins(pcct_sino, detector, ...) → combined ideal sinogram
  │     │ Convert bins back to counts: N_bin = I0_bin × exp(-sino_bin)
  │     │ Sum: N_total = Σ N_bin
  │     │ Combined: sino = -log(N_total / I0_total)
  │     └── NOTE: I0_bins come from _compute_bin_I0() with DRM weighting
  │
  ├── apply_bhc!(combined_ideal, bhc)  ← SAME coefficients as EICT!
  │
  ├── apply_pcct_noise!(pcct_sino, ...)  ← noise on PER-BIN sinograms
  │
  ├── _combine_pcct_bins(pcct_sino, ...) → combined noisy sinogram
  │
  ├── apply_bhc!(combined_noisy, bhc)
  │
  └── Material decomposition (for VMI)
```

### EICT Path (simulate! → EICTWorkspace)

```
simulate!(ws::EICTWorkspace, phantom, scanner, protocol, sim_opts, recon_opts)
  │
  ├── _forward_project_poly!(ws.sinogram, phantom.mask, geom; ...)
  │     │ For each energy bin:
  │     │   μ_volume = compute attenuation at energy E
  │     │   L_E = Siddon ray trace
  │     │   I_total += w_E × exp(-L_E)
  │     └── sinogram = -log(I_total)
  │
  ├── apply_physics_effects!(ws.sinogram, config, geom)
  │     │ fill_factor, flat_filter, bowtie, scatter, crosstalk,
  │     │ focal_spot, heel_effect, lag, bhc
  │     └── NOTE: 13 effects applied in specific order
  │
  ├── sim_detect(ws.sinogram, geom, protocol; ...)
  │     │ compute_detector_I0() → quantum noise
  │     └── sinogram_noisy
  │
  └── Apply BHC (inside apply_physics_effects, before noise)
```

### KEY DIFFERENCE: BHC Application

```
EICT: physics_effects → BHC → noise → done
       (BHC on polychromatic sinogram — CORRECT)

PCCT: pcct_forward → combine → BHC → noise → combine → BHC → done
       (BHC on combined sinogram — MAYBE WRONG?)
       (Also: BHC applied TWICE — once ideal, once noisy!)
```

### KEY DIFFERENCE: Physics Effects

EICT applies 13 effects via apply_physics_effects!(). PCCT applies a SUBSET.
The PCCT path may be MISSING effects like fill_factor, flat_filter, bowtie,
scatter, crosstalk, focal_spot, heel_effect, lag that the EICT path has.
These effects modify the sinogram values, which changes BHC behavior.

---

## Known Issues (Confirmed)

1. **Z-coverage mismatch**: EICT 8.0cm vs PCCT 5.12cm → different slice index = different anatomy. This is EXPECTED and NOT a bug.

2. **Water calibration z_cm**: Uses GE pitch (0.625mm) for PCCT. For uniform water cylinder, impact on μ_water is minimal.

3. **volume_fov**: Correctly threaded through all workspace paths. Public API missing it (separate bug).

4. **Previous fixes applied**: StandardFilter default, pixel_row_size fix, phantom.fov threading.

---

## Absolute Rules

1. **NEVER** make code changes during DIAGNOSTIC stories — read-only
2. **NEVER** guess at sinogram/attenuation values — compute them from the actual formula
3. **NEVER** claim "this matches EICT" without showing the numbers side-by-side
4. **NEVER** go more than 10 minutes without a checkpoint commit
5. **NEVER** exit with zero commits
6. **ALWAYS** document exact line numbers and file paths
7. **ALWAYS** compare EICT and PCCT paths for the same operation
8. **ALWAYS** consider whether an effect is position-dependent (cupping requires position dependence)
9. **ALWAYS** check if an effect exists in BOTH EICT and PCCT (if yes, it cancels out)
10. **ALWAYS** think about the physical mechanism: does this effect create higher attenuation at edges?

---

## Mistake Prevention

### Cupping vs Non-Cupping Artifacts

Cupping = radially varying intensity error:
- Center of reconstruction appears darker than it should
- Edges appear brighter (or vice versa for inverted cupping)
- Caused by POSITION-DEPENDENT errors in the signal chain

NOT cupping:
- Uniform noise increase (all positions equally affected)
- Overall HU offset (uniform shift)
- Streak artifacts (angle-dependent, not radial)

### Think About Position Dependence

For each hypothesis, ask: "Does this effect vary with position in the sinogram?"
- If YES → could cause cupping → investigate
- If NO → cannot cause cupping → rule out

Examples:
- Wrong I0 (uniform) → NO cupping → rule out
- BHC polynomial on wrong sinogram type → YES cupping (polynomial's correction varies with path length) → investigate
- Missing flat filter → NO cupping (adds uniform attenuation) → rule out
- Missing bowtie filter → YES cupping (fan-angle dependent) → investigate

---

## Working Directory

`/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`
