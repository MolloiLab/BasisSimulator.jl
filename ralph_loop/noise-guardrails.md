# Noise Diagnosis Guardrails

> Read this BEFORE doing any work. Key code locations and rules for the 2x noise investigation.

---

## The Core Rule

**READ code, TRACE data flow, COMPARE implementations. Do NOT guess.**

Every diagnostic finding must include:
1. Exact file path and line number
2. Exact numeric value or formula
3. Comparison to expected/CatSim value
4. Verdict: does this explain 2x noise?

---

## Key File Locations

### Noise Pipeline (Most Relevant)

| File | Key Function | Line | Purpose |
|------|-------------|------|---------|
| `src/detector/detector_noise.jl` | `add_quantum_noise!()` | 586 | Core noise: λ = I0*exp(-p), noise = √λ*randn |
| `src/detector/detector_noise.jl` | `compute_detector_I0()` | 724 | I0 from geometry (MISSING η_det!) |
| `src/detector/detector_noise.jl` | `mA_to_I0()` | 259 | I0 from mA (HAS η_det) |
| `src/detector/detector_noise.jl` | `sim_detect()` | 767 | Applies noise via compute_detector_I0 |
| `src/detector/detector_noise.jl` | `apply_detector_model!()` | 682 | blur→noise→electronic |
| `src/detector/physics_pipeline.jl` | `apply_physics_effects!()` | 409 | All 13 effects in order |
| `src/detector/physics_pipeline.jl` | `full_physics_config()` | 325 | ALL effects, I0=1e6 HARDCODED |

### Forward Projection

| File | Key Function | Purpose |
|------|-------------|---------|
| `src/projection/polychromatic.jl` | `_forward_project_poly!()` | Beer-Lambert spectral integration |
| `src/projection/polychromatic.jl` | `forward_project()` | Unified API |
| `src/projection/siddon.jl` | `siddon_forward_project!()` | Ray tracing |

### Reconstruction (Noise Amplification)

| File | Key Function | Purpose |
|------|-------------|---------|
| `src/reconstruction/fbp/fdk.jl` | `fdk_reconstruct()` | FDK entry point |
| `src/reconstruction/core/filtering.jl` | `apply_ramp_filter!()` | Ramp filter (amplifies noise) |
| `src/reconstruction/core/filtering.jl` | `apply_cosine_weights!()` | Pre-filter weighting |
| `src/reconstruction/core/backprojection.jl` | `voxel_backproject!()` | Weighted backprojection |

### Spectrum

| File | Key Function | Purpose |
|------|-------------|---------|
| `src/source/spectrum.jl` | `load_spectrum()` | Load .dat files |
| `src/source/spectrum.jl` | `downsample_spectrum()` | Bin spectrum |

### Driver (API Layer)

| File | Key Function | Line | Purpose |
|------|-------------|------|---------|
| `src/api/driver.jl` | `_simulate_axial_single()` | 258 | Main simulation pipeline |
| `src/api/driver.jl` | `build_physics_config()` | 1298 | Constructs PhysicsConfig from SimOptions |
| `src/api/driver.jl` | `build_physics_config() line 1310` | — | **noise=nothing** (noise via sim_detect, not physics) |
| `src/api/options.jl` | `SimOptions()` | 95 | Fidelity presets |

### Verification Notebook

| File | Purpose |
|------|---------|
| `verification/notebooks/01_single_kvp_verification.jl` | THE primary CatSim comparison |

---

## Critical Architecture Insights

### Noise Path in Driver API (simulate())

```
simulate() → _simulate_axial_single()
  → build_physics_config()          # noise=nothing ALWAYS (line 1310)
  → forward_project(mask, geom; physics=config)
     → _forward_project_poly!()     # Beer-Lambert
     → apply_physics_effects!()     # 13 effects, but noise=nothing
  → sim_detect(sino_ideal, geom, protocol)   # Noise added HERE
     → compute_detector_I0()        # I0 from mA/geometry (no η_det!)
     → add_quantum_noise!(copy, model)
```

**Key insight**: In the driver path, noise is NEVER double-applied. But the I0 computation
uses `compute_detector_I0()` which is MISSING η_det (quantum efficiency ~0.85).

### Noise Path in Direct API (forward_project with physics)

```
forward_project(mask, geom; physics=full_physics_config())
  → _forward_project_poly!()
  → apply_physics_effects!()
     → apply_detector_model!()      # blur + noise + electronic noise
        → I0 = 1e6 (HARDCODED!)
```

**Key insight**: If notebook 01 uses direct API with `full_physics_config()`,
I0 is hardcoded to 1e6 regardless of mA/geometry.

### Noise Path in Notebook 01 (MOST LIKELY)

The notebook uses `SimOptions(fidelity=:high)` and `simulate()`. So:
1. Physics effects applied WITHOUT noise (noise=nothing in config)
2. `sim_detect()` adds noise with I0 from compute_detector_I0()
3. I0 is based on protocol mA/views/rotation + geometry

### Two Different I0 Functions (POTENTIAL BUG SOURCE)

| Function | η_det | Used By |
|----------|-------|---------|
| `compute_detector_I0()` (line 724) | NOT included | `sim_detect()` → driver |
| `mA_to_I0()` (line 259) | Included (0.85) | `clinical_detector_model()` |

Both should agree. The difference is ~15% in I0, which translates to ~7% in noise.
This alone doesn't explain 2x, but it's a real discrepancy.

---

## Known Unit Conventions

| Quantity | Siddon/Projection | CTGeometry | Scanner | detector_noise.jl |
|----------|-------------------|------------|---------|-------------------|
| Distance | mm | cm | mm | mm |
| Pixel size | mm | cm | mm | — |
| Attenuation μ | mm⁻¹ | — | — | — |
| Energy | keV | keV | — | — |
| I0 | photons/pixel/view | — | — | photons/pixel/view |

**CRITICAL**: Units convert at boundaries. Watch for:
- pixel_size in CTGeometry is cm, but in Scanner is mm
- compute_detector_I0() converts geom.pixel_size from cm to mm (*10)
- Magnification: pixel_at_detector = pixel_at_isocenter × (SDD/SID)

---

## Absolute Rules

1. **NEVER** make code changes during DIAGNOSTIC stories — read-only
2. **NEVER** guess at I0 values — compute them from the actual formula
3. **NEVER** claim "this matches" without showing the numbers side-by-side
4. **NEVER** go more than 10 minutes without a checkpoint commit
5. **NEVER** exit with zero commits
6. **ALWAYS** document exact line numbers and file paths
7. **ALWAYS** compute expected values from first principles when possible
8. **ALWAYS** compare to CatSim/XCIST behavior when available
9. **ALWAYS** estimate the noise factor contribution (e.g., "this adds 1.2x noise")
10. **ALWAYS** consider whether the effect also exists in CatSim (it might cancel out)

---

## Mistake Prevention

### Don't Chase Red Herrings

If an effect exists in BOTH BasisSimulator AND CatSim, it cancels out and cannot
explain the 2x difference. Only differences between the two implementations matter.

Example: Both BasisSimulator and CatSim use Poisson noise with Gaussian approximation
for high photon counts. This is NOT the cause of the discrepancy.

### Think About Factors, Not Absolutes

The noise scales as: σ ∝ 1/√I0 × (FDK_norm) × (ramp_gain) × (1/μ_water)

A 2x discrepancy requires one of these factors to be 2x off, OR several factors
to each be 1.3-1.5x off (since 1.4 × 1.4 ≈ 2.0).

### Most Likely Root Causes (Ranked)

1. **FDK normalization factor** — 2π/N vs π/N is exactly 2x
2. **Ramp filter Δ units** — cm vs mm is 10x, but squared = 100x (too much, unless partial)
3. **I0 computation** — missing η_det + wrong pixel area = maybe 1.5x
4. **Double-filtered spectrum** — two layers of Al filtration = ~1.5x more noise
5. **μ_water wrong** — if empirical μ_water is 2x too low, σ_HU is 2x too high
6. **Physics effects before noise** — compound effect of fill_factor, flat_filter, bowtie, scatter_correction

---

## CatSim Reference

CatSim/XCIST noise implementation (for comparison):
- `pyfiles/Detection_EI.py`: Main detector simulation
- `clib_build/src/rndpoi.c`: Poisson random number generator
- Noise model: Exact Poisson for small λ, Gaussian approximation for λ > 10
- I0: Computed from source flux, mA, exposure time, detector area, distance
- FDK: Standard TIGRE-compatible FDK with Ram-Lak filter

CatSim source (if available on this machine): `/Users/daleblack/Documents/dev/CatSim/`

---

## Working Directory

`/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`
