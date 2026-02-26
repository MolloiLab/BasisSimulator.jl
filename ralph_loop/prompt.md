# BasisSimulator — Two-Material Beam Hardening Correction

You are an autonomous agent implementing a two-material (water + bone) beam hardening correction algorithm in the BasisSimulator.jl CT simulator package.

## STEP 0: Check Environment

```bash
ls /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/src/BasisSimulator.jl
ls /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl/src/correction/beam_hardening_correction.jl
```

---

## STEP 1: Read State (Do This First — EVERY iteration)

```
1. ralph_loop/prd.json       — Stories with status and priorities
2. ralph_loop/progress.md    — What was done previously
3. ralph_loop/guardrails.md  — Rules you MUST follow
```

For BHC-CORE, also read:
```
4. src/correction/beam_hardening_correction.jl  — Existing water-only BHC code
5. src/object/materials.jl                      — Material registry
6. src/object/attenuation.jl                    — Attenuation coefficient functions
```

For BHC-SCRIPT, also read:
```
7. ralph_loop/scripts/bhc_two_material_test.jl  — EXISTING reference script (rewrite to use core functions)
```

---

## STEP 2: Pick a Story

Select the highest-priority **unblocked** story:
- `"status": "open"` (or `"in_progress"` — finish it first)
- All `"blockers"` stories have `"status": "done"`

Priority order: lower number first. Set chosen story to `"in_progress"` in prd.json immediately.

---

## STEP 3: Execute

### CHECKPOINT RULE (CRITICAL)

**Commit early and often.** Every agent iteration may be killed at any time.

```
CHECKPOINT after:
  - Every function added
  - Every file created/modified
  - Every 10 minutes of work
  - BEFORE running long commands

A checkpoint = update progress.md + commit:
  1. Append findings to ralph_loop/progress.md
  2. Update ralph_loop/prd.json if status changed
  3. git add ralph_loop/ src/
  4. git commit -m "[STORY-ID]: [what was done]"
```

---

## ALGORITHM KNOWLEDGE — Martinez/Fessler 2022 "2DCalBH"

This is the complete algorithm you are implementing. READ THIS CAREFULLY.

### Background

Water-only polynomial BHC corrects "cupping" artifacts in uniform objects but leaves
dark-band/streak artifacts between dense structures (e.g., bone, calcium inserts).
The two-material approach corrects BOTH effects.

### Algorithm Steps

**Pass 1 (Water-only BHC — already exists in codebase):**
1. `calibrate_bhc(energies, weights; order=5)` → `BeamHardeningCorrection`
2. `apply_bhc(sinogram, bhc)` → water-corrected sinogram
3. `fdk_reconstruct(sino_water, geom, matrix_size)` → preliminary image
4. Convert to HU: `hu = 1000 * (recon - μ_water_ref) / μ_water_ref`

**Pass 2 (Bone correction — NEW):**

**Step 1: Smooth tissue fraction decomposition (Elbakri/Fessler 2003)**

For each voxel in the preliminary HU image, compute bone fraction using C1 smoothstep:

```julia
function bone_fraction_smooth(hu; hu_low=100.0, hu_high=500.0)
    hu <= hu_low && return 0.0
    hu >= hu_high && return 1.0
    t = (hu - hu_low) / (hu_high - hu_low)
    return t * t * (3.0 - 2.0 * t)  # 3t² - 2t³
end
```

- Below `hu_low` (100 HU): 100% soft tissue (water-like)
- Above `hu_high` (500 HU): 100% bone-like
- Between: smooth cubic interpolation (NO hard segmentation)

Create bone-weighted μ image: `bone_μ = bone_fraction .* recon_μ`

**Step 2: Forward-project bone image (Siddon)**

```julia
p_b = Array(siddon_forward_project(bone_μ_3d, geom; volume_extent=phantom_extent))
```

Soft tissue line integral from the water-corrected sinogram:
```julia
p_s = sino_water_corrected - p_b
```

**Step 3: Convert to equivalent material path lengths**

```julia
L_w = p_s ./ μ_water_ref   # cm of water
L_b = p_b ./ μ_bone_ref    # cm of bone
L_w .= max.(L_w, 0.0)      # clamp negatives
L_b .= max.(L_b, 0.0)
```

**Step 4: Compute exact 2-material polychromatic correction**

For each ray (indexed by `idx`):

```julia
# Polychromatic forward model
I_poly = Σ w_norm[e] * exp(-μ_water_E[e] * L_w[idx] - μ_bone_E[e] * L_b[idx])
F_2mat = -log(I_poly)

# Monochromatic reference
p_mono = μ_water_ref * L_w[idx] + μ_bone_ref * L_b[idx]

# Full BH correction (captures both water and bone hardening)
correction[idx] = p_mono - F_2mat
```

Use `Threads.@threads` for the outer loop. Use Float64 internally for precision.

**Step 5: Apply correction to RAW sinogram**

```julia
sino_corrected = sino_raw + correction
```

The correction is applied to the ORIGINAL (raw polychromatic) sinogram, NOT the
water-corrected sinogram. This captures the FULL beam hardening (both water and bone
components) in a single pass.

### Key Physics Insight

With a known spectrum (simulation), Step 4 is EXACT — no approximation. The only
approximation is the tissue fraction decomposition in Step 1, which uses the preliminary
(water-BHC) reconstruction. But this is very robust — even 10% errors in tissue
fractions have negligible effect on the correction (Joseph & Spital 1978).

### Material Data

- **Water**: `XA.Materials.water`
- **Cortical bone**: `XA.Materials.corticalbone` (NO underscore — this is critical!)
- Attenuation: `ustrip(u"cm^-1", XA.linear_attenuation_coeff(material, E * u"keV"))`
- Or use `compute_μ_at_energy(material, energy_keV)` from attenuation.jl

### Typical Values at 70 keV Reference

- μ_water ≈ 0.1928 cm⁻¹
- μ_bone ≈ 0.4935 cm⁻¹
- Bone/water ratio: 2.56× at 70 keV, 6.8× at 30 keV (this energy dependence causes BH)

---

## BHC-CORE Story: Implementation Details

### Where to Add Code

Add ALL new code at the END of `src/correction/beam_hardening_correction.jl`, BEFORE no
other files. Keep all existing water-only code untouched.

### New Exports (add to existing export block near top of file)

```julia
export bone_fraction_smooth, TwoMaterialBHC
export calibrate_bhc_two_material, apply_bhc_two_material
```

### Struct: TwoMaterialBHC

```julia
struct TwoMaterialBHC
    water_bhc::BeamHardeningCorrection
    energies::Vector{Float64}
    w_norm::Vector{Float64}
    μ_water_E::Vector{Float64}
    μ_bone_E::Vector{Float64}
    μ_water_ref::Float64
    μ_bone_ref::Float64
    reference_energy_keV::Float64
    hu_low::Float64
    hu_high::Float64
end
```

### Function: calibrate_bhc_two_material

```julia
function calibrate_bhc_two_material(
    energies::Vector,
    weights::Vector;
    order::Int = 5,
    max_path_cm::Real = 50.0,
    n_points::Int = 100,
    reference_energy_keV::Real = 70.0,
    hu_low::Real = 100.0,
    hu_high::Real = 500.0
)
    # 1. Calibrate water-only BHC
    water_bhc = calibrate_bhc(energies, weights;
        order=order, max_path_cm=max_path_cm,
        n_points=n_points, reference_energy_keV=reference_energy_keV)

    # 2. Compute attenuation data
    import XrayAttenuation as XA
    using Unitful: ustrip, @u_str
    water_mat = XA.Materials.water
    bone_mat = XA.Materials.corticalbone

    E_vec = Float64.(energies)
    w_norm = Float64.(weights) ./ sum(Float64.(weights))

    μ_water_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, E * u"keV")) for E in E_vec]
    μ_bone_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, E * u"keV")) for E in E_vec]
    μ_water_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, reference_energy_keV * u"keV"))
    μ_bone_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, reference_energy_keV * u"keV"))

    return TwoMaterialBHC(water_bhc, E_vec, w_norm, μ_water_E, μ_bone_E,
                          μ_water_ref, μ_bone_ref, Float64(reference_energy_keV),
                          Float64(hu_low), Float64(hu_high))
end
```

IMPORTANT: The `import XrayAttenuation as XA` and `using Unitful` are already at the
top of materials.jl which is loaded before this file. Check how the existing code
handles imports — use the same pattern. The file already has `import AcceleratedKernels as AK`.
You may need to add `import XrayAttenuation as XA` and `using Unitful: ustrip, @u_str`
at the top of beam_hardening_correction.jl if not already there. Check what's available
in the module scope by reading `src/BasisSimulator.jl`.

### Function: apply_bhc_two_material

```julia
function apply_bhc_two_material(
    sinogram_raw::AbstractArray{T, 3},
    bhc_2mat::TwoMaterialBHC,
    geom,           # CTGeometry
    matrix_size::Tuple{Int,Int,Int};
    volume_extent=nothing
) where T <: AbstractFloat
    # Pass 1: Water-only BHC → FDK → preliminary HU image
    sino_water = apply_bhc(sinogram_raw, bhc_2mat.water_bhc)
    recon = Array(fdk_reconstruct(sino_water, geom, matrix_size))
    hu = 1000.0f0 .* (recon .- T(bhc_2mat.μ_water_ref)) ./ T(bhc_2mat.μ_water_ref)

    # Pass 2: Bone correction
    # Step 1: Smooth tissue fraction decomposition
    nx, ny, nz = size(recon)
    bone_μ_3d = zeros(Float32, nx, ny, nz)
    for s in 1:nz
        for j in 1:ny, i in 1:nx
            bf = bone_fraction_smooth(hu[i,j,s]; hu_low=bhc_2mat.hu_low, hu_high=bhc_2mat.hu_high)
            bone_μ_3d[i,j,s] = Float32(bf * recon[i,j,s])
        end
    end

    # Step 2: Forward-project bone image
    p_b = Array(siddon_forward_project(bone_μ_3d, geom; volume_extent=volume_extent))

    # Soft tissue line integral
    sino_water_cpu = Array(sino_water)
    p_s = sino_water_cpu .- p_b

    # Step 3: Material path lengths
    L_w = p_s ./ Float32(bhc_2mat.μ_water_ref)
    L_b = p_b ./ Float32(bhc_2mat.μ_bone_ref)
    L_w .= max.(L_w, 0.0f0)
    L_b .= max.(L_b, 0.0f0)

    # Step 4: Exact 2-material correction
    sino_raw_cpu = Array(sinogram_raw)
    correction = zeros(Float32, size(sino_raw_cpu))
    w_norm_64 = Float64.(bhc_2mat.w_norm)
    μ_w_64 = Float64.(bhc_2mat.μ_water_E)
    μ_b_64 = Float64.(bhc_2mat.μ_bone_E)
    μ_water_ref = bhc_2mat.μ_water_ref
    μ_bone_ref = bhc_2mat.μ_bone_ref

    Threads.@threads for idx in eachindex(sino_raw_cpu)
        Lw = Float64(L_w[idx])
        Lb = Float64(L_b[idx])
        (Lw < 1e-6 && Lb < 1e-6) && continue

        I_poly = 0.0
        for e in eachindex(w_norm_64)
            I_poly += w_norm_64[e] * exp(-μ_w_64[e] * Lw - μ_b_64[e] * Lb)
        end
        F_2mat = I_poly > 0.0 ? -log(I_poly) : 0.0
        p_mono = μ_water_ref * Lw + μ_bone_ref * Lb
        correction[idx] = Float32(p_mono - F_2mat)
    end

    # Step 5: Apply to RAW sinogram
    return sino_raw_cpu .+ correction
end
```

---

## BHC-SCRIPT Story: Test Script Details

### Location

`ralph_loop/scripts/bhc_two_material_test.jl`

A reference version already exists at this path. READ IT FIRST, then REWRITE it to:
- Use `calibrate_bhc_two_material()` instead of inline calibration
- Use `apply_bhc_two_material()` instead of inline algorithm
- Keep the same visualization structure (2×4 grid, colorbars, annotation)
- Keep `mkpath(output_dir)` and `save(output_path, fig, px_per_unit=2)`

### Key API Calls

```julia
# Spectrum
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)

# Phantom & geometry
phantom = create_gammex_472(n_voxels=256, n_slices=16, fov_cm=35.0, z_cm=2.0)
geom = create_aquilion_one(n_angles=720, n_rows=16, n_cols=736, fov_cm=35.0, z_cm=2.0)
materials = get_region_materials()

# Forward project (polychromatic, no corrections)
sino_raw = forward_project(phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    volume_extent=phantom.extent)

# Water-only BHC
bhc_water = calibrate_bhc(energies, weights; order=5, reference_energy_keV=70.0)
sino_water = apply_bhc(sino_raw, bhc_water)

# Two-material BHC
bhc_2mat = calibrate_bhc_two_material(energies, weights;
    order=5, reference_energy_keV=70.0)
sino_2mat = apply_bhc_two_material(sino_raw, bhc_2mat, geom, (256, 256, 16);
    volume_extent=phantom.extent)

# Reconstruct all three
recon_none  = Array(fdk_reconstruct(sino_raw, geom, matrix_size))
recon_water = Array(fdk_reconstruct(sino_water, geom, matrix_size))
recon_2mat  = Array(fdk_reconstruct(sino_2mat, geom, matrix_size))

# Convert to HU
μ_water_ref = bhc_2mat.μ_water_ref
hu_none  = 1000f0 .* (recon_none  .- μ_water_ref) ./ μ_water_ref
hu_water = 1000f0 .* (recon_water .- μ_water_ref) ./ μ_water_ref
hu_2mat  = 1000f0 .* (recon_2mat  .- μ_water_ref) ./ μ_water_ref
```

### Figure Layout

```
Row 0: Title "Two-Material Beam Hardening Correction — Gammex 472 at 120 kVp"
Row 1: [No BHC soft] [Water BHC soft] [Water+Bone BHC soft] [Colorbar HU]
Row 2: [No BHC wide] [Water BHC wide] [Bone Correction Map] [Colorbar ΔHU]
Row 3: Annotation with center ROI HU values
```

- Soft tissue window: W:400 L:0 → colorrange (-200, 200)
- Wide window: W:2000 L:0 → colorrange (-1000, 1000)
- Bone correction diff map: RdBu colormap, ±50 HU range
- Figure size: (1800, 900)

---

## BHC-RUN Story: Running the Script

```bash
cd /Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl
julia --project=. ralph_loop/scripts/bhc_two_material_test.jl
```

Timeout: This may take 5-15 minutes (JIT compilation + GPU forward projection + FDK).

Verify output:
```bash
ls -la ralph_loop/results/bhc_two_material.png
# Should be >10KB
```

If it errors, debug and fix. Common issues:
- Import errors: check that exports are in the right place
- Material name: it's `corticalbone` not `cortical_bone`
- GPU arrays: use `Array()` to move to CPU before indexing/plotting
- `volume_extent`: pass `phantom.extent` to forward_project and apply_bhc_two_material

---

## STEP 4: Document

After every checkpoint, append to `ralph_loop/progress.md`:

```markdown
### [STORY-ID] — Iteration N (YYYY-MM-DD)
- **Done:** [What was accomplished]
- **Files:** [Files created/modified]
- **Next:** [What to do next if not done]
```

---

## STEP 5: Commit

Working directory: `/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl`

```bash
git add ralph_loop/ src/
git commit -m "[STORY-ID]: [description]"
```

---

## STEP 6: Exit

- Story completed → Exit normally
- All stories done → Output: `RALPH_COMPLETE`
- Blocked → Output: `RALPH_BLOCKED: [reason]`

---

## Key Files

- **BHC source**: `src/correction/beam_hardening_correction.jl`
- **Materials**: `src/object/materials.jl` (XA.Materials.water, XA.Materials.corticalbone)
- **Attenuation**: `src/object/attenuation.jl` (compute_μ_at_energy)
- **Forward projection**: `src/projection/siddon.jl` (siddon_forward_project)
- **Reconstruction**: `src/reconstruction/fbp/fdk.jl` (fdk_reconstruct)
- **Polychromatic FP**: `src/projection/polychromatic.jl` (forward_project)
- **Main module**: `src/BasisSimulator.jl` (check imports/exports)
- **Test script**: `ralph_loop/scripts/bhc_two_material_test.jl`
- **Output PNG**: `ralph_loop/results/bhc_two_material.png`

## GPU Closure Rule (from CLAUDE.md)

NEVER conditionally assign a variable then capture it in an AK.foreachindex closure.
Use `let` to capture with concrete type. Example:

```julia
let coeffs = coeffs, order = order
    AK.foreachindex(sinogram) do idx
        # ... use coeffs, order ...
    end
end
```

This applies to ALL variables captured from conditional branches (if/else, ternary).
