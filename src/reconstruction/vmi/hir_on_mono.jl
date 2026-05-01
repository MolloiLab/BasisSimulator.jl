"""
HIR-on-Mono+ — per-energy iterative refinement of a VMI volume.

Pipeline (per target energy E):
  1. HU sanitization     : NaN/Inf → -1000 (air); clamp to [-1000, 3000].
  2. HU → μ at energy E  : μ_data = μw_E · (1 + HU_capped / 1000).
  3. GPU forward project : sino_data = A · μ_data.
  4. HU → μ at energy E  : μ_init = μw_E · (1 + HU_mono / 1000).
  5. HIR reconstruct     : warm-start from `μ_init`, refine against
                           `sino_data` with Huber-prior PWLS-OS.
  6. μ → HU              : final HU at energy E.

The pre-Mono+ "capped" volume provides natural data noise for HIR to
iterate against; the Mono+ image (smoothed VMI) provides the warm start
so HIR doesn't have to claw away from a noisy FDK init.

This is the "HIR equivalent" output of the image-domain VMI pipeline —
the Mono+ image alone is the "FBP equivalent".

# Public API
- [`apply_hir_on_mono`](@ref) — single-energy wrapper.

# Important
The function expects to run on a Metal/CUDA-resident workspace (HIR is
GPU-only).  Pass the GPU array constructor via the `gpu_arr_type` kwarg.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    apply_hir_on_mono(capped_HU, mono_HU;
                      energy_keV::Real,
                      geom::CTGeometry,
                      recon_size::NTuple{3, Int},
                      gpu_arr_type,
                      filter::Union{FilterType, Symbol} = StandardFilter(),
                      water_material  = XA.Materials.water,
                      hir_strength::Int            = 3,
                      hir_lambda::Real             = 4.0f0,
                      hir_nepochs::Int             = 2,
                      hir_n_subsets::Int           = 12,
                      hir_huber_delta::Real        = 0.06f0,
                      hir_relaxation::Real         = 0.35f0,
                      hir_target_noise_reduction::Tuple{Int, Int} = (25, 35),
                      hu_clamp_lo::Real            = -1000f0,
                      hu_clamp_hi::Real            = 3000f0,
                      hu_air_replace::Real         = -1000f0,
                      sim_noise_floor_hu::Union{Nothing, Real} = nothing,
                      verbose::Bool                = true)
        -> Array{Float32, 3}

Run HIR-on-Mono+ for a single VMI energy.

# Required positional arguments
- `capped_HU::AbstractArray{Float32, 3}` : pre-Mono+ VMI HU volume at the
  target energy.  Provides the noisy data sinogram via forward project.
- `mono_HU::AbstractArray{Float32, 3}`   : Mono+ HU volume at the target
  energy.  Used as the warm-start `init_volume` for HIR.

# Required keyword arguments
- `energy_keV`        : VMI target energy (keV).
- `geom`              : CTGeometry for the recon (matches the data
                        sinogram dimensions).
- `recon_size`        : `(nx, ny, nz)` recon volume shape — must equal
                        `size(capped_HU)` and `size(mono_HU)`.
- `gpu_arr_type`      : GPU array constructor.  `MtlArray` for Metal,
                        `CuArray` for CUDA, `ROCArray` for ROCm.

# HIR hyperparams (kwargs)
- `hir_strength`              : `1..5` strength preset (default `3`).
- `hir_lambda`                : Huber regularization weight (default `4.0`).
- `hir_nepochs`               : OS-PWLS epochs (default `2`).
- `hir_n_subsets`             : ordered subsets per epoch (default `12`).
- `hir_huber_delta`           : Huber penalty edge threshold (default `0.06`).
- `hir_relaxation`            : SIRT relaxation (default `0.35`).
- `hir_target_noise_reduction`: `(min%, max%)` documentation tuple (default `(25, 35)`).

# Optional kwargs
- `filter`                    : FDK apodization filter for `create_hir_recon_workspace`
                                (default `StandardFilter()`).
- `water_material`            : XA material for `compute_μ_at_energy` (default `XA.Materials.water`).
- `hu_clamp_lo`/`hu_clamp_hi` : HU sanitization clamp range (default `[-1000, 3000]`).
- `hu_air_replace`            : value to substitute for NaN/±Inf voxels (default `-1000`).
- `sim_noise_floor_hu`        : if non-nothing, calls
                                `BS.add_system_noise_floor!(hu, sim_noise_floor_hu)`
                                on the final HIR HU output.  Useful for
                                matching a SE/poly noise-floor calibration.
- `verbose`                   : log per-energy σ before/after.

# Returns
A freshly allocated `Array{Float32, 3}` of HIR-refined HU at energy E.
"""
function apply_hir_on_mono(
        capped_HU::AbstractArray{Float32, 3},
        mono_HU::AbstractArray{Float32, 3};
        energy_keV::Real,
        geom::CTGeometry,
        recon_size::NTuple{3, Int},
        gpu_arr_type,
        filter::Union{FilterType, Symbol} = StandardFilter(),
        water_material  = XA.Materials.water,
        hir_strength::Int                       = 3,
        hir_lambda::Real                        = 4.0f0,
        hir_nepochs::Int                        = 2,
        hir_n_subsets::Int                      = 12,
        hir_huber_delta::Real                   = 0.06f0,
        hir_relaxation::Real                    = 0.35f0,
        hir_niter::Int                          = 30,
        hir_target_noise_reduction::Tuple{Int, Int} = (25, 35),
        hu_clamp_lo::Real                       = -1000f0,
        hu_clamp_hi::Real                       = 3000f0,
        hu_air_replace::Real                    = -1000f0,
        sim_noise_floor_hu::Union{Nothing, Real} = nothing,
        verbose::Bool                           = true,
    )
    size(capped_HU) == size(mono_HU) ||
        error("apply_hir_on_mono: capped_HU shape $(size(capped_HU)) ≠ mono_HU shape $(size(mono_HU))")
    size(capped_HU) == recon_size ||
        error("apply_hir_on_mono: input shape $(size(capped_HU)) ≠ recon_size $(recon_size)")

    μw_E = Float32(compute_μ_at_energy(water_material, Float64(energy_keV)))

    # ── HU sanitization (NaN/Inf → air, clamp range) ──
    rep   = Float32(hu_air_replace)
    lo    = Float32(hu_clamp_lo); hi = Float32(hu_clamp_hi)
    hu_capped = clamp.(replace(capped_HU, NaN => rep, Inf => rep, -Inf => rep), lo, hi)
    hu_mono   = clamp.(replace(mono_HU,   NaN => rep, Inf => rep, -Inf => rep), lo, hi)

    # ── HU → μ at E (GPU-resident) ──
    μ_data_gpu = gpu_arr_type(@. μw_E * (1f0 + hu_capped / 1000f0))
    sino_gpu   = forward_project(μ_data_gpu, geom)
    init_gpu   = gpu_arr_type(@. μw_E * (1f0 + hu_mono / 1000f0))

    # ── HIR ──
    ws = create_hir_recon_workspace(sino_gpu, geom, recon_size;
        strength = hir_strength, filter = filter)
    ws.params = HIRParams(
        Int(hir_strength), Float32(hir_lambda), Int(hir_niter),
        Int(hir_nepochs),  Int(hir_n_subsets),
        Float32(hir_huber_delta), Float32(hir_relaxation),
        hir_target_noise_reduction)
    reconstruct!(ws, sino_gpu, geom, recon_size; init_volume = init_gpu)
    recon_μ = Array(ws.volume)

    # ── μ → HU + optional noise floor ──
    recon_hu = Float32.(to_hounsfield(recon_μ; μ_water = μw_E))
    if sim_noise_floor_hu !== nothing
        add_system_noise_floor!(recon_hu, Float64(sim_noise_floor_hu))
    end

    if verbose
        nz = size(mono_HU, 3); mid_z = max(1, nz ÷ 2)
        nx = size(mono_HU, 1)
        i0, i1 = max(1, nx ÷ 2 - 50), min(nx, nx ÷ 2 + 50)
        ny = size(mono_HU, 2)
        j0, j1 = max(1, ny ÷ 2 - 50), min(ny, ny ÷ 2 + 50)
        roi_b = mono_HU[i0:i1, j0:j1, mid_z]
        roi_a = recon_hu[i0:i1, j0:j1, mid_z]
        @info "[HIR-on-Mono+] $(Int(round(energy_keV))) keV  μw_E=$(round(μw_E, sigdigits = 3)) cm⁻¹  σ=$(round(std(roi_b), digits = 1)) → $(round(std(roi_a), digits = 1)) HU"
    end

    ws = nothing; sino_gpu = nothing; init_gpu = nothing
    μ_data_gpu = nothing; hu_capped = nothing; hu_mono = nothing
    GC.gc(true)

    recon_hu
end


export apply_hir_on_mono
