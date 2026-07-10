# =============================================================================
# Polychromatic Forward Projection with CatSim Signal Chain
# =============================================================================
#
# Memory-efficient polychromatic CT simulation using Beer-Lambert physics
# with optional CatSim-exact signal chain built-in.
#
# NOT from TIGRE - TIGRE is monochromatic only.
# This implements proper spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
#
# Algorithm Overview
# ------------------
# Polychromatic projection models the energy-dependent attenuation of X-rays
# as they traverse a heterogeneous medium. Unlike monochromatic projection
# (single energy), polychromatic simulation accounts for the broad energy
# spectrum produced by an X-ray tube and the energy-dependence of material
# attenuation coefficients.
#
# Mathematical Foundation: Beer-Lambert Law
# -----------------------------------------
# For a monochromatic beam of intensity I₀ at energy E traversing a path:
#
#     I(E) = I₀(E) × exp(-∫ μ(E, l) dl)
#
# where μ(E, l) is the linear attenuation coefficient (mm⁻¹) at energy E and
# position l along the ray path.
#
# For a polychromatic beam with spectral distribution w(E) (photon fluence
# weights normalized such that Σ w(E) = 1), the transmitted intensity is:
#
#     I_total = ∫ w(E) × I₀(E) × exp(-∫ μ(E, l) dl) dE
#
# In discretized form with N energy bins:
#
#     I_total = Σₑ wₑ × exp(-∫ μₑ(l) dl) = Σₑ wₑ × exp(-Lₑ)
#
# where wₑ is the normalized weight for energy bin e, and Lₑ is the line
# integral of attenuation at energy e computed via ray tracing.
#
# The measured projection value is converted back to an "effective" line
# integral via the log transform:
#
#     p = -log(I_total / I₀) = -log(Σₑ wₑ × exp(-Lₑ))
#
# Spectral Integration Method
# ---------------------------
# This implementation uses an energy-sequential approach for memory efficiency:
#
# 1. For each energy bin e in spectrum:
#    a. Create μ-volume: μᵢⱼₖ(e) = μ(material[i,j,k], E_e) for all voxels
#    b. Forward project: Lₑ = Siddon ray-trace through μ-volume
#    c. Accumulate: I_total += wₑ × exp(-Lₑ)
#
# 2. After all energies: p = -log(I_total)
#
# This approach requires O(1) additional memory per energy bin (one μ-volume),
# versus O(N) memory for storing all energy-specific projections simultaneously.
#
# Beam Hardening
# --------------
# The nonlinear log-sum-exp operation causes beam hardening artifacts:
# - p_poly ≠ Σₑ wₑ × pₑ (projection is NOT a linear combination)
# - Low-energy photons are preferentially absorbed, "hardening" the beam
# - Results in cupping artifacts in reconstructed images
# - Corrected via polynomial BHC (beam hardening correction) post-projection
#
# CatSim Signal Chain (when enabled via kwargs):
# 1. Polychromatic forward projection (Beer-Lambert)
# 2. Physics pipeline (scatter, crosstalk, focal spot, noise, lag)
# 3. Heel effect (intensity domain)
# 4. DAS model (gain + electronic noise)
# 5. Air scan calibration (CatSim-exact: noise-free air reference)
# 6. Low signal correction (replace negatives with smoothed neighbors)
# 7. Log transform
# 8. Beam hardening correction
#
# Physical Units
# --------------
# - Energy bins: keV
# - Spectral weights: normalized photon fluence (dimensionless, sum to 1)
# - Attenuation coefficients μ: mm⁻¹
# - Line integrals: dimensionless (mm⁻¹ × mm)
# - Intensity: relative transmitted photon counts (dimensionless)
#
# GPU Compatibility (via AcceleratedKernels.jl)
# ---------------------------------------------
# - ✅ Metal (Apple Silicon) - Primary development platform
# - ✅ CUDA (NVIDIA GPUs)
# - ✅ ROCm (AMD GPUs)
# - ✅ Intel oneAPI
# - ✅ CPU fallback (multi-threaded)
#
# The algorithm auto-detects backend from array type. Pass MtlArray (Metal),
# CuArray (CUDA), ROCArray (ROCm), or regular Array (CPU).
#
# References
# ----------
# 1. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent
#    Advances." 3rd ed. SPIE Press; 2015. Chapter 3: X-ray Production and
#    Interactions. doi:10.1117/3.2197756
#
# 2. Buzug TM. "Computed Tomography: From Photon Statistics to Modern Cone-Beam
#    CT." Springer; 2008. Section 5.3: Polychromatic Effects and Beam Hardening.
#    doi:10.1007/978-3-540-39408-2
#
# 3. De Man B, Nuyts J, Dupont P, Marchal G, Suetens P. "An iterative maximum-
#    likelihood polychromatic algorithm for CT." IEEE Trans Med Imaging.
#    2001;20(10):999-1008. doi:10.1109/42.959297
#
# 4. Joseph PM, Spital RD. "A method for correcting bone induced artifacts in
#    computed tomography scanners." J Comput Assist Tomogr. 1978;2(1):100-108.
#    doi:10.1097/00004728-197801000-00017
#
# 5. Hubbell JH, Seltzer SM. "Tables of X-Ray Mass Attenuation Coefficients and
#    Mass Energy-Absorption Coefficients." NIST Standard Reference Database 126.
#    https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
#
# Uses AcceleratedKernels.jl for GPU/CPU acceleration.
#
# =============================================================================

import AcceleratedKernels as AK

# No public exports — the polychromatic forward model is driven by the
# api/driver simulate! pipeline, which calls `_forward_project_poly!`
# and `_apply_physics_no_noise!` directly.  `create_μ_volume!` is
# consumed by detector/photon_counting.jl + workspace ctors.

# =============================================================================
# Create μ Volume from Material Mask
# =============================================================================

"""
    create_μ_volume!(μ_volume, mask, materials, energy_keV) -> μ_volume

Create a 3D attenuation coefficient volume for a single energy from a material
mask. This is a helper function for polychromatic projection that maps material
indices to energy-specific attenuation coefficients.

# Algorithm

For each voxel in the volume:
1. Read material index m from mask (0-indexed UInt8)
2. Look up material composition: composition = materials[m + 1]
3. Compute attenuation: μᵢⱼₖ = μ(composition, E) using XrayAttenuation.jl

The attenuation lookup uses NIST XCOM cross-section data interpolated to the
specified energy via `compute_μ_at_energy()`.

# Arguments

- `μ_volume::AbstractArray{T,3}`: Output attenuation volume of size `[nx, ny, nz]`,
  modified in place. After execution, contains linear attenuation coefficients
  μ in mm⁻¹ for each voxel at the specified energy.

- `mask::AbstractArray{UInt8,3}`: Input material index volume of size `[nx, ny, nz]`.
  Values are 0-indexed region indices (0 = region 1, 1 = region 2, etc.).
  Typically created by phantom generators with material segmentation.

- `materials::Vector`: Vector of material definitions indexed by region.
  Each material should be compatible with `compute_μ_at_energy(material, E)`.
  Typically from `get_region_materials()` which returns Gammex 472 or custom
  tissue compositions.

- `energy_keV::Real`: X-ray photon energy in keV for attenuation lookup.
  Valid range: typically 10-150 keV for diagnostic CT.

# Returns

- `μ_volume::AbstractArray{T,3}`: The modified input array (same reference)

# GPU Compatibility

Uses `AK.foreachindex` for parallel execution on any backend:

| Array Type | Backend | Notes |
|------------|---------|-------|
| `Array` | CPU | Multi-threaded via Julia |
| `MtlArray` | Metal | Apple Silicon GPU |
| `CuArray` | CUDA | NVIDIA GPU |
| `ROCArray` | ROCm | AMD GPU |

# Example

```julia
using BasisSimulator

# Create a simple 2-material phantom mask
mask = zeros(UInt8, 128, 128, 32)
mask[40:90, 40:90, :] .= 1  # Water region
# Region 0 = air (background), Region 1 = water

# Get material definitions
materials = get_region_materials()  # Returns [air, water, ...]

# Create μ-volume at 60 keV
μ_volume = zeros(Float32, 128, 128, 32)
create_μ_volume!(μ_volume, mask, materials, 60.0)

# Result: μ_volume contains ~0.0 for air, ~0.02 mm⁻¹ for water at 60 keV
println("Water μ at 60 keV: ", μ_volume[65, 65, 16], " mm⁻¹")
```

# Notes

- The lookup table is computed on CPU then transferred to GPU for efficiency
- UInt8 mask values are converted to 1-based Julia indices internally
- Attenuation values are linearly interpolated from NIST XCOM data
- For very large volumes, this is the memory bottleneck (one full μ-volume)

# See Also

- [`forward_project!`](@ref): High-level projection interface
- [`get_region_materials`](@ref): Standard phantom material definitions
- [`compute_μ_at_energy`](@ref): Single-material attenuation lookup
"""
function create_μ_volume!(
        μ_volume::AbstractArray{T, 3},
        mask::AbstractArray{<:Unsigned, 3},
        materials::Vector,
        energy_keV::Real;
        ws_μ_lut_cpu::Union{Nothing, Vector{T}} = nothing,
        ws_μ_lut_gpu = nothing,
        ws_μ_table = nothing,   # pre-computed μ[region, energy] matrix
        energy_idx::Int = 0,     # index into μ_table columns
        ws_μ_table_gpu = nothing # GPU copy of μ_table for zero-copy fast path
    ) where {T <: AbstractFloat}

    # Fast path: μ_table already on GPU — skip CPU→GPU copy per energy bin
    if ws_μ_table_gpu !== nothing && energy_idx > 0
        let tbl = ws_μ_table_gpu, e = energy_idx
            AK.foreachindex(mask) do idx
                region_idx = Int(mask[idx]) + 1
                μ_volume[idx] = tbl[region_idx, e]
            end
        end
        return μ_volume
    end

    # Pre-compute μ for all regions at this energy (on CPU)
    n_regions = length(materials)
    μ_at_energy_cpu = ws_μ_lut_cpu !== nothing ? ws_μ_lut_cpu : Vector{T}(undef, n_regions)
    if ws_μ_table !== nothing && energy_idx > 0
        # Use pre-computed table (zero-alloc path)
        for i in 1:n_regions
            μ_at_energy_cpu[i] = ws_μ_table[i, energy_idx]
        end
    else
        for i in 1:n_regions
            μ_at_energy_cpu[i] = T(compute_μ_at_energy(materials[i], Float64(energy_keV)))
        end
    end

    # Transfer lookup table to same device as mask (GPU or CPU)
    μ_at_energy = ws_μ_lut_gpu !== nothing ? ws_μ_lut_gpu : similar(mask, T, n_regions)
    copyto!(μ_at_energy, μ_at_energy_cpu)

    # Use AcceleratedKernels.jl for parallel execution
    AK.foreachindex(mask) do idx
        region_idx = Int(mask[idx]) + 1  # Convert 0-based region to 1-based array index
        μ_volume[idx] = μ_at_energy[region_idx]
    end

    return μ_volume
end


"""
Apply physics effects except noise (used by simulate! signal chain).
"""
function _apply_physics_no_noise!(
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry,
        config::PhysicsConfig;
        # Workspace kwargs for zero-allocation path
        ws_output = nothing,             # sinogram-sized scratch (shared by convolution effects)
        ws_scatter_kernel = nothing,     # pre-computed scatter kernel (GPU)
        ws_optical_crosstalk_kernel = nothing, # pre-computed optical crosstalk 3x3 kernel (GPU)
        ws_focal_spot_kernel = nothing,  # pre-computed focal spot kernel (GPU)
        ws_lag_output = nothing,         # sinogram-sized scratch for lag (needs separate from ws_output)
        ws_lag_intensity = nothing,      # sinogram-sized intensity scratch for lag
        ws_lag_coeffs = nothing,         # pre-computed lag coefficients (GPU)
        ws_scatter_temp = nothing,       # sinogram-sized scratch for separable scatter (SPEED-BUILD-002)
        ws_scatter_kernel_1d = nothing   # 1D Gaussian scatter kernel (separable path)
    ) where {T}

    # Apply deterministic physics effects only
    # Skip noise since DAS model handles it

    # Fill factor
    if config.fill_factor !== nothing
        apply_fill_factor!(sinogram, config.fill_factor)
    end

    # NOTE: Scatter injection and correction are now handled in simulate!() using
    # the unified per-energy scatter model (compute_scatter_energy_weights +
    # inject_scatter!/inject_scatter_bins!). Scatter is added BEFORE noise and
    # corrected AFTER noise, matching the clinical scanner signal chain.
    # See: driver.jl simulate!(ws::EICTWorkspace) and simulate!(ws::PCCTWorkspace)

    # Optical crosstalk
    if config.optical_crosstalk !== nothing
        apply_optical_crosstalk!(
            sinogram, config.optical_crosstalk;
            ws_output = ws_output, ws_kernel = ws_optical_crosstalk_kernel
        )
    end

    # Focal spot blur
    if config.focal_spot !== nothing
        apply_focal_spot_blur!(
            sinogram, config.focal_spot, geom;
            ws_output = ws_output, ws_kernel = ws_focal_spot_kernel
        )
    end

    # Detector efficiency η(E) is now applied in the spectral sum inside
    # _forward_project_poly! (ws_η kwarg) and in the noise I₀ scaling,
    # so no separate sinogram-domain application is needed here.

    # NOTE: Skip noise - handled by DAS model in signal chain

    # Detector lag
    if config.lag !== nothing
        apply_lag!(
            sinogram, config.lag;
            ws_output = ws_lag_output, ws_intensity = ws_lag_intensity, ws_coeffs = ws_lag_coeffs
        )
    end

    return sinogram
end


"""
    _forward_project_poly!(sinogram, mask, geom, energies, weights, materials) -> sinogram

Internal implementation of polychromatic forward projection using Beer-Lambert
physics with energy-sequential integration.

# Algorithm

Implements the polychromatic Beer-Lambert formula:

    I_total = Σₑ wₑ × exp(-Lₑ)

where:
- wₑ = normalized weight for energy bin e
- Lₑ = Siddon line integral through μ(E_e) volume

The algorithm proceeds energy-by-energy for memory efficiency:

```
I_transmitted = 0
for e in 1:N_energies:
    μ_volume = create_μ_volume(mask, materials, E_e)  # Energy-specific μ
    L_e = siddon_project(μ_volume)                     # Ray trace
    I_transmitted += w_e × exp(-L_e)                   # Accumulate intensity
end
sinogram = -log(max(I_transmitted, ε))                 # Convert to projection
```

# Mathematical Formulation

For a polychromatic beam passing through heterogeneous material:

    I/I₀ = ∫ w(E) × exp(-∫ μ(E, l) dl) dE

Discretized with N energy bins:

    I/I₀ = Σₑ₌₁ᴺ wₑ × exp(-Σᵢ μᵢ(Eₑ) × Δlᵢ)

The projection (negative log-intensity) is:

    p = -log(Σₑ wₑ × exp(-Lₑ))

Note: This is NOT equal to Σₑ wₑ × Lₑ (weighted average of monochromatic
projections) due to the nonlinearity of log-sum-exp. This nonlinearity
causes beam hardening artifacts.

# Arguments

- `sinogram::AbstractArray{T,3}`: Output array [n_cols, n_rows, n_angles], modified
  in place with line integral values.

- `mask::AbstractArray{UInt8,3}`: Material index volume [nx, ny, nz]. Values 0-255
  index into materials array (0-indexed, converted internally).

- `geom::CTGeometry`: Scanner geometry with source/detector positions and FOV.

- `energies::Vector`: Energy bin centers in keV. Length N_energies.
  Example: [20.0, 30.0, ..., 120.0] for 10 keV bins.

- `weights::Vector`: Photon fluence weights per energy bin. Will be normalized
  internally to sum to 1.0. Typically from X-ray tube spectrum.

- `materials::Vector`: Material definitions for each region index. Must include
  all indices present in mask. From `get_region_materials()`.

# Returns

- `sinogram::AbstractArray{T,3}`: The modified sinogram (same reference)

# Memory Usage

- One temporary μ-volume: O(nx × ny × nz) elements
- One temporary monochromatic sinogram: O(n_cols × n_rows × n_angles)
- One intensity accumulator: O(n_cols × n_rows × n_angles)
- Total: ~3× sinogram size additional memory, independent of N_energies

# Notes

- Weights are normalized internally (sum to 1.0)
- Epsilon clamping (1e-10) prevents log(0) errors
- GPU execution via AcceleratedKernels.jl when arrays are GPU arrays

# See Also

- [`forward_project!`](@ref): High-level interface
- [`create_μ_volume!`](@ref): Energy-specific μ volume creation
- [`siddon_forward_project!`](@ref): Monochromatic ray tracing
"""
function _forward_project_poly!(
        sinogram::AbstractArray{T, 3},
        mask::AbstractArray{<:Unsigned, 3},
        geom::CTGeometry,
        energies::Vector,
        weights::Vector,
        materials::Vector;
        # Workspace kwargs for zero-allocation path
        ws_μ_volume = nothing,
        ws_sino_mono = nothing,
        ws_I_transmitted = nothing,
        ws_weights_norm::Union{Nothing, Vector{T}} = nothing,
        ws_μ_lut_cpu::Union{Nothing, Vector{T}} = nothing,
        ws_μ_lut_gpu = nothing,
        ws_μ_table = nothing,
        ws_μ_table_gpu = nothing,
        # Siddon geometry arrays (avoid re-allocating per energy)
        ws_source_positions = nothing,
        ws_detector_centers = nothing,
        ws_detector_u = nothing,
        ws_detector_v = nothing,
        # Override volume bounds for phantom FOV
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        # Detector efficiency η(E) per energy bin (weights spectral sum)
        ws_η::Union{Nothing, Vector{Float64}} = nothing,
        # Bowtie spectral transmission [n_cols, n_rows, n_energies]
        ws_bowtie_spectral = nothing,
        # Fused kernel: pre-computed wη on GPU [n_energies], enables single-pass projection
        ws_wη_gpu = nothing,
        # Control flag: true = fused single-pass kernel, false = sequential energy loop
        # Default false: 234-bin fused kernel causes massive register spilling on GPU (3.5× slower).
        # Tiled fusion (K=16) will replace this — see SPEED-BUILD-V2-002.
        fused::Bool = false,
        # Ray tracer: :dd_fast (default/fastest general path), :dd (DEPRECATED), or :siddon (compatibility, aliases).
        projector::Symbol = :dd_fast
    ) where {T <: AbstractFloat}

    n_energies = length(energies)

    # =========================================================================
    # FUSED PATH: single AK.foreachindex kernel, traces mask ONCE
    # =========================================================================
    if fused && ws_μ_table_gpu !== nothing
        @info "FUSED PATH: n_energies=$n_energies, mask=$(size(mask)), sino=$(size(sinogram))" maxlog = 1
        # Build wη on GPU if not provided via workspace
        wη_dev = if ws_wη_gpu !== nothing
            ws_wη_gpu
        else
            weights_norm = ws_weights_norm !== nothing ? ws_weights_norm : T.(weights ./ sum(weights))
            η_vec = ws_η !== nothing ? ws_η : ones(Float64, n_energies)
            wη_cpu = T.(weights_norm .* η_vec)
            _buf = similar(sinogram, T, n_energies)
            copyto!(_buf, wη_cpu)
            _buf
        end

        _project_fused_poly!(
            projector,
            sinogram, mask, geom, ws_μ_table_gpu, wη_dev, Val(n_energies);
            volume_extent = volume_extent,
            ws_source_positions = ws_source_positions,
            ws_detector_centers = ws_detector_centers,
            ws_detector_u = ws_detector_u,
            ws_detector_v = ws_detector_v,
            ws_bowtie_spectral = ws_bowtie_spectral
        )

        return sinogram
    end

    # =========================================================================
    # TILED PATH (SPEED-BUILD-V2-002): K=16 per tile, 15 tiles for 234 bins
    # Default when μ_table_gpu + wη_gpu available. 3.66× faster than unfused.
    # =========================================================================
    if ws_μ_table_gpu !== nothing && ws_wη_gpu !== nothing
        # ---------------------------------------------------------------------
        # SINGLE-PASS PATH (:dd_fast, ≤32 materials): the path-length DD kernel
        # accumulates per-MATERIAL path lengths (energy-independent registers),
        # so all n_energies convert once per cell — ONE volume walk instead of
        # n_tiles.  Identical DD footprint/weights; float ordering only.
        # :dd and :siddon keep the tiled path below (their kernels hold
        # per-energy registers → K=16 required).
        # ---------------------------------------------------------------------
        if projector === :dd_fast && size(ws_μ_table_gpu, 1) <= _PLEN_MAX_MATERIALS
            @info "SINGLE-PASS PATH (:dd_fast path-length): n_energies=$n_energies, mask=$(size(mask)), sino=$(size(sinogram))" maxlog = 1
            dd_fast_fused_poly_project!(
                sinogram, mask, geom,
                ws_μ_table_gpu, ws_wη_gpu, Val(n_energies);
                volume_extent = volume_extent,
                ws_source_positions = ws_source_positions,
                ws_detector_centers = ws_detector_centers,
                ws_detector_u = ws_detector_u,
                ws_detector_v = ws_detector_v,
                ws_bowtie_spectral = ws_bowtie_spectral
            )
            return sinogram
        end

        @info "TILED PATH: K=16, n_energies=$n_energies, mask=$(size(mask)), sino=$(size(sinogram))" maxlog = 1

        # Tile parameters — fused poly projector (Val(16), `projector`-selected)
        # + subset copies per tile. Avoids runtime offset arithmetic that causes
        # 71% GPU slowdown vs compile-time constant indices.
        TILE_K = 16
        n_energies_padded = cld(n_energies, TILE_K) * TILE_K
        n_tiles = n_energies_padded ÷ TILE_K

        # Use I_transmitted buffer for accumulation
        I_transmitted = ws_I_transmitted !== nothing ? ws_I_transmitted : similar(sinogram)
        fill!(I_transmitted, zero(T))

        # Subset buffers (small allocations: μ=1.7KB, wη=64B, bt=512KB)
        n_regions = size(ws_μ_table_gpu, 1)
        μ_sub = similar(sinogram, T, n_regions, TILE_K)
        wη_sub = similar(sinogram, T, TILE_K)
        has_bt = ws_bowtie_spectral !== nothing
        bt_sub = has_bt ? similar(sinogram, T, size(sinogram, 1), size(sinogram, 2), TILE_K) : nothing

        for tile_idx in 1:n_tiles
            ts = (tile_idx - 1) * TILE_K + 1
            te = ts + TILE_K - 1

            # Copy subset of μ_table and wη for this tile
            copyto!(μ_sub, @view ws_μ_table_gpu[:, ts:te])
            copyto!(wη_sub, @view ws_wη_gpu[ts:te])
            if has_bt
                copyto!(bt_sub, @view ws_bowtie_spectral[:, :, ts:te])
            end

            # Use proven-fast fused kernel with Val(16) — 95ms/tile on Metal
            _project_fused_poly!(
                projector,
                sinogram, mask, geom,
                μ_sub, wη_sub, Val(TILE_K);
                volume_extent = volume_extent,
                ws_source_positions = ws_source_positions,
                ws_detector_centers = ws_detector_centers,
                ws_detector_u = ws_detector_u,
                ws_detector_v = ws_detector_v,
                ws_bowtie_spectral = bt_sub
            )

            # Accumulate: undo -log, add partial Beer-Lambert sum
            let I_trans = I_transmitted
                AK.foreachindex(sinogram) do idx
                    I_trans[idx] += exp(-sinogram[idx])
                end
            end
        end

        # Final -log
        let I_trans = I_transmitted, eps_val = T(1.0e-10)
            AK.foreachindex(sinogram) do idx
                sinogram[idx] = -log(max(I_trans[idx], eps_val))
            end
        end

        return sinogram
    end

    # =========================================================================
    # UNFUSED PATH: sequential energy loop (fallback when no μ_table_gpu)
    # =========================================================================

    # Normalize weights (use pre-computed if available)
    weights_norm = ws_weights_norm !== nothing ? ws_weights_norm : T.(weights ./ sum(weights))

    # Use workspace buffers or allocate
    μ_volume = ws_μ_volume !== nothing ? ws_μ_volume : similar(sinogram, T, size(mask))
    sino_mono = ws_sino_mono !== nothing ? ws_sino_mono : similar(sinogram)
    I_transmitted = ws_I_transmitted !== nothing ? ws_I_transmitted : similar(sinogram)

    # Initialize transmitted intensity
    fill!(I_transmitted, zero(T))

    # Loop over energies (memory-efficient approach)
    for e_idx in 1:n_energies
        # Create μ volume for this energy
        create_μ_volume!(
            μ_volume, mask, materials, energies[e_idx];
            ws_μ_lut_cpu = ws_μ_lut_cpu, ws_μ_lut_gpu = ws_μ_lut_gpu,
            ws_μ_table = ws_μ_table, energy_idx = e_idx,
            ws_μ_table_gpu = ws_μ_table_gpu
        )

        # Forward project at this energy
        fill!(sino_mono, zero(T))
        _project_mono!(
            projector,
            sino_mono, μ_volume, geom;
            ws_source_positions = ws_source_positions,
            ws_detector_centers = ws_detector_centers,
            ws_detector_u = ws_detector_u,
            ws_detector_v = ws_detector_v,
            volume_extent = volume_extent
        )

        # Accumulate Beer-Lambert: I += w × η(E) × T_bt(E,col,row) × exp(-line_integral)
        w = weights_norm[e_idx]
        # η(E) reshapes effective spectrum — accounts for detector absorption + fluorescence escape
        η_e = ws_η !== nothing ? T(ws_η[e_idx]) : one(T)

        if ws_bowtie_spectral !== nothing
            # Per-pixel bowtie transmission from [n_cols, n_rows, n_energies] array
            nc = size(sinogram, 1)
            nr = size(sinogram, 2)
            nc_nr = nc * nr
            bt_offset = (e_idx - 1) * nc_nr
            let w = w, η_e = η_e, bt = ws_bowtie_spectral, bt_off = bt_offset, nc = nc, nr = nr, I_trans = I_transmitted, sm = sino_mono
                AK.foreachindex(I_trans) do idx
                    idx_0 = Int32(idx - 1)
                    col = (idx_0 % Int32(nc)) + Int32(1)
                    row = ((idx_0 ÷ Int32(nc)) % Int32(nr)) + Int32(1)
                    bt_val = bt[col + (row - 1) * nc + bt_off]
                    I_trans[idx] += w * η_e * bt_val * exp(-sm[idx])
                end
            end
        else
            # Scalar-weight path (no bowtie)
            let w_eff = w * η_e, I_trans = I_transmitted, sm = sino_mono
                AK.foreachindex(I_trans) do idx
                    I_trans[idx] += w_eff * exp(-sm[idx])
                end
            end
        end
    end

    # Convert back to line integral: sinogram = -log(I / I₀)
    let I_trans = I_transmitted, eps_val = T(1.0e-10)
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = -log(max(I_trans[idx], eps_val))
        end
    end

    return sinogram
end
