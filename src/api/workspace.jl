# =============================================================================
# PCCTWorkspace — Pre-allocated workspace for zero-allocation simulate!()
# =============================================================================
#
# All buffers needed by the PCCT simulation pipeline are allocated once here.
# simulate!() writes into these buffers with zero allocations on repeated calls.
#
# GPU-Agnostic: Uses similar(phantom.mask, T, shape) for backend detection.
# Never references Metal.jl, CUDA.jl, or any specific GPU backend.

export PCCTWorkspace, create_workspace
export EICTWorkspace, create_eict_workspace
export FDKReconWorkspace, create_fdk_recon_workspace
export HIRReconWorkspace, create_hir_recon_workspace

"""
    PCCTWorkspace{T, A3, A1}

Pre-allocated workspace for zero-allocation PCCT simulation.

Type parameters:
- `T`: Element type (typically Float32)
- `A3`: 3D array type matching GPU backend (e.g., Array{T,3} for CPU)
- `A1`: 1D array type matching GPU backend (e.g., Vector{T} for CPU)

All GPU-side buffers match the backend of the phantom mask passed to
`create_workspace`. CPU-side buffers are always plain `Array`.

Pre-computed setup data (geometry, spectrum, physics config, detector) is
computed once in `create_workspace()` and stored here for zero-alloc reuse.
"""
mutable struct PCCTWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A1<:AbstractArray{T,1}, A2<:AbstractArray{T,2}}
    # ─── Forward projection (GPU-side) ───
    bins::Vector{A3}           # n_bins sinogram buffers (output of forward projection)
    μ_volume::A3               # attenuation volume, reused per energy
    sino_buf::A3               # forward projection scratch, reused per energy

    # ─── Spatial kernel scratch (GPU-side) ───
    scratch::A3                # ONE buffer for spatial kernels

    # ─── Combine (GPU-side) ───
    combined::A3               # _combine_pcct_bins output (reused ideal + noisy)

    # ─── Noise CPU staging (Phase 1: CPU RNG) ───
    noise_staging::Array{T,3}  # CPU buffer for GPU↔CPU noise transfer
    noise_buf::Array{T,3}      # randn output buffer (CPU)

    # ─── Electronic noise ───
    enoise_cpu::Vector{T}      # randn output for electronic noise (CPU, flat)
    enoise_gpu::A1             # electronic noise GPU transfer buffer (flat)

    # ─── Result staging (CPU) ───
    sino_ideal_out::Array{T,3}  # final ideal sinogram for return
    sino_noisy_out::Array{T,3}  # final noisy sinogram for return

    # ─── Pre-computed CPU vectors/matrices ───
    η::Vector{Float64}          # quantum efficiency vector (n_energies)
    R::Matrix{Float64}          # spectral response matrix (n_energies × n_bins)
    R_energies::Vector{Float64} # energy grid for R matrix
    I0_bins::Vector{Float64}    # per-bin I0 values for combine (n_bins)
    I0_bins_norm::Vector{Float64} # per-bin I0 values for fwd proj normalization (n_bins)
    thresholds_T::Vector{T}     # T-typed thresholds (n_bins)

    # ─── RNG state ───
    rng::MersenneTwister        # pre-allocated RNG (reset with seed each call)

    # ─── Detector physics precomputed (small buffers) ───
    μ_values::Vector{T}                       # VMI attenuation coefficients (n_materials)

    # ─── Noise I0 ───
    noise_I0::Vector{Float64}   # per-bin I0 for noise model (n_bins)

    # ─── μ lookup table (pre-computed for all energies × materials) ───
    μ_lut_cpu::Vector{T}                     # μ LUT CPU buffer (n_regions)
    μ_lut_gpu::A1                            # μ LUT GPU buffer (n_regions, matches mask backend)
    μ_table::Matrix{T}                       # pre-computed μ[region, energy] (n_regions × n_energies)

    # ─── BHC coefficients (pre-allocated, small) ───
    bhc_coeffs_cpu::Vector{T}                # BHC polynomial coefficients (CPU)
    bhc_coeffs_gpu::A1                       # BHC polynomial coefficients (GPU/backend)

    # ─── Pre-computed geometry arrays (T-typed, same backend as mask) ───
    geom_source_positions::A2                # (3, n_angles) source positions
    geom_detector_centers::A2                # (3, n_angles) detector centers
    geom_detector_u::A2                      # (3, n_angles) detector u vectors
    geom_detector_v::A2                      # (3, n_angles) detector v vectors

    # ─── Native-resolution buffers (for spatial binning path) ───
    native_bins::Union{Nothing, Vector{A3}}  # per-bin at native dexel res (nothing if bf==1)
    native_sino_buf::Union{Nothing, A3}      # Siddon scratch at native res
    native_scratch::Union{Nothing, A3}       # scratch at native res
    native_geom::Union{Nothing, CTGeometry}  # native-res geometry (nothing if bf==1)
    native_geom_source_positions::Union{Nothing, A2}
    native_geom_detector_centers::Union{Nothing, A2}
    native_geom_detector_u::Union{Nothing, A2}
    native_geom_detector_v::Union{Nothing, A2}

    # ─── Tube physics scratch (for combined sinogram) ───
    tube_physics_scratch::Union{Nothing, A3}  # sinogram-sized scratch for scatter/focal spot

    # ─── Tiled spectral projection (fused PCCT forward projection) ───
    μ_table_gpu::A2                          # GPU copy of μ_table [n_regions, n_energies_padded]
    W_matrix_gpu::A2                         # spectral weight matrix [n_energies_padded, n_bins] on GPU
    outputs_flat::A1                         # flattened output buffer [n_elements * n_bins] on GPU
    native_outputs_flat::Union{Nothing, A1}  # native-res flattened output (nothing if bf==1)
    source_spectral_gpu::Union{Nothing, A3}  # heel × bowtie [n_cols, n_rows, n_energies_padded]

    # ─── Pre-computed pileup LUT ───
    pileup_S::Matrix{Float64}               # spectral migration matrix S[n_bins, n_bins]
    pileup_count_factor::Float64            # count-loss factor (N_recorded / N_true)

    # ─── Pre-computed scatter bin fractions (energy-resolved scatter) ───
    scatter_bin_fractions::Vector{Float64}   # fractional scatter per bin (sums to 1.0)

    # ─── Pre-computed setup data (computed once, reused) ───
    geom::CTGeometry                         # CT geometry (binned resolution)
    energies::Vector{Float64}                # downsampled spectrum energies
    weights::Vector{Float64}                 # downsampled spectrum weights
    config::PhysicsConfig                    # physics configuration
    pcct_detector::PhotonCountingDetector{Float64}  # PCCT detector
    mats::Vector{XA.Material}                # resolved materials
    use_detector_fx::Bool                    # whether detector effects are applied
    use_corrections::Bool                    # whether corrections are applied
    kVp::Float64                             # max energy (kVp)
end

"""
    create_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T=Float32)

Create a pre-allocated workspace for zero-allocation `simulate!()` calls.

Uses `similar(phantom.mask, T, shape)` for GPU-side buffers, which auto-detects
the correct GPU backend (Metal, CUDA, ROCm, or CPU).

Pre-computes all setup data (geometry, spectrum, physics config, detector,
spectral response matrices) so that `simulate!()` has zero allocations.

# Arguments
- `scanner`: Scanner specification (provides detector geometry)
- `protocol`: CT protocol (provides number of views)
- `sim_opts`: Simulation options (provides fidelity and effect toggles)
- `recon_opts`: Reconstruction options (provides vmi_basis for n_materials)
- `phantom`: Phantom struct (provides mask for backend detection and volume shape)
- `T`: Element type, default Float32

# Returns
A `PCCTWorkspace{T, A3, A1}` with all buffers and setup data pre-computed.

# Example
```julia
ws = create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
# Second call: zero allocations
result2 = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
```
"""
function create_workspace(scanner, protocol, sim_opts, recon_opts, phantom;
                          T::Type{<:AbstractFloat}=Float32, mask=nothing,
                          materials::Union{Nothing, Vector}=nothing)
    # --- Pre-compute geometry first (collimation derives n_rows) ---
    geom = CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm, z_cm=recon_opts.z_cm, collimation_mm=protocol.collimation_mm)

    sino_shape = (geom.n_cols, geom.n_rows, geom.n_angles)
    vol_shape = size(phantom.mask)
    n_bins = length(scanner.energy_thresholds)
    n_materials = length(recon_opts.vmi_basis)
    n_elements = prod(sino_shape)
    # n_energies is set after resolve_spectrum below

    # GPU-side buffers — similar() matches the provided mask's backend
    # If mask kwarg is provided (e.g. GPU-converted mask), use it for similar();
    # otherwise fall back to phantom.mask (CPU arrays)
    ref_mask = mask === nothing ? phantom.mask : mask
    bins = [similar(ref_mask, T, sino_shape) for _ in 1:n_bins]
    μ_volume = similar(ref_mask, T, vol_shape)
    sino_buf = similar(ref_mask, T, sino_shape)
    scratch = similar(ref_mask, T, sino_shape)
    combined = similar(ref_mask, T, sino_shape)
    enoise_gpu = similar(ref_mask, T, n_elements)

    # VMI decomposition buffers (GPU-side, unified with dual-kVp)

    # CPU-side buffers
    noise_staging = zeros(T, sino_shape)
    noise_buf = zeros(T, sino_shape)
    enoise_cpu = Vector{T}(undef, n_elements)
    sino_ideal_out = zeros(T, sino_shape)
    sino_noisy_out = zeros(T, sino_shape)
    energies, weights_vec = resolve_spectrum(sim_opts, protocol; scanner=scanner)
    n_energies = length(energies)
    config = build_physics_config(scanner, sim_opts, energies, weights_vec; phantom=phantom)
    pcct_detector = _build_pcct_detector(scanner)
    mats = _resolve_materials(phantom, materials)
    use_detector_fx = sim_opts.fidelity in (:eict, :pcct)
    use_corrections = sim_opts.use_pcct_corrections
    kVp = Float64(maximum(energies))
    thresholds = pcct_detector.energy_thresholds_keV
    # Pre-compute spectral response data
    η_vec = quantum_efficiency_vector(pcct_detector.material, pcct_detector.thickness_mm, energies)
    R_mat = compute_mc_drm(pcct_detector, kVp)
    R_energies_vec = collect(range(1.0, Float64(kVp), length=size(R_mat, 1)))

    # Recalibrate BHC for PCCT using effective spectrum weights.
    # The PCCT combined sinogram uses effective weights w_eff = w × η × Σ_b R[E,b]
    # because the R matrix drops low-energy photons (row sum < 1.0), making the
    # effective spectrum harder. BHC must be calibrated to this harder spectrum.
    if config.bhc !== nothing
        # Map each spectrum energy to nearest DRM grid row to get R row sums
        n_R = size(R_mat, 1)
        R_row_sums = [begin
            r_idx = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
            sum(R_mat[r_idx, :])
        end for E in energies]
        w_eff = Float64.(weights_vec) .* Float64.(η_vec) .* R_row_sums
        ref_energy = sum(Float64.(energies) .* w_eff) / sum(w_eff)
        pcct_bhc = calibrate_bhc(energies, w_eff;
                                  order=5, reference_energy_keV=ref_energy)
        config = PhysicsConfig(
            config.fill_factor,
            config.scatter, config.scatter_correction,
            config.optical_crosstalk, config.focal_spot, config.detector_efficiency,
            config.lag, config.noise_seed, config.energy_keV,
            config.heel_effect, pcct_bhc)
    end

    # Pre-compute I0_bins for normalization (forward projection)
    I0_bins_norm_vec = if use_detector_fx && !use_corrections
        _compute_degraded_I0(pcct_detector, energies, weights_vec, η_vec, thresholds, kVp, 1e6, 1e8; R=R_mat)
    else
        [_compute_bin_I0(pcct_detector, energies, weights_vec, η_vec, thresholds, b,
                          kVp, 1e6; R=R_mat) for b in 1:n_bins]
    end

    # Pre-compute I0_bins for combine (may differ from norm if corrections change the path)
    I0_bins_combine = if use_detector_fx && !use_corrections
        copy(I0_bins_norm_vec)
    else
        [_compute_bin_I0(pcct_detector, energies, weights_vec, η_vec, thresholds, b,
                          kVp, 1e6; R=R_mat) for b in 1:n_bins]
    end

    # Pre-compute T-typed thresholds
    thresholds_T_vec = T.(thresholds)

    rng = MersenneTwister(0)

    # Detector physics precomputed buffers
    μ_values = zeros(T, n_materials)
    noise_I0 = zeros(Float64, n_bins)

    # μ lookup table — pre-compute μ for all regions × all energies
    n_regions = length(mats)
    μ_lut_cpu = Vector{T}(undef, n_regions)
    μ_lut_gpu = similar(ref_mask, T, n_regions)
    μ_table = zeros(T, n_regions, n_energies)
    for (e_idx, E) in enumerate(energies)
        for r in 1:n_regions
            μ_table[r, e_idx] = T(compute_μ_at_energy(mats[r], Float64(E)))
        end
    end

    # BHC coefficients pre-allocated
    bhc_n_coeffs = config.bhc !== nothing ? length(get_bhc_coefficients(config.bhc)) : 1
    bhc_coeffs_cpu = if config.bhc !== nothing
        T.(get_bhc_coefficients(config.bhc))
    else
        zeros(T, 1)
    end
    bhc_coeffs_gpu = similar(ref_mask, T, length(bhc_coeffs_cpu))
    copyto!(bhc_coeffs_gpu, bhc_coeffs_cpu)

    # Pre-computed geometry arrays (T-typed, same backend as mask) for siddon_forward_project!
    geom_source_positions = similar(ref_mask, T, size(geom.source_positions)...)
    copyto!(geom_source_positions, T.(geom.source_positions))
    geom_detector_centers = similar(ref_mask, T, size(geom.detector_centers)...)
    copyto!(geom_detector_centers, T.(geom.detector_centers))
    geom_detector_u = similar(ref_mask, T, size(geom.detector_u)...)
    copyto!(geom_detector_u, T.(geom.detector_u))
    geom_detector_v = similar(ref_mask, T, size(geom.detector_v)...)
    copyto!(geom_detector_v, T.(geom.detector_v))

    # Pre-compute 2×2 decomposition matrix for pseudo-dual-energy VMI
    max_keV = 120.0
    # --- Native-resolution geometry and buffers (for spatial binning path) ---
    bf = scanner.binning_factor
    if bf > 1
        magnification = scanner.source_to_detector / scanner.source_to_isocenter
        native_col_iso_mm = scanner.native_dexel_col_mm / magnification
        native_row_iso_mm = scanner.native_dexel_row_mm / magnification
        # Build native geometry directly — same source/detector positions as binned
        # but with bf× more cols and rows at native dexel pitch
        _native_n_cols = geom.n_cols * bf
        _native_n_rows = geom.n_rows * bf
        _native_pixel_size = native_col_iso_mm / 10.0  # mm → cm
        _native_pixel_row_size = native_row_iso_mm / 10.0  # mm → cm
        _native_geom = CTGeometry(
            geom.SAD, geom.SDD, geom.n_angles, _native_n_rows, _native_n_cols,
            _native_pixel_size, _native_pixel_row_size,
            geom.angles,
            geom.source_positions, geom.detector_centers,
            geom.detector_u, geom.detector_v,
            geom.fov  # same recon FOV
        )
        native_sino_shape = (_native_geom.n_cols, _native_geom.n_rows, _native_geom.n_angles)
        _native_bins = [similar(ref_mask, T, native_sino_shape) for _ in 1:n_bins]
        _native_sino_buf = similar(ref_mask, T, native_sino_shape)
        _native_scratch = similar(ref_mask, T, native_sino_shape)
        # Pre-computed geometry arrays at native resolution
        _n_src = similar(ref_mask, T, size(_native_geom.source_positions)...)
        copyto!(_n_src, T.(_native_geom.source_positions))
        _n_det = similar(ref_mask, T, size(_native_geom.detector_centers)...)
        copyto!(_n_det, T.(_native_geom.detector_centers))
        _n_u = similar(ref_mask, T, size(_native_geom.detector_u)...)
        copyto!(_n_u, T.(_native_geom.detector_u))
        _n_v = similar(ref_mask, T, size(_native_geom.detector_v)...)
        copyto!(_n_v, T.(_native_geom.detector_v))
    else
        _native_geom = nothing
        _native_bins = nothing
        _native_sino_buf = nothing
        _native_scratch = nothing
        _n_src = nothing
        _n_det = nothing
        _n_u = nothing
        _n_v = nothing
    end

    # Tube physics scratch buffer (binned resolution, for scatter/focal spot on combined sinogram)
    tube_scratch = similar(ref_mask, T, sino_shape)

    # ─── Tiled spectral projection buffers (fused PCCT forward projection) ───
    # Pad μ_table to multiple of 16 for tiled projection (same as EICT path)
    TILE_K = 16
    n_energies_padded = cld(n_energies, TILE_K) * TILE_K
    _μ_table_gpu = similar(ref_mask, T, n_regions, n_energies_padded)
    fill!(_μ_table_gpu, zero(T))
    copyto!(view(_μ_table_gpu, :, 1:n_energies), μ_table)

    # Build W matrix: W[e, b] = I0 * w[e] * η[e] * R[e, b]
    # Uses I0=1e6 consistent with pcct_forward_project default
    _I0 = 1e6
    n_R = size(R_mat, 1)
    W_cpu = zeros(T, n_energies_padded, n_bins)
    for e_idx in 1:n_energies
        E_float = Float64(energies[e_idx])
        w = Float64(weights_vec[e_idx])
        if w < 1e-12
            continue
        end
        r_idx = clamp(round(Int, (E_float - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        for b in 1:n_bins
            W_cpu[e_idx, b] = T(_I0 * w * η_vec[e_idx] * R_mat[r_idx, b])
        end
    end
    _W_matrix_gpu = similar(ref_mask, T, n_energies_padded, n_bins)
    copyto!(_W_matrix_gpu, W_cpu)

    # Source spectral: fold center-pixel bowtie into W matrix
    # The bowtie's dominant effect is spectral hardening (energy-dependent), which is
    # nearly uniform across the detector. The per-pixel spatial variation (~5-10%) is
    # secondary. By folding center-pixel bowtie into W, we capture the hardening
    # without the GPU Float32 precision issues of the per-pixel spectral path.
    bowtie_filter_pcct = resolve_bowtie_filter(scanner.bowtie_filter)
    if bowtie_filter_pcct !== nothing && bowtie_filter_pcct.name != "none"
        bt_cpu = compute_bowtie_attenuation_spectral(bowtie_filter_pcct, geom, Float64.(energies))
        center_col = sino_shape[1] ÷ 2
        center_row = sino_shape[2] ÷ 2
        for e_idx in 1:n_energies
            bt_center = Float64(bt_cpu[center_col, center_row, e_idx])
            for b in 1:n_bins
                W_cpu[e_idx, b] *= T(bt_center)
            end
        end
        copyto!(_W_matrix_gpu, W_cpu)
    end
    _source_spectral_gpu = nothing  # Not needed — bowtie folded into W

    # Flattened output buffer for spectral projection
    _outputs_flat = similar(ref_mask, T, n_elements * n_bins)

    # Native-resolution flattened output (for spatial binning path)
    _native_outputs_flat = if bf > 1
        native_n_elements = prod(native_sino_shape)
        similar(ref_mask, T, native_n_elements * n_bins)
    else
        nothing
    end

    # ─── Pre-compute pileup migration matrix (MC-based, one-time cost) ───
    # Pileup is per native dexel (not per binned pixel).
    # I0 from compute_detector_I0 is per binned pixel per view.
    # Count rate per dexel = (I0 / bf²) / time_per_view  [photons/s]
    _I0_physics = compute_detector_I0(geom, protocol, sum(weights_vec))
    _time_per_view = protocol.rotation_time / protocol.views
    _count_rate_per_dexel = (_I0_physics / Float64(bf * bf)) / _time_per_view
    _τ_ns = Float64(pcct_detector.dead_time_ns)
    _aτ = _count_rate_per_dexel * _τ_ns * 1e-9

    _pileup_S = if pcct_detector.dead_time_ns > 0 && _aτ > 0.01
        w_norm = Float64.(weights_vec) ./ sum(Float64.(weights_vec))
        compute_mc_pileup_matrix(
            pcct_detector.energy_thresholds_keV,
            w_norm, Float64.(energies),
            _count_rate_per_dexel, _τ_ns;
            n_trials=5000, seed=42)
    else
        Matrix{Float64}(I, n_bins, n_bins)  # Identity = no pileup
    end
    _pileup_cf = seminonparalyzable_count_factor(_aτ)

    # Pre-compute scatter bin fractions (how scatter distributes across PCCT bins)
    _scatter_bin_fracs = if config.scatter !== nothing
        compute_scatter_bin_fractions(Float64.(energies), Float64.(weights_vec),
                                      Float64.(η_vec), R_mat, kVp)
    else
        fill(1.0 / n_bins, n_bins)
    end

    return PCCTWorkspace{T, typeof(sino_buf), typeof(enoise_gpu), typeof(geom_source_positions)}(
        bins, μ_volume, sino_buf, scratch,
        combined,
        noise_staging, noise_buf, enoise_cpu, enoise_gpu,
        sino_ideal_out, sino_noisy_out,
        η_vec, R_mat, R_energies_vec, I0_bins_combine, I0_bins_norm_vec, thresholds_T_vec, rng,
        μ_values, noise_I0,
        μ_lut_cpu, μ_lut_gpu, μ_table,
        bhc_coeffs_cpu, bhc_coeffs_gpu,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        _native_bins, _native_sino_buf, _native_scratch,
        _native_geom, _n_src, _n_det, _n_u, _n_v,
        tube_scratch,
        _μ_table_gpu, _W_matrix_gpu, _outputs_flat, _native_outputs_flat, _source_spectral_gpu,
        _pileup_S, _pileup_cf,
        _scatter_bin_fracs,
        geom, energies, weights_vec, config, pcct_detector, mats,
        use_detector_fx, use_corrections, kVp
    )
end

# =============================================================================
# EICTWorkspace — Pre-allocated workspace for zero-allocation EICT simulate!()
# =============================================================================
#
# Energy-Integrating CT (single-kVp). Much simpler than PCCT — ~16 fields.
# Buffers for polychromatic forward projection + quantum noise.

"""
    EICTWorkspace{T, A3, A1}

Pre-allocated workspace for zero-allocation EICT single-kVp simulation.

Type parameters:
- `T`: Element type (typically Float32)
- `A3`: 3D array type matching GPU backend
- `A1`: 1D array type matching GPU backend
"""
mutable struct EICTWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A2<:AbstractArray{T,2}, A1<:AbstractArray{T,1}}
    # ─── Forward projection (GPU-side) ───
    sinogram::A3          # output sinogram (n_cols, n_rows, n_angles)
    μ_volume::A3          # attenuation volume, reused per energy (nx, ny, nz)
    sino_mono::A3         # monochromatic sinogram scratch (sino shape)
    I_transmitted::A3     # Beer-Lambert accumulator (sino shape)

    # ─── Signal chain scratch (GPU-side) ───
    air_scan::A3          # CatSim air scan buffer (sino shape)

    # ─── Physics effect scratch buffers (GPU-side, shared) ───
    physics_output::A3    # sinogram-sized scratch for convolution effects (shared)
    lag_intensity::A3     # sinogram-sized scratch for lag intensity computation

    # ─── Pre-computed physics kernels (GPU-side) ───
    scatter_kernel::Union{Nothing, A2}          # scatter convolution kernel (2D fallback)
    scatter_correct_kernel::Union{Nothing, A2}  # scatter correction kernel (2D fallback)
    scatter_kernel_1d::Union{Nothing, A1}       # 1D Gaussian scatter kernel (separable path)
    scatter_correct_kernel_1d::Union{Nothing, A1} # 1D correction kernel (separable path)
    scatter_temp::A3                            # sinogram-shaped scratch for separable convolution
    optical_crosstalk_kernel::Union{Nothing, A2} # 3×3 optical crosstalk kernel
    focal_spot_kernel::Union{Nothing, A2}       # focal spot blur kernel
    bowtie_spectral::Union{Nothing, A3}         # [n_cols, n_rows, n_energies] spectral transmission
    bowtie_air_reference::Union{Nothing, A2}    # [n_cols, n_rows] spectral air I₀
    lag_coeffs::Union{Nothing, A1}              # lag coefficients (n_frames)

    # ─── Noise (CPU + GPU) ───
    noise_rand_cpu::Vector{T}  # randn output (n_elements)
    noise_rand_gpu::A1         # GPU transfer buffer (n_elements)
    enoise_rand_cpu::Vector{T} # electronic noise randn output (n_elements)
    enoise_rand_gpu::A1        # electronic noise GPU transfer buffer (n_elements)

    # ─── Pre-computed vectors ───
    weights_norm::Vector{T}    # T.(weights ./ sum(weights))
    μ_lut_cpu::Vector{T}       # μ LUT CPU buffer (n_regions)
    μ_lut_gpu::A1              # μ LUT GPU buffer (matches mask backend)
    μ_table::Matrix{T}         # pre-computed μ[region, energy] (n_regions × n_energies)
    μ_table_gpu::A2            # GPU copy of μ_table for zero-copy create_μ_volume!
    η_vec::Vector{Float64}     # detector efficiency η(E) per energy bin
    wη_gpu::A1                 # Pre-computed weights_norm .* η on GPU [n_energies] (fused kernel)
    bhc_coeffs_gpu::A1         # BHC polynomial coefficients (GPU/backend)

    # ─── Pre-computed geometry arrays (T-typed, same backend as mask) ───
    geom_source_positions::A2      # (3, n_angles) source positions
    geom_detector_centers::A2      # (3, n_angles) detector centers
    geom_detector_u::A2            # (3, n_angles) detector u vectors
    geom_detector_v::A2            # (3, n_angles) detector v vectors

    # ─── Pre-computed setup data ───
    geom::CTGeometry
    energies::Vector{Float64}
    weights::Vector{Float64}
    config::PhysicsConfig
    mats::Vector
    rng::MersenneTwister
    # Signal chain config (extracted from PhysicsConfig for zero-alloc)
    bhc::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}}

    # ─── Result staging (CPU) ───
    sino_ideal_out::Array{T,3}
    sino_noisy_out::Array{T,3}
end

"""
    create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T=Float32, materials=nothing)

Create a pre-allocated workspace for zero-allocation EICT single-kVp `simulate!()`.
"""
function create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom;
                                T::Type{<:AbstractFloat}=Float32,
                                materials::Union{Nothing, Vector}=nothing)
    # Geometry
    geom = CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm, z_cm=recon_opts.z_cm, collimation_mm=protocol.collimation_mm)

    # Spectrum (pass scanner for IPEM pipeline)
    energies, weights_vec = resolve_spectrum(sim_opts, protocol; scanner=scanner)
    n_energies = length(energies)

    # Physics config
    config = build_physics_config(scanner, sim_opts, energies, weights_vec; phantom=phantom)

    # Materials
    mats = _resolve_materials(phantom, materials)
    n_regions = length(mats)

    # Dimensions
    sino_shape = (geom.n_cols, geom.n_rows, geom.n_angles)
    vol_shape = size(phantom.mask)
    n_elements = prod(sino_shape)

    # Reference array for similar() — matches GPU backend
    ref = phantom.mask

    # GPU-side buffers
    sinogram = similar(ref, T, sino_shape)
    μ_volume = similar(ref, T, vol_shape)
    sino_mono = similar(ref, T, sino_shape)
    I_transmitted = similar(ref, T, sino_shape)
    air_scan = similar(ref, T, sino_shape)

    # Physics effect scratch buffers (sinogram-sized, shared)
    physics_output = similar(ref, T, sino_shape)
    lag_intensity = similar(ref, T, sino_shape)

    # Pre-compute physics kernels (GPU-side)
    # These depend only on config and are constant across calls.
    scatter_kernel = if config.scatter !== nothing
        k_cpu = T.(create_scatter_kernel_spatial(config.scatter))
        k_gpu = similar(ref, T, size(k_cpu)...)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    scatter_correct_kernel = if config.scatter_correction !== nothing
        sc_temp = ScatterModel(
            config.scatter_correction.correction_coefficient,
            config.scatter_correction.scale_factor,
            config.scatter_correction.kernel_fwhm,
            config.scatter_correction.kernel_type
        )
        k_cpu = T.(create_scatter_kernel_spatial(sc_temp))
        k_gpu = similar(ref, T, size(k_cpu)...)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    # 1D scatter kernels for separable Gaussian convolution (SPEED-BUILD-002)
    scatter_kernel_1d = if config.scatter !== nothing
        k1d_cpu = create_scatter_kernel_1d(config.scatter)
        if k1d_cpu !== nothing
            k1d = similar(ref, T, length(k1d_cpu))
            copyto!(k1d, T.(k1d_cpu))
            k1d
        else
            nothing
        end
    else
        nothing
    end

    scatter_correct_kernel_1d = if config.scatter_correction !== nothing
        sc_temp2 = ScatterModel(
            config.scatter_correction.correction_coefficient,
            config.scatter_correction.scale_factor,
            config.scatter_correction.kernel_fwhm,
            config.scatter_correction.kernel_type
        )
        k1d_cpu = create_scatter_kernel_1d(sc_temp2)
        if k1d_cpu !== nothing
            k1d = similar(ref, T, length(k1d_cpu))
            copyto!(k1d, T.(k1d_cpu))
            k1d
        else
            nothing
        end
    else
        nothing
    end

    # Scratch buffer for separable scatter convolution intermediate results
    scatter_temp = similar(ref, T, sino_shape)

    optical_crosstalk_kernel = if config.optical_crosstalk !== nothing
        k_cpu = T.(create_optical_crosstalk_kernel(config.optical_crosstalk))
        k_gpu = similar(ref, T, 3, 3)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    focal_spot_kernel = if config.focal_spot !== nothing
        blur_fwhm = compute_focal_spot_blur_fwhm(config.focal_spot, geom, geom.SAD)
        if blur_fwhm[1] >= 0.1 || blur_fwhm[2] >= 0.1
            k_cpu = T.(create_focal_spot_kernel_spatial(config.focal_spot, blur_fwhm))
            k_gpu = similar(ref, T, size(k_cpu)...)
            copyto!(k_gpu, k_cpu)
            k_gpu
        else
            nothing
        end
    else
        nothing
    end

    lag_coeffs_buf = if config.lag !== nothing && !isempty(config.lag.amplitudes)
        n_frames = min(20, sino_shape[3])
        c_cpu = T.(compute_lag_coefficients(config.lag, n_frames))
        c_gpu = similar(ref, T, n_frames)
        copyto!(c_gpu, c_cpu)
        c_gpu
    else
        nothing
    end

    # Noise buffers
    noise_rand_cpu = Vector{T}(undef, n_elements)
    noise_rand_gpu = similar(ref, T, n_elements)
    enoise_rand_cpu = Vector{T}(undef, n_elements)
    enoise_rand_gpu = similar(ref, T, n_elements)

    # Pre-computed weights
    w_sum = sum(weights_vec)
    weights_norm = T.(weights_vec ./ w_sum)

    # μ lookup table
    μ_lut_cpu = Vector{T}(undef, n_regions)
    μ_lut_gpu = similar(ref, T, n_regions)
    μ_table = zeros(T, n_regions, n_energies)
    for (e_idx, E) in enumerate(energies)
        for r in 1:n_regions
            μ_table[r, e_idx] = T(compute_μ_at_energy(mats[r], Float64(E)))
        end
    end

    # Upload μ_table to GPU, padded to multiple of 16 for tiled projection (SPEED-BUILD-V2-002)
    # Extra columns are zero-filled — zero wη weights ensure they contribute nothing.
    n_energies_padded = cld(n_energies, 16) * 16
    μ_table_gpu = similar(ref, T, n_regions, n_energies_padded)
    fill!(μ_table_gpu, zero(T))
    copyto!(view(μ_table_gpu, :, 1:n_energies), μ_table)

    # Detector efficiency η(E) per energy bin
    η_vec = if config.detector_efficiency !== nothing
        compute_eid_efficiency_vector(config.detector_efficiency, energies)
    else
        ones(Float64, n_energies)
    end

    # Pre-computed weights_norm .* η, padded to match μ_table (tiled projection)
    wη_cpu = zeros(T, n_energies_padded)
    wη_cpu[1:n_energies] .= T.(weights_norm .* η_vec)
    wη_gpu_buf = similar(ref, T, n_energies_padded)
    copyto!(wη_gpu_buf, wη_cpu)

    # Source spectral transmission: bowtie × heel effect (both per-pixel, per-energy)
    # These are pre-patient source effects applied during spectral forward projection.
    bowtie_filter = resolve_bowtie_filter(scanner.bowtie_filter)
    bowtie_spectral_gpu = nothing
    bowtie_air_ref_gpu = nothing

    # Start with heel effect spectral transmission (energy-dependent, per-column)
    heel_trans = if config.heel_effect !== nothing
        compute_heel_spectral(config.heel_effect, geom, Float64.(energies))
    else
        nothing
    end

    # Bowtie spectral transmission (energy-dependent, per-pixel)
    bowtie_trans = if bowtie_filter !== nothing && bowtie_filter.name != "none"
        compute_bowtie_attenuation_spectral(bowtie_filter, geom, Float64.(energies))
    else
        nothing
    end

    # Combine: source_transmission = bowtie × heel (both are [n_cols, n_rows, n_energies])
    has_source_spectral = bowtie_trans !== nothing || heel_trans !== nothing
    if has_source_spectral
        if bowtie_trans !== nothing && heel_trans !== nothing
            trans_cpu = bowtie_trans .* heel_trans  # Element-wise multiply
        elseif bowtie_trans !== nothing
            trans_cpu = bowtie_trans
        else
            trans_cpu = heel_trans
        end

        # Pad to n_energies_padded
        bt_padded_cpu = zeros(T, size(trans_cpu, 1), size(trans_cpu, 2), n_energies_padded)
        bt_padded_cpu[:, :, 1:n_energies] .= T.(trans_cpu)
        bt_gpu = similar(ref, T, size(bt_padded_cpu)...)
        copyto!(bt_gpu, bt_padded_cpu)
        bowtie_spectral_gpu = bt_gpu

        # Air reference: I₀(col,row) = Σ w_norm(E) × T_source(E,col,row) × η(E)
        w_norm = weights_vec ./ sum(weights_vec)
        air_ref_cpu = zeros(T, sino_shape[1], sino_shape[2])
        for e in 1:n_energies
            η_e = η_vec[e]
            for row in 1:sino_shape[2], col in 1:sino_shape[1]
                air_ref_cpu[col, row] += T(w_norm[e] * trans_cpu[col, row, e] * η_e)
            end
        end
        ar_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(ar_gpu, air_ref_cpu)
        bowtie_air_ref_gpu = ar_gpu
    end

    # BHC coefficients
    bhc_coeffs_cpu = if config.bhc !== nothing
        T.(get_bhc_coefficients(config.bhc))
    else
        zeros(T, 1)
    end
    bhc_coeffs_gpu = similar(ref, T, length(bhc_coeffs_cpu))
    copyto!(bhc_coeffs_gpu, bhc_coeffs_cpu)

    # Extract signal chain config from PhysicsConfig (pre-computed)
    # Note: heel_effect is now in spectral domain (folded into bowtie_spectral above)
    bhc_effect = config.bhc

    # Pre-computed geometry arrays (T-typed, same backend as mask) for siddon_forward_project!
    geom_source_positions = similar(ref, T, size(geom.source_positions)...)
    copyto!(geom_source_positions, T.(geom.source_positions))
    geom_detector_centers = similar(ref, T, size(geom.detector_centers)...)
    copyto!(geom_detector_centers, T.(geom.detector_centers))
    geom_detector_u = similar(ref, T, size(geom.detector_u)...)
    copyto!(geom_detector_u, T.(geom.detector_u))
    geom_detector_v = similar(ref, T, size(geom.detector_v)...)
    copyto!(geom_detector_v, T.(geom.detector_v))

    # RNG
    rng = MersenneTwister(0)

    # CPU staging
    sino_ideal_out = zeros(T, sino_shape)
    sino_noisy_out = zeros(T, sino_shape)

    return EICTWorkspace{T, typeof(sinogram), typeof(geom_source_positions), typeof(noise_rand_gpu)}(
        sinogram, μ_volume, sino_mono, I_transmitted, air_scan,
        physics_output, lag_intensity,
        scatter_kernel, scatter_correct_kernel,
        scatter_kernel_1d, scatter_correct_kernel_1d, scatter_temp,
        optical_crosstalk_kernel, focal_spot_kernel,
        bowtie_spectral_gpu, bowtie_air_ref_gpu, lag_coeffs_buf,
        noise_rand_cpu, noise_rand_gpu, enoise_rand_cpu, enoise_rand_gpu,
        weights_norm, μ_lut_cpu, μ_lut_gpu, μ_table, μ_table_gpu, η_vec, wη_gpu_buf, bhc_coeffs_gpu,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        geom, energies, weights_vec, config, mats, rng,
        bhc_effect,
        sino_ideal_out, sino_noisy_out
    )
end

# =============================================================================
# FDKReconWorkspace — Pre-allocated workspace for zero-allocation reconstruct!()
# =============================================================================

"""
    FDKReconWorkspace{T, A3, A2, A1}

Pre-allocated workspace for zero-allocation FDK reconstruction.

Type parameters:
- `T`: Element type (Float32, Float64)
- `A3`: 3D array type (Array{T,3}, MtlArray{T,3}, CuArray{T,3})
- `A2`: 2D array type for geometry arrays (Matrix{T}, MtlArray{T,2}, etc.)
- `A1`: 1D array type for filter kernel

Create with [`create_fdk_recon_workspace`](@ref).
"""
mutable struct FDKReconWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A2<:AbstractArray{T,2}, A1<:AbstractArray{T,1}}
    # ─── Output ───
    volume::A3                # Reconstructed volume (nx, ny, nz)

    # ─── Filtering scratch ───
    filtered::A3              # Cosine-weighted + filtered sinogram (sino shape)
    conv_scratch::A3          # Convolution output scratch (sino shape)
    filter_kernel::A1         # Spatial domain filter kernel

    # ─── Backprojection geometry (GPU-side, pre-computed) ───
    bp_source_positions::A2   # Geometry arrays for backproject! [3, n_angles]
    bp_detector_centers::A2
    bp_detector_u::A2
    bp_detector_v::A2
end

"""
    create_fdk_recon_workspace(sinogram, geom, volume_size; T=eltype(sinogram), filter=StandardFilter(), cutoff=1.0)

Create a pre-allocated workspace for zero-allocation FDK `reconstruct!()`.

`filter` can be a `FilterType` struct (e.g., `StandardFilter()`) or a `Symbol`
(e.g., `:standard`, `:ram_lak`).
"""
function create_fdk_recon_workspace(
    sinogram::AbstractArray{<:AbstractFloat, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    T::Type{<:AbstractFloat} = eltype(sinogram),
    filter::Union{FilterType, Symbol} = StandardFilter(),
    cutoff::Float64 = 1.0
)
    filter = filter isa Symbol ? filter_from_symbol(filter) : filter
    sino_shape = size(sinogram)

    # Output volume
    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))

    # Filtering scratch buffers
    filtered = similar(sinogram, T, sino_shape...)
    conv_scratch = similar(sinogram, T, sino_shape...)

    # Pre-compute filter kernel on GPU
    pixel_size = T(geom.pixel_size)
    n_cols = size(sinogram, 1)
    raw_size = max(Int(ceil(n_cols * cutoff)), 32)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), n_cols)
    kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size)
    filter_kernel = similar(sinogram, T, kernel_size_int)
    copyto!(filter_kernel, kernel_cpu)

    # Pre-compute geometry arrays on GPU (for backprojection)
    bp_source_positions = similar(sinogram, T, size(geom.source_positions)...)
    copyto!(bp_source_positions, T.(geom.source_positions))
    bp_detector_centers = similar(sinogram, T, size(geom.detector_centers)...)
    copyto!(bp_detector_centers, T.(geom.detector_centers))
    bp_detector_u = similar(sinogram, T, size(geom.detector_u)...)
    copyto!(bp_detector_u, T.(geom.detector_u))
    bp_detector_v = similar(sinogram, T, size(geom.detector_v)...)
    copyto!(bp_detector_v, T.(geom.detector_v))

    return FDKReconWorkspace{T, typeof(volume), typeof(bp_source_positions), typeof(filter_kernel)}(
        volume, filtered, conv_scratch, filter_kernel,
        bp_source_positions, bp_detector_centers, bp_detector_u, bp_detector_v
    )
end

# =============================================================================
# HIRReconWorkspace — Pre-allocated workspace for zero-allocation Hybrid IR reconstruct!()
# =============================================================================

"""
    HIRReconWorkspace{T, A3, A2, A1}

Pre-allocated workspace for zero-allocation Hybrid IR reconstruction.

Includes FDK initialization buffers + PWLS iteration buffers.
Geometry arrays are shared between forward projection and backprojection.

Type parameters:
- `T`: Element type (Float32, Float64)
- `A3`: 3D array type (Array{T,3}, MtlArray{T,3}, CuArray{T,3})
- `A2`: 2D array type for geometry arrays
- `A1`: 1D array type for filter kernel

Create with [`create_hir_recon_workspace`](@ref).
"""
mutable struct HIRReconWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A2<:AbstractArray{T,2}, A1<:AbstractArray{T,1}}
    # ─── Output / iterate ───
    volume::A3                # (nx, ny, nz) — FDK result → PWLS iterate

    # ─── FDK filtering scratch ───
    filtered::A3              # (sino shape)
    conv_scratch::A3          # (sino shape)
    filter_kernel::A1         # (~n_cols elements)

    # ─── Shared geometry (GPU-side, used by fwd_proj + backproj) ───
    geom_source_positions::A2 # (3, n_angles)
    geom_detector_centers::A2
    geom_detector_u::A2
    geom_detector_v::A2

    # ─── PWLS pre-computed weights (computed once in create) ───
    W_proj::A3                # Projection domain weights (sino shape)
    V_inv::A3                 # Image domain weights (vol shape)

    # ─── PWLS iteration scratch (reused every iteration) ───
    stat_weights::A3          # Statistical weights (sino shape)
    Ax::A3                    # Forward projection scratch (sino shape)
    correction::A3            # Backprojection scratch (vol shape)
    reg_grad::A3              # Regularization gradient (vol shape)

    # ─── Ordered subsets (pre-computed) ───
    subsets::Vector{Vector{Int}}            # angle indices per subset
    subset_geometries::Vector{CTGeometry}   # pre-built geometry per subset
    subset_geom_source_positions::Vector{A2}  # GPU geometry arrays per subset
    subset_geom_detector_centers::Vector{A2}
    subset_geom_detector_u::Vector{A2}
    subset_geom_detector_v::Vector{A2}
    subset_sino_buf::A3                     # (n_cols, n_rows, max_subset_size)
    subset_Ax_buf::A3                       # same shape for forward projection
    subset_W_proj_buf::A3                   # same shape for projection weights
    subset_stat_weights_buf::A3             # same shape for statistical weights

    # ─── Pre-computed HIR params ───
    params::HIRParams
end

"""
    create_hir_recon_workspace(sinogram, geom, volume_size; T=eltype(sinogram), strength=3, filter=StandardFilter(), cutoff=1.0)

Create a pre-allocated workspace for zero-allocation Hybrid IR `reconstruct!()`.

`filter` can be a `FilterType` struct or a `Symbol` (e.g., `:standard`).
"""
function create_hir_recon_workspace(
    sinogram::AbstractArray{<:AbstractFloat, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    T::Type{<:AbstractFloat} = eltype(sinogram),
    strength::Int = 3,
    filter::Union{FilterType, Symbol} = StandardFilter(),
    cutoff::Float64 = 1.0
)
    filter = filter isa Symbol ? filter_from_symbol(filter) : filter
    sino_shape = size(sinogram)

    # Output volume / iterate
    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))

    # FDK filtering scratch
    filtered = similar(sinogram, T, sino_shape...)
    conv_scratch = similar(sinogram, T, sino_shape...)

    # Pre-compute filter kernel on GPU
    pixel_size = T(geom.pixel_size)
    n_cols = size(sinogram, 1)
    raw_size = max(Int(ceil(n_cols * cutoff)), 32)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), n_cols)
    kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size)
    filter_kernel = similar(sinogram, T, kernel_size_int)
    copyto!(filter_kernel, kernel_cpu)

    # Pre-compute geometry arrays on GPU (shared by fwd_proj + backproj)
    geom_source_positions = similar(sinogram, T, size(geom.source_positions)...)
    copyto!(geom_source_positions, T.(geom.source_positions))
    geom_detector_centers = similar(sinogram, T, size(geom.detector_centers)...)
    copyto!(geom_detector_centers, T.(geom.detector_centers))
    geom_detector_u = similar(sinogram, T, size(geom.detector_u)...)
    copyto!(geom_detector_u, T.(geom.detector_u))
    geom_detector_v = similar(sinogram, T, size(geom.detector_v)...)
    copyto!(geom_detector_v, T.(geom.detector_v))

    # Pre-compute SIRT-style normalization weights (W_proj and V_inv)
    # These are expensive but only computed once
    W_proj_cpu = compute_projection_weights(geom, volume_size, T)
    W_proj = similar(sinogram, T, size(W_proj_cpu)...)
    copyto!(W_proj, W_proj_cpu)

    V_inv_cpu = compute_image_weights(geom, volume_size, T)
    V_inv = similar(sinogram, T, size(V_inv_cpu)...)
    copyto!(V_inv, V_inv_cpu)

    # PWLS iteration scratch buffers
    stat_weights = similar(sinogram, T, sino_shape...)
    Ax = similar(sinogram, T, sino_shape...)
    correction = similar(sinogram, T, volume_size...)
    reg_grad = similar(sinogram, T, volume_size...)

    # HIR params
    params = get_hir_params(strength)

    # Ordered subsets pre-computation
    n_subsets = params.n_subsets
    if n_subsets > 0
        subsets = create_ordered_subsets(geom.n_angles, n_subsets)
        subset_geometries = [create_subset_geometry(geom, indices) for indices in subsets]

        # Pre-compute GPU geometry arrays for each subset
        subset_geom_src = Vector{typeof(geom_source_positions)}(undef, n_subsets)
        subset_geom_det = Vector{typeof(geom_source_positions)}(undef, n_subsets)
        subset_geom_u = Vector{typeof(geom_source_positions)}(undef, n_subsets)
        subset_geom_v = Vector{typeof(geom_source_positions)}(undef, n_subsets)
        for s in 1:n_subsets
            sg = subset_geometries[s]
            subset_geom_src[s] = similar(sinogram, T, size(sg.source_positions)...)
            copyto!(subset_geom_src[s], T.(sg.source_positions))
            subset_geom_det[s] = similar(sinogram, T, size(sg.detector_centers)...)
            copyto!(subset_geom_det[s], T.(sg.detector_centers))
            subset_geom_u[s] = similar(sinogram, T, size(sg.detector_u)...)
            copyto!(subset_geom_u[s], T.(sg.detector_u))
            subset_geom_v[s] = similar(sinogram, T, size(sg.detector_v)...)
            copyto!(subset_geom_v[s], T.(sg.detector_v))
        end

        # Allocate subset buffers sized for the largest subset
        max_subset_size = maximum(length(s) for s in subsets)
        subset_sino_shape = (sino_shape[1], sino_shape[2], max_subset_size)
        subset_sino_buf = similar(sinogram, T, subset_sino_shape...)
        subset_Ax_buf = similar(sinogram, T, subset_sino_shape...)
        subset_W_proj_buf = similar(sinogram, T, subset_sino_shape...)
        subset_stat_weights_buf = similar(sinogram, T, subset_sino_shape...)
    else
        # Legacy mode: no subsets
        subsets = Vector{Int}[]
        subset_geometries = CTGeometry[]
        subset_geom_src = typeof(geom_source_positions)[]
        subset_geom_det = typeof(geom_source_positions)[]
        subset_geom_u = typeof(geom_source_positions)[]
        subset_geom_v = typeof(geom_source_positions)[]
        # Allocate minimal buffers (won't be used)
        subset_sino_buf = similar(sinogram, T, 1, 1, 1)
        subset_Ax_buf = similar(sinogram, T, 1, 1, 1)
        subset_W_proj_buf = similar(sinogram, T, 1, 1, 1)
        subset_stat_weights_buf = similar(sinogram, T, 1, 1, 1)
    end

    return HIRReconWorkspace{T, typeof(volume), typeof(geom_source_positions), typeof(filter_kernel)}(
        volume, filtered, conv_scratch, filter_kernel,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        W_proj, V_inv,
        stat_weights, Ax, correction, reg_grad,
        subsets, subset_geometries,
        subset_geom_src, subset_geom_det, subset_geom_u, subset_geom_v,
        subset_sino_buf, subset_Ax_buf, subset_W_proj_buf, subset_stat_weights_buf,
        params
    )
end
