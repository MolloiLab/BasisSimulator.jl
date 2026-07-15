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
mutable struct PCCTWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}, A1 <: AbstractArray{T, 1}, A2 <: AbstractArray{T, 2}}
    # ─── Forward projection (GPU-side) ───
    bins::Vector{A3}           # n_bins sinogram buffers (output of forward projection)
    μ_volume::A3               # attenuation volume, reused per energy
    sino_buf::A3               # forward projection scratch, reused per energy

    # ─── Spatial kernel scratch (GPU-side) ───
    scratch::A3                # ONE buffer for spatial kernels

    # ─── Combine (GPU-side) ───
    combined::A3               # scratch for `combined_primary` in scatter step

    # ─── Noise CPU staging (Phase 1: CPU RNG) ───
    noise_staging::Array{T, 3}  # CPU buffer for GPU↔CPU noise transfer
    noise_buf::Array{T, 3}      # randn output buffer (CPU)

    # ─── Pre-computed CPU vectors/matrices ───
    η::Vector{Float64}          # quantum efficiency vector (n_energies)
    R::Matrix{Float64}          # spectral response matrix (n_energies × n_bins)
    R_energies::Vector{Float64} # energy grid for R matrix
    I0_bins::Vector{Float64}    # per-bin I0 values for combine (n_bins)
    I0_bins_norm::Vector{Float64} # per-bin I0 values for fwd proj normalization (n_bins)
    thresholds_T::Vector{T}     # T-typed thresholds (n_bins)

    # ─── RNG state ───
    rng::MersenneTwister        # pre-allocated RNG (reset with seed each call)

    # ─── Noise I0 ───
    noise_I0::Vector{Float64}   # per-bin I0 for noise model (n_bins)

    # ─── μ lookup table (pre-computed for all energies × materials) ───
    μ_lut_cpu::Vector{T}                     # μ LUT CPU buffer (n_regions)
    μ_lut_gpu::A1                            # μ LUT GPU buffer (n_regions, matches mask backend)
    μ_table::Matrix{T}                       # pre-computed μ[region, energy] (n_regions × n_energies)

    # ─── Pre-computed geometry arrays (T-typed, same backend as mask) ───
    geom_source_positions::A2                # (3, n_angles) source positions
    geom_detector_centers::A2                # (3, n_angles) detector centers
    geom_detector_u::A2                      # (3, n_angles) detector u vectors
    geom_detector_v::A2                      # (3, n_angles) detector v vectors

    # ─── Native-resolution buffers (for spatial binning path, bf > 1) ───
    native_bins::Union{Nothing, Vector{A3}}  # per-bin at native dexel res
    native_sino_buf::Union{Nothing, A3}      # Siddon scratch at native res
    native_geom::Union{Nothing, CTGeometry}  # native-res geometry
    native_geom_source_positions::Union{Nothing, A2}
    native_geom_detector_centers::Union{Nothing, A2}
    native_geom_detector_u::Union{Nothing, A2}
    native_geom_detector_v::Union{Nothing, A2}

    # ─── Tube physics scratch (for combined sinogram) ───
    tube_physics_scratch::Union{Nothing, A3}  # sinogram-sized scratch for scatter/focal spot
    focal_spot_kernel::Union{Nothing, A2}     # pre-computed focal-spot kernel (nothing when blur off/negligible)

    # ─── Tiled spectral projection (fused PCCT forward projection) ───
    μ_table_gpu::A2                          # GPU copy of μ_table [n_regions, n_energies_padded]
    W_matrix_gpu::A2                         # spectral weight matrix [n_energies_padded, n_bins] on GPU
    outputs_flat::A1                         # flattened output buffer [n_elements * n_bins] on GPU
    native_outputs_flat::Union{Nothing, A1}  # native-res flattened output (nothing if bf==1)

    # ─── Pulse pileup (full MC-LUT spectral migration) ───
    # `pileup_S` is the MC-derived n_bins × n_bins matrix returned by
    # `compute_mc_pileup_matrix`: S[i,j] = fraction of true-bin-j counts
    # recorded in bin-i.  Column sums ≤ 1 — the deficit is the count loss
    # from pileup, so a single S × counts multiply captures both spectral
    # migration AND count loss.  `simulate!` applies S in count domain and
    # re-normalizes against `I0_bins` (truth) so the returned bins remain
    # `-log(recorded / I0_truth)` and the round-trip
    # `I0_b · exp(-bin) = recorded count` stays valid for downstream
    # count-domain math (scatter correction, bin combine, …).
    use_pcct_pileup::Bool                              # toggle from sim_opts.use_pcct_pileup
    pileup_S::Union{Nothing, Matrix{Float64}}          # (n_bins × n_bins), nothing when pileup off

    # ─── Pre-computed setup data (computed once, reused) ───
    geom::CTGeometry                         # CT geometry (binned resolution)
    energies::Vector{Float64}                # downsampled spectrum energies
    weights::Vector{Float64}                 # downsampled spectrum weights
    config::PhysicsConfig                    # physics configuration
    pcct_detector::PhotonCountingDetector{Float64}  # PCCT detector
    mats::Vector{XA.Material}                # resolved materials
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
- `recon_opts`: Reconstruction options (provides fov_cm / z_cm for CTGeometry)
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
function create_workspace(
        scanner, protocol, sim_opts, recon_opts, phantom;
        T::Type{<:AbstractFloat} = Float32,
        extended_collimation::Bool = false,
    )
    # --- Pre-compute geometry first (collimation derives n_rows) ---
    geom = CTGeometry(scanner; n_angles = protocol.views, fov_cm = recon_opts.fov_cm, z_cm = recon_opts.z_cm, collimation_mm = protocol.collimation_mm, extended_collimation = extended_collimation, pitch = protocol.pitch, n_rotations = protocol.n_rotations)

    sino_shape = (geom.n_cols, geom.n_rows, geom.n_angles)
    vol_shape = size(phantom.mask)
    n_bins = length(scanner.energy_thresholds)
    n_elements = prod(sino_shape)
    # n_energies is set after resolve_source_spectrum_without_bowtie below

    # GPU-side buffers — similar() matches phantom.mask's backend
    ref_mask = phantom.mask
    bins = [similar(ref_mask, T, sino_shape) for _ in 1:n_bins]
    μ_volume = similar(ref_mask, T, vol_shape)
    sino_buf = similar(ref_mask, T, sino_shape)
    scratch = similar(ref_mask, T, sino_shape)
    combined = similar(ref_mask, T, sino_shape)

    # CPU-side buffers (Phase 1 noise: CPU RNG → GPU staging)
    noise_staging = zeros(T, sino_shape)
    noise_buf = zeros(T, sino_shape)
    energies, weights_vec = resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    n_energies = length(energies)
    config = build_physics_config(scanner, sim_opts, energies, weights_vec; phantom = phantom)
    pcct_detector = _build_pcct_detector(scanner)
    mats = phantom.materials
    kVp = Float64(maximum(energies))
    thresholds = pcct_detector.energy_thresholds_keV
    # Pre-compute spectral response data
    η_vec = quantum_efficiency_vector(pcct_detector.material, pcct_detector.thickness_mm, energies)
    R_mat = compute_mc_drm(pcct_detector, kVp)
    R_energies_vec = collect(range(1.0, Float64(kVp), length = size(R_mat, 1)))

    # BHC, scatter correction, and pile-up correction are all decoupled —
    # applied at the notebook level via dedicated `apply_*` functions.

    # Pre-compute per-bin I0 (DRM-weighted spectrum × η) in PHYSICAL counts.
    # `_I0_anchor` is the protocol-derived incident photons per ray per view
    # divided by the raw spectrum sum, so N_air(b) = I0_phys·Σ(ŵ·η·R) — true
    # detected counts.  The SAME anchor scales the forward W matrix below;
    # the two sites are a coupled invariant (a global count-unit rescale):
    # change them TOGETHER or the -log(N/I0) round trip breaks.  Previously
    # both used a hardcoded 1.0e6 → ~7e11 fake "counts"/ray, which (a) let
    # the scatter-correction eps floor mint deterministic p≈50 plateau rays
    # and (b) left the noise model's low-count Poisson branch dead.
    _I0_anchor = compute_detector_I0(geom, protocol, sum(weights_vec)) /
        max(sum(weights_vec), 1.0e-30)
    # AIR CALIBRATION: per-bin I0 = the forward kernel's own air response,
    # Σ_e W[e,b] (all paths zero → exp(0) = 1) — guarantees p_air ≡ 0 by
    # construction, like a real scanner's air cal.  (`_compute_bin_I0`
    # disagreed with the kernel's actual air output by 13–26 % per bin —
    # measured air rays read p = 0.24/0.22/0.17/0.13 instead of 0, a
    # per-bin DC that biased every PCCT decomposition.)  Filled right after
    # W_cpu is built below.
    I0_bins_norm_vec = Float64[]
    I0_bins_combine = I0_bins_norm_vec === nothing ? Float64[] : Float64[]

    # Pre-compute T-typed thresholds
    thresholds_T_vec = T.(thresholds)

    rng = MersenneTwister(0)

    # Detector physics precomputed buffers
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
            geom.fov,  # same recon FOV
            geom.pitch, geom.table_feed, geom.detector_shape
        )
        native_sino_shape = (_native_geom.n_cols, _native_geom.n_rows, _native_geom.n_angles)
        _native_bins = [similar(ref_mask, T, native_sino_shape) for _ in 1:n_bins]
        _native_sino_buf = similar(ref_mask, T, native_sino_shape)
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
    # SAME physical anchor as I0_bins_norm_vec above (coupled invariant).
    _I0 = _I0_anchor
    n_R = size(R_mat, 1)
    W_cpu = zeros(T, n_energies_padded, n_bins)
    for e_idx in 1:n_energies
        E_float = Float64(energies[e_idx])
        w = Float64(weights_vec[e_idx])
        if w < 1.0e-12
            continue
        end
        r_idx = clamp(round(Int, (E_float - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        for b in 1:n_bins
            W_cpu[e_idx, b] = T(_I0 * w * η_vec[e_idx] * R_mat[r_idx, b])
        end
    end
    _W_matrix_gpu = similar(ref_mask, T, n_energies_padded, n_bins)

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

    # AIR CALIBRATION — after ALL W shaping (η, DRM, bowtie-centre fold):
    # the fused kernel's air output is exactly Σ_e W[e,b], so this
    # normalization guarantees p_air ≡ 0 per bin by construction.  (The old
    # _compute_bin_I0 normalization ignored the bowtie fold → air rays read
    # p = 0.13–0.24 per bin, a DC on every ray.)
    append!(I0_bins_norm_vec, [sum(Float64.(W_cpu[1:n_energies, b])) for b in 1:n_bins])
    append!(I0_bins_combine, I0_bins_norm_vec)
    copyto!(_W_matrix_gpu, W_cpu)

    # Flattened output buffer for spectral projection
    _outputs_flat = similar(ref_mask, T, n_elements * n_bins)

    # Native-resolution flattened output (for spatial binning path)
    _native_outputs_flat = if bf > 1
        native_n_elements = prod(native_sino_shape)
        similar(ref_mask, T, native_n_elements * n_bins)
    else
        nothing
    end

    # ─── Pre-compute MC-LUT pulse-pileup migration matrix S ───
    # S[i,j] = P(true-bin-j count is recorded in bin-i).  Column sums ≤ 1, the
    # deficit being the count loss, so a single S×counts multiply applies
    # spectral migration AND count loss in one shot — no analytical Taguchi /
    # semi-non-paralyzable fallback.
    #
    # Pileup is per native dexel (not per binned pixel).
    # I0 from compute_detector_I0 is per binned pixel per view.
    # Count rate per dexel = (I0 / bf²) / time_per_view  [photons/s]
    _use_pileup = sim_opts.use_pcct_pileup &&
        pcct_detector.dead_time_ns > 0
    _pileup_S = if _use_pileup
        _I0_physics_pileup = compute_detector_I0(geom, protocol, sum(weights_vec))
        _time_per_view_pileup = protocol.rotation_time / protocol.views
        _count_rate_per_dexel = (_I0_physics_pileup / Float64(bf * bf)) / _time_per_view_pileup
        _τ_ns = Float64(pcct_detector.dead_time_ns)
        w_norm = Float64.(weights_vec) ./ sum(Float64.(weights_vec))
        compute_mc_pileup_matrix(
            pcct_detector.energy_thresholds_keV,
            w_norm, Float64.(energies),
            _count_rate_per_dexel, _τ_ns;
            n_trials = 5000, seed = 42
        )
    else
        nothing
    end

    # Scatter bin fractions are now computed on-the-fly in simulate!() using
    # compute_scatter_energy_weights() + compute_scatter_bin_weights() (unified per-energy model)

    # ─── Pre-compute focal-spot blur kernel (opt-in; :pcct preset default off) ───
    # Mirrors the EICT workspace: kernel is built once here so the per-bin
    # blur in simulate! stays allocation-free.
    _pcct_focal_kernel = if config.focal_spot !== nothing
        _blur_fwhm = compute_focal_spot_blur_fwhm(config.focal_spot, geom, geom.SAD)
        if _blur_fwhm[1] >= 0.1 || _blur_fwhm[2] >= 0.1
            _k_cpu = T.(create_focal_spot_kernel_spatial(config.focal_spot, _blur_fwhm))
            _k_gpu = similar(ref_mask, T, size(_k_cpu)...)
            copyto!(_k_gpu, _k_cpu)
            _k_gpu
        else
            nothing
        end
    else
        nothing
    end

    return PCCTWorkspace{T, typeof(sino_buf), typeof(μ_lut_gpu), typeof(geom_source_positions)}(
        bins, μ_volume, sino_buf, scratch,
        combined,
        noise_staging, noise_buf,
        η_vec, R_mat, R_energies_vec, I0_bins_combine, I0_bins_norm_vec, thresholds_T_vec, rng,
        noise_I0,
        μ_lut_cpu, μ_lut_gpu, μ_table,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        _native_bins, _native_sino_buf,
        _native_geom, _n_src, _n_det, _n_u, _n_v,
        tube_scratch, _pcct_focal_kernel,
        _μ_table_gpu, _W_matrix_gpu, _outputs_flat, _native_outputs_flat,
        _use_pileup, _pileup_S,
        geom, energies, weights_vec, config, pcct_detector, mats,
        kVp
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
mutable struct EICTWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 2}, A1 <: AbstractArray{T, 1}}
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
    scatter_kernel_1d::Union{Nothing, A1}       # 1D Gaussian scatter kernel (separable path)
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

    # ─── Pre-computed noise constants (scanner + spectrum derived) ───
    η_eff::T          # sum(weights_norm .* η_vec) — spectrum-averaged detector efficiency
    σ_e_photon::T     # electronic_noise / (mean_E_keV * detection_gain) — DAS electronic σ
end

"""
    create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom;
                          T=Float32, spectrum_override=nothing)

Create a pre-allocated workspace for zero-allocation EICT single-kVp `simulate!()`.

# Keyword arguments
- `spectrum_override::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}} = nothing` —
  bypass the polychromatic IPEM lookup and use a custom `(energies, weights)` pair
  for the source spectrum.  When `nothing` (default), the spectrum is loaded from
  the IPEM tables via `resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner)`.

  **Important — `weights` is ABSOLUTE photon flux** (photons / mAs / mm² at 750 mm),
  matching what `load_spectrum_unfiltered` returns.  `compute_detector_I0` reads
  `sum(weights)` to size the per-pixel I0 budget, so passing `[1.0]` yields ~1
  photon / mAs / mm² total — far below the noise clamp, leading to a uniform
  saturated recon.

  For a monoenergetic injection at `E_mono` with the same flux budget as a
  polychromatic protocol at the same kVp, fetch the polychromatic flux total and
  put it all at the mono bin:

  ```julia
  _, w_poly = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner)
  spectrum_override = ([E_mono], [sum(Float64.(w_poly))])
  ```

  Bowtie + heel + detector response evaluate at the override's actual energy
  grid, so per-ray spectral physics still runs.  See
  `docs/notebooks/03b_dual_keV_monoe.jl` for the complete monoenergetic example.
"""
function create_eict_workspace(
        scanner, protocol, sim_opts, recon_opts, phantom;
        T::Type{<:AbstractFloat} = Float32,
        spectrum_override::Union{
            Nothing,
            Tuple{AbstractVector, AbstractVector},
        } = nothing,
        extended_collimation::Bool = false,
    )
    # Geometry
    geom = CTGeometry(scanner; n_angles = protocol.views, fov_cm = recon_opts.fov_cm, z_cm = recon_opts.z_cm, collimation_mm = protocol.collimation_mm, extended_collimation = extended_collimation, pitch = protocol.pitch, n_rotations = protocol.n_rotations)

    # Spectrum — IPEM polychromatic by default; spectrum_override lets a
    # caller inject a custom (energies, weights) pair (e.g. `([70.0], [1.0])`
    # for a monoenergetic 70 keV beam).  Accepts any AbstractVector pair and
    # materializes to `Vector{Float64}` internally.
    energies, weights_vec = if spectrum_override === nothing
        resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    else
        e, w = spectrum_override
        length(e) == length(w) ||
            throw(ArgumentError("spectrum_override: length(energies)=$(length(e)) != length(weights)=$(length(w))"))
        Vector{Float64}(e), Vector{Float64}(w)
    end
    n_energies = length(energies)

    # Physics config
    config = build_physics_config(scanner, sim_opts, energies, weights_vec; phantom = phantom)

    # Materials
    mats = phantom.materials
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

    # BHC is decoupled — applied at notebook level

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

    # Pre-compute scanner+spectrum-derived noise constants — scalars used every
    # simulate! call.  Lets the hot path avoid taking `scanner` and re-summing.
    mean_E_keV = sum(weights_norm[i] * T(energies[i]) for i in 1:length(energies))
    η_eff_T = sum(weights_norm[i] * T(η_vec[i])    for i in 1:length(η_vec))
    σ_e_photon = T(scanner.electronic_noise) / (mean_E_keV * T(scanner.detection_gain))

    return EICTWorkspace{T, typeof(sinogram), typeof(geom_source_positions), typeof(noise_rand_gpu)}(
        sinogram, μ_volume, sino_mono, I_transmitted, air_scan,
        physics_output, lag_intensity,
        scatter_kernel,
        scatter_kernel_1d, scatter_temp,
        optical_crosstalk_kernel, focal_spot_kernel,
        bowtie_spectral_gpu, bowtie_air_ref_gpu, lag_coeffs_buf,
        noise_rand_cpu, noise_rand_gpu, enoise_rand_cpu, enoise_rand_gpu,
        weights_norm, μ_lut_cpu, μ_lut_gpu, μ_table, μ_table_gpu, η_vec, wη_gpu_buf,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        geom, energies, weights_vec, config, mats, rng,
        η_eff_T, σ_e_photon
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
mutable struct FDKReconWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 2}, A1 <: AbstractArray{T, 1}}
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
    raw_size = max(Int(ceil(2 * n_cols * cutoff)), 64)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), 2 * n_cols - 1)
    kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size)
    if is_arc(geom) && !is_helical(geom)
        # equiangular fan filter correction (helical WFBP filters rebinned
        # PARALLEL rows, which need the plain ramp)
        equiangular_kernel_scale!(kernel_cpu, geom.pixel_size / geom.SAD)
    end
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

"""
    calibrate_pcct_poly_bhc(ws; order=5, max_path_cm=50, n_points=100)

Build water BHC for an I₀-weighted combination of all PCCT bins from the
workspace's exact detected spectrum (`source × η × DRM × bowtie`). The same
per-column model and `μ_water_ref` contract as [`calibrate_bhc_water`](@ref)
is returned.
"""
function calibrate_pcct_poly_bhc(
        ws::PCCTWorkspace;
        order::Int = 5,
        max_path_cm::Real = 50.0,
        n_points::Int = 100,
    )
    n_E = length(ws.energies)
    energies = Float64.(ws.energies)
    W = Float64.(Array(ws.W_matrix_gpu))[1:n_E, :]
    w_detected = vec(sum(W; dims = 2))
    sum(w_detected) > 0 || error("PCCT detected spectrum has zero total weight")
    weights_per_col = repeat(reshape(w_detected, :, 1), 1, ws.geom.n_cols)
    reference_energy_keV = sum(energies .* w_detected) / sum(w_detected)
    return calibrate_bhc_water(
        energies, weights_per_col;
        order, max_path_cm, n_points, reference_energy_keV,
    )
end

export calibrate_pcct_poly_bhc

# =============================================================================
# HIRReconWorkspace — Pre-allocated workspace for zero-allocation Hybrid IR reconstruct!()
# =============================================================================

# Cone rays through a requested axial ROI continue through the object beyond
# the ROI's terminal planes.  The iterative forward model must represent that
# chord or it implicitly inserts air immediately outside the saved volume and
# creates a systematic terminal residual.  Keep the requested voxel spacing
# and expand symmetrically by the worst-case near-to-far cone magnification.
# This is an internal computational domain; `ws.volume` remains the exact grid
# requested by the caller.
function _hir_axial_support(
        geom::CTGeometry, volume_size::NTuple{3, Int}, enabled::Bool,
    )
    nx, ny, nz = volume_size
    if !enabled || is_helical(geom) || nz == 1
        return geom, volume_size, 1:nz
    end

    radius = min(geom.fov[1], geom.fov[2]) / 2
    radius < geom.SAD || throw(ArgumentError(
        "HIR reconstruction radius $radius cm must be smaller than SAD $(geom.SAD) cm"))
    cone_ratio = (geom.SAD + radius) / (geom.SAD - radius)
    dz = geom.fov[3] / nz
    work_nz = max(nz, ceil(Int, geom.fov[3] * cone_ratio / dz))
    # Equal halo on both sides preserves the requested voxel-center coordinates.
    isodd(work_nz - nz) && (work_nz += 1)
    work_nz == nz && return geom, volume_size, 1:nz

    work_fov = (geom.fov[1], geom.fov[2], work_nz * dz)
    work_geom = CTGeometry(
        geom.SAD, geom.SDD, geom.n_angles, geom.n_rows, geom.n_cols,
        geom.pixel_size, geom.pixel_row_size,
        geom.angles, geom.source_positions, geom.detector_centers,
        geom.detector_u, geom.detector_v, work_fov,
        geom.pitch, geom.table_feed, geom.detector_shape,
    )
    z0 = (work_nz - nz) ÷ 2 + 1
    return work_geom, (nx, ny, work_nz), z0:(z0 + nz - 1)
end

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
mutable struct HIRReconWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}, A2 <: AbstractArray{T, 2}, A1 <: AbstractArray{T, 1}, AI <: AbstractVector{Int32}}
    # ─── Output / iterate ───
    volume::A3                # exact caller-requested output grid
    work_volume::A3           # PWLS iterate with internal axial support halo
    work_geom::CTGeometry
    output_z::UnitRange{Int}

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
    # `data_weights` holds W_proj ⊙ stat_weights, folded once per reconstruct!
    # call (the product is loop-invariant), so the subset residual kernel reads
    # one array instead of two.
    data_weights::A3          # W_proj ⊙ statistical weights (sino shape)
    correction::A3            # Backprojection scratch (vol shape)
    reg_grad::A3              # Regularization gradient (vol shape)

    # ─── Ordered subsets (pre-computed) ───
    subsets::Vector{Vector{Int}}            # angle indices per subset
    subset_geometries::Vector{CTGeometry}   # pre-built geometry per subset
    subset_geom_source_positions::Vector{A2}  # GPU geometry arrays per subset
    subset_geom_detector_centers::Vector{A2}
    subset_geom_detector_u::Vector{A2}
    subset_geom_detector_v::Vector{A2}
    subset_angle_idx::Vector{AI}            # device-side global angle index per subset
    subset_Ax_buf::A3                       # (n_cols, n_rows, max_subset_size)

    # ─── Pre-computed HIR params ───
    params::HIRParams

    # ─── Forward-projection ray tracer (must match the sim's projector) ───
    projector::Symbol         # :dd_fast (default), :dd (DEPRECATED), or :siddon
end

"""
    create_hir_recon_workspace(sinogram, geom, volume_size; T=eltype(sinogram), strength=60, filter=StandardFilter(), cutoff=1.0)

Create a pre-allocated workspace for zero-allocation Hybrid IR `reconstruct!()`.

`strength` is the IR strength **percentage**, `0`–`100` in steps of 10 — the one
dial for the whole HIR pipeline (`0` = pure FBP, `60` = standard clinical,
`100` = maximum noise reduction).  See [`get_hir_params`](@ref) for the
strength → noise-reduction table.

`filter` can be a `FilterType` struct or a `Symbol` (e.g., `:standard`).

`projector` (`:dd_fast` default, `:dd` DEPRECATED, or `:siddon`) selects the forward ray tracer used
for the IR system matrix (`A·x` and `W = 1/(A·1)`).  Set it to the SAME value
as the simulation's `SimOptions(; projector=…)` so the recon inverts the
operator that generated the data.
"""
function create_hir_recon_workspace(
        sinogram::AbstractArray{<:AbstractFloat, 3},
        geom::CTGeometry,
        volume_size::NTuple{3, Int};
        T::Type{<:AbstractFloat} = eltype(sinogram),
        strength::Integer = 60,
        filter::Union{FilterType, Symbol} = StandardFilter(),
        cutoff::Float64 = 1.0,
        projector::Symbol = :dd_fast
    )
    _validate_projector(projector)
    filter = filter isa Symbol ? filter_from_symbol(filter) : filter
    sino_shape = size(sinogram)

    # Strength zero is contractually plain FBP and needs no iterative support
    # domain.  Helical HIR retains its existing WFBP-domain behavior.
    params = get_hir_params(strength)
    work_geom, work_size, output_z = _hir_axial_support(
        geom, volume_size, params.nepochs > 0)

    # Exact public output plus the internal iterative domain.  Alias them when
    # no halo is needed so strength=0/helical paths allocate nothing extra.
    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))
    work_volume = work_size == volume_size ? volume : similar(sinogram, T, work_size...)
    work_volume === volume || fill!(work_volume, zero(T))

    # FDK filtering scratch
    filtered = similar(sinogram, T, sino_shape...)
    conv_scratch = similar(sinogram, T, sino_shape...)

    # Pre-compute filter kernel on GPU
    pixel_size = T(geom.pixel_size)
    n_cols = size(sinogram, 1)
    raw_size = max(Int(ceil(2 * n_cols * cutoff)), 64)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), 2 * n_cols - 1)
    kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size)
    if is_arc(geom) && !is_helical(geom)
        # equiangular fan filter correction (helical WFBP filters rebinned
        # PARALLEL rows, which need the plain ramp)
        equiangular_kernel_scale!(kernel_cpu, geom.pixel_size / geom.SAD)
    end
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

    # Pre-compute SIRT-style normalization weights (W_proj and V_inv).
    # Each is one full projection, so `like = sinogram` pins them to the
    # backend that already owns the data — computing them on the host and
    # copying over dominated workspace construction (~83 s of 99 s at
    # 834×8×500 → 512×512×8).  W_proj = 1/(A·1) uses the selected forward
    # projector so the system matrix matches the sim's.
    W_proj = compute_projection_weights(work_geom, work_size, T; projector = projector, like = sinogram)
    V_inv = compute_image_weights(work_geom, work_size, T; like = sinogram)

    # PWLS iteration scratch buffers
    data_weights = similar(sinogram, T, sino_shape...)
    correction = similar(sinogram, T, work_size...)
    reg_grad = similar(sinogram, T, work_size...)

    # Ordered subsets pre-computation.  `get_hir_params` always returns
    # n_subsets = 12 at every strength, so the OS-PWLS path is the only path.
    # (At strength = 0 the subsets go unused: nepochs = 0 short-circuits to FBP.)
    n_subsets = params.n_subsets
    n_subsets > 0 || error("HIRParams.n_subsets must be > 0; got $n_subsets")
    subsets = create_ordered_subsets(geom.n_angles, n_subsets)
    subset_geometries = [create_subset_geometry(work_geom, indices) for indices in subsets]

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

    # Device-side global angle index per subset.  The residual kernel reads the
    # full sinogram / data_weights through this map instead of copying each
    # subset into a staging buffer every sub-iteration (that was 3 copies ×
    # n_subsets × nepochs, ~23 % of `reconstruct!`, and 3 extra sinogram-subset
    # buffers).  Int32 keeps the index small; nc·nr·n_angles ≪ typemax(Int32).
    idx_proto = similar(sinogram, Int32, 0)
    subset_angle_idx = Vector{typeof(idx_proto)}(undef, n_subsets)
    for s in 1:n_subsets
        subset_angle_idx[s] = similar(sinogram, Int32, length(subsets[s]))
        copyto!(subset_angle_idx[s], Int32.(subsets[s]))
    end

    # Only the forward-projection target still needs a staging buffer.
    max_subset_size = maximum(length(s) for s in subsets)
    subset_Ax_buf = similar(sinogram, T, sino_shape[1], sino_shape[2], max_subset_size)

    return HIRReconWorkspace{T, typeof(volume), typeof(geom_source_positions), typeof(filter_kernel), typeof(idx_proto)}(
        volume, work_volume, work_geom, output_z,
        filtered, conv_scratch, filter_kernel,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        W_proj, V_inv,
        data_weights, correction, reg_grad,
        subsets, subset_geometries,
        subset_geom_src, subset_geom_det, subset_geom_u, subset_geom_v,
        subset_angle_idx, subset_Ax_buf,
        params,
        projector
    )
end
