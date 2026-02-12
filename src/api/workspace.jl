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
export EICTDualWorkspace, create_eict_dual_workspace
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
    scratch::A3                # ONE buffer for all neighbor kernels
    total_counts::A3           # for anti-coincidence (sum across bins)

    # ─── Combine (GPU-side) ───
    combined::A3               # _combine_pcct_bins output (reused ideal + noisy)

    # ─── VMI decomposition (GPU-side, unified with dual-kVp) ───
    vmi_sino_low::A3           # combined low-energy sinogram (bins 1:2 averaged)
    vmi_sino_high::A3          # combined high-energy sinogram (bins 3:4 averaged)
    vmi_material1::A3          # first basis material map (GPU)
    vmi_material2::A3          # second basis material map (GPU)
    vmi_inv_a11::T             # pre-computed 2×2 inverse decomposition matrix
    vmi_inv_a12::T
    vmi_inv_a21::T
    vmi_inv_a22::T
    vmi_sino::A3               # VMI synthesis output (GPU, reused across energies)

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
    charge_sharing_probs::Vector{Float64}     # per-bin charge sharing probabilities
    pileup_counts::Vector{Float64}            # per-bin mean counts for pileup
    pileup_migration::Matrix{Float64}         # spectral migration matrix (n_bins × n_bins)
    correction_pileup_counts::Vector{Float64} # correction path counts
    correction_migration::Matrix{Float64}     # correction migration matrix
    μ_values::Vector{T}                       # VMI attenuation coefficients (n_materials)

    # ─── Noise I0 ───
    noise_I0::Vector{Float64}   # per-bin I0 for noise model (n_bins)

    # ─── Pileup scratch buffers (for compute_spectral_migration_matrix) ───
    pileup_S::Matrix{Float64}           # migration matrix scratch (n_bins × n_bins)
    pileup_thresh::Vector{Float64}      # Float64 copy of thresholds (n_bins)
    pileup_E_low::Vector{Float64}       # bin lower edges (n_bins)
    pileup_E_high::Vector{Float64}      # bin upper edges (n_bins)
    pileup_E_centers::Vector{Float64}   # bin center energies (n_bins)
    pileup_w::Vector{Float64}           # normalized bin weights (n_bins)

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

    # ─── Pre-computed setup data (computed once, reused) ───
    geom::CTGeometry                         # CT geometry
    energies::Vector{Float64}                # downsampled spectrum energies
    weights::Vector{Float64}                 # downsampled spectrum weights
    config::PhysicsConfig                    # physics configuration
    pcct_detector::PhotonCountingDetector{Float64}  # PCCT detector
    mats::Vector{XA.Material}                # resolved materials
    use_detector_fx::Bool                    # whether detector effects are applied
    use_corrections::Bool                    # whether corrections are applied
    kVp::Float64                             # max energy (kVp)
    basis_tuple::Tuple                       # VMI basis materials tuple
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
- `sim_opts`: Simulation options (provides n_energy_bins)
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
    sino_shape = (scanner.detector_cols, scanner.detector_rows, protocol.views)
    vol_shape = size(phantom.mask)
    n_bins = length(scanner.energy_thresholds)
    n_materials = length(recon_opts.vmi_basis)
    n_elements = prod(sino_shape)
    n_energies = sim_opts.n_energy_bins

    # GPU-side buffers — similar() matches the provided mask's backend
    # If mask kwarg is provided (e.g. GPU-converted mask), use it for similar();
    # otherwise fall back to phantom.mask (CPU arrays)
    ref_mask = mask === nothing ? phantom.mask : mask
    bins = [similar(ref_mask, T, sino_shape) for _ in 1:n_bins]
    μ_volume = similar(ref_mask, T, vol_shape)
    sino_buf = similar(ref_mask, T, sino_shape)
    scratch = similar(ref_mask, T, sino_shape)
    total_counts = similar(ref_mask, T, sino_shape)
    combined = similar(ref_mask, T, sino_shape)
    enoise_gpu = similar(ref_mask, T, n_elements)

    # VMI decomposition buffers (GPU-side, unified with dual-kVp)
    vmi_sino_low = similar(ref_mask, T, sino_shape)
    vmi_sino_high = similar(ref_mask, T, sino_shape)
    vmi_material1 = similar(ref_mask, T, sino_shape)
    vmi_material2 = similar(ref_mask, T, sino_shape)
    vmi_sino = similar(ref_mask, T, sino_shape)

    # CPU-side buffers
    noise_staging = zeros(T, sino_shape)
    noise_buf = zeros(T, sino_shape)
    enoise_cpu = Vector{T}(undef, n_elements)
    sino_ideal_out = zeros(T, sino_shape)
    sino_noisy_out = zeros(T, sino_shape)

    # --- Pre-compute setup data (done once, reused by simulate!()) ---
    geom = CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm, z_cm=nothing)
    e_full, w_full = load_spectrum(Int(protocol.kVp))
    energies, weights_vec = downsample_spectrum(e_full, w_full, sim_opts.n_energy_bins)
    config = build_physics_config(scanner, sim_opts, energies, weights_vec; phantom=phantom)
    pcct_detector = _build_pcct_detector(scanner)
    mats = _resolve_materials(phantom, materials)
    use_detector_fx = sim_opts.fidelity in (:medium, :high, :pcct)
    use_corrections = sim_opts.use_pcct_corrections
    kVp = Float64(maximum(energies))
    thresholds = pcct_detector.energy_thresholds_keV
    basis_tuple = Tuple(recon_opts.vmi_basis)

    # Pre-compute spectral response data
    η_vec = quantum_efficiency_vector(pcct_detector.material, pcct_detector.thickness_mm, energies)
    R_mat = compute_spectral_response_matrix(
        pcct_detector.material, pcct_detector.thickness_mm, thresholds, kVp;
        energy_resolution_keV=pcct_detector.energy_resolution_keV,
        pixel_size_mm=pcct_detector.pixel_size_mm,
        include_fluorescence=true, include_tailing=true,
        n_energy_points=length(energies)
    )
    R_energies_vec = collect(range(1.0, kVp, length=n_energies))

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
    charge_sharing_probs = zeros(Float64, n_bins)
    pileup_counts = zeros(Float64, n_bins)
    pileup_migration = zeros(Float64, n_bins, n_bins)
    correction_pileup_counts = zeros(Float64, n_bins)
    correction_migration = zeros(Float64, n_bins, n_bins)
    μ_values = zeros(T, n_materials)
    noise_I0 = zeros(Float64, n_bins)

    # Pileup scratch buffers (for compute_spectral_migration_matrix)
    pileup_S = zeros(Float64, n_bins, n_bins)
    pileup_thresh = Float64.(thresholds)
    pileup_E_low = zeros(Float64, n_bins)
    pileup_E_high = zeros(Float64, n_bins)
    for b in 1:n_bins
        pileup_E_low[b] = pileup_thresh[b]
        pileup_E_high[b] = b < n_bins ? pileup_thresh[b+1] : Float64(kVp)
    end
    pileup_E_centers = zeros(Float64, n_bins)
    for b in 1:n_bins
        pileup_E_centers[b] = (pileup_E_low[b] + pileup_E_high[b]) / 2.0
    end
    pileup_w = zeros(Float64, n_bins)

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
    bhc_n_coeffs = config.bhc !== nothing ? length(config.bhc.coefficients) : 1
    bhc_coeffs_cpu = if config.bhc !== nothing
        T.(config.bhc.coefficients)
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

    # Pre-compute charge sharing probabilities (avoid re-computing each call)
    if use_detector_fx && pcct_detector.enable_charge_sharing && pcct_detector.material == CDTE_MATERIAL
        _cs_geom = PCCTDetectorGeometry(
            "runtime",
            Float64.(pcct_detector.pixel_size_mm),
            Float64(pcct_detector.thickness_mm),
            800.0, 680.0, 1.5, 3.0
        )
        _cs_fluor = compute_cdte_fluorescence_model(pcct_detector.pixel_size_mm, pcct_detector.thickness_mm)
        for b in 1:n_bins
            E_low = thresholds[b]
            E_high = b < n_bins ? thresholds[b+1] : 140.0
            E_center = (E_low + E_high) / 2.0
            σ = mean_charge_cloud_sigma_mm(E_center, _cs_geom)
            p_cloud = charge_sharing_probability(σ, pcct_detector.pixel_size_mm)
            p_fluor = fluorescence_sharing_boost(E_center, _cs_fluor)
            charge_sharing_probs[b] = min(p_cloud + p_fluor, 0.7)
        end
    end

    # Pre-compute 2×2 decomposition matrix for pseudo-dual-energy VMI
    max_keV = 120.0
    _bin_energies = T.(compute_pcct_bin_energies(thresholds; max_keV=max_keV, kvp=Int(protocol.kVp)))
    _split_bin = 2
    E_low = Float64(sum(_bin_energies[1:_split_bin]) / _split_bin)
    E_high = Float64(sum(_bin_energies[_split_bin+1:end]) / (n_bins - _split_bin))
    _basis_2 = length(basis_tuple) >= 2 ? (basis_tuple[1], basis_tuple[2]) : (:water, :iodine)
    vmi_inv_a11, vmi_inv_a12, vmi_inv_a21, vmi_inv_a22 = compute_decomposition_matrix(_basis_2, E_low, E_high; T=T)

    return PCCTWorkspace{T, typeof(sino_buf), typeof(enoise_gpu), typeof(geom_source_positions)}(
        bins, μ_volume, sino_buf, scratch, total_counts,
        combined,
        vmi_sino_low, vmi_sino_high, vmi_material1, vmi_material2,
        vmi_inv_a11, vmi_inv_a12, vmi_inv_a21, vmi_inv_a22,
        vmi_sino,
        noise_staging, noise_buf, enoise_cpu, enoise_gpu,
        sino_ideal_out, sino_noisy_out,
        η_vec, R_mat, R_energies_vec, I0_bins_combine, I0_bins_norm_vec, thresholds_T_vec, rng,
        charge_sharing_probs, pileup_counts, pileup_migration,
        correction_pileup_counts, correction_migration, μ_values, noise_I0,
        pileup_S, pileup_thresh, pileup_E_low, pileup_E_high, pileup_E_centers, pileup_w,
        μ_lut_cpu, μ_lut_gpu, μ_table,
        bhc_coeffs_cpu, bhc_coeffs_gpu,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        geom, energies, weights_vec, config, pcct_detector, mats,
        use_detector_fx, use_corrections, kVp, basis_tuple
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
    scatter_kernel::Union{Nothing, A2}          # scatter convolution kernel
    scatter_correct_kernel::Union{Nothing, A2}  # scatter correction kernel
    crosstalk_kernel::Union{Nothing, A2}        # 3×3 crosstalk kernel
    optical_crosstalk_kernel::Union{Nothing, A2} # 3×3 optical crosstalk kernel
    focal_spot_kernel::Union{Nothing, A2}       # focal spot blur kernel
    flat_filter_projection::Union{Nothing, A2}  # 2D flat filter projection (n_cols × n_rows)
    bowtie_projection::Union{Nothing, A2}       # 2D bowtie projection (n_cols × n_rows)
    lag_coeffs::Union{Nothing, A1}              # lag coefficients (n_frames)

    # ─── Noise (CPU + GPU) ───
    noise_rand_cpu::Vector{T}  # randn output (n_elements)
    noise_rand_gpu::A1         # GPU transfer buffer (n_elements)

    # ─── Pre-computed vectors ───
    weights_norm::Vector{T}    # T.(weights ./ sum(weights))
    μ_lut_cpu::Vector{T}       # μ LUT CPU buffer (n_regions)
    μ_lut_gpu::A1              # μ LUT GPU buffer (matches mask backend)
    μ_table::Matrix{T}         # pre-computed μ[region, energy] (n_regions × n_energies)
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
    heel_effect::Union{Nothing, HeelEffect}
    das_model::Union{Nothing, DASModel}
    bhc::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}}
    has_signal_chain::Bool

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
    geom = CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm, z_cm=nothing)

    # Spectrum
    energies, weights_vec = resolve_spectrum(sim_opts, protocol)
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

    crosstalk_kernel = if config.crosstalk !== nothing
        k_cpu = T.(create_crosstalk_kernel_3x3(config.crosstalk))
        k_gpu = similar(ref, T, 3, 3)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

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

    flat_filter_proj = if config.flat_filter !== nothing
        transmission_cpu = compute_flat_filter_attenuation(config.flat_filter, geom; energy_keV=config.energy_keV)
        fp_cpu = T.(-log.(transmission_cpu))
        fp_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(fp_gpu, fp_cpu)
        fp_gpu
    else
        nothing
    end

    bowtie_proj = if config.bowtie_filter !== nothing
        transmission_cpu = compute_bowtie_attenuation(config.bowtie_filter, geom; energy_keV=config.energy_keV)
        bp_cpu = T.(-log.(transmission_cpu))
        bp_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(bp_gpu, bp_cpu)
        bp_gpu
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

    # BHC coefficients
    bhc_coeffs_cpu = if config.bhc !== nothing
        T.(config.bhc.coefficients)
    else
        zeros(T, 1)
    end
    bhc_coeffs_gpu = similar(ref, T, length(bhc_coeffs_cpu))
    copyto!(bhc_coeffs_gpu, bhc_coeffs_cpu)

    # Extract signal chain config from PhysicsConfig (pre-computed)
    heel = config.heel_effect
    das = config.das_model
    bhc_effect = config.bhc
    has_sc = heel !== nothing || das !== nothing || bhc_effect !== nothing

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
        scatter_kernel, scatter_correct_kernel, crosstalk_kernel,
        optical_crosstalk_kernel, focal_spot_kernel, flat_filter_proj,
        bowtie_proj, lag_coeffs_buf,
        noise_rand_cpu, noise_rand_gpu,
        weights_norm, μ_lut_cpu, μ_lut_gpu, μ_table, bhc_coeffs_gpu,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        geom, energies, weights_vec, config, mats, rng,
        heel, das, bhc_effect, has_sc,
        sino_ideal_out, sino_noisy_out
    )
end

# =============================================================================
# EICTDualWorkspace — Pre-allocated workspace for zero-allocation EICT dual-kVp simulate!()
# =============================================================================
#
# Dual-kVp Energy-Integrating CT. Runs single-kVp pipeline twice (low + high),
# then material decomposition. Shared scratch buffers (μ_volume, sino_mono,
# I_transmitted) between passes since they run sequentially.

"""
    EICTDualWorkspace{T, A3, A2, A1}

Pre-allocated workspace for zero-allocation EICT dual-kVp simulation.

Type parameters:
- `T`: Element type (typically Float32)
- `A3`: 3D array type matching GPU backend
- `A2`: 2D array type matching GPU backend
- `A1`: 1D array type matching GPU backend

Key design: μ_volume, sino_mono, I_transmitted are SHARED between low and high kVp
passes because they run sequentially. This saves 3 large buffer allocations.
"""
mutable struct EICTDualWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A2<:AbstractArray{T,2}, A1<:AbstractArray{T,1}}
    # ─── Forward projection scratch (GPU-side, SHARED between low/high) ───
    μ_volume::A3          # attenuation volume, reused per energy (nx, ny, nz)
    sino_mono::A3         # monochromatic sinogram scratch (sino shape)
    I_transmitted::A3     # Beer-Lambert accumulator (sino shape)

    # ─── Per-pass output (GPU-side) ───
    sino_low::A3          # low-kVp forward projection output (sino shape)
    sino_high::A3         # high-kVp forward projection output (sino shape)

    # ─── Signal chain scratch (GPU-side, SHARED) ───
    air_scan::A3          # CatSim air scan buffer (sino shape)

    # ─── Physics effect scratch buffers (GPU-side, shared) ───
    physics_output::A3    # sinogram-sized scratch for convolution effects
    lag_intensity::A3     # sinogram-sized scratch for lag intensity

    # ─── Pre-computed physics kernels (GPU-side, SHARED — energy-independent) ───
    scatter_kernel::Union{Nothing, A2}
    scatter_correct_kernel::Union{Nothing, A2}
    crosstalk_kernel::Union{Nothing, A2}
    optical_crosstalk_kernel::Union{Nothing, A2}
    focal_spot_kernel::Union{Nothing, A2}
    lag_coeffs::Union{Nothing, A1}

    # ─── Per-kVp pre-computed projections (GPU-side, energy-dependent) ───
    flat_filter_projection_low::Union{Nothing, A2}
    flat_filter_projection_high::Union{Nothing, A2}
    bowtie_projection_low::Union{Nothing, A2}
    bowtie_projection_high::Union{Nothing, A2}

    # ─── Noise (CPU + GPU, reused between low/high) ───
    noise_rand_cpu::Vector{T}
    noise_rand_gpu::A1

    # ─── Material decomposition output (GPU-side) ───
    material1::A3         # first basis material (sino shape)
    material2::A3         # second basis material (sino shape)

    # ─── Pre-computed vectors (per-kVp) ───
    weights_norm_low::Vector{T}
    weights_norm_high::Vector{T}
    μ_lut_cpu::Vector{T}       # shared (same region count)
    μ_lut_gpu::A1              # shared
    μ_table_low::Matrix{T}     # μ[region, energy] for low kVp
    μ_table_high::Matrix{T}    # μ[region, energy] for high kVp
    bhc_coeffs_gpu_low::A1     # BHC coefficients for low kVp
    bhc_coeffs_gpu_high::A1    # BHC coefficients for high kVp

    # ─── Pre-computed geometry arrays (T-typed, same backend as mask) ───
    geom_source_positions::A2
    geom_detector_centers::A2
    geom_detector_u::A2
    geom_detector_v::A2

    # ─── Pre-computed setup data ───
    geom::CTGeometry
    energies_low::Vector{Float64}
    weights_low::Vector{Float64}
    energies_high::Vector{Float64}
    weights_high::Vector{Float64}
    config_low::PhysicsConfig
    config_high::PhysicsConfig
    mats::Vector
    rng::MersenneTwister

    # ─── Signal chain config (per-kVp, extracted from PhysicsConfig) ───
    heel_effect::Union{Nothing, HeelEffect}
    das_model::Union{Nothing, DASModel}
    bhc_low::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}}
    bhc_high::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}}
    has_signal_chain::Bool

    # ─── Pre-computed decomposition matrix inverse elements ───
    inv_a11::T
    inv_a12::T
    inv_a21::T
    inv_a22::T
    basis::Tuple{Symbol, Symbol}

    # ─── Result staging (CPU) ───
    sino_ideal_out_low::Array{T,3}    # CPU: ideal low-kVp sinogram
    sino_ideal_out_high::Array{T,3}   # CPU: ideal high-kVp sinogram
    sino_noisy_out_low::Array{T,3}    # CPU: noisy low-kVp sinogram
    sino_noisy_out_high::Array{T,3}   # CPU: noisy high-kVp sinogram
end

"""
    create_eict_dual_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T=Float32, materials=nothing)

Create a pre-allocated workspace for zero-allocation EICT dual-kVp `simulate!()`.
"""
function create_eict_dual_workspace(scanner, protocol, sim_opts, recon_opts, phantom;
                                     T::Type{<:AbstractFloat}=Float32,
                                     materials::Union{Nothing, Vector}=nothing)
    # Geometry (shared)
    geom = CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm, z_cm=nothing)

    # Create per-kVp protocols (same as _simulate_axial_dual)
    protocol_low = CTProtocol(
        mA = protocol.mA_low > 0 ? protocol.mA_low : protocol.mA,
        kVp = protocol.kVp_low,
        views = protocol.views,
        rotation_time = protocol.rotation_time,
        flux_density = protocol.flux_density,
        spectrum_path = nothing,
        dual_energy = false
    )
    protocol_high = CTProtocol(
        mA = protocol.mA,
        kVp = protocol.kVp,
        views = protocol.views,
        rotation_time = protocol.rotation_time,
        flux_density = protocol.flux_density,
        spectrum_path = nothing,
        dual_energy = false
    )

    # Spectra for both kVps
    energies_low, weights_low = resolve_spectrum(sim_opts, protocol_low)
    energies_high, weights_high = resolve_spectrum(sim_opts, protocol_high)

    # Physics configs for both kVps
    config_low = build_physics_config(scanner, sim_opts, energies_low, weights_low; phantom=phantom)
    config_high = build_physics_config(scanner, sim_opts, energies_high, weights_high; phantom=phantom)

    # Materials
    mats = _resolve_materials(phantom, materials)
    n_regions = length(mats)

    # Dimensions
    sino_shape = (geom.n_cols, geom.n_rows, geom.n_angles)
    vol_shape = size(phantom.mask)
    n_elements = prod(sino_shape)
    n_energies_low = length(energies_low)
    n_energies_high = length(energies_high)

    # Reference array for similar()
    ref = phantom.mask

    # ─── GPU-side buffers ───
    # Shared scratch
    μ_volume = similar(ref, T, vol_shape)
    sino_mono = similar(ref, T, sino_shape)
    I_transmitted = similar(ref, T, sino_shape)
    air_scan = similar(ref, T, sino_shape)
    physics_output = similar(ref, T, sino_shape)
    lag_intensity = similar(ref, T, sino_shape)

    # Per-pass output
    sino_low = similar(ref, T, sino_shape)
    sino_high = similar(ref, T, sino_shape)

    # Material decomposition output
    material1 = similar(ref, T, sino_shape)
    material2 = similar(ref, T, sino_shape)

    # ─── Shared physics kernels (energy-independent) ───
    # Use config_high for kernel computation (same as config_low for these)
    scatter_kernel = if config_high.scatter !== nothing
        k_cpu = T.(create_scatter_kernel_spatial(config_high.scatter))
        k_gpu = similar(ref, T, size(k_cpu)...)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    scatter_correct_kernel = if config_high.scatter_correction !== nothing
        sc_temp = ScatterModel(
            config_high.scatter_correction.correction_coefficient,
            config_high.scatter_correction.scale_factor,
            config_high.scatter_correction.kernel_fwhm,
            config_high.scatter_correction.kernel_type
        )
        k_cpu = T.(create_scatter_kernel_spatial(sc_temp))
        k_gpu = similar(ref, T, size(k_cpu)...)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    crosstalk_kernel = if config_high.crosstalk !== nothing
        k_cpu = T.(create_crosstalk_kernel_3x3(config_high.crosstalk))
        k_gpu = similar(ref, T, 3, 3)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    optical_crosstalk_kernel = if config_high.optical_crosstalk !== nothing
        k_cpu = T.(create_optical_crosstalk_kernel(config_high.optical_crosstalk))
        k_gpu = similar(ref, T, 3, 3)
        copyto!(k_gpu, k_cpu)
        k_gpu
    else
        nothing
    end

    focal_spot_kernel = if config_high.focal_spot !== nothing
        blur_fwhm = compute_focal_spot_blur_fwhm(config_high.focal_spot, geom, geom.SAD)
        if blur_fwhm[1] >= 0.1 || blur_fwhm[2] >= 0.1
            k_cpu = T.(create_focal_spot_kernel_spatial(config_high.focal_spot, blur_fwhm))
            k_gpu = similar(ref, T, size(k_cpu)...)
            copyto!(k_gpu, k_cpu)
            k_gpu
        else
            nothing
        end
    else
        nothing
    end

    lag_coeffs_buf = if config_high.lag !== nothing && !isempty(config_high.lag.amplitudes)
        n_frames = min(20, sino_shape[3])
        c_cpu = T.(compute_lag_coefficients(config_high.lag, n_frames))
        c_gpu = similar(ref, T, n_frames)
        copyto!(c_gpu, c_cpu)
        c_gpu
    else
        nothing
    end

    # ─── Per-kVp energy-dependent projections ───
    flat_filter_proj_low = if config_low.flat_filter !== nothing
        transmission_cpu = compute_flat_filter_attenuation(config_low.flat_filter, geom; energy_keV=config_low.energy_keV)
        fp_cpu = T.(-log.(transmission_cpu))
        fp_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(fp_gpu, fp_cpu)
        fp_gpu
    else
        nothing
    end

    flat_filter_proj_high = if config_high.flat_filter !== nothing
        transmission_cpu = compute_flat_filter_attenuation(config_high.flat_filter, geom; energy_keV=config_high.energy_keV)
        fp_cpu = T.(-log.(transmission_cpu))
        fp_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(fp_gpu, fp_cpu)
        fp_gpu
    else
        nothing
    end

    bowtie_proj_low = if config_low.bowtie_filter !== nothing
        transmission_cpu = compute_bowtie_attenuation(config_low.bowtie_filter, geom; energy_keV=config_low.energy_keV)
        bp_cpu = T.(-log.(transmission_cpu))
        bp_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(bp_gpu, bp_cpu)
        bp_gpu
    else
        nothing
    end

    bowtie_proj_high = if config_high.bowtie_filter !== nothing
        transmission_cpu = compute_bowtie_attenuation(config_high.bowtie_filter, geom; energy_keV=config_high.energy_keV)
        bp_cpu = T.(-log.(transmission_cpu))
        bp_gpu = similar(ref, T, sino_shape[1], sino_shape[2])
        copyto!(bp_gpu, bp_cpu)
        bp_gpu
    else
        nothing
    end

    # ─── Noise buffers ───
    noise_rand_cpu = Vector{T}(undef, n_elements)
    noise_rand_gpu = similar(ref, T, n_elements)

    # ─── Pre-computed weights ───
    w_sum_low = sum(weights_low)
    weights_norm_low = T.(weights_low ./ w_sum_low)
    w_sum_high = sum(weights_high)
    weights_norm_high = T.(weights_high ./ w_sum_high)

    # ─── μ lookup tables ───
    μ_lut_cpu = Vector{T}(undef, n_regions)
    μ_lut_gpu = similar(ref, T, n_regions)

    μ_table_low = zeros(T, n_regions, n_energies_low)
    for (e_idx, E) in enumerate(energies_low)
        for r in 1:n_regions
            μ_table_low[r, e_idx] = T(compute_μ_at_energy(mats[r], Float64(E)))
        end
    end

    μ_table_high = zeros(T, n_regions, n_energies_high)
    for (e_idx, E) in enumerate(energies_high)
        for r in 1:n_regions
            μ_table_high[r, e_idx] = T(compute_μ_at_energy(mats[r], Float64(E)))
        end
    end

    # ─── BHC coefficients (per-kVp) ───
    bhc_coeffs_cpu_low = if config_low.bhc !== nothing
        T.(config_low.bhc.coefficients)
    else
        zeros(T, 1)
    end
    bhc_coeffs_gpu_low = similar(ref, T, length(bhc_coeffs_cpu_low))
    copyto!(bhc_coeffs_gpu_low, bhc_coeffs_cpu_low)

    bhc_coeffs_cpu_high = if config_high.bhc !== nothing
        T.(config_high.bhc.coefficients)
    else
        zeros(T, 1)
    end
    bhc_coeffs_gpu_high = similar(ref, T, length(bhc_coeffs_cpu_high))
    copyto!(bhc_coeffs_gpu_high, bhc_coeffs_cpu_high)

    # ─── Geometry arrays ───
    geom_source_positions = similar(ref, T, size(geom.source_positions)...)
    copyto!(geom_source_positions, T.(geom.source_positions))
    geom_detector_centers = similar(ref, T, size(geom.detector_centers)...)
    copyto!(geom_detector_centers, T.(geom.detector_centers))
    geom_detector_u = similar(ref, T, size(geom.detector_u)...)
    copyto!(geom_detector_u, T.(geom.detector_u))
    geom_detector_v = similar(ref, T, size(geom.detector_v)...)
    copyto!(geom_detector_v, T.(geom.detector_v))

    # ─── Signal chain config (shared — heel/das are energy-independent) ───
    heel = config_high.heel_effect
    das = config_high.das_model
    bhc_eff_low = config_low.bhc
    bhc_eff_high = config_high.bhc
    has_sc = heel !== nothing || das !== nothing || bhc_eff_low !== nothing || bhc_eff_high !== nothing

    # ─── Pre-computed decomposition matrix inverse ───
    basis = length(recon_opts.vmi_basis) >= 2 ? Tuple(recon_opts.vmi_basis[1:2]) : (:water, :iodine)
    e_eff_low = get_effective_energy(Int(protocol.kVp_low))
    e_eff_high = get_effective_energy(Int(protocol.kVp))
    inv_a11, inv_a12, inv_a21, inv_a22 = compute_decomposition_matrix(basis, e_eff_low, e_eff_high; T=T)

    # RNG
    rng = MersenneTwister(0)

    # CPU staging (both low and high kVp)
    sino_ideal_out_low = zeros(T, sino_shape)
    sino_ideal_out_high = zeros(T, sino_shape)
    sino_noisy_out_low = zeros(T, sino_shape)
    sino_noisy_out_high = zeros(T, sino_shape)

    return EICTDualWorkspace{T, typeof(sino_low), typeof(geom_source_positions), typeof(noise_rand_gpu)}(
        μ_volume, sino_mono, I_transmitted,
        sino_low, sino_high,
        air_scan, physics_output, lag_intensity,
        scatter_kernel, scatter_correct_kernel, crosstalk_kernel,
        optical_crosstalk_kernel, focal_spot_kernel, lag_coeffs_buf,
        flat_filter_proj_low, flat_filter_proj_high,
        bowtie_proj_low, bowtie_proj_high,
        noise_rand_cpu, noise_rand_gpu,
        material1, material2,
        weights_norm_low, weights_norm_high,
        μ_lut_cpu, μ_lut_gpu, μ_table_low, μ_table_high,
        bhc_coeffs_gpu_low, bhc_coeffs_gpu_high,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        geom, energies_low, weights_low, energies_high, weights_high,
        config_low, config_high, mats, rng,
        heel, das, bhc_eff_low, bhc_eff_high, has_sc,
        inv_a11, inv_a12, inv_a21, inv_a22, basis,
        sino_ideal_out_low, sino_ideal_out_high, sino_noisy_out_low, sino_noisy_out_high
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
    create_fdk_recon_workspace(sinogram, geom, volume_size; T=eltype(sinogram), filter=RampFilter(), cutoff=1.0)

Create a pre-allocated workspace for zero-allocation FDK `reconstruct!()`.
"""
function create_fdk_recon_workspace(
    sinogram::AbstractArray{<:AbstractFloat, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    T::Type{<:AbstractFloat} = eltype(sinogram),
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
)
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
    create_hir_recon_workspace(sinogram, geom, volume_size; T=eltype(sinogram), strength=3, filter=RampFilter(), cutoff=1.0)

Create a pre-allocated workspace for zero-allocation Hybrid IR `reconstruct!()`.
"""
function create_hir_recon_workspace(
    sinogram::AbstractArray{<:AbstractFloat, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    T::Type{<:AbstractFloat} = eltype(sinogram),
    strength::Int = 3,
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
)
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
