# =============================================================================
# Zero-Allocation foreachindex — CPU plain loop / GPU via AK.foreachindex
# =============================================================================
#
# AK.foreachindex on CPU with multithreading allocates ~5,888 bytes per call
# for task spawning overhead. On GPU backends (Metal, CUDA, ROCm), this overhead
# is minimal (~16 bytes) and handled by KernelAbstractions.jl.
#
# _foreachindex! dispatches:
#   - Array (CPU) → plain @inbounds for loop (zero allocations)
#   - GPU arrays  → AK.foreachindex (backend-agnostic via KA.jl)
#
# This is the key to achieving @allocated == 0 on CPU while preserving
# GPU acceleration when a GPU backend is loaded.

"""
    _foreachindex!(f, A::Array)

Zero-allocating CPU version: plain sequential loop over all indices.
"""
@inline function _foreachindex!(f, A::Array)
    @inbounds for idx in eachindex(A)
        f(idx)
    end
    return nothing
end

"""
    _foreachindex!(f, A::AbstractArray)

GPU fallback: delegates to AK.foreachindex for any non-Array type
(e.g., MtlArray, CuArray, ROCArray — detected via AcceleratedKernels.jl).
"""
@inline function _foreachindex!(f, A::AbstractArray)
    AK.foreachindex(f, A)
    return nothing
end

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

    # ─── VMI synthesis (CPU-side, matches material_maps backend) ───
    vmi_sino::Array{T,3}       # synthesize_vmi output (CPU, reused across energies)

    # ─── Noise CPU staging (Phase 1: CPU RNG) ───
    noise_staging::Array{T,3}  # CPU buffer for GPU↔CPU noise transfer
    noise_buf::Array{T,3}      # randn output buffer (CPU)

    # ─── Electronic noise ───
    enoise_cpu::Vector{T}      # randn output for electronic noise (CPU, flat)
    enoise_gpu::A1             # electronic noise GPU transfer buffer (flat)

    # ─── Material decomposition (CPU) ───
    bins_cpu::Vector{Array{T,3}}       # CPU copies of bins for decomposition
    material_maps::Vector{Array{T,3}}  # n_materials output arrays (CPU)
    decomp_pixel_buf::Vector{T}        # per-pixel gather buffer (n_bins)

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

    # ─── Pre-computed decomposition data ───
    A_pinv::Matrix{T}                        # pseudo-inverse for material decomposition
    basis_vec::Vector{Symbol}                # basis material symbols

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

    # VMI sinogram — CPU (matches material_maps backend)
    vmi_sino = zeros(T, sino_shape)

    # CPU-side buffers
    noise_staging = zeros(T, sino_shape)
    noise_buf = zeros(T, sino_shape)
    enoise_cpu = Vector{T}(undef, n_elements)
    bins_cpu = [zeros(T, sino_shape) for _ in 1:n_bins]
    material_maps_buf = [zeros(T, sino_shape) for _ in 1:n_materials]
    decomp_pixel_buf = zeros(T, n_bins)
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

    # Pre-compute material decomposition pseudo-inverse (avoids allocation each call)
    basis_vec = collect(Symbol, basis_tuple)
    max_keV = 120.0
    _bin_energies = zeros(T, n_bins)
    for i in 1:n_bins
        lower = thresholds[i]
        upper = i < n_bins ? thresholds[i+1] : max_keV
        _bin_energies[i] = T((lower + upper) / 2)
    end
    _A_mat = zeros(T, n_bins, n_materials)
    for (j, mat_sym) in enumerate(basis_vec)
        for i in 1:n_bins
            _A_mat[i, j] = T(get_material_attenuation_pcct(mat_sym, Float64(_bin_energies[i])))
        end
    end
    A_pinv_mat = Matrix{T}(pinv(_A_mat))

    return PCCTWorkspace{T, typeof(sino_buf), typeof(enoise_gpu), typeof(geom_source_positions)}(
        bins, μ_volume, sino_buf, scratch, total_counts,
        combined, vmi_sino,
        noise_staging, noise_buf, enoise_cpu, enoise_gpu,
        bins_cpu, material_maps_buf, decomp_pixel_buf,
        sino_ideal_out, sino_noisy_out,
        η_vec, R_mat, R_energies_vec, I0_bins_combine, I0_bins_norm_vec, thresholds_T_vec, rng,
        charge_sharing_probs, pileup_counts, pileup_migration,
        correction_pileup_counts, correction_migration, μ_values, noise_I0,
        pileup_S, pileup_thresh, pileup_E_low, pileup_E_high, pileup_E_centers, pileup_w,
        μ_lut_cpu, μ_lut_gpu, μ_table,
        bhc_coeffs_cpu, bhc_coeffs_gpu,
        geom_source_positions, geom_detector_centers, geom_detector_u, geom_detector_v,
        A_pinv_mat, basis_vec,
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
mutable struct EICTWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A1<:AbstractArray{T,1}}
    # ─── Forward projection (GPU-side) ───
    sinogram::A3          # output sinogram (n_cols, n_rows, n_angles)
    μ_volume::A3          # attenuation volume, reused per energy (nx, ny, nz)
    sino_mono::A3         # monochromatic sinogram scratch (sino shape)
    I_transmitted::A3     # Beer-Lambert accumulator (sino shape)

    # ─── Signal chain scratch (GPU-side) ───
    air_scan::A3          # CatSim air scan buffer (sino shape)

    # ─── Noise (CPU + GPU) ───
    noise_rand_cpu::Vector{T}  # randn output (n_elements)
    noise_rand_gpu::A1         # GPU transfer buffer (n_elements)

    # ─── Pre-computed vectors ───
    weights_norm::Vector{T}    # T.(weights ./ sum(weights))
    μ_lut_cpu::Vector{T}       # μ LUT CPU buffer (n_regions)
    μ_lut_gpu::A1              # μ LUT GPU buffer (matches mask backend)
    μ_table::Matrix{T}         # pre-computed μ[region, energy] (n_regions × n_energies)
    bhc_coeffs_gpu::A1         # BHC polynomial coefficients (GPU/backend)

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

    # RNG
    rng = MersenneTwister(0)

    # CPU staging
    sino_ideal_out = zeros(T, sino_shape)
    sino_noisy_out = zeros(T, sino_shape)

    return EICTWorkspace{T, typeof(sinogram), typeof(noise_rand_gpu)}(
        sinogram, μ_volume, sino_mono, I_transmitted, air_scan,
        noise_rand_cpu, noise_rand_gpu,
        weights_norm, μ_lut_cpu, μ_lut_gpu, μ_table, bhc_coeffs_gpu,
        geom, energies, weights_vec, config, mats, rng,
        heel, das, bhc_effect, has_sc,
        sino_ideal_out, sino_noisy_out
    )
end
