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

"""
    PCCTWorkspace{T, A3, A1}

Pre-allocated workspace for zero-allocation PCCT simulation.

Type parameters:
- `T`: Element type (typically Float32)
- `A3`: 3D array type matching GPU backend (e.g., Array{T,3} for CPU)
- `A1`: 1D array type matching GPU backend (e.g., Vector{T} for CPU)

All GPU-side buffers match the backend of the phantom mask passed to
`create_workspace`. CPU-side buffers are always plain `Array`.
"""
struct PCCTWorkspace{T<:AbstractFloat, A3<:AbstractArray{T,3}, A1<:AbstractArray{T,1}}
    # ─── Forward projection (GPU-side) ───
    bins::Vector{A3}           # n_bins sinogram buffers (output of forward projection)
    μ_volume::A3               # attenuation volume, reused per energy
    sino_buf::A3               # forward projection scratch, reused per energy

    # ─── Spatial kernel scratch (GPU-side) ───
    scratch::A3                # ONE buffer for all neighbor kernels
    total_counts::A3           # for anti-coincidence (sum across bins)

    # ─── Combine (GPU-side) ───
    combined::A3               # _combine_pcct_bins output (reused ideal + noisy)

    # ─── VMI synthesis (GPU-side) ───
    vmi_sino::A3               # synthesize_vmi output (reused across energies)

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

    # ─── Pre-computed CPU vectors/matrices (filled in simulate!()) ───
    η::Vector{Float64}          # quantum efficiency vector (n_energies)
    R::Matrix{Float64}          # spectral response matrix (n_energies × n_bins)
    R_energies::Vector{Float64} # energy grid for R matrix
    I0_bins::Vector{Float64}    # per-bin I0 values (n_bins)
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
end

"""
    create_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T=Float32)

Create a pre-allocated workspace for zero-allocation `simulate!()` calls.

Uses `similar(phantom.mask, T, shape)` for GPU-side buffers, which auto-detects
the correct GPU backend (Metal, CUDA, ROCm, or CPU).

# Arguments
- `scanner`: Scanner specification (provides detector geometry)
- `protocol`: CT protocol (provides number of views)
- `sim_opts`: Simulation options (provides n_energy_bins)
- `recon_opts`: Reconstruction options (provides vmi_basis for n_materials)
- `phantom`: Phantom struct (provides mask for backend detection and volume shape)
- `T`: Element type, default Float32

# Returns
A `PCCTWorkspace{T, A3, A1}` with all buffers pre-allocated.

# Example
```julia
ws = create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
# Second call: zero allocations
result2 = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
```
"""
function create_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T::Type{<:AbstractFloat}=Float32)
    sino_shape = (scanner.detector_cols, scanner.detector_rows, protocol.views)
    vol_shape = size(phantom.mask)
    n_bins = length(scanner.energy_thresholds)
    n_materials = length(recon_opts.vmi_basis)
    n_elements = prod(sino_shape)
    n_energies = sim_opts.n_energy_bins

    # GPU-side buffers — similar() matches phantom.mask backend
    mask = phantom.mask
    bins = [similar(mask, T, sino_shape) for _ in 1:n_bins]
    μ_volume = similar(mask, T, vol_shape)
    sino_buf = similar(mask, T, sino_shape)
    scratch = similar(mask, T, sino_shape)
    total_counts = similar(mask, T, sino_shape)
    combined = similar(mask, T, sino_shape)
    vmi_sino = similar(mask, T, sino_shape)
    enoise_gpu = similar(mask, T, n_elements)

    # CPU-side buffers
    noise_staging = zeros(T, sino_shape)
    noise_buf = zeros(T, sino_shape)
    enoise_cpu = Vector{T}(undef, n_elements)
    bins_cpu = [zeros(T, sino_shape) for _ in 1:n_bins]
    material_maps = [zeros(T, sino_shape) for _ in 1:n_materials]
    decomp_pixel_buf = zeros(T, n_bins)
    sino_ideal_out = zeros(T, sino_shape)
    sino_noisy_out = zeros(T, sino_shape)

    # Pre-computed vectors/matrices (allocated once, filled during simulate!())
    η = zeros(Float64, n_energies)
    R = zeros(Float64, n_energies, n_bins)
    R_energies = zeros(Float64, n_energies)
    I0_bins = zeros(Float64, n_bins)
    thresholds_T = zeros(T, n_bins)
    rng = MersenneTwister(0)

    # Detector physics precomputed buffers
    charge_sharing_probs = zeros(Float64, n_bins)
    pileup_counts = zeros(Float64, n_bins)
    pileup_migration = zeros(Float64, n_bins, n_bins)
    correction_pileup_counts = zeros(Float64, n_bins)
    correction_migration = zeros(Float64, n_bins, n_bins)
    μ_values = zeros(T, n_materials)
    noise_I0 = zeros(Float64, n_bins)

    return PCCTWorkspace{T, typeof(sino_buf), typeof(enoise_gpu)}(
        bins, μ_volume, sino_buf, scratch, total_counts,
        combined, vmi_sino,
        noise_staging, noise_buf, enoise_cpu, enoise_gpu,
        bins_cpu, material_maps, decomp_pixel_buf,
        sino_ideal_out, sino_noisy_out,
        η, R, R_energies, I0_bins, thresholds_T, rng,
        charge_sharing_probs, pileup_counts, pileup_migration,
        correction_pileup_counts, correction_migration, μ_values, noise_I0
    )
end
