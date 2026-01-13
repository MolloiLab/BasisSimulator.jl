"""
    API.jl

Unified high-level API for CT simulation and reconstruction.

Two main functions:
- `simulate_sinogram()` - Forward projection with all physical effects
- `reconstruct()` - Volume reconstruction with multiple methods

All functions are designed for Reactant compilation and use sensible clinical defaults.
"""

# =============================================================================
# Default Physical Effect Models
# =============================================================================

"""Default physical effect models for realistic CT simulation."""
const DEFAULT_FLAT_FILTER = flat_filter_al_cu(2.5, 0.1)
const DEFAULT_BOWTIE_FILTER = bowtie_filter_medium_body()
const DEFAULT_SCATTER_MODEL = default_scatter_model(scale_factor=0.5)
const DEFAULT_DETECTOR_MODEL = default_detector_model(blur_fwhm=1.0, I0=2e5, electronic_noise_std=20.0)
const DEFAULT_CROSSTALK_MODEL = crosstalk_medium()
const DEFAULT_LAG_MODEL = lag_gadox()
const DEFAULT_OPTICAL_CROSSTALK = optical_crosstalk_typical()
const DEFAULT_FILL_FACTOR = fill_factor_standard()
const DEFAULT_FOCAL_SPOT = focal_spot_medium()

# =============================================================================
# Unified Forward Projection
# =============================================================================

"""
    simulate_sinogram(phantom, geom; kwargs...) -> Array{Float32,3}

Simulate realistic CT sinogram with all physical effects.

This is the main forward projection function - polychromatic by default with
clinical-realistic detector physics.

# Arguments
- `phantom::Phantom`: Phantom with μ values and material mask
- `geom::CTGeometry`: Scanner geometry

# Keyword Arguments (Physical Effects - all enabled by default)
- `polychromatic::Bool=true`: Use polychromatic X-ray spectrum (includes beam hardening)
- `kVp::Int=120`: Tube voltage for spectrum generation
- `n_energy_bins::Int=20`: Number of energy bins for polychromatic simulation
- `flat_filter=DEFAULT_FLAT_FILTER`: Flat filtration (Al+Cu), `nothing` to disable
- `bowtie_filter=DEFAULT_BOWTIE_FILTER`: Bowtie filter, `nothing` to disable
- `scatter=DEFAULT_SCATTER_MODEL`: Scatter model, `nothing` to disable
- `detector=DEFAULT_DETECTOR_MODEL`: Detector model (noise, blur), `nothing` to disable
- `crosstalk=DEFAULT_CROSSTALK_MODEL`: Electronic crosstalk, `nothing` to disable
- `lag=DEFAULT_LAG_MODEL`: Detector afterglow, `nothing` to disable
- `optical_crosstalk=DEFAULT_OPTICAL_CROSSTALK`: Optical crosstalk, `nothing` to disable
- `fill_factor=DEFAULT_FILL_FACTOR`: Detector fill factor, `nothing` to disable
- `focal_spot=DEFAULT_FOCAL_SPOT`: Focal spot blur, `nothing` to disable
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility

# Returns
Sinogram array [n_cols, n_rows, n_angles] with all effects applied.

# Example
```julia
# Full realistic simulation (default)
sino = simulate_sinogram(phantom, geom)

# Ideal simulation (no effects)
sino = simulate_sinogram(phantom, geom;
    polychromatic=false,
    flat_filter=nothing,
    bowtie_filter=nothing,
    scatter=nothing,
    detector=nothing,
    crosstalk=nothing,
    lag=nothing,
    optical_crosstalk=nothing,
    fill_factor=nothing,
    focal_spot=nothing
)

# Custom: polychromatic + noise only
sino = simulate_sinogram(phantom, geom;
    flat_filter=nothing,
    bowtie_filter=nothing,
    scatter=nothing,
    crosstalk=nothing,
    lag=nothing
)
```
"""
function simulate_sinogram(
    phantom::Phantom,
    geom::CTGeometry;
    # Polychromatic settings
    polychromatic::Bool=true,
    kVp::Int=120,
    n_energy_bins::Int=20,
    # Physical effect models (all enabled by default)
    flat_filter::Union{FlatFilter,Nothing}=DEFAULT_FLAT_FILTER,
    bowtie_filter::Union{BowtieFilter,Nothing}=DEFAULT_BOWTIE_FILTER,
    scatter::Union{ScatterModel,Nothing}=DEFAULT_SCATTER_MODEL,
    detector::Union{DetectorModel,Nothing}=DEFAULT_DETECTOR_MODEL,
    crosstalk::Union{CrosstalkModel,Nothing}=DEFAULT_CROSSTALK_MODEL,
    lag::Union{LagModel,Nothing}=DEFAULT_LAG_MODEL,
    optical_crosstalk::Union{OpticalCrosstalkModel,Nothing}=DEFAULT_OPTICAL_CROSSTALK,
    fill_factor::Union{FillFactorModel,Nothing}=DEFAULT_FILL_FACTOR,
    focal_spot::Union{FocalSpot,Nothing}=DEFAULT_FOCAL_SPOT,
    seed::Union{Int,Nothing}=nothing
)
    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Step 1: Forward projection (polychromatic or monochromatic)
    if polychromatic
        sinogram = forward_project_polychromatic(phantom, geom, kVp; n_bins=n_energy_bins)
    else
        sinogram = forward_project_raymarching(phantom, geom)
    end

    # Step 2: Source effects (applied to projection values)
    if flat_filter !== nothing
        sinogram = apply_flat_filter(sinogram, flat_filter, geom)
    end

    if bowtie_filter !== nothing
        sinogram = apply_bowtie_filter(sinogram, bowtie_filter, geom)
    end

    # Step 3: Scatter (adds to sinogram)
    if scatter !== nothing
        sinogram = add_scatter(sinogram, scatter)
    end

    # Step 4: Detector effects
    # Note: These are applied in intensity domain internally

    if fill_factor !== nothing
        sinogram = apply_fill_factor(sinogram, fill_factor)
    end

    if optical_crosstalk !== nothing
        sinogram = apply_optical_crosstalk(sinogram, optical_crosstalk)
    end

    if crosstalk !== nothing
        sinogram = apply_crosstalk(sinogram, crosstalk)
    end

    if lag !== nothing
        sinogram = apply_lag(sinogram, lag)
    end

    if focal_spot !== nothing
        sinogram = apply_focal_spot_blur(sinogram, focal_spot, geom)
    end

    # Step 5: Detector noise (quantum + electronic)
    if detector !== nothing
        sinogram = apply_detector_model(sinogram, detector)
    end

    return Float32.(sinogram)
end

# =============================================================================
# Unified Reconstruction
# =============================================================================

"""
    reconstruct(sinogram, geom, output_size, fov; kwargs...) -> Array{Float32,3}

Reconstruct volume from sinogram using specified method.

Supports FDK (analytical) and iterative methods (SIRT, CGLS).

# Arguments
- `sinogram::AbstractArray{<:Real,3}`: Projection data [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Scanner geometry
- `output_size::NTuple{3,Int}`: Output volume dimensions (nx, ny, nz)
- `fov::NTuple{3,Float64}`: Field of view in cm

# Keyword Arguments
- `method::Symbol=:fdk`: Reconstruction method (:fdk, :sirt, :cgls)
- `kernel::ReconKernel=RampKernel()`: Reconstruction kernel (FDK only)
- `bhc::Union{WaterBHC,Nothing}=nothing`: Beam hardening correction
- `n_iterations::Int=10`: Number of iterations (SIRT/CGLS only)
- `regularization::Union{Nothing,Symbol}=nothing`: Regularization type (:tv, :l1, :l2)
- `regularization_weight::Float64=0.01`: Regularization strength
- `non_negativity::Bool=true`: Enforce non-negative values (iterative only)
- `verbose::Bool=false`: Print progress information

# Returns
Reconstructed volume [nx, ny, nz] containing μ values (cm⁻¹).

# Example
```julia
# FDK reconstruction (fast, default)
recon = reconstruct(sino, geom, (256, 256, 64), (20.0, 20.0, 6.4))

# FDK with soft kernel (reduced noise)
recon = reconstruct(sino, geom, output_size, fov; kernel=kernel_soft())

# SIRT iterative (noise robust)
recon = reconstruct(sino, geom, output_size, fov; method=:sirt, n_iterations=20)

# CGLS iterative (fast convergence)
recon = reconstruct(sino, geom, output_size, fov; method=:cgls, n_iterations=15)

# With beam hardening correction
recon = reconstruct(sino, geom, output_size, fov; bhc=water_bhc_120kVp())
```
"""
function reconstruct(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64};
    # Method selection
    method::Symbol=:fdk,
    # FDK parameters
    kernel::ReconKernel=RampKernel(),
    # Beam hardening correction
    bhc::Union{WaterBHC,Nothing}=nothing,
    # Iterative parameters
    n_iterations::Int=10,
    regularization::Union{Nothing,Symbol}=nothing,
    regularization_weight::Float64=0.01,
    non_negativity::Bool=true,
    # Options
    verbose::Bool=false
) where T

    # Step 1: Apply beam hardening correction if specified
    sino_corrected = if bhc !== nothing
        verbose && @info "Applying beam hardening correction..."
        apply_water_bhc(Float32.(sinogram), bhc)
    else
        Float32.(sinogram)
    end

    # Step 2: Reconstruct using specified method
    volume = if method == :fdk
        verbose && @info "FDK reconstruction..."
        _reconstruct_fdk(sino_corrected, geom, output_size, fov, kernel)

    elseif method == :sirt
        verbose && @info "SIRT reconstruction ($n_iterations iterations)..."
        _reconstruct_sirt(sino_corrected, geom, output_size, fov, n_iterations,
                         regularization, regularization_weight, non_negativity, verbose)

    elseif method == :cgls
        verbose && @info "CGLS reconstruction ($n_iterations iterations)..."
        _reconstruct_cgls(sino_corrected, geom, output_size, fov, n_iterations, verbose)

    else
        error("Unknown reconstruction method: $method. Use :fdk, :sirt, or :cgls")
    end

    return volume
end

# =============================================================================
# Internal Reconstruction Methods
# =============================================================================

"""FDK reconstruction using ray marching backprojection."""
function _reconstruct_fdk(
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64},
    kernel::ReconKernel
)
    return fdk_reconstruct_raymarching(sinogram, geom, output_size, fov; kernel=kernel)
end

"""SIRT reconstruction with optional regularization."""
function _reconstruct_sirt(
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64},
    n_iterations::Int,
    regularization::Union{Nothing,Symbol},
    regularization_weight::Float64,
    non_negativity::Bool,
    verbose::Bool
)
    nx, ny, nz = output_size
    n_voxels = prod(output_size)
    b = vec(sinogram)

    # Compute ray geometry for forward projection
    recon_ray_geom = compute_ray_geometry(geom, fov, output_size)
    recon_voxel_size = Tuple(Float32.(fov ./ output_size))
    recon_min_voxel = minimum(fov ./ output_size)
    recon_step_size = Float32(recon_min_voxel * 0.5)
    # n_samples must cover full ray path from source through volume
    half_diagonal = sqrt(sum((fov ./ 2) .^ 2))
    ray_path_to_far_edge = geom.SAD + half_diagonal
    recon_n_samples = ceil(Int, ray_path_to_far_edge / recon_step_size) + 10

    # Compute backprojection geometry
    bp_geom = compute_backproj_geometry(geom)
    dx, dy, dz = fov ./ output_size
    voxel_x = Float32.(range(-fov[1]/2 + dx/2, fov[1]/2 - dx/2, length=nx))
    voxel_y = Float32.(range(-fov[2]/2 + dy/2, fov[2]/2 - dy/2, length=ny))
    voxel_z = Float32.(range(-fov[3]/2 + dz/2, fov[3]/2 - dz/2, length=nz))

    # Compute SIRT normalization factors
    # R: row sums (forward project ones)
    ones_vol = ones(Float32, output_size...)
    R_sino = forward_project_raymarching_vectorized(
        ones_vol, recon_ray_geom.origins, recon_ray_geom.directions,
        recon_ray_geom.vol_min, recon_voxel_size, recon_n_samples, recon_step_size
    )
    R_sirt = map(r -> r > 1f-8 ? 1f0 / r : 0f0, vec(R_sino))

    # C: column sums (backproject ones)
    ones_sino = ones(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    C_vol = backproject_raw_kernel(
        ones_sino, bp_geom.source_positions, bp_geom.detector_centers,
        bp_geom.detector_u, bp_geom.detector_v, bp_geom.sd_axis,
        voxel_x, voxel_y, voxel_z,
        bp_geom.SAD, bp_geom.SDD, bp_geom.pixel_size_det,
        bp_geom.n_cols, bp_geom.n_rows
    )
    C_sirt = map(c -> c > 1f-8 ? 1f0 / c : 0f0, vec(C_vol))

    # Initialize solution
    x = zeros(Float32, n_voxels)

    # SIRT iterations
    for iter in 1:n_iterations
        verbose && @info "  SIRT iteration $iter / $n_iterations"

        # Forward project: A*x
        x_vol = reshape(x, output_size)
        Ax = forward_project_raymarching_vectorized(
            x_vol, recon_ray_geom.origins, recon_ray_geom.directions,
            recon_ray_geom.vol_min, recon_voxel_size, recon_n_samples, recon_step_size
        )

        # Weighted residual: R * (b - Ax)
        residual_weighted = (b .- vec(Ax)) .* R_sirt

        # Backproject: A^T * R * (b - Ax)
        residual_sino = reshape(residual_weighted, geom.n_cols, geom.n_rows, geom.n_angles)
        correction = backproject_raw_kernel(
            residual_sino, bp_geom.source_positions, bp_geom.detector_centers,
            bp_geom.detector_u, bp_geom.detector_v, bp_geom.sd_axis,
            voxel_x, voxel_y, voxel_z,
            bp_geom.SAD, bp_geom.SDD, bp_geom.pixel_size_det,
            bp_geom.n_cols, bp_geom.n_rows
        )

        # Update: x + C * correction
        x .+= vec(correction) .* C_sirt

        # Optional regularization
        if regularization !== nothing
            x_vol = reshape(x, output_size)
            if regularization == :tv
                # Gradient descent step for TV
                grad = _tv_gradient(x_vol)
                x .-= regularization_weight .* vec(grad)
            elseif regularization == :l1
                # Soft thresholding
                x .= sign.(x) .* max.(abs.(x) .- regularization_weight, 0f0)
            elseif regularization == :l2
                # Shrinkage
                x .*= 1f0 / (1f0 + regularization_weight)
            end
        end

        # Non-negativity constraint
        if non_negativity
            x .= max.(x, 0f0)
        end
    end

    return reshape(x, output_size...)
end

"""CGLS reconstruction."""
function _reconstruct_cgls(
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64},
    n_iterations::Int,
    verbose::Bool
)
    nx, ny, nz = output_size
    n_voxels = prod(output_size)
    b = vec(sinogram)

    # Compute ray geometry
    recon_ray_geom = compute_ray_geometry(geom, fov, output_size)
    recon_voxel_size = Tuple(Float32.(fov ./ output_size))
    recon_min_voxel = minimum(fov ./ output_size)
    recon_step_size = Float32(recon_min_voxel * 0.5)
    # n_samples must cover full ray path from source through volume
    half_diagonal = sqrt(sum((fov ./ 2) .^ 2))
    ray_path_to_far_edge = geom.SAD + half_diagonal
    recon_n_samples = ceil(Int, ray_path_to_far_edge / recon_step_size) + 10

    # Compute backprojection geometry
    bp_geom = compute_backproj_geometry(geom)
    dx, dy, dz = fov ./ output_size
    voxel_x = Float32.(range(-fov[1]/2 + dx/2, fov[1]/2 - dx/2, length=nx))
    voxel_y = Float32.(range(-fov[2]/2 + dy/2, fov[2]/2 - dy/2, length=ny))
    voxel_z = Float32.(range(-fov[3]/2 + dz/2, fov[3]/2 - dz/2, length=nz))

    # Initialize
    x = zeros(Float32, n_voxels)
    r = copy(b)  # Initial residual (x=0)

    # Initial gradient: s = A^T * r
    r_sino = reshape(r, geom.n_cols, geom.n_rows, geom.n_angles)
    s = vec(backproject_raw_kernel(
        r_sino, bp_geom.source_positions, bp_geom.detector_centers,
        bp_geom.detector_u, bp_geom.detector_v, bp_geom.sd_axis,
        voxel_x, voxel_y, voxel_z,
        bp_geom.SAD, bp_geom.SDD, bp_geom.pixel_size_det,
        bp_geom.n_cols, bp_geom.n_rows
    ))

    p_dir = copy(s)
    gamma = sum(s .* s)

    for iter in 1:n_iterations
        verbose && @info "  CGLS iteration $iter / $n_iterations"

        # q = A * p
        p_vol = reshape(p_dir, output_size)
        q = vec(forward_project_raymarching_vectorized(
            p_vol, recon_ray_geom.origins, recon_ray_geom.directions,
            recon_ray_geom.vol_min, recon_voxel_size, recon_n_samples, recon_step_size
        ))

        # alpha = gamma / ||q||²
        q_norm_sq = sum(q .* q)
        if q_norm_sq < 1e-16
            break
        end
        alpha = gamma / q_norm_sq

        # Update x and r
        x .+= alpha .* p_dir
        r .-= alpha .* q

        # Update gradient: s = A^T * r
        r_sino = reshape(r, geom.n_cols, geom.n_rows, geom.n_angles)
        s = vec(backproject_raw_kernel(
            r_sino, bp_geom.source_positions, bp_geom.detector_centers,
            bp_geom.detector_u, bp_geom.detector_v, bp_geom.sd_axis,
            voxel_x, voxel_y, voxel_z,
            bp_geom.SAD, bp_geom.SDD, bp_geom.pixel_size_det,
            bp_geom.n_cols, bp_geom.n_rows
        ))

        # Update gamma and search direction
        gamma_new = sum(s .* s)
        beta = gamma_new / gamma
        p_dir = s .+ beta .* p_dir
        gamma = gamma_new
    end

    return reshape(x, output_size...)
end

"""Compute TV gradient for regularization."""
function _tv_gradient(volume::Array{Float32,3}; epsilon::Float32=1f-8)
    nx, ny, nz = size(volume)
    grad = zeros(Float32, nx, ny, nz)

    for iz in 2:nz-1, iy in 2:ny-1, ix in 2:nx-1
        # Forward differences
        dx_f = volume[ix+1, iy, iz] - volume[ix, iy, iz]
        dy_f = volume[ix, iy+1, iz] - volume[ix, iy, iz]
        dz_f = volume[ix, iy, iz+1] - volume[ix, iy, iz]

        # Backward differences
        dx_b = volume[ix, iy, iz] - volume[ix-1, iy, iz]
        dy_b = volume[ix, iy, iz] - volume[ix, iy-1, iz]
        dz_b = volume[ix, iy, iz] - volume[ix, iy, iz-1]

        # TV gradient
        norm_f = sqrt(dx_f^2 + dy_f^2 + dz_f^2 + epsilon)
        norm_b = sqrt(dx_b^2 + dy_b^2 + dz_b^2 + epsilon)

        grad[ix, iy, iz] = -(dx_f + dy_f + dz_f) / norm_f + (dx_b + dy_b + dz_b) / norm_b
    end

    return grad
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    simulate_and_reconstruct(phantom, geom, output_size; kwargs...) -> (sinogram, volume)

Complete CT simulation pipeline: forward projection with effects, then reconstruction.

# Arguments
- `phantom::Phantom`: Input phantom
- `geom::CTGeometry`: Scanner geometry
- `output_size::NTuple{3,Int}`: Output volume size

# Keyword Arguments
All kwargs from `simulate_sinogram()` and `reconstruct()` are supported.

# Returns
Tuple of (sinogram, reconstructed_volume)

# Example
```julia
sino, recon = simulate_and_reconstruct(phantom, geom, (256, 256, 64))
```
"""
function simulate_and_reconstruct(
    phantom::Phantom,
    geom::CTGeometry,
    output_size::NTuple{3,Int};
    # simulate_sinogram kwargs
    polychromatic::Bool=true,
    kVp::Int=120,
    n_energy_bins::Int=20,
    flat_filter::Union{FlatFilter,Nothing}=DEFAULT_FLAT_FILTER,
    bowtie_filter::Union{BowtieFilter,Nothing}=DEFAULT_BOWTIE_FILTER,
    scatter::Union{ScatterModel,Nothing}=DEFAULT_SCATTER_MODEL,
    detector::Union{DetectorModel,Nothing}=DEFAULT_DETECTOR_MODEL,
    crosstalk::Union{CrosstalkModel,Nothing}=DEFAULT_CROSSTALK_MODEL,
    lag::Union{LagModel,Nothing}=DEFAULT_LAG_MODEL,
    optical_crosstalk::Union{OpticalCrosstalkModel,Nothing}=DEFAULT_OPTICAL_CROSSTALK,
    fill_factor::Union{FillFactorModel,Nothing}=DEFAULT_FILL_FACTOR,
    focal_spot::Union{FocalSpot,Nothing}=DEFAULT_FOCAL_SPOT,
    seed::Union{Int,Nothing}=nothing,
    # reconstruct kwargs
    method::Symbol=:fdk,
    kernel::ReconKernel=RampKernel(),
    bhc::Union{WaterBHC,Nothing}=nothing,
    n_iterations::Int=10,
    regularization::Union{Nothing,Symbol}=nothing,
    regularization_weight::Float64=0.01,
    non_negativity::Bool=true,
    verbose::Bool=false
)
    # Forward projection
    sinogram = simulate_sinogram(phantom, geom;
        polychromatic=polychromatic, kVp=kVp, n_energy_bins=n_energy_bins,
        flat_filter=flat_filter, bowtie_filter=bowtie_filter, scatter=scatter,
        detector=detector, crosstalk=crosstalk, lag=lag,
        optical_crosstalk=optical_crosstalk, fill_factor=fill_factor,
        focal_spot=focal_spot, seed=seed
    )

    # Reconstruction
    volume = reconstruct(sinogram, geom, output_size, phantom.fov;
        method=method, kernel=kernel, bhc=bhc,
        n_iterations=n_iterations, regularization=regularization,
        regularization_weight=regularization_weight, non_negativity=non_negativity,
        verbose=verbose
    )

    return sinogram, volume
end

# =============================================================================
# Exports
# =============================================================================

export simulate_sinogram, reconstruct, simulate_and_reconstruct
export DEFAULT_FLAT_FILTER, DEFAULT_BOWTIE_FILTER, DEFAULT_SCATTER_MODEL
export DEFAULT_DETECTOR_MODEL, DEFAULT_CROSSTALK_MODEL, DEFAULT_LAG_MODEL
export DEFAULT_OPTICAL_CROSSTALK, DEFAULT_FILL_FACTOR, DEFAULT_FOCAL_SPOT
