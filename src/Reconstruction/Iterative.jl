"""
    Reconstruction/Iterative.jl

Iterative reconstruction algorithms for CT.

Implements XLA-compatible iterative methods:
- SIRT (Simultaneous Iterative Reconstruction Technique)
- CGLS (Conjugate Gradient Least Squares)

All algorithms use pre-computed geometry for XLA/Reactant compatibility.

References:
- SIRT: Gregor & Benson (2008), doi:10.1109/TMI.2008.923696
- CGLS: Hestenes & Stiefel (1952), Paige & Saunders (1982)
"""

# =============================================================================
# SIRT Normalization Factors
# =============================================================================

"""
    SIRTNormalization

Pre-computed normalization factors for SIRT algorithm.

# Fields
- `C`: Column normalization (inverse sum of rays hitting each voxel) [nx*ny*nz]
- `R`: Row normalization (inverse sum of voxels hit by each ray) [n_cols*n_rows*n_angles]
"""
struct SIRTNormalization
    C::Vector{Float32}  # Per-voxel normalization
    R::Vector{Float32}  # Per-ray normalization
end

"""
    compute_sirt_normalization(proj_geom, bp_geom)

Compute SIRT normalization factors from pre-computed geometries.

The normalization ensures proper scaling:
- C: How many rays contribute to each voxel (column sums of system matrix)
- R: How many voxels each ray passes through (row sums of system matrix)

These are computed by projecting/backprojecting a volume of ones.
"""
function compute_sirt_normalization(proj_geom::ProjectionGeometry, bp_geom::BackprojectionGeometry)
    nx, ny, nz = bp_geom.nx, bp_geom.ny, bp_geom.nz
    n_total_voxels = nx * ny * nz
    n_total_rays = bp_geom.n_cols * bp_geom.n_rows * bp_geom.n_angles

    # Compute R: Forward project a volume of ones
    # This gives sum of path lengths through each ray
    ones_volume = ones(Float32, n_total_voxels)
    R_raw = project_volume(reshape(ones_volume, nx, ny, nz), proj_geom)
    R = vec(R_raw)

    # Avoid division by zero
    R = map(r -> r > Float32(1e-8) ? Float32(1.0) / r : Float32(0.0), R)

    # Compute C: Backproject a sinogram of ones
    # This gives how many rays hit each voxel (weighted by path length)
    ones_sino = ones(Float32, n_total_rays)
    C_raw = backproject_volume(ones_sino, bp_geom)
    C = vec(C_raw)

    # Avoid division by zero
    C = map(c -> c > Float32(1e-8) ? Float32(1.0) / c : Float32(0.0), C)

    return SIRTNormalization(C, R)
end

# =============================================================================
# SIRT Algorithm
# =============================================================================

"""
    sirt_step(x, b, proj_geom, bp_geom, norm; relaxation=Float32(1.0))

Perform one SIRT iteration.

Update: x^(k+1) = x^(k) + relaxation * C * A^T * R * (b - A*x^(k))

# Arguments
- `x`: Current volume estimate (flattened) [nx*ny*nz]
- `b`: Measured sinogram (flattened) [n_cols*n_rows*n_angles]
- `proj_geom`: Pre-computed forward projection geometry
- `bp_geom`: Pre-computed backprojection geometry
- `norm`: SIRT normalization factors
- `relaxation`: Relaxation parameter (0 < λ ≤ 2, default: 1.0)

# Returns
Updated volume estimate (flattened)

This function is XLA-compatible.
"""
function sirt_step(
    x::AbstractVector{T},
    b::AbstractVector{T},
    proj_geom::ProjectionGeometry,
    bp_geom::BackprojectionGeometry,
    norm::SIRTNormalization;
    relaxation::Float32=Float32(1.0)
) where T
    nx, ny, nz = bp_geom.nx, bp_geom.ny, bp_geom.nz

    # Forward project current estimate: A*x
    x_vol = reshape(x, nx, ny, nz)
    Ax = vec(project_volume(x_vol, proj_geom))

    # Compute residual: b - A*x
    residual = b .- Ax

    # Apply row normalization: R * residual
    residual_weighted = residual .* norm.R

    # Backproject: A^T * R * residual
    correction_vol = backproject_volume(residual_weighted, bp_geom)
    correction = vec(correction_vol)

    # Apply column normalization: C * A^T * R * residual
    correction_weighted = correction .* norm.C

    # Update: x + relaxation * correction
    x_new = x .+ relaxation .* correction_weighted

    return x_new
end

"""
    sirt_reconstruct(sinogram, proj_geom, bp_geom; kwargs...)

SIRT (Simultaneous Iterative Reconstruction Technique) reconstruction.

# Arguments
- `sinogram`: Measured projections [n_cols, n_rows, n_angles]
- `proj_geom`: Pre-computed forward projection geometry
- `bp_geom`: Pre-computed backprojection geometry

# Keyword Arguments
- `n_iterations::Int=50`: Maximum number of iterations (default: 50)
- `relaxation::Float32=1.0`: Relaxation parameter (0 < λ ≤ 2)
- `x0::Union{Nothing, Array}=nothing`: Initial estimate (default: zeros)
- `min_change::Float32=1e-6`: Early stopping threshold (relative change)
- `verbose::Bool=false`: Print progress

# Returns
NamedTuple with:
- `volume`: Reconstructed volume [nx, ny, nz]
- `iterations`: Number of iterations performed
- `converged`: Whether early stopping was triggered

# Notes
- SIRT is robust to noise but converges slowly
- Typical iteration count: 20-100
- Higher relaxation (up to ~1.9) can speed convergence but may oscillate

# Example
```julia
proj_geom = precompute_projection_geometry(geom, fov, voxel_size, vol_size)
bp_geom = precompute_backprojection_geometry(geom, vol_size, fov)
result = sirt_reconstruct(sinogram, proj_geom, bp_geom; n_iterations=50)
volume = result.volume
```
"""
function sirt_reconstruct(
    sinogram::AbstractArray{T,3},
    proj_geom::ProjectionGeometry,
    bp_geom::BackprojectionGeometry;
    n_iterations::Int=50,
    relaxation::Float32=Float32(1.0),
    x0::Union{Nothing, AbstractArray}=nothing,
    min_change::Float32=Float32(1e-6),
    verbose::Bool=false
) where T
    @assert 0 < relaxation <= 2 "Relaxation must be in (0, 2]"
    @assert n_iterations > 0 "n_iterations must be positive"

    nx, ny, nz = bp_geom.nx, bp_geom.ny, bp_geom.nz
    n_voxels = nx * ny * nz

    # Initialize
    if x0 === nothing
        x = zeros(Float32, n_voxels)
    else
        x = vec(Float32.(x0))
    end

    b = vec(Float32.(sinogram))

    # Compute normalization factors
    norm = compute_sirt_normalization(proj_geom, bp_geom)

    # Iterate
    converged = false
    actual_iterations = n_iterations

    for iter in 1:n_iterations
        x_old = copy(x)
        x = sirt_step(x, b, proj_geom, bp_geom, norm; relaxation=relaxation)

        # Check convergence
        change = norm_relative_change(x, x_old)

        if verbose && (iter % 10 == 0 || iter == 1)
            residual_norm = compute_residual_norm(x, b, proj_geom, bp_geom)
            println("SIRT iter $iter: residual=$residual_norm, change=$change")
        end

        if change < min_change
            converged = true
            actual_iterations = iter
            verbose && println("SIRT converged at iteration $iter (change=$change)")
            break
        end
    end

    volume = reshape(x, nx, ny, nz)

    return (
        volume = volume,
        iterations = actual_iterations,
        converged = converged
    )
end

# =============================================================================
# CGLS Algorithm
# =============================================================================

"""
    cgls_reconstruct(sinogram, proj_geom, bp_geom; kwargs...)

CGLS (Conjugate Gradient Least Squares) reconstruction.

Minimizes ||Ax - b||² using conjugate gradients.

# Arguments
- `sinogram`: Measured projections [n_cols, n_rows, n_angles]
- `proj_geom`: Pre-computed forward projection geometry
- `bp_geom`: Pre-computed backprojection geometry

# Keyword Arguments
- `n_iterations::Int=20`: Maximum number of iterations (default: 20)
- `x0::Union{Nothing, Array}=nothing`: Initial estimate (default: zeros)
- `min_residual::Float32=1e-6`: Early stopping threshold (relative residual)
- `verbose::Bool=false`: Print progress

# Returns
NamedTuple with:
- `volume`: Reconstructed volume [nx, ny, nz]
- `iterations`: Number of iterations performed
- `converged`: Whether early stopping was triggered
- `residual`: Final residual norm

# Notes
- CGLS converges faster than SIRT but is less robust to noise
- Typical iteration count: 5-30
- Semi-convergent: may need early stopping to avoid noise amplification

# Example
```julia
proj_geom = precompute_projection_geometry(geom, fov, voxel_size, vol_size)
bp_geom = precompute_backprojection_geometry(geom, vol_size, fov)
result = cgls_reconstruct(sinogram, proj_geom, bp_geom; n_iterations=20)
volume = result.volume
```
"""
function cgls_reconstruct(
    sinogram::AbstractArray{T,3},
    proj_geom::ProjectionGeometry,
    bp_geom::BackprojectionGeometry;
    n_iterations::Int=20,
    x0::Union{Nothing, AbstractArray}=nothing,
    min_residual::Float32=Float32(1e-6),
    verbose::Bool=false
) where T
    @assert n_iterations > 0 "n_iterations must be positive"

    nx, ny, nz = bp_geom.nx, bp_geom.ny, bp_geom.nz
    n_voxels = nx * ny * nz

    b = vec(Float32.(sinogram))

    # Initialize x
    if x0 === nothing
        x = zeros(Float32, n_voxels)
    else
        x = vec(Float32.(x0))
    end

    # Initial residual: r = b - A*x
    x_vol = reshape(x, nx, ny, nz)
    Ax = vec(project_volume(x_vol, proj_geom))
    r = b .- Ax

    # Initial gradient: s = A^T * r
    s = vec(backproject_volume(r, bp_geom))

    # Initial search direction
    p = copy(s)

    # Initial gamma = ||s||²
    gamma = sum(s .* s)
    gamma0 = gamma  # For relative residual check

    converged = false
    actual_iterations = n_iterations
    final_residual = sqrt(gamma)

    for iter in 1:n_iterations
        # q = A * p
        p_vol = reshape(p, nx, ny, nz)
        q = vec(project_volume(p_vol, proj_geom))

        # alpha = gamma / ||q||²
        q_norm_sq = sum(q .* q)
        if q_norm_sq < 1e-16
            verbose && println("CGLS: q_norm_sq too small, stopping")
            actual_iterations = iter
            break
        end
        alpha = gamma / q_norm_sq

        # Update x: x = x + alpha * p
        x = x .+ alpha .* p

        # Update residual: r = r - alpha * q
        r = r .- alpha .* q

        # Update gradient: s = A^T * r
        s = vec(backproject_volume(r, bp_geom))

        # Update gamma
        gamma_new = sum(s .* s)

        # Check convergence
        relative_residual = sqrt(gamma_new / gamma0)
        final_residual = sqrt(gamma_new)

        if verbose && (iter % 5 == 0 || iter == 1)
            println("CGLS iter $iter: residual=$final_residual, relative=$relative_residual")
        end

        if relative_residual < min_residual
            converged = true
            actual_iterations = iter
            verbose && println("CGLS converged at iteration $iter")
            break
        end

        # Update search direction: p = s + beta * p
        beta = gamma_new / gamma
        p = s .+ beta .* p

        gamma = gamma_new
    end

    volume = reshape(x, nx, ny, nz)

    return (
        volume = volume,
        iterations = actual_iterations,
        converged = converged,
        residual = final_residual
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    norm_relative_change(x_new, x_old)

Compute relative change between iterations.
"""
function norm_relative_change(x_new::AbstractVector, x_old::AbstractVector)
    diff_norm = sqrt(sum((x_new .- x_old).^2))
    old_norm = sqrt(sum(x_old.^2))
    return old_norm > 1e-10 ? diff_norm / old_norm : diff_norm
end

"""
    compute_residual_norm(x, b, proj_geom, bp_geom)

Compute ||b - Ax|| for monitoring convergence.
"""
function compute_residual_norm(
    x::AbstractVector{T},
    b::AbstractVector{T},
    proj_geom::ProjectionGeometry,
    bp_geom::BackprojectionGeometry
) where T
    nx, ny, nz = bp_geom.nx, bp_geom.ny, bp_geom.nz
    x_vol = reshape(x, nx, ny, nz)
    Ax = vec(project_volume(x_vol, proj_geom))
    residual = b .- Ax
    return sqrt(sum(residual.^2))
end

# =============================================================================
# Convenience Wrappers
# =============================================================================

"""
    iterative_reconstruct(sinogram, geom, output_size, fov; method=:sirt, kwargs...)

Convenience wrapper for iterative reconstruction.

# Arguments
- `sinogram`: Measured projections [n_cols, n_rows, n_angles]
- `geom`: CT scanner geometry
- `output_size`: Output volume size (nx, ny, nz)
- `fov`: Field of view (fx, fy, fz) in cm
- `method`: Reconstruction method (:sirt or :cgls)

# Keyword Arguments
Passed to the underlying algorithm. Common options:
- `n_iterations`: Maximum iterations (SIRT: 50, CGLS: 20)
- `verbose`: Print progress

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
recon = iterative_reconstruct(sinogram, geom, (64,64,64), phantom.fov; method=:sirt)
```
"""
function iterative_reconstruct(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64};
    method::Symbol=:sirt,
    kwargs...
) where T
    # Pre-compute geometries
    voxel_size = (fov[1]/output_size[1], fov[2]/output_size[2], fov[3]/output_size[3])
    proj_geom = precompute_projection_geometry(geom, fov, voxel_size, output_size)
    bp_geom = precompute_backprojection_geometry(geom, output_size, fov)

    if method == :sirt
        result = sirt_reconstruct(sinogram, proj_geom, bp_geom; kwargs...)
    elseif method == :cgls
        result = cgls_reconstruct(sinogram, proj_geom, bp_geom; kwargs...)
    else
        error("Unknown method: $method. Use :sirt or :cgls")
    end

    return result.volume
end

# =============================================================================
# Exports
# =============================================================================

export SIRTNormalization, compute_sirt_normalization
export sirt_step, sirt_reconstruct
export cgls_reconstruct
export iterative_reconstruct
