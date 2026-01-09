"""
    Reconstruction/Iterative.jl

Iterative reconstruction algorithms for CT.

Includes:
- SIRT (Simultaneous Iterative Reconstruction Technique)
- MLEM (Maximum Likelihood Expectation Maximization)
- TV Regularization (Total Variation)

# Advantages over FDK

1. **Handles incomplete data**: Sparse angles, limited angles, truncated FOV
2. **Noise control**: Regularization suppresses noise amplification
3. **Prior knowledge**: Can incorporate constraints (non-negativity, smoothness)
4. **Artifact reduction**: Less susceptible to metal artifacts, beam hardening

# Disadvantages

1. **Computational cost**: 10-100x slower than FDK
2. **Convergence**: Requires careful stopping criteria
3. **Parameter tuning**: λ (regularization), iterations, step size

# References

**SIRT:**
- Gilbert, P. (1972). J. Theor. Biol., 36(1), 105-117.
  "Iterative methods for the three-dimensional reconstruction of an object"
- Andersen, A. H., & Kak, A. C. (1984). Ultrason. Imaging, 6(1), 81-94.
  "Simultaneous algebraic reconstruction technique (SART)"

**MLEM:**
- Shepp, L. A., & Vardi, Y. (1982). IEEE Trans. Med. Imaging, 1(2), 113-122.
  "Maximum likelihood reconstruction for emission tomography"
- Lange, K., & Carson, R. (1984). J. Comp. Assist. Tomogr., 8(2), 306-316.
  "EM reconstruction algorithms for emission and transmission tomography"

**TV Regularization:**
- Rudin, L. I., et al. (1992). Physica D, 60(1-4), 259-268.
  "Nonlinear total variation based noise removal algorithms"
- Sidky, E. Y., et al. (2008). Phys. Med. Biol., 53(17), 4777.
  "Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT"
"""

using LinearAlgebra
using SparseArrays

# ==============================================================================
# SIRT (Simultaneous Iterative Reconstruction Technique)
# ==============================================================================

"""
    reconstruct_sirt(
        sinogram::Array{Float64, 3},
        geometry::CTGeometry,
        image_size::Int = 256;
        n_iterations::Int = 20,
        relaxation::Float64 = 1.0
    )::Array{Float64, 3}

Reconstruct CT volume using SIRT (Simultaneous Iterative Reconstruction Technique).

**Status**: Placeholder for future implementation.

# Algorithm

SIRT is a simple iterative method that minimizes ||Ax - b||² where:
- A = system matrix (geometry of projection lines)
- x = image to reconstruct
- b = measured sinogram

**Update rule**:
```
x^(k+1) = x^(k) + λ × C × A^T × R × (b - A × x^(k))
```

where:
- λ = relaxation parameter (0 < λ ≤ 1)
- C = column normalization diagonal matrix
- R = row normalization diagonal matrix

# Arguments

- `sinogram::Array{Float64, 3}` - Measured projection data [rows, cols, angles]
- `geometry::CTGeometry` - Scanner geometry
- `image_size::Int = 256` - Reconstructed image size (pixels per side)
- `n_iterations::Int = 20` - Number of SIRT iterations
- `relaxation::Float64 = 1.0` - Relaxation parameter (step size)

# Returns

- `volume::Array{Float64, 3}` - Reconstructed 3D volume

# Example

```julia
# Reconstruct with 20 SIRT iterations
volume = reconstruct_sirt(
    sinogram,
    geometry,
    image_size=256,
    n_iterations=20,
    relaxation=0.8  # Slightly damped for stability
)
```

# Convergence

**Stopping criteria**:
1. Maximum iterations reached
2. Relative change < tolerance: ||x^(k+1) - x^(k)|| / ||x^(k)|| < ε
3. Data fit plateaus: ||A×x^(k) - b|| stops decreasing

**Typical values**:
- n_iterations = 10-50 (more for noisy data)
- relaxation = 0.5-1.0 (smaller = more stable, slower)

# Computational Cost

**Memory**: O(n_angles × n_pixels²) for system matrix (sparse)
**Time per iteration**: O(n_angles × n_pixels²) for forward/back projection

For 256² image, 360 angles:
- Memory: ~500 MB (sparse)
- Time: ~5 seconds per iteration (CPU)

# References

- Gilbert (1972) - Original SIRT paper
- Andersen & Kak (1984) - SART (algebraic variant)
- Van der Sluis & van der Vorst (1990) - Numerical properties
"""
function reconstruct_sirt(
        sinogram::Array{Float64, 3},
        geometry::CTGeometry,
        image_size::Int = 256;
        n_iterations::Int = 20,
        relaxation::Float64 = 1.0
    )::Array{Float64, 3}

    error("""
    SIRT reconstruction requires a forward projector (A×x operation).

    Full implementation requires:
    1. Forward projection: Ray tracing through arbitrary volume
       - Not currently available (simulate_ct_scan requires phantom structure)
       - Would need to implement volume-based ray tracer
       - Estimated effort: 4-6 hours

    2. Back projection: Transpose of forward operator
       - Could approximate with FDK backprojection
       - But without proper forward projector, algorithm won't converge

    3. System matrix: Sparse representation of A
       - Memory prohibitive for realistic sizes (100+ GB)
       - Requires on-the-fly projection operators

    Recommended for future implementation:
    - Implement volume-based forward projector in RayTracing.jl
    - Add to PhantomData: conversion from arbitrary Float64 volume
    - Then enable SIRT, MLEM, TV in iterative reconstruction

    Timeline: Phase 4+ (after GECATSIM validation)
    Priority: Medium (FDK sufficient for most use cases)

    Alternative: Use MIRT.jl (Michigan Image Reconstruction Toolbox)
    for production iterative reconstruction needs.
    """)
end

# ==============================================================================
# MLEM (Maximum Likelihood Expectation Maximization)
# ==============================================================================

"""
    reconstruct_mlem(
        sinogram::Array{Float64, 3},
        geometry::CTGeometry,
        image_size::Int = 256;
        n_iterations::Int = 20,
        epsilon::Float64 = 1e-10
    )::Array{Float64, 3}

Reconstruct CT volume using MLEM (Maximum Likelihood Expectation Maximization).

**Status**: Placeholder for future implementation.

# Algorithm

MLEM maximizes the likelihood of the data given Poisson statistics:

**Update rule**:
```
x^(k+1)_j = (x^(k)_j / Σ_i A_ij) × Σ_i A_ij × (b_i / [A × x^(k)]_i)
```

where:
- A_ij = probability that emission from voxel j detected in bin i
- b_i = measured counts in detector bin i
- x_j = activity in voxel j

# Arguments

- `sinogram::Array{Float64, 3}` - Measured projection data (counts)
- `geometry::CTGeometry` - Scanner geometry
- `image_size::Int = 256` - Reconstructed image size
- `n_iterations::Int = 20` - Number of EM iterations
- `epsilon::Float64 = 1e-10` - Small constant to avoid division by zero

# Returns

- `volume::Array{Float64, 3}` - Reconstructed 3D volume

# Properties

**Advantages**:
1. Guarantees non-negativity (x ≥ 0)
2. Preserves photon counts (Σ_j x_j = constant)
3. Statistically optimal for low counts (Poisson noise)

**Disadvantages**:
1. Slow convergence
2. Amplifies noise at high iterations
3. Requires accurate Poisson model

# Example

```julia
# Low-dose reconstruction with Poisson statistics
volume = reconstruct_mlem(
    sinogram,
    geometry,
    image_size=256,
    n_iterations=10  # Stop early to avoid noise amplification
)
```

# Stopping Criteria

**Objective function** (log-likelihood):
```
L = Σ_i b_i × log([A×x]_i) - [A×x]_i
```

Stop when:
1. L stops increasing
2. Maximum iterations reached
3. Image quality metric plateaus (SSIM, CNR, etc.)

# References

- Shepp & Vardi (1982) - Original MLEM for medical imaging
- Lange & Carson (1984) - EM for transmission tomography
- Fessler (1994) - Penalized-likelihood image reconstruction
"""
function reconstruct_mlem(
        sinogram::Array{Float64, 3},
        geometry::CTGeometry,
        image_size::Int = 256;
        n_iterations::Int = 20,
        epsilon::Float64 = 1e-10
    )::Array{Float64, 3}

    error("""
    MLEM reconstruction requires forward/back projection operators.

    Same limitations as SIRT - needs volume-based ray tracer.

    MLEM update rule:
    ```
    x^(k+1)_j = (x^(k)_j / Σ_i A_ij) × Σ_i A_ij × (b_i / [A×x^(k)]_i)
    ```

    Advantages over SIRT:
    - Guarantees non-negativity (x ≥ 0)
    - Statistical optimality for Poisson noise
    - Preserves photon counts

    Implementation requirements (same as SIRT):
    1. Volume-based forward projector
    2. Transpose operator (backprojection)
    3. Convergence monitoring

    Timeline: Phase 4+ (after forward projector implementation)
    Estimated effort: 2-3 hours (given SIRT infrastructure)

    Alternative: Use MIRT.jl or other established packages
    """)
end

# ==============================================================================
# TV Regularization (Total Variation)
# ==============================================================================

"""
    reconstruct_tv(
        sinogram::Array{Float64, 3},
        geometry::CTGeometry,
        image_size::Int = 256;
        n_iterations::Int = 50,
        lambda::Float64 = 0.01,
        tv_iterations::Int = 5
    )::Array{Float64, 3}

Reconstruct CT volume using TV (Total Variation) regularization.

**Status**: Placeholder for future implementation.

# Algorithm

TV regularization minimizes:
```
E(x) = ||A×x - b||² + λ × TV(x)
```

where TV(x) = Σ |∇x| is the total variation (sum of image gradients).

**Properties**:
- Preserves edges (piecewise-constant assumption)
- Reduces noise while maintaining sharp boundaries
- Ideal for sparse-angle CT (e.g., 20-60 projections)

# Update Rule (Gradient Descent)

```
x^(k+1) = x^(k) - α × [A^T(Ax - b) + λ × ∇TV(x)]
```

where ∇TV(x) is the gradient of total variation.

# Arguments

- `sinogram::Array{Float64, 3}` - Measured projection data
- `geometry::CTGeometry` - Scanner geometry
- `image_size::Int = 256` - Reconstructed image size
- `n_iterations::Int = 50` - Number of outer iterations
- `lambda::Float64 = 0.01` - TV regularization strength
- `tv_iterations::Int = 5` - Number of TV minimization sub-iterations

# Returns

- `volume::Array{Float64, 3}` - Reconstructed 3D volume

# Parameter Selection

**Lambda (λ)** controls smoothness:
- λ = 0: No regularization (standard least-squares)
- λ → small (0.001-0.01): Mild smoothing, preserve fine detail
- λ → large (0.1-1.0): Strong smoothing, cartoon-like images

**Rule of thumb**:
```
λ ≈ σ_noise² / ||∇x_true||
```

# Example

```julia
# Sparse-angle reconstruction (60 projections)
sparse_sinogram = sinogram[:, :, 1:6:360]  # Every 6th angle
sparse_geometry = create_geometry(num_projections=60)

# Reconstruct with TV regularization
volume = reconstruct_tv(
    sparse_sinogram,
    sparse_geometry,
    image_size=256,
    n_iterations=100,
    lambda=0.01  # Tuned for 60 projections
)
```

# Applications

1. **Sparse-angle CT**: 20-100 projections instead of 360-720
2. **Limited-angle CT**: Missing angular range (e.g., ±90°)
3. **Low-dose CT**: Noise reduction with edge preservation
4. **Metal artifact reduction**: Inpainting missing data

# References

- Rudin et al. (1992) Physica D - Original TV denoising
- Sidky & Pan (2008) Phys Med Biol - Compressed sensing CT
- Chen et al. (2008) IEEE TMI - Prior image constrained compressed sensing
"""
function reconstruct_tv(
        sinogram::Array{Float64, 3},
        geometry::CTGeometry,
        image_size::Int = 256;
        n_iterations::Int = 50,
        lambda::Float64 = 0.01,
        tv_iterations::Int = 5
    )::Array{Float64, 3}

    error("""
    TV reconstruction requires projection operators + optimization solver.

    Same forward/back projection limitations as SIRT/MLEM, plus:

    Objective function:
    ```
    E(x) = ||A×x - b||² + λ × TV(x)
    TV(x) = Σ |∇x|  (total variation)
    ```

    Implementation requirements:
    1. Forward/back projection (same as SIRT)
    2. Gradient operators (∇x, ∇y, ∇z)
    3. Optimization solver:
       - Split Bregman iteration (recommended)
       - ADMM (Alternating Direction Method of Multipliers)
       - Gradient descent with line search

    Use cases:
    - Sparse-angle CT (20-100 projections)
    - Limited-angle CT (±90° missing wedge)
    - Low-dose CT (edge-preserving denoising)
    - Metal artifact reduction

    Complexity: High (optimization + projectors)
    Timeline: Phase 5+ (research implementation)
    Estimated effort: 6-8 hours

    Alternatives:
    - MIRT.jl (Michigan Image Reconstruction Toolbox)
    - ImagePhantoms.jl + optimization packages
    - Standalone TV solvers (Condat, Chambolle-Pock)
    """)
end

# ==============================================================================
# Exports
# ==============================================================================

export reconstruct_sirt
export reconstruct_mlem
export reconstruct_tv
