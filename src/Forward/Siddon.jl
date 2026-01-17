# =============================================================================
# Siddon Forward Projection (TIGRE-style, AcceleratedKernels.jl)
# =============================================================================
#
# Direct port of TIGRE's Siddon algorithm using AcceleratedKernels.jl
# for backend-agnostic GPU/CPU execution.
#
# Algorithm Overview
# ------------------
# Siddon's algorithm (1985) computes exact radiological path lengths through a
# voxelized volume in O(N) time, where N is the number of voxels intersected
# by the ray. The algorithm uses parametric ray representation to efficiently
# determine ray-voxel intersections without testing every voxel.
#
# Mathematical Foundation
# -----------------------
# A ray from source S to detector D is represented parametrically:
#
#     P(t) = S + t × (D - S),  t ∈ [0, 1]
#
# The line integral L through the volume is:
#
#     L = ∫ μ(l) dl = Σᵢ μᵢ × lᵢ
#
# where μᵢ is the attenuation coefficient of voxel i and lᵢ is the physical
# path length (in mm) through that voxel.
#
# The algorithm proceeds by:
# 1. Computing parametric t values where ray intersects volume boundaries
# 2. Finding entry/exit points via t_enter = max(t_min) and t_exit = min(t_max)
# 3. Using 3D-DDA (Digital Differential Analyzer) to traverse voxels in order
# 4. Accumulating path-length weighted attenuation values
#
# Physical Units
# --------------
# - Source, detector positions: mm
# - Voxel sizes: mm
# - Attenuation coefficients μ: mm⁻¹ (if volume contains μ values)
# - Line integral output: dimensionless (μ × mm / mm) or mm (if volume contains 1s)
#
# Implementation Notes
# --------------------
# - Port of TIGRE's Siddon_projection.cu with modifications for Julia/AK
# - Uses Int32 throughout for GPU compatibility (Metal limitation)
# - Epsilon handling prevents division by zero for axis-aligned rays
# - Maximum iteration guard prevents infinite loops from numerical issues
#
# GPU Compatibility (via AcceleratedKernels.jl)
# ---------------------------------------------
# - ✅ Metal (Apple Silicon) - Primary development platform
# - ✅ CUDA (NVIDIA GPUs)
# - ✅ ROCm (AMD GPUs)
# - ✅ Intel oneAPI
# - ✅ CPU fallback (multi-threaded)
#
# The algorithm auto-detects backend from array type - no code changes needed.
# Simply pass MtlArray (Metal), CuArray (CUDA), or regular Array (CPU).
#
# References
# ----------
# 1. Siddon RL. "Fast calculation of the exact radiological path for a
#    three-dimensional CT array." Med Phys. 1985;12(2):252-255.
#    doi:10.1118/1.595715
#
# 2. Jacobs F, Sundermann E, De Sutter B, et al. "A fast algorithm to calculate
#    the exact radiological path through a pixel or voxel space." J Comput
#    Assist Tomogr. 1998;22(6):1003-1008. doi:10.1097/00004728-199811000-00027
#
# 3. Biguri A, Dosanjh M, Hancock S, Soleimani M. "TIGRE: A MATLAB-GPU toolbox
#    for CBCT image reconstruction." Biomed Phys Eng Express. 2016;2(5):055010.
#    doi:10.1088/2057-1976/2/5/055010
#
# =============================================================================

import AcceleratedKernels as AK

export siddon_forward_project!, siddon_forward_project

# =============================================================================
# Single Ray Trace (inlined into the main loop)
# =============================================================================

"""
    siddon_trace_ray(volume, src_x, src_y, src_z, det_x, det_y, det_z,
                     vol_min_x, vol_min_y, vol_min_z, vol_max_x, vol_max_y, vol_max_z,
                     voxel_size_x, voxel_size_y, voxel_size_z, nx, ny, nz) -> T

Trace a single ray through a 3D voxelized volume using Siddon's algorithm,
computing the exact line integral of attenuation coefficients.

This is an internal function called by [`siddon_forward_project!`](@ref).
It implements the core ray-tracing logic from Siddon (1985), ported from
TIGRE's Siddon_projection.cu with adaptations for Julia and AcceleratedKernels.jl.

# Algorithm

The algorithm computes the line integral L = Σᵢ μᵢ × lᵢ by:

1. **Parametric intersection**: Find t values where ray P(t) = S + t(D-S)
   intersects each axis-aligned plane of the voxel grid

2. **Entry/exit determination**: t_enter = max(t_x_min, t_y_min, t_z_min),
   t_exit = min(t_x_max, t_y_max, t_z_max)

3. **3D-DDA traversal**: Step through voxels in order of intersection,
   accumulating μᵢ × lᵢ for each voxel visited

4. **Path length computation**: lᵢ = (t_next - t_current) × ||D - S||

# Arguments

- `volume::AbstractArray{T,3}`: Voxelized attenuation volume [nx, ny, nz].
  Values are linear attenuation coefficients μ in mm⁻¹.
- `src_x, src_y, src_z::T`: X-ray source position in mm (world coordinates)
- `det_x, det_y, det_z::T`: Detector pixel center position in mm (world coordinates)
- `vol_min_x, vol_min_y, vol_min_z::T`: Volume minimum corner in mm
- `vol_max_x, vol_max_y, vol_max_z::T`: Volume maximum corner in mm
- `voxel_size_x, voxel_size_y, voxel_size_z::T`: Voxel dimensions in mm
- `nx, ny, nz::Int32`: Volume dimensions (Int32 for GPU compatibility)

# Returns

- `T`: Line integral value. If volume contains attenuation coefficients μ (mm⁻¹),
  the result is dimensionless (suitable for Beer-Lambert: I = I₀ × exp(-L)).
  If volume contains material indices or densities, interpret accordingly.

# Notes

- Uses `@inline` for GPU kernel efficiency
- Int32 dimensions required for Metal GPU compatibility
- Epsilon padding (1e-10) prevents division by zero for axis-aligned rays
- Maximum iteration guard (nx + ny + nz + 10) prevents infinite loops
- Rays missing the volume return zero (no intersection)

# References

1. Siddon RL. "Fast calculation of the exact radiological path for a
   three-dimensional CT array." Med Phys. 1985;12(2):252-255.
   doi:10.1118/1.595715

2. Jacobs F, et al. "A fast algorithm to calculate the exact radiological
   path through a pixel or voxel space." JCAT. 1998;22(6):1003-1008.
   doi:10.1097/00004728-199811000-00027

# See Also

- [`siddon_forward_project!`](@ref): High-level in-place projection
- [`siddon_forward_project`](@ref): Allocating projection
"""
@inline function siddon_trace_ray(
    volume::AbstractArray{T, 3},
    src_x::T, src_y::T, src_z::T,
    det_x::T, det_y::T, det_z::T,
    vol_min_x::T, vol_min_y::T, vol_min_z::T,
    vol_max_x::T, vol_max_y::T, vol_max_z::T,
    voxel_size_x::T, voxel_size_y::T, voxel_size_z::T,
    nx::Int32, ny::Int32, nz::Int32
) where T

    # Ray direction (not normalized - we use parametric form)
    ray_x = det_x - src_x
    ray_y = det_y - src_y
    ray_z = det_z - src_z

    # Ray length for proper path length scaling
    ray_length = sqrt(ray_x^2 + ray_y^2 + ray_z^2)

    # Avoid division by zero
    eps = T(1e-10)
    ray_x = abs(ray_x) < eps ? (ray_x >= zero(T) ? eps : -eps) : ray_x
    ray_y = abs(ray_y) < eps ? (ray_y >= zero(T) ? eps : -eps) : ray_y
    ray_z = abs(ray_z) < eps ? (ray_z >= zero(T) ? eps : -eps) : ray_z

    # Compute parametric t where ray intersects volume boundaries
    # Following TIGRE: t = (boundary - source) / ray
    t_x_min = (vol_min_x - src_x) / ray_x
    t_x_max = (vol_max_x - src_x) / ray_x
    t_y_min = (vol_min_y - src_y) / ray_y
    t_y_max = (vol_max_y - src_y) / ray_y
    t_z_min = (vol_min_z - src_z) / ray_z
    t_z_max = (vol_max_z - src_z) / ray_z

    # Sort to get entry/exit (TIGRE style)
    if t_x_min > t_x_max
        t_x_min, t_x_max = t_x_max, t_x_min
    end
    if t_y_min > t_y_max
        t_y_min, t_y_max = t_y_max, t_y_min
    end
    if t_z_min > t_z_max
        t_z_min, t_z_max = t_z_max, t_z_min
    end

    # Ray enters at max of entry t, exits at min of exit t
    t_enter = max(t_x_min, t_y_min, t_z_min)
    t_exit = min(t_x_max, t_y_max, t_z_max)

    # Check if ray misses volume
    if t_enter >= t_exit || t_exit <= zero(T)
        return zero(T)
    end

    # Clamp to positive t
    t_enter = max(t_enter, zero(T))

    # Entry point
    entry_x = src_x + t_enter * ray_x
    entry_y = src_y + t_enter * ray_y
    entry_z = src_z + t_enter * ray_z

    # Initial voxel indices (0-based for computation, Int32 for GPU)
    ix = unsafe_trunc(Int32, floor((entry_x - vol_min_x) / voxel_size_x))
    iy = unsafe_trunc(Int32, floor((entry_y - vol_min_y) / voxel_size_y))
    iz = unsafe_trunc(Int32, floor((entry_z - vol_min_z) / voxel_size_z))

    # Clamp to valid range
    ix = clamp(ix, Int32(0), nx - Int32(1))
    iy = clamp(iy, Int32(0), ny - Int32(1))
    iz = clamp(iz, Int32(0), nz - Int32(1))

    # Step direction (TIGRE: iu, ju, ku)
    step_x = ray_x >= zero(T) ? Int32(1) : Int32(-1)
    step_y = ray_y >= zero(T) ? Int32(1) : Int32(-1)
    step_z = ray_z >= zero(T) ? Int32(1) : Int32(-1)

    # Delta t to cross one voxel (TIGRE: axu, ayu, azu)
    dt_x = abs(voxel_size_x / ray_x)
    dt_y = abs(voxel_size_y / ray_y)
    dt_z = abs(voxel_size_z / ray_z)

    # t to next boundary
    if ray_x >= zero(T)
        t_next_x = t_enter + (vol_min_x + T(ix + Int32(1)) * voxel_size_x - entry_x) / ray_x
    else
        t_next_x = t_enter + (vol_min_x + T(ix) * voxel_size_x - entry_x) / ray_x
    end

    if ray_y >= zero(T)
        t_next_y = t_enter + (vol_min_y + T(iy + Int32(1)) * voxel_size_y - entry_y) / ray_y
    else
        t_next_y = t_enter + (vol_min_y + T(iy) * voxel_size_y - entry_y) / ray_y
    end

    if ray_z >= zero(T)
        t_next_z = t_enter + (vol_min_z + T(iz + Int32(1)) * voxel_size_z - entry_z) / ray_z
    else
        t_next_z = t_enter + (vol_min_z + T(iz) * voxel_size_z - entry_z) / ray_z
    end

    # Traverse voxels and accumulate (TIGRE main loop)
    t_current = t_enter
    line_integral = zero(T)

    # Maximum iterations to prevent infinite loops (use Int32)
    max_iter = nx + ny + nz + Int32(10)
    iter = Int32(0)

    while t_current < t_exit && iter < max_iter
        iter += Int32(1)

        # Check bounds
        if ix < Int32(0) || ix >= nx || iy < Int32(0) || iy >= ny || iz < Int32(0) || iz >= nz
            break
        end

        # Next boundary crossing
        t_next = min(t_next_x, t_next_y, t_next_z, t_exit)

        # Path length through this voxel
        path_length = (t_next - t_current) * ray_length

        if path_length > eps
            # Get voxel value (1-based indexing)
            voxel_val = volume[ix + Int32(1), iy + Int32(1), iz + Int32(1)]
            line_integral += voxel_val * path_length
        end

        # Step to next voxel (TIGRE style comparison)
        if t_next_x <= t_next_y && t_next_x <= t_next_z
            ix += step_x
            t_next_x += dt_x
        elseif t_next_y <= t_next_z
            iy += step_y
            t_next_y += dt_y
        else
            iz += step_z
            t_next_z += dt_z
        end

        t_current = t_next
    end

    return line_integral
end

# =============================================================================
# High-Level Interface using AcceleratedKernels.jl
# =============================================================================

"""
    siddon_forward_project!(sinogram, volume, geom) -> sinogram

Compute forward projection (sinogram) from a 3D attenuation volume using
Siddon's exact ray-tracing algorithm. This is an in-place operation that
modifies the sinogram array directly.

# Algorithm

Implements Siddon (1985) for CT forward projection:

1. For each projection angle θ and detector pixel (u, v):
   - Compute source position: S(θ) from gantry geometry
   - Compute detector pixel center: D(θ, u, v)
   - Trace ray from S to D through volume
   - Store line integral in sinogram

The line integral for each ray is:

    p(θ, u, v) = ∫ μ(x, y, z) dl = Σᵢ μᵢ × lᵢ

where μᵢ is the attenuation of voxel i and lᵢ is the path length through it.

# Arguments

- `sinogram::AbstractArray{T,3}`: Output sinogram array of size
  `[n_cols, n_rows, n_angles]`, modified in place. Will contain line integrals.
  - `n_cols`: Number of detector columns (transaxial direction)
  - `n_rows`: Number of detector rows (axial direction)
  - `n_angles`: Number of projection angles

- `volume::AbstractArray{T,3}`: Input attenuation volume of size `[nx, ny, nz]`.
  Values are linear attenuation coefficients μ in mm⁻¹. The volume is assumed
  to be centered at the isocenter with extent defined by `geom.fov`.

- `geom::CTGeometry`: CT scanner geometry containing:
  - `fov::Tuple{T,T,T}`: Field of view (x, y, z) in mm
  - `SAD::T`: Source-to-axis (isocenter) distance in mm
  - `SDD::T`: Source-to-detector distance in mm
  - `pixel_size::T`: Detector pixel size at isocenter in mm
  - `source_positions::Matrix{T}`: Source positions [3, n_angles]
  - `detector_centers::Matrix{T}`: Detector centers [3, n_angles]
  - `detector_u::Matrix{T}`: Detector u-direction vectors [3, n_angles]
  - `detector_v::Matrix{T}`: Detector v-direction vectors [3, n_angles]

# Returns

- `sinogram::AbstractArray{T,3}`: The modified sinogram array (same as input)

# GPU Compatibility

Automatically selects compute backend based on array type:

| Array Type | Backend | Performance |
|------------|---------|-------------|
| `Array` | CPU (multi-threaded) | Baseline |
| `MtlArray` | Metal (Apple Silicon) | ~50-100× faster |
| `CuArray` | CUDA (NVIDIA) | ~50-100× faster |
| `ROCArray` | ROCm (AMD) | ~50-100× faster |

No code changes needed - just pass appropriate array types.

# Example

```julia
using BasisSimulator
using Metal  # or: using CUDA

# Create geometry for GE Revolution Apex-like scanner
scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=360, fov=(350.0, 350.0, 40.0))

# Create a simple water cylinder phantom
phantom = zeros(Float32, 256, 256, 64)
for i in 1:256, j in 1:256, k in 1:64
    x, y = (i - 128.5) * 1.37, (j - 128.5) * 1.37  # mm from center
    if x^2 + y^2 < 100^2  # 100mm radius cylinder
        phantom[i, j, k] = 0.02f0  # μ_water ≈ 0.02 mm⁻¹ at 60 keV
    end
end

# GPU forward projection
phantom_gpu = MtlArray(phantom)  # or CuArray
sinogram_gpu = similar(phantom_gpu, Float32, geom.n_cols, geom.n_rows, geom.n_angles)
fill!(sinogram_gpu, 0f0)

siddon_forward_project!(sinogram_gpu, phantom_gpu, geom)

# Result: sinogram_gpu contains line integrals for FDK reconstruction
```

# Performance Notes

- O(N) complexity per ray, where N = number of voxels intersected
- Typical CT ray traverses ~256-512 voxels
- GPU parallelization over all (n_cols × n_rows × n_angles) rays
- Memory bandwidth limited on GPU; compute-limited on CPU

# References

1. Siddon RL. "Fast calculation of the exact radiological path for a
   three-dimensional CT array." Med Phys. 1985;12(2):252-255.
   doi:10.1118/1.595715

2. Biguri A, et al. "TIGRE: A MATLAB-GPU toolbox for CBCT image
   reconstruction." Biomed Phys Eng Express. 2016;2(5):055010.
   doi:10.1088/2057-1976/2/5/055010

# See Also

- [`siddon_forward_project`](@ref): Allocating version (creates sinogram)
- [`polychromatic_forward_project`](@ref): Spectral projection with energy dependence
- [`fdk_reconstruct`](@ref): Filtered backprojection reconstruction
"""
function siddon_forward_project!(
    sinogram::AbstractArray{T, 3},
    volume::AbstractArray{T, 3},
    geom::CTGeometry
) where T <: AbstractFloat

    # Get dimensions as Int32 for GPU compatibility
    nx = Int32(size(volume, 1))
    ny = Int32(size(volume, 2))
    nz = Int32(size(volume, 3))
    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Pre-compute volume parameters (typed constants for GPU)
    vol_min_x = T(-geom.fov[1] / 2)
    vol_min_y = T(-geom.fov[2] / 2)
    vol_min_z = T(-geom.fov[3] / 2)
    vol_max_x = T(geom.fov[1] / 2)
    vol_max_y = T(geom.fov[2] / 2)
    vol_max_z = T(geom.fov[3] / 2)
    voxel_size_x = T(geom.fov[1]) / T(nx)
    voxel_size_y = T(geom.fov[2]) / T(ny)
    voxel_size_z = T(geom.fov[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)

    # Pre-compute detector center offset for GPU
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Extract geometry and convert to same array type as volume (GPU compatibility)
    source_positions = similar(volume, T, size(geom.source_positions)...)
    copyto!(source_positions, T.(geom.source_positions))
    detector_centers = similar(volume, T, size(geom.detector_centers)...)
    copyto!(detector_centers, T.(geom.detector_centers))
    detector_u = similar(volume, T, size(geom.detector_u)...)
    copyto!(detector_u, T.(geom.detector_u))
    detector_v = similar(volume, T, size(geom.detector_v)...)
    copyto!(detector_v, T.(geom.detector_v))

    # Use AcceleratedKernels.jl to parallelize over all rays
    AK.foreachindex(sinogram) do idx
        # Convert linear index to (col, row, angle) using integer arithmetic
        # Use Int32 throughout to avoid boxing
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)
        angle = (idx_0 ÷ n_rows) + Int32(1)

        # Source position for this angle
        src_x = source_positions[1, angle]
        src_y = source_positions[2, angle]
        src_z = source_positions[3, angle]

        # Detector center and orientation
        dcx = detector_centers[1, angle]
        dcy = detector_centers[2, angle]
        dcz = detector_centers[3, angle]

        dux = detector_u[1, angle]
        duy = detector_u[2, angle]
        duz = detector_u[3, angle]

        dvx = detector_v[1, angle]
        dvy = detector_v[2, angle]
        dvz = detector_v[3, angle]

        # Compute detector pixel position
        u_offset = (T(col) - col_center) * pixel_size * magnification
        v_offset = (T(row) - row_center) * pixel_size * magnification

        det_x = dcx + u_offset * dux + v_offset * dvx
        det_y = dcy + u_offset * duy + v_offset * dvy
        det_z = dcz + u_offset * duz + v_offset * dvz

        # Trace ray and store result
        sinogram[idx] = siddon_trace_ray(
            volume,
            src_x, src_y, src_z,
            det_x, det_y, det_z,
            vol_min_x, vol_min_y, vol_min_z,
            vol_max_x, vol_max_y, vol_max_z,
            voxel_size_x, voxel_size_y, voxel_size_z,
            nx, ny, nz
        )
    end

    return sinogram
end

"""
    siddon_forward_project(volume, geom) -> sinogram

Compute forward projection (sinogram) from a 3D attenuation volume using
Siddon's exact ray-tracing algorithm. This is an allocating version that
creates and returns a new sinogram array.

# Algorithm

Implements Siddon (1985) for CT forward projection. See
[`siddon_forward_project!`](@ref) for detailed algorithm description.

# Arguments

- `volume::AbstractArray{T,3}`: Input attenuation volume of size `[nx, ny, nz]`.
  Values are linear attenuation coefficients μ in mm⁻¹.

- `geom::CTGeometry`: CT scanner geometry (see [`siddon_forward_project!`](@ref)
  for full description of required fields).

# Returns

- `sinogram::AbstractArray{T,3}`: Newly allocated sinogram of size
  `[n_cols, n_rows, n_angles]` containing line integrals. The array is
  allocated on the same device as `volume` (CPU, Metal, CUDA, etc.).

# GPU Compatibility

The returned sinogram is allocated on the same device as the input volume:

```julia
# CPU version
sinogram = siddon_forward_project(Array(phantom), geom)  # returns Array

# GPU version (Metal)
sinogram = siddon_forward_project(MtlArray(phantom), geom)  # returns MtlArray

# GPU version (CUDA)
sinogram = siddon_forward_project(CuArray(phantom), geom)  # returns CuArray
```

# Example

```julia
using BasisSimulator

# Create scanner and geometry
scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=180, fov=(300.0, 300.0, 32.0))

# Create uniform water phantom (μ ≈ 0.02 mm⁻¹)
phantom = fill(0.02f0, 128, 128, 32)

# Forward projection - automatically allocates sinogram
sinogram = siddon_forward_project(phantom, geom)

# sinogram is Float32 Array of size (geom.n_cols, geom.n_rows, 180)
println("Sinogram size: ", size(sinogram))
println("Mean projection value: ", mean(sinogram))
```

# Performance Notes

For repeated projections (e.g., iterative reconstruction), prefer the in-place
version [`siddon_forward_project!`](@ref) to avoid repeated allocations.

# References

1. Siddon RL. "Fast calculation of the exact radiological path for a
   three-dimensional CT array." Med Phys. 1985;12(2):252-255.
   doi:10.1118/1.595715

2. Biguri A, et al. "TIGRE: A MATLAB-GPU toolbox for CBCT image
   reconstruction." Biomed Phys Eng Express. 2016;2(5):055010.
   doi:10.1088/2057-1976/2/5/055010

# See Also

- [`siddon_forward_project!`](@ref): In-place version (avoids allocation)
- [`polychromatic_forward_project`](@ref): Spectral projection with energy dependence
- [`fdk_reconstruct`](@ref): Filtered backprojection reconstruction
"""
function siddon_forward_project(
    volume::AbstractArray{T, 3},
    geom::CTGeometry
) where T <: AbstractFloat

    # Allocate output on same device as input
    sinogram = similar(volume, T, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, zero(T))

    return siddon_forward_project!(sinogram, volume, geom)
end
