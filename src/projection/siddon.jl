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

export siddon_forward_project!, siddon_forward_project, siddon_fused_poly_project!, siddon_tiled_poly_project!, siddon_fused_spectral_project!

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

        # Branchless DDA step (SPEED-BUILD-003)
        mx = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
        my = Int32(Int32(1) - mx) * Int32(t_next_y <= t_next_z)
        mz = Int32(1) - mx - my
        ix += mx * step_x
        iy += my * step_y
        iz += mz * step_z
        t_next_x += T(mx) * dt_x
        t_next_y += T(my) * dt_y
        t_next_z += T(mz) * dt_z

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
    geom::CTGeometry;
    # Pre-allocated geometry arrays (optional — allocate internally if not provided)
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    # Override volume bounds (use phantom's physical extent instead of geom.fov)
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing
) where T <: AbstractFloat

    # Get dimensions as Int32 for GPU compatibility
    nx = Int32(size(volume, 1))
    ny = Int32(size(volume, 2))
    nz = Int32(size(volume, 3))
    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Pre-compute volume parameters (typed constants for GPU)
    # Use volume_extent (phantom's physical dimensions) if provided,
    # otherwise fall back to geom.fov (reconstruction FOV).
    # These differ when the phantom extent ≠ reconstruction FOV.
    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vol_min_x = T(-vol_bounds[1] / 2)
    vol_min_y = T(-vol_bounds[2] / 2)
    vol_min_z = T(-vol_bounds[3] / 2)
    vol_max_x = T(vol_bounds[1] / 2)
    vol_max_y = T(vol_bounds[2] / 2)
    vol_max_z = T(vol_bounds[3] / 2)
    voxel_size_x = T(vol_bounds[1]) / T(nx)
    voxel_size_y = T(vol_bounds[2]) / T(ny)
    voxel_size_z = T(vol_bounds[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)

    # Pre-compute detector center offset for GPU
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Extract geometry and convert to same array type as volume (GPU compatibility)
    # Use pre-allocated workspace buffers if provided to avoid allocation
    source_positions = if ws_source_positions !== nothing
        ws_source_positions
    else
        _sp = similar(volume, T, size(geom.source_positions)...)
        copyto!(_sp, T.(geom.source_positions))
        _sp
    end
    detector_centers = if ws_detector_centers !== nothing
        ws_detector_centers
    else
        _dc = similar(volume, T, size(geom.detector_centers)...)
        copyto!(_dc, T.(geom.detector_centers))
        _dc
    end
    detector_u = if ws_detector_u !== nothing
        ws_detector_u
    else
        _du = similar(volume, T, size(geom.detector_u)...)
        copyto!(_du, T.(geom.detector_u))
        _du
    end
    detector_v = if ws_detector_v !== nothing
        ws_detector_v
    else
        _dv = similar(volume, T, size(geom.detector_v)...)
        copyto!(_dv, T.(geom.detector_v))
        _dv
    end

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
        v_offset = (T(row) - row_center) * pixel_row_size * magnification

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
    geom::CTGeometry;
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing
) where T <: AbstractFloat

    # Allocate output on same device as input
    sinogram = similar(volume, T, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, zero(T))

    return siddon_forward_project!(sinogram, volume, geom; volume_extent=volume_extent)
end

# =============================================================================
# Fused Polychromatic Siddon Projection (SPEED-BUILD-001)
# =============================================================================
#
# Single-kernel fused polychromatic projection: traces each ray ONCE through
# the UInt16 material mask and accumulates line integrals for ALL energy bins
# via μ_table lookup. Replaces 90+ sequential kernel launches with 1.
#
# Bandwidth reduction: ~30x (reads mask once instead of Float32 volume 30x)
# Kernel launch reduction: 90+ → 1
#
# Uses NTuple{N_E,T} accumulators for register-resident energy storage.
# N_E is a compile-time constant via Val{N_E} for loop unrolling.
#
# IMPORTANT: Uses @generated functions for NTuple manipulation to avoid the
# catastrophic slowdown from ntuple(f, Val(N)) closure overhead on CPU.
# @generated expands to explicit tuple(old[1]+..., old[2]+..., ...) at compile
# time, producing clean LLVM IR that both CPU and GPU compilers optimize well.
# =============================================================================

"""
    _fused_accum_energies(accums, μ_tbl, mat, path_length)

@generated helper: accumulate path_length × μ for all energies.
Expands at compile time to: tuple(accums[1] + μ_tbl[mat,1]*pl, accums[2] + μ_tbl[mat,2]*pl, ...)
"""
@generated function _fused_accum_energies(accums::NTuple{N,T}, μ_tbl, mat::Int32, pl::T) where {N, T}
    exprs = [:(accums[$i] + @inbounds(μ_tbl[mat, $i]) * pl) for i in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _fused_beer_lambert(accums, wη)

@generated helper: Beer-Lambert sum without bowtie.
Returns Σ_e wη[e] * exp(-accums[e]).
"""
@generated function _fused_beer_lambert(accums::NTuple{N,T}, wη) where {N, T}
    if N == 0
        return :(zero(T))
    end
    # Chain binary + to avoid varargs allocation on GPU
    ex = :((@inbounds wη[1]) * exp(-accums[1]))
    for i in 2:N
        ex = :($ex + (@inbounds wη[$i]) * exp(-accums[$i]))
    end
    return ex
end

"""
    _fused_beer_lambert_bt(accums, wη, bt, bt_base, ncnr)

@generated helper: Beer-Lambert sum with bowtie spectral transmission.
bt_base = col + (row-1)*nc, ncnr = n_cols*n_rows.
Returns Σ_e wη[e] * bt[bt_base + (e-1)*ncnr] * exp(-accums[e]).
"""
@generated function _fused_beer_lambert_bt(accums::NTuple{N,T}, wη, bt, bt_base::Int32, ncnr::Int32) where {N, T}
    if N == 0
        return :(zero(T))
    end
    # Chain binary + to avoid varargs allocation on GPU
    ex = :((@inbounds wη[1]) * (@inbounds bt[bt_base + Int32(0) * ncnr]) * exp(-accums[1]))
    for i in 2:N
        ex = :($ex + (@inbounds wη[$i]) * (@inbounds bt[bt_base + Int32($(i - 1)) * ncnr]) * exp(-accums[$i]))
    end
    return ex
end

"""
    siddon_fused_poly_project!(sinogram, mask, geom, μ_table_gpu, wη_gpu, Val(N_E); kwargs...)

Fused polychromatic forward projection: single DDA pass through material mask,
accumulating line integrals for all energy bins simultaneously.

Mathematically equivalent to the unfused path (sequential energy loop), but
traces each ray only ONCE instead of N_E times. The material mask (UInt16) is
read once per voxel, and μ values for all energies are looked up from a
[n_materials × n_energies] table that fits in GPU cache.

# Arguments
- `sinogram::AbstractArray{T,3}`: Output [n_cols, n_rows, n_angles]. Written in-place
  with -log(I_total) values.
- `mask::AbstractArray{<:Unsigned,3}`: Material index volume [nx, ny, nz]. 0-indexed
  region indices (UInt8 or UInt16). Supports up to 65,535 materials with UInt16.
- `geom::CTGeometry`: Scanner geometry.
- `μ_table_gpu::AbstractArray{T,2}`: Attenuation lookup table [n_materials, n_energies]
  on GPU. μ_table[mat+1, e] gives μ in mm⁻¹ for material `mat` at energy bin `e`.
- `wη_gpu::AbstractArray{T,1}`: Pre-computed `weights_norm .* η` on GPU [n_energies].
- `::Val{N_E}`: Number of energy bins as compile-time constant for loop unrolling.

# Keyword Arguments
- `volume_extent`: Override volume physical bounds (for phantoms with extent ≠ recon FOV).
- `ws_source_positions`, `ws_detector_centers`, `ws_detector_u`, `ws_detector_v`:
  Pre-allocated geometry arrays on GPU (avoids per-call allocation).
- `ws_bowtie_spectral`: Per-pixel bowtie spectral transmission [n_cols, n_rows, n_energies]
  on GPU. If nothing, no bowtie correction applied.
"""
function siddon_fused_poly_project!(
    sinogram::AbstractArray{T, 3},
    mask::AbstractArray{<:Unsigned, 3},
    geom::CTGeometry,
    μ_table_gpu::AbstractArray{T, 2},
    wη_gpu::AbstractArray{T, 1},
    ::Val{N_E};
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    ws_bowtie_spectral = nothing
) where {T <: AbstractFloat, N_E}

    # Volume dimensions (Int32 for GPU)
    nx = Int32(size(mask, 1))
    ny = Int32(size(mask, 2))
    nz = Int32(size(mask, 3))
    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Volume bounds (use phantom extent if provided, else recon FOV)
    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vol_min_x = T(-vol_bounds[1] / 2)
    vol_min_y = T(-vol_bounds[2] / 2)
    vol_min_z = T(-vol_bounds[3] / 2)
    vol_max_x = T(vol_bounds[1] / 2)
    vol_max_y = T(vol_bounds[2] / 2)
    vol_max_z = T(vol_bounds[3] / 2)
    voxel_size_x = T(vol_bounds[1]) / T(nx)
    voxel_size_y = T(vol_bounds[2]) / T(ny)
    voxel_size_z = T(vol_bounds[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Geometry arrays on GPU (use workspace or allocate)
    source_positions = if ws_source_positions !== nothing
        ws_source_positions
    else
        _sp = similar(sinogram, T, size(geom.source_positions)...)
        copyto!(_sp, T.(geom.source_positions))
        _sp
    end
    detector_centers = if ws_detector_centers !== nothing
        ws_detector_centers
    else
        _dc = similar(sinogram, T, size(geom.detector_centers)...)
        copyto!(_dc, T.(geom.detector_centers))
        _dc
    end
    detector_u = if ws_detector_u !== nothing
        ws_detector_u
    else
        _du = similar(sinogram, T, size(geom.detector_u)...)
        copyto!(_du, T.(geom.detector_u))
        _du
    end
    detector_v = if ws_detector_v !== nothing
        ws_detector_v
    else
        _dv = similar(sinogram, T, size(geom.detector_v)...)
        copyto!(_dv, T.(geom.detector_v))
        _dv
    end

    # Bowtie layout constants
    nc_nr = n_cols * n_rows
    has_bowtie = ws_bowtie_spectral !== nothing

    # Capture all variables in let block for GPU closure correctness
    let mask=mask, μ_tbl=μ_table_gpu, wη=wη_gpu,
        sp=source_positions, dc=detector_centers, du=detector_u, dv=detector_v,
        bt=ws_bowtie_spectral,
        vmx=vol_min_x, vmy=vol_min_y, vmz=vol_min_z,
        vMx=vol_max_x, vMy=vol_max_y, vMz=vol_max_z,
        vsx=voxel_size_x, vsy=voxel_size_y, vsz=voxel_size_z,
        nx=nx, ny=ny, nz=nz, nc=n_cols, nr=n_rows,
        mag=magnification, ps=pixel_size, prs=pixel_row_size,
        cc=col_center, rc=row_center, ncnr=nc_nr, hbt=has_bowtie

        AK.foreachindex(sinogram) do idx
            # ─── Index decomposition (mod/div, no CartesianIndices) ───
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            # ─── Source position ───
            src_x = sp[1, angle]
            src_y = sp[2, angle]
            src_z = sp[3, angle]

            # ─── Detector pixel position ───
            dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
            dux = du[1, angle]; duy = du[2, angle]; duz = du[3, angle]
            dvx = dv[1, angle]; dvy = dv[2, angle]; dvz = dv[3, angle]

            u_offset = (T(col) - cc) * ps * mag
            v_offset = (T(row) - rc) * prs * mag

            det_x = dcx + u_offset * dux + v_offset * dvx
            det_y = dcy + u_offset * duy + v_offset * dvy
            det_z = dcz + u_offset * duz + v_offset * dvz

            # ═══════════════════════════════════════════════════════════
            # DDA SETUP (from siddon_trace_ray — identical geometry)
            # ═══════════════════════════════════════════════════════════
            ray_x = det_x - src_x
            ray_y = det_y - src_y
            ray_z = det_z - src_z
            ray_length = sqrt(ray_x^2 + ray_y^2 + ray_z^2)

            eps = T(1e-10)
            ray_x = abs(ray_x) < eps ? (ray_x >= zero(T) ? eps : -eps) : ray_x
            ray_y = abs(ray_y) < eps ? (ray_y >= zero(T) ? eps : -eps) : ray_y
            ray_z = abs(ray_z) < eps ? (ray_z >= zero(T) ? eps : -eps) : ray_z

            t_x_min = (vmx - src_x) / ray_x
            t_x_max = (vMx - src_x) / ray_x
            t_y_min = (vmy - src_y) / ray_y
            t_y_max = (vMy - src_y) / ray_y
            t_z_min = (vmz - src_z) / ray_z
            t_z_max = (vMz - src_z) / ray_z

            if t_x_min > t_x_max; t_x_min, t_x_max = t_x_max, t_x_min end
            if t_y_min > t_y_max; t_y_min, t_y_max = t_y_max, t_y_min end
            if t_z_min > t_z_max; t_z_min, t_z_max = t_z_max, t_z_min end

            t_enter = max(t_x_min, t_y_min, t_z_min)
            t_exit  = min(t_x_max, t_y_max, t_z_max)

            # Ray misses volume → zero output
            if t_enter >= t_exit || t_exit <= zero(T)
                sinogram[idx] = zero(T)
                return
            end

            t_enter = max(t_enter, zero(T))

            entry_x = src_x + t_enter * ray_x
            entry_y = src_y + t_enter * ray_y
            entry_z = src_z + t_enter * ray_z

            # Initial voxel (0-based, Int32)
            ix = unsafe_trunc(Int32, floor((entry_x - vmx) / vsx))
            iy = unsafe_trunc(Int32, floor((entry_y - vmy) / vsy))
            iz = unsafe_trunc(Int32, floor((entry_z - vmz) / vsz))
            ix = clamp(ix, Int32(0), nx - Int32(1))
            iy = clamp(iy, Int32(0), ny - Int32(1))
            iz = clamp(iz, Int32(0), nz - Int32(1))

            # Step direction
            step_x = ray_x >= zero(T) ? Int32(1) : Int32(-1)
            step_y = ray_y >= zero(T) ? Int32(1) : Int32(-1)
            step_z = ray_z >= zero(T) ? Int32(1) : Int32(-1)

            # Delta-t per voxel
            dt_x = abs(vsx / ray_x)
            dt_y = abs(vsy / ray_y)
            dt_z = abs(vsz / ray_z)

            # t to next boundary
            if ray_x >= zero(T)
                t_next_x = t_enter + (vmx + T(ix + Int32(1)) * vsx - entry_x) / ray_x
            else
                t_next_x = t_enter + (vmx + T(ix) * vsx - entry_x) / ray_x
            end
            if ray_y >= zero(T)
                t_next_y = t_enter + (vmy + T(iy + Int32(1)) * vsy - entry_y) / ray_y
            else
                t_next_y = t_enter + (vmy + T(iy) * vsy - entry_y) / ray_y
            end
            if ray_z >= zero(T)
                t_next_z = t_enter + (vmz + T(iz + Int32(1)) * vsz - entry_z) / ray_z
            else
                t_next_z = t_enter + (vmz + T(iz) * vsz - entry_z) / ray_z
            end

            # ═══════════════════════════════════════════════════════════
            # ENERGY ACCUMULATORS (NTuple in registers)
            # ═══════════════════════════════════════════════════════════
            accums = ntuple(_ -> zero(T), Val(N_E))

            # ═══════════════════════════════════════════════════════════
            # DDA TRAVERSAL — single pass, all energies accumulated
            # ═══════════════════════════════════════════════════════════
            t_current = t_enter
            max_iter = nx + ny + nz + Int32(10)
            iter = Int32(0)

            while t_current < t_exit && iter < max_iter
                iter += Int32(1)

                # Bounds check
                if ix < Int32(0) || ix >= nx || iy < Int32(0) || iy >= ny || iz < Int32(0) || iz >= nz
                    break
                end

                t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
                path_length = (t_next - t_current) * ray_length

                if path_length > eps
                    # Read material index from mask (UInt8 or UInt16 → Int32)
                    mat = Int32(mask[ix + Int32(1), iy + Int32(1), iz + Int32(1)]) + Int32(1)

                    # Accumulate line integrals for ALL energies (@generated unroll)
                    accums = _fused_accum_energies(accums, μ_tbl, mat, path_length)
                end

                # Branchless DDA step (SPEED-BUILD-003)
                mx = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
                my = Int32(Int32(1) - mx) * Int32(t_next_y <= t_next_z)
                mz = Int32(1) - mx - my
                ix += mx * step_x
                iy += my * step_y
                iz += mz * step_z
                t_next_x += T(mx) * dt_x
                t_next_y += T(my) * dt_y
                t_next_z += T(mz) * dt_z

                t_current = t_next
            end

            # ═══════════════════════════════════════════════════════════
            # BEER-LAMBERT SUMMATION (@generated unroll, replaces 30 kernels)
            # ═══════════════════════════════════════════════════════════
            I_total = if hbt
                bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
                _fused_beer_lambert_bt(accums, wη, bt, bt_base, ncnr)
            else
                _fused_beer_lambert(accums, wη)
            end

            sinogram[idx] = -log(max(I_total, T(1e-10)))
        end
    end

    return sinogram
end

# =============================================================================
# Tiled Polychromatic Siddon Projection (SPEED-BUILD-V2-002)
# =============================================================================
#
# Tiled variant of the fused kernel: processes K energy bins per tile (K=16).
# Each tile traces DDA once through the material mask and accumulates K energy
# bins. Multiple tiles are called sequentially to cover all 234 energy bins.
#
# Why K=16: 16 Float32 accumulators = 64 bytes — fits in GPU registers without
# spilling. Val(234) requires 936 bytes → massive register spilling on Metal.
# K=16 × 15 tiles = 240 bins (234 real + 6 zero-padded).
#
# Key difference from fused kernel:
# - Outputs partial Beer-Lambert sum ADDED to I_buffer (not -log to sinogram)
# - Takes tile_start offset for μ_table/wη/bowtie indexing
# - After all tiles, caller applies -log(I_buffer) to get sinogram
# =============================================================================

"""
    _tiled_accum_energies(accums, μ_tbl, mat, path_length, ts)

@generated helper: accumulate path_length × μ for K energies starting at column `ts`.
Expands to: tuple(accums[1] + μ_tbl[mat, ts+0]*pl, accums[2] + μ_tbl[mat, ts+1]*pl, ...)
"""
@generated function _tiled_accum_energies(accums::NTuple{K,T}, μ_tbl, mat::Int32, pl::T, ts::Int32) where {K, T}
    exprs = [:(accums[$i] + @inbounds(μ_tbl[mat, ts + Int32($(i - 1))]) * pl) for i in 1:K]
    return :(tuple($(exprs...)))
end

"""
    _tiled_beer_lambert(accums, wη, ts)

@generated helper: partial Beer-Lambert sum without bowtie, starting at index `ts`.
Returns Σ_e wη[ts+e-1] * exp(-accums[e]) for e=1:K.
"""
@generated function _tiled_beer_lambert(accums::NTuple{K,T}, wη, ts::Int32) where {K, T}
    if K == 0
        return :(zero(T))
    end
    ex = :((@inbounds wη[ts]) * exp(-accums[1]))
    for i in 2:K
        ex = :($ex + (@inbounds wη[ts + Int32($(i - 1))]) * exp(-accums[$i]))
    end
    return ex
end

"""
    _tiled_beer_lambert_bt(accums, wη, bt, bt_base, ncnr, ts)

@generated helper: partial Beer-Lambert sum with bowtie, starting at index `ts`.
Bowtie layout: bt[col + (row-1)*nc + (energy-1)*nc*nr], energy 1-based global index.
For tile starting at `ts`, global energy index = ts + e - 1.
"""
@generated function _tiled_beer_lambert_bt(accums::NTuple{K,T}, wη, bt, bt_base::Int32, ncnr::Int32, ts::Int32) where {K, T}
    if K == 0
        return :(zero(T))
    end
    # e=1: global energy = ts, bowtie offset = (ts-1)*ncnr
    ex = :((@inbounds wη[ts]) * (@inbounds bt[bt_base + (ts - Int32(1)) * ncnr]) * exp(-accums[1]))
    for i in 2:K
        # e=i: global energy = ts+i-1, bowtie offset = (ts+i-2)*ncnr
        ex = :($ex + (@inbounds wη[ts + Int32($(i - 1))]) * (@inbounds bt[bt_base + (ts + Int32($(i - 2))) * ncnr]) * exp(-accums[$i]))
    end
    return ex
end

"""
    siddon_tiled_poly_project!(I_buffer, mask, geom, μ_table_gpu, wη_gpu, Val(K), tile_start; kwargs...)

Tiled polychromatic forward projection: single DDA pass through material mask,
accumulating K energy bins (starting at `tile_start`) and ADDING partial
Beer-Lambert sum to `I_buffer`.

Called in a loop over tiles to process all energy bins. After all tiles,
the caller applies `-log(I_buffer)` to get the sinogram.

# Arguments
- `I_buffer::AbstractArray{T,3}`: Output [n_cols, n_rows, n_angles]. Partial
  Beer-Lambert sum is ADDED to existing values (must be zero-initialized before first tile).
- `mask::AbstractArray{<:Unsigned,3}`: Material index volume [nx, ny, nz].
- `geom::CTGeometry`: Scanner geometry.
- `μ_table_gpu::AbstractArray{T,2}`: Attenuation lookup table [n_materials, n_energies_padded].
- `wη_gpu::AbstractArray{T,1}`: Pre-computed `weights_norm .* η` [n_energies_padded].
- `::Val{K}`: Tile size (compile-time constant, typically 16).
- `tile_start::Int32`: 1-based start index into energy dimension for this tile.

# Keyword Arguments
- `volume_extent`: Override volume physical bounds.
- `ws_source_positions`, `ws_detector_centers`, `ws_detector_u`, `ws_detector_v`:
  Pre-allocated geometry arrays on GPU.
- `ws_bowtie_spectral`: Per-pixel bowtie spectral transmission [n_cols, n_rows, n_energies_padded].
"""
function siddon_tiled_poly_project!(
    I_buffer::AbstractArray{T, 3},
    mask::AbstractArray{<:Unsigned, 3},
    geom::CTGeometry,
    μ_table_gpu::AbstractArray{T, 2},
    wη_gpu::AbstractArray{T, 1},
    ::Val{K},
    tile_start::Int32;
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    ws_bowtie_spectral = nothing
) where {T <: AbstractFloat, K}

    # Volume dimensions (Int32 for GPU)
    nx = Int32(size(mask, 1))
    ny = Int32(size(mask, 2))
    nz = Int32(size(mask, 3))
    n_cols = Int32(size(I_buffer, 1))
    n_rows = Int32(size(I_buffer, 2))
    n_angles = Int32(size(I_buffer, 3))

    # Volume bounds
    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vol_min_x = T(-vol_bounds[1] / 2)
    vol_min_y = T(-vol_bounds[2] / 2)
    vol_min_z = T(-vol_bounds[3] / 2)
    vol_max_x = T(vol_bounds[1] / 2)
    vol_max_y = T(vol_bounds[2] / 2)
    vol_max_z = T(vol_bounds[3] / 2)
    voxel_size_x = T(vol_bounds[1]) / T(nx)
    voxel_size_y = T(vol_bounds[2]) / T(ny)
    voxel_size_z = T(vol_bounds[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Geometry arrays on GPU (use workspace or allocate)
    source_positions = if ws_source_positions !== nothing
        ws_source_positions
    else
        _sp = similar(I_buffer, T, size(geom.source_positions)...)
        copyto!(_sp, T.(geom.source_positions))
        _sp
    end
    detector_centers = if ws_detector_centers !== nothing
        ws_detector_centers
    else
        _dc = similar(I_buffer, T, size(geom.detector_centers)...)
        copyto!(_dc, T.(geom.detector_centers))
        _dc
    end
    detector_u = if ws_detector_u !== nothing
        ws_detector_u
    else
        _du = similar(I_buffer, T, size(geom.detector_u)...)
        copyto!(_du, T.(geom.detector_u))
        _du
    end
    detector_v = if ws_detector_v !== nothing
        ws_detector_v
    else
        _dv = similar(I_buffer, T, size(geom.detector_v)...)
        copyto!(_dv, T.(geom.detector_v))
        _dv
    end

    # Bowtie layout constants
    nc_nr = n_cols * n_rows
    has_bowtie = ws_bowtie_spectral !== nothing

    # Capture all variables in let block for GPU closure correctness
    let mask=mask, μ_tbl=μ_table_gpu, wη=wη_gpu,
        sp=source_positions, dc=detector_centers, du=detector_u, dv=detector_v,
        bt=ws_bowtie_spectral, ts=tile_start,
        vmx=vol_min_x, vmy=vol_min_y, vmz=vol_min_z,
        vMx=vol_max_x, vMy=vol_max_y, vMz=vol_max_z,
        vsx=voxel_size_x, vsy=voxel_size_y, vsz=voxel_size_z,
        nx=nx, ny=ny, nz=nz, nc=n_cols, nr=n_rows,
        mag=magnification, ps=pixel_size, prs=pixel_row_size,
        cc=col_center, rc=row_center, ncnr=nc_nr, hbt=has_bowtie

        AK.foreachindex(I_buffer) do idx
            # ─── Index decomposition ───
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            # ─── Source position ───
            src_x = sp[1, angle]
            src_y = sp[2, angle]
            src_z = sp[3, angle]

            # ─── Detector pixel position ───
            dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
            dux = du[1, angle]; duy = du[2, angle]; duz = du[3, angle]
            dvx = dv[1, angle]; dvy = dv[2, angle]; dvz = dv[3, angle]

            u_offset = (T(col) - cc) * ps * mag
            v_offset = (T(row) - rc) * prs * mag

            det_x = dcx + u_offset * dux + v_offset * dvx
            det_y = dcy + u_offset * duy + v_offset * dvy
            det_z = dcz + u_offset * duz + v_offset * dvz

            # ═══════════════════════════════════════════════════════════
            # DDA SETUP
            # ═══════════════════════════════════════════════════════════
            ray_x = det_x - src_x
            ray_y = det_y - src_y
            ray_z = det_z - src_z
            ray_length = sqrt(ray_x^2 + ray_y^2 + ray_z^2)

            eps = T(1e-10)
            ray_x = abs(ray_x) < eps ? (ray_x >= zero(T) ? eps : -eps) : ray_x
            ray_y = abs(ray_y) < eps ? (ray_y >= zero(T) ? eps : -eps) : ray_y
            ray_z = abs(ray_z) < eps ? (ray_z >= zero(T) ? eps : -eps) : ray_z

            t_x_min = (vmx - src_x) / ray_x
            t_x_max = (vMx - src_x) / ray_x
            t_y_min = (vmy - src_y) / ray_y
            t_y_max = (vMy - src_y) / ray_y
            t_z_min = (vmz - src_z) / ray_z
            t_z_max = (vMz - src_z) / ray_z

            if t_x_min > t_x_max; t_x_min, t_x_max = t_x_max, t_x_min end
            if t_y_min > t_y_max; t_y_min, t_y_max = t_y_max, t_y_min end
            if t_z_min > t_z_max; t_z_min, t_z_max = t_z_max, t_z_min end

            t_enter = max(t_x_min, t_y_min, t_z_min)
            t_exit  = min(t_x_max, t_y_max, t_z_max)

            # ═══════════════════════════════════════════════════════════
            # ENERGY ACCUMULATORS (K registers — fits without spilling)
            # ═══════════════════════════════════════════════════════════
            accums = ntuple(_ -> zero(T), Val(K))

            # ═══════════════════════════════════════════════════════════
            # DDA TRAVERSAL (skip if ray misses — accums stay zero)
            # ═══════════════════════════════════════════════════════════
            if t_enter < t_exit && t_exit > zero(T)
                t_enter = max(t_enter, zero(T))

                entry_x = src_x + t_enter * ray_x
                entry_y = src_y + t_enter * ray_y
                entry_z = src_z + t_enter * ray_z

                ix = unsafe_trunc(Int32, floor((entry_x - vmx) / vsx))
                iy = unsafe_trunc(Int32, floor((entry_y - vmy) / vsy))
                iz = unsafe_trunc(Int32, floor((entry_z - vmz) / vsz))
                ix = clamp(ix, Int32(0), nx - Int32(1))
                iy = clamp(iy, Int32(0), ny - Int32(1))
                iz = clamp(iz, Int32(0), nz - Int32(1))

                step_x = ray_x >= zero(T) ? Int32(1) : Int32(-1)
                step_y = ray_y >= zero(T) ? Int32(1) : Int32(-1)
                step_z = ray_z >= zero(T) ? Int32(1) : Int32(-1)

                dt_x = abs(vsx / ray_x)
                dt_y = abs(vsy / ray_y)
                dt_z = abs(vsz / ray_z)

                if ray_x >= zero(T)
                    t_next_x = t_enter + (vmx + T(ix + Int32(1)) * vsx - entry_x) / ray_x
                else
                    t_next_x = t_enter + (vmx + T(ix) * vsx - entry_x) / ray_x
                end
                if ray_y >= zero(T)
                    t_next_y = t_enter + (vmy + T(iy + Int32(1)) * vsy - entry_y) / ray_y
                else
                    t_next_y = t_enter + (vmy + T(iy) * vsy - entry_y) / ray_y
                end
                if ray_z >= zero(T)
                    t_next_z = t_enter + (vmz + T(iz + Int32(1)) * vsz - entry_z) / ray_z
                else
                    t_next_z = t_enter + (vmz + T(iz) * vsz - entry_z) / ray_z
                end

                t_current = t_enter
                max_iter = nx + ny + nz + Int32(10)
                iter = Int32(0)

                while t_current < t_exit && iter < max_iter
                    iter += Int32(1)

                    if ix < Int32(0) || ix >= nx || iy < Int32(0) || iy >= ny || iz < Int32(0) || iz >= nz
                        break
                    end

                    t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
                    path_length = (t_next - t_current) * ray_length

                    if path_length > eps
                        mat = Int32(mask[ix + Int32(1), iy + Int32(1), iz + Int32(1)]) + Int32(1)
                        accums = _tiled_accum_energies(accums, μ_tbl, mat, path_length, ts)
                    end

                    # Branchless DDA step
                    mx = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
                    my = Int32(Int32(1) - mx) * Int32(t_next_y <= t_next_z)
                    mz = Int32(1) - mx - my
                    ix += mx * step_x
                    iy += my * step_y
                    iz += mz * step_z
                    t_next_x += T(mx) * dt_x
                    t_next_y += T(my) * dt_y
                    t_next_z += T(mz) * dt_z

                    t_current = t_next
                end
            end

            # ═══════════════════════════════════════════════════════════
            # PARTIAL BEER-LAMBERT SUM — ADD to I_buffer
            # (accums are zero if ray missed → adds wη partial sum correctly)
            # ═══════════════════════════════════════════════════════════
            I_partial = if hbt
                bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
                _tiled_beer_lambert_bt(accums, wη, bt, bt_base, ncnr, ts)
            else
                _tiled_beer_lambert(accums, wη, ts)
            end

            I_buffer[idx] += I_partial
        end
    end

    return I_buffer
end

# =============================================================================
# Single-Kernel Multi-Tile Projection (SPEED-BUILD-V2-002 optimization)
# =============================================================================
#
# All tiles processed in ONE kernel launch. Each thread:
# 1. Sets up DDA once
# 2. Re-traverses voxels for each tile, accumulating K=16 energies per tile
# 3. Writes I_buffer once (not 15 times)
#
# Saves: 14 kernel launch overheads, 14 I_buffer read-modify-writes,
# 14 DDA setup repetitions, Metal command buffer barriers.
# =============================================================================

"""
    siddon_multitile_poly_project!(I_buffer, mask, geom, μ_table_gpu, wη_gpu, Val(K), n_tiles; kwargs...)

Multi-tile polychromatic forward projection in a single kernel launch.
Each thread traces DDA once per tile (re-traversing same voxels), accumulating
K energy bins per tile. All tiles processed per thread, I_buffer written once.

# Arguments
- `I_buffer::AbstractArray{T,3}`: Output [n_cols, n_rows, n_angles]. Written with
  total Beer-Lambert sum (not -log; caller applies -log after).
- `mask::AbstractArray{<:Unsigned,3}`: Material index volume [nx, ny, nz].
- `geom::CTGeometry`: Scanner geometry.
- `μ_table_gpu::AbstractArray{T,2}`: Attenuation lookup table [n_materials, n_energies_padded].
- `wη_gpu::AbstractArray{T,1}`: Pre-computed `weights_norm .* η` [n_energies_padded].
- `::Val{K}`: Tile size (compile-time, typically 16).
- `n_tiles::Int32`: Number of tiles (runtime).
"""
function siddon_multitile_poly_project!(
    I_buffer::AbstractArray{T, 3},
    mask::AbstractArray{<:Unsigned, 3},
    geom::CTGeometry,
    μ_table_gpu::AbstractArray{T, 2},
    wη_gpu::AbstractArray{T, 1},
    ::Val{K},
    n_tiles::Int32;
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    ws_bowtie_spectral = nothing
) where {T <: AbstractFloat, K}

    nx = Int32(size(mask, 1))
    ny = Int32(size(mask, 2))
    nz = Int32(size(mask, 3))
    n_cols = Int32(size(I_buffer, 1))
    n_rows = Int32(size(I_buffer, 2))
    n_angles = Int32(size(I_buffer, 3))

    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vol_min_x = T(-vol_bounds[1] / 2)
    vol_min_y = T(-vol_bounds[2] / 2)
    vol_min_z = T(-vol_bounds[3] / 2)
    vol_max_x = T(vol_bounds[1] / 2)
    vol_max_y = T(vol_bounds[2] / 2)
    vol_max_z = T(vol_bounds[3] / 2)
    voxel_size_x = T(vol_bounds[1]) / T(nx)
    voxel_size_y = T(vol_bounds[2]) / T(ny)
    voxel_size_z = T(vol_bounds[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    source_positions = if ws_source_positions !== nothing
        ws_source_positions
    else
        _sp = similar(I_buffer, T, size(geom.source_positions)...)
        copyto!(_sp, T.(geom.source_positions))
        _sp
    end
    detector_centers = if ws_detector_centers !== nothing
        ws_detector_centers
    else
        _dc = similar(I_buffer, T, size(geom.detector_centers)...)
        copyto!(_dc, T.(geom.detector_centers))
        _dc
    end
    detector_u = if ws_detector_u !== nothing
        ws_detector_u
    else
        _du = similar(I_buffer, T, size(geom.detector_u)...)
        copyto!(_du, T.(geom.detector_u))
        _du
    end
    detector_v = if ws_detector_v !== nothing
        ws_detector_v
    else
        _dv = similar(I_buffer, T, size(geom.detector_v)...)
        copyto!(_dv, T.(geom.detector_v))
        _dv
    end

    nc_nr = n_cols * n_rows
    has_bowtie = ws_bowtie_spectral !== nothing
    tile_K = Int32(K)

    let mask=mask, μ_tbl=μ_table_gpu, wη=wη_gpu,
        sp=source_positions, dc=detector_centers, du=detector_u, dv=detector_v,
        bt=ws_bowtie_spectral, ntiles=n_tiles, tK=tile_K,
        vmx=vol_min_x, vmy=vol_min_y, vmz=vol_min_z,
        vMx=vol_max_x, vMy=vol_max_y, vMz=vol_max_z,
        vsx=voxel_size_x, vsy=voxel_size_y, vsz=voxel_size_z,
        nx=nx, ny=ny, nz=nz, nc=n_cols, nr=n_rows,
        mag=magnification, ps=pixel_size, prs=pixel_row_size,
        cc=col_center, rc=row_center, ncnr=nc_nr, hbt=has_bowtie

        AK.foreachindex(I_buffer) do idx
            # ─── Index decomposition ───
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            # ─── Source position ───
            src_x = sp[1, angle]
            src_y = sp[2, angle]
            src_z = sp[3, angle]

            # ─── Detector pixel position ───
            dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
            dux = du[1, angle]; duy = du[2, angle]; duz = du[3, angle]
            dvx = dv[1, angle]; dvy = dv[2, angle]; dvz = dv[3, angle]

            u_offset = (T(col) - cc) * ps * mag
            v_offset = (T(row) - rc) * prs * mag

            det_x = dcx + u_offset * dux + v_offset * dvx
            det_y = dcy + u_offset * duy + v_offset * dvy
            det_z = dcz + u_offset * duz + v_offset * dvz

            # ═══════════════════════════════════════════════════════════
            # DDA SETUP (once per ray, shared across all tiles)
            # ═══════════════════════════════════════════════════════════
            ray_x = det_x - src_x
            ray_y = det_y - src_y
            ray_z = det_z - src_z
            ray_length = sqrt(ray_x^2 + ray_y^2 + ray_z^2)

            eps = T(1e-10)
            ray_x = abs(ray_x) < eps ? (ray_x >= zero(T) ? eps : -eps) : ray_x
            ray_y = abs(ray_y) < eps ? (ray_y >= zero(T) ? eps : -eps) : ray_y
            ray_z = abs(ray_z) < eps ? (ray_z >= zero(T) ? eps : -eps) : ray_z

            t_x_min = (vmx - src_x) / ray_x
            t_x_max = (vMx - src_x) / ray_x
            t_y_min = (vmy - src_y) / ray_y
            t_y_max = (vMy - src_y) / ray_y
            t_z_min = (vmz - src_z) / ray_z
            t_z_max = (vMz - src_z) / ray_z

            if t_x_min > t_x_max; t_x_min, t_x_max = t_x_max, t_x_min end
            if t_y_min > t_y_max; t_y_min, t_y_max = t_y_max, t_y_min end
            if t_z_min > t_z_max; t_z_min, t_z_max = t_z_max, t_z_min end

            t_enter = max(t_x_min, t_y_min, t_z_min)
            t_exit  = min(t_x_max, t_y_max, t_z_max)

            # Ray misses volume → I_buffer = sum of all wη (exp(0)=1)
            if t_enter >= t_exit || t_exit <= zero(T)
                I_sum = zero(T)
                for tile_idx in Int32(1):ntiles
                    ts = (tile_idx - Int32(1)) * tK + Int32(1)
                    accums_miss = ntuple(_ -> zero(T), Val(K))
                    I_sum += if hbt
                        bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
                        _tiled_beer_lambert_bt(accums_miss, wη, bt, bt_base, ncnr, ts)
                    else
                        _tiled_beer_lambert(accums_miss, wη, ts)
                    end
                end
                I_buffer[idx] = I_sum
                return
            end

            t_enter = max(t_enter, zero(T))

            # Save DDA start state for re-traversal
            entry_x = src_x + t_enter * ray_x
            entry_y = src_y + t_enter * ray_y
            entry_z = src_z + t_enter * ray_z

            ix_start = clamp(unsafe_trunc(Int32, floor((entry_x - vmx) / vsx)), Int32(0), nx - Int32(1))
            iy_start = clamp(unsafe_trunc(Int32, floor((entry_y - vmy) / vsy)), Int32(0), ny - Int32(1))
            iz_start = clamp(unsafe_trunc(Int32, floor((entry_z - vmz) / vsz)), Int32(0), nz - Int32(1))

            step_x = ray_x >= zero(T) ? Int32(1) : Int32(-1)
            step_y = ray_y >= zero(T) ? Int32(1) : Int32(-1)
            step_z = ray_z >= zero(T) ? Int32(1) : Int32(-1)

            dt_x = abs(vsx / ray_x)
            dt_y = abs(vsy / ray_y)
            dt_z = abs(vsz / ray_z)

            t_next_x_start = if ray_x >= zero(T)
                t_enter + (vmx + T(ix_start + Int32(1)) * vsx - entry_x) / ray_x
            else
                t_enter + (vmx + T(ix_start) * vsx - entry_x) / ray_x
            end
            t_next_y_start = if ray_y >= zero(T)
                t_enter + (vmy + T(iy_start + Int32(1)) * vsy - entry_y) / ray_y
            else
                t_enter + (vmy + T(iy_start) * vsy - entry_y) / ray_y
            end
            t_next_z_start = if ray_z >= zero(T)
                t_enter + (vmz + T(iz_start + Int32(1)) * vsz - entry_z) / ray_z
            else
                t_enter + (vmz + T(iz_start) * vsz - entry_z) / ray_z
            end

            max_iter = nx + ny + nz + Int32(10)

            # Bowtie base (computed once, used in all tiles)
            bt_base = if hbt
                Int32(col) + (Int32(row) - Int32(1)) * nc
            else
                Int32(0)
            end

            # ═══════════════════════════════════════════════════════════
            # TILE LOOP — re-traverse DDA for each tile
            # ═══════════════════════════════════════════════════════════
            I_sum = zero(T)

            for tile_idx in Int32(1):ntiles
                ts = (tile_idx - Int32(1)) * tK + Int32(1)
                accums = ntuple(_ -> zero(T), Val(K))

                # Restore DDA state
                ix = ix_start; iy = iy_start; iz = iz_start
                t_current = t_enter
                t_next_x = t_next_x_start
                t_next_y = t_next_y_start
                t_next_z = t_next_z_start
                iter = Int32(0)

                while t_current < t_exit && iter < max_iter
                    iter += Int32(1)

                    if ix < Int32(0) || ix >= nx || iy < Int32(0) || iy >= ny || iz < Int32(0) || iz >= nz
                        break
                    end

                    t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
                    path_length = (t_next - t_current) * ray_length

                    if path_length > eps
                        mat = Int32(mask[ix + Int32(1), iy + Int32(1), iz + Int32(1)]) + Int32(1)
                        accums = _tiled_accum_energies(accums, μ_tbl, mat, path_length, ts)
                    end

                    mx = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
                    my = Int32(Int32(1) - mx) * Int32(t_next_y <= t_next_z)
                    mz = Int32(1) - mx - my
                    ix += mx * step_x
                    iy += my * step_y
                    iz += mz * step_z
                    t_next_x += T(mx) * dt_x
                    t_next_y += T(my) * dt_y
                    t_next_z += T(mz) * dt_z

                    t_current = t_next
                end

                # Partial Beer-Lambert sum for this tile
                I_sum += if hbt
                    _tiled_beer_lambert_bt(accums, wη, bt, bt_base, ncnr, ts)
                else
                    _tiled_beer_lambert(accums, wη, ts)
                end
            end

            I_buffer[idx] = I_sum
        end
    end

    return I_buffer
end

# =============================================================================
# Spectral Multi-Output Tiled Projection (PCCT)
# =============================================================================
#
# Unified spectral forward projection for photon-counting CT. Uses the same
# tiled DDA approach as EICT (K=16 energy accumulators per tile), but writes
# to N_bins output channels instead of 1.
#
# For each ray, the kernel:
# 1. Traverses the material mask via DDA (identical to EICT tiled kernel)
# 2. Accumulates K=16 energy-bin line integrals in NTuple registers
# 3. For each output bin b: computes Σ_e W[e,b] * exp(-accum[e])
# 4. ADDS the partial sum to outputs_flat[idx + (b-1)*n_elements]
#
# The W matrix encodes: I0 * weight * η * R[e,b] for each energy/bin pair,
# pre-computed on CPU and uploaded as [n_energies_padded, n_bins] to GPU.
#
# outputs_flat is a 1D buffer laid out as [sino1..., sino2..., ...sinoN],
# where each sino has n_elements = n_cols * n_rows * n_angles entries.
# =============================================================================

"""
    _tiled_beer_lambert_col(accums, W, b, ts)

@generated helper: partial Beer-Lambert sum for a single output bin `b`,
using K tiled energy accumulators starting at energy index `ts` in W.

Returns Σ_e W[ts+e-1, b] * exp(-accums[e]) for e=1:K.
"""
@generated function _tiled_beer_lambert_col(accums::NTuple{K,T}, W, b::Int32, ts::Int32) where {K, T}
    if K == 0
        return :(zero(T))
    end
    ex = :((@inbounds W[ts, b]) * exp(-accums[1]))
    for i in 2:K
        ex = :($ex + (@inbounds W[ts + Int32($(i - 1)), b]) * exp(-accums[$i]))
    end
    return ex
end

"""
    _tiled_beer_lambert_col_bt(accums, W, b, ts, bt, bt_base, ncnr)

@generated helper: multi-output Beer-Lambert sum with source spectral modulation.
Returns Σ_e W[ts+e-1, b] * bt[bt_base + (ts+e-2)*ncnr] * exp(-accums[e]).
"""
@generated function _tiled_beer_lambert_col_bt(accums::NTuple{K,T}, W, b::Int32, ts::Int32, bt, bt_base::Int32, ncnr::Int32) where {K, T}
    if K == 0
        return :(zero(T))
    end
    ex = :((@inbounds W[ts, b]) * (@inbounds bt[bt_base + (ts - Int32(1)) * ncnr]) * exp(-accums[1]))
    for i in 2:K
        ex = :($ex + (@inbounds W[ts + Int32($(i - 1)), b]) * (@inbounds bt[bt_base + (ts + Int32($(i - 2))) * ncnr]) * exp(-accums[$i]))
    end
    return ex
end

"""
    siddon_fused_spectral_project!(pilot, outputs_flat, n_bins, mask, geom,
        μ_table_gpu, W_gpu, Val(K), tile_start; kwargs...)

Tiled spectral forward projection for PCCT: single DDA pass through material
mask, accumulating K energy bins, then writing partial Beer-Lambert sums to
n_bins output channels.

Each tile processes K energies (starting at `tile_start`). Multiple tiles are
called sequentially. After all tiles, caller applies -log(outputs/I0_bin) per
bin.

# Arguments
- `pilot::AbstractArray{T,3}`: Sinogram-shaped reference array [n_cols, n_rows, n_angles].
  Used only for `AK.foreachindex` iteration count and backend detection. Not written.
- `outputs_flat::AbstractArray{T,1}`: Flattened output [n_elements * n_bins].
  Layout: [sino_bin1..., sino_bin2..., ...]. Partial Beer-Lambert sums are
  ADDED to existing values (must be zero-initialized before first tile).
- `n_bins::Int32`: Number of output bins (e.g., 4 for NAEOTOM Alpha).
- `mask::AbstractArray{<:Unsigned,3}`: Material index volume [nx, ny, nz].
- `geom::CTGeometry`: Scanner geometry (provides detector layout + magnification).
- `μ_table_gpu::AbstractArray{T,2}`: Attenuation lookup [n_materials, n_energies_padded].
- `W_gpu::AbstractArray{T,2}`: Weight matrix [n_energies_padded, n_bins].
  W[e,b] = I0 * w[e] * η[e] * R[e,b] for energy e, bin b.
- `::Val{K}`: Tile size (compile-time, typically 16).
- `tile_start::Int32`: 1-based start index into energy dimension for this tile.

# Keyword Arguments
- `volume_extent`: Override volume physical bounds (phantom extent).
- `ws_source_positions`, `ws_detector_centers`, `ws_detector_u`, `ws_detector_v`:
  Pre-allocated geometry arrays on GPU.
"""
function siddon_fused_spectral_project!(
    pilot::AbstractArray{T, 3},
    outputs_flat::AbstractArray{T, 1},
    n_bins::Int32,
    mask::AbstractArray{<:Unsigned, 3},
    geom::CTGeometry,
    μ_table_gpu::AbstractArray{T, 2},
    W_gpu::AbstractArray{T, 2},
    ::Val{K},
    tile_start::Int32;
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    ws_bowtie_spectral = nothing
) where {T <: AbstractFloat, K}

    # Volume dimensions (Int32 for GPU)
    nx = Int32(size(mask, 1))
    ny = Int32(size(mask, 2))
    nz = Int32(size(mask, 3))
    n_cols = Int32(size(pilot, 1))
    n_rows = Int32(size(pilot, 2))
    n_angles = Int32(size(pilot, 3))
    n_elem = n_cols * n_rows * n_angles

    # Volume bounds
    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vol_min_x = T(-vol_bounds[1] / 2)
    vol_min_y = T(-vol_bounds[2] / 2)
    vol_min_z = T(-vol_bounds[3] / 2)
    vol_max_x = T(vol_bounds[1] / 2)
    vol_max_y = T(vol_bounds[2] / 2)
    vol_max_z = T(vol_bounds[3] / 2)
    voxel_size_x = T(vol_bounds[1]) / T(nx)
    voxel_size_y = T(vol_bounds[2]) / T(ny)
    voxel_size_z = T(vol_bounds[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Geometry arrays on GPU (use workspace or allocate)
    source_positions = if ws_source_positions !== nothing
        ws_source_positions
    else
        _sp = similar(mask, T, size(geom.source_positions)...)
        copyto!(_sp, T.(geom.source_positions))
        _sp
    end
    detector_centers = if ws_detector_centers !== nothing
        ws_detector_centers
    else
        _dc = similar(mask, T, size(geom.detector_centers)...)
        copyto!(_dc, T.(geom.detector_centers))
        _dc
    end
    detector_u = if ws_detector_u !== nothing
        ws_detector_u
    else
        _du = similar(mask, T, size(geom.detector_u)...)
        copyto!(_du, T.(geom.detector_u))
        _du
    end
    detector_v = if ws_detector_v !== nothing
        ws_detector_v
    else
        _dv = similar(mask, T, size(geom.detector_v)...)
        copyto!(_dv, T.(geom.detector_v))
        _dv
    end

    # Bowtie/heel spectral layout constants
    nc_nr = n_cols * n_rows
    has_source_spectral = ws_bowtie_spectral !== nothing

    # Capture all variables in let block for GPU closure correctness
    let mask=mask, μ_tbl=μ_table_gpu, W=W_gpu, ts=tile_start,
        sp=source_positions, dc=detector_centers, du=detector_u, dv=detector_v,
        vmx=vol_min_x, vmy=vol_min_y, vmz=vol_min_z,
        vMx=vol_max_x, vMy=vol_max_y, vMz=vol_max_z,
        vsx=voxel_size_x, vsy=voxel_size_y, vsz=voxel_size_z,
        nx=nx, ny=ny, nz=nz, nc=n_cols, nr=n_rows, na=n_angles,
        mag=magnification, ps=pixel_size, prs=pixel_row_size,
        cc=col_center, rc=row_center,
        ne=n_elem, nb=n_bins, oflat=outputs_flat,
        bt=ws_bowtie_spectral, ncnr=nc_nr, hbt=has_source_spectral

        AK.foreachindex(pilot) do idx
            # ─── Index decomposition (mod/div, no CartesianIndices) ───
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            # ─── Source position ───
            src_x = sp[1, angle]
            src_y = sp[2, angle]
            src_z = sp[3, angle]

            # ─── Detector pixel position ───
            dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
            dux = du[1, angle]; duy = du[2, angle]; duz = du[3, angle]
            dvx = dv[1, angle]; dvy = dv[2, angle]; dvz = dv[3, angle]

            u_offset = (T(col) - cc) * ps * mag
            v_offset = (T(row) - rc) * prs * mag

            det_x = dcx + u_offset * dux + v_offset * dvx
            det_y = dcy + u_offset * duy + v_offset * dvy
            det_z = dcz + u_offset * duz + v_offset * dvz

            # ═══════════════════════════════════════════════════════════
            # DDA SETUP (identical to tiled EICT kernel)
            # ═══════════════════════════════════════════════════════════
            ray_x = det_x - src_x
            ray_y = det_y - src_y
            ray_z = det_z - src_z
            ray_length = sqrt(ray_x^2 + ray_y^2 + ray_z^2)

            eps = T(1e-10)
            ray_x = abs(ray_x) < eps ? (ray_x >= zero(T) ? eps : -eps) : ray_x
            ray_y = abs(ray_y) < eps ? (ray_y >= zero(T) ? eps : -eps) : ray_y
            ray_z = abs(ray_z) < eps ? (ray_z >= zero(T) ? eps : -eps) : ray_z

            t_x_min = (vmx - src_x) / ray_x
            t_x_max = (vMx - src_x) / ray_x
            t_y_min = (vmy - src_y) / ray_y
            t_y_max = (vMy - src_y) / ray_y
            t_z_min = (vmz - src_z) / ray_z
            t_z_max = (vMz - src_z) / ray_z

            if t_x_min > t_x_max; t_x_min, t_x_max = t_x_max, t_x_min end
            if t_y_min > t_y_max; t_y_min, t_y_max = t_y_max, t_y_min end
            if t_z_min > t_z_max; t_z_min, t_z_max = t_z_max, t_z_min end

            t_enter = max(t_x_min, t_y_min, t_z_min)
            t_exit  = min(t_x_max, t_y_max, t_z_max)

            # ═══════════════════════════════════════════════════════════
            # ENERGY ACCUMULATORS (K registers — fits without spilling)
            # ═══════════════════════════════════════════════════════════
            accums = ntuple(_ -> zero(T), Val(K))

            # ═══════════════════════════════════════════════════════════
            # DDA TRAVERSAL (skip if ray misses — accums stay zero)
            # ═══════════════════════════════════════════════════════════
            if t_enter < t_exit && t_exit > zero(T)
                t_enter = max(t_enter, zero(T))

                entry_x = src_x + t_enter * ray_x
                entry_y = src_y + t_enter * ray_y
                entry_z = src_z + t_enter * ray_z

                ix = unsafe_trunc(Int32, floor((entry_x - vmx) / vsx))
                iy = unsafe_trunc(Int32, floor((entry_y - vmy) / vsy))
                iz = unsafe_trunc(Int32, floor((entry_z - vmz) / vsz))
                ix = clamp(ix, Int32(0), nx - Int32(1))
                iy = clamp(iy, Int32(0), ny - Int32(1))
                iz = clamp(iz, Int32(0), nz - Int32(1))

                step_x = ray_x >= zero(T) ? Int32(1) : Int32(-1)
                step_y = ray_y >= zero(T) ? Int32(1) : Int32(-1)
                step_z = ray_z >= zero(T) ? Int32(1) : Int32(-1)

                dt_x = abs(vsx / ray_x)
                dt_y = abs(vsy / ray_y)
                dt_z = abs(vsz / ray_z)

                if ray_x >= zero(T)
                    t_next_x = t_enter + (vmx + T(ix + Int32(1)) * vsx - entry_x) / ray_x
                else
                    t_next_x = t_enter + (vmx + T(ix) * vsx - entry_x) / ray_x
                end
                if ray_y >= zero(T)
                    t_next_y = t_enter + (vmy + T(iy + Int32(1)) * vsy - entry_y) / ray_y
                else
                    t_next_y = t_enter + (vmy + T(iy) * vsy - entry_y) / ray_y
                end
                if ray_z >= zero(T)
                    t_next_z = t_enter + (vmz + T(iz + Int32(1)) * vsz - entry_z) / ray_z
                else
                    t_next_z = t_enter + (vmz + T(iz) * vsz - entry_z) / ray_z
                end

                t_current = t_enter
                max_iter = nx + ny + nz + Int32(10)
                iter = Int32(0)

                while t_current < t_exit && iter < max_iter
                    iter += Int32(1)

                    if ix < Int32(0) || ix >= nx || iy < Int32(0) || iy >= ny || iz < Int32(0) || iz >= nz
                        break
                    end

                    t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
                    path_length = (t_next - t_current) * ray_length

                    if path_length > eps
                        mat = Int32(mask[ix + Int32(1), iy + Int32(1), iz + Int32(1)]) + Int32(1)
                        accums = _tiled_accum_energies(accums, μ_tbl, mat, path_length, ts)
                    end

                    # Branchless DDA step
                    mx = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
                    my = Int32(Int32(1) - mx) * Int32(t_next_y <= t_next_z)
                    mz = Int32(1) - mx - my
                    ix += mx * step_x
                    iy += my * step_y
                    iz += mz * step_z
                    t_next_x += T(mx) * dt_x
                    t_next_y += T(my) * dt_y
                    t_next_z += T(mz) * dt_z

                    t_current = t_next
                end
            end

            # ═══════════════════════════════════════════════════════════
            # MULTI-OUTPUT BEER-LAMBERT — ADD partial sum to each bin
            # With optional source spectral modulation (heel × bowtie)
            # ═══════════════════════════════════════════════════════════
            if hbt
                bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
                for b in Int32(1):nb
                    I_partial = _tiled_beer_lambert_col_bt(accums, W, b, ts, bt, bt_base, ncnr)
                    oflat[idx + (b - Int32(1)) * ne] += I_partial
                end
            else
                for b in Int32(1):nb
                    I_partial = _tiled_beer_lambert_col(accums, W, b, ts)
                    oflat[idx + (b - Int32(1)) * ne] += I_partial
                end
            end
        end
    end

    return outputs_flat
end
