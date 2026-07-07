# =============================================================================
# FDK Reconstruction (Feldkamp-Davis-Kress Algorithm)
# =============================================================================
#
# Complete GPU-native cone-beam CT reconstruction using the Feldkamp-Davis-Kress
# (FDK) algorithm with backend-agnostic acceleration via AcceleratedKernels.jl.
#
# Algorithm Overview
# ------------------
# The FDK algorithm (Feldkamp et al., 1984) extends the 2D fan-beam filtered
# backprojection (FBP) algorithm to 3D cone-beam geometry. It is the standard
# algorithm for circular-orbit cone-beam CT reconstruction and provides exact
# reconstruction for the mid-plane while introducing cone-beam artifacts at
# larger cone angles (>5-10°).
#
# Mathematical Foundation
# -----------------------
# For a cone-beam projection p(θ, u, v) where:
#   - θ: projection angle (gantry rotation)
#   - u: detector column position (fan direction)
#   - v: detector row position (cone direction)
#
# The FDK reconstruction proceeds in three steps:
#
# 1. COSINE WEIGHTING (pre-filtering)
#    Pre-weight projections to account for detector geometry:
#
#        p̃(θ, u, v) = p(θ, u, v) × D / √(D² + u² + v²)
#
#    where D is the source-to-detector distance (SDD). This weight equals
#    cos(γ) where γ is the angle between the ray and the central ray.
#
# 2. RAMP FILTERING (1D convolution per row)
#    Apply the ramp (Ram-Lak) filter along each detector row:
#
#        p̂(θ, u, v) = p̃(θ, u, v) * h(u)
#
#    The ramp filter h(u) in spatial domain (Kak & Slaney):
#        h[0] = 1/(4Δ²)
#        h[n] = 0                    for even n ≠ 0
#        h[n] = -1/(π²n²Δ²)          for odd n
#
#    Optional windowing functions (Shepp-Logan, Cosine, Hamming, Hann)
#    reduce noise amplification at the cost of spatial resolution.
#
# 3. WEIGHTED BACKPROJECTION (voxel-driven)
#    For each voxel (x, y, z), accumulate weighted contributions:
#
#        f(x, y, z) = (1/2) ∫₀²π [D² / (D - s)²] × p̂(θ, u*, v*) dθ
#
#    where (u*, v*) is the detector coordinates of the ray through (x, y, z),
#    s = x·sin(θ) - y·cos(θ) is the signed distance to the source plane,
#    and D²/(D-s)² is the FDK distance weight.
#
# Physical Interpretation
# -----------------------
# - Cosine weighting: Corrects for oblique ray path lengths on detector
# - Ramp filter: High-pass filter that deconvolves blur from projection
# - Distance weighting: Accounts for 1/r² intensity falloff in cone-beam
#
# Cone-Beam Artifacts
# -------------------
# FDK is an approximate algorithm for circular orbits. Artifacts include:
# - Cone-beam artifacts: Increasing distortion away from mid-plane
# - Intensity drop-off: Objects near edges of z-FOV appear darker
# - Rule of thumb: Cone angle < 10° for clinically acceptable images
#
# For large cone angles, consider iterative reconstruction (SIRT, CGLS)
# or exact algorithms (Katsevich) for helical acquisition.
#
# Physical Units
# --------------
# - Source positions, detector positions: mm
# - Field of view (FOV): mm (as (fx, fy, fz) tuple)
# - Pixel size: mm (at isocenter)
# - Voxel size: mm (derived from FOV / volume_size)
# - Input sinogram: dimensionless line integrals (μ × path length)
# - Output volume: linear attenuation coefficients μ (mm⁻¹)
#
# To convert to Hounsfield Units (HU):
#   HU = 1000 × (μ - μ_water) / μ_water
#
# Implementation Notes
# --------------------
# - Port of TIGRE's voxel-driven FDK with modifications for Julia/AK
# - Spatial domain filtering (no FFT) for full GPU native execution
# - Int32 indexing throughout for GPU compatibility (Metal limitation)
# - Bilinear interpolation in backprojection for sub-pixel accuracy
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
# 1. Feldkamp LA, Davis LC, Kress JW. "Practical cone-beam algorithm."
#    J Opt Soc Am A. 1984;1(6):612-619. doi:10.1364/JOSAA.1.000612
#    [Original FDK algorithm paper]
#
# 2. Kak AC, Slaney M. "Principles of Computerized Tomographic Imaging."
#    IEEE Press, 1988. Chapter 3: Algorithms for Reconstruction with
#    Nondiffracting Sources. Available: http://www.slaney.org/pct/
#    [Comprehensive treatment of CT reconstruction including FDK]
#
# 3. Biguri A, Dosanjh M, Hancock S, Soleimani M. "TIGRE: A MATLAB-GPU
#    toolbox for CBCT image reconstruction." Biomed Phys Eng Express.
#    2016;2(5):055010. doi:10.1088/2057-1976/2/5/055010
#    [TIGRE implementation that this code is ported from]
#
# 4. Turbell H. "Cone-Beam Reconstruction Using Filtered Backprojection."
#    PhD Thesis, Linköping University, 2001. ISBN 91-7219-919-9.
#    [Detailed analysis of FDK artifacts and exact algorithms]
#
# =============================================================================

import AcceleratedKernels as AK

export fdk_reconstruct, apply_fov_mask!

"""
    fdk_reconstruct(sinogram, geom, volume_size; filter=StandardFilter(), cutoff=1.0)

Perform FDK (Feldkamp-Davis-Kress) cone-beam CT reconstruction from a sinogram.

This is the primary reconstruction function for circular-orbit cone-beam CT.
It implements the complete FDK algorithm: cosine weighting, ramp filtering,
and weighted backprojection, all executed on GPU via AcceleratedKernels.jl.

# Algorithm

The FDK algorithm reconstructs a 3D volume f(x,y,z) from cone-beam projections
p(θ,u,v) in three steps (Feldkamp et al., 1984):

**Step 1: Cosine Weighting**

Pre-weight projections to correct for oblique ray geometry:

    p̃(θ, u, v) = p(θ, u, v) × D / √(D² + u² + v²)

where D is the source-to-detector distance. This weight equals cos(γ) where
γ is the angle between the ray and the central ray.

**Step 2: Ramp Filtering**

Apply 1D convolution with ramp (Ram-Lak) filter along each detector row:

    p̂(θ, u, v) = p̃(θ, u, v) * h(u)

The discrete ramp filter kernel h[n] (Kak & Slaney, 1988):
- h[0] = 1/(4Δ²)
- h[n] = 0 for even n ≠ 0
- h[n] = -1/(π²n²Δ²) for odd n

where Δ is the detector pixel spacing.

**Step 3: Weighted Backprojection**

For each voxel (x, y, z), integrate weighted filtered projections:

    f(x, y, z) = (π/N) Σθ [D² / (D - s)²] × p̂(θ, u*, v*)

where:
- (u*, v*) are detector coordinates of the ray through voxel (x, y, z)
- s = x·sin(θ) - y·cos(θ) is signed distance to source plane
- D²/(D-s)² is the FDK distance weight
- N is the number of projection angles

# Arguments

- `sinogram::AbstractArray{T,3}`: Input sinogram of size `[n_cols, n_rows, n_angles]`.
  Contains line integrals (dimensionless, typically from forward projection).
  - `n_cols`: Number of detector columns (transaxial/fan direction)
  - `n_rows`: Number of detector rows (axial/cone direction)
  - `n_angles`: Number of projection angles

- `geom::CTGeometry`: CT scanner geometry containing:
  - `SAD::T`: Source-to-axis (isocenter) distance in mm
  - `SDD::T`: Source-to-detector distance in mm
  - `fov::Tuple{T,T,T}`: Field of view (x, y, z) in mm
  - `pixel_size::T`: Detector pixel size at isocenter in mm
  - `angles::Vector{T}`: Projection angles in radians
  - `source_positions::Matrix{T}`: Source positions [3, n_angles] in mm
  - `detector_centers::Matrix{T}`: Detector centers [3, n_angles] in mm
  - `detector_u::Matrix{T}`: Detector u-direction vectors [3, n_angles]
  - `detector_v::Matrix{T}`: Detector v-direction vectors [3, n_angles]

- `volume_size::NTuple{3,Int}`: Output volume dimensions (nx, ny, nz).
  Voxel size is computed as FOV / volume_size.

# Keyword Arguments

- `filter::FilterType = StandardFilter()`: Reconstruction filter type. Options:
  - `StandardFilter()`: CatSim-matched apodized ramp (clinical default, balanced noise/resolution)
  - `SoftFilter()`: Stronger smoothing than Standard (lower noise, softer edges)
  - `BoneFilter()`: Boosted mid-frequencies (sharper bone edges, higher noise)
  - `RampFilter()`: Pure Ram-Lak (sharpest, highest noise)
  - `SheppLoganFilter()`: Ramp × sinc (moderate smoothing)
  - `CosineFilter()`: Ramp × cos² (smooth)
  - `HammingFilter()`: Ramp × Hamming window (low noise)
  - `HannFilter()`: Ramp × Hann window (lowest noise, softest)

- `cutoff::Float64 = 1.0`: Frequency cutoff as fraction of Nyquist frequency.
  Range: 0.0 to 1.0. Lower values reduce noise but also spatial resolution.
  Typical values: 0.8-1.0 for diagnostic imaging, 0.5-0.7 for dose reduction.

# Returns

- `volume::AbstractArray{T,3}`: Reconstructed volume of size `[nx, ny, nz]`.
  Contains linear attenuation coefficients μ (mm⁻¹). Convert to HU:
  `HU = 1000 × (μ - μ_water) / μ_water`

  The volume is allocated on the same device as the input sinogram
  (CPU Array, Metal MtlArray, CUDA CuArray, etc.).

# GPU Compatibility

Automatically selects compute backend based on array type:

| Array Type | Backend | Typical Speedup |
|------------|---------|-----------------|
| `Array`    | CPU (multi-threaded) | 1× (baseline) |
| `MtlArray` | Metal (Apple Silicon) | 20-50× |
| `CuArray`  | CUDA (NVIDIA) | 50-100× |
| `ROCArray` | ROCm (AMD) | 50-100× |

No code changes needed - just pass appropriate array types.

# Example

```julia
using BasisSimulator
using Metal  # or: using CUDA

# Create scanner and geometry
scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=360, fov=(350.0, 350.0, 160.0))

# Create a water cylinder phantom
phantom = zeros(Float32, 256, 256, 64)
μ_water = 0.02f0  # mm⁻¹ at ~60 keV
for i in 1:256, j in 1:256, k in 1:64
    x, y = (i - 128.5) * 1.37, (j - 128.5) * 1.37  # mm from center
    if x^2 + y^2 < 100^2  # 100mm radius cylinder
        phantom[i, j, k] = μ_water
    end
end

# Forward projection on GPU
phantom_gpu = MtlArray(phantom)
sinogram_gpu = siddon_forward_project(phantom_gpu, geom)

# FDK reconstruction
recon_gpu = fdk_reconstruct(sinogram_gpu, geom, (256, 256, 64))

# Convert to Hounsfield Units
recon_cpu = Array(recon_gpu)
recon_hu = 1000f0 .* (recon_cpu .- μ_water) ./ μ_water

# Water region should have HU ≈ 0
water_mask = phantom .> 0
println("Mean HU in water: ", mean(recon_hu[water_mask]))  # Should be ≈ 0 ± 20
```

# Filter Selection Guide

| Filter | Noise | Resolution | Use Case |
|--------|-------|------------|----------|
| RampFilter | Highest | Best | High-contrast imaging |
| SheppLoganFilter | High | Good | General purpose |
| CosineFilter | Medium | Medium | Soft tissue |
| HammingFilter | Low | Reduced | Low-dose protocols |
| HannFilter | Lowest | Most reduced | Pediatric/dose-critical |

# Cone-Beam Artifacts

FDK is an approximate algorithm for circular orbits. Be aware of:

- **Cone-beam artifacts**: Shading/streaks increase with cone angle
- **Intensity drop-off**: Objects near z-FOV edges appear darker
- **Practical limit**: Cone angle < 10° for diagnostic quality

For wider cone angles or higher accuracy, consider:
- `sirt_reconstruct` / `cgls_reconstruct`: Iterative methods (slower, fewer artifacts)
- Helical acquisition with Katsevich algorithm (exact reconstruction)

# Performance Notes

- Filtering: O(n_cols × n_rows × n_angles × kernel_size)
- Backprojection: O(nx × ny × nz × n_angles)
- Memory: Allocates filtered sinogram + output volume
- Typical 256³ reconstruction: ~1-2 seconds on GPU, ~30-60 seconds on CPU

For repeated reconstructions with same geometry, consider pre-computing
the filter kernel with lower-level functions in Filtering.jl.

# References

1. Feldkamp LA, Davis LC, Kress JW. "Practical cone-beam algorithm."
   J Opt Soc Am A. 1984;1(6):612-619. doi:10.1364/JOSAA.1.000612

2. Kak AC, Slaney M. "Principles of Computerized Tomographic Imaging."
   IEEE Press, 1988. Chapter 3. Available: http://www.slaney.org/pct/

3. Biguri A, et al. "TIGRE: A MATLAB-GPU toolbox for CBCT image
   reconstruction." Biomed Phys Eng Express. 2016;2(5):055010.
   doi:10.1088/2057-1976/2/5/055010

4. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and
   Recent Advances." 3rd ed. SPIE Press, 2015. Chapter 3.
   ISBN: 978-1628418255

# See Also

- [`filter_sinogram`](@ref): Apply FDK filtering separately
- [`backproject`](@ref): Apply backprojection separately
- [`sirt_reconstruct`](@ref): Iterative SIRT reconstruction
- [`cgls_reconstruct`](@ref): Iterative CGLS reconstruction
- [`siddon_forward_project`](@ref): Forward projection (inverse operation)
"""
function fdk_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    filter::FilterType = StandardFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    # Helical trajectories route to the rebinned WFBP chain (Stierstorfer
    # 2004 family) — the circular FDK weighting below assumes a full-orbit
    # circular scan.
    if is_helical(geom)
        return wfbp_helical_reconstruct(sinogram, geom, volume_size;
            filter = filter, cutoff = cutoff)
    end

    # Step 1: Filter sinogram (includes cosine weighting)
    # GPU-native spatial domain filtering - no CPU transfer needed
    filtered = filter_sinogram(sinogram, geom; filter=filter, cutoff=cutoff)

    # Step 2: Backproject
    volume = backproject(filtered, geom, volume_size)

    # Step 3: Mask outside FOV (clinical convention)
    apply_fov_mask!(volume, geom)

    return volume
end

"""
    fdk_reconstruct(sinogram, geom, volume_size, fov; filter=StandardFilter(), cutoff=1.0)

FDK reconstruction with explicit field-of-view (FOV) specification.

This variant allows reconstructing to a different FOV than specified in the
scanner geometry. Useful for zoomed reconstructions, targeted ROI imaging,
or when the reconstruction grid should cover a different region than the
original acquisition geometry.

# Arguments

- `sinogram::AbstractArray{T,3}`: Input sinogram of size `[n_cols, n_rows, n_angles]`

- `geom::CTGeometry`: CT scanner geometry (see [`fdk_reconstruct`](@ref))

- `volume_size::NTuple{3,Int}`: Output volume dimensions (nx, ny, nz)

- `fov::NTuple{3,Float64}`: Reconstruction field of view in mm as (fx, fy, fz).
  The reconstructed volume will cover:
  - x: [-fx/2, +fx/2] mm
  - y: [-fy/2, +fy/2] mm
  - z: [-fz/2, +fz/2] mm

  Voxel size = FOV / volume_size.

# Keyword Arguments

- `filter::FilterType = StandardFilter()`: Reconstruction filter type
- `cutoff::Float64 = 1.0`: Frequency cutoff (0-1)

# Returns

- `volume::AbstractArray{T,3}`: Reconstructed volume of size `[nx, ny, nz]`

# Example

```julia
using BasisSimulator

# Standard reconstruction at full FOV
recon_full = fdk_reconstruct(sinogram, geom, (256, 256, 64))

# Zoomed reconstruction: smaller FOV, same voxel count = higher resolution
# This focuses on the central 150mm × 150mm × 80mm region
recon_zoom = fdk_reconstruct(sinogram, geom, (256, 256, 64), (150.0, 150.0, 80.0))

# Reduced FOV for faster reconstruction (coarser voxels)
recon_quick = fdk_reconstruct(sinogram, geom, (128, 128, 32), (350.0, 350.0, 160.0))
```

# Use Cases

- **Zoomed reconstruction**: Smaller FOV + same voxels = finer detail in ROI
- **Extended FOV**: Larger FOV for peripheral structures
- **Asymmetric FOV**: Different z-coverage than transaxial
- **Dose studies**: Match FOV exactly to dose measurement region

# Notes

- The FOV is centered at the isocenter (0, 0, 0)
- Voxels outside the original acquisition cone will have artifacts
- Zooming beyond the acquired data doesn't improve true resolution

# See Also

- [`fdk_reconstruct`](@ref): Standard reconstruction using geometry FOV
"""
function fdk_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int},
    fov::NTuple{3, Float64};
    filter::FilterType = StandardFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    # Create a copy of geometry with modified FOV
    geom_fov = CTGeometry(
        geom.SAD, geom.SDD,
        geom.n_angles, geom.n_rows, geom.n_cols, geom.pixel_size, geom.pixel_row_size,
        geom.angles, geom.source_positions, geom.detector_centers,
        geom.detector_u, geom.detector_v,
        fov,  # Use specified FOV
        geom.pitch, geom.table_feed, geom.detector_shape
    )

    return fdk_reconstruct(sinogram, geom_fov, volume_size; filter=filter, cutoff=cutoff)
end

"""
    apply_fov_mask!(volume, geom; sentinel_μ=-0.04)

Mask voxels outside the circular reconstruction FOV to a sentinel value.

Clinical CT scanners set voxels outside the reconstruction circle to a fixed
value (typically -1024 or -2048 HU equivalent) to suppress FDK edge artifacts.
The default sentinel_μ = -0.04 cm⁻¹ gives approximately -2048 HU for typical
μ_water ≈ 0.02 cm⁻¹.

# Arguments
- `volume`: 3D reconstruction volume (nx, ny, nz), modified in-place
- `geom`: CT geometry (provides FOV for circle radius)

# Keyword Arguments
- `sentinel_μ`: Raw μ value to assign outside FOV (default: -0.04 cm⁻¹)
"""
function apply_fov_mask!(volume::AbstractArray{T,3}, geom::CTGeometry;
                         sentinel_μ::Real = T(-0.04)) where T
    nx, ny, nz = size(volume)
    fov_x, fov_y = geom.fov[1], geom.fov[2]

    # FOV radius (use the smaller of x/y for a circular mask)
    radius = T(min(fov_x, fov_y) / 2)
    radius_sq = radius * radius

    voxel_x = T(fov_x / nx)
    voxel_y = T(fov_y / ny)
    sentinel = T(sentinel_μ)

    AK.foreachindex(volume) do idx
        ci = CartesianIndices(volume)[idx]
        ix, iy, iz = Tuple(ci)

        # Voxel center in physical coordinates (centered at origin)
        x = (T(ix) - T(0.5) - T(nx) / T(2)) * voxel_x
        y = (T(iy) - T(0.5) - T(ny) / T(2)) * voxel_y

        if x * x + y * y > radius_sq
            volume[idx] = sentinel
        end
    end

    return volume
end
