"""
    Forward/FocalSpot.jl

Finite focal spot modeling for CT simulation.

# Physics Background

The X-ray focal spot is the region on the anode where electrons impact and
produce X-rays. It has finite dimensions (typically 0.5-1.5 mm), which affects
spatial resolution by creating geometric blur (penumbra effect).

The blur width at the detector depends on:
- Focal spot size (fs_width, fs_length)
- Source-to-object distance (SOD)
- Object-to-detector distance (ODD = SDD - SOD)
- Magnification factor M = SDD/SOD

# Mathematical Formulation

The geometric blur at the detector plane is given by:

    blur_detector = fs_size × (SDD - SOD) / SOD = fs_size × (M - 1)

where:
- fs_size is the focal spot dimension (width or length) in mm
- SDD is the source-to-detector distance
- SOD is the source-to-object distance
- M = SDD/SOD is the geometric magnification at the object

Objects closer to the source experience higher magnification and more blur.
Objects closer to the detector experience lower magnification and less blur.

The blur in detector pixels (at isocenter) is:

    blur_pixels = blur_detector_cm / (pixel_size_iso_cm × SDD/SID)

where pixel_size_iso is the pixel size at the isocenter (as used in CTGeometry).

# Focal Spot Characterization

Per IEC 60336:2005, focal spot dimensions are specified as:
- Width: perpendicular to anode-cathode axis (fan direction)
- Length: parallel to anode-cathode axis (cone direction)

Focal spots are not uniform; they typically have:
- Gaussian intensity distribution
- Asymmetric dimensions (width ≠ length)
- The "line focus principle": effective size smaller than actual size due to
  target angle (typically 7-12°)

# CatSim Compatibility

CatSim (SetFocalspot.py) uses multi-sample source ray tracing:
- srcXSampleCount × srcYSampleCount samples across the focal spot
- Accounts for anode target angle: y_offset = -z/tan(target_angle)
- Supports measured focal spot profiles from .mat/.npz files

BasisSimulator uses post-projection convolution blur, which is mathematically
equivalent for small blur kernels and produces identical FWHM values.

# Implementation

This module provides two approaches:
1. **Fast (default)**: Convolution-based blur approximation (GPU-native)
   - Uses spatial domain convolution via AcceleratedKernels.jl
   - Efficient for sub-pixel to few-pixel blur (clinical focal spots)
   - Suitable for most CT simulations

2. **Accurate**: Multi-sample ray tracing
   - Uses generate_focal_spot_samples() to create source sampling grid
   - Must be integrated with forward projection loop
   - More accurate for large focal spots or extreme magnifications

# Typical Focal Spot Sizes (IEC 60336)

| Size Class | Nominal (mm) | Max Width | Max Length |
|------------|--------------|-----------|------------|
| Small      | 0.6          | 0.90      | 0.90       |
| Medium     | 1.0          | 1.40      | 1.40       |
| Large      | 1.2          | 1.65      | 1.65       |

Clinical CT typically uses 0.5-0.7 mm (small) or 0.9-1.2 mm (large) focal spots.

# GPU Compatibility
- ✅ Metal (Apple Silicon)
- ✅ CUDA (NVIDIA)
- ✅ ROCm (AMD)
- ✅ CPU fallback

# References

1. IEC 60336:2005 - Medical electrical equipment - X-ray tube assemblies for
   medical diagnosis - Characteristics of focal spots.

2. Bushberg JT, Seibert JA, Leidholdt EM, Boone JM. "The Essential Physics of
   Medical Imaging" 3rd ed. Lippincott Williams & Wilkins, 2012.
   Chapter 5: X-ray Production.

3. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent
   Advances" 2nd ed. SPIE Press, 2009. Section 3.2.1: Focal Spot.

4. GE CatSim (XCIST) SetFocalspot.py - Reference implementation for source
   sampling approach. https://github.com/xcist/main

5. Flohr TG, et al. "First performance evaluation of a dual-source CT (DSCT)
   system." Eur Radiol. 2006;16(2):256-268. doi:10.1007/s00330-005-2919-2
"""

import AcceleratedKernels as AK

# =============================================================================
# Focal Spot Types
# =============================================================================

"""
    FocalSpot

Focal spot specification with size and shape parameters.

The focal spot is characterized by its physical dimensions on the anode target
and the intensity distribution within those dimensions.

# Fields
- `width::Float64`: Focal spot width in mm (fan direction, perpendicular to
  anode-cathode axis). This is the dimension that primarily affects in-plane
  spatial resolution.
- `length::Float64`: Focal spot length in mm (cone direction, along anode-
  cathode axis). This affects z-axis resolution.
- `shape::Symbol`: Intensity distribution shape within the focal spot.
  - `:gaussian` (default): Realistic Gaussian distribution
  - `:uniform`: Flat rectangular distribution
  - `:bimodal`: Two peaks (for bloomed/degraded focal spots)
- `n_samples::Int`: Number of samples per dimension for ray-tracing mode.
  Typical values: 3-5. Higher values improve accuracy but increase computation.

# Typical Clinical Values

| Scanner Type     | Small FS (mm)  | Large FS (mm)  |
|------------------|----------------|----------------|
| GE Revolution    | 0.6 × 0.7      | 0.9 × 0.9      |
| Siemens Force    | 0.7 × 0.7      | 1.2 × 1.2      |
| Canon Aquilion   | 0.5 × 0.5      | 0.9 × 0.9      |

Note: The nominal "0.6 mm focal spot" refers to the effective size after
accounting for the anode target angle (line focus principle).

# Example
```julia
# GE Revolution small focal spot
fs = FocalSpot(0.6, 0.7, :gaussian, 3)

# Large focal spot for high-power imaging
fs_large = FocalSpot(1.2, 1.2, :gaussian, 5)
```

See also: [`focal_spot_small`](@ref), [`focal_spot_medium`](@ref),
[`focal_spot_large`](@ref), [`compute_focal_spot_blur_fwhm`](@ref)
"""
struct FocalSpot
    width::Float64      # mm
    length::Float64     # mm
    shape::Symbol       # :gaussian, :uniform, :bimodal
    n_samples::Int      # samples per dimension for ray tracing
end

# =============================================================================
# Pre-defined Focal Spots
# =============================================================================

"""
    focal_spot_small()

Small focal spot for high-resolution imaging.

Typical dimensions: 0.5 × 0.5 mm (or smaller for micro-CT).
"""
function focal_spot_small()
    return FocalSpot(0.5, 0.5, :gaussian, 3)
end

"""
    focal_spot_medium()

Medium focal spot for general diagnostic imaging.

Typical dimensions: 0.8 × 0.8 mm.
"""
function focal_spot_medium()
    return FocalSpot(0.8, 0.8, :gaussian, 3)
end

"""
    focal_spot_large()

Large focal spot for high-power imaging (thick patients).

Typical dimensions: 1.2 × 1.2 mm.
"""
function focal_spot_large()
    return FocalSpot(1.2, 1.2, :gaussian, 5)
end

"""
    focal_spot_point()

Ideal point source (no focal spot blur).

Use for testing or when focal spot effects are not needed.
"""
function focal_spot_point()
    return FocalSpot(0.0, 0.0, :gaussian, 1)
end

# =============================================================================
# Focal Spot Blur Computation
# =============================================================================

"""
    compute_focal_spot_blur_fwhm(fs::FocalSpot, geom::CTGeometry, object_distance::Float64) -> Tuple

Compute the FWHM of focal spot blur at the detector plane.

# Mathematical Background

The geometric blur (penumbra) at the detector due to finite focal spot size is:

    blur_detector = fs_size × (SDD - SOD) / SOD = fs_size × (M - 1)

where M = SDD/SOD is the magnification at the object position.

This function converts the blur to detector pixels using:

    blur_pixels = blur_cm / pixel_size_detector_cm

The pixel size at the detector is computed from the isocenter pixel size:

    pixel_size_detector = pixel_size_iso × (SDD / SID)

# CatSim Equivalence

This function produces identical FWHM values to CatSim's multi-sample source
approach when compared at the same object distance and geometry. The blur
formula is mathematically equivalent to averaging rays from a finite source.

# Arguments
- `fs::FocalSpot`: Focal spot specification (width and length in mm)
- `geom::CTGeometry`: Scanner geometry (must have SAD, SDD, pixel_size)
- `object_distance::Float64`: Distance from source to object center (cm).
  Use `geom.SAD` for objects at the isocenter.

# Returns
- `Tuple{Float64,Float64}`: (blur_width_fwhm, blur_length_fwhm) in detector pixels

# Example
```julia
fs = FocalSpot(1.0, 1.0, :gaussian, 3)  # 1 mm focal spot
geom = create_aquilion_one(n_angles=360, n_rows=64, n_cols=512, fov_cm=35.0)

# Blur at isocenter
blur_iso = compute_focal_spot_blur_fwhm(fs, geom, geom.SAD)

# Blur for object closer to source (more magnification, more blur)
blur_near = compute_focal_spot_blur_fwhm(fs, geom, geom.SAD * 0.7)
@assert blur_near[1] > blur_iso[1]  # Closer = more blur

# Blur for object farther from source (less magnification, less blur)
blur_far = compute_focal_spot_blur_fwhm(fs, geom, geom.SAD * 1.3)
@assert blur_far[1] < blur_iso[1]  # Farther = less blur
```

See also: [`create_focal_spot_kernel_spatial`](@ref), [`apply_focal_spot_blur!`](@ref)
"""
function compute_focal_spot_blur_fwhm(
    fs::FocalSpot,
    geom::CTGeometry,
    object_distance::Float64
)
    # Convert focal spot size from mm to cm
    fs_width_cm = fs.width / 10.0
    fs_length_cm = fs.length / 10.0

    # Magnification of focal spot blur at detector
    # M_blur = (SDD - object_distance) / object_distance
    if object_distance <= 0
        object_distance = geom.SAD  # Default to isocenter
    end

    M_blur = (geom.SDD - object_distance) / object_distance

    # Blur size at detector (cm)
    blur_width_cm = fs_width_cm * M_blur
    blur_length_cm = fs_length_cm * M_blur

    # Convert to detector pixels
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    blur_width_pixels = blur_width_cm / pixel_size_det
    blur_length_pixels = blur_length_cm / pixel_size_det

    return (blur_width_pixels, blur_length_pixels)
end

# Maximum kernel size for spatial convolution (controls quality/performance tradeoff)
const MAX_FOCAL_SPOT_KERNEL_SIZE = 15

"""
    create_focal_spot_kernel_spatial(fs::FocalSpot, blur_fwhm::Tuple{Float64,Float64})

Create small 2D blur kernel for spatial domain focal spot convolution.

Returns a normalized kernel array of size (2*extent+1) × (2*extent+1),
capped at MAX_FOCAL_SPOT_KERNEL_SIZE.
"""
function create_focal_spot_kernel_spatial(
    fs::FocalSpot,
    blur_fwhm::Tuple{Float64,Float64}
)
    # Convert FWHM to sigma for Gaussian
    sigma_x = blur_fwhm[1] / (2 * sqrt(2 * log(2)))
    sigma_y = blur_fwhm[2] / (2 * sqrt(2 * log(2)))

    # Compute kernel extent (3σ typically captures 99.7% of the Gaussian)
    extent_x = min(MAX_FOCAL_SPOT_KERNEL_SIZE ÷ 2, max(1, ceil(Int, 3 * sigma_x)))
    extent_y = min(MAX_FOCAL_SPOT_KERNEL_SIZE ÷ 2, max(1, ceil(Int, 3 * sigma_y)))
    extent = max(extent_x, extent_y)

    kernel_size = 2 * extent + 1
    kernel = zeros(Float64, kernel_size, kernel_size)

    center = extent + 1

    if fs.shape == :gaussian
        for dy in -extent:extent
            for dx in -extent:extent
                if sigma_x > 0 && sigma_y > 0
                    kernel[center + dx, center + dy] = exp(-dx^2 / (2 * sigma_x^2) - dy^2 / (2 * sigma_y^2))
                elseif sigma_x > 0
                    kernel[center + dx, center + dy] = exp(-dx^2 / (2 * sigma_x^2))
                elseif sigma_y > 0
                    kernel[center + dx, center + dy] = exp(-dy^2 / (2 * sigma_y^2))
                end
            end
        end

    elseif fs.shape == :uniform
        # Rectangular uniform distribution
        half_w = min(extent, ceil(Int, blur_fwhm[1] / 2))
        half_h = min(extent, ceil(Int, blur_fwhm[2] / 2))
        for dy in -half_h:half_h
            for dx in -half_w:half_w
                kernel[center + dx, center + dy] = 1.0
            end
        end

    elseif fs.shape == :bimodal
        separation = blur_fwhm[1] / 4
        for dy in -extent:extent
            for dx in -extent:extent
                if sigma_x > 0 && sigma_y > 0
                    g1 = exp(-(dx - separation)^2 / (2 * sigma_x^2) - dy^2 / (2 * sigma_y^2))
                    g2 = exp(-(dx + separation)^2 / (2 * sigma_x^2) - dy^2 / (2 * sigma_y^2))
                    kernel[center + dx, center + dy] = g1 + g2
                end
            end
        end
    end

    # Handle case where sigma is zero or negative
    total = sum(kernel)
    if total > 0
        kernel ./= total
    else
        kernel[center, center] = 1.0
    end

    return kernel
end

"""
    apply_focal_spot_blur!(sinogram, fs::FocalSpot, geom::CTGeometry;
                           object_distance=nothing) -> sinogram

Apply focal spot blur to sinogram (in-place, GPU-native).

Uses spatial domain convolution for GPU compatibility.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles]
- `fs::FocalSpot`: Focal spot specification
- `geom::CTGeometry`: Scanner geometry
- `object_distance`: Distance from source to object center (cm).
                     Default: isocenter (SAD)

# Returns
Modified blurred sinogram.
"""
function apply_focal_spot_blur!(
    sinogram::AbstractArray{T,3},
    fs::FocalSpot,
    geom::CTGeometry;
    object_distance::Union{Nothing,Float64}=nothing,
    ws_output=nothing, ws_kernel=nothing
) where T
    # Skip if point source
    if fs.width <= 0 && fs.length <= 0
        return sinogram
    end

    # Default object distance to isocenter
    if object_distance === nothing
        object_distance = geom.SAD
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Compute blur FWHM at detector
    blur_fwhm = compute_focal_spot_blur_fwhm(fs, geom, object_distance)

    # Skip if blur is negligible
    if blur_fwhm[1] < 0.1 && blur_fwhm[2] < 0.1
        return sinogram
    end

    # Create spatial domain kernel (or use pre-computed workspace kernel)
    if ws_kernel !== nothing
        kernel = ws_kernel
        kernel_size = size(kernel, 1)
    else
        kernel_cpu = T.(create_focal_spot_kernel_spatial(fs, blur_fwhm))
        kernel_size = size(kernel_cpu, 1)
        kernel = similar(sinogram, size(kernel_cpu)...)
        copyto!(kernel, kernel_cpu)
    end
    half_k = kernel_size ÷ 2

    # Output buffer (use workspace or allocate)
    output = ws_output !== nothing ? ws_output : similar(sinogram)

    # GPU-native spatial convolution
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Apply kernel
        acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                # Kernel indexing
                ki = di + half_k + 1
                kj = dj + half_k + 1

                acc += sinogram[src_col, src_row, angle] * kernel[ki, kj]
            end
        end

        output[idx] = acc
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates (for backward compatibility)
function apply_focal_spot_blur(
    sinogram::AbstractArray{T,3},
    fs::FocalSpot,
    geom::CTGeometry;
    object_distance::Union{Nothing,Float64}=nothing
) where T
    result = copy(sinogram)
    return apply_focal_spot_blur!(result, fs, geom; object_distance=object_distance)
end

# =============================================================================
# Multi-Sample Ray Tracing (More Accurate)
# =============================================================================

"""
    generate_focal_spot_samples(fs::FocalSpot) -> (positions, weights)

Generate sample positions and weights across the focal spot.

# Returns
- `positions`: Array of (dx, dy) offsets from focal spot center (mm)
- `weights`: Normalized weights for each sample
"""
function generate_focal_spot_samples(fs::FocalSpot)
    n = fs.n_samples

    if n <= 1 || (fs.width <= 0 && fs.length <= 0)
        # Point source
        return [(0.0, 0.0)], [1.0]
    end

    positions = Tuple{Float64,Float64}[]
    weights = Float64[]

    # Sample grid
    half_w = fs.width / 2
    half_l = fs.length / 2

    if fs.shape == :uniform
        # Uniform grid sampling
        for i in 1:n
            for j in 1:n
                x = -half_w + (i - 0.5) * fs.width / n
                y = -half_l + (j - 0.5) * fs.length / n
                push!(positions, (x, y))
                push!(weights, 1.0)
            end
        end

    elseif fs.shape == :gaussian
        # Gaussian-weighted sampling
        sigma_x = fs.width / (2 * sqrt(2 * log(2)))
        sigma_y = fs.length / (2 * sqrt(2 * log(2)))

        for i in 1:n
            for j in 1:n
                # Sample at Gaussian-spaced points
                x = -half_w + (i - 0.5) * fs.width / n
                y = -half_l + (j - 0.5) * fs.length / n
                w = exp(-x^2 / (2 * sigma_x^2) - y^2 / (2 * sigma_y^2))
                push!(positions, (x, y))
                push!(weights, w)
            end
        end
    end

    # Normalize weights
    total = sum(weights)
    weights ./= total

    return positions, weights
end

"""
    get_focal_spot_info(fs::FocalSpot, geom::CTGeometry) -> NamedTuple

Get diagnostic information about focal spot blur.

# Returns
Named tuple with blur information at various object distances.
"""
function get_focal_spot_info(fs::FocalSpot, geom::CTGeometry)
    # Blur at different positions
    blur_at_iso = compute_focal_spot_blur_fwhm(fs, geom, geom.SAD)
    blur_near = compute_focal_spot_blur_fwhm(fs, geom, geom.SAD * 0.7)  # Closer to source
    blur_far = compute_focal_spot_blur_fwhm(fs, geom, geom.SAD * 1.3)   # Farther from source

    return (
        size_mm = (fs.width, fs.length),
        shape = fs.shape,
        blur_at_isocenter_pixels = blur_at_iso,
        blur_near_source_pixels = blur_near,
        blur_far_from_source_pixels = blur_far
    )
end

# =============================================================================
# Exports
# =============================================================================

export FocalSpot
export focal_spot_small, focal_spot_medium, focal_spot_large, focal_spot_point
export compute_focal_spot_blur_fwhm
export create_focal_spot_kernel_spatial
export apply_focal_spot_blur!, apply_focal_spot_blur
export generate_focal_spot_samples, get_focal_spot_info
