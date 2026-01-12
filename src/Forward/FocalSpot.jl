"""
    Forward/FocalSpot.jl

Finite focal spot modeling for CT simulation.

The X-ray focal spot has finite size, which affects spatial resolution.
A larger focal spot causes more geometric blur (penumbra effect).

The blur width at the detector depends on:
- Focal spot size (fs)
- Source-to-object distance (SOD)
- Object-to-detector distance (ODD)
- Magnification factor M = SDD/SOD

Blur width at detector ≈ fs × (SDD - SOD) / SOD = fs × (M - 1)

This module provides two approaches:
1. Fast: Convolution-based blur approximation
2. Accurate: Multi-sample ray tracing (slower but more accurate)
"""

using FFTW

# =============================================================================
# Focal Spot Types
# =============================================================================

"""
    FocalSpot

Focal spot specification with size and shape parameters.

# Fields
- `width`: Focal spot width in mm (fan direction, perpendicular to anode-cathode)
- `length`: Focal spot length in mm (cone direction, along anode-cathode)
- `shape`: Distribution shape (:gaussian, :uniform, :bimodal)
- `n_samples`: Number of samples for ray-tracing mode (per dimension)
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

Compute the FWHM of focal spot blur at the detector.

The blur depends on the object position between source and detector:
    blur_fwhm = fs_size × (SDD - object_distance) / object_distance

Objects closer to the source have more magnification and more blur.

# Arguments
- `fs::FocalSpot`: Focal spot specification
- `geom::CTGeometry`: Scanner geometry
- `object_distance::Float64`: Distance from source to object (cm)

# Returns
Tuple of (blur_width_fwhm, blur_length_fwhm) in detector pixels.
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

"""
    create_focal_spot_kernel(fs::FocalSpot, blur_fwhm::Tuple, n_cols::Int, n_rows::Int)

Create 2D blur kernel for focal spot convolution.

# Arguments
- `fs::FocalSpot`: Focal spot specification
- `blur_fwhm`: (width_fwhm, length_fwhm) in pixels
- `n_cols`, `n_rows`: Kernel dimensions

# Returns
Normalized 2D kernel array.
"""
function create_focal_spot_kernel(
    fs::FocalSpot,
    blur_fwhm::Tuple{Float64,Float64},
    n_cols::Int,
    n_rows::Int
)
    kernel = zeros(Float64, n_cols, n_rows)

    # Convert FWHM to sigma for Gaussian
    sigma_x = blur_fwhm[1] / (2 * sqrt(2 * log(2)))
    sigma_y = blur_fwhm[2] / (2 * sqrt(2 * log(2)))

    # Center
    cx = (n_cols + 1) / 2
    cy = (n_rows + 1) / 2

    if fs.shape == :gaussian
        for j in 1:n_rows, i in 1:n_cols
            dx = i - cx
            dy = j - cy
            if sigma_x > 0 && sigma_y > 0
                kernel[i, j] = exp(-dx^2 / (2 * sigma_x^2) - dy^2 / (2 * sigma_y^2))
            elseif sigma_x > 0
                kernel[i, j] = exp(-dx^2 / (2 * sigma_x^2))
            elseif sigma_y > 0
                kernel[i, j] = exp(-dy^2 / (2 * sigma_y^2))
            else
                kernel[i, j] = (i == round(Int, cx) && j == round(Int, cy)) ? 1.0 : 0.0
            end
        end

    elseif fs.shape == :uniform
        # Rectangular uniform distribution
        half_w = blur_fwhm[1] / 2
        half_h = blur_fwhm[2] / 2
        for j in 1:n_rows, i in 1:n_cols
            dx = abs(i - cx)
            dy = abs(j - cy)
            if dx <= half_w && dy <= half_h
                kernel[i, j] = 1.0
            end
        end

    elseif fs.shape == :bimodal
        # Bi-modal (double peak) typical of some X-ray tubes
        # Two peaks separated by ~half the focal spot width
        separation = blur_fwhm[1] / 4
        for j in 1:n_rows, i in 1:n_cols
            dx = i - cx
            dy = j - cy
            if sigma_x > 0 && sigma_y > 0
                # Two offset Gaussians
                g1 = exp(-(dx - separation)^2 / (2 * sigma_x^2) - dy^2 / (2 * sigma_y^2))
                g2 = exp(-(dx + separation)^2 / (2 * sigma_x^2) - dy^2 / (2 * sigma_y^2))
                kernel[i, j] = g1 + g2
            end
        end
    end

    # Normalize
    total = sum(kernel)
    if total > 0
        kernel ./= total
    else
        kernel[round(Int, cx), round(Int, cy)] = 1.0
    end

    return kernel
end

"""
    apply_focal_spot_blur(sinogram, fs::FocalSpot, geom::CTGeometry;
                          object_distance=nothing) -> Array

Apply focal spot blur to sinogram using convolution.

This is the fast approximation method - applies a fixed blur based on
average object position. For more accuracy, use multi-sample ray tracing.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles]
- `fs::FocalSpot`: Focal spot specification
- `geom::CTGeometry`: Scanner geometry
- `object_distance`: Distance from source to object center (cm).
                     Default: isocenter (SAD)

# Returns
Blurred sinogram.
"""
function apply_focal_spot_blur(
    sinogram::AbstractArray{T,3},
    fs::FocalSpot,
    geom::CTGeometry;
    object_distance::Union{Nothing,Float64}=nothing
) where T
    # Skip if point source
    if fs.width <= 0 && fs.length <= 0
        return sinogram
    end

    # Default object distance to isocenter
    if object_distance === nothing
        object_distance = geom.SAD
    end

    n_cols, n_rows, n_angles = size(sinogram)

    # Compute blur FWHM at detector
    blur_fwhm = compute_focal_spot_blur_fwhm(fs, geom, object_distance)

    # Skip if blur is negligible
    if blur_fwhm[1] < 0.1 && blur_fwhm[2] < 0.1
        return sinogram
    end

    # Create blur kernel
    kernel = create_focal_spot_kernel(fs, blur_fwhm, n_cols, n_rows)

    # FFT of kernel
    kernel_fft = fft(kernel)

    # Apply blur to each angle
    result = similar(sinogram)
    for angle in 1:n_angles
        proj = sinogram[:, :, angle]
        proj_fft = fft(proj)
        blurred = real(ifft(proj_fft .* kernel_fft))
        result[:, :, angle] = T.(blurred)
    end

    return result
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
export compute_focal_spot_blur_fwhm, apply_focal_spot_blur
export generate_focal_spot_samples, get_focal_spot_info
