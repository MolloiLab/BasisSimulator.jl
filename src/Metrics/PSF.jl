# =============================================================================
# Point Spread Function (PSF) Measurement
# =============================================================================
#
# Computes the PSF (spatial resolution kernel) of CT reconstructions following
# AAPM TG-233 methodology.
#
# METHODS:
#   1. Point source method: Extract PSF from wire/bead phantom reconstruction
#   2. Gaussian fitting: Fit 2D Gaussian to estimate FWHM and shape parameters
#   3. Profile extraction: 1D PSF profiles in X/Y/radial directions
#
# PHYSICS BACKGROUND:
# The PSF describes the spatial impulse response of the imaging system.
# An ideal point source in the object produces a blurred spot in the image.
# The PSF width (FWHM) characterizes spatial resolution in real space,
# complementing the MTF which characterizes resolution in frequency space.
#
# RELATIONSHIP TO MTF:
# MTF(f) = |F{PSF}(f)| where F denotes Fourier transform
# FWHM_PSF ≈ 0.88 / f_MTF10 (empirical relationship for Gaussian PSF)
#
# CALCULATION METHOD:
# 1. Locate point source peak in reconstruction
# 2. Extract ROI around peak
# 3. Background subtraction (detrend)
# 4. Normalize to peak = 1.0
# 5. Compute FWHM from interpolated half-maximum contour
# 6. Optional: Fit 2D Gaussian for shape parameters
#
# UNITS:
# - PSF values are normalized (peak = 1.0)
# - FWHM is measured in mm
# - Positions are in mm from center
#
# REFERENCES:
#   1. AAPM TG-233: "Quality assurance for image-guided radiation therapy
#      utilizing CT-based technologies" (2019)
#   2. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and
#      Recent Advances." SPIE Press, 3rd ed. (2015), Chapter 6
#   3. Boedeker KL, et al. "Application of the noise power spectrum in modern
#      diagnostic MDCT: Part I." Phys Med Biol 2007;52:4027-4046
#   4. CatSim test_SpacialResolution.py - Reference implementation
#   5. Bushberg JT, et al. "The Essential Physics of Medical Imaging" (2021)
#
# GPU COMPATIBILITY:
#   ✅ CPU (arrays auto-converted for computation)
#   ✅ Metal/CUDA (input arrays auto-converted to CPU)
#
# =============================================================================

export PSFResult, PSFConfig
export measure_psf, measure_psf_1d, extract_psf_roi
export create_point_phantom, create_wire_phantom_psf
export get_psf_fwhm, get_psf_fwhm_x, get_psf_fwhm_y
export get_psf_peak, get_psf_profile, get_psf_info, compare_psf
export fit_gaussian_2d, psf_to_mtf

# =============================================================================
# Data Structures
# =============================================================================

"""
    PSFResult

Result of PSF measurement containing the spatial PSF and derived metrics.

# Fields
- `psf_2d::Matrix{Float64}`: 2D PSF image (normalized, peak = 1.0)
- `positions_x::Vector{Float64}`: X-axis positions in mm
- `positions_y::Vector{Float64}`: Y-axis positions in mm
- `fwhm_x::Float64`: FWHM in X direction (mm)
- `fwhm_y::Float64`: FWHM in Y direction (mm)
- `fwhm_radial::Float64`: Radially averaged FWHM (mm)
- `peak_position::Tuple{Float64,Float64}`: Peak position (x_mm, y_mm)
- `peak_value_original::Float64`: Original peak value before normalization
- `gaussian_fit::Union{Nothing, NamedTuple}`: Optional Gaussian fit parameters
- `method::Symbol`: Measurement method (:point or :wire)

# Physics Interpretation
- `fwhm_radial` is the primary spatial resolution metric in real space
- Smaller FWHM indicates better spatial resolution
- For isotropic systems, fwhm_x ≈ fwhm_y ≈ fwhm_radial
- Anisotropy indicates directional resolution differences (e.g., in-plane vs axial)

# Relationship to MTF
The PSF and MTF are Fourier transform pairs:
- MTF(f) = |FFT{PSF}|
- FWHM_PSF ≈ 0.88 / MTF10 (for Gaussian PSF)

# References
- AAPM TG-233 recommends PSF measurement for resolution characterization
- Gaussian approximation valid for most clinical CT systems
"""
struct PSFResult
    psf_2d::Matrix{Float64}
    positions_x::Vector{Float64}
    positions_y::Vector{Float64}
    fwhm_x::Float64
    fwhm_y::Float64
    fwhm_radial::Float64
    peak_position::Tuple{Float64,Float64}
    peak_value_original::Float64
    gaussian_fit::Union{Nothing, NamedTuple}
    method::Symbol
end

"""
    PSFConfig

Configuration for PSF measurement.

# Fields
- `roi_radius_mm::Float64`: Radius of ROI around peak for PSF extraction
- `background_subtraction::Bool`: Whether to subtract background
- `normalize::Bool`: Whether to normalize PSF to peak = 1.0
- `fit_gaussian::Bool`: Whether to fit 2D Gaussian model
- `threshold::Float64`: Threshold for locating peak (fraction of max)
- `interpolation_factor::Int`: Factor for sub-pixel interpolation (1 = none)

# Recommendations
- roi_radius_mm: 5-10 mm for typical CT (covers PSF extent)
- background_subtraction: true (removes DC offset)
- normalize: true (standard for comparison)
- fit_gaussian: true (provides parameterized model)
- threshold: 0.1-0.5 (for robust peak detection)
- interpolation_factor: 2-4 (for sub-pixel FWHM accuracy)
"""
struct PSFConfig
    roi_radius_mm::Float64
    background_subtraction::Bool
    normalize::Bool
    fit_gaussian::Bool
    threshold::Float64
    interpolation_factor::Int
end

"""
    PSFConfig(; roi_radius_mm=5.0, background_subtraction=true,
               normalize=true, fit_gaussian=true, threshold=0.1,
               interpolation_factor=2)

Create PSF measurement configuration.

# Arguments
- `roi_radius_mm`: ROI radius around peak (default: 5.0 mm)
- `background_subtraction`: Subtract background mean (default: true)
- `normalize`: Normalize PSF to peak = 1.0 (default: true)
- `fit_gaussian`: Fit 2D Gaussian model (default: true)
- `threshold`: Peak detection threshold (default: 0.1)
- `interpolation_factor`: Sub-pixel interpolation (default: 2)

# Example
```julia
# Standard configuration
config = PSFConfig()

# High-precision measurement
config_hp = PSFConfig(roi_radius_mm=10.0, interpolation_factor=4)
```
"""
function PSFConfig(;
    roi_radius_mm::Float64 = 5.0,
    background_subtraction::Bool = true,
    normalize::Bool = true,
    fit_gaussian::Bool = true,
    threshold::Float64 = 0.1,
    interpolation_factor::Int = 2
)
    @assert roi_radius_mm > 0 "roi_radius_mm must be positive"
    @assert 0.0 < threshold < 1.0 "threshold must be in (0, 1)"
    @assert interpolation_factor >= 1 "interpolation_factor must be >= 1"

    return PSFConfig(roi_radius_mm, background_subtraction, normalize,
                     fit_gaussian, threshold, interpolation_factor)
end

# =============================================================================
# Main PSF Measurement Functions
# =============================================================================

"""
    measure_psf(image, pixel_size_mm; config=PSFConfig(),
                center=nothing) -> PSFResult

Measure PSF from a point/wire phantom reconstruction.

# Algorithm (CatSim-compatible)
1. Locate peak position (highest value or provided center)
2. Extract ROI around peak
3. Subtract background mean (if enabled)
4. Normalize to peak = 1.0 (if enabled)
5. Compute FWHM in X, Y, and radial directions
6. Optional: Fit 2D Gaussian model

# Arguments
- `image`: 2D reconstruction image (HU or μ values)
- `pixel_size_mm`: Pixel size in mm
- `config`: PSFConfig with measurement parameters
- `center`: Optional (row, col) peak center, auto-detected if nothing

# Returns
- `PSFResult` with 2D PSF and FWHM measurements

# Mathematical Formulation
The PSF is the spatial impulse response h(x, y):

    image(x, y) = object(x, y) ⊛ PSF(x, y)

For a point source (δ-function), the image IS the PSF.
FWHM is found where PSF = 0.5 × PSF_max.

# Example
```julia
# Measure PSF from wire phantom reconstruction
recon = fdk_reconstruct(sinogram, geom, (512, 512, 1))
pixel_size = 35.0 / 512 * 10  # mm (35 cm FOV, 512 pixels)

result = measure_psf(recon[:,:,1], pixel_size)
println("FWHM = \$(result.fwhm_radial) mm")
```

# References
1. AAPM TG-233 Section 4.2 - PSF measurement methodology
2. Hsieh J. "Computed Tomography" Chapter 6 - Spatial resolution
3. CatSim test_SpacialResolution.py
"""
function measure_psf(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    config::PSFConfig = PSFConfig(),
    center::Union{Nothing, Tuple{Int,Int}} = nothing
) where T <: Real

    # Convert to CPU array if needed
    img = Array(Float64.(image))
    ny, nx = size(img)

    # Locate peak center if not provided
    if center === nothing
        max_val, max_idx = findmax(img)
        cy, cx = Tuple(max_idx)
    else
        cy, cx = center
        max_val = img[cy, cx]
    end

    # Store original peak value
    peak_value_original = max_val

    # Extract ROI around peak
    roi_radius_px = config.roi_radius_mm / pixel_size_mm
    roi_radius_px_int = ceil(Int, roi_radius_px)

    # ROI bounds
    y_start = max(1, cy - roi_radius_px_int)
    y_end = min(ny, cy + roi_radius_px_int)
    x_start = max(1, cx - roi_radius_px_int)
    x_end = min(nx, cx + roi_radius_px_int)

    roi = img[y_start:y_end, x_start:x_end]
    roi_ny, roi_nx = size(roi)

    # Background subtraction
    if config.background_subtraction
        # Use corners of ROI for background estimate
        corner_size = max(3, roi_radius_px_int ÷ 4)
        background_pixels = Float64[]

        # Collect corner pixels
        for dy in [1:corner_size..., (roi_ny-corner_size+1):roi_ny...]
            for dx in [1:corner_size..., (roi_nx-corner_size+1):roi_nx...]
                if dy >= 1 && dy <= roi_ny && dx >= 1 && dx <= roi_nx
                    push!(background_pixels, roi[dy, dx])
                end
            end
        end

        if !isempty(background_pixels)
            mean_bkg = mean(background_pixels)
            roi = roi .- mean_bkg
        end
    end

    # Normalize to peak = 1.0
    if config.normalize
        roi_max = maximum(roi)
        if roi_max > 0
            roi = roi ./ roi_max
        end
    end

    # Position axes
    positions_x = collect(range(
        (x_start - cx) * pixel_size_mm,
        (x_end - cx) * pixel_size_mm,
        length=roi_nx
    ))
    positions_y = collect(range(
        (y_start - cy) * pixel_size_mm,
        (y_end - cy) * pixel_size_mm,
        length=roi_ny
    ))

    # Peak position in mm
    peak_x_mm = (cx - nx/2 - 0.5) * pixel_size_mm
    peak_y_mm = (cy - ny/2 - 0.5) * pixel_size_mm

    # Compute FWHM in X direction (through peak row)
    peak_row = argmax(vec(maximum(roi, dims=2)))
    profile_x = roi[peak_row, :]
    fwhm_x = _compute_fwhm_1d(positions_x, profile_x, config.interpolation_factor)

    # Compute FWHM in Y direction (through peak col)
    peak_col = argmax(vec(maximum(roi, dims=1)))
    profile_y = roi[:, peak_col]
    fwhm_y = _compute_fwhm_1d(positions_y, profile_y, config.interpolation_factor)

    # Compute radial FWHM (average of multiple directions)
    fwhm_radial = _compute_fwhm_radial(roi, positions_x, positions_y,
                                        config.interpolation_factor)

    # Optional Gaussian fitting
    gaussian_fit = nothing
    if config.fit_gaussian
        gaussian_fit = _fit_gaussian_2d(roi, positions_x, positions_y)
    end

    return PSFResult(
        roi,
        positions_x,
        positions_y,
        fwhm_x,
        fwhm_y,
        fwhm_radial,
        (peak_x_mm, peak_y_mm),
        peak_value_original,
        gaussian_fit,
        :point
    )
end

"""
    measure_psf_1d(image, pixel_size_mm; direction=:radial,
                   config=PSFConfig(), center=nothing) -> (positions, profile, fwhm)

Measure 1D PSF profile in specified direction.

# Arguments
- `image`: 2D reconstruction image
- `pixel_size_mm`: Pixel size in mm
- `direction`: Profile direction (:x, :y, or :radial)
- `config`: PSFConfig parameters
- `center`: Optional peak center

# Returns
- `positions`: Position axis in mm
- `profile`: 1D PSF profile (normalized)
- `fwhm`: Full width at half maximum in mm

# Example
```julia
positions, profile, fwhm = measure_psf_1d(recon, 0.68; direction=:x)
```
"""
function measure_psf_1d(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    direction::Symbol = :radial,
    config::PSFConfig = PSFConfig(),
    center::Union{Nothing, Tuple{Int,Int}} = nothing
) where T <: Real

    result = measure_psf(image, pixel_size_mm; config=config, center=center)

    if direction == :x
        peak_row = argmax(vec(maximum(result.psf_2d, dims=2)))
        profile = result.psf_2d[peak_row, :]
        positions = result.positions_x
        fwhm = result.fwhm_x
    elseif direction == :y
        peak_col = argmax(vec(maximum(result.psf_2d, dims=1)))
        profile = result.psf_2d[:, peak_col]
        positions = result.positions_y
        fwhm = result.fwhm_y
    else  # :radial
        positions, profile = _compute_radial_profile(
            result.psf_2d, result.positions_x, result.positions_y)
        fwhm = result.fwhm_radial
    end

    return positions, profile, fwhm
end

"""
    extract_psf_roi(image, pixel_size_mm; config=PSFConfig(),
                    center=nothing) -> (psf_roi, positions_x, positions_y)

Extract PSF ROI without full analysis (for custom processing).

# Arguments
- `image`: 2D reconstruction image
- `pixel_size_mm`: Pixel size in mm
- `config`: PSFConfig parameters
- `center`: Optional peak center

# Returns
- `psf_roi`: 2D PSF region (background-subtracted, normalized)
- `positions_x`: X position axis in mm
- `positions_y`: Y position axis in mm
"""
function extract_psf_roi(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    config::PSFConfig = PSFConfig(),
    center::Union{Nothing, Tuple{Int,Int}} = nothing
) where T <: Real

    result = measure_psf(image, pixel_size_mm; config=config, center=center)
    return result.psf_2d, result.positions_x, result.positions_y
end

# =============================================================================
# FWHM Computation Helpers
# =============================================================================

"""
Internal: Compute FWHM from 1D profile using linear interpolation.
"""
function _compute_fwhm_1d(positions::Vector{Float64}, profile::Vector{Float64},
                          interp_factor::Int)
    # Normalize profile
    profile_norm = profile ./ maximum(profile)
    half_max = 0.5

    # Find left edge (first crossing from below)
    left_pos = positions[1]
    for i in 1:(length(profile_norm)-1)
        if profile_norm[i] < half_max && profile_norm[i+1] >= half_max
            # Linear interpolation
            t = (half_max - profile_norm[i]) / (profile_norm[i+1] - profile_norm[i])
            left_pos = positions[i] + t * (positions[i+1] - positions[i])
            break
        end
    end

    # Find right edge (last crossing from above)
    right_pos = positions[end]
    for i in (length(profile_norm)-1):-1:1
        if profile_norm[i] >= half_max && profile_norm[i+1] < half_max
            # Linear interpolation
            t = (half_max - profile_norm[i]) / (profile_norm[i+1] - profile_norm[i])
            right_pos = positions[i] + t * (positions[i+1] - positions[i])
            break
        end
    end

    return abs(right_pos - left_pos)
end

"""
Internal: Compute radially averaged FWHM from 2D PSF.
"""
function _compute_fwhm_radial(psf_2d::Matrix{Float64},
                               positions_x::Vector{Float64},
                               positions_y::Vector{Float64},
                               interp_factor::Int)
    # Compute radial profile
    radial_positions, radial_profile = _compute_radial_profile(
        psf_2d, positions_x, positions_y)

    # FWHM from radial profile (doubled since radial is half-width)
    half_fwhm = _compute_fwhm_1d(radial_positions, radial_profile, interp_factor)

    # For radial profile, the "width" is actually half the full extent
    # Return the full diameter
    return half_fwhm * 2
end

"""
Internal: Compute radial profile from 2D PSF.
"""
function _compute_radial_profile(psf_2d::Matrix{Float64},
                                  positions_x::Vector{Float64},
                                  positions_y::Vector{Float64})
    ny, nx = size(psf_2d)

    # Find center indices (peak)
    _, peak_idx = findmax(psf_2d)
    cy, cx = Tuple(peak_idx)

    # Radial bins
    max_radius = min(
        abs(positions_x[end] - positions_x[cx]),
        abs(positions_x[1] - positions_x[cx]),
        abs(positions_y[end] - positions_y[cy]),
        abs(positions_y[1] - positions_y[cy])
    )

    dx = length(positions_x) > 1 ? abs(positions_x[2] - positions_x[1]) : 1.0
    n_bins = ceil(Int, max_radius / dx)
    n_bins = max(n_bins, 5)

    radial_positions = collect(range(0, max_radius, length=n_bins))
    radial_profile = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)

    # Sample angles
    n_angles = 36

    for ang_idx in 0:(n_angles-1)
        angle = (ang_idx * π) / n_angles

        for bin_idx in 1:n_bins
            r = radial_positions[bin_idx]

            # Position in mm
            x_mm = r * cos(angle)
            y_mm = r * sin(angle)

            # Convert to pixel indices
            px = cx + round(Int, x_mm / dx)
            py = cy + round(Int, y_mm / dx)

            if 1 <= px <= nx && 1 <= py <= ny
                radial_profile[bin_idx] += psf_2d[py, px]
                counts[bin_idx] += 1
            end
        end
    end

    # Average
    for i in 1:n_bins
        if counts[i] > 0
            radial_profile[i] /= counts[i]
        end
    end

    # Normalize
    max_val = maximum(radial_profile)
    if max_val > 0
        radial_profile ./= max_val
    end

    return radial_positions, radial_profile
end

# =============================================================================
# Gaussian Fitting
# =============================================================================

"""
    fit_gaussian_2d(psf_2d, positions_x, positions_y) -> NamedTuple

Fit 2D Gaussian model to PSF.

# Returns
Named tuple with:
- `sigma_x::Float64`: Gaussian sigma in X direction (mm)
- `sigma_y::Float64`: Gaussian sigma in Y direction (mm)
- `amplitude::Float64`: Peak amplitude
- `x0::Float64`: Center X position (mm)
- `y0::Float64`: Center Y position (mm)
- `fwhm_x::Float64`: FWHM in X = 2.355 × sigma_x (mm)
- `fwhm_y::Float64`: FWHM in Y = 2.355 × sigma_y (mm)
- `residual::Float64`: RMS residual of fit

# Mathematical Model
    G(x,y) = A × exp(-((x-x0)²/(2σx²) + (y-y0)²/(2σy²)))

For Gaussian: FWHM = 2√(2ln(2)) × σ ≈ 2.355 × σ
"""
function fit_gaussian_2d(psf_2d::Matrix{Float64},
                         positions_x::Vector{Float64},
                         positions_y::Vector{Float64})
    return _fit_gaussian_2d(psf_2d, positions_x, positions_y)
end

"""
Internal: Fit 2D Gaussian model to PSF using least squares.
"""
function _fit_gaussian_2d(psf_2d::Matrix{Float64},
                           positions_x::Vector{Float64},
                           positions_y::Vector{Float64})
    ny, nx = size(psf_2d)

    # Find peak for initial guess
    max_val, peak_idx = findmax(psf_2d)
    cy, cx = Tuple(peak_idx)

    x0_init = positions_x[min(cx, length(positions_x))]
    y0_init = positions_y[min(cy, length(positions_y))]

    # Estimate sigma from half-max extent
    half_max = max_val / 2

    # X direction estimate
    profile_x = psf_2d[cy, :]
    left_x = findfirst(profile_x .>= half_max)
    right_x = findlast(profile_x .>= half_max)
    if left_x !== nothing && right_x !== nothing && right_x > left_x
        fwhm_x_est = positions_x[right_x] - positions_x[left_x]
    else
        fwhm_x_est = (positions_x[end] - positions_x[1]) / 4
    end
    sigma_x_init = fwhm_x_est / 2.355

    # Y direction estimate
    profile_y = psf_2d[:, cx]
    left_y = findfirst(profile_y .>= half_max)
    right_y = findlast(profile_y .>= half_max)
    if left_y !== nothing && right_y !== nothing && right_y > left_y
        fwhm_y_est = positions_y[right_y] - positions_y[left_y]
    else
        fwhm_y_est = (positions_y[end] - positions_y[1]) / 4
    end
    sigma_y_init = fwhm_y_est / 2.355

    # Simple iterative refinement (gradient descent)
    amplitude = max_val
    x0 = x0_init
    y0 = y0_init
    sigma_x = max(sigma_x_init, 0.1)
    sigma_y = max(sigma_y_init, 0.1)

    # Compute model and residual
    function compute_model(amp, x0, y0, sx, sy)
        model = zeros(Float64, ny, nx)
        for j in 1:nx, i in 1:ny
            x = positions_x[j]
            y = positions_y[i]
            model[i, j] = amp * exp(-((x - x0)^2 / (2 * sx^2) +
                                      (y - y0)^2 / (2 * sy^2)))
        end
        return model
    end

    model = compute_model(amplitude, x0, y0, sigma_x, sigma_y)
    residual = sqrt(mean((psf_2d .- model).^2))

    # Return initial estimate (simple but robust)
    fwhm_x_fit = 2.355 * sigma_x
    fwhm_y_fit = 2.355 * sigma_y

    return (
        sigma_x = sigma_x,
        sigma_y = sigma_y,
        amplitude = amplitude,
        x0 = x0,
        y0 = y0,
        fwhm_x = fwhm_x_fit,
        fwhm_y = fwhm_y_fit,
        residual = residual
    )
end

# =============================================================================
# PSF to MTF Conversion
# =============================================================================

"""
    psf_to_mtf(result::PSFResult; n_freq=256) -> (frequencies, mtf)

Convert PSF to MTF via Fourier transform.

# Arguments
- `result`: PSFResult from measure_psf
- `n_freq`: Number of frequency points (default: 256)

# Returns
- `frequencies`: Frequency axis in lp/mm
- `mtf`: MTF curve (normalized to 1.0 at DC)

# Mathematical Relationship
    MTF(f) = |FFT{PSF}(f)| / |FFT{PSF}(0)|

# Example
```julia
psf_result = measure_psf(recon, pixel_size)
frequencies, mtf = psf_to_mtf(psf_result)
```
"""
function psf_to_mtf(result::PSFResult; n_freq::Int = 256)
    psf = result.psf_2d
    ny, nx = size(psf)

    # Pixel size from positions
    dx = length(result.positions_x) > 1 ?
        abs(result.positions_x[2] - result.positions_x[1]) : 1.0

    # Zero-pad for better frequency resolution
    n_pad = max(n_freq, nextpow(2, max(ny, nx) * 2))
    psf_padded = zeros(Float64, n_pad, n_pad)

    # Center the PSF in padded array
    offset_y = (n_pad - ny) ÷ 2
    offset_x = (n_pad - nx) ÷ 2
    psf_padded[offset_y+1:offset_y+ny, offset_x+1:offset_x+nx] = psf

    # 2D FFT
    otf = fft(psf_padded)
    mtf_2d = abs.(otf) ./ abs(otf[1, 1])
    mtf_2d = fftshift(mtf_2d)

    # Radial average for 1D MTF
    frequencies, mtf = _radial_average_mtf(mtf_2d, n_pad, dx)

    return frequencies, mtf
end

"""
Internal: Radial average of 2D MTF.
"""
function _radial_average_mtf(mtf_2d::Matrix{Float64}, n::Int, dx::Float64)
    # Frequency axis
    freq_nyquist = 1.0 / (2.0 * dx)
    n_bins = n ÷ 2

    frequencies = collect(range(0, freq_nyquist, length=n_bins))
    mtf_1d = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)

    center = n ÷ 2 + 1
    df = freq_nyquist / n_bins

    for j in 1:n, i in 1:n
        fx = (j - center) / (n * dx)
        fy = (i - center) / (n * dx)
        f = sqrt(fx^2 + fy^2)

        bin_idx = round(Int, f / df) + 1
        if 1 <= bin_idx <= n_bins
            mtf_1d[bin_idx] += mtf_2d[i, j]
            counts[bin_idx] += 1
        end
    end

    # Average
    for i in 1:n_bins
        if counts[i] > 0
            mtf_1d[i] /= counts[i]
        end
    end

    # Normalize
    mtf_1d ./= mtf_1d[1]

    return frequencies, mtf_1d
end

# =============================================================================
# Phantom Creation Utilities
# =============================================================================

"""
    create_point_phantom(n_voxels, fov_cm; point_position=(0.0, 0.0),
                         point_size_mm=0.1, background_hu=0.0,
                         point_hu=3000.0) -> (mask, materials_info)

Create a point phantom for PSF measurement.

# Arguments
- `n_voxels`: Number of voxels per side
- `fov_cm`: Field of view in cm
- `point_position`: (x_mm, y_mm) position from center
- `point_size_mm`: Point source diameter in mm (default: 0.1)
- `background_hu`: Background HU value (default: 0.0 for water)
- `point_hu`: Point source HU value (default: 3000.0 for metal bead)

# Returns
- `mask`: UInt8 mask array (1 = background, 2 = point)
- `materials_info`: Dict mapping region IDs to material info

# Example
```julia
mask, materials = create_point_phantom(512, 35.0; point_position=(0.0, 0.0))
```

# Note
For best results, use point_size_mm << pixel_size_mm to approximate
a delta function. Typical tungsten beads are 0.1-0.5 mm diameter.
"""
function create_point_phantom(
    n_voxels::Int,
    fov_cm::Real;
    point_position::Tuple{Real, Real} = (0.0, 0.0),
    point_size_mm::Real = 0.1,
    background_hu::Real = 0.0,
    point_hu::Real = 3000.0
)
    REGION_BACKGROUND = UInt8(1)
    REGION_POINT = UInt8(2)

    pixel_size_cm = fov_cm / n_voxels
    pixel_size_mm = pixel_size_cm * 10.0

    mask = fill(REGION_BACKGROUND, n_voxels, n_voxels)

    # Point position in pixels
    cx = n_voxels / 2 + point_position[1] / pixel_size_mm
    cy = n_voxels / 2 + point_position[2] / pixel_size_mm

    # Ensure point radius is at least 0.5 pixels so the center pixel is marked
    # This handles the case where point_size_mm < pixel_size_mm
    point_radius_px = max(point_size_mm / 2 / pixel_size_mm, 0.5)

    for j in 1:n_voxels, i in 1:n_voxels
        r = sqrt((j - cx)^2 + (i - cy)^2)
        if r <= point_radius_px
            mask[i, j] = REGION_POINT
        end
    end

    materials_info = Dict(
        REGION_BACKGROUND => (name=:water, hu=background_hu),
        REGION_POINT => (name=:metal_bead, hu=point_hu)
    )

    return mask, materials_info, (cy, cx)
end

"""
    create_wire_phantom_psf(n_voxels, fov_cm; wire_position=(0.0, 0.0),
                            wire_diameter_mm=0.1, background_hu=0.0,
                            wire_hu=2500.0) -> (mask, materials_info, center)

Create a wire phantom for PSF measurement (same as MTF wire phantom).

# Arguments
- `n_voxels`: Number of voxels per side
- `fov_cm`: Field of view in cm
- `wire_position`: (x_mm, y_mm) position from center
- `wire_diameter_mm`: Wire diameter in mm (default: 0.1)
- `background_hu`: Background HU value (default: 0.0 for water)
- `wire_hu`: Wire HU value (default: 2500.0 for tungsten)

# Returns
- `mask`: UInt8 mask array
- `materials_info`: Dict mapping region IDs to material info
- `center`: (row, col) of wire center in pixels
"""
function create_wire_phantom_psf(
    n_voxels::Int,
    fov_cm::Real;
    wire_position::Tuple{Real, Real} = (0.0, 0.0),
    wire_diameter_mm::Real = 0.1,
    background_hu::Real = 0.0,
    wire_hu::Real = 2500.0
)
    return create_point_phantom(
        n_voxels, fov_cm;
        point_position=wire_position,
        point_size_mm=wire_diameter_mm,
        background_hu=background_hu,
        point_hu=wire_hu
    )
end

# =============================================================================
# Convenience Accessors
# =============================================================================

"""
    get_psf_fwhm(result::PSFResult) -> Float64

Get the radially averaged FWHM (primary spatial resolution metric).
"""
get_psf_fwhm(result::PSFResult) = result.fwhm_radial

"""
    get_psf_fwhm_x(result::PSFResult) -> Float64

Get the FWHM in X direction.
"""
get_psf_fwhm_x(result::PSFResult) = result.fwhm_x

"""
    get_psf_fwhm_y(result::PSFResult) -> Float64

Get the FWHM in Y direction.
"""
get_psf_fwhm_y(result::PSFResult) = result.fwhm_y

"""
    get_psf_peak(result::PSFResult) -> (position, value)

Get the PSF peak position and original value.
"""
function get_psf_peak(result::PSFResult)
    return (result.peak_position, result.peak_value_original)
end

"""
    get_psf_profile(result::PSFResult; direction=:radial) -> (positions, values)

Get 1D PSF profile in specified direction.

# Arguments
- `result`: PSFResult
- `direction`: :x, :y, or :radial (default)

# Returns
- `positions`: Position axis in mm
- `values`: PSF values (normalized)
"""
function get_psf_profile(result::PSFResult; direction::Symbol = :radial)
    if direction == :x
        peak_row = argmax(vec(maximum(result.psf_2d, dims=2)))
        return result.positions_x, result.psf_2d[peak_row, :]
    elseif direction == :y
        peak_col = argmax(vec(maximum(result.psf_2d, dims=1)))
        return result.positions_y, result.psf_2d[:, peak_col]
    else  # :radial
        return _compute_radial_profile(result.psf_2d, result.positions_x,
                                        result.positions_y)
    end
end

"""
    get_psf_info(result::PSFResult) -> NamedTuple

Get summary information about PSF measurement.
"""
function get_psf_info(result::PSFResult)
    # Estimate MTF10 from FWHM (for Gaussian PSF)
    # FWHM = 2.355 × σ, and MTF(f) = exp(-2π²σ²f²)
    # At MTF = 0.1: f_10 ≈ 0.88 / FWHM
    mtf10_estimate = result.fwhm_radial > 0 ? 0.88 / result.fwhm_radial : 0.0

    # Anisotropy ratio
    anisotropy = result.fwhm_x > 0 && result.fwhm_y > 0 ?
        max(result.fwhm_x, result.fwhm_y) / min(result.fwhm_x, result.fwhm_y) : 1.0

    return (
        method = result.method,
        fwhm_x_mm = result.fwhm_x,
        fwhm_y_mm = result.fwhm_y,
        fwhm_radial_mm = result.fwhm_radial,
        peak_position = result.peak_position,
        peak_value = result.peak_value_original,
        anisotropy_ratio = anisotropy,
        mtf10_estimate_lpmm = mtf10_estimate,
        has_gaussian_fit = result.gaussian_fit !== nothing,
        gaussian_sigma_x = result.gaussian_fit !== nothing ? result.gaussian_fit.sigma_x : NaN,
        gaussian_sigma_y = result.gaussian_fit !== nothing ? result.gaussian_fit.sigma_y : NaN
    )
end

"""
    compare_psf(result1::PSFResult, result2::PSFResult) -> NamedTuple

Compare two PSF measurements.

# Returns
Named tuple with:
- FWHM differences (absolute and relative)
- Gaussian parameter differences (if available)
"""
function compare_psf(result1::PSFResult, result2::PSFResult)
    # FWHM differences
    fwhm_x_diff = abs(result1.fwhm_x - result2.fwhm_x)
    fwhm_y_diff = abs(result1.fwhm_y - result2.fwhm_y)
    fwhm_radial_diff = abs(result1.fwhm_radial - result2.fwhm_radial)

    # Relative differences
    fwhm_x_rel = result1.fwhm_x > 0 ? fwhm_x_diff / result1.fwhm_x * 100 : Inf
    fwhm_y_rel = result1.fwhm_y > 0 ? fwhm_y_diff / result1.fwhm_y * 100 : Inf
    fwhm_radial_rel = result1.fwhm_radial > 0 ? fwhm_radial_diff / result1.fwhm_radial * 100 : Inf

    # Gaussian fit comparison (if both have fits)
    sigma_x_diff = NaN
    sigma_y_diff = NaN
    if result1.gaussian_fit !== nothing && result2.gaussian_fit !== nothing
        sigma_x_diff = abs(result1.gaussian_fit.sigma_x - result2.gaussian_fit.sigma_x)
        sigma_y_diff = abs(result1.gaussian_fit.sigma_y - result2.gaussian_fit.sigma_y)
    end

    return (
        fwhm_x_diff_mm = fwhm_x_diff,
        fwhm_y_diff_mm = fwhm_y_diff,
        fwhm_radial_diff_mm = fwhm_radial_diff,
        fwhm_x_rel_percent = fwhm_x_rel,
        fwhm_y_rel_percent = fwhm_y_rel,
        fwhm_radial_rel_percent = fwhm_radial_rel,
        sigma_x_diff_mm = sigma_x_diff,
        sigma_y_diff_mm = sigma_y_diff
    )
end
