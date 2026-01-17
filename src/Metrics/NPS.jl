# =============================================================================
# Noise Power Spectrum (NPS) Measurement
# =============================================================================
#
# Computes the NPS (noise texture characterization) of CT reconstructions
# following AAPM TG-233 methodology.
#
# METHODS:
#   1. 2D NPS: Full 2D power spectrum from uniform ROI patches
#   2. 1D NPS: Radially averaged power spectrum for isotropic systems
#
# PHYSICS BACKGROUND:
# The NPS describes the spatial frequency distribution of noise in an image.
# For quantum-limited systems, noise variance = ∫ NPS(f) df.
# The NPS shape indicates noise texture (coarse vs fine grained).
# Peak NPS frequency indicates dominant noise spatial scale.
#
# CALCULATION METHOD (AAPM TG-233):
# 1. Extract multiple non-overlapping ROIs from uniform region
# 2. Subtract mean from each ROI (detrend)
# 3. Apply 2D FFT to each ROI
# 4. Average |FFT|² across all ROIs
# 5. Normalize by ROI area and pixel size to get NPS in proper units
#
# UNITS:
# NPS is measured in HU² × mm² (or HU² × cm²)
# This represents noise variance per unit spatial frequency bandwidth
#
# REFERENCES:
#   1. AAPM TG-233: "Quality assurance for image-guided radiation therapy
#      utilizing CT-based technologies" (2019)
#   2. Boedeker KL, et al. "Application of the noise power spectrum in modern
#      diagnostic MDCT: Part I." Phys Med Biol 2007;52:4027-4046
#   3. Solomon JB, Samei E. "Quantum noise properties of CT images with anatomical
#      textured backgrounds: application to lung cancer detection." Med Phys 2012
#   4. Siewerdsen JH. "Cone-beam CT with a flat-panel detector: From image science
#      to image-guided surgery." Nucl Instrum Methods A 2011;648:S241-S250
#   5. ICRU Report 87: "Radiation Dose and Image-Quality Assessment in CT" (2013)
#
# GPU COMPATIBILITY:
#   ✅ CPU (via FFTW.jl)
#   ✅ Metal/CUDA (Arrays auto-converted to CPU for FFT)
#
# =============================================================================

export NPSResult, NPSConfig
export measure_nps, measure_nps_2d, radial_average_nps
export create_uniform_phantom_nps
export get_nps_peak, get_nps_integral, get_nps_info, get_nps_value, compare_nps
export nps_variance, nps_peak_frequency, nps_peak_value

# =============================================================================
# Data Structures
# =============================================================================

"""
    NPSResult

Result of NPS measurement containing the frequency axes and NPS values.

# Fields
- `frequencies::Vector{Float64}`: Spatial frequency axis for 1D NPS (lp/mm or lp/cm)
- `nps_1d::Vector{Float64}`: Radially averaged 1D NPS values (HU² × mm² or HU² × cm²)
- `nps_2d::Matrix{Float64}`: Full 2D NPS (optional, may be empty for memory efficiency)
- `freq_x::Vector{Float64}`: X-axis frequencies for 2D NPS
- `freq_y::Vector{Float64}`: Y-axis frequencies for 2D NPS
- `peak_frequency::Float64`: Frequency at peak NPS (lp/mm or lp/cm)
- `peak_value::Float64`: Peak NPS value
- `integrated_nps::Float64`: Total integrated NPS (equals noise variance)
- `n_rois::Int`: Number of ROIs used in measurement
- `roi_size::Tuple{Int,Int}`: Size of each ROI in pixels
- `unit::Symbol`: Frequency unit (:lp_mm or :lp_cm)

# Physics Interpretation
- `integrated_nps` equals the noise variance σ² when properly normalized
- `peak_frequency` indicates the dominant spatial scale of noise texture
- A flat NPS indicates white (uncorrelated) noise
- NPS peak at low frequencies indicates correlated (coarse) noise texture
- NPS peak at high frequencies indicates fine-grained noise texture

# References
- AAPM TG-233 recommends NPS measurement at multiple dose levels
- ICRU Report 87 defines standard NPS calculation methodology
"""
struct NPSResult
    frequencies::Vector{Float64}
    nps_1d::Vector{Float64}
    nps_2d::Matrix{Float64}
    freq_x::Vector{Float64}
    freq_y::Vector{Float64}
    peak_frequency::Float64
    peak_value::Float64
    integrated_nps::Float64
    n_rois::Int
    roi_size::Tuple{Int,Int}
    unit::Symbol
end

"""
    NPSConfig

Configuration for NPS measurement.

# Fields
- `roi_size::Int`: Size of square ROI in pixels (power of 2 recommended, e.g., 64, 128)
- `n_rois::Int`: Number of ROIs to average (higher = smoother NPS)
- `overlap::Float64`: Fractional overlap between ROIs (0.0 = none, 0.5 = 50%)
- `detrend::Symbol`: Detrending method (:mean, :linear, :none)
- `window::Symbol`: Window function (:none, :hann, :hamming)
- `include_2d::Bool`: Whether to store full 2D NPS (memory intensive)

# Recommendations (AAPM TG-233)
- roi_size: 64 or 128 pixels for typical clinical CT resolution
- n_rois: At least 16, preferably 32-64 for smooth NPS
- overlap: 0.0-0.5 depending on available image area
- detrend: :mean is standard, :linear can remove residual trends
- window: Usually :none for NPS, but :hann can reduce spectral leakage
"""
struct NPSConfig
    roi_size::Int
    n_rois::Int
    overlap::Float64
    detrend::Symbol
    window::Symbol
    include_2d::Bool
end

"""
    NPSConfig(; roi_size=64, n_rois=16, overlap=0.0, detrend=:mean,
               window=:none, include_2d=false)

Create NPS measurement configuration.

# Arguments
- `roi_size`: Square ROI size in pixels (default: 64, power of 2 recommended)
- `n_rois`: Number of ROIs to average (default: 16)
- `overlap`: ROI overlap fraction (default: 0.0 = no overlap)
- `detrend`: Detrending method (default: :mean)
- `window`: Window function (default: :none)
- `include_2d`: Store full 2D NPS (default: false for memory efficiency)

# Example
```julia
# Standard configuration
config = NPSConfig()

# High-quality measurement with more ROIs
config_hq = NPSConfig(roi_size=128, n_rois=32, overlap=0.5)
```
"""
function NPSConfig(;
    roi_size::Int = 64,
    n_rois::Int = 16,
    overlap::Float64 = 0.0,
    detrend::Symbol = :mean,
    window::Symbol = :none,
    include_2d::Bool = false
)
    @assert roi_size > 0 "roi_size must be positive"
    @assert n_rois > 0 "n_rois must be positive"
    @assert 0.0 <= overlap < 1.0 "overlap must be in [0, 1)"
    @assert detrend in (:mean, :linear, :none) "detrend must be :mean, :linear, or :none"
    @assert window in (:none, :hann, :hamming) "window must be :none, :hann, or :hamming"

    return NPSConfig(roi_size, n_rois, overlap, detrend, window, include_2d)
end

# =============================================================================
# Main NPS Measurement Functions
# =============================================================================

"""
    measure_nps(image, pixel_size_mm; config=NPSConfig(), roi_center=nothing,
                roi_radius_mm=nothing, unit=:lp_mm) -> NPSResult

Measure NPS from a uniform region of a CT reconstruction.

# Algorithm (AAPM TG-233 / ICRU 87)
1. Extract uniform ROI region (specified by center/radius or full image)
2. Extract multiple non-overlapping square patches
3. For each patch:
   a. Subtract mean (detrend)
   b. Apply window function (optional)
   c. Compute 2D FFT
   d. Compute |FFT|² (power spectrum)
4. Average power spectra across all patches
5. Normalize: NPS = (Δx × Δy / (Nx × Ny)) × mean(|FFT|²)
6. Radially average for 1D NPS

# Arguments
- `image`: 2D reconstruction image in HU (after conversion from μ)
- `pixel_size_mm`: Pixel size in mm
- `config`: NPSConfig with measurement parameters
- `roi_center`: Optional (row, col) center for circular ROI restriction
- `roi_radius_mm`: Optional radius for circular ROI restriction
- `unit`: Output frequency unit (:lp_mm or :lp_cm)

# Returns
- `NPSResult` with 1D/2D NPS and derived metrics

# Mathematical Formulation
The NPS is the Fourier transform of the noise autocovariance:

    NPS(fx, fy) = lim_{A→∞} (1/A) × |∫∫ ΔI(x,y) × exp(-2πi(fx×x + fy×y)) dx dy|²

For discrete images with N pixels of size Δ:

    NPS(k,l) = (Δx × Δy / (Nx × Ny)) × ⟨|FFT{ΔI}(k,l)|²⟩

where ΔI = I - ⟨I⟩ is the detrended image and ⟨⟩ denotes ensemble average.

# Example
```julia
# Measure NPS from water phantom reconstruction
recon_hu = 1000f0 .* (recon .- μ_water) ./ μ_water
pixel_size = 35.0 / 512 * 10  # mm (35 cm FOV, 512 pixels)

result = measure_nps(recon_hu, pixel_size;
    config=NPSConfig(roi_size=64, n_rois=32))

println("Peak NPS frequency: \$(result.peak_frequency) lp/mm")
println("Noise variance: \$(result.integrated_nps) HU²")
```

# References
1. AAPM TG-233 Section 4.3 - NPS measurement methodology
2. ICRU Report 87 Section 4.2.2 - Definition and computation of NPS
3. Boedeker KL, et al. Phys Med Biol 2007;52:4027-4046
"""
function measure_nps(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    config::NPSConfig = NPSConfig(),
    roi_center::Union{Nothing, Tuple{Int,Int}} = nothing,
    roi_radius_mm::Union{Nothing, Real} = nothing,
    unit::Symbol = :lp_mm
) where T <: Real

    # Convert to CPU array and Float64
    img = Array(Float64.(image))
    ny, nx = size(img)

    # Extract uniform region if specified
    if roi_center !== nothing && roi_radius_mm !== nothing
        cy, cx = roi_center
        roi_radius_px = roi_radius_mm / pixel_size_mm

        # Create circular mask
        mask = zeros(Bool, ny, nx)
        for j in 1:nx, i in 1:ny
            r = sqrt((i - cy)^2 + (j - cx)^2)
            if r <= roi_radius_px
                mask[i, j] = true
            end
        end

        # Find bounding box of masked region
        rows_mask = findall(any(mask, dims=2)[:])
        cols_mask = findall(any(mask, dims=1)[:])

        if isempty(rows_mask) || isempty(cols_mask)
            error("No pixels found in specified ROI")
        end

        img_roi = img[minimum(rows_mask):maximum(rows_mask),
                      minimum(cols_mask):maximum(cols_mask)]
    else
        img_roi = img
    end

    ny_roi, nx_roi = size(img_roi)
    roi_size = config.roi_size

    # Validate ROI size fits in image
    if roi_size > min(ny_roi, nx_roi)
        roi_size = min(ny_roi, nx_roi) ÷ 2
        if roi_size < 8
            error("Image too small for NPS measurement. Need at least 16×16 pixels.")
        end
    end

    # Extract ROI patches
    rois = _extract_rois(img_roi, roi_size, config.n_rois, config.overlap)
    actual_n_rois = length(rois)

    if actual_n_rois == 0
        error("Could not extract any ROIs. Image may be too small.")
    end

    # Create window function
    window = _create_window(roi_size, config.window)

    # Compute NPS from each ROI
    power_sum = zeros(Float64, roi_size, roi_size)

    for roi in rois
        # Detrend
        roi_detrended = _detrend_roi(roi, config.detrend)

        # Apply window
        roi_windowed = roi_detrended .* window

        # FFT
        roi_fft = fft(roi_windowed)

        # Power spectrum
        power = abs2.(roi_fft)

        # Accumulate
        power_sum .+= power
    end

    # Average power spectrum
    power_avg = power_sum ./ actual_n_rois

    # Normalize to get NPS in HU² × mm²
    # NPS = (Δx × Δy) × power / (Nx × Ny)
    # where Δx, Δy are pixel sizes in mm
    # and Nx, Ny are ROI dimensions in pixels
    normalization = (pixel_size_mm^2) / (roi_size^2)
    nps_2d = power_avg .* normalization

    # Shift to center DC
    nps_2d_shifted = fftshift(nps_2d)

    # Frequency axes
    freq_nyquist = 1.0 / (2.0 * pixel_size_mm)  # lp/mm
    freq_axis = fftshift(fftfreq(roi_size, 1.0 / pixel_size_mm))

    # Radial average for 1D NPS
    radial_freqs, nps_1d = _radial_average_nps(nps_2d_shifted, freq_axis)

    # Only positive frequencies
    center_idx = length(radial_freqs) ÷ 2 + 1
    radial_freqs = radial_freqs[center_idx:end]
    nps_1d = nps_1d[center_idx:end]

    # Convert units if requested
    if unit == :lp_cm
        radial_freqs .*= 10.0
        freq_axis_out = freq_axis .* 10.0
        # NPS units become HU² × cm²
        nps_1d ./= 100.0  # mm² to cm²
        nps_2d_shifted ./= 100.0
    else
        freq_axis_out = freq_axis
    end

    # Find peak (excluding DC)
    dc_idx = 1
    if length(nps_1d) > 1
        # Skip DC (index 1) and find peak
        peak_idx = argmax(nps_1d[2:end]) + 1
        peak_frequency = radial_freqs[peak_idx]
        peak_value = nps_1d[peak_idx]
    else
        peak_frequency = 0.0
        peak_value = nps_1d[1]
    end

    # Integrate NPS to get variance
    # For discrete: σ² = Σ NPS × Δf
    df = length(radial_freqs) > 1 ? radial_freqs[2] - radial_freqs[1] : 1.0
    # Use trapezoidal integration for radially averaged NPS
    # Account for radial weighting: ∫₀^∞ NPS(f) × 2πf df for 2D
    # For 1D radial average, integrate directly
    integrated_nps = sum(nps_1d) * df

    # Build result
    nps_2d_out = config.include_2d ? nps_2d_shifted : zeros(Float64, 0, 0)
    freq_x_out = config.include_2d ? collect(freq_axis_out) : Float64[]
    freq_y_out = config.include_2d ? collect(freq_axis_out) : Float64[]

    return NPSResult(
        radial_freqs,
        nps_1d,
        nps_2d_out,
        freq_x_out,
        freq_y_out,
        peak_frequency,
        peak_value,
        integrated_nps,
        actual_n_rois,
        (roi_size, roi_size),
        unit
    )
end

"""
    measure_nps_2d(image, pixel_size_mm; config=NPSConfig(), unit=:lp_mm) -> (nps_2d, freq_x, freq_y)

Measure full 2D NPS (for anisotropic noise analysis).

This is a convenience wrapper that forces 2D NPS storage.

# Returns
- `nps_2d`: 2D NPS matrix (HU² × mm² or HU² × cm²)
- `freq_x`: X-axis frequencies
- `freq_y`: Y-axis frequencies
"""
function measure_nps_2d(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    config::NPSConfig = NPSConfig(),
    unit::Symbol = :lp_mm
) where T <: Real

    config_2d = NPSConfig(
        roi_size = config.roi_size,
        n_rois = config.n_rois,
        overlap = config.overlap,
        detrend = config.detrend,
        window = config.window,
        include_2d = true
    )

    result = measure_nps(image, pixel_size_mm; config=config_2d, unit=unit)

    return result.nps_2d, result.freq_x, result.freq_y
end

"""
    radial_average_nps(nps_2d, freq_axis) -> (radial_freqs, nps_1d)

Compute radially averaged 1D NPS from 2D NPS.

# Arguments
- `nps_2d`: Centered 2D NPS (DC at center)
- `freq_axis`: Frequency axis (same for x and y)

# Returns
- `radial_freqs`: Radial frequency axis
- `nps_1d`: Radially averaged NPS values
"""
function radial_average_nps(nps_2d::Matrix{Float64}, freq_axis::Vector{Float64})
    return _radial_average_nps(nps_2d, freq_axis)
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
Extract multiple non-overlapping ROIs from image.
"""
function _extract_rois(img::Matrix{Float64}, roi_size::Int, target_n_rois::Int, overlap::Float64)
    ny, nx = size(img)

    # Effective step size accounting for overlap
    step = round(Int, roi_size * (1 - overlap))
    step = max(step, 1)

    # Number of ROIs that fit
    n_rois_y = (ny - roi_size) ÷ step + 1
    n_rois_x = (nx - roi_size) ÷ step + 1

    total_available = n_rois_y * n_rois_x

    rois = Matrix{Float64}[]

    if total_available <= target_n_rois
        # Use all available ROIs
        for iy in 0:(n_rois_y-1)
            for ix in 0:(n_rois_x-1)
                row_start = 1 + iy * step
                col_start = 1 + ix * step

                if row_start + roi_size - 1 <= ny && col_start + roi_size - 1 <= nx
                    roi = img[row_start:row_start+roi_size-1, col_start:col_start+roi_size-1]
                    push!(rois, copy(roi))
                end
            end
        end
    else
        # Random selection of ROIs
        all_positions = [(iy, ix) for iy in 0:(n_rois_y-1) for ix in 0:(n_rois_x-1)]
        selected = all_positions[randperm(length(all_positions))[1:target_n_rois]]

        for (iy, ix) in selected
            row_start = 1 + iy * step
            col_start = 1 + ix * step

            if row_start + roi_size - 1 <= ny && col_start + roi_size - 1 <= nx
                roi = img[row_start:row_start+roi_size-1, col_start:col_start+roi_size-1]
                push!(rois, copy(roi))
            end
        end
    end

    return rois
end

"""
Create window function for spectral analysis.
"""
function _create_window(size::Int, window_type::Symbol)
    if window_type == :none
        return ones(Float64, size, size)
    end

    # 1D window
    if window_type == :hann
        w1d = [0.5 * (1 - cos(2π * i / (size - 1))) for i in 0:(size-1)]
    elseif window_type == :hamming
        w1d = [0.54 - 0.46 * cos(2π * i / (size - 1)) for i in 0:(size-1)]
    else
        w1d = ones(Float64, size)
    end

    # 2D separable window
    return w1d * w1d'
end

"""
Detrend ROI using specified method.
"""
function _detrend_roi(roi::Matrix{Float64}, method::Symbol)
    if method == :none
        return roi
    elseif method == :mean
        return roi .- mean(roi)
    elseif method == :linear
        # Remove 2D linear trend (plane fit)
        ny, nx = size(roi)

        # Create coordinate matrices
        x = repeat(1:nx, 1, ny)'
        y = repeat(1:ny, 1, nx)

        # Flatten
        x_flat = vec(x)
        y_flat = vec(y)
        z_flat = vec(roi)

        # Solve for plane: z = a*x + b*y + c
        A = [x_flat y_flat ones(length(x_flat))]
        coeffs = A \ z_flat

        # Remove trend
        trend = coeffs[1] * x + coeffs[2] * y .+ coeffs[3]
        return roi .- trend
    else
        return roi .- mean(roi)
    end
end

"""
Radially average 2D NPS to get 1D NPS (internal).
"""
function _radial_average_nps(nps_2d::Matrix{Float64}, freq_axis::AbstractVector{Float64})
    ny, nx = size(nps_2d)

    # Create radial bins
    freq_array = collect(freq_axis)
    n_bins = length(freq_array) ÷ 2 + 1

    # Frequency spacing
    df = length(freq_array) > 1 ? abs(freq_array[2] - freq_array[1]) : 1.0

    # Radial frequency bins
    radial_freqs = collect(range(0, stop=(n_bins-1)*df, length=n_bins))
    nps_1d = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)

    # Center indices
    cy = ny ÷ 2 + 1
    cx = nx ÷ 2 + 1

    for j in 1:nx, i in 1:ny
        # Radial distance in frequency
        fx = freq_array[min(j, length(freq_array))]
        fy = freq_array[min(i, length(freq_array))]
        r = sqrt(fx^2 + fy^2)

        # Bin index
        bin_idx = round(Int, r / df) + 1
        if 1 <= bin_idx <= n_bins
            nps_1d[bin_idx] += nps_2d[i, j]
            counts[bin_idx] += 1
        end
    end

    # Average
    for i in 1:n_bins
        if counts[i] > 0
            nps_1d[i] /= counts[i]
        end
    end

    return radial_freqs, nps_1d
end

"""
Generate FFT frequencies (matches FFTW convention).
"""
function fftfreq(n::Int, sample_rate::Real)
    # Returns frequencies for n-point FFT with given sample rate
    # Follows numpy.fft.fftfreq convention
    results = zeros(Float64, n)
    if n % 2 == 0
        # Even
        for i in 0:(n÷2-1)
            results[i+1] = i * sample_rate / n
        end
        for i in (n÷2):(n-1)
            results[i+1] = (i - n) * sample_rate / n
        end
    else
        # Odd
        for i in 0:((n-1)÷2)
            results[i+1] = i * sample_rate / n
        end
        for i in ((n+1)÷2):(n-1)
            results[i+1] = (i - n) * sample_rate / n
        end
    end
    return results
end

# =============================================================================
# Phantom Creation
# =============================================================================

"""
    create_uniform_phantom_nps(n_voxels, fov_cm; hu_value=0.0, noise_std=30.0) -> image_hu

Create a uniform phantom with Gaussian noise for NPS testing.

# Arguments
- `n_voxels`: Number of voxels per side
- `fov_cm`: Field of view in cm
- `hu_value`: Mean HU value (default: 0.0 for water)
- `noise_std`: Standard deviation of Gaussian noise in HU (default: 30.0)

# Returns
- `image_hu`: 2D image in HU with Gaussian noise

# Example
```julia
# Create 256×256 water phantom with 30 HU noise
phantom = create_uniform_phantom_nps(256, 35.0; hu_value=0.0, noise_std=30.0)
```
"""
function create_uniform_phantom_nps(
    n_voxels::Int,
    fov_cm::Real;
    hu_value::Real = 0.0,
    noise_std::Real = 30.0
)
    # Create uniform image with Gaussian noise
    return fill(Float64(hu_value), n_voxels, n_voxels) .+ noise_std .* randn(n_voxels, n_voxels)
end

# =============================================================================
# Convenience Accessors and Utilities
# =============================================================================

"""
    nps_peak_frequency(result::NPSResult) -> Float64

Get the frequency at peak NPS.
"""
nps_peak_frequency(result::NPSResult) = result.peak_frequency

"""
    nps_peak_value(result::NPSResult) -> Float64

Get the peak NPS value.
"""
nps_peak_value(result::NPSResult) = result.peak_value

"""
    nps_variance(result::NPSResult) -> Float64

Get the integrated NPS (equals noise variance σ²).
"""
nps_variance(result::NPSResult) = result.integrated_nps

"""
    get_nps_peak(result::NPSResult) -> (frequency, value)

Get the peak NPS frequency and value.
"""
function get_nps_peak(result::NPSResult)
    return (result.peak_frequency, result.peak_value)
end

"""
    get_nps_integral(result::NPSResult; f_min=0.0, f_max=Inf) -> Float64

Get integrated NPS over specified frequency range.

The integral of NPS equals the noise variance within that frequency band.
"""
function get_nps_integral(result::NPSResult; f_min::Real=0.0, f_max::Real=Inf)
    # Find frequency indices in range
    mask = (result.frequencies .>= f_min) .& (result.frequencies .<= f_max)
    freqs_in_range = result.frequencies[mask]
    nps_in_range = result.nps_1d[mask]

    if length(freqs_in_range) < 2
        return sum(nps_in_range)
    end

    # Trapezoidal integration
    df = freqs_in_range[2] - freqs_in_range[1]
    return sum(nps_in_range) * df
end

"""
    get_nps_info(result::NPSResult) -> NamedTuple

Get summary information about NPS measurement.
"""
function get_nps_info(result::NPSResult)
    # Find frequency at 50% of peak
    if result.peak_value > 0
        half_max = result.peak_value / 2
        f_half_max = 0.0
        for i in 1:length(result.nps_1d)
            if result.nps_1d[i] >= half_max
                f_half_max = result.frequencies[i]
            end
        end
    else
        f_half_max = 0.0
    end

    return (
        unit = result.unit,
        peak_frequency = result.peak_frequency,
        peak_value = result.peak_value,
        integrated_nps = result.integrated_nps,
        noise_std = sqrt(abs(result.integrated_nps)),  # σ = √(variance)
        n_rois = result.n_rois,
        roi_size = result.roi_size,
        nyquist = result.frequencies[end],
        f_half_max = f_half_max
    )
end

"""
    get_nps_value(result::NPSResult, frequency::Float64) -> Float64

Get the NPS value at a specified frequency (interpolated).
"""
function get_nps_value(result::NPSResult, frequency::Float64)
    if frequency <= result.frequencies[1]
        return result.nps_1d[1]
    elseif frequency >= result.frequencies[end]
        return result.nps_1d[end]
    end

    # Linear interpolation
    for i in 1:(length(result.frequencies)-1)
        if result.frequencies[i] <= frequency && result.frequencies[i+1] > frequency
            t = (frequency - result.frequencies[i]) /
                (result.frequencies[i+1] - result.frequencies[i])
            return (1 - t) * result.nps_1d[i] + t * result.nps_1d[i+1]
        end
    end

    return 0.0
end

"""
    compare_nps(result1::NPSResult, result2::NPSResult;
                test_frequencies=[]) -> NamedTuple

Compare two NPS measurements.

# Returns
Named tuple with:
- Peak frequency and value differences
- Integrated NPS (variance) differences
- NPS values at specified test frequencies
"""
function compare_nps(
    result1::NPSResult,
    result2::NPSResult;
    test_frequencies::Vector{Float64} = Float64[]
)
    # Peak frequency difference
    peak_freq_diff = abs(result1.peak_frequency - result2.peak_frequency)
    peak_freq_rel = result1.peak_frequency > 0 ?
        peak_freq_diff / result1.peak_frequency * 100 : Inf

    # Peak value difference
    peak_val_diff = abs(result1.peak_value - result2.peak_value)
    peak_val_rel = result1.peak_value > 0 ?
        peak_val_diff / result1.peak_value * 100 : Inf

    # Integrated NPS (variance) difference
    var_diff = abs(result1.integrated_nps - result2.integrated_nps)
    var_rel = result1.integrated_nps > 0 ?
        var_diff / result1.integrated_nps * 100 : Inf

    # Noise std difference
    std1 = sqrt(abs(result1.integrated_nps))
    std2 = sqrt(abs(result2.integrated_nps))
    std_diff = abs(std1 - std2)
    std_rel = std1 > 0 ? std_diff / std1 * 100 : Inf

    # Comparison at specified frequencies
    freq_comparison = Float64[]
    for f in test_frequencies
        v1 = get_nps_value(result1, f)
        v2 = get_nps_value(result2, f)
        push!(freq_comparison, abs(v1 - v2))
    end

    return (
        peak_freq_diff = peak_freq_diff,
        peak_freq_rel_percent = peak_freq_rel,
        peak_val_diff = peak_val_diff,
        peak_val_rel_percent = peak_val_rel,
        variance_diff = var_diff,
        variance_rel_percent = var_rel,
        noise_std_diff = std_diff,
        noise_std_rel_percent = std_rel,
        test_frequencies = test_frequencies,
        freq_differences = freq_comparison
    )
end
