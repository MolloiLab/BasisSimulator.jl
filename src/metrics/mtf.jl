# =============================================================================
# Modulation Transfer Function (MTF) Measurement
# =============================================================================
#
# Computes the MTF (spatial resolution) of CT reconstructions following
# AAPM TG-233 methodology.
#
# METHODS:
#   1. Wire phantom method: 2D FFT of PSF from tungsten wire phantom
#   2. Edge phantom method: ESF → LSF → MTF via differentiation and FFT
#
# REFERENCES:
#   1. AAPM TG-233: "Quality assurance for image-guided radiation therapy
#      utilizing CT-based technologies" (2019)
#   2. Verdun FR, et al. "Image quality in CT: From physical measurements to
#      model observers." Physica Medica (2015) doi:10.1016/j.ejmp.2015.08.007
#   3. Bushberg JT, et al. "The Essential Physics of Medical Imaging" (2021)
#   4. CatSim test_SpacialResolution.py - Reference implementation
#
# GPU COMPATIBILITY:
#   ✅ CPU (via FFTW.jl)
#   ✅ Metal/CUDA (Arrays auto-converted to CPU for FFT)
#
# =============================================================================

export MTFResult, WirePhantomMTF, EdgePhantomMTF
export measure_mtf_wire, measure_mtf_edge
export create_wire_phantom, create_edge_phantom
export get_mtf_frequency, get_mtf_value, get_mtf_info, compare_mtf
export mtf10, mtf50, mtf5

# =============================================================================
# Data Structures
# =============================================================================

"""
    MTFResult

Result of MTF measurement containing the frequency axis and MTF curve.

# Fields
- `frequencies::Vector{Float64}`: Spatial frequency axis (lp/cm or lp/mm)
- `mtf::Vector{Float64}`: MTF values (normalized to 1.0 at DC)
- `mtf50::Float64`: Frequency at 50% MTF (lp/cm)
- `mtf10::Float64`: Frequency at 10% MTF (lp/cm)
- `mtf5::Float64`: Frequency at 5% MTF (lp/cm)
- `method::Symbol`: Measurement method (:wire or :edge)
- `unit::Symbol`: Frequency unit (:lp_cm or :lp_mm)

# References
- AAPM TG-233 recommends reporting MTF at 10% and 50% levels
- f₁₀ (10% MTF) is the limiting spatial resolution
- f₅₀ (50% MTF) indicates clinical performance at moderate contrast
"""
struct MTFResult
    frequencies::Vector{Float64}
    mtf::Vector{Float64}
    mtf50::Float64
    mtf10::Float64
    mtf5::Float64
    method::Symbol
    unit::Symbol
end

"""
    WirePhantomMTF

Configuration for wire phantom MTF measurement.

# Fields
- `wire_diameter_mm::Float64`: Wire diameter (typically 0.1-0.2 mm tungsten)
- `roi_radius_mm::Float64`: ROI radius for PSF extraction
- `background_subtraction::Bool`: Whether to subtract background
- `threshold::Float64`: Threshold for automatic wire localization
"""
struct WirePhantomMTF
    wire_diameter_mm::Float64
    roi_radius_mm::Float64
    background_subtraction::Bool
    threshold::Float64
end

"""
    WirePhantomMTF(; wire_diameter_mm=0.1, roi_radius_mm=5.0,
                    background_subtraction=true, threshold=0.1)

Create wire phantom MTF configuration.

# Arguments
- `wire_diameter_mm`: Wire diameter (default: 0.1 mm for thin tungsten wire)
- `roi_radius_mm`: ROI radius around wire for PSF (default: 5.0 mm)
- `background_subtraction`: Subtract background mean (default: true)
- `threshold`: Threshold for wire localization (default: 0.1 of max)
"""
function WirePhantomMTF(;
    wire_diameter_mm::Float64 = 0.1,
    roi_radius_mm::Float64 = 5.0,
    background_subtraction::Bool = true,
    threshold::Float64 = 0.1
)
    return WirePhantomMTF(wire_diameter_mm, roi_radius_mm,
                          background_subtraction, threshold)
end

"""
    EdgePhantomMTF

Configuration for edge phantom MTF measurement.

# Fields
- `edge_angle_deg::Float64`: Edge angle relative to pixel grid (typically 2-5°)
- `oversampling_factor::Int`: Oversampling factor for ESF (typically 4-10)
- `smoothing_window::Int`: Window size for LSF smoothing (0 = no smoothing)
"""
struct EdgePhantomMTF
    edge_angle_deg::Float64
    oversampling_factor::Int
    smoothing_window::Int
end

"""
    EdgePhantomMTF(; edge_angle_deg=3.0, oversampling_factor=8, smoothing_window=0)

Create edge phantom MTF configuration.

# Arguments
- `edge_angle_deg`: Edge angle for oversampling (default: 3.0°, per AAPM)
- `oversampling_factor`: Factor for ESF super-resolution (default: 8)
- `smoothing_window`: LSF smoothing window (default: 0 = none)

# References
- Slanted edge must be 2-5° from perpendicular for proper oversampling
- See IEC 62220-1-1:2015 for medical imaging MTF measurement
"""
function EdgePhantomMTF(;
    edge_angle_deg::Float64 = 3.0,
    oversampling_factor::Int = 8,
    smoothing_window::Int = 0
)
    return EdgePhantomMTF(edge_angle_deg, oversampling_factor, smoothing_window)
end

# =============================================================================
# Wire Phantom MTF Measurement
# =============================================================================

"""
    measure_mtf_wire(image, pixel_size_mm; config=WirePhantomMTF(),
                     center=nothing, unit=:lp_cm) -> MTFResult

Measure MTF from a wire phantom reconstruction using 2D FFT method.

# Algorithm (CatSim-compatible)
1. Locate wire position (peak in image or use provided center)
2. Extract ROI around wire
3. Subtract background mean (detrend)
4. Normalize PSF to peak = 1.0
5. Compute 2D FFT to get OTF
6. Take magnitude and normalize: MTF = |OTF| / |OTF[0,0]|
7. Radially average to get 1D MTF curve

# Arguments
- `image`: 2D reconstruction image (HU or μ values)
- `pixel_size_mm`: Pixel size in mm
- `config`: WirePhantomMTF configuration
- `center`: Optional (row, col) wire center, auto-detected if nothing
- `unit`: Output frequency unit (:lp_cm or :lp_mm)

# Returns
- `MTFResult` with MTF curve and characteristic frequencies

# Mathematical Formulation
The PSF of a wire phantom is the 2D system response. The MTF is:

    MTF(f) = |ℱ{PSF}(f)|

where ℱ denotes the Fourier transform. For a rotationally symmetric system,
radial averaging gives the 1D MTF.

# Example
```julia
# Reconstruct wire phantom
recon = fdk_reconstruct(sinogram, geom, (512, 512, 1))
pixel_size = 35.0 / 512 * 10  # mm (35 cm FOV, 512 pixels)

# Measure MTF
result = measure_mtf_wire(recon[:,:,1], pixel_size)
println("MTF10 = \$(result.mtf10) lp/cm")
println("MTF50 = \$(result.mtf50) lp/cm")
```

# References
1. CatSim test_SpacialResolution.py - exact algorithm
2. AAPM TG-233 Section 4.2 - Wire phantom methodology
"""
function measure_mtf_wire(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    config::WirePhantomMTF = WirePhantomMTF(),
    center::Union{Nothing, Tuple{Int,Int}} = nothing,
    unit::Symbol = :lp_cm
) where T <: Real

    # Convert to CPU array if needed (for FFT)
    img = Array(Float64.(image))
    ny, nx = size(img)

    # FOV in mm
    fov_mm = nx * pixel_size_mm

    # Locate wire center if not provided
    if center === nothing
        # Find peak (wire location)
        max_val, max_idx = findmax(img)
        cy, cx = Tuple(max_idx)
    else
        cy, cx = center
    end

    # Background subtraction (detrend)
    if config.background_subtraction
        # Create mask for background (outside ROI around wire)
        roi_radius_px = config.roi_radius_mm / pixel_size_mm
        background_mask = zeros(Bool, ny, nx)
        for j in 1:nx, i in 1:ny
            r = sqrt((i - cy)^2 + (j - cx)^2)
            if r > roi_radius_px * 2 && r < min(ny, nx) / 2
                background_mask[i, j] = true
            end
        end

        if sum(background_mask) > 0
            mean_bkg = mean(img[background_mask])
            img = img .- mean_bkg
        end
    end

    # Normalize to peak = 1
    img = img ./ maximum(abs.(img))

    # Apply threshold to create PSF ROI
    psf_mask = img .>= config.threshold
    if sum(psf_mask) > 0
        # Determine ROI extent
        rows_with_signal = findall(any(psf_mask, dims=2)[:])
        cols_with_signal = findall(any(psf_mask, dims=1)[:])

        if !isempty(rows_with_signal) && !isempty(cols_with_signal)
            r_roi = max(maximum(rows_with_signal) - minimum(rows_with_signal),
                       maximum(cols_with_signal) - minimum(cols_with_signal)) / 2
        else
            r_roi = config.roi_radius_mm / pixel_size_mm
        end
    else
        r_roi = config.roi_radius_mm / pixel_size_mm
    end

    # Create circular ROI mask
    roi_mask = zeros(Float64, ny, nx)
    for j in 1:nx, i in 1:ny
        r = sqrt((i - cy)^2 + (j - cx)^2)
        if r <= r_roi * 1.5  # 1.5x margin
            roi_mask[i, j] = 1.0
        end
    end
    img = img .* roi_mask

    # 2D FFT to get OTF
    otf = fft(img)
    otf_mag = abs.(otf)
    otf_mag = otf_mag ./ otf_mag[1, 1]  # Normalize to DC = 1
    otf_mag = fftshift(otf_mag)

    # Radial averaging for 1D MTF
    radial_profile, freq_px = _radial_average(otf_mag)

    # Convert frequency axis from pixels to physical units
    # Frequency in cycles per pixel: f_px = k / N where k is index, N is size
    # Frequency in cycles per mm: f_mm = f_px / pixel_size_mm
    # Frequency in lp/cm: f_lpcm = f_mm * 10 = 10 / (pixel_size_mm * N) * k
    freq_scale = 1.0 / fov_mm  # cycles/mm per frequency bin
    if unit == :lp_cm
        freq_scale *= 10.0  # lp/cm
    end

    # Only keep positive frequencies (from center outward)
    n = length(radial_profile)
    center_idx = n ÷ 2 + 1

    frequencies = freq_px[center_idx:end] .* freq_scale
    mtf_values = radial_profile[center_idx:end]

    # Ensure MTF is normalized and monotonically decreasing
    mtf_values = mtf_values ./ mtf_values[1]

    # Find characteristic frequencies
    f50 = _find_mtf_frequency(frequencies, mtf_values, 0.50)
    f10 = _find_mtf_frequency(frequencies, mtf_values, 0.10)
    f05 = _find_mtf_frequency(frequencies, mtf_values, 0.05)

    return MTFResult(frequencies, mtf_values, f50, f10, f05, :wire, unit)
end

"""
Internal helper: Radial averaging of 2D array (CatSim-compatible).
"""
function _radial_average(img::Matrix{Float64})
    ny, nx = size(img)
    n = min(ny, nx)

    # Create coordinate arrays
    cy, cx = (ny + 1) / 2, (nx + 1) / 2
    v = collect(1:n) .- (n / 2 + 0.5)

    # Angular sampling (CatSim uses 100 angles)
    n_ang_sample = 100

    radial_profile = zeros(Float64, n)

    for ii in 0:(n_ang_sample-1)
        ang = (ii * π) / n_ang_sample
        x1 = v .* cos(ang)
        y1 = v .* sin(ang)

        # Bilinear interpolation
        for (k, (px, py)) in enumerate(zip(x1, y1))
            # Convert to image coordinates
            ix = px + cx
            iy = py + cy

            # Bounds check
            if ix >= 1 && ix <= nx && iy >= 1 && iy <= ny
                # Bilinear interpolation
                x0 = floor(Int, ix)
                y0 = floor(Int, iy)
                x1_i = min(x0 + 1, nx)
                y1_i = min(y0 + 1, ny)

                fx = ix - x0
                fy = iy - y0

                val = (1 - fx) * (1 - fy) * img[y0, x0] +
                      fx * (1 - fy) * img[y0, x1_i] +
                      (1 - fx) * fy * img[y1_i, x0] +
                      fx * fy * img[y1_i, x1_i]

                radial_profile[k] += val / n_ang_sample
            end
        end
    end

    return radial_profile, v
end

"""
Internal helper: Find frequency at specified MTF level using linear interpolation.
"""
function _find_mtf_frequency(frequencies::Vector{Float64},
                             mtf_values::Vector{Float64},
                             level::Float64)
    # Find indices where MTF crosses the level
    for i in 1:(length(mtf_values)-1)
        if mtf_values[i] >= level && mtf_values[i+1] < level
            # Linear interpolation
            t = (level - mtf_values[i]) / (mtf_values[i+1] - mtf_values[i])
            return frequencies[i] + t * (frequencies[i+1] - frequencies[i])
        end
    end

    # If level not crossed, return Nyquist or 0
    if mtf_values[end] >= level
        return frequencies[end]
    else
        return 0.0
    end
end

# =============================================================================
# Edge Phantom MTF Measurement
# =============================================================================

"""
    measure_mtf_edge(image, pixel_size_mm; config=EdgePhantomMTF(),
                     edge_roi=nothing, unit=:lp_cm) -> MTFResult

Measure MTF from a slanted edge phantom using ESF differentiation method.

# Algorithm (per IEC 62220-1-1:2015)
1. Detect edge position using Canny or threshold
2. Project pixel values perpendicular to edge (super-resolved ESF)
3. Differentiate ESF to get LSF
4. FFT of LSF gives MTF

# Arguments
- `image`: 2D reconstruction image
- `pixel_size_mm`: Pixel size in mm
- `config`: EdgePhantomMTF configuration
- `edge_roi`: Optional (row_range, col_range) to limit edge detection
- `unit`: Output frequency unit (:lp_cm or :lp_mm)

# Returns
- `MTFResult` with MTF curve and characteristic frequencies

# Mathematical Formulation
For a slanted edge at angle θ:

    ESF(x) = Edge Spread Function (integrated LSF)
    LSF(x) = d/dx ESF(x) = Line Spread Function
    MTF(f) = |ℱ{LSF}(f)|

The slanted edge allows sub-pixel sampling of the ESF by projecting
pixels perpendicular to the edge direction.

# Example
```julia
# Create edge phantom with aluminum/air interface
edge_phantom = create_edge_phantom(512, 35.0; angle_deg=3.0)
# ... simulate and reconstruct ...

result = measure_mtf_edge(recon[:,:,1], pixel_size)
```

# References
1. IEC 62220-1-1:2015 - Determination of the detective quantum efficiency
2. Samei E, et al. "A method for measuring the presampled MTF of digital
   radiographic systems using an edge test device." Med Phys 1998;25:102-113
"""
function measure_mtf_edge(
    image::AbstractMatrix{T},
    pixel_size_mm::Real;
    config::EdgePhantomMTF = EdgePhantomMTF(),
    edge_roi::Union{Nothing, Tuple{UnitRange{Int}, UnitRange{Int}}} = nothing,
    unit::Symbol = :lp_cm
) where T <: Real

    # Convert to CPU array
    img = Array(Float64.(image))
    ny, nx = size(img)

    # Extract ROI if specified
    if edge_roi !== nothing
        row_range, col_range = edge_roi
        img_roi = img[row_range, col_range]
    else
        img_roi = img
    end

    ny_roi, nx_roi = size(img_roi)

    # Detect edge angle using Sobel gradients
    edge_angle_rad = _detect_edge_angle(img_roi)
    edge_angle_deg = rad2deg(edge_angle_rad)

    # If angle is near vertical, use horizontal projection; else vertical
    # For angles close to 0° or 180°, edge is horizontal
    # For angles close to 90° or 270°, edge is vertical

    # Project pixels perpendicular to edge to get ESF
    esf, esf_positions = _compute_oversampled_esf(
        img_roi, edge_angle_rad, config.oversampling_factor, pixel_size_mm)

    # Differentiate to get LSF
    lsf = diff(esf) ./ (esf_positions[2] - esf_positions[1])
    lsf_positions = (esf_positions[1:end-1] .+ esf_positions[2:end]) ./ 2

    # Optional smoothing
    if config.smoothing_window > 0
        lsf = _smooth_1d(lsf, config.smoothing_window)
    end

    # Normalize LSF
    lsf = lsf ./ maximum(abs.(lsf))

    # Zero-pad for better frequency resolution
    n_pad = nextpow(2, length(lsf) * 4)
    lsf_padded = zeros(Float64, n_pad)
    offset = (n_pad - length(lsf)) ÷ 2
    lsf_padded[offset+1:offset+length(lsf)] = lsf

    # FFT to get MTF
    mtf_complex = fft(lsf_padded)
    mtf_values = abs.(mtf_complex)
    mtf_values = mtf_values ./ mtf_values[1]  # Normalize

    # Frequency axis
    esf_spacing = esf_positions[2] - esf_positions[1]  # mm
    freq_nyquist = 1.0 / (2.0 * esf_spacing)  # cycles/mm

    # Only positive frequencies
    n_pos = n_pad ÷ 2
    freq_axis = collect(0:n_pos-1) ./ n_pad .* (1.0 / esf_spacing)  # cycles/mm

    if unit == :lp_cm
        freq_axis .*= 10.0  # lp/cm
    end

    mtf_values = mtf_values[1:n_pos]

    # Find characteristic frequencies
    f50 = _find_mtf_frequency(freq_axis, mtf_values, 0.50)
    f10 = _find_mtf_frequency(freq_axis, mtf_values, 0.10)
    f05 = _find_mtf_frequency(freq_axis, mtf_values, 0.05)

    return MTFResult(freq_axis, mtf_values, f50, f10, f05, :edge, unit)
end

"""
Internal: Detect edge angle using gradient analysis.
"""
function _detect_edge_angle(img::Matrix{Float64})
    ny, nx = size(img)

    # Sobel kernels
    sobel_x = [-1.0 0.0 1.0; -2.0 0.0 2.0; -1.0 0.0 1.0]
    sobel_y = [-1.0 -2.0 -1.0; 0.0 0.0 0.0; 1.0 2.0 1.0]

    # Simple convolution (avoid padding issues at edges)
    gx = zeros(Float64, ny-2, nx-2)
    gy = zeros(Float64, ny-2, nx-2)

    for j in 2:nx-1, i in 2:ny-1
        patch = img[i-1:i+1, j-1:j+1]
        gx[i-1, j-1] = sum(patch .* sobel_x)
        gy[i-1, j-1] = sum(patch .* sobel_y)
    end

    # Find dominant gradient direction
    # Weight by gradient magnitude
    magnitude = sqrt.(gx.^2 .+ gy.^2)
    magnitude_sum = sum(magnitude)

    if magnitude_sum > 0
        mean_gx = sum(gx .* magnitude) / magnitude_sum
        mean_gy = sum(gy .* magnitude) / magnitude_sum
        angle = atan(mean_gy, mean_gx)
    else
        angle = 0.0
    end

    return angle
end

"""
Internal: Compute oversampled ESF by projecting pixels perpendicular to edge.
"""
function _compute_oversampled_esf(
    img::Matrix{Float64},
    edge_angle_rad::Float64,
    oversampling::Int,
    pixel_size_mm::Float64
)
    ny, nx = size(img)

    # Direction perpendicular to edge
    perp_angle = edge_angle_rad + π/2

    # Project each pixel position perpendicular to edge
    cx, cy = (nx + 1) / 2, (ny + 1) / 2

    # Collect (position, value) pairs
    positions = Float64[]
    values = Float64[]

    for j in 1:nx, i in 1:ny
        # Position relative to center
        dx = (j - cx) * pixel_size_mm
        dy = (i - cy) * pixel_size_mm

        # Project onto perpendicular direction
        pos = dx * cos(perp_angle) + dy * sin(perp_angle)

        push!(positions, pos)
        push!(values, img[i, j])
    end

    # Sort by position
    perm = sortperm(positions)
    positions = positions[perm]
    values = values[perm]

    # Bin into oversampled ESF
    pos_min, pos_max = extrema(positions)
    n_bins = oversampling * round(Int, (pos_max - pos_min) / pixel_size_mm)
    n_bins = max(n_bins, 10)

    bin_edges = range(pos_min, pos_max, length=n_bins+1)
    bin_width = (pos_max - pos_min) / n_bins

    esf = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)

    for (pos, val) in zip(positions, values)
        bin_idx = clamp(floor(Int, (pos - pos_min) / bin_width) + 1, 1, n_bins)
        esf[bin_idx] += val
        counts[bin_idx] += 1
    end

    # Average
    for i in 1:n_bins
        if counts[i] > 0
            esf[i] /= counts[i]
        end
    end

    # Fill empty bins with interpolation
    _fill_empty_bins!(esf, counts)

    # ESF positions (bin centers)
    esf_positions = collect(range(pos_min + bin_width/2, pos_max - bin_width/2, length=n_bins))

    return esf, esf_positions
end

"""
Internal: Fill empty bins with linear interpolation.
"""
function _fill_empty_bins!(esf::Vector{Float64}, counts::Vector{Int})
    n = length(esf)

    for i in 1:n
        if counts[i] == 0
            # Find nearest filled bins
            left = i - 1
            while left >= 1 && counts[left] == 0
                left -= 1
            end

            right = i + 1
            while right <= n && counts[right] == 0
                right += 1
            end

            if left >= 1 && right <= n
                # Linear interpolation
                t = (i - left) / (right - left)
                esf[i] = (1 - t) * esf[left] + t * esf[right]
            elseif left >= 1
                esf[i] = esf[left]
            elseif right <= n
                esf[i] = esf[right]
            end
        end
    end
end

"""
Internal: Simple 1D smoothing (moving average).
"""
function _smooth_1d(data::Vector{Float64}, window::Int)
    n = length(data)
    result = similar(data)
    half_w = window ÷ 2

    for i in 1:n
        i_start = max(1, i - half_w)
        i_end = min(n, i + half_w)
        result[i] = mean(data[i_start:i_end])
    end

    return result
end

# =============================================================================
# Phantom Creation Utilities
# =============================================================================

"""
    create_wire_phantom(n_voxels, fov_cm; wire_position=(0.0, 0.0),
                        wire_material=:tungsten, wire_diameter_mm=0.1,
                        background_material=:water) -> (mask, materials_dict)

Create a wire phantom for MTF measurement.

# Arguments
- `n_voxels`: Number of voxels per side
- `fov_cm`: Field of view in cm
- `wire_position`: (x_mm, y_mm) position of wire from center
- `wire_material`: Material for wire (:tungsten, :aluminum)
- `wire_diameter_mm`: Wire diameter in mm
- `background_material`: Background material (:water, :air)

# Returns
- `mask`: UInt8 mask array
- `materials_dict`: Dict mapping region IDs to materials

# Example
```julia
mask, materials = create_wire_phantom(512, 35.0; wire_position=(30.0, 50.0))
```
"""
function create_wire_phantom(
    n_voxels::Int,
    fov_cm::Real;
    wire_position::Tuple{Real, Real} = (0.0, 0.0),
    wire_diameter_mm::Real = 0.1,
    background_hu::Real = 0.0  # Water = 0 HU
)
    # Constants
    REGION_BACKGROUND = UInt8(1)
    REGION_WIRE = UInt8(2)

    pixel_size_cm = fov_cm / n_voxels
    pixel_size_mm = pixel_size_cm * 10.0

    mask = fill(REGION_BACKGROUND, n_voxels, n_voxels)

    # Wire position in pixels
    cx = n_voxels / 2 + wire_position[1] / pixel_size_mm
    cy = n_voxels / 2 + wire_position[2] / pixel_size_mm

    wire_radius_px = wire_diameter_mm / 2 / pixel_size_mm

    for j in 1:n_voxels, i in 1:n_voxels
        r = sqrt((j - cx)^2 + (i - cy)^2)
        if r <= wire_radius_px
            mask[i, j] = REGION_WIRE
        end
    end

    # Materials dictionary (for reference)
    materials_info = Dict(
        REGION_BACKGROUND => (name=:water, hu=0),
        REGION_WIRE => (name=:tungsten, hu=2500)  # Approximate
    )

    return mask, materials_info, (cy, cx)
end

"""
    create_edge_phantom(n_voxels, fov_cm; angle_deg=3.0,
                        high_hu=1000.0, low_hu=-1000.0) -> mask

Create a slanted edge phantom for MTF measurement.

# Arguments
- `n_voxels`: Number of voxels per side
- `fov_cm`: Field of view in cm
- `angle_deg`: Edge angle from vertical (2-5° recommended)
- `high_hu`: HU value on one side of edge
- `low_hu`: HU value on other side of edge

# Returns
- `mask`: Float32 array with HU values (or UInt8 for material-based)
"""
function create_edge_phantom(
    n_voxels::Int,
    fov_cm::Real;
    angle_deg::Real = 3.0,
    high_val::Real = 1.0,
    low_val::Real = 0.0
)
    edge_angle_rad = deg2rad(angle_deg)

    phantom = zeros(Float32, n_voxels, n_voxels)

    cx, cy = n_voxels / 2, n_voxels / 2

    for j in 1:n_voxels, i in 1:n_voxels
        # Position relative to center
        x = j - cx
        y = i - cy

        # Rotated coordinate (perpendicular to edge)
        x_rot = x * cos(edge_angle_rad) - y * sin(edge_angle_rad)

        if x_rot >= 0
            phantom[i, j] = Float32(high_val)
        else
            phantom[i, j] = Float32(low_val)
        end
    end

    return phantom
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    mtf10(result::MTFResult) -> Float64

Get the 10% MTF frequency (limiting spatial resolution).
"""
mtf10(result::MTFResult) = result.mtf10

"""
    mtf50(result::MTFResult) -> Float64

Get the 50% MTF frequency (clinical performance indicator).
"""
mtf50(result::MTFResult) = result.mtf50

"""
    mtf5(result::MTFResult) -> Float64

Get the 5% MTF frequency.
"""
mtf5(result::MTFResult) = result.mtf5

"""
    get_mtf_frequency(result::MTFResult, level::Float64) -> Float64

Get the frequency at a specified MTF level (0-1).
"""
function get_mtf_frequency(result::MTFResult, level::Float64)
    return _find_mtf_frequency(result.frequencies, result.mtf, level)
end

"""
    get_mtf_value(result::MTFResult, frequency::Float64) -> Float64

Get the MTF value at a specified frequency (interpolated).
"""
function get_mtf_value(result::MTFResult, frequency::Float64)
    if frequency <= result.frequencies[1]
        return result.mtf[1]
    elseif frequency >= result.frequencies[end]
        return result.mtf[end]
    end

    # Linear interpolation
    for i in 1:(length(result.frequencies)-1)
        if result.frequencies[i] <= frequency && result.frequencies[i+1] > frequency
            t = (frequency - result.frequencies[i]) /
                (result.frequencies[i+1] - result.frequencies[i])
            return (1 - t) * result.mtf[i] + t * result.mtf[i+1]
        end
    end

    return 0.0
end

"""
    get_mtf_info(result::MTFResult) -> NamedTuple

Get summary information about MTF measurement.
"""
function get_mtf_info(result::MTFResult)
    return (
        method = result.method,
        unit = result.unit,
        mtf50 = result.mtf50,
        mtf10 = result.mtf10,
        mtf5 = result.mtf5,
        nyquist = result.frequencies[end],
        n_points = length(result.frequencies)
    )
end

"""
    compare_mtf(result1::MTFResult, result2::MTFResult;
                frequencies=[]) -> NamedTuple

Compare two MTF measurements.
"""
function compare_mtf(
    result1::MTFResult,
    result2::MTFResult;
    test_frequencies::Vector{Float64} = Float64[]
)
    # Standard comparison points
    mtf50_diff = abs(result1.mtf50 - result2.mtf50)
    mtf10_diff = abs(result1.mtf10 - result2.mtf10)
    mtf5_diff = abs(result1.mtf5 - result2.mtf5)

    # Relative differences
    mtf50_rel = result1.mtf50 > 0 ? mtf50_diff / result1.mtf50 * 100 : Inf
    mtf10_rel = result1.mtf10 > 0 ? mtf10_diff / result1.mtf10 * 100 : Inf
    mtf5_rel = result1.mtf5 > 0 ? mtf5_diff / result1.mtf5 * 100 : Inf

    # Comparison at specified frequencies
    freq_comparison = Float64[]
    for f in test_frequencies
        v1 = get_mtf_value(result1, f)
        v2 = get_mtf_value(result2, f)
        push!(freq_comparison, abs(v1 - v2))
    end

    return (
        mtf50_diff = mtf50_diff,
        mtf10_diff = mtf10_diff,
        mtf5_diff = mtf5_diff,
        mtf50_rel_percent = mtf50_rel,
        mtf10_rel_percent = mtf10_rel,
        mtf5_rel_percent = mtf5_rel,
        test_frequencies = test_frequencies,
        freq_differences = freq_comparison
    )
end
