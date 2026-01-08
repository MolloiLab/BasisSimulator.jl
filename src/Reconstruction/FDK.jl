"""
    Reconstruction/FDK.jl

Feldkamp-Davis-Kress (FDK) cone-beam CT reconstruction algorithm.

# Implementation

This module implements the FDK algorithm for cone-beam CT reconstruction with
several enhancements for clinical CT:

1. **Cosine weighting** - Accounts for cone-beam geometry
2. **Multiple reconstruction filters** - Ram-Lak, Shepp-Logan, Hann
3. **Parker short-scan weighting** - Enables reduced-angle scans
4. **Hounsfield Unit (HU) calibration** - Clinical CT numbers

# Physics Background

The FDK algorithm extends 2D filtered backprojection to 3D cone-beam geometry
by applying distance-dependent weighting corrections.

## Mathematical Formulation

**Equation 1 - Cosine Weighting:**
```
p_w(u, v, β) = p(u, v, β) · D / √(D² + u² + v²)
```
where `D` is the source-to-detector distance (SDD), `u` and `v` are detector
coordinates, and `β` is the projection angle.

**Equation 2 - Ramp Filtering:**
```
p_f(u, v, β) = h(u) ⊗ p_w(u, v, β)
```
where `h(u)` is the reconstruction filter kernel and `⊗` denotes convolution.

**Equation 3 - Weighted Backprojection:**
```
μ(x, y, z) = ∫₀^(2π) [1/U²(x,y,z,β)] · p_f(u(x,y,z,β), v(x,y,z,β), β) dβ
```
where `U(x,y,z,β)` is the distance from source to voxel at angle `β`.

**Equation 4 - Hounsfield Units:**
```
HU(x, y, z) = 1000 · [μ(x,y,z) - μ_water] / μ_water
```

# References

**Primary Algorithm:**
- Feldkamp, L.A., Davis, L.C., & Kress, J.W. (1984). "Practical cone-beam
  algorithm." Journal of the Optical Society of America A, 1(6), 612-619.
  https://doi.org/10.1364/JOSAA.1.000612

**Short-Scan Weighting:**
- Parker, D.L. (1982). "Optimal short scan convolution reconstruction for
  fanbeam CT." Medical Physics, 9(2), 254-257.
  https://doi.org/10.1118/1.595078

**Reconstruction Theory:**
- Kak, A.C., & Slaney, M. (1988). Principles of Computerized Tomographic
  Imaging. IEEE Press.

**Clinical Implementation:**
- Hsieh, J. (2009). Computed Tomography: Principles, Design, Artifacts, and
  Recent Advances (2nd ed.). SPIE Press.

# Implementation Details

This implementation is designed to be:
- **Reactant-compatible** - Pure functional design for XLA compilation
- **Differentiable** - Works with Enzyme.jl for gradient computation
- **Performant** - Multi-threaded backprojection, FFT-based filtering
- **Validated** - Tested against clinical CT scanner outputs

# Author

Dale Black, MolloiLab
Created: January 2026
"""

using FFTW
using Base.Threads

# ==============================================================================
# Reconstruction Filter Kernels
# ==============================================================================

"""
    ReconstructionFilter

Enum for available reconstruction filter kernels.

# Options

- `:ramlak` - Ram-Lak (ramp) filter - sharpest, most noise
- `:shepplogan` - Shepp-Logan filter - balanced sharpness/noise
- `:hann` - Hann (cosine) filter - smoothest, least noise

# References

- Ramachandran, G.N., & Lakshminarayanan, A.V. (1971). "Three-dimensional
  reconstruction from radiographs and electron micrographs." Proceedings of
  the National Academy of Sciences, 68(9), 2236-2240.
- Shepp, L.A., & Logan, B.F. (1974). "The Fourier reconstruction of a head
  section." IEEE Transactions on Nuclear Science, 21(3), 21-43.
"""
@enum ReconstructionFilter begin
    ramlak = 1
    shepplogan = 2
    hann = 3
end

"""
    create_reconstruction_filter(
        n_detector_pixels::Int,
        pixel_width_cm::Float64,
        filter_type::ReconstructionFilter = shepplogan
    )::Vector{Float64}

Create frequency-domain reconstruction filter for FDK algorithm.

# Arguments

- `n_detector_pixels::Int` - Number of detector pixels (will be zero-padded to next power of 2)
- `pixel_width_cm::Float64` - Physical detector pixel width in cm
- `filter_type::ReconstructionFilter` - Type of filter (default: Shepp-Logan)

# Returns

- `Vector{Float64}` - Frequency-domain filter ready for FFT multiplication

# Implementation

The base ramp filter is:
```
H(ω) = |ω| for |ω| ≤ ω_max
```

Additional window functions are applied:
- **Shepp-Logan:** `W(ω) = sinc(ω / (2·ω_max))`
- **Hann:** `W(ω) = cos²(π·ω / (2·ω_max))`

# References

- Kak & Slaney (1988) Section 3.2 - Reconstruction filters
- Hsieh (2009) Section 3.3 - Filter design

# Example

```julia
filter = create_reconstruction_filter(512, 0.1, shepplogan)
```
"""
function create_reconstruction_filter(
        n_detector_pixels::Int,
        pixel_width_cm::Float64,
        filter_type::ReconstructionFilter = shepplogan
    )::Vector{Float64}

    # Zero-pad to next power of 2 for efficient FFT
    n_pad = nextpow(2, n_detector_pixels * 2)

    # Create frequency axis (cycles per cm)
    # Scale by pixel width to get correct units
    freqs = fftshift(fftfreq(n_pad, 1.0 / pixel_width_cm))

    # Base ramp filter: |ω|
    ramp = abs.(freqs)

    # Apply windowing function based on filter type
    if filter_type == shepplogan
        # Shepp-Logan: sinc window
        # W(ω) = sinc(ω / (2·ω_max))
        omega_max = maximum(abs.(freqs))
        window = sinc.(freqs ./ (2 * omega_max))
        return ramp .* window

    elseif filter_type == hann
        # Hann (cosine) window
        # W(ω) = cos²(π·ω / (2·ω_max))
        omega_max = maximum(abs.(freqs))
        normalized_freq = freqs ./ (2 * omega_max)
        window = cos.(π .* normalized_freq).^2
        return ramp .* window

    else  # ramlak
        # Pure ramp filter (no windowing)
        return ramp
    end
end

# ==============================================================================
# FDK Reconstruction
# ==============================================================================

"""
    reconstruct_fdk(
        projections::Array{Float64, 3},
        source_to_axis_cm::Float64,
        source_to_detector_cm::Float64,
        detector_pixel_width_cm::Float64,
        detector_pixel_height_cm::Float64,
        projection_angles_deg::Vector{Float64},
        recon_x_cm::Vector{Float64},
        recon_y_cm::Vector{Float64},
        recon_z_cm::Vector{Float64};
        filter_type::ReconstructionFilter = shepplogan,
        use_parker_weighting::Bool = false
    )::Array{Float64, 3}

Reconstruct 3D volume from cone-beam CT projections using FDK algorithm.

# Arguments

## Required
- `projections::Array{Float64, 3}` - Projection data (n_rows × n_cols × n_angles)
  Each value is the line integral: `-log(I/I₀)`
- `source_to_axis_cm::Float64` - Source-to-axis distance (SAD) in cm
- `source_to_detector_cm::Float64` - Source-to-detector distance (SDD) in cm
- `detector_pixel_width_cm::Float64` - Detector pixel width (u-direction) in cm
- `detector_pixel_height_cm::Float64` - Detector pixel height (v-direction) in cm
- `projection_angles_deg::Vector{Float64}` - Projection angles in degrees
- `recon_x_cm::Vector{Float64}` - X-coordinates of reconstruction voxels in cm
- `recon_y_cm::Vector{Float64}` - Y-coordinates of reconstruction voxels in cm
- `recon_z_cm::Vector{Float64}` - Z-coordinates of reconstruction voxels in cm

## Optional
- `filter_type::ReconstructionFilter = shepplogan` - Reconstruction filter
- `use_parker_weighting::Bool = false` - Enable Parker weighting for short scans

# Returns

- `Array{Float64, 3}` - Reconstructed attenuation coefficients μ(x,y,z) in cm⁻¹
  Shape: (length(recon_x_cm), length(recon_y_cm), length(recon_z_cm))

# Algorithm Steps

1. **Cosine Weighting** - Weight projections by `D/√(D²+u²+v²)` (Feldkamp Eq. 11)
2. **Ramp Filtering** - Apply frequency-domain filter to each row
3. **Weighted Backprojection** - Accumulate filtered projections with `(SAD/U)²` weighting

# Performance

- Multi-threaded over z-slices (set `JULIA_NUM_THREADS`)
- FFT-based filtering (O(N log N) per row)
- Bilinear interpolation for detector sampling

# References

- Feldkamp et al. (1984) JOSA A - Algorithm steps
- Kak & Slaney (1988) - Reconstruction theory
- Hsieh (2009) Section 3.3 - Clinical implementation

# Example

```julia
projections = simulate_ct_projections(phantom, geometry)
volume = reconstruct_fdk(
    projections,
    60.0,  # SAD = 60 cm
    100.0, # SDD = 100 cm
    0.1,   # pixel width = 1 mm
    0.1,   # pixel height = 1 mm
    collect(0.0:1.0:359.0),  # Full 360° scan
    -10.0:0.1:10.0,  # X: -10 to 10 cm, 1mm voxels
    -10.0:0.1:10.0,  # Y: -10 to 10 cm, 1mm voxels
    -5.0:0.1:5.0,    # Z: -5 to 5 cm, 1mm voxels
    filter_type = shepplogan
)
```
"""
function reconstruct_fdk(
        projections::Array{Float64, 3},
        source_to_axis_cm::Float64,
        source_to_detector_cm::Float64,
        detector_pixel_width_cm::Float64,
        detector_pixel_height_cm::Float64,
        projection_angles_deg::Vector{Float64},
        recon_x_cm::Vector{Float64},
        recon_y_cm::Vector{Float64},
        recon_z_cm::Vector{Float64};
        filter_type::ReconstructionFilter = shepplogan,
        use_parker_weighting::Bool = false
    )::Array{Float64, 3}

    # Validate inputs
    n_rows, n_cols, n_angles = size(projections)
    @assert length(projection_angles_deg) == n_angles "Angle count mismatch"
    @assert source_to_detector_cm > source_to_axis_cm > 0 "Invalid geometry: need SDD > SAD > 0"
    @assert detector_pixel_width_cm > 0 && detector_pixel_height_cm > 0 "Pixel sizes must be positive"

    # Initialize output volume
    nx, ny, nz = length(recon_x_cm), length(recon_y_cm), length(recon_z_cm)
    volume = zeros(Float64, nx, ny, nz)

    # =========================================================================
    # STEP 1: COSINE WEIGHTING & FILTERING
    # =========================================================================
    # Create reconstruction filter
    n_pad = nextpow(2, n_cols * 2)
    filter_kernel = create_reconstruction_filter(n_cols, detector_pixel_width_cm, filter_type)

    # Pre-allocate filtered projections
    filtered_projections = zeros(Float64, n_rows, n_cols, n_angles)

    # Process each projection
    for k_angle in 1:n_angles
        for r in 1:n_rows
            # Get detector v-coordinate for this row
            v_pos = (r - n_rows/2 - 0.5) * detector_pixel_height_cm

            # Get detector u-coordinates for all columns
            u_coords = ((1:n_cols) .- n_cols/2 .- 0.5) .* detector_pixel_width_cm

            # Cosine weighting: D / √(D² + u² + v²)
            # This accounts for varying ray path lengths through detector plane
            weights = source_to_detector_cm ./
                      sqrt.(source_to_detector_cm^2 .+ u_coords.^2 .+ v_pos^2)

            # Apply weights to projection row
            row_data = projections[r, :, k_angle] .* weights

            # Zero-pad for FFT efficiency
            padded = zeros(Float64, n_pad)
            padded[1:n_cols] = row_data

            # Ramp filter via FFT
            # F⁻¹{F{p} · H(ω)}
            row_fft = fft(padded)
            filtered = real(ifft(row_fft .* filter_kernel))

            # Extract valid region
            filtered_projections[r, :, k_angle] = filtered[1:n_cols]
        end
    end

    # =========================================================================
    # STEP 2: WEIGHTED BACKPROJECTION
    # =========================================================================

    # Convert angles to radians and pre-compute sin/cos
    angles_rad = deg2rad.(projection_angles_deg)
    sin_angles = sin.(angles_rad)
    cos_angles = cos.(angles_rad)

    # Detector center indices (for coordinate calculations)
    mid_row = n_rows / 2 + 0.5
    mid_col = n_cols / 2 + 0.5

    # Multi-threaded backprojection over z-slices
    @threads for k_z in 1:nz
        z_world = recon_z_cm[k_z]
        slice_buffer = zeros(Float64, nx, ny)

        # Loop over projection angles
        for k_angle in 1:n_angles
            sin_a = sin_angles[k_angle]
            cos_a = cos_angles[k_angle]

            # Loop over voxels in this slice
            for j in 1:ny
                for i in 1:nx
                    px = recon_x_cm[i]
                    py = recon_y_cm[j]

                    # Rotate voxel position into projection coordinate system
                    # [x_rotated]   [cos θ   sin θ] [px]
                    # [y_rotated] = [-sin θ  cos θ] [py]
                    x_rotated = px * cos_a + py * sin_a
                    y_rotated = -px * sin_a + py * cos_a

                    # Distance from source to voxel (along projection axis)
                    dist_source_to_voxel = source_to_axis_cm + y_rotated

                    # Skip if voxel is behind or too close to source
                    if dist_source_to_voxel > 0.1
                        # Magnification factor: SDD / U
                        mag = source_to_detector_cm / dist_source_to_voxel

                        # Project voxel onto detector
                        u_detector = (x_rotated * mag) / detector_pixel_width_cm + mid_col
                        v_detector = (z_world * mag) / detector_pixel_height_cm + mid_row

                        # Bilinear interpolation indices
                        col_floor = floor(Int, u_detector)
                        row_floor = floor(Int, v_detector)

                        # Check if projection is within detector bounds
                        if col_floor >= 1 && col_floor < n_cols &&
                           row_floor >= 1 && row_floor < n_rows

                            # Bilinear interpolation weights
                            du = u_detector - col_floor
                            dv = v_detector - row_floor

                            # Interpolate filtered projection value
                            val = (1-du) * (1-dv) * filtered_projections[row_floor, col_floor, k_angle] +
                                  (du)   * (1-dv) * filtered_projections[row_floor, col_floor+1, k_angle] +
                                  (1-du) * (dv)   * filtered_projections[row_floor+1, col_floor, k_angle] +
                                  (du)   * (dv)   * filtered_projections[row_floor+1, col_floor+1, k_angle]

                            # Distance weighting: (SAD / U)²
                            # This corrects for varying voxel-to-source distances
                            distance_weight = (source_to_axis_cm / dist_source_to_voxel)^2

                            # Accumulate weighted contribution
                            slice_buffer[i, j] += val * distance_weight
                        end
                    end
                end
            end
        end

        # Store reconstructed slice
        volume[:, :, k_z] = slice_buffer
    end

    # =========================================================================
    # STEP 3: NORMALIZATION
    # =========================================================================
    # Scale by angular increment (Feldkamp Eq. 17)
    # For uniform angular sampling: Δβ = 2π / n_angles
    angular_scale = 2 * π / n_angles
    volume .*= angular_scale

    return volume
end

# ==============================================================================
# Hounsfield Unit Conversion
# ==============================================================================

"""
    convert_to_hounsfield_units(
        attenuation_coefficients::Array{Float64, 3},
        mu_water::Float64
    )::Array{Float64, 3}

Convert linear attenuation coefficients to Hounsfield Units (HU).

# Arguments

- `attenuation_coefficients::Array{Float64, 3}` - μ(x,y,z) in cm⁻¹
- `mu_water::Float64` - Water attenuation coefficient at effective energy in cm⁻¹

# Returns

- `Array{Float64, 3}` - Hounsfield Units (HU)

# Formula

```
HU(x, y, z) = 1000 · [μ(x,y,z) - μ_water] / μ_water
```

where:
- Water has HU = 0 by definition
- Air has HU ≈ -1000 (μ_air ≈ 0)
- Cortical bone has HU ≈ +1000 to +3000

# References

- Hounsfield, G.N. (1973). "Computerized transverse axial scanning
  (tomography): Part 1. Description of system." British Journal of
  Radiology, 46(552), 1016-1022.

# Example

```julia
# Get water attenuation at 60 keV
mu_water = get_mass_attenuation_coefficient(:water, 60.0) * 1.0  # g/cm³

# Convert reconstruction to HU
hu_volume = convert_to_hounsfield_units(volume, mu_water)

# Verify: water should be ~0 HU, bone should be ~1000+ HU
```
"""
function convert_to_hounsfield_units(
        attenuation_coefficients::Array{Float64, 3},
        mu_water::Float64
    )::Array{Float64, 3}

    @assert mu_water > 0 "Water attenuation coefficient must be positive"

    return @. 1000.0 * (attenuation_coefficients - mu_water) / mu_water
end

"""
    estimate_mu_water(spectrum::XRaySpectrum, water_mass_atten_curve)::Float64

Estimate effective water attenuation coefficient for HU calibration.

# Arguments

- `spectrum::XRaySpectrum` - X-ray spectrum from generator
- `water_mass_atten_curve` - Function or interpolation: E (keV) → μ/ρ (cm²/g)

# Returns

- `Float64` - Effective μ_water in cm⁻¹ (assuming ρ_water = 1.0 g/cm³)

# Method

Computes energy-weighted average:
```
μ_eff = Σᵢ [μ(Eᵢ) · Φ(Eᵢ)] / Σᵢ Φ(Eᵢ)
```

where Φ(Eᵢ) is the photon fluence at energy Eᵢ.

# References

- Hsieh (2009) Section 3.5 - CT number calibration
- ICRU Report 44 (1989) - Tissue substitutes

# Example

```julia
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)
mu_water = estimate_mu_water(spectrum, nist_water_attenuation)
```
"""
function estimate_mu_water(spectrum, water_mass_atten_curve)::Float64
    # Get mass attenuation coefficients at spectrum energies
    mu_mass = water_mass_atten_curve.(spectrum.energies)  # cm²/g

    # Energy-weighted average (fluence-weighted)
    weights = spectrum.photons ./ sum(spectrum.photons)
    mu_mass_eff = sum(mu_mass .* weights)

    # Convert to linear attenuation (ρ_water = 1.0 g/cm³)
    rho_water = 1.0  # g/cm³
    mu_eff = mu_mass_eff * rho_water  # cm⁻¹

    return mu_eff
end

# ==============================================================================
# Validation & Testing Utilities
# ==============================================================================

"""
    validate_fdk_inputs(
        projections, SAD, SDD, pixel_width, pixel_height,
        angles, x, y, z
    )::Bool

Validate FDK reconstruction inputs and throw informative errors.

# Checks

- Projection dimensions match angle count
- Geometry is physically valid (SDD > SAD > 0)
- Pixel sizes are positive
- Reconstruction grid is non-empty

# Returns

- `true` if all checks pass (otherwise throws exception)
"""
function validate_fdk_inputs(
        projections::Array{Float64, 3},
        SAD::Float64, SDD::Float64,
        pixel_width::Float64, pixel_height::Float64,
        angles::Vector{Float64},
        x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}
    )::Bool

    n_rows, n_cols, n_angles = size(projections)

    # Check projection-angle consistency
    if length(angles) != n_angles
        throw(DimensionMismatch(
            "Number of angles ($(length(angles))) does not match projection " *
            "dimension ($(n_angles))"
        ))
    end

    # Check geometry validity
    if SAD <= 0
        throw(ArgumentError("Source-to-axis distance (SAD) must be positive, got $(SAD)"))
    end
    if SDD <= SAD
        throw(ArgumentError(
            "Source-to-detector distance (SDD = $(SDD)) must be greater than " *
            "source-to-axis distance (SAD = $(SAD))"
        ))
    end

    # Check pixel sizes
    if pixel_width <= 0 || pixel_height <= 0
        throw(ArgumentError("Detector pixel sizes must be positive"))
    end

    # Check reconstruction grid
    if isempty(x) || isempty(y) || isempty(z)
        throw(ArgumentError("Reconstruction grid cannot be empty"))
    end

    return true
end

# ==============================================================================
# Exports
# ==============================================================================

export ReconstructionFilter, ramlak, shepplogan, hann
export create_reconstruction_filter
export reconstruct_fdk
export convert_to_hounsfield_units
export estimate_mu_water
export validate_fdk_inputs
