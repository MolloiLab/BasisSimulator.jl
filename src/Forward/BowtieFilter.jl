"""
    Forward/BowtieFilter.jl

Bowtie filter modeling for CT simulation.

The bowtie filter is placed between the X-ray source and patient to:
1. Reduce peripheral dose (thinner patient regions get less exposure)
2. Equalize signal across the detector (compensate for varying path lengths)
3. Reduce scatter by pre-hardening the beam at edges

The filter is thicker at the center and thinner at the edges, creating
an angle-dependent attenuation profile.
"""

using Statistics

# =============================================================================
# Bowtie Filter Types
# =============================================================================

"""
    BowtieFilter

Bowtie filter specification with angle-dependent thickness profile.

The filter attenuates the beam based on fan angle from central ray:
    I(θ) = I₀ × exp(-μ × t(θ))

where t(θ) is the filter thickness at angle θ.
"""
struct BowtieFilter
    # Filter material linear attenuation coefficient at reference energy (cm⁻¹)
    # Typical materials: aluminum (~0.5 at 60 keV), PMMA, Teflon
    μ_ref::Float64

    # Half-angles (degrees) at which thickness is defined
    angles::Vector{Float64}

    # Filter thickness (cm) at each angle
    thickness::Vector{Float64}

    # Filter name/type
    name::String
end

# =============================================================================
# Pre-defined Bowtie Filters
# =============================================================================

"""
    bowtie_filter_large_body()

Large body bowtie filter for adult abdomen/pelvis imaging.

Provides strong peripheral attenuation for large FOV scans.
Based on typical clinical scanner profiles.
"""
function bowtie_filter_large_body()
    # Aluminum-equivalent filter
    μ_al_60keV = 0.50  # cm⁻¹ at 60 keV

    # Angle profile (symmetric, degrees from center)
    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    # Thickness profile (cm) - thickest at center
    thickness = [2.5, 2.3, 1.8, 1.2, 0.6, 0.3, 0.1]

    return BowtieFilter(μ_al_60keV, angles, thickness, "large_body")
end

"""
    bowtie_filter_medium_body()

Medium body bowtie filter for average adult imaging.
"""
function bowtie_filter_medium_body()
    μ_al_60keV = 0.50

    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    thickness = [1.8, 1.6, 1.2, 0.8, 0.4, 0.2, 0.05]

    return BowtieFilter(μ_al_60keV, angles, thickness, "medium_body")
end

"""
    bowtie_filter_small_body()

Small body bowtie filter for pediatric or head imaging.
"""
function bowtie_filter_small_body()
    μ_al_60keV = 0.50

    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    thickness = [1.2, 1.0, 0.7, 0.4, 0.2, 0.1, 0.02]

    return BowtieFilter(μ_al_60keV, angles, thickness, "small_body")
end

"""
    bowtie_filter_head()

Head bowtie filter optimized for brain imaging.

Flatter profile since head is more circular.
"""
function bowtie_filter_head()
    μ_al_60keV = 0.50

    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    thickness = [0.8, 0.75, 0.6, 0.4, 0.25, 0.1, 0.02]

    return BowtieFilter(μ_al_60keV, angles, thickness, "head")
end

"""
    bowtie_filter_none()

No bowtie filter (flat field).

Use for testing or when bowtie is not desired.
"""
function bowtie_filter_none()
    angles = [0.0, 30.0]
    thickness = [0.0, 0.0]

    return BowtieFilter(0.0, angles, thickness, "none")
end

# =============================================================================
# Bowtie Attenuation Computation
# =============================================================================

"""
    get_bowtie_thickness(filter::BowtieFilter, fan_angle::Float64) -> Float64

Get filter thickness at given fan angle (degrees) by linear interpolation.

Fan angle is measured from the central ray (0 = center, positive = either side).
"""
function get_bowtie_thickness(filter::BowtieFilter, fan_angle::Float64)
    abs_angle = abs(fan_angle)

    # Handle out-of-range angles
    if abs_angle >= filter.angles[end]
        return filter.thickness[end]
    end

    # Linear interpolation
    for i in 1:(length(filter.angles)-1)
        if abs_angle <= filter.angles[i+1]
            # Interpolate between angles[i] and angles[i+1]
            t = (abs_angle - filter.angles[i]) / (filter.angles[i+1] - filter.angles[i])
            return filter.thickness[i] + t * (filter.thickness[i+1] - filter.thickness[i])
        end
    end

    return filter.thickness[end]
end

"""
    compute_bowtie_attenuation(filter::BowtieFilter, geom::CTGeometry) -> Array{Float64,2}

Compute bowtie filter attenuation factors for all detector pixels.

Returns a 2D array [n_cols, n_rows] of transmission factors (0 to 1).
These factors multiply the incident flux before projection through the patient.

# Arguments
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry

# Returns
Array of transmission factors to multiply with sinogram.
"""
function compute_bowtie_attenuation(filter::BowtieFilter, geom::CTGeometry)
    n_cols = geom.n_cols
    n_rows = geom.n_rows

    # Compute fan angle for each detector column
    # Fan angle = atan(detector_u_offset / SAD)
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    transmission = zeros(Float64, n_cols, n_rows)

    for col in 1:n_cols
        # Detector offset from center (in cm at detector plane)
        u_offset = (col - (n_cols + 1) / 2) * pixel_size_det

        # Fan angle in degrees
        fan_angle = atand(u_offset / geom.SDD)

        # Get filter thickness at this angle
        thickness = get_bowtie_thickness(filter, fan_angle)

        # Compute transmission: T = exp(-μ × t)
        trans = exp(-filter.μ_ref * thickness)

        # Same transmission for all rows (bowtie is 1D in fan direction)
        for row in 1:n_rows
            transmission[col, row] = trans
        end
    end

    return transmission
end

"""
    apply_bowtie_filter(sinogram, filter::BowtieFilter, geom::CTGeometry) -> Array

Apply bowtie filter attenuation to sinogram.

The bowtie filter reduces intensity at peripheral detector positions,
which affects both the signal level and noise characteristics.

This should be applied BEFORE adding noise (in intensity domain) or
can be applied as an additive correction in projection domain.

# Arguments
- `sinogram`: Sinogram in projection (line-integral) domain [n_cols, n_rows, n_angles]
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry

# Returns
Sinogram with bowtie filter effect added.
"""
function apply_bowtie_filter(
    sinogram::AbstractArray{T,3},
    filter::BowtieFilter,
    geom::CTGeometry
) where T
    # Skip if no filter
    if filter.name == "none"
        return sinogram
    end

    # Compute bowtie transmission
    transmission = compute_bowtie_attenuation(filter, geom)

    # In projection domain, bowtie adds to the line integral:
    # p_total = p_patient + p_bowtie
    # where p_bowtie = -log(transmission) = μ × t
    bowtie_projection = -log.(transmission)

    # Add bowtie attenuation to each angle
    n_angles = size(sinogram, 3)
    result = similar(sinogram)

    for angle in 1:n_angles
        result[:, :, angle] = sinogram[:, :, angle] .+ T.(bowtie_projection)
    end

    return result
end

"""
    apply_bowtie_to_intensity(intensity, filter::BowtieFilter, geom::CTGeometry) -> Array

Apply bowtie filter to intensity-domain data (before log transform).

This is the physically correct approach: bowtie attenuates the beam
before it reaches the patient.

# Arguments
- `intensity`: Intensity data [n_cols, n_rows, n_angles] (pre-log)
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry

# Returns
Attenuated intensity data.
"""
function apply_bowtie_to_intensity(
    intensity::AbstractArray{T,3},
    filter::BowtieFilter,
    geom::CTGeometry
) where T
    if filter.name == "none"
        return intensity
    end

    transmission = T.(compute_bowtie_attenuation(filter, geom))

    # Multiply intensity by transmission
    n_angles = size(intensity, 3)
    result = similar(intensity)

    for angle in 1:n_angles
        result[:, :, angle] = intensity[:, :, angle] .* transmission
    end

    return result
end

"""
    get_bowtie_profile(filter::BowtieFilter, geom::CTGeometry) -> Vector{Float64}

Get the 1D bowtie transmission profile across detector columns.

Useful for visualization and debugging.
"""
function get_bowtie_profile(filter::BowtieFilter, geom::CTGeometry)
    transmission = compute_bowtie_attenuation(filter, geom)
    return transmission[:, 1]  # Same for all rows
end

# =============================================================================
# Exports
# =============================================================================

export BowtieFilter
export bowtie_filter_large_body, bowtie_filter_medium_body
export bowtie_filter_small_body, bowtie_filter_head, bowtie_filter_none
export get_bowtie_thickness, compute_bowtie_attenuation
export apply_bowtie_filter, apply_bowtie_to_intensity, get_bowtie_profile
