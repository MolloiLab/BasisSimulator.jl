"""
    Forward/BowtieFilter.jl

Bowtie filter modeling for CT simulation.

The bowtie filter is placed between the X-ray source and patient to:
1. Reduce peripheral dose (thinner patient regions get less exposure)
2. Equalize signal across the detector (compensate for varying path lengths)
3. Reduce scatter by pre-hardening the beam at edges

The filter is thicker at the center and thinner at the edges, creating
an angle-dependent attenuation profile.

Implementation follows CatSim/XCIST approach:
- Multi-material support (Al, graphite, Cu, Ti)
- Energy-dependent attenuation μ(E)
- Cone angle geometric correction
- File-based profile loading

GPU-native implementation using AcceleratedKernels.jl.
"""

import AcceleratedKernels as AK
using Statistics
using DelimitedFiles

# =============================================================================
# Bowtie Filter Types
# =============================================================================

"""
    BowtieFilter

Bowtie filter specification with angle-dependent thickness profile.

The filter attenuates the beam based on fan angle from central ray:
    I(θ, E) = I₀(E) × exp(-Σᵢ μᵢ(E) × tᵢ(θ))

where tᵢ(θ) is the thickness of material i at angle θ.

# Fields
- `angles`: Fan angles (radians) at which thickness is defined
- `thickness`: Matrix of thicknesses [n_angles, n_materials] in cm
- `materials`: Vector of material names (for μ lookup)
- `name`: Filter name/type
"""
struct BowtieFilter
    angles::Vector{Float64}           # Fan angles in radians
    thickness::Matrix{Float64}        # [n_angles, n_materials] in cm
    materials::Vector{String}         # Material names
    name::String
end

# =============================================================================
# Pre-defined Bowtie Filters (CatSim-style)
# =============================================================================

"""
    bowtie_filter_large_body()

Large body bowtie filter for adult abdomen/pelvis imaging.

Uses aluminum as primary material with profile based on typical clinical scanners.
"""
function bowtie_filter_large_body()
    # Angles in radians (symmetric profile, 0 = center)
    angles_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    angles = deg2rad.(angles_deg)

    # Aluminum thickness profile (cm) - thickest at center
    # This creates the characteristic "bowtie" attenuation pattern
    al_thickness = [2.5, 2.3, 1.8, 1.2, 0.6, 0.3, 0.1]

    # Single material for simplicity (aluminum equivalent)
    thickness = reshape(al_thickness, :, 1)
    materials = ["Al"]

    return BowtieFilter(angles, thickness, materials, "large_body")
end

"""
    bowtie_filter_medium_body()

Medium body bowtie filter for average adult imaging.
"""
function bowtie_filter_medium_body()
    angles_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    angles = deg2rad.(angles_deg)

    al_thickness = [1.8, 1.6, 1.2, 0.8, 0.4, 0.2, 0.05]
    thickness = reshape(al_thickness, :, 1)
    materials = ["Al"]

    return BowtieFilter(angles, thickness, materials, "medium_body")
end

"""
    bowtie_filter_small_body()

Small body bowtie filter for pediatric imaging.
"""
function bowtie_filter_small_body()
    angles_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    angles = deg2rad.(angles_deg)

    al_thickness = [1.2, 1.0, 0.7, 0.4, 0.2, 0.1, 0.02]
    thickness = reshape(al_thickness, :, 1)
    materials = ["Al"]

    return BowtieFilter(angles, thickness, materials, "small_body")
end

"""
    bowtie_filter_head()

Head bowtie filter optimized for brain imaging.

Flatter profile since head is more circular.
"""
function bowtie_filter_head()
    angles_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    angles = deg2rad.(angles_deg)

    al_thickness = [0.8, 0.75, 0.6, 0.4, 0.25, 0.1, 0.02]
    thickness = reshape(al_thickness, :, 1)
    materials = ["Al"]

    return BowtieFilter(angles, thickness, materials, "head")
end

"""
    bowtie_filter_none()

No bowtie filter (flat field).
"""
function bowtie_filter_none()
    angles = [0.0, 0.5]  # radians
    thickness = zeros(2, 1)
    materials = ["Al"]

    return BowtieFilter(angles, thickness, materials, "none")
end

"""
    bowtie_filter_multimaterial()

Multi-material bowtie filter following CatSim convention.

Uses Al, graphite, Cu, Ti for more realistic spectral shaping.
"""
function bowtie_filter_multimaterial()
    angles_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    angles = deg2rad.(angles_deg)

    # Thickness profiles for each material (cm)
    # Al - primary structural material
    al = [1.5, 1.3, 1.0, 0.6, 0.3, 0.15, 0.05]
    # Graphite - low-Z for beam hardening
    graphite = [0.3, 0.25, 0.2, 0.12, 0.06, 0.03, 0.01]
    # Cu - high-Z for additional hardening at center
    cu = [0.02, 0.015, 0.01, 0.005, 0.002, 0.001, 0.0]
    # Ti - intermediate Z
    ti = [0.05, 0.04, 0.03, 0.015, 0.008, 0.004, 0.001]

    thickness = hcat(al, graphite, cu, ti)
    materials = ["Al", "graphite", "Cu", "Ti"]

    return BowtieFilter(angles, thickness, materials, "multimaterial")
end

# =============================================================================
# File-based Bowtie Loading (CatSim compatible)
# =============================================================================

"""
    load_bowtie_filter(filepath::String; name::String="custom") -> BowtieFilter

Load bowtie filter from a text file (CatSim format).

File format:
- Lines starting with # or % are comments
- First column: fan angle (degrees)
- Subsequent columns: thickness (mm) for each material
- Materials assumed: Al, graphite, Cu, Ti (in that order)

# Example file content:
```
# angle(deg)  Al(mm)  graphite(mm)  Cu(mm)  Ti(mm)
0.0    15.0    3.0    0.2    0.5
5.0    13.0    2.5    0.15   0.4
10.0   10.0    2.0    0.1    0.3
...
```
"""
function load_bowtie_filter(filepath::String; name::String="custom")
    # Read file, skipping comment lines
    data = readdlm(filepath, comments=true, comment_char='#')

    # First column is angles (degrees), rest are thicknesses (mm)
    angles_deg = Float64.(data[:, 1])
    angles = deg2rad.(angles_deg)

    # Convert thickness from mm to cm
    thickness_mm = Float64.(data[:, 2:end])
    thickness = thickness_mm ./ 10.0

    # Determine materials based on number of columns
    n_materials = size(thickness, 2)
    if n_materials >= 4
        materials = ["Al", "graphite", "Cu", "Ti"]
    elseif n_materials == 3
        materials = ["Al", "graphite", "Cu"]
    elseif n_materials == 2
        materials = ["Al", "graphite"]
    else
        materials = ["Al"]
    end

    return BowtieFilter(angles, thickness, materials[1:n_materials], name)
end

# =============================================================================
# Material Attenuation Coefficients
# =============================================================================

# NIST XCOM-based linear attenuation coefficients (cm⁻¹)
# Data points for interpolation: (energy_keV, μ in cm⁻¹)
const BOWTIE_MU_DATA = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}(
    # Aluminum (ρ = 2.70 g/cm³)
    # Data from NIST XCOM
    "Al" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [3.44, 1.13, 0.75, 0.63, 0.61, 0.55, 0.51, 0.49, 0.46]
    ),
    # Graphite/Carbon (ρ = 1.70 g/cm³ for graphite)
    "graphite" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [0.56, 0.31, 0.27, 0.26, 0.26, 0.25, 0.24, 0.24, 0.23]
    ),
    # Copper (ρ = 8.96 g/cm³)
    # K-edge at 8.98 keV
    "Cu" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [94.5, 31.9, 14.5, 7.77, 4.67, 2.14, 1.22, 0.81, 0.52]
    ),
    # Titanium (ρ = 4.51 g/cm³)
    # K-edge at 4.97 keV
    "Ti" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [17.0, 5.75, 2.74, 1.56, 1.02, 0.54, 0.37, 0.30, 0.25]
    )
)

# Alias for carbon
BOWTIE_MU_DATA["C"] = BOWTIE_MU_DATA["graphite"]

"""
    get_bowtie_mu(material::String, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for bowtie material at given energy.

Returns μ in cm⁻¹.

Uses NIST XCOM-based lookup tables with log-linear interpolation.
"""
function get_bowtie_mu(material::String, energy_keV::Float64)
    # Get data for material (default to Al if unknown)
    if !haskey(BOWTIE_MU_DATA, material)
        material = "Al"
    end

    energies, mus = BOWTIE_MU_DATA[material]
    E = clamp(energy_keV, energies[1], energies[end])

    # Log-linear interpolation (μ varies roughly linearly with log(E) in diagnostic range)
    log_E = log(E)
    log_energies = log.(energies)
    log_mus = log.(mus)

    # Find interpolation interval
    idx = 1
    for i in 1:(length(energies)-1)
        if log_E >= log_energies[i] && log_E <= log_energies[i+1]
            idx = i
            break
        end
    end

    # Linear interpolation in log space
    t = (log_E - log_energies[idx]) / (log_energies[idx+1] - log_energies[idx])
    log_mu = log_mus[idx] + t * (log_mus[idx+1] - log_mus[idx])

    return exp(log_mu)
end

"""
    get_bowtie_mu_reference(material::String) -> Float64

Get reference linear attenuation coefficient at 60 keV.
"""
function get_bowtie_mu_reference(material::String)
    return get_bowtie_mu(material, 60.0)
end

# =============================================================================
# Bowtie Attenuation Computation
# =============================================================================

"""
    interpolate_thickness(filter::BowtieFilter, fan_angle::Float64) -> Vector{Float64}

Interpolate filter thickness at given fan angle for all materials.

Fan angle in radians, measured from central ray (0 = center).
Returns thickness vector in cm for each material.
"""
function interpolate_thickness(filter::BowtieFilter, fan_angle::Float64)
    abs_angle = abs(fan_angle)
    n_materials = size(filter.thickness, 2)

    # Handle out-of-range angles
    if abs_angle >= filter.angles[end]
        return filter.thickness[end, :]
    end

    # Linear interpolation
    for i in 1:(length(filter.angles)-1)
        if abs_angle <= filter.angles[i+1]
            t = (abs_angle - filter.angles[i]) / (filter.angles[i+1] - filter.angles[i])
            return filter.thickness[i, :] .+ t .* (filter.thickness[i+1, :] .- filter.thickness[i, :])
        end
    end

    return filter.thickness[end, :]
end

# Legacy function for backward compatibility
function get_bowtie_thickness(filter::BowtieFilter, fan_angle_deg::Float64)
    thickness_vec = interpolate_thickness(filter, deg2rad(fan_angle_deg))
    return sum(thickness_vec)  # Return total equivalent thickness
end

"""
    compute_bowtie_attenuation(filter::BowtieFilter, geom::CTGeometry;
                               energy_keV::Float64=60.0) -> Array{Float64,2}

Compute bowtie filter attenuation factors for all detector pixels.

Returns a 2D array [n_cols, n_rows] of transmission factors (0 to 1).

# Arguments
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry
- `energy_keV`: Reference energy for μ calculation (default: 60 keV)

# Returns
Array of transmission factors.
"""
function compute_bowtie_attenuation(
    filter::BowtieFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows

    # Get μ for each material at reference energy
    μ_vec = [get_bowtie_mu(mat, energy_keV) for mat in filter.materials]

    # Compute detector pixel positions
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    transmission = zeros(Float64, n_cols, n_rows)

    for row in 1:n_rows
        # Cone angle for this row (alpha in CatSim)
        v_offset = (row - (n_rows + 1) / 2) * pixel_size_det
        cone_angle = atan(v_offset / geom.SDD)
        cos_alpha = cos(cone_angle)

        for col in 1:n_cols
            # Fan angle for this column (gamma in CatSim)
            u_offset = (col - (n_cols + 1) / 2) * pixel_size_det
            fan_angle = atan(u_offset / geom.SDD)

            # Get thickness at this fan angle for all materials
            thickness_vec = interpolate_thickness(filter, fan_angle)

            # Apply cone angle correction (path length increases with cone angle)
            # This follows CatSim: t_corrected = t / cos(alpha)
            thickness_corrected = thickness_vec ./ cos_alpha

            # Compute total attenuation: exp(-Σ μᵢ × tᵢ)
            μt_total = sum(μ_vec .* thickness_corrected)
            transmission[col, row] = exp(-μt_total)
        end
    end

    return transmission
end

"""
    compute_bowtie_attenuation_spectral(filter::BowtieFilter, geom::CTGeometry,
                                        energies::Vector{Float64}) -> Array{Float64,3}

Compute energy-dependent bowtie attenuation for polychromatic simulation.

Returns [n_cols, n_rows, n_energies] transmission factors.

# Arguments
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry
- `energies`: Vector of energies in keV

# Returns
3D array of transmission factors per energy bin.
"""
function compute_bowtie_attenuation_spectral(
    filter::BowtieFilter,
    geom::CTGeometry,
    energies::Vector{Float64}
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_energies = length(energies)

    # Compute μ for each material at each energy
    n_materials = length(filter.materials)
    μ_matrix = zeros(n_materials, n_energies)
    for (i, mat) in enumerate(filter.materials)
        for (j, E) in enumerate(energies)
            μ_matrix[i, j] = get_bowtie_mu(mat, E)
        end
    end

    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    transmission = zeros(Float64, n_cols, n_rows, n_energies)

    for row in 1:n_rows
        v_offset = (row - (n_rows + 1) / 2) * pixel_size_det
        cone_angle = atan(v_offset / geom.SDD)
        cos_alpha = cos(cone_angle)

        for col in 1:n_cols
            u_offset = (col - (n_cols + 1) / 2) * pixel_size_det
            fan_angle = atan(u_offset / geom.SDD)

            thickness_vec = interpolate_thickness(filter, fan_angle)
            thickness_corrected = thickness_vec ./ cos_alpha

            # Compute transmission at each energy
            for k in 1:n_energies
                μt_total = sum(μ_matrix[:, k] .* thickness_corrected)
                transmission[col, row, k] = exp(-μt_total)
            end
        end
    end

    return transmission
end

"""
    apply_bowtie_filter!(sinogram, filter::BowtieFilter, geom::CTGeometry;
                         energy_keV::Float64=60.0) -> sinogram

Apply bowtie filter attenuation to sinogram (in-place, GPU-native).

In projection domain, bowtie adds to the line integral:
    p_total = p_patient + p_bowtie

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles]
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry
- `energy_keV`: Reference energy (default: 60 keV)

# Returns
Modified sinogram with bowtie filter effect added.
"""
function apply_bowtie_filter!(
    sinogram::AbstractArray{T,3},
    filter::BowtieFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    # Skip if no filter
    if filter.name == "none"
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Compute bowtie transmission on CPU (done once)
    transmission_cpu = compute_bowtie_attenuation(filter, geom; energy_keV=energy_keV)

    # In projection domain: p_bowtie = -log(transmission)
    bowtie_projection_cpu = T.(-log.(transmission_cpu))

    # Transfer to GPU (same type as sinogram)
    bowtie_projection = similar(sinogram, n_cols, n_rows)
    copyto!(bowtie_projection, bowtie_projection_cpu)

    # GPU-native element-wise operation
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, _ = Tuple(ci)
        proj_idx = col + (row - 1) * n_cols
        sinogram[idx] += bowtie_projection[proj_idx]
    end

    return sinogram
end

"""
    apply_bowtie_to_intensity!(intensity, filter::BowtieFilter, geom::CTGeometry;
                               energy_keV::Float64=60.0) -> intensity

Apply bowtie filter to intensity-domain data (in-place, GPU-native).

This is the physically correct approach: bowtie attenuates the beam
before it reaches the patient.

# Arguments
- `intensity`: Intensity data [n_cols, n_rows, n_angles] (pre-log)
- `filter::BowtieFilter`: Bowtie filter specification
- `geom::CTGeometry`: Scanner geometry
- `energy_keV`: Reference energy (default: 60 keV)

# Returns
Modified attenuated intensity data.
"""
function apply_bowtie_to_intensity!(
    intensity::AbstractArray{T,3},
    filter::BowtieFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    if filter.name == "none"
        return intensity
    end

    n_cols = size(intensity, 1)
    n_rows = size(intensity, 2)

    # Compute transmission on CPU (done once)
    transmission_cpu = T.(compute_bowtie_attenuation(filter, geom; energy_keV=energy_keV))

    # Transfer to GPU (same type as intensity)
    transmission = similar(intensity, n_cols, n_rows)
    copyto!(transmission, transmission_cpu)

    # GPU-native element-wise operation
    AK.foreachindex(intensity) do idx
        ci = CartesianIndices(intensity)[idx]
        col, row, _ = Tuple(ci)
        trans_idx = col + (row - 1) * n_cols
        intensity[idx] *= transmission[trans_idx]
    end

    return intensity
end

# Convenience wrappers that allocate (for backward compatibility during transition)
function apply_bowtie_filter(
    sinogram::AbstractArray{T,3},
    filter::BowtieFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    result = copy(sinogram)
    return apply_bowtie_filter!(result, filter, geom; energy_keV=energy_keV)
end

function apply_bowtie_to_intensity(
    intensity::AbstractArray{T,3},
    filter::BowtieFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    result = copy(intensity)
    return apply_bowtie_to_intensity!(result, filter, geom; energy_keV=energy_keV)
end

"""
    get_bowtie_profile(filter::BowtieFilter, geom::CTGeometry;
                       energy_keV::Float64=60.0) -> Vector{Float64}

Get the 1D bowtie transmission profile across detector columns.

Returns transmission at the central row.
"""
function get_bowtie_profile(filter::BowtieFilter, geom::CTGeometry;
                            energy_keV::Float64=60.0)
    transmission = compute_bowtie_attenuation(filter, geom; energy_keV=energy_keV)
    mid_row = geom.n_rows ÷ 2 + 1
    return transmission[:, mid_row]
end

"""
    get_bowtie_info(filter::BowtieFilter) -> NamedTuple

Get diagnostic information about bowtie filter.
"""
function get_bowtie_info(filter::BowtieFilter)
    max_thickness = maximum(sum(filter.thickness, dims=2))
    min_thickness = minimum(sum(filter.thickness, dims=2))

    return (
        name = filter.name,
        n_materials = length(filter.materials),
        materials = filter.materials,
        angle_range_deg = (rad2deg(filter.angles[1]), rad2deg(filter.angles[end])),
        thickness_range_cm = (min_thickness, max_thickness),
        n_angles = length(filter.angles)
    )
end

# =============================================================================
# Exports
# =============================================================================

export BowtieFilter
export bowtie_filter_large_body, bowtie_filter_medium_body
export bowtie_filter_small_body, bowtie_filter_head, bowtie_filter_none
export bowtie_filter_multimaterial, load_bowtie_filter
export get_bowtie_thickness, interpolate_thickness
export compute_bowtie_attenuation, compute_bowtie_attenuation_spectral
export apply_bowtie_filter!, apply_bowtie_to_intensity!
export apply_bowtie_filter, apply_bowtie_to_intensity
export get_bowtie_profile, get_bowtie_info
export get_bowtie_mu, get_bowtie_mu_reference
