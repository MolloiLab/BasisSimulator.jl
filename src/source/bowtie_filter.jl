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

Spectral-domain bowtie attenuation is applied during spectrum weighting,
not as a sinogram-domain post-processing step.
"""

using DelimitedFiles

# =============================================================================
# Built-in Bowtie Data Directory
# =============================================================================

"""Directory containing built-in bowtie filter data (CatSim/XCIST format)."""
const BOWTIE_DIR = joinpath(@__DIR__, "..", "bowtie")

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

# =============================================================================
# Built-in Bowtie Loader (CatSim/XCIST data bundled with package)
# =============================================================================

"""
    load_builtin_bowtie(size::String; bowtie_dir::AbstractString=BOWTIE_DIR) -> BowtieFilter

Load a built-in bowtie filter profile from the bundled CatSim/XCIST data.

The package ships with GE Revolution bowtie profiles (889 data points each,
4-material columns) from the CatSim/XCIST open-source CT simulator.

# Arguments
- `size::String`: Filter size — `"large"`, `"medium"`, or `"small"`
- `bowtie_dir::AbstractString`: Directory containing bowtie data files (default: bundled data)

# Returns
`BowtieFilter` with 889-point asymmetric profile.

# Example
```julia
filter = load_builtin_bowtie("large")
length(filter.angles)  # 889
```
"""
function load_builtin_bowtie(size::String; bowtie_dir::AbstractString=BOWTIE_DIR)
    filepath = joinpath(bowtie_dir, "$(size).txt")
    isfile(filepath) || error("Built-in bowtie file not found: $filepath")
    return load_catsim_bowtie(filepath; name="catsim_$(size)")
end

# =============================================================================
# GE Revolution Apex Bowtie Filters (from CatSim/XCIST)
#
# These are the actual GE Revolution bowtie profiles from CatSim/XCIST:
# 889 data points, asymmetric, 4-material columns (Al, graphite, Cu, Ti).
# Licensed BSD 3-Clause by GE Precision HealthCare.
# =============================================================================

"""
    ge_revolution_bowtie_large() -> BowtieFilter

GE Revolution large body bowtie filter (889-point CatSim/XCIST profile).

This filter is used for adult body imaging (abdomen, pelvis, chest).
Data from GE Precision HealthCare via CatSim/XCIST (BSD 3-Clause).
"""
ge_revolution_bowtie_large() = load_builtin_bowtie("large")

"""
    ge_revolution_bowtie_medium() -> BowtieFilter

GE Revolution medium body bowtie filter (889-point CatSim/XCIST profile).

This filter is used for head imaging and medium-sized patients.
Data from GE Precision HealthCare via CatSim/XCIST (BSD 3-Clause).
"""
ge_revolution_bowtie_medium() = load_builtin_bowtie("medium")

"""
    ge_revolution_bowtie_small() -> BowtieFilter

GE Revolution small body bowtie filter (889-point CatSim/XCIST profile).

This filter is used for pediatric and small patient imaging.
Data from GE Precision HealthCare via CatSim/XCIST (BSD 3-Clause).
"""
ge_revolution_bowtie_small() = load_builtin_bowtie("small")

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
# Boone Super-Gaussian Analytical Model
# =============================================================================

"""
    bowtie_filter_supergaussian(; kwargs...) -> BowtieFilter

Generate a bowtie filter profile using the Boone super-Gaussian analytical model.

The super-Gaussian provides a flexible parametric model for bowtie filters
with only 4 shape parameters:

    t(θ) = t_center + Δt × (1 - exp(-0.5 × (|θ|/σ)^n))

where θ is the fan angle, σ controls the width, and n controls the shape.

# Keyword Arguments
- `t_center::Float64 = 0.1`: Minimum thickness at center (cm)
- `delta_t::Float64 = 3.0`: Additional thickness at field edge (cm)
- `sigma_deg::Float64 = 15.0`: Angular width parameter (degrees)
- `n::Float64 = 4.0`: Shape exponent
  - `n=2`: Gaussian rolloff (head, round cross-section)
  - `n=4-6`: Flat center plateau with gradual rise (body imaging)
  - `n=8-10`: Nearly flat-top, aggressive peripheral filtering
- `material::String = "Al"`: Filter material
- `n_angles::Int = 100`: Number of angle sample points (positive half)
- `max_angle_deg::Float64 = 30.0`: Maximum fan angle (degrees)
- `name::String = "supergaussian"`: Filter name

# Returns
`BowtieFilter` with symmetric profile (positive angles only, 0 to max).

# Reference
Boone JM. "Method for evaluating bow tie filter angle-dependent attenuation
in CT: Theory and simulation results." Med Phys. 2010;37(1):40-48.

# Example
```julia
# Body filter: flat center, gradual rise
body = bowtie_filter_supergaussian(t_center=0.1, delta_t=3.0, sigma_deg=15.0, n=4.0)

# Head filter: Gaussian rolloff
head = bowtie_filter_supergaussian(t_center=0.05, delta_t=1.5, sigma_deg=12.0, n=2.0)

# Aggressive peripheral filter: flat-top
flat = bowtie_filter_supergaussian(t_center=0.1, delta_t=4.0, sigma_deg=10.0, n=10.0)
```
"""
function bowtie_filter_supergaussian(;
    t_center::Float64 = 0.1,
    delta_t::Float64 = 3.0,
    sigma_deg::Float64 = 15.0,
    n::Float64 = 4.0,
    material::String = "Al",
    n_angles::Int = 100,
    max_angle_deg::Float64 = 30.0,
    name::String = "supergaussian"
)
    sigma_rad = deg2rad(sigma_deg)
    angles_rad = collect(range(0.0, deg2rad(max_angle_deg), length=n_angles))

    # Super-Gaussian thickness profile: t(θ) = t_center + Δt × (1 - exp(-0.5 × (|θ|/σ)^n))
    al_thickness = [t_center + delta_t * (1.0 - exp(-0.5 * (θ / sigma_rad)^n)) for θ in angles_rad]

    thickness = reshape(al_thickness, :, 1)
    materials = [material]

    return BowtieFilter(angles_rad, thickness, materials, name)
end

# =============================================================================
# Symbol-based Bowtie Resolver
# =============================================================================

"""
    resolve_bowtie_filter(name::Symbol; kVp::Int=120) -> BowtieFilter

Resolve a bowtie filter symbol to a `BowtieFilter` object.

This is the primary entry point for the driver to look up bowtie filters by name.
Built-in CatSim profiles (889-point GE Revolution data) are preferred over the
generic factory functions.

# Supported Names
- `:large_body` / `:ge_revolution_large` → CatSim large body (889 pts)
- `:medium_body` / `:ge_revolution_medium` → CatSim medium body (889 pts)
- `:small_body` / `:ge_revolution_small` → CatSim small body (889 pts)
- `:head` → Generic head profile (7 pts)
- `:none` → No bowtie filter (flat field)

# Arguments
- `name::Symbol`: Filter name
- `kVp::Int`: Tube voltage (unused — CatSim profiles are physical thickness, not kVp-dependent)

# Returns
`BowtieFilter` ready for use with spectral bowtie attenuation computation.

# Example
```julia
filter = resolve_bowtie_filter(:large_body)
filter = resolve_bowtie_filter(:ge_revolution_large)
filter = resolve_bowtie_filter(:none)
```
"""
function resolve_bowtie_filter(name::Symbol; kVp::Int=120)
    # Built-in CatSim profiles (GE Revolution, 889 data points)
    name == :large_body && return load_builtin_bowtie("large")
    name == :medium_body && return load_builtin_bowtie("medium")
    name == :small_body && return load_builtin_bowtie("small")

    # Generic factory functions
    name == :head && return bowtie_filter_head()
    name == :none && return bowtie_filter_none()

    # GE Revolution aliases → same CatSim data
    name == :ge_revolution_large && return load_builtin_bowtie("large")
    name == :ge_revolution_medium && return load_builtin_bowtie("medium")
    name == :ge_revolution_small && return load_builtin_bowtie("small")

    error("Unknown bowtie filter: :$name. " *
          "Supported: :large_body, :medium_body, :small_body, :head, :none, " *
          ":ge_revolution_large, :ge_revolution_medium, :ge_revolution_small")
end

# =============================================================================
# File-based Bowtie Loading (CatSim compatible)
# =============================================================================

"""
    load_bowtie_filter(filepath::String; name::String="custom") -> BowtieFilter

Load bowtie filter from a text file (legacy format).

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

See also [`load_catsim_bowtie`](@ref) for native CatSim/XCIST format.
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

"""
    load_catsim_bowtie(filepath::String; name::String="catsim") -> BowtieFilter

Load bowtie filter from CatSim/XCIST native format.

This function loads bowtie profiles in the exact format used by CatSim/XCIST:
- Lines starting with # are comments
- First column: fan angle in **radians** (not degrees!)
- Columns 2-5: thickness in **cm** for Al, graphite, Cu, Ti

# CatSim File Format (from gecatsim/bowtie/*.txt):
```
# angle, materials in cm ('al', 'graphite', 'cu', 'ti')
-0.479582   3.537235   0.000000   0.000000   0.000000
-0.478503   3.535054   0.000000   0.000000   0.000000
...
```

# Arguments
- `filepath::String`: Path to CatSim bowtie file
- `name::String`: Filter name (default: "catsim")

# Returns
`BowtieFilter` with angles in radians and thickness in cm.

# Example
```julia
# Load CatSim's medium body bowtie
filter = load_catsim_bowtie("/path/to/gecatsim/bowtie/medium.txt", name="catsim_medium")
```

# Implementation Notes
- CatSim bowtie files are pre-computed for specific fan angle ranges
- Angles are NOT symmetric; negative angles are included explicitly
- For verification, compare profiles using `interpolate_thickness` and manual checks
"""
function load_catsim_bowtie(filepath::String; name::String="catsim")
    # Read file, skipping comment lines
    data = readdlm(filepath, comments=true, comment_char='#')

    # CatSim format: column 1 = angle (radians), columns 2-5 = thickness (cm)
    angles_rad = Float64.(data[:, 1])

    # Thickness is already in cm in CatSim format
    thickness = Float64.(data[:, 2:end])

    # CatSim uses: Al, graphite, Cu, Ti
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

    return BowtieFilter(angles_rad, thickness, materials[1:n_materials], name)
end


# =============================================================================
# Material Attenuation Coefficients
# =============================================================================

# Attenuation μ(E) for bowtie/flat filter materials — delegates to the
# centralised get_filter_mu() in spectrum.jl (XrayAttenuation.jl / NIST XCOM).

"""
    get_bowtie_mu(material::String, energy_keV::Float64) -> Float64

Get linear attenuation coefficient (cm⁻¹) for a filter/bowtie material at the
given energy.  Delegates to `get_filter_mu()` from spectrum.jl.
"""
const get_bowtie_mu = get_filter_mu

"""
    get_bowtie_mu_reference(material::String) -> Float64

Get reference linear attenuation coefficient at 60 keV.
"""
function get_bowtie_mu_reference(material::String)
    return get_filter_mu(material, 60.0)
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
    pixel_row_size_det = geom.pixel_row_size * (geom.SDD / geom.SAD)

    transmission = zeros(Float64, n_cols, n_rows)

    for row in 1:n_rows
        # Cone angle for this row (alpha in CatSim)
        v_offset = (row - (n_rows + 1) / 2) * pixel_row_size_det
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
    pixel_row_size_det = geom.pixel_row_size * (geom.SDD / geom.SAD)
    transmission = zeros(Float64, n_cols, n_rows, n_energies)

    for row in 1:n_rows
        v_offset = (row - (n_rows + 1) / 2) * pixel_row_size_det
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


# =============================================================================
# Exports
# =============================================================================

export BowtieFilter
export bowtie_filter_large_body, bowtie_filter_medium_body
export bowtie_filter_small_body, bowtie_filter_head, bowtie_filter_none
export bowtie_filter_multimaterial, load_bowtie_filter, load_catsim_bowtie
export ge_revolution_bowtie_large, ge_revolution_bowtie_medium, ge_revolution_bowtie_small
export load_builtin_bowtie, resolve_bowtie_filter, bowtie_filter_supergaussian
export interpolate_thickness
export compute_bowtie_attenuation, compute_bowtie_attenuation_spectral
export get_bowtie_mu, get_bowtie_mu_reference
