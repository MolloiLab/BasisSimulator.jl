"""
    src/source/bowtie_filter.jl

Bowtie filter modeling for CT simulation.

## Provenance

The bowtie data and algorithm in this file are a Julia port of the
CatSim / XCIST X-ray filter module:

  - Algorithm: `gecatsim/pyfiles/Xray_Filter.py :: bowtie_filter()`
    https://github.com/xcist/main
  - Data:      `gecatsim/bowtie/{large,medium,small}.txt`
    Bundled byte-identical at `src/bowtie/{large,medium,small}.txt`
    (888 fan-angle samples × 4 material columns [Al, graphite, Cu, Ti];
    angle in radians, thickness in cm).
  - License:   BSD 3-Clause, Copyright 2024 GE Precision HealthCare.
    https://github.com/xcist/main/tree/master/license

## Physics (mirrors `Xray_Filter.py:bowtie_filter()` line-for-line)

The bowtie filter is placed between the X-ray source and patient to
(i) reduce peripheral dose, (ii) equalize the detector signal by
compensating for varying patient path lengths, and (iii) harden the
edge beam to suppress scatter. CatSim's body bowties are thinnest at
the center of fan (γ ≈ 0, ~0.1 cm Al) and thickest at the edges
(|γ| ≈ 0.48 rad, ~3.5 cm Al).

Per detector pixel at fan angle γ and cone angle α:

    t(γ, m)    = linear interp of t₀[:, m] on γ₀ → γ      # per material
    t(γ, α, m) = t(γ, m) / cos(α)                          # cone correction
    T(γ, α, E) = exp(-Σ_m μ_m(E) · t(γ, α, m))             # Beer-Lambert

where `μ_m(E)` is from XrayAttenuation.jl (NIST XCOM database) via
`get_filter_mu()` in `spectrum.jl`. The spectral-domain attenuation is
applied during spectrum weighting — not as a sinogram-domain post-step.
"""

using DelimitedFiles

# =============================================================================
# Built-in Bowtie Data Directory
# =============================================================================
#
# `src/bowtie/{large,medium,small}.txt` are byte-identical copies of the
# `gecatsim/bowtie/*.txt` files (BSD 3-Clause, Copyright 2024 GE Precision
# HealthCare). The upstream copyright header is preserved verbatim in each
# file as the in-band acknowledgment.
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
# Pre-defined Bowtie Filters (head / none — body bowties come from the
# bundled CatSim .txt data via load_builtin_bowtie)
# =============================================================================

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

The package ships with GE Revolution bowtie profiles (888 data points each,
4-material columns) from the CatSim/XCIST open-source CT simulator.

# Arguments
- `size::String`: Filter size — `"large"`, `"medium"`, or `"small"`
- `bowtie_dir::AbstractString`: Directory containing bowtie data files (default: bundled data)

# Returns
`BowtieFilter` with 888-point asymmetric profile.

# Example
```julia
filter = load_builtin_bowtie("large")
length(filter.angles)  # 888
```
"""
function load_builtin_bowtie(size::String; bowtie_dir::AbstractString = BOWTIE_DIR)
    filepath = joinpath(bowtie_dir, "$(size).txt")
    isfile(filepath) || error("Built-in bowtie file not found: $filepath")
    return load_catsim_bowtie(filepath; name = "catsim_$(size)")
end

# =============================================================================
# Symbol-based Bowtie Resolver
# =============================================================================

"""
    resolve_bowtie_filter(name::Symbol; kVp::Int=120) -> BowtieFilter

Resolve a bowtie filter symbol to a `BowtieFilter` object.

This is the primary entry point for the driver to look up bowtie filters by name.

# Supported Names
- `:large_body` / `:ge_revolution_large` → CatSim large body (888 pts)
- `:medium_body` / `:ge_revolution_medium` → CatSim medium body (888 pts)
- `:small_body` / `:ge_revolution_small` → CatSim small body (888 pts)
- `:head` → Generic head profile (7 pts, flat-ish for circular cross-section)
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
function resolve_bowtie_filter(name::Symbol; kVp::Int = 120)
    # Built-in CatSim profiles (GE Revolution, 888 data points)
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

    error(
        "Unknown bowtie filter: :$name. " *
            "Supported: :large_body, :medium_body, :small_body, :head, :none, " *
            ":ge_revolution_large, :ge_revolution_medium, :ge_revolution_small"
    )
end

# =============================================================================
# File-based Bowtie Loading (CatSim/XCIST native format)
# =============================================================================

"""
    load_catsim_bowtie(filepath::String; name::String="catsim") -> BowtieFilter

Load bowtie filter from CatSim/XCIST native format.

Mirrors the file reader in `gecatsim/pyfiles/Xray_Filter.py:bowtie_filter()`
(line ~51, `np.loadtxt(bowtieFile, comments=['#', '%'])`). Material order
[Al, graphite, Cu, Ti] is fixed by the upstream convention.

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
function load_catsim_bowtie(filepath::String; name::String = "catsim")
    # Read file, skipping comment lines
    data = readdlm(filepath, comments = true, comment_char = '#')

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
    for i in 1:(length(filter.angles) - 1)
        if abs_angle <= filter.angles[i + 1]
            t = (abs_angle - filter.angles[i]) / (filter.angles[i + 1] - filter.angles[i])
            return filter.thickness[i, :] .+ t .* (filter.thickness[i + 1, :] .- filter.thickness[i, :])
        end
    end

    return filter.thickness[end, :]
end

"""
    compute_bowtie_attenuation_spectral(filter::BowtieFilter, geom::CTGeometry,
                                        energies::Vector{Float64}) -> Array{Float64,3}

Compute energy-dependent bowtie attenuation for polychromatic simulation.

Direct port of `gecatsim/pyfiles/Xray_Filter.py :: bowtie_filter()`
(BSD 3-Clause, GE Precision HealthCare via CatSim/XCIST). The CatSim
reference (lines ~56-67) does, in pseudocode:

    muT = 0
    for material in [Al, graphite, Cu, Ti]:
        mu     = GetMu(material, Evec)                       # μ_m(E)
        t      = interp1d(γ₀, t₀[:, m])(γ_det) / cos(α_det)  # thickness/cos(α)
        muT   += outer(t, mu)                                # [pixel, E]
    trans = exp(-muT)

`get_filter_mu()` plays the role of CatSim's `GetMu`; `interpolate_thickness`
plays the role of `scipy.interpolate.interp1d(kind='linear', ...)`.

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
export bowtie_filter_head, bowtie_filter_none
export load_catsim_bowtie, load_builtin_bowtie
export resolve_bowtie_filter
export interpolate_thickness
export compute_bowtie_attenuation_spectral
export get_bowtie_mu
