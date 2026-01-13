"""
    Forward/FlatFilter.jl

Flat (inherent) filtration modeling for CT simulation.

The flat filter is placed at the X-ray source before the bowtie filter to:
1. Remove low-energy photons (reduce patient dose)
2. Harden the beam spectrum
3. Reduce beam hardening artifacts

Unlike the bowtie filter, the flat filter has uniform thickness across the beam.
Common materials: aluminum (Al), copper (Cu), titanium (Ti).

Implementation follows CatSim/XCIST approach:
- Multiple filter layers supported
- Energy-dependent attenuation
- Geometric correction for oblique rays
"""

# =============================================================================
# Flat Filter Types
# =============================================================================

"""
    FlatFilter

Flat (inherent) filter specification.

# Fields
- `materials`: Vector of material names
- `thicknesses`: Vector of thicknesses in mm for each material
- `name`: Filter name/description
"""
struct FlatFilter
    materials::Vector{String}
    thicknesses::Vector{Float64}  # mm
    name::String
end

# =============================================================================
# Pre-defined Flat Filters
# =============================================================================

"""
    flat_filter_none()

No flat filtration.
"""
function flat_filter_none()
    return FlatFilter(String[], Float64[], "none")
end

"""
    flat_filter_al(thickness_mm::Float64=2.5)

Aluminum flat filter (standard for diagnostic CT).

Typical values: 2-7 mm Al equivalent.
"""
function flat_filter_al(thickness_mm::Float64=2.5)
    return FlatFilter(["Al"], [thickness_mm], "Al_$(thickness_mm)mm")
end

"""
    flat_filter_cu(thickness_mm::Float64=0.1)

Copper flat filter (for additional beam hardening).

Typical values: 0.1-0.5 mm Cu.
"""
function flat_filter_cu(thickness_mm::Float64=0.1)
    return FlatFilter(["Cu"], [thickness_mm], "Cu_$(thickness_mm)mm")
end

"""
    flat_filter_al_cu(al_mm::Float64=2.5, cu_mm::Float64=0.1)

Combined aluminum and copper filter.

Common clinical configuration for dose reduction.
"""
function flat_filter_al_cu(al_mm::Float64=2.5, cu_mm::Float64=0.1)
    return FlatFilter(["Al", "Cu"], [al_mm, cu_mm], "Al_$(al_mm)mm_Cu_$(cu_mm)mm")
end

"""
    flat_filter_ti(thickness_mm::Float64=0.5)

Titanium flat filter.

Used in some modern CT systems.
"""
function flat_filter_ti(thickness_mm::Float64=0.5)
    return FlatFilter(["Ti"], [thickness_mm], "Ti_$(thickness_mm)mm")
end

"""
    flat_filter_custom(materials::Vector{String}, thicknesses::Vector{Float64})

Create custom flat filter with specified materials and thicknesses.

# Arguments
- `materials`: Vector of material names (e.g., ["Al", "Cu"])
- `thicknesses`: Vector of thicknesses in mm

# Example
```julia
filter = flat_filter_custom(["Al", "Cu", "Ti"], [3.0, 0.2, 0.1])
```
"""
function flat_filter_custom(materials::Vector{String}, thicknesses::Vector{Float64})
    @assert length(materials) == length(thicknesses) "Materials and thicknesses must have same length"
    @assert all(thicknesses .>= 0) "Thicknesses must be non-negative"

    name = join(["$(m)_$(t)mm" for (m, t) in zip(materials, thicknesses)], "_")
    return FlatFilter(materials, thicknesses, name)
end

# =============================================================================
# Flat Filter Attenuation
# =============================================================================

"""
    compute_flat_filter_attenuation(filter::FlatFilter, geom::CTGeometry;
                                    energy_keV::Float64=60.0) -> Array{Float64,2}

Compute flat filter transmission for all detector pixels.

The flat filter has uniform thickness but rays at oblique angles
travel longer paths through the filter.

# Arguments
- `filter::FlatFilter`: Flat filter specification
- `geom::CTGeometry`: Scanner geometry
- `energy_keV`: Reference energy for μ calculation

# Returns
2D array [n_cols, n_rows] of transmission factors.
"""
function compute_flat_filter_attenuation(
    filter::FlatFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows

    if isempty(filter.materials)
        return ones(Float64, n_cols, n_rows)
    end

    # Get μ for each material
    μ_vec = [get_bowtie_mu(mat, energy_keV) for mat in filter.materials]

    # Convert thicknesses from mm to cm
    t_cm = filter.thicknesses ./ 10.0

    # Compute total μt product for perpendicular path
    μt_perpendicular = sum(μ_vec .* t_cm)

    # Compute geometric correction for each detector pixel
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    transmission = zeros(Float64, n_cols, n_rows)

    for row in 1:n_rows
        # Cone angle (alpha)
        v_offset = (row - (n_rows + 1) / 2) * pixel_size_det
        cos_alpha = cos(atan(v_offset / geom.SDD))

        for col in 1:n_cols
            # Fan angle (gamma)
            u_offset = (col - (n_cols + 1) / 2) * pixel_size_det
            cos_gamma = cos(atan(u_offset / geom.SDD))

            # Path length correction: 1 / (cos(alpha) * cos(gamma))
            # Following CatSim: cosineFactors = 1/cos(gammas)/cos(alphas)
            path_factor = 1.0 / (cos_alpha * cos_gamma)

            # Transmission = exp(-μt × path_factor)
            transmission[col, row] = exp(-μt_perpendicular * path_factor)
        end
    end

    return transmission
end

"""
    compute_flat_filter_attenuation_spectral(filter::FlatFilter, geom::CTGeometry,
                                             energies::Vector{Float64}) -> Array{Float64,3}

Compute energy-dependent flat filter transmission.

# Returns
3D array [n_cols, n_rows, n_energies] of transmission factors.
"""
function compute_flat_filter_attenuation_spectral(
    filter::FlatFilter,
    geom::CTGeometry,
    energies::Vector{Float64}
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_energies = length(energies)

    if isempty(filter.materials)
        return ones(Float64, n_cols, n_rows, n_energies)
    end

    # Compute μ for each material at each energy
    n_materials = length(filter.materials)
    μ_matrix = zeros(n_materials, n_energies)
    for (i, mat) in enumerate(filter.materials)
        for (j, E) in enumerate(energies)
            μ_matrix[i, j] = get_bowtie_mu(mat, E)
        end
    end

    # Convert thicknesses to cm
    t_cm = filter.thicknesses ./ 10.0

    # Compute μt for each energy
    μt_vec = [sum(μ_matrix[:, k] .* t_cm) for k in 1:n_energies]

    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    transmission = zeros(Float64, n_cols, n_rows, n_energies)

    for row in 1:n_rows
        v_offset = (row - (n_rows + 1) / 2) * pixel_size_det
        cos_alpha = cos(atan(v_offset / geom.SDD))

        for col in 1:n_cols
            u_offset = (col - (n_cols + 1) / 2) * pixel_size_det
            cos_gamma = cos(atan(u_offset / geom.SDD))
            path_factor = 1.0 / (cos_alpha * cos_gamma)

            for k in 1:n_energies
                transmission[col, row, k] = exp(-μt_vec[k] * path_factor)
            end
        end
    end

    return transmission
end

"""
    apply_flat_filter(sinogram, filter::FlatFilter, geom::CTGeometry;
                      energy_keV::Float64=60.0) -> Array

Apply flat filter attenuation to sinogram.

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles]
- `filter::FlatFilter`: Flat filter specification
- `geom::CTGeometry`: Scanner geometry
- `energy_keV`: Reference energy (default: 60 keV)

# Returns
Sinogram with flat filter effect added.
"""
function apply_flat_filter(
    sinogram::AbstractArray{T,3},
    filter::FlatFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    if isempty(filter.materials)
        return sinogram
    end

    # Compute transmission
    transmission = compute_flat_filter_attenuation(filter, geom; energy_keV=energy_keV)

    # In projection domain: add -log(transmission)
    filter_projection = -log.(transmission)

    n_angles = size(sinogram, 3)
    result = similar(sinogram)

    for angle in 1:n_angles
        result[:, :, angle] = sinogram[:, :, angle] .+ T.(filter_projection)
    end

    return result
end

"""
    apply_flat_filter_to_intensity(intensity, filter::FlatFilter, geom::CTGeometry;
                                   energy_keV::Float64=60.0) -> Array

Apply flat filter to intensity-domain data (before log transform).

# Arguments
- `intensity`: Intensity data [n_cols, n_rows, n_angles]
- `filter::FlatFilter`: Flat filter specification
- `geom::CTGeometry`: Scanner geometry

# Returns
Attenuated intensity data.
"""
function apply_flat_filter_to_intensity(
    intensity::AbstractArray{T,3},
    filter::FlatFilter,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    if isempty(filter.materials)
        return intensity
    end

    transmission = T.(compute_flat_filter_attenuation(filter, geom; energy_keV=energy_keV))

    n_angles = size(intensity, 3)
    result = similar(intensity)

    for angle in 1:n_angles
        result[:, :, angle] = intensity[:, :, angle] .* transmission
    end

    return result
end

"""
    get_flat_filter_info(filter::FlatFilter) -> NamedTuple

Get diagnostic information about flat filter.
"""
function get_flat_filter_info(filter::FlatFilter)
    if isempty(filter.materials)
        return (
            name = "none",
            n_materials = 0,
            materials = String[],
            thicknesses_mm = Float64[],
            total_al_equivalent_mm = 0.0
        )
    end

    # Compute Al-equivalent thickness at 60 keV
    μ_al = get_bowtie_mu("Al", 60.0)
    al_equiv = 0.0
    for (mat, t) in zip(filter.materials, filter.thicknesses)
        μ_mat = get_bowtie_mu(mat, 60.0)
        al_equiv += t * (μ_mat / μ_al)
    end

    return (
        name = filter.name,
        n_materials = length(filter.materials),
        materials = filter.materials,
        thicknesses_mm = filter.thicknesses,
        total_al_equivalent_mm = al_equiv
    )
end

# =============================================================================
# Exports
# =============================================================================

export FlatFilter
export flat_filter_none, flat_filter_al, flat_filter_cu
export flat_filter_al_cu, flat_filter_ti, flat_filter_custom
export compute_flat_filter_attenuation, compute_flat_filter_attenuation_spectral
export apply_flat_filter, apply_flat_filter_to_intensity
export get_flat_filter_info
