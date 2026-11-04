# =============================================================================
# Detector materials
# =============================================================================
# One description of a material feeds both sides of every comparison: Geant4 builds its
# G4Material from it, and XrayAttenuation (NIST XCOM) computes the Beer-Lambert reference from
# the same mass fractions and density. A mismatch between the two curves is then physics,
# never two different recipes for the same crystal.
# =============================================================================

"""
    DetectorMaterial(name; density, formula = nothing, mass_fractions = nothing)

A detector material: a `name`, a `density` in g/cm³, and its composition, either as a chemical
`formula` (`"CdTe"`, `"Gd2O2S"`) or as `mass_fractions`, a `Dict` from element symbol to mass
fraction (`Dict("Tb" => 0.287, "Lu" => 0.316, …)`). Mass fractions are normalised to sum to one.

The same object builds the Geant4 material and the XCOM attenuation of
[`linear_attenuation`](@ref), so Monte Carlo and Beer-Lambert always describe the same crystal.
The presets live in [`DETECTOR_MATERIALS`](@ref).
"""
struct DetectorMaterial
    name::String
    density::Float64                      # g/cm³
    mass_fractions::Dict{Int, Float64}     # Z => mass fraction (sums to 1)
end

function DetectorMaterial(
        name::AbstractString; density::Real, formula = nothing,
        mass_fractions = nothing
    )
    (formula === nothing) == (mass_fractions === nothing) &&
        throw(ArgumentError("give exactly one of `formula` or `mass_fractions`"))
    fractions = if formula !== nothing
        XA.formula_to_mass_fractions(String(formula))
    else
        Dict{Int, Float64}(element_z(String(s)) => Float64(w) for (s, w) in mass_fractions)
    end
    total = sum(values(fractions))
    fractions = Dict(z => w / total for (z, w) in fractions)
    return DetectorMaterial(String(name), Float64(density), fractions)
end

Base.show(io::IO, m::DetectorMaterial) = print(
    io, "DetectorMaterial(\"", m.name, "\", ",
    m.density, " g/cm³, ", composition_string(m), ")"
)

element_z(symbol::String) = XA.get_element_by_symbol(symbol).Z
element_symbol(z::Int) = XA.get_element_by_Z(z).symbol

"""Composition as `"Cd 47.0% · Te 53.0%"`, heaviest element first."""
function composition_string(m::DetectorMaterial)
    parts = sort!(collect(m.mass_fractions); by = first, rev = true)
    return join(
        ("$(element_symbol(z)) $(round(100w; digits = 1))%" for (z, w) in parts),
        " · ",
    )
end

"""
    DETECTOR_MATERIALS

Preset [`DetectorMaterial`](@ref)s, keyed by `Symbol`:

| key | material | density (g/cm³) | used by |
|---|---|---|---|
| `:CdTe` | cadmium telluride | 5.85 | photon-counting CT (NAEOTOM Alpha) |
| `:CsI` | cesium iodide | 4.51 | flat-panel imagers; the Hajdok benchmarks |
| `:aSe` | amorphous selenium | 4.27 | mammography; the Hajdok benchmarks |
| `:Gd2O2S` | gadolinium oxysulfide (UFC) | 7.44 | Siemens Force / Flash (BasisSimulator's UFC table) |
| `:UFC` | gadolinium oxysulfide, as Siemens documents UFC | 7.31 | Siemens Force / Flash (UFC data sheet: 7.29–7.33) |
| `:Gemstone` | Ce:(Tb,Lu)₃Al₅O₁₂ garnet | 7.0 | GE Revolution Apex (Hamid's MCNP composition) |
| `:TiO2_epoxy` | epoxy loaded with 40 % TiO₂ by mass | 1.8 | reflector septa between scintillator dexels (assumed) |
| `:Al` | aluminium | 2.699 | filters; the half-value layer ([`half_value_layer`](@ref)) |
| `:Water` | water | 1.0 | patients and phantoms for scatter ([`CylinderPhantom`](@ref)) |
| `:PMMA` | polymethyl methacrylate | 1.19 | phantoms |
| `:Lung` | lung, ICRP (inflated) | 0.26 | heterogeneous phantoms |
| `:CorticalBone` | cortical bone, ICRP | 1.92 | heterogeneous phantoms |
| `:Si` | silicon | 2.33 | silicon photon-counting detectors |
| `:W` | tungsten | 19.3 | anti-scatter collimators ([`Collimator`](@ref)) |

The Gd₂O₂S and Gemstone entries copy the density and composition BasisSimulator already uses
for those crystals (`src/detector/detector_efficiency.jl`), so a Geant4 table made here is
compared with the existing Monte Carlo LUTs on the same material.
"""
const DETECTOR_MATERIALS = Dict{Symbol, DetectorMaterial}(
    :CdTe => DetectorMaterial("CdTe"; density = 5.85, formula = "CdTe"),
    :CsI => DetectorMaterial("CsI"; density = 4.51, formula = "CsI"),
    :aSe => DetectorMaterial("a-Se"; density = 4.27, formula = "Se"),
    :Gd2O2S => DetectorMaterial("Gd2O2S"; density = 7.44, formula = "Gd2O2S"),
    :Gemstone => DetectorMaterial(
        "Gemstone"; density = 7.0, mass_fractions = Dict(
            "Tb" => 0.287, "Lu" => 0.316, "Al" => 0.163, "O" => 0.231, "Ce" => 0.002
        )
    ),
    :UFC => DetectorMaterial("UFC"; density = 7.31, formula = "Gd2O2S"),
    :TiO2_epoxy => DetectorMaterial(
        "TiO2-epoxy"; density = 1.8, mass_fractions = Dict(
            "Ti" => 0.24, "O" => 0.286, "C" => 0.432, "H" => 0.042
        )
    ),
    :Al => DetectorMaterial("Al"; density = 2.699, formula = "Al"),
    :Water => DetectorMaterial("Water"; density = 1.0, formula = "H2O"),
    :PMMA => DetectorMaterial("PMMA"; density = 1.19, formula = "C5H8O2"),
    :Lung => DetectorMaterial(
        "Lung ICRP"; density = 0.26, mass_fractions = Dict(
            "H" => 0.103, "C" => 0.105, "N" => 0.031, "O" => 0.749, "Na" => 0.002,
            "P" => 0.002, "S" => 0.003, "Cl" => 0.003, "K" => 0.002,
        )
    ),
    :CorticalBone => DetectorMaterial(
        "Cortical bone ICRP"; density = 1.92, mass_fractions = Dict(
            "H" => 0.034, "C" => 0.155, "N" => 0.042, "O" => 0.435, "Na" => 0.001,
            "Mg" => 0.002, "P" => 0.103, "S" => 0.003, "Ca" => 0.225,
        )
    ),
    :Si => DetectorMaterial("Si"; density = 2.33, formula = "Si"),
    :W => DetectorMaterial("W"; density = 19.3, formula = "W"),
)

DetectorMaterial(key::Symbol) = DETECTOR_MATERIALS[key]

"""XrayAttenuation's view of a [`DetectorMaterial`](@ref) (same density and mass fractions)."""
xa_material(m::DetectorMaterial) =
    XA.Material(m.name, 0.5, 0.0u"eV", m.density * u"g/cm^3", copy(m.mass_fractions))

"""
    linear_attenuation(material, energy_keV; kind = :total) -> μ in 1/mm

NIST XCOM linear attenuation coefficient of a [`DetectorMaterial`](@ref) at `energy_keV`.
`kind` is `:total` (coherent included, what removes a photon from a narrow beam),
`:photoelectric`, `:compton` or `:rayleigh`.
"""
function linear_attenuation(m::DetectorMaterial, energy_keV::Real; kind::Symbol = :total)
    process = kind === :total ? XA.TotalWithCoherent() :
        kind === :photoelectric ? XA.PhotoelectricAbsorption() :
        kind === :compton ? XA.IncoherentScattering() :
        kind === :rayleigh ? XA.CoherentScattering() :
        throw(ArgumentError("kind must be :total, :photoelectric, :compton or :rayleigh"))
    μρ = sum(
        w * ustrip(
                XA.mass_attenuation_coeff(
                    XA.get_element_by_Z(z), energy_keV * u"keV",
                    process
                )
            )
            for (z, w) in m.mass_fractions
    )              # cm²/g
    return μρ * m.density / 10                           # 1/mm
end

"""
    beer_lambert_efficiency(material, thickness_mm, energy_keV; cross_sections = :xcom) -> η

The analytic answer, `η = 1 − exp(−μ(E)·d)`: the probability that a photon interacts at all
in a slab `thickness_mm` thick at normal incidence. With `cross_sections = :xcom` (the default)
`μ` is NIST XCOM's, which is what BasisSimulator's `:beer_lambert` mode uses; with `:geant4` it
is Geant4's own ([`geant4_attenuation`](@ref)), the attenuation the Monte Carlo samples.

It says nothing about where the energy then goes: a photon that interacts and sends its
K-fluorescence back out counts as fully detected here. The Monte Carlo
[`interaction_probability`](@ref) must reproduce the `:geant4` version to within counting
statistics; [`energy_absorption_efficiency`](@ref) shows what it leaves out.
"""
function beer_lambert_efficiency(
        m::DetectorMaterial, thickness_mm::Real, energy_keV::Real;
        cross_sections::Symbol = :xcom
    )
    μ = cross_sections === :xcom ? linear_attenuation(m, energy_keV) :
        cross_sections === :geant4 ? geant4_attenuation(m, energy_keV) :
        throw(ArgumentError("cross_sections must be :xcom or :geant4"))
    return 1 - exp(-μ * thickness_mm)
end

"""
    k_edges(material; min_fraction = 0.01) -> Vector of (element, edge_keV, kalpha_keV, yield)

The K-shell edges of the elements of `material` that fall in the diagnostic range (5–150 keV)
and make up at least `min_fraction` of its mass (so a 0.2 % dopant is left out),
with each element's Kα fluorescence energy and K-fluorescence yield. These are where the Monte
Carlo efficiency departs from Beer-Lambert: just above an edge the photoelectric effect makes
a Kα photon in a fraction `yield` of the absorptions, and that photon can leave.
"""
function k_edges(m::DetectorMaterial; min_fraction::Real = 0.01)
    edges = NamedTuple{(:element, :edge_keV, :kalpha_keV, :yield), Tuple{String, Float64, Float64, Float64}}[]
    for z in sort!(collect(keys(m.mass_fractions)))
        m.mass_fractions[z] >= min_fraction || continue
        edge = XA.k_binding_energy(z)
        5 <= edge <= 150 || continue
        push!(
            edges, (
                element = element_symbol(z), edge_keV = edge,
                kalpha_keV = XA.k_alpha_energy(z), yield = XA.k_fluorescence_yield(z),
            )
        )
    end
    return edges
end

# -----------------------------------------------------------------------------
# Geant4 side
# -----------------------------------------------------------------------------

"""Build (once per process and name) the Geant4 material for `m`."""
function g4_material(m::DetectorMaterial)
    existing = G4Material!GetMaterial("BasisMonteCarlo_" * m.name, false)
    existing != C_NULL && return existing
    nist = G4NistManager!Instance()
    mat = G4Material(
        "BasisMonteCarlo_" * m.name; density = m.density * g / cm3,
        ncomponents = length(m.mass_fractions)
    )
    for (z, w) in sort!(collect(m.mass_fractions))
        AddElement(mat, FindOrBuildElement(nist, z); fractionmass = w)
    end
    return CxxPtr(move!(mat))
end

"""
    DetectorMaterial(material::XrayAttenuation.Material)

The same material as XrayAttenuation (and so BasisSimulator) describes it: its name, density and
mass fractions. A phantom built from BasisSimulator's materials is then the same phantom in Geant4.
"""
DetectorMaterial(m::XA.Material) = DetectorMaterial(
    m.name;
    density = ustrip(u"g/cm^3", m.density),
    mass_fractions = Dict(element_symbol(z) => Float64(w) for (z, w) in m.composition),
)
