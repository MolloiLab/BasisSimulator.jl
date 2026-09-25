# =============================================================================
# Two-basis (water + iodine) VMI synthesis from basis images
# =============================================================================
#
# The last stage of the spectral chain (`synthesize_vmi_stack` / `vmi_pipeline` in
# nchannel.jl): turn a reconstructed (water, iodine) basis-image pair into virtual
# monoenergetic HU at a target energy.

"""
    synth_vmi_2basis!(HU_E, c_water, c_iodine; energy_keV,
                      water_material  = XA.Materials.water,
                      iodine_material = XA.Elements.Iodine)
        -> HU_E

Per-energy VMI synthesis from a two-basis (water + iodine) image-domain
decomposition, written into `HU_E`:

    μ(E)  = c_water · μρ_water(E) + c_iodine · 1e-3 · μρ_iodine(E)
    HU(E) = 1000·(μ(E) − μρ_water(E)) / μρ_water(E)
          = 1000·(c_water − 1) + c_iodine · α(E),    α(E) = μρ_iodine(E) / μρ_water(E)

`μρ` are the mass attenuation coefficients of `water_material` and `iodine_material`
at `energy_keV`. `c_water` is in g/mL (≈ 1 for water) and `c_iodine` in mg/mL.
`HU_E`, `c_water` and `c_iodine` are 3-D `Float32` arrays of the same size (an error
is thrown otherwise); any array backend works.
"""
function synth_vmi_2basis!(
        HU_E::AbstractArray{Float32, 3},
        c_water::AbstractArray{Float32, 3},
        c_iodine::AbstractArray{Float32, 3};
        energy_keV::Real,
        water_material  = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
    )
    size(HU_E) == size(c_water) == size(c_iodine) ||
        error("synth_vmi_2basis!: shapes must match")
    μρ_w = compute_mass_μ_at_energy(water_material,  Float64(energy_keV))
    μρ_I = compute_mass_μ_at_energy(iodine_material, Float64(energy_keV))
    α_E  = Float32(μρ_I / μρ_w)
    @. HU_E = 1.0f3 * (c_water - 1.0f0) + c_iodine * α_E
    HU_E
end

"""
    synth_vmi_2basis(c_water, c_iodine; energy_keV,
                     water_material = XA.Materials.water,
                     iodine_material = XA.Elements.Iodine) -> AbstractArray{Float32, 3}

Allocating form of [`synth_vmi_2basis!`](@ref): a new `Float32` volume like `c_water` (same
size and array backend) with the virtual monoenergetic HU at `energy_keV`,
`HU(E) = 1000·(c_water − 1) + c_iodine · μρ_iodine(E)/μρ_water(E)`, `c_water` in g/mL and
`c_iodine` in mg/mL (3-D `Float32` arrays of the same size).
"""
function synth_vmi_2basis(
        c_water::AbstractArray{Float32, 3},
        c_iodine::AbstractArray{Float32, 3};
        energy_keV::Real,
        water_material  = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
    )
    out = similar(c_water)
    synth_vmi_2basis!(out, c_water, c_iodine;
        energy_keV      = energy_keV,
        water_material  = water_material,
        iodine_material = iodine_material,
    )
end


export synth_vmi_2basis!, synth_vmi_2basis
