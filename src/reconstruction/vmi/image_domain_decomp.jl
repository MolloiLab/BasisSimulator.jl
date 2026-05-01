"""
Ding 2012 image-domain dual-energy material decomposition.

Reference: Ding et al. (2012) — image-domain DE decomp via least-squares
fit of a linear model relating measured rod HU at low/high kVp scans to
nominal iodine concentrations:

    c_iodine(voxel) = a₀ + a₁ · HU_low(voxel) + a₂ · HU_high(voxel)

The three coefficients `(a₀, a₁, a₂)` are fit from a small set of
calibration ROIs (water + 7 iodine rods on a Gammex 472-style phantom)
via simple LSQ on the design matrix `[1, HU_low, HU_high]`.

The same machinery generalizes to PCCT bin pairs (`HU_low_bin`,
`HU_high_bin`) — Ding's model only requires two HU "channels" with
different effective spectra.

# Public API
- [`fit_ding_coeffs`](@ref) — LSQ fit on rod ROIs.
- [`apply_ding_decomp`](@ref) — broadcast coeffs over a volume pair.
- [`apply_ding_decomp!`](@ref) — in-place variant.
- [`synth_vmi_image_domain`](@ref) — per-energy VMI synthesis from
  `(HU_low, c_iodine)`.
- [`synth_vmi_image_domain!`](@ref) — in-place variant.
"""

# =============================================================================
# Calibration fit
# =============================================================================

"""
    fit_ding_coeffs(rod_HU_low, rod_HU_high, c_known)
        -> NamedTuple{(:coeffs, :α_low, :α_high, :rms, :pred_c)}

LSQ-fit the Ding image-domain decomposition coefficients from per-rod
HU at low/high kVp (or low/high PCCT bins) and known iodine
concentrations.

# Arguments (positional)
- `rod_HU_low::AbstractVector`  : per-rod HU at the LOW-kVp / low-bin scan
- `rod_HU_high::AbstractVector` : per-rod HU at the HIGH-kVp / high-bin scan
- `c_known::AbstractVector`     : per-rod nominal iodine concentration
                                  (mg/mL).  0 for the water anchor; 7
                                  iodine rods at 2.0..20.0 mg/mL is a
                                  good 8-point set.

All three vectors must be the same length.

# Returns
NamedTuple:
- `coeffs::Vector{Float32}` : `[a₀, a₁, a₂]` — the 3-vector fit.
- `α_low::Float32`          : iodine HU-per-(mg/mL) sensitivity at low —
                              `Σ c·HU_low / Σ c²` over iodine rods only.
                              Used for VMI synthesis.
- `α_high::Float32`         : same, at high.
- `rms::Float64`            : LSQ residual RMS in mg/mL.
- `pred_c::Vector{Float64}` : predicted concentrations (LSQ output).
"""
function fit_ding_coeffs(
        rod_HU_low::AbstractVector,
        rod_HU_high::AbstractVector,
        c_known::AbstractVector,
    )
    n = length(c_known)
    (length(rod_HU_low) == n && length(rod_HU_high) == n) ||
        error("fit_ding_coeffs: input vectors must share length, got " *
              "HU_low=$(length(rod_HU_low)), HU_high=$(length(rod_HU_high)), c=$(n)")

    HU_low_f  = Float64.(rod_HU_low)
    HU_high_f = Float64.(rod_HU_high)
    c_f       = Float64.(c_known)

    # Ding Eq 3 LSQ:  A · coeffs = c   where A = [1, HU_low, HU_high]
    A = hcat(ones(n), HU_low_f, HU_high_f)
    coeffs = A \ c_f
    pred_c = A * coeffs
    rms    = sqrt(mean((pred_c .- c_f) .^ 2))

    # Iodine HU/(mg/mL) sensitivity slope through origin (water rod has
    # c=0, contributes nothing).  Used by `synth_vmi_image_domain` for
    # iodine-contribution swap at the target energy.
    iod_idx = findall(c -> c > 0, c_f)
    Σc²    = sum(c_f[iod_idx] .^ 2)
    α_low  = sum(c_f[iod_idx] .* HU_low_f[iod_idx])  / Σc²
    α_high = sum(c_f[iod_idx] .* HU_high_f[iod_idx]) / Σc²

    (coeffs = Float32.(coeffs),
     α_low  = Float32(α_low),
     α_high = Float32(α_high),
     rms    = rms,
     pred_c = pred_c)
end


# =============================================================================
# Apply Ding decomp — voxel-wise c_iodine map
# =============================================================================

"""
    apply_ding_decomp!(c_iodine::AbstractArray{Float32, 3},
                       HU_low::AbstractArray{Float32, 3},
                       HU_high::AbstractArray{Float32, 3},
                       coeffs)
        -> c_iodine

In-place voxel-wise Ding decomposition:
    c_iodine[v] = a₀ + a₁ · HU_low[v] + a₂ · HU_high[v]

`coeffs` must be the 3-vector output of [`fit_ding_coeffs`](@ref) (or any
3-element iterable of Float32).
"""
function apply_ding_decomp!(
        c_iodine::AbstractArray{Float32, 3},
        HU_low::AbstractArray{Float32, 3},
        HU_high::AbstractArray{Float32, 3},
        coeffs,
    )
    size(c_iodine) == size(HU_low) == size(HU_high) ||
        error("apply_ding_decomp!: shapes must match")
    length(coeffs) == 3 || error("apply_ding_decomp!: expected 3-vector coeffs, got $(length(coeffs))")
    a0 = Float32(coeffs[1]); a1 = Float32(coeffs[2]); a2 = Float32(coeffs[3])
    @. c_iodine = a0 + a1 * HU_low + a2 * HU_high
    c_iodine
end

"""
    apply_ding_decomp(HU_low, HU_high, coeffs) -> Array{Float32, 3}

Allocating wrapper around [`apply_ding_decomp!`](@ref).  Returns a
freshly allocated `c_iodine` map.
"""
function apply_ding_decomp(
        HU_low::AbstractArray{Float32, 3},
        HU_high::AbstractArray{Float32, 3},
        coeffs,
    )
    out = similar(HU_low)
    apply_ding_decomp!(out, HU_low, HU_high, coeffs)
end


# =============================================================================
# VMI synthesis (image domain, per energy)
# =============================================================================

"""
    synth_vmi_image_domain!(HU_E::AbstractArray{Float32, 3},
                            HU_low::AbstractArray{Float32, 3},
                            c_iodine::AbstractArray{Float32, 3};
                            energy_keV::Real,
                            α_iod_low_cal::Real,
                            water_material  = XA.Materials.water,
                            iodine_material = XA.Elements.Iodine)
        -> HU_E

Per-energy VMI synthesis in image domain:
    HU_E[v] = HU_low[v] + c_iodine[v] · (α_iod_E_phys − α_iod_low_cal)
where `α_iod_E_phys = μρ_iodine(E) / μρ_water(E)` is the physics-based
iodine HU-per-(mg/mL) sensitivity at the target energy, and
`α_iod_low_cal` is the empirical sensitivity at the LOW basis (returned
as `α_low` from [`fit_ding_coeffs`](@ref)).

Intuition:
  HU_low contains the iodine contribution at the low-basis sensitivity
  (= c_iodine · α_iod_low_cal).  Subtract that and add the iodine
  contribution at the target energy → HU_E.

# Required keyword arguments
- `energy_keV`          : VMI target energy (keV).
- `α_iod_low_cal`       : low-basis empirical iodine HU/(mg/mL).

# Optional kwargs
- `water_material`, `iodine_material` : XrayAttenuation materials for
  μ/ρ lookup.  Defaults to `XA.Materials.water` + `XA.Elements.Iodine`.
"""
function synth_vmi_image_domain!(
        HU_E::AbstractArray{Float32, 3},
        HU_low::AbstractArray{Float32, 3},
        c_iodine::AbstractArray{Float32, 3};
        energy_keV::Real,
        α_iod_low_cal::Real,
        water_material  = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
    )
    size(HU_E) == size(HU_low) == size(HU_compatible_check_size(HU_low, c_iodine)) ||
        error("synth_vmi_image_domain!: shapes must match")
    μρ_w_E = compute_mass_μ_at_energy(water_material,  Float64(energy_keV))
    μρ_I_E = compute_mass_μ_at_energy(iodine_material, Float64(energy_keV))
    α_E_phys = Float32(μρ_I_E / μρ_w_E)
    Δα = α_E_phys - Float32(α_iod_low_cal)
    @. HU_E = HU_low + c_iodine * Δα
    HU_E
end

# Tiny helper to keep the size check legible above.
HU_compatible_check_size(a, b) = (size(a) == size(b) ? b : error("shape mismatch $(size(a)) vs $(size(b))"))

"""
    synth_vmi_image_domain(HU_low, c_iodine; energy_keV, α_iod_low_cal,
                           water_material  = XA.Materials.water,
                           iodine_material = XA.Elements.Iodine)
        -> Array{Float32, 3}

Allocating wrapper around [`synth_vmi_image_domain!`](@ref).  Returns a
freshly allocated `HU_E` volume at the target keV.
"""
function synth_vmi_image_domain(
        HU_low::AbstractArray{Float32, 3},
        c_iodine::AbstractArray{Float32, 3};
        energy_keV::Real,
        α_iod_low_cal::Real,
        water_material  = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
    )
    out = similar(HU_low)
    synth_vmi_image_domain!(out, HU_low, c_iodine;
        energy_keV     = energy_keV,
        α_iod_low_cal  = α_iod_low_cal,
        water_material  = water_material,
        iodine_material = iodine_material,
    )
end


export fit_ding_coeffs,
       apply_ding_decomp, apply_ding_decomp!,
       synth_vmi_image_domain, synth_vmi_image_domain!
