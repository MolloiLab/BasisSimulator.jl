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


# =============================================================================
# 2-basis (water + iodine) μ-domain VMI pipeline — selectable Ding form
# =============================================================================

"""
    eval_cal(L, H, c, form::Symbol)

Scalar calibration evaluator — maps a single `(HU_low, HU_high)` pair to
a basis fraction (typically `c_iodine` in mg/mL).  Two forms supported:

- `:linear` (Ding-2012, 3 params):   `c[1] + c[2]·L + c[3]·H`
- `:rational_quadratic` (Ding-2020, 8 params):
        `(c[1] + c[2]·L + c[3]·H + c[4]·L² + c[5]·L·H + c[6]·H²)
       / (1 + c[7]·L + c[8]·H)`

The denominator should stay strictly positive over the HU range covered
by the calibration data — the BFGS optimizer doesn't enforce this, so
sanity-check post-fit if you're rolling your own.
"""
@inline function eval_cal(L, H, c, form::Symbol)
    Base.depwarn("Image-domain decomposition calibrations are QUARANTINED: constants are stale (flat+Siddon era) and the rational-quadratic denominators have poles inside the clinical HU domain (audit B1/B6). Use the projection-domain Cong chain.", :eval_cal)
    if form === :linear
        return c[1] + c[2]*L + c[3]*H
    elseif form === :rational_quadratic
        num = c[1] + c[2]*L + c[3]*H + c[4]*L*L + c[5]*L*H + c[6]*H*H
        den = oneunit(c[1]) + c[7]*L + c[8]*H
        return num / den
    else
        error("eval_cal: unknown form $(form) — expected :linear or :rational_quadratic")
    end
end


"""
    apply_cal!(out, HU_low, HU_high, coeffs; form = :linear) -> out
    apply_cal(HU_low, HU_high, coeffs; form = :linear)        -> Array{Float32, 3}

Voxel-wise calibration: writes `eval_cal(HU_low[v], HU_high[v], coeffs;
form)` into every voxel of `out`.  Useful for both `c_iodine` and any
other Ding-style basis fraction map.
"""
function apply_cal!(
        out::AbstractArray{Float32, 3},
        HU_low::AbstractArray{Float32, 3},
        HU_high::AbstractArray{Float32, 3},
        coeffs;
        form::Symbol = :linear,
    )
    Base.depwarn("Image-domain decomposition calibrations are QUARANTINED: constants are stale (flat+Siddon era) and the rational-quadratic denominators have poles inside the clinical HU domain (audit B1/B6). Use the projection-domain Cong chain.", :apply_cal!)
    size(out) == size(HU_low) == size(HU_high) ||
        error("apply_cal!: shapes must match")
    if form === :linear
        a0 = Float32(coeffs[1]); a1 = Float32(coeffs[2]); a2 = Float32(coeffs[3])
        @. out = a0 + a1 * HU_low + a2 * HU_high
    elseif form === :rational_quadratic
        a0 = Float32(coeffs[1]); a1 = Float32(coeffs[2]); a2 = Float32(coeffs[3])
        a3 = Float32(coeffs[4]); a4 = Float32(coeffs[5]); a5 = Float32(coeffs[6])
        b1 = Float32(coeffs[7]); b2 = Float32(coeffs[8])
        @. out = (a0 + a1*HU_low + a2*HU_high +
                  a3*HU_low*HU_low + a4*HU_low*HU_high + a5*HU_high*HU_high) /
                 (1.0f0 + b1*HU_low + b2*HU_high)
    else
        error("apply_cal!: unknown form $(form) — expected :linear or :rational_quadratic")
    end
    out
end

apply_cal(HU_low::AbstractArray{Float32, 3},
          HU_high::AbstractArray{Float32, 3},
          coeffs;
          form::Symbol = :linear) =
    apply_cal!(similar(HU_low), HU_low, HU_high, coeffs; form = form)


"""
    synth_vmi_2basis!(HU_E, c_water, c_iodine; energy_keV,
                      water_material  = XA.Materials.water,
                      iodine_material = XA.Elements.Iodine)
        -> HU_E
    synth_vmi_2basis(c_water, c_iodine; ...)
        -> Array{Float32, 3}

Per-energy VMI synthesis from a 2-basis (water + iodine) image-domain
decomposition (McCollough 2015 / Yu 2012 textbook form):

    μ(E)  = c_water · μρ_water(E) + c_iodine · 1e-3 · μρ_iodine(E)
    HU(E) = 1000·(μ(E) − μρ_water(E)) / μρ_water(E)
          = 1000·(c_water − 1) + c_iodine · α_phys(E)

where `α_phys(E) = μρ_iodine(E) / μρ_water(E)`.  Replaces the
iodine-only perturbation form `HU_low + c_iod·(α_E − α_low_cal)` —
mathematically equivalent when `c_water = 1 + (HU_low − α_low_cal·c_iod)/1000`
(the algebraic identity used in nb03 / nb04 §13's `de_decomp`), but
makes the basis-pair structure explicit.

`c_water` is in g/mL (≈1 for water, ≈1.5–2 for calcium-equivalent rods).
`c_iodine` is in mg/mL.
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


"""
    decomp_2basis(HU_low, HU_high, cal) -> NamedTuple{(:c_iodine, :c_water)}

Convenience wrapper that runs the 2-basis decomposition for a calibration
NamedTuple as returned by [`de_vmi_cal_2basis_for`](@ref) /
[`pcct_vmi_cal_2basis_for`](@ref).  Equivalent to:

```julia
c_iodine = apply_cal(HU_low, HU_high, cal.coeffs_iodine; form = cal.cal_form)
c_water  = @. 1f0 + (HU_low − cal.α_iod_low_cal · c_iodine) / 1000f0
```

The `c_water` identity ties basis-fraction noise to ONE HU term
(`HU_low`), preserving the iodine-perturbation form's noise behavior
while still feeding the proper textbook 2-basis μ-domain synth via
[`synth_vmi_2basis`](@ref).
"""
function decomp_2basis(
        HU_low::AbstractArray{Float32, 3},
        HU_high::AbstractArray{Float32, 3},
        cal::NamedTuple,
    )
    c_iodine = apply_cal(HU_low, HU_high, cal.coeffs_iodine; form = cal.cal_form)
    α_low_f32 = Float32(cal.α_iod_low_cal)
    c_water = @. 1.0f0 + (HU_low - α_low_f32 * c_iodine) / 1000.0f0
    (c_iodine = c_iodine, c_water = c_water)
end


export eval_cal, apply_cal!, apply_cal,
       synth_vmi_2basis!, synth_vmi_2basis,
       decomp_2basis
