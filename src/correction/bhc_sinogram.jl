# =============================================================================
# Sinogram-Domain Beam Hardening Correction (BHC)
# =============================================================================
#
# Water-based polynomial BHC + bowtie-aware two-material (water + bone) BHC,
# both operating on the log-line-integral sinogram before reconstruction.
#
# ## Physics Background
#
# X-ray beams from clinical CT scanners are polychromatic (bremsstrahlung +
# characteristic lines). As the beam traverses tissue, lower-energy photons
# are preferentially absorbed, causing the mean energy to increase ("harden").
# This energy shift causes the measured line integral to be less than what
# would be measured with a monochromatic beam of the same effective energy,
# producing cupping artifacts (~50–100 HU in uncorrected water) and inaccurate
# HU values for dense materials.
#
# ## Correction Method
#
# **Water-based polynomial BHC** — the standard approach used in clinical CT,
# including CatSim/XCIST (`Prep_BHC_Accurate.py`):
#
#   1. Generate calibration curve: simulate polychromatic transmission through
#      water of varying thickness (0 to ~50 cm).
#   2. For each path d, compute the measured polychromatic line integral
#      `p_meas = -log(Σ wᵢ exp(-μᵢ d))` and the monochromatic-equivalent
#      `p_true = μ_ref · d` at the reference energy.
#   3. Fit `p_true = a₀ + a₁ p_meas + … + aₙ p_meas^n` by least squares.
#   4. Apply the polynomial to each sinogram value before reconstruction.
#
# **Bowtie awareness** — CatSim fits per-detector polynomial coefficients to
# account for bowtie-induced fan-angle path variation. BasisSim does the same
# via `TwoMaterialBHCPerColumn`, fitting one polynomial per detector column
# from the per-column spectrum produced by `resolve_source_spectrum_with_bowtie`.
#
# **Two-material (water + bone) refinement** — water-only BHC under-corrects
# bone, leaving dark-band artifacts between dense structures. The two-material
# scheme adds an image-domain bone-fraction segmentation pass (Elbakri & Fessler
# 2003 smoothstep) and a second polychromatic-forward subtraction step.
#
# ## Provenance
#
# Algorithm port pointer for the polynomial-evaluation kernel:
#   - CatSim/XCIST `gecatsim/pyfiles/Prep_BHC_Accurate.py` lines 18-23
#     (Horner-style loop over coefficients, in-place per view).
#     BSD 3-Clause, GE Precision HealthCare.
#
# ## References
#
# 1. Joseph PM, Spital RD. "A method for correcting bone induced artifacts
#    in computed tomography scanners." J Comput Assist Tomogr.
#    1978;2(1):100-108. doi:10.1097/00004728-197801000-00017
# 2. Herman GT. "Correction for beam hardening in computed tomography."
#    Phys Med Biol. 1979;24(1):81-106. doi:10.1088/0031-9155/24/1/008
# 3. Brooks RA, Di Chiro G. "Beam hardening in x-ray reconstructive
#    tomography." Phys Med Biol. 1976;21(3):390-398.
#    doi:10.1088/0031-9155/21/3/004
# 4. Elbakri IA, Fessler JA. "Segmentation-free statistical image
#    reconstruction for polyenergetic x-ray computed tomography with
#    experimental validation." Phys Med Biol. 2003;48(15):2453-2477.
#    (smooth two-material fraction)
# 5. Martinez C, Fessler JA, et al. "Bowtie-aware two-material beam
#    hardening correction." Phys Med Biol. 2022;67(11). (2DCalBH per-column)
# 6. CatSim/XCIST: https://github.com/xcist/main — `Prep_BHC_Accurate.py`.
# =============================================================================

import AcceleratedKernels as AK
import XrayAttenuation as XA
using Unitful: ustrip, @u_str

export BHCPolynomial, BeamHardeningCorrection, TwoMaterialBHCPerColumn
export calibrate_bhc, apply_bhc!
export calibrate_bhc_two_material, apply_bhc_two_material
export bhc_spectrum_per_column, compute_polychromatic_μ_water

# =============================================================================
# BHC Types
# =============================================================================

"""
    BHCPolynomial

Polynomial beam hardening correction coefficients.

    p_corrected = a₀ + a₁·p + a₂·p² + … + aₙ·pⁿ

`p` is the measured polychromatic line integral, `p_corrected` the
monochromatic-equivalent value at `reference_energy_keV`.

# Fields
- `coefficients::Vector{Float64}`: `[a₀, a₁, …, aₙ]` ascending order.
- `order::Int`: polynomial order (`length(coefficients) - 1`). Typical 3-5.
- `reference_energy_keV::Float64`: reference energy for the monochromatic
  target. **Diagnostic metadata** — not read by the apply kernel.

CatSim (`Prep_BHC_Accurate.py`) uses the same polynomial formulation; default
order in CatSim is 5.
"""
struct BHCPolynomial
    coefficients::Vector{Float64}
    order::Int
    reference_energy_keV::Float64
end

"""
    BeamHardeningCorrection

Polynomial BHC + raw calibration data, returned by `calibrate_bhc`.

# Fields
- `polynomial::BHCPolynomial`: fitted correction polynomial.
- `calibration_paths::Vector{Float64}`: water path lengths (cm). **Diagnostic.**
- `calibration_measured::Vector{Float64}`: polychromatic line integrals at
  each path. **Diagnostic.**
- `calibration_true::Vector{Float64}`: monochromatic line integrals at each
  path. **Diagnostic.**

The three `calibration_*` fields are stashed for offline plotting / quality
assessment and are not read by any apply path.
"""
struct BeamHardeningCorrection
    polynomial::BHCPolynomial
    calibration_paths::Vector{Float64}
    calibration_measured::Vector{Float64}
    calibration_true::Vector{Float64}
end

"""
    TwoMaterialBHCPerColumn

Bowtie-aware two-material BHC. One polynomial per detector column, each fit
from the spectrum that column actually saw (tube × filters × bowtie at that
fan angle), plus the per-column normalized spectrum and global μ refs needed
for the bone-segmentation second stage.

# Fields
- `water_bhc_per_col::Vector{BeamHardeningCorrection}`: per-column water poly.
  Length equals the sinogram's `n_col`.
- `energies::Vector{Float64}`: energy grid (keV).
- `w_norm_per_col::Matrix{Float64}`: `[n_E, n_col]` normalized spectrum.
- `μ_water_E`, `μ_bone_E::Vector{Float64}`: per-energy μ for water / cortical
  bone (cm⁻¹).
- `μ_water_ref`, `μ_bone_ref::Float64`: global μ at `reference_energy_keV`.
  The mono-equivalent sinogram produced by the per-column polynomials targets
  this single global reference so FBP downstream sees one effective energy.
- `reference_energy_keV::Float64`: **diagnostic metadata** (not read by apply).
- `hu_low`, `hu_high::Float64`: bone-fraction smoothstep thresholds (HU).
"""
struct TwoMaterialBHCPerColumn
    water_bhc_per_col::Vector{BeamHardeningCorrection}
    energies::Vector{Float64}
    w_norm_per_col::Matrix{Float64}      # [n_E, n_col]
    μ_water_E::Vector{Float64}
    μ_bone_E::Vector{Float64}
    μ_water_ref::Float64
    μ_bone_ref::Float64
    reference_energy_keV::Float64
    hu_low::Float64
    hu_high::Float64
end

# =============================================================================
# Water Calibration Curve Generation
# =============================================================================

"""
    generate_water_calibration_curve(energies, weights; max_path_cm=50.0, n_points=100, reference_energy_keV=70.0)

Generate the BHC calibration curve `(paths, measured, true_values)` by
simulating polychromatic transmission through water of varying thickness.

For each path `d` in `range(0, max_path_cm; length=n_points)`:

  - `measured[i] = -log(Σ wᵢ_norm · exp(-μ_water(Eᵢ) · d))`
  - `true_values[i] = μ_water(reference_energy_keV) · d`

`weights` are normalized internally. μ(E) comes from XrayAttenuation.jl
(NIST XCOM).
"""
function generate_water_calibration_curve(
        energies::Vector,
        weights::Vector;
        max_path_cm::Real = 50.0,
        n_points::Int = 100,
        reference_energy_keV::Real = 70.0,
    )
    water = get_material(:water)
    μ_water = [compute_μ_at_energy(water, Float64(e)) for e in energies]
    μ_water_ref = compute_μ_at_energy(water, Float64(reference_energy_keV))

    weights_norm = weights ./ sum(weights)
    paths = collect(range(0.0, Float64(max_path_cm), length = n_points))

    measured = zeros(Float64, n_points)
    true_values = zeros(Float64, n_points)
    for (i, path) in enumerate(paths)
        I_poly = sum(weights_norm .* exp.(-μ_water .* path))
        measured[i] = -log(max(I_poly, 1.0e-10))
        true_values[i] = μ_water_ref * path
    end
    return (paths, measured, true_values)
end

# =============================================================================
# Polynomial Fitting (least-squares via normal equations)
# =============================================================================

"""
    fit_polynomial(x, y, order) -> coefficients

Fit `y ≈ a₀ + a₁·x + … + aₙ·xⁿ` by ordinary least squares, returning
`[a₀, a₁, …, aₙ]`.
"""
function fit_polynomial(x::Vector, y::Vector, order::Int)
    n = length(x)
    V = zeros(n, order + 1)
    for i in 1:n, j in 0:order
        V[i, j + 1] = x[i]^j
    end
    return (V' * V) \ (V' * y)
end

# =============================================================================
# Single-Spectrum BHC Calibration (inner helper for per-column flow)
# =============================================================================

"""
    calibrate_bhc(energies, weights; order=3, max_path_cm=50.0, n_points=100, reference_energy_keV=70.0)
        -> BeamHardeningCorrection

Calibrate water-based BHC from a single 1-D source spectrum by fitting a
polynomial to the water calibration curve.

This is the inner helper used by the per-column two-material flow; one call
per detector column. Notebooks should use the high-level
`calibrate_bhc_two_material(sim_opts, protocol; …)` entry instead, which
handles the bowtie + per-column dispatch.
"""
function calibrate_bhc(
        energies::Vector,
        weights::Vector;
        order::Int = 3,
        max_path_cm::Real = 50.0,
        n_points::Int = 100,
        reference_energy_keV::Real = 70.0,
    )
    paths, measured, true_values = generate_water_calibration_curve(
        energies, weights;
        max_path_cm = max_path_cm,
        n_points = n_points,
        reference_energy_keV = reference_energy_keV,
    )
    coefficients = fit_polynomial(measured, true_values, order)
    polynomial = BHCPolynomial(coefficients, order, Float64(reference_energy_keV))
    return BeamHardeningCorrection(polynomial, paths, measured, true_values)
end

# =============================================================================
# BHC Application (per-column polynomial, GPU kernel)
# =============================================================================
#
# Direct port of `gecatsim/pyfiles/Prep_BHC_Accurate.py` lines 18-23: Horner-
# style polynomial evaluation in-place per view. Difference: BasisSim stacks
# coefficients into a [order+1, n_col] device matrix once and indexes per-ray
# via `col = (idx-1) % n_col + 1`; CatSim loops over views in Python and uses
# numpy broadcasting.
# =============================================================================

"""
    apply_bhc!(sinogram, bhcs_per_col::AbstractVector{BeamHardeningCorrection}; ws_coeffs_gpu=nothing)

Per-column (bowtie-aware) sinogram BHC. Convenience wrapper that unwraps the
`BeamHardeningCorrection` objects and delegates to the `BHCPolynomial`-vector
method below.
"""
function apply_bhc!(
        sinogram::AbstractArray{T, 3},
        bhcs_per_col::AbstractVector{BeamHardeningCorrection};
        ws_coeffs_gpu = nothing,
    ) where {T <: AbstractFloat}
    polys = [b.polynomial for b in bhcs_per_col]
    return apply_bhc!(sinogram, polys; ws_coeffs_gpu = ws_coeffs_gpu)
end

"""
    apply_bhc!(sinogram, polys_per_col::AbstractVector{BHCPolynomial}; ws_coeffs_gpu=nothing) -> sinogram

Per-column (bowtie-aware) BHC: applies a different polynomial to each
detector column, indexed off the sinogram's BS storage `[n_col, n_row, n_view]`
layout. All polynomials must share the same `order`.
"""
function apply_bhc!(
        sinogram::AbstractArray{T, 3},
        polys_per_col::AbstractVector{BHCPolynomial};
        ws_coeffs_gpu = nothing,
    ) where {T <: AbstractFloat}

    n_col = size(sinogram, 1)
    length(polys_per_col) == n_col ||
        error("apply_bhc!: polys_per_col length $(length(polys_per_col)) ≠ n_col $n_col")
    order = polys_per_col[1].order
    for c in 2:n_col
        polys_per_col[c].order == order ||
            error("apply_bhc!: polynomial orders must match across columns (col $c has order $(polys_per_col[c].order), expected $order)")
    end

    coeffs_cpu = zeros(T, order + 1, n_col)
    for col_idx in 1:n_col
        coeffs_cpu[:, col_idx] .= T.(polys_per_col[col_idx].coefficients)
    end
    coeffs = if ws_coeffs_gpu !== nothing
        copyto!(ws_coeffs_gpu, coeffs_cpu)
        ws_coeffs_gpu
    else
        cbuf = similar(sinogram, T, order + 1, n_col)
        copyto!(cbuf, coeffs_cpu)
        cbuf
    end

    # NOTE: renamed the local `col` to avoid collision with any outer `c`
    # scoped variable — same-name assignments in closures trigger Julia's
    # Core.Box heuristic and break GPU kernel compile with "non-bitstype
    # argument" errors.
    let coeffs = coeffs, order = order, n_col = n_col
        AK.foreachindex(sinogram) do idx
            col = ((idx - 1) % n_col) + 1
            p = sinogram[idx]
            p_corrected = coeffs[1, col]
            p_power = one(T)
            for i in 1:order
                p_power *= p
                p_corrected += coeffs[i + 1, col] * p_power
            end
            sinogram[idx] = p_corrected
        end
    end
    return sinogram
end

# =============================================================================
# Two-Material (Water + Bone) BHC — Per-Column (Bowtie-Aware)
# =============================================================================

"""
    calibrate_bhc_two_material(sim_opts, protocol; scanner, geom, kwargs...)
        -> TwoMaterialBHCPerColumn

**High-level entry — bowtie-aware by construction.**
Auto-resolves the bowtie-hardened source spectrum via
`resolve_source_spectrum_with_bowtie`, collapses to per-column weights via
`bhc_spectrum_per_column`, and dispatches to the per-column polynomial fit.
This captures the column-dependent spectral shaping any modern bowtie filter
imposes — a single global polynomial leaves residual radial cupping behind
that this path eliminates.

If `reference_energy_keV` is `nothing` (default), it's set to the column-mean
spectrum's mean energy so all columns share a single mono-equivalent target.

# Example
```julia
bhc = calibrate_bhc_two_material(
    sim_opts, protocol;
    scanner = scanner, geom = ws.geom,
    order   = 2,
    hu_low  = 450.0,
    hu_high = 600.0,
)
```

    calibrate_bhc_two_material(energies, weights_per_col::AbstractMatrix; kwargs...)
        -> TwoMaterialBHCPerColumn

Lower-level variant when you already have a `[n_E, n_col]` per-column spectrum
matrix — same path the high-level method dispatches into.
"""
function calibrate_bhc_two_material(
        sim_opts,
        protocol;
        scanner,
        geom,
        order::Int = 5,
        max_path_cm::Real = 50.0,
        n_points::Int = 100,
        reference_energy_keV::Union{Nothing, Real} = nothing,
        hu_low::Real = 100.0,
        hu_high::Real = 500.0,
    )
    e, ŵ = resolve_source_spectrum_with_bowtie(
        sim_opts, protocol; scanner = scanner, geom = geom,
    )
    e, w_col = bhc_spectrum_per_column(e, ŵ)

    ref_E = if reference_energy_keV === nothing
        w_mean = vec(sum(w_col; dims = 2)) ./ size(w_col, 2)
        sum(e .* w_mean) / sum(w_mean)
    else
        Float64(reference_energy_keV)
    end

    return calibrate_bhc_two_material(
        e, w_col;
        order = order,
        max_path_cm = max_path_cm,
        n_points = n_points,
        reference_energy_keV = ref_E,
        hu_low = hu_low,
        hu_high = hu_high,
    )
end

function calibrate_bhc_two_material(
        energies::Vector,
        weights_per_col::AbstractMatrix;
        order::Int = 5,
        max_path_cm::Real = 50.0,
        n_points::Int = 100,
        reference_energy_keV::Real = 70.0,
        hu_low::Real = 100.0,
        hu_high::Real = 500.0,
    )
    n_E, n_col = size(weights_per_col)
    length(energies) == n_E ||
        error("calibrate_bhc_two_material: energies length $(length(energies)) ≠ weights_per_col n_E $n_E")

    water_bhc_per_col = Vector{BeamHardeningCorrection}(undef, n_col)
    w_norm_per_col = zeros(Float64, n_E, n_col)
    for c in 1:n_col
        w_c = Float64.(@view weights_per_col[:, c])
        w_norm_per_col[:, c] .= w_c ./ sum(w_c)
        water_bhc_per_col[c] = calibrate_bhc(
            energies, w_c;
            order = order, max_path_cm = max_path_cm,
            n_points = n_points, reference_energy_keV = reference_energy_keV,
        )
    end

    water_mat = XA.Materials.water
    bone_mat = XA.Materials.corticalbone
    E_vec = Float64.(energies)
    μ_water_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, E * u"keV")) for E in E_vec]
    μ_bone_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, E * u"keV")) for E in E_vec]
    μ_water_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, reference_energy_keV * u"keV"))
    μ_bone_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, reference_energy_keV * u"keV"))

    @info "[calibrate_bhc_two_material] bowtie-aware: $n_col columns × order-$order polynomials, ref_E = $reference_energy_keV keV"
    return TwoMaterialBHCPerColumn(
        water_bhc_per_col, E_vec, w_norm_per_col,
        μ_water_E, μ_bone_E,
        μ_water_ref, μ_bone_ref,
        Float64(reference_energy_keV),
        Float64(hu_low), Float64(hu_high),
    )
end

"""
    bhc_spectrum_per_column(e, w) -> (e, w_per_col)

Normalize a source-spectrum array to the `[n_E, n_col]` shape that
`calibrate_bhc_two_material(::Vector, ::Matrix; …)` expects.

- 1-D `w[n_E]` → returned as-is (caller must duplicate per column).
- 2-D `w[n_E, n_col]` or `w[n_col, n_E]` → auto-transpose to `[n_E, n_col]`.
- 3-D bowtie-aware `w[n_col, n_row, n_E]` (from
  `resolve_source_spectrum_with_bowtie`) → take the center detector row,
  return `[n_E, n_col]`. Bowtie variation is primarily in-plane (fan angle);
  row-direction variation is small.
"""
function bhc_spectrum_per_column(e::AbstractVector, w::AbstractArray)
    if ndims(w) == 1
        return e, w
    elseif ndims(w) == 2
        size(w, 1) == length(e) && return e, w
        size(w, 2) == length(e) && return e, permutedims(w)
        error("bhc_spectrum_per_column: 2D spectrum shape $(size(w)) doesn't match energies length $(length(e))")
    elseif ndims(w) == 3
        n_col, n_row, n_E = size(w)
        n_E == length(e) ||
            error("bhc_spectrum_per_column: 3D spectrum last-dim $n_E ≠ energies length $(length(e))")
        mid_r = n_row ÷ 2 + 1
        out = Array{Float64}(undef, n_E, n_col)
        @inbounds for c in 1:n_col, k in 1:n_E
            out[k, c] = Float64(w[c, mid_r, k])
        end
        return e, out
    else
        error("bhc_spectrum_per_column: unsupported ndims $(ndims(w))")
    end
end

"""
    compute_polychromatic_μ_water(sim_opts, protocol; scanner, geom, water_path_cm) -> Float64

Spectrum-analytic polychromatic-effective `μ_water` for an in-phantom voxel —
the analytic counterpart to "average μ in a solid-water ROI of the FBP volume".
Use this as the HU-conversion reference for image-domain pipelines that don't
run BHC.

Pipeline:
1. Resolve the bowtie-aware source spectrum via
   `resolve_source_spectrum_with_bowtie` — picks up the scanner flat filter,
   `protocol.additional_filters`, and the bowtie attenuation.
2. Collapse the per-ray ŵ to the central-column 1D fluence profile via
   `bhc_spectrum_per_column`.
3. **Pre-harden** the spectrum by `water_path_cm` of solid water at each
   energy bin via Beer-Lambert (low-E photons drop out preferentially,
   raising the mean energy — same effect a center-of-phantom voxel sees).
4. Return the fluence-weighted μ_water on the hardened spectrum.

# Keyword arguments
- `scanner::Scanner` — flat filter + bowtie geometry.
- `geom::CTGeometry` — for bowtie attenuation per column.
- `water_path_cm::Real` — total chord-length of solid water seen by an FBP
  line integral through the reference voxel. For a center voxel, the chord
  crosses the **full phantom diameter** (one entry + one exit half), so pass
  the phantom *diameter*, e.g. `33.0` for a Gammex 472 (33 cm body), not
  the radius.
"""
function compute_polychromatic_μ_water(
        sim_opts,
        protocol;
        scanner,
        geom,
        water_path_cm::Real,
    )
    e, ŵ = resolve_source_spectrum_with_bowtie(
        sim_opts, protocol;
        scanner = scanner, geom = geom, include_bowtie = true,
        label = "compute_polychromatic_μ_water $(Int(protocol.kVp)) kVp",
    )
    _, w_per_col = bhc_spectrum_per_column(e, ŵ)
    mid_c = size(w_per_col, 2) ÷ 2 + 1
    w_entr = Float64.(@view w_per_col[:, mid_c])
    μ_per_E = Float64[compute_μ_at_energy(XA.Materials.water, Float64(eᵢ)) for eᵢ in e]
    w_hard = w_entr .* exp.(-μ_per_E .* Float64(water_path_cm))
    return sum(w_hard .* μ_per_E) / sum(w_hard)
end

"""
    apply_bhc_two_material(sinogram_raw, bhc::TwoMaterialBHCPerColumn, geom, matrix_size)

Bowtie-aware sinogram-domain two-material BHC.

# Algorithm (3 stages)
1. **Per-column water polynomial** — apply `bhc.water_bhc_per_col` to the
   sinogram, giving a water-only mono-equivalent estimate.
2. **Bone-fraction segmentation** — FDK-reconstruct the water-corrected sino,
   convert μ → HU using `bhc.μ_water_ref`, and compute a smooth bone fraction
   via the Elbakri/Fessler 2003 C1-continuous smoothstep `t²·(3 - 2t)` on
   `[hu_low, hu_high]`. Forward-project the bone-fraction-weighted μ image.
3. **Per-column polychromatic forward subtraction** — for each pixel, solve
   the polychromatic forward model with per-column normalized spectrum
   `w_norm_per_col` and add the (mono − poly) residual to the original
   sinogram.

# Note on geometry
Stage 2's internal FDK + Siddon round-trip both use `geom`'s recon FOV. An
earlier signature accepted `volume_extent=phantom.extent` and forwarded it to
the forward projector only — the matching FDK call always used `geom`'s FOV,
which caused a grid mismatch and bone-shaped halo artifacts whenever the two
extents differed (e.g. nb02's XCAT body, where `phantom.extent` ≠ recon FOV).
The kwarg has been removed.
"""
function apply_bhc_two_material(
        sinogram_raw::AbstractArray{T, 3},
        bhc::TwoMaterialBHCPerColumn,
        geom,
        matrix_size::Tuple{Int, Int, Int},
    ) where {T <: AbstractFloat}

    n_col = size(sinogram_raw, 1)
    length(bhc.water_bhc_per_col) == n_col ||
        error("apply_bhc_two_material(PerColumn): polys $(length(bhc.water_bhc_per_col)) ≠ sino n_col $n_col")

    # Stage 1 — per-column polynomial water BHC.
    sino_water = similar(sinogram_raw)
    copyto!(sino_water, sinogram_raw)
    apply_bhc!(sino_water, bhc.water_bhc_per_col)
    recon_gpu = fdk_reconstruct(sino_water, geom, matrix_size)

    μ_w_ref = T(bhc.μ_water_ref)
    μ_b_ref = T(bhc.μ_bone_ref)
    hu_low_T = T(bhc.hu_low)
    hu_high_T = T(bhc.hu_high)

    # Stage 2 — bone-fraction segmentation in image domain.
    # Smooth bone fraction via Elbakri/Fessler 2003 C1 smoothstep
    # (3t² - 2t³ on [hu_low, hu_high]); avoids hard-segmentation artifacts.
    bone_μ_gpu = similar(recon_gpu)
    let recon_gpu = recon_gpu, bone_μ_gpu = bone_μ_gpu,
            μ_w_ref = μ_w_ref, hu_low_T = hu_low_T, hu_high_T = hu_high_T
        AK.foreachindex(recon_gpu) do idx
            μ_val = recon_gpu[idx]
            hu_val = T(1000) * (μ_val - μ_w_ref) / μ_w_ref
            bf = if hu_val <= hu_low_T
                zero(T)
            elseif hu_val >= hu_high_T
                one(T)
            else
                t = (hu_val - hu_low_T) / (hu_high_T - hu_low_T)
                t * t * (T(3) - T(2) * t)
            end
            bone_μ_gpu[idx] = bf * μ_val
        end
    end

    p_b_gpu = siddon_forward_project(bone_μ_gpu, geom)

    p_s_gpu = similar(sino_water)
    copyto!(p_s_gpu, sino_water)
    let p_s_gpu = p_s_gpu, p_b_gpu = p_b_gpu
        AK.foreachindex(p_s_gpu) do idx
            p_s_gpu[idx] = p_s_gpu[idx] - p_b_gpu[idx]
        end
    end

    # Stage 3 — per-column polychromatic forward correction.  The w_norm
    # matrix is [n_E, n_col]; kernel indexes w_norm[e, c] for col c.
    n_e = length(bhc.energies)
    μ_w_cpu = T.(bhc.μ_water_E)
    μ_b_cpu = T.(bhc.μ_bone_E)
    w_norm_cpu = T.(bhc.w_norm_per_col)  # [n_E, n_col]
    μ_w_gpu = similar(sinogram_raw, T, n_e);            copyto!(μ_w_gpu, μ_w_cpu)
    μ_b_gpu = similar(sinogram_raw, T, n_e);            copyto!(μ_b_gpu, μ_b_cpu)
    w_norm_gpu = similar(sinogram_raw, T, n_e, n_col);  copyto!(w_norm_gpu, w_norm_cpu)

    sino_out = similar(sinogram_raw)
    copyto!(sino_out, sinogram_raw)

    let sino_out = sino_out, p_s_gpu = p_s_gpu, p_b_gpu = p_b_gpu,
            w_norm_gpu = w_norm_gpu, μ_w_gpu = μ_w_gpu, μ_b_gpu = μ_b_gpu,
            μ_w_ref = μ_w_ref, μ_b_ref = μ_b_ref, n_e = n_e, n_col = n_col
        AK.foreachindex(sino_out) do idx
            col = ((idx - 1) % n_col) + 1
            Lw = max(p_s_gpu[idx] / μ_w_ref, zero(T))
            Lb = max(p_b_gpu[idx] / μ_b_ref, zero(T))
            if Lw > T(1.0e-6) || Lb > T(1.0e-6)
                I_poly = zero(T)
                for e in 1:n_e
                    I_poly += w_norm_gpu[e, col] * exp(-μ_w_gpu[e] * Lw - μ_b_gpu[e] * Lb)
                end
                F_2mat = I_poly > zero(T) ? -log(I_poly) : zero(T)
                p_mono = μ_w_ref * Lw + μ_b_ref * Lb
                sino_out[idx] = sino_out[idx] + (p_mono - F_2mat)
            end
        end
    end

    return Array(sino_out)
end
