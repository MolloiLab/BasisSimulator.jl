"""
PCCT polynomial calibration for dual-material (water, iodine) sinogram-domain
decomposition (Alvarez & Macovski 1976).

Pipeline:
  1. Build per-bin effective spectra from the source spectrum folded with
     detector quantum efficiency and the Monte-Carlo detector response
     matrix (DRM) columns for the chosen bin group (low / high).
  2. Forward-project a synthetic step-wedge (water × iodine grid) through
     each bin's effective spectrum to collect
     `(t_water, t_iodine) → (p_low, p_high)` samples.
  3. Fit the **inverse** 4th-order polynomial
     `(p_low, p_high) → (t_water, t_iodine)` by least squares.

The resulting polynomial coefficients are consumed per-ray by
`apply_pcct_vmi_poly!` to seed the RWLS/PWLS warm start with a cheap,
beam-hardening-linearised material decomp.

Reference:
  Alvarez, Macovski (1976) *Phys Med Biol* 21(5):733–744.
  Cardinal & Fenster (1990) Chebyshev-spaced calibration grid.
"""

"""
    calibrate_pcct_vmi_poly(scanner, protocol;
                            sim_opts,
                            low_bins = 1:2,
                            high_bins = nothing,
                            order = 4,
                            n_water = 40, n_iodine = 25,
                            max_water_cm = 50.0, max_iodine_g_cm2 = 0.15,
                            water_material = XA.Materials.water,
                            iodine_material = XA.Elements.Iodine,
                            verbose = true) -> NamedTuple

Build the polynomial calibration mapping `(p_low, p_high) → (t_water, t_iodine)`.

# Arguments
- `scanner`   : `BS.Scanner` — supplies PCCT detector geometry for DRM.
- `protocol`  : `BS.CTProtocol` — kVp + filter chain for the source spectrum.

# Keyword arguments
- `sim_opts`        : `BS.SimOptions`; its flags affect the resolved spectrum.
- `low_bins`        : bin indices grouped as the "low" channel (default `1:2`).
- `high_bins`       : bin indices for the "high" channel.  `nothing` →
  `(last(low_bins)+1):n_bins`.
- `order`           : bivariate polynomial total order (default 4).
- `n_water`, `n_iodine` : Chebyshev grid sizes in water / iodine directions.
- `max_water_cm`    : maximum water path in the step-wedge (cm).
- `max_iodine_g_cm2`: maximum iodine area density (g/cm²).
- `water_material`, `iodine_material` : `XA` materials for `μ/ρ` lookup.

Returns a NamedTuple with fields matching the notebook:
`(coeffs_w, coeffs_I, terms, E_low, E_high, rms_water, rms_iodine,
  low_bins, high_bins, order)`.
"""
function calibrate_pcct_vmi_poly(
        scanner,
        protocol;
        sim_opts,
        low_bins = 1:2,
        high_bins = nothing,
        order::Integer = 4,
        n_water::Integer = 40,
        n_iodine::Integer = 25,
        max_water_cm::Real = 50.0,
        max_iodine_g_cm2::Real = 0.15,
        water_material = XA.Materials.water,
        iodine_material = XA.Elements.Iodine,
        verbose::Bool = true,
    )
    e_full, w_full = resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)

    pcct_det = _build_pcct_detector(scanner)
    kVp = Float64(maximum(e_full))
    R_mat = compute_mc_drm(pcct_det, kVp)
    η_vec = quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)

    # Map each spectrum energy to its nearest DRM row.
    n_R = size(R_mat, 1)
    drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)

    n_bins = size(R_mat, 2)
    high_bins_resolved = high_bins === nothing ? ((last(low_bins) + 1):n_bins) : high_bins

    # w_eff(E) = w(E) · η(E) · Σ_{b ∈ group} R(E, b)
    e = Float64.(e_full)
    w_low  = [Float64(w_full[i]) * Float64(η_vec[i]) *
              sum(R_mat[drm_row(e[i]), b] for b in low_bins)            for i in eachindex(e)]
    w_high = [Float64(w_full[i]) * Float64(η_vec[i]) *
              sum(R_mat[drm_row(e[i]), b] for b in high_bins_resolved)  for i in eachindex(e)]

    # Probability-weighted spectra (sum to 1 for the forward model).
    wn_l = w_low  ./ sum(w_low)
    wn_h = w_high ./ sum(w_high)

    μρ_w = [compute_mass_μ_at_energy(water_material,  E) for E in e]
    μρ_I = [compute_mass_μ_at_energy(iodine_material, E) for E in e]

    # Chebyshev-spaced calibration grid (Cardinal & Fenster 1990).  Pad with
    # 0 to anchor the polynomial at the origin.
    cheb(n, xmax) = [xmax / 2 * (1 - cos((2m - 1) / (2n) * π)) for m in 1:n]
    tw_vec = vcat(0.0, cheb(n_water  - 1, Float64(max_water_cm)))
    tI_vec = vcat(0.0, cheb(n_iodine - 1, Float64(max_iodine_g_cm2)))

    # Forward model per grid point:
    #   p_m = -log(Σ wn_m(E) · exp(-μρ_w(E)·tw - μρ_I(E)·tI))
    N = length(tw_vec) * length(tI_vec)
    p_low    = zeros(N)
    p_high   = zeros(N)
    t_water  = zeros(N)
    t_iodine = zeros(N)
    idx = 0
    for tI in tI_vec, tw in tw_vec
        idx += 1
        t_water[idx]  = tw
        t_iodine[idx] = tI
        tr_l = sum(wn_l[i] * exp(-μρ_w[i] * tw - μρ_I[i] * tI) for i in eachindex(wn_l))
        tr_h = sum(wn_h[i] * exp(-μρ_w[i] * tw - μρ_I[i] * tI) for i in eachindex(wn_h))
        p_low[idx]  = -log(max(tr_l, 1e-30))
        p_high[idx] = -log(max(tr_h, 1e-30))
    end

    # Bivariate total-order-`order` polynomial basis: (i + j) ≤ order.
    terms = [(i, j) for i in 0:order for j in 0:(order - i)]
    A_mat = hcat([p_low .^ i .* p_high .^ j for (i, j) in terms]...)
    coeffs_w = A_mat \ t_water
    coeffs_I = A_mat \ t_iodine

    pred_w = A_mat * coeffs_w
    pred_I = A_mat * coeffs_I
    rms_water  = sqrt(mean((pred_w .- t_water)  .^ 2))
    rms_iodine = sqrt(mean((pred_I .- t_iodine) .^ 2))

    E_low  = sum(e .* w_low)  / sum(w_low)
    E_high = sum(e .* w_high) / sum(w_high)

    if verbose
        @info "[calibrate_pcct_vmi_poly] RMS water=$(round(rms_water, sigdigits = 3)) cm, iodine=$(round(rms_iodine, sigdigits = 3)) g/cm²"
        @info "  effective mean energies: low=$(round(E_low, digits = 1)) keV (bins $(collect(low_bins))), high=$(round(E_high, digits = 1)) keV (bins $(collect(high_bins_resolved)))"
    end

    (coeffs_w  = coeffs_w,
     coeffs_I  = coeffs_I,
     terms     = terms,
     E_low     = E_low,
     E_high    = E_high,
     rms_water = rms_water,
     rms_iodine = rms_iodine,
     low_bins  = collect(low_bins),
     high_bins = collect(high_bins_resolved),
     order     = Int(order))
end

"""
    apply_pcct_vmi_poly!(sino_water, sino_iodine, sino_low, sino_high, cal)

Per-ray polynomial evaluation of the calibration `cal` produced by
`calibrate_pcct_vmi_poly`.  All four arrays must share shape and element
type (default `Float32`).  Negative evaluations are clipped to 0.

Mutates `sino_water` and `sino_iodine` in place.
"""
function apply_pcct_vmi_poly!(
        sino_water::AbstractArray{T, 3},
        sino_iodine::AbstractArray{T, 3},
        sino_low::AbstractArray{T, 3},
        sino_high::AbstractArray{T, 3},
        cal,
    ) where {T <: AbstractFloat}
    coeffs_w = cal.coeffs_w
    coeffs_I = cal.coeffs_I
    terms    = cal.terms

    @inline function _eval_poly(coeffs, terms, pl::Float64, ph::Float64)
        s = 0.0
        @inbounds for k in eachindex(coeffs)
            i, j = terms[k]
            s += coeffs[k] * pl ^ i * ph ^ j
        end
        s
    end

    @inbounds Threads.@threads for idx in eachindex(sino_low)
        pl = Float64(sino_low[idx])
        ph = Float64(sino_high[idx])
        aw = _eval_poly(coeffs_w, terms, pl, ph)
        aI = _eval_poly(coeffs_I, terms, pl, ph)
        sino_water[idx]  = T(max(aw, 0.0))
        sino_iodine[idx] = T(max(aI, 0.0))
    end
    (sino_water, sino_iodine)
end

"""
    apply_pcct_vmi_poly(sino_low, sino_high, cal) -> (sino_water, sino_iodine)

Allocating wrapper around `apply_pcct_vmi_poly!`.  Returns a fresh
`(sino_water, sino_iodine)` pair with the same shape and element type as
the inputs.
"""
function apply_pcct_vmi_poly(
        sino_low::AbstractArray{T, 3},
        sino_high::AbstractArray{T, 3},
        cal,
    ) where {T <: AbstractFloat}
    sino_water  = similar(sino_low)
    sino_iodine = similar(sino_low)
    apply_pcct_vmi_poly!(sino_water, sino_iodine, sino_low, sino_high, cal)
    (sino_water, sino_iodine)
end

export calibrate_pcct_vmi_poly, apply_pcct_vmi_poly!, apply_pcct_vmi_poly
