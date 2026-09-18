# CT dose: CTDI100, CTDIw, CTDIvol and DLP from the beam the simulator actually uses.
#
# The inputs are the simulator's own tube-side quantities — the absolute IPEM-78 source spectrum
# (photons / mAs / mm²), the flat and added filtration, the bowtie thickness profile, the
# source-to-isocentre distance and the nominal collimation. Nothing about the detector enters:
# an energy-integrating and a photon-counting scan of the same beam deliver the same dose.
#
# Method: a small, deterministic Monte Carlo photon transport in the standard PMMA CTDI phantom
# (16 cm head, 32 cm body, 15 cm long). Primary attenuation alone is not enough — at the centre
# of the body phantom scattered photons deliver about seven times the primary dose — so the
# scatter has to be transported, and a cylinder of one material makes that cheap. Air kerma is
# scored with a collision estimator in (r, z) rings; the gantry rotation is obtained exactly by
# symmetry (the source is fixed and the rotation-averaged dose to a chamber at radius d is a
# weighted sum over the rings it sweeps). One million histories give CTDIw to about 0.4 % in
# about a second on one core, and the result is cached per beam.
#
# Definitions (IEC 60601-2-44):
#   CTDI100 = (1 / N·T) ∫_{-50}^{+50 mm} D_air(z) dz          single rotation, dose to air
#   CTDIw   = ⅓ CTDI100(centre) + ⅔ CTDI100(periphery, 1 cm below the surface)
#   CTDIvol = CTDIw / pitch (helical),  CTDIw · N·T / Δd (axial, table increment Δd)
#   DLP     = CTDIvol · scan length
# For N·T > 40 mm the 100 mm chamber no longer collects the profile, and the IEC wide-beam rule
# is applied: CTDI100(N·T) = CTDI100(20 mm) · CTDI_free-air(N·T) / CTDI_free-air(20 mm).
#
# What is NOT modelled, in decreasing order of effect on the absolute number: the difference
# between the IPEM-78 tube output and a particular real tube (tens of percent, and strongly
# dependent on total filtration); the over-beaming of a real collimator (pass `overbeam_mm`);
# the heel effect; the patient table; Rayleigh scattering and electron binding in Compton
# scattering (under 0.2 % on CTDIw). Use `dose_calibration` to pin a scanner to a measured
# CTDIvol; the uncalibrated beam quantities are reported alongside.

const _KEV_PER_G_TO_MGY = 1.602176634e-10      # 1 keV/g = 1.602e-13 Gy
const _CTDI_CHAMBER_MM = 100.0                 # the pencil ionisation chamber's active length
const _ELECTRON_REST_KEV = 510.99895

# NIST NISTIR-5632 (Hubbell & Seltzer) mass energy-absorption coefficient of dry air, cm²/g.
const _MUEN_AIR_E = [5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
const _MUEN_AIR = [
    39.31, 22.7, 9.446, 4.742, 1.334, 0.5389, 0.1537, 0.06833, 0.04098, 0.03041, 0.02407,
    0.02325, 0.02496, 0.02672,
]

"""
Mass energy-absorption coefficient of dry air (cm²/g) at `E` keV, log-log interpolated on the
NIST table. Outside it the value is extrapolated: below 5 keV as `E^-3`, above 200 keV as the
last tabulated value. Neither matters for a diagnostic beam.
"""
function muen_rho_air(E::Real)
    xs, ys = _MUEN_AIR_E, _MUEN_AIR
    E <= xs[1] && return ys[1] * (xs[1] / E)^3
    i = searchsortedlast(xs, E)
    i >= length(xs) && return ys[end]
    t = log(E / xs[i]) / log(xs[i + 1] / xs[i])
    return exp(log(ys[i]) + t * (log(ys[i + 1]) - log(ys[i])))
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# The beam
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    DoseSource

The tube-side description of a beam, enough to compute its dose: the absolute fluence spectrum
at isocentre after flat and added filtration but before the bowtie (`phi_iso`, photons per mAs
per mm² per energy bin of `E`), the bowtie, the source-to-isocentre distance and the nominal
collimation `N·T` at isocentre. Build it with [`dose_source`](@ref); every workspace carries
one as `ws.dose_source`.
"""
struct DoseSource
    E::Vector{Float64}
    phi_iso::Vector{Float64}
    SAD_mm::Float64
    bowtie_name::Symbol
    bowtie::BowtieFilter
    nominal_collimation_mm::Float64
    kVp::Float64
    filters::Vector{Tuple{String, Float64}}
end

"""
    dose_source(scanner, protocol; spectrum = nothing) -> DoseSource

The beam of `scanner` run with `protocol`. By default the spectrum is the one the simulator
uses (IPEM-78 tungsten anode, Beer-Lambert through the scanner's flat filter and the protocol's
added filters). Pass `spectrum = (energies_keV, weights)` — in the simulator's units, photons
per mAs per mm² AT THE DETECTOR — when a workspace was built with a spectrum override.
"""
function dose_source(scanner::Scanner, protocol::CTProtocol; spectrum = nothing)
    filters = Tuple{String, Float64}[]
    scanner.flat_filter_thickness > 0 && push!(
        filters, (String(scanner.flat_filter_material), Float64(scanner.flat_filter_thickness))
    )
    append!(filters, protocol.additional_filters)
    SAD = Float64(scanner.source_to_isocenter)
    E, phi = if spectrum === nothing
        e, w = load_spectrum_unfiltered(Int(protocol.kVp); anode_angle = protocol.anode_angle)
        filter_spectrum(e, w; filters = filters, sdd_mm = SAD)
    else
        e, w = spectrum
        collect(Float64, e), collect(Float64, w) .* (Float64(scanner.source_to_detector) / SAD)^2
    end
    NT = protocol.collimation_mm === nothing ?
        Float64(scanner.detector_rows * scanner.detector_row_size) : Float64(protocol.collimation_mm)
    return DoseSource(
        collect(Float64, E), collect(Float64, phi), SAD, scanner.bowtie_filter,
        resolve_bowtie_filter(scanner.bowtie_filter), NT, Float64(protocol.kVp), filters,
    )
end

# Bowtie transmission per energy bin at fan angle γ (rad).
function _bowtie_transmission(src::DoseSource, γ::Real)
    t = interpolate_thickness(src.bowtie, Float64(γ))
    return [
        exp(-sum(get_bowtie_mu(src.bowtie.materials[m], E) * t[m] for m in eachindex(t); init = 0.0))
            for E in src.E
    ]
end

"""
    air_kerma_free_in_air(src::DoseSource; fan_angle = 0.0) -> Float64

Air kerma per mAs at isocentre, free in air, behind the bowtie at `fan_angle` (rad), in mGy/mAs:
`K = Σ_E Φ(E) · T_bowtie(E) · E · (μ_en/ρ)_air(E)`.
"""
function air_kerma_free_in_air(src::DoseSource; fan_angle::Real = 0.0)
    T = _bowtie_transmission(src, fan_angle)
    return sum(
        src.phi_iso[k] * 100.0 * T[k] * src.E[k] * muen_rho_air(src.E[k]) for k in eachindex(src.E)
    ) * _KEV_PER_G_TO_MGY
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Photon transport in PMMA
# ─────────────────────────────────────────────────────────────────────────────────────────────

const _DOSE_DE = 0.5                           # keV, cross-section table step
const _DOSE_EMAX = 160.0
const _DOSE_CUTOFF_KEV = 5.0
const _PMMA_DENSITY = 1.19

# Linear attenuation of PMMA on the table grid: total (photoelectric + incoherent) and the
# photoelectric share. Rayleigh scattering is left out: it changes CTDIw by under 0.2 %.
const _PMMA_TABLE = Ref{Union{Nothing, NamedTuple}}(nothing)
function _pmma_table()
    table = _PMMA_TABLE[]
    table === nothing || return table
    Es = collect(_DOSE_DE:_DOSE_DE:_DOSE_EMAX)
    partial(E, kind) = Unitful.ustrip(XA.mass_attenuation_coeff(XA.Materials.pmma, max(E, 1.0) * XA.keV, kind))
    pe = [max(partial(E, XA.PhotoelectricAbsorption()), 0.0) * _PMMA_DENSITY for E in Es]
    incoherent = [partial(E, XA.IncoherentScattering()) * _PMMA_DENSITY for E in Es]
    total = pe .+ incoherent
    table = (mu_total = total, p_photoelectric = pe ./ total, muen_air = muen_rho_air.(Es))
    _PMMA_TABLE[] = table
    return table
end

@inline function _table_lookup(v::Vector{Float64}, E::Float64)
    x = E / _DOSE_DE
    i = clamp(floor(Int, x), 1, length(v) - 1)
    return v[i] + (x - i) * (v[i + 1] - v[i])
end

# Klein-Nishina sampling (free electron): returns (E'/E, cos θ).
@inline function _klein_nishina(rng, E::Float64)
    k = E / _ELECTRON_REST_KEV
    ε0 = 1 / (1 + 2k)
    a1 = -log(ε0)
    a2 = (1 - ε0^2) / 2
    while true
        ε = rand(rng) < a1 / (a1 + a2) ? exp(-rand(rng) * a1) : sqrt(ε0^2 + (1 - ε0^2) * rand(rng))
        t = (1 - ε) / (k * ε)
        (1 - ε * t * (2 - t) / (1 + ε^2)) >= rand(rng) && return ε, 1 - t
    end
end

@inline function _deflect(ux, uy, uz, cθ, φ)
    sθ = sqrt(max(1 - cθ^2, 0.0))
    cφ, sφ = cos(φ), sin(φ)
    abs(uz) > 0.99999 && return sθ * cφ, sθ * sφ, cθ * sign(uz)
    s = sqrt(1 - uz^2)
    return (
        ux * cθ + sθ * (ux * uz * cφ - uy * sφ) / s,
        uy * cθ + sθ * (uy * uz * cφ + ux * sφ) / s,
        uz * cθ - s * sθ * cφ,
    )
end

const _DOSE_DR = 0.1                           # cm, radial ring width
const _DOSE_DZ = 0.5                           # cm, axial bin width
const _DOSE_PHANTOM_LENGTH = 15.0              # cm
const _DOSE_N_CHUNKS = 32                      # fixed, so the result does not depend on nthreads

# One chunk of histories: air-kerma tallies (total, primary) in (r, z) rings, unnormalised.
function _dose_chunk!(
        total::Matrix{Float64}, primary::Matrix{Float64}, energies::Vector{Float64},
        SAD::Float64, mu_total::Vector{Float64}, p_photoelectric::Vector{Float64},
        muen_air::Vector{Float64}, cdf::Vector{Float64}, bowtie_T::Matrix{Float64},
        γ_max::Float64, R::Float64, beam_cm::Float64, n::Int, rng,
    )
    half_length = _DOSE_PHANTOM_LENGTH / 2
    n_γ = size(bowtie_T, 1)
    for _ in 1:n
        kE = min(searchsortedfirst(cdf, rand(rng)), length(cdf))
        E = energies[kE]
        γ = (2rand(rng) - 1) * γ_max
        z0 = (rand(rng) - 0.5) * beam_cm
        jγ = abs(γ) / γ_max * (n_γ - 1) + 1
        j0 = min(floor(Int, jγ), n_γ - 1)
        weight = bowtie_T[j0, kE] + (jγ - j0) * (bowtie_T[j0 + 1, kE] - bowtie_T[j0, kE])
        # source fixed at (0, -SAD, 0); the ray aims at (SAD·sinγ, 0 + …, z0) in the iso plane
        ux, uy, uz = SAD * sin(γ), SAD * cos(γ), z0
        norm_u = sqrt(ux^2 + uy^2 + uz^2)
        ux /= norm_u; uy /= norm_u; uz /= norm_u
        x, y, z = 0.0, -SAD, 0.0
        a = ux^2 + uy^2
        b = x * ux + y * uy
        disc = b^2 - a * (x^2 + y^2 - R^2)
        disc <= 0 && continue
        t = (-b - sqrt(disc)) / a
        x += t * ux; y += t * uy; z += t * uz
        abs(z) > half_length && continue
        is_primary = true
        while true
            μ = _table_lookup(mu_total, E)
            step = -log(rand(rng)) / μ
            x += step * ux; y += step * uy; z += step * uz
            r = hypot(x, y)
            (r >= R || abs(z) >= half_length) && break
            ir = floor(Int, r / _DOSE_DR) + 1
            iz = floor(Int, (z + half_length) / _DOSE_DZ) + 1
            score = weight * E * _table_lookup(muen_air, E) / μ
            total[ir, iz] += score
            is_primary && (primary[ir, iz] += score)
            rand(rng) < _table_lookup(p_photoelectric, E) && break
            is_primary = false
            ε, cθ = _klein_nishina(rng, E)
            E *= ε
            E < _DOSE_CUTOFF_KEV && break
            ux, uy, uz = _deflect(ux, uy, uz, cθ, 2π * rand(rng))
        end
    end
    return nothing
end

# Rotation-averaged weights of the radial rings for a chamber disc of radius `a` centred at
# radius `d`: the fraction of each ring's circumference the chamber covers.
function _chamber_weights(n_r, d, a)
    weights = zeros(n_r)
    for ir in 1:n_r, sub in 1:20
        r = ((ir - 1) + (sub - 0.5) / 20) * _DOSE_DR
        covered = if d == 0
            r < a ? 1.0 : 0.0
        elseif abs(r - d) >= a
            0.0
        else
            acos(clamp((r^2 + d^2 - a^2) / (2r * d), -1, 1)) / π
        end
        weights[ir] += 2π * r * covered * _DOSE_DR / 20
    end
    return weights ./ sum(weights)
end

# CTDI100 per mAs from a ring map: chamber-weighted profile, integrated over ±50 mm, over N·T.
function _ctdi100_from_map(K, d, a, NT_mm)
    n_r, n_z = size(K)
    weights = _chamber_weights(n_r, d, a)
    integral = 0.0
    for iz in 1:n_z
        zc = -_DOSE_PHANTOM_LENGTH / 2 + (iz - 0.5) * _DOSE_DZ
        abs(zc) < 5.0 || continue
        integral += sum(weights[ir] * K[ir, iz] for ir in 1:n_r) * _DOSE_DZ
    end
    return integral / (NT_mm / 10)
end

const _CTDI_CACHE = Dict{UInt64, NamedTuple}()
const _CTDI_CACHE_LOCK = ReentrantLock()

"""
    ctdi100(src::DoseSource; phantom = :body32, beam_width_mm = src.nominal_collimation_mm,
            nominal_collimation_mm = src.nominal_collimation_mm,
            n_histories = 1_000_000, seed = 1) -> NamedTuple

CTDI100 per mAs (mGy/mAs) at the centre and at the periphery (1 cm below the surface) of the
`:body32` (32 cm) or `:head16` (16 cm) PMMA CTDI phantom, by Monte Carlo transport of the beam.
`beam_width_mm` is the actual beam width at isocentre; the integral is divided by
`nominal_collimation_mm`. Returns `(center, periphery, primary_center, primary_periphery,
ctdi_w, rel_stat_uncertainty)`.

The result is deterministic for a given `seed`, whatever the number of threads, and is cached.
"""
function ctdi100(
        src::DoseSource; phantom::Symbol = :body32,
        beam_width_mm::Real = src.nominal_collimation_mm,
        nominal_collimation_mm::Real = src.nominal_collimation_mm,
        n_histories::Integer = 1_000_000, seed::Integer = 1,
    )
    R = phantom === :body32 ? 16.0 : phantom === :head16 ? 8.0 :
        throw(ArgumentError("phantom must be :body32 or :head16, got :$(phantom)"))
    # Keyed on everything the transport reads. The bowtie enters by its materials and
    # thickness profile, not by its name: a DoseSource assembled by hand can carry one name and
    # another filter, and answering that from the cache would be wrong by a factor of two.
    key = hash(
        (
            src.E, src.phi_iso, src.SAD_mm, src.bowtie.materials, src.bowtie.thickness,
            src.bowtie_name, R, Float64(beam_width_mm), Float64(nominal_collimation_mm),
            Int(n_histories), Int(seed),
        )
    )
    cached = lock(() -> get(_CTDI_CACHE, key, nothing), _CTDI_CACHE_LOCK)
    cached === nothing || return cached

    SAD = src.SAD_mm / 10
    γ_max = asin(R / SAD) * 1.0001
    n_r = round(Int, R / _DOSE_DR)
    n_z = round(Int, _DOSE_PHANTOM_LENGTH / _DOSE_DZ)
    cdf = cumsum(src.phi_iso)
    fluence = cdf[end]
    cdf ./= fluence
    n_γ = 257
    bowtie_T = zeros(n_γ, length(src.E))
    for j in 1:n_γ
        bowtie_T[j, :] .= _bowtie_transmission(src, (j - 1) / (n_γ - 1) * γ_max)
    end
    table = _pmma_table()                        # build once, before the threads start
    mu_total, p_photoelectric, muen_air = table.mu_total, table.p_photoelectric, table.muen_air

    per_chunk = cld(Int(n_histories), _DOSE_N_CHUNKS)
    totals = [zeros(n_r, n_z) for _ in 1:_DOSE_N_CHUNKS]
    primaries = [zeros(n_r, n_z) for _ in 1:_DOSE_N_CHUNKS]
    Threads.@threads for chunk in 1:_DOSE_N_CHUNKS
        _dose_chunk!(
            totals[chunk], primaries[chunk], src.E, SAD, mu_total, p_photoelectric, muen_air,
            cdf, bowtie_T, γ_max, R, beam_width_mm / 10, per_chunk,
            Random.Xoshiro(seed + 7919 * chunk),
        )
    end

    # photons per mAs into the sampled aperture (arc length × beam width at isocentre, mm²)
    photons_per_mAs = fluence * (2γ_max * src.SAD_mm) * beam_width_mm
    ring_volume = [π * ((ir * _DOSE_DR)^2 - ((ir - 1) * _DOSE_DR)^2) * _DOSE_DZ for ir in 1:n_r]
    normalise(K, n) = K .* (photons_per_mAs / n * _KEV_PER_G_TO_MGY) ./ ring_volume
    weighted(K) = begin
        # the centre is flat over r < 1 cm, which has a quarter of the variance of r < 0.5 cm
        c = _ctdi100_from_map(K, 0.0, 1.0, nominal_collimation_mm)
        p = _ctdi100_from_map(K, R - 1.0, 0.5, nominal_collimation_mm)
        (c, p, c / 3 + 2p / 3)
    end
    n_total = per_chunk * _DOSE_N_CHUNKS
    center, periphery, ctdi_w = weighted(normalise(sum(totals), n_total))
    primary_center, primary_periphery, _ = weighted(normalise(sum(primaries), n_total))
    per_chunk_w = [weighted(normalise(K, per_chunk))[3] for K in totals]
    rel_stat = Statistics.std(per_chunk_w) / sqrt(_DOSE_N_CHUNKS) / ctdi_w

    result = (; center, periphery, primary_center, primary_periphery, ctdi_w, rel_stat_uncertainty = rel_stat)
    lock(() -> (_CTDI_CACHE[key] = result), _CTDI_CACHE_LOCK)
    return result
end

"Forget every memoised CTDI100 result (see [`ctdi100`](@ref))."
empty_ctdi_cache!() = lock(() -> (empty!(_CTDI_CACHE); nothing), _CTDI_CACHE_LOCK)

# ─────────────────────────────────────────────────────────────────────────────────────────────
# The report
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    DoseReport

The dose of one acquisition. `ctdi_vol_mGy` and `dlp_mGy_cm` are the headline numbers; the rest
records how they were obtained so they can be audited.
"""
struct DoseReport
    ctdi_vol_mGy::Float64
    dlp_mGy_cm::Float64
    ctdi_w_mGy_per_100mAs::Float64
    ctdi100_center_mGy_per_100mAs::Float64
    ctdi100_periphery_mGy_per_100mAs::Float64
    air_kerma_free_in_air_mGy_per_100mAs::Float64   # centre ray, behind the bowtie, at isocentre
    phantom::Symbol
    kVp::Float64
    mAs_per_rotation::Float64
    nominal_collimation_mm::Float64
    beam_width_mm::Float64
    pitch::Union{Nothing, Float64}
    table_increment_mm::Float64
    n_rotations::Float64
    scan_length_cm::Float64
    wide_beam_reference_mm::Union{Nothing, Float64}  # 20.0 when the IEC N·T > 40 mm rule was used
    filters::Vector{Tuple{String, Float64}}
    bowtie::Symbol
    dose_calibration::Float64
    n_histories::Int                                 # as transported: rounded up to whole chunks
    rel_stat_uncertainty::Float64
end

function Base.show(io::IO, ::MIME"text/plain", r::DoseReport)
    mode = r.pitch === nothing ? "axial, increment $(round(r.table_increment_mm; digits = 2)) mm" :
        "helical, pitch $(r.pitch)"
    print(
        io,
        "DoseReport ($(r.phantom), $(Int(round(r.kVp))) kVp, $(round(r.mAs_per_rotation; digits = 1)) mAs/rotation, $(mode))\n",
        "  CTDIvol = $(round(r.ctdi_vol_mGy; digits = 2)) mGy    DLP = $(round(r.dlp_mGy_cm; digits = 2)) mGy·cm over $(round(r.scan_length_cm; digits = 2)) cm\n",
        "  CTDIw = $(round(r.ctdi_w_mGy_per_100mAs; digits = 3)) mGy/100 mAs (centre $(round(r.ctdi100_center_mGy_per_100mAs; digits = 3)), periphery $(round(r.ctdi100_periphery_mGy_per_100mAs; digits = 3)); ±$(round(100r.rel_stat_uncertainty; digits = 2)) % statistical)\n",
        "  free-in-air kerma at isocentre = $(round(r.air_kerma_free_in_air_mGy_per_100mAs; digits = 2)) mGy/100 mAs\n",
        "  N·T = $(r.nominal_collimation_mm) mm, beam $(r.beam_width_mm) mm, filters $(r.filters), bowtie :$(r.bowtie)",
    )
    r.wide_beam_reference_mm === nothing ||
        print(io, "\n  IEC wide-beam rule applied (reference beam $(r.wide_beam_reference_mm) mm)")
    r.dose_calibration == 1 || print(io, "\n  scaled by dose_calibration = $(r.dose_calibration)")
    return nothing
end

"""
    compute_dose(src::DoseSource, protocol::CTProtocol; phantom = :body32, overbeam_mm = 0.0,
                 table_increment_mm = nothing, n_tubes = 1, dose_calibration = 1.0,
                 n_histories = 1_000_000, seed = 1) -> DoseReport
    compute_dose(scanner::Scanner, protocol::CTProtocol; kwargs...) -> DoseReport

CTDIvol and DLP of an acquisition.

- `phantom`: `:body32` (the body convention) or `:head16`.
- `overbeam_mm`: how much wider than `N·T` the real beam is at isocentre. The simulator's beam is
  exactly the collimation, so the default is 0; a clinical collimator over-beams by about 3 mm,
  which matters most for narrow collimations.
- `table_increment_mm`: axial table step between rotations; defaults to `N·T` (contiguous).
  Ignored for helical protocols, which use `protocol.pitch`.
- `n_tubes`: dual-source scanners run `n_tubes` identical beams.
- `dose_calibration`: multiplies the result, to pin a scanner to a measured CTDIvol.

Helical: `CTDIvol = CTDIw · mAs / pitch`, scan length `pitch · N·T · n_rotations`.
Axial:   `CTDIvol = CTDIw · mAs · N·T / Δd`, scan length `n_rotations · Δd`.
"""
function compute_dose(
        src::DoseSource, protocol::CTProtocol;
        phantom::Symbol = :body32, overbeam_mm::Real = 0.0,
        table_increment_mm::Union{Nothing, Real} = nothing, n_tubes::Integer = 1,
        dose_calibration::Real = 1.0, n_histories::Integer = 1_000_000, seed::Integer = 1,
    )
    NT = src.nominal_collimation_mm
    beam = NT + overbeam_mm
    wide_reference = NT > 40 ? 20.0 : nothing
    mc = if wide_reference === nothing
        ctdi100(src; phantom, beam_width_mm = beam, nominal_collimation_mm = NT, n_histories, seed)
    else
        # IEC 60601-2-44 A1: measure at the reference beam, scale by the free-in-air ratio,
        # which for a uniform beam is the ratio of (beam width / N·T).
        reference = ctdi100(
            src; phantom, beam_width_mm = wide_reference + overbeam_mm,
            nominal_collimation_mm = wide_reference, n_histories, seed,
        )
        # CTDI_free-air(N·T) = D₀ · min(N·T, 100) / (N·T): the pencil chamber is 100 mm long, so
        # the free-in-air integral stops growing once the beam is wider than it. Using the
        # unsaturated ratio over-reports a 160 mm beam by 1.6x.
        scale = (min(beam, _CTDI_CHAMBER_MM) / NT) /
            (min(wide_reference + overbeam_mm, _CTDI_CHAMBER_MM) / wide_reference)
        (;
            center = reference.center * scale, periphery = reference.periphery * scale,
            primary_center = reference.primary_center * scale,
            primary_periphery = reference.primary_periphery * scale,
            ctdi_w = reference.ctdi_w * scale,
            rel_stat_uncertainty = reference.rel_stat_uncertainty,
        )
    end

    mAs = protocol.mA * protocol.rotation_time
    helical = protocol.pitch !== nothing
    increment = helical ? protocol.pitch * NT :
        (table_increment_mm === nothing ? NT : Float64(table_increment_mm))
    increment > 0 || throw(ArgumentError("table increment must be positive, got $(increment) mm"))
    ctdi_vol = mc.ctdi_w * mAs * NT / increment * n_tubes * dose_calibration
    scan_length_cm = protocol.n_rotations * increment / 10
    return DoseReport(
        ctdi_vol, ctdi_vol * scan_length_cm,
        100mc.ctdi_w, 100mc.center, 100mc.periphery, 100air_kerma_free_in_air(src),
        phantom, src.kVp, mAs, NT, beam,
        helical ? Float64(protocol.pitch) : nothing, increment, protocol.n_rotations, scan_length_cm,
        wide_reference, src.filters, src.bowtie_name, Float64(dose_calibration),
        cld(Int(n_histories), _DOSE_N_CHUNKS) * _DOSE_N_CHUNKS, mc.rel_stat_uncertainty,
    )
end

compute_dose(scanner::Scanner, protocol::CTProtocol; spectrum = nothing, kwargs...) =
    compute_dose(dose_source(scanner, protocol; spectrum), protocol; kwargs...)

"""
    compute_ctdi_vol(scanner, protocol; kwargs...) -> Float64

CTDIvol in mGy; see [`compute_dose`](@ref) for the keywords and the full report.
"""
compute_ctdi_vol(scanner::Scanner, protocol::CTProtocol; kwargs...) =
    compute_dose(scanner, protocol; kwargs...).ctdi_vol_mGy

"""
    compute_dlp(report::DoseReport) -> Float64
    compute_dlp(scanner, protocol; kwargs...) -> Float64

Dose-length product in mGy·cm: CTDIvol times the scan length.
"""
compute_dlp(report::DoseReport) = report.dlp_mGy_cm
compute_dlp(scanner::Scanner, protocol::CTProtocol; kwargs...) =
    compute_dose(scanner, protocol; kwargs...).dlp_mGy_cm

"""
    dose_report(ws, protocol; kwargs...) -> Union{DoseReport, Nothing}

The dose of the acquisition a workspace simulates, from the beam stored in `ws.dose_source`, or
`nothing` when that workspace has no beam in absolute units (a monoenergetic or otherwise
overridden spectrum). Every `simulate!` result carries this report as `result.dose`; pass
`dose_kwargs` there to reach the keywords of [`compute_dose`](@ref), which this forwards —
the defaults are a single tube and the 32 cm body phantom.
"""
function dose_report(ws, protocol::CTProtocol; kwargs...)
    ws.dose_source === nothing && return nothing
    return compute_dose(ws.dose_source, protocol; kwargs...)
end

export DoseSource, DoseReport, dose_source, air_kerma_free_in_air, ctdi100
export compute_dose, compute_ctdi_vol, compute_dlp, dose_report, muen_rho_air
export empty_ctdi_cache!
