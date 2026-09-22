# K-channel projection-domain material decomposition and the VMI chain built on it.
#
# The estimator is a nested profile Poisson maximum-likelihood fit of a material-direct basis
# pair — iodine `A` and water `C`, both in g/cm² — to `K` measured channels per ray:
#
#     y_k = I0_k · exp(-h_k)                                   corrected counts
#     λ_k(A, C) = Σ_e Φ[e, k] · exp(-μρ_I[e]·A - μρ_W[e]·C)    exact discrete polychromatic mean
#
# `K` is the number of channels the acquisition delivers: four energy windows of a
# photon-counting detector, or two acquisitions of a dual-kVp / dual-source scan. Nothing is
# rebinned into low/high; every native channel enters the likelihood.
#
# Per ray: a K-channel effective-energy linear initialiser; a bisection on the monotone aggregate
# count equation to place water; an outer Newton iteration on iodine that uses the Fisher
# Schur-complement profile curvature, with an inner Newton iteration on water; a final
# re-profile of water; and quality flags. The kernel is one `AK.foreachindex` body, so it runs
# unchanged on CPU, CUDA, Metal, ROCm and oneAPI arrays.
#
# Ported from the published worked examples (`docs/notebooks/03_dual_kvp_switching_vmi.jl`,
# `04_pcct_vmi.jl`) in the form hardened in MolloiLab/basis-vmi. Two things differ from notebook
# 04 on purpose: detector rows are combined by summing COUNTS, not transmission (the two agree
# only when `I0` is constant across rows, which a bowtie breaks), and channel groups are merged
# the same way.

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    NChannelControls(; kwargs...)

Numerical controls of the K-channel estimator. The defaults are the published ones.

- `iodine_bounds = (-0.10, 0.40)` g/cm². Scaled to per-ray noise, not to geometry, so negative
  values are admitted: clipping at zero would bias the mean of a noisy zero upward.
- `water_bounds = (-2, 50)` g/cm². The ceiling must exceed the longest water-equivalent path
  through the object; 50 leaves about 1.5× margin for a 33 cm body.
- `outer_iterations = 16`, `inner_iterations = 12`, `bisection_iterations = 28`
- `max_iodine_step = 0.05`, `max_water_step = 5` (per Newton step, g/cm²)
- `parameter_tolerance = 5e-5` (relative), `fisher_condition_limit = 1e8`
- `air_gate = 0` — rays whose every `|h_k|` is below this are set to zero without solving.
  Zero disables the gate; it is off in every published configuration.
- `tile_views = 8` — views sent to the backend per kernel launch.
"""
Base.@kwdef struct NChannelControls
    iodine_bounds::Tuple{Float32, Float32} = (-0.1f0, 0.4f0)
    water_bounds::Tuple{Float32, Float32} = (-2.0f0, 50.0f0)
    outer_iterations::Int = 16
    inner_iterations::Int = 12
    bisection_iterations::Int = 28
    max_iodine_step::Float32 = 0.05f0
    max_water_step::Float32 = 5.0f0
    parameter_tolerance::Float32 = 5.0f-5
    fisher_condition_limit::Float32 = 1.0f8
    air_gate::Float32 = 0.0f0
    tile_views::Int = 8
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Spectral basis
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    spectral_basis(; energies, response, I0, tolerance = 5e-5) -> NamedTuple

Assemble the estimator's spectral input from an ABSOLUTE per-energy response.

- `energies`: length `nE` grid, keV.
- `response`: `(nc, nr, nE, K)`; `nc` or `nr` may be 1, meaning one response serves every ray.
- `I0`: `(nc, nr, K)` absolute air counts, matching the ray axes of `response`.
- `tolerance`: the largest relative disagreement allowed between `sum(response; dims = 3)` and
  `I0`. The likelihood needs absolute responses, not independently normalised spectra, so a
  disagreement is an error rather than something to renormalise away.

Returns `(E, Φ, μρ_I, μρ_W, I0, μI_eff, μW_eff, normal_II, normal_IW, normal_WW, I0_relerr,
n_channels, ray_resolved)`. The effective-energy quantities initialise the solver only.
"""
function spectral_basis(; energies, response, I0, tolerance::Real = 5.0e-5)
    E = Float32.(collect(energies))
    Φ = Float32.(response)
    I0f = Float32.(I0)
    ndims(Φ) == 4 || throw(DimensionMismatch("response must be (nc, nr, nE, K), got $(size(Φ))"))
    size(Φ, 3) == length(E) ||
        throw(DimensionMismatch("response energy axis ($(size(Φ, 3))) ≠ energies ($(length(E)))"))
    size(Φ)[[1, 2, 4]] == size(I0f) ||
        throw(DimensionMismatch("response ray/channel axes $(size(Φ)[[1, 2, 4]]) ≠ I0 $(size(I0f))"))

    μρ_I = Float32[compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(e)) for e in E]
    μρ_W = Float32[compute_mass_μ_at_energy(XA.Materials.water, Float64(e)) for e in E]

    I0_from_Φ = dropdims(sum(Float64.(Φ); dims = 3); dims = 3)
    I0_relerr = maximum(abs.(I0_from_Φ .- Float64.(I0f)) ./ max.(Float64.(I0f), eps(Float64)))
    I0_relerr < tolerance || error(
        "Applied response and I0 disagree (max relative error = $(I0_relerr), " *
            "tolerance = $(tolerance))."
    )

    # Effective-energy linearisation: initialiser only, never the model.
    Φsum = max.(dropdims(sum(Φ; dims = 3); dims = 3), eps(Float32))
    μI_eff = dropdims(sum(Φ .* reshape(μρ_I, 1, 1, length(E), 1); dims = 3); dims = 3) ./ Φsum
    μW_eff = dropdims(sum(Φ .* reshape(μρ_W, 1, 1, length(E), 1); dims = 3); dims = 3) ./ Φsum

    return (
        E = E, Φ = Φ, μρ_I = μρ_I, μρ_W = μρ_W, I0 = I0f,
        μI_eff = μI_eff, μW_eff = μW_eff,
        normal_II = dropdims(sum(abs2, μI_eff; dims = 3); dims = 3),
        normal_IW = dropdims(sum(μI_eff .* μW_eff; dims = 3); dims = 3),
        normal_WW = dropdims(sum(abs2, μW_eff; dims = 3); dims = 3),
        I0_relerr = I0_relerr,
        n_channels = size(Φ, 4),
        ray_resolved = size(Φ, 1) > 1 || size(Φ, 2) > 1,
    )
end

"""
    spectral_basis_from_bins(; energies, W_applied, I0, transmission = nothing, tolerance = 5e-5)

Basis for the `K` energy windows of one photon-counting acquisition. `W_applied[e, k]` is the
detected response the forward model applied to the unattenuated central beam and
`I0[col, row, k]` the air count of every ray and bin. With `transmission[col, row, e]` — the
source transmission (bowtie × heel) the simulation applied per ray — the response is
ray-resolved, `W_applied[e, k]·transmission[col, row, e]`, which is exactly what each ray saw;
without it one response serves every ray and `I0` must then be constant across the fan.
"""
function spectral_basis_from_bins(; energies, W_applied, I0, transmission = nothing, tolerance::Real = 5.0e-5)
    nE, K = size(W_applied)
    I0a = Float32.(Array(I0))
    ndims(I0a) == 3 || throw(DimensionMismatch("I0 is per ray, [n_cols, n_rows, K]; got $(size(I0a))"))
    response = if transmission === nothing
        # one response for every ray: I0 must not vary across the fan
        flat = maximum(I0a; dims = (1, 2)) .- minimum(I0a; dims = (1, 2))
        all(flat .<= tolerance .* maximum(I0a; dims = (1, 2))) ||
            error("I0 varies across the fan (a bowtie) but no per-ray transmission was given")
        reshape(Float32.(W_applied), 1, 1, nE, K), reshape(I0a[1, 1, :], 1, 1, K)
    else
        tr = Float32.(Array(transmission))[:, :, 1:nE]
        Φ = Array{Float32}(undef, size(tr, 1), size(tr, 2), nE, K)
        for k in 1:K, e in 1:nE
            @views Φ[:, :, e, k] .= Float32(W_applied[e, k]) .* tr[:, :, e]
        end
        Φ, I0a
    end
    return spectral_basis(energies = energies, response = response[1], I0 = response[2], tolerance = tolerance)
end

"""
    spectral_basis(ws::PCCTWorkspace; I0 = ws.I0, tolerance = 5e-5)

Basis straight from a photon-counting workspace: the response the simulation applied
(`ws.W_matrix_gpu` on `ws.energies`, through the per-ray source transmission
`ws.bowtie_spectral` when the scanner has a bowtie) and the air response of every ray. Because
this is the model that generated the data, the decomposition inverts exactly what was simulated
and needs no calibration scan.
"""
function spectral_basis(ws::PCCTWorkspace; I0 = ws.I0, tolerance::Real = 5.0e-5)
    energies = Float64.(ws.energies)
    nE = length(energies)
    W_applied = Float64.(Array(ws.W_matrix_gpu))[1:nE, :]
    return spectral_basis_from_bins(; energies, W_applied, I0, transmission = _binned_transmission(ws, nE), tolerance)
end

# The source transmission each BINNED ray applied, summed over its native dexels: on the native
# path (bf > 1) a binned pixel's counts are the sum of bf × bf dexels, so its response is the sum
# of theirs — bf² times a single dexel's when the transmission is flat, and exactly the binned sum
# of the native table when it is not. With no table and no binning, `nothing` (one response for
# every ray).
function _binned_transmission(ws::PCCTWorkspace, nE)
    bf = ws.native_geom === nothing ? 1 : ws.native_geom.n_cols ÷ ws.geom.n_cols
    if bf == 1
        return ws.bowtie_spectral === nothing ? nothing : Array(ws.bowtie_spectral)[:, :, 1:nE]
    end
    nc, nr = ws.geom.n_cols, ws.geom.n_rows
    native = ws.native_bowtie_spectral === nothing ? nothing : Array(ws.native_bowtie_spectral)
    out = zeros(Float32, nc, nr, nE)
    for e in 1:nE, r in 1:nr, c in 1:nc
        out[c, r, e] = native === nothing ? Float32(bf * bf) :
            sum(@view native[((c - 1) * bf + 1):(c * bf), ((r - 1) * bf + 1):(r * bf), e])
    end
    return out
end

"""
    spectral_basis_from_acquisitions(; acquisitions, tolerance = 5e-5)

Basis for `K` separate acquisitions: rapid kVp switching, or two tubes. Each entry of
`acquisitions` is a NamedTuple with

- `energies`: that acquisition's own grid, keV;
- `response`: `(nc, nr, nE)` relative detected spectrum per ray;
- `I0_ray`: `(nc, nr)` absolute air counts for that acquisition.

Grids are merged onto their sorted union, so 80 and 140 kVp coexist without resampling either.
Each response is normalised over energy and scaled by its own `I0_ray`, which is where mA, duty
cycle and rotation time enter: they change how many photons an acquisition delivers, not the
shape of its spectrum.
"""
function spectral_basis_from_acquisitions(; acquisitions, tolerance::Real = 5.0e-5)
    K = length(acquisitions)
    K > 0 || throw(ArgumentError("need at least one acquisition"))
    E = sort!(unique(vcat([collect(Float64.(a.energies)) for a in acquisitions]...)))
    index_of = Dict(e => i for (i, e) in enumerate(E))
    nc, nr = size(first(acquisitions).response)[1:2]

    Φ = zeros(Float32, nc, nr, length(E), K)
    I0 = zeros(Float32, nc, nr, K)
    for (k, acquisition) in pairs(acquisitions)
        size(acquisition.response)[1:2] == (nc, nr) || throw(
            DimensionMismatch(
                "acquisition $(k) ray axes $(size(acquisition.response)[1:2]) ≠ $((nc, nr))"
            )
        )
        weights = Float64.(acquisition.response)
        weights ./= max.(sum(weights; dims = 3), eps(Float64))
        air = Float64.(acquisition.I0_ray)
        scaled = weights .* reshape(air, nc, nr, 1)
        for (source, e) in pairs(Float64.(acquisition.energies))
            @views Φ[:, :, index_of[e], k] .= Float32.(scaled[:, :, source])
        end
        I0[:, :, k] .= Float32.(air)
    end
    return spectral_basis(energies = E, response = Φ, I0 = I0, tolerance = tolerance)
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Channel preparation — always in counts
# ─────────────────────────────────────────────────────────────────────────────────────────────

_channel_I0(I0, k, nc, nr) =
    size(I0, 1) == 1 && size(I0, 2) == 1 ? fill(Float64(I0[1, 1, k]), nc, nr) : Float64.(I0[:, :, k])

"""
    reduce_detector_rows(; channels, basis, rows = :, floor_counts = 1e-12)

Combine detector rows into one by **summing counts**, carrying `Φ` and `I0` through the same
sum.

For an object invariant over the active longitudinal extent the rows are repeated measurements
of one in-plane ray, so `Y_k = Σ_r I0_{k,r} · exp(-h_{k,r})` is again Poisson with the summed
rate and the likelihood is unchanged except that `Φ_k` and `I0_k` are the summed ones. Summing
transmission and re-logging gives the same answer only when `I0` does not vary across rows; it
does whenever a bowtie is modelled, so the sum is taken in counts unconditionally.

Returns `(channels, basis, n_rows, selected_rows)`.
"""
function reduce_detector_rows(; channels, basis, rows = (:), floor_counts::Real = 1.0e-12)
    K = length(channels)
    K == basis.n_channels ||
        throw(DimensionMismatch("channels ($(K)) ≠ basis channels ($(basis.n_channels))"))
    nc = size(first(channels), 1)
    selected = rows === (:) ? (1:size(first(channels), 2)) : rows

    Φ = Float32.(basis.Φ)
    I0 = Float32.(basis.I0)
    ray_resolved = size(Φ, 1) > 1 || size(Φ, 2) > 1

    new_channels = map(1:K) do k
        I0_k = size(I0, 1) == 1 && size(I0, 2) == 1 ?
            fill(I0[1, 1, k], nc, length(selected)) : Float32.(I0[:, selected, k])
        counts = dropdims(
            sum(
                Float64.(reshape(I0_k, nc, length(selected), 1)) .*
                    exp.(-Float64.(view(channels[k], :, selected, :)));
                dims = 2,
            ); dims = 2
        )
        air = dropdims(sum(Float64.(I0_k); dims = 2); dims = 2)
        h = Float32.(-log.(max.(counts, floor_counts) ./ reshape(air, nc, 1)))
        reshape(h, nc, 1, size(h, 2))
    end

    summed_Φ = ray_resolved ?
        sum(Float64.(view(Φ, :, selected, :, :)); dims = 2) : Float64.(Φ) .* length(selected)
    summed_I0 = ray_resolved ?
        sum(Float64.(view(I0, :, selected, :)); dims = 2) : Float64.(I0) .* length(selected)

    new_basis = spectral_basis(
        energies = basis.E,
        response = reshape(summed_Φ, size(summed_Φ, 1), 1, length(basis.E), K),
        I0 = reshape(summed_I0, size(summed_I0, 1), 1, K),
    )
    return (
        channels = new_channels, basis = new_basis,
        n_rows = length(selected), selected_rows = selected,
    )
end

"""
    merge_channels(; channels, basis, groups, floor_counts = 1e-12)

Sum channels into `groups`, in counts, carrying `Φ` and `I0` with them: counts add,
transmission does not, so merged `Φ` and `I0` are summed over the group and never averaged.
`groups = [1:2, 3:4]` collapses four photon-counting windows into a low and a high channel,
which is how a two-measurement comparator is given the same acquisition.

The groups must partition the channels, each appearing exactly once: a channel in two groups
would enter the likelihood twice as though it were an independent measurement, and one in none
would be discarded silently.

Returns `(channels, basis, groups)`.
"""
function merge_channels(; channels, basis, groups, floor_counts::Real = 1.0e-12)
    K = length(channels)
    K == basis.n_channels ||
        throw(DimensionMismatch("channels ($(K)) ≠ basis channels ($(basis.n_channels))"))
    # Every channel exactly once. An overlapping group would enter the likelihood twice as if it
    # were an independent measurement; a missing one would be dropped without a word.
    flat = reduce(vcat, [collect(g) for g in groups]; init = Int[])
    sort(flat) == collect(1:K) || throw(
        ArgumentError(
            "groups must partition the $(K) channels, each exactly once; got $(groups)"
        )
    )
    Φ = Float64.(basis.Φ)
    I0 = Float64.(basis.I0)
    nc, nr = size(first(channels))[1:2]

    new_channels = map(groups) do group
        counts = zeros(Float64, size(first(channels)))
        air = zeros(Float64, nc, nr)
        for k in group
            I0_k = _channel_I0(I0, k, nc, nr)
            counts .+= reshape(I0_k, nc, nr, 1) .* exp.(-Float64.(channels[k]))
            air .+= I0_k
        end
        Float32.(-log.(max.(counts, floor_counts) ./ reshape(air, nc, nr, 1)))
    end

    merged_Φ = cat([sum(view(Φ, :, :, :, group); dims = 4) for group in groups]...; dims = 4)
    merged_I0 = cat([sum(view(I0, :, :, group); dims = 3) for group in groups]...; dims = 3)
    new_basis = spectral_basis(energies = basis.E, response = merged_Φ, I0 = merged_I0)
    return (channels = new_channels, basis = new_basis, groups = groups)
end

"""
    prepare_channels(; channels, basis, merge_groups = nothing, reduce_rows = false, rows = :)

Everything the decomposition needs done to its inputs first: sum channel groups, then sum
detector rows. Both reductions add counts and rebuild the response so that `sum(Φ[:, k])`
still equals that channel's `I0`. Returns `(channels, basis, n_rows)`.
"""
function prepare_channels(;
        channels, basis, merge_groups = nothing, reduce_rows::Bool = false, rows = (:),
    )
    !reduce_rows && rows !== (:) && throw(ArgumentError(
        "rows selects the detector rows to sum, and does nothing without reduce_rows = true"))
    working_channels, working_basis = channels, basis
    if merge_groups !== nothing
        merged = merge_channels(
            channels = working_channels, basis = working_basis, groups = merge_groups,
        )
        working_channels, working_basis = merged.channels, merged.basis
    end
    n_rows = nothing
    if reduce_rows
        reduced = reduce_detector_rows(
            channels = working_channels, basis = working_basis, rows = rows,
        )
        working_channels, working_basis = reduced.channels, reduced.basis
        n_rows = reduced.n_rows
    end
    return (channels = working_channels, basis = working_basis, n_rows = n_rows)
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# The estimator kernel
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    nchannel_profile_tile!(sino_I, sino_W, fisher_AA, fisher_AC, fisher_CC, quality_flag,
                           score_norm, outer_count, inner_count, hs::NTuple{K},
                           Φ, μρ_I, μρ_W, I0, μI_eff, μW_eff,
                           normal_II, normal_IW, normal_WW, controls::NChannelControls)

One tile of the K-channel profile-likelihood estimator; every array lives on one backend.
`hs` holds the `K` corrected log-transmission tiles `(n_col, n_row, n_view_tile)`.

Quality flag bits: `1` iodine at a bound, `2` water at a bound, `4` not converged,
`8` ill-conditioned Fisher matrix or invalid initialiser, `16` the summed counts are not
attainable anywhere inside the bounds, `32` a non-finite result.

Counts may be fractional after detector corrections, so nothing here assumes integers.
"""
function nchannel_profile_tile!(
        sino_I, sino_W, fisher_AA, fisher_AC, fisher_CC,
        quality_flag, score_norm, outer_count, inner_count,
        hs::NTuple{K, Any},
        Φ, μρ_I, μρ_W, I0, μI_eff, μW_eff,
        normal_II, normal_IW, normal_WW, controls::NChannelControls,
    ) where {K}
    nE = length(μρ_I)
    A_lo, A_hi = controls.iodine_bounds
    C_lo, C_hi = controls.water_bounds
    n_outer, n_inner = controls.outer_iterations, controls.inner_iterations
    n_bisect = controls.bisection_iterations
    A_step, C_step = controls.max_iodine_step, controls.max_water_step
    parameter_tolerance = controls.parameter_tolerance
    fisher_condition_limit = controls.fisher_condition_limit
    air_gate = controls.air_gate

    AK.foreachindex(sino_I) do idx
        # Ray coordinates, so a response that varies across the detector (a bowtie, or
        # per-acquisition air scaling) is looked up per ray. A response that does not vary is
        # stored with singleton ray axes; these collapse to 1, bit-identical to indexing Φ[e, k].
        ncol = size(sino_I, 1)
        nrow = size(sino_I, 2)
        col = mod1(idx, ncol)
        row = mod1(cld(idx, ncol), nrow)
        cΦ = size(Φ, 1) == 1 ? 1 : col
        rΦ = size(Φ, 2) == 1 ? 1 : row
        cI = size(I0, 1) == 1 ? 1 : col
        rI = size(I0, 2) == 1 ? 1 : row

        max_abs_h = 0.0f0
        for k in 1:K
            max_abs_h = max(max_abs_h, abs(hs[k][idx]))
        end
        if max_abs_h < air_gate
            sino_I[idx] = 0.0f0
            sino_W[idx] = 0.0f0
            fisher_AA[idx] = 0.0f0
            fisher_AC[idx] = 0.0f0
            fisher_CC[idx] = 0.0f0
            quality_flag[idx] = UInt8(0)
            score_norm[idx] = 0.0f0
            outer_count[idx] = UInt8(0)
            inner_count[idx] = UInt8(0)
            return
        end

        # K-channel linear initialiser; every iteration below is polychromatic.
        rhs_I, rhs_W = 0.0f0, 0.0f0
        for k in 1:K
            rhs_I += μI_eff[cI, rI, k] * hs[k][idx]
            rhs_W += μW_eff[cI, rI, k] * hs[k][idx]
        end
        nII = normal_II[cI, rI]
        nIW = normal_IW[cI, rI]
        nWW = normal_WW[cI, rI]
        det0_raw = nII * nWW - nIW * nIW
        initializer_valid = isfinite(det0_raw) && det0_raw > 1.0f-12
        det0 = initializer_valid ? det0_raw : 1.0f0
        A = initializer_valid ?
            clamp((nWW * rhs_I - nIW * rhs_W) / det0, A_lo, A_hi) : clamp(0.0f0, A_lo, A_hi)
        C = initializer_valid ?
            clamp((nII * rhs_W - nIW * rhs_I) / det0, C_lo, C_hi) : clamp(20.0f0, C_lo, C_hi)

        # The aggregate count equation is monotone in water, so a bisection on it stabilises the
        # fast solver's initial water value at its current iodine value.
        y_total = 0.0f0
        for k in 1:K
            y_total += max(I0[cI, rI, k] * exp(-hs[k][idx]), 1.0f-6)
        end
        croot_lo, croot_hi = C_lo, C_hi
        total_lo, total_hi = 0.0f0, 0.0f0
        attainable_max, attainable_min = 0.0f0, 0.0f0
        # Energies the channel does not respond to are skipped, here and below. It is the same
        # sum, but a spectrum grid reaches down to a few keV where Φ is exactly zero and
        # exp(+μ·|C_lo|) overflows Float32: 0 · Inf = NaN would poison the bracketing test and
        # the feasibility flag for every ray (the published notebook recomputes that flag on the
        # host in Float64 for this reason).
        for k in 1:K, e in 1:nE
            ϕ = Φ[cΦ, rΦ, e, k]
            ϕ > 0.0f0 || continue
            total_lo += ϕ * exp(-μρ_I[e] * A - μρ_W[e] * croot_lo)
            total_hi += ϕ * exp(-μρ_I[e] * A - μρ_W[e] * croot_hi)
            attainable_max += ϕ * exp(-μρ_I[e] * A_lo - μρ_W[e] * C_lo)
            attainable_min += ϕ * exp(-μρ_I[e] * A_hi - μρ_W[e] * C_hi)
        end
        aggregate_bracketed = total_lo ≥ y_total && total_hi ≤ y_total
        aggregate_feasible = attainable_max ≥ y_total && attainable_min ≤ y_total
        if aggregate_bracketed
            for _ in 1:n_bisect
                mid = (croot_lo + croot_hi) / 2.0f0
                total_mid = 0.0f0
                for k in 1:K, e in 1:nE
                    ϕ = Φ[cΦ, rΦ, e, k]
                    ϕ > 0.0f0 || continue
                    total_mid += ϕ * exp(-μρ_I[e] * A - μρ_W[e] * mid)
                end
                if total_mid > y_total
                    croot_lo = mid
                else
                    croot_hi = mid
                end
            end
            C = (croot_lo + croot_hi) / 2.0f0
        end

        converged = false
        used_outer = 0
        used_inner = 0
        for outer_iter in 1:n_outer
            used_outer = outer_iter
            # Inner scalar solve: C*(A) = argmin_C L(A, C).
            for _ in 1:n_inner
                used_inner += 1
                gC, FCC = 0.0f0, 0.0f0
                for k in 1:K
                    λ, dC = 0.0f0, 0.0f0
                    @inbounds for e in 1:nE
                        ϕ = Φ[cΦ, rΦ, e, k]
                        ϕ > 0.0f0 || continue
                        z = ϕ * exp(-μρ_I[e] * A - μρ_W[e] * C)
                        λ += z
                        dC -= μρ_W[e] * z
                    end
                    λ = max(λ, 1.0f-6)
                    y = max(I0[cI, rI, k] * exp(-hs[k][idx]), 1.0f-6)
                    gC += (1.0f0 - y / λ) * dC
                    FCC += dC * dC / λ
                end
                raw_C_step = gC / max(FCC, 1.0f-12)
                C_new = clamp(C - clamp(raw_C_step, -C_step, C_step), C_lo, C_hi)
                C_done = abs(C_new - C) <= parameter_tolerance * (1.0f0 + abs(C))
                C = C_new
                C_done && break
            end

            # Envelope gradient and Fisher Schur-complement profile curvature.
            gA, FAA, FAC, FCC = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            for k in 1:K
                λ, dA, dC = 0.0f0, 0.0f0, 0.0f0
                @inbounds for e in 1:nE
                    ϕ = Φ[cΦ, rΦ, e, k]
                    ϕ > 0.0f0 || continue
                    z = ϕ * exp(-μρ_I[e] * A - μρ_W[e] * C)
                    λ += z
                    dA -= μρ_I[e] * z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1.0f-6)
                y = max(I0[cI, rI, k] * exp(-hs[k][idx]), 1.0f-6)
                gA += (1.0f0 - y / λ) * dA
                FAA += dA * dA / λ
                FAC += dA * dC / λ
                FCC += dC * dC / λ
            end
            Hprof = max(FAA - FAC * FAC / max(FCC, 1.0f-12), 1.0f-12)
            A_new = clamp(A - clamp(gA / Hprof, -A_step, A_step), A_lo, A_hi)
            converged = abs(A_new - A) <= parameter_tolerance * (1.0f0 + abs(A))
            A = A_new
            converged && break
        end

        # Re-profile water at the final iodine iterate.
        c_converged = false
        for _ in 1:n_inner
            used_inner += 1
            gC, FCC = 0.0f0, 0.0f0
            for k in 1:K
                λ, dC = 0.0f0, 0.0f0
                @inbounds for e in 1:nE
                    ϕ = Φ[cΦ, rΦ, e, k]
                    ϕ > 0.0f0 || continue
                    z = ϕ * exp(-μρ_I[e] * A - μρ_W[e] * C)
                    λ += z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1.0f-6)
                y = max(I0[cI, rI, k] * exp(-hs[k][idx]), 1.0f-6)
                gC += (1.0f0 - y / λ) * dC
                FCC += dC * dC / λ
            end
            C_new = clamp(C - clamp(gC / max(FCC, 1.0f-12), -C_step, C_step), C_lo, C_hi)
            C_done = abs(C_new - C) <= parameter_tolerance * (1.0f0 + abs(C))
            C = C_new
            if C_done
                c_converged = true
                break
            end
        end
        converged &= c_converged

        # The final score and Fisher conditioning are recorded; they are never silently turned
        # into image regularisation.
        gA, gC, FAA, FAC, FCC = 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0
        for k in 1:K
            λ, dA, dC = 0.0f0, 0.0f0, 0.0f0
            @inbounds for e in 1:nE
                ϕ = Φ[cΦ, rΦ, e, k]
                ϕ > 0.0f0 || continue
                z = ϕ * exp(-μρ_I[e] * A - μρ_W[e] * C)
                λ += z
                dA -= μρ_I[e] * z
                dC -= μρ_W[e] * z
            end
            λ = max(λ, 1.0f-6)
            y = max(I0[cI, rI, k] * exp(-hs[k][idx]), 1.0f-6)
            gA += (1.0f0 - y / λ) * dA
            gC += (1.0f0 - y / λ) * dC
            FAA += dA * dA / λ
            FAC += dA * dC / λ
            FCC += dC * dC / λ
        end
        score = sqrt(gA * gA + gC * gC) / sqrt(max(FAA + FCC, 1.0f-12))
        score_norm[idx] = score
        fisher_det = max(FAA * FCC - FAC * FAC, 0.0f0)
        fisher_trace = FAA + FCC
        fisher_disc = sqrt(max(fisher_trace * fisher_trace - 4.0f0 * fisher_det, 0.0f0))
        eig_max_raw = max((fisher_trace + fisher_disc) / 2.0f0, 1.0f-12)
        eig_min = max(fisher_det / eig_max_raw, 1.0f-12)
        eig_max = max(eig_max_raw, eig_min)
        ill_conditioned = eig_max / eig_min > fisher_condition_limit

        tol = 2.0f-4
        hit_A = A <= A_lo + tol || A >= A_hi - tol
        hit_C = C <= C_lo + tol || C >= C_hi - tol
        invalid_model = !(
            isfinite(A) && isfinite(C) && isfinite(score) &&
                isfinite(FAA) && isfinite(FAC) && isfinite(FCC)
        )
        quality_flag[idx] =
            UInt8(hit_A ? 1 : 0) |
            UInt8(hit_C ? 2 : 0) |
            UInt8(converged ? 0 : 4) |
            UInt8(ill_conditioned || !initializer_valid ? 8 : 0) |
            UInt8(aggregate_feasible ? 0 : 16) |
            UInt8(invalid_model ? 32 : 0)
        outer_count[idx] = UInt8(min(used_outer, 255))
        inner_count[idx] = UInt8(min(used_inner, 255))
        fisher_AA[idx], fisher_AC[idx], fisher_CC[idx] = FAA, FAC, FCC
        sino_I[idx], sino_W[idx] = A, C
    end
    return nothing
end

# Collapse the per-ray quality maps into the few numbers that describe a decomposition. A
# per-ray Fisher matrix or score map is sinogram-shaped; returning several of them from every
# call is how a long run ends out of memory hours after the mistake was made.
function _nchannel_flag_summary(flags, score, outer, inner)
    n = length(flags)
    bit(mask) = count(f -> (f & mask) != 0, flags) / n
    return (
        n_rays = n,
        frac_bound_iodine = bit(UInt8(1)),
        frac_bound_water = bit(UInt8(2)),
        frac_not_converged = bit(UInt8(4)),
        frac_ill_conditioned = bit(UInt8(8)),
        frac_infeasible = bit(UInt8(16)),
        frac_invalid = bit(UInt8(32)),
        score_norm_max = maximum(score),
        score_norm_mean = sum(score) / n,
        outer_mean = sum(Float64, outer) / n,
        inner_mean = sum(Float64, inner) / n,
    )
end

"""
    decompose_nchannel(; channels, basis, controls = NChannelControls(), to_backend = identity,
                       tile_views = controls.tile_views, keep_diagnostics = false)

The K-channel estimator over a full sinogram. `channels` is a vector of `K` corrected
log-transmission arrays `(n_col, n_row, n_view)`; `basis` comes from [`spectral_basis`](@ref) or
one of its builders; `to_backend` uploads an array to the compute backend (`CuArray`,
`MtlArray`, …; the default stays on the CPU).

Views are tiled so the backend never holds a whole sinogram of workspace, and every tile's
buffers are released without a full garbage collection.

Returns `(sino_iodine, sino_water, quality, elapsed_s, method = :nchannel, n_channels)` with
both sinograms in g/cm². `keep_diagnostics = true` adds the per-ray maps.
"""
function decompose_nchannel(;
        channels, basis, controls::NChannelControls = NChannelControls(),
        to_backend = identity, tile_views::Integer = controls.tile_views,
        keep_diagnostics::Bool = false,
    )
    K = length(channels)
    K == basis.n_channels ||
        throw(DimensionMismatch("channels ($(K)) ≠ basis channels ($(basis.n_channels))"))
    K >= 2 || throw(
        ArgumentError(
            "the estimator solves for two materials and needs at least two channels, got $(K)"
        )
    )
    shape = size(first(channels))
    all(size(h) == shape for h in channels) ||
        throw(DimensionMismatch("every channel must share one sinogram shape"))

    sino_I = Array{Float32}(undef, shape)
    sino_W = Array{Float32}(undef, shape)
    flags = Array{UInt8}(undef, shape)
    score = Array{Float32}(undef, shape)
    fisher_AA = Array{Float32}(undef, shape)
    fisher_AC = Array{Float32}(undef, shape)
    fisher_CC = Array{Float32}(undef, shape)
    outer = Array{UInt8}(undef, shape)
    inner = Array{UInt8}(undef, shape)

    constants = map(
        to_backend,
        (
            basis.Φ, basis.μρ_I, basis.μρ_W, basis.I0, basis.μI_eff, basis.μW_eff,
            basis.normal_II, basis.normal_IW, basis.normal_WW,
        ),
    )

    elapsed = try
        @elapsed for vrange in tile_ranges(shape[3], tile_views)
            hs = ntuple(k -> to_backend(Float32.(channels[k][:, :, vrange])), K)
            I_dev, W_dev = similar(hs[1]), similar(hs[1])
            flag_dev = similar(hs[1], UInt8)
            score_dev = similar(hs[1], Float32)
            AA_dev = similar(hs[1], Float32)
            AC_dev = similar(hs[1], Float32)
            CC_dev = similar(hs[1], Float32)
            outer_dev = similar(hs[1], UInt8)
            inner_dev = similar(hs[1], UInt8)
            try
                nchannel_profile_tile!(
                    I_dev, W_dev, AA_dev, AC_dev, CC_dev,
                    flag_dev, score_dev, outer_dev, inner_dev, hs,
                    constants..., controls,
                )
                sino_I[:, :, vrange] .= Array(I_dev)
                sino_W[:, :, vrange] .= Array(W_dev)
                flags[:, :, vrange] .= Array(flag_dev)
                score[:, :, vrange] .= Array(score_dev)
                fisher_AA[:, :, vrange] .= Array(AA_dev)
                fisher_AC[:, :, vrange] .= Array(AC_dev)
                fisher_CC[:, :, vrange] .= Array(CC_dev)
                outer[:, :, vrange] .= Array(outer_dev)
                inner[:, :, vrange] .= Array(inner_dev)
            finally
                # collect = false: the buffers are returned deterministically, and a full GC
                # per tile costs more than the tile does.
                release_backend!(
                    (
                        hs..., I_dev, W_dev, flag_dev, score_dev,
                        AA_dev, AC_dev, CC_dev, outer_dev, inner_dev,
                    ); collect = false
                )
            end
        end
    finally
        release_backend!(constants; collect = false)
    end

    base = (
        sino_iodine = sino_I, sino_water = sino_W,
        quality = _nchannel_flag_summary(flags, score, outer, inner),
        elapsed_s = elapsed, method = :nchannel, n_channels = K,
    )
    return keep_diagnostics ? merge(
            base, (
                diagnostics = (
                    quality_flag = flags, score_norm = score,
                    fisher = (AA = fisher_AA, AC = fisher_AC, CC = fisher_CC),
                    outer_iterations = outer, inner_iterations = inner,
                ),
            )
        ) : base
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# The two-measurement comparator, on the same inputs
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    cong_material_basis(basis) -> NamedTuple

Translate a two-channel [`spectral_basis`](@ref) into the `(ŵ, p, q)` form [`apply_cong!`](@ref)
wants. `Φ` carries absolute per-energy air counts; Cong wants the shape only, normalised so
`Σ_E ŵ = 1`, because the absolute scale enters through the measured line integrals. `p` and `q`
are the iodine and water mass attenuations: the same material-direct pair the K-channel
estimator uses, so both invert the identical physical basis. A response with singleton ray axes
collapses to a 1-D spectrum; a ray-resolved one stays 3-D, which is how a bowtie reaches the
solver.
"""
function cong_material_basis(basis)
    Φ = Float64.(basis.Φ)
    size(Φ, 4) == 2 ||
        throw(ArgumentError("Cong needs exactly two channels, got $(size(Φ, 4))"))
    shape(k) = begin
        w = Φ[:, :, :, k]
        w ./= max.(sum(w; dims = 3), eps(Float64))
        size(w, 1) == 1 && size(w, 2) == 1 ? Float32.(vec(w)) : Float32.(w)
    end
    p = Float32.(basis.μρ_I)
    q = Float32.(basis.μρ_W)
    return (ŵ_L = shape(1), p_L = p, q_L = q, ŵ_H = shape(2), p_H = p, q_H = q)
end

"""
    decompose_cong(; channels, basis, to_backend = identity, newton_max_iter = 12,
                   newton_tol = eps(Float32), y_max_factor = 0.99, y_max_cap = 1f7)

The Cong 2022 two-measurement estimator ([`apply_cong!`](@ref)) behind the same signature and
return shape as [`decompose_nchannel`](@ref), so one configuration can be sent through either.
Exactly two channels are required; pass `merge_groups` to [`prepare_channels`](@ref) to sum a
photon-counting acquisition into two.
"""
function decompose_cong(;
        channels, basis, to_backend = identity,
        newton_max_iter::Integer = 12, newton_tol::Real = eps(Float32),
        y_max_factor::Real = 0.99, y_max_cap::Real = 1.0f7,
    )
    length(channels) == 2 || throw(
        ArgumentError(
            "the Cong estimator is defined for two measurements, got $(length(channels)); " *
                "pass merge_groups to sum channels into two"
        )
    )
    material = cong_material_basis(basis)
    low = to_backend(Float32.(channels[1]))
    high = to_backend(Float32.(channels[2]))
    sino_y, sino_c = similar(low), similar(low)
    fill!(sino_y, 0.0f0)
    fill!(sino_c, 0.0f0)
    cong_ws = nothing
    iodine, water, elapsed = try
        cong_ws = create_cong_workspace(low, material)
        t = @elapsed apply_cong!(
            cong_ws, sino_y, sino_c, low, high;
            water_basis = (a = 0.0f0, c = 1.0f0),
            newton_max_iter, newton_tol, y_max_factor, y_max_cap,
        )
        Array(sino_y), Array(sino_c), t
    finally
        # the Cong workspace uploads its own basis arrays, so it is released too
        release_backend!((low, high, sino_y, sino_c); collect = false)
        cong_ws === nothing || release_backend!(cong_ws; collect = false)
    end
    return (
        sino_iodine = iodine, sino_water = water,
        quality = (
            n_rays = length(iodine),
            frac_invalid = count(!isfinite, iodine) / length(iodine),
            frac_nonfinite_water = count(!isfinite, water) / length(water),
        ),
        elapsed_s = elapsed, method = :cong, n_channels = 2,
    )
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Reconstruction and synthesis
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    angular_antialias_response(; n_views, matrix_nx) -> Vector{Float64}

Deterministic angular response: unit gain through mode `⌈π·matrix_nx/4⌉`, then a raised-cosine
roll-off across the oversampling margin only. It is all ones when the acquisition does not
oversample the target grid, in which case the filter is an identity and is skipped. This is not
a tunable denoiser: a 1200-view acquisition oversamples what a 512-pixel grid can represent, and
the roll-off discards only the angular frequencies that grid cannot carry without aliasing.
"""
function angular_antialias_response(; n_views::Integer, matrix_nx::Integer)
    pass_mode = min(n_views ÷ 2, ceil(Int, π * matrix_nx / 4))
    margin = n_views ÷ 2 - pass_mode
    margin <= 0 && return ones(Float64, n_views)
    return [
        let mode = min(j - 1, n_views - (j - 1))
                mode <= pass_mode ? 1.0 : 0.5 * (1 + cos(π * (mode - pass_mode) / margin))
        end
            for j in 1:n_views
    ]
end

"""
    reconstruct_basis_slice(sino, geom, matrix_size; to_backend = identity,
                            filter = SoftFilter(), n_rows = geom.n_rows, antialias = true,
                            method = :fbp, hir_strength = 60, projector = :dd_fast,
                            hir_weights = :uniform, scale = 1)

Reconstruct one sinogram, whether it carries a measured channel or an estimated basis material.

`method = :fbp` is FDK as basis-vmi does it: a single-row sinogram (what
[`reduce_detector_rows`](@ref) leaves) is repeated to `n_rows` and backprojected onto the
requested grid with `filter`.

`method = :hir` is the penalized iterative reconstructor at `hir_strength` (0-100, see
[`get_hir_params`](@ref)). Three things make it sound on a basis sinogram, none of which FDK
needs:

- **The weights.** HIR's default data weighting is `exp(-y)`, the Poisson heuristic for a
  log-transmission. A basis sinogram is g/cm², not a transmission — through a body it reads 20 to
  30 for water and a fraction of one for iodine — so that weighting would switch the data term
  off inside the object for one material and leave it uniform for the other. `hir_weights` is
  what is passed to `reconstruct!` instead: `:uniform`, or the per-ray inverse-variance map from
  the decomposition's Fisher information (see [`vmi_pipeline`](@ref)).
- **The units.** HIR's Huber threshold and regularisation strength were tuned on attenuation
  images in cm⁻¹. `scale` multiplies the sinogram before reconstruction and divides the image
  after, so with `scale = μ/ρ` of the material at a reference energy the reconstruction happens
  in μ-equivalent units and the strength dial means what its table says. FDK is linear and needs
  no such thing.
- **The geometry.** A single-row sinogram is NOT repeated: it is reconstructed through a one-row
  geometry with a volume one row thick. Repeating one row to `n_rows` asserts that every row saw
  the same thing, which a one-slice forward model cannot reproduce at the outer rows; FDK
  ignores the contradiction, PWLS would fit it.

`hir_weights` and `scale` are ignored by `:fbp`.
"""
function reconstruct_basis_slice(
        sino::AbstractArray{<:Real, 3}, geom, matrix_size;
        to_backend = identity, filter = SoftFilter(), n_rows::Integer = geom.n_rows,
        antialias::Bool = true, method::Symbol = :fbp, hir_strength::Integer = 60,
        projector::Symbol = :dd_fast, hir_weights::Union{Symbol, AbstractArray} = :uniform,
        scale::Real = 1,
    )
    method in (:fbp, :hir) ||
        throw(ArgumentError("method must be :fbp or :hir, got :$(method)"))
    size(sino, 2) in (1, n_rows) || throw(DimensionMismatch(
        "the sinogram has $(size(sino, 2)) rows; the geometry has $(n_rows) — pass every row, or one"))
    n_views = size(sino, 3)
    working = Float32.(sino)
    if antialias
        response = angular_antialias_response(n_views = n_views, matrix_nx = matrix_size[1])
        if !all(==(1.0), response)
            spectrum = FFTW.fft(Float64.(sino), 3)
            working = Float32.(real.(FFTW.ifft(spectrum .* reshape(response, 1, 1, n_views), 3)))
        end
    end

    if method === :fbp
        repeated = size(working, 2) == n_rows ? working : repeat(working, 1, n_rows, 1)
        sino_dev = to_backend(repeated)
        ws = create_fdk_recon_workspace(sino_dev, geom, matrix_size; filter = filter)
        try
            return Float32.(Array(reconstruct!(ws, sino_dev, geom)))
        finally
            release_backend!(ws)
            release_backend!(sino_dev)
        end
    end

    single_row = size(working, 2) == 1 && geom.n_rows != 1
    single_row && matrix_size[3] != 1 && throw(ArgumentError(
        "a single-row sinogram reconstructs one slice; matrix_size has $(matrix_size[3])"))
    recon_geom = single_row ? _single_row_geometry(geom) : geom
    hir_weights isa AbstractArray && size(hir_weights) != size(working) && throw(DimensionMismatch(
        "hir_weights has size $(size(hir_weights)); the sinogram is $(size(working))"))
    s = Float32(scale)
    sino_dev = to_backend(working .* s)
    weights_dev = hir_weights isa AbstractArray ? to_backend(Float32.(hir_weights)) : hir_weights
    ws = create_hir_recon_workspace(
        sino_dev, recon_geom, matrix_size;
        filter = filter, strength = hir_strength, projector = projector,
    )
    try
        image = reconstruct!(ws, sino_dev, recon_geom; weights = weights_dev)
        return Float32.(Array(image)) ./ s
    finally
        release_backend!(ws)
        release_backend!(sino_dev)
        weights_dev isa AbstractArray && release_backend!(weights_dev)
    end
end

# One detector row at the isocentre plane: the same trajectory, fan and pitch, a field of view
# one row thick. What a reduced-row sinogram actually measured.
function _single_row_geometry(geom::CTGeometry)
    return CTGeometry(
        geom.SAD, geom.SDD, geom.n_angles, 1, geom.n_cols,
        geom.pixel_size, geom.pixel_row_size,
        geom.angles, geom.source_positions, geom.detector_centers,
        geom.detector_u, geom.detector_v, (geom.fov[1], geom.fov[2], geom.pixel_row_size),
        geom.pitch, geom.table_feed, geom.detector_shape, geom.column_offset,
    )
end

# Inverse-variance weights for the two basis sinograms from the K-channel Fisher information,
# in the μ-equivalent units the reconstruction runs in. Σ = F⁻¹, so var(A) = F_CC/det and
# var(C) = F_AA/det; scaling a sinogram by s scales its variance by s². Normalised so the
# best-determined ray has weight 1 — the (0, 1] range HIR's strength table was calibrated on
# with transmission weights. A ray the estimator could not determine (F singular) gets none.
function _basis_weights(fisher, scale_iodine::Real, scale_water::Real)
    AA, AC, CC = Float64.(fisher.AA), Float64.(fisher.AC), Float64.(fisher.CC)
    det = @. AA * CC - AC * AC
    w_I = @. ifelse(det > 0, det / max(CC, eps()) / scale_iodine^2, 0.0)
    w_W = @. ifelse(det > 0, det / max(AA, eps()) / scale_water^2, 0.0)
    normalise(w) = (m = maximum(w); m > 0 ? Float32.(w ./ m) : ones(Float32, size(w)))
    return normalise(w_I), normalise(w_W)
end

"""
    synthesize_vmi_stack(water, iodine, energies) -> Array{Float32,4}

Virtual monoenergetic images in HU from a reconstructed basis pair: `(nx, ny, nz, n_energies)`,
every slice of the pair at every energy. `water` is a density in g/mL and `iodine` a mass density
in g/cm³; the latter is converted to mg/mL for [`synth_vmi_2basis`](@ref).
"""
function synthesize_vmi_stack(water::AbstractArray{<:Real, 3}, iodine::AbstractArray{<:Real, 3}, energies)
    size(water) == size(iodine) ||
        throw(DimensionMismatch("water $(size(water)) and iodine $(size(iodine)) differ"))
    stack = Array{Float32}(undef, size(water)..., length(energies))
    water32 = Float32.(water)
    iodine_mg_mL = Float32.(iodine) .* 1000.0f0
    for (index, energy) in pairs(collect(energies))
        stack[:, :, :, index] .= synth_vmi_2basis(water32, iodine_mg_mL; energy_keV = Float64(energy))
    end
    return stack
end

# ─────────────────────────────────────────────────────────────────────────────────────────────
# The whole chain
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    vmi_pipeline(; channels, basis, geom, to_backend = identity, kwargs...)

Measured channels to virtual monoenergetic images:

    channels ─┬─ merge_channels ─┬─ reduce_detector_rows ─┬─ decompose_nchannel ─┐
              └──────────────────┘                        └─ decompose_cong ─────┤
        └─ tlbf_denoise ─ reconstruct_basis_slice ─ apply_acnr_kalender! ─ synthesize_vmi_stack

Every stage's settings are keywords, so the function makes no decision the caller cannot see
and override, which is what makes it usable as the inner call of an ablation sweep.

Required: `channels` (vector of `K` corrected log-transmission sinograms), `basis`
([`spectral_basis`](@ref)), `geom` (the acquisition geometry), `matrix_size`.

Stages, all optional:

- `method = :nchannel` or `:cong`; `controls = NChannelControls()`.
- `merge_groups = nothing` — channel groups summed before decomposing, e.g. `[1:2, 3:4]`.
- `reduce_rows = false`, `rows = :` — sum detector rows in counts, for a z-invariant object whose
  slices are all the same. Without it every row is kept and the volume is reconstructed slice by
  slice onto `matrix_size`.
- `use_tlbf = false`, `tlbf_alpha1`, `tlbf_alpha2`, `tlbf_radius` — photon-counting only; filters
  each detector row in its own (column, view) plane.
- `use_acnr = true`, `acnr_passes = 4`, `acnr_beta_max = 20`, `acnr_hp_sigma_px = 1.5`,
  `acnr_window = 4`.
- `matrix_size` (required) — the reconstruction grid, `(nx, ny, nz)`; this function never sees a
  workspace's `ReconOptions`, so the caller states the grid. One slice when the rows were reduced.
  `fbp_filter = SoftFilter()`, `antialias = true`, `recon_rows = geom.n_rows`.
- `recon_method = :fbp` (the published chain) or `:hir`, with `hir_strength = 60`,
  `recon_projector = :dd_fast` and `hir_reference_kev = 70`. `:hir` reconstructs the basis pair
  with the penalized iterative reconstructor and nothing else changes: T-LBF, ACNR and the
  synthesis run as before. Each material is reconstructed in μ-equivalent units at the reference
  energy, weighted by its own inverse variance from the K-channel Fisher information (uniformly
  for `:cong`, which has none), through a one-row geometry when the rows were reduced — see
  [`reconstruct_basis_slice`](@ref) for why each of those is needed. Run once each way for the
  analytic and iterative stacks of one acquisition.
- `vmi_energies = (40, 70, 100, 140)`.
- `keep_sinograms` adds the decomposed pair; `keep_diagnostics` adds the estimator's per-ray
  maps (Fisher information, quality flags) for `:nchannel`.

The published photon-counting configuration is
`vmi_pipeline(; channels, basis, geom, to_backend, reduce_rows = true, use_tlbf = true)`.

Returns `(vmis, energies, images = (water, iodine), quality, elapsed_s, settings)`, with `vmis`
`(nx, ny, nz, n_energies)` in HU.
"""
function vmi_pipeline(;
        channels, basis, geom, to_backend = identity,
        method::Symbol = :nchannel,
        controls::NChannelControls = NChannelControls(),
        merge_groups = nothing,
        reduce_rows::Bool = false, rows = (:),
        use_tlbf::Bool = false,
        tlbf_alpha1::Real = 0.9, tlbf_alpha2::Real = 24.635648571666497,
        tlbf_radius::Integer = 2,
        use_acnr::Bool = true,
        acnr_passes::Integer = 4, acnr_beta_max::Real = 20.0,
        acnr_hp_sigma_px::Real = 1.5, acnr_window::Integer = 4,
        matrix_size,
        fbp_filter = SoftFilter(),
        antialias::Bool = true,
        recon_rows::Integer = geom.n_rows,
        recon_method::Symbol = :fbp,
        hir_strength::Integer = 60,
        recon_projector::Symbol = :dd_fast,
        hir_reference_kev::Real = 70.0,
        vmi_energies = (40, 70, 100, 140),
        tile_views::Integer = controls.tile_views,
        keep_sinograms::Bool = false,
        keep_diagnostics::Bool = false,
    )
    method in (:nchannel, :cong) ||
        throw(ArgumentError("method must be :nchannel or :cong, got $(method)"))
    recon_method in (:fbp, :hir) ||
        throw(ArgumentError("recon_method must be :fbp or :hir, got $(recon_method)"))
    hir = recon_method === :hir

    prepared = prepare_channels(; channels, basis, merge_groups, reduce_rows, rows)
    working_channels, working_basis = prepared.channels, prepared.basis

    decomposition = if method === :nchannel
        decompose_nchannel(
            channels = working_channels, basis = working_basis, controls = controls,
            to_backend = to_backend, tile_views = tile_views,
            # the Fisher information is the variance HIR weights the basis pair by
            keep_diagnostics = keep_diagnostics || hir,
        )
    else
        decompose_cong(channels = working_channels, basis = working_basis, to_backend = to_backend)
    end

    sino_iodine, sino_water = decomposition.sino_iodine, decomposition.sino_water
    if use_tlbf
        filtered = tlbf_denoise(
            sino_iodine, sino_water,
            total_expected_counts(
                sino_iodine, sino_water, working_basis.Φ, working_basis.μρ_I,
                working_basis.μρ_W; to_backend = to_backend,
            ),
            total_measured_counts(working_channels, working_basis.I0);
            alpha1 = tlbf_alpha1, alpha2 = tlbf_alpha2, radius = tlbf_radius,
            to_backend = to_backend,
        )
        sino_iodine, sino_water = filtered.sino_iodine, filtered.sino_water
    end

    # HIR runs each material in μ-equivalent units at the reference energy, weighted by that
    # material's inverse variance; FDK is linear and takes the sinograms as they are.
    scale_water = hir ? compute_mass_μ_at_energy(XA.Materials.water, Float64(hir_reference_kev)) : 1.0
    scale_iodine = hir ? compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(hir_reference_kev)) : 1.0
    weights_iodine, weights_water = if hir && method === :nchannel
        _basis_weights(decomposition.diagnostics.fisher, scale_iodine, scale_water)
    else
        (:uniform, :uniform)
    end
    reconstruct(one_sino, weights, scale) = reconstruct_basis_slice(
        one_sino, geom, matrix_size;
        to_backend = to_backend, filter = fbp_filter, n_rows = recon_rows, antialias = antialias,
        method = recon_method, hir_strength = hir_strength, projector = recon_projector,
        hir_weights = weights, scale = scale,
    )
    water_image = reconstruct(sino_water, weights_water, scale_water)
    iodine_image = reconstruct(sino_iodine, weights_iodine, scale_iodine)

    acnr_settings = nothing
    if use_acnr && acnr_passes > 0
        apply_acnr_kalender!(
            water_image, iodine_image;
            hp_sigma_px = acnr_hp_sigma_px, window = acnr_window,
            passes = acnr_passes, beta_max = acnr_beta_max,
        )
        acnr_settings = (
            hp_sigma_px = acnr_hp_sigma_px, window = acnr_window,
            passes = acnr_passes, beta_max = acnr_beta_max,
        )
    end

    result = (
        vmis = synthesize_vmi_stack(water_image, iodine_image, vmi_energies),
        energies = collect(vmi_energies),
        images = (water = water_image, iodine = iodine_image),
        quality = decomposition.quality,
        elapsed_s = decomposition.elapsed_s,
        settings = (
            method = method, n_channels = working_basis.n_channels,
            merge_groups = merge_groups, reduce_rows = reduce_rows, n_rows = prepared.n_rows,
            tlbf = use_tlbf ?
                (alpha1 = tlbf_alpha1, alpha2 = tlbf_alpha2, radius = tlbf_radius) : nothing,
            acnr = acnr_settings,
            recon = (;
                method = recon_method, matrix_size, antialias, recon_rows, filter = fbp_filter,
                hir = hir ? (;
                    strength = hir_strength, projector = recon_projector,
                    reference_kev = hir_reference_kev,
                    weights = method === :nchannel ? :fisher : :uniform,
                ) : nothing,
            ),
            controls = controls,
        ),
    )
    keep_sinograms && (result = merge(result, (sinograms = (iodine = sino_iodine, water = sino_water),)))
    keep_diagnostics && method === :nchannel &&
        (result = merge(result, (diagnostics = decomposition.diagnostics,)))
    return result
end

export NChannelControls, spectral_basis, spectral_basis_from_bins, spectral_basis_from_acquisitions
export reduce_detector_rows, merge_channels, prepare_channels
export nchannel_profile_tile!, decompose_nchannel, cong_material_basis, decompose_cong
export angular_antialias_response, reconstruct_basis_slice, synthesize_vmi_stack, vmi_pipeline
