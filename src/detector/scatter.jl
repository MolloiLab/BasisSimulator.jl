"""
    src/detector/scatter.jl

Patient scatter in `simulate!`: the Geant4 kernels of [`scatter_kernels`](@ref) superposed onto the
patient's own rays, for both detector families.

Scatter is smooth, so it is computed on coarse rays (`ScatterPlan`, built with the workspace):
views every `_SCATTER_VIEW_STEP_DEG` of gantry angle, rays every `_SCATTER_RAY_STEP_DEG` across the
detector's fan, and groups of detector rows `_SCATTER_GROUP_ROWS` library rows tall. The phantom is
walked along those rays for each material's path length (`dd_fast_material_paths!`), which gives
each ray its water-equivalent length (at 70 keV, the library's reference), its length through the
patient's outline (every material but gases) and its materials' Compton and Rayleigh scattering
and attenuation against water's. The superposition fills the library's bins (2° columns, rows of
10–20 mm at the detector) view by view, in the units of the detector's signal; each dexel then
reads its value interpolated between bins and views. The view stands in its water ellipse as
BasisMonteCarlo notebook 11 does (`view = :mean`, the geometric mean of the ellipse measured from
the water-equivalent lengths and the one measured from the outline).

Units: an energy integrator's scatter is added to its mean signal `S₁` (units of `E_ref`) and to
its second moment `S₂`, each photon weighted by the detector's `m₁` and `m₂/m₁` at the photon's
energy; a photon counter's is added to each bin's counts, a counted photon split over the bins as
the response table splits a photon of its energy (`R(E, b) / η(E)`, the library counting each
photon the sensor absorbs once).

The correction (`PCCTScanner(; scatter_correction)` for a photon counter; always, in the DAS, for an
energy integrator) estimates the scatter from the measured data alone: each coarse ray's
water-equivalent length from its measured transmission (scatter included, inverted through the
ray's own spectrum), the scatter of that water patient superposed, the estimate subtracted and the
lengths measured again (`_SCATTER_CORRECTION_PASSES` times). It never sees the phantom, so it is
realistic, not exact: it treats the patient as water and the view's outline as its
water-equivalent lengths' (`view = :wet`).

Outside the library's grids (thickness 140–500 mm, aspect 0.6–2.5, 20–140 keV) the nearest
kernels stand in, with a warning once per session; an ellipse thinner than the thinnest kernel's
takes that kernel scaled by the water the ray crosses against what the kernel's own ray crosses.
"""

# The coarse rays' sampling
const _SCATTER_VIEW_STEP_DEG = 10.0
const _SCATTER_RAY_STEP_DEG = 0.25
const _SCATTER_GROUP_ROWS = 3
const _SCATTER_CORRECTION_PASSES = 2

# Materials below this density (g/cm³) are gases: outside the patient's outline, and not scattering
const _SCATTER_GAS_DENSITY = 0.01

"""
    ScatterPlan

What `simulate!` needs to superpose the scatter library on one workspace's acquisition, built with
the workspace: the coarse rays (a `CTGeometry` of their own, their fan angles and z groups, each
library row's share of a group's beam), every ray's source spectrum and measured-signal weights,
the detector's weight of a photon in each output channel (an energy integrator's `S₁` and `S₂`, a
photon counter's bins), the materials' scattering and attenuation tables, and on the device the
walk's buffer, the scatter field (library bins × channels × coarse views), the correction's
estimate and the maps from bins and coarse views to dexels and views.
"""
struct ScatterPlan{A4, A3, AI, AT}
    kernels::ScatterKernels
    geom::CTGeometry
    views::Vector{Int}
    angle_deg::Vector{Float64}
    z_mm::Vector{Float64}
    row_share::Matrix{Float64}
    spectrum::Array{Float64, 3}
    detected::Array{Float64, 3}
    channels::Matrix{Float64}
    shares::Vector{Tuple{Int, Int, Float64}}
    air::Matrix{Float64}
    water_mu_per_mm::Vector{Float64}
    gas::Vector{Bool}
    waterlike::Vector{Bool}
    mu70_per_mm::Vector{Float64}
    mu70_water_per_mm::Float64
    compton::Matrix{Float64}
    rayleigh_wide::Matrix{Float64}
    rayleigh_forward::Matrix{Float64}
    compton_water::Vector{Float64}
    rayleigh_water::Vector{Float64}
    band_mu::Matrix{Float64}
    band_mu_water::Vector{Float64}
    ray_column::Vector{Tuple{Int, Int, Float64}}
    group_row::Vector{Tuple{Int, Int, Float64}}
    paths::A4
    measured::A3
    field::A4
    estimate::A4
    footprint_cols::AI
    footprint_rows::AI
    view_index::AI
    col_lo::AI
    col_hi::AI
    col_w::AT
    row_lo::AI
    row_hi::AI
    row_w::AT
    view_lo::AI
    view_hi::AI
    view_w::AT
end

_n_channels(p::ScatterPlan) = size(p.channels, 2)

# Linear interpolation of `x` between the centres of the bins with `edges`, clamped at the ends.
_bin_bracket(edges, x) = _bracket((edges[1:(end - 1)] .+ edges[2:end]) ./ 2, x)

_device_copy(ref, x::AbstractArray) = copyto!(similar(ref, eltype(x), size(x)...), x)

# The source transmission (bowtie × heel effect, `[cols, rows, energies]`) at the rays of a
# geometry, or `nothing` without either.
_scatter_source(bowtie, heel, energies) = function (g::CTGeometry)
    E = Float64.(energies)
    b = bowtie !== nothing && bowtie.name != "none" ? compute_bowtie_attenuation_spectral(bowtie, g, E) : nothing
    h = heel !== nothing ? compute_heel_spectral(heel, g, E) : nothing
    return b === nothing ? h : (h === nothing ? b : b .* h)
end

"""
    _scatter_plan(kernels, geom, energies, weights; source, detection, channels, measured_channels,
                  air, materials, ref, T)

The [`ScatterPlan`](@ref) of one workspace. `source(g)` is the source transmission (bowtie × heel,
`[cols, rows, energies]`, or `nothing`) at the rays of a geometry `g`; `detection[energy]` weighs
the measured signal of a photon (an energy integrator's `m₁`, a photon counter's counts per
photon); `channels[energy, channel]` weighs a scattered photon in each output channel, the first
`measured_channels` of them summing to the measured signal the correction reads; `air`
(`[cols, rows]`) is the measured signal of each dexel in air, in the same units as the data the
correction reads.
"""
function _scatter_plan(
        k::ScatterKernels,
        geom::CTGeometry,
        energies::AbstractVector,
        weights::AbstractVector;
        source,
        detection::AbstractVector,
        channels::AbstractMatrix,
        measured_channels::Int,
        air::AbstractMatrix,
        materials,
        ref,
        T::Type,
    )
    length(materials) <= _PLEN_MAX_MATERIALS || throw(
        ArgumentError(
            "scatter walks each material's path length, at most $(_PLEN_MAX_MATERIALS) materials; " *
                "the phantom has $(length(materials)): call compact_materials(phantom) first"
        )
    )
    sad_mm, sdd_mm = geom.SAD * 10, geom.SDD * 10
    if abs(sad_mm / k.source_to_isocenter_mm - 1) > 0.02 || abs(sdd_mm / k.source_to_detector_mm - 1) > 0.02
        @warn "the scatter kernels \"$(k.name)\" were made for a source–isocentre distance of " *
            "$(k.source_to_isocenter_mm) mm and a source–detector distance of $(k.source_to_detector_mm) mm; " *
            "this scanner's are $(sad_mm) and $(sdd_mm) mm" maxlog = 1
    end
    E = Float64.(energies)
    w̄ = Float64.(weights) ./ sum(Float64.(weights))
    outside = sum(w̄[(E .< first(k.energies_keV) - 0.5) .| (E .> last(k.energies_keV) + 0.5)])
    outside > 1.0e-3 && @warn "$(round(100 * outside; digits = 1)) % of the spectrum lies outside the scatter " *
        "kernels' $(first(k.energies_keV))–$(last(k.energies_keV)) keV; those photons take the nearest " *
        "energy's kernels" maxlog = 1

    # ─── The coarse rays: across the detector's fan, in groups of rows, every few views ───
    cc = column_center(Float64, geom)
    dexel_angle(c) = rad2deg(
        is_arc(geom) ? (c - cc) * geom.pixel_size / geom.SAD : atan((c - cc) * geom.pixel_size / geom.SAD)
    )
    γ_dexel = [dexel_angle(c) for c in 1:geom.n_cols]
    γ_min, γ_max = dexel_angle(0.5), dexel_angle(geom.n_cols + 0.5)
    n_cc = max(1, ceil(Int, (γ_max - γ_min) / _SCATTER_RAY_STEP_DEG))
    Δγ = (γ_max - γ_min) / n_cc
    angle_deg = [γ_min + (i - 0.5) * Δγ for i in 1:n_cc]

    mag = geom.SDD / geom.SAD
    z_dexel = [(r - (geom.n_rows + 1) / 2) * geom.pixel_row_size * 10 * mag for r in 1:geom.n_rows]
    width = geom.n_rows * geom.pixel_row_size * 10 * mag
    rows_lib = k.row_edges_mm
    row_bin = rows_lib[2] - rows_lib[1]
    n_g = clamp(round(Int, width / (_SCATTER_GROUP_ROWS * row_bin)), 1, geom.n_rows)
    h = width / n_g
    z_lo = [-width / 2 + (g - 1) * h for g in 1:n_g]
    z_mm = z_lo .+ h / 2
    row_share = [
        max(0.0, min(rows_lib[r + 1], z_lo[g] + h) - max(rows_lib[r], z_lo[g])) / row_bin
            for r in 1:(length(rows_lib) - 1), g in 1:n_g
    ]

    n_v = geom.n_angles
    Δθ = n_v > 1 ? abs(geom.angles[2] - geom.angles[1]) : 2π
    step = max(1, floor(Int, deg2rad(_SCATTER_VIEW_STEP_DEG) / Δθ))
    views = collect(1:step:n_v)
    last(views) == n_v || push!(views, n_v)
    n_cv = length(views)

    rays = CTGeometry(
        geom.SAD,
        geom.SDD,
        n_cv,
        n_g,
        n_cc,
        deg2rad(Δγ) * geom.SAD,
        geom.n_rows * geom.pixel_row_size / n_g,
        geom.angles[views],
        geom.source_positions[:, views],
        geom.detector_centers[:, views],
        geom.detector_u[:, views],
        geom.detector_v[:, views],
        geom.fov,
        geom.pitch,
        geom.table_feed,
        :arc,
        (0.5 - γ_min / Δγ) - (n_cc + 1) / 2,
    )

    # each coarse ray's footprint on the detector: the dexels it stands for
    fp_cols = zeros(Int32, 2, n_cc)
    for i in 1:n_cc
        inside = findall(c -> abs(γ_dexel[c] - angle_deg[i]) <= Δγ / 2, 1:geom.n_cols)
        isempty(inside) && (inside = [argmin(abs.(γ_dexel .- angle_deg[i]))])
        fp_cols[:, i] .= (first(inside), last(inside))
    end
    fp_rows = zeros(Int32, 2, n_g)
    for g in 1:n_g
        inside = findall(r -> z_lo[g] <= z_dexel[r] < z_lo[g] + h, 1:geom.n_rows)
        isempty(inside) && (inside = [argmin(abs.(z_dexel .- z_mm[g]))])
        fp_rows[:, g] .= (first(inside), last(inside))
    end
    footprint_mean(A, i, g) = sum(
        A[c, r] for c in fp_cols[1, i]:fp_cols[2, i], r in fp_rows[1, g]:fp_rows[2, g]
    ) / ((fp_cols[2, i] - fp_cols[1, i] + 1) * (fp_rows[2, g] - fp_rows[1, g] + 1))

    # ─── Every coarse ray's spectrum, and the weights of the signal it measures ───
    trans = source(rays)
    cols = k.column_edges_deg
    ray_share = Δγ / (cols[2] - cols[1])
    nE = length(E)
    spectrum = zeros(nE, n_cc, n_g)
    detected = zeros(nE, n_cc, n_g)
    for g in 1:n_g, i in 1:n_cc
        for e in 1:nE
            t = trans === nothing ? 1.0 : trans[i, g, e]
            spectrum[e, i, g] = w̄[e] * t * ray_share
            detected[e, i, g] = w̄[e] * t * detection[e]
        end
        s = sum(@view detected[:, i, g])
        s > 0 && (detected[:, i, g] ./= s)
    end
    shares = [_bracket(k.energies_keV, Ek) for Ek in E]
    air_coarse = [footprint_mean(air, i, g) for i in 1:n_cc, g in 1:n_g]

    # ─── The materials against water, at the spectrum's energies ───
    water = XA.Materials.water
    μ(m, x, kind) = ustrip(u"cm^-1", XA.linear_attenuation_coeff(m, x * u"keV", kind)) / 10
    bins = _rayleigh_bins(k)
    all_bins = eachindex(k.scatter_angle_edges_deg[2:end])
    band_keV(b) = max(1, 5(b - 1)):min(150, 5b - 1)
    nbands = size(k.paths, 1)
    band_mu_of(m) = [sum(compute_μ_at_energy(m, Float64(x)) / 10 for x in band_keV(b)) / length(band_keV(b)) for b in 1:nbands]
    n_m = length(materials)
    compton = zeros(n_m, nE)
    rayleigh_wide = zeros(n_m, nE)
    rayleigh_forward = zeros(n_m, nE)
    band_mu = zeros(n_m, nbands)
    gas = falses(n_m)
    mu70 = zeros(n_m)
    for (j, m) in enumerate(materials)
        density = hasproperty(m, :density) ? ustrip(u"g/cm^3", m.density) : 1.0
        gas[j] = density < _SCATTER_GAS_DENSITY
        mu70[j] = compute_μ_at_energy(m, 70.0) / 10
        fractions = hasproperty(m, :composition) ? m.composition : nothing
        for (e, Ek) in enumerate(E)
            gc, gw, gf = if fractions === nothing
                1.0, 1.0, 1.0
            else
                _angular_factor(k, fractions, water.composition, 1, Ek, all_bins),
                    _angular_factor(k, fractions, water.composition, 2, Ek, bins.wide),
                    _angular_factor(k, fractions, water.composition, 2, Ek, bins.forward)
            end
            compton[j, e] = μ(m, Ek, XA.IncoherentScattering()) * gc
            rr = μ(m, Ek, XA.CoherentScattering())
            rayleigh_wide[j, e] = rr * gw
            rayleigh_forward[j, e] = rr * gf
        end
        band_mu[j, :] .= band_mu_of(m)
    end
    compton_water = [μ(water, Ek, XA.IncoherentScattering()) for Ek in E]
    rayleigh_water = [μ(water, Ek, XA.CoherentScattering()) for Ek in E]
    band_mu_water = band_mu_of(water)
    mu70_water = compute_μ_at_energy(water, 70.0) / 10
    waterlike = [
        compton[j, :] == compton_water && rayleigh_wide[j, :] == rayleigh_water &&
            rayleigh_forward[j, :] == rayleigh_water && band_mu[j, :] == band_mu_water &&
            mu70[j] == mu70_water
            for j in 1:n_m
    ]

    # ─── From the library's bins and the coarse views to dexels and views ───
    col_maps = [_bin_bracket(cols, γ) for γ in γ_dexel]
    row_maps = [_bin_bracket(rows_lib, z) for z in z_dexel]
    view_maps = map(1:n_v) do j
        q = clamp(searchsortedlast(views, j), 1, n_cv)
        q == n_cv ? (q, q, 0.0) : (q, q + 1, (j - views[q]) / (views[q + 1] - views[q]))
    end
    n_ch = size(channels, 2)
    nlc, nlr = length(cols) - 1, length(rows_lib) - 1
    device_int(v) = _device_copy(ref, Int32.(v))
    device_real(v) = _device_copy(ref, T.(v))
    return ScatterPlan(
        k,
        rays,
        views,
        angle_deg,
        z_mm,
        row_share,
        spectrum,
        detected,
        Float64.(channels),
        shares,
        air_coarse,
        [compute_μ_at_energy(water, Ek) / 10 for Ek in E],
        collect(gas),
        waterlike,
        mu70,
        mu70_water,
        compton,
        rayleigh_wide,
        rayleigh_forward,
        compton_water,
        rayleigh_water,
        band_mu,
        band_mu_water,
        [_bin_bracket(cols, γ) for γ in angle_deg],
        [_bin_bracket(rows_lib, z) for z in z_mm],
        similar(ref, T, n_m, n_cc, n_g, n_cv),
        similar(ref, T, n_cc, n_g, n_cv),
        fill!(similar(ref, T, nlc, nlr, n_ch, n_cv), zero(T)),
        fill!(similar(ref, T, nlc, nlr, measured_channels, n_cv), zero(T)),
        device_int(vec(fp_cols)),
        device_int(vec(fp_rows)),
        device_int(views),
        device_int(first.(col_maps)),
        device_int(getindex.(col_maps, 2)),
        device_real(last.(col_maps)),
        device_int(first.(row_maps)),
        device_int(getindex.(row_maps, 2)),
        device_real(last.(row_maps)),
        device_int(first.(view_maps)),
        device_int(getindex.(view_maps, 2)),
        device_real(last.(view_maps)),
    )
end

# =============================================================================
# The rays of one coarse view and group
# =============================================================================

# From the walk's path lengths (cm, `[material, column, group, view]`): each ray's water-equivalent
# length, its outline and its materials' ratios to water.
function _phantom_rays(p::ScatterPlan, L::AbstractArray, v::Int, g::Int)
    n_cc = length(p.angle_deg)
    nE = size(p.spectrum, 1)
    nbands = length(p.band_mu_water)
    wet = zeros(n_cc)
    outline = zeros(n_cc)
    plain = true
    for i in 1:n_cc, m in axes(L, 1)
        p.gas[m] && continue
        Lm = 10 * Float64(L[m, i, g, v])
        Lm > 0 || continue
        wet[i] += Lm * p.mu70_per_mm[m] / p.mu70_water_per_mm
        outline[i] += Lm
        p.waterlike[m] || (plain = false)
    end
    spectrum = view(p.spectrum, :, :, g)
    plain && return (;
        angle_deg = p.angle_deg, wet_mm = wet, outline_mm = outline, spectrum,
        compton = nothing, rayleigh_wide = nothing, rayleigh_forward = nothing, delta_mu_per_mm = nothing,
    )

    compton = ones(nE, n_cc)
    rayleigh_wide = ones(nE, n_cc)
    rayleigh_forward = ones(nE, n_cc)
    δμ = zeros(nbands, n_cc)
    for i in 1:n_cc
        wet[i] > 0 || continue
        fill!(view(compton, :, i), 0.0)
        fill!(view(rayleigh_wide, :, i), 0.0)
        fill!(view(rayleigh_forward, :, i), 0.0)
        for m in axes(L, 1)
            p.gas[m] && continue
            Lm = 10 * Float64(L[m, i, g, v])
            Lm > 0 || continue
            for e in 1:nE
                compton[e, i] += Lm * p.compton[m, e]
                rayleigh_wide[e, i] += Lm * p.rayleigh_wide[m, e]
                rayleigh_forward[e, i] += Lm * p.rayleigh_forward[m, e]
            end
            scale = p.mu70_per_mm[m] / p.mu70_water_per_mm
            for b in 1:nbands
                δμ[b, i] += (Lm / wet[i]) * (p.band_mu[m, b] - p.band_mu_water[b] * scale)
            end
        end
        for e in 1:nE
            compton[e, i] /= wet[i] * p.compton_water[e]
            rayleigh_wide[e, i] /= wet[i] * p.rayleigh_water[e]
            rayleigh_forward[e, i] /= wet[i] * p.rayleigh_water[e]
        end
    end
    return (;
        angle_deg = p.angle_deg, wet_mm = wet, outline_mm = outline, spectrum,
        compton, rayleigh_wide, rayleigh_forward, delta_mu_per_mm = δμ,
    )
end

# The water patient the correction estimates: water-equivalent lengths alone.
_water_rays(p::ScatterPlan, wet::AbstractMatrix, g::Int) = (;
    angle_deg = p.angle_deg,
    wet_mm = wet[:, g],
    outline_mm = wet[:, g],
    spectrum = view(p.spectrum, :, :, g),
    compton = nothing,
    rayleigh_wide = nothing,
    rayleigh_forward = nothing,
    delta_mu_per_mm = nothing,
)

# The water-equivalent length (mm) whose transmission, through the ray's own spectrum and detector
# weights, is `t` (relative to air): Newton's method on the logarithm.
function _water_length(p::ScatterPlan, t::Real, i::Int, g::Int)
    t >= 1 && return 0.0
    target = log(max(t, 1.0e-6))
    w = view(p.detected, :, i, g)
    μ = p.water_mu_per_mm
    μ̄ = sum(w[e] * μ[e] for e in eachindex(w))
    x = -target / μ̄
    for _ in 1:30
        s0, s1 = 0.0, 0.0
        for e in eachindex(w)
            q = w[e] * exp(-μ[e] * x)
            s0 += q
            s1 += q * μ[e]
        end
        s0 > 0 || break
        step = (log(s0) - target) / (s1 / s0)
        x = max(x + step, 0.0)
        abs(step) < 1.0e-4 && break
    end
    return x
end

# =============================================================================
# The scatter of every coarse view
# =============================================================================

# `rays(v, g)` gives the rays of coarse view `v` and group `g`; `view` is :mean (the geometric mean
# of the wet and the outline ellipses' scatter) or :wet. Fills `field[column, row, channel, view]`
# for the channels of `channels[energy, channel]`.
function _scatter_views!(field::Array{Float64, 4}, p::ScatterPlan, rays, channels::AbstractMatrix; view::Symbol)
    k = p.kernels
    n_cv = size(field, 4)
    n_ch = size(channels, 2)
    n_g = size(p.row_share, 2)
    clamped = Threads.Atomic{Int}(0)
    chunks = collect(Iterators.partition(1:n_cv, cld(n_cv, max(1, Threads.nthreads()))))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            scratch = _SuperposeScratch(k, n_ch)
            wet_field = zeros(size(field, 1), size(field, 2), n_ch)
            outline_field = similar(wet_field)
            for v in chunk
                group_rays = [rays(v, g) for g in 1:n_g]
                fill!(wet_field, 0.0)
                for (g, r) in enumerate(group_rays)
                    n = _superpose!(
                        wet_field,
                        k,
                        merge(r, (; ellipse_mm = r.wet_mm)),
                        Base.view(p.row_share, :, g),
                        channels,
                        p.shares,
                        scratch,
                    )
                    Threads.atomic_add!(clamped, n)
                end
                if view === :mean && any(r -> r.outline_mm != r.wet_mm, group_rays)
                    fill!(outline_field, 0.0)
                    for (g, r) in enumerate(group_rays)
                        _superpose!(
                            outline_field,
                            k,
                            merge(r, (; ellipse_mm = r.outline_mm)),
                            Base.view(p.row_share, :, g),
                            channels,
                            p.shares,
                            scratch,
                        )
                    end
                    field[:, :, :, v] .= sqrt.(wet_field .* outline_field)
                else
                    field[:, :, :, v] .= wet_field
                end
            end
        end
    end
    foreach(wait, tasks)
    clamped[] > 0 && @warn "scatter: $(clamped[]) rays' ellipses lie outside the kernels' " *
        "$(first(k.thicknesses_mm))–$(last(k.thicknesses_mm)) mm and aspects " *
        "$(first(k.aspects))–$(last(k.aspects)); the nearest kernels stand in" maxlog = 1
    return field
end

"""
    _scatter_forward!(plan, phantom) -> plan.field

The scatter of `phantom` on every coarse view, from its own rays: the walk of each material's
path length along the coarse rays, then the superposition (`view = :mean`). Fills `plan.field` on
the device.
"""
function _scatter_forward!(p::ScatterPlan, phantom)
    extent = hasproperty(phantom, :extent) && phantom.extent !== nothing ? Tuple(Float64.(phantom.extent)) : nothing
    dd_fast_material_paths!(p.paths, phantom.mask, p.geom; volume_extent = extent)
    L = Array(p.paths)
    field = zeros(size(p.field)...)
    _scatter_views!(field, p, (v, g) -> _phantom_rays(p, L, v, g), p.channels; view = :mean)
    copyto!(p.field, eltype(p.field).(field))
    return p.field
end

# The value of a field (library bins × channels × coarse views, on the host) at a coarse ray.
function _field_at_ray(p::ScatterPlan, field::Array{Float64, 4}, ch::Int, i::Int, g::Int, v::Int)
    c0, c1, wc = p.ray_column[i]
    r0, r1, wr = p.group_row[g]
    return (1 - wr) * ((1 - wc) * field[c0, r0, ch, v] + wc * field[c1, r0, ch, v]) +
        wr * ((1 - wc) * field[c0, r1, ch, v] + wc * field[c1, r1, ch, v])
end

"""
    _scatter_estimate!(plan, measured) -> plan.estimate

The correction's estimate of the scatter, from the data alone. `measured[column, group, view]` is
each coarse ray's mean measured signal over its footprint, scatter included, in the units in
which `plan.air` is its air value and the fields are the scatter (the signal of an energy
integrator's `S₁`, or a photon counter's counts summed over its bins). Each pass measures every
ray's water-equivalent length from its transmission less the last estimate, and superposes the
scatter of that water patient (`view = :wet`). Fills `plan.estimate` on the device, its channels
the measured signal's: `S₁` alone for an energy integrator, every bin for a photon counter.
"""
function _scatter_estimate!(p::ScatterPlan, measured::AbstractArray)
    n_cc, n_g, n_cv = size(measured)
    n_est = size(p.estimate, 3)
    channels = p.channels[:, 1:n_est]
    estimate = zeros(size(p.estimate)...)
    wet = zeros(n_cc, n_g, n_cv)
    for pass in 1:_SCATTER_CORRECTION_PASSES
        Threads.@threads for v in 1:n_cv
            for g in 1:n_g, i in 1:n_cc
                s = pass == 1 ? 0.0 : sum(_field_at_ray(p, estimate, ch, i, g, v) for ch in 1:n_est)
                t = (Float64(measured[i, g, v]) - s) / max(p.air[i, g], 1.0e-30)
                x = _water_length(p, t, i, g)
                # below a millimetre of water the ray is taken to miss the patient (the air calibration's
                # own scale), as the forward model leaves the gases out
                wet[i, g, v] = x < 1 ? 0.0 : x
            end
        end
        fill!(estimate, 0.0)
        _scatter_views!(estimate, p, (v, g) -> _water_rays(p, view(wet, :, :, v), g), channels; view = :wet)
    end
    copyto!(p.estimate, eltype(p.estimate).(estimate))
    return p.estimate
end

# =============================================================================
# On the device
# =============================================================================

# The mean of `src[col, row, view]` over each coarse ray's footprint, into `out[column, group,
# view]`; with `i0` (`[cols, rows, …]`), of the counts `i0[col, row, plane] · exp(-src)`, added to
# what `out` holds.
function _reduce_footprints!(
        out::AbstractArray{T, 3},
        p::ScatterPlan,
        src::AbstractArray{T, 3};
        i0 = nothing,
        plane::Int = 1,
        accumulate::Bool = false,
    ) where {T}
    accumulate || fill!(out, zero(T))
    n_cc, n_g = Int32(size(out, 1)), Int32(size(out, 2))
    nc, nr = Int32(size(src, 1)), Int32(size(src, 2))
    has_i0 = i0 !== nothing
    let out = out, src = src, fc = p.footprint_cols, fr = p.footprint_rows, vi = p.view_index,
            i0 = has_i0 ? i0 : p.col_w, h = has_i0, off = Int32(plane - 1) * nc * nr,
            n_cc = n_cc, n_g = n_g, nc = nc
        AK.foreachindex(out) do idx
            q = Int32(idx - 1)
            i = q % n_cc + Int32(1)
            g = (q ÷ n_cc) % n_g + Int32(1)
            v = q ÷ (n_cc * n_g) + Int32(1)
            iv = vi[v]
            acc = zero(T)
            c0, c1 = fc[Int32(2) * i - Int32(1)], fc[Int32(2) * i]
            r0, r1 = fr[Int32(2) * g - Int32(1)], fr[Int32(2) * g]
            for r in r0:r1, c in c0:c1
                x = src[c, r, iv]
                acc += h ? i0[c + (r - Int32(1)) * nc + off] * exp(-x) : x
            end
            out[idx] += acc / T((c1 - c0 + Int32(1)) * (r1 - r0 + Int32(1)))
        end
    end
    return out
end

# A field's value at dexel (col, row) of view `iv`: bilinear between the library's bins,
# linear between coarse views.
@inline function _field_at(F, ch, col, row, iv, cl, chi, cw, rl, rh, rw, vl, vh, vw)
    c0, c1, wc = cl[col], chi[col], cw[col]
    r0, r1, wr = rl[row], rh[row], rw[row]
    v0, v1, wv = vl[iv], vh[iv], vw[iv]
    o = one(wc)
    lo = (o - wr) * ((o - wc) * F[c0, r0, ch, v0] + wc * F[c1, r0, ch, v0]) +
        wr * ((o - wc) * F[c0, r1, ch, v0] + wc * F[c1, r1, ch, v0])
    hi = (o - wr) * ((o - wc) * F[c0, r0, ch, v1] + wc * F[c1, r0, ch, v1]) +
        wr * ((o - wc) * F[c0, r1, ch, v1] + wc * F[c1, r1, ch, v1])
    return (o - wv) * lo + wv * hi
end

# An energy integrator's signal with the scatter added: S₁ (as `-log S₁` in `sinogram`) and the
# dispersion S₂/S₁.
function _add_eict_scatter!(sinogram::AbstractArray{T, 3}, dispersion::AbstractArray{T, 3}, p::ScatterPlan) where {T}
    nc, nr = Int32(size(sinogram, 1)), Int32(size(sinogram, 2))
    let s = sinogram, d = dispersion, F = p.field, cl = p.col_lo, chi = p.col_hi, cw = p.col_w,
            rl = p.row_lo, rh = p.row_hi, rw = p.row_w, vl = p.view_lo, vh = p.view_hi, vw = p.view_w,
            nc = nc, nr = nr
        AK.foreachindex(s) do idx
            q = Int32(idx - 1)
            col = q % nc + Int32(1)
            row = (q ÷ nc) % nr + Int32(1)
            iv = q ÷ (nc * nr) + Int32(1)
            s1 = _field_at(F, Int32(1), col, row, iv, cl, chi, cw, rl, rh, rw, vl, vh, vw)
            s2 = _field_at(F, Int32(2), col, row, iv, cl, chi, cw, rl, rh, rw, vl, vh, vw)
            S1 = exp(-s[idx])
            total = S1 + s1
            d[idx] = (d[idx] * S1 + s2) / total
            s[idx] = -log(total)
        end
    end
    return sinogram
end

# An energy integrator's DAS: the measured signal (`sinogram`, relative to `I0v`) less the
# estimated scatter, held at one quantum, as a log line integral.
function _subtract_eict_scatter!(sinogram::AbstractArray{T, 3}, p::ScatterPlan, I0v::T) where {T}
    nc, nr = Int32(size(sinogram, 1)), Int32(size(sinogram, 2))
    let s = sinogram, F = p.estimate, cl = p.col_lo, chi = p.col_hi, cw = p.col_w,
            rl = p.row_lo, rh = p.row_hi, rw = p.row_w, vl = p.view_lo, vh = p.view_hi, vw = p.view_w,
            nc = nc, nr = nr, floor = one(T) / I0v
        AK.foreachindex(s) do idx
            q = Int32(idx - 1)
            col = q % nc + Int32(1)
            row = (q ÷ nc) % nr + Int32(1)
            iv = q ÷ (nc * nr) + Int32(1)
            e = _field_at(F, Int32(1), col, row, iv, cl, chi, cw, rl, rh, rw, vl, vh, vw)
            s[idx] = -log(max(s[idx] - e, floor))
        end
    end
    return sinogram
end

# A photon counter's bins with the scatter added (`sign = 1`, the expected counts before
# detection) or the estimate subtracted (`sign = -1`, held at one count): `I0[col, row, b]` is
# each ray's air response, `photons` the incident photons per pixel the fields are relative to.
function _pcct_scatter!(bins, I0::AbstractArray{T, 3}, p::ScatterPlan, photons::Real, sign::Int) where {T}
    nc, nr = Int32(size(I0, 1)), Int32(size(I0, 2))
    m = nc * nr
    F = sign > 0 ? p.field : p.estimate
    floor = sign > 0 ? T(1.0e-30) : one(T)
    for (b, bin) in enumerate(bins)
        let s = bin, i0 = I0, F = F, cl = p.col_lo, chi = p.col_hi, cw = p.col_w,
                rl = p.row_lo, rh = p.row_hi, rw = p.row_w, vl = p.view_lo, vh = p.view_hi, vw = p.view_w,
                nc = nc, nr = nr, m = m, off = Int32(b - 1) * m, ch = Int32(b), n = T(sign * photons), floor = floor
            AK.foreachindex(s) do idx
                q = Int32(idx - 1)
                col = q % nc + Int32(1)
                row = (q ÷ nc) % nr + Int32(1)
                iv = q ÷ m + Int32(1)
                i0b = i0[q % m + Int32(1) + off]
                N = i0b * exp(-s[idx]) + n * _field_at(F, ch, col, row, iv, cl, chi, cw, rl, rh, rw, vl, vh, vw)
                s[idx] = -log(max(N, floor) / i0b)
            end
        end
    end
    return bins
end
