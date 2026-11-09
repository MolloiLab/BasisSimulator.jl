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

The kernels bring only their multiple scatter: each kernel's own single scatter, a quadrature in
its water ellipse, is taken out when the library is loaded, and each coarse ray adds its own, the
same quadrature through the patient as its view's rays describe it (each ray's water-equivalent
length spread over its outline chord; BasisMonteCarlo's `single_scatter` in its `RayStrips`),
weighed by the same photons and escape factors, at the ray's own columns (`_scatter_single!`, on
the device). A thorax's mediastinum stops the scatter a ray beside it sends inward, and its lungs
let out what it sends outward, as water kernels cannot; for a patient that is its own water
ellipse the two single scatters are the same. In single precision the quadrature's yes/no
decisions (which strip, inside it or not, which cell of water's angular tables) are taken again in
double-float near their lines, so that they fall as in double precision.

Units: an energy integrator's scatter is added to its mean signal `S₁` (units of `E_ref`) and to
its second moment `S₂`, each photon weighted by the detector's `m₁` and `m₂/m₁` at the photon's
energy; a photon counter's is added to each bin's counts, a counted photon split over the bins as
the response table splits a photon of its energy (`R(E, b) / η(E)`, the library counting each
photon the sensor absorbs once).

The correction (the scanner's `scatter_correction`, on by default for both families: a photon
counter's counts, an energy integrator's DAS) estimates the scatter from the measured data alone: each coarse ray's
water-equivalent length from its measured transmission (scatter included, inverted through the
ray's own spectrum), the scatter of that water patient superposed, the estimate subtracted and the
lengths measured again (`_SCATTER_CORRECTION_PASSES` times). It never sees the phantom, so it is
realistic, not exact: it treats the patient as water and the view's outline as its
water-equivalent lengths' (`view = :wet`, and the rays' single scatter through that water).

Everything after the plan is built runs on the workspace's backend (the GPU, or Julia's threads on
the CPU): the walk, each ray's lengths and photons, the superposition (`scatter_kernels.jl`), the
correction's inversion, and the maps to dexels. The plan keeps its own copy of the library there.

Outside the library's grids (thickness 140–500 mm, aspect 0.6–2.5, 20–140 keV) the nearest
kernels stand in, with a warning once per session; an ellipse thinner than the thinnest kernel's
takes that kernel's multiple scatter scaled by the water the ray crosses against what the kernel's
own ray crosses (its single scatter is the ray's own).
"""

# The coarse rays' sampling
const _SCATTER_VIEW_STEP_DEG = 10.0
const _SCATTER_RAY_STEP_DEG = 0.25
const _SCATTER_GROUP_ROWS = 3
const _SCATTER_CORRECTION_PASSES = 2

# The superposition's buffers hold at most this many coarse rays: views beyond go in chunks
const _SCATTER_CHUNK_RAYS = 32768

# Materials below this density (g/cm³) are gases: outside the patient's outline, and not scattering
const _SCATTER_GAS_DENSITY = 0.01

"""
    ScatterPlan

What `simulate!` needs to superpose the scatter library on one workspace's acquisition, built with
the workspace: the coarse rays (a `CTGeometry` of their own, their fan angles and z groups), the
detector's weight of a photon in each output channel (`channels[energy, channel]`: an energy
integrator's `S₁` and `S₂`, a photon counter's bins), and on the device the walk's buffer, the
scatter field (library bins × channels × coarse views), the correction's estimate, the maps from
bins and coarse views to dexels and views, and

- `library`: the kernels (their multiple scatter), their column moments and path lengths, the
  grids, and the single-scatter quadrature's tables (`single`);
- `rays`: each coarse ray's photons by library energy and channel through water, by spectral
  energy for its materials, the weights of its measured signal and its air value, the materials'
  scattering and attenuation against water's, the maps from bins to coarse rays, and the rays'
  places and directions as double-floats and the library columns their own single scatter reaches;
- `scratch`: the superposition's buffers (`scatter_kernels.jl`).
"""
struct ScatterPlan{A4, A3, AI, AT, LB, RY, SC}
    kernels::ScatterKernels
    geom::CTGeometry
    views::Vector{Int}
    angle_deg::Vector{Float64}
    z_mm::Vector{Float64}
    channels::Matrix{Float64}
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
    library::LB
    rays::RY
    scratch::SC
end

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
                  correction, air, materials, ref, T)

The [`ScatterPlan`](@ref) of one workspace. `source(g)` is the source transmission (bowtie × heel,
`[cols, rows, energies]`, or `nothing`) at the rays of a geometry `g`; `detection[energy]` weighs
the measured signal of a photon (an energy integrator's `m₁`, a photon counter's counts per
photon); `channels[energy, channel]` weighs a scattered photon in each output channel, the first
`measured_channels` of them summing to the measured signal the correction reads (whose buffers are
empty without the `correction`); `air`
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
        correction::Bool,
        air::AbstractMatrix,
        materials,
        ref,
        T::Type,
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
    # a view's rays describe its patient (the ellipse, the strips): one ray describes none
    n_cc >= 2 || throw(
        ArgumentError(
            "the scatter superposition needs a fan of at least two coarse rays ($(2 * _SCATTER_RAY_STEP_DEG)°); this detector spans $(γ_max - γ_min)°"
        )
    )
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

    # ─── Every coarse ray's photons, and the weights of the signal it measures ───
    # `spectrum[e, i, g]`: the photons of each spectral energy, the ray's share of a library bin;
    # `lib_share[e, energy]`: each spectral energy between the library's
    trans = source(rays)
    cols = k.column_edges_deg
    ray_share = Δγ / (cols[2] - cols[1])
    nE = length(E)
    nE_lib = length(k.energies_keV)
    n_ch = size(channels, 2)
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
    lib_share = zeros(nE, nE_lib)
    for (e, Ek) in enumerate(E)
        g1, g2, t = _bracket(k.energies_keV, Ek)
        lib_share[e, g1] += 1 - t
        lib_share[e, g2] += t
    end
    lib_range = zeros(Int32, 2, nE_lib)
    for g in 1:nE_lib
        used = findall(!iszero, view(lib_share, :, g))
        lib_range[:, g] .= isempty(used) ? (1, 0) : (first(used), last(used))
    end
    photons = [spectrum[e, i, g] * channels[e, ch] for e in 1:nE, ch in 1:n_ch, i in 1:n_cc, g in 1:n_g]
    # through water, every process weighed alike
    water_photons = zeros(nE_lib, n_ch, n_cc, n_g)
    for g in 1:n_g, i in 1:n_cc, ch in 1:n_ch, e in 1:nE
        base = photons[e, ch, i, g]
        base == 0 && continue
        for gl in 1:nE_lib
            lib_share[e, gl] == 0 || (water_photons[gl, ch, i, g] += base * lib_share[e, gl])
        end
    end
    air_coarse = [footprint_mean(air, i, g) for i in 1:n_cc, g in 1:n_g]

    # ─── The materials against water, at the spectrum's energies ───
    # `kind`: 0 a gas, 1 like water, 2 another; `ratio[m, e, process]`: the material's Compton,
    # wide- and forward-Rayleigh scattering against water's; `dmu[m, band]`: its attenuation in
    # each band of the library's path energies against water's of the same water-equivalent length
    water = XA.Materials.water
    μ(m, x, kind) = ustrip(u"cm^-1", XA.linear_attenuation_coeff(m, x * u"keV", kind)) / 10
    bins = _rayleigh_bins(k)
    all_bins = eachindex(k.scatter_angle_edges_deg[2:end])
    band_keV(b) = max(1, 5(b - 1)):min(150, 5b - 1)
    nbands = size(k.paths, 1)
    band_mu_of(m) = [sum(compute_μ_at_energy(m, Float64(x)) / 10 for x in band_keV(b)) / length(band_keV(b)) for b in 1:nbands]
    n_m = length(materials)
    compton_water = [μ(water, Ek, XA.IncoherentScattering()) for Ek in E]
    rayleigh_water = [μ(water, Ek, XA.CoherentScattering()) for Ek in E]
    band_mu_water = band_mu_of(water)
    mu70_water = compute_μ_at_energy(water, 70.0) / 10
    kind = zeros(Int32, n_m)
    wet_per_mm = zeros(n_m)
    ratio = zeros(n_m, nE, 3)
    dmu = zeros(n_m, nbands)
    for (j, m) in enumerate(materials)
        density = hasproperty(m, :density) ? ustrip(u"g/cm^3", m.density) : 1.0
        mu70 = compute_μ_at_energy(m, 70.0) / 10
        fractions = hasproperty(m, :composition) ? m.composition : nothing
        for (e, Ek) in enumerate(E)
            gc, gw, gf = if fractions === nothing
                1.0, 1.0, 1.0
            else
                _angular_factor(k, fractions, water.composition, 1, Ek, all_bins),
                    _angular_factor(k, fractions, water.composition, 2, Ek, bins.wide),
                    _angular_factor(k, fractions, water.composition, 2, Ek, bins.forward)
            end
            rr = μ(m, Ek, XA.CoherentScattering())
            ratio[j, e, 1] = μ(m, Ek, XA.IncoherentScattering()) * gc / compton_water[e]
            ratio[j, e, 2] = rr * gw / rayleigh_water[e]
            ratio[j, e, 3] = rr * gf / rayleigh_water[e]
        end
        band_mu = band_mu_of(m)
        wet_per_mm[j] = mu70 / mu70_water
        dmu[j, :] .= band_mu .- band_mu_water .* wet_per_mm[j]
        like_water = all(==(1.0), view(ratio, j, :, :)) && band_mu == band_mu_water && mu70 == mu70_water
        kind[j] = density < _SCATTER_GAS_DENSITY ? 0 : like_water ? 1 : 2
    end
    # What the rays need from the walk is linear in each material's path length: its water-equivalent
    # length, its outline, whether it crosses a material other than water, and its materials'
    # attenuation and scattering against water's. The walk accumulates those sums (`lanes`) instead
    # of every material's length, as few as reproduce them all (`_scatter_feature_lanes`).
    lanes = _scatter_feature_lanes(kind, wet_per_mm, dmu, ratio)

    # ─── From the library's bins and the coarse views to dexels and views ───
    col_maps = [_bin_bracket(cols, γ) for γ in γ_dexel]
    row_maps = [_bin_bracket(rows_lib, z) for z in z_dexel]
    view_maps = map(1:n_v) do j
        q = clamp(searchsortedlast(views, j), 1, n_cv)
        q == n_cv ? (q, q, 0.0) : (q, q + 1, (j - views[q]) / (views[q + 1] - views[q]))
    end
    ray_maps = [_bin_bracket(cols, γ) for γ in angle_deg]
    group_maps = [_bin_bracket(rows_lib, z) for z in z_mm]
    nlc, nlr = length(cols) - 1, length(rows_lib) - 1
    all(≈(cols[2] - cols[1]), diff(cols)) || throw(
        ArgumentError(
            "the scatter kernels \"$(k.name)\" have columns of unequal widths; the superposition needs them uniform"
        )
    )
    nE_lib <= 64 || throw(ArgumentError("the scatter kernels \"$(k.name)\" have $(nE_lib) energies; the superposition takes at most 64"))
    device_int(v) = _device_copy(ref, Int32.(v))
    device_real(v) = _device_copy(ref, T.(v))
    # the library itself as it is (Float32): on the host, the kernels' own arrays
    device_library(x) = ref isa Array ? x : _device_copy(ref, x)
    pair(maps) = device_int(reduce(hcat, [[m[1], m[2]] for m in maps]))
    # each value as the two floats of the workspace's type that sum to it
    float_pairs(x) = reduce(hcat, [[T(v), T(v - T(v))] for v in x])
    node_deg = rad2deg.(asin.(clamp.(vec(k.node_offsets_mm) ./ k.source_to_isocenter_mm, -1, 1)))
    sid = k.source_to_isocenter_mm

    # The rays' own single scatter: the quadrature's tables, in the workspace's type (water's
    # angular tables as read), and the library columns it reaches from each coarse ray's own
    tables = _single_tables(
        T;
        cols,
        rows = rows_lib,
        sdd = k.source_to_detector_mm,
        energies = k.energies_keV,
        water = (mu = k.water_mu_per_mm, compton = k.water_compton_per_mm, rayleigh = k.water_rayleigh_per_mm),
        water_pdf = k.water_pdf,
        detector = (efficiency = k.efficiency, mu = k.detector_mu_per_mm, depth = k.detector_depth_mm),
        energy_weighted = k.energy_weighted,
        grid = k.grid,
        grid_height = k.grid_height_mm,
        grid_opening = k.grid_opening_mm,
        single = k.single_scatter,
        forward_deg = k.forward_rayleigh_deg,
    )
    C = k.single_scatter.columns_each_side
    n_win = k.grid === :none ? nlc : min(nlc, 2C + 1)
    window0 = [k.grid === :none ? 0 : clamp(searchsortedlast(cols, γ), 1, nlc) - C - 1 for γ in angle_deg]
    xs = [sid * sind(γ) for γ in angle_deg]
    ray_pairs, boundary_pairs = _ray_pairs(T, angle_deg, sid)

    library = (;
        kernels = device_library(vec(k.kernels)),
        moments = device_library(vec(k.moments)),
        paths = device_library(vec(k.paths)),
        node_deg = device_real(float_pairs(node_deg)),
        thicknesses = device_real(k.thicknesses_mm),
        log_aspects = device_real(log.(k.aspects)),
        impacts = device_real(k.impact_fractions),
        cols = device_real(cols),
        column_width = T(cols[2] - cols[1]),
        grid = T.(
            (
                first(k.thicknesses_mm),
                last(k.thicknesses_mm),
                first(k.aspects),
                last(k.aspects),
                last(k.impact_fractions),
            )
        ),
        sid2 = _pair(T, sid),
        π2 = _pair(T, Float64(π)),
        single = map(x -> x isa AbstractArray ? _device_copy(ref, x) : x, tables),
    )
    ray_consts = (;
        xs = device_real(xs),
        dxs = device_real(_ray_widths(xs)),
        angle_deg = device_real(angle_deg),
        angle_first = T(first(angle_deg)),
        angle_step = T(Δγ),
        pairs = device_real(ray_pairs),
        boundaries = device_real(boundary_pairs),
        window0 = device_int(window0),
        angle = device_real(float_pairs(angle_deg)),
        row_share = device_real(row_share),
        water_photons = device_real(water_photons),
        photons = device_real(photons),
        lib_share = device_real(lib_share),
        lib_range = _device_copy(ref, lib_range),
        detected = device_real(detected),
        water_mu_per_mm = device_real([compute_μ_at_energy(water, Ek) / 10 for Ek in E]),
        air = device_real(air_coarse),
        kind = _device_copy(ref, kind),
        lanes_Ct = lanes.Ct === nothing ? nothing : device_real(lanes.Ct),
        lanes_Gt = lanes.G === nothing ? nothing : device_real(permutedims(lanes.G)),
        other = any(==(2), kind),
        ray_cols = pair(ray_maps),
        ray_col_w = device_real(last.(ray_maps)),
        group_rows = pair(group_maps),
        group_row_w = device_real(last.(group_maps)),
    )
    n_ray = n_cc * n_g * n_cv
    n_chunk = n_cc * n_g * clamp(_SCATTER_CHUNK_RAYS ÷ (n_cc * n_g), 1, n_cv)       # whole views
    # the forward model's ellipses: the water-equivalent one, and the outline's through a material
    # other than water (`_scatter_forward!`), their single scatters' weights side by side
    n_ellipses = ray_consts.other ? 2 : 1
    scratch = (;
        # every ray's features (`[feature, column, group, view]`): the lane sums themselves with a
        # lane per feature, else the lane sums times the features (one matrix product)
        features = lanes.G === nothing ? nothing : similar(ref, T, size(lanes.G, 2), n_cc, n_g, n_cv),
        wet = similar(ref, T, n_cc, n_g, n_cv),
        outline = similar(ref, T, n_cc, n_g, n_cv),
        plain = similar(ref, Bool, n_ray),
        dmu = fill!(similar(ref, T, nbands, n_ray), zero(T)),
        weights = similar(ref, T, nE_lib, 3, n_ch, n_chunk),
        own = similar(ref, T, n_chunk, nE_lib, 3, n_ellipses * n_ch),
        ellipse = ntuple(_ -> similar(ref, T, 3, n_g, n_cv), n_ellipses),
        shapes = ntuple(_ -> similar(ref, T, 5, n_ray), n_ellipses),
        strips = similar(ref, T, 4, n_ray),
        clamped = fill!(similar(ref, Int32, n_ray), Int32(0)),
        corner = fill!(similar(ref, Int32, 8, n_chunk), Int32(0)),
        corner_weight = similar(ref, T, 8, n_chunk),
        single = similar(ref, T, n_chunk, nlr, n_ellipses * n_ch, n_win),
        factor = similar(ref, T, n_ch, 8, n_chunk),
        mix = similar(ref, T, 3nE_lib, n_ch, 8, n_chunk),
        boxes = similar(ref, T, 3, nlc, n_ch, 8, n_chunk),
        window = similar(ref, Int32, 2, n_ch, 8, n_chunk),
        rows = similar(ref, T, nlr, nlc, n_ch, n_g, n_chunk ÷ (n_cc * n_g)),
    )
    return ScatterPlan(
        k,
        rays,
        views,
        angle_deg,
        z_mm,
        Float64.(channels),
        similar(ref, T, lanes.Ct === nothing ? size(lanes.G, 1) : size(lanes.Ct, 1), n_cc, n_g, n_cv),
        correction ? similar(ref, T, n_cc, n_g, n_cv) : similar(ref, T, 0, 0, 0),
        fill!(similar(ref, T, nlc, nlr, n_ch, n_cv), zero(T)),
        fill!(similar(ref, T, (correction ? (nlc, nlr, measured_channels, n_cv) : (0, 0, 0, 0))...), zero(T)),
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
        library,
        ray_consts,
        scratch,
    )
end

# =============================================================================
# The rays of every coarse view
# =============================================================================

"""
    _scatter_feature_lanes(kind, wet_per_mm, dmu, ratio) -> (; Ct, G)

The walk's lanes for the scatter's rays. The features, per material (gases zero): its water-equivalent
length per mm (1), its outline (2), whether it is other than water (3), its attenuation against
water's in each band (`dmu`, 3 + b) and its Compton and wide/forward Rayleigh scattering against
water's at each spectral energy (`ratio`, 3 + nbands + (process − 1)·nE + e). A ray's feature is the
sum of its materials' path lengths times their features; the walk accumulates `Ct · L` (lanes ×
materials) and `G` (lanes × features) gives every feature from those sums, or `G = nothing` when
the sums are the features:

- up to 16 materials: one lane per material, each voxel's length selected into its own (`Ct =
  nothing`, `G` the features) — v0.16.3's arithmetic, and the fastest walk;
- more: features 1–3 as lanes of their own (exact: the water test reads feature 3's zero), and the
  others on an orthonormal basis: the singular vectors of the feature matrix, each feature scaled
  to its largest entry, down to 1e-8 of the largest singular value (below Float32's precision;
  1024 mixtures of 8 tissues: 25 lanes). The lanes' coefficients `U·S` and the map back `V` are
  orthonormal, so the map back amplifies no rounding (a basis of the features' own columns did,
  moving the scatter by up to a percent). It is used when it reproduces every feature to 1e-7 of
  its scale in at most `_PLEN_MAX_MATERIALS` lanes;
- otherwise one lane per material (no more materials than features) or per feature (`G = nothing`).
"""
function _scatter_feature_lanes(kind, wet_per_mm, dmu, ratio)
    n_m = length(kind)
    solid = kind .!= 0
    F = hcat(
        wet_per_mm .* solid, Float64.(solid), Float64.(kind .== 2), dmu .* solid,
        reshape(ratio, n_m, :) .* solid,
    )
    Q = size(F, 2)
    n_m <= _DIRECT_MAX_MATERIALS[] && return (Ct = nothing, G = F)        # a lane per material, selected
    rest = F[:, 4:Q]
    scale = [max(maximum(abs, c), floatmin(Float64)) for c in eachcol(rest)]
    restn = rest ./ scale'
    f = svd(restn)
    residual(r) = maximum(abs, f.U[:, 1:r] * Diagonal(f.S[1:r]) * f.V[:, 1:r]' .- restn; init = 0.0)
    r = min(count(>(_SCATTER_SVD_TOL * f.S[1]), f.S), length(f.S))
    while r < min(length(f.S), _PLEN_MAX_MATERIALS - 3) && residual(r) > 1.0e-7
        r += 1
    end
    if 3 + r <= _PLEN_MAX_MATERIALS && residual(r) <= 1.0e-7
        Ct = vcat(permutedims(F[:, 1:3]), permutedims(f.U[:, 1:r] .* f.S[1:r]'))
        G = zeros(3 + r, Q)
        G[1:3, 1:3] = Matrix{Float64}(I, 3, 3)
        G[4:(3 + r), 4:Q] = permutedims(f.V[:, 1:r] .* scale)
        return (; Ct, G)
    end
    return n_m <= Q ? (Ct = Matrix{Float64}(I, n_m, n_m), G = F) : (Ct = permutedims(F), G = nothing)
end

const _SCATTER_SVD_TOL = 1.0e-8

# Every ray's features (`_scatter_feature_lanes`) from the walk's lane sums: the sums themselves
# (a lane per feature), or one matrix product (a lane per material).
function _scatter_features!(p::ScatterPlan)
    F = p.scratch.features
    F === nothing && return p.paths
    Gt = p.rays.lanes_Gt
    mul!(reshape(F, size(Gt, 1), :), Gt, reshape(p.paths, size(Gt, 2), :))
    return F
end

# From the rays' features (`[feature, column, group, view]`): each ray's water-equivalent length and
# outline (mm), whether it crosses nothing but water-like materials, and if not its materials'
# attenuation against water's (`dmu[band, ray]`).
function _scatter_phantom_rays!(p::ScatterPlan, F)
    s = p.scratch
    Q = size(F, 1)
    nb = size(s.dmu, 1)
    let wet = s.wet, outline = s.outline, plain = s.plain, dmu = s.dmu, F = F, Q = Q, nb = nb
        AK.foreachindex(wet) do ray
            T = eltype(wet)
            base = Q * (ray - 1)
            w = 10 * F[1 + base]
            wet[ray] = w
            outline[ray] = 10 * F[2 + base]
            water = !(F[3 + base] > 0)
            plain[ray] = water
            if !water && w > 0
                for b in 1:nb
                    dmu[b, ray] = 10 * F[3 + b + base] / w
                end
            end
        end
    end
    return p
end

# `weights[energy, process, ch, ray]` for the first `n_used` channels of the `n_rays` rays from
# `ray0 + 1` on (a chunk): each ray's photons by library energy, process and channel, its
# materials' scattering against water's at each spectral energy (`plain[ray]`: water, every process
# alike).
function _scatter_weights!(p::ScatterPlan, n_used::Int, ray0::Int, n_rays::Int)
    s, r = p.scratch, p.rays
    nE_lib, n_ch = size(s.weights, 1), size(s.weights, 3)
    n_cc, n_g = size(s.wet, 1), size(s.wet, 2)
    F = s.features === nothing ? p.paths : s.features
    Q = size(F, 1)
    nE = size(r.photons, 1)
    nb = size(s.dmu, 1)
    let weights = s.weights, wet = s.wet, plain = s.plain, F = F, water = r.water_photons,
            photons = r.photons, lib_share = r.lib_share, lib_range = r.lib_range,
            nE_lib = nE_lib, n_ch = n_ch, n_cc = n_cc, n_g = n_g, Q = Q, nE = nE, nb = nb, n_used = n_used, ray0 = ray0
        _foreachindex(1:(nE_lib * 3 * n_ch * n_rays), AK.get_backend(weights)) do idx
            T = eltype(weights)
            q = idx - 1
            gl = q % nE_lib + 1
            q ÷= nE_lib
            pr = q % 3 + 1
            q ÷= 3
            ch = q % n_ch + 1
            ray = q ÷ n_ch + 1 + ray0
            i = (ray - 1) % n_cc + 1
            g = ((ray - 1) ÷ n_cc) % n_g + 1
            val = zero(T)
            if ch <= n_used
                if plain[ray]
                    val = water[gl, ch, i, g]
                elseif wet[ray] > 0
                    w = wet[ray]
                    for e in lib_range[1, gl]:lib_range[2, gl]
                        share = lib_share[e, gl]
                        share == 0 && continue
                        f = 10 * F[3 + nb + (pr - 1) * nE + e + Q * (ray - 1)] / w
                        val += photons[e, ch, i, g] * share * f
                    end
                end
            end
            weights[idx] = val
        end
    end
    return p
end

# The views in chunks, as many as the superposition's buffers hold: `(v0, n_v)`, the `n_v` views
# from `v0 + 1` on.
function _scatter_chunks(p::ScatterPlan)
    n_cv, per_chunk = size(p.scratch.wet, 3), size(p.scratch.rows, 5)
    return [(v0, min(per_chunk, n_cv - v0)) for v0 in 0:per_chunk:(n_cv - 1)]
end

# The corners of every ray of the chunk (`_scatter_corners!`) for the ellipse `e`'s shapes.
function _superpose_corners!(p::ScatterPlan, e::Int, n_used::Int, ray0::Int, n_rays::Int)
    s = p.scratch
    return _scatter_corners!(
        s.corner,
        s.corner_weight,
        s.factor,
        s.mix,
        s.boxes,
        s.window,
        s.shapes[e],
        s.weights,
        s.plain,
        s.dmu,
        p.rays.angle,
        p.library,
        n_used,
        ray0,
        n_rays,
    )
end

# The scatter of the `n_v` coarse views from `v0 + 1` on (one of `_scatter_chunks`) into
# `out[column, row, ch, view]` (its channels the first of the plan's): the rays' impacts from
# `scratch.wet`, the patient for their own single scatter from `scratch.wet` spread over `outline`,
# each view's ellipse measured from `ellipses[1]`, or the geometric mean of the scatters in the
# ellipses of `ellipses[1]` and `ellipses[2]`. It marks the rays outside the library's grids by the
# first ellipse in `scratch.clamped`.
#
# The rays' own single scatter is one pass over the patient for every ellipse (their weights side
# by side in its channels); with `single = false` it is the last call's (`scratch.single`), not
# computed again.
function _superpose!(
        out::AbstractArray,
        p::ScatterPlan,
        ellipses::Tuple,
        outline::AbstractArray,
        v0::Int,
        n_v::Int;
        single::Bool = true,
    )
    s, lib, r = p.scratch, p.library, p.rays
    n_used = size(out, 3)
    n_cc, n_g = size(s.wet, 1), size(s.wet, 2)
    ray0, n_rays = v0 * n_cc * n_g, n_v * n_cc * n_g
    n_e = length(ellipses)

    # each ellipse's rays, and the chunk's photons by library energy
    for (e, lengths) in enumerate(ellipses)
        _scatter_ellipses!(s.ellipse[e], r.xs, r.dxs, lengths)
        _scatter_ray_shapes!(s.shapes[e], s.clamped, s.wet, s.ellipse[e], r.xs, r.angle_deg, lib.grid, e == 1)
    end
    _scatter_weights!(p, n_used, ray0, n_rays)

    # the rays' own single scatter, every ellipse's weights at once (the first's corners last: its
    # deposit is next)
    if single
        _scatter_strips!(s.strips, r.pairs, s.wet, outline, lib.sid2, lib.π2)
        for e in n_e:-1:1
            _superpose_corners!(p, e, n_used, ray0, n_rays)
            _scatter_single_weights!(s.own, s.corner_weight, s.mix, s.plain, n_used, (e - 1) * n_used, ray0, n_rays)
        end
        _scatter_single!(
            s.single,
            s.own,
            s.strips,
            r.angle_deg,
            r.pairs,
            r.boundaries,
            r.window0,
            r.angle_first,
            r.angle_step,
            lib.single,
            ray0,
            n_rays,
            Val(n_e * n_used),
        )
    else
        _superpose_corners!(p, 1, n_used, ray0, n_rays)
    end

    # each ellipse's scatter, the second's combined with the first's
    for e in 1:n_e
        e > 1 && _superpose_corners!(p, e, n_used, ray0, n_rays)
        _scatter_deposit!(
            s.rows,
            s.corner,
            s.factor,
            s.mix,
            s.boxes,
            s.window,
            s.single,
            (e - 1) * n_used,
            r.window0,
            s.shapes[e],
            s.plain,
            lib,
            n_cc,
            n_used,
            ray0,
            n_v * n_g,
        )
        _scatter_rows!(out, s.rows, r.row_share, v0, n_v; combine = e > 1)
    end
    return out
end

# The warning for the rays the superpositions since the last forward scatter found outside the
# library's grids (`scratch.clamped`). It reads the device, so `simulate!` calls it last.
function _warn_clamped(p::ScatterPlan)
    k = p.kernels
    n = Int(sum(p.scratch.clamped))
    n > 0 && @warn "scatter: $(n) rays' ellipses lie outside the kernels' " *
        "$(first(k.thicknesses_mm))–$(last(k.thicknesses_mm)) mm and aspects " *
        "$(first(k.aspects))–$(last(k.aspects)); the nearest kernels stand in" maxlog = 1
    return n
end

"""
    _scatter_forward!(plan, phantom) -> plan.field

The scatter of `phantom` on every coarse view, from its own rays: the walk of each material's
path length along the coarse rays, then the superposition (`view = :mean`). Fills `plan.field` on
the device.
"""
function _scatter_forward!(p::ScatterPlan, phantom)
    extent = hasproperty(phantom, :extent) && phantom.extent !== nothing ? Tuple(Float64.(phantom.extent)) : nothing
    s = p.scratch
    dd_fast_lane_sums!(p.paths, phantom.mask, p.geom, p.rays.lanes_Ct; volume_extent = extent)
    _scatter_phantom_rays!(p, _scatter_features!(p))
    fill!(s.clamped, Int32(0))
    # the geometric mean with the outline's ellipse, which is another only through a material other
    # than water (a gas is in neither)
    ellipses = p.rays.other ? (s.wet, s.outline) : (s.wet,)
    for (v0, n_v) in _scatter_chunks(p)
        _superpose!(p.field, p, ellipses, s.outline, v0, n_v)
    end
    return p.field
end

# Each coarse ray's water-equivalent length (mm) from its measured signal, `measured / photons`
# against its air value, less the estimate's scatter at the ray when `subtract`: the length whose
# transmission through the ray's own spectrum and detector weights is that (Newton's method on the
# logarithm). Below a millimetre of water the ray is taken to miss the patient (the air
# calibration's own scale), as the forward model leaves the gases out.
function _scatter_water_lengths!(p::ScatterPlan, photons::Real, subtract::Bool)
    s, r = p.scratch, p.rays
    n_cc, n_g = size(s.wet, 1), size(s.wet, 2)
    nE = size(r.detected, 1)
    photons = eltype(s.wet)(photons)
    let wet = s.wet, measured = p.measured, estimate = p.estimate, air = r.air, detected = r.detected,
            μ = r.water_mu_per_mm, rc = r.ray_cols, rcw = r.ray_col_w, gr = r.group_rows, grw = r.group_row_w,
            n_cc = n_cc, n_g = n_g, nE = nE, n_est = size(p.estimate, 3), photons = photons, subtract = subtract
        AK.foreachindex(wet) do ray
            T = eltype(wet)
            o = one(T)
            q = ray - 1
            i = q % n_cc + 1
            g = (q ÷ n_cc) % n_g + 1
            v = q ÷ (n_cc * n_g) + 1
            scatter = zero(T)
            if subtract
                c0, c1, wc = rc[1, i], rc[2, i], rcw[i]
                r0, r1, wr = gr[1, g], gr[2, g], grw[g]
                for ch in 1:n_est
                    scatter += (o - wr) * ((o - wc) * estimate[c0, r0, ch, v] + wc * estimate[c1, r0, ch, v]) +
                        wr * ((o - wc) * estimate[c0, r1, ch, v] + wc * estimate[c1, r1, ch, v])
                end
            end
            t = (measured[ray] / photons - scatter) / max(air[i, g], T(1.0e-30))
            x = zero(T)
            if t < 1
                target = log(max(t, T(1.0e-6)))
                base = nE * (i - 1 + n_cc * (g - 1))
                μ̄ = zero(T)
                for e in 1:nE
                    μ̄ += detected[base + e] * μ[e]
                end
                x = -target / μ̄
                for _ in 1:30
                    s0, s1 = zero(T), zero(T)
                    for e in 1:nE
                        w = detected[base + e] * exp(-μ[e] * x)
                        s0 += w
                        s1 += w * μ[e]
                    end
                    s0 > 0 || break
                    step = (log(s0) - target) / (s1 / s0)
                    x = max(x + step, zero(T))
                    abs(step) < T(1.0e-4) && break
                end
            end
            wet[ray] = x < 1 ? zero(T) : x
        end
    end
    return p
end

"""
    _scatter_estimate!(plan, photons = 1) -> plan.estimate

The correction's estimate of the scatter, from the data alone. `plan.measured[column, group,
view] / photons` is each coarse ray's mean measured signal over its footprint, scatter included,
in the units in which the plan's air value is its air value and the fields are the scatter (the
signal of an energy integrator's `S₁`, or a photon counter's counts summed over its bins, per
incident photon). Each pass measures every ray's water-equivalent length from its transmission
less the last estimate, and superposes the scatter of that water patient (`view = :wet`); the rays'
own single scatter is the first pass's (a view's estimate depends on its own views alone, so each
chunk of views takes its passes in turn). Fills
`plan.estimate` on the device, its channels the measured signal's: `S₁` alone for an energy
integrator, every bin for a photon counter.
"""
function _scatter_estimate!(p::ScatterPlan, photons::Real = 1)
    s = p.scratch
    fill!(s.plain, true)
    for (v0, n_v) in _scatter_chunks(p), pass in 1:_SCATTER_CORRECTION_PASSES
        _scatter_water_lengths!(p, photons, pass > 1)
        # a water patient: its outline is its water-equivalent lengths
        _superpose!(p.estimate, p, (s.wet,), s.wet, v0, n_v; single = pass == 1)
    end
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
