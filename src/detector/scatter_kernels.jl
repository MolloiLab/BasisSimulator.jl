"""
    src/detector/scatter_kernels.jl

The Geant4 scatter library, read as plain data. BasisMonteCarlo (`lib/BasisMonteCarlo`, notebook
10) fires pencil beams through water elliptical cylinders and records, on each scanner's detector,
the scatter every ray makes: kernels indexed by the ellipse's thickness along the ray `T`, its
aspect `r` (width ÷ thickness), the ray's impact `u` (its offset as a fraction of the half-width
the tilted ray sees) and the photon energy `E`, with the part sent by a Compton scattering, by a
wide Rayleigh scattering and by a forward (< 10°) Rayleigh scattering kept apart. The library is a
TOML header and a little-endian Float32 `.bin`; this file carries its own copy of BasisMonteCarlo's
reader (`read_scatter_library`), so BasisSimulator never loads BasisMonteCarlo.

Each scanner's entry ships as its own Julia artifact (`SCATTER_KERNEL_ARTIFACTS`), fetched lazily by
[`scatter_kernels`](@ref) and cached per process. A scanner names its entry with its
`scatter_kernels` field.

The superposition (`_superpose!`) is BasisMonteCarlo's `library_scan`, ported: each view stands in
the water ellipse of its rays' water-equivalent lengths; each ray takes the kernel of the impact
whose chord is its own water-equivalent length, its Compton, wide- and forward-Rayleigh parts scaled
by its materials' scattering against water's and by how much less of each gets out of them, and is
moved across the fan to the ray's own angle, column by column as boxes of the width the scatter
landed with.
"""

import TOML

# =============================================================================
# The library on disk
# =============================================================================

"""
    read_scatter_library(header_path; entry = nothing) -> Dict{String, Any}

Read a scatter library written by BasisMonteCarlo's `write_scatter_library`, with nothing but Base
and TOML: the header's contents, with `"arrays"` replaced by
`Dict(scanner => Dict(kind => Array{Float32}))`. With `entry`, only that scanner's arrays and the
shared `elements` table are read. A copy of BasisMonteCarlo's reference reader.
"""
function read_scatter_library(header_path::AbstractString; entry::Union{Nothing, AbstractString} = nothing)
    header = TOML.parsefile(header_path)
    data_path = joinpath(dirname(header_path), header["data_file"])
    arrays = Dict{String, Dict{String, Array{Float32}}}()
    open(data_path) do io
        for item in header["arrays"]
            entry === nothing || item["scanner"] in (entry, "elements") || continue
            A = Array{Float32}(undef, Tuple(item["shape"]))
            seek(io, item["offset_bytes"])
            read!(io, A)
            A .= ltoh.(A)
            get!(arrays, item["scanner"], Dict{String, Array{Float32}}())[item["kind"]] = A
        end
    end
    header["arrays"] = arrays
    return header
end

"""
    ScatterKernels

One scanner's entry of the scatter library, as [`scatter_kernels`](@ref) returns it. All per
incident photon of a pencil beam; signal in keV for an energy-integrating detector
(`energy_weighted`), in counts for a photon-counting one.

- `name`, `path`: the entry and the header it was read from.
- `source_to_isocenter_mm`, `source_to_detector_mm`, `column_edges_deg` (across the fan from the
  ray, 2° bins), `row_edges_mm` (along z at the detector): the scanner the kernels were made on.
- `thicknesses_mm`, `aspects`, `impact_fractions`, `energies_keV`: the grids.
- `kernels[row, column, energy, process, impact, aspect, thickness]`: the scatter whose last
  scattering was a Compton (process 1), a Rayleigh wider than `forward_rayleigh_deg` (2) or a
  forward Rayleigh (3), and all of it (4).
- `moments[column, k, energy, process, impact, aspect, thickness]`: each column's scatter (k = 1)
  and its first and second moments of where in the column it landed, in degrees from the
  column's centre (k = 2, 3).
- `paths[band, process, energy, impact, aspect, thickness]`: the mean length (mm) the detected
  photons of each process (1 Compton, 2 Rayleigh) travelled in the water with an energy in each
  5-keV band of `path_energy_edges_keV`.
- `angles[process, bin, energy]`: the detected scatter by its last scattering's process and angle
  (bins of `scatter_angle_edges_deg`); `element_scattering[element, process, bin, keV]` (Z of
  `element_Z`, 1–150 keV): each element's Compton and Rayleigh cross-sections split over those
  bins, cm²/g.
- `node_offsets_mm[impact, aspect, thickness]`: where each kernel's own ray passes its ellipse's
  axis.
"""
struct ScatterKernels
    name::String
    path::String
    energy_weighted::Bool
    source_to_isocenter_mm::Float64
    source_to_detector_mm::Float64
    column_edges_deg::Vector{Float64}
    row_edges_mm::Vector{Float64}
    thicknesses_mm::Vector{Float64}
    aspects::Vector{Float64}
    impact_fractions::Vector{Float64}
    energies_keV::Vector{Float64}
    kernels::Array{Float32, 7}
    moments::Array{Float32, 7}
    paths::Array{Float32, 6}
    angles::Array{Float32, 3}
    water_mu_per_mm::Vector{Float64}
    scatter_angle_edges_deg::Vector{Float64}
    forward_rayleigh_deg::Float64
    path_energy_edges_keV::Vector{Float64}
    element_Z::Vector{Int}
    element_scattering::Array{Float32, 4}
    node_offsets_mm::Array{Float64, 3}
end

function ScatterKernels(header::Dict, name::AbstractString, path::AbstractString)
    header["format"] == "BasisSimulator scatter library" ||
        throw(ArgumentError("$(path) is not a BasisSimulator scatter library"))
    header["format_version"] == 2 ||
        throw(ArgumentError("$(path) is scatter library format $(header["format_version"]); BasisSimulator reads format 2"))
    names = [s["name"] for s in header["scanners"]]
    k = findfirst(==(name), names)
    k === nothing && throw(
        ArgumentError("the scatter library $(path) has no entry \"$(name)\"; its entries are $(names)")
    )
    s = header["scanners"][k]
    A = header["arrays"][name]
    for kind in ("scatter", "rayleigh", "forward", "paths", "offsets", "spreads", "angles")
        haskey(A, kind) || throw(
            ArgumentError("the entry \"$(name)\" of $(path) has no `$(kind)` array (a v2 library has it)")
        )
    end
    axes = header["axes"]
    thicknesses = Float64.(axes["thickness_mm"])
    aspects = Float64.(axes["aspect"])
    impacts = Float64.(axes["impact_fraction"])
    energies = Float64.(axes["energy_keV"])

    # Compton, wide Rayleigh, forward Rayleigh and all the scatter, each (row, column) slice next to
    # its energies and processes, so that one corner of the grid is one contiguous block
    S, R, F = A["scatter"], A["rayleigh"], A["forward"]
    O, Sd = A["offsets"], A["spreads"]
    nc, nr, nu, na, nT, nE = size(S)
    kernels = Array{Float32}(undef, nr, nc, nE, 4, nu, na, nT)
    moments = Array{Float32}(undef, nc, 3, nE, 4, nu, na, nT)
    for iT in 1:nT, ia in 1:na, iu in 1:nu, iE in 1:nE
        total = view(S, :, :, iu, ia, iT, iE)
        rayleigh = view(R, :, :, iu, ia, iT, iE)
        forward = view(F, :, :, iu, ia, iT, iE)
        kernels[:, :, iE, 1, iu, ia, iT] .= transpose(total .- rayleigh)
        kernels[:, :, iE, 2, iu, ia, iT] .= transpose(rayleigh .- forward)
        kernels[:, :, iE, 3, iu, ia, iT] .= transpose(forward)
        kernels[:, :, iE, 4, iu, ia, iT] .= transpose(total)
        for p in 1:4, c in 1:nc
            mass = sum(view(kernels, :, c, iE, p, iu, ia, iT))
            o, sd = O[c, iu, ia, iT, iE], Sd[c, iu, ia, iT, iE]
            moments[c, 1, iE, p, iu, ia, iT] = mass
            moments[c, 2, iE, p, iu, ia, iT] = mass * o
            moments[c, 3, iE, p, iu, ia, iT] = mass * (sd^2 + o^2)
        end
    end

    # where each kernel's own ray passes the axis of its ellipse
    node_offsets = [
        _kernel_offset(T / 2, _kernel_half_width(T, r), u)
            for u in impacts, r in aspects, T in thicknesses
    ]
    scattering = header["arrays"]["elements"]["scattering"]
    return ScatterKernels(
        String(name),
        String(path),
        Bool(s["energy_weighted"]),
        Float64(s["source_to_isocenter_mm"]),
        Float64(s["source_to_detector_mm"]),
        Float64.(s["column_edges_deg"]),
        Float64.(s["row_edges_mm"]),
        thicknesses,
        aspects,
        impacts,
        energies,
        kernels,
        moments,
        permutedims(A["paths"], (2, 1, 6, 3, 4, 5)),
        A["angles"],
        Float64.(header["water_mu_per_mm"]),
        Float64.(header["scatter_angle_edges_deg"]),
        Float64(header["forward_rayleigh_deg"]),
        Float64.(header["path_energy_edges_keV"]),
        Int.(header["element_Z"]),
        scattering,
        node_offsets,
    )
end

Base.show(io::IO, k::ScatterKernels) = print(
    io,
    "ScatterKernels(\"$(k.name)\": T $(first(k.thicknesses_mm))–$(last(k.thicknesses_mm)) mm, ",
    "aspect $(first(k.aspects))–$(last(k.aspects)), $(first(k.energies_keV))–$(last(k.energies_keV)) keV)",
)

# =============================================================================
# The artifacts: one header + data file per library entry
# =============================================================================

struct ScatterKernelArtifact
    file::String        # the header's file name inside the artifact
    url::String         # the release asset (a .zip of the header and its .bin)
    sha256::String      # SHA-256 of the .zip
    tree_hash::String   # git-tree-sha1 of the extracted directory (the artifact store's key)
end

# RELEASE: the assets are attached to the v0.16.0 release. Fill `sha256` and `tree_hash` from
# `lib/BasisMonteCarlo/scripts/split_scatter_library.jl`, which writes the per-scanner files and
# prints both; an entry whose `sha256` is empty is not released yet.
const _SCATTER_RELEASE_URL = "https://github.com/MolloiLab/BasisSimulator.jl/releases/download/v0.16.0"

_scatter_artifact(stem) = ScatterKernelArtifact(
    "$(stem).toml",
    "$(_SCATTER_RELEASE_URL)/$(stem).zip",
    "",     # RELEASE: placeholder
    "",     # RELEASE: placeholder
)

"""
    SCATTER_KERNEL_ARTIFACTS

The scatter library's entries and the artifact each ships as: `Dict(entry => (file, url,
sha256, tree_hash))`. A scanner's `scatter_kernels` field names one of these.
"""
const SCATTER_KERNEL_ARTIFACTS = Dict{String, ScatterKernelArtifact}(
    "SOMATOM Force" => _scatter_artifact("somatom_force"),
    "SOMATOM Definition Flash" => _scatter_artifact("somatom_definition_flash"),
    "NAEOTOM Alpha" => _scatter_artifact("naeotom_alpha"),
    "Revolution Apex" => _scatter_artifact("revolution_apex"),
    "Brilliance iCT" => _scatter_artifact("brilliance_ict"),
    "Aquilion ONE" => _scatter_artifact("aquilion_one"),
)

const _SCATTER_KERNELS = Dict{String, ScatterKernels}()
const _SCATTER_KERNELS_LOCK = ReentrantLock()

"""
    scatter_kernels(entry::AbstractString; path = nothing, quiet = false) -> ScatterKernels
    scatter_kernels(scanner::Scanner) -> ScatterKernels

The Geant4 scatter kernels of one library entry (`"SOMATOM Force"`, `"NAEOTOM Alpha"`, …; the keys
of `SCATTER_KERNEL_ARTIFACTS`), or of the entry a scanner names in its `scatter_kernels` field.
`simulate!` calls this when `SimOptions(use_scatter = true)`.

Loaded once per process and cached. The first call fetches the entry's artifact (about 250 MB)
into Julia's artifact store (`~/.julia/artifacts/`) unless it is already there, verifying it by
SHA-256. `path` reads a local copy instead, the header (`.toml`) of a library BasisMonteCarlo
wrote, with its `.bin` beside it; the entry is cached under `entry`, so every scanner naming it
uses that copy for the rest of the session.
"""
function scatter_kernels(entry::AbstractString; path::Union{Nothing, AbstractString} = nothing, quiet::Bool = false)
    return lock(_SCATTER_KERNELS_LOCK) do
        cached = get(_SCATTER_KERNELS, entry, nothing)
        if cached !== nothing && (path === nothing || abspath(path) == cached.path)
            return cached
        end
        header = if path === nothing
            joinpath(_download_scatter_kernels(entry; quiet), SCATTER_KERNEL_ARTIFACTS[entry].file)
        else
            abspath(path)
        end
        isfile(header) || throw(ArgumentError("no scatter library header at $(header)"))
        kernels = ScatterKernels(read_scatter_library(header; entry), entry, header)
        _SCATTER_KERNELS[String(entry)] = kernels
        return kernels
    end
end

function scatter_kernels(scanner::Scanner)
    isempty(scanner.scatter_kernels) && throw(
        ArgumentError(
            "this scanner names no scatter kernels (`scatter_kernels = \"\"`), so " *
                "`SimOptions(use_scatter = true)` has nothing to simulate. The library has kernels for " *
                "SOMATOM_FORCE and SOMATOM_FORCE_B (\"SOMATOM Force\"), SOMATOM_DEFINITION_FLASH and " *
                "SOMATOM_DEFINITION_FLASH_B (\"SOMATOM Definition Flash\"), REVOLUTION_APEX " *
                "(\"Revolution Apex\") and NAEOTOM_ALPHA and NAEOTOM_ALPHA_UHR (\"NAEOTOM Alpha\"), and " *
                "for \"Brilliance iCT\" and \"Aquilion ONE\". Start from one of those constants, name " *
                "an entry (`$(nameof(typeof(scanner)))(s; scatter_kernels = \"SOMATOM Force\")`), or " *
                "set `use_scatter = false`.",
        )
    )
    return scatter_kernels(scanner.scatter_kernels)
end

# The directory holding the entry's files, from Julia's artifact store, downloading it if needed.
function _download_scatter_kernels(entry::AbstractString; quiet::Bool = false)
    haskey(SCATTER_KERNEL_ARTIFACTS, entry) || throw(
        ArgumentError(
            "no scatter library entry \"$(entry)\"; the entries are $(sort!(collect(keys(SCATTER_KERNEL_ARTIFACTS)))). " *
                "A library of your own loads with `scatter_kernels(\"$(entry)\"; path = \"…/library.toml\")`."
        )
    )
    a = SCATTER_KERNEL_ARTIFACTS[entry]
    isempty(a.sha256) && throw(
        ArgumentError(
            "the scatter kernels of \"$(entry)\" are not released yet (BasisSimulator v0.16.0 attaches " *
                "them to its GitHub release). Load a local copy of the library BasisMonteCarlo wrote with " *
                "`scatter_kernels(\"$(entry)\"; path = \"…/$(a.file)\")`, or set `use_scatter = false`."
        )
    )
    hash = Base.SHA1(a.tree_hash)
    Artifacts.artifact_exists(hash) && return Artifacts.artifact_path(hash)

    quiet || @info "Downloading the scatter kernels of \"$(entry)\" into the Julia artifact store…"
    dest = Artifacts.artifact_path(hash)
    mkpath(dirname(dest))
    # staged on the store's own filesystem, so that the install is a rename
    staging = mktempdir(dirname(dest))
    try
        archive = joinpath(staging, basename(a.url))
        Downloads.download(a.url, archive)
        got = open(io -> bytes2hex(SHA.sha256(io)), archive)
        got == a.sha256 || error(
            "SHA-256 mismatch for the scatter kernels of \"$(entry)\"\n  expected $(a.sha256)\n  got      $(got)"
        )
        extracted = joinpath(staging, "extracted")
        mkpath(extracted)
        run(pipeline(`$(p7zip_jll.p7zip()) x -y -o$(extracted) $(archive)`; stdout = devnull))
        Artifacts.artifact_exists(hash) && return Artifacts.artifact_path(hash)   # a concurrent download won
        mv(extracted, dest)
    finally
        rm(staging; recursive = true, force = true)
    end
    return dest
end

# =============================================================================
# The kernels' geometry (BasisMonteCarlo's `shape_library.jl`)
# =============================================================================

# The widest a kernel's ellipse is made (half-width, mm), and the distance from a kernel's focal spot
# to its ellipse's axis: BasisMonteCarlo's KERNEL_MAX_HALF_WIDTH_MM and KERNEL_SOURCE_MM.
const _KERNEL_MAX_HALF_WIDTH_MM = 350.0
const _KERNEL_SOURCE_MM = 600.0

_kernel_half_width(thickness_mm, aspect) = min(aspect * thickness_mm / 2, _KERNEL_MAX_HALF_WIDTH_MM)

# The distance (mm) from the axis of the ellipse of semi-axes (b across, a along the view) at which
# the kernel ray of `impact_fraction` passes it: that fraction of the half-width the tilted ray sees.
function _kernel_offset(a, b, impact_fraction)
    offset = impact_fraction * b
    for _ in 1:30
        s = clamp(offset / _KERNEL_SOURCE_MM, -1.0, 1.0)
        next = impact_fraction * sqrt(b^2 * (1 - s^2) + a^2 * s^2)
        # BasisMonteCarlo runs 30 steps; the fixed point is reached to rounding long before
        next == offset && return next
        offset = next
    end
    return offset
end

function _kernel_direction(a, b, impact_fraction)
    s = _kernel_offset(a, b, impact_fraction) / _KERNEL_SOURCE_MM
    return (s, sqrt(1 - s^2))
end

# Length (mm) of the ray from `origin` along `direction` (x–y plane) inside an ellipse.
function _ellipse_chord(origin, direction, centre, semi_axes)
    d = direction ./ hypot(direction[1], direction[2])
    ox, oy = (origin[1] - centre[1]) / semi_axes[1], (origin[2] - centre[2]) / semi_axes[2]
    ux, uy = d[1] / semi_axes[1], d[2] / semi_axes[2]
    a = ux^2 + uy^2
    b = 2 * (ox * ux + oy * uy)
    c = ox^2 + oy^2 - 1
    disc = b^2 - 4a * c
    disc <= 0 && return 0.0
    t1, t2 = (-b - sqrt(disc)) / (2a), (-b + sqrt(disc)) / (2a)
    return max(0.0, t2) - max(0.0, t1)
end

_kernel_chord(a, b, impact_fraction) = _ellipse_chord(
    (0.0, -_KERNEL_SOURCE_MM),
    _kernel_direction(a, b, impact_fraction),
    (0.0, 0.0),
    (b, a),
)

# The impact fraction in (0, 1) whose kernel ray crosses the ellipse (b, a) for `length_mm`
# (bisection, to 1e-9; 0 when the ray through the axis is no longer than that).
function _kernel_impact(a, b, length_mm)
    _kernel_chord(a, b, 0.0) <= length_mm && return 0.0
    lo, hi = 0.0, 1.0
    for _ in 1:30
        mid = (lo + hi) / 2
        _kernel_chord(a, b, mid) > length_mm ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# Linear interpolation weights of `x` on the sorted grid `xs`, clamped: (i, i + 1, weight of i + 1).
function _bracket(xs, x)
    x <= first(xs) && return (1, 1, 0.0)
    x >= last(xs) && return (length(xs), length(xs), 0.0)
    i = searchsortedlast(xs, x)
    return (i, i + 1, (x - xs[i]) / (xs[i + 1] - xs[i]))
end

# The water ellipse of one view: its chords across the view have the patient's water (the zeroth
# moment of the rays' lengths across the fan), centre (the first) and spread (the second). A ray's
# place across the view is where it passes the isocentre, `xs = sid · sin(angle)`. Returns the
# semi-axes along the view (a) and across it (b) and the centre, or `nothing` for an empty view.
function _view_ellipse(xs, lengths)
    n = length(xs)
    n < 2 && return nothing
    m0 = 0.0
    m1 = 0.0
    for i in 1:n
        dx = i == 1 ? xs[2] - xs[1] : i == n ? xs[n] - xs[n - 1] : (xs[i + 1] - xs[i - 1]) / 2
        m0 += lengths[i] * dx
        m1 += xs[i] * lengths[i] * dx
    end
    m0 > 0 || return nothing
    x0 = m1 / m0
    spread = 0.0
    for i in 1:n
        dx = i == 1 ? xs[2] - xs[1] : i == n ? xs[n] - xs[n - 1] : (xs[i + 1] - xs[i - 1]) / 2
        spread += (xs[i] - x0)^2 * lengths[i] * dx
    end
    b = 2 * sqrt(spread / m0)
    b > 0 || return nothing
    return (a = m0 / (π * b), b = b, centre = x0)
end

# =============================================================================
# Materials against water (BasisMonteCarlo's `material_scattering_factors` / `angular_factors`)
# =============================================================================

# How much more (or less) a material of `fractions` (Z => mass fraction) scatters into the angles
# the scanner detects than its cross-section says, against water's, for one process (1 Compton,
# 2 Rayleigh), at `keV`, over the angle bins `bins`: Σ f(bin) P_m(bin) / P_water(bin) / Σ f(bin),
# `f` the scanner's profile of detected scatter and `P` each bin's share of the process's angular
# distribution. 1 for water.
function _angular_factor(k::ScatterKernels, fractions, water, process::Int, keV::Real, bins)
    E = clamp(round(Int, keV), 1, size(k.element_scattering, 4))
    nbins = size(k.element_scattering, 3)
    i1, i2, t = _bracket(k.energies_keV, Float64(E))
    f = (1 - t) .* Float64.(k.angles[process, :, i1]) .+ t .* Float64.(k.angles[process, :, i2])
    f = sum(f) > 0 ? f ./ sum(f) : fill(1 / nbins, nbins)
    split(fr) = sum(
        w .* Float64.(k.element_scattering[findfirst(==(Z), k.element_Z), process, :, E])
            for (Z, w) in fr
    )
    m, w = split(fractions), split(water)
    sm, sw = m ./ sum(m), w ./ sum(w)
    weight = sum(f[b] for b in bins)
    weight > 0 || return 1.0
    return sum(f[b] * (sw[b] > 0 ? sm[b] / sw[b] : 1.0) for b in bins) / weight
end

# The bins of Rayleigh's forward (< forward_rayleigh_deg) and wide scatter.
function _rayleigh_bins(k::ScatterKernels)
    upper = k.scatter_angle_edges_deg[2:end]
    forward = findall(<=(k.forward_rayleigh_deg), upper)
    return (forward = forward, wide = setdiff(eachindex(upper), forward))
end

# =============================================================================
# The superposition (BasisMonteCarlo's `library_scan`, one view)
# =============================================================================

# Scratch of one task: the per-ray weights by library energy, process and channel, the weights of
# one corner of the grid, that corner's mixed kernel and column moments, how much of each process
# gets out of the ray's materials, and the scatter of the rays relative to their own row.
struct _SuperposeScratch
    W0::Array{Float64, 3}
    Wc::Matrix{Float32}
    part::Matrix{Float32}
    mom::Matrix{Float32}
    escape::Matrix{Float64}
    acc::Array{Float64, 3}
end

function _SuperposeScratch(k::ScatterKernels, n_channels::Int)
    nc, nr = length(k.column_edges_deg) - 1, length(k.row_edges_mm) - 1
    nE = length(k.energies_keV)
    return _SuperposeScratch(
        zeros(nE, 3, n_channels),
        zeros(Float32, 4nE, n_channels),
        zeros(Float32, nr * nc, n_channels),
        zeros(Float32, 3nc, n_channels),
        ones(nE, 2),
        zeros(nr, nc, n_channels),
    )
end

# The rays of one view (one z group of it), across the fan in order of angle: `angle_deg`, the
# water-equivalent length `wet_mm`, the length `ellipse_mm` the view's ellipse is measured from
# (the water-equivalent length or the outline's), the photons of each spectral energy `spectrum`
# (energy × ray: the source spectrum through the bowtie, times the ray's share of a library bin),
# the materials' ratios to water by process at each spectral energy (`compton`, `rayleigh_wide`,
# `rayleigh_forward`, energy × ray; `nothing` for water) and `delta_mu_per_mm` (band × ray, or
# `nothing`).
#
# Adds the view's scatter to `out[column, row, channel]` (the library's bins): each channel's
# weight of a photon of each spectral energy is `channels[energy, channel]`, each library row's
# share of the rays' beam `row_share`. `shares[energy]` places each spectral energy between the
# library's (`_bracket`). Returns the number of rays clamped onto the library's grid.
function _superpose!(
        out::AbstractArray{Float64, 3},
        k::ScatterKernels,
        rays,
        row_share::AbstractVector{Float64},
        channels::AbstractMatrix{Float64},
        shares::AbstractVector,
        scratch::_SuperposeScratch,
    )
    cols = k.column_edges_deg
    nc, nr = length(cols) - 1, length(k.row_edges_mm) - 1
    nE = length(k.energies_keV)
    nu, na = length(k.impact_fractions), length(k.aspects)
    n_ch = size(channels, 2)
    sid = k.source_to_isocenter_mm
    w_col = cols[2] - cols[1]
    mid = (nr + 1) ÷ 2
    L = nc * nr
    block = L * nE * 4
    mblock = 3nc * nE * 4
    kernels, moments = vec(k.kernels), vec(k.moments)
    log_aspects = log.(k.aspects)
    W0, Wc, part, mom, escape, acc = scratch.W0, scratch.Wc, scratch.part, scratch.mom, scratch.escape, scratch.acc
    nbands = size(k.paths, 1)
    T_min, T_max = first(k.thicknesses_mm), last(k.thicknesses_mm)
    u_last = last(k.impact_fractions)
    # water: every process weighed alike, so the kernel of all the scatter (process 4) serves
    plain = rays.compton === nothing && rays.rayleigh_wide === nothing &&
        rays.rayleigh_forward === nothing && rays.delta_mu_per_mm === nothing
    slots = plain ? ((3nE + 1):(4nE)) : (1:(3nE))

    xs = [sid * sind(γ) for γ in rays.angle_deg]
    ellipse = _view_ellipse(xs, rays.ellipse_mm)
    ellipse === nothing && return 0
    a, b, x0 = ellipse.a, ellipse.b, ellipse.centre
    clamped = 0
    fill!(acc, 0.0)

    for i in eachindex(rays.angle_deg)
        wet = rays.wet_mm[i]
        wet > 0 || continue
        γ = rays.angle_deg[i]

        # The ray's offset from the ellipse's axis, perpendicular to it; the offset whose kernel ray
        # crosses the ray's own water-equivalent length; a ray longer than the ellipse is thick
        # gets an ellipse as thick as itself
        offset = sid * sind(γ) - x0 * cosd(γ)
        side = offset >= 0 ? 1.0 : -1.0
        a_ray = max(a, wet / 2)
        u = _kernel_impact(a_ray, b, wet) * side
        aspect = b / a_ray
        T_ray = 2a_ray
        if !(T_min <= T_ray <= T_max) || !(first(k.aspects) <= aspect <= last(k.aspects))
            clamped += 1
        end

        # Below the grid's thinnest ellipse the thinnest kernel stands in, scaled by the water the
        # ray crosses against what that kernel's own ray crosses
        amount = 1.0
        if T_ray < T_min
            r_node = clamp(aspect, first(k.aspects), last(k.aspects))
            chord = _kernel_chord(T_min / 2, _kernel_half_width(T_min, r_node), abs(u))
            amount = min(1.0, wet / max(chord, 1.0))
        end

        # The spectrum and the ray's ratios at each of its energies, gathered onto the library's
        # energies, by process and channel
        fill!(W0, 0.0)
        for e in axes(rays.spectrum, 1)
            s = rays.spectrum[e, i]
            s > 0 || continue
            g1, g2, t = shares[e]
            fc = rays.compton === nothing ? 1.0 : rays.compton[e, i]
            fw = rays.rayleigh_wide === nothing ? 1.0 : rays.rayleigh_wide[e, i]
            ff = rays.rayleigh_forward === nothing ? 1.0 : rays.rayleigh_forward[e, i]
            for ch in 1:n_ch
                base = s * channels[e, ch]
                base == 0 && continue
                for (g, wt) in ((g1, 1 - t), (g2, t))
                    wt == 0 && continue
                    W0[g, 1, ch] += base * wt * fc
                    W0[g, 2, ch] += base * wt * fw
                    W0[g, 3, ch] += base * wt * ff
                end
            end
        end

        # The grid's kernels around the ray, each moved so that its own ray lies under this one
        # before they are blended
        iT = _bracket(k.thicknesses_mm, T_ray)
        ir = _bracket(log_aspects, log(aspect))
        iu = _bracket(k.impact_fractions, abs(u))
        graze = abs(u) > u_last ? sqrt(max(0.0, 1 - min(abs(u), 1.0)^2)) / sqrt(1 - u_last^2) : 1.0
        for (tT, wT) in ((iT[1], 1 - iT[3]), (iT[2], iT[3])),
                (tr, wr) in ((ir[1], 1 - ir[3]), (ir[2], ir[3])),
                (tu, wu) in ((iu[1], 1 - iu[3]), (iu[2], iu[3]))

            wg = wT * wr * wu
            wg == 0 && continue
            corner = ((tT - 1) * na + (tr - 1)) * nu + (tu - 1)

            # this corner's weights: how much less of each process's scatter gets out of the ray's
            # materials than of the kernel's water
            if plain
                for ch in 1:n_ch, g in 1:nE
                    Wc[3nE + g, ch] = Float32(W0[g, 1, ch])
                end
            else
                if rays.delta_mu_per_mm === nothing
                    fill!(escape, 1.0)
                else
                    for pp in 1:2, g in 1:nE
                        acc_path = 0.0
                        for band in 1:nbands
                            acc_path += k.paths[band, pp, g, tu, tr, tT] * rays.delta_mu_per_mm[band, i]
                        end
                        escape[g, pp] = exp(-acc_path)
                    end
                end
                for ch in 1:n_ch, p in 1:3, g in 1:nE
                    Wc[g + (p - 1) * nE, ch] = Float32(W0[g, p, ch] * escape[g, min(p, 2)])
                end
            end

            # the mixed kernel and its column moments
            fill!(part, 0.0f0)
            fill!(mom, 0.0f0)
            for j in slots
                base = corner * block + (j - 1) * L
                mbase = corner * mblock + (j - 1) * 3nc
                for ch in 1:n_ch
                    w = Wc[j, ch]
                    w == 0 && continue
                    @inbounds @simd for q in 1:L
                        part[q, ch] += w * kernels[base + q]
                    end
                    @inbounds @simd for q in 1:(3nc)
                        mom[q, ch] += w * moments[mbase + q]
                    end
                end
            end

            # this kernel's own ray, seen from this scanner
            node_angle = asind(clamp(k.node_offsets_mm[tu, tr, tT] / sid, -1.0, 1.0)) * side
            factor = wg * graze * amount

            # Each column moved by the ray's angle from the kernel's own, as what it holds: the
            # scatter centred where it landed in the column and spread as it landed, a box of that
            # width laid over the library's columns (the mirror image for a ray left of the axis)
            for ch in 1:n_ch, ii in 1:nc
                src = u < 0 ? nc + 1 - ii : ii
                m0 = Float64(mom[src, ch])
                m0 > 0 || continue
                m1 = (u < 0 ? -1.0 : 1.0) * Float64(mom[nc + src, ch])
                m2 = Float64(mom[2nc + src, ch])
                centre = m1 / m0
                spread = sqrt(max(m2 / m0 - centre^2, 0.0))
                x = cols[ii] + w_col / 2 + centre + (γ - node_angle)
                half = min(sqrt(3) * spread, w_col / 2)
                lo, hi = x - half, x + half
                for ci in max(1, searchsortedlast(cols, lo)):min(nc, searchsortedlast(cols, hi))
                    share = hi > lo ? (min(hi, cols[ci + 1]) - max(lo, cols[ci])) / (hi - lo) : 1.0
                    share > 0 || continue
                    f = factor * share
                    q0 = (src - 1) * nr
                    @inbounds @simd for dr in 1:nr
                        acc[dr, ci, ch] += f * part[q0 + dr, ch]
                    end
                end
            end
        end
    end

    # the rays' scatter, relative to their own row, laid over each row of their beam
    for r0 in eachindex(row_share)
        share = row_share[r0]
        share > 0 || continue
        for ch in 1:n_ch, ci in 1:nc, ri in max(1, r0 - mid + 1):min(nr, r0 - mid + nr)
            @inbounds out[ci, ri, ch] += share * acc[ri - r0 + mid, ci, ch]
        end
    end
    return clamped
end
