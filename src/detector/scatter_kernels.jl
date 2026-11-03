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

The superposition (the kernels at the end of this file, run by `_superpose!` in `scatter.jl`) is
BasisMonteCarlo's `library_scan`, ported to run on every coarse ray of every view at once on the
workspace's backend: each view stands in the water ellipse of its rays' water-equivalent lengths;
each ray takes the kernel of the impact whose chord is its own water-equivalent length, its
Compton, wide- and forward-Rayleigh parts scaled by its materials' scattering against water's and
by how much less of each gets out of them, and is moved across the fan to the ray's own angle,
column by column as boxes of the width the scatter landed with.
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

# RELEASE: the assets are a data release of their own, tag `scatter-kernels-v2` (made before the
# package release, so that the hashes below can be registered with it; RELEASING.md). Fill `sha256`
# and `tree_hash` from `lib/BasisMonteCarlo/scripts/split_scatter_library.jl --archive`, which
# writes the per-scanner files and prints both; an entry whose `sha256` is empty is not released.
const _SCATTER_RELEASE_URL = "https://github.com/MolloiLab/BasisSimulator.jl/releases/download/scatter-kernels-v2"

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
            "the scatter kernels of \"$(entry)\" are not released yet (the GitHub release " *
                "`scatter-kernels-v2` will carry them). Load a local copy of the library BasisMonteCarlo wrote with " *
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
#
# Generic in the float type: the superposition runs these on the device, in the workspace's type.

# The widest a kernel's ellipse is made (half-width, mm), and the distance from a kernel's focal spot
# to its ellipse's axis: BasisMonteCarlo's KERNEL_MAX_HALF_WIDTH_MM and KERNEL_SOURCE_MM.
const _KERNEL_MAX_HALF_WIDTH_MM = 350.0
const _KERNEL_SOURCE_MM = 600.0

_kernel_half_width(thickness_mm, aspect) = min(
    aspect * thickness_mm / 2,
    oftype(thickness_mm, _KERNEL_MAX_HALF_WIDTH_MM),
)

# The half-width of the ellipse of semi-axes (b across, a along the view) across the kernel ray of
# `impact_fraction`, which passes the axis at that fraction of it: the ray from the focal spot is
# tilted by the offset it passes at, so the two are found together (a fixed point).
function _kernel_seen_half_width(a, b, impact_fraction)
    source = oftype(a, _KERNEL_SOURCE_MM)
    h = b
    for _ in 1:30
        s = clamp(impact_fraction * h / source, -one(a), one(a))
        next = sqrt(b^2 * (1 - s^2) + a^2 * s^2)
        # BasisMonteCarlo runs 30 steps; the fixed point is reached to rounding long before
        next == h && return next
        h = next
    end
    return h
end

# The distance (mm) from the axis at which the kernel ray of `impact_fraction` passes it.
_kernel_offset(a, b, impact_fraction) = impact_fraction * _kernel_seen_half_width(a, b, impact_fraction)

# The length (mm) of the kernel ray of impact `u` inside its ellipse: a line passing the centre of
# an ellipse at `u` times its half-width `h` across the line crosses it for 2ab·√(1 − u²)/h.
# `q = √(1 − u²)` is passed by `_kernel_impact`, exact near the edge where 1 − u² is not.
_kernel_chord(a, b, u, q = sqrt(max(zero(u), 1 - u^2))) = 2a * b * q / _kernel_seen_half_width(a, b, u)

# `(u, √(1 − u²))` for u = k·2⁻ⁿ, from 1 − u = (2ⁿ − k)·2⁻ⁿ, exact in any float type near the edge.
@inline function _dyadic_impact(::Type{T}, k::Int, n::Int) where {T}
    scale = T(2.0^-n)
    w = T((1 << n) - k) * scale
    return (T(k) * scale, sqrt(w * (2 - w)))
end

# The impact fraction `u` in [0, 1) whose kernel ray crosses the ellipse (b, a) for `length_mm`,
# and `√(1 − u²)`: bisection to 2⁻³⁰ on dyadic steps (BasisMonteCarlo's `kernel_impact`); 0 when
# the ray through the axis (2a long) is no longer than that.
function _kernel_impact(a, b, length_mm)
    T = typeof(a)
    2a <= length_mm && return (zero(T), one(T))
    lo, hi = 0, 1 << 30
    for _ in 1:30
        mid = (lo + hi) >> 1
        u, q = _dyadic_impact(T, mid, 30)
        _kernel_chord(a, b, u, q) > length_mm ? (lo = mid) : (hi = mid)
    end
    return _dyadic_impact(T, lo + hi, 31)
end

# Linear interpolation weights of `x` on the sorted grid `xs`, clamped: (i, i + 1, weight of i + 1).
function _bracket(xs, x)
    n = length(xs)
    w = zero(promote_type(eltype(xs), typeof(x)))
    x <= xs[1] && return (1, 1, w)
    x >= xs[n] && return (n, n, w)
    # the last i with xs[i] <= x
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        xs[mid] <= x ? (lo = mid) : (hi = mid)
    end
    return (lo, lo + 1, (x - xs[lo]) / (xs[lo + 1] - xs[lo]))
end

# The width across the view each ray stands for, from where the rays pass the isocentre.
_ray_widths(xs) = [
    i == 1 ? xs[2] - xs[1] : i == length(xs) ? xs[end] - xs[end - 1] : (xs[i + 1] - xs[i - 1]) / 2
        for i in eachindex(xs)
]

# `(sum, compensation) + x`, Kahan's: the device sums in single precision.
@inline function _kahan(sum, c, x)
    y = x - c
    t = sum + y
    return (t, (t - sum) - y)
end

# The water ellipse of one view: its chords across the view have the patient's water (the zeroth
# moment of the rays' lengths across the fan), centre (the first) and spread (the second). A ray's
# place across the view is where it passes the isocentre, `xs = sid · sin(angle)`, and the width
# it stands for `dxs` (`_ray_widths(xs)`). Returns the semi-axes along the view (a) and across it
# (b) and the centre; `b = 0` for an empty view.
function _view_ellipse(xs, dxs, lengths)
    n = length(xs)
    R = promote_type(eltype(xs), eltype(lengths))
    empty = (a = zero(R), b = zero(R), centre = zero(R))
    n < 2 && return empty
    m0, c0 = zero(R), zero(R)
    m1, c1 = zero(R), zero(R)
    for i in 1:n
        m0, c0 = _kahan(m0, c0, lengths[i] * dxs[i])
        m1, c1 = _kahan(m1, c1, xs[i] * lengths[i] * dxs[i])
    end
    m0 > 0 || return empty
    x0 = m1 / m0
    spread, c2 = zero(R), zero(R)
    for i in 1:n
        spread, c2 = _kahan(spread, c2, (xs[i] - x0)^2 * lengths[i] * dxs[i])
    end
    b = 2 * sqrt(spread / m0)
    b > 0 || return empty
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
# The superposition (BasisMonteCarlo's `library_scan`), every coarse ray at once
# =============================================================================
#
# Each step is one kernel over all the coarse views and groups, one thread per output, on the
# backend of its arrays (the device, or Julia's threads on the CPU), in the workspace's float type.
# A ray is `[column, group, view]` of the coarse rays, linearly; a view-group is `[group, view]`.
#
# 1. `_scatter_ellipses!`: each view-group's water ellipse, from its rays' lengths.
# 2. `_scatter_ray_shapes!`: each ray's own ellipse (thickness, aspect), its impact, and how much of
#    the nearest kernels it takes.
# 3. `_scatter_corners!`: for each of the 8 kernels of the grid around a ray (its corners) and each
#    channel: the weights of the kernel's energies and processes (the ray's photons and how much
#    less of each process's scatter gets out of its materials than of the kernel's water), and,
#    for each of the kernel's columns, where it lands once moved to the ray's angle: a box of the
#    width the scatter landed with (degrees from a library column's left edge, so that single
#    precision keeps the slivers of it a column takes).
# 4. `_scatter_deposit!`: each library bin (row relative to the rays' own, column, channel) of each
#    view-group gathers the boxes over it, each the kernel's column mixed by the corner's weights.
# 5. `_scatter_rows!`: each library bin of each view, the groups' scatter laid over the rows of
#    their beams.

# `a + b` and its rounding error, exactly (Knuth's two-sum).
@inline function _two_sum(a, b)
    s = a + b
    bb = s - a
    return (s, (a - (s - bb)) + (b - bb))
end

# `ellipse[(a, b, centre), group, view]` from `lengths[column, group, view]`, the rays passing the
# isocentre at `xs` (mm), each standing for a width `dxs`.
function _scatter_ellipses!(
        ellipse::AbstractArray{T, 3},
        xs::AbstractVector,
        dxs::AbstractVector,
        lengths::AbstractArray{T, 3},
    ) where {T}
    n_cc = size(lengths, 1)
    let ellipse = ellipse, xs = xs, dxs = dxs, lengths = lengths, n_cc = n_cc
        _foreachindex(1:(size(lengths, 2) * size(lengths, 3)), AK.get_backend(ellipse)) do gv
            e = _view_ellipse(xs, dxs, view(lengths, ((gv - 1) * n_cc + 1):(gv * n_cc)))
            ellipse[3gv - 2] = e.a
            ellipse[3gv - 1] = e.b
            ellipse[3gv] = e.centre
        end
    end
    return ellipse
end

# `shapes[(thickness, aspect, impact, side, weight), ray]`: each ray's own ellipse (as thick as
# the view's, or as the ray's water-equivalent length `wet` if longer), its signed impact, the side
# of the axis it passes on, and the weight of the kernels standing in for it (0: no scatter).
# With `mark`, `clamped[ray]` is set to 1 for a ray outside the library's grids (and otherwise left
# as it is). `grid` = (thinnest, thickest, narrowest, widest, last impact).
function _scatter_ray_shapes!(
        shapes::AbstractMatrix{T},
        clamped::AbstractArray{Int32},
        wet::AbstractArray{T, 3},
        ellipse::AbstractArray{T, 3},
        xs::AbstractVector{T},
        angle_deg::AbstractVector{T},
        grid::NTuple{5, T},
        mark::Bool,
    ) where {T}
    n_cc = size(wet, 1)
    let shapes = shapes, clamped = clamped, wet = wet, ellipse = ellipse, xs = xs, angle_deg = angle_deg,
            grid = grid, n_cc = n_cc, mark = mark
        AK.foreachindex(wet) do ray
            T_min, T_max, r_min, r_max, u_last = grid
            o = one(T)
            q = ray - 1
            i = q % n_cc + 1
            gv = q ÷ n_cc
            L = wet[ray]
            a, b, x0 = ellipse[3gv + 1], ellipse[3gv + 2], ellipse[3gv + 3]
            thickness, aspect, u, side, weight = zero(T), zero(T), zero(T), o, zero(T)
            outside = Int32(0)
            if L > 0 && b > 0
                # The ray's offset from the ellipse's axis, perpendicular to it; the impact whose
                # kernel ray crosses the ray's own water-equivalent length; a ray longer than the
                # ellipse is thick gets an ellipse as thick as itself
                offset = xs[i] - x0 * cos(deg2rad(angle_deg[i]))
                side = offset >= 0 ? o : -o
                a_ray = max(a, L / 2)
                u, q = _kernel_impact(a_ray, b, L)
                u *= side
                aspect = b / a_ray
                thickness = 2a_ray
                inside = T_min <= thickness <= T_max && r_min <= aspect <= r_max
                outside = inside ? Int32(0) : Int32(1)

                # Below the grid's thinnest ellipse the thinnest kernel stands in, scaled by the
                # water the ray crosses against what that kernel's own ray crosses
                amount = o
                if thickness < T_min
                    r_node = clamp(aspect, r_min, r_max)
                    chord = _kernel_chord(T_min / 2, _kernel_half_width(T_min, r_node), abs(u), q)
                    amount = min(o, L / max(chord, o))
                end
                graze = abs(u) > u_last ? q / sqrt(o - u_last^2) : o
                weight = graze * amount
            end
            shapes[1, ray] = thickness
            shapes[2, ray] = aspect
            shapes[3, ray] = u
            shapes[4, ray] = side
            shapes[5, ray] = weight
            mark && outside == 1 && (clamped[ray] = outside)
        end
    end
    return shapes
end

# For each channel `ch <= n_used`, corner `c` (0–7: bits thickness, aspect, impact) and ray:
# `corner[c, ray]`, the kernel's place in the grid (0-based); `factor[ch, c, ray]`, its weight
# (0: none); `mix[slot, ch, c, ray]`, the weights of its library energies and processes (`3nE`
# slots of Compton, wide and forward Rayleigh, or for a ray of water `nE` slots of all the
# scatter); `boxes[(edge, lo, hi), column, ch, c, ray]`, where each of its columns (in the order
# of the ray's side) lands, from the left edge of library column `edge` (degrees; `lo = hi = Inf`
# for none); `window[(min, max), ch, c, ray]`, the offsets of the library columns those boxes
# reach from their own (one wider each side, for rounding).
#
# The library's columns are uniform, `lib.column_width` wide. `angle[(hi, lo), column]` and
# `node[(hi, lo), kernel]` are the coarse rays' angles and the kernels' own rays' (degrees, from the
# axis on their side), each the pair of floats that sums to its value: their difference is kept to
# the rounding of a column's width, not of the fan's.
#
# `weights[energy, process, ch, ray]` are the ray's photons by library energy, process and
# channel; `dmu[band, ray]` its materials' attenuation against water's (not read for a ray of
# water, `plain[ray]`).
#
# For the `n_rays` rays from `ray0 + 1` on (a chunk of the views): `shapes`, `plain`, `dmu` are
# every ray's, the others the chunk's.
function _scatter_corners!(
        corner::AbstractMatrix{Int32},
        factor::AbstractArray{T, 3},
        mix::AbstractArray{T, 4},
        boxes::AbstractArray{T, 5},
        window::AbstractArray{Int32, 4},
        shapes::AbstractMatrix{T},
        weights::AbstractArray{T, 4},
        plain::AbstractVector{Bool},
        dmu::AbstractMatrix{T},
        angle::AbstractMatrix{T},
        lib,
        n_used::Int,
        ray0::Int,
        n_rays::Int,
    ) where {T}
    nE, n_ch = size(weights, 1), size(weights, 3)
    nc = length(lib.cols) - 1
    n_cc = size(angle, 2)
    nu, na = length(lib.impacts), length(lib.log_aspects)
    nb = size(dmu, 1)
    let corner = corner, factor = factor, mix = mix, boxes = boxes, window = window, shapes = shapes,
            weights = weights, plain = plain, dmu = dmu, angle = angle,
            moments = lib.moments, paths = lib.paths, node = lib.node_deg, thick = lib.thicknesses,
            log_r = lib.log_aspects, impacts = lib.impacts, w_col = lib.column_width,
            nE = nE, n_ch = n_ch, nc = nc, n_cc = n_cc, nu = nu, na = na, nb = nb, n_used = n_used, ray0 = ray0
        _foreachindex(1:(n_ch * 8 * n_rays), AK.get_backend(factor)) do idx
            o = one(T)
            q = idx - 1
            ch = q % n_ch + 1
            c = (q ÷ n_ch) % 8
            local_ray = q ÷ (8n_ch) + 1
            ray = local_ray + ray0
            weight = shapes[5, ray]
            factor[idx] = zero(T)
            window[1, ch, c + 1, local_ray] = Int32(1)
            window[2, ch, c + 1, local_ray] = Int32(0)
            @inbounds if ch <= n_used && weight > 0
                u = shapes[3, ray]
                iT = _bracket(thick, shapes[1, ray])
                ir = _bracket(log_r, log(shapes[2, ray]))
                iu = _bracket(impacts, abs(u))
                bT, br, bu = (c >> 2) & 1, (c >> 1) & 1, c & 1
                tT, wT = bT == 0 ? (iT[1], o - iT[3]) : (iT[2], iT[3])
                tr, wr = br == 0 ? (ir[1], o - ir[3]) : (ir[2], ir[3])
                tu, wu = bu == 0 ? (iu[1], o - iu[3]) : (iu[2], iu[3])
                wg = wT * wr * wu
                if wg != 0
                    k = ((tT - 1) * na + (tr - 1)) * nu + (tu - 1)
                    corner[c + 1, local_ray] = Int32(k)

                    # this corner's weights: how much less of each process's scatter gets out of the
                    # ray's materials than of the kernel's water
                    water = plain[ray]
                    if water
                        for g in 1:nE
                            mix[g, ch, c + 1, local_ray] = weights[g, 1, ch, local_ray]
                        end
                    else
                        for pp in 1:2, g in 1:nE
                            path = zero(T)
                            base = nb * (pp - 1 + 2 * (g - 1 + nE * k))
                            for band in 1:nb
                                path += paths[base + band] * dmu[band, ray]
                            end
                            escape = exp(-path)
                            if pp == 1
                                mix[g, ch, c + 1, local_ray] = weights[g, 1, ch, local_ray] * escape
                            else
                                mix[g + nE, ch, c + 1, local_ray] = weights[g, 2, ch, local_ray] * escape
                                mix[g + 2nE, ch, c + 1, local_ray] = weights[g, 3, ch, local_ray] * escape
                            end
                        end
                    end
                    slot0, n_slots = water ? (3nE, nE) : (0, 3nE)

                    # The ray's angle from this kernel's own ray, seen from this scanner, in whole
                    # columns and the rest
                    side = shapes[4, ray]
                    i = (ray - 1) % n_cc + 1
                    s_hi, s_err = _two_sum(angle[1, i], -side * node[1, k + 1])
                    s_lo = s_err + (angle[2, i] - side * node[2, k + 1])
                    whole = round(s_hi / w_col)
                    rest = s_hi - whole * w_col
                    mirror = u < 0
                    mbase = 3nc * 4nE * k
                    d_lo, d_hi = typemax(Int32), typemin(Int32)

                    # Each column moved by the ray's angle from the kernel's own, as what it holds:
                    # the scatter centred where it landed in the column and spread as it landed, a
                    # box of that width (the mirror image for a ray left of the axis)
                    for ii in 1:nc
                        src = mirror ? nc + 1 - ii : ii
                        m0, m1, m2 = zero(T), zero(T), zero(T)
                        for j in 1:n_slots
                            w = mix[j, ch, c + 1, local_ray]
                            w == 0 && continue
                            b = mbase + 3nc * (slot0 + j - 1) + src
                            m0 += w * moments[b]
                            m1 += w * moments[b + nc]
                            m2 += w * moments[b + 2nc]
                        end
                        edge, lo, hi = zero(T), T(Inf), T(Inf)
                        if m0 > 0
                            centre = (mirror ? -m1 : m1) / m0
                            spread = sqrt(max(m2 / m0 - centre^2, zero(T)))
                            # the box's centre from the left edge of column `ii + whole`
                            x = ((w_col / 2 + centre) + rest) + s_lo
                            half = min(sqrt(T(3)) * spread, w_col / 2)
                            lo, hi = x - half, x + half
                            edge = T(ii) + whole
                            first_col = max(1, unsafe_trunc(Int, edge + floor(lo / w_col)))
                            last_col = min(nc, unsafe_trunc(Int, edge + floor(hi / w_col)))
                            if first_col <= last_col
                                d_lo = min(d_lo, Int32(first_col - ii - 1))
                                d_hi = max(d_hi, Int32(last_col - ii + 1))
                            end
                        end
                        boxes[1, ii, ch, c + 1, local_ray] = edge
                        boxes[2, ii, ch, c + 1, local_ray] = lo
                        boxes[3, ii, ch, c + 1, local_ray] = hi
                    end
                    if d_lo <= d_hi
                        window[1, ch, c + 1, local_ray] = d_lo
                        window[2, ch, c + 1, local_ray] = d_hi
                        factor[idx] = wg * weight
                    end
                end
            end
        end
    end
    return factor
end

# `rows[row, column, ch, group, view]`: the scatter of a view-group's rays relative to their own
# row, each library bin gathering the corners' boxes over its column, for the `n_gv` view-groups
# of the chunk whose rays start after `ray0` (`shapes`, `plain`: every ray's; the rest the chunk's).
function _scatter_deposit!(
        rows::AbstractArray{T, 5},
        corner::AbstractMatrix{Int32},
        factor::AbstractArray{T, 3},
        mix::AbstractArray{T, 4},
        boxes::AbstractArray{T, 5},
        window::AbstractArray{Int32, 4},
        shapes::AbstractMatrix{T},
        plain::AbstractVector{Bool},
        lib,
        n_cc::Int,
        n_used::Int,
        ray0::Int,
        n_gv::Int,
    ) where {T}
    nr, nc, n_ch = size(rows, 1), size(rows, 2), size(rows, 3)
    nE = size(mix, 1) ÷ 3
    L = nr * nc
    block = L * 4nE
    lanes = 32 * cld(nr, 32)
    let rows = rows, corner = corner, factor = factor, mix = mix, boxes = boxes, window = window,
            shapes = shapes, plain = plain, kernels = lib.kernels, w_col = lib.column_width,
            nr = Int32(nr), nc = Int32(nc), n_ch = Int32(n_ch), nE = Int32(nE), L = Int32(L), block = Int32(block),
            n_cc = Int32(n_cc), n_used = Int32(n_used), lanes = Int32(lanes), ray0 = Int32(ray0)
        # the rows in whole warps: a warp is one bin's column, all its threads taking the same branches
        _foreachindex(1:(lanes * nc * n_ch * n_gv), AK.get_backend(rows)) do idx
            q = Int32(idx - 1)
            dr = q % lanes + Int32(1)
            dr > nr && return
            q ÷= lanes
            ci = q % nc + Int32(1)
            q ÷= nc
            ch = q % n_ch + Int32(1)
            gv = q ÷ n_ch
            acc, comp = zero(T), zero(T)
            @inbounds if ch <= n_used
                for i in Int32(1):n_cc
                    ray = gv * n_cc + i
                    mirror = shapes[3, ray + ray0] < 0
                    slot0, n_slots = plain[ray + ray0] ? (Int32(3) * nE, nE) : (Int32(0), Int32(3) * nE)
                    for c in Int32(1):Int32(8)
                        f = factor[ch, c, ray]
                        f == 0 && continue
                        base = corner[c, ray] * block + dr + slot0 * L
                        for ii in max(Int32(1), ci - window[2, ch, c, ray]):min(nc, ci - window[1, ch, c, ray])
                            # the box from the left edge of this bin's column, and its share of it
                            off = w_col * (boxes[1, ii, ch, c, ray] - T(ci))
                            lo, hi = boxes[2, ii, ch, c, ray], boxes[3, ii, ch, c, ray]
                            l, h = off + lo, off + hi
                            share = hi > lo ? (min(h, w_col) - max(l, zero(T))) / (hi - lo) :
                                (zero(T) <= l < w_col ? one(T) : zero(T))
                            share > 0 || continue
                            src = mirror ? nc + Int32(1) - ii : ii
                            kq = base + (src - Int32(1)) * nr
                            s = zero(T)
                            for j in Int32(1):n_slots
                                s += mix[j, ch, c, ray] * kernels[kq + (j - Int32(1)) * L]
                            end
                            acc, comp = _kahan(acc, comp, f * share * s)
                        end
                    end
                end
            end
            rows[dr + nr * (ci - Int32(1) + nc * (ch - Int32(1) + n_ch * gv))] = acc
        end
    end
    return rows
end

# `out[column, row, ch, view]` for the `n_v` views from `v0 + 1` on (the chunk's, `rows`): the
# scatter of each view, its groups' `rows` laid over each row of their beams (`row_share[row,
# group]`); with `combine`, the geometric mean of that and what `out` holds.
function _scatter_rows!(
        out::AbstractArray{T, 4},
        rows::AbstractArray{T, 5},
        row_share::AbstractMatrix{T},
        v0::Int,
        n_v::Int;
        combine::Bool,
    ) where {T}
    nc, nr, n_ch = size(out, 1), size(out, 2), size(out, 3)
    n_g = size(rows, 4)
    let out = out, rows = rows, row_share = row_share, nc = nc, nr = nr, n_ch = n_ch, n_g = n_g,
            mid = (nr + 1) ÷ 2, combine = combine, v0 = v0
        _foreachindex(1:(nc * nr * n_ch * n_v), AK.get_backend(out)) do q1
            q = q1 - 1
            ci = q % nc + 1
            q ÷= nc
            ri = q % nr + 1
            q ÷= nr
            ch = q % n_ch + 1
            v = q ÷ n_ch + 1
            idx = q1 + nc * nr * n_ch * v0
            s = zero(T)
            for g in 1:n_g, r0 in 1:nr
                share = row_share[r0, g]
                dr = ri - r0 + mid
                share > 0 && 1 <= dr <= nr || continue
                s += share * rows[dr, ci, ch, g, v]
            end
            out[idx] = combine ? sqrt(out[idx] * s) : s
        end
    end
    return out
end
