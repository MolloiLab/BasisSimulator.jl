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
column by column as boxes of the width the scatter landed with (the mirror image for a ray left of
the ellipse's axis).

A kernel's ray is surrounded by water as its ellipse places it; a patient's is not (a thorax is
dense in the middle and light at the sides, so the scatter a ray beside the mediastinum sends
inward is stopped there and what it sends out through a lung escapes). So each kernel keeps only
its multiple scatter: its own single scatter, a quadrature in its water ellipse
(BasisMonteCarlo's `single_scatter`), is taken out when the library is loaded. Each ray adds back
its own single scatter, the same quadrature in the patient as the view's rays describe it
(`_RayStrips`: each ray's water-equivalent length spread over its outline), scaled by the same
photons and escape factors, at its own columns.
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
shared `elements` tables are read. A copy of BasisMonteCarlo's reference reader.
"""
function read_scatter_library(
    header_path::AbstractString;
    entry::Union{Nothing, AbstractString} = nothing,
)
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
- `kernels[row, column, energy, process, impact, aspect, thickness]`: the multiple scatter (each
  process's scatter less the kernel's own single scatter, never below none) whose last scattering
  was a Compton (process 1), a Rayleigh wider than `forward_rayleigh_deg` (2) or a forward
  Rayleigh (3), and all of it (4).
- `moments[column, k, energy, process, impact, aspect, thickness]`: each column's multiple scatter
  (k = 1) and its first and second moments of where in the column it landed, in degrees from the
  column's centre (k = 2, 3), the library's offsets and spreads of its scatter there.
- `paths[band, process, energy, impact, aspect, thickness]`: the mean length (mm) the detected
  photons of each process (1 Compton, 2 Rayleigh) travelled in the water with an energy in each
  5-keV band of `path_energy_edges_keV`.
- `angles[process, bin, energy]`: the detected scatter by its last scattering's process and angle
  (bins of `scatter_angle_edges_deg`); `element_scattering[element, process, bin, keV]` (Z of
  `element_Z`, 1–150 keV): each element's Compton and Rayleigh cross-sections split over those
  bins, cm²/g.
- `node_offsets_mm[impact, aspect, thickness]`: where each kernel's own ray passes its ellipse's
  axis.
- For the single scatter: water's attenuation on a 1-keV grid from 1 keV, in all and by Compton
  and Rayleigh scattering (`water_mu_per_mm`, `water_compton_per_mm`, `water_rayleigh_per_mm`,
  per mm); `water_pdf[cell, keV, process]`, water's chance per steradian of scattering (1
  Compton, 2 Rayleigh) in equal cells of the cosine of the angle from −1; the detector's
  `efficiency` (signal per photon at each keV, before the energy weighting), its attenuation
  `detector_mu_per_mm` and depth `detector_depth_mm` (for the obliquity); the anti-scatter `grid`
  (`:none`, `:across` the fan or `:both` ways), its `grid_height_mm` and `grid_opening_mm`
  (across, along; `Inf` for none); and the quadrature, `single_scatter` = (depths along the ray,
  points per column, points per row, columns either side of the ray's own).
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
    water_compton_per_mm::Vector{Float64}
    water_rayleigh_per_mm::Vector{Float64}
    water_pdf::Array{Float32, 3}
    efficiency::Vector{Float64}
    detector_mu_per_mm::Vector{Float64}
    detector_depth_mm::Float64
    grid::Symbol
    grid_height_mm::Float64
    grid_opening_mm::NTuple{2, Float64}
    single_scatter::@NamedTuple{
        depths::Int,
        points_per_column::Int,
        points_per_row::Int,
        columns_each_side::Int,
    }
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
        throw(ArgumentError(
            "$(path) is scatter library format $(header["format_version"]); BasisSimulator reads format 2",
        ))
    names = [s["name"] for s in header["scanners"]]
    k = findfirst(==(name), names)
    k === nothing &&
        throw(ArgumentError(
            "the scatter library $(path) has no entry \"$(name)\"; its entries are $(names)",
        ))
    s = header["scanners"][k]
    A = header["arrays"][name]
    for kind in ("scatter", "rayleigh", "forward", "paths", "offsets", "spreads", "angles")
        haskey(A, kind) ||
            throw(ArgumentError(
                "the entry \"$(name)\" of $(path) has no `$(kind)` array (a v2 library has it)",
            ))
    end
    # the single scatter's data: a library written before BasisMonteCarlo swapped each kernel's
    # single scatter for the patient's has none
    elements = header["arrays"]["elements"]
    missing_data = vcat(
        [
            key
            for key in (
                "water_compton_per_mm",
                "water_rayleigh_per_mm",
                "water_pdf_cells",
                "single_scatter",
            )
            if !haskey(header, key)
        ],
        [
            key
            for key in (
                "efficiency",
                "detector_mu_per_mm",
                "detector_depth_mm",
                "grid",
                "grid_height_mm",
                "grid_opening_mm",
            )
            if !haskey(s, key)
        ],
        haskey(elements, "water_pdf") ? String[] : ["elements/water_pdf"],
    )
    isempty(missing_data) ||
        throw(ArgumentError(
            "the scatter library $(path) has no $(join(missing_data, ", ")): it predates the single-scatter " *
                "quadrature (BasisMonteCarlo's `single_scatter`); write it again with BasisMonteCarlo's `write_scatter_library`",
        ))
    axes = header["axes"]
    thicknesses = Float64.(axes["thickness_mm"])
    aspects = Float64.(axes["aspect"])
    impacts = Float64.(axes["impact_fraction"])
    energies = Float64.(axes["energy_keV"])
    sid = Float64(s["source_to_isocenter_mm"])

    kernels = _kernel_processes(A["scatter"], A["rayleigh"], A["forward"])
    nr, nc, nE, _, _, _, _ = size(kernels)

    # where each kernel's own ray passes the axis of its ellipse
    node_offsets = [
        _kernel_offset(T / 2, _kernel_half_width(T, r), u)
        for u in impacts, r in aspects, T in thicknesses
    ]

    # Each kernel's multiple scatter: each process's scatter less the kernel's own single scatter
    # in its water ellipse, never below none, and all of it their sum
    grid = Symbol(s["grid"])
    grid in (:none, :across, :both) ||
        throw(ArgumentError(
            "the scatter library $(path) has grid `$(grid)`; a grid is none, across or both",
        ))
    opening = Tuple(x < 0 ? Inf : Float64(x) for x in s["grid_opening_mm"])
    quadrature = header["single_scatter"]
    single = (;
        depths = Int(quadrature["depths"]),
        points_per_column = Int(quadrature["points_per_column"]),
        points_per_row = Int(quadrature["points_per_row"]),
        columns_each_side = Int(quadrature["columns_each_side"]),
    )
    water_pdf = elements["water_pdf"]
    size(water_pdf, 1) == header["water_pdf_cells"] ||
        throw(ArgumentError(
            "the scatter library $(path) has $(size(water_pdf, 1)) cells of `water_pdf`, its header $(header["water_pdf_cells"])",
        ))
    water = (
        mu = Float64.(header["water_mu_per_mm"]),
        compton = Float64.(header["water_compton_per_mm"]),
        rayleigh = Float64.(header["water_rayleigh_per_mm"]),
    )
    detector = (
        efficiency = Float64.(s["efficiency"]),
        mu = Float64.(s["detector_mu_per_mm"]),
        depth = Float64(s["detector_depth_mm"]),
    )
    tables = _single_tables(
        Float64;
        cols = Float64.(s["column_edges_deg"]),
        rows = Float64.(s["row_edges_mm"]),
        sdd = Float64(s["source_to_detector_mm"]),
        energies,
        water,
        water_pdf,
        detector,
        energy_weighted = Bool(s["energy_weighted"]),
        grid,
        grid_height = Float64(s["grid_height_mm"]),
        grid_opening = opening,
        single,
        forward_deg = Float64(header["forward_rayleigh_deg"]),
    )
    nodes = vec([
        (
            T / 2,
            _kernel_half_width(T, r),
            rad2deg(asin(clamp(node_offsets[iu, ia, iT] / sid, -1, 1))),
        )
        for (iu, u) in enumerate(impacts), (ia, r) in enumerate(aspects), (
            iT,
            T,
        ) in enumerate(thicknesses)
    ])
    _remove_single_scatter!(reshape(kernels, nr, nc, nE, 4, :), tables, nodes, sid)
    moments = _kernel_moments(kernels, A["offsets"], A["spreads"])

    return ScatterKernels(
        String(name),
        String(path),
        Bool(s["energy_weighted"]),
        sid,
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
        water.mu,
        water.compton,
        water.rayleigh,
        water_pdf,
        detector.efficiency,
        detector.mu,
        detector.depth,
        grid,
        Float64(s["grid_height_mm"]),
        opening,
        single,
        Float64.(header["scatter_angle_edges_deg"]),
        Float64(header["forward_rayleigh_deg"]),
        Float64.(header["path_energy_edges_keV"]),
        Int.(header["element_Z"]),
        elements["scattering"],
        node_offsets,
    )
end

# `kernels[row, column, energy, process, impact, aspect, thickness]` from the library's
# `scatter[column, row, impact, aspect, thickness, energy]`, `rayleigh` and `forward`: Compton (the
# scatter less its Rayleigh part), wide Rayleigh (less its forward part) and forward Rayleigh, each
# (row, column) slice next to its energies and processes, so that one corner of the grid is one
# contiguous block; process 4 (all of it) is filled with the multiple scatter
# (`_remove_single_scatter!`).
function _kernel_processes(
    S::AbstractArray{Float32, 6},
    R::AbstractArray{Float32, 6},
    F::AbstractArray{Float32, 6},
)
    nc, nr, nu, na, nT, nE = size(S)
    kernels = Array{Float32}(undef, nr, nc, nE, 4, nu, na, nT)
    for iT in 1:nT, ia in 1:na, iu in 1:nu, iE in 1:nE, c in 1:nc, r in 1:nr
        total, rayleigh, forward = S[c, r, iu, ia, iT, iE],
            R[c, r, iu, ia, iT, iE],
            F[c, r, iu, ia, iT, iE]
        kernels[r, c, iE, 1, iu, ia, iT] = total - rayleigh
        kernels[r, c, iE, 2, iu, ia, iT] = rayleigh - forward
        kernels[r, c, iE, 3, iu, ia, iT] = forward
    end
    return kernels
end

# `moments[column, k, energy, process, impact, aspect, thickness]`: each column's scatter (k = 1)
# and its first and second moments about the column's centre (k = 2, 3), as the kernel's scatter
# landed there (`offsets` and `spreads[column, impact, aspect, thickness, energy]`, degrees).
function _kernel_moments(
    kernels::Array{Float32, 7},
    O::AbstractArray{Float32, 5},
    Sd::AbstractArray{Float32, 5},
)
    _, nc, nE, n_p, nu, na, nT = size(kernels)
    moments = Array{Float32}(undef, nc, 3, nE, n_p, nu, na, nT)
    for iT in 1:nT, ia in 1:na, iu in 1:nu, iE in 1:nE, p in 1:n_p, c in 1:nc
        mass = sum(view(kernels, :, c, iE, p, iu, ia, iT))
        o, sd = O[c, iu, ia, iT, iE], Sd[c, iu, ia, iT, iE]
        moments[c, 1, iE, p, iu, ia, iT] = mass
        moments[c, 2, iE, p, iu, ia, iT] = mass * o
        moments[c, 3, iE, p, iu, ia, iT] = mass * (sd^2 + o^2)
    end
    return moments
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
    file::String   # the header's file name inside the artifact
    url::String    # the release asset (a .zip of the header and its .bin)
    sha256::String # SHA-256 of the .zip
    # git-tree-sha1 of the extracted directory (the artifact store's key)
    tree_hash::String
end

# RELEASE: the assets are a data release of their own, tag `scatter-kernels-v2` (made before the
# package release, so that the hashes below can be registered with it; RELEASING.md). Fill `sha256`
# and `tree_hash` from `lib/BasisMonteCarlo/scripts/split_scatter_library.jl --archive`, which
# writes the per-scanner files and prints both. These are the kernels of notebook 10's run of
# 2026-11-03 (scorecard 52 of 52; notebook 11, 37 of 37).
const _SCATTER_RELEASE_URL = "https://github.com/MolloiLab/BasisSimulator.jl/releases/download/scatter-kernels-v2"

_scatter_artifact(stem, sha256, tree_hash) = ScatterKernelArtifact(
    "$(stem).toml",
    "$(_SCATTER_RELEASE_URL)/$(stem).zip",
    sha256,
    tree_hash,
)

"""
    SCATTER_KERNEL_ARTIFACTS

The scatter library's entries and the artifact each ships as: `Dict(entry => (file, url,
sha256, tree_hash))`. A scanner's `scatter_kernels` field names one of these.
"""
const SCATTER_KERNEL_ARTIFACTS = Dict{String, ScatterKernelArtifact}(
    "SOMATOM Force" => _scatter_artifact(
        "somatom_force",
        "bf75c5d4a2cd795ffba12ac94de326bbf3113b9a99c243885c79e7168a831b6a",
        "ff300294a6c65e083e0b1356fba7d2fa19910948",
    ),
    "SOMATOM Definition Flash" => _scatter_artifact(
        "somatom_definition_flash",
        "ced5ba8c10f24e47db93341cdcf7db75adb1d3fea52abf990e005e438c0e24d6",
        "33201d6836e85d8d989259d328d87203bb4a12cc",
    ),
    "NAEOTOM Alpha" => _scatter_artifact(
        "naeotom_alpha",
        "9790e51c326bb6431cda259f3e94adbbb49f3c6a3ec91d3988637669b807388d",
        "f0a3a0c2a492fb4fbda03b99ac78afdcaa92ab21",
    ),
    "Revolution Apex" => _scatter_artifact(
        "revolution_apex",
        "13578a4f09eca822930ec50e075811211ed1da5a32f276378cff81ba8fe91397",
        "93e888558045216c18e4609df3d9113af12e468d",
    ),
    "Brilliance iCT" => _scatter_artifact(
        "brilliance_ict",
        "1ed67fbd220260923021a18240b6fb4118121575176d908338bc7164a996e78a",
        "d66216a3a5d3216cba3435d908555e49f0234779",
    ),
    "Aquilion ONE" => _scatter_artifact(
        "aquilion_one",
        "f0c4d8a373c21bdd8fd9fa1ca2d1d822b66d6395337460f9faa987e4cce5d2ac",
        "9b5e3229471e8cb2da9196b2c7f1f2ddcdfc5757",
    ),
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
function scatter_kernels(
    entry::AbstractString;
    path::Union{Nothing, AbstractString} = nothing,
    quiet::Bool = false,
)
    return lock(_SCATTER_KERNELS_LOCK) do
        cached = get(_SCATTER_KERNELS, entry, nothing)
        if cached !== nothing && (path === nothing || abspath(path) == cached.path)
            return cached
        end
        header = if path === nothing
            joinpath(
                _download_scatter_kernels(entry; quiet),
                SCATTER_KERNEL_ARTIFACTS[entry].file,
            )
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
    isempty(scanner.scatter_kernels) &&
        throw(ArgumentError(
            "this scanner names no scatter kernels (`scatter_kernels = \"\"`), so " *
                "`SimOptions(use_scatter = true)` has nothing to simulate. The library has kernels for " *
                "SOMATOM_FORCE and SOMATOM_FORCE_B (\"SOMATOM Force\"), SOMATOM_DEFINITION_FLASH and " *
                "SOMATOM_DEFINITION_FLASH_B (\"SOMATOM Definition Flash\"), REVOLUTION_APEX " *
                "(\"Revolution Apex\") and NAEOTOM_ALPHA and NAEOTOM_ALPHA_UHR (\"NAEOTOM Alpha\"), and " *
                "for \"Brilliance iCT\" and \"Aquilion ONE\". Start from one of those constants, name " *
                "an entry (`$(nameof(typeof(scanner)))(s; scatter_kernels = \"SOMATOM Force\")`), or " *
                "set `use_scatter = false`.",
        ))
    return scatter_kernels(scanner.scatter_kernels)
end

# The directory holding the entry's files, from Julia's artifact store, downloading it if needed.
function _download_scatter_kernels(entry::AbstractString; quiet::Bool = false)
    haskey(SCATTER_KERNEL_ARTIFACTS, entry) ||
        throw(ArgumentError(
            "no scatter library entry \"$(entry)\"; the entries are $(sort!(collect(keys(SCATTER_KERNEL_ARTIFACTS)))). " *
                "A library of your own loads with `scatter_kernels(\"$(entry)\"; path = \"…/library.toml\")`.",
        ))
    a = SCATTER_KERNEL_ARTIFACTS[entry]
    isempty(a.sha256) &&
        throw(ArgumentError(
            "the scatter kernels of \"$(entry)\" are not released yet (the GitHub release " *
                "`scatter-kernels-v2` will carry them). Load a local copy of the library BasisMonteCarlo wrote with " *
                "`scatter_kernels(\"$(entry)\"; path = \"…/$(a.file)\")`, or set `use_scatter = false`.",
        ))
    hash = Base.SHA1(a.tree_hash)
    Artifacts.artifact_exists(hash) && return Artifacts.artifact_path(hash)

    quiet ||
        @info "Downloading the scatter kernels of \"$(entry)\" into the Julia artifact store…"
    dest = Artifacts.artifact_path(hash)
    mkpath(dirname(dest))
    # staged on the store's own filesystem, so that the install is a rename
    staging = mktempdir(dirname(dest))
    try
        archive = joinpath(staging, basename(a.url))
        Downloads.download(a.url, archive)
        got = open(io -> bytes2hex(SHA.sha256(io)), archive)
        got == a.sha256 ||
            error(
                "SHA-256 mismatch for the scatter kernels of \"$(entry)\"\n  expected $(a.sha256)\n  got      $(got)",
            )
        extracted = joinpath(staging, "extracted")
        mkpath(extracted)
        run(pipeline(
            `$(p7zip_jll.p7zip()) x -y -o$(extracted) $(archive)`;
            stdout = devnull,
        ))
        # a concurrent download won
        Artifacts.artifact_exists(hash) && return Artifacts.artifact_path(hash)
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
_kernel_offset(a, b, impact_fraction) = impact_fraction *
    _kernel_seen_half_width(a, b, impact_fraction)

# The length (mm) of the kernel ray of impact `u` inside its ellipse: a line passing the centre of
# an ellipse at `u` times its half-width `h` across the line crosses it for 2ab·√(1 − u²)/h.
# `q = √(1 − u²)` is passed by `_kernel_impact`, exact near the edge where 1 − u² is not.
_kernel_chord(a, b, u, q = _complement(u)) = 2a * b * q / _kernel_seen_half_width(a, b, u)

# √(1 − x²), for x in [0, 1]
@inline _complement(x) = sqrt(max(zero(x), (one(x) - x) * (one(x) + x)))

# The impact fraction `u` in [0, 1) whose kernel ray crosses the ellipse (b, a) for `length_mm`,
# and `q = √(1 − u²)`, both to the float type's precision (BasisMonteCarlo's `kernel_impact`):
# bisection on u while u ≤ q, and nearer the edge on q, whose digits 1 − u² would lose (a 0.03 mm
# chord of a 16 cm circle has q ≈ 1e-4). 0 when the ray through the axis (2a long) is no longer than
# that, to rounding.
function _kernel_impact(a, b, length_mm)
    T = typeof(a)
    o = one(T)
    2a <= length_mm * (o + T(1.0e-9)) && return (zero(T), o)
    half = sqrt(T(0.5))
    near_axis = _kernel_chord(a, b, half, half) <= length_mm
    # the chord falls with u and rises with q
    lo, hi = zero(T), half
    for _ in 1:80
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        x, y = near_axis ? (mid, _complement(mid)) : (_complement(mid), mid)
        (_kernel_chord(a, b, x, y) > length_mm) == near_axis ? (lo = mid) : (hi = mid)
    end
    mid = (lo + hi) / 2
    return near_axis ? (mid, _complement(mid)) : (_complement(mid), mid)
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
    i == 1 ?
        xs[2] - xs[1] :
        i == length(xs) ? xs[end] - xs[end - 1] : (xs[i + 1] - xs[i - 1]) / 2
    for i in eachindex(xs)
]

# For rays at `angle_deg` from the central ray (sorted, at least two): `rays[(x, width, sin, cos), ray]`,
# where each passes the isocentre `sid` mm from the focal spot, the width across the view it stands
# for and its direction, and `boundaries[(sin, cos), k]`, the direction halfway between rays k and
# k + 1, each value the (hi, lo) pair of `T`s that sums to its Float64 (`_scatter_strips!`, `_RayStrips`)
function _ray_pairs(::Type{T}, angle_deg, sid) where {T}
    xs = [sid * sind(γ) for γ in angle_deg]
    rays = reduce(
        hcat,
        [
            [_pair(T, x)..., _pair(T, w)..., _pair(T, sind(γ))..., _pair(T, cosd(γ))...]
            for (x, w, γ) in zip(xs, _ray_widths(xs), angle_deg)
        ],
    )
    halfway = (angle_deg[1:(end - 1)] .+ angle_deg[2:end]) ./ 2
    boundaries = reduce(
        hcat,
        [[_pair(T, sind(β))..., _pair(T, cosd(β))...] for β in halfway],
    )
    return rays, boundaries
end

# `(sum, compensation) + x`, Kahan's: the device sums in single precision.
@inline function _kahan(sum, c, x)
    y = x - c
    t = sum + y
    return (t, (t - sum) - y)
end

# Double-float arithmetic: a value as the pair (hi, lo) of floats of the workspace's type that sum
# to it, about twice its digits (Dekker's, with the fused multiply-add). The single scatter's yes/no
# decisions (which strip a point is in, whether it is inside it, which cell of the angular tables)
# are taken again in it when single precision is too near the line to tell, so that they fall as
# in double precision (and on every backend: Metal has no Float64).
@inline function _two_prod(a, b)
    p = a * b
    return (p, fma(a, b, -p))
end
@inline function _quick_two_sum(a, b)
    s = a + b
    return (s, b - (s - a))
end
@inline function _dd_add(a::NTuple{2}, b::NTuple{2})
    s = a[1] + b[1]
    v = s - a[1]
    e = (a[1] - (s - v)) + (b[1] - v)
    return _quick_two_sum(s, e + (a[2] + b[2]))
end
@inline _dd_sub(a::NTuple{2}, b::NTuple{2}) = _dd_add(a, (-b[1], -b[2]))
@inline function _dd_mul(a::NTuple{2}, b::NTuple{2})
    p, e = _two_prod(a[1], b[1])
    return _quick_two_sum(p, e + (a[1] * b[2] + a[2] * b[1]))
end
@inline function _dd_div(a::NTuple{2}, b::NTuple{2})
    q = a[1] / b[1]
    r = _dd_sub(a, _dd_mul((q, zero(q)), b))
    return _quick_two_sum(q, r[1] / b[1])
end
@inline function _dd_sqrt(a::NTuple{2})
    s = sqrt(a[1])
    s > 0 || return (s, zero(s))
    r = _dd_sub(a, _two_prod(s, s))
    return _quick_two_sum(s, r[1] / (2s))
end
@inline _dd(x) = (x, zero(x))
# a Float64 as the pair of `T`s that sum to it (on the host)
_pair(::Type{T}, x::Real) where {T} = (T(x), T(Float64(x) - Float64(T(x))))

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
# Single scatter (BasisMonteCarlo's `single_scatter`)
# =============================================================================
#
# The scatter of a pencil ray's first scattering on the scanner's detector, a deterministic
# quadrature of what forced detection integrates along the ray: the ray leaves the focal spot (the
# origin; the central ray along +y) at an angle across the fan, at z = 0, through a medium that
# scatters and attenuates as water of its water-equivalent density. At each of `depths` sites along
# it: its chance per mm of each scattering, the attenuation on the way there and, toward
# `points_per_column × points_per_row` points of every bin of the detector within
# `columns_each_side` columns of the ray's own (a grid lets no single scatter through beyond; every
# column without a grid): water's angular distribution, the way out (its in-plane water-equivalent
# length lengthened by the elevation), the grid, the detector's efficiency (with the obliquity) and,
# for an energy integrator, the energy. Compton, wide Rayleigh and forward Rayleigh (an angle below
# `forward_rayleigh_deg`) apart. Generic in the float type: the device runs it in the workspace's.

const _ELECTRON_REST_KEV = 510.99895

# The medium of a kernel: water, an elliptical cylinder of semi-axes `a` along the central ray and `b`
# across it, its axis `axis` mm from the focal spot on the central ray (BasisMonteCarlo's `WaterEllipse`).
struct _WaterEllipse{T}
    a::T
    b::T
    axis::T
end

# A patient as one view-group's coarse rays describe it (BasisMonteCarlo's `RayStrips`): ray `i`'s
# water-equivalent length spread evenly over its outline chord, which starts `strips[1:2, base + i]`
# mm from the focal spot (a double-float), is `strips[3, …]` long and has the density `strips[4, …]`.
# The `n` rays are at `first + (i − 1) · step` degrees; a point belongs to the ray nearest its
# direction from the focal spot, `boundaries[:, k]` (sin, cos of the angle halfway between rays k and
# k + 1, double-floats) the line between them. Uniform along z.
struct _RayStrips{A, B, T}
    strips::A
    boundaries::B
    base::Int
    n::Int
    first::T
    step::T
end

# Where the ray at `γ` degrees from the focal spot enters the medium (mm from the focal spot, a
# double-float), how long it is inside, and its density there (BasisMonteCarlo's `ray_segment`)
@inline function _ray_segment(m::_WaterEllipse, γ)
    z = zero(m.a)
    sγ, cγ = sincos(deg2rad(γ))
    qy = -m.axis / m.a
    ux, uy = sγ / m.b, cγ / m.a
    A, B, C = ux^2 + uy^2, qy * uy, qy^2 - 1
    disc = B^2 - A * C
    disc > 0 || return (_dd(z), z, z)
    return (_dd((-B - sqrt(disc)) / A), 2sqrt(disc) / A, one(z))
end

# (the strip of a coarse ray: by its number, or by its angle)
@inline function _ray_segment(m::_RayStrips, i::Int)
    j = m.base + i
    return ((m.strips[1, j], m.strips[2, j]), m.strips[3, j], m.strips[4, j])
end
@inline _ray_segment(m::_RayStrips, γ::AbstractFloat) = _ray_segment(
    m,
    clamp(unsafe_trunc(Int, round((γ - m.first) / m.step)), 0, m.n - 1) + 1,
)

# The water-equivalent length (mm) from the in-plane point `P` to the edge of the medium along the
# unit direction `ω` (BasisMonteCarlo's `exit_wet`; both double-floats, `((x hi, x lo), (y hi, y lo))`):
# through an ellipse, its chord beyond the point (BasisMonteCarlo's `ellipse_chord`); through the
# strips, the density every 2 mm, out of the patient at the first point outside it after one inside.
@inline function _exit_wet(m::_WaterEllipse, P, ω)
    x, y, s, c = P[1][1], P[2][1], ω[1][1], ω[2][1]
    ox, oy = x / m.b, (y - m.axis) / m.a
    ux, uy = s / m.b, c / m.a
    A = ux^2 + uy^2
    B = 2 * (ox * ux + oy * uy)
    C = ox^2 + oy^2 - 1
    disc = B^2 - 4A * C
    disc <= 0 && return zero(x)
    t1, t2 = (-B - sqrt(disc)) / (2A), (-B + sqrt(disc)) / (2A)
    return max(zero(x), t2) - max(zero(x), t1)
end

# A point of the way out, `t` mm from `P` along `ω`, as a double-float
@inline _dd_along(P, ω, t) = (
    _dd_add(P[1], _dd_mul(_dd(t), ω[1])),
    _dd_add(P[2], _dd_mul(_dd(t), ω[2])),
)

@inline function _exit_wet(m::_RayStrips, P, ω)
    T = typeof(P[1][1])
    step = T(2)
    # mm, and steps: well beyond single precision's error on either
    near = T(1.0e-3)
    total, t, inside = zero(T), step / 2, false
    while t < T(1000)
        x, y = P[1][1] + t * ω[1][1], P[2][1] + t * ω[2][1]
        # the nearest ray (BasisMonteCarlo's `nearest_ray`, a tie to the later one), in double-float
        # when the point is near the line between two
        f = (rad2deg(atan(x, y)) - m.first) / m.step + T(0.5)
        k = floor(f)
        i = clamp(unsafe_trunc(Int, clamp(k, -one(k), oftype(k, m.n))), 0, m.n - 1) + 1
        frac = f - k
        b = frac < T(0.5) ? i - 1 : i
        if 0 <= k < m.n && min(frac, 1 - frac) < near && 1 <= b < m.n
            X, Y = _dd_along(P, ω, t)
            s = _dd_sub(
                _dd_mul(X, (m.boundaries[3, b], m.boundaries[4, b])),
                _dd_mul(Y, (m.boundaries[1, b], m.boundaries[2, b])),
            )
            i = s[1] >= 0 ? b + 1 : b
        end
        # inside its strip, in double-float when the point is near either end of it
        j = m.base + i
        entry, L = m.strips[1, j], m.strips[3, j]
        r = sqrt(x^2 + y^2)
        within = r - entry >= 0 && entry + L - r >= 0
        if L > 0 && min(abs(r - entry), abs(entry + L - r)) < near
            X, Y = _dd_along(P, ω, t)
            r2 = _dd_add(_dd_mul(X, X), _dd_mul(Y, Y))
            E = (entry, m.strips[2, j])
            F = _dd_add(E, _dd(L))
            within = _dd_sub(r2, _dd_mul(E, E))[1] >= 0 &&
                _dd_sub(_dd_mul(F, F), r2)[1] >= 0
        end
        ρ = L > 0 && within ? m.strips[4, j] : zero(T)
        ρ > 0 && (inside = true)
        inside && ρ == 0 && break
        total += ρ * step
        t += step
    end
    return total
end

# The fraction a focused grid lets through at `α` (rad) from its focus, one way
@inline _grid_acceptance(α, height, opening) = isinf(opening) ?
    one(α) :
    max(zero(α), 1 - height * abs(tan(α)) / opening)

# Linear interpolation of a table on the 1-keV grid from 1 keV, clamped (BasisMonteCarlo's `on_kev_grid`)
@inline function _on_kev_grid(table, E)
    n = length(table)
    e = clamp(E, one(E), oftype(E, n))
    i = min(unsafe_trunc(Int, floor(e)), n - 1)
    return table[i] + (e - i) * (table[i + 1] - table[i])
end

# The single-scatter quadrature's inputs in the float type `T`, as plain arrays and numbers: the
# library's columns and rows, the direction of each column point (`column_points[(sin, cos) as
# (hi, lo) pairs, point, column]`), the scanner's source–detector distance (and as a pair, `sdd2`),
# water's attenuation on the 1-keV grid (`total`) and at the library's `energies` (`mu[energy, (all,
# Compton, Rayleigh)]`), its angular tables at each of the library's energies (`pdf[cell, energy,
# (Compton, Rayleigh)]`, between the 1-keV rows read), the detector's efficiency, attenuation × depth
# (`mud`, 0 for no obliquity) and 1/(1 − e^{−mud}), and the same at each library energy (`at_energy[energy,
# (efficiency, mud, norm, E / mₑc²)]`: what the quadrature takes once per energy, not per point), the grid (0 none, 1 across, 2 both), the quadrature's counts, the forward
# Rayleigh's (1 − cos θ)/2 (and as a pair), and whether water scatters at all.
function _single_tables(
    ::Type{T};
    cols,
    rows,
    sdd,
    energies,
    water,
    water_pdf,
    detector,
    energy_weighted,
    grid,
    grid_height,
    grid_opening,
    single,
    forward_deg,
) where {T}
    E = Float64.(energies)
    mu = [
        _on_kev_grid(table, Ek)
        for Ek in E, table in (water.mu, water.compton, water.rayleigh)
    ]
    n_keV = size(water_pdf, 2)
    pdf_row = [min(floor(Int, clamp(Ek, 1, n_keV)), n_keV - 1) for Ek in E]
    pdf_t = T.(clamp.(E, 1, n_keV) .- pdf_row)
    pdf = [
        (one(T) - pdf_t[e]) * T(water_pdf[j, pdf_row[e], p]) +
            pdf_t[e] * T(water_pdf[j, pdf_row[e] + 1, p])
        for j in axes(water_pdf, 1), e in eachindex(E), p in 1:2
    ]
    mud = [detector.depth > 0 && μ > 0 ? μ * detector.depth : 0.0 for μ in detector.mu]
    efficiency, mud, norm = T.(detector.efficiency),
        T.(mud),
        T.([m > 0 ? 1 / (1 - exp(-m)) : 0.0 for m in mud])
    k = [clamp(unsafe_trunc(Int, round(T(Ek))), 1, length(efficiency)) for Ek in E]
    at_energy = hcat(efficiency[k], mud[k], norm[k], T.(E) ./ T(_ELECTRON_REST_KEV))
    Q, nc = single.points_per_column, length(cols) - 1
    column_points = zeros(T, 4, Q, nc)
    for c in 1:nc, q in 1:Q
        s, co = sincos(deg2rad(Float64(
            T(cols[c]) + (q - T(0.5)) / Q * (T(cols[2]) - T(cols[1])),
        )))
        column_points[:, q, c] .= (_pair(T, s)..., _pair(T, co)...)
    end
    forward = (1 - cosd(forward_deg)) / 2
    return (;
        cols = T.(cols),
        rows = T.(rows),
        column_points,
        sdd = T(sdd),
        sdd2 = _pair(T, sdd),
        energies = T.(E),
        total = T.(water.mu),
        mu = T.(mu),
        pdf,
        efficiency,
        mud,
        norm,
        at_energy,
        energy_weighted = Bool(energy_weighted),
        grid = grid === :none ? 0 : grid === :across ? 1 : 2,
        grid_height = T(grid_height),
        opening = T.(grid_opening),
        depths = single.depths,
        points_per_column = single.points_per_column,
        points_per_row = single.points_per_row,
        columns_each_side = single.columns_each_side,
        forward = _pair(T, forward),
        scatters = any(>(0), view(mu, :, 2)) || any(>(0), view(mu, :, 3)),
    )
end

# The detector's signal per photon of `E` keV arriving at `obliquity` (the cosine to its normal): its
# efficiency `η`, its path (attenuation × depth `m`, `norm` = 1/(1 − e^{−m})) lengthened by the
# obliquity, and the energy for an energy integrator; `_single_efficiency` looks them up at `E`
@inline function _detected(η, m, norm, E, obliquity, energy_weighted)
    m > 0 && (η *= (1 - exp(-m / obliquity)) * norm)
    return energy_weighted ? η * E : η
end

@inline function _single_efficiency(E, obliquity, efficiency, mud, norm, energy_weighted)
    k = clamp(unsafe_trunc(Int, round(E)), 1, length(efficiency))
    return _detected(efficiency[k], mud[k], norm[k], E, obliquity, energy_weighted)
end

# One column point of the detector (`D` mm from the focal spot, `Δ` rad across the fan from the ray)
# seen from a site of the ray `t` mm from the focal spot, in the plane: the length ℓ of the way to it
# and its part `along` the column point's direction, 1 − cos φ for the angle φ it turns off the ray,
# and the grid's acceptance across the fan (1 without one). Each is BasisMonteCarlo's (from the
# site's and the point's coordinates) written in Δ, so that nothing cancels in single precision:
# the fan angle `atan(sγ dy − cγ dx, sγ dx + cγ dy)` is `atan(t sin Δ, D − t cos Δ)`, and the way
# to the point is `D cos Δ − t` along the ray and `D sin Δ` across it.
@inline function _single_column(t, Δ, D, grid, height, opening)
    sΔ, cΔ = sincos(Δ)
    along = D - t * cΔ
    ℓ = sqrt(along^2 + (t * sΔ)^2)
    g = grid == 0 ? one(t) : _grid_acceptance(atan(t * sΔ, along), height, opening)
    c = (D * cΔ - t) / ℓ
    turn = c > 0 ? (D * sΔ / ℓ)^2 / (1 + c) : 1 - c
    return ℓ, along, turn, g
end

# One point of the detector `z` mm along the rows from a column point (`_single_column`): its solid
# angle (`area` mm² of detector) times the grid's acceptance `g` (0: none of it through), the
# scattering angle as (1 − cos θ)/2, cos θ = cos ε cos φ for the elevation ε (which keeps its digits
# near the ray, where the forward Rayleigh peak's cells are narrow, as 1 − cos θ in single precision
# does not), the obliquity at the detector, and the factor that lengthens the in-plane way out by
# the elevation. The elevation's angle off the grid's focus is BasisMonteCarlo's
# `atan(z, ℓ) − atan(z, D)` as one arctangent (the difference of two near ones loses its digits).
@inline function _single_point(ℓ, along, turn, z, D, area, g, grid, height, opening)
    L2 = ℓ^2 + z^2
    L = sqrt(L2)
    grid == 2 && (g *= _grid_acceptance(atan(z * (D - ℓ), ℓ * D + z^2), height, opening))
    facing = along / L
    obliquity = max(
        oftype(L, 1.0e-3),
        abs((facing + (z / D) * (z / L)) / sqrt(1 + (z / D)^2)),
    )
    cos_ε = ℓ / L
    half_versine = ((z / L)^2 / (1 + cos_ε) + cos_ε * turn) / 2
    return area * abs(facing) / L2 * g, half_versine, obliquity, L / ℓ
end

# The unit direction from the site `P` to the column point (sγ, cγ) `D` mm from the focal spot, all
# double-floats: the way out's
@inline function _dd_way(P, sγ, cγ, D)
    dx = _dd_sub(_dd_mul(D, sγ), P[1])
    dy = _dd_sub(_dd_mul(D, cγ), P[2])
    ℓ = _dd_sqrt(_dd_add(_dd_mul(dx, dx), _dd_mul(dy, dy)))
    return (_dd_div(dx, ℓ), _dd_div(dy, ℓ))
end

# (1 − cos θ)/2 as `_single_point` has it, in double-float: from the site `t` mm along the ray, the
# angle Δ between the ray and the column point (its sine and cosine), D and z
@inline function _dd_half_versine(t, sΔ, cΔ, D, z)
    o = _dd(one(z))
    dot = _dd_sub(_dd_mul(D, cΔ), t)
    cross = _dd_mul(D, sΔ)
    ℓ2 = _dd_add(_dd_mul(dot, dot), _dd_mul(cross, cross))
    ℓ = _dd_sqrt(ℓ2)
    c = _dd_div(dot, ℓ)
    turn = c[1] > 0 ?
        _dd_div(_dd_div(_dd_mul(cross, cross), ℓ2), _dd_add(o, c)) :
        _dd_sub(o, c)
    z2 = _two_prod(z, z)
    L2 = _dd_add(ℓ2, z2)
    cos_ε = _dd_div(ℓ, _dd_sqrt(L2))
    v = _dd_add(_dd_div(_dd_div(z2, L2), _dd_add(o, cos_ε)), _dd_mul(cos_ε, turn))
    return (v[1] / 2, v[2] / 2)
end

# The cell of water's angular tables (equal in cos θ from −1) holding the angle whose (1 − cos θ)/2
# is `v`, and the Rayleigh scattering's process (3 forward, below `forward`, else 2), each again in
# double-float (`_dd_half_versine(t, sΔ, cΔ, D, z)`) when single precision is too near the line to tell
@inline function _single_cell(v, cells, forward, t, sΔ, cΔ, D, z)
    f = v * cells
    frac = f - floor(f)
    near = oftype(v, 3.0e-3)
    if min(frac, 1 - frac) < near || abs(v - forward[1]) < near * forward[1]
        v2 = _dd_half_versine(t, sΔ, cΔ, D, z)
        g = _dd_mul(v2, _dd(oftype(v, cells)))
        c = ceil(g[1])
        c = g[1] == c && g[2] > 0 ? c + 1 : c
        return clamp(cells + 1 - unsafe_trunc(Int, c), 1, cells),
            _dd_sub(v2, forward)[1] < 0 ? 3 : 2
    end
    return clamp(cells + 1 - unsafe_trunc(Int, ceil(f)), 1, cells), v < forward[1] ? 3 : 2
end

# The Compton and the Rayleigh scatter (per mm of path, per steradian, signal per photon) of a site
# at water-equivalent depth `τ` toward a point of the detector at the angle whose (1 − cos θ)/2 is
# `v` (cell `j`), its way out `path` long, at the library's energy `e`
@inline function _single_energy(t, e, j, v, τ, path, obliquity)
    E = t.energies[e]
    μT, μC, μR = t.mu[e, 1], t.mu[e, 2], t.mu[e, 3]
    η, m, norm, recoil = t.at_energy[e, 1],
        t.at_energy[e, 2],
        t.at_energy[e, 3],
        t.at_energy[e, 4]
    o = one(E)
    rayleigh = μR *
        t.pdf[j, e, 2] *
        exp(-μT * (τ + path)) *
        _detected(η, m, norm, E, obliquity, t.energy_weighted)
    E′ = E / (o + recoil * 2v)
    compton = μC *
        t.pdf[j, e, 1] *
        exp(-μT * τ - _on_kev_grid(t.total, E′) * path) *
        _single_efficiency(E′, obliquity, t.efficiency, t.mud, t.norm, t.energy_weighted)
    return compton, rayleigh
end

# The single scatter of the pencil ray at `angle_deg` through `medium`, added to `out[row, column,
# energy, process]` (Compton, wide Rayleigh, forward Rayleigh; per incident photon of each of the
# library's energies), the columns binned at the scanner's own: BasisMonteCarlo's `single_scatter`.
function _single_scatter!(out::AbstractArray{<:Real, 4}, medium, angle_deg, t)
    T = eltype(t.energies)
    cols, rows = t.cols, t.rows
    nc, nr = length(cols) - 1, length(rows) - 1
    t.scatters || return out
    u = T(angle_deg)
    entry, L, ρ = _ray_segment(medium, u)
    (L > 0 && ρ > 0) || return out
    M, Q, Z, C = t.depths, t.points_per_column, t.points_per_row, t.columns_each_side
    Δc, Δr = cols[2] - cols[1], rows[2] - rows[1]
    D = t.sdd
    dt = L / M
    s64, c64 = sincos(deg2rad(Float64(angle_deg)))
    su2, cu2 = _pair(T, s64), _pair(T, c64)
    own = clamp(searchsortedlast(cols, angle_deg), 1, nc)
    window = t.grid == 0 ? (1:nc) : (max(1, own - C):min(nc, own + C))
    area = D * deg2rad(Δc / Q) * (Δr / Z)
    for m in 1:M
        # the scattering site (a double-float: the way out's decisions are taken from it), and the
        # water-equivalent length the photon crossed to reach it
        depth = _dd_div(_two_prod(T(m) - T(0.5), L), _dd(T(M)))
        site = _dd_add(entry, depth)
        P = (_dd_mul(site, su2), _dd_mul(site, cu2))
        τ = ρ * depth[1]
        for ci in window, qc in 1:Q
            γ = cols[ci] + (qc - T(0.5)) / Q * Δc
            ℓ, along, turn, g_fan = _single_column(
                site[1],
                deg2rad(u - γ),
                D,
                t.grid,
                t.grid_height,
                t.opening[1],
            )
            g_fan > 0 || continue
            sγ = (t.column_points[1, qc, ci], t.column_points[2, qc, ci])
            cγ = (t.column_points[3, qc, ci], t.column_points[4, qc, ci])
            exit = _exit_wet(medium, P, _dd_way(P, sγ, cγ, t.sdd2))
            sΔ = _dd_sub(_dd_mul(su2, cγ), _dd_mul(cu2, sγ))
            cΔ = _dd_add(_dd_mul(cu2, cγ), _dd_mul(su2, sγ))
            for ri in 1:nr, qz in 1:Z
                z = rows[ri] + (qz - T(0.5)) / Z * Δr
                w, v, obliquity, lengthen = _single_point(
                    ℓ,
                    along,
                    turn,
                    z,
                    D,
                    area,
                    g_fan,
                    t.grid,
                    t.grid_height,
                    t.opening[2],
                )
                w > 0 || continue
                j, p = _single_cell(v, size(t.pdf, 1), t.forward, site, sΔ, cΔ, t.sdd2, z)
                for e in eachindex(t.energies)
                    compton, rayleigh = _single_energy(
                        t,
                        e,
                        j,
                        v,
                        τ,
                        exit * lengthen,
                        obliquity,
                    )
                    out[ri, ci, e, 1] += ρ * dt * w * compton
                    out[ri, ci, e, p] += ρ * dt * w * rayleigh
                end
            end
        end
    end
    return out
end

# Each kernel's multiple scatter, in place of its scatter: `kernels[row, column, energy, process,
# node]`, each process (Compton, wide and forward Rayleigh) less the node's own single scatter in its
# water ellipse at its own ray, never below none, and all of it (process 4) their sum. `nodes[node]`
# is (a, b, angle_deg): the ellipse's semi-axes along and across the ray, its axis `sid` from the focal
# spot, and the angle of the kernel's own ray from the central ray. Once per library entry, on the
# host's threads.
function _remove_single_scatter!(kernels::AbstractArray{Float32, 5}, tables, nodes, sid)
    nr, nc, nE = size(kernels, 1), size(kernels, 2), size(kernels, 3)
    let kernels = kernels, tables = tables, nodes = nodes, sid = sid
        _foreachindex(eachindex(nodes), AK.get_backend(kernels)) do node
            a, b, angle = nodes[node]
            single = zeros(nr, nc, nE, 3)
            _single_scatter!(single, _WaterEllipse(a, b, sid), angle, tables)
            for e in 1:nE, c in 1:nc, r in 1:nr
                total = 0.0
                for p in 1:3
                    left = max(Float64(kernels[r, c, e, p, node]) - single[r, c, e, p], 0.0)
                    kernels[r, c, e, p, node] = left
                    total += left
                end
                kernels[r, c, e, 4, node] = total
            end
        end
    end
    return kernels
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
    f = (1 - t) .* Float64.(k.angles[process, :, i1]) .+
        t .* Float64.(k.angles[process, :, i2])
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
# 3. `_scatter_strips!`: each view-group's patient as its rays describe it, for their single scatter.
# 4. `_scatter_corners!`: for each of the 8 kernels of the grid around a ray (its corners) and each
#    channel: the weights of the kernel's energies and processes (the ray's photons and how much
#    less of each process's scatter gets out of its materials than of the kernel's water), and,
#    for each of the kernel's columns, where its multiple scatter lands once moved to the ray's
#    angle: a box of the width the scatter landed with (degrees from a library column's left edge,
#    so that single precision keeps the slivers of it a column takes).
# 5. `_scatter_single_weights!`: each ray's photons by energy and process, scaled by the escape
#    factors of its corners.
# 6. `_scatter_single!`: each ray's own single scatter in the patient, at its own columns.
# 7. `_scatter_deposit!`: each library bin (row relative to the rays' own, column, channel) of each
#    view-group gathers the boxes over it, each the kernel's column mixed by the corner's weights,
#    and its rays' own single scatter.
# 8. `_scatter_rows!`: each library bin of each view, the groups' scatter laid over the rows of
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
        _foreachindex(
            1:(size(lengths, 2) * size(lengths, 3)),
            AK.get_backend(ellipse),
        ) do gv
            e = _view_ellipse(xs, dxs, view(lengths, ((gv - 1) * n_cc + 1):(gv * n_cc)))
            ellipse[3gv - 2] = e.a
            ellipse[3gv - 1] = e.b
            ellipse[3gv] = e.centre
        end
    end
    return ellipse
end

# `shapes[(thickness, aspect, impact, side, weight), ray]`: each ray's own ellipse (as thick as
# the view's, or as the ray's water-equivalent length `wet` if longer, then at impact 0), its
# impact, the side of the axis it passes on (-1: left, its kernels mirrored), and the weight of the
# kernels standing in for it (0: no scatter).
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
    let shapes = shapes,
        clamped = clamped,
        wet = wet,
        ellipse = ellipse,
        xs = xs,
        angle_deg = angle_deg,
        grid = grid,
        n_cc = n_cc,
        mark = mark
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
                aspect = b / a_ray
                thickness = 2a_ray
                inside = T_min <= thickness <= T_max && r_min <= aspect <= r_max
                outside = inside ? Int32(0) : Int32(1)

                # Below the grid's thinnest ellipse the thinnest kernel stands in, scaled by the
                # water the ray crosses against what that kernel's own ray crosses
                amount = o
                if thickness < T_min
                    r_node = clamp(aspect, r_min, r_max)
                    chord = _kernel_chord(
                        T_min / 2,
                        _kernel_half_width(T_min, r_node),
                        u,
                        q,
                    )
                    amount = min(o, L / max(chord, o))
                end
                graze = u > u_last ? q / sqrt(o - u_last^2) : o
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
# `corner[c, ray]`, the kernel's place in the grid (0-based); `corner_weight[c, ray]`, its share of
# the ray (the interpolation's weight alone, 0: none); `factor[ch, c, ray]`, its weight with the
# ray's (0: no scatter lands); `mix[slot, ch, c, ray]`, the weights of its library energies and processes (`3nE`
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
    corner_weight::AbstractMatrix{T},
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
    let corner = corner,
        corner_weight = corner_weight,
        factor = factor,
        mix = mix,
        boxes = boxes,
        window = window,
        shapes = shapes,
        weights = weights,
        plain = plain,
        dmu = dmu,
        angle = angle,
        moments = lib.moments,
        paths = lib.paths,
        node = lib.node_deg,
        thick = lib.thicknesses,
        log_r = lib.log_aspects,
        impacts = lib.impacts,
        w_col = lib.column_width,
        nE = nE,
        n_ch = n_ch,
        nc = nc,
        n_cc = n_cc,
        nu = nu,
        na = na,
        nb = nb,
        n_used = n_used,
        ray0 = ray0
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
            ch == 1 && (corner_weight[c + 1, local_ray] = zero(T))
            @inbounds if ch <= n_used && weight > 0
                u = shapes[3, ray]
                iT = _bracket(thick, shapes[1, ray])
                ir = _bracket(log_r, log(shapes[2, ray]))
                iu = _bracket(impacts, u)
                bT, br, bu = (c >> 2) & 1, (c >> 1) & 1, c & 1
                tT, wT = bT == 0 ? (iT[1], o - iT[3]) : (iT[2], iT[3])
                tr, wr = br == 0 ? (ir[1], o - ir[3]) : (ir[2], ir[3])
                tu, wu = bu == 0 ? (iu[1], o - iu[3]) : (iu[2], iu[3])
                wg = wT * wr * wu
                if wg != 0
                    k = ((tT - 1) * na + (tr - 1)) * nu + (tu - 1)
                    corner[c + 1, local_ray] = Int32(k)
                    ch == 1 && (corner_weight[c + 1, local_ray] = wg)

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
                                mix[g, ch, c + 1, local_ray] = weights[
                                    g,
                                    1,
                                    ch,
                                    local_ray,
                                ] *
                                    escape
                            else
                                mix[g + nE, ch, c + 1, local_ray] = weights[
                                    g,
                                    2,
                                    ch,
                                    local_ray,
                                ] *
                                    escape
                                mix[g + 2nE, ch, c + 1, local_ray] = weights[
                                    g,
                                    3,
                                    ch,
                                    local_ray,
                                ] *
                                    escape
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
                    mirror = side < 0
                    mbase = 3nc * 4nE * k
                    d_lo, d_hi = typemax(Int32), typemin(Int32)

                    # Each column moved by the ray's angle from the kernel's own, as what it holds:
                    # the scatter centred where it landed in the column and spread as it landed, a
                    # box of that width (the mirror image for a ray left of the axis, by its side
                    # even through the axis)
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

# `strips[(entry hi, entry lo, outline, density), ray]`: each view-group's patient as its rays
# describe it (BasisMonteCarlo's `RayStrips`): each ray's water-equivalent length `wet` spread evenly
# over its length through the outline (`outline`), that length centred where the view-group's
# outline ellipse (its moments of the outline lengths, BasisMonteCarlo's `view_ellipse`) centres the
# ray's chord, at the point of the ray nearest the ellipse's centre in the ellipse's own measure
# (on the ray at the isocentre without an ellipse). In double-float throughout (the strips' ends
# decide what a point is in, `_exit_wet`): `rays[(x, width, sin, cos) as (hi, lo) pairs, ray]`, where
# each coarse ray passes the isocentre (`sid` mm from the focal spot, a pair, as `π` is), the width
# across the view it stands for, and its direction.
function _scatter_strips!(
    strips::AbstractMatrix{T},
    rays::AbstractMatrix{T},
    wet::AbstractArray{T, 3},
    outline::AbstractArray{T, 3},
    sid::NTuple{2, T},
    π2::NTuple{2, T},
) where {T}
    n_cc = size(wet, 1)
    let strips = strips,
        rays = rays,
        wet = wet,
        outline = outline,
        sid = sid,
        π2 = π2,
        n_cc = n_cc
        _foreachindex(1:(size(wet, 2) * size(wet, 3)), AK.get_backend(strips)) do gv
            base = (gv - 1) * n_cc
            z = _dd(zero(T))
            m0, m1 = z, z
            @inbounds for i in 1:n_cc
                w = _dd_mul(_dd(outline[base + i]), (rays[3, i], rays[4, i]))
                m0 = _dd_add(m0, w)
                m1 = _dd_add(m1, _dd_mul((rays[1, i], rays[2, i]), w))
            end
            a2, b2, x0 = z, z, z
            if m0[1] > 0
                x0 = _dd_div(m1, m0)
                spread = z
                @inbounds for i in 1:n_cc
                    d = _dd_sub((rays[1, i], rays[2, i]), x0)
                    spread = _dd_add(
                        spread,
                        _dd_mul(
                            _dd_mul(d, d),
                            _dd_mul(_dd(outline[base + i]), (rays[3, i], rays[4, i])),
                        ),
                    )
                end
                b = _dd_mul(_dd(T(2)), _dd_sqrt(_dd_div(spread, m0)))
                if b[1] > 0
                    a = _dd_div(m0, _dd_mul(π2, b))
                    a2, b2 = _dd_mul(a, a), _dd_mul(b, b)
                end
            end
            @inbounds for i in 1:n_cc
                sγ, cγ = (rays[5, i], rays[6, i]), (rays[7, i], rays[8, i])
                L = outline[base + i]
                middle = if a2[1] > 0 && b2[1] > 0
                    _dd_div(
                        _dd_add(
                            _dd_div(_dd_mul(x0, sγ), b2),
                            _dd_div(_dd_mul(sid, cγ), a2),
                        ),
                        _dd_add(_dd_div(_dd_mul(sγ, sγ), b2), _dd_div(_dd_mul(cγ, cγ), a2)),
                    )
                else
                    _dd_mul(sid, cγ)
                end
                entry = _dd_sub(middle, _dd(L / 2))
                strips[1, base + i] = entry[1]
                strips[2, base + i] = entry[2]
                strips[3, base + i] = L
                strips[4, base + i] = L > 0 ? wet[base + i] / L : zero(T)
            end
        end
    end
    return strips
end

# `own[ray, energy, process, own0 + ch]` for the first `n_used` channels of the `n_rays` rays from
# `ray0 + 1` on: each ray's photons by library energy and process (`mix`, `_scatter_corners!`'s from
# `_scatter_weights!`'s), each scaled by how much less of its scatter gets out of the ray's materials
# than of the water of its corners' kernels, mixed by the corners' shares of the ray alone
# (`corner_weight`; BasisMonteCarlo's escape factors): the weights of the ray's own single scatter.
# The ray is the fastest index: `_scatter_single!`'s warps read 32 rays' weights at once.
function _scatter_single_weights!(
    own::AbstractArray{T, 4},
    corner_weight::AbstractMatrix{T},
    mix::AbstractArray{T, 4},
    plain::AbstractVector{Bool},
    n_used::Int,
    own0::Int,
    ray0::Int,
    n_rays::Int,
) where {T}
    nE = size(own, 2)
    let own = own,
        corner_weight = corner_weight,
        mix = mix,
        plain = plain,
        nE = nE,
        n_rays = n_rays,
        own0 = own0,
        ray0 = ray0
        _foreachindex(1:(n_rays * nE * n_used), AK.get_backend(own)) do idx
            q = idx - 1
            local_ray = q % n_rays + 1
            q ÷= n_rays
            g = q % nE + 1
            ch = q ÷ nE + 1
            # a ray of water's kernels hold all their scatter, each process weighed alike
            water = plain[local_ray + ray0]
            @inbounds for p in 1:3
                slot = water ? g : g + (p - 1) * nE
                s = zero(T)
                for c in 1:8
                    w = corner_weight[c, local_ray]
                    w == 0 && continue
                    s += w * mix[slot, ch, c, local_ray]
                end
                own[local_ray, g, p, own0 + ch] = s
            end
        end
    end
    return own
end

# The channels' sums `acc` with one energy `e` of a point added: its Compton and its Rayleigh scatter
# (wide, process 2, or forward, 3: `p`) weighed by the ray's `own[ray, e, process, ch]`. Written out
# channel by channel: a closure (`ntuple(k -> …)`) is not inlined into a GPU kernel, and the call
# passes `acc` through memory once per energy and point.
@generated function _single_add(acc::NTuple{N}, own, e, p, ray, compton, rayleigh) where {N}
    sums = [
        :(
            @inbounds(
                acc[$k] + (own[ray, e, 1, $k] * compton + own[ray, e, p, $k] * rayleigh),
            )
        )
        for k in 1:N
    ]
    return quote
        Base.@_inline_meta
        tuple($(sums...))
    end
end

# `single[ray, row, ch, column]` for the `n_rays` rays from `ray0 + 1` on: each ray's own single
# scatter (`_single_scatter!`) through the patient its view-group's rays describe (`strips`), its
# energies and processes weighed by `weights` (`_scatter_single_weights!`) into the first `N`
# channels, on the library's rows relative to the ray's own and its columns from `window0[i] + 1`
# (its own column and `columns_each_side` either side behind a grid, every column without one; a
# ray at the coarse ray `i`). `t` is the quadrature's tables on the device (`_single_tables`),
# `angle_deg` the coarse rays' angles, `first` + (i − 1) `step`, `rays` their directions
# (`_scatter_strips!`) and `boundaries` the lines between them (`_RayStrips`). A thread for each
# column of a ray, a warp's 32 threads one column of 32 rays: alike, they take the same branches.
function _scatter_single!(
    single::AbstractArray{T, 4},
    weights::AbstractArray{T, 4},
    strips::AbstractMatrix{T},
    angle_deg::AbstractVector{T},
    rays::AbstractMatrix{T},
    boundaries::AbstractMatrix{T},
    window0::AbstractVector{Int32},
    first::T,
    step::T,
    t,
    ray0::Int,
    n_rays::Int,
    ::Val{N},
) where {T, N}
    nr, nc, n_win = size(single, 2), length(t.cols) - 1, size(single, 4)
    nE = size(weights, 2)
    n_cc = length(angle_deg)
    let single = single,
        weights = weights,
        strips = strips,
        angle_deg = angle_deg,
        rays = rays,
        boundaries = boundaries,
        window0 = window0,
        first = first,
        step = step,
        cols = t.cols,
        rows = t.rows,
        column_points = t.column_points,
        energies = t.energies,
        total = t.total,
        mu = t.mu,
        pdf = t.pdf,
        at_energy = t.at_energy,
        efficiency = t.efficiency,
        mud = t.mud,
        norm = t.norm,
        energy_weighted = t.energy_weighted,
        grid = t.grid,
        height = t.grid_height,
        opening = t.opening,
        M = t.depths,
        Q = t.points_per_column,
        Z = t.points_per_row,
        forward = t.forward,
        scatters = t.scatters,
        D = t.sdd,
        D2 = t.sdd2,
        nr = nr,
        nc = nc,
        n_win = n_win,
        nE = nE,
        n_cc = n_cc,
        ray0 = ray0,
        n_rays = n_rays
        _foreachindex(1:(32 * n_win * cld(n_rays, 32)), AK.get_backend(single)) do idx
            q = idx - 1
            lane = q % 32
            q ÷= 32
            w = q % n_win + 1
            local_ray = (q ÷ n_win) * 32 + lane + 1
            local_ray > n_rays && return
            @inbounds for k in 1:N, ri in 1:nr
                single[local_ray, ri, k, w] = zero(T)
            end
            ray = local_ray + ray0
            i = (ray - 1) % n_cc + 1
            ci = w + Int(window0[i])
            (scatters && 1 <= ci <= nc) || return
            medium = _RayStrips(strips, boundaries, ray - i, n_cc, first, step)
            entry, L, ρ = _ray_segment(medium, i)
            (L > 0 && ρ > 0) || return
            # the library energies the ray has photons of
            used = UInt64(0)
            @inbounds for e in 1:nE, k in 1:N, p in 1:3
                weights[local_ray, e, p, k] != 0 && (used |= UInt64(1) << (e - 1))
            end
            used == 0 && return
            tables = (; energies, total, mu, pdf, at_energy, efficiency, mud, norm, energy_weighted)
            cells = size(pdf, 1)
            Δc, Δr = cols[2] - cols[1], rows[2] - rows[1]
            area = D * deg2rad(Δc / Q) * (Δr / Z)
            u = angle_deg[i]
            su2, cu2 = (rays[5, i], rays[6, i]), (rays[7, i], rays[8, i])
            dt = L / M
            @inbounds for m in 1:M
                # the scattering site (a double-float: the way out's decisions are taken from it), and
                # the water-equivalent length the photon crossed to reach it
                depth = _dd_div(_two_prod(T(m) - T(0.5), L), _dd(T(M)))
                site = _dd_add(entry, depth)
                P = (_dd_mul(site, su2), _dd_mul(site, cu2))
                τ = ρ * depth[1]
                for qc in 1:Q
                    γ = cols[ci] + (qc - T(0.5)) / Q * Δc
                    ℓ, along, turn, g_fan = _single_column(
                        site[1],
                        deg2rad(u - γ),
                        D,
                        grid,
                        height,
                        opening[1],
                    )
                    g_fan > 0 || continue
                    sγ = (column_points[1, qc, ci], column_points[2, qc, ci])
                    cγ = (column_points[3, qc, ci], column_points[4, qc, ci])
                    exit = _exit_wet(medium, P, _dd_way(P, sγ, cγ, D2))
                    sΔ = _dd_sub(_dd_mul(su2, cγ), _dd_mul(cu2, sγ))
                    cΔ = _dd_add(_dd_mul(cu2, cγ), _dd_mul(su2, sγ))
                    for ri in 1:nr
                        acc = ntuple(_ -> zero(T), Val(N))
                        for qz in 1:Z
                            z = rows[ri] + (qz - T(0.5)) / Z * Δr
                            wp, v, obliquity, lengthen = _single_point(
                                ℓ,
                                along,
                                turn,
                                z,
                                D,
                                area,
                                g_fan,
                                grid,
                                height,
                                opening[2],
                            )
                            wp > 0 || continue
                            j, p = _single_cell(v, cells, forward, site, sΔ, cΔ, D2, z)
                            s = ρ * dt * wp
                            path = exit * lengthen
                            for e in 1:nE
                                (used >> (e - 1)) & 1 == 0 && continue
                                compton, rayleigh = _single_energy(
                                    tables,
                                    e,
                                    j,
                                    v,
                                    τ,
                                    path,
                                    obliquity,
                                )
                                acc = _single_add(
                                    acc,
                                    weights,
                                    e,
                                    p,
                                    local_ray,
                                    s * compton,
                                    s * rayleigh,
                                )
                            end
                        end
                        for k in 1:N
                            single[local_ray, ri, k, w] += acc[k]
                        end
                    end
                end
            end
        end
    end
    return single
end

# `rows[row, column, ch, group, view]`: the scatter of a view-group's rays relative to their own
# row, each library bin gathering the corners' boxes over its column and the rays' own single
# scatter (`single`, `_scatter_single!`; ray `i` holds the columns from `window0[i] + 1`), for the
# `n_gv` view-groups of the chunk whose rays start after `ray0` (`shapes`, `plain`: every ray's; the
# rest the chunk's).
function _scatter_deposit!(
    rows::AbstractArray{T, 5},
    corner::AbstractMatrix{Int32},
    factor::AbstractArray{T, 3},
    mix::AbstractArray{T, 4},
    boxes::AbstractArray{T, 5},
    window::AbstractArray{Int32, 4},
    single::AbstractArray{T, 4},
    single0::Int,
    window0::AbstractVector{Int32},
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
    let rows = rows,
        corner = corner,
        factor = factor,
        mix = mix,
        boxes = boxes,
        window = window,
        single = single,
        single0 = Int32(single0),
        window0 = window0,
        n_win = Int32(size(single, 4)),
        shapes = shapes,
        plain = plain,
        kernels = lib.kernels,
        w_col = lib.column_width,
        nr = Int32(nr),
        nc = Int32(nc),
        n_ch = Int32(n_ch),
        nE = Int32(nE),
        L = Int32(L),
        block = Int32(block),
        n_cc = Int32(n_cc),
        n_used = Int32(n_used),
        lanes = Int32(lanes),
        ray0 = Int32(ray0)
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
                    # the ray's own single scatter, where it lands
                    w = ci - window0[i]
                    if Int32(1) <= w <= n_win
                        acc, comp = _kahan(acc, comp, single[ray, dr, single0 + ch, w])
                    end
                    mirror = shapes[4, ray + ray0] < 0
                    slot0, n_slots = plain[ray + ray0] ?
                        (Int32(3) * nE, nE) :
                        (Int32(0), Int32(3) * nE)
                    for c in Int32(1):Int32(8)
                        f = factor[ch, c, ray]
                        f == 0 && continue
                        base = corner[c, ray] * block + dr + slot0 * L
                        for ii in max(Int32(1), ci - window[2, ch, c, ray]):min(
                                    nc,
                                    ci - window[1, ch, c, ray],
                                )
                            # the box from the left edge of this bin's column, and its share of it
                            off = w_col * (boxes[1, ii, ch, c, ray] - T(ci))
                            lo, hi = boxes[2, ii, ch, c, ray], boxes[3, ii, ch, c, ray]
                            l, h = off + lo, off + hi
                            share = hi > lo ?
                                (min(h, w_col) - max(l, zero(T))) / (hi - lo) :
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
    let out = out,
        rows = rows,
        row_share = row_share,
        nc = nc,
        nr = nr,
        n_ch = n_ch,
        n_g = n_g,
        mid = (nr + 1) ÷ 2,
        combine = combine,
        v0 = v0
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
