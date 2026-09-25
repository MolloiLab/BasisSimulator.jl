# api_sections.jl — the section map of the generated API reference (docs/src/routes/api/index.jl).
#
# Pure data plus one resolver function, with no package dependencies. The docs route and the
# package test suite (test/docs.jl) both `include` this file, so the page and the test always
# agree on where every exported name goes.
#
# Each exported name is resolved to a section in three steps, first match wins:
#   1. `API_NAME_SECTION`  — explicit per-name overrides (e.g. the five structs, which live in
#                            different files but read best together);
#   2. `API_PREFIX_SECTION` — name-prefix rules (e.g. every `REGION_*` label);
#   3. `API_FILE_SECTION`  — the path, relative to `src/`, of the file holding the name's
#                            docstring: an exact file, or else the longest matching folder prefix.
# A name that matches none of those lands in the "Other" section. It is never dropped, but
# test/docs.jl fails, so a new source file has to be mapped here deliberately.

"""Ordered sections of the API page: `(id, title, intro)`. The intro is Markdown."""
const API_SECTIONS = [
    (id = "overview", title = "Overview", intro = """
Every simulation walks the same pipeline:

```text
Phantom             ─┐
EICT/PCCTScanner    ─┤
CTProtocol          ─┼─▶ create_*_workspace → simulate! → reconstruct! → to_hounsfield → recon (HU)
SimOptions          ─┤
ReconOptions        ─┘
```

All five stages are plain Julia structs. The backend (CPU or GPU) is chosen by where the phantom
mask lives: wrap it in a `CuArray`, `MtlArray` or `ROCArray` for a GPU, or leave it a plain `Array`
for the CPU. The same code path runs everywhere.

Everything below is generated from the package's docstrings when the site is built, so it cannot
drift from the code. Each entry links to its source on GitHub. The worked examples show these
functions inside real pipelines.
"""),
    (id = "five-structs", title = "The five-struct API", intro = """
These are the load-bearing types. Every public entry point takes some combination of them. Four
are concrete structs with named fields. The scanner is the one abstraction: an abstract `Scanner`
with one concrete family per detector type (`EICTScanner` for scintillators, `PCCTScanner` for
photon counting). That split means a photon-counting parameter can't be set on a scintillator,
and a scintillator parameter can't be set on a photon-counting detector.
"""),
    (id = "phantom", title = "Phantom & materials", intro = """
Built-in phantoms (Gammex 472, the XCIST voxelized XCAT chests), phantoms built from any labeled
mask, material lookup, and the attenuation / HU helpers. Materials are `XrayAttenuation.Material`s,
re-exported as `XA`.
"""),
    (id = "geometry", title = "Geometry & coordinate mapping", intro = """
`CTGeometry` pre-computes the source and detector trajectory of every view, axial or helical,
for arc or flat detectors. The affine helpers map between phantom voxels, world coordinates and
the reconstruction grid.
"""),
    (id = "spectrum", title = "Spectrum & source", intro = """
The X-ray spectrum is resolved once per simulation from the scanner's flat filter, the protocol's
additional filters and, optionally, the bowtie. These helpers are usually called inside
`simulate!`. They're public because beam-hardening correction and μ_water calibration need the
same spectrum the simulator uses. Focal spot, heel effect and protocol helpers also live here.
"""),
    (id = "forward", title = "Forward projection", intro = """
Allocate a workspace once and reuse it across `simulate!` calls. After JIT warm-up the hot path
doesn't allocate. Below the workspace API sit the ray-driven (Siddon) and distance-driven
(DD, `dd_fast`) projectors, the cached per-material path lengths, and the memory-budget helpers.
"""),
    (id = "dose", title = "Dose & CTDI", intro = """
CTDI is measured the way IEC 60601-2-44 defines it. The beam is transported by Monte Carlo
through a PMMA cylinder (32 cm body or 16 cm head), and the air kerma is integrated over a
100 mm pencil chamber at the centre and at the four peripheral holes. Because it's simulated
rather than looked up, it follows the spectrum, filtration and bowtie of the scanner being
simulated.
"""),
    (id = "reconstruction", title = "Reconstruction", intro = """
Analytic FDK for a circular orbit, weighted FBP for a helical one, and penalized iterative HIR
(Hybrid IR). The volume reconstructors share a single `reconstruct!` entry point, which
dispatches on the workspace type. The filters, back-projectors and iterative-reconstruction
utilities are documented alongside.
"""),
    (id = "vmi", title = "VMI & spectral decomposition", intro = """
Virtual monoenergetic imaging from any set of spectral channels (dual-kVp, dual-source or
photon-counting bins). `vmi_pipeline` is the whole chain: an optional `SpectralHYPR` denoiser,
channel merging, K-channel (`:nchannel`) or Cong (`:cong`) projection-domain decomposition, basis
reconstruction, and synthesis of the VMI stack. Each stage is also public on its own.
"""),
    (id = "denoising", title = "Denoising", intro = """
Projection-domain and image-domain noise reduction. This includes generalized HYPR-LR
(`SpectralHYPR`, `hypr_lr`, `image_hypr`), ACNR, the RSKR joint bilateral filter, T-LBF,
the sinogram SVD / SF-JSD denoisers and the median-z filter.
"""),
    (id = "corrections", title = "Corrections & calibration", intro = """
Beam-hardening correction (water and two-material, in the sinogram or the image domain), radial
cupping correction, PCCT pile-up correction, and the calibration helpers.
"""),
    (id = "detector", title = "Detector physics", intro = """
Both detector families use Monte Carlo–derived response models. The energy-integrating (EICT)
path uses per-energy efficiency lookup tables, fill factor, optical crosstalk and lag. The
photon-counting (PCCT) path uses the full detector response matrix and the pile-up model.
Scatter estimation and injection are also here.
"""),
    (id = "constants", title = "Constants & labels", intro = """
The semantic region labels that the built-in phantom masks store (air, water, the Gammex 472
calcium and iodine inserts). Each is a `RegionLabel` value.
"""),
    (id = "other", title = "Other", intro = """
Exported names whose source file isn't mapped to a section yet. Map them in
`docs/src/api_sections.jl`. The test suite fails while this section has entries.
"""),
]

"""Names shown first in their section, in this order (the rest follow by source file and line)."""
const API_LEADS = [
    "Phantom", "Scanner", "EICTScanner", "PCCTScanner", "ScannerGeometry", "CTProtocol", "SimOptions", "ReconOptions",
    "create_eict_workspace", "create_workspace", "simulate!",
    "reconstruct!", "create_fdk_recon_workspace", "create_hir_recon_workspace",
    "vmi_pipeline", "spectral_basis", "SpectralHYPR",
    "compute_dose", "DoseReport",
    "create_gammex_472", "load_xcat_phantom", "download_xcat_phantom",
    "CTGeometry",
]

"""Per-name overrides, checked first."""
const API_NAME_SECTION = Dict{String, String}(
    # The five structs — spread across files, read together.
    "Phantom"         => "five-structs",
    "Scanner"         => "five-structs",
    "EICTScanner"     => "five-structs",
    "PCCTScanner"     => "five-structs",
    "ScannerGeometry" => "five-structs",
    "CTProtocol"      => "five-structs",
    "SimOptions"      => "five-structs",
    "ReconOptions"    => "five-structs",
    # The reconstruction side of the workspace API lives in src/api/ with the simulation side.
    "reconstruct!"                => "reconstruction",
    "create_fdk_recon_workspace"  => "reconstruction",
    "create_hir_recon_workspace"  => "reconstruction",
    "FDKReconWorkspace"           => "reconstruction",
    "HIRReconWorkspace"           => "reconstruction",
    # Spectrum resolution and PCCT BHC calibration live in src/api/ but belong with their topic.
    "apply_bowtie_to_spectrum"               => "spectrum",
    "resolve_source_spectrum_full"           => "spectrum",
    "resolve_source_spectrum_with_bowtie"    => "spectrum",
    "resolve_source_spectrum_without_bowtie" => "spectrum",
    "calibrate_pcct_poly_bhc"                => "corrections",
    # Re-exported dependency.
    "XA" => "phantom",
)

"""Name-prefix rules, checked second (longest prefix wins)."""
const API_PREFIX_SECTION = Dict{String, String}(
    "REGION_" => "constants",
    "RegionLabel" => "constants",
)

"""Source file or folder (relative to `src/`) → section, checked last (longest match wins)."""
const API_FILE_SECTION = Dict{String, String}(
    "api/"                             => "forward",
    "object/"                          => "phantom",
    "phantoms/"                        => "phantom",
    "geometry/"                        => "geometry",
    "source/"                          => "spectrum",
    "source/dose.jl"                   => "dose",
    "projection/"                      => "forward",
    "reconstruction/workspace/"        => "forward",
    "reconstruction/core/"             => "reconstruction",
    "reconstruction/fbp/"              => "reconstruction",
    "reconstruction/hybrid_ir/"        => "reconstruction",
    "reconstruction/ir/"               => "reconstruction",
    "reconstruction/vmi/"              => "vmi",
    "denoising/"                       => "denoising",
    "correction/"                      => "corrections",
    "detector/"                        => "detector",
)

"""
    api_section(name::AbstractString, src_path::Union{Nothing,AbstractString}) -> String

Section id for exported `name`, whose docstring lives in `src_path` (relative to `src/`, `/`
separators; `nothing` when the name has no docstring). Returns `"other"` when nothing matches.
"""
function api_section(name::AbstractString, src_path::Union{Nothing, AbstractString})
    haskey(API_NAME_SECTION, name) && return API_NAME_SECTION[name]
    best = ""
    for p in keys(API_PREFIX_SECTION)
        startswith(name, p) && length(p) > length(best) && (best = p)
    end
    isempty(best) || return API_PREFIX_SECTION[best]
    src_path === nothing && return "other"
    haskey(API_FILE_SECTION, src_path) && return API_FILE_SECTION[src_path]
    for p in keys(API_FILE_SECTION)
        endswith(p, "/") && startswith(src_path, p) && length(p) > length(best) && (best = p)
    end
    return isempty(best) ? "other" : API_FILE_SECTION[best]
end

"""
    api_doc_location(docstr_data) -> (path, line)

`(path relative to src/, line)` of a `Base.Docs.DocStr`'s `data` dict, or `(nothing, 0)`.
"""
function api_doc_location(data)
    p = get(data, :path, nothing)
    p === nothing && return (nothing, 0)
    s = replace(string(p), '\\' => '/')
    i = findlast("/src/", s)
    rel = i === nothing ? basename(s) : s[last(i)+1:end]
    return (rel, Int(get(data, :linenumber, 0)))
end
