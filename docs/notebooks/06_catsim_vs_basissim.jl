### A Pluto.jl notebook ###
# v0.3.0

using Markdown
using InteractiveUtils

# ╔═╡ 06000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 06000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 06000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 06000001-0000-4000-8000-000000000004
using Printf: @sprintf

# ╔═╡ 06000001-0000-4000-8000-000000000010
md"""
# CatSim vs BasisSimulator: Agreement and Runtime

**One scanner, one protocol, one Gammex 472 phantom, simulated and reconstructed three ways:
XCIST/CatSim (GE's open-source reference simulator, Python + C), BasisSimulator on the CPU, and
BasisSimulator on the GPU.**

This is a cross-validation of the forward model and the reconstruction against an independent
implementation. Both simulators are configured to the same beam and the same physics, both apply
a water beam-hardening correction referenced to the same monoenergetic water μ, and both
reconstruct with FDK and a `standard` kernel. The comparison therefore reads in HU: the
per-rod table below puts CatSim, BasisSimulator and the XrayAttenuation theory side by side. The
runtime of each pipeline comes after.

Noise is off in all three runs. The two simulators calibrate tube flux independently, so a noisy
comparison would mix a flux-model difference into what should be a projector and reconstruction
check.

```
Gammex 472 @ 128² × 8 (2.7 mm voxels)
   → GE Apex Elite geometry · 120 kVp / 200 mA / 500 views · 7 mm Al · large-body bowtie
   →┬→ CatSim (Python):  forward project → water BHC → FDK
    ├→ BasisSim CPU:     simulate! → water BHC → reconstruct!
    └→ BasisSim GPU:     simulate! → water BHC → reconstruct!   (same code, device mask)
   → images · differences · per-rod HU · runtime
```
"""

# ╔═╡ 06000001-0000-4000-8000-000000000020
md"""
## Notebook setup

This is the only docs notebook that uses Python. `docs/CondaPkg.toml` pins `gecatsim` to the
MolloiLab fork of XCIST (`git+https://github.com/MolloiLab/main`), which adds the Gammex 472
material definitions, together with numpy and pydicom. The first `import PythonCall` in the
`docs/` environment makes CondaPkg install Python and those packages into `docs/.CondaPkg/`
(a few minutes and about 1 GB, once). If PythonCall or gecatsim cannot be loaded, the CatSim
cells skip with a notice and the BasisSimulator CPU and GPU runs still execute.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 06000001-0000-4000-8000-000000000031
# ╠═╡ show_logs = false
import CairoMakie as Mke

# ╔═╡ 06000001-0000-4000-8000-000000000032
# ╠═╡ show_logs = false
# PythonCall, loaded so that a missing Python environment degrades to "CatSim skipped"
PC = try
    Base.require(Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall"))
catch err
    @warn "PythonCall could not be loaded; the CatSim cells will skip" exception = (err, catch_backtrace())
    nothing
end;

# ╔═╡ 06000001-0000-4000-8000-000000000033
import PlutoUI

# ╔═╡ 06000001-0000-4000-8000-000000000034
PlutoUI.TableOfContents()

# ╔═╡ 06000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()   # CuArray / MtlArray / ROCArray / oneArray, or Array on a CPU-only host
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 06000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 06000001-0000-4000-8000-000000000060
const HAS_GECATSIM = PC !== nothing && try
    PC.pyimport("gecatsim")
    true
catch err
    @warn "gecatsim is not importable; the CatSim cells will skip" exception = err
    false
end;

# ╔═╡ 06000001-0000-4000-8000-000000000070
HAS_GECATSIM ? Markdown.parse("""
    **gecatsim** $(PC.pyconvert(String, PC.pyimport("importlib.metadata").version("gecatsim"))) loaded
    (Python $(PC.pyconvert(String, PC.pyimport("platform").python_version()))): the CatSim cells run.
    """) : md"""
    !!! warning "gecatsim not available: CatSim cells skipped"
        `import PythonCall` or `pyimport("gecatsim")` failed. Instantiate `docs/` so that CondaPkg
        can install the Python environment (see `docs/CondaPkg.toml`), then re-run. The
        BasisSimulator CPU and GPU runs below still execute.
    """

# ╔═╡ 06000001-0000-4000-8000-000000000080
# ╠═╡ show_logs = false
# Two import stubs for the MolloiLab gecatsim fork.
#
# `gecatsim.reconstruction.pyfiles.recon` imports the iterative reconstructions (ART, SIRT, CGLS)
# at module load, and those import `gecatsim.pyfiles.C_DD3Back` / `C_DD3WBack`, which the fork
# does not ship (only `C_DD3Proj*.py`). Without them the FDK reconstruction cannot even be
# imported. We write a stub of each missing module into the installed package: its function
# raises if called, which FDK never does. Idempotent: re-running is a no-op once they exist.
gecatsim_patched = !HAS_GECATSIM ? false : let
    spec = PC.pyimport("importlib.util").find_spec("gecatsim")
    pyfiles_dir = joinpath(dirname(PC.pyconvert(String, spec.origin)), "pyfiles")
    wrote = String[]
    for name in ("C_DD3Back", "C_DD3WBack")
        path = joinpath(pyfiles_dir, "$(name).py")
        isfile(path) && continue
        write(path, """
            # Stub written by BasisSimulator.jl docs notebook 06: the fork does not ship this module.
            def $(replace(name, "C_" => ""))(*args, **kwargs):
                raise NotImplementedError("$(name) is a stub; FDK does not need it, iterative recons do.")
            """)
        push!(wrote, name)
    end
    @info "[gecatsim] import stubs: $(isempty(wrote) ? "already present" : "wrote " * join(wrote, ", "))"
    true
end;

# ╔═╡ 06000001-0000-4000-8000-000000000090
# Two runtime patches for CatSim inside Pluto.
#
# 1. Speed: `fdk_equiAngle.float3Darray2pointer` / `float3Dpointer2array` copy a numpy volume to
#    and from a C triple pointer element by element in Python — millions of ctypes assignments
#    before and after the C FDK. The replacements alias one row pointer per (i, j) into the numpy
#    buffer and copy back with one `memmove` per row: the same C ABI and the same numbers.
# 2. Output: `run_all()` and `recon()` print hundreds of lines. Under PythonCall in a Pluto worker
#    that flood can block the captured stdout pipe, so both are wrapped in
#    `contextlib.redirect_stdout`.
gecatsim_fdk_patched = !HAS_GECATSIM ? false : let
    PC.pyexec(
        """
        import ctypes, contextlib, io
        import numpy as np
        import gecatsim as _gecatsim
        import gecatsim.reconstruction.pyfiles.fdk_equiAngle as _fdk
        import gecatsim.reconstruction.pyfiles.recon as _recon_mod

        FLOAT = ctypes.c_float
        PtrFLOAT = ctypes.POINTER(FLOAT)
        PtrPtrFLOAT = ctypes.POINTER(PtrFLOAT)

        def _fast_arr2ptr(arr):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            n0, n1, _ = arr.shape
            out = (PtrPtrFLOAT * n0)()
            for i in range(n0):
                row = (PtrFLOAT * n1)()
                for j in range(n1):
                    row[j] = arr[i, j].ctypes.data_as(PtrFLOAT)
                out[i] = row
            out._keepalive_ = arr          # the row pointers alias arr's buffer
            return out

        def _fast_ptr2arr(ptr, n, m, o):
            out = np.empty((n, m, o), dtype=np.float32)
            nbytes = o * ctypes.sizeof(FLOAT)
            for i in range(n):
                for j in range(m):
                    ctypes.memmove(out[i, j].ctypes.data_as(PtrFLOAT), ptr[i][j], nbytes)
            return out

        _fdk.float3Darray2pointer = _fast_arr2ptr
        _fdk.float3Dpointer2array = _fast_ptr2arr

        if not getattr(_gecatsim.CatSim, '_basissim_quiet', False):
            _orig_run_all = _gecatsim.CatSim.run_all
            def _quiet_run_all(self, *args, **kwargs):
                with contextlib.redirect_stdout(io.StringIO()):
                    return _orig_run_all(self, *args, **kwargs)
            _gecatsim.CatSim.run_all = _quiet_run_all
            _gecatsim.CatSim._basissim_quiet = True

        if not getattr(_recon_mod, '_basissim_quiet', False):
            _orig_recon = _recon_mod.recon
            def _quiet_recon(ct, *args, **kwargs):
                with contextlib.redirect_stdout(io.StringIO()):
                    return _orig_recon(ct, *args, **kwargs)
            _recon_mod.recon = _quiet_recon
            _recon_mod._basissim_quiet = True
        """,
        Main,
    )
    true
end;

# ╔═╡ 060000f1-0000-4000-8000-000000000001
md"""
## Scan and phantom set-up

One scanner, one protocol and one phantom, shared verbatim by the three pipelines, and a thin
wrapper that hands them to CatSim.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
md"""
### 1. Scanner

The GE Revolution Apex Elite geometry of notebook 01, with its large-body bowtie. BasisSimulator's
`:ge_revolution_large` bowtie is read from the same `large.txt` table CatSim ships, so both
simulators see the same bowtie. Electronic noise is zero, as noise is off everywhere here.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000010
scanner = BS.EICTScanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 0,
    detection_gain = 10.0,
);

# ╔═╡ 06000003-0000-4000-8000-000000000001
md"""
### 2. Protocol, physics and grid

120 kVp / 200 mA, 500 views in a 1 s rotation, 4 mm of collimation, 4.5 mm of extra aluminium.
The physics is set to what both simulators model by default: polychromatic transport through
the filters and bowtie, energy-dependent detector efficiency, fill factor and focal-spot blur.
Noise, scatter, detector lag and the heel effect are switched off in BasisSimulator because the
CatSim defaults have them off (CatSim leaves its scatter, lag and crosstalk callbacks empty). The
reconstruction is 256 × 256 over 35 cm, 0.625 mm slices.

!!! info "Point views on both sides"
    The other notebooks model a clinical scanner's detector integrating each view while the
    gantry turns (`SimOptions(; view_samples = 5)`, notebook 01). This one does not: it is a
    like-for-like check of the projector and the reconstruction, so both simulators sample each
    view at a single angle. BasisSimulator keeps its default `view_samples = 1`, and the wrapper
    sets CatSim's `physics.viewSampleCount` (2 by default) to 1.
"""

# ╔═╡ 06000003-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 200.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 4.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 06000003-0000-4000-8000-000000000020
sim_opts = BS.SimOptions(
    seed = 1234, projector = :dd_fast,
    use_noise = false, use_scatter = false, use_lag = false, use_heel_effect = false,
);

# ╔═╡ 06000003-0000-4000-8000-000000000030
recon_opts = BS.ReconOptions(
    matrix_size = (256, 256, round(Int, protocol.collimation_mm / 0.625)),
    fov_cm = 35.0,
    z_cm = protocol.collimation_mm / 10,
);

# ╔═╡ 06000004-0000-4000-8000-000000000001
md"""
### 3. The CatSim wrapper

A few Julia functions map `EICTScanner` / `CTProtocol` / `ReconOptions` / `Phantom` onto
CatSim's configuration:

- **Geometry.** BasisSimulator gives the detector pitch at the isocentre; CatSim wants it at the
  detector face, so it is multiplied by `SDD/SID`. Every column is its own module with no gap
  (`detectorColsPerMod = 1`, `detectorColSkip = 0`); without that CatSim braids or squashes the
  sinogram.
- **Beam.** CatSim's 120 kVp tungsten spectrum for a 10° target (the protocol's anode angle),
  the scanner's flat filter plus the protocol's filters, the same `large.txt` bowtie, the same
  target angle and focal spot, and no graphite detector prefilter (BasisSimulator models none).
- **Views.** One angular sample per view (`physics.viewSampleCount = 1`), the point views of
  BasisSimulator's `view_samples = 1`.
- **Preprocessing.** CatSim's own water BHC, `Prep_BHC_Accurate`: a degree-5 polynomial per
  detector cell fitted to air scans through 1–50 cm of water, mapped to the same monoenergetic
  water μ that BasisSimulator's `calibrate_bhc_water` uses. HU conversion uses that water μ too.
"""

# ╔═╡ 06000004-0000-4000-8000-000000000010
const _catsim_state = Dict{Symbol, Any}();

# ╔═╡ 06000004-0000-4000-8000-000000000020
function catsim_init()
    # Reference both patch flags so Pluto runs the stubs and the FDK patch first.
    (gecatsim_patched && gecatsim_fdk_patched) || error("the gecatsim patches did not run")
    if isempty(_catsim_state)
        _catsim_state[:xc] = PC.pyimport("gecatsim")
        _catsim_state[:recon] = PC.pyimport("gecatsim.reconstruction.pyfiles.recon")
        origin = PC.pyconvert(String, PC.pyimport("importlib.util").find_spec("gecatsim").origin)
        _catsim_state[:cfg] = joinpath(dirname(origin), "examples", "cfg")
    end
    return _catsim_state[:xc], _catsim_state[:recon], _catsim_state[:cfg]
end;

# ╔═╡ 06000004-0000-4000-8000-000000000030
function catsim_create_simulation()
    xc, _, cfg = catsim_init()
    return xc.CatSim(
        joinpath(cfg, "Phantom_Sample.cfg"),
        joinpath(cfg, "Scanner_Sample_generic.cfg"),
        joinpath(cfg, "Protocol_Sample_axial.cfg"),
    )
end;

# ╔═╡ 06000004-0000-4000-8000-000000000040
function catsim_configure_scanner!(ct, scanner, protocol)
    magnification = scanner.source_to_detector / scanner.source_to_isocenter
    n_rows = protocol.collimation_mm === nothing ? scanner.detector_rows :
        round(Int, protocol.collimation_mm / scanner.detector_row_size)

    ct.scanner.sid = scanner.source_to_isocenter
    ct.scanner.sdd = scanner.source_to_detector
    ct.scanner.detectorColCount = scanner.detector_cols
    ct.scanner.detectorRowCount = n_rows
    ct.scanner.detectorColSize = scanner.detector_col_size * magnification   # isocentre → face
    ct.scanner.detectorRowSize = scanner.detector_row_size * magnification
    ct.scanner.detectorColOffset = scanner.detector_col_offset
    ct.scanner.detectorColsPerMod = 1          # every column its own module …
    ct.scanner.detectorRowsPerMod = n_rows
    ct.scanner.detectorColSkip = 0.0           # … with no inter-module gap
    ct.scanner.detectorRowSkip = 0.0

    ct.scanner.targetAngle = scanner.target_angle
    ct.scanner.focalspotWidth = scanner.focal_spot_width
    ct.scanner.focalspotLength = scanner.focal_spot_length
    ct.scanner.detectorMaterial = "Lumex"
    ct.scanner.detectorDepth = scanner.detector_depth
    ct.scanner.detectorColFillFraction = scanner.fill_factor_col
    ct.scanner.detectorRowFillFraction = scanner.fill_factor_row
    ct.scanner.detectorPrefilter = PC.pylist([])
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000050
function catsim_configure_protocol!(ct, scanner, protocol; μ_water_cm)
    ct.protocol.mA = protocol.mA
    ct.protocol.viewsPerRotation = protocol.views
    ct.protocol.viewCount = protocol.views
    ct.protocol.stopViewId = protocol.views - 1
    ct.protocol.rotationTime = protocol.rotation_time
    ct.protocol.spectrumFilename = "tungsten_tar$(protocol.anode_angle).0_$(Int(protocol.kVp))_filt.dat"
    scanner.bowtie_filter in (:ge_revolution_large, :large_body) ||
        error("this wrapper maps only the large-body bowtie")
    ct.protocol.bowtie = "large.txt"
    catsim_name(m) = Dict("aluminum" => "Al", "Al" => "Al", "copper" => "Cu", "Cu" => "Cu")[string(m)]
    filters = Any[catsim_name(scanner.flat_filter_material), scanner.flat_filter_thickness]
    for (m, t) in protocol.additional_filters
        push!(filters, catsim_name(m), t)
    end
    ct.protocol.flatFilter = PC.pylist(filters)

    ct.physics.viewSampleCount = 1      # point views, like BasisSimulator's view_samples = 1
    ct.physics.enableQuantumNoise = 0
    ct.physics.enableElectronicNoise = 0
    ct.physics.callback_post_log = "Prep_BHC_Accurate"   # CatSim's water BHC
    ct.physics.EffectiveMu = μ_water_cm                   # cm⁻¹, the mono target of the fit
    ct.physics.BHC_poly_order = 5
    ct.physics.BHC_max_length_mm = 500
    ct.physics.BHC_length_step_mm = 10
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000060
function catsim_configure_recon!(ct, recon_opts; μ_water_cm)
    xc, _, cfg = catsim_init()
    xc.source_cfg(joinpath(cfg, "Recon_Sample_2d.cfg"), ct)
    n_slices = recon_opts.matrix_size[3]
    ct.recon.fov = recon_opts.fov_cm * 10.0
    ct.recon.imageSize = recon_opts.matrix_size[1]
    ct.recon.sliceCount = n_slices
    ct.recon.sliceThickness = recon_opts.z_cm * 10.0 / n_slices
    ct.recon.reconType = "fdk_equiAngle"
    ct.recon.kernelType = "standard"
    ct.recon.unit = "HU"
    ct.recon.mu = μ_water_cm / 10.0           # cm⁻¹ → mm⁻¹
    ct.recon.huOffset = -1000
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000070
function catsim_configure_phantom!(ct, json_path)
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.filename = json_path
    ct.phantom.scale = 1.0
    ct.phantom.centerOffset = PC.pylist([0.0, 0.0, 0.0])
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000080
function catsim_forward_project(ct; results_name)
    ct.resultsName = results_name
    ct.run_all()
    return nothing
end;

# ╔═╡ 06000004-0000-4000-8000-000000000090
function catsim_reconstruct_fdk(ct; results_name)
    _, recon_mod, _ = catsim_init()
    ct.resultsName = results_name
    ct.recon.filename = results_name
    ct.do_Recon = 1
    recon_mod.recon(ct)
    n = Int(PC.pyconvert(Float64, ct.recon.imageSize))
    nz = Int(PC.pyconvert(Float64, ct.recon.sliceCount))
    file = "$(results_name)_$(n)x$(n)x$(nz).raw"
    isfile(file) || error("CatSim wrote no reconstruction")
    return copy(reshape(reinterpret(Float32, read(file)), (n, n, nz)))
end;

# ╔═╡ 06000005-0000-4000-8000-000000000001
md"""
#### Phantom → CatSim voxelized JSON

`create_gammex_472` labels the solid-water body 3, the calcium rods 10–16 and the iodine rods
20–26; the MolloiLab fork defines the matching `Gammex472_*` materials. CatSim's
`Phantom_Voxelized` reads a JSON header plus one Float32 volume-fraction map per material; for a
hard-segmented mask each map is the indicator `mask .== label`.
"""

# ╔═╡ 06000005-0000-4000-8000-000000000010
const REGION_TO_CATSIM = Dict{Int, String}(
    1 => "water", 2 => "water", 3 => "water",     # air region label 1 is empty in this phantom
    10 => "Gammex472_Ca_50", 11 => "Gammex472_Ca_100", 12 => "Gammex472_Ca_200",
    13 => "Gammex472_Ca_300", 14 => "Gammex472_Ca_400", 15 => "Gammex472_Ca_500",
    16 => "Gammex472_Ca_600",
    20 => "Gammex472_I_2_0", 21 => "Gammex472_I_2_5", 22 => "Gammex472_I_5_0",
    23 => "Gammex472_I_7_5", 24 => "Gammex472_I_10_0", 25 => "Gammex472_I_15_0",
    26 => "Gammex472_I_20_0",
);

# ╔═╡ 06000005-0000-4000-8000-000000000030
function export_phantom_for_catsim(phantom, output_dir, name)
    mask = Array(phantom.mask)
    nx, ny, nz = size(mask)
    vx, vy, vz = phantom.voxel_size .* 10.0          # cm → mm
    mkpath(output_dir)
    entries = NamedTuple[]
    for lbl in sort(unique(mask))
        haskey(REGION_TO_CATSIM, Int(lbl)) || continue
        fname = "$(name)_mat$(Int(lbl)).density_"
        write(joinpath(output_dir, fname), Float32.(mask .== lbl))
        push!(entries, (mat = REGION_TO_CATSIM[Int(lbl)], file = fname))
    end
    n = length(entries)
    list(x) = "[" * join(x, ", ") * "]"
    strs(x) = "[" * join(("\"$v\"" for v in x), ", ") * "]"
    json = """
    {
      "n_materials": $(n),
      "mat_name": $(strs(e.mat for e in entries)),
      "volumefractionmap_filename": $(strs(e.file for e in entries)),
      "volumefractionmap_datatype": $(strs(fill("float", n))),
      "cols": $(list(fill(nx, n))), "rows": $(list(fill(ny, n))), "slices": $(list(fill(nz, n))),
      "x_size": $(list(fill(vx, n))), "y_size": $(list(fill(vy, n))), "z_size": $(list(fill(vz, n))),
      "x_offset": $(list(fill((nx + 1) / 2, n))), "y_offset": $(list(fill((ny + 1) / 2, n))),
      "z_offset": $(list(fill((nz + 1) / 2, n))),
      "density_scale": $(list(fill(1.0, n)))
    }
    """
    path = joinpath(output_dir, "$(name).json")
    write(path, json)
    return path
end;

# ╔═╡ 06000006-0000-4000-8000-000000000001
md"""
### 4. The Gammex 472 phantom

`n_voxels = 128` gives 2.7 mm voxels over 35 cm: coarse enough that CatSim's voxelized projector
finishes in minutes, fine enough to resolve the 28 mm rods. Eight slices cover the 4 mm beam.
"""

# ╔═╡ 06000006-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 128,
    n_slices = 8,
    fov_cm = 35.0,
    z_cm = protocol.collimation_mm / 10.0,
);

# ╔═╡ 06000006-0000-4000-8000-000000000020
phantom_gpu = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 06000007-0000-4000-8000-000000000001
md"""
### 5. The shared water reference

`calibrate_bhc_water` resolves BasisSimulator's detected spectrum per detector column and returns
the per-column correction with its monoenergetic reference (the spectrum's mean energy). That
reference water μ is given to CatSim as the target of its own BHC fit and used for the HU
conversion of all three reconstructions.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000010
geom_inspect = BS.CTGeometry(
    scanner;
    n_angles = protocol.views,
    fov_cm = recon_opts.fov_cm,
    z_cm = recon_opts.z_cm,
    collimation_mm = protocol.collimation_mm,
);

# ╔═╡ 06000007-0000-4000-8000-000000000020
bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = geom_inspect);

# ╔═╡ 06000007-0000-4000-8000-000000000030
Markdown.parse("""
**Water reference:** $(round(bhc.reference_energy_keV; digits = 1)) keV, water μ = $(round(bhc.μ_water_ref; digits = 5)) cm⁻¹
""")

# ╔═╡ 060000f2-0000-4000-8000-000000000001
md"""
## Run the three pipelines

Each timing covers the whole job from the phantom to an HU volume: forward projection,
preprocessing and BHC, FDK. Both BasisSimulator runs are compiled by one untimed call first, so
the timings are steady-state. CatSim's includes writing and reading its intermediate files and
fitting its BHC polynomials.
"""

# ╔═╡ 06000008-0000-4000-8000-000000000001
md"""
### 1. CatSim
"""

# ╔═╡ 06000008-0000-4000-8000-000000000010
# ╠═╡ show_logs = false
catsim_result = !HAS_GECATSIM ? nothing : let
    work_dir = mktempdir(; prefix = "basissim_catsim_06_")
    tag = joinpath(work_dir, "gammex472")
    local recon
    elapsed = @elapsed begin
        json = export_phantom_for_catsim(phantom_cpu, work_dir, "gammex472")
        ct = catsim_create_simulation()
        catsim_configure_phantom!(ct, json)
        catsim_configure_scanner!(ct, scanner, protocol)
        catsim_configure_protocol!(ct, scanner, protocol; μ_water_cm = bhc.μ_water_ref)
        catsim_configure_recon!(ct, recon_opts; μ_water_cm = bhc.μ_water_ref)
        catsim_forward_project(ct; results_name = tag)
        recon = catsim_reconstruct_fdk(ct; results_name = tag)
    end
    rm(work_dir; recursive = true, force = true)
    (recon = recon, elapsed = elapsed)
end;

# ╔═╡ 06000009-0000-4000-8000-000000000001
md"""
### 2. BasisSimulator, CPU and GPU

One function, two phantoms: `phantom_cpu.mask` is an `Array`, so the first run stays on the host;
`phantom_gpu.mask` lives on the GPU, so the second runs there. Nothing else differs.
"""

# ╔═╡ 06000009-0000-4000-8000-000000000005
"""
    basissim_pipeline(phantom) -> HU volume

`create_workspace` → `simulate!` → water BHC → FDK → HU, on the phantom's backend.
"""
function basissim_pipeline(phantom)
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol, sim_opts; report_dose = false)
    sino = BS.apply_bhc_water(ws.sinogram, bhc)
    ws_fdk = BS.create_fdk_recon_workspace(sino, ws.geom, recon_opts.matrix_size; filter = :standard)
    μ = Array(BS.reconstruct!(ws_fdk, sino, ws.geom))
    return Float32.(BS.to_hounsfield(μ; μ_water = bhc.μ_water_ref))
end;

# ╔═╡ 06000009-0000-4000-8000-000000000010
basissim_cpu_result = let
    basissim_pipeline(phantom_cpu); GC.gc(true)                 # compile, untimed
    elapsed = @elapsed recon = basissim_pipeline(phantom_cpu)
    GC.gc(true)
    (recon = recon, elapsed = elapsed)
end;

# ╔═╡ 0600000a-0000-4000-8000-000000000010
basissim_gpu_result = let
    basissim_pipeline(phantom_gpu); GC.gc(true)                 # compile, untimed
    elapsed = @elapsed recon = basissim_pipeline(phantom_gpu)
    GC.gc(true)
    (recon = recon, elapsed = elapsed)
end;

# ╔═╡ 060000f3-0000-4000-8000-000000000001
md"""
## Results
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000001
md"""
### Image agreement

The central slice of each reconstruction in one HU window, and the difference of each
BasisSimulator image from CatSim's. CatSim writes its image with its own axis convention; it is
brought onto BasisSimulator's orientation by the flip or transpose (of the eight in-plane
symmetries) that best matches the BasisSimulator image, which the figure reports. The thin rings
along the body and rod edges are where the two images differ in edge sharpness; the flat
interiors are what the per-rod table below measures.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000005
catsim_aligned = catsim_result === nothing ? nothing : let
    ref = basissim_gpu_result.recon
    k = size(ref, 3) ÷ 2 + 1
    cands = [
        ("as written", identity),
        ("x reversed", v -> reverse(v; dims = 1)),
        ("y reversed", v -> reverse(v; dims = 2)),
        ("x and y reversed", v -> reverse(v; dims = (1, 2))),
        ("transposed", v -> permutedims(v, (2, 1, 3))),
        ("transposed, x reversed", v -> reverse(permutedims(v, (2, 1, 3)); dims = 1)),
        ("transposed, y reversed", v -> reverse(permutedims(v, (2, 1, 3)); dims = 2)),
        ("transposed, x and y reversed", v -> reverse(permutedims(v, (2, 1, 3)); dims = (1, 2))),
    ]
    a = vec(Float64.(ref[:, :, k]))
    corr(b) = (x = a .- mean(a); y = b .- mean(b); sum(x .* y) / sqrt(sum(x .^ 2) * sum(y .^ 2)))
    scores = [corr(vec(Float64.(f(catsim_result.recon)[:, :, k]))) for (_, f) in cands]
    i = argmax(scores)
    (recon = cands[i][2](catsim_result.recon), transform = cands[i][1], correlation = scores[i])
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000010
let
    k = size(basissim_gpu_result.recon, 3) ÷ 2 + 1
    win = (-200, 600)
    fmt(t) = @sprintf("%.2f s", t)
    panels = Any[]
    catsim_aligned === nothing ||
        push!(panels, ("CatSim", "Python + C · $(fmt(catsim_result.elapsed))", catsim_aligned.recon))
    push!(panels, ("BasisSimulator.jl (CPU)", "Julia · $(fmt(basissim_cpu_result.elapsed))", basissim_cpu_result.recon))
    push!(panels, ("BasisSimulator.jl ($(GPU_BACKEND.name))", "Julia · $(fmt(basissim_gpu_result.elapsed))", basissim_gpu_result.recon))

    n = length(panels)
    fig = Mke.Figure(size = (n * 480 + 100, catsim_aligned === nothing ? 560 : 1040))
    local hm
    for (c, (title, sub, vol)) in enumerate(panels)
        ax = Mke.Axis(fig[1, c]; title, subtitle = sub, aspect = Mke.DataAspect(), titlesize = 24, subtitlesize = 18)
        hm = Mke.heatmap!(ax, vol[:, :, k]; colormap = :grays, colorrange = win)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(fig[1, n + 1], hm; label = "HU", width = 14, labelsize = 18)
    if catsim_aligned !== nothing
        local hd
        for (c, (title, _, vol)) in enumerate(panels[2:end])
            ax = Mke.Axis(fig[2, c + 1]; title = "$(title) − CatSim", aspect = Mke.DataAspect(), titlesize = 20)
            d = vol[:, :, k] .- catsim_aligned.recon[:, :, k]
            nx, ny = size(d)
            d = [(i - (nx + 1) / 2)^2 + (j - (ny + 1) / 2)^2 <= (min(nx, ny) / 2 - 1)^2 ? d[i, j] : NaN32
                 for i in 1:nx, j in 1:ny]   # inside the reconstruction circle only
            hd = Mke.heatmap!(ax, d; colormap = :RdBu, colorrange = (-100, 100), nan_color = :white)
            Mke.hidedecorations!(ax)
        end
        Mke.Colorbar(fig[2, n + 1], hd; label = "ΔHU", width = 14, labelsize = 18)
        Mke.Label(fig[2, 1], "CatSim image: $(catsim_aligned.transform)\ncorrelation with BasisSim $(round(catsim_aligned.correlation; digits = 4))";
            fontsize = 18, tellwidth = false, tellheight = false)
        Mke.rowsize!(fig.layout, 2, Mke.Aspect(2, 1.0))
    end
    Mke.save(joinpath(@__DIR__, "..", "assets", "catsim_vs_basissim_mosaic.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 0600000b-0000-4000-8000-000000000020
md"""
### Per-rod HU

Mean HU inside each rod (the phantom labels resampled onto the reconstruction grid, eroded by one
voxel) on the central slice, next to the rod's theoretical monoenergetic HU at the water
reference energy. A water-only BHC leaves dense calcium and iodine rods below their theory in
**both** simulators (the rod's own beam hardening beyond the water curve; notebook 01 discusses
it). What this table tests is whether the two simulators agree with each other.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000030
let
    labels = BS.resample_to_recon(phantom_cpu, geom_inspect, recon_opts.matrix_size; method = :nearest)
    k = size(labels, 3) ÷ 2 + 1
    nx, ny = size(labels, 1), size(labels, 2)
    refE, μw = bhc.reference_energy_keV, bhc.μ_water_ref
    roi(lab) = [CartesianIndex(i, j, k) for j in 2:(ny - 1), i in 2:(nx - 1)
                if all(labels[i + di, j + dj, k] == lab for di in -1:1, dj in -1:1)]
    cs = catsim_aligned === nothing ? nothing : catsim_aligned.recon
    rows = String[]; diffs = Float64[]; rel = Float64[]
    for lab in [3; 10:16; 20:26]
        idx = roi(UInt8(lab))
        length(idx) < 5 && continue
        mat = phantom_cpu.materials[lab + 1]
        theory = lab == 3 ? 0.0 : 1000 * (BS.compute_μ_at_energy(mat, refE) - μw) / μw
        bs = mean(basissim_gpu_result.recon[idx])
        c = cs === nothing ? NaN : mean(cs[idx])
        cs === nothing || push!(diffs, bs - c)
        (cs === nothing || lab == 3) || push!(rel, 100 * (bs - c) / theory)
        push!(rows, "| $(lab == 3 ? "solid water body" : mat.name) | $(length(idx)) | $(round(theory; digits = 0)) | " *
                    (cs === nothing ? "—" : "$(round(c; digits = 1))") * " | $(round(bs; digits = 1)) | " *
                    (cs === nothing ? "—" : "$(round(bs - c; digits = 1))") * " | " *
                    (cs === nothing || lab == 3 ? "" : "$(round(100 * (bs - c) / theory; digits = 1)) %") * " |")
    end
    cpu_gpu = maximum(abs.(basissim_cpu_result.recon .- basissim_gpu_result.recon))
    summary = isempty(diffs) ? "CatSim skipped." :
        "Water: BasisSim − CatSim = $(round(diffs[1]; digits = 1)) HU. Over the $(length(rel)) rods the difference is " *
        "$(round(minimum(rel); digits = 1)) to $(round(maximum(rel); digits = 1)) % of the rod's theoretical HU. " *
        "Both simulators read every dense rod below its theory, and they differ mainly in how far: " *
        "the two use independent source-spectrum models (IPEM tables in BasisSimulator, CatSim's own " *
        "tungsten spectra) and independently fitted water corrections, and a spectrum difference " *
        "would show exactly there, in the beam-hardening residual that grows with rod density."
    Markdown.parse("""
    | region | voxels | theory (HU) | CatSim (HU) | BasisSim (HU) | BasisSim − CatSim | ÷ theory |
    |:--|--:|--:|--:|--:|--:|--:|
    $(join(rows, "\n"))

    $(summary) BasisSimulator CPU vs GPU: largest voxel difference $(@sprintf("%.3g", cpu_gpu)) HU
    over the whole volume (the same code on two backends; floating-point summation order is all
    that differs).
    """)
end

# ╔═╡ 0600000c-0000-4000-8000-000000000001
md"""
### Runtime

Wall-clock of each end-to-end pipeline, on a log scale so the GPU bar stays visible. The CPU bar
is the like-for-like comparison with CatSim (both run on the host); the GPU bar is what
BasisSimulator is built for. This page was rendered on an NVIDIA RTX PRO 6000 (CUDA).
"""

# ╔═╡ 0600000c-0000-4000-8000-000000000005
let
    rows = [
        ("CatSim\n(Python + C)", catsim_result === nothing ? NaN : catsim_result.elapsed, Mke.RGBf(0.4, 0.4, 0.45)),
        ("BasisSimulator.jl\nCPU", basissim_cpu_result.elapsed, Mke.RGBf(0.95, 0.55, 0.1)),
        ("BasisSimulator.jl\n$(GPU_BACKEND.name)", basissim_gpu_result.elapsed, Mke.RGBf(0.13, 0.59, 0.85)),
    ]
    ok = findall(r -> !isnan(r[2]), rows)
    ref_t = catsim_result === nothing ? nothing : catsim_result.elapsed
    fig = Mke.Figure(size = (1180, 620))
    ax = Mke.Axis(fig[1, 1];
        title = "End-to-end time: forward projection + BHC + FDK",
        subtitle = "120 kVp · 200 mA · 500 views · noise-free · Gammex 472 at 128² × 8 → 256² × 8",
        ylabel = "wall-clock (s)", xticks = (1:3, [r[1] for r in rows]), yscale = log10,
        titlesize = 30, subtitlesize = 20, ylabelsize = 22, xticklabelsize = 20, yticklabelsize = 16)
    Mke.barplot!(ax, ok, [rows[i][2] for i in ok]; color = [rows[i][3] for i in ok],
        strokecolor = :black, strokewidth = 1, width = 0.65)
    for i in ok
        t = rows[i][2]
        label = ref_t === nothing ? @sprintf("%.2f s", t) :
            (t == ref_t ? @sprintf("%.2f s\n(reference)", t) : @sprintf("%.2f s\n(%.1f× faster)", t, ref_t / t))
        Mke.text!(ax, i, t * 1.18; text = label, align = (:center, :bottom), fontsize = 20, font = :bold)
    end
    ys = [rows[i][2] for i in ok]
    Mke.ylims!(ax, minimum(ys) * 0.7, maximum(ys) * 3.5)
    Mke.save(joinpath(@__DIR__, "..", "assets", "catsim_vs_basissim_runtime_bar.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000010
let
    t_cs = catsim_result === nothing ? nothing : catsim_result.elapsed
    speed(t) = t_cs === nothing ? "—" : @sprintf("%.1f×", t_cs / t)
    Markdown.parse("""
    | pipeline | wall-clock | speed-up vs CatSim |
    |:--|--:|--:|
    | CatSim (voxelized projector, Python + C) | $(t_cs === nothing ? "— (skipped)" : @sprintf("%.2f s", t_cs)) | $(t_cs === nothing ? "—" : "1.0× (reference)") |
    | BasisSimulator.jl, CPU ($(Threads.nthreads()) Julia threads) | $(@sprintf("%.2f s", basissim_cpu_result.elapsed)) | $(speed(basissim_cpu_result.elapsed)) |
    | BasisSimulator.jl, $(GPU_BACKEND.name) | $(@sprintf("%.2f s", basissim_gpu_result.elapsed)) | $(speed(basissim_gpu_result.elapsed)) |
    """)
end

# ╔═╡ 0600000d-0000-4000-8000-000000000001
md"""
## Why BasisSimulator is faster

- **No disk round trip.** CatSim writes `.air`, `.offset`, `.scan` and `.prep` files and reads
  the `.prep` back for FDK, then writes the image volume. BasisSimulator keeps the sinogram on the
  device from forward projection through FBP.
- **One volume walk for the whole spectrum.** The default `:dd_fast` projector accumulates the
  path length through each material once and weights all energies from it, instead of tracing
  every energy separately.
- **The same kernels on any backend.** CatSim's voxelized projector is C on the host.
  BasisSimulator's kernels are written once with AcceleratedKernels.jl and run on CUDA, Metal,
  ROCm, oneAPI or the CPU, as the CPU and GPU runs above show.

The phantom here is deliberately small so that CatSim finishes in minutes. For a larger phantom
or a clinical-size detector, scale `n_voxels` and the scanner; the wrapper forwards any
`EICTScanner` / `CTProtocol` pair, with the bowtie restriction noted in `catsim_configure_protocol!`.
"""

# ╔═╡ Cell order:
# ╟─06000001-0000-4000-8000-000000000010
# ╟─06000001-0000-4000-8000-000000000020
# ╟─06000001-0000-4000-8000-000000000001
# ╟─06000001-0000-4000-8000-000000000002
# ╟─06000001-0000-4000-8000-000000000003
# ╟─06000001-0000-4000-8000-000000000004
# ╠═06000001-0000-4000-8000-000000000030
# ╟─06000001-0000-4000-8000-000000000031
# ╠═06000001-0000-4000-8000-000000000032
# ╟─06000001-0000-4000-8000-000000000033
# ╟─06000001-0000-4000-8000-000000000034
# ╠═06000001-0000-4000-8000-000000000040
# ╟─06000001-0000-4000-8000-000000000050
# ╠═06000001-0000-4000-8000-000000000060
# ╟─06000001-0000-4000-8000-000000000070
# ╟─06000001-0000-4000-8000-000000000080
# ╟─06000001-0000-4000-8000-000000000090
# ╟─060000f1-0000-4000-8000-000000000001
# ╟─06000002-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000010
# ╟─06000003-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000010
# ╠═06000003-0000-4000-8000-000000000020
# ╠═06000003-0000-4000-8000-000000000030
# ╟─06000004-0000-4000-8000-000000000001
# ╟─06000004-0000-4000-8000-000000000010
# ╟─06000004-0000-4000-8000-000000000020
# ╟─06000004-0000-4000-8000-000000000030
# ╠═06000004-0000-4000-8000-000000000040
# ╠═06000004-0000-4000-8000-000000000050
# ╠═06000004-0000-4000-8000-000000000060
# ╟─06000004-0000-4000-8000-000000000070
# ╟─06000004-0000-4000-8000-000000000080
# ╟─06000004-0000-4000-8000-000000000090
# ╟─06000005-0000-4000-8000-000000000001
# ╟─06000005-0000-4000-8000-000000000010
# ╟─06000005-0000-4000-8000-000000000030
# ╟─06000006-0000-4000-8000-000000000001
# ╠═06000006-0000-4000-8000-000000000010
# ╠═06000006-0000-4000-8000-000000000020
# ╟─06000007-0000-4000-8000-000000000001
# ╠═06000007-0000-4000-8000-000000000010
# ╠═06000007-0000-4000-8000-000000000020
# ╟─06000007-0000-4000-8000-000000000030
# ╟─060000f2-0000-4000-8000-000000000001
# ╟─06000008-0000-4000-8000-000000000001
# ╠═06000008-0000-4000-8000-000000000010
# ╟─06000009-0000-4000-8000-000000000001
# ╠═06000009-0000-4000-8000-000000000005
# ╠═06000009-0000-4000-8000-000000000010
# ╠═0600000a-0000-4000-8000-000000000010
# ╟─060000f3-0000-4000-8000-000000000001
# ╟─0600000b-0000-4000-8000-000000000001
# ╟─0600000b-0000-4000-8000-000000000005
# ╟─0600000b-0000-4000-8000-000000000010
# ╟─0600000b-0000-4000-8000-000000000020
# ╟─0600000b-0000-4000-8000-000000000030
# ╟─0600000c-0000-4000-8000-000000000001
# ╟─0600000c-0000-4000-8000-000000000005
# ╟─0600000c-0000-4000-8000-000000000010
# ╟─0600000d-0000-4000-8000-000000000001
