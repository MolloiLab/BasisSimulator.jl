### A Pluto.jl notebook ###
# v0.2.3

using Markdown
using InteractiveUtils

# ╔═╡ 06000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    # nb06 adds PythonCall to the docs Project.toml.  If your local Manifest
    # predates that, `Pkg.instantiate()` errors with "X is a direct dependency,
    # but does not appear in the manifest".  `Pkg.resolve()` first picks up the
    # new direct deps and writes them to the manifest, then `instantiate()`
    # actually installs them (and triggers CondaPkg to fetch Python + gecatsim
    # — 5–10 min on a fresh setup).
    Pkg.resolve()
    Pkg.instantiate()
end

# ╔═╡ 06000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 06000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 06000001-0000-4000-8000-000000000004
using Printf: @sprintf

# ╔═╡ 06000001-0000-4000-8000-000000000010
md"""
# CatSim vs BasisSimulator: Qualitative Match and Runtime

**Same scanner, same protocol, same Gammex 472 phantom.  Three
forward-projection + FDK pipelines run side-by-side: XCIST/CatSim
(industry reference, Python), BasisSimulator on CPU, BasisSimulator on
GPU.  We compare the recon images qualitatively, then the runtimes —
the second is the point of the simulator.**

!!! warning "🚧 Heavy install — read this first"
    This notebook is the only one in the docs gallery that uses Python.
    `docs/CondaPkg.toml` pins gecatsim to the **MolloiLab fork**
    (`git+https://github.com/MolloiLab/main`) for the Gammex 472 material
    definitions, and `docs/Project.toml` adds **PythonCall** + **CondaPkg**.
    On your **first** `Pkg.instantiate()` inside `docs/`, CondaPkg will
    download Python + numpy + pydicom + gecatsim — typically 5–10 minutes
    and ~1 GB on disk.  CI does NOT render this notebook (the static
    HTML in `docs/notebooks-static/` is regenerated locally and shipped
    verbatim), so CI doesn't pay this cost.

    If you don't have CatSim installed and just want the BasisSim CPU vs
    GPU comparison: comment out the `import PythonCall as PC` cell — every
    CatSim cell short-circuits to a "skipped" notice.

This notebook intentionally uses a **heavily downsampled** Gammex 472
(`n_voxels = 128`, ~2.7 mm voxels) so CatSim finishes in minutes rather
than hours.  Clinical-fidelity reference scans typically run at 1750³
voxels (≈ 5 billion); that takes hours per protocol and isn't
appropriate for a docs example.

Pipeline:

```
Gammex 472 @ 128³ (2.7 mm)
   → GE Apex Elite scanner + 120 kVp / 200 mA / 500-view protocol
   →┬→ CatSim (Python): forward project → FDK
    ├→ BasisSim CPU:   simulate! → reconstruct!
    └→ BasisSim GPU:   simulate! → reconstruct!  (same code path, GPU phantom mask)
   → 1×3 mid-slice mosaic with shared HU window
   → runtime / speedup table
```
"""

# ╔═╡ 06000001-0000-4000-8000-000000000020
md"""
## Notebook Setup

Same project + GPU detection idiom as nb02 / nb04 / nb05, plus the
Python-side `import PythonCall as PC` for gecatsim.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 06000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 06000001-0000-4000-8000-000000000032
# ╠═╡ show_logs = false
import PythonCall as PC

# ╔═╡ 06000001-0000-4000-8000-000000000033
import PlutoUI

# ╔═╡ 06000001-0000-4000-8000-000000000034
PlutoUI.TableOfContents()

# ╔═╡ 06000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 06000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 06000001-0000-4000-8000-000000000060
const HAS_GECATSIM = try
    PC.pyimport("gecatsim")
    true
catch err
    @warn "gecatsim not importable — CatSim cells will short-circuit" err
    false
end;

# ╔═╡ 06000001-0000-4000-8000-000000000070
HAS_GECATSIM ? md"""
    **gecatsim located** — CatSim cells will run.
    """ : md"""
    !!! warning "gecatsim not available — CatSim cells skipped"
        `pyimport("gecatsim")` failed.  Run `Pkg.instantiate()` inside
        `docs/` to trigger CondaPkg's gecatsim fetch (5–10 min) and re-run
        this cell.  The BasisSim CPU + GPU panels will still execute.
    """

# ╔═╡ 06000001-0000-4000-8000-000000000080
# Self-healing patch for the MolloiLab gecatsim fork.
#
# The fork ships `C_DD3Back_mm.py` and `C_DD3WBack_mm.py` (the "_mm" mm-units
# that `gecatsim.reconstruction.pyfiles.{art,sirt,cgls}_equiAngle` import at
# module load time.  Without those files, `pyimport("gecatsim.reconstruction.
# pyfiles.recon")` blows up at import time even though we only ever call FDK
# (which doesn't need DD3Back).
#
# Fix: write a no-op stub of each missing module into the installed gecatsim
# package's `pyfiles/` directory.  The stub's symbols `raise` if actually
# called, so iterative recons would fail loudly — but FDK never touches them.
#
# This is idempotent — re-running the cell is a no-op once the stubs exist.
# The right long-term fix is a PR to `github.com/MolloiLab/main` (the fork)
# that either ships a real `C_DD3Back.py` or updates the iterative recon imports.
gecatsim_patched = !HAS_GECATSIM ? false : let
        spec_mod = PC.pyimport("importlib.util")
        gecatsim_spec = spec_mod.find_spec("gecatsim")
        gecatsim_init_path = PC.pyconvert(String, gecatsim_spec.origin)
        pyfiles_dir = joinpath(dirname(gecatsim_init_path), "pyfiles")

        function _write_stub(name::String)
            path = joinpath(pyfiles_dir, "$(name).py")
            if isfile(path)
                return false
        end
            open(path, "w") do io
                print(
                    io, """
                    # Auto-generated stub by BasisSimulator.jl docs notebook 06.
                    # The MolloiLab gecatsim fork is missing this legacy module — only
                    # `$(name)_mm.py` (different signature) ships.  This stub lets
                    # `gecatsim.reconstruction.pyfiles.recon` import successfully so
                    # FDK recon works.  Iterative recons (ART/SART/CGLS) will raise.
                    def $(replace(name, "C_" => ""))(*args, **kwargs):
                        raise NotImplementedError(
                            "$(name) is a stub — install a real one from upstream "
                            "xcist/main or fix the MolloiLab fork.  FDK recon does "
                            "not need this; iterative recons (ART/SART/CGLS) do."
                        )
                    """
                )
        end
            return true
    end

        wrote_back = _write_stub("C_DD3Back")
        wrote_wback = _write_stub("C_DD3WBack")

        if wrote_back || wrote_wback
            @info "[gecatsim patch] wrote stub(s): C_DD3Back=$(wrote_back), C_DD3WBack=$(wrote_wback) → $(pyfiles_dir)"
    else
            @info "[gecatsim patch] all stubs already present at $(pyfiles_dir) — no-op"
    end
        true
end;

# ╔═╡ 06000001-0000-4000-8000-000000000090
# Speed patch for gecatsim's FDK recon.
#
# `gecatsim.reconstruction.pyfiles.fdk_equiAngle` ships two helper functions —
# `float3Darray2pointer` (numpy → C triple-pointer) and `float3Dpointer2array`
# (the inverse) — that walk the array element-by-element in pure Python.  For a
# 834×6×500 sinogram that's 2.5 M Python-level ctypes assignments before the
# C `fbp` even starts; on a 128³ Gammex 472 run it dominates wallclock by
# 60–90 minutes.  The recon does eventually finish — but you'd never know,
# because Python `print()`s from inside Pluto/PythonCall don't show up in the
# cell.
#
# Fix: monkey-patch both helpers to use one row-pointer per slice (driven by
# `arr.ctypes.data_as`) and a single `ctypes.memmove` per slice on the way back.
# Same C ABI, ~1000× fewer Python iterations.  Also enable line-buffered
# stdout so the upstream `print("* In C...")` lines flush to your terminal.
gecatsim_fdk_patched = !HAS_GECATSIM ? false : let
        PC.pyexec(
            """
            import ctypes
            import numpy as np
            import gecatsim.reconstruction.pyfiles.fdk_equiAngle as _fdk

            FLOAT       = ctypes.c_float
            PtrFLOAT    = ctypes.POINTER(FLOAT)
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
                # Keep `arr` alive — the row pointers alias into its buffer.
                out._keepalive_ = arr
                return out

            def _fast_ptr2arr(ptr, n, m, o):
                out = np.empty((n, m, o), dtype=np.float32)
                nbytes = o * ctypes.sizeof(FLOAT)
                for i in range(n):
                    for j in range(m):
                        ctypes.memmove(
                            out[i, j].ctypes.data_as(PtrFLOAT),
                            ptr[i][j],
                            nbytes,
                        )
                return out

            _fdk.float3Darray2pointer = _fast_arr2ptr
            _fdk.float3Dpointer2array = _fast_ptr2arr

            # Silence CatSim's chatty stdout — `run_all()` and `recon()` together
            # emit ~600+ buffered print() lines (per-material C-allocation logs,
            # 500 tqdm view ticks, FDK stage banners).  When PythonCall is talking
            # to a Pluto worker, that flood overflows the captured stdout pipe and
            # Pluto's "drain output before marking cell done" logic blocks on a
            # pipe that never empties — the cell hangs forever even though the
            # actual work finished.  Wrap both entry points in
            # `contextlib.redirect_stdout(io.StringIO())` so the prints get
            # absorbed in-process and never hit the pipe.  Plain `julia --project`
            # doesn't have a captured pipe, which is why scripts run fine.
            import contextlib, io
            import gecatsim as _gecatsim
            import gecatsim.reconstruction.pyfiles.recon as _recon_mod

            if not getattr(_gecatsim.CatSim, '_basissim_silenced_run_all', False):
                _orig_run_all = _gecatsim.CatSim.run_all
                def _quiet_run_all(self, *args, **kwargs):
                    with contextlib.redirect_stdout(io.StringIO()):
                        return _orig_run_all(self, *args, **kwargs)
                _gecatsim.CatSim.run_all = _quiet_run_all
                _gecatsim.CatSim._basissim_silenced_run_all = True

            if not getattr(_recon_mod, '_basissim_silenced_recon', False):
                _orig_recon = _recon_mod.recon
                def _quiet_recon(ct, *args, **kwargs):
                    with contextlib.redirect_stdout(io.StringIO()):
                        return _orig_recon(ct, *args, **kwargs)
                _recon_mod.recon = _quiet_recon
                _recon_mod._basissim_silenced_recon = True
            """,
            Main,
        )
        @info "[gecatsim FDK patch] vectorized float3D{array2pointer,pointer2array} + stdout silencing installed"
        true
end;

# ╔═╡ 060000f1-0000-4000-8000-000000000001
md"""
## Scan and Phantom Setup

One scanner, one protocol, one phantom, shared verbatim by all three
pipelines.  These cells also build the CatSim wrapper layer that hands
BasisSimulator's phantom to gecatsim.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
md"""
### 01. Scanner: GE Revolution Apex Elite

Same hardware as nb02 / nb03 / nb05.  The GE Apex Elite is the
clinical scanner the CatSim reference is configured against, so it's
the right scanner to put on equal footing with CatSim's projector.

The wrapper layer in §3 converts BasisSimulator's **isocenter-pitch**
detector geometry to CatSim's **face-pitch** convention via the
magnification factor `SDD/SID`.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000010
scanner = BS.Scanner(
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
### 02. Protocol and Sim/Recon Options

Single 120 kVp / 200 mA / 500-view acquisition, 4 mm collimation, 35 cm
recon FOV.  Recon matrix is `(256, 256, n_z)` to keep the comparison
quick — the goal is qualitative-similarity verification, not clinical
fidelity.
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
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234, projector = :dd_fast);

# ╔═╡ 06000003-0000-4000-8000-000000000030
recon_opts = let
    slice_thickness_mm = 0.625
    n_z = max(1, round(Int, protocol.collimation_mm / slice_thickness_mm))
    BS.ReconOptions(
        matrix_size = (256, 256, n_z),
        fov_cm = 35.0,
        z_cm = protocol.collimation_mm / 10.0,
    )
end;

# ╔═╡ 06000004-0000-4000-8000-000000000001
md"""
### 03. CatSim Wrapper Layer

Eight Julia functions wrap gecatsim so it accepts `BS.Scanner` /
`BS.CTProtocol` / `BS.ReconOptions` / `BS.Phantom` directly.  The
wrappers only do struct-field forwarding + two CatSim quirks
(`detectorColsPerMod = 1`, `detectorColSkip = 0` — without these you
get braided / squashed sinograms).
"""

# ╔═╡ 06000004-0000-4000-8000-000000000010
const _catsim_ref = Ref{PC.Py}();

# ╔═╡ 06000004-0000-4000-8000-000000000011
const _recon_mod_ref = Ref{PC.Py}();

# ╔═╡ 06000004-0000-4000-8000-000000000012
const _np_ref = Ref{PC.Py}();

# ╔═╡ 06000004-0000-4000-8000-000000000013
const _cfg_path_ref = Ref("");

# ╔═╡ 06000004-0000-4000-8000-000000000020
function catsim_init()
    # Force-reference both patch flags so Pluto runs the stub-writer AND the
    # FDK speed patch before us.
    gecatsim_patched     || error("gecatsim patch did not run — see §0")
    gecatsim_fdk_patched || error("gecatsim FDK speed patch did not run — see §0")

    if !isassigned(_catsim_ref)
        _catsim_ref[] = PC.pyimport("gecatsim")
        _recon_mod_ref[] = PC.pyimport("gecatsim.reconstruction.pyfiles.recon")
        _np_ref[] = PC.pyimport("numpy")

        spec = PC.pyimport("importlib.util")
        gecatsim_spec = spec.find_spec("gecatsim")
        gecatsim_path = PC.pyconvert(String, gecatsim_spec.origin)
        base_path = dirname(dirname(gecatsim_path))
        _cfg_path_ref[] = joinpath(base_path, "gecatsim", "examples", "cfg")
    end
    return _catsim_ref[], _recon_mod_ref[], _np_ref[], _cfg_path_ref[]
end

# ╔═╡ 06000004-0000-4000-8000-000000000030
function catsim_create_simulation(;
        phantom_cfg = "Phantom_Sample.cfg",
        scanner_cfg = "Scanner_Sample_generic.cfg",
        protocol_cfg = "Protocol_Sample_axial.cfg",
    )
    xc, _, _, cfg_path = catsim_init()
    return xc.CatSim(
        joinpath(cfg_path, phantom_cfg),
        joinpath(cfg_path, scanner_cfg),
        joinpath(cfg_path, protocol_cfg),
    )
end

# ╔═╡ 06000004-0000-4000-8000-000000000040
function catsim_configure_scanner!(ct, scanner, protocol)
    magnification = scanner.source_to_detector / scanner.source_to_isocenter

    n_active_rows = if protocol.collimation_mm !== nothing
        round(Int, protocol.collimation_mm / scanner.detector_row_size)
    else
        scanner.detector_rows
    end

    ct.scanner.sid = scanner.source_to_isocenter
    ct.scanner.sdd = scanner.source_to_detector
    ct.scanner.detectorColCount = scanner.detector_cols
    ct.scanner.detectorRowCount = n_active_rows
    ct.scanner.detectorColSize = scanner.detector_col_size * magnification  # iso → face
    ct.scanner.detectorRowSize = scanner.detector_row_size * magnification

    # Prevent "braided" sinograms — every pixel its own module.
    ct.scanner.detectorColsPerMod = 1
    ct.scanner.detectorRowsPerMod = n_active_rows

    # Prevent "squashed" sinograms — zero inter-module gap.
    ct.scanner.detectorColSkip = 0.0
    ct.scanner.detectorRowSkip = 0.0
    return ct
end

# ╔═╡ 06000004-0000-4000-8000-000000000050
function catsim_configure_protocol!(ct, protocol)
    ct.protocol.mA = protocol.mA
    ct.protocol.viewsPerRotation = protocol.views
    ct.protocol.viewCount = protocol.views
    ct.protocol.stopViewId = protocol.views - 1
    ct.protocol.rotationTime = protocol.rotation_time
    ct.protocol.spectrumFilename = "tungsten_tar7.0_$(Int(protocol.kVp))_filt.dat"
    return ct
end

# ╔═╡ 06000004-0000-4000-8000-000000000060
function catsim_configure_recon!(ct, recon_opts; μ_water_cm = nothing)
    xc, _, _, cfg_path = catsim_init()
    xc.source_cfg(joinpath(cfg_path, "Recon_Sample_2d.cfg"), ct)

    n_slices = recon_opts.matrix_size[3]
    slice_thick_mm = recon_opts.z_cm * 10.0 / n_slices    # cm → mm

    ct.recon.fov = recon_opts.fov_cm * 10.0    # cm → mm
    ct.recon.imageSize = recon_opts.matrix_size[1]
    ct.recon.sliceCount = n_slices
    ct.recon.sliceThickness = slice_thick_mm

    # `Recon_Sample_2d.cfg` doesn't set `reconType` — pin it to FDK so we don't
    # accidentally hit the (broken) iterative recons in the MolloiLab fork.
    ct.recon.reconType = "fdk_equiAngle"

    ct.recon.unit = "HU"
    ct.recon.mu = μ_water_cm !== nothing ? μ_water_cm / 10.0 : 0.02   # cm⁻¹ → mm⁻¹
    ct.recon.huOffset = -1000
    return ct
end

# ╔═╡ 06000004-0000-4000-8000-000000000070
function catsim_configure_phantom!(ct, json_path; scale = 1.0, offset = [0.0, 0.0, 0.0])
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.filename = json_path
    ct.phantom.scale = scale
    ct.phantom.centerOffset = PC.pylist(offset)
    return ct
end

# ╔═╡ 06000004-0000-4000-8000-000000000080
function catsim_forward_project(ct; results_name = "catsim_out")
    ct.resultsName = results_name
    ct.run_all()

    rows = Int(PC.pyconvert(Float64, ct.scanner.detectorRowCount))
    cols = Int(PC.pyconvert(Float64, ct.scanner.detectorColCount))
    views = Int(PC.pyconvert(Float64, ct.protocol.viewCount))

    raw_bytes = read("$(results_name).prep")
    sino_flat = reinterpret(Float32, raw_bytes)
    return reshape(sino_flat, (cols, rows, views))
end

# ╔═╡ 06000004-0000-4000-8000-000000000090
function catsim_reconstruct_fdk(ct; results_name = "catsim_out")
    _, recon_mod, _, _ = catsim_init()
    ct.resultsName = results_name
    ct.recon.filename = ct.resultsName
    ct.do_Recon = 1

    recon_mod.recon(ct)

    nx = Int(PC.pyconvert(Float64, ct.recon.imageSize))
    ny = Int(PC.pyconvert(Float64, ct.recon.imageSize))
    nz = Int(PC.pyconvert(Float64, ct.recon.sliceCount))

    recon_file = "$(results_name)_$(nx)x$(ny)x$(nz).raw"
    isfile(recon_file) || error("CatSim recon file not found: $recon_file")
    raw_bytes = read(recon_file)
    return reshape(reinterpret(Float32, raw_bytes), (nx, ny, nz))
end

# ╔═╡ 06000004-0000-4000-8000-0000000000a0
function catsim_cleanup(results_name)
    for ext in (".air", ".offset", ".scan", ".prep")
        f = results_name * ext
        isfile(f) && rm(f)
    end
    parent = dirname(results_name)
    parent == "" && (parent = ".")
    base = basename(results_name)
    for f in readdir(parent)
        if startswith(f, base) && endswith(f, ".raw")
            rm(joinpath(parent, f))
        end
    end
    return nothing
end

# ╔═╡ 06000005-0000-4000-8000-000000000001
md"""
#### CatSim Wrapper: Label → Material Name

`BS.create_gammex_472` mask labels: 1 = solid water body, 10–16 = Ca
inserts, 20–26 = I inserts.  CatSim wants string material names, and
the **MolloiLab gecatsim fork** ships the matching `Gammex472_*` entries.
This dict is what the `export_phantom_for_catsim` step (§3c) writes
into the phantom JSON.
"""

# ╔═╡ 06000005-0000-4000-8000-000000000010
const REGION_TO_CATSIM = Dict{Int, String}(
    1 => "water",                   # solid water body  (≈ water in CatSim)
    2 => "water",                   # pure water vials
    3 => "water",                   # SW reference rods
    10 => "Gammex472_Ca_50",
    11 => "Gammex472_Ca_100",
    12 => "Gammex472_Ca_200",
    13 => "Gammex472_Ca_300",
    14 => "Gammex472_Ca_400",
    15 => "Gammex472_Ca_500",
    16 => "Gammex472_Ca_600",
    20 => "Gammex472_I_2_0",
    21 => "Gammex472_I_2_5",
    22 => "Gammex472_I_5_0",
    23 => "Gammex472_I_7_5",
    24 => "Gammex472_I_10_0",
    25 => "Gammex472_I_15_0",
    26 => "Gammex472_I_20_0",
);

# ╔═╡ 06000005-0000-4000-8000-000000000020
md"""
#### CatSim Wrapper: Phantom → Voxelized JSON

CatSim's `Phantom_Voxelized` callback wants:

1. A JSON header listing one entry per material with that material's
   density-fraction map filename, shape, voxel size (mm), and offset.
2. A binary `.density_` file per material — Float32 column-major, one
   value per voxel ∈ [0, 1] giving that material's volume fraction.

For a hard-segmented mask (every voxel belongs to exactly one label)
each density map is just the binary indicator `mask .== label`.
"""

# ╔═╡ 06000005-0000-4000-8000-000000000030
function export_phantom_for_catsim(phantom, output_dir, basename_str)
    mask_cpu = phantom.mask isa Array ? phantom.mask : Array(phantom.mask)
    nx, ny, nz = size(mask_cpu)
    vx, vy, vz = phantom.voxel_size .* 10.0    # cm → mm

    mkpath(output_dir)
    unique_labels = sort(unique(mask_cpu))
    filter!(l -> l != 0, unique_labels)

    json_materials = String[]; json_filenames = String[]; json_datatypes = String[]
    json_cols = Int[];      json_rows = Int[];      json_slices = Int[]
    json_xsize = Float64[]; json_ysize = Float64[]; json_zsize = Float64[]
    json_xoffset = Float64[]; json_yoffset = Float64[]; json_zoffset = Float64[]
    json_densscale = Float64[]

    for lbl in unique_labels
        lbl_int = Int(lbl)
        haskey(REGION_TO_CATSIM, lbl_int) || continue
        mat_name = REGION_TO_CATSIM[lbl_int]

        density_map = Float32.(mask_cpu .== lbl)
        fname = "$(basename_str)_mat$(lbl_int).density_"
        write(joinpath(output_dir, fname), density_map)

        push!(json_materials, mat_name)
        push!(json_filenames, fname)
        push!(json_datatypes, "float")
        push!(json_cols, nx);   push!(json_rows, ny);   push!(json_slices, nz)
        push!(json_xsize, vx);  push!(json_ysize, vy);  push!(json_zsize, vz)
        push!(json_xoffset, (nx + 1) / 2.0)
        push!(json_yoffset, (ny + 1) / 2.0)
        push!(json_zoffset, (nz + 1) / 2.0)
        push!(json_densscale, 1.0)
    end

    json_data = Dict(
        "n_materials" => length(json_materials),
        "mat_name" => json_materials,
        "volumefractionmap_filename" => json_filenames,
        "volumefractionmap_datatype" => json_datatypes,
        "cols" => json_cols,
        "rows" => json_rows,
        "slices" => json_slices,
        "x_size" => json_xsize,
        "y_size" => json_ysize,
        "z_size" => json_zsize,
        "x_offset" => json_xoffset,
        "y_offset" => json_yoffset,
        "z_offset" => json_zoffset,
        "density_scale" => json_densscale,
    )

    json_path = joinpath(output_dir, "$(basename_str).json")
    open(json_path, "w") do io
        println(io, "{")
        items = collect(json_data)
        for (i, (k, v)) in enumerate(items)
            val_str = if v isa Vector{String}
                "[\"" * join(v, "\", \"") * "\"]"
            elseif v isa Vector
                "[" * join(v, ", ") * "]"
            else
                string(v)
            end
            comma = i < length(items) ? "," : ""
            println(io, "  \"$k\": $val_str$comma")
        end
        println(io, "}")
    end
    return json_path
end

# ╔═╡ 06000006-0000-4000-8000-000000000001
md"""
### 04. Build the Gammex 472 Phantom

`n_voxels = 128` → 2.7 mm voxels at 35 cm FOV.  Coarse enough that
CatSim's voxelized projector finishes in ~1 minute on a laptop CPU but
still resolves the 28 mm rods.  `n_slices = 8` matches the recon slab.
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
### 05. Theoretical μ_water for HU Calibration

All three pipelines use the same per-kVp μ_water reference so HU
baselines line up.  `BS.compute_polychromatic_μ_water` returns a
spectrum-weighted, phantom-hardened μ — the **resolved source spectrum**
(post-bowtie, post-flat-filter) is integrated against `exp(-μ_water · L)`
for `L = ` the actual phantom diameter.  Diameter `L` is pulled from the
voxelized phantom via [`BS.estimate_phantom_diameter_cm`](@ref) — same
approach as nb04, no hardcoded 33 cm.
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
μ_water_120 = let
    # Phantom-hardened μ_water: pull the *actual* body diameter from the
    # voxelized phantom (no hardcoded 33 cm) so μ_water tracks any change to
    # `n_voxels` / `fov_cm` automatically.  Same approach as nb04.
    voxel_size_mm = phantom_cpu.voxel_size .* 10.0
    phantom_diam_cm = BS.estimate_phantom_diameter_cm(phantom_cpu.mask, voxel_size_mm)
    μ = BS.compute_polychromatic_μ_water(
        sim_opts, protocol;
        scanner = scanner,
        geom = geom_inspect,
        water_path_cm = phantom_diam_cm,
    )
    @info "μ_water (120 kVp, $(round(phantom_diam_cm, digits = 1)) cm hardening) = $(round(μ, digits = 5)) cm⁻¹"
    μ
end;

# ╔═╡ 060000f2-0000-4000-8000-000000000001
md"""
## Run Both Simulators

The same forward-project → FDK job three ways: CatSim (the Python
reference), BasisSimulator on CPU, and BasisSimulator on GPU.
"""

# ╔═╡ 06000008-0000-4000-8000-000000000001
md"""
### 01. Run CatSim
"""

# ╔═╡ 06000008-0000-4000-8000-000000000010
catsim_result = let
    if !HAS_GECATSIM
        nothing
    else
        work_dir = mktempdir(; prefix = "basissim_catsim_06_")
        json_path = export_phantom_for_catsim(phantom_cpu, work_dir, "gammex472")

        @info "[CatSim] running 120 kVp / 200 mA / 500 views on Gammex 472 (n_voxels=128)…"
        tag = joinpath(work_dir, "gammex472_run")

        elapsed = @elapsed begin
            ct = catsim_create_simulation()
            catsim_configure_phantom!(ct, json_path)
            catsim_configure_scanner!(ct, scanner, protocol)
            catsim_configure_protocol!(ct, protocol)
            catsim_configure_recon!(ct, recon_opts; μ_water_cm = μ_water_120)

            sino = catsim_forward_project(ct; results_name = tag)
            recon = catsim_reconstruct_fdk(ct; results_name = tag)
        end

        catsim_cleanup(tag)
        rm(work_dir; recursive = true, force = true)

        @info "[CatSim] forward proj + FDK total = $(round(elapsed, digits = 2)) s"
        (recon = recon, elapsed = elapsed)
    end
end;

# ╔═╡ 06000009-0000-4000-8000-000000000001
md"""
### 02. Run BasisSimulator (CPU)

`phantom_cpu.mask` is a regular `Array{UInt8, 3}`, so the EICT workspace
runs everything on the CPU side.
"""

# ╔═╡ 06000009-0000-4000-8000-000000000010
basissim_cpu_result = let
    @info "[BasisSim CPU] warm-up (excluded from timing)…"
    let
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_cpu)
        BS.simulate!(ws, phantom_cpu, protocol, sim_opts)
        ws_fdk = BS.create_fdk_recon_workspace(
            ws.sinogram, ws.geom, recon_opts.matrix_size; filter = :standard,
        )
        BS.reconstruct!(ws_fdk, ws.sinogram, ws.geom)
        ws = nothing; ws_fdk = nothing
    end
    GC.gc(true)

    @info "[BasisSim CPU] timing run…"
    local recon_μ
    elapsed = @elapsed begin
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_cpu)
        BS.simulate!(ws, phantom_cpu, protocol, sim_opts)
        ws_fdk = BS.create_fdk_recon_workspace(
            ws.sinogram, ws.geom, recon_opts.matrix_size; filter = :standard,
        )
        recon_μ = Array(BS.reconstruct!(ws_fdk, ws.sinogram, ws.geom))
        ws = nothing; ws_fdk = nothing
    end
    GC.gc(true)

    recon_HU = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_120))
    @info "[BasisSim CPU] forward proj + FBP = $(round(elapsed, digits = 2)) s"
    (recon = recon_HU, elapsed = elapsed)
end;

# ╔═╡ 0600000a-0000-4000-8000-000000000001
md"""
### 03. Run BasisSimulator (GPU)

Same scanner / protocol / phantom geometry as the CPU run, but
`phantom_gpu.mask` lives on the detected GPU backend
(**$(GPU_BACKEND.name)**).  The simulator's hot path is GPU-aware: the
forward-projection kernel and FBP filter both stream off the GPU mask
without an extra CPU round-trip.
"""

# ╔═╡ 0600000a-0000-4000-8000-000000000010
basissim_gpu_result = let
    @info "[BasisSim $(GPU_BACKEND.name)] warm-up (excluded from timing)…"
    let
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
        BS.simulate!(ws, phantom_gpu, protocol, sim_opts)
        ws_fdk = BS.create_fdk_recon_workspace(
            ws.sinogram, ws.geom, recon_opts.matrix_size; filter = :standard,
        )
        BS.reconstruct!(ws_fdk, ws.sinogram, ws.geom)
        ws = nothing; ws_fdk = nothing
    end
    GC.gc(true)

    @info "[BasisSim $(GPU_BACKEND.name)] timing run…"
    local recon_μ
    elapsed = @elapsed begin
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
        BS.simulate!(ws, phantom_gpu, protocol, sim_opts)
        ws_fdk = BS.create_fdk_recon_workspace(
            ws.sinogram, ws.geom, recon_opts.matrix_size; filter = :standard,
        )
        recon_μ = Array(
            BS.reconstruct!(ws_fdk, ws.sinogram, ws.geom)
        )
        ws = nothing; ws_fdk = nothing
    end
    GC.gc(true)

    recon_HU = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_120))
    @info "[BasisSim $(GPU_BACKEND.name)] forward proj + FBP = $(round(elapsed, digits = 2)) s"
    (recon = recon_HU, elapsed = elapsed)
end;

# ╔═╡ 060000f3-0000-4000-8000-000000000001
md"""
## Results

Qualitative image agreement first, then the runtime comparison that is the
whole point of a native-Julia GPU simulator.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000001
md"""
### Qualitative Comparison

Mid-slice of all three reconstructions, shared HU window
(-200, 600) so the rod contrast lines up visually.  CatSim handles HU
internally (`recon.unit = "HU"`, `huOffset = -1000`); BasisSim's recons
go through `BS.to_hounsfield(...; μ_water = μ_water_120)` with the same
spectrum-weighted μ_water — so HU baselines should match across all
three panels.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000010
let
    fmt_s(x) = @sprintf("%.2f s", x)
    fmt_speedup(s) = @sprintf("%.1f× faster", s)
    ref_t = catsim_result === nothing ? nothing : catsim_result.elapsed

    cpu_sub = ref_t === nothing ?
        "Julia · $(fmt_s(basissim_cpu_result.elapsed))" :
        "Julia · $(fmt_s(basissim_cpu_result.elapsed)) · $(fmt_speedup(ref_t / basissim_cpu_result.elapsed))"
    gpu_sub = ref_t === nothing ?
        "$(GPU_BACKEND.name) · $(fmt_s(basissim_gpu_result.elapsed))" :
        "$(GPU_BACKEND.name) · $(fmt_s(basissim_gpu_result.elapsed)) · $(fmt_speedup(ref_t / basissim_gpu_result.elapsed))"

    n_panels = catsim_result === nothing ? 2 : 3
    fig = CM.Figure(size = (n_panels * 540 + 90, 600))

    title_kwargs = (titlesize = 28, subtitlesize = 20)

    col = 1
    last_hm = nothing
    if catsim_result !== nothing
        ax = CM.Axis(
            fig[1, col];
            title = "CatSim",
            subtitle = "Python · $(fmt_s(catsim_result.elapsed)) · reference",
            aspect = CM.DataAspect(), yreversed = true,
            title_kwargs...,
        )
        last_hm = CM.heatmap!(
            ax, catsim_result.recon[:, :, size(catsim_result.recon, 3) ÷ 2 + 1];
            colormap = :grays, colorrange = (-200, 600),
        )
        CM.hidedecorations!(ax)
        col += 1
    end

    ax_cpu = CM.Axis(
        fig[1, col];
        title = "BasisSimulator.jl (CPU)",
        subtitle = cpu_sub,
        aspect = CM.DataAspect(), yreversed = true,
        title_kwargs...,
    )
    last_hm = CM.heatmap!(
        ax_cpu, basissim_cpu_result.recon[:, :, size(basissim_cpu_result.recon, 3) ÷ 2 + 1];
        colormap = :grays, colorrange = (-200, 600),
    )
    CM.hidedecorations!(ax_cpu)
    col += 1

    ax_gpu = CM.Axis(
        fig[1, col];
        title = "BasisSimulator.jl ($(GPU_BACKEND.name))",
        subtitle = gpu_sub,
        aspect = CM.DataAspect(), yreversed = true,
        title_kwargs...,
    )
    last_hm = CM.heatmap!(
        ax_gpu, basissim_gpu_result.recon[:, :, size(basissim_gpu_result.recon, 3) ÷ 2 + 1];
        colormap = :grays, colorrange = (-200, 600),
    )
    CM.hidedecorations!(ax_gpu)

    CM.Colorbar(fig[1, col + 1], last_hm; label = "HU", width = 14, labelsize = 18)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "catsim_vs_basissim_mosaic.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000001
md"""
### Runtime: Bar Chart and Table

End-to-end **forward projection + FBP** wallclock for each pipeline.
Log-y so the GPU bar doesn't disappear next to the CatSim bar.  The
CPU bar is the apples-to-apples comparison (both run on the host
CPU); the GPU bar is what BasisSim is actually built for.
"""

# ╔═╡ 0600000c-0000-4000-8000-000000000005
let
    rows = NamedTuple[]
    push!(
        rows, (
            label = "CatSim\n(Python)",
            seconds = catsim_result === nothing ? NaN : catsim_result.elapsed,
            color = CM.RGBf(0.4, 0.4, 0.45),
        )
    )
    push!(
        rows, (
            label = "BasisSimulator.jl\nCPU",
            seconds = basissim_cpu_result.elapsed,
            color = CM.RGBf(0.95, 0.55, 0.1),
        )
    )
    push!(
        rows, (
            label = "BasisSimulator.jl\n$(GPU_BACKEND.name)",
            seconds = basissim_gpu_result.elapsed,
            color = CM.RGBf(0.13, 0.59, 0.85),
        )
    )

    valid_idx = findall(r -> !isnan(r.seconds), rows)
    xs = collect(1:length(rows))
    ys = [isnan(r.seconds) ? 1.0 : r.seconds for r in rows]
    cs = [r.color for r in rows]

    ref_t = catsim_result === nothing ? nothing : catsim_result.elapsed

    fig = CM.Figure(size = (1180, 620))
    ax = CM.Axis(
        fig[1, 1];
        title = "End-to-End Timing (Forward projection + FBP)",
        subtitle = "120 kVp · 200 mA · 500 views · 128³ Gammex 472",
        xlabel = "",
        ylabel = "Timing (s)",
        xticks = (xs, [r.label for r in rows]),
        yscale = log10,
        titlesize = 32,
        subtitlesize = 24,
        ylabelsize = 22,
        xlabelsize = 22,
        xticklabelsize = 20,
        yticklabelsize = 16,
    )

    CM.barplot!(
        ax, xs[valid_idx], ys[valid_idx];
        color = cs[valid_idx],
        strokecolor = :black, strokewidth = 1,
        width = 0.65,
    )

    # annotate each bar with elapsed time + speedup vs CatSim
    for i in valid_idx
        r = rows[i]
        time_txt = @sprintf("%.2f s", r.seconds)
        annot = if ref_t === nothing
            time_txt
        elseif r.seconds == ref_t
            "$time_txt\n(reference)"
        else
            @sprintf("%s\n(%.1f× faster)", time_txt, ref_t / r.seconds)
        end
        CM.text!(
            ax, i, r.seconds * 1.18;
            text = annot, align = (:center, :bottom),
            fontsize = 20, font = :bold,
        )
    end

    # pad the y-range so the bold text annotations don't clip
    y_hi = maximum(ys[valid_idx]) * 3.5
    y_lo = minimum(ys[valid_idx]) * 0.7
    CM.ylims!(ax, y_lo, y_hi)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "catsim_vs_basissim_runtime_bar.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000010
let
    cs_label = catsim_result === nothing ? "—  *(skipped)*" :
        @sprintf("%.2f s", catsim_result.elapsed)
    cs_speedup = catsim_result === nothing ? "—" : "1.00× (reference)"
    cpu_speedup = catsim_result === nothing ? "—" :
        @sprintf("%.2f×", catsim_result.elapsed / basissim_cpu_result.elapsed)
    gpu_speedup = catsim_result === nothing ? "—" :
        @sprintf("%.2f×", catsim_result.elapsed / basissim_gpu_result.elapsed)

    gpu_row_label = GPU_BACKEND.name == "CPU" ?
        "BasisSim *GPU*  (no GPU detected — CPU run)" :
        "BasisSim GPU ($(GPU_BACKEND.name))"

    Markdown.parse(
        """
        | pipeline | wallclock | speedup vs CatSim |
        |----------|-----------|-------------------|
        | CatSim (Python, voxelized projector) | $(cs_label) | $(cs_speedup) |
        | BasisSim CPU                         | $(@sprintf("%.2f s", basissim_cpu_result.elapsed)) | $(cpu_speedup) |
        | $(gpu_row_label)                     | $(@sprintf("%.2f s", basissim_gpu_result.elapsed)) | $(gpu_speedup) |

        Numbers are end-to-end forward projection + FBP for one 120 kVp / 200 mA / 500-view scan
        on a 128³ Gammex 472 phantom into a $(recon_opts.matrix_size[1])×$(recon_opts.matrix_size[2])×$(recon_opts.matrix_size[3])
        HU recon.  **Both BasisSim runs were JIT-warmed once before the timing pass** so what's
        reported is steady-state hot-cache wallclock, comparable to CatSim's C-kernel runtime
        (no warm-up needed).  Re-running this notebook on different hardware will give different
        numbers but the ordering (CatSim > BasisSim CPU > BasisSim GPU on wallclock) holds.
        """
    )
end

# ╔═╡ 060000f4-0000-4000-8000-000000000001
md"""
## Summary

Where the speedup comes from, and where to take the comparison next.
"""

# ╔═╡ 0600000d-0000-4000-8000-000000000001
md"""
### Why BasisSimulator.jl Is Faster

A few specific things that show up in the runtime difference:

- **No disk round-trip per scan.** CatSim writes `*.air`, `*.offset`,
  `*.scan`, `*.prep`, then reloads `*.prep` from disk for FDK, then
  writes `*.raw` with the recon volume.  Every scan is several hundred
  MB of disk I/O.  BasisSim keeps the sinogram on the GPU between
  forward projection and FBP — zero disk round-trip until the user asks
  for the recon array.
- **Fused per-energy projection.**  `BS.simulate!` runs the spectral
  weighting as a fused kernel pass over the polychromatic spectrum
  rather than one ray-trace per energy bin.  Same physics, fewer kernel
  launches.
- **GPU forward projection.**  CatSim's `C_Projector_Voxelized` is a C
  kernel that runs on the host CPU.  BasisSim's Siddon ray-tracer is
  written in `AcceleratedKernels.jl` and dispatches to whichever GPU
  backend is loaded (Metal, CUDA, ROCm) — same Julia source, different
  hardware.
- **HU conversion is a single broadcast.**  No per-slice file write,
  no `huOffset` arithmetic on every voxel — just `to_hounsfield(recon_μ;
  μ_water)`.

For docs we kept the phantom small (128³) on purpose — the speedup
ratio against CatSim grows with phantom resolution, since the GPU
forward-projection kernel's wallclock barely moves while CatSim's
ray-tracing time scales linearly.
"""

# ╔═╡ 0600000e-0000-4000-8000-000000000001
md"""
### Where to Go Next

- For a **full multi-protocol regression** (multiple dose levels × kVp
  values) at clinical fidelity (1750³ phantom, hours of CatSim runtime
  per protocol), scale up `n_voxels` and loop over multiple
  `CTProtocol` instances.  The wrapper layer in §3 already accepts any
  `BS.Scanner` / `BS.CTProtocol` pair.
- For the **per-rod HU regression** (measured vs theoretical via
  `XrayAttenuation`), see notebooks 03 (dual-kVp VMI) and 04 (PCCT VMI)
  — same Gammex 472, same XrayAttenuation comparison flow, no Python
  involved.
- For **using your own scanner / phantom** with the wrapper layer:
  drop in a different `BS.Scanner` and `BS.create_phantom_from_mask(...)`
  — the wrappers don't care about scanner brand or phantom geometry,
  they only forward struct fields.
"""

# ╔═╡ Cell order:
# ╟─06000001-0000-4000-8000-000000000010
# ╟─06000001-0000-4000-8000-000000000020
# ╠═06000001-0000-4000-8000-000000000001
# ╠═06000001-0000-4000-8000-000000000002
# ╠═06000001-0000-4000-8000-000000000003
# ╠═06000001-0000-4000-8000-000000000004
# ╠═06000001-0000-4000-8000-000000000030
# ╠═06000001-0000-4000-8000-000000000031
# ╠═06000001-0000-4000-8000-000000000032
# ╠═06000001-0000-4000-8000-000000000033
# ╠═06000001-0000-4000-8000-000000000034
# ╠═06000001-0000-4000-8000-000000000040
# ╟─06000001-0000-4000-8000-000000000050
# ╠═06000001-0000-4000-8000-000000000060
# ╟─06000001-0000-4000-8000-000000000070
# ╠═06000001-0000-4000-8000-000000000080
# ╠═06000001-0000-4000-8000-000000000090
# ╟─060000f1-0000-4000-8000-000000000001
# ╟─06000002-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000010
# ╟─06000003-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000010
# ╠═06000003-0000-4000-8000-000000000020
# ╠═06000003-0000-4000-8000-000000000030
# ╟─06000004-0000-4000-8000-000000000001
# ╠═06000004-0000-4000-8000-000000000010
# ╠═06000004-0000-4000-8000-000000000011
# ╠═06000004-0000-4000-8000-000000000012
# ╠═06000004-0000-4000-8000-000000000013
# ╠═06000004-0000-4000-8000-000000000020
# ╠═06000004-0000-4000-8000-000000000030
# ╠═06000004-0000-4000-8000-000000000040
# ╠═06000004-0000-4000-8000-000000000050
# ╠═06000004-0000-4000-8000-000000000060
# ╠═06000004-0000-4000-8000-000000000070
# ╠═06000004-0000-4000-8000-000000000080
# ╠═06000004-0000-4000-8000-000000000090
# ╠═06000004-0000-4000-8000-0000000000a0
# ╟─06000005-0000-4000-8000-000000000001
# ╠═06000005-0000-4000-8000-000000000010
# ╟─06000005-0000-4000-8000-000000000020
# ╠═06000005-0000-4000-8000-000000000030
# ╟─06000006-0000-4000-8000-000000000001
# ╠═06000006-0000-4000-8000-000000000010
# ╠═06000006-0000-4000-8000-000000000020
# ╟─06000007-0000-4000-8000-000000000001
# ╠═06000007-0000-4000-8000-000000000010
# ╠═06000007-0000-4000-8000-000000000020
# ╟─060000f2-0000-4000-8000-000000000001
# ╟─06000008-0000-4000-8000-000000000001
# ╠═06000008-0000-4000-8000-000000000010
# ╟─06000009-0000-4000-8000-000000000001
# ╠═06000009-0000-4000-8000-000000000010
# ╟─0600000a-0000-4000-8000-000000000001
# ╠═0600000a-0000-4000-8000-000000000010
# ╟─060000f3-0000-4000-8000-000000000001
# ╟─0600000b-0000-4000-8000-000000000001
# ╟─0600000b-0000-4000-8000-000000000010
# ╟─0600000c-0000-4000-8000-000000000001
# ╟─0600000c-0000-4000-8000-000000000005
# ╟─0600000c-0000-4000-8000-000000000010
# ╟─060000f4-0000-4000-8000-000000000001
# ╟─0600000d-0000-4000-8000-000000000001
# ╟─0600000e-0000-4000-8000-000000000001
