# The device agrees with the CPU, in the docs environment, before any notebook renders.
#
#     julia --project=docs docs/gpu_agreement.jl      # exit 0: agree; exit 1: a device answer differs
#
# A GPU kernel that compiles wrongly gives wrong numbers with no error, and only on some toolchains:
# the docs render on their own lockfile (`docs/Manifest-v*.toml`), which the GPU test suite does not
# run on. Every case here compared the CPU with the device and found a toolchain that got it wrong:
#
# - notebook 02's XCAT slab (10 materials, one slice of 1385 × 917), scatter off: CUDA.jl 6.1 with
#   AcceleratedKernels 0.4 / GPUCompiler 1.x halved the line integrals (0.16.7's docs);
# - the scatter estimate at 3, 12, 16 and 17 materials: Julia 1.13 with CUDA.jl 5 got 16 wrong
#   (0.16.4 to 0.16.6);
# - a photon-counting scan, its four bins.
#
# Noise off, the test suite's synthetic scatter library; about five minutes. `render_notebooks.sh`
# runs it before rendering on a GPU and stops if it fails.

import BasisSimulator as BS
using Printf

const REPO = dirname(@__DIR__)
const TOL = 1.0e-4 # Float32, rounding only; a miscompile is off by percent or more

# the device: CUDA here; any backend whose package is in the environment would do
import CUDA
CUDA.functional() ||
    (println("gpu_agreement: no functional GPU, nothing to compare"); exit(0))
const DEVICE = CUDA.CuArray
println(
    "gpu_agreement: Julia ",
    VERSION,
    ", CUDA.jl ",
    pkgversion(CUDA),
    ", AcceleratedKernels ",
    pkgversion(BS.AK),
    ", ",
    CUDA.name(CUDA.device()),
)

# the test suite's synthetic scatter library and its scan helpers (everything before its first testset)
include(joinpath(REPO, "test", "scatter_library.jl"))
let src = read(joinpath(REPO, "test", "scatter.jl"), String)
    include_string(Main, src[1:(first(findfirst("@testset", src)) - 1)])
end

failures = String[]

function compare!(name, cpu, gpu)
    d = maximum(abs.(gpu .- cpu)) / max(maximum(abs.(cpu)), eps())
    ok = isfinite(d) && d <= TOL
    @printf("  %-48s max rel |GPU − CPU| = %.2e  %s\n", name, d, ok ? "ok" : "DIFFERS")
    ok || push!(failures, name)
    return nothing
end

# 1. notebook 02's XCAT slab on the Apex, scatter off and on
let m = BS.xcat_default_materials()
    xcat = BS.load_xcat_male_slab(; materials = m, quiet = true)
    protocol = BS.CTProtocol(
        kVp = 120,
        mA = 250.0,
        views = 24,
        collimation_mm = 5.0,
        additional_filters = [("Al", 4.5)],
    )
    recon = BS.ReconOptions(matrix_size = (512, 512, 8), fov_cm = 35.0, z_cm = 0.5)
    for scatter in (false, true)
        opts = BS.SimOptions(seed = 1, use_noise = false, use_scatter = scatter)
        sino(dev) = let ph = BS.Phantom(dev(xcat.mask), xcat.materials, xcat.voxel_size_cm)
            ws = BS.create_workspace(BS.REVOLUTION_APEX, protocol, opts, recon, ph)
            BS.simulate!(ws, ph, protocol, opts; report_dose = false)
            Array(ws.sinogram)
        end
        compare!(
            "XCAT slab, $(length(xcat.materials)) materials, scatter $(scatter ? "on" : "off")",
            sino(identity),
            sino(DEVICE),
        )
    end
end

# 2. the scatter estimate by number of materials (the compacted Gammex 472, labels merged down)
let g = BS.compact_materials(BS.create_gammex_472(
    n_voxels = 160,
    n_slices = 8,
    fov_cm = 40.0,
    z_cm = 2.0,
))
    full = BS.create_gammex_472(n_voxels = 160, n_slices = 8, fov_cm = 40.0, z_cm = 2.0)
    cases = [
        [
            (K, copy(g.mask), Dict{Int, Any}(i - 1 => g.materials[i] for i in 1:K))
            for K in (3, 12, 16)
        ]...,
        (
            length(full.materials),
            full.mask,
            Dict{Int, Any}(i - 1 => full.materials[i] for i in eachindex(full.materials))
        ),
    ]
    for (K, mask, materials) in cases
        mask[mask .>= K] .= 0x01
        cpu = _eict_scatter_scan(_SCATTER_EICT, BS.Phantom(mask, materials, g.voxel_size))
        gpu = _eict_scatter_scan(
            _SCATTER_EICT,
            BS.Phantom(DEVICE(mask), materials, g.voxel_size),
        )
        compare!(
            "Gammex, $(K) materials, scatter estimate",
            Array(cpu.ws.scatter.estimate),
            Array(gpu.ws.scatter.estimate),
        )
        compare!("Gammex, $(K) materials, sinogram", cpu.sinogram, gpu.sinogram)
    end
end

# 3. a photon-counting scan with scatter: its four bins
let scanner = BS.PCCTScanner(
    BS.NAEOTOM_ALPHA;
    detector_rows = 4,
    detector_cols = 96,
    detector_col_size = 3.0,
    detector_row_size = 2.0,
    detector_col_offset = 0.0,
    scatter_kernels = "BS test counter",
)
    phantom = _scatter_phantom(; a_cm = 30.0, lungs = true)
    materials = Dict{Int, Any}(i - 1 => m for (i, m) in enumerate(phantom.materials))
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 50.0, views = 24)
    opts = _scatter_opts()
    bins(dev) = let ph = BS.Phantom(dev(phantom.mask), materials, phantom.voxel_size)
        calibration = BS.calibrate(scanner, protocol, _SCATTER_RECON, ph; sim_opts = opts)
        ws = BS.create_workspace(scanner, protocol, opts, _SCATTER_RECON, ph; calibration)
        r = BS.simulate!(ws, ph, protocol, opts; report_dose = false)
        cat((Array(b) for b in r.pcct_sino.bins)...; dims = 4)
    end
    compare!("photon counting, 4 bins, scatter on", bins(identity), bins(DEVICE))
end

if isempty(failures)
    println("gpu_agreement: the device agrees with the CPU")
    exit(0)
else
    println("gpu_agreement: the device DIFFERS from the CPU in: ", join(failures, "; "))
    exit(1)
end
