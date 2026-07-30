# Standalone Metal/unified-memory soak test. Run explicitly:
#   julia --project=docs test/memory_stability_pcct.jl
#
# This is intentionally not included in runtests.jl: it compiles and executes
# the complete PCCT spectral projector several times.

using BasisSimulator
using Metal

const BS = BasisSimulator

scanner = BS.Scanner(
    source_to_isocenter = 540.0,
    source_to_detector = 1080.0,
    detector_rows = 8,
    detector_cols = 64,
    detector_row_size = 1.0,
    detector_col_size = 1.0,
    detector_type = :photon_counting,
    detector_material = :CdTe,
    detector_depth = 1.6,
    n_energy_bins = 4,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0],
    dead_time_ns = 25.0,
)
protocol = BS.CTProtocol(
    mA = 2.5, kVp = 120.0, views = 16, rotation_time = 0.5,
)
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    use_noise = false,
    use_scatter = false,
    use_lag = false,
    use_focal_spot = false,
    use_optical_crosstalk = false,
    use_pcct_pileup = false,
)
recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
phantom_cpu = BS.create_gammex_472(
    n_voxels = 32, fov_cm = 20.0, z_cm = 2.0,
)
phantom = BS.Phantom(
    Metal.MtlArray(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
)

baseline = BS.backend_memory_snapshot(phantom.mask)
rows = NamedTuple[]
for iteration in 1:5
    ws = BS.create_workspace(
        scanner, protocol, sim_opts, recon_opts, phantom,
    )
    checksum = NaN
    peak = nothing
    released = 0
    try
        result = BS.simulate!(ws, phantom, protocol, sim_opts)
        checksum = sum(sum, (Array(bin) for bin in result.pcct_sino.bins))
        peak = BS.backend_memory_snapshot(first(ws.bins))
    finally
        released = BS.release_backend!(ws)
    end
    post = Int(Metal.device().currentAllocatedSize)
    push!(rows, (; iteration, checksum, released,
        peak_bytes = peak.device_allocated_bytes, post_bytes = post))
    println(last(rows))
end

posts = getproperty.(rows, :post_bytes)
tolerance = 16 * 2^20
maximum(posts) - minimum(posts) <= tolerance || error(
    "post-release Metal allocation drift exceeded $(tolerance) bytes: $posts",
)
last(posts) <= baseline.device_allocated_bytes + tolerance || error(
    "final Metal allocation failed to return near baseline: " *
    "baseline=$(baseline.device_allocated_bytes), posts=$posts",
)
println((status = :pass, baseline_bytes = baseline.device_allocated_bytes,
    post_release_bytes = posts))
