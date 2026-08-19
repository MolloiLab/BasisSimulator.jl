# Exact 04d scan-geometry unified-memory soak test.
#
# This intentionally runs outside Pluto so changing the simulation does not
# queue every reactive denoising descendant. It is not part of runtests.jl.
#
#   julia --project=docs test/memory_stability_pcct_representative.jl

using BasisSimulator
using Metal
using Statistics: mean

const BS = BasisSimulator
const N_CYCLES = parse(Int, get(ENV, "PCCT_MEMORY_CYCLES", "3"))
const DRIFT_TOLERANCE_BYTES = 64 * 2^20

phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
)
phantom = BS.Phantom(
    Metal.MtlArray(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
)

scanner = let
    native_col_mm = 0.275
    native_row_mm = 0.322
    sid = 610.0
    sdd = 1113.0
    magnification = sdd / sid
    bf = 2
    pixel_col_iso = (native_col_mm * bf) / magnification
    pixel_row_iso = (native_row_mm * bf) / magnification
    n_cols = ceil(Int, 360.0 / pixel_col_iso)
    BS.Scanner(
        source_to_isocenter = sid,
        source_to_detector = sdd,
        detector_rows = 144,
        detector_cols = n_cols,
        detector_row_size = pixel_row_iso,
        detector_col_size = pixel_col_iso,
        detector_row_offset = 0.0,
        detector_col_offset = pixel_col_iso / 2,
        focal_spot_width = 0.4,
        focal_spot_length = 0.5,
        target_angle = 7.0,
        gantry_rotation_time = 0.5,
        scan_diameter = 360.0,
        gantry_aperture = 820.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        detection_gain = 1.0,
        electronic_noise = 0.0,
        detector_type = :photon_counting,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        energy_resolution = 10.0,
        charge_sharing_fwhm = 0.08,
        dead_time_ns = 5.0,
        pixel_mode = :standard,
        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor = bf,
    )
end

protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
)
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    projector = :dd_fast,
    use_fill_factor = false,
    use_detector_efficiency = false,
    use_optical_crosstalk = false,
    use_focal_spot = false,
    use_lag = false,
    use_heel_effect = false,
    use_scatter = false,
    use_noise = true,
    use_pcct_scatter = true,
    use_pcct_scatter_correction = true,
    use_pcct_pileup = true,
    use_pcct_pileup_correction = true,
    pcct_noise_reduction = 0.0,
)
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 12),
    fov_cm = 35.0,
    z_cm = 0.5,
)

baseline = BS.backend_memory_snapshot(phantom.mask)
rows = NamedTuple[]
for iteration in 1:N_CYCLES
    pre = BS.backend_memory_snapshot(phantom.mask)
    ws = nothing
    cpu_bins = nothing
    released = 0
    checksum = NaN
    peak = nothing
    elapsed = @elapsed try
        ws = BS.create_workspace(
            scanner, protocol, sim_opts, recon_opts, phantom,
        )
        # capture_raw_counts=false: this script pins the reusable-workspace
        # memory contract; raw-count capture allocates per call by design.
        result = BS.simulate!(
            ws, phantom, protocol, sim_opts; capture_raw_counts = false,
        )
        cpu_bins = [Array(bin) for bin in result.pcct_sino.bins]
        checksum = sum(mean, cpu_bins)
        peak = BS.backend_memory_snapshot(first(ws.bins))
    finally
        ws === nothing || (released = BS.release_backend!(ws))
    end
    cpu_bins = nothing
    GC.gc(true)
    post = BS.backend_memory_snapshot(phantom.mask)
    row = (; iteration, elapsed, checksum, released,
        pre_host_free = pre.host_free_bytes,
        peak_host_free = peak.host_free_bytes,
        post_host_free = post.host_free_bytes,
        pre_device = pre.device_allocated_bytes,
        peak_device = peak.device_allocated_bytes,
        post_device = post.device_allocated_bytes)
    push!(rows, row)
    println(row)
    flush(stdout)
end

posts = getproperty.(rows, :post_device)
maximum(posts) - minimum(posts) <= DRIFT_TOLERANCE_BYTES || error(
    "post-release Metal allocation drift exceeded tolerance: $posts",
)
last(posts) <= baseline.device_allocated_bytes + DRIFT_TOLERANCE_BYTES || error(
    "final Metal allocation failed to return near baseline: " *
    "baseline=$(baseline.device_allocated_bytes), posts=$posts",
)
all(>(0), getproperty.(rows, :released)) || error(
    "one or more representative workspaces released no GPU buffers",
)
println((status = :pass, cycles = N_CYCLES,
    baseline_device = baseline.device_allocated_bytes,
    post_device = posts,
    rows))
