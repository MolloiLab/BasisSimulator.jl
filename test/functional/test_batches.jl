# Batch-as-data composition: Σ_b eict_batch_vol → eict_vol_to_hu ≡ eict_forward (full dense and
# windowed projector, with and without noise), and the batch programs' inputs are plain arrays.
@testset "Functional — view batches as data reproduce eict_forward" begin
    scanner = BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 64, detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 14, rotation_time = 0.5)
    recon = BS.ReconOptions(matrix_size = (24, 24, 2), fov_cm = 20.0, z_cm = 0.4)
    phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 24, n_slices = 2, fov_cm = 20.0, z_cm = 0.4))
    fr = BSF.onehot_fractions(phantom.mask, length(phantom.materials))
    for (noise, budget) in ((false, 512), (true, 512), (false, 0.3))        # 0.3 MB forces the windowed projector (a full view is ≈0.45 MB)
        opts = BS.SimOptions(use_noise = noise, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 7)
        pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon; view_batch = :auto, batch_budget_mb = budget)
        bs = BSF.eict_batches(pipe)
        @test sum(length(b.views) for b in bs) == 14
        budget < 1 && @test all(b.windowed for b in bs)
        ε, ε_e = noise ? BSF.draw_eict_noise(pipe; seed = 7) : (nothing, nothing)
        ref = BSF.eict_forward(fr, pipe, ε, ε_e)
        got = BSF.eict_forward_batched(fr, pipe, ε, ε_e)
        @test size(got) == size(ref)
        @test maximum(abs.(got .- ref)) ≤ 2e-2                                  # HU, summation order
        d = BSF.batch_data(bs[1], pipe)
        @test size(d.table, 2) == length(bs[1].views) && size(d.gt) == (12, length(bs[1].views))
    end
end
