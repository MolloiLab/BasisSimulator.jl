# The label-volume input: material_paths / eict_forward / pcct_forward on the integer label volume
# (one-hot chunks formed on device, `batching.mat` materials at a time) ≡ the dense one-hot fractions.
@testset "Functional — label-volume input ≡ one-hot fractions, material batches" begin
    scanner = BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 64, detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 12, rotation_time = 0.5)
    opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false)
    recon = BS.ReconOptions(matrix_size = (24, 24, 2), fov_cm = 20.0, z_cm = 0.4)
    phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 24, n_slices = 2, fov_cm = 20.0, z_cm = 0.4))
    n_mat = length(phantom.materials)
    for mb in (1, 3, n_mat)
        pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon; view_batch = 4)
        pipe = BSF.EICTPipeline{Float32, typeof(pipe.eict), typeof(pipe.fbp)}(pipe.geom, pipe.vol_shape, pipe.n_mat, pipe.volume_extent,
            pipe.eict, pipe.fbp, pipe.μ_water, pipe.recon_shape, merge(pipe.batching, (mat = mb,)), pipe.unrolled, pipe.loop)
        fr = BSF.onehot_fractions(phantom.mask, n_mat)
        P_fr = BSF.material_paths(fr, pipe); P_lb = BSF.material_paths(phantom.mask, pipe)
        @test size(P_lb) == size(P_fr)
        @test maximum(abs.(P_lb .- P_fr)) ≤ 1e-5 * max(1, maximum(abs.(P_fr)))
        hu_fr = BSF.eict_forward(fr, pipe); hu_lb = BSF.forward(phantom.mask, pipe)
        @test maximum(abs.(hu_lb .- hu_fr)) ≤ 1e-2
    end
    @test_throws DimensionMismatch BSF.material_paths(phantom.mask[1:end-1, :, :], BSF.eict_pipeline(phantom, scanner, protocol, opts, recon; view_batch = 4))
end
