# Gantry rotation during a view (`SimOptions(; view_samples, view_arc)`): the geometry rotation it
# samples, the sub-view offsets, and the blur itself on the energy-integrating and photon-counting
# paths — the intensity averaged over each view's arc, the workspace geometry restored afterwards.
# Uses `_toy_eict_setup` / `_toy_pcct_setup` from api.jl.

@testset "rotate_geometry turns every view about the axis" begin
    scanner = BS.EICTScanner(source_to_isocenter = 541.0, source_to_detector = 949.0,
        detector_rows = 4, detector_cols = 32, detector_row_size = 1.0, detector_col_size = 1.0)
    g = BS.CTGeometry(scanner; n_angles = 36, fov_cm = 20.0)
    Δ = g.angles[2] - g.angles[1]
    r = BS.rotate_geometry(g, Δ)
    # turned by one view spacing, view k is where view k + 1 was
    for M in (:source_positions, :detector_centers, :detector_u, :detector_v)
        @test getfield(r, M)[:, 1:(end - 1)] ≈ getfield(g, M)[:, 2:end] atol = 1.0e-9
    end
    @test r.angles ≈ g.angles .+ Δ
    @test BS.rotate_geometry(g, 2π).source_positions ≈ g.source_positions atol = 1.0e-9
    h = BS.CTGeometry(scanner; n_angles = 36, fov_cm = 20.0, z_cm = 1.0, pitch = 1.0)
    # on a helix the gantry z advances with the angle
    rh = BS.rotate_geometry(h, Δ)
    @test rh.source_positions[3, 1:(end - 1)] ≈ h.source_positions[3, 2:end] atol = 1.0e-9
end

@testset "view_sample_offsets sample the arc by the midpoint rule" begin
    scanner = BS.EICTScanner(source_to_isocenter = 541.0, source_to_detector = 949.0,
        detector_rows = 4, detector_cols = 32, detector_row_size = 1.0, detector_col_size = 1.0)
    g = BS.CTGeometry(scanner; n_angles = 40, fov_cm = 20.0)
    Δ = g.angles[2] - g.angles[1]
    @test BS.view_sample_offsets(g, BS.SimOptions()) == [0.0]
    @test BS.view_sample_offsets(g, BS.SimOptions(view_samples = 3)) ≈ [-Δ / 3, 0.0, Δ / 3]
    @test BS.view_sample_offsets(g, BS.SimOptions(view_samples = 2, view_arc = 0.5)) ≈ [-Δ / 8, Δ / 8]
    @test_throws ArgumentError BS.SimOptions(view_samples = 0)
    @test_throws ArgumentError BS.SimOptions(view_arc = 0.0)
    @test_throws ArgumentError BS.SimOptions(view_arc = 1.5)
end

# A small scan of the Gammex through either detector family, with `views` point views or
# `views` views integrated over their arcs (`sim_opts` keywords).
function _vi_setup(kind; views, kwargs...)
    scanner = kind === :eict ?
        BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4,
            detector_cols = 64, detector_row_size = 1.0, detector_col_size = 1.0,
            detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0) :
        BS.PCCTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4,
            detector_cols = 64, detector_row_size = 1.0, detector_col_size = 1.0,
            detector_material = :CdTe, detector_depth = 1.6, n_energy_bins = 4,
            energy_thresholds = [20.0, 35.0, 55.0, 70.0], pileup = false)
    protocol = BS.CTProtocol(mA = 2.5, kVp = 120.0, views = views, rotation_time = 0.5)
    sim_opts = BS.SimOptions(; merge((use_noise = false, use_scatter = false, use_lag = false,
        use_focal_spot = false, use_optical_crosstalk = false), kwargs)...)
    recon_opts = BS.ReconOptions(matrix_size = (32, 32, 2), fov_cm = 20.0)
    ph = BS.create_gammex_472(n_voxels = 64, fov_cm = 20.0, z_cm = 1.0)
    phantom = BS.Phantom(to_gpu(ph.mask), ph.materials, ph.voxel_size, ph.origin, ph.extent)
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    return (; protocol, sim_opts, phantom, ws)
end
_vi_run(f; kw...) = let r = BS.simulate!(f.ws, f.phantom, f.protocol, f.sim_opts; report_dose = false, kw...)
    f.ws isa BS.PCCTWorkspace ? [Array(b) for b in r.pcct_sino.bins] : [Array(f.ws.sinogram)]
end

@testset "view integration is the arc average of point views ($(kind))" for kind in (:eict, :pcct)
    # m sub-views of each of n views sit exactly on the point views of an m·n-view scan, at
    # angle j·Δ + (s − (m+1)/2)·Δ/m: averaging those transmissions is the definition
    n, m = 24, 5
    fine = _vi_setup(kind; views = n * m)
    arc = _vi_setup(kind; views = n, view_samples = m)
    geom_before = Array(arc.ws.geom_source_positions)
    pf = _vi_run(fine)
    pa = _vi_run(arc)
    @test Array(arc.ws.geom_source_positions) == geom_before       # the workspace geometry restored
    for (f, a) in zip(pf, pa)
        idx(j, s) = mod(m * (j - 1) + (s - (m + 1) ÷ 2), n * m) + 1
        ref = similar(a)
        for j in 1:n
            ref[:, :, j] .= -log.(sum(exp.(-Float64.(f[:, :, idx(j, s)])) for s in 1:m) ./ m)
        end
        @test a ≈ ref rtol = 1.0e-4
        @test sum(abs, diff(Float64.(a); dims = 3)) < sum(abs, diff(Float64.(f[:, :, 1:m:end]); dims = 3))
    end
    # a vanishing arc is the point view — away from the exactly diagonal views, where the
    # distance-driven projector switches its driving axis and a rotation of 1e-7 rad flips it
    point = _vi_run(_vi_setup(kind; views = n))
    tiny = _vi_run(_vi_setup(kind; views = n, view_samples = 3, view_arc = 1.0e-6))
    offdiag = [j for j in 1:n if mod(8 * (j - 1), n) != 0]
    @test all(t[:, :, offdiag] ≈ p[:, :, offdiag] for (t, p) in zip(tiny, point))
end

@testset "a projection shared by noise draws ($(kind))" for kind in (:eict, :pcct)
    # everything before the noise is the same for every draw: re-using it is the full simulation
    kw = (views = 24, view_samples = 3, use_scatter = true, use_focal_spot = true)
    clean = _vi_setup(kind; kw..., use_noise = false)
    r0 = BS.simulate!(clean.ws, clean.phantom, clean.protocol, clean.sim_opts; report_dose = false,
        keep_projection = true)
    for seed in (7, 8)
        full = _vi_setup(kind; kw..., use_noise = true, seed = seed)
        reuse = _vi_setup(kind; kw..., use_noise = true, seed = seed)
        @test _vi_run(reuse; projection = r0.projection) == _vi_run(full)
    end
    # and the noise-free draw from it is the noise-free simulation
    @test _vi_run(_vi_setup(kind; kw..., use_noise = false); projection = r0.projection) == _vi_run(clean)
end

@testset "a path cache is of point views" begin
    point = _vi_setup(:eict; views = 24)
    paths = BS.material_paths(point.ws, point.phantom)
    @test only(_vi_run(point; paths = paths)) ≈ only(_vi_run(_vi_setup(:eict; views = 24))) rtol = 1.0e-5
    arc = _vi_setup(:eict; views = 24, view_samples = 3)
    @test_throws ArgumentError _vi_run(arc; paths = paths)
end
