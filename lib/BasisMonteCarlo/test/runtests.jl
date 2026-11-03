using Test
using BasisMonteCarlo
using Random: Xoshiro

@testset "BasisMonteCarlo" begin
    @testset "materials" begin
        cdte = DetectorMaterial(:CdTe)
        @test sum(values(cdte.mass_fractions)) ≈ 1
        @test Set(e.element for e in k_edges(cdte)) == Set(["Cd", "Te"])
        @test isempty(filter(e -> e.element == "Ce", k_edges(DetectorMaterial(:Gemstone))))
        @test 0 < beer_lambert_efficiency(cdte, 1.0, 100) < 1
        @test linear_attenuation(cdte, 60; kind = :total) >
            linear_attenuation(cdte, 60; kind = :photoelectric)
    end

    det = SlabDetector(:CdTe, 1.0)

    @testset "interaction probability matches Beer-Lambert" begin
        for E in (40.0, 100.0)
            r = simulate_monoenergetic(det, E; n = 50_000)
            bl = beer_lambert_efficiency(det.material, det.thickness_mm, E)
            # 4σ of counting statistics plus 1% for Geant4-vs-XCOM cross sections
            @test abs(interaction_probability(r) - bl) <
                4standard_error(r, interaction_probability) + 0.01bl
        end
    end

    @testset "energy is conserved" begin
        r = simulate_monoenergetic(det, 60.0; n = 20_000)
        @test energy_budget(r).total ≈ 1 atol = 1.0e-4
        @test 0 < energy_absorption_efficiency(r) < quantum_efficiency(r) <= interaction_probability(r)
        @test 0 < swank_factor(r) <= 1
    end

    @testset "same seed, same photons" begin
        a = simulate_monoenergetic(det, 50.0; n = 2_000, seed = 3)
        b = simulate_monoenergetic(det, 50.0; n = 2_000, seed = 3)
        @test a.edep_keV == b.edep_keV
    end

    @testset "geometry changes re-initialise" begin
        thin = simulate_monoenergetic(SlabDetector(:CdTe, 0.1), 100.0; n = 20_000)
        thick = simulate_monoenergetic(SlabDetector(:CdTe, 2.0), 100.0; n = 20_000)
        @test interaction_probability(thin) < interaction_probability(thick)
    end

    @testset "track recording" begin
        t = record_tracks(det, 60.0; n = 5)
        @test !isempty(t.steps)
        @test any(s -> s.process == "phot", t.steps)
    end

    @testset "dexel readout" begin
        readout = DexelReadout(;
            pitch_x_mm = 0.275,
            pitch_y_mm = 0.322,
            cloud_sigma_um = 30.0,
            transport = nothing,
        )
        beam = AreaBeam(; width_mm = 0.275, height_mm = 0.322)
        run = simulate_monoenergetic(
            SlabDetector(:CdTe, 1.6),
            60.0;
            n = 5_000,
            beam,
            record_deposits = true,
        )
        response = dexel_response(run, readout)

        # The 3 × 3 neighbourhood collects at most what the photon left in the crystal
        collected = vec(sum(response.dexel_energy_keV; dims = (2, 3)))
        @test all(collected .<= run.edep_keV .+ 1.0f-3)
        @test sum(collected) / sum(run.edep_keV) > 0.95

        # Charge sharing: one photon can count in two dexels, and the centre counts most
        counts = threshold_counts(response, [20.0, 35.0])
        @test counts.total[1] > counts.per_dexel[2, 2, 1]
        @test counts.per_dexel[2, 2, 1] == maximum(counts.per_dexel[:, :, 1])
        @test counts.total[1] > counts.total[2]
    end

    @testset "parallel sweeps give the same photons" begin
        jobs = [(det, E) for E in (40.0, 60.0, 80.0)]
        serial = simulate_sweep(jobs; n = 2_000, seed = 5)
        start_workers(2)
        @test length(workers()) == 2
        @test start_workers(2) == workers()
        parallel = simulate_sweep(jobs; n = 2_000, seed = 5)
        @test all(a.edep_keV == b.edep_keV for (a, b) in zip(serial, parallel))
        stop_workers()
        @test isempty(workers())
    end

    @testset "a checkpointed sweep resumes with the same photons" begin
        jobs = [(det, E) for E in (40.0, 60.0, 80.0)]
        plain = simulate_sweep(jobs; n = 2_000, seed = 5)
        dir = mktempdir()

        first = simulate_sweep(jobs; n = 2_000, seed = 5, checkpoint = dir, checkpoint_batch = 2)
        @test length(filter(endswith(".jls"), readdir(dir))) == 2
        @test all(a.edep_keV == b.edep_keV for (a, b) in zip(plain, first))

        # a sweep that died after its first batch: only the second is rerun
        rm(joinpath(dir, last(sort(readdir(dir)))))
        resumed = simulate_sweep(jobs; n = 2_000, seed = 5, checkpoint = dir, checkpoint_batch = 2)
        @test all(a.edep_keV == b.edep_keV for (a, b) in zip(plain, resumed))

        # other photons are other batches
        other = simulate_sweep(jobs; n = 1_000, seed = 5, checkpoint = dir, checkpoint_batch = 2)
        @test length(filter(endswith(".jls"), readdir(dir))) == 4
        @test sum(other[1].edep_keV) != sum(plain[1].edep_keV)
        @test !any(endswith(".partial"), readdir(dir))
    end

    @testset "a material can leave the geometry and come back" begin
        # Regression: Geant4 crashed in the electron multiple scattering when a material returned
        gos = DetectorMaterial(:Gd2O2S)
        geant4_attenuation(gos, 60.0)
        simulate_monoenergetic(SlabDetector(:UFC, 1.4), 60.0; n = 200)
        run = simulate_monoenergetic(SlabDetector(gos, 0.12), 55.0; n = 2_000)
        @test energy_absorption_efficiency(run) > 0
    end

    @testset "energy-integrating dexels" begin
        pitch = (1.0, 1.1)
        septa = Septa(;
            pitch_x_mm = pitch[1],
            pitch_y_mm = pitch[2],
            width_x_mm = 0.1,
            width_y_mm = 0.1,
            material = DetectorMaterial(:TiO2_epoxy),
            cells = 9,
        )
        slab = SlabDetector(:UFC, 1.4; width_mm = 12.0, septa)
        readout = DexelEnergyReadout(
            DexelReadout(;
                pitch_x_mm = pitch[1],
                pitch_y_mm = pitch[2],
                cloud_sigma_um = 0.01,
                transport = nothing,
            )
        )
        beam = AreaBeam(; width_mm = 0.9, height_mm = 1.0)
        d = only(simulate_sweep([(slab, 60.0)]; n = 5_000, beam, readout))
        @test d isa DexelEnergy

        # The dexel the photons land on keeps less than the whole ceramic, the 3 × 3 at most all of it
        @test dexel_efficiency(d) < energy_absorption_efficiency(d)
        @test sum(d.dexel_keV) <= d.crystal_keV + 1.0e-6
        @test 0 < dexel_swank(d) <= swank_factor(d) <= 1
    end

    @testset "scintillator arc with a 1-D grid" begin
        arc = ArcDetector(;
            columns = 16,
            rows = 4,
            radius_mm = 1085.6,
            thickness_mm = 1.4,
            dexel_mm = (0.9, 1.0),
            group = (1, 1),
            gap_mm = (0.1, 0.1),
            septum_material = DetectorMaterial(:TiO2_epoxy),
            grid_height_mm = 10.0,
            grid_walls = :across,
            grid_wall_mm = (0.08, 0.08),
            material = DetectorMaterial(:UFC),
        )
        run = simulate_monoenergetic(
            arc,
            60.0;
            n = 5_000,
            beam = focal_spot(arc),
            record_deposits = true,
        )
        signal = arc_signal(run, ArcEnergyReadout())

        # Every deposit is in a dexel's ceramic, so the dexels share exactly the energy deposited
        @test sum(signal.energy_keV) ≈ sum(Float64, run.edep_keV) rtol = 1.0e-6
        @test signal.photon_keV ≈ 5_000 * 60.0
        @test all(>(0), sum(signal.energy_keV; dims = 2))
    end

    @testset "half-value layer" begin
        # One energy: the HVL is ln 2 / μ of aluminium
        al = DetectorMaterial(:Al)
        @test half_value_layer([60.0], [1.0]) ≈ log(2) / linear_attenuation(al, 60.0) rtol = 1.0e-6
        @test air_mu_en(30.0) ≈ 0.1537
    end

    @testset "phantoms: water-equivalent geometry" begin
        water = CylinderPhantom(300.0; centre_mm = (0.0, 600.0))
        @test water_equivalent_length(water, (0.0, 0.0), (0.0, 1.0)) ≈ 300.0 rtol = 1.0e-9
        @test water_equivalent_diameter(water) ≈ 300.0 rtol = 0.01

        # A ray through an insert: the lengths add up to the chord through the body
        rod = PhantomInsert(; material = DetectorMaterial(:CorticalBone), x_mm = 0.0, y_mm = 0.0, semi_axes_mm = (20.0, 20.0))
        p = CylinderPhantom(300.0; centre_mm = (0.0, 600.0), inserts = [rod])
        lengths = material_lengths(p, (0.0, 0.0), (0.0, 1.0))
        @test sum(last, lengths) ≈ 300.0
        @test last(lengths[2]) ≈ 40.0
        @test water_equivalent_length(p, (0.0, 0.0), (0.0, 1.0)) > 300.0
    end

    @testset "scatter: kernel geometry and the primary" begin
        @test BasisMonteCarlo.grid_acceptance(0.0, 10.0, 1.0) == 1.0
        @test BasisMonteCarlo.grid_acceptance(atan(0.05), 10.0, 1.0) ≈ 0.5
        @test BasisMonteCarlo.grid_acceptance(atan(0.2), 10.0, 1.0) == 0.0

        s = ScatterScanner(;
            name = "test",
            source_to_isocenter_mm = 600.0,
            source_to_detector_mm = 1080.0,
            fan_deg = 50.0,
            z_half_mm = 50.0,
            column_bin_deg = 1.0,
            row_bin_mm = 10.0,
        )
        D, u, E = 100.0, 0.5, 60.0
        run = simulate_monoenergetic(CylinderPhantom(D), E; n = 20_000, beam = kernel_ray(D, u), seed = 4)
        tally = only(scatter_tally(run, ScatterReadout([s])))

        # The primary lands in the column of the ray's angle, seen from the scanner's focal spot
        columns = vec(sum(tally.primary; dims = 2))
        expected_angle = rad2deg(asin(u * D / 2 / 600.0))
        @test tally.column_edges_deg[argmax(columns)] <= expected_angle < tally.column_edges_deg[argmax(columns) + 1]

        # ... and as often as Beer–Lambert says (energy-weighted, efficiency 1)
        chord = D * sqrt(1 - u^2)
        transmission = exp(-geant4_attenuation(DetectorMaterial(:Water), E) * chord)
        @test sum(tally.primary) / (run.n * E) ≈ transmission rtol = 0.05
        @test sum(tally.scatter) > 0
    end

    @testset "scatter: forced detection agrees with counting" begin
        water, bone = DetectorMaterial(:Water), DetectorMaterial(:CorticalBone)

        # The way out of a cylinder, and the angular distributions normalised over the sphere
        @test exit_length(CylinderPhantom(100.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) ≈ 50.0
        @test exit_length(CylinderPhantom(100.0; length_mm = 20.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)) ≈ 10.0
        tables = scattering_tables(water)
        for process in (1, 2), E in (20, 70, 140)
            @test 2π * (2 / tables.cells) * sum(tables.pdf[:, E, process]) ≈ 1.0 rtol = 1.0e-9
        end

        s = ScatterScanner(;
            name = "test",
            source_to_isocenter_mm = 600.0,
            source_to_detector_mm = 1080.0,
            fan_deg = 50.0,
            z_half_mm = 100.0,
        )
        readout = ForcedScatterReadout([s]; attenuation = forced_attenuation([water, bone]))
        rod = PhantomInsert(; material = bone, x_mm = 15.0, y_mm = 0.0, semi_axes_mm = (15.0, 15.0))

        # A kernel (its first scatterings integrated) and a phantom with an insert (all sampled):
        # the same photons counted one by one and by forced detection
        for (phantom, beam) in (
                (CylinderPhantom(150.0), kernel_ray(150.0, 0.3)),
                (CylinderPhantom(150.0; inserts = [rod]), kernel_ray(150.0, 0.1)),
            )
            run = simulate_monoenergetic(phantom, 60.0; n = 100_000, beam, seed = 5, record_vertices = true)
            tally = only(BasisMonteCarlo.reduce_run(run, readout))
            @test sum(tally.scatter) ≈ sum(tally.analog_scatter) rtol = 0.05
            @test tally.primary == only(scatter_tally(run, ScatterReadout([s]))).primary
        end
    end

    @testset "BasisSimulator v0.15.1's tables, kept for comparison" begin
        for t in BASISSIMULATOR_V0_15_1_EFFICIENCY
            @test t.energies == 1.0:140.0
            @test all(0 .< t.efficiency .<= 1)
        end
        # the Gd K-edge: the Force's efficiency drops between 50 and 51 keV
        ufc = BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc
        @test ufc.efficiency[51] < ufc.efficiency[50] - 0.2
        response = basissimulator_v0_15_1_response()
        @test response["energies_keV"] == 1.0:140.0
        @test size(response["R_total"]) == (140, length(response["thresholds_keV"]))
    end

    @testset "BasisSimulator v0.15.1's analytic scatter, kept for comparison" begin
        # Elliptical water phantoms across a fan, with a row-dependent ramp, and a synthetic
        # spectrum. The numbers were recorded from BasisSimulator v0.15.1's own
        # geometry_aware_scatter_model, estimate_scatter_field! and compute_scatter_energy_weights
        # on the same inputs (which the port matched bit for bit).
        function ellipse_sinogram(n_cols, views, sid, col, a, b)
            s = zeros(Float64, n_cols, 64, length(views))
            for (v, θ) in enumerate(views), r in 1:64, i in 1:n_cols
                t = sid * sin((i - (n_cols + 1) / 2) * (col / sid))
                ρ2 = (a * cos(θ))^2 + (b * sin(θ))^2
                L = abs(t) < sqrt(ρ2) ? 2 * a * b * sqrt(ρ2 - t^2) / ρ2 : 0.0
                s[i, r, v] = 0.019 * L * (1 + 0.3 * ((r - 32.5) / 64)^2)
            end
            return s
        end
        energies = collect(20.0:10.0:120.0)
        weights = [exp(-((E - 60) / 25)^2) for E in energies]
        cases = [
            # the library's SOMATOM Force entry (notebook 11): 2° columns
            (
                sid = 595.0,
                sdd = 1085.6,
                col = 595.0 * deg2rad(2.0),
                n_cols = 25,
                a = 150.0,
                b = 150.0,
                diameter_cm = 30.0,
                views = [0.0],
                recorded = [
                    (13, 32, 1) => 0.00031693335508647453,
                    (6, 1, 1) => 0.0038412715479984607,
                    (20, 64, 1) => 0.00384127154799846,
                ],
            ),
            # the library's Aquilion ONE entry (notebook 11)
            (
                sid = 600.0,
                sdd = 1072.0,
                col = 600.0 * deg2rad(2.0),
                n_cols = 25,
                a = 175.0,
                b = 125.0,
                diameter_cm = 29.6,
                views = [0.0, π / 3],
                recorded = [
                    (13, 32, 1) => 0.00071925506610508,
                    (6, 1, 1) => 0.0030834678829173264,
                    (20, 64, 2) => 0.00042855329010831674,
                ],
            ),
            # BasisSimulator's Force dexels (notebook 09): the kernel at its 63-dexel cap, and line
            # integrals past the clamp at 20
            (
                sid = 595.0,
                sdd = 1085.6,
                col = 0.561,
                n_cols = 920,
                a = 250.0,
                b = 200.0,
                diameter_cm = 45.0,
                views = [0.0, π / 2],
                recorded = [
                    (460, 32, 1) => 0.00010307154413174238,
                    (460, 32, 2) => 1.8746872182146465e-5,
                    (100, 5, 2) => 0.0035231987611380636,
                ],
            ),
        ]
        for c in cases
            sinogram = ellipse_sinogram(c.n_cols, c.views, c.sid, c.col, c.a, c.b)
            c.n_cols == 920 && (sinogram[460, 30:34, 2] .= 25.0)
            scatter = basissimulator_v0_15_1_scatter(
                sinogram;
                source_to_isocenter_mm = c.sid,
                source_to_detector_mm = c.sdd,
                detector_col_size_mm = c.col,
                phantom_diameter_cm = c.diameter_cm,
                energies,
                weights,
            )
            @test size(scatter) == size(sinogram)
            for (index, value) in c.recorded
                @test scatter[index...] ≈ value rtol = 1.0e-12
            end
        end
    end

    @testset "scatter library: each ray in its own cylinder" begin
        # A synthetic library whose kernel is the cylinder's diameter, in one bin: the scatter a
        # scan adds up is then the diameter each ray was given, weighted by its share
        s = ScatterScanner(;
            name = "test",
            source_to_isocenter_mm = 600.0,
            source_to_detector_mm = 1080.0,
            fan_deg = 41.0,
            z_half_mm = 50.0,
            column_bin_deg = 1.0,
            row_bin_mm = 10.0,
        )
        nc, nr = length(column_edges(s)) - 1, length(row_edges(s)) - 1
        diameters, impacts, energies = [200.0, 300.0, 400.0, 500.0], [0.0, 0.5, 0.95], [60.0, 80.0]
        S = zeros(Float32, nc, nr, length(impacts), length(diameters), length(energies))
        for (a, D) in enumerate(diameters)
            S[(nc + 1) ÷ 2, (nr + 1) ÷ 2, :, a, :] .= D
        end
        lib = ScatterLibrary(
            [s],
            diameters,
            impacts,
            energies,
            Dict("test" => S),
            Dict("test" => ones(Float32, length(impacts), length(diameters), length(energies))),
            fill(0.02, 150),
            Dict{String, Any}(),
        )
        scan(wed, rays) = library_scan(lib, "test", wed, rays; collimation_mm = 20.0, spectrum = ([70.0], [1.0]))
        ray(γ, wet) = (angle_deg = γ, wet_mm = wet, weight = 0.1)

        # A cylinder: no ray is longer than its diameter, so every ray keeps the WED's cylinder
        R = 150.0
        chord(γ) = 2sqrt(R^2 - (600.0 * sind(γ))^2)
        cylinder = [ray(γ, chord(γ)) for γ in (-12.0, -5.0, 0.0, 3.0, 11.0)]
        @test sum(scan(2R, cylinder).scatter) ≈ 2R * sum(r.weight for r in cylinder) rtol = 1.0e-5

        # A ray through 400 mm of a patient whose WED is 300 mm gets a 400 mm cylinder, not a
        # 300 mm one with its length cut short
        tall = [ray(-8.0, 250.0), ray(0.0, 400.0)]
        @test sum(scan(300.0, tall).scatter) ≈ 0.1 * (300.0 + 400.0) rtol = 1.0e-5
    end

    @testset "scatter library v2: water ellipses, Compton and Rayleigh apart" begin
        # A view's water ellipse from its rays' chords: an ellipse of water gives itself back
        ellipse = CylinderPhantom(; semi_axes_mm = (130.0, 180.0), centre_mm = (20.0, 600.0))
        rays = [
            (angle_deg = γ, wet_mm = BasisMonteCarlo.ellipse_chord((0.0, 0.0), (sind(γ), cosd(γ)), ellipse.centre_mm, ellipse.semi_axes_mm))
                for γ in -25:0.02:25
        ]
        e = view_ellipse(rays, 600.0)
        @test e.thickness_mm ≈ 360.0 rtol = 0.01
        @test e.width_mm ≈ 260.0 rtol = 0.03
        @test e.centre_mm ≈ 20.0 atol = 0.5

        # The kernels' geometry: an ellipse thickness × aspect × thickness; the ray from a focal spot,
        # passing the axis at its impact of the half-width the tilted ray sees, tilted as a fan's ray
        # is; and the impact of a given chord
        @test kernel_phantom(300.0, 1.25).semi_axes_mm == (187.5, 150.0)
        ray = kernel_ray(300.0, 1.25, 0.4)
        o, d = ray.origin_mm, ray.direction
        offset = abs(o[1] * d[2] - o[2] * d[1])
        t = asin(offset / BasisMonteCarlo.KERNEL_SOURCE_MM)
        @test offset ≈ 0.4 * sqrt(187.5^2 * cos(t)^2 + 150.0^2 * sin(t)^2) rtol = 1.0e-6
        @test d[1] ≈ offset / BasisMonteCarlo.KERNEL_SOURCE_MM
        # impact 1 grazes the edge and 0.99 still crosses it, for ellipses wider and thicker than wide
        for (a, b) in ((130.0, 260.0), (175.0, 130.0), (130.0, 130.0))
            @test BasisMonteCarlo.kernel_chord(a, b, 1.0) < 1.0e-3 * a
            @test BasisMonteCarlo.kernel_chord(a, b, 0.99) > 0.1 * a
        end
        @test BasisMonteCarlo.kernel_chord(150.0, 187.5, 0.0) ≈ 300.0
        u = BasisMonteCarlo.kernel_impact(150.0, 187.5, 200.0)
        @test BasisMonteCarlo.kernel_chord(150.0, 187.5, u) ≈ 200.0 rtol = 1.0e-6

        # Forced detection keeps the Rayleigh part apart: part of the scatter, never more
        water = DetectorMaterial(:Water)
        s = ScatterScanner(;
            name = "test",
            source_to_isocenter_mm = 600.0,
            source_to_detector_mm = 1080.0,
            fan_deg = 50.0,
            z_half_mm = 100.0,
        )
        a = forced_attenuation([water])[water.name]
        readout = ForcedScatterReadout(
            [s];
            water_mu_per_mm = a.total,
            water_compton_per_mm = a.compton,
            water_rayleigh_per_mm = a.rayleigh,
        )
        run = simulate_monoenergetic(kernel_phantom(150.0, 0.8), 60.0; n = 20_000, beam = kernel_ray(150.0, 0.8, 0.3), seed = 7, record_vertices = true)
        tally = only(BasisMonteCarlo.reduce_run(run, readout))
        @test 0 < sum(tally.rayleigh) < sum(tally.scatter)
        @test all(tally.rayleigh .<= tally.scatter .+ 1.0e-9)
        @test 0 < sum(tally.forward) < sum(tally.rayleigh)
        @test all(tally.forward .<= tally.rayleigh .+ 1.0e-9)

        # The lengths the detected photons travelled: the incident leg in the band of 60 keV, none
        # above it (a scattering never adds energy), and on average about the ellipse's size
        band = BasisMonteCarlo.path_band(60.0)
        @test all(tally.paths[:, (band + 1):end] .== 0)
        @test tally.paths[1, band] > 0
        mean_path = sum(tally.paths[1, :]) / (sum(tally.scatter) - sum(tally.rayleigh))
        @test 50 < mean_path < 600

        # Where in its column the scatter landed: within the column, on average and in spread
        mass = vec(sum(tally.scatter; dims = 2))
        hit = findall(>(0), mass)
        @test all(abs.(tally.column_moments[1, hit] ./ mass[hit]) .<= s.column_bin_deg / 2)
        @test all(tally.column_moments[2, hit] ./ mass[hit] .<= (s.column_bin_deg / 2)^2)

        # A ray through water makes exactly the scattering its kernel stands for: every ratio 1, and
        # it attenuates as water at every energy
        p = CylinderPhantom(200.0; centre_mm = (0.0, 600.0))
        r = process_ratios(p, (0.0, 0.0), (0.0, 1.0), Dict(water.name => a), 200.0)
        @test r.compton[60] ≈ 1.0
        @test r.rayleigh[60] ≈ 1.0
        @test r.rayleigh_forward[60] ≈ 1.0
        @test all(abs.(r.delta_mu_per_mm) .< 1.0e-12)

        # Bone attenuates far more than its water-equivalent length of water at 40 keV, as much at 70
        bone = DetectorMaterial(:CorticalBone)
        attenuation = forced_attenuation([water, bone])
        p_bone = CylinderPhantom(; material = bone, semi_axes_mm = (50.0, 50.0), centre_mm = (0.0, 600.0))
        wet = water_equivalent_length(p_bone, (0.0, 0.0), (0.0, 1.0))
        δ = process_ratios(p_bone, (0.0, 0.0), (0.0, 1.0), attenuation, wet).delta_mu_per_mm
        @test δ[BasisMonteCarlo.path_band(40.0)] > 0.005
        @test abs(δ[BasisMonteCarlo.path_band(70.0)]) < 0.001

        # Blending the kernels around a ray: each moved so its own ray lies under this one. Synthetic
        # kernels, a bump centred on each kernel's own ray, give one bump on the ray, no wider (the bump
        # spans a few of the 2° columns, so moving it by part of a column hardly widens it)
        thicknesses, aspects, impacts, energies = [200.0, 300.0], [1.0, 1.5], [0.0, 0.2, 0.4, 0.6, 0.8, 0.99], [50.0, 70.0]
        cols, rows = column_edges(s), row_edges(s)
        γ = (cols[1:(end - 1)] .+ cols[2:end]) ./ 2
        width = 6.0
        S = zeros(Float32, length(γ), length(rows) - 1, length(impacts), length(aspects), length(thicknesses), length(energies))
        for (iu, u) in enumerate(impacts), (ir, ar) in enumerate(aspects), (iT, T) in enumerate(thicknesses)
            own = asind(BasisMonteCarlo.kernel_offset(T / 2, BasisMonteCarlo.kernel_half_width(T, ar), u) / s.source_to_isocenter_mm)
            S[:, :, iu, ir, iT, :] .= exp.(-((γ .- own) ./ width) .^ 2)
        end
        nb = length(BasisMonteCarlo.PATH_ENERGY_EDGES_KEV) - 1
        lib = ShapeScatterLibrary(
            [s],
            thicknesses,
            aspects,
            impacts,
            energies,
            Dict(s.name => S),
            Dict(s.name => zero(S)),
            Dict(s.name => zero(S)),
            Dict(s.name => ones(Float32, length(impacts), length(aspects), length(thicknesses), length(energies))),
            Dict(s.name => zeros(Float32, 2, nb, length(impacts), length(aspects), length(thicknesses), length(energies))),
            Dict(s.name => zeros(Float32, length(γ), length(impacts), length(aspects), length(thicknesses), length(energies))),
            Dict(s.name => fill(Float32(s.column_bin_deg / sqrt(12)), length(γ), length(impacts), length(aspects), length(thicknesses), length(energies))),
            Dict(s.name => ones(Float32, 2, length(BasisMonteCarlo.SCATTER_ANGLE_EDGES_DEG) - 1, length(energies))),
            a.total,
            # a water that does not scatter: these kernels are made up, with no single scatter of
            # their own to give up, so the scans below test the blending alone
            zeros(150),
            zeros(150),
            Dict{String, Any}(),
        )
        function bump(angle, wet, ellipse)
            out = BasisMonteCarlo.library_scan(
                lib,
                s.name,
                [(angle_deg = angle, wet_mm = wet, weight = 1.0)];
                collimation_mm = 40.0,
                spectrum = ([60.0], [1.0]),
                ellipse,
            )
            profile = vec(sum(out.scatter; dims = 2))
            centre = sum(profile .* γ) / sum(profile)
            return centre, sqrt(sum(profile .* (γ .- centre) .^ 2) / sum(profile))
        end
        # a ray on a kernel of the grid, nothing blended, against one among eight of them
        on_grid = bump(
            asind(0.4 * 100.0 / s.source_to_isocenter_mm),
            BasisMonteCarlo.kernel_chord(100.0, 100.0, 0.4),
            (thickness_mm = 200.0, width_mm = 200.0, centre_mm = 0.0),
        )
        target = 8.0
        blended = bump(target, 180.0, (thickness_mm = 250.0, width_mm = 300.0, centre_mm = 0.0))
        @test blended[1] ≈ target atol = 0.05
        @test blended[2] ≈ on_grid[2] rtol = 0.05

        # A kernel whose scatter landed at one point of its column (no spread) moves there whole: a ray
        # anywhere in a column puts it all in that column, none in its neighbours
        O = zeros(Float32, length(γ), length(impacts), length(aspects), length(thicknesses), length(energies))
        point = ShapeScatterLibrary(
            [s],
            thicknesses,
            aspects,
            impacts,
            energies,
            Dict(s.name => S),
            Dict(s.name => zero(S)),
            Dict(s.name => zero(S)),
            lib.primary,
            lib.paths,
            Dict(s.name => O),
            Dict(s.name => zero(O)),
            lib.angles,
            a.total,
            zeros(150),
            zeros(150),
            Dict{String, Any}(),
        )
        S .= 0
        for (iu, u) in enumerate(impacts), (ir, ar) in enumerate(aspects), (iT, T) in enumerate(thicknesses)
            own = asind(BasisMonteCarlo.kernel_offset(T / 2, BasisMonteCarlo.kernel_half_width(T, ar), u) / s.source_to_isocenter_mm)
            c = searchsortedlast(cols, own)
            S[c, :, iu, ir, iT, :] .= 1
            O[c, iu, ir, iT, :] .= own - γ[c]
        end
        for target in (8.3, 9.7)
            out = BasisMonteCarlo.library_scan(
                point,
                s.name,
                [(angle_deg = target, wet_mm = 180.0, weight = 1.0)];
                collimation_mm = 40.0,
                spectrum = ([60.0], [1.0]),
                ellipse = (thickness_mm = 250.0, width_mm = 300.0, centre_mm = 0.0),
            )
            profile = vec(sum(out.scatter; dims = 2))
            @test profile[searchsortedlast(cols, target)] ≈ sum(profile) rtol = 1.0e-9
        end

        # A ray left of the axis takes its kernels' mirror image, by its side, even through the axis
        # (impact 0): a kernel at impact 0 whose scatter lies 2° to one side of its ray lands 2°
        # farther out on either side
        S .= 0
        for (iu, u) in enumerate(impacts), (ir, ar) in enumerate(aspects), (iT, T) in enumerate(thicknesses)
            own = asind(BasisMonteCarlo.kernel_offset(T / 2, BasisMonteCarlo.kernel_half_width(T, ar), u) / s.source_to_isocenter_mm)
            S[:, :, iu, ir, iT, :] .= exp.(-((γ .- own .- 2.0) ./ width) .^ 2)
        end
        @test BasisMonteCarlo.kernel_impact(150.0, 187.5, BasisMonteCarlo.kernel_chord(150.0, 187.5, 0.0)) == 0.0
        for side in (1, -1)
            out = BasisMonteCarlo.library_scan(
                lib,
                s.name,
                [(angle_deg = 8.0 * side, wet_mm = 300.0, weight = 1.0)];
                collimation_mm = 40.0,
                spectrum = ([60.0], [1.0]),
                ellipse = (thickness_mm = 250.0, width_mm = 300.0, centre_mm = 0.0),
            )
            profile = vec(sum(out.scatter; dims = 2))
            @test sum(profile .* γ) / sum(profile) ≈ 10.0 * side atol = 0.3
        end

        # Single scatter by quadrature: in a kernel's water ellipse as forced detection integrates it
        # along the kernel's ray, and the same in that ellipse as its rays describe it
        T, ar, u, E = 260.0, 1.25, 0.4, 60.0
        phantom, beam = kernel_phantom(T, ar), kernel_ray(T, ar, u)
        run = (beam = beam, energy_keV = E, n = 200_000, photon_energy_keV = nothing, seed = 1)
        cols_s, rows_s = column_edges(s), row_edges(s)
        blank = zeros(length(cols_s) - 1, length(rows_s) - 1)
        analog = ScatterTally(s.name, run.n, 0.0, cols_s, rows_s, blank, blank, zeros(150), blank, blank, zeros(2, 7), blank, zeros(2, 30), zeros(2, length(cols_s) - 1))
        frames = [BasisMonteCarlo.detector_frame(s, run, phantom, analog)]
        BasisMonteCarlo.first_scatter!(frames, run, phantom, BasisMonteCarlo.phantom_media(phantom, readout), readout, Xoshiro(3))
        forced = frames[1].scatter ./ run.n
        water_parts = (total = a.total, compton = a.compton, rayleigh = a.rayleigh, tables = BasisMonteCarlo.cached_scattering_tables(water))
        a_k, b_k = T / 2, BasisMonteCarlo.kernel_half_width(T, ar)
        node = asind(BasisMonteCarlo.kernel_offset(a_k, b_k, u) / s.source_to_isocenter_mm)
        q = only(BasisMonteCarlo.single_scatter(BasisMonteCarlo.WaterEllipse(a_k, b_k, s.source_to_isocenter_mm), s, node, [E], water_parts))
        quadrature = q.compton .+ q.wide .+ q.forward
        @test sum(quadrature) ≈ sum(forced) rtol = 0.03
        @test sum(q.forward) ≈ sum(frames[1].forward) / run.n rtol = 0.03
        centre = (0.0, s.source_to_isocenter_mm)
        strips = BasisMonteCarlo.RayStrips(
            [
                (angle_deg = g, wet_mm = BasisMonteCarlo.ellipse_chord((0.0, 0.0), (sind(g), cosd(g)), centre, (b_k, a_k)))
                    for g in -30:0.1:30
            ],
            s.source_to_isocenter_mm,
        )
        r = only(BasisMonteCarlo.single_scatter(strips, s, node, [E], water_parts))
        @test sum(r.compton .+ r.wide .+ r.forward) ≈ sum(quadrature) rtol = 0.02

        # A single ray describes no patient: it stands in its ellipse, and its kernels keep their own
        # single scatter (through a water that scatters, the scan is the same as through one that
        # does not)
        S .= 1.0f-3
        water_lib = ShapeScatterLibrary(
            [s],
            thicknesses,
            aspects,
            impacts,
            energies,
            Dict(s.name => S),
            Dict(s.name => zero(S)),
            Dict(s.name => zero(S)),
            lib.primary,
            lib.paths,
            lib.offsets,
            lib.spreads,
            lib.angles,
            a.total,
            a.compton,
            a.rayleigh,
            Dict{String, Any}(),
        )
        one_ray(library) = BasisMonteCarlo.library_scan(
            library,
            s.name,
            [(angle_deg = 3.0, wet_mm = 200.0, weight = 1.0)];
            collimation_mm = 40.0,
            spectrum = ([60.0], [1.0]),
            ellipse = (thickness_mm = 250.0, width_mm = 300.0, centre_mm = 0.0),
        ).scatter
        @test sum(one_ray(lib)) > 0
        @test one_ray(water_lib) == one_ray(lib)
    end

    @testset "charge transport: induction and trapping" begin
        p = ChargeTransport()
        L = p.thickness_mm
        wx, wy = 0.275, 0.322
        # The weighting potentials of all the pixels add up to a whole-face electrode's: z / L
        for z in (0.3, 1.2)
            s = sum(weighting_potential(i * wx, j * wy, L - z, wx, wy, L) for i in -40:40, j in -40:40)
            @test s ≈ z / L atol = 0.005
        end
        @test weighting_potential(0.0, 0.0, 1.0e-4, wx, wy, L) ≈ 1 atol = 0.01
        @test abs(weighting_potential(0.0, 0.0, L, wx, wy, L)) < 1.0e-12

        # Narrower clouds than v0.15.1's 30 µm: more photons keep their full energy in one dexel
        crystal = SlabDetector(:CdTe, 1.6)
        beam = AreaBeam(; width_mm = wx, height_mm = wy)
        run = simulate_monoenergetic(crystal, 60.0; n = 5_000, beam, seed = 2, record_deposits = true)
        # the charge is transported by default; the fixed cloud is asked for
        @test DexelReadout(; pitch_x_mm = wx, pitch_y_mm = wy).transport == ChargeTransport()
        fixed = dexel_response(run, DexelReadout(; pitch_x_mm = wx, pitch_y_mm = wy, transport = nothing))
        moved = dexel_response(run, DexelReadout(; pitch_x_mm = wx, pitch_y_mm = wy))
        full(r) = count(>(58.0f0), r.dexel_energy_keV[:, 2, 2])
        @test full(moved) > full(fixed)

        # Nothing is created and little is lost: over 9 × 9 dexels the recorded charge is the
        # deposited energy less trapping (a few %)
        wide = dexel_response(run, DexelReadout(; pitch_x_mm = wx, pitch_y_mm = wy, neighborhood = 9, transport = ChargeTransport()))
        ratio = sum(Float64, wide.dexel_energy_keV) / sum(Float64, run.deposits.energy_keV)
        @test 0.95 < ratio <= 1.0
    end

    @testset "electronics: shaped pulses, comparators, variation" begin
        # A CR-RC pulse is 2.446 τ wide at half its height and peaks at 1 after τ
        @test BasisMonteCarlo.shaper_fwhm(1) ≈ 2.4464 atol = 1.0e-3
        @test BasisMonteCarlo.shaper_pulse(1.0, 1.0, 1) ≈ 1

        # Pulses far apart: every pulse above a threshold counts, none below
        values, cdf = [50.0], [1.0]
        alone = PulseShaper(; fwhm_ns = 10.0, dead_ns = 20.0)
        @test pulse_counts(Xoshiro(1), alone, 1.0e-6, 500, values, cdf, [20.0, 60.0]) == [500, 0]

        # Narrow pulses and a long dead time: a non-paralysable counter, n / (1 + rate × dead)
        rate, dead = 0.01, 50.0
        narrow = PulseShaper(; fwhm_ns = 0.5, dead_ns = dead)
        counted = pulse_counts(Xoshiro(2), narrow, rate, 20_000, values, cdf, [20.0]; dt_ns = 0.05)[1]
        @test counted / 20_000 ≈ 1 / (1 + rate * dead) rtol = 0.03

        # Two pulses that overlap count once with a plain comparator, twice when it retriggers
        piled(retrigger) = pulse_counts(Xoshiro(3), PulseShaper(; fwhm_ns = 10.0, dead_ns = 0.0, retrigger), 0.2, 20_000, values, cdf, [20.0])[1]
        @test piled(false) < piled(true)

        # Without noise, a gain g at threshold T counts what the nominal dexel counts at T / g; a
        # threshold offset δ counts what it counts at T + δ
        heights, p = collect(10.0:1.0:100.0), fill(1 / 91, 91)
        quiet = (; electronic_noise_keV = 1.0e-6, fano = 0.0)
        @test counts_above(heights, p, [40.5]; variation = DexelVariation(; gain = 1.1), quiet...) ≈
            counts_above(heights, p, [40.5 / 1.1]; quiet...)
        @test counts_above(heights, p, [40.5]; variation = DexelVariation(; threshold_offset_keV = 2.0), quiet...) ≈
            counts_above(heights, p, [42.5]; quiet...)
        @test counts_above(heights, p, [0.0])[1] ≈ 1 atol = 1.0e-6

        # A dexel's pulses per photon: at least one when the photon is absorbed (60 keV in 1.6 mm CdTe), at
        # most one in each dexel of the 3 × 3
        crystal = SlabDetector(:CdTe, 1.6)
        run = simulate_monoenergetic(crystal, 60.0; n = 2_000, beam = AreaBeam(; width_mm = 0.275, height_mm = 0.322), seed = 4, record_deposits = true)
        h = pulse_heights(dexel_response(run, DexelReadout(; pitch_x_mm = 0.275, pitch_y_mm = 0.322, transport = nothing)))
        @test interaction_probability(run) < sum(h) <= 9
    end

    @testset "physics is fixed per process" begin
        @test_throws ErrorException simulate_monoenergetic(
            det, 60.0; n = 10,
            physics = PhysicsSettings(fluorescence = false)
        )
    end
end
