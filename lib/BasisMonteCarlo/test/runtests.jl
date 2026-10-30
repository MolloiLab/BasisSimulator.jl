using Test
using BasisMonteCarlo

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

    @testset "physics is fixed per process" begin
        @test_throws ErrorException simulate_monoenergetic(
            det, 60.0; n = 10,
            physics = PhysicsSettings(fluorescence = false)
        )
    end
end
