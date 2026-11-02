# The Geant4 scatter library in `simulate!` (src/detector/scatter_kernels.jl, src/detector/scatter.jl),
# on a small synthetic library written here in the library's own on-disk format (a TOML header and
# little-endian Float32 arrays): the reader's round trip, the interpolation and its clamping, the
# error for a scanner with no kernels, a centred water cylinder's symmetric scatter, linearity in
# mAs, the two detector families' units, the data-driven correction, and CPU/GPU agreement.
# The library is scatter_library.jl's (`_SCATTER_TEST_ENTRY`, `"BS test counter"`).

# A water cylinder (diameter in cm) or ellipse on a 0.25 cm grid, centred or moved across x.
function _scatter_phantom(; a_cm, b_cm = a_cm, shift_cm = 0.0, n = 160, nz = 8)
    vox = (0.25, 0.25, 0.25)
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    for k in 1:nz, j in 1:n, i in 1:n
        x, y = (i - c) * vox[1] - shift_cm, (j - c) * vox[2]
        (x / (a_cm / 2))^2 + (y / (b_cm / 2))^2 <= 1 && (mask[i, j, k] = 0x01)
    end
    return BS.Phantom(mask, Dict{Int, Any}(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water), vox)
end

_scatter_opts(; kw...) = BS.SimOptions(;
    use_noise = false,
    use_scatter = true,
    use_heel_effect = false,
    use_focal_spot = false,
    use_lag = false,
    seed = 1,
    kw...,
)

const _SCATTER_EICT = BS.EICTScanner(
    BS.SOMATOM_FORCE;
    detector_rows = 4,
    detector_cols = 230,
    detector_col_size = 2.244,
    detector_row_size = 2.0,
    detector_col_offset = 0.0,
    electronic_noise = 0.0,
    scatter_kernels = _SCATTER_TEST_ENTRY,
)
const _SCATTER_RECON = BS.ReconOptions(; matrix_size = (128, 128, 1), fov_cm = 36.0, z_cm = 0.5)

# The EICT scan's S₁ (signal relative to the air beam, as exp(-sinogram) before the air
# calibration) with and without scatter, noise-free, read off the workspace.
function _eict_scatter_scan(scanner, phantom; mAs = 200.0, views = 36, kw...)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs, views)
    opts = _scatter_opts(; kw...)
    ws = BS.create_workspace(scanner, protocol, opts, _SCATTER_RECON, phantom)
    BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
    return (; ws, protocol, sinogram = Array(ws.sinogram))
end

@testset "the library on disk: reader round trip" begin
    header = BS.read_scatter_library(_SCATTER_TEST_HEADER)
    @test header["format_version"] == 2
    A = header["arrays"][_SCATTER_TEST_ENTRY]
    @test size(A["scatter"]) == (25, 11, 3, 3, 3, 3)
    γ = collect(-24.0:2.0:24.0)
    @test A["scatter"][13, 6, 1, 2, 2, 2] ≈ Float32(1.0e-4 * exp(-(γ[13] / 12)^2)) rtol = 1.0e-6
    @test A["forward"] ≈ 0.1f0 .* A["scatter"]
    @test header["arrays"]["elements"]["scattering"][8, 1, 3, 70] ≈ 1.0f0 + 0.08f0 + 0.3f0
    only_entry = BS.read_scatter_library(_SCATTER_TEST_COUNTING; entry = "BS test counter")
    @test sort(collect(keys(only_entry["arrays"]))) == ["BS test counter", "elements"]

    k = BS.scatter_kernels(_SCATTER_TEST_ENTRY)
    @test k isa BS.ScatterKernels && k.energy_weighted && k.path == abspath(_SCATTER_TEST_HEADER)
    @test BS.scatter_kernels(_SCATTER_TEST_ENTRY) === k                         # cached
    # Compton, wide Rayleigh and forward Rayleigh: the scatter less its Rayleigh part, the Rayleigh
    # less its forward part, the forward part
    S = A["scatter"][:, :, 2, 3, 1, 2]
    @test k.kernels[:, :, 2, 1, 2, 3, 1] ≈ transpose(0.75f0 .* S)                # (row, column)
    @test k.kernels[:, :, 2, 2, 2, 3, 1] ≈ transpose(0.15f0 .* S)
    @test k.kernels[:, :, 2, 3, 2, 3, 1] ≈ transpose(0.1f0 .* S)
    @test k.kernels[:, :, 2, 4, 2, 3, 1] ≈ transpose(S)
    @test k.moments[:, 1, 2, 1, 2, 3, 1] ≈ vec(sum(0.75f0 .* S; dims = 2))
    @test_throws ArgumentError BS.scatter_kernels("No such scanner"; path = _SCATTER_TEST_HEADER)
    @test_throws ArgumentError BS.scatter_kernels("SOMATOM Force; not an entry")
end

@testset "the kernels' geometry and interpolation" begin
    # the impact whose chord is a given length, in the kernel's own (tilted) ray
    a, b = 150.0, 200.0
    for L in (50.0, 150.0, 290.0)
        u, q = BS._kernel_impact(a, b, L)
        @test q ≈ sqrt(1 - u^2)
        @test BS._kernel_chord(a, b, u) ≈ L rtol = 1.0e-6
        # in single precision too, and at the edge, where 1 − u² is below its rounding
        u32, q32 = BS._kernel_impact(Float32(a), Float32(b), Float32(L))
        @test u32 ≈ u rtol = 1.0e-5
        @test q32 ≈ q rtol = 1.0e-5
    end
    edge = BS._kernel_chord(a, b, 1 - 1.0e-8)
    @test BS._kernel_impact(Float32(a), Float32(b), Float32(edge))[2] ≈ BS._kernel_impact(a, b, edge)[2] rtol = 0.02
    @test BS._kernel_impact(a, b, 400.0) == (0.0, 1.0)                          # longer than the axis ray
    @test BS._kernel_offset(a, b, 1.0) > BS._kernel_offset(a, b, 0.5) > 0
    # clamped linear interpolation
    @test BS._bracket([1.0, 2.0, 4.0], 3.0) == (2, 3, 0.5)
    @test BS._bracket([1.0, 2.0, 4.0], 0.0) == (1, 1, 0.0)
    @test BS._bracket([1.0, 2.0, 4.0], 9.0) == (3, 3, 0.0)
    # a view's water ellipse from its rays' lengths: an ellipse of water gives itself back
    sid = 595.0
    γ = range(-25, 25; length = 501)
    xs = sid .* sind.(γ)
    lengths = [abs(x) < 120 ? 2 * 90 * sqrt(1 - (x / 120)^2) : 0.0 for x in xs]
    e = BS._view_ellipse(xs, BS._ray_widths(xs), lengths)
    @test e.a ≈ 90 rtol = 2.0e-3
    @test e.b ≈ 120 rtol = 2.0e-3
    @test abs(e.centre) < 1.0e-6
end

@testset "a scanner with no kernels" begin
    plain = BS.EICTScanner(_SCATTER_EICT; scatter_kernels = "")
    phantom = _scatter_phantom(; a_cm = 20.0, n = 16)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 100.0, views = 8)
    err = try
        BS.create_workspace(plain, protocol, _scatter_opts(), _SCATTER_RECON, phantom)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("SOMATOM_FORCE", sprint(showerror, err)) && occursin("use_scatter = false", sprint(showerror, err))
    # copies of a clinical scanner keep its kernels
    @test BS.SOMATOM_FORCE_B.scatter_kernels == BS.SOMATOM_FORCE.scatter_kernels == "SOMATOM Force"
    @test BS.NAEOTOM_ALPHA_UHR.scatter_kernels == "NAEOTOM Alpha"
    @test BS.EICTScanner().scatter_kernels == ""
    @test !BS.SimOptions().use_scatter
end

@testset "superposition: symmetry, linearity in mAs, units" begin
    phantom = _scatter_phantom(; a_cm = 30.0)
    on = _eict_scatter_scan(_SCATTER_EICT, phantom)
    off = _eict_scatter_scan(_SCATTER_EICT, phantom; use_scatter = false)
    plan = on.ws.scatter
    field = Array(plan.field)

    # a centred water cylinder: every view alike (to its voxels), the profile symmetric across the fan
    @test all(isapprox(field[:, :, :, v], field[:, :, :, 1]; rtol = 0.01) for v in axes(field, 4))
    profile = field[:, (size(field, 2) + 1) ÷ 2, 1, 1]
    @test profile ≈ reverse(profile) rtol = 1.0e-4
    @test maximum(profile) > 0

    # the DAS subtracts what it estimates from the data: close to what was added, not exactly
    estimate = Array(plan.estimate)[:, :, 1, 1]
    centre = (size(field, 2) + 1) ÷ 2
    @test estimate[:, centre] ≈ field[:, centre, 1, 1] rtol = 0.05
    residual = on.sinogram .- off.sinogram
    @test 0 < maximum(abs.(residual)) < 0.01

    # linear in mAs: the scatter is in the units of the signal per incident photon
    S1(ws) = exp.(-Array(ws.sinogram))
    p4 = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 36)
    ws_a = BS.create_workspace(_SCATTER_EICT, p4, _scatter_opts(), _SCATTER_RECON, phantom)
    ws_b = BS.create_workspace(_SCATTER_EICT, BS.CTProtocol(p4; mAs = 400.0), _scatter_opts(), _SCATTER_RECON, phantom)
    BS.simulate!(ws_a, phantom, p4, _scatter_opts(); report_dose = false)
    BS.simulate!(ws_b, phantom, BS.CTProtocol(p4; mAs = 400.0), _scatter_opts(); report_dose = false)
    @test Array(ws_a.scatter.field) ≈ Array(ws_b.scatter.field)
    @test Array(ws_a.sinogram) ≈ Array(ws_b.sinogram) rtol = 1.0e-5

    # the energy integrator's units: S₁ in E_ref and S₂ as m₂/m₁ at each energy
    k = BS.scatter_kernels(_SCATTER_TEST_ENTRY)
    @test plan.channels[:, 1] ≈ fill(1 / sum(on.ws.weights_norm .* on.ws.m1), length(on.ws.energies)) rtol = 1.0e-5
    @test plan.channels[:, 2] ./ plan.channels[:, 1] ≈ (on.ws.m2 ./ on.ws.m1) ./ sum(on.ws.weights_norm .* on.ws.m1) rtol = 1.0e-5
end

@testset "an empty field gets no scatter" begin
    air = _scatter_phantom(; a_cm = 0.0, n = 8)
    on = _eict_scatter_scan(_SCATTER_EICT, air)
    off = _eict_scatter_scan(_SCATTER_EICT, air; use_scatter = false)
    @test all(iszero, Array(on.ws.scatter.field))
    @test maximum(abs.(on.sinogram .- off.sinogram)) < 1.0e-6
end

@testset "outside the grid: the nearest kernels, scaled below the thinnest" begin
    air = _scatter_phantom(; a_cm = 0.0, n = 8)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 100.0, views = 8)
    plan = BS.create_workspace(_SCATTER_EICT, protocol, _scatter_opts(), _SCATTER_RECON, air).scatter
    sid = plan.kernels.source_to_isocenter_mm
    T = eltype(plan.scratch.wet)

    # a centred water cylinder of radius R (mm) on every coarse ray, superposed as the correction's
    # water patient
    scan(R) = let s = plan.scratch
        wet = [abs(sid * sind(γ)) < R ? 2 * sqrt(R^2 - (sid * sind(γ))^2) : 0.0 for γ in plan.angle_deg]
        copyto!(s.wet, repeat(T.(wet), 1, size(s.wet, 2), size(s.wet, 3)))
        fill!(s.plain, true)
        BS._scatter_weights!(plan, 1)
        out = similar(plan.estimate)
        n = BS._superpose!(out, plan, s.wet)
        (; out = Array(out), n)
    end
    thin, inside, thickest = scan(35.0), scan(80.0), scan(400.0)
    @test thin.n > 0 && thickest.n > 0 && inside.n == 0
    @test 0 < sum(thin.out) < sum(inside.out)               # less water, less scatter
    @test all(isfinite, thickest.out) && sum(thickest.out) > 0          # the thickest kernels stand in
end

@testset "photon counter: each bin's counts, corrected from the counts" begin
    base = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA;
        detector_rows = 4,
        detector_cols = 96,
        detector_col_size = 3.0,
        detector_row_size = 2.0,
        detector_col_offset = 0.0,
        bowtie_filter = :none,
        pileup = false,
        gain_spread = 0.0,
        threshold_spread_keV = 0.0,
        scatter_correction = false,
        scatter_kernels = "BS test counter",
    )
    phantom = _scatter_phantom(; a_cm = 30.0)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 50.0, views = 24)
    scan(scanner; kw...) = let opts = _scatter_opts(; kw...)
        ws = BS.create_workspace(scanner, protocol, opts, _SCATTER_RECON, phantom)
        r = BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
        (; ws, counts = [Float64.(Array(c)) for c in r.raw_counts], bins = [Array(b) for b in r.pcct_sino.bins])
    end
    on = scan(base)
    off = scan(base; use_scatter = false)
    corrected = scan(BS.PCCTScanner(base; scatter_correction = true))
    plan = on.ws.scatter

    # a scattered photon's bins: the response table's share of a photon of its energy, per photon
    # the library counted
    det = on.ws.pcct_detector
    e = findfirst(>=(60.0), on.ws.energies)
    E = on.ws.energies[e]
    n_R = size(on.ws.R, 1)
    ri = clamp(round(Int, (E - 1) / (on.ws.kVp - 1) * (n_R - 1)) + 1, 1, n_R)
    @test plan.channels[e, :] ≈ on.ws.R[ri, :] ./ BS.quantum_efficiency(det.material, det.thickness_mm, E)
    @test size(plan.estimate, 3) == 4

    # the counts the scatter adds, bin by bin, at the centre of the detector
    photons = BS.compute_detector_I0(on.ws.geom, protocol, sum(on.ws.weights))
    c, r = 48, 2
    field = Array(plan.field)
    for b in 1:4
        added = on.counts[b][c, r, 1] - off.counts[b][c, r, 1]
        @test added > 0
        @test added ≈ photons * ((field[13, 6, b, 1] + field[13, 7, b, 1]) / 2) rtol = 0.05
    end
    # corrected from the counts alone: most of the scatter gone, not all
    before = maximum(maximum(abs.(on.bins[b] .- off.bins[b])) for b in 1:4)
    after = maximum(maximum(abs.(corrected.bins[b] .- off.bins[b])) for b in 1:4)
    @test after < before / 4
    @test after > 0
end

@testset "CPU and GPU agree" begin
    if !(@isdefined(HAS_GPU)) || !HAS_GPU
        @warn "Skipping the scatter CPU/GPU agreement: no GPU backend."
    else
        cpu_phantom = _scatter_phantom(; a_cm = 30.0, b_cm = 20.0, shift_cm = 2.0)
        gpu_phantom = BS.Phantom(to_gpu(cpu_phantom.mask), cpu_phantom.materials, cpu_phantom.voxel_size, cpu_phantom.origin, cpu_phantom.extent)
        cpu = _eict_scatter_scan(_SCATTER_EICT, cpu_phantom)
        gpu = _eict_scatter_scan(_SCATTER_EICT, gpu_phantom)
        @test Array(gpu.ws.scatter.field) ≈ Array(cpu.ws.scatter.field) rtol = 1.0e-3
        @test Array(gpu.ws.scatter.estimate) ≈ Array(cpu.ws.scatter.estimate) rtol = 1.0e-3
        @test gpu.sinogram ≈ cpu.sinogram rtol = 1.0e-4
    end
end
