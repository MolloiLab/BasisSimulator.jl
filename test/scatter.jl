# The Geant4 scatter library in `simulate!` (src/detector/scatter_kernels.jl, src/detector/scatter.jl),
# on a small synthetic library written here in the library's own on-disk format (a TOML header and
# little-endian Float32 arrays): the reader's round trip, the interpolation and its clamping, the
# kernels' single scatter swapped for the rays' own, the mirror image by the ray's side, a grazing
# ray's impact, the error for a scanner with no kernels, a centred water cylinder's symmetric
# scatter, linearity in mAs, the two detector families' units, the data-driven correction, and
# CPU/GPU agreement. The library is scatter_library.jl's (`_SCATTER_TEST_ENTRY`, `"BS test counter"`,
# `_SCATTER_AS_MADE_ENTRY`).

# A water cylinder (diameter in cm) or ellipse on a 0.25 cm grid, centred or moved across x; with
# `lungs`, two lungs (0.26 g/cm³) either side of a water mediastinum.
function _scatter_phantom(;
    a_cm,
    b_cm = a_cm,
    shift_cm = 0.0,
    lungs = false,
    n = 160,
    nz = 8,
)
    vox = (0.25, 0.25, 0.25)
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    for k in 1:nz, j in 1:n, i in 1:n
        x, y = (i - c) * vox[1] - shift_cm, (j - c) * vox[2]
        (x / (a_cm / 2))^2 + (y / (b_cm / 2))^2 <= 1 || continue
        mask[i, j, k] = 0x01
        lungs &&
            ((abs(x) - a_cm / 4) / (a_cm / 7))^2 + (y / (b_cm / 3))^2 <= 1 &&
            (mask[i, j, k] = 0x02)
    end
    materials = Dict{Int, Any}(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water)
    lungs && (materials[2] = BS.XA.Materials.lung)
    return BS.Phantom(mask, materials, vox)
end

_scatter_opts(; kw...) = BS.SimOptions(;
    use_noise = false,
    use_scatter = true,
    use_heel_effect = false,
    use_focal_spot = false,
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
    use_lag = false,
    scatter_kernels = _SCATTER_TEST_ENTRY,
)
const _SCATTER_RECON = BS.ReconOptions(;
    matrix_size = (128, 128, 1),
    fov_cm = 36.0,
    z_cm = 0.5,
)

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
    @test size(A["scatter"]) == (25, 11, 21, 6, 5, 3)
    bump, single = _SCATTER_TEST_LIBRARY.bump, _SCATTER_TEST_LIBRARY.single
    @test A["scatter"] ≈
        Float32.(bump .+ dropdims(sum(single; dims = 7); dims = 7)) rtol = 1.0e-6
    @test A["forward"] ≈ Float32.(0.1 .* bump .+ single[:, :, :, :, :, :, 3]) rtol = 1.0e-6
    @test header["arrays"]["elements"]["scattering"][8, 1, 3, 70] ≈ 1.0f0 + 0.08f0 + 0.3f0
    # water's angular tables: per steradian, each normalised over the sphere
    pdf = header["arrays"]["elements"]["water_pdf"]
    @test size(pdf) == (header["water_pdf_cells"], 150, 2)
    @test all(≈(1; rtol = 1.0e-5), 2π * (2 / size(pdf, 1)) .* sum(Float64.(pdf); dims = 1))
    only_entry = BS.read_scatter_library(_SCATTER_TEST_COUNTING; entry = "BS test counter")
    @test sort(collect(keys(only_entry["arrays"]))) == ["BS test counter", "elements"]

    k = BS.scatter_kernels(_SCATTER_TEST_ENTRY)
    @test k isa BS.ScatterKernels &&
        k.energy_weighted &&
        k.path == abspath(_SCATTER_TEST_HEADER)
    @test BS.scatter_kernels(_SCATTER_TEST_ENTRY) === k # cached
    @test k.grid === :both &&
        k.grid_opening_mm == (0.94, 1.01) &&
        k.detector_depth_mm == 1.4
    @test k.single_scatter ==
        (depths = 32, points_per_column = 4, points_per_row = 2, columns_each_side = 8)
    # Compton, wide Rayleigh and forward Rayleigh, the multiple scatter of each (its single scatter
    # taken out: the synthetic bump), all of it their sum, and the columns' moments of what is left
    for (p, share) in enumerate((0.75, 0.15, 0.1))
        # (row, column) first
        left = permutedims(k.kernels[:, :, :, p, :, :, :], (2, 1, 4, 5, 6, 3))
        @test maximum(abs.(left .- share .* bump)) < 1.0e-4 * maximum(bump)
    end
    @test k.kernels[:, :, :, 4, :, :, :] ≈
        dropdims(sum(k.kernels[:, :, :, 1:3, :, :, :]; dims = 4); dims = 4)
    @test k.moments[:, 1, 2, 1, 2, 3, 1] ≈
        vec(sum(k.kernels[:, :, 2, 1, 2, 3, 1]; dims = 1))
    # a water that does not scatter takes nothing out
    as_made = BS.scatter_kernels(_SCATTER_AS_MADE_ENTRY)
    @test as_made.kernels[:, :, 2, 4, 2, 3, 1] ≈ transpose(A["scatter"][:, :, 2, 3, 1, 2])
    # a library from before the single scatter's quadrature is refused, saying what it lacks
    old = BS.read_scatter_library(_SCATTER_TEST_HEADER)
    delete!(old["arrays"]["elements"], "water_pdf")
    delete!(old, "single_scatter")
    err = try
        BS.ScatterKernels(old, _SCATTER_TEST_ENTRY, _SCATTER_TEST_HEADER)
    catch e
        e
    end
    @test err isa ArgumentError &&
        occursin("single_scatter", err.msg) &&
        occursin("elements/water_pdf", err.msg)
    @test_throws ArgumentError BS.scatter_kernels(
        "No such scanner";
        path = _SCATTER_TEST_HEADER,
    )
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
    @test BS._kernel_impact(Float32(a), Float32(b), Float32(edge))[2] ≈
        BS._kernel_impact(a, b, edge)[2] rtol = 1.0e-5
    # longer than the axis ray
    @test BS._kernel_impact(a, b, 400.0) == (0.0, 1.0)
    # as long, to rounding
    @test BS._kernel_impact(a, b, 2a * (1 - 1.0e-10)) == (0.0, 1.0)
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

@testset "a grazing ray's impact, to the float type's precision" begin
    # a 16 cm circle crossed for a millimetre down to 30 µm: √(1 − u²) against its exact value (the
    # chord's own formula bisected in BigFloat), in double and in single precision
    a = b = 160.1
    for L in (1.0, 0.3, 0.1, 0.03)
        lo, hi = big(0.0), big(1.0)
        for _ in 1:200
            mid = (lo + hi) / 2
            BS._kernel_chord(big(a), big(b), sqrt(1 - mid^2), mid) < L ?
                (lo = mid) :
                (hi = mid)
        end
        q_exact = (lo + hi) / 2
        u, q = BS._kernel_impact(a, b, L)
        @test q ≈ q_exact rtol = 1.0e-12
        @test u ≈ sqrt(1 - q_exact^2) rtol = 1.0e-15
        _, q32 = BS._kernel_impact(Float32(a), Float32(b), Float32(L))
        @test q32 ≈ q_exact rtol = 1.0e-6
    end
end

@testset "single scatter: each kernel's own swapped for the ray's own in the patient" begin
    k = BS.scatter_kernels(_SCATTER_TEST_ENTRY)
    tables = BS._single_tables(
        Float64;
        cols = k.column_edges_deg,
        rows = k.row_edges_mm,
        sdd = k.source_to_detector_mm,
        energies = k.energies_keV,
        water = (
            mu = k.water_mu_per_mm,
            compton = k.water_compton_per_mm,
            rayleigh = k.water_rayleigh_per_mm,
        ),
        water_pdf = k.water_pdf,
        detector = (
            efficiency = k.efficiency,
            mu = k.detector_mu_per_mm,
            depth = k.detector_depth_mm,
        ),
        energy_weighted = k.energy_weighted,
        grid = k.grid,
        grid_height = k.grid_height_mm,
        grid_opening = k.grid_opening_mm,
        single = k.single_scatter,
        forward_deg = k.forward_rayleigh_deg,
    )
    sid = k.source_to_isocenter_mm

    # The quadrature in a kernel's water ellipse, and through the strips that ellipse's own rays
    # (every 0.1°) describe: the same, but for the strips' width at a grazing ray
    for (T, aspect, u) in (
            (300.0, 1.0, 0.0),
            (300.0, 1.0, 0.5),
            (140.0, 2.5, 0.5),
            (500.0, 0.6, 0.5),
        )
        a, b = T / 2, BS._kernel_half_width(T, aspect)
        angle = asind(BS._kernel_offset(a, b, u) / sid)
        ellipse, strips = zeros(11, 25, 3, 3), zeros(11, 25, 3, 3)
        BS._single_scatter!(ellipse, BS._WaterEllipse(a, b, sid), angle, tables)
        γs = collect(-40.0:0.1:40.0)
        chord(γ) = BS._exit_wet(
            BS._WaterEllipse(a, b, sid),
            (BS._dd(0.0), BS._dd(0.0)),
            (BS._dd(sind(γ)), BS._dd(cosd(γ))),
        )
        wet = reshape(chord.(γs), :, 1, 1)
        pairs, boundaries = BS._ray_pairs(Float64, γs, sid)
        described = zeros(4, length(γs))
        BS._scatter_strips!(
            described,
            pairs,
            wet,
            wet,
            BS._pair(Float64, sid),
            BS._pair(Float64, Float64(π)),
        )
        BS._single_scatter!(
            strips,
            BS._RayStrips(described, boundaries, 0, length(γs), γs[1], 0.1),
            angle,
            tables,
        )
        @test sum(ellipse) > 0
        @test sum(strips) ≈ sum(ellipse) rtol = 0.02
        # the forward peak, bin by bin
        @test strips[:, :, :, 3] ≈ ellipse[:, :, :, 3] rtol = 0.02
    end

    # A water patient that is its kernels' water: the same field as kernels that keep their own
    # single scatter (a water that does not scatter in the quadrature), to the quadrature's and the
    # grid's error; a thorax-like patient, lungs either side of a water mediastinum: less of it in
    # the middle (what a ray beside the mediastinum sends inward stops there), more beside it (what
    # it sends out through a lung escapes)
    as_made = BS.EICTScanner(_SCATTER_EICT; scatter_kernels = _SCATTER_AS_MADE_ENTRY)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 36)
    function field(scanner, phantom)
        p = BS.create_workspace(
            scanner,
            protocol,
            _scatter_opts(),
            _SCATTER_RECON,
            phantom,
        ).scatter
        BS._scatter_forward!(p, phantom)
        return Array(p.field)[:, :, 1, 1]
    end
    water = _scatter_phantom(; a_cm = 30.0)
    swapped, kept = field(_SCATTER_EICT, water), field(as_made, water)
    @test sum(swapped) ≈ sum(kept) rtol = 0.015
    strong = kept .> 0.1 * maximum(kept)
    @test maximum(abs.(swapped[strong] ./ kept[strong] .- 1)) < 0.06
    centre = 12:14
    mid = (size(kept, 2) + 1) ÷ 2
    @test sum(swapped[centre, mid]) ≈ sum(kept[centre, mid]) rtol = 0.01
    lungs = _scatter_phantom(; a_cm = 30.0, lungs = true)
    swapped, kept = field(_SCATTER_EICT, lungs), field(as_made, lungs)
    @test sum(swapped[centre, mid]) < 0.9 * sum(kept[centre, mid])
    @test sum(swapped[[9, 10, 16, 17], mid]) > 1.05 * sum(kept[[9, 10, 16, 17], mid])
end

@testset "a ray left of the axis takes its kernels' mirror image, by its side" begin
    # A centred water cylinder of 12 cm radius on every coarse ray, and one ray 6° to one side crossing
    # more water than the ellipse is thick, so at its own ellipse's axis (impact 0): its kernels lie 2°
    # to its side of it, so their mirror image matters there too. The two sides are mirror images.
    air = _scatter_phantom(; a_cm = 0.0, n = 8)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 100.0, views = 8)
    plan = BS.create_workspace(
        _SCATTER_EICT,
        protocol,
        _scatter_opts(),
        _SCATTER_RECON,
        air,
    ).scatter
    sid = plan.kernels.source_to_isocenter_mm
    T = eltype(plan.scratch.wet)
    function scan(side)
        s = plan.scratch
        wet = [
            abs(sid * sind(γ)) < 120 ? 2 * sqrt(120^2 - (sid * sind(γ))^2) : 0.0
            for γ in plan.angle_deg
        ]
        i = argmin(abs.(plan.angle_deg .- 6.0 * side))
        wet[i] = 330.0
        copyto!(s.wet, repeat(T.(wet), 1, size(s.wet, 2), size(s.wet, 3)))
        fill!(s.plain, true)
        out = similar(
            plan.field,
            size(plan.field, 1),
            size(plan.field, 2),
            1,
            size(plan.field, 4),
        )
        for (v0, n_v) in BS._scatter_chunks(plan)
            BS._superpose!(out, plan, (s.wet,), s.wet, v0, n_v)
        end
        return (; out = Array(out), shape = Array(s.shapes[1])[:, i])
    end
    left, right = scan(-1), scan(1)
    @test left.shape[3] == 0 &&
        left.shape[4] == -1 &&
        right.shape[3] == 0 &&
        right.shape[4] == 1
    @test left.out ≈ reverse(right.out; dims = 1) rtol = 1.0e-4
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
    @test occursin("SOMATOM_FORCE", sprint(showerror, err)) &&
        occursin("use_scatter = false", sprint(showerror, err))
    # copies of a clinical scanner keep its kernels
    @test BS.SOMATOM_FORCE_B.scatter_kernels ==
        BS.SOMATOM_FORCE.scatter_kernels ==
        "SOMATOM Force"
    @test BS.NAEOTOM_ALPHA_UHR.scatter_kernels == "NAEOTOM Alpha"
    @test BS.EICTScanner().scatter_kernels == ""
    # on by default
    @test BS.SimOptions().use_scatter
    # so the default options on a scanner without kernels throw the same clear error
    @test_throws ArgumentError BS.create_workspace(
        plain,
        protocol,
        BS.SimOptions(),
        _SCATTER_RECON,
        phantom,
    )
end

@testset "superposition: symmetry, linearity in mAs, units" begin
    phantom = _scatter_phantom(; a_cm = 30.0)
    on = _eict_scatter_scan(_SCATTER_EICT, phantom)
    off = _eict_scatter_scan(_SCATTER_EICT, phantom; use_scatter = false)
    plan = on.ws.scatter
    field = Array(plan.field)

    # a centred water cylinder: every view alike (to its voxels), the profile symmetric across the fan
    @test all(
        isapprox(field[:, :, :, v], field[:, :, :, 1]; rtol = 0.01)
        for v in axes(field, 4)
    )
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
    ws_b = BS.create_workspace(
        _SCATTER_EICT,
        BS.CTProtocol(p4; mAs = 400.0),
        _scatter_opts(),
        _SCATTER_RECON,
        phantom,
    )
    BS.simulate!(ws_a, phantom, p4, _scatter_opts(); report_dose = false)
    BS.simulate!(
        ws_b,
        phantom,
        BS.CTProtocol(p4; mAs = 400.0),
        _scatter_opts();
        report_dose = false,
    )
    @test Array(ws_a.scatter.field) ≈ Array(ws_b.scatter.field)
    @test Array(ws_a.sinogram) ≈ Array(ws_b.sinogram) rtol = 1.0e-5

    # the energy integrator's units: S₁ in E_ref and S₂ as m₂/m₁ at each energy
    k = BS.scatter_kernels(_SCATTER_TEST_ENTRY)
    @test plan.channels[:, 1] ≈
        fill(1 / sum(on.ws.weights_norm .* on.ws.m1), length(on.ws.energies)) rtol = 1.0e-5
    @test plan.channels[:, 2] ./ plan.channels[:, 1] ≈
        (on.ws.m2 ./ on.ws.m1) ./ sum(on.ws.weights_norm .* on.ws.m1) rtol = 1.0e-5
end

@testset "the energy integrator's scatter correction, on and off" begin
    phantom = _scatter_phantom(; a_cm = 30.0)
    corrected = _eict_scatter_scan(_SCATTER_EICT, phantom)
    kept = _eict_scatter_scan(
        BS.EICTScanner(_SCATTER_EICT; scatter_correction = false),
        phantom,
    )
    free = _eict_scatter_scan(_SCATTER_EICT, phantom; use_scatter = false)
    @test corrected.ws.scatter_correction && !kept.ws.scatter_correction
    # the same scatter reaches the detector either way
    @test Array(kept.ws.scatter.field) == Array(corrected.ws.scatter.field)
    # on: the scatter-corrected projection is near the scatter-free one; off: the scatter stays
    # (the line integrals fall behind the patient, most where the primary is weakest)
    before = maximum(abs.(kept.sinogram .- free.sinogram))
    after = maximum(abs.(corrected.sinogram .- free.sinogram))
    @test before > 0.005
    @test all(kept.sinogram .<= free.sinogram .+ 1.0e-6)
    @test after < before / 4
end

@testset "any number of materials: copies of a material scatter as the material" begin
    # water and lung, each split across many labels of the same material: the walk carries one lane
    # per material (fewer materials than features) or one per feature (more), and either way the
    # scatter is the two-material phantom's
    small = _scatter_phantom(; a_cm = 30.0, lungs = true)
    m = Array(small.mask)
    function copies(n)
        mask = zeros(UInt16, size(m))
        for k in eachindex(m)
            m[k] == 0 && continue
            mask[k] = UInt16(m[k] == 1 ? 1 + 2 * (k % (n ÷ 2)) : 2 + 2 * (k % (n ÷ 2)))
        end
        materials = Dict{Int, Any}(0 => BS.XA.Materials.air)
        for l in 1:n
            materials[l] = isodd(l) ? BS.XA.Materials.water : BS.XA.Materials.lung
        end
        return BS.Phantom(mask, materials, small.voxel_size)
    end
    ref = _eict_scatter_scan(_SCATTER_EICT, small)
    # one lane per material; the orthonormal lanes
    for n in (8, 300)
        big = _eict_scatter_scan(_SCATTER_EICT, copies(n))
        lanes = big.ws.scatter.rays
        # one lane per material up to 16 (air is one), else water and lung: rank 2 beyond the exact three
        # 3 + 2, padded to 4k
        @test n < 16 ? lanes.lanes_Ct === nothing : size(lanes.lanes_Ct, 1) == 8
        # to rounding, but for the few rays whose rounding tips a discrete choice (a kernel's
        # nearest grid node): 3e-4 there, inside the library's own Monte Carlo precision
        @test isapprox(
            Array(big.ws.scatter.field),
            Array(ref.ws.scatter.field);
            rtol = 1.0e-3,
        )
        @test isapprox(big.sinogram, ref.sinogram; rtol = 1.0e-5)
    end
end

@testset "the scatter's lanes reproduce every feature" begin
    rng = MersenneTwister(855)
    nb, nE = 4, 10
    features(kind, wet, dmu, ratio) = (
        s = kind .!= 0;
        hcat(
            wet .* s,
            Float64.(s),
            Float64.(kind .== 2),
            dmu .* s,
            reshape(ratio, length(kind), :) .* s,
        )
    )
    function check(kind, wet, dmu, ratio)
        l = BS._scatter_feature_lanes(kind, wet, dmu, ratio)
        F = features(kind, wet, dmu, ratio)
        rebuilt = l.G === nothing ?
            permutedims(l.Ct) :
            l.Ct === nothing ? l.G : permutedims(l.Ct) * l.G
        scale = [max(maximum(abs, c), 1.0e-300) for c in eachcol(F)]
        @test maximum(abs.(rebuilt .- F) ./ scale') <= 1.0e-7
        # the water test's lane is exact
        @test rebuilt[:, 3] == F[:, 3]
        return l
    end
    kinds(n) = Int32[k == 1 ? 0 : (isodd(k) ? 1 : 2) for k in 1:n]
    # mixtures of four materials: few lanes whatever their number
    base = (rand(rng, 4), rand(rng, 4, nb), 1 .+ rand(rng, 4, nE, 3))
    for n in (10, 300)
        w = rand(rng, n, 4)
        w ./= sum(w; dims = 2)
        l = check(
            kinds(n),
            w * base[1],
            w * base[2],
            reshape(w * reshape(base[3], 4, :), n, nE, 3),
        )
        @test n <= 16 ? l.Ct === nothing && size(l.G, 1) == n : size(l.Ct, 1) == 3 + 4
    end
    # features of a rank no 64 lanes hold: a lane per material (no more materials than features)
    # or per feature
    nE = 30
    Q = 3 + nb + 3nE
    for n in (80, 200)
        l = check(kinds(n), rand(rng, n), rand(rng, n, nb), rand(rng, n, nE, 3))
        @test size(l.Ct, 1) == min(n, Q)
        @test (l.G === nothing) == (n > Q)
    end
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
    plan = BS.create_workspace(
        _SCATTER_EICT,
        protocol,
        _scatter_opts(),
        _SCATTER_RECON,
        air,
    ).scatter
    sid = plan.kernels.source_to_isocenter_mm
    T = eltype(plan.scratch.wet)

    # a centred water cylinder of radius R (mm) on every coarse ray, superposed as the correction's
    # water patient
    scan(R) = let s = plan.scratch
        wet = [
            abs(sid * sind(γ)) < R ? 2 * sqrt(R^2 - (sid * sind(γ))^2) : 0.0
            for γ in plan.angle_deg
        ]
        copyto!(s.wet, repeat(T.(wet), 1, size(s.wet, 2), size(s.wet, 3)))
        fill!(s.plain, true)
        out = similar(
            plan.field,
            size(plan.field, 1),
            size(plan.field, 2),
            1,
            size(plan.field, 4),
        )
        fill!(s.clamped, Int32(0))
        for (v0, n_v) in BS._scatter_chunks(plan)
            BS._superpose!(out, plan, (s.wet,), s.wet, v0, n_v)
        end
        (; out = Array(out), n = Int(sum(s.clamped)))
    end
    thin, inside, thickest = scan(35.0), scan(80.0), scan(400.0)
    @test thin.n > 0 && thickest.n > 0 && inside.n == 0
    @test 0 < sum(thin.out) < sum(inside.out) # less water, less scatter
    # the thickest kernels stand in
    @test all(isfinite, thickest.out) && sum(thickest.out) > 0
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
        (; ws, counts = [Float64.(Array(c)) for c in r.raw_counts], bins = [
            Array(b)
            for b in r.pcct_sino.bins
        ])
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
    @test plan.channels[e, :] ≈
        on.ws.R[ri, :] ./ BS.quantum_efficiency(det.material, det.thickness_mm, E)
    @test size(corrected.ws.scatter.estimate, 3) == 4
    # no correction, no buffers for it
    @test isempty(plan.estimate) && isempty(plan.measured)

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
        # an off-centre water ellipse with a bone insert: the materials' path as well as water's
        water = _scatter_phantom(; a_cm = 30.0, b_cm = 20.0, shift_cm = 2.0)
        mask = copy(water.mask)
        mask[70:80, 75:85, :] .= 0x02
        materials = Dict{Int, Any}(
            0 => BS.XA.Materials.air,
            1 => BS.XA.Materials.water,
            2 => BS.XA.Materials.corticalbone,
        )
        cpu_phantom = BS.Phantom(mask, materials, water.voxel_size)
        gpu_phantom = BS.Phantom(to_gpu(mask), materials, water.voxel_size)
        cpu = _eict_scatter_scan(_SCATTER_EICT, cpu_phantom)
        gpu = _eict_scatter_scan(_SCATTER_EICT, gpu_phantom)
        @test cpu.ws.scatter.rays.other
        @test Array(gpu.ws.scatter.field) ≈ Array(cpu.ws.scatter.field) rtol = 1.0e-5
        @test Array(gpu.ws.scatter.estimate) ≈ Array(cpu.ws.scatter.estimate) rtol = 1.0e-5
        @test gpu.sinogram ≈ cpu.sinogram rtol = 1.0e-5
    end
end

# The 16 materials of the compacted Gammex 472: under Julia 1.13, CUDA.jl 5 miscompiled the scatter
# estimate's per-material walk at exactly 16 lanes (0.12 in log transmission against the CPU; 15
# lanes were exact), so the three-material phantom above did not catch it. CUDA.jl 6 is the compat
# floor since 0.16.7.
@testset "CPU and GPU agree: 16 materials" begin
    if !(@isdefined(HAS_GPU)) || !HAS_GPU
        @warn "Skipping the 16-material scatter CPU/GPU agreement: no GPU backend."
    else
        g = BS.compact_materials(BS.create_gammex_472(
            n_voxels = 160,
            n_slices = 8,
            fov_cm = 40.0,
            z_cm = 2.0,
        ))
        @test length(g.materials) == 16
        materials = Dict{Int, Any}(i - 1 => m for (i, m) in enumerate(g.materials))
        cpu = _eict_scatter_scan(_SCATTER_EICT, BS.Phantom(g.mask, materials, g.voxel_size))
        gpu = _eict_scatter_scan(
            _SCATTER_EICT,
            BS.Phantom(to_gpu(g.mask), materials, g.voxel_size),
        )
        @test Array(gpu.ws.scatter.estimate) ≈ Array(cpu.ws.scatter.estimate) rtol = 1.0e-5
        @test gpu.sinogram ≈ cpu.sinogram rtol = 1.0e-5
    end
end

# A photon counter on a left-right symmetric patient: in a view along an axis the ray through the
# middle passes the ellipse's axis, and the side its kernels lean to was the sign of a rounding
# (±1e-4 mm), of opposite signs on the CPU and the GPU, so the device's scatter was the CPU's mirror
# image (2.6e-3 in log transmission). Within a thousandth of the ellipse's width every device takes
# the same side since 0.16.8.
@testset "CPU and GPU agree: photon counter, symmetric patient" begin
    if !(@isdefined(HAS_GPU)) || !HAS_GPU
        @warn "Skipping the photon-counter scatter CPU/GPU agreement: no GPU backend."
    else
        scanner = BS.PCCTScanner(
            BS.NAEOTOM_ALPHA;
            detector_rows = 4,
            detector_cols = 96,
            detector_col_size = 3.0,
            detector_row_size = 2.0,
            detector_col_offset = 0.0,
            pileup = false,
            gain_spread = 0.0,
            threshold_spread_keV = 0.0,
            scatter_kernels = "BS test counter",
        )
        phantom = _scatter_phantom(; a_cm = 30.0, lungs = true)
        materials = Dict{Int, Any}(i - 1 => m for (i, m) in enumerate(phantom.materials))
        protocol = BS.CTProtocol(; kVp = 120.0, mAs = 50.0, views = 24)
        opts = _scatter_opts()
        scan(device) = let ph = BS.Phantom(
            device(phantom.mask),
            materials,
            phantom.voxel_size,
        )
            ws = BS.create_workspace(scanner, protocol, opts, _SCATTER_RECON, ph)
            r = BS.simulate!(ws, ph, protocol, opts; report_dose = false)
            (; field = Array(ws.scatter.field), bins = [Array(b) for b in r.pcct_sino.bins])
        end
        cpu, gpu = scan(identity), scan(to_gpu)
        @test gpu.field ≈ cpu.field rtol = 1.0e-5
        @test all(isapprox(g, c; rtol = 1.0e-5) for (g, c) in zip(gpu.bins, cpu.bins))
    end
end
