### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 171294a2-26bd-49e2-ac92-9df48ae5444f
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 69358294-97f2-4782-94d7-c29c747c45f4
using Markdown: @md_str, Markdown

# ╔═╡ 9ae27110-5c47-442b-a98e-d137599570f2
using Statistics: mean, std, quantile, median, cov, cor

# ╔═╡ d3054785-9e00-4094-a491-088ce63be9dc
md"""
# Photon-Counting CT Virtual Monoenergetic Imaging

Siemens Naeotom Alpha photon-counting CT simulation (140 kVp / 174 mA,
4-threshold acquisition, Gammex 472 phantom) with a tried cong approach direct in the notebook
"""

# ╔═╡ 3d515abe-f3d9-4ce5-96c7-bef7da9bf294
md"""
## Notebook Setup
"""

# ╔═╡ 492bb299-678d-4e6f-8c21-1e9178cc2beb
import PlutoUI

# ╔═╡ 9f8d5cd4-147e-4359-95bc-cc096a53f0e7
import BasisSimulator as BS

# ╔═╡ 2ff539c9-a678-403c-b629-8068a332a0e9
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
PlutoUI.TableOfContents()

# ╔═╡ 86c52e9e-7987-4504-93e6-128017f5e703
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 551f84fe-d7b4-48f9-a475-0c63178a6ede
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 59a5079b-a711-4f28-b3d6-665f0d91fb72
md"""
## Scan Setup and Simulation
"""

# ╔═╡ 4d519e7e-e412-45d8-a54e-97d8f9fe5cf7
md"""
### 01. `Phantom()` Struct
**Gammex 472**
"""

# ╔═╡ 939dcda3-9be5-46c8-aaa1-ded273e8cf04
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 5248ba55-965a-41f7-845c-99616018b475
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ a9d9212b-782c-4552-8d23-dcf74c052826
md"""
### 02. `Scanner()` Struct
**Siemens Naeotom Alpha (PCCT, 4-threshold)**

CdTe direct-conversion detector with native dexels 0.275 × 0.322 mm at
the detector face (2×2 binned in DAS).  Energy thresholds
`T = [20, 35, 55, 70] keV` define 4 bins:

| Bin | Range (keV) |
|-----|-------------|
| 1   | 20 – 35     |
| 2   | 35 – 55     |
| 3   | 55 – 70     |
| 4   | > 70        |
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
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
end;

# ╔═╡ ba6a1636-c0bc-4881-8ba7-d0e9dc8d90a8
md"""
### 03. `CTProtocol()` Struct
**140 kVp Photon-Counting**

Clinical 140 kVp single-energy acquisition.
`additional_filters = [("Ti", 0.9)]` is the Vectron tube's inherent
0.9 mm titanium window on top of the 3 mm Al flat filter.
"""

# ╔═╡ f9c0af7a-addd-4249-96fb-b9078765fbd1
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 07b0ef06-7c00-4f26-80fd-def32a742597
md"""
### 04. `SimOptions()` & `ReconOptions()`

`fidelity = :pcct` switches the simulator into the photon-counting path
(per-bin sinograms + DRM + Compton scatter modeling).
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
sim_opts = BS.SimOptions(
    fidelity = :pcct,
    seed = 1234,
    projector = :dd_fast,  # same anti-aliased DD physics, single-pass fused kernels (~47× faster poly)

    # ─── Inert for PCCT (flag exists but does nothing) ───
    use_fill_factor = false,
    use_detector_efficiency = false,
    use_optical_crosstalk = false,
    use_focal_spot = false,
    use_lag = false,
    use_heel_effect = false,

    # ─── Active for PCCT — all applied INSIDE simulate!() ───
    use_scatter = false,                  # EICT scatter flag — OFF (PCCT uses use_pcct_scatter)
    use_noise = true,                     # quantum noise (src :count, nr below)
    use_pcct_scatter = true,              # PCCT scatter injection
    use_pcct_scatter_correction = true,   # PCCT model-based scatter correction
    use_pcct_pileup = true,               # PCCT MC pile-up forward
    use_pcct_pileup_correction = true,    # PCCT pile-up correction (inverse S)
    # DETECTOR-LEVEL CORRECTION SURROGATE — explicitly NOT a recon-level
    # (QIR/iterative) stand-in: this chain is pure FBP end to end, and its
    # ACCURACY does not depend on this knob (pure chain, nr = 0, noise off:
    # rods within ~3 % of NIST, solid water < 3 HU).  The simulator
    # Monte-Carlo models the detector DEGRADATIONS (charge sharing,
    # fluorescence escape, pulse pileup, spectral distortion via the MC DRM)
    # but not the vendor's DETECTOR-side correction algorithms for them —
    # anti-coincidence/charge-sharing event reconstruction, count-rate
    # linearization beyond our inverse-S, threshold/spectral-distortion
    # compensation.  Those corrections recover count statistics at the
    # detector output; nr = 0.0 stands in for that recovery and nothing
    # else.
    pcct_noise_reduction = 0.0,
)

# ╔═╡ 08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
recon_opts = let
    slice_thickness_mm = 0.4
    n_recon_slices = max(1, round(Int, protocol.collimation_mm / slice_thickness_mm))
    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = protocol.collimation_mm / 10.0,
    )
end;

# ╔═╡ 5ecd97c6-ad47-4558-886d-22ed45eda97d
md"""
### 05. Forward Project: `simulate!`

A single `BS.simulate!` call produces the 4 per-bin log-line-integral
sinograms with the **complete PCCT physics + corrections**, all gated by the
`sim_opts` flags above:

```
forward → scatter inject → quantum noise → pile-up fwd →
pile-up correction → scatter correction
```

Scatter (`use_pcct_scatter` + `use_pcct_scatter_correction`) and pile-up
(`use_pcct_pileup` + `use_pcct_pileup_correction`) now happen inside
`simulate!()` — no decoupled notebook-level correction steps.  Bins are
`-log(N_recorded / I0_truth[b])`; `I0_bins` is the truth per-bin air baseline.
"""

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# === Forward project + full PCCT physics + corrections via simulate!() ===
# One src call: forward → scatter inject → noise → pile-up fwd → pile-up
# correction → scatter correction, all gated by the `sim_opts` flags.
sim_bins = let
    @info "Simulating: $(Int(protocol.kVp)) kVp / $(round(protocol.mA, digits = 1)) mA (PCCT 4-bin) — full physics + corrections via simulate!()"
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol, sim_opts)

    bins = [Array(b) for b in result.pcct_sino.bins]
    I0_bins = copy(result.I0_bins)
    geom = ws.geom
    # The EXACT per-bin detected spectra the forward applied (w·η·DRM with
    # the workspace's MC-LUT η and the centre-pixel bowtie fold).  The
    # decomposition basis consumes THESE — the inversion's forward model is
    # the model simulate! actually applied, by construction.
    energies = Float64.(ws.energies)
    W_applied = Float64.(Array(ws.W_matrix_gpu))[1:length(ws.energies), :]

    ws = nothing; result = nothing
    GC.gc(true)
    (bins = bins, I0_bins = I0_bins, geom = geom,
     energies = energies, W_applied = W_applied)
end;

# ╔═╡ 3e5c24af-1bb2-457e-b2a8-236aeb1fcff3
let
    n_row = size(sim_bins.bins[1], 2)
    mid_r = n_row ÷ 2 + 1

    # sino layout is (n_col, n_row, n_view); transpose so heatmap x = view, y = col
    bin_titles = ("Bin 1", "Bin 2", "Bin 3", "Bin 4")
    bin_subs = ("20 – 35 keV", "35 – 55 keV", "55 – 70 keV", "> 70 keV")
    slices = [permutedims(sim_bins.bins[k][:, mid_r, :], (2, 1)) for k in 1:4]

    # Dynamic shared range across all 4 bins — q1/q99 percentile clipping
    all_v = vcat([vec(s) for s in slices]...)
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    for k in 1:4
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = Mke.Axis(
            fig[r, c]; title = bin_titles[k], subtitle = bin_subs[k],
            axis_kwargs...,
        )
        Mke.heatmap!(ax, slices[k]; colormap = :viridis, colorrange = sino_window)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ ebf6817f-baef-48c9-9e7d-8ae303c41ad0
md"""
#### Intermediate FBP per Bin (μ-domain sanity check)

A quick per-bin FDK on the raw simulator sinograms — *before* the SVD
denoise, the bin combine, or the Cong decomposition — so we can
eyeball that the photon-counting forward model is producing physically
sensible images at each energy band.

Output is the linear attenuation coefficient **μ (cm⁻¹)**, the natural
unit of the FBP — no HU conversion yet.  Lower energy bins should
register higher μ for the same attenuator (μ rolls off with E), and
the same rod ordering should be visible across all four bins.
"""

# ╔═╡ be97f48a-9b4e-40e6-aff1-20e6392d73b3
sim_bins_fbp = let
    matrix_size = recon_opts.matrix_size
    geom = sim_bins.geom

    fdk_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.75, 0.6, 0.2, 0.001),
    )

    function _fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = fdk_filter,
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon)
    end

    [_fbp(b) for b in sim_bins.bins]
end;

# ╔═╡ 9533c523-9ff2-4094-808a-ff9e7424a3e9
let
    n_z = size(sim_bins_fbp[1], 3)
    mid_z = n_z ÷ 2 + 1

    bin_titles = ("Bin 1", "Bin 2", "Bin 3", "Bin 4")
    bin_subs = ("20 – 35 keV", "35 – 55 keV", "55 – 70 keV", "> 70 keV")
    slices = [sim_bins_fbp[k][:, :, mid_z] for k in 1:4]

    # Dynamic shared range across all 4 bins — q1/q99 percentile clipping
    all_v = vcat([vec(s) for s in slices]...)
    mu_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "x", ylabel = "y",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
        aspect = Mke.DataAspect(),
    )

    for k in 1:4
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = Mke.Axis(
            fig[r, c]; title = bin_titles[k], subtitle = bin_subs[k],
            axis_kwargs...,
        )
        Mke.heatmap!(ax, slices[k]; colormap = :viridis, colorrange = mu_window)
    end
    Mke.Colorbar(
        fig[1:2, 3]; colormap = :viridis, colorrange = mu_window,
        label = "μ (cm⁻¹)", width = 16, labelsize = 22, ticklabelsize = 18,
    )
    fig
end

# ╔═╡ 6dce791e-d934-4f23-95fb-5fa08f45a612
md"""
## Native 4-Channel Material Decomposition

The four photon-counting windows remain four separate measurements. This
section does **not** form the synthetic `123 | 4` pair used by the older
two-channel Cong workflow.

For each ray, the corrected count-domain measurement is

```math
y_k = I_{0,k}\exp(-h_k), \qquad k=1,\ldots,4,
```

and the native-channel forward model is

```math
\lambda_k(A,C)=\sum_e \Phi_k(E_e)
\exp[-\mu_{\rho,I}(E_e)A-\mu_{\rho,W}(E_e)C].
```

``A`` and ``C`` are iodine and water area densities (g/cm²).
`sim_bins.W_applied[:, k]` supplies the absolute ``\Phi_k`` actually used
by the simulation, including source, quantum efficiency, MC-DRM, and bowtie
spectral shaping.

The implementation is profiled: for each outer iodine value it solves the
scalar water subproblem, then updates iodine using the envelope gradient and
the Schur complement of the four-channel Fisher matrix. There is no native-bin
merge and no simultaneous two-variable update.

!!! note "Corrected-count likelihood"
    Pile-up inversion and scatter subtraction produce corrected, generally
    fractional pseudo-counts. The independent-Poisson objective is therefore
    a quasi-likelihood for the present fully corrected simulation. It is the
    exact independent-Poisson likelihood when those nonlinear correction
    stages are disabled.
"""

# ╔═╡ 4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
nchannel_basis = let
    E = Float32.(sim_bins.energies)
    Φ = Float32.(sim_bins.W_applied)
    μρ_I = Float32[
        BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(e))
        for e in E
    ]
    μρ_W = Float32[
        BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(e))
        for e in E
    ]

    # The likelihood needs absolute responses, not independently normalized spectra.
    I0_from_Φ = vec(sum(Float64.(Φ); dims = 1))
    I0_relerr = maximum(abs.(
        I0_from_Φ .- Float64.(sim_bins.I0_bins)
    ) ./ max.(Float64.(sim_bins.I0_bins), eps(Float64)))
    I0_relerr < 5e-5 || error(
        "Applied response and I0 disagree (max relative error = $(I0_relerr))."
    )

    # Four-channel effective-energy linearization, used only as an initializer.
    Φsum = vec(sum(Φ; dims = 1))
    μI_eff = Float32[
        sum(view(Φ, :, k) .* μρ_I) / Φsum[k] for k in 1:4
    ]
    μW_eff = Float32[
        sum(view(Φ, :, k) .* μρ_W) / Φsum[k] for k in 1:4
    ]

    (
        E = E, Φ = Φ, μρ_I = μρ_I, μρ_W = μρ_W,
        I0 = Float32.(sim_bins.I0_bins),
        μI_eff = μI_eff, μW_eff = μW_eff,
        normal_II = sum(abs2, μI_eff),
        normal_IW = sum(μI_eff .* μW_eff),
        normal_WW = sum(abs2, μW_eff),
        I0_relerr = I0_relerr,
    )
end;

# ╔═╡ a27649c0-e623-473e-bfbe-60fc8dd6d976
md"""
### Profile-solver controls

The bounds include a narrow negative guard band. Physical line integrals are
non-negative, but retaining small negative noise excursions avoids the
positive bias that hard rectification would spread through a later ramp
filter.
"""

# ╔═╡ 4985581f-616d-4bb7-ab9b-967d7250b28b
nchannel_controls = (
    iodine_bounds = (-0.10f0, 0.40f0), # g/cm²
    water_bounds = (-2.0f0, 50.0f0),   # g/cm²
    outer_iterations = 7,
    inner_iterations = 6,
    max_iodine_step = 0.05f0,
    max_water_step = 5.0f0,
    air_gate = 5.0f-3,
    tile_views = 8,
);

# ╔═╡ 73371177-0498-4eda-897b-651c94f43e83
function nchannel_profile_tile!(
    sino_I, sino_W, boundary_flag, h1, h2, h3, h4,
    Φ, μρ_I, μρ_W, I0, μI_eff, μW_eff,
    normal_II::Float32, normal_IW::Float32, normal_WW::Float32, controls,
)
    nE = length(μρ_I)
    A_lo, A_hi = controls.iodine_bounds
    C_lo, C_hi = controls.water_bounds
    n_outer, n_inner = controls.outer_iterations, controls.inner_iterations
    A_step, C_step = controls.max_iodine_step, controls.max_water_step
    air_gate = controls.air_gate

    BS.AK.foreachindex(sino_I) do idx
        h_1, h_2, h_3, h_4 = h1[idx], h2[idx], h3[idx], h4[idx]
        if max(max(abs(h_1), abs(h_2)), max(abs(h_3), abs(h_4))) < air_gate
            sino_I[idx] = 0f0
            sino_W[idx] = 0f0
            boundary_flag[idx] = UInt8(0)
            return
        end

        # Corrected counts. These may be fractional after detector correction.
        y1 = max(I0[1] * exp(-h_1), 1f-6)
        y2 = max(I0[2] * exp(-h_2), 1f-6)
        y3 = max(I0[3] * exp(-h_3), 1f-6)
        y4 = max(I0[4] * exp(-h_4), 1f-6)

        # Four-channel linear initializer; all iterations below are polychromatic.
        rhs_I = μI_eff[1]*h_1 + μI_eff[2]*h_2 +
                μI_eff[3]*h_3 + μI_eff[4]*h_4
        rhs_W = μW_eff[1]*h_1 + μW_eff[2]*h_2 +
                μW_eff[3]*h_3 + μW_eff[4]*h_4
        det0 = max(normal_II*normal_WW - normal_IW*normal_IW, 1f-12)
        A = clamp((normal_WW*rhs_I - normal_IW*rhs_W) / det0, A_lo, A_hi)
        C = clamp((normal_II*rhs_W - normal_IW*rhs_I) / det0, C_lo, C_hi)

        for _ in 1:n_outer
            # Inner scalar solve: C*(A) = argmin_C L(A,C).
            for _ in 1:n_inner
                gC, FCC = 0f0, 0f0
                for k in 1:4
                    λ, dC = 0f0, 0f0
                    @inbounds for e in 1:nE
                        z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                        λ += z
                        dC -= μρ_W[e] * z
                    end
                    λ = max(λ, 1f-6)
                    y = k == 1 ? y1 : k == 2 ? y2 : k == 3 ? y3 : y4
                    gC += (1f0 - y/λ) * dC
                    FCC += dC*dC / λ
                end
                C = clamp(C - clamp(gC/max(FCC, 1f-12), -C_step, C_step), C_lo, C_hi)
            end

            # Envelope gradient and Fisher Schur-complement profile curvature.
            gA, FAA, FAC, FCC = 0f0, 0f0, 0f0, 0f0
            for k in 1:4
                λ, dA, dC = 0f0, 0f0, 0f0
                @inbounds for e in 1:nE
                    z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dA -= μρ_I[e] * z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = k == 1 ? y1 : k == 2 ? y2 : k == 3 ? y3 : y4
                gA += (1f0 - y/λ) * dA
                FAA += dA*dA / λ
                FAC += dA*dC / λ
                FCC += dC*dC / λ
            end
            Hprof = max(FAA - FAC*FAC/max(FCC, 1f-12), 1f-12)
            A = clamp(A - clamp(gA/Hprof, -A_step, A_step), A_lo, A_hi)
        end

        # Re-profile water at the final iodine iterate.
        for _ in 1:n_inner
            gC, FCC = 0f0, 0f0
            for k in 1:4
                λ, dC = 0f0, 0f0
                @inbounds for e in 1:nE
                    z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = k == 1 ? y1 : k == 2 ? y2 : k == 3 ? y3 : y4
                gC += (1f0 - y/λ) * dC
                FCC += dC*dC / λ
            end
            C = clamp(C - clamp(gC/max(FCC, 1f-12), -C_step, C_step), C_lo, C_hi)
        end

        tol = 2f-4
        hit_A = A <= A_lo + tol || A >= A_hi - tol
        hit_C = C <= C_lo + tol || C >= C_hi - tol
        boundary_flag[idx] = UInt8(hit_A ? 1 : 0) | UInt8(hit_C ? 2 : 0)
        sino_I[idx], sino_W[idx] = A, C
    end
    nothing
end

# ╔═╡ ad092cf9-14b2-4f32-bad4-1e7962961fc2
sino_basis_nchannel = let
    shape = size(sim_bins.bins[1])
    sino_I = Array{Float32}(undef, shape)
    sino_W = Array{Float32}(undef, shape)
    flags = Array{UInt8}(undef, shape)

    Φ_gpu = to_gpu(nchannel_basis.Φ)
    μρ_I_gpu, μρ_W_gpu = to_gpu(nchannel_basis.μρ_I), to_gpu(nchannel_basis.μρ_W)
    I0_gpu = to_gpu(nchannel_basis.I0)
    μI_eff_gpu, μW_eff_gpu = to_gpu(nchannel_basis.μI_eff), to_gpu(nchannel_basis.μW_eff)

    elapsed = @elapsed for vrange in BS.tile_ranges(shape[3], nchannel_controls.tile_views)
        hs = [to_gpu(Float32.(sim_bins.bins[k][:,:,vrange])) for k in 1:4]
        I_gpu, W_gpu = similar(hs[1]), similar(hs[1])
        flag_gpu = similar(hs[1], UInt8)
        nchannel_profile_tile!(
            I_gpu, W_gpu, flag_gpu, hs[1], hs[2], hs[3], hs[4],
            Φ_gpu, μρ_I_gpu, μρ_W_gpu, I0_gpu, μI_eff_gpu, μW_eff_gpu,
            nchannel_basis.normal_II, nchannel_basis.normal_IW,
            nchannel_basis.normal_WW, nchannel_controls,
        )
        sino_I[:,:,vrange] .= Array(I_gpu)
        sino_W[:,:,vrange] .= Array(W_gpu)
        flags[:,:,vrange] .= Array(flag_gpu)
    end

    boundary_fraction = count(x -> !iszero(x), flags) / length(flags)
    @info "Native 4-channel profile decomposition complete" elapsed_s=round(elapsed,digits=1) boundary_fraction
    (
        sino_iodine = sino_I, sino_water = sino_W, boundary_flag = flags,
        geom = sim_bins.geom, elapsed_s = elapsed,
    )
end;

# ╔═╡ 831c3a54-76d2-4dfc-b93a-faf3661518a3
md"""
### Two Material Sinograms
"""

# ╔═╡ 7bfdb7d7-6a46-4803-88dd-8c08d41a3780
let
    mid_r = size(sino_basis_nchannel.sino_iodine, 2) ÷ 2 + 1
    slice_I = permutedims(sino_basis_nchannel.sino_iodine[:,mid_r,:], (2,1))
    slice_W = permutedims(sino_basis_nchannel.sino_water[:,mid_r,:], (2,1))
    qrange(x) = (Float64(quantile(vec(x), 0.01)), Float64(quantile(vec(x), 0.99)))

    fig = Mke.Figure(size = (1400, 580))
    panels = (
        (1, 1, 2, "Iodine Basis Sinogram", "g/cm²", slice_I, qrange(slice_I)),
        (1, 3, 4, "Water Basis Sinogram", "g/cm²", slice_W, qrange(slice_W)),
    )
    for (r, c, cb, title, label, image, range) in panels
        ax = Mke.Axis(fig[r,c]; title, xlabel="View", ylabel="Detector Column")
        Mke.heatmap!(ax, image; colormap=:viridis, colorrange=range)
        Mke.Colorbar(fig[r,cb]; colormap=:viridis, colorrange=range, label, width=16)
    end
    fig
end

# ╔═╡ c456f690-518d-4fb9-941e-a5662b101ff5
md"""
## Pre-Reconstruction Statistical Verification

Before applying FBP, filtering, or VMI synthesis, evaluate the final material
estimate against the complete native four-bin likelihood. For every active
ray this block records

```math
\Sigma_{WI,\mathrm{CRLB}}=F^{-1},\qquad
F=J^\mathsf{T}\operatorname{diag}(\lambda_k^{-1})J,
```

the water--iodine correlation, the four-bin Poisson deviance, and the
Fisher-normalized score magnitude. The decomposition itself is not modified.

The empirical covariance check uses the central repeated detector rows of
this axially uniform phantom as independent realizations of nearly identical
rays. Their per-ray across-row mean is removed before pooling. This is a
single-simulation diagnostic; repeated full noise realizations remain the
strict validation experiment.
"""

# ╔═╡ f9168282-ad86-446c-ac04-1aac09d4c1a2
function nchannel_fisher_tile!(
    var_I, cov_IW, var_W, deviance, score2,
    sino_I, sino_W, h1, h2, h3, h4,
    Φ, μρ_I, μρ_W, I0, air_gate::Float32,
)
    nE = length(μρ_I)
    BS.AK.foreachindex(sino_I) do idx
        h_1, h_2, h_3, h_4 = h1[idx], h2[idx], h3[idx], h4[idx]
        if max(max(abs(h_1), abs(h_2)), max(abs(h_3), abs(h_4))) < air_gate
            var_I[idx] = NaN32
            cov_IW[idx] = NaN32
            var_W[idx] = NaN32
            deviance[idx] = NaN32
            score2[idx] = NaN32
            return
        end

        A, C = sino_I[idx], sino_W[idx]
        y1 = max(I0[1] * exp(-h_1), 1f-6)
        y2 = max(I0[2] * exp(-h_2), 1f-6)
        y3 = max(I0[3] * exp(-h_3), 1f-6)
        y4 = max(I0[4] * exp(-h_4), 1f-6)
        FAA, FAC, FCC = 0f0, 0f0, 0f0
        gA, gC, D = 0f0, 0f0, 0f0

        # Deliberately loop over every native channel: no bin is selected or merged.
        for k in 1:4
            λ, dA, dC = 0f0, 0f0, 0f0
            @inbounds for e in 1:nE
                z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                λ += z
                dA -= μρ_I[e] * z
                dC -= μρ_W[e] * z
            end
            λ = max(λ, 1f-6)
            y = k == 1 ? y1 : k == 2 ? y2 : k == 3 ? y3 : y4
            q = 1f0 - y/λ
            gA += q*dA
            gC += q*dC
            FAA += dA*dA/λ
            FAC += dA*dC/λ
            FCC += dC*dC/λ
            D += 2f0 * (λ - y + y*log(y/λ))
        end

        detF = max(FAA*FCC - FAC*FAC, 1f-20)
        vI = FCC/detF
        cIW = -FAC/detF
        vW = FAA/detF
        var_I[idx], cov_IW[idx], var_W[idx] = vI, cIW, vW
        deviance[idx] = max(D, 0f0)
        score2[idx] = max(vI*gA*gA + 2f0*cIW*gA*gC + vW*gC*gC, 0f0)
    end
    nothing
end

# ╔═╡ 54bc4490-08c6-4d61-a687-d491976b2761
nchannel_fisher = let
    shape = size(sino_basis_nchannel.sino_iodine)
    fields = (
        var_iodine = Array{Float32}(undef, shape),
        cov_iodine_water = Array{Float32}(undef, shape),
        var_water = Array{Float32}(undef, shape),
        deviance = Array{Float32}(undef, shape),
        score2 = Array{Float32}(undef, shape),
    )
    Φ_gpu = to_gpu(nchannel_basis.Φ)
    μI_gpu = to_gpu(nchannel_basis.μρ_I)
    μW_gpu = to_gpu(nchannel_basis.μρ_W)
    I0_gpu = to_gpu(nchannel_basis.I0)

    elapsed = @elapsed for vrange in BS.tile_ranges(
        shape[3], nchannel_controls.tile_views,
    )
        hs = [
            to_gpu(Float32.(sim_bins.bins[k][:,:,vrange]))
            for k in 1:4
        ]
        I_gpu = to_gpu(
            Float32.(sino_basis_nchannel.sino_iodine[:,:,vrange])
        )
        W_gpu = to_gpu(
            Float32.(sino_basis_nchannel.sino_water[:,:,vrange])
        )
        outs = ntuple(_ -> similar(I_gpu), 5)
        nchannel_fisher_tile!(
            outs..., I_gpu, W_gpu, hs...,
            Φ_gpu, μI_gpu, μW_gpu, I0_gpu, nchannel_controls.air_gate,
        )
        for (dest, src) in zip(values(fields), outs)
            dest[:,:,vrange] .= Array(src)
        end
    end
    @info "Four-bin Fisher verification complete" elapsed_s=round(elapsed,digits=1)
    merge(fields, (elapsed_s=elapsed,))
end;

# ╔═╡ a0930dd5-c28e-45a1-9bb6-4be300ba19d2
nchannel_sinogram_verification = let
    I = sino_basis_nchannel.sino_iodine
    W = sino_basis_nchannel.sino_water
    nrow = size(I,2)
    rows = max(1, nrow ÷ 4):min(nrow, nrow - nrow ÷ 4)
    flags = sino_basis_nchannel.boundary_flag

    residual_I = Float64[]
    residual_W = Float64[]
    predicted = NTuple{3,Float64}[]
    for view in axes(I,3), col in axes(I,1)
        valid_rows = [
            r for r in rows
            if flags[col,r,view] == 0x00 &&
               isfinite(nchannel_fisher.var_iodine[col,r,view])
        ]
        length(valid_rows) ≥ 4 || continue
        mean_I = mean(I[col,valid_rows,view])
        mean_W = mean(W[col,valid_rows,view])
        # Exclude air while retaining water-only and insert-bearing object rays.
        mean_W > 1.0f0 || continue
        n = length(valid_rows)
        mean_removal = 1.0 - 1.0/n
        for r in valid_rows
            push!(residual_I, Float64(I[col,r,view] - mean_I))
            push!(residual_W, Float64(W[col,r,view] - mean_W))
            push!(predicted, (
                mean_removal*Float64(nchannel_fisher.var_iodine[col,r,view]),
                mean_removal*Float64(nchannel_fisher.cov_iodine_water[col,r,view]),
                mean_removal*Float64(nchannel_fisher.var_water[col,r,view]),
            ))
        end
    end
    length(residual_I) > 100 || error("Too few valid repeated-row samples")

    centered_I = residual_I .- mean(residual_I)
    centered_W = residual_W .- mean(residual_W)
    denom = length(residual_I) - 1
    empirical = [
        sum(abs2,centered_I)/denom sum(centered_I .* centered_W)/denom
        sum(centered_I .* centered_W)/denom sum(abs2,centered_W)/denom
    ]
    predicted_mean = [
        mean(first.(predicted)) mean(getindex.(predicted,2))
        mean(getindex.(predicted,2)) mean(last.(predicted))
    ]
    empirical_rho = empirical[1,2] / sqrt(empirical[1,1]*empirical[2,2])
    predicted_rho = predicted_mean[1,2] /
        sqrt(predicted_mean[1,1]*predicted_mean[2,2])
    finite_deviance = filter(isfinite, vec(nchannel_fisher.deviance))
    finite_score2 = filter(isfinite, vec(nchannel_fisher.score2))

    (
        rows=rows, n_samples=length(residual_I),
        empirical=empirical, predicted=predicted_mean,
        variance_ratio=[
            empirical[1,1]/predicted_mean[1,1],
            empirical[2,2]/predicted_mean[2,2],
        ],
        empirical_rho=empirical_rho, predicted_rho=predicted_rho,
        deviance_median=median(finite_deviance),
        deviance_p95=quantile(finite_deviance,0.95),
        score2_median=median(finite_score2),
        score2_p95=quantile(finite_score2,0.95),
        boundary_fraction=count(x -> !iszero(x),flags)/length(flags),
    )
end;

# ╔═╡ 7d987c7a-50ac-4827-9145-d07efe14747a
let
    d = nchannel_sinogram_verification
    labels = ["Iodine variance","Water variance"]
    empirical = [d.empirical[1,1],d.empirical[2,2]]
    predicted = [d.predicted[1,1],d.predicted[2,2]]

    fig = Mke.Figure(size=(1180,500))
    ax1 = Mke.Axis(
        fig[1,1];
        title="Empirical Repeated-Row Covariance vs CRLB",
        subtitle="$(d.n_samples) samples; bars are variances",
        xticks=(1:2,labels), ylabel="Variance (basis units²)",
        yscale=log10,
    )
    Mke.barplot!(
        ax1,[0.82,1.82],predicted;
        color=:steelblue,strokecolor=:black,strokewidth=1,
    )
    Mke.barplot!(
        ax1,[1.18,2.18],empirical;
        color=:darkorange,strokecolor=:black,strokewidth=1,
    )
    Mke.axislegend(
        ax1,
        [
            Mke.PolyElement(color=:steelblue),
            Mke.PolyElement(color=:darkorange),
        ],
        ["Mean F⁻¹ prediction","Empirical residual"];
        position=:lt,
    )

    ax2 = Mke.Axis(
        fig[1,2];
        title="Four-Bin Fit Health",
        xlabel="Diagnostic", ylabel="Value",
        xticks=(
            1:4,
            ["ρ(W,I) predicted","ρ(W,I) empirical","Deviance p95","Score² p95"],
        ),
    )
    values = [
        d.predicted_rho,d.empirical_rho,d.deviance_p95,d.score2_p95,
    ]
    Mke.barplot!(
        ax2,1:4,values;
        color=[:steelblue,:darkorange,:seagreen,:mediumpurple],
        strokecolor=:black,strokewidth=1,
    )
    Mke.hlines!(ax2,[0.0]; color=:black,linewidth=1)
    for (i,value) in pairs(values)
        Mke.text!(
            ax2,i,value;text="$(round(value,digits=3))",
            align=(:center,value ≥ 0 ? :bottom : :top),
            offset=(0,value ≥ 0 ? 5 : -5),
        )
    end
    fig
end

# ╔═╡ 2d6e6dda-5a23-42c8-8744-116029c2db9a
let
    d = nchannel_sinogram_verification
    md"""
    **Four-bin pre-FBP verification**

    - Native channels in the evaluated likelihood: **4 / 4**
    - Boundary-contact fraction: **$(round(100d.boundary_fraction,digits=3))%**
    - Empirical / CRLB variance ratio — iodine: **$(round(d.variance_ratio[1],digits=3))×**, water: **$(round(d.variance_ratio[2],digits=3))×**
    - Water--iodine correlation — predicted: **$(round(d.predicted_rho,digits=4))**, empirical: **$(round(d.empirical_rho,digits=4))**
    - Four-bin deviance — median: **$(round(Float64(d.deviance_median),digits=3))**, p95: **$(round(Float64(d.deviance_p95),digits=3))**
    - Fisher-normalized score² — median: **$(round(Float64(d.score2_median),digits=4))**, p95: **$(round(Float64(d.score2_p95),digits=4))**

    A variance ratio near one supports an efficient decomposition. A much
    larger empirical value points to avoidable solver/model noise before any
    denoising is considered. A small normalized score indicates that the
    final estimates are close to a stationary point of the complete
    likelihood.
    """
end

# ╔═╡ f8688ab5-5e22-432e-bf04-45d8a7763576
md"""
## Experimental Fisher-ACNR → VMI Path

This is a standalone experimental path. The original direct-FBP workflow
below remains unchanged as a control.

1. Reconstruct water and iodine with the **same** `SoftFilter`.
2. Use the mean four-bin Fisher covariance to define the 140-keV reference
   mode \(g\), the orthogonal material-separation mode \(d\), and
   \(r=d-\beta g\), so the predicted \(\operatorname{Cov}(g,r)=0\).
3. Estimate residual-mode noise from adjacent axial-replicate differences
   and apply the closed-form radial Wiener gain \(H=\max(1-N/P,0)\).
4. Select the largest residual-retention factor \(s\in[0,1]\) satisfying
   the solid-water monotonic-noise inequalities. This is solved from their
   quadratic variance curves, not manually tuned.
5. Preserve the 140-keV reference mode pixel-for-pixel and synthesize every
   VMI from the same corrected material pair.
"""

# ╔═╡ 40b39aa1-2aed-4804-80db-cab52a8e4964
FFTW = BS.FFTW

# ╔═╡ 308003a4-fe8e-4690-a492-2bc786e96965
fisher_common_fbp = let
    geom = sino_basis_nchannel.geom
    matrix_size = recon_opts.matrix_size
    common_filter = BS.SoftFilter()

    function common_fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu,geom,matrix_size; filter=common_filter,
        )
        volume = Float32.(Array(BS.reconstruct!(ws,sino_gpu,geom)))
        ws = nothing
        sino_gpu = nothing
        GC.gc(true)
        volume
    end

    W = common_fbp(sino_basis_nchannel.sino_water)
    I = common_fbp(sino_basis_nchannel.sino_iodine)

    # A direct mixed-sinogram FBP verifies linear commutation at E_ref.
    E_ref = 140.0
    μW_ref = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E_ref)
    μI_ref = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine,E_ref)
    mixed_sino = @. Float32(
        μW_ref*sino_basis_nchannel.sino_water +
        μI_ref*sino_basis_nchannel.sino_iodine
    )
    mixed_direct = common_fbp(mixed_sino)
    mixed_after = @. Float32(μW_ref*W + μI_ref*I)
    commutation_relerr = maximum(abs,mixed_direct .- mixed_after) /
        max(maximum(abs,mixed_direct),eps(Float32))

    (
        vol_water=W,vol_iodine=I,geom=geom,filter=common_filter,
        E_ref=E_ref,μW_ref=μW_ref,μI_ref=μI_ref,
        commutation_relerr=commutation_relerr,
    )
end;

# ╔═╡ f8138d36-d35a-403a-a824-26ab60d75727
function fisher_radial_wiener(residual::AbstractArray{<:Real,3})
    nx,ny,nz = size(residual)
    nrad = floor(Int,hypot(nx ÷ 2,ny ÷ 2)) + 1
    power_sum = zeros(Float64,nrad)
    noise_sum = zeros(Float64,nrad)
    power_count = zeros(Int,nrad)
    noise_count = zeros(Int,nrad)

    radial_bin(i,j) = min(
        floor(Int,hypot(min(i-1,nx-(i-1)),min(j-1,ny-(j-1)))) + 1,
        nrad,
    )

    for z in 1:nz
        spectrum = FFTW.fft(Float64.(@view residual[:,:,z]))
        for j in 1:ny, i in 1:nx
            b = radial_bin(i,j)
            power_sum[b] += abs2(spectrum[i,j])
            power_count[b] += 1
        end
    end
    for z in 1:(nz-1)
        noise_sample = (
            Float64.(@view residual[:,:,z+1]) .-
            Float64.(@view residual[:,:,z])
        ) ./ sqrt(2.0)
        spectrum = FFTW.fft(noise_sample)
        for j in 1:ny, i in 1:nx
            b = radial_bin(i,j)
            noise_sum[b] += abs2(spectrum[i,j])
            noise_count[b] += 1
        end
    end

    P = power_sum ./ max.(power_count,1)
    N = noise_sum ./ max.(noise_count,1)
    gain_radial = clamp.(1.0 .- N ./ max.(P,eps(Float64)),0.0,1.0)
    gain_radial[1] = 1.0
    gain = [
        gain_radial[radial_bin(i,j)] for i in 1:nx, j in 1:ny
    ]
    filtered = similar(residual,Float32)
    Threads.@threads for z in 1:nz
        spectrum = FFTW.fft(Float64.(@view residual[:,:,z]))
        filtered[:,:,z] .= Float32.(
            real.(FFTW.ifft(spectrum .* gain))
        )
    end
    (
        filtered=filtered,gain_radial=gain_radial,
        power_radial=P,noise_radial=N,
    )
end

# ╔═╡ 94b9c44f-90a5-4dad-95d8-3b47d4036575
function largest_monotonic_retention(variance_polynomials)
    # Each tuple is (constant, linear, quadratic) for σ²_E(s).
    boundaries = Float64[0.0,1.0]
    inequalities = NTuple{3,Float64}[]
    for j in 1:(length(variance_polynomials)-1)
        lo,hi = variance_polynomials[j],variance_polynomials[j+1]
        q = (hi[1]-lo[1],hi[2]-lo[2],hi[3]-lo[3])
        push!(inequalities,q)
        c,b,a = q
        if abs(a) < 1e-14
            abs(b) > 1e-14 && push!(boundaries,-c/b)
        else
            Δ = b*b - 4a*c
            if Δ ≥ 0
                root = sqrt(Δ)
                push!(boundaries,(-b-root)/(2a),(-b+root)/(2a))
            end
        end
    end
    boundaries = sort(unique(clamp.(filter(
        x -> isfinite(x) && -1e-10 ≤ x ≤ 1+1e-10,boundaries,
    ),0.0,1.0)))
    feasible(s) = all(
        q -> q[1]+q[2]*s+q[3]*s*s ≤ 1e-10,inequalities,
    )
    candidates = Float64[]
    for x in boundaries
        feasible(x) && push!(candidates,x)
    end
    for j in 1:(length(boundaries)-1)
        midpoint = (boundaries[j]+boundaries[j+1])/2
        feasible(midpoint) && push!(candidates,boundaries[j+1])
    end
    isempty(candidates) ?
        (retention=1.0,feasible=false) :
        (retention=maximum(candidates),feasible=true)
end

# ╔═╡ ca67864e-a436-4bf5-be0e-1924019885f5
fisher_acnr = let
    W,I = fisher_common_fbp.vol_water,fisher_common_fbp.vol_iodine
    μW,μI = fisher_common_fbp.μW_ref,fisher_common_fbp.μI_ref

    # x=[W,I]. Reorder the stored projection covariance, which is [I,W].
    Σ_sino = nchannel_sinogram_verification.predicted
    Σ = [
        Σ_sino[2,2] Σ_sino[2,1]
        Σ_sino[1,2] Σ_sino[1,1]
    ]
    norm_c = hypot(μW,μI)
    u = [μW/norm_c,μI/norm_c]
    v = [-u[2],u[1]]
    Σgg = sum(u .* (Σ*u))
    Σdg = sum(v .* (Σ*u))
    β = Σdg/Σgg

    g = @. Float32(u[1]*W + u[2]*I)
    d = @. Float32(v[1]*W + v[2]*I)
    r = @. Float32(d - β*g)
    wiener = fisher_radial_wiener(r)
    rf = wiener.filtered

    mask_mid = phantom_cpu.mask[:,:,size(phantom_cpu.mask,3) ÷ 2]
    water_mask = collect(BS.erode_mask_2d(
        mask_mid .== UInt8(BS.REGION_SOLID_WATER); erode_px=12.0,
    ))
    roi_g = Float64.(g[water_mask,:])
    roi_r = Float64.(rf[water_mask,:])
    gc = roi_g .- mean(roi_g)
    rc = roi_r .- mean(roi_r)
    var_g = sum(abs2,gc)/(length(gc)-1)
    cov_gr = sum(gc .* rc)/(length(gc)-1)
    var_r = sum(abs2,rc)/(length(rc)-1)
    energies = [40.0,70.0,100.0,140.0]

    # HU noise at E is qg*g + s*qr*rf; constants do not affect variance.
    mode_coefficients = map(energies) do E
        ratio = BS.compute_mass_μ_at_energy(
            BS.XA.Elements.Iodine,E,
        ) / BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        cHU = [1000.0,1000.0*ratio]
        b = sum(cHU .* v)
        a = sum(cHU .* u) + β*b
        (a=a,b=b)
    end
    variance_polynomials = [
        (
            q.a^2*var_g,
            2q.a*q.b*cov_gr,
            q.b^2*var_r,
        )
        for q in mode_coefficients
    ]
    monotonic = largest_monotonic_retention(variance_polynomials)
    s = monotonic.retention

    d_corrected = @. Float32(β*g + s*rf)
    W_corrected = @. Float32(u[1]*g + v[1]*d_corrected)
    I_corrected = @. Float32(u[2]*g + v[2]*d_corrected)
    reference_before = @. Float32(μW*W + μI*I)
    reference_after = @. Float32(μW*W_corrected + μI*I_corrected)
    reference_relerr = maximum(abs,reference_after .- reference_before) /
        max(maximum(abs,reference_before),eps(Float32))

    predicted_sd = [
        sqrt(max(p[1]+p[2]*s+p[3]*s*s,0.0))
        for p in variance_polynomials
    ]
    (
        vol_water=W_corrected,vol_iodine=I_corrected,
        vol_water_raw=W,vol_iodine_raw=I,
        reference_mode=g,residual_mode=r,residual_wiener=rf,
        u=u,v=v,β=β,retention=s,
        monotonic_feasible=monotonic.feasible,
        predicted_roi_sd_HU=predicted_sd,energies=energies,
        reference_relerr=reference_relerr,
        commutation_relerr=fisher_common_fbp.commutation_relerr,
        wiener_gain_radial=wiener.gain_radial,
        wiener_power_radial=wiener.power_radial,
        wiener_noise_radial=wiener.noise_radial,
        water_mask=water_mask,
    )
end;

# ╔═╡ 42dafcec-17be-4ed4-8e6a-3185ee9f0b0b
fisher_acnr_vmi = let
    energies = fisher_acnr.energies
    raw = Dict(
        E => BS.synth_vmi_2basis(
            fisher_acnr.vol_water_raw,
            fisher_acnr.vol_iodine_raw .* 1000.0f0;
            energy_keV=E,
        )
        for E in energies
    )
    denoised = Dict(
        E => BS.synth_vmi_2basis(
            fisher_acnr.vol_water,
            fisher_acnr.vol_iodine .* 1000.0f0;
            energy_keV=E,
        )
        for E in energies
    )
    (raw=raw,denoised=denoised,energies=energies)
end;

# ╔═╡ fa1fb5e8-64ee-4e9b-8ce7-dd8eb00f0e22
let
    sample = fisher_acnr_vmi.denoised[40.0]
    mid_z = size(sample,3) ÷ 2 + 1
    fig = Mke.Figure(size=(1180,1180))
    for (k,E) in enumerate(fisher_acnr_vmi.energies)
        row = (k-1) ÷ 2 + 1
        col = (k-1) % 2 + 1
        ax = Mke.Axis(
            fig[row,col]; title="Fisher-ACNR · $(Int(E)) keV",
            aspect=Mke.DataAspect(),titlesize=28,
        )
        Mke.heatmap!(
            ax,fisher_acnr_vmi.denoised[E][:,:,mid_z];
            colormap=:grays,colorrange=(-200.0,500.0),
        )
        Mke.hidedecorations!(ax)
    end
    fig
end

# ╔═╡ c2f2b9a2-aead-4de1-9132-a8d6bc10ded6
fisher_acnr_metrics = let
    mask = fisher_acnr.water_mask
    energies = fisher_acnr_vmi.energies
    mean_roi(volume) = mean(volume[mask,:])
    sd_roi(volume) = std(volume[mask,:])
    raw_mean = [mean_roi(fisher_acnr_vmi.raw[E]) for E in energies]
    denoised_mean = [mean_roi(fisher_acnr_vmi.denoised[E]) for E in energies]
    raw_sd = [sd_roi(fisher_acnr_vmi.raw[E]) for E in energies]
    denoised_sd = [sd_roi(fisher_acnr_vmi.denoised[E]) for E in energies]
    (
        energies=energies,raw_mean=raw_mean,denoised_mean=denoised_mean,
        raw_sd=raw_sd,denoised_sd=denoised_sd,
        mean_change=denoised_mean .- raw_mean,
        monotonic_measured=all(diff(denoised_sd) .≤ 1f-4),
    )
end;

# ╔═╡ efbb5ea0-a2ad-4d94-a145-34bc65a059cc
let
    d = fisher_acnr_metrics
    fig = Mke.Figure(size=(1180,500))
    ax1 = Mke.Axis(
        fig[1,1];title="Solid-Water Noise",
        subtitle="Common-kernel raw vs Fisher-ACNR",
        xlabel="VMI Energy (keV)",ylabel="σ (HU)",
        xticks=(d.energies,string.(Int.(d.energies))),
    )
    Mke.lines!(
        ax1,d.energies,d.raw_sd;
        color=:darkorange,linewidth=3,label="Raw",
    )
    Mke.scatter!(ax1,d.energies,d.raw_sd;color=:darkorange,markersize=14)
    Mke.lines!(
        ax1,d.energies,d.denoised_sd;
        color=:dodgerblue,linewidth=3,label="Fisher-ACNR",
    )
    Mke.scatter!(
        ax1,d.energies,d.denoised_sd;color=:dodgerblue,markersize=14,
    )
    Mke.axislegend(ax1;position=:rt)

    ax2 = Mke.Axis(
        fig[1,2];title="Solid-Water Mean HU",
        subtitle="Bias must remain unchanged",
        xlabel="VMI Energy (keV)",ylabel="Mean HU",
        xticks=(d.energies,string.(Int.(d.energies))),
    )
    Mke.lines!(
        ax2,d.energies,d.raw_mean;
        color=:darkorange,linewidth=3,label="Raw",
    )
    Mke.scatter!(ax2,d.energies,d.raw_mean;color=:darkorange,markersize=14)
    Mke.lines!(
        ax2,d.energies,d.denoised_mean;
        color=:dodgerblue,linewidth=3,label="Fisher-ACNR",
    )
    Mke.scatter!(
        ax2,d.energies,d.denoised_mean;color=:dodgerblue,markersize=14,
    )
    Mke.hlines!(ax2,[0.0];color=:black,linestyle=:dash)
    Mke.axislegend(ax2;position=:rt)
    fig
end

# ╔═╡ 22c17c01-1162-49e8-80d4-3b013f1e0672
let
    a,d = fisher_acnr,fisher_acnr_metrics
    md"""
    **Experimental Fisher-ACNR verification**

    - Common-kernel FBP/VMI commutation relative error: **$(round(Float64(a.commutation_relerr),sigdigits=4))**
    - 140-keV reference-mode relative change: **$(round(Float64(a.reference_relerr),sigdigits=4))**
    - Fisher regression coefficient β: **$(round(a.β,digits=5))**
    - Automatically selected residual retention \(s\): **$(round(a.retention,digits=5))**
    - Monotonic constraint feasible while preserving the reference: **$(a.monotonic_feasible)**
    - Measured denoised ROI noise monotonic: **$(d.monotonic_measured)**
    - Raw ROI noise (HU): **$(join(round.(d.raw_sd,digits=1),", "))**
    - Fisher-ACNR ROI noise (HU): **$(join(round.(d.denoised_sd,digits=1),", "))**
    - Maximum absolute water-ROI mean change: **$(round(Float64(maximum(abs,d.mean_change)),digits=4)) HU**

    The reference and commutation errors should be near floating-point
    precision. If monotonic feasibility is false, filtering only the
    separation mode cannot satisfy the requested curve without also changing
    the protected reference mode.
    """
end

# ╔═╡ 3d019ead-ea76-49b6-bc64-a6e534b9612a
md"""
### Adversarial implementation audit

The first Fisher-ACNR result is not accepted at face value. These controls
test the most consequential assumptions:

- **units/order:** direct textbook HU algebra must match
  `synth_vmi_2basis`;
- **covariance propagation:** projection-domain Fisher \(\beta\) is compared
  with image-ROI \(\beta\) after common-kernel FBP;
- **noise-spectrum validity:** adjacent-slice correlation and Wiener-gain
  occupancy reveal whether difference NPS underestimates coherent noise;
- **method ceiling:** setting the residual mode identically to zero gives the
  lowest noise this one-protected-mode construction can possibly reach.
"""

# ╔═╡ d13ac73b-b554-4a42-a1e1-29feb2365514
fisher_acnr_adversarial = let
    W,I = fisher_common_fbp.vol_water,fisher_common_fbp.vol_iodine
    mask = fisher_acnr.water_mask
    u,v = fisher_acnr.u,fisher_acnr.v
    g = @. Float32(u[1]*W + u[2]*I)
    d = @. Float32(v[1]*W + v[2]*I)

    roi_g = Float64.(g[mask,:])
    roi_d = Float64.(d[mask,:])
    gc = roi_g .- mean(roi_g)
    dc = roi_d .- mean(roi_d)
    β_image = sum(gc .* dc) / sum(abs2,gc)
    r_image = @. Float32(d - β_image*g)
    wiener_image = fisher_radial_wiener(r_image)

    function volumes_for(β,residual)
        d_new = @. Float32(β*g + residual)
        W_new = @. Float32(u[1]*g + v[1]*d_new)
        I_new = @. Float32(u[2]*g + v[2]*d_new)
        (W=W_new,I=I_new)
    end
    empirical_wiener = volumes_for(β_image,wiener_image.filtered)
    zero_residual = volumes_for(β_image,zeros(Float32,size(r_image)))
    energies = fisher_acnr.energies

    function noise_curve(pair)
        [
            std(BS.synth_vmi_2basis(
                pair.W,pair.I .* 1000.0f0; energy_keV=E,
            )[mask,:])
            for E in energies
        ]
    end
    empirical_wiener_sd = noise_curve(empirical_wiener)
    zero_residual_sd = noise_curve(zero_residual)

    # Direct algebra/unit-order identity check at all requested energies.
    synth_error = maximum(energies) do E
        via_api = BS.synth_vmi_2basis(
            W,I .* 1000.0f0; energy_keV=E,
        )
        ratio = Float32(BS.compute_mass_μ_at_energy(
            BS.XA.Elements.Iodine,E,
        ) / BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E))
        direct = @. Float32(1000.0f0*(W-1.0f0) + 1000.0f0*I*ratio)
        maximum(abs,via_api .- direct)
    end

    # Lag-one correlation after removing each voxel's axial mean.
    r_centered = Float64.(r_image[mask,:])
    r_centered .-= mean(r_centered;dims=2)
    lag_left = @view r_centered[:,1:end-1]
    lag_right = @view r_centered[:,2:end]
    lag_num = sum(lag_left .* lag_right)
    lag_den = sqrt(
        sum(abs2,lag_left) * sum(abs2,lag_right)
    )
    axial_lag1 = lag_num / lag_den

    gains = wiener_image.gain_radial[2:end]
    roi_W = Float64.(W[mask,:])
    roi_I = Float64.(I[mask,:])
    roi_W .-= mean(roi_W)
    roi_I .-= mean(roi_I)
    basis_rho = sum(roi_W .* roi_I) /
        sqrt(sum(abs2,roi_W)*sum(abs2,roi_I))
    object_counts = [
        begin
            h = sim_bins.bins[k]
            selected = h .> 0.1f0
            median(Float64.(
                nchannel_basis.I0[k] .* exp.(-h[selected])
            ))
        end
        for k in 1:4
    ]

    (
        β_fisher=fisher_acnr.β,β_image=β_image,
        β_relative_error=abs(β_image-fisher_acnr.β)/max(abs(β_image),eps()),
        empirical_wiener_sd=empirical_wiener_sd,
        zero_residual_sd=zero_residual_sd,
        synth_max_abs_error_HU=synth_error,
        axial_lag1=axial_lag1,
        gain_median=median(gains),
        gain_zero_fraction=count(x -> x ≤ 1e-6,gains)/length(gains),
        gain_high_fraction=count(x -> x ≥ 0.9,gains)/length(gains),
        basis_roi_sd=(water=std(W[mask,:]),iodine=std(I[mask,:])),
        basis_roi_rho=basis_rho,
        I0_counts=Float64.(nchannel_basis.I0),
        median_object_counts=object_counts,
    )
end;

# ╔═╡ 0e3e1191-edfb-4e5d-8845-6ba9be895538
let
    a = fisher_acnr_adversarial
    energies = fisher_acnr.energies
    fig = Mke.Figure(size=(900,500))
    ax = Mke.Axis(
        fig[1,1];title="Adversarial Noise Floors",
        subtitle="Tests covariance propagation and method ceiling",
        xlabel="VMI Energy (keV)",ylabel="Solid-water σ (HU)",
        xticks=(energies,string.(Int.(energies))),
    )
    Mke.lines!(
        ax,energies,fisher_acnr_metrics.raw_sd;
        color=:black,linewidth=3,label="Common-kernel raw",
    )
    Mke.lines!(
        ax,energies,fisher_acnr_metrics.denoised_sd;
        color=:dodgerblue,linewidth=3,label="Projection-Fisher β",
    )
    Mke.lines!(
        ax,energies,a.empirical_wiener_sd;
        color=:darkorange,linewidth=3,label="Image-ROI β",
    )
    Mke.lines!(
        ax,energies,a.zero_residual_sd;
        color=:crimson,linewidth=3,linestyle=:dash,
        label="Residual deleted (ceiling)",
    )
    for curve in (
        fisher_acnr_metrics.raw_sd,fisher_acnr_metrics.denoised_sd,
        a.empirical_wiener_sd,a.zero_residual_sd,
    )
        Mke.scatter!(ax,energies,curve;markersize=11)
    end
    Mke.axislegend(ax;position=:rt)
    fig
end

# ╔═╡ bf6152a1-c29f-4292-a197-f2e2cdd7e9e0
let
    a = fisher_acnr_adversarial
    md"""
    **Adversarial audit results**

    - VMI algebra/unit maximum error: **$(round(Float64(a.synth_max_abs_error_HU),sigdigits=4)) HU**
    - Projection-Fisher β: **$(round(a.β_fisher,digits=5))**
    - Empirical post-FBP image β: **$(round(a.β_image,digits=5))**
    - β relative disagreement: **$(round(100a.β_relative_error,digits=2))%**
    - Image-β Wiener noise (HU): **$(join(round.(a.empirical_wiener_sd,digits=1),", "))**
    - Zero-residual method floor (HU): **$(join(round.(a.zero_residual_sd,digits=1),", "))**
    - Residual axial lag-one correlation: **$(round(a.axial_lag1,digits=4))**
    - Wiener radial gain median: **$(round(a.gain_median,digits=4))**
    - Gain bins at zero / above 0.9: **$(round(100a.gain_zero_fraction,digits=1))% / $(round(100a.gain_high_fraction,digits=1))%**
    - Raw common-kernel basis ROI SD — water: **$(round(a.basis_roi_sd.water,digits=4)) g/cm³**, iodine: **$(round(a.basis_roi_sd.iodine,digits=5)) g/cm³**
    - Raw image-domain basis correlation: **$(round(a.basis_roi_rho,digits=4))**
    - Per-bin air counts: **$(join(round.(a.I0_counts,digits=1),", "))**
    - Median transmitted object counts: **$(join(round.(a.median_object_counts,digits=1),", "))**

    A large β disagreement identifies invalid covariance propagation in the
    first implementation. A still-unacceptable zero-residual floor proves
    that preserving one noisy reference mode cannot solve the absolute-noise
    problem even with perfect removal of the other mode.
    """
end

# ╔═╡ 7f515bef-0690-43aa-8d54-e61019c60380
md"""
### Upstream exposure and count-statistics tests

These tests determine whether the denoiser is being asked to compensate for
an exposure/count problem:

1. independently reconstruct the protocol-derived incident photons per
   detector pixel and compare them with the detected four-bin air counts;
2. compare repeated-row log-count variance with the Poisson prediction
   \(\operatorname{Var}[-\log(Y/I_0)]\approx1/\lambda\);
3. resample fitted rays at \(1\times\), \(4\times\), and \(16\times\) counts
   with independent Poisson noise, repeat the complete four-bin profile
   solve, and test the required \(1/\sqrt{N}\) material-noise scaling.
"""

# ╔═╡ 4975fdf4-5479-454c-a2e8-04a0bd755859
nchannel_exposure_audit = let
    spectrum_E,spectrum_weights = BS.resolve_source_spectrum_without_bowtie(
        sim_opts,protocol;scanner=scanner,
    )
    incident_I0 = BS.compute_detector_I0(
        sim_bins.geom,protocol,sum(spectrum_weights),
    )
    magnification = sim_bins.geom.SDD/sim_bins.geom.SAD
    detector_col_mm = sim_bins.geom.pixel_size*10*magnification
    detector_row_mm = sim_bins.geom.pixel_row_size*10*magnification
    detector_area_mm2 = detector_col_mm*detector_row_mm
    total_detected_I0 = sum(Float64.(sim_bins.I0_bins))
    (
        total_mAs=protocol.mA*protocol.rotation_time,
        mAs_per_view=protocol.mA*protocol.rotation_time/protocol.views,
        detector_col_mm=detector_col_mm,
        detector_row_mm=detector_row_mm,
        detector_area_mm2=detector_area_mm2,
        incident_I0=incident_I0,
        detected_I0=total_detected_I0,
        detected_fraction=total_detected_I0/incident_I0,
        bin_I0=Float64.(sim_bins.I0_bins),
        spectrum_energy_range=extrema(spectrum_E),
    )
end;

# ╔═╡ 0cb20a11-9250-43d3-beac-2bb9e0560a7f
nchannel_log_variance_test = let
    nrow = size(sim_bins.bins[1],2)
    rows = max(1,nrow ÷ 4):min(nrow,nrow-nrow ÷ 4)
    observed = zeros(Float64,4)
    predicted = zeros(Float64,4)
    samples = zeros(Int,4)
    for k in 1:4
        h = sim_bins.bins[k]
        I0 = Float64(sim_bins.I0_bins[k])
        for view in axes(h,3),col in axes(h,1)
            values = Float64.(h[col,rows,view])
            mean(values) > 0.1 || continue
            counts = I0 .* exp.(-values)
            h_centered = values .- mean(values)
            n = length(values)
            observed[k] += sum(abs2,h_centered)
            predicted[k] += (1-1/n)*sum(1.0 ./ max.(counts,1.0))
            samples[k] += n
        end
    end
    (
        observed_variance=observed ./ max.(samples .- 1,1),
        predicted_variance=predicted ./ max.(samples .- 1,1),
        ratio=observed ./ max.(predicted,eps(Float64)),
        samples=samples,
    )
end;

# ╔═╡ 12527954-7d88-4b5d-9958-41b797affee4
function nchannel_poisson_dose_trial(
    A_truth,C_truth,λ_truth,scale::Float32,seed::Int;
    controls=nchannel_controls,
)
    rng = BS.Random.MersenneTwister(seed)
    hs = map(1:4) do k
        I0 = scale*nchannel_basis.I0[k]
        Float32[
            -log(Float32(max(BS._poisson_sample(
                rng,Float64(scale*λ),
            ),1))/I0)
            for λ in λ_truth[k]
        ]
    end
    hs = [reshape(h,size(A_truth)) for h in hs]
    hs_gpu = to_gpu.(hs)
    I_gpu,W_gpu = similar(hs_gpu[1]),similar(hs_gpu[1])
    flags_gpu = similar(hs_gpu[1],UInt8)
    nchannel_profile_tile!(
        I_gpu,W_gpu,flags_gpu,hs_gpu...,
        to_gpu(scale .* nchannel_basis.Φ),
        to_gpu(nchannel_basis.μρ_I),to_gpu(nchannel_basis.μρ_W),
        to_gpu(scale .* nchannel_basis.I0),
        to_gpu(nchannel_basis.μI_eff),to_gpu(nchannel_basis.μW_eff),
        nchannel_basis.normal_II,nchannel_basis.normal_IW,
        nchannel_basis.normal_WW,controls,
    )
    (
        error_I=Array(I_gpu) .- A_truth,
        error_W=Array(W_gpu) .- C_truth,
        flags=Array(flags_gpu),
    )
end

# ╔═╡ 0b50613e-d9bd-4905-9964-dcf14e04c5e9
nchannel_dose_scaling_test = let
    full_I = sino_basis_nchannel.sino_iodine
    full_W = sino_basis_nchannel.sino_water
    cols = 1:8:size(full_I,1)
    row_mid = size(full_I,2) ÷ 2 + 1
    rows = (row_mid-3):(row_mid+4)
    views = 1:12:size(full_I,3)
    A = Float32.(full_I[cols,rows,views])
    C = Float32.(full_W[cols,rows,views])
    original_flags = sino_basis_nchannel.boundary_flag[cols,rows,views]
    valid = (original_flags .== 0x00) .& (C .> 1.0f0)

    λ = map(1:4) do k
        λk = zeros(Float32,size(A))
        for e in eachindex(nchannel_basis.E)
            @. λk += nchannel_basis.Φ[e,k] * exp(
                -nchannel_basis.μρ_I[e]*A -
                nchannel_basis.μρ_W[e]*C
            )
        end
        λk
    end
    base_var_I = mean(
        nchannel_fisher.var_iodine[cols,rows,views][valid]
    )
    base_cov_IW = mean(
        nchannel_fisher.cov_iodine_water[cols,rows,views][valid]
    )
    base_var_W = mean(
        nchannel_fisher.var_water[cols,rows,views][valid]
    )

    scales = Float32[1,4,16]
    seeds = 4101:4106
    rows_out = map(scales) do scale
        errors_I = Float64[]
        errors_W = Float64[]
        boundary_count = 0
        valid_count = 0
        for seed in seeds
            trial = nchannel_poisson_dose_trial(A,C,λ,scale,seed)
            keep = valid .& (trial.flags .== 0x00)
            append!(errors_I,Float64.(trial.error_I[keep]))
            append!(errors_W,Float64.(trial.error_W[keep]))
            boundary_count += count(x -> !iszero(x),trial.flags[valid])
            valid_count += count(valid)
        end
        centered_I = errors_I .- mean(errors_I)
        centered_W = errors_W .- mean(errors_W)
        covariance = [
            mean(abs2,centered_I) mean(centered_I .* centered_W)
            mean(centered_I .* centered_W) mean(abs2,centered_W)
        ]
        (
            scale=Float64(scale),n=length(errors_I),
            bias_I=mean(errors_I),bias_W=mean(errors_W),
            sd_I=std(errors_I),sd_W=std(errors_W),
            covariance=covariance,
            rho=covariance[1,2]/sqrt(covariance[1,1]*covariance[2,2]),
            fisher_ratio_I=covariance[1,1]/(base_var_I/scale),
            fisher_ratio_W=covariance[2,2]/(base_var_W/scale),
            fisher_rho=(base_cov_IW/scale) /
                sqrt((base_var_I/scale)*(base_var_W/scale)),
            boundary_fraction=boundary_count/max(valid_count,1),
        )
    end
    (
        results=rows_out,
        iodine_scaling=[
            r.sd_I*sqrt(r.scale)/rows_out[1].sd_I for r in rows_out
        ],
        water_scaling=[
            r.sd_W*sqrt(r.scale)/rows_out[1].sd_W for r in rows_out
        ],
        subset_shape=size(A),valid_rays=count(valid),
        repetitions=length(seeds),
        trial_data=(A=A,C=C,λ=λ,valid=valid),
    )
end;

# ╔═╡ 804d3ec2-8e56-4496-a44f-882bb8d426b4
nchannel_optimizer_precision_test = let
    data = nchannel_dose_scaling_test.trial_data
    scale = 16.0f0
    tight_controls = merge(
        nchannel_controls,
        (outer_iterations=16,inner_iterations=12),
    )
    function summarize(controls)
        error_I,error_W = Float64[],Float64[]
        boundary,total = 0,0
        for seed in 5101:5103
            trial = nchannel_poisson_dose_trial(
                data.A,data.C,data.λ,scale,seed;controls,
            )
            keep = data.valid .& (trial.flags .== 0x00)
            append!(error_I,Float64.(trial.error_I[keep]))
            append!(error_W,Float64.(trial.error_W[keep]))
            boundary += count(x -> !iszero(x),trial.flags[data.valid])
            total += count(data.valid)
        end
        (
            sd_I=std(error_I),sd_W=std(error_W),
            bias_I=mean(error_I),bias_W=mean(error_W),
            boundary_fraction=boundary/total,
        )
    end
    (
        standard=summarize(nchannel_controls),
        tight=summarize(tight_controls),
        tight_controls=tight_controls,
    )
end;

# ╔═╡ b16d08c9-29d8-4998-adcd-3ad25f93a1f0
let
    e = nchannel_exposure_audit
    l = nchannel_log_variance_test
    d = nchannel_dose_scaling_test
    md"""
    **Exposure and count-statistics results**

    - Total exposure: **$(round(e.total_mAs,digits=2)) mAs**
    - Exposure per view: **$(round(e.mAs_per_view,digits=5)) mAs**
    - Binned detector pixel: **$(round(e.detector_col_mm,digits=4)) × $(round(e.detector_row_mm,digits=4)) mm**
    - Detector area: **$(round(e.detector_area_mm2,digits=4)) mm²**
    - Incident air photons/pixel/view: **$(round(e.incident_I0,digits=1))**
    - Detected four-bin air counts: **$(round(e.detected_I0,digits=1))**
    - Detected/incident fraction after bowtie, QE, and DRM: **$(round(100e.detected_fraction,digits=2))%**
    - Per-bin empirical/Poisson log-variance ratio: **$(join(round.(l.ratio,digits=3),", "))**
    - Dose-sweep subset: **$(d.valid_rays) rays × $(d.repetitions) noise realizations**
    - Iodine \(σ\sqrt{N}\) relative to 1×: **$(join(round.(d.iodine_scaling,digits=3),", "))**
    - Water \(σ\sqrt{N}\) relative to 1×: **$(join(round.(d.water_scaling,digits=3),", "))**
    - Iodine SD by count scale (g/cm²): **$(join(round.([r.sd_I for r in d.results],digits=6),", "))**
    - Water SD by count scale (g/cm²): **$(join(round.([r.sd_W for r in d.results],digits=5),", "))**
    - Iodine bias by count scale (g/cm²): **$(join(round.([r.bias_I for r in d.results],digits=6),", "))**
    - Water bias by count scale (g/cm²): **$(join(round.([r.bias_W for r in d.results],digits=5),", "))**
    - Dose-sweep boundary fractions: **$(join(round.([r.boundary_fraction for r in d.results],digits=4),", "))**
    - Repeated-seed iodine variance / mean \(F^{-1}\): **$(join(round.([r.fisher_ratio_I for r in d.results],digits=3),", "))**
    - Repeated-seed water variance / mean \(F^{-1}\): **$(join(round.([r.fisher_ratio_W for r in d.results],digits=3),", "))**
    - Repeated-seed empirical material correlation: **$(join(round.([r.rho for r in d.results],digits=3),", "))**
    - Fisher-predicted material correlation: **$(join(round.([r.fisher_rho for r in d.results],digits=3),", "))**
    - 16× standard/tight iodine SD: **$(round(nchannel_optimizer_precision_test.standard.sd_I,digits=6)) / $(round(nchannel_optimizer_precision_test.tight.sd_I,digits=6)) g/cm²**
    - 16× standard/tight water SD: **$(round(nchannel_optimizer_precision_test.standard.sd_W,digits=5)) / $(round(nchannel_optimizer_precision_test.tight.sd_W,digits=5)) g/cm²**
    - 16× standard/tight iodine bias: **$(round(nchannel_optimizer_precision_test.standard.bias_I,digits=6)) / $(round(nchannel_optimizer_precision_test.tight.bias_I,digits=6)) g/cm²**
    - 16× standard/tight water bias: **$(round(nchannel_optimizer_precision_test.standard.bias_W,digits=5)) / $(round(nchannel_optimizer_precision_test.tight.bias_W,digits=5)) g/cm²**

    Ideal Poisson behavior gives a log-variance ratio near one and a flat
    normalized \(σ\sqrt{N}\) sequence near one. Deviations isolate corrected
    count statistics, solver bounds, or model mismatch from ordinary quantum
    noise.
    """
end

# ╔═╡ c0c45d8d-3c5f-46af-8e65-03a83b20ee47
let
    d = nchannel_dose_scaling_test
    scales = [r.scale for r in d.results]
    fig = Mke.Figure(size=(1000,480))
    ax = Mke.Axis(
        fig[1,1];title="Controlled Poisson Dose Scaling",
        subtitle="Flat at 1.0 is ideal 1/√N behavior",
        xlabel="Count scale",ylabel="Normalized σ√N",
        xticks=(scales,["$(Int(s))×" for s in scales]),
    )
    Mke.lines!(
        ax,scales,d.iodine_scaling;
        color=:darkorange,linewidth=3,label="Iodine",
    )
    Mke.scatter!(ax,scales,d.iodine_scaling;color=:darkorange,markersize=14)
    Mke.lines!(
        ax,scales,d.water_scaling;
        color=:dodgerblue,linewidth=3,label="Water",
    )
    Mke.scatter!(ax,scales,d.water_scaling;color=:dodgerblue,markersize=14)
    Mke.hlines!(ax,[1.0];color=:black,linestyle=:dash)
    Mke.axislegend(ax;position=:rt)
    fig
end

# ╔═╡ 63d11d7a-8ff4-4e86-89c7-312199ef95ef
md"""
## All-photon composite-guided VMI experiment

The protected-mode experiment above is retained as a negative control. Its
zero-residual oracle proved that a noisy decomposed VMI cannot serve as the
protected structural image.

This experiment instead follows the photon statistics:

1. Sum the four **exclusive count bins before the logarithm**,
   \(Y_\Sigma=\sum_k I_{0,k}\exp(-h_k)\).
2. Reconstruct \(g=-\log(Y_\Sigma/I_{0,\Sigma})\) with the same common
   `SoftFilter` used for both material bases.
3. In positive linear-attenuation units, form for every energy
   \[
   V_E^{\rm out}=G\,
   \frac{L[V_E^{\rm raw}]}{L[G]+\epsilon}.
   \]
4. Use one radially symmetric low-pass transfer function and one cutoff for
   the complete energy series. No energy-specific denoising strength is
   allowed.

The adjacent detector rows are **not** summed: this notebook targets a native
approximately 0.4-mm slice, and those rows are different cone rays. The
axially averaged guide below is an explicitly labelled oracle diagnostic,
not the accepted native-slice result.
"""

# ╔═╡ 1662cde2-263f-473f-ae6e-f517df96058b
nchannel_all_photon_guide = let
    I0 = Float64.(sim_bins.I0_bins)
    I0sum = sum(I0)
    counts_sum = zeros(Float32,size(sim_bins.bins[1]))
    for k in 1:4
        @. counts_sum += Float32(I0[k]) * exp(-sim_bins.bins[k])
    end
    hsum = @. Float32(-log(max(counts_sum,1.0f0)/Float32(I0sum)))

    sino_gpu = to_gpu(hsum)
    ws = BS.create_fdk_recon_workspace(
        sino_gpu,sim_bins.geom,recon_opts.matrix_size;
        filter=BS.SoftFilter(),
    )
    measured = Float32.(Array(BS.reconstruct!(ws,sino_gpu,sim_bins.geom)))
    ws = nothing
    sino_gpu = nothing
    GC.gc(true)

    # The phantom is axially constant over the reconstructed central slab.
    # Averaging all reconstructed slices therefore gives a high-dose/oracle
    # guide with unchanged in-plane sampling but deliberately sacrificed
    # native z resolution. It is only a diagnostic ceiling.
    axial_mean = dropdims(mean(measured,dims=3),dims=3)
    oracle = repeat(axial_mean,1,1,size(measured,3))
    (
        sino=hsum,measured=measured,oracle=Float32.(oracle),
        I0_sum=I0sum,median_object_counts=median(counts_sum),
    )
end;

# ╔═╡ d8841fe7-c89f-4e73-930a-f12e42ed258e
function nchannel_radial_lowpass(
    volume::AbstractArray{<:Real,3},cutoff_cycles_per_pixel::Real,
)
    nx,ny,nz = size(volume)
    cutoff = Float64(cutoff_cycles_per_pixel)
    transfer = [
        exp(-0.5 * (
            hypot(min(i-1,nx-(i-1))/nx,min(j-1,ny-(j-1))/ny) /
            max(cutoff,eps(Float64))
        )^4)
        for i in 1:nx,j in 1:ny
    ]
    out = similar(volume,Float32)
    Threads.@threads for z in 1:nz
        spectrum = FFTW.fft(Float64.(@view volume[:,:,z]))
        out[:,:,z] .= Float32.(real.(FFTW.ifft(spectrum .* transfer)))
    end
    out
end

# ╔═╡ 220dc45d-83c6-4e2e-bcee-b65830337a61
function nchannel_hypr_series(guide,cutoff;energies=fisher_acnr.energies)
    W = fisher_common_fbp.vol_water
    I = fisher_common_fbp.vol_iodine
    low_guide = nchannel_radial_lowpass(guide,cutoff)
    water_mask = fisher_acnr.water_mask
    # A 5-sigma denominator floor is derived from the measured guide noise.
    guide_roi = Float64.(guide[water_mask,:])
    epsilon = 5std(guide_roi) / sqrt(length(guide_roi))
    denominator = max.(low_guide,Float32(epsilon))
    object_mask = low_guide .> Float32(5epsilon)

    raw_mu = Dict{Float64,Array{Float32,3}}()
    output_mu = Dict{Float64,Array{Float32,3}}()
    for E in energies
        μW = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        μI = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine,E)
        raw = @. Float32(μW*W + μI*I)
        ratio = nchannel_radial_lowpass(raw,cutoff) ./ denominator
        output = @. Float32(ifelse(object_mask,guide*ratio,raw))
        raw_mu[E] = raw
        output_mu[E] = output
    end
    (
        raw_mu=raw_mu,output_mu=output_mu,low_guide=low_guide,
        epsilon=epsilon,object_mask=object_mask,cutoff=Float64(cutoff),
    )
end

# ╔═╡ 2a246b35-0810-4c50-a208-b85d8b4bc63f
nchannel_hypr_sweep = let
    energies = fisher_acnr.energies
    water_mask = fisher_acnr.water_mask
    cutoffs = collect(0.03:0.02:0.31)
    μwater = Dict(
        E => BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        for E in energies
    )
    raw = nchannel_hypr_series(
        nchannel_all_photon_guide.measured,last(cutoffs);energies,
    )
    raw_sd = [
        1000std(raw.raw_mu[E][water_mask,:])/μwater[E] for E in energies
    ]
    raw_mean = [
        mean(raw.raw_mu[E][water_mask,:]) for E in energies
    ]

    function summarize(guide,cutoff)
        result = nchannel_hypr_series(guide,cutoff;energies)
        sd = [
            1000std(result.output_mu[E][water_mask,:])/μwater[E]
            for E in energies
        ]
        bias = [
            1000(
                mean(result.output_mu[E][water_mask,:])-raw_mean[j]
            )/μwater[E]
            for (j,E) in enumerate(energies)
        ]
        # Gradient-energy retention is an adversarial edge-transfer screen.
        # It is evaluated on the central slice after removing the DC level.
        function gradient_energy(image)
            z = size(image,3) ÷ 2 + 1
            s = Float64.(@view image[:,:,z])
            gx = diff(s,dims=1)
            gy = diff(s,dims=2)
            sqrt(sum(abs2,gx)+sum(abs2,gy))
        end
        edge_ratio = [
            gradient_energy(result.output_mu[E]) /
            max(gradient_energy(result.raw_mu[E]),eps(Float64))
            for E in energies
        ]
        (
            cutoff=Float64(cutoff),sd=sd,bias_HU=bias,
            edge_energy_ratio=edge_ratio,result=result,
            monotonic=all(diff(sd) .≤ 0),
        )
    end

    measured = [summarize(nchannel_all_photon_guide.measured,c) for c in cutoffs]
    oracle = [summarize(nchannel_all_photon_guide.oracle,c) for c in cutoffs]
    target = [60.0,40.0,35.0,35.0]
    acceptable(row) = row.monotonic &&
        all(row.sd .≤ target) &&
        all(abs.(row.bias_HU) .≤ 10.0) &&
        all(abs.(row.edge_energy_ratio .- 1.0) .≤ 0.05)
    measured_ok = findall(acceptable,measured)
    oracle_ok = findall(acceptable,oracle)
    measured_pick = isempty(measured_ok) ?
        measured[argmin([sum(abs2,r.sd .- target) for r in measured])] :
        measured[last(measured_ok)]
    oracle_pick = isempty(oracle_ok) ?
        oracle[argmin([sum(abs2,r.sd .- target) for r in oracle])] :
        oracle[last(oracle_ok)]
    (
        energies=energies,cutoffs=cutoffs,raw_sd=raw_sd,
        measured=measured,oracle=oracle,
        measured_pick=measured_pick,oracle_pick=oracle_pick,
        measured_has_acceptable=!isempty(measured_ok),
        oracle_has_acceptable=!isempty(oracle_ok),
    )
end;

# ╔═╡ 62f127a2-eb84-4ddd-a758-02eda218c5a6
let
    s = nchannel_hypr_sweep
    fig = Mke.Figure(size=(1100,500))
    ax = Mke.Axis(
        fig[1,1];title="Composite-guided VMI adversarial sweep",
        subtitle="One common cutoff across all energies",
        xlabel="VMI energy (keV)",ylabel="Solid-water σ (HU)",
        xticks=(s.energies,string.(Int.(s.energies))),
    )
    Mke.lines!(ax,s.energies,s.raw_sd;color=:black,linewidth=3,label="Raw")
    Mke.scatter!(ax,s.energies,s.raw_sd;color=:black,markersize=12)
    for (row,color,label) in (
        (s.measured_pick,:dodgerblue,"Measured composite"),
        (s.oracle_pick,:darkorange,"Axial-average oracle"),
    )
        Mke.lines!(ax,s.energies,row.sd;color,linewidth=3,label)
        Mke.scatter!(ax,s.energies,row.sd;color,markersize=12)
    end
    Mke.axislegend(ax;position=:rt)
    fig
end

# ╔═╡ dbe133d0-ffde-450d-97dd-e1309401f70b
let
    s = nchannel_hypr_sweep
    m,o = s.measured_pick,s.oracle_pick
    md"""
    **Composite-guide gate results**

    - Summed-bin air counts: **$(round(nchannel_all_photon_guide.I0_sum,digits=1))**
    - Median summed transmitted counts: **$(round(nchannel_all_photon_guide.median_object_counts,digits=1))**
    - Raw common-kernel noise (HU): **$(join(round.(s.raw_sd,digits=1),", "))**
    - Measured-guide selected cutoff: **$(round(m.cutoff,digits=3)) cycles/pixel**
    - Measured-guide noise (HU): **$(join(round.(m.sd,digits=1),", "))**
    - Measured-guide mean shifts (HU): **$(join(round.(m.bias_HU,digits=1),", "))**
    - Measured-guide gradient-energy ratios: **$(join(round.(m.edge_energy_ratio,digits=3),", "))**
    - Measured guide passes all gates: **$(s.measured_has_acceptable)**
    - Oracle-guide selected cutoff: **$(round(o.cutoff,digits=3)) cycles/pixel**
    - Oracle-guide noise (HU): **$(join(round.(o.sd,digits=1),", "))**
    - Oracle-guide mean shifts (HU): **$(join(round.(o.bias_HU,digits=1),", "))**
    - Oracle-guide gradient-energy ratios: **$(join(round.(o.edge_energy_ratio,digits=3),", "))**
    - Oracle guide passes all gates: **$(s.oracle_has_acceptable)**

    The oracle is diagnostic only. If it succeeds while the measured guide
    fails, the limitation is guide photon statistics. If both fail, the
    multiplicative contrast-transfer model or frequency split is rejected.
    Gradient energy is only a screening metric; a passing candidate still
    requires insert-specific TTF/MTF validation before acceptance.
    """
end

# ╔═╡ 393f8547-f96f-44ac-9665-f284297a0bfd
function nchannel_edge_width(image,label::UInt8=UInt8(26))
    truth = phantom_cpu.mask[:,:,size(phantom_cpu.mask,3) ÷ 2]
    idx = findall(==(label),truth)
    cx = mean(Float64(ci[1]) for ci in idx)
    cy = mean(Float64(ci[2]) for ci in idx)
    area_radius = sqrt(length(idx)/π)
    averaged = dropdims(mean(Float64.(image),dims=3),dims=3)
    spacing = 0.20
    distances = collect(-4.0:spacing:4.0)
    sums = zeros(Float64,length(distances))
    counts = zeros(Int,length(distances))
    for j in axes(averaged,2),i in axes(averaged,1)
        d = hypot(i-cx,j-cy)-area_radius
        -4.0 ≤ d ≤ 4.0 || continue
        b = clamp(round(Int,(d+4.0)/spacing)+1,1,length(distances))
        sums[b] += averaged[i,j]
        counts[b] += 1
    end
    profile = sums ./ max.(counts,1)
    # Reject remote threshold crossings caused by noise: smooth only along
    # the edge normal and choose the crossing nearest the known boundary.
    smooth = similar(profile)
    for i in eachindex(profile)
        lo,hi = max(firstindex(profile),i-2),min(lastindex(profile),i+2)
        smooth[i] = mean(@view profile[lo:hi])
    end
    inside = mean(profile[distances .≤ -2.5])
    outside = mean(profile[distances .≥ 2.5])
    normalized = (smooth .- inside) ./ (outside-inside)
    function crossing(level)
        candidates = Float64[]
        for i in 1:(length(distances)-1)
            y1,y2 = normalized[i],normalized[i+1]
            if (y1-level)*(y2-level) ≤ 0 && y1 != y2
                push!(candidates,distances[i] +
                    spacing*(level-y1)/(y2-y1))
            end
        end
        isempty(candidates) ? NaN :
            candidates[argmin(abs.(candidates))]
    end
    x10,x90 = crossing(0.1),crossing(0.9)
    (
        width_10_90=abs(x90-x10),distances=distances,
        profile=profile,normalized=normalized,
        center=(cx,cy),area_radius=area_radius,
    )
end

# ╔═╡ 18d6335a-d6a0-4742-a63e-c40250dd78b6
function nchannel_local_wiener_guide(guide,cutoff)
    local_mean = nchannel_radial_lowpass(guide,cutoff)
    local_second = nchannel_radial_lowpass(guide .* guide,cutoff)
    local_variance = max.(local_second .- local_mean .* local_mean,0.0f0)
    mask = fisher_acnr.water_mask
    noise_variance = Float32(std(Float64.(guide[mask,:]))^2)
    gain = @. Float32(clamp(
        1.0f0-noise_variance/max(local_variance,eps(Float32)),0.0f0,1.0f0,
    ))
    denoised = @. Float32(local_mean + gain*(guide-local_mean))
    (
        denoised=denoised,gain=gain,noise_variance=noise_variance,
        cutoff=Float64(cutoff),
    )
end

# ╔═╡ 9415717e-0f1e-472f-ab4e-ad5c643b9f33
nchannel_adaptive_guide_sweep = let
    energies = fisher_acnr.energies
    mask = fisher_acnr.water_mask
    μwater = Dict(
        E => BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        for E in energies
    )
    raw_reference = nchannel_hypr_series(
        nchannel_all_photon_guide.measured,0.31;energies,
    )
    raw_width = Dict(
        E => nchannel_edge_width(raw_reference.raw_mu[E]).width_10_90
        for E in energies
    )
    measured_guide_width = nchannel_edge_width(
        nchannel_all_photon_guide.measured,
    ).width_10_90
    cutoffs = collect(0.03:0.02:0.31)
    rows = map(cutoffs) do cutoff
        guide_filter = nchannel_local_wiener_guide(
            nchannel_all_photon_guide.measured,cutoff,
        )
        result = nchannel_hypr_series(
            guide_filter.denoised,cutoff;energies,
        )
        sd = [
            1000std(result.output_mu[E][mask,:])/μwater[E]
            for E in energies
        ]
        widths = Dict(
            E => nchannel_edge_width(result.output_mu[E]).width_10_90
            for E in energies
        )
        width_ratio = [widths[E]/raw_width[E] for E in energies]
        guide_width = nchannel_edge_width(
            guide_filter.denoised,
        ).width_10_90
        guide_width_ratio = guide_width/measured_guide_width
        output_to_guide_width = [widths[E]/guide_width for E in energies]
        mean_shift = [
            1000(
                mean(result.output_mu[E][mask,:]) -
                mean(result.raw_mu[E][mask,:])
            )/μwater[E]
            for E in energies
        ]
        (
            cutoff=Float64(cutoff),sd=sd,width_ratio=width_ratio,
            guide_width_ratio=guide_width_ratio,
            output_to_guide_width=output_to_guide_width,
            mean_shift_HU=mean_shift,result=result,
            guide=guide_filter,
            monotonic=all(diff(sd) .≤ 0),
        )
    end
    target = [60.0,40.0,35.0,35.0]
    valid(row) = row.monotonic &&
        all(row.sd .≤ target) &&
        all(abs.(row.mean_shift_HU) .≤ 10.0) &&
        abs(row.guide_width_ratio-1.0) ≤ 0.05 &&
        all(abs.(row.output_to_guide_width .- 1.0) .≤ 0.05)
    valid_indices = findall(valid,rows)
    score(row) = sum(abs2,row.sd .- target) +
        1e4(
            abs2(row.guide_width_ratio-1.0) +
            sum(abs2,row.output_to_guide_width .- 1.0)
        )
    selected = isempty(valid_indices) ?
        rows[argmin(score.(rows))] : rows[last(valid_indices)]
    (
        energies=energies,rows=rows,selected=selected,
        has_acceptable=!isempty(valid_indices),raw_width=raw_width,
        measured_guide_width=measured_guide_width,
    )
end;

# ╔═╡ 56702bee-3c47-4328-b920-c004cc1ca5ff
let
    a = nchannel_adaptive_guide_sweep.selected
    md"""
    **Noise-derived adaptive-guide gate**

    - Selected common cutoff: **$(round(a.cutoff,digits=3)) cycles/pixel**
    - VMI noise (HU): **$(join(round.(a.sd,digits=1),", "))**
    - Water mean shifts (HU): **$(join(round.(a.mean_shift_HU,digits=1),", "))**
    - Denoised/measured guide 10–90% edge-width ratio:
      **$(round(a.guide_width_ratio,digits=3))**
    - VMI/denoised-guide edge-width ratios:
      **$(join(round.(a.output_to_guide_width,digits=3),", "))**
    - VMI/raw-VMI edge-width ratios (noise-sensitive diagnostic):
      **$(join(round.(a.width_ratio,digits=3),", "))**
    - Monotonic noise: **$(a.monotonic)**
    - Passes absolute-noise, bias, and ±5% edge-width gates:
      **$(nchannel_adaptive_guide_sweep.has_acceptable)**

    The local Wiener gain is calculated from the measured composite variance;
    it is not an energy-specific or visually selected strength. The insert
    edge is averaged over all axial replicates solely to estimate the TTF
    robustly; the processed native slices themselves are not averaged.
    """
end

# ╔═╡ 764394d6-8103-4af9-816a-dfe8c12fc6da
let
    h = nchannel_hypr_sweep
    a = nchannel_adaptive_guide_sweep
    rows = String[
        "| cutoff | measured σ HU | oracle σ HU | adaptive σ HU | adaptive guide edge ratio |",
        "|---:|:---|:---|:---|---:|",
    ]
    for i in eachindex(h.cutoffs)
        measured_text = join(round.(h.measured[i].sd,digits=0),",")
        oracle_text = join(round.(h.oracle[i].sd,digits=0),",")
        adaptive_text = join(round.(a.rows[i].sd,digits=0),",")
        push!(rows,
            "| $(round(h.cutoffs[i],digits=2)) | " *
            "$measured_text | $oracle_text | $adaptive_text | " *
            "$(round(a.rows[i].guide_width_ratio,digits=2)) |"
        )
    end
    md"""
    **Common-cutoff audit (40, 70, 100, 140 keV)**

    $(join(rows,"\n"))
    """
end

# ╔═╡ bfd8f2d8-b708-478b-9d84-6b02e1eb9de3
md"""
### Likelihood-gated native-volume guide

The axial-average oracle is now replaced by a falsifiable native-volume
estimator. For target slice \(z\), candidate slice \(q\), and a fixed
3×3 decision patch \(P\), define

```math
d_{zq}^2(\mathbf r)
=\frac{1}{|P|}\sum_{\mathbf p\in P}
[G_z(\mathbf r+\mathbf p)-G_q(\mathbf r+\mathbf p)]^2.
```

Two independent measurements of the same signal have expected squared
difference \(2\sigma_G^2\). The non-negative signal discrepancy and its
likelihood weight are therefore

```math
\delta_{zq}^2=\max(d_{zq}^2-2\sigma_G^2,0),
\qquad
w_{zq}=\exp\!\left[-\frac{|P|\delta_{zq}^2}{4\sigma_G^2}\right].
```

Only the **co-located** value \(G_q(\mathbf r)\) is combined. The patch
controls statistical compatibility but contributes no neighboring pixel
values, so the operation introduces no direct in-plane convolution.
The noise variance comes from adjacent-slice differences in the eroded
water ROI. There is no fitted range bandwidth or energy-specific strength.
"""

# ╔═╡ 34c4bda9-3ca6-4f1c-99c3-f146697fcaf3
function nchannel_box3_mean_squared_difference(a,b)
    nx,ny = size(a)
    out = zeros(Float32,nx,ny)
    Threads.@threads for j in 1:ny
        for i in 1:nx
            total = 0.0f0
            count = 0
            for dj in -1:1,di in -1:1
                ii,jj = i+di,j+dj
                if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny
                    difference = Float32(a[ii,jj]-b[ii,jj])
                    total += difference*difference
                    count += 1
                end
            end
            out[i,j] = total/count
        end
    end
    out
end

# ╔═╡ b278c84b-7506-45c6-9f83-2c603a64ebbb
function nchannel_likelihood_gated_guide(
    guide::AbstractArray{<:Real,3};noise_variance=nothing,
)
    nx,ny,nz = size(guide)
    mask = fisher_acnr.water_mask
    σ² = if isnothing(noise_variance)
        differences = Float64[]
        for z in 1:(nz-1)
            append!(
                differences,
                Float64.(
                    @view(guide[:,:,z+1])[mask] .-
                    @view(guide[:,:,z])[mask]
                ),
            )
        end
        mean(abs2,differences)/2
    else
        Float64(noise_variance)
    end
    numerator = zeros(Float32,nx,ny,nz)
    denominator = zeros(Float32,nx,ny,nz)
    patch_samples = 9.0f0
    σ²f = Float32(max(σ²,eps(Float32)))
    for z in 1:nz,q in 1:nz
        distance² = nchannel_box3_mean_squared_difference(
            @view(guide[:,:,z]),@view(guide[:,:,q]),
        )
        weight = @. Float32(exp(
            -patch_samples*max(distance²-2σ²f,0.0f0)/(4σ²f)
        ))
        @views @. numerator[:,:,z] += weight*guide[:,:,q]
        @views @. denominator[:,:,z] += weight
    end
    denoised = numerator ./ max.(denominator,eps(Float32))
    effective_samples = @. Float32(
        denominator^2 / max(denominator,eps(Float32))
    )
    (
        denoised=denoised,noise_variance=σ²,
        noise_sd=sqrt(σ²),effective_samples=effective_samples,
        patch_samples=Int(patch_samples),
    )
end

# ╔═╡ d7f0f806-452d-4028-a635-f1415290a09e
nchannel_likelihood_guide = nchannel_likelihood_gated_guide(
    nchannel_all_photon_guide.measured,
);

# ╔═╡ f01f8544-e949-40b1-a5f0-13d8d04f222a
nchannel_likelihood_hypr_sweep = let
    energies = fisher_acnr.energies
    mask = fisher_acnr.water_mask
    μwater = Dict(
        E => BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        for E in energies
    )
    cutoffs = collect(0.015:0.005:0.08)
    rows = map(cutoffs) do cutoff
        result = nchannel_hypr_series(
            nchannel_likelihood_guide.denoised,cutoff;energies,
        )
        sd = [
            1000std(result.output_mu[E][mask,:])/μwater[E]
            for E in energies
        ]
        mean_shift = [
            1000(
                mean(result.output_mu[E][mask,:]) -
                mean(result.raw_mu[E][mask,:])
            )/μwater[E]
            for E in energies
        ]
        guide_width_ratio = nchannel_edge_width(
            nchannel_likelihood_guide.denoised,
        ).width_10_90 / nchannel_edge_width(
            nchannel_all_photon_guide.measured,
        ).width_10_90
        output_width = [
            nchannel_edge_width(result.output_mu[E]).width_10_90
            for E in energies
        ]
        output_spread = maximum(output_width)/minimum(output_width)
        (
            cutoff=Float64(cutoff),result=result,sd=sd,
            mean_shift_HU=mean_shift,
            guide_width_ratio=guide_width_ratio,
            output_width=output_width,output_spread=output_spread,
            monotonic=all(diff(sd) .≤ 1.0),
        )
    end
    target = [50.0,30.0,27.5,25.0]
    valid(row) = row.monotonic &&
        all(row.sd .≤ [60.0,40.0,35.0,35.0]) &&
        all(abs.(row.mean_shift_HU) .≤ 10.0) &&
        abs(row.guide_width_ratio-1.0) ≤ 0.05 &&
        row.output_spread ≤ 1.05
    valid_indices = findall(valid,rows)
    score(row) = sum(abs2,row.sd .- target) +
        1e4abs2(row.guide_width_ratio-1.0) +
        1e4abs2(row.output_spread-1.0)
    selected = isempty(valid_indices) ?
        rows[argmin(score.(rows))] : rows[last(valid_indices)]
    (
        energies=energies,cutoffs=cutoffs,rows=rows,selected=selected,
        has_acceptable=!isempty(valid_indices),
    )
end;

# ╔═╡ e756839a-54c4-43d3-9e05-9463ee9ddd0e
nchannel_axial_edge_adversary = let
    guide = nchannel_all_photon_guide.measured
    mask = fisher_acnr.water_mask
    water_level = mean(Float64.(guide[mask,:]))
    nz = size(guide,3)
    step_index = nz ÷ 2
    contrasts_HU = [50.0,100.0,300.0]
    rows = map(contrasts_HU) do contrast
        synthetic = copy(guide)
        Δ = Float32(water_level*contrast/1000)
        for z in (step_index+1):nz
            @views synthetic[:,:,z][mask] .+= Δ
        end
        filtered = nchannel_likelihood_gated_guide(
            synthetic;
            noise_variance=nchannel_likelihood_guide.noise_variance,
        ).denoised
        before = [
            mean(Float64.(@view synthetic[:,:,z])[mask]) for z in 1:nz
        ]
        after = [
            mean(Float64.(@view filtered[:,:,z])[mask]) for z in 1:nz
        ]
        measured_step = mean(after[(step_index+1):end]) -
            mean(after[1:step_index])
        leakage = (
            after[step_index+1]-after[step_index]
        ) / max(Δ,eps(Float32))
        (
            contrast_HU=contrast,before=before,after=after,
            recovered_step_fraction=measured_step/Δ,
            adjacent_step_fraction=leakage,
        )
    end
    rows
end;

# ╔═╡ 484246a0-4240-44d8-884c-c074794cda57
let
    s = nchannel_likelihood_hypr_sweep
    r = s.selected
    axial = nchannel_axial_edge_adversary
    md"""
    **Likelihood-gated native-volume result**

    - Estimated native-guide noise SD:
      **$(round(nchannel_likelihood_guide.noise_sd,sigdigits=4)) μ units**
    - Selected common split: **$(round(r.cutoff,digits=3)) cycles/pixel**
    - VMI noise (HU): **$(join(round.(r.sd,digits=1),", "))**
    - Water mean shifts (HU):
      **$(join(round.(r.mean_shift_HU,digits=1),", "))**
    - Denoised/measured guide iodine-edge width ratio:
      **$(round(r.guide_width_ratio,digits=3))**
    - Cross-energy VMI edge-width spread:
      **$(round(r.output_spread,digits=3))**
    - Axial 50/100/300-HU recovered step fractions:
      **$(join(round.([x.recovered_step_fraction for x in axial],digits=3),", "))**
    - Axial adjacent-slice step fractions:
      **$(join(round.([x.adjacent_step_fraction for x in axial],digits=3),", "))**
    - Passes current magnitude, monotonicity, bias, and edge gates:
      **$(s.has_acceptable)**
    """
end

# ╔═╡ e6a9ed03-b6ec-4241-826f-dcbfb34b3631
md"""
### Dose–slice-resolution feasibility boundary

No denoiser can be judged independently of the resolution and photon budget
of the comparator. This sweep uses the all-photon composite and a fixed
0.03-cycle/pixel HYPR split while changing only the number of statistically
independent 0.4-mm axial replicas in the guide. It is a diagnostic of the
required photon factor:

- `n = 1` is the native 0.4-mm slice;
- averaging `n` replicas corresponds to an `n`-fold photon increase, or to
  an \(0.4n\)-mm slab for this axially uniform phantom;
- the in-plane FBP kernel and sampling remain unchanged;
- longitudinal resolution is **not** claimed to remain unchanged.

This explicitly distinguishes a true denoising result from a thicker-slice
or higher-dose result.
"""

# ╔═╡ 542ddc41-7d19-497e-b530-f3414736db3b
nchannel_resolution_dose_boundary = let
    guide = nchannel_all_photon_guide.measured
    energies = fisher_acnr.energies
    mask = fisher_acnr.water_mask
    μwater = Dict(
        E => BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        for E in energies
    )
    nz = size(guide,3)
    rows = map(1:nz) do n
        slab = dropdims(mean(@view(guide[:,:,1:n]),dims=3),dims=3)
        slab_volume = repeat(slab,1,1,nz)
        result = nchannel_hypr_series(slab_volume,0.03;energies)
        sd = [
            1000std(result.output_mu[E][mask,:])/μwater[E]
            for E in energies
        ]
        width_ratio = nchannel_edge_width(slab_volume).width_10_90 /
            nchannel_edge_width(guide).width_10_90
        (
            replicas=n,thickness_mm=0.4n,
            equivalent_mAs=n*nchannel_exposure_audit.total_mAs,
            sd=sd,width_ratio=width_ratio,result=result,
        )
    end
    requested = [60.0,40.0,35.0,35.0]
    meets = findall(row ->
        all(row.sd .≤ requested) &&
        all(diff(row.sd) .≤ 1.0),
        rows,
    )
    (
        rows=rows,
        first_meeting=isempty(meets) ? nothing : rows[first(meets)],
        requested=requested,
    )
end;

# ╔═╡ 8e6551a4-5481-4812-bbf7-707fd27c289d
let
    b = nchannel_resolution_dose_boundary
    rows = String[
        "| replicas | slab (mm) | equivalent mAs | σ HU at 40/70/100/140 | in-plane edge ratio |",
        "|---:|---:|---:|:---|---:|",
    ]
    for r in b.rows
        noise_text = join(round.(r.sd,digits=1),", ")
        push!(rows,
            "| $(r.replicas) | $(round(r.thickness_mm,digits=1)) | " *
            "$(round(r.equivalent_mAs,digits=0)) | $noise_text | " *
            "$(round(r.width_ratio,digits=3)) |"
        )
    end
    first_text = if isnothing(b.first_meeting)
        "No tested photon factor meets the requested envelope."
    else
        r = b.first_meeting
        "First factor meeting the magnitude envelope: **$(r.replicas)×**, " *
        "equivalent to **$(round(r.thickness_mm,digits=1)) mm** in this " *
        "uniform phantom or **$(round(r.equivalent_mAs,digits=0)) mAs** " *
        "at native thickness."
    end
    md"""
    **Resolution/dose boundary**

    $first_text

    $(join(rows,"\n"))
    """
end

# ╔═╡ 2ee9014d-aebc-4a25-a1ca-30c5ba3a24b6
md"""
### Resolution-matched 4.8-mm candidate audit

The following candidate is evaluated at a declared 4.8-mm longitudinal
resolution, matching the 12-replica photon support used by its all-photon
guide. It preserves the original four exclusive energy bins, the complete
four-bin Cong likelihood, the common in-plane `SoftFilter`, and the common
0.03-cycle/pixel HYPR split. It does **not** claim native 0.4-mm z
resolution.
"""

# ╔═╡ 79332028-9529-4837-892b-39e5fc3c81a2
nchannel_slab_candidate = let
    energies = fisher_acnr.energies
    guide = nchannel_all_photon_guide.oracle
    mask = fisher_acnr.water_mask
    μwater = Dict(
        E => BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        for E in energies
    )
    cutoffs = collect(0.005:0.005:0.060)
    trials = map(cutoffs) do cutoff
        result = nchannel_hypr_series(guide,cutoff;energies)
        sd = [
            1000std(result.output_mu[E][mask,:])/μwater[E]
            for E in energies
        ]
        (cutoff=cutoff,result=result,sd=sd)
    end
    target = [60.0,40.0,35.0,35.0]
    admissible = findall(t ->
        all(t.sd .≤ target) && all(diff(t.sd) .≤ 0),
        trials,
    )
    selected = isempty(admissible) ?
        trials[argmin([
            sum(max.(t.sd .- target,0).^2) +
            100sum(max.(diff(t.sd),0).^2)
            for t in trials
        ])] :
        trials[last(admissible)]
    hu = Dict{Float64,Array{Float32,3}}()
    for E in energies
        μw = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        hu[E] = @. Float32(1000*(selected.result.output_mu[E]/μw-1))
    end
    (
        energies=energies,HU=hu,mu=selected.result.output_mu,
        noise_sd_HU=selected.sd,thickness_mm=4.8,
        cutoff=selected.cutoff,guide=guide,trials=trials,
        has_admissible=!isempty(admissible),
    )
end;

# ╔═╡ c414d52f-b563-43a0-b0e0-85fc9ea73e1c
nchannel_slab_quantitative_audit = let
    candidate = nchannel_slab_candidate
    truth = phantom_cpu.mask[:,:,size(phantom_cpu.mask,3) ÷ 2]
    materials = phantom_cpu.materials
    labels = (
        Ca=UInt8[10,11,12,13,14,15,16],
        I=UInt8[20,21,22,23,24,25,26],
    )
    function core_roi(label)
        idx = findall(==(label),truth)
        cx = mean(Float64(ci[1]) for ci in idx)
        cy = mean(Float64(ci[2]) for ci in idx)
        [
            CartesianIndex(i,j)
            for j in axes(truth,2),i in axes(truth,1)
            if (i-cx)^2+(j-cy)^2 ≤ 8.0^2
        ]
    end
    function roi_mean(volume,roi)
        mean(Float64(volume[ci,z]) for z in axes(volume,3),ci in roi)
    end
    function linear_fit(x,y)
        xc = x .- mean(x)
        slope = sum(xc .* (y .- mean(y)))/sum(abs2,xc)
        (slope=slope,intercept=mean(y)-slope*mean(x))
    end
    groups = Dict{Symbol,Any}()
    for group in (:Ca,:I)
        measured = zeros(Float64,length(labels[group]),length(candidate.energies))
        theoretical = similar(measured)
        for (i,label) in pairs(labels[group])
            roi = core_roi(label)
            material = materials[Int(label)+1]
            for (j,E) in pairs(candidate.energies)
                measured[i,j] = roi_mean(candidate.HU[E],roi)
                μ = BS.compute_μ_at_energy(material,E)
                μw = BS.compute_μ_at_energy(BS.XA.Materials.water,E)
                theoretical[i,j] = 1000*(μ-μw)/μw
            end
        end
        fits = [
            linear_fit(theoretical[:,j],measured[:,j])
            for j in axes(measured,2)
        ]
        groups[group]=(measured=measured,theoretical=theoretical,fits=fits)
    end

    water_mask = fisher_acnr.water_mask
    samples = hcat([
        vec(Float64.(candidate.HU[E][water_mask,:]))
        for E in candidate.energies
    ]...)
    covariance = cov(samples)
    correlation = cor(samples)

    edge_widths = Dict(
        E => nchannel_edge_width(candidate.mu[E]).width_10_90
        for E in candidate.energies
    )
    guide_width = nchannel_edge_width(candidate.guide).width_10_90
    edge_ratios = [edge_widths[E]/guide_width for E in candidate.energies]
    (
        groups=groups,covariance=covariance,correlation=correlation,
        edge_widths=edge_widths,guide_width=guide_width,
        edge_ratios=edge_ratios,
    )
end;

# ╔═╡ 602c2ef8-7364-4d26-8308-72a073f44485
function nchannel_radial_nps(volume,halfwidth::Int=48)
    nx,ny,nz = size(volume)
    cx,cy = nx ÷ 2 + 1,ny ÷ 2 + 1
    mean_image = dropdims(mean(Float64.(volume),dims=3),dims=3)
    npx,npy = 2halfwidth,2halfwidth
    X = hcat(
        ones(npx*npy),
        repeat(collect(1.0:npx),npy),
        repeat(collect(1.0:npy)',npx)[:],
    )
    spectrum = zeros(Float64,npx,npy)
    variance_sum = 0.0
    for z in 1:nz
        patch = Float64.(
            @view volume[
                (cx-halfwidth):(cx+halfwidth-1),
                (cy-halfwidth):(cy+halfwidth-1),z,
            ]
        ) .- @view mean_image[
            (cx-halfwidth):(cx+halfwidth-1),
            (cy-halfwidth):(cy+halfwidth-1),
        ]
        residual = vec(patch)-X*(X\vec(patch))
        spectrum .+= abs2.(FFTW.fft(reshape(residual,npx,npy)))/(npx*npy)
        variance_sum += sum(abs2,residual)/(length(residual)-3)
    end
    spectrum ./= nz
    nrad = floor(Int,hypot(npx ÷ 2,npy ÷ 2))+1
    sums,counts = zeros(Float64,nrad),zeros(Int,nrad)
    for j in 1:npy,i in 1:npx
        radius = floor(Int,hypot(
            min(i-1,npx-(i-1)),min(j-1,npy-(j-1)),
        ))+1
        sums[radius] += spectrum[i,j]
        counts[radius] += 1
    end
    radial = sums ./ max.(counts,1)
    frequencies = collect(0:(nrad-1))/npx
    (
        frequency=frequencies,power=radial,
        integral=variance_sum/nz,
        peak_frequency=frequencies[argmax(radial[2:end])+1],
    )
end

# ╔═╡ 784aaf59-2422-408c-b6ab-2004981f1c00
nchannel_slab_nps = Dict(
    E => nchannel_radial_nps(nchannel_slab_candidate.HU[E])
    for E in nchannel_slab_candidate.energies
);

# ╔═╡ 8a08befc-af91-4ebc-b75d-f520885515d6
let
    c = nchannel_slab_candidate
    q = nchannel_slab_quantitative_audit
    slopes(group) = [x.slope for x in q.groups[group].fits]
    intercepts(group) = [x.intercept for x in q.groups[group].fits]
    md"""
    **4.8-mm candidate quantitative gates**

    - VMI noise (HU): **$(join(round.(c.noise_sd_HU,digits=1),", "))**
    - Selected common split: **$(round(c.cutoff,digits=3)) cycles/pixel**
    - An admissible common split exists: **$(c.has_admissible)**
    - Strictly monotonic: **$(all(diff(c.noise_sd_HU) .≤ 0))**
    - Iodine NIST slopes: **$(join(round.(slopes(:I),digits=3),", "))**
    - Iodine intercepts (HU):
      **$(join(round.(intercepts(:I),digits=1),", "))**
    - Calcium NIST slopes:
      **$(join(round.(slopes(:Ca),digits=3),", "))**
    - Calcium intercepts (HU):
      **$(join(round.(intercepts(:Ca),digits=1),", "))**
    - VMI/guide iodine-edge width ratios:
      **$(join(round.(q.edge_ratios,digits=3),", "))**
    - NPS peak frequencies (cycles/pixel):
      **$(join(round.([nchannel_slab_nps[E].peak_frequency for E in c.energies],digits=3),", "))**
    - Cross-energy water-noise correlation, 40↔140 keV:
      **$(round(q.correlation[1,end],digits=3))**
    """
end

# ╔═╡ d8b8608f-6fa8-4c5f-85ee-17e3877efb4c
let
    c = nchannel_slab_candidate
    z = size(first(values(c.HU)),3) ÷ 2 + 1
    fig = Mke.Figure(size=(1180,920))
    for (i,E) in enumerate(c.energies)
        row,col = divrem(i-1,2) .+ (1,1)
        ax = Mke.Axis(
            fig[row,col];title="$(Int(E)) keV · 4.8 mm",
            aspect=Mke.DataAspect(),
        )
        Mke.heatmap!(
            ax,c.HU[E][:,:,z];colormap=:grays,
            colorrange=(-200,800),
        )
        Mke.hidedecorations!(ax)
    end
    fig
end

# ╔═╡ 2d447ccd-b408-42a6-8a1b-af770e3277e4
md"""
## Direct FBP of the Material Sinograms

Each decomposed material sinogram is reconstructed independently with the
same FDK geometry, matrix, and per-basis filters used by
`04_pcct_vmi.jl`: a soft custom iodine filter and `SoftFilter()` for water.
No ACNR or empirical calibration is inserted between decomposition and
reconstruction.

Area density (g/cm²) therefore reconstructs to volume density (g/cm³).
"""

# ╔═╡ 1ff5e801-ef54-45ff-b0a8-e780e2e6cb63
nchannel_basis_volumes = let
    geom = sino_basis_nchannel.geom
    matrix_size = recon_opts.matrix_size
    iodine_filter = BS.CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.40, 0.12, 0.03, 0.001),
    )
    water_filter = BS.SoftFilter()

    function direct_fbp(sino_cpu, filter)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter,
        )
        volume = Float32.(Array(BS.reconstruct!(ws, sino_gpu, geom)))
        ws = nothing
        sino_gpu = nothing
        GC.gc(true)
        volume
    end

    (
        vol_iodine = direct_fbp(
            sino_basis_nchannel.sino_iodine, iodine_filter,
        ),
        vol_water = direct_fbp(
            sino_basis_nchannel.sino_water, water_filter,
        ),
        geom = geom,
    )
end;

# ╔═╡ 65a40efc-c9ff-41d4-8f42-46d6737119d5
let
    mid_z = size(nchannel_basis_volumes.vol_water, 3) ÷ 2 + 1
    slice_I = nchannel_basis_volumes.vol_iodine[:,:,mid_z]
    slice_W = nchannel_basis_volumes.vol_water[:,:,mid_z]
    qrange(x) = (
        Float64(quantile(vec(x), 0.01)),
        Float64(quantile(vec(x), 0.99)),
    )

    fig = Mke.Figure(size = (1400, 580))
    panels = (
        (1, 1, 2, "Iodine Basis FBP", "g/cm³", slice_I, qrange(slice_I)),
        (1, 3, 4, "Water Basis FBP", "g/cm³", slice_W, qrange(slice_W)),
    )
    for (r, c, cb, title, label, image, range) in panels
        ax = Mke.Axis(fig[r,c]; title, aspect=Mke.DataAspect())
        Mke.heatmap!(ax, image; colormap=:viridis, colorrange=range)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[r,cb]; colormap=:viridis, colorrange=range, label, width=16)
    end
    fig
end

# ╔═╡ 7707308d-e855-4111-9367-ef4ffc66da8b
md"""
## Direct VMI Synthesis

The reconstructed basis densities are mixed analytically at each requested
monoenergetic energy:

```math
\mu(E)=c_W(\mu/\rho)_W(E)+c_I(\mu/\rho)_I(E).
```

`BS.synth_vmi_2basis` expects iodine concentration in mg/mL, so the
reconstructed iodine density is converted from g/cm³ by multiplying by
1000. The result is returned directly in HU relative to monoenergetic water.
"""

# ╔═╡ 9161e7bf-eaa9-4d48-9d93-ab743ff3a0c2
nchannel_vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 5cc1f9ba-4a5f-4a78-86dd-e0d844a09a28
nchannel_vmi_HU = let
    iodine_mg_per_mL = nchannel_basis_volumes.vol_iodine .* 1000.0f0
    Dict(
        E => BS.synth_vmi_2basis(
            nchannel_basis_volumes.vol_water, iodine_mg_per_mL;
            energy_keV = E,
        )
        for E in nchannel_vmi_energies
    )
end;

# ╔═╡ c0231062-8372-44c3-a3f0-574d39a1c326
let
    HU_window = (-200.0, 500.0)
    sample = nchannel_vmi_HU[first(nchannel_vmi_energies)]
    mid_z = size(sample, 3) ÷ 2 + 1
    fig = Mke.Figure(size = (1180, 1180))

    for (k, E) in enumerate(nchannel_vmi_energies)
        r = (k - 1) ÷ 2 + 1
        c = (k - 1) % 2 + 1
        ax = Mke.Axis(
            fig[r,c]; title="$(Int(E)) keV VMI",
            aspect=Mke.DataAspect(), titlesize=30,
        )
        Mke.heatmap!(
            ax, nchannel_vmi_HU[E][:,:,mid_z];
            colormap=:grays, colorrange=HU_window,
        )
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1:2,3]; colormap=:grays, colorrange=HU_window,
        label="HU", width=16,
    )
    fig
end

# ╔═╡ 3e8869e6-4f74-4ac8-a048-10d55aa76e20
md"""
## NIST HU Verification

Measured rod HU is the mean of an 8-pixel-radius core ROI across every
reconstructed slice. The theoretical value is calculated independently from
the phantom material and NIST attenuation data:

```math
\mathrm{HU}_{r,E}
=1000\frac{\mu_r(E)-\mu_{\mathrm{water}}(E)}
{\mu_{\mathrm{water}}(E)}.
```

Solid lines and markers are the four-channel VMI measurements. Dashed lines
are the corresponding NIST predictions; no fit or empirical correction is
applied.
"""

# ╔═╡ 594208fc-89bc-4c7b-b989-623a0c432207
begin
    NCHANNEL_ROD_LABELS = (
        Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
        I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
    )
    NCHANNEL_ROD_NAMES = (
        Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
        I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
    )
end

# ╔═╡ f2bd7681-b328-48c4-b578-71d01f1bd5eb
nchannel_rod_data = let
    materials = phantom_cpu.materials
    mask_2d = phantom_cpu.mask[:,:,size(phantom_cpu.mask,3) ÷ 2]
    nx, ny = size(mask_2d)
    ROI_RADIUS_PX = 8

    function rod_roi(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("No phantom voxels found for rod label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        r² = Float64(ROI_RADIUS_PX)^2
        [
            CartesianIndex(i,j)
            for j in max(1,floor(Int,cy-ROI_RADIUS_PX)):min(ny,ceil(Int,cy+ROI_RADIUS_PX))
            for i in max(1,floor(Int,cx-ROI_RADIUS_PX)):min(nx,ceil(Int,cx+ROI_RADIUS_PX))
            if (i-cx)^2 + (j-cy)^2 ≤ r²
        ]
    end

    labels_all = vcat(
        collect(NCHANNEL_ROD_LABELS.Ca), collect(NCHANNEL_ROD_LABELS.I),
    )
    rois = Dict(label => rod_roi(label) for label in labels_all)
    μ_water = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
        for E in nchannel_vmi_energies
    )

    function measured_hu(volume, label)
        roi = rois[label]
        sum(volume[ci,z] for z in axes(volume,3), ci in roi) /
            (size(volume,3) * length(roi))
    end
    theoretical_hu(material, E) =
        1000.0 * (
            BS.compute_μ_at_energy(material, E) - μ_water[E]
        ) / μ_water[E]

    out = Dict{Symbol,NamedTuple}()
    for group in (:Ca, :I)
        labels = NCHANNEL_ROD_LABELS[group]
        measured = zeros(Float64, length(labels), length(nchannel_vmi_energies))
        theoretical = similar(measured)
        for (i,label) in pairs(labels), (j,E) in pairs(nchannel_vmi_energies)
            measured[i,j] = measured_hu(nchannel_vmi_HU[E], label)
            theoretical[i,j] = theoretical_hu(materials[Int(label)+1], E)
        end
        out[group] = (
            labels, names=NCHANNEL_ROD_NAMES[group], measured, theoretical,
        )
    end
    out
end;

# ╔═╡ 48ef9fbf-aa1d-4535-81db-70d9551667a6
let
    fig = Mke.Figure(size=(1180,580))
    panels = (
        (group=:Ca, title="Calcium rods", subtitle="50–600 mg/mL",
         cmap=Mke.cgrad(:Oranges,7; categorical=true), ylim=(0,4200)),
        (group=:I, title="Iodine rods", subtitle="2–20 mg/mL",
         cmap=Mke.cgrad(:GnBu,7; categorical=true), ylim=(0,1500)),
    )

    for (col,p) in pairs(panels)
        ax = Mke.Axis(
            fig[1,col]; title=p.title, subtitle=p.subtitle,
            xlabel="VMI energy (keV)", ylabel="HU",
            xticks=nchannel_vmi_energies,
        )
        Mke.ylims!(ax,p.ylim...)
        d = nchannel_rod_data[p.group]
        rod_lines = Any[]
        for i in eachindex(d.names)
            color = p.cmap[i]
            Mke.scatterlines!(
                ax,nchannel_vmi_energies,vec(d.measured[i,:]);
                color,linewidth=2.5,markersize=9,
            )
            Mke.lines!(
                ax,nchannel_vmi_energies,vec(d.theoretical[i,:]);
                color,linewidth=1.6,linestyle=:dash,
            )
            push!(rod_lines,Mke.LineElement(color=color,linewidth=2.5))
        end
        Mke.axislegend(
            ax,
            vcat(
                [Mke.MarkerElement(color=:black,marker=:circle,markersize=9),
                 Mke.LineElement(color=:black,linewidth=1.6,linestyle=:dash)],
                rod_lines,
            ),
            vcat(["Measured","NIST theoretical"],collect(d.names));
            position=:rt,framevisible=true,rowgap=1,
        )
    end
    fig
end

# ╔═╡ df5f06e2-6075-40b0-9eec-660c75002e9a
md"""
### Solid-Water ROI by VMI Energy

The same deeply eroded solid-water region used by the established PCCT
notebook is overlaid on the 70 keV image. Its mean HU is then measured at
40, 70, 100, and 140 keV. Energy-dependent drift indicates spectral or
decomposition bias; an energy-independent offset indicates a water-basis
scale bias.
"""

# ╔═╡ 7e8bbfaf-35f1-4fc9-8ddd-58c4ca50ca6d
nchannel_water_roi = let
    mask_2d = phantom_cpu.mask[:,:,size(phantom_cpu.mask,3) ÷ 2]
    raw = mask_2d .== UInt8(BS.REGION_SOLID_WATER)
    eroded = collect(BS.erode_mask_2d(raw; erode_px=12.0))
    count(eroded) > 0 || error("The 12-pixel solid-water erosion removed the ROI")
    (mask=eroded, indices=findall(eroded))
end;

# ╔═╡ e2ccf044-c0b8-414a-ad09-f5e40a0b71b5
let
    fig = Mke.Figure(size=(1180,580))
    HU_window = (-200.0,500.0)
    mid_z = size(nchannel_vmi_HU[70.0],3) ÷ 2 + 1
    bg = nchannel_vmi_HU[70.0][:,:,mid_z]
    overlay = Float32[b ? 1.0f0 : NaN32 for b in nchannel_water_roi.mask]

    ax1 = Mke.Axis(
        fig[1,1]; title="Eroded Water Region",
        subtitle="Overlaid on 70 keV VMI",aspect=Mke.DataAspect(),
    )
    Mke.heatmap!(ax1,bg; colormap=:grays,colorrange=HU_window)
    Mke.heatmap!(
        ax1,overlay; colormap=:reds,alpha=0.5,nan_color=(:white,0.0),
    )
    Mke.hidedecorations!(ax1)

    function mean_water_hu(volume)
        sum(volume[ci,z] for z in axes(volume,3), ci in nchannel_water_roi.indices) /
            (size(volume,3) * length(nchannel_water_roi.indices))
    end
    water_hu = [
        mean_water_hu(nchannel_vmi_HU[E]) for E in nchannel_vmi_energies
    ]
    colors = Mke.cgrad(
        :plasma,length(nchannel_vmi_energies); categorical=true,
    )
    ax2 = Mke.Axis(
        fig[1,2]; title="Water Region Mean HU",subtitle="Per VMI Energy",
        xlabel="VMI Energy (keV)",ylabel="HU",
        xticks=(
            collect(eachindex(nchannel_vmi_energies)),
            ["$(Int(E))" for E in nchannel_vmi_energies],
        ),
    )
    Mke.barplot!(
        ax2,eachindex(water_hu),water_hu;
        color=[colors[i] for i in eachindex(water_hu)],
        strokecolor=:black,strokewidth=1,
    )
    Mke.hlines!(ax2,[0.0]; color=:black,linewidth=1,linestyle=:dash)
    for (i,h) in pairs(water_hu)
        Mke.text!(
            ax2,i,h;text="$(round(h,digits=1)) HU",
            align=(:center,h ≥ 0 ? :bottom : :top),
            offset=(0,h ≥ 0 ? 4 : -4),
        )
    end
    y_max = max(15.0,1.2*maximum(abs,water_hu))
    Mke.ylims!(ax2,-y_max,y_max)
    fig
end

# ╔═╡ 3f35047e-00ba-44a1-a8a6-d8325305736d
md"""
### Solid-Water Noise Magnitude

Noise is the voxelwise standard deviation in the same eroded solid-water
ROI, pooled across reconstructed slices. The line shape makes a monotonic
decrease or a U-shaped VMI noise trend directly visible.
"""

# ╔═╡ 1f0cfed9-bd4c-43bc-a042-0de34a55598e
let
    energies = sort(collect(nchannel_vmi_energies))
    noise_hu = [
        std(nchannel_vmi_HU[E][nchannel_water_roi.mask, :])
        for E in energies
    ]

    decreasing = all(diff(noise_hu) .< 0)
    increasing = all(diff(noise_hu) .> 0)
    i_min = argmin(noise_hu)
    u_shaped =
        1 < i_min < length(noise_hu) &&
        all(diff(noise_hu[1:i_min]) .< 0) &&
        all(diff(noise_hu[i_min:end]) .> 0)
    trend = decreasing ? "monotonic decreasing" :
        increasing ? "monotonic increasing" :
        u_shaped ? "U-shaped" : "non-monotonic / mixed"

    fig = Mke.Figure(size=(760,480))
    ax = Mke.Axis(
        fig[1,1];
        title="Solid-Water VMI Noise",
        subtitle="Observed trend: $trend",
        xlabel="VMI Energy (keV)",
        ylabel="Noise σ (HU)",
        xticks=(energies, ["$(Int(E))" for E in energies]),
    )
    Mke.lines!(ax,energies,noise_hu; color=:dodgerblue,linewidth=3)
    Mke.scatter!(
        ax,energies,noise_hu;
        color=:dodgerblue,strokecolor=:black,strokewidth=1,markersize=16,
    )
    for (E,σ) in zip(energies,noise_hu)
        Mke.text!(
            ax,E,σ;
            text="$(round(σ,digits=1)) HU",
            align=(:center,:bottom),offset=(0,8),
        )
    end
    Mke.ylims!(ax,0,nothing)
    fig
end

# ╔═╡ Cell order:
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
# ╠═171294a2-26bd-49e2-ac92-9df48ae5444f
# ╠═69358294-97f2-4782-94d7-c29c747c45f4
# ╠═9ae27110-5c47-442b-a98e-d137599570f2
# ╠═492bb299-678d-4e6f-8c21-1e9178cc2beb
# ╠═9f8d5cd4-147e-4359-95bc-cc096a53f0e7
# ╠═2ff539c9-a678-403c-b629-8068a332a0e9
# ╠═320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
# ╠═86c52e9e-7987-4504-93e6-128017f5e703
# ╟─551f84fe-d7b4-48f9-a475-0c63178a6ede
# ╟─59a5079b-a711-4f28-b3d6-665f0d91fb72
# ╟─4d519e7e-e412-45d8-a54e-97d8f9fe5cf7
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╟─a9d9212b-782c-4552-8d23-dcf74c052826
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╟─ba6a1636-c0bc-4881-8ba7-d0e9dc8d90a8
# ╠═f9c0af7a-addd-4249-96fb-b9078765fbd1
# ╟─07b0ef06-7c00-4f26-80fd-def32a742597
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─3e5c24af-1bb2-457e-b2a8-236aeb1fcff3
# ╟─ebf6817f-baef-48c9-9e7d-8ae303c41ad0
# ╠═be97f48a-9b4e-40e6-aff1-20e6392d73b3
# ╟─9533c523-9ff2-4094-808a-ff9e7424a3e9
# ╟─6dce791e-d934-4f23-95fb-5fa08f45a612
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╟─a27649c0-e623-473e-bfbe-60fc8dd6d976
# ╠═4985581f-616d-4bb7-ab9b-967d7250b28b
# ╠═73371177-0498-4eda-897b-651c94f43e83
# ╠═ad092cf9-14b2-4f32-bad4-1e7962961fc2
# ╟─831c3a54-76d2-4dfc-b93a-faf3661518a3
# ╟─7bfdb7d7-6a46-4803-88dd-8c08d41a3780
# ╟─c456f690-518d-4fb9-941e-a5662b101ff5
# ╠═f9168282-ad86-446c-ac04-1aac09d4c1a2
# ╠═54bc4490-08c6-4d61-a687-d491976b2761
# ╠═a0930dd5-c28e-45a1-9bb6-4be300ba19d2
# ╟─7d987c7a-50ac-4827-9145-d07efe14747a
# ╟─2d6e6dda-5a23-42c8-8744-116029c2db9a
# ╟─f8688ab5-5e22-432e-bf04-45d8a7763576
# ╠═40b39aa1-2aed-4804-80db-cab52a8e4964
# ╠═308003a4-fe8e-4690-a492-2bc786e96965
# ╠═f8138d36-d35a-403a-a824-26ab60d75727
# ╠═94b9c44f-90a5-4dad-95d8-3b47d4036575
# ╠═ca67864e-a436-4bf5-be0e-1924019885f5
# ╠═42dafcec-17be-4ed4-8e6a-3185ee9f0b0b
# ╟─fa1fb5e8-64ee-4e9b-8ce7-dd8eb00f0e22
# ╠═c2f2b9a2-aead-4de1-9132-a8d6bc10ded6
# ╟─efbb5ea0-a2ad-4d94-a145-34bc65a059cc
# ╟─22c17c01-1162-49e8-80d4-3b013f1e0672
# ╟─3d019ead-ea76-49b6-bc64-a6e534b9612a
# ╠═d13ac73b-b554-4a42-a1e1-29feb2365514
# ╟─0e3e1191-edfb-4e5d-8845-6ba9be895538
# ╟─bf6152a1-c29f-4292-a197-f2e2cdd7e9e0
# ╟─7f515bef-0690-43aa-8d54-e61019c60380
# ╠═4975fdf4-5479-454c-a2e8-04a0bd755859
# ╠═0cb20a11-9250-43d3-beac-2bb9e0560a7f
# ╠═12527954-7d88-4b5d-9958-41b797affee4
# ╠═0b50613e-d9bd-4905-9964-dcf14e04c5e9
# ╠═804d3ec2-8e56-4496-a44f-882bb8d426b4
# ╟─b16d08c9-29d8-4998-adcd-3ad25f93a1f0
# ╟─c0c45d8d-3c5f-46af-8e65-03a83b20ee47
# ╟─63d11d7a-8ff4-4e86-89c7-312199ef95ef
# ╠═1662cde2-263f-473f-ae6e-f517df96058b
# ╠═d8841fe7-c89f-4e73-930a-f12e42ed258e
# ╠═220dc45d-83c6-4e2e-bcee-b65830337a61
# ╠═2a246b35-0810-4c50-a208-b85d8b4bc63f
# ╟─62f127a2-eb84-4ddd-a758-02eda218c5a6
# ╟─dbe133d0-ffde-450d-97dd-e1309401f70b
# ╠═393f8547-f96f-44ac-9665-f284297a0bfd
# ╠═18d6335a-d6a0-4742-a63e-c40250dd78b6
# ╠═9415717e-0f1e-472f-ab4e-ad5c643b9f33
# ╟─56702bee-3c47-4328-b920-c004cc1ca5ff
# ╟─764394d6-8103-4af9-816a-dfe8c12fc6da
# ╟─bfd8f2d8-b708-478b-9d84-6b02e1eb9de3
# ╠═34c4bda9-3ca6-4f1c-99c3-f146697fcaf3
# ╠═b278c84b-7506-45c6-9f83-2c603a64ebbb
# ╠═d7f0f806-452d-4028-a635-f1415290a09e
# ╠═f01f8544-e949-40b1-a5f0-13d8d04f222a
# ╠═e756839a-54c4-43d3-9e05-9463ee9ddd0e
# ╟─484246a0-4240-44d8-884c-c074794cda57
# ╟─e6a9ed03-b6ec-4241-826f-dcbfb34b3631
# ╠═542ddc41-7d19-497e-b530-f3414736db3b
# ╟─8e6551a4-5481-4812-bbf7-707fd27c289d
# ╟─2ee9014d-aebc-4a25-a1ca-30c5ba3a24b6
# ╠═79332028-9529-4837-892b-39e5fc3c81a2
# ╠═c414d52f-b563-43a0-b0e0-85fc9ea73e1c
# ╠═602c2ef8-7364-4d26-8308-72a073f44485
# ╠═784aaf59-2422-408c-b6ab-2004981f1c00
# ╟─8a08befc-af91-4ebc-b75d-f520885515d6
# ╟─d8b8608f-6fa8-4c5f-85ee-17e3877efb4c
# ╟─2d447ccd-b408-42a6-8a1b-af770e3277e4
# ╠═1ff5e801-ef54-45ff-b0a8-e780e2e6cb63
# ╟─65a40efc-c9ff-41d4-8f42-46d6737119d5
# ╟─7707308d-e855-4111-9367-ef4ffc66da8b
# ╠═9161e7bf-eaa9-4d48-9d93-ab743ff3a0c2
# ╠═5cc1f9ba-4a5f-4a78-86dd-e0d844a09a28
# ╟─c0231062-8372-44c3-a3f0-574d39a1c326
# ╟─3e8869e6-4f74-4ac8-a048-10d55aa76e20
# ╠═594208fc-89bc-4c7b-b989-623a0c432207
# ╠═f2bd7681-b328-48c4-b578-71d01f1bd5eb
# ╟─48ef9fbf-aa1d-4535-81db-70d9551667a6
# ╟─df5f06e2-6075-40b0-9eec-660c75002e9a
# ╠═7e8bbfaf-35f1-4fc9-8ddd-58c4ca50ca6d
# ╟─e2ccf044-c0b8-414a-ad09-f5e40a0b71b5
# ╟─3f35047e-00ba-44a1-a8a6-d8325305736d
# ╟─1f0cfed9-bd4c-43bc-a042-0de34a55598e
