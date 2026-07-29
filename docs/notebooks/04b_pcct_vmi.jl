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
    # No oracle/noise-reduction surrogate is used.
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

# ╔═╡ b3e1d768-eb02-4c2a-9363-c84077fedc32
md"""
### Pre-decomposition 5-mm row-count combination

For the present axial phantom the object is invariant over the active
longitudinal detector extent. The row-center cone factor is
\[
s_r=\sqrt{1+(z_r/\mathrm{SAD})^2}.
\]
The maximum departure from unity is reported below. When this departure is
negligible, the rows are repeated independent measurements of the same
in-plane ray to that stated tolerance. We may then sum **raw counts by native
energy bin before the logarithm**:
\[
Y_{k,\Sigma}=\sum_r I_{0,k}e^{-h_{k,r}},\qquad
h_{k,\Sigma}=-\log\frac{Y_{k,\Sigma}}{R I_{0,k}}.
\]
The four-bin likelihood is unchanged except that both \(\Phi_k\) and \(I_{0,k}\)
are multiplied by \(R\). This improves the photon support seen by the
nonlinear estimator while retaining all four spectral bins and the profiled
univariate solve. It represents a declared 5-mm slice, not native 0.4-mm
longitudinal resolution.
"""

# ╔═╡ 50ed35bb-df61-4862-a99b-ea37380f30d8
nchannel_slab_counts = let
    available_rows = size(sim_bins.bins[1],2)
    nrows = round(
        Int,protocol.collimation_mm/(10*sim_bins.geom.pixel_row_size),
    )
    nrows ≤ available_rows || error("Nominal active rows exceed simulated rows.")
    first_row = (available_rows-nrows) ÷ 2 + 1
    selected_rows = first_row:(first_row+nrows-1)
    row_positions = (
        collect(selected_rows) .- (available_rows+1)/2
    ) .* sim_bins.geom.pixel_row_size
    cone_scales = sqrt.(1 .+ (row_positions ./ sim_bins.geom.SAD).^2)
    bins = map(1:4) do k
        summed_transmission = dropdims(
            sum(
                exp.(-Float64.(sim_bins.bins[k][:,selected_rows,:])),
                dims=2,
            ),dims=2,
        )
        h = @. Float32(-log(max(summed_transmission,1e-12)/nrows))
        reshape(h,size(h,1),1,size(h,2))
    end
    (
        bins=bins,nrows=nrows,selected_rows=selected_rows,
        available_rows=available_rows,cone_scales=cone_scales,
        max_cone_relerr=maximum(abs.(cone_scales .- 1)),
        thickness_mm=nrows*10*sim_bins.geom.pixel_row_size,
    )
end;

# ╔═╡ b86a9c50-cb10-44b2-af2e-06bdd50943b7
sino_basis_nchannel_slab = let
    shape = size(nchannel_slab_counts.bins[1])
    sino_I = Array{Float32}(undef,shape)
    sino_W = Array{Float32}(undef,shape)
    flags = Array{UInt8}(undef,shape)
    scale = Float32(nchannel_slab_counts.nrows)
    Φ_gpu = to_gpu(scale .* nchannel_basis.Φ)
    μρ_I_gpu = to_gpu(nchannel_basis.μρ_I)
    μρ_W_gpu = to_gpu(nchannel_basis.μρ_W)
    I0_gpu = to_gpu(scale .* nchannel_basis.I0)
    μI_eff_gpu = to_gpu(nchannel_basis.μI_eff)
    μW_eff_gpu = to_gpu(nchannel_basis.μW_eff)
    elapsed = @elapsed for vrange in BS.tile_ranges(
        shape[3],nchannel_controls.tile_views,
    )
        hs = [
            to_gpu(Float32.(nchannel_slab_counts.bins[k][:,:,vrange]))
            for k in 1:4
        ]
        I_gpu,W_gpu = similar(hs[1]),similar(hs[1])
        flag_gpu = similar(hs[1],UInt8)
        nchannel_profile_tile!(
            I_gpu,W_gpu,flag_gpu,hs[1],hs[2],hs[3],hs[4],
            Φ_gpu,μρ_I_gpu,μρ_W_gpu,I0_gpu,μI_eff_gpu,μW_eff_gpu,
            nchannel_basis.normal_II,nchannel_basis.normal_IW,
            nchannel_basis.normal_WW,nchannel_controls,
        )
        sino_I[:,:,vrange] .= Array(I_gpu)
        sino_W[:,:,vrange] .= Array(W_gpu)
        flags[:,:,vrange] .= Array(flag_gpu)
    end
    (
        sino_iodine=sino_I,sino_water=sino_W,boundary_flag=flags,
        geom=sim_bins.geom,elapsed_s=elapsed,
    )
end;

# ╔═╡ ddfde8bb-ddff-44bb-8b70-b397725402cf
nchannel_slab_common_fbp = let
    nrows = sim_bins.geom.n_rows
    function common_fbp(sino_one_row)
        repeated = repeat(sino_one_row,1,nrows,1)
        sino_gpu = to_gpu(Float32.(repeated))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu,sim_bins.geom,recon_opts.matrix_size;
            filter=BS.SoftFilter(),
        )
        out = Float32.(Array(BS.reconstruct!(ws,sino_gpu,sim_bins.geom)))
        ws = nothing
        sino_gpu = nothing
        GC.gc(true)
        out
    end
    W = common_fbp(sino_basis_nchannel_slab.sino_water)
    I = common_fbp(sino_basis_nchannel_slab.sino_iodine)
    energies = [40.0,70.0,100.0,140.0]
    mu = Dict{Float64,Array{Float32,3}}()
    for E in energies
        μW = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
        μI = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine,E)
        mu[E] = @. Float32(μW*W + μI*I)
    end
    (vol_water=W,vol_iodine=I,mu=mu,energies=energies)
end;

# ╔═╡ aa1c494e-ccbf-4a0c-8d44-5f12d47f6e8c
md"""
### Matched-support all-photon guide after row-count combination

This repeats the composite-guided test with one crucial correction: the guide
and the material likelihood now use the same centered 4.94-mm raw-count
support. The guide is reconstructed with the same `SoftFilter`; only one
central valid slice is evaluated, because the row-combined sinogram defines
one slab estimate.
"""

# ╔═╡ af6ea59b-06a0-4527-9b6d-344e54dc3bee
nchannel_slab_guide = let
    I0 = Float64.(nchannel_basis.I0)
    weighted = zeros(Float64,size(nchannel_slab_counts.bins[1]))
    for k in 1:4
        weighted .+= I0[k] .* exp.(-nchannel_slab_counts.bins[k])
    end
    I0sum = sum(I0)
    h = Float32.(-log.(max.(weighted,1e-12)./I0sum))
    repeated = repeat(h,1,sim_bins.geom.n_rows,1)
    sino_gpu = to_gpu(repeated)
    ws = BS.create_fdk_recon_workspace(
        sino_gpu,sim_bins.geom,recon_opts.matrix_size;
        filter=BS.SoftFilter(),
    )
    volume = Float32.(Array(BS.reconstruct!(ws,sino_gpu,sim_bins.geom)))
    ws = nothing
    sino_gpu = nothing
    GC.gc(true)
    finite_counts = [
        count(isfinite,@view volume[:,:,z]) for z in axes(volume,3)
    ]
    z = argmax(finite_counts)
    (sino=h,volume=volume,slice=volume[:,:,z],z=z)
end;

# ╔═╡ 2d447ccd-b408-42a6-8a1b-af770e3277e4
md"""
## Generalized Four-Bin Cong Output

All four native photon-counting bins enter the profiled likelihood. The
14 geometrically equivalent detector rows are summed as raw counts before
the logarithm, matching the declared 4.94-mm slice support. The resulting
water and iodine sinograms are reconstructed with the same `SoftFilter`.

Area density (g/cm²) therefore reconstructs to volume density (g/cm³).
"""

# ╔═╡ 1ff5e801-ef54-45ff-b0a8-e780e2e6cb63
nchannel_basis_volumes = (
    vol_iodine=nchannel_slab_common_fbp.vol_iodine,
    vol_water=nchannel_slab_common_fbp.vol_water,
    geom=sino_basis_nchannel_slab.geom,
);

# ╔═╡ 65a40efc-c9ff-41d4-8f42-46d6737119d5
let
    mid_z = nchannel_slab_guide.z
    slice_I = nchannel_basis_volumes.vol_iodine[:,:,mid_z]
    slice_W = nchannel_basis_volumes.vol_water[:,:,mid_z]
    qrange(x) = let y=filter(isfinite,vec(x))
        (Float64(quantile(y,0.01)),Float64(quantile(y,0.99)))
    end

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
nchannel_vmi_raw_HU = let
    iodine_mg_per_mL = nchannel_basis_volumes.vol_iodine .* 1000.0f0
    Dict(
        E => BS.synth_vmi_2basis(
            nchannel_basis_volumes.vol_water, iodine_mg_per_mL;
            energy_keV = E,
        )
        for E in nchannel_vmi_energies
    )
end;

# ╔═╡ 1c55fa7b-e29a-4464-81f5-a75a6fce7228
md"""
## Fixed All-Energy Denoising

One all-photon guide, one common `σ = 5` pixel frequency split, one 9×9
local covariance window, and one common gain cap `β ≤ 6` are used for the
entire VMI series. No denoising parameter or reconstruction kernel is chosen
independently by keV.

The low-frequency quantitative content always comes from the four-bin Cong
VMI. Only the shared high-frequency structure is carried by the summed-bin
guide.
"""

# ╔═╡ a2ce06aa-31e8-4533-8617-40cb0f02b292
nchannel_vmi_HU = let
    energies = nchannel_vmi_energies
    z = nchannel_slab_guide.z
    raw_mu = [
        begin
            μW = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
            hu = nchannel_vmi_raw_HU[E][:,:,z]
            image = @. Float32(μW*(hu/1000+1))
            reshape(image,size(hu)...,1)
        end
        for E in energies
    ]

    mask_2d = phantom_cpu.mask[:,:,size(phantom_cpu.mask,3) ÷ 2]
    water_mask = collect(BS.erode_mask_2d(
        mask_2d .== UInt8(BS.REGION_SOLID_WATER);erode_px=12.0,
    ))
    guide = Float32.(nchannel_slab_guide.slice)
    guide_water = mean(filter(isfinite,vec(guide[water_mask])))
    μW70 = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,70.0)
    guide_proxy = reshape(
        Float32.(guide .* (μW70/guide_water)),size(guide)...,1,
    )

    volumes = [raw_mu...,guide_proxy]
    workspace = BS.create_mono_plus_workspace(
        raw_mu[1];n_energies=length(volumes),
    )
    result = BS.apply_mono_plus_regression!(
        workspace,volumes,[energies...,-1.0];
        E_noise_opt=-1.0,σ_lp_px=5.0,window=4,
        beta_max=6.0,verbose=false,
    )
    Dict(
        E => begin
            μW = BS.compute_mass_μ_at_energy(BS.XA.Materials.water,E)
            @. Float32(1000*(result.volumes[j]/μW-1))
        end
        for (j,E) in pairs(energies)
    )
end;

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

# ╔═╡ a6d34bb4-9af4-46ab-8b20-20cd4dbfe33f
md"""
### NIST HU Linearity

The same rod measurements are plotted against their independently calculated
NIST HU values. Each VMI energy receives one least-squares line, while the
dashed identity line shows perfect quantitative agreement.

Slope near one and intercept near zero indicate correct quantitative scaling;
\(R^2\) measures linearity, and RMSE reports the remaining HU error directly.
Calcium and iodine are shown separately because their HU ranges differ
substantially.
"""

# ╔═╡ dbb6d6e4-e9a0-4b99-933d-d6937a486622
let
    fig = Mke.Figure(size=(1000,1200))
    energy_colors = Dict(
        40.0 => Mke.RGBf(0.85,0.27,0.10),
        70.0 => Mke.RGBf(0.95,0.65,0.13),
        100.0 => Mke.RGBf(0.13,0.59,0.85),
        140.0 => Mke.RGBf(0.10,0.27,0.65),
    )

    function fit_lr(x,y)
        x̄,ȳ = mean(x),mean(y)
        sxx = sum((x .- x̄).^2)
        β = sum((x .- x̄).*(y .- ȳ)) / sxx
        α = ȳ - β*x̄
        residuals = y .- (α .+ β.*x)
        ss_res = sum(abs2,residuals)
        ss_tot = sum(abs2,y .- ȳ)
        (
            slope=β,
            intercept=α,
            r²=1-ss_res/ss_tot,
            rmse=sqrt(ss_res/length(y)),
        )
    end

    panels = (
        (:Ca,"Calcium Rods","50–600 mg/mL"),
        (:I,"Iodine Rods","2–20 mg/mL"),
    )
    for (row,(group,title,subtitle)) in pairs(panels)
        d = nchannel_rod_data[group]
        ax = Mke.Axis(
            fig[row,1];
            title,subtitle,
            xlabel="Theoretical HU",ylabel="Measured HU",
            titlesize=32,subtitlesize=24,
            xlabelsize=22,ylabelsize=22,
            xticklabelsize=16,yticklabelsize=16,
        )
        lim_lo = min(0.0,minimum(d.measured),minimum(d.theoretical))
        lim_hi = max(maximum(d.measured),maximum(d.theoretical))*1.05
        Mke.lines!(
            ax,[lim_lo,lim_hi],[lim_lo,lim_hi];
            color=:black,linestyle=:dash,linewidth=2,label="Unity (y = x)",
        )

        for (j,E) in pairs(nchannel_vmi_energies)
            x = Float64.(vec(d.theoretical[:,j]))
            y = Float64.(vec(d.measured[:,j]))
            color = energy_colors[E]
            fit = fit_lr(x,y)
            xrange = [minimum(x),maximum(x)]
            sign_str = fit.intercept ≥ 0 ? "+" : "−"
            label = "$(Int(E)) keV: y = $(round(fit.slope,digits=3))·x " *
                "$sign_str $(round(abs(fit.intercept),digits=1)) HU   " *
                "R² = $(round(fit.r²,digits=4))   " *
                "RMSE = $(round(fit.rmse,digits=1)) HU"
            Mke.scatter!(ax,x,y; color,markersize=11)
            Mke.lines!(
                ax,xrange,fit.intercept .+ fit.slope.*xrange;
                color,linewidth=2,label,
            )
        end
        Mke.xlims!(ax,lim_lo,lim_hi)
        Mke.ylims!(ax,lim_lo,lim_hi)
        Mke.axislegend(
            ax; position=:rb,framevisible=true,labelsize=16,
            padding=(6,6,6,6),rowgap=1,
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

# ╔═╡ c0231062-8372-44c3-a3f0-574d39a1c326
let
    HU_window = (-200.0, 500.0)
    sample = nchannel_vmi_HU[first(nchannel_vmi_energies)]
    mid_z = size(sample, 3) ÷ 2 + 1
    fig = Mke.Figure(size = (1180, 1180))

    for (k, E) in enumerate(nchannel_vmi_energies)
        r = (k - 1) ÷ 2 + 1
        c = (k - 1) % 2 + 1
        σ = std(nchannel_vmi_HU[E][nchannel_water_roi.mask,:])
        ax = Mke.Axis(
            fig[r,c]; title="$(Int(E)) keV VMI · $(round(σ,digits=1)) HU",
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
# ╠═171294a2-26bd-49e2-ac92-9df48ae5444f
# ╠═69358294-97f2-4782-94d7-c29c747c45f4
# ╠═9ae27110-5c47-442b-a98e-d137599570f2
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
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
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╟─a27649c0-e623-473e-bfbe-60fc8dd6d976
# ╠═4985581f-616d-4bb7-ab9b-967d7250b28b
# ╠═73371177-0498-4eda-897b-651c94f43e83
# ╟─b3e1d768-eb02-4c2a-9363-c84077fedc32
# ╠═50ed35bb-df61-4862-a99b-ea37380f30d8
# ╠═b86a9c50-cb10-44b2-af2e-06bdd50943b7
# ╠═ddfde8bb-ddff-44bb-8b70-b397725402cf
# ╟─aa1c494e-ccbf-4a0c-8d44-5f12d47f6e8c
# ╠═af6ea59b-06a0-4527-9b6d-344e54dc3bee
# ╟─2d447ccd-b408-42a6-8a1b-af770e3277e4
# ╠═1ff5e801-ef54-45ff-b0a8-e780e2e6cb63
# ╟─65a40efc-c9ff-41d4-8f42-46d6737119d5
# ╟─7707308d-e855-4111-9367-ef4ffc66da8b
# ╠═9161e7bf-eaa9-4d48-9d93-ab743ff3a0c2
# ╠═5cc1f9ba-4a5f-4a78-86dd-e0d844a09a28
# ╟─1c55fa7b-e29a-4464-81f5-a75a6fce7228
# ╠═a2ce06aa-31e8-4533-8617-40cb0f02b292
# ╟─c0231062-8372-44c3-a3f0-574d39a1c326
# ╟─3e8869e6-4f74-4ac8-a048-10d55aa76e20
# ╠═594208fc-89bc-4c7b-b989-623a0c432207
# ╠═f2bd7681-b328-48c4-b578-71d01f1bd5eb
# ╟─48ef9fbf-aa1d-4535-81db-70d9551667a6
# ╟─a6d34bb4-9af4-46ab-8b20-20cd4dbfe33f
# ╟─dbb6d6e4-e9a0-4b99-933d-d6937a486622
# ╟─df5f06e2-6075-40b0-9eec-660c75002e9a
# ╠═7e8bbfaf-35f1-4fc9-8ddd-58c4ca50ca6d
# ╟─e2ccf044-c0b8-414a-ad09-f5e40a0b71b5
# ╟─3f35047e-00ba-44a1-a8a6-d8325305736d
# ╟─1f0cfed9-bd4c-43bc-a042-0de34a55598e
