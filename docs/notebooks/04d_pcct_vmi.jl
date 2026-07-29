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
using Statistics: mean, std, var, quantile, median, cov, cor

# ╔═╡ 27f065e9-f97c-4392-a83f-e3638682152d
using LinearAlgebra: Symmetric, Diagonal, eigen, eigvals, det, dot, norm, svd,
    svdvals, I

# ╔═╡ a8fc42d0-ee5b-44ea-b4dc-8de368209e44
using Random: MersenneTwister, randn

# ╔═╡ d3054785-9e00-4094-a491-088ce63be9dc
md"""
# Cong Followed by Total-Likelihood Bilateral Filtering

Siemens Naeotom Alpha photon-counting CT simulation (140 kVp / 174 mA,
four native energy windows, Gammex 472 phantom).

This feasibility notebook preserves the verified four-bin Cong decomposition,
then tests Lee's 2025 total-likelihood bilateral filter (T-LBF) on the completed
water and iodine sinograms.
"""

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## Focused feasibility objective

The four native bins, detector corrections, row-count combination, and
all-bin profiled Cong solve remain unchanged. Filtering begins only after Cong.
The collapsed total count is used only to calculate a shared neighborhood
weight; it is never used to estimate the two materials.

Executable TNV, CGLS, LSMR, RSKR, and earlier denoising experiments were
removed from this active notebook and remain recoverable from Git history at
checkpoint `37b7241`.
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

# ╔═╡ 5b50d9e8-3d67-4f0c-a9e8-92b7021c6cc5
pcct_memory_checkpoint = let
    before = BS.backend_memory_snapshot(phantom.mask)
    GC.gc(true)
    yield()
    GC.gc(true)
    after = BS.backend_memory_snapshot(phantom.mask)
    (
        before,
        after,
        reclaimed_device_bytes =
            before.device_allocated_bytes - after.device_allocated_bytes,
        device_pressure_after =
            after.device_allocated_bytes / after.device_working_set_bytes,
    )
end

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
end

# ╔═╡ f9c0af7a-addd-4249-96fb-b9078765fbd1
protocol = BS.CTProtocol(
    kVp = 140,
    mA = 174.0,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

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
    try
        result = BS.simulate!(ws, phantom, protocol, sim_opts)
        bins = [Array(b) for b in result.pcct_sino.bins]
        I0_bins = copy(result.I0_bins)
        geom = ws.geom
        # The EXACT per-bin detected spectra the forward applied (w·η·DRM with
        # the workspace's MC-LUT η and the centre-pixel bowtie fold).
        energies = Float64.(ws.energies)
        W_applied = Float64.(Array(ws.W_matrix_gpu))[1:length(ws.energies), :]
        mc_count_moments = BS.compute_mc_count_moments(
            ws.pcct_detector, ws.energies, ws.weights;
            η=ws.η, R=ws.R,
        )
        pileup_S = result.pileup_S
        returned_eltype = eltype(first(bins))
        memory_before_release = BS.backend_memory_snapshot(first(ws.bins))
        (bins = bins, I0_bins = I0_bins, geom = geom,
         energies = energies, W_applied = W_applied,
         mc_count_moments = mc_count_moments,
         pileup_S = pileup_S, returned_eltype = returned_eltype,
         memory = (before_release = memory_before_release,))
    finally
        BS.release_backend!(ws)
    end
end;

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## 1. Simulator output contract

This contract was traced through `simulate!(::PCCTWorkspace)` and the called
PCCT detector routines rather than inferred from variable names.

- Each returned channel is a floating-point **corrected negative-log
  transmission**, not an integer raw-count array:
  \(h_k=-\log(\widetilde y_k/I_{0,k})\). Consequently
  \(\widetilde y_k=I_{0,k}e^{-h_k}\) is called a **corrected count-domain
  equivalent**, never a raw Poisson count.
- `I0_bins[k]` is the expected air count per detector element and view in
  native differential window \(k\), in photons/counts. The same absolute
  response is stored in `ws.W_matrix_gpu`; its energy sum must reproduce
  `I0_bins`.
- Before detector corrections, the four thresholds define mutually exclusive
  differential windows. Charge sharing is contained in the Monte-Carlo
  detector response and count covariance; it is not a later image-domain
  operation.
- The implemented order is: primary polychromatic projection through the
  applied MC response → optional focal-spot blur → count-domain scatter
  injection → MC-covariance quantum noise → pileup forward migration →
  inverse pileup correction → scatter correction → negative-log conversion.
- The bundled Monte-Carlo LUT supplies both the mean detector response and
  second moments. `compute_mc_count_moments` exposes per-incident-photon
  `mean`, `fano`, `correlation`, and compound-Poisson `covariance`.

Pileup inversion, scatter correction, and the final logarithm can introduce
fractional values and alter covariance. Therefore the independent-Poisson
objective below is explicitly a **Poisson quasi-likelihood** unless repeated
realizations validate that covariance model. The calibrated MC moments are
reported as a separate detector-statistics model, not silently substituted
for post-correction covariance.
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

    # K-channel effective-energy linearization, used only as an initializer.
    Φsum = vec(sum(Φ; dims = 1))
    μI_eff = Float32[
        sum(view(Φ, :, k) .* μρ_I) / Φsum[k] for k in axes(Φ, 2)
    ]
    μW_eff = Float32[
        sum(view(Φ, :, k) .* μρ_W) / Φsum[k] for k in axes(Φ, 2)
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

# ╔═╡ dffbbf26-cb0d-45c2-b4ba-621e52b40108
nchannel_simulator_contract = let
    moments=sim_bins.mc_count_moments
    (
        returned_quantity=:corrected_negative_log_transmission,
        corrected_count_equivalent=true,
        raw_integer_counts=false,
        native_bins=length(sim_bins.bins),
        returned_eltype=sim_bins.returned_eltype,
        I0_units="expected photons / detector element / view",
        I0_bins=Float64.(sim_bins.I0_bins),
        response_I0_max_relative_error=nchannel_basis.I0_relerr,
        mutually_exclusive_before_correction=true,
        independent_poisson_after_correction=false,
        mc_mean=moments.mean,
        mc_fano=moments.fano,
        mc_correlation=moments.correlation,
        mc_covariance_per_incident_photon=moments.covariance,
        pileup_correction_enabled=sim_opts.use_pcct_pileup_correction,
        scatter_correction_enabled=sim_opts.use_pcct_scatter_correction,
        statistical_mode=:poisson_quasi_likelihood,
    )
end;

# ╔═╡ 67547db5-68e4-4648-badc-610ec40fef9f
detector_moment_summary = let
    moments=sim_bins.mc_count_moments
    off_diagonal=[
        abs(moments.correlation[i,j])
        for j in axes(moments.correlation,2),i in axes(moments.correlation,1)
        if i!=j
    ]
    (
        fano=moments.fano,
        correlation=moments.correlation,
        maximum_absolute_off_diagonal_correlation=maximum(off_diagonal),
        scalar_fano_range=extrema(moments.fano),
    )
end

# ╔═╡ 11fd3c9c-392b-4ee4-adcb-e757021401f7
md"""
## 2. Effective spectral response

The decomposition uses the exact absolute response
`sim_bins.W_applied == Array(ws.W_matrix_gpu)`, not four independently
normalized spectra. Its maximum response-versus-`I0_bins` relative
discrepancy is
**$(nchannel_simulator_contract.response_I0_max_relative_error)**.

The detector MC Fano factors are
`$(round.(nchannel_simulator_contract.mc_fano; digits=4))`. The off-diagonal
entries in the reported MC correlation matrix make independence a testable
approximation rather than an assumption.
"""

# ╔═╡ 41c4d6c2-69cc-453e-9ad5-35026e027a93
md"""
## 3. Generic K-channel forward model

`nchannel_forward` has no four-bin indexing. It returns the exact channel
means, first derivatives, and optional second derivatives for any
`K = size(Φ,2)`. The audit below checks its analytic derivatives against
central finite differences at randomized physical iodine/water paths.
"""

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
    outer_iterations = 16,              # numerical ceiling, not regularization
    inner_iterations = 12,              # numerical ceiling, not regularization
    max_iodine_step = 0.05f0,
    max_water_step = 5.0f0,
    parameter_tolerance = 5.0f-5,
    fisher_condition_limit = 1.0f8,
    # Disabled for the production result: the shortcut is discontinuous and
    # the bounded-reference audit showed measurable likelihood gaps in air.
    # A separate threshold sweep below quantifies it as an optional speed path.
    air_gate = 0.0f0,
    tile_views = 8,
);

# ╔═╡ 73371177-0498-4eda-897b-651c94f43e83
begin
    """
        nchannel_forward(A, C, Φ, μI, μW; second=false)

    Exact discrete polychromatic K-channel mean and analytic derivatives.
    `Φ[e,k]` is the absolute air-count contribution, `A` is iodine area
    density, and `C` is water area density, both in g/cm².
    """
    function nchannel_forward(A,C,Φ,μI,μW;second=false)
        nE,K = size(Φ)
        nE == length(μI) == length(μW) ||
            throw(DimensionMismatch("Energy dimensions disagree"))
        λ=zeros(Float64,K)
        dA=zeros(Float64,K)
        dC=zeros(Float64,K)
        dAA=second ? zeros(Float64,K) : nothing
        dAC=second ? zeros(Float64,K) : nothing
        dCC=second ? zeros(Float64,K) : nothing
        @inbounds for k in 1:K, e in 1:nE
            z=Float64(Φ[e,k])*exp(-Float64(μI[e])*A-Float64(μW[e])*C)
            mi,mw=Float64(μI[e]),Float64(μW[e])
            λ[k]+=z; dA[k]-=mi*z; dC[k]-=mw*z
            if second
                dAA[k]+=mi*mi*z; dAC[k]+=mi*mw*z; dCC[k]+=mw*mw*z
            end
        end
        second ? (;λ,dA,dC,dAA,dAC,dCC) : (;λ,dA,dC)
    end

    function nchannel_golden_minimize(f,lo,hi;iterations=80)
        lo == hi && return (x=lo,value=f(lo))
        ϕ=(sqrt(5.0)-1)/2
        x1=hi-ϕ*(hi-lo); x2=lo+ϕ*(hi-lo)
        f1,f2=f(x1),f(x2)
        for _ in 1:iterations
            if f1 ≤ f2
                hi,x2,f2=x2,x1,f1
                x1=hi-ϕ*(hi-lo); f1=f(x1)
            else
                lo,x1,f1=x1,x2,f2
                x2=lo+ϕ*(hi-lo); f2=f(x2)
            end
        end
        candidates=((x1,f1),(x2,f2),(lo,f(lo)),(hi,f(hi)))
        x,value=argmin(last,candidates)
        (;x,value)
    end

    function nchannel_scalar_global(f,bounds;grid_points=129,iterations=80)
        grid=collect(range(bounds...;length=grid_points))
        values=f.(grid)
        basins=Tuple{Float64,Float64,Int}[]
        for i in 2:(length(grid)-1)
            isfinite(values[i]) &&
                values[i] ≤ values[i-1] && values[i] ≤ values[i+1] &&
                push!(basins,(grid[i-1],grid[i+1],i))
        end
        candidates=NamedTuple[
            (x=grid[1],value=values[1],basin=0),
            (x=grid[end],value=values[end],basin=length(grid)),
        ]
        for (lo,hi,i) in basins
            r=nchannel_golden_minimize(f,lo,hi;iterations)
            push!(candidates,(x=r.x,value=r.value,basin=i))
        end
        if isempty(basins)
            i=argmin(values)
            push!(candidates,(x=grid[i],value=values[i],basin=i))
        end
        best=candidates[argmin(getproperty.(candidates,:value))]
        (;best,candidates,grid,values,basin_count=length(basins))
    end

    function nchannel_solve_total_C(
        A,y_total,Φ,μI,μW,water_bounds;
        bisection_iterations=80,
    )
        lo,hi=Float64.(water_bounds)
        residual(C)=sum(nchannel_forward(A,C,Φ,μI,μW).λ)-y_total
        rlo,rhi=residual(lo),residual(hi)
        if !(isfinite(rlo)&&isfinite(rhi)&&rlo≥0&&rhi≤0)
            return (success=false,C=NaN,residual=NaN,bracket=(rlo,rhi))
        end
        for _ in 1:bisection_iterations
            mid=(lo+hi)/2
            residual(mid)>0 ? (lo=mid) : (hi=mid)
        end
        C=(lo+hi)/2
        (success=true,C,residual=residual(C),bracket=(rlo,rhi))
    end

    """
    Exact monotone aggregate-channel Cong-like reference. The inner root is
    guaranteed when bracketed; the complete outer iodine interval is scanned,
    every detected basin is refined, and both endpoints are evaluated.
    """
    function nchannel_cong_constrained_reference(
        y,Φ,μI,μW;
        iodine_bounds=(-0.10,0.40),water_bounds=(-2.0,50.0),
        grid_points=129,bisection_iterations=80,golden_iterations=80,
    )
        K=length(y)
        size(Φ,2)==K || throw(DimensionMismatch("Channel count mismatch"))
        yv=Float64.(y); y_total=sum(yv)
        roots=Dict{Float64,NamedTuple}()
        root(A)=get!(roots,Float64(A)) do
            nchannel_solve_total_C(
                A,y_total,Φ,μI,μW,water_bounds;
                bisection_iterations,
            )
        end
        function objective(A)
            r=root(A)
            r.success || return Inf
            λ=max.(nchannel_forward(A,r.C,Φ,μI,μW).λ,eps(Float64))
            π=λ/sum(λ)
            -sum(yv.*log.(π))
        end
        search=nchannel_scalar_global(
            objective,iodine_bounds;
            grid_points,iterations=golden_iterations,
        )
        A=search.best.x; r=root(A)
        (
            iodine=A,water=r.C,objective=search.best.value,
            root_bracketed=r.success,total_residual=r.residual,
            selected_basin=search.best.basin,
            basin_count=search.basin_count,
            boundary_contact=(
                A≈first(iodine_bounds) || A≈last(iodine_bounds) ||
                r.C≈first(water_bounds) || r.C≈last(water_bounds)
            ),
        )
    end

    nchannel_poisson_quasi_nll(A,C,y,Φ,μI,μW)=let
        λ=max.(nchannel_forward(A,C,Φ,μI,μW).λ,eps(Float64))
        sum(λ-Float64.(y).*log.(λ))
    end

    """
    Slow bounded all-channel profile quasi-likelihood reference. No
    allocation assumptions or local optimizer starting point enter the
    correctness claim: every sampled inner and outer basin plus endpoints is
    evaluated.
    """
    function nchannel_profile_reference(
        y,Φ,μI,μW;
        iodine_bounds=(-0.10,0.40),water_bounds=(-2.0,50.0),
        outer_grid_points=65,inner_grid_points=65,iterations=80,
    )
        inner_cache=Dict{Float64,NamedTuple}()
        function inner(A)
            get!(inner_cache,Float64(A)) do
                f(C)=nchannel_poisson_quasi_nll(A,C,y,Φ,μI,μW)
                s=nchannel_scalar_global(
                    f,water_bounds;grid_points=inner_grid_points,iterations,
                )
                (C=s.best.x,value=s.best.value,basin=s.best.basin,
                 basin_count=s.basin_count)
            end
        end
        outer(A)=inner(A).value
        s=nchannel_scalar_global(
            outer,iodine_bounds;grid_points=outer_grid_points,iterations,
        )
        A=s.best.x; inn=inner(A)
        f=nchannel_forward(A,inn.C,Φ,μI,μW)
        λ=max.(f.λ,eps(Float64)); yv=Float64.(y)
        gA=sum((1 .- yv./λ).*f.dA)
        gC=sum((1 .- yv./λ).*f.dC)
        FAA=sum(f.dA.^2 ./ λ)
        FCC=sum(f.dC.^2 ./ λ)
        (
            iodine=A,water=inn.C,objective=inn.value,
            score_norm=hypot(gA,gC)/sqrt(max(FAA+FCC,eps(Float64))),
            selected_outer_basin=s.best.basin,
            outer_basin_count=s.basin_count,
            selected_inner_basin=inn.basin,
            inner_basin_count=inn.basin_count,
            boundary_contact=(
                A≈first(iodine_bounds)||A≈last(iodine_bounds)||
                inn.C≈first(water_bounds)||inn.C≈last(water_bounds)
            ),
        )
    end

    function nchannel_profile_tile!(
    sino_I, sino_W, fisher_AA, fisher_AC, fisher_CC,
    quality_flag, score_norm, outer_count, inner_count,
    hs::NTuple{K},
    Φ, μρ_I, μρ_W, I0, μI_eff, μW_eff,
    normal_II::Float32, normal_IW::Float32, normal_WW::Float32, controls,
) where {K}
    nE = length(μρ_I)
    A_lo, A_hi = controls.iodine_bounds
    C_lo, C_hi = controls.water_bounds
    n_outer, n_inner = controls.outer_iterations, controls.inner_iterations
    A_step, C_step = controls.max_iodine_step, controls.max_water_step
    parameter_tolerance = controls.parameter_tolerance
    fisher_condition_limit = controls.fisher_condition_limit
    air_gate = controls.air_gate

    BS.AK.foreachindex(sino_I) do idx
        max_abs_h = 0f0
        for k in 1:K
            max_abs_h = max(max_abs_h,abs(hs[k][idx]))
        end
        if max_abs_h < air_gate
            sino_I[idx] = 0f0
            sino_W[idx] = 0f0
            fisher_AA[idx] = 0f0
            fisher_AC[idx] = 0f0
            fisher_CC[idx] = 0f0
            quality_flag[idx] = UInt8(0)
            score_norm[idx] = 0f0
            outer_count[idx] = UInt8(0)
            inner_count[idx] = UInt8(0)
            return
        end

        # K-channel linear initializer; all iterations below are polychromatic.
        rhs_I, rhs_W = 0f0, 0f0
        for k in 1:K
            rhs_I += μI_eff[k]*hs[k][idx]
            rhs_W += μW_eff[k]*hs[k][idx]
        end
        det0_raw = normal_II*normal_WW - normal_IW*normal_IW
        initializer_valid = isfinite(det0_raw) && det0_raw > 1f-12
        det0 = initializer_valid ? det0_raw : 1f0
        A = initializer_valid ?
            clamp((normal_WW*rhs_I-normal_IW*rhs_W)/det0,A_lo,A_hi) :
            clamp(0f0,A_lo,A_hi)
        C = initializer_valid ?
            clamp((normal_II*rhs_W-normal_IW*rhs_I)/det0,C_lo,C_hi) :
            clamp(20f0,C_lo,C_hi)

        # Guaranteed monotone aggregate equation, used here only to stabilize
        # the fast solver's initial water value at its current iodine value.
        y_total=0f0
        for k in 1:K
            y_total += max(I0[k]*exp(-hs[k][idx]),1f-6)
        end
        croot_lo,croot_hi=C_lo,C_hi
        total_lo,total_hi=0f0,0f0
        for k in 1:K, e in 1:nE
            total_lo += Φ[e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_lo)
            total_hi += Φ[e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_hi)
        end
        aggregate_bracketed=total_lo≥y_total && total_hi≤y_total
        attainable_max,attainable_min=0f0,0f0
        for k in 1:K, e in 1:nE
            attainable_max += Φ[e,k]*exp(
                -μρ_I[e]*A_lo-μρ_W[e]*C_lo,
            )
            attainable_min += Φ[e,k]*exp(
                -μρ_I[e]*A_hi-μρ_W[e]*C_hi,
            )
        end
        aggregate_feasible =
            attainable_max≥y_total && attainable_min≤y_total
        if aggregate_bracketed
            for _ in 1:28
                mid=(croot_lo+croot_hi)/2f0
                total_mid=0f0
                for k in 1:K, e in 1:nE
                    total_mid += Φ[e,k]*exp(-μρ_I[e]*A-μρ_W[e]*mid)
                end
                if total_mid>y_total
                    croot_lo=mid
                else
                    croot_hi=mid
                end
            end
            C=(croot_lo+croot_hi)/2f0
        end

        converged = false
        used_outer=0
        used_inner=0
        for outer_iter in 1:n_outer
            used_outer=outer_iter
            # Inner scalar solve: C*(A) = argmin_C L(A,C).
            for _ in 1:n_inner
                used_inner+=1
                gC, FCC = 0f0, 0f0
                for k in 1:K
                    λ, dC = 0f0, 0f0
                    @inbounds for e in 1:nE
                        z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                        λ += z
                        dC -= μρ_W[e] * z
                    end
                    λ = max(λ, 1f-6)
                    # Corrected counts may be fractional after detector correction.
                    y = max(I0[k]*exp(-hs[k][idx]),1f-6)
                    gC += (1f0 - y/λ) * dC
                    FCC += dC*dC / λ
                end
                raw_C_step = gC/max(FCC,1f-12)
                C_new = clamp(
                    C-clamp(raw_C_step,-C_step,C_step),C_lo,C_hi,
                )
                C_done = abs(C_new-C) <= parameter_tolerance*(1f0+abs(C))
                C = C_new
                C_done && break
            end

            # Envelope gradient and Fisher Schur-complement profile curvature.
            gA, FAA, FAC, FCC = 0f0, 0f0, 0f0, 0f0
            for k in 1:K
                λ, dA, dC = 0f0, 0f0, 0f0
                @inbounds for e in 1:nE
                    z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dA -= μρ_I[e] * z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[k]*exp(-hs[k][idx]),1f-6)
                gA += (1f0 - y/λ) * dA
                FAA += dA*dA / λ
                FAC += dA*dC / λ
                FCC += dC*dC / λ
            end
            Hprof = max(FAA - FAC*FAC/max(FCC, 1f-12), 1f-12)
            raw_A_step = gA/Hprof
            A_new = clamp(
                A-clamp(raw_A_step,-A_step,A_step),A_lo,A_hi,
            )
            converged = abs(A_new-A) <= parameter_tolerance*(1f0+abs(A))
            A = A_new
            converged && break
        end

        # Re-profile water at the final iodine iterate.
        c_converged = false
        for _ in 1:n_inner
            used_inner+=1
            gC, FCC = 0f0, 0f0
            for k in 1:K
                λ, dC = 0f0, 0f0
                @inbounds for e in 1:nE
                    z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[k]*exp(-hs[k][idx]),1f-6)
                gC += (1f0 - y/λ) * dC
                FCC += dC*dC / λ
            end
            C_new = clamp(
                C-clamp(gC/max(FCC,1f-12),-C_step,C_step),C_lo,C_hi,
            )
            C_done = abs(C_new-C) <= parameter_tolerance*(1f0+abs(C))
            C = C_new
            if C_done
                c_converged = true
                break
            end
        end
        converged &= c_converged

        # Final score and Fisher conditioning are recorded; they are not silently
        # converted into image regularization.
        gA, gC, FAA, FAC, FCC = 0f0, 0f0, 0f0, 0f0, 0f0
        for k in 1:K
            λ, dA, dC = 0f0, 0f0, 0f0
            @inbounds for e in 1:nE
                z = Φ[e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                λ += z
                dA -= μρ_I[e]*z
                dC -= μρ_W[e]*z
            end
            λ = max(λ,1f-6)
            y = max(I0[k]*exp(-hs[k][idx]),1f-6)
            gA += (1f0-y/λ)*dA
            gC += (1f0-y/λ)*dC
            FAA += dA*dA/λ
            FAC += dA*dC/λ
            FCC += dC*dC/λ
        end
        score_norm[idx] = sqrt(gA*gA+gC*gC) /
            sqrt(max(FAA+FCC,1f-12))
        fisher_det = max(FAA*FCC-FAC*FAC,0f0)
        fisher_trace = FAA+FCC
        fisher_disc = sqrt(max(fisher_trace*fisher_trace-4f0*fisher_det,0f0))
        eig_max_raw = max((fisher_trace+fisher_disc)/2f0,1f-12)
        eig_min = max(fisher_det/eig_max_raw,1f-12)
        eig_max = max(eig_max_raw,eig_min)
        ill_conditioned = eig_max/eig_min > fisher_condition_limit

        tol = 2f-4
        hit_A = A <= A_lo + tol || A >= A_hi - tol
        hit_C = C <= C_lo + tol || C >= C_hi - tol
        invalid_model = !(
            isfinite(A)&&isfinite(C)&&isfinite(score_norm[idx])&&
            isfinite(FAA)&&isfinite(FAC)&&isfinite(FCC)
        )
        quality_flag[idx] =
            UInt8(hit_A ? 1 : 0) |
            UInt8(hit_C ? 2 : 0) |
            UInt8(converged ? 0 : 4) |
            UInt8(ill_conditioned || !initializer_valid ? 8 : 0) |
            UInt8(aggregate_feasible ? 0 : 16) |
            UInt8(invalid_model ? 32 : 0)
        outer_count[idx]=UInt8(min(used_outer,255))
        inner_count[idx]=UInt8(min(used_inner,255))
        fisher_AA[idx],fisher_AC[idx],fisher_CC[idx] = FAA,FAC,FCC
        sino_I[idx], sino_W[idx] = A, C
    end
    nothing
    end
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
negligible, the rows are repeated measurements of the same in-plane ray to
that stated tolerance. The simulator supplies corrected log transmissions, so
we sum their **corrected count-domain equivalents by native energy bin before
reapplying the logarithm**:
\[
Y_{k,\Sigma}=\sum_r I_{0,k}e^{-h_{k,r}},\qquad
h_{k,\Sigma}=-\log\frac{Y_{k,\Sigma}}{R I_{0,k}}.
\]
For ideal independent Poisson counts the \(K\)-channel likelihood is unchanged
except that every \(\Phi_k\) and \(I_{0,k}\)
are multiplied by \(R\). This improves the photon support seen by the
nonlinear estimator while retaining all native spectral bins and the profiled
univariate solve. It represents a declared 5-mm slice, not native 0.4-mm
longitudinal resolution. With fractional detector-corrected outputs, the same
operation preserves the corrected mean but the Poisson interpretation is a
quasi-likelihood unless post-correction covariance is validated.

The 1200-view acquisition also oversamples the angular sampling requirement
of the 512-pixel reconstruction,
\(N_{\theta,\mathrm{required}}=\lceil\pi N/2\rceil=805\). Before the common
FBP, both material sinograms therefore receive the same deterministic angular
anti-alias projection: Fourier modes through
\(\lceil\pi N/4\rceil\) pass with gain exactly one, followed by a raised-cosine
roll-off confined to the angular oversampling margin. This is not a denoising
parameter or an energy-dependent kernel; it discards angular noise that the
target image grid cannot represent without aliasing.
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
    bins = map(eachindex(sim_bins.bins)) do k
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
    score_norm = Array{Float32}(undef,shape)
    fisher_AA = Array{Float32}(undef,shape)
    fisher_AC = Array{Float32}(undef,shape)
    fisher_CC = Array{Float32}(undef,shape)
    outer_iterations = Array{UInt8}(undef,shape)
    inner_iterations = Array{UInt8}(undef,shape)
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
            for k in eachindex(nchannel_slab_counts.bins)
        ]
        I_gpu,W_gpu = similar(hs[1]),similar(hs[1])
        flag_gpu = similar(hs[1],UInt8)
        score_gpu = similar(hs[1],Float32)
        fisher_AA_gpu = similar(hs[1],Float32)
        fisher_AC_gpu = similar(hs[1],Float32)
        fisher_CC_gpu = similar(hs[1],Float32)
        outer_gpu = similar(hs[1],UInt8)
        inner_gpu = similar(hs[1],UInt8)
        nchannel_profile_tile!(
            I_gpu,W_gpu,fisher_AA_gpu,fisher_AC_gpu,fisher_CC_gpu,
            flag_gpu,score_gpu,outer_gpu,inner_gpu,Tuple(hs),
            Φ_gpu,μρ_I_gpu,μρ_W_gpu,I0_gpu,μI_eff_gpu,μW_eff_gpu,
            nchannel_basis.normal_II,nchannel_basis.normal_IW,
            nchannel_basis.normal_WW,nchannel_controls,
        )
        sino_I[:,:,vrange] .= Array(I_gpu)
        sino_W[:,:,vrange] .= Array(W_gpu)
        flags[:,:,vrange] .= Array(flag_gpu)
        score_norm[:,:,vrange] .= Array(score_gpu)
        fisher_AA[:,:,vrange] .= Array(fisher_AA_gpu)
        fisher_AC[:,:,vrange] .= Array(fisher_AC_gpu)
        fisher_CC[:,:,vrange] .= Array(fisher_CC_gpu)
        outer_iterations[:,:,vrange] .= Array(outer_gpu)
        inner_iterations[:,:,vrange] .= Array(inner_gpu)
    end
    # Recompute the aggregate feasibility bit on the host from the global
    # attainable count range. This is independent of the initializer's
    # single-A bracket and avoids backend-specific boolean lowering in QC.
    Φ_host=scale.*Float64.(nchannel_basis.Φ)
    attainable_max=sum(nchannel_forward(
        first(nchannel_controls.iodine_bounds),
        first(nchannel_controls.water_bounds),
        Φ_host,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    ).λ)
    attainable_min=sum(nchannel_forward(
        last(nchannel_controls.iodine_bounds),
        last(nchannel_controls.water_bounds),
        Φ_host,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    ).λ)
    ytotal=zeros(Float64,shape)
    for k in eachindex(nchannel_slab_counts.bins)
        ytotal .+= scale*nchannel_basis.I0[k].*
            exp.(-Float64.(nchannel_slab_counts.bins[k]))
    end
    feasible=(ytotal.≤attainable_max).&(ytotal.≥attainable_min)
    flags[feasible] .&= 0xef
    flags[.!feasible] .|= 0x10
    (
        sino_iodine=sino_I,sino_water=sino_W,quality_flag=flags,
        fisher=(AA=fisher_AA,AC=fisher_AC,CC=fisher_CC),
        score_norm,outer_iterations,inner_iterations,
        geom=sim_bins.geom,elapsed_s=elapsed,
    )
end;

# ╔═╡ 30f13da5-f7e2-4f5f-9494-d264a4e4a355
cong_ray_covariance = let
    F = sino_basis_nchannel_slab.fisher
    detF = @. F.AA*F.CC - F.AC*F.AC
    valid = @. isfinite(detF) && detF > 0 &&
        (sino_basis_nchannel_slab.quality_flag & 0x2c) == 0

    C_AA = fill(Float32(Inf),size(detF))
    C_AC = zeros(Float32,size(detF))
    C_CC = fill(Float32(Inf),size(detF))
    C_AA[valid] .= F.CC[valid] ./ detF[valid]
    C_AC[valid] .= -F.AC[valid] ./ detF[valid]
    C_CC[valid] .= F.AA[valid] ./ detF[valid]

    (
        precision=F,
        covariance=(AA=C_AA,AC=C_AC,CC=C_CC),
        valid,
        valid_fraction=count(valid)/length(valid),
        model=:local_poisson_quasilikelihood,
        boundary_policy=:excluded_pending_bootstrap_calibration,
    )
end;

# ╔═╡ f9ca428e-9acf-494e-8f77-34502d68e0ad
md"""
## 7. Full Cong covariance

The final profiled Cong iterate now retains the complete per-ray Fisher matrix
\((F_{AA},F_{AC},F_{CC})\), including its off-diagonal term. Its inverse is
the working two-material covariance for the first TNV implementation.

This is explicitly a local quasi-likelihood covariance because corrected
counts need not be independent Poisson variables. Bound-hitting,
nonconverged, or ill-conditioned rays are excluded until a small bootstrap
calibration replaces this approximation. No repeated-scan sweep is part of
the primary execution path.
"""

# ╔═╡ 92692d46-1b57-4daa-b78d-c5aef609ed86
nchannel_decomposition_checkpoint = (
    simulator_contract=nchannel_simulator_contract,
    channel_count=length(nchannel_slab_counts.bins),
    sinogram_size=size(sino_basis_nchannel_slab.sino_water),
    finite_materials=all(isfinite,sino_basis_nchannel_slab.sino_water) &&
        all(isfinite,sino_basis_nchannel_slab.sino_iodine),
    covariance_valid_fraction=cong_ray_covariance.valid_fraction,
    elapsed_s=sino_basis_nchannel_slab.elapsed_s,
)

# ╔═╡ ddfde8bb-ddff-44bb-8b70-b397725402cf
begin
    nchannel_fbp_matrix_size=(512,512,1)
    nchannel_fbp_nview=size(sino_basis_nchannel_slab.sino_water,3)
    nchannel_fbp_pass_mode=min(
        nchannel_fbp_nview÷2,
        ceil(Int,π*nchannel_fbp_matrix_size[1]/4),
    )
    nchannel_fbp_angular_response=[
        let mode=min(j-1,nchannel_fbp_nview-(j-1))
            mode≤nchannel_fbp_pass_mode ? 1.0 :
            0.5*(1+cos(
                π*(mode-nchannel_fbp_pass_mode)/
                (nchannel_fbp_nview÷2-nchannel_fbp_pass_mode),
            ))
        end
        for j in 1:nchannel_fbp_nview
    ]

    function nchannel_common_fbp_slice(sino_one_row)
        spectrum=BS.FFTW.fft(Float64.(sino_one_row),3)
        antialiased=Float32.(real.(BS.FFTW.ifft(
            spectrum.*reshape(
                nchannel_fbp_angular_response,1,1,nchannel_fbp_nview,
            ),3,
        )))
        repeated=repeat(antialiased,1,sim_bins.geom.n_rows,1)
        sino_gpu=to_gpu(repeated)
        ws=BS.create_fdk_recon_workspace(
            sino_gpu,sim_bins.geom,nchannel_fbp_matrix_size;
            filter=BS.SoftFilter(),
        )
        try
            Float32.(Array(BS.reconstruct!(
                ws,sino_gpu,sim_bins.geom,
            )))
        finally
            BS.release_backend!(ws)
        end
    end

    nchannel_slab_common_fbp=(
        vol_water=nchannel_common_fbp_slice(
            sino_basis_nchannel_slab.sino_water,
        ),
        vol_iodine=nchannel_common_fbp_slice(
            sino_basis_nchannel_slab.sino_iodine,
        ),
        angular_response=nchannel_fbp_angular_response,
        pass_mode=nchannel_fbp_pass_mode,
        kernel=:SoftFilter,
    )
end

# ╔═╡ 2d447ccd-b408-42a6-8a1b-af770e3277e4
md"""
## Generalized \(K\)-Channel Profile-Likelihood Output

All native photon-counting bins enter the profiled likelihood. The
14 geometrically equivalent detector rows are summed in the corrected
count domain before
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
    mid_z = size(nchannel_basis_volumes.vol_water,3) ÷ 2 + 1
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
## 11. Raw VMI synthesis

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
begin
    nchannel_validation_energies=collect(40.0:5.0:190.0)
    nchannel_vmi_energies=[40.0,70.0,100.0,140.0]
end;

# ╔═╡ 5cc1f9ba-4a5f-4a78-86dd-e0d844a09a28
nchannel_vmi_raw_HU = let
    iodine_mg_per_mL = nchannel_basis_volumes.vol_iodine .* 1000.0f0
    Dict(
        E => BS.synth_vmi_2basis(
            nchannel_basis_volumes.vol_water, iodine_mg_per_mL;
            energy_keV = E,
        )
        for E in nchannel_validation_energies
    )
end;

# ╔═╡ 707aec65-d9d6-4462-a662-c669beb8525b
begin
function pcct_sim_options(;use_noise,seed)
    BS.SimOptions(
        fidelity=:pcct,seed=seed,projector=:dd_fast,
        use_fill_factor=false,use_detector_efficiency=false,
        use_optical_crosstalk=false,use_focal_spot=false,
        use_lag=false,use_heel_effect=false,use_scatter=false,
        use_noise=use_noise,use_pcct_scatter=true,
        use_pcct_scatter_correction=true,use_pcct_pileup=true,
        use_pcct_pileup_correction=true,pcct_noise_reduction=0.0,
    )
end

function simulate_corrected_pcct_bins(;use_noise,seed)
    options=pcct_sim_options(;use_noise,seed)
    workspace=BS.create_workspace(
        scanner,protocol,options,recon_opts,phantom,
    )
    payload=nothing
    elapsed=@elapsed try
        result=BS.simulate!(workspace,phantom,protocol,options)
        payload=(
            bins=[Array(bin) for bin in result.pcct_sino.bins],
            I0_bins=copy(result.I0_bins),geom=workspace.geom,
        )
    finally
        BS.release_backend!(workspace)
    end
    merge(payload,(elapsed_s=elapsed,))
end

function cong_from_corrected_bins(native_bins)
    selected=nchannel_slab_counts.selected_rows
    nrows=nchannel_slab_counts.nrows
    slab=map(eachindex(native_bins)) do k
        summed=dropdims(sum(
            exp.(-Float64.(native_bins[k][:,selected,:]));dims=2,
        );dims=2)
        h=@. Float32(-log(max(summed,1e-12)/nrows))
        reshape(h,size(h,1),1,size(h,2))
    end
    shape=size(slab[1])
    sino_I=Array{Float32}(undef,shape);sino_W=similar(sino_I)
    flags=Array{UInt8}(undef,shape);score=similar(sino_I)
    FAA=similar(sino_I);FAC=similar(sino_I);FCC=similar(sino_I)
    outer=Array{UInt8}(undef,shape);inner=similar(outer)
    scale=Float32(nrows)
    persistent=(
        Φ=to_gpu(scale.*nchannel_basis.Φ),
        μI=to_gpu(nchannel_basis.μρ_I),
        μW=to_gpu(nchannel_basis.μρ_W),
        I0=to_gpu(scale.*nchannel_basis.I0),
        μI_eff=to_gpu(nchannel_basis.μI_eff),
        μW_eff=to_gpu(nchannel_basis.μW_eff),
    )
    elapsed=@elapsed try
        for vrange in BS.tile_ranges(shape[3],nchannel_controls.tile_views)
            hs=Tuple(
                to_gpu(Float32.(slab[k][:,:,vrange]))
                for k in eachindex(slab)
            )
            I_gpu,W_gpu=similar(hs[1]),similar(hs[1])
            flag_gpu=similar(hs[1],UInt8);score_gpu=similar(hs[1])
            FAA_gpu,FAC_gpu,FCC_gpu=
                similar(hs[1]),similar(hs[1]),similar(hs[1])
            outer_gpu,inner_gpu=
                similar(hs[1],UInt8),similar(hs[1],UInt8)
            tile_objects=(
                hs,I_gpu,W_gpu,flag_gpu,score_gpu,
                FAA_gpu,FAC_gpu,FCC_gpu,outer_gpu,inner_gpu,
            )
            try
                nchannel_profile_tile!(
                    I_gpu,W_gpu,FAA_gpu,FAC_gpu,FCC_gpu,
                    flag_gpu,score_gpu,outer_gpu,inner_gpu,hs,
                    persistent.Φ,persistent.μI,persistent.μW,persistent.I0,
                    persistent.μI_eff,persistent.μW_eff,
                    nchannel_basis.normal_II,nchannel_basis.normal_IW,
                    nchannel_basis.normal_WW,nchannel_controls,
                )
                sino_I[:,:,vrange].=Array(I_gpu)
                sino_W[:,:,vrange].=Array(W_gpu)
                flags[:,:,vrange].=Array(flag_gpu)
                score[:,:,vrange].=Array(score_gpu)
                FAA[:,:,vrange].=Array(FAA_gpu)
                FAC[:,:,vrange].=Array(FAC_gpu)
                FCC[:,:,vrange].=Array(FCC_gpu)
                outer[:,:,vrange].=Array(outer_gpu)
                inner[:,:,vrange].=Array(inner_gpu)
            finally
                BS.release_backend!(tile_objects)
            end
        end
    finally
        BS.release_backend!(persistent)
    end
    Φ_host=scale.*Float64.(nchannel_basis.Φ)
    attainable_max=sum(nchannel_forward(
        first(nchannel_controls.iodine_bounds),
        first(nchannel_controls.water_bounds),
        Φ_host,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    ).λ)
    attainable_min=sum(nchannel_forward(
        last(nchannel_controls.iodine_bounds),
        last(nchannel_controls.water_bounds),
        Φ_host,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    ).λ)
    ytotal=zeros(Float64,shape)
    for k in eachindex(slab)
        ytotal .+= scale*nchannel_basis.I0[k].*exp.(-Float64.(slab[k]))
    end
    feasible=(ytotal.≤attainable_max).&(ytotal.≥attainable_min)
    flags[feasible].&=0xef
    flags[.!feasible].|=0x10
    (
        sino_iodine=sino_I,sino_water=sino_W,
        slab_bins=slab,
        quality_flag=flags,score_norm=score,
        fisher=(AA=FAA,AC=FAC,CC=FCC),
        outer_iterations=outer,inner_iterations=inner,
        elapsed_s=elapsed,
    )
end
end

# ╔═╡ 067e491c-b305-46d0-abf9-bad0431f7b5e
matched_noiseless_pipeline = let
    GC.gc(true)
    simulated=simulate_corrected_pcct_bins(
        use_noise=false,seed=Int(0x04d0000),
    )
    decomposed=cong_from_corrected_bins(simulated.bins)
    (
        simulated,cong=decomposed,
        elapsed_s=simulated.elapsed_s+decomposed.elapsed_s,
    )
end

# ╔═╡ 8a23843f-3414-4641-a4c0-8800ec52cd7c
md"""
## Lee 2025 total-likelihood bilateral filter

For every center ray, the corrected four-bin count equivalents are summed only
to evaluate Lee's total likelihood. Each neighboring **completed Cong material
pair** is evaluated with the center measurement and the same MC-DRM-weighted
forward model. One normalized scalar weight is applied jointly to iodine and
water. The collapsed total count is never used for material estimation.

The implementation follows Lee 2025 with a \(5\times5\) sinogram window,
\(\alpha_1=0.9\), circular view wrapping, and nonwrapping detector boundaries.
The simulator returns corrected fractional count equivalents rather than raw
independent Poisson counts; that is the only likelihood-model deviation in
this feasibility test.
"""

# ╔═╡ f3de45a6-4818-4ee1-ad56-c65797119dee
begin
function total_measured_counts(slab_bins,I0,nrows)
    total=zeros(Float32,size(first(slab_bins)))
    for bin in eachindex(slab_bins)
        @. total+=Float32(nrows*I0[bin])*exp(-slab_bins[bin])
    end
    total
end

function total_expected_counts(
    sino_iodine,sino_water,Φ_total,μI,μW,
)
    I_gpu=to_gpu(Float32.(sino_iodine))
    W_gpu=to_gpu(Float32.(sino_water))
    Φ_gpu=to_gpu(Float32.(Φ_total))
    μI_gpu=to_gpu(Float32.(μI))
    μW_gpu=to_gpu(Float32.(μW))
    output_gpu=similar(I_gpu)
    try
        BS.AK.foreachindex(output_gpu) do idx
            total=0f0
            @inbounds for energy in eachindex(Φ_gpu)
                total+=Φ_gpu[energy]*exp(
                    -μI_gpu[energy]*I_gpu[idx]-
                    μW_gpu[energy]*W_gpu[idx],
                )
            end
            output_gpu[idx]=max(total,1f-6)
        end
        Array(output_gpu)
    finally
        BS.release_backend!((
            I_gpu,W_gpu,Φ_gpu,μI_gpu,μW_gpu,output_gpu,
        ))
    end
end

function tlbf_delta_sample(expected,measured;maximum_centers=4096)
    nchannel,_,nview=size(expected)
    stride=max(1,cld(length(expected),maximum_centers))
    values=Float32[]
    for linear in 1:stride:length(expected)
        channel=mod1(linear,nchannel)
        view=mod1(cld(linear,nchannel),nview)
        center_expected=max(expected[channel,1,view],1f-6)
        Y=measured[channel,1,view]
        center=-center_expected+Y*log(center_expected)
        for dv in -2:2,dc in -2:2
            dc==0&&dv==0&&continue
            neighbor_channel=channel+dc
            1≤neighbor_channel≤nchannel||continue
            neighbor_view=mod1(view+dv,nview)
            neighbor_expected=max(
                expected[neighbor_channel,1,neighbor_view],1f-6,
            )
            candidate=-neighbor_expected+Y*log(neighbor_expected)
            push!(values,abs(candidate-center))
        end
    end
    values
end

function tlbf_filter_pair(
    sino_iodine,sino_water,expected,measured,alpha2;
    alpha1=0.9,
)
    I_gpu=to_gpu(Float32.(sino_iodine))
    W_gpu=to_gpu(Float32.(sino_water))
    expected_gpu=to_gpu(Float32.(expected))
    measured_gpu=to_gpu(Float32.(measured))
    output_I=similar(I_gpu)
    output_W=similar(W_gpu)
    nchannel,_,nview=size(sino_iodine)
    a1=Float32(alpha1)
    a2=Float32(alpha2)
    try
        BS.AK.foreachindex(output_I) do idx
            channel=Int32(mod1(idx,nchannel))
            view=Int32(mod1(cld(idx,nchannel),nview))
            center_expected=max(expected_gpu[idx],1f-6)
            Y=measured_gpu[idx]
            center_likelihood=
                -center_expected+Y*log(center_expected)
            weight_sum=0f0
            iodine_sum=0f0
            water_sum=0f0
            for dv in Int32(-2):Int32(2),dc in Int32(-2):Int32(2)
                neighbor_channel=channel+dc
                (
                    neighbor_channel<Int32(1)||
                    neighbor_channel>Int32(nchannel)
                )&&continue
                neighbor_view=mod1(view+dv,Int32(nview))
                neighbor_idx=
                    Int(neighbor_channel)+
                    (Int(neighbor_view)-1)*nchannel
                spatial=exp(
                    -Float32(dc*dc+dv*dv)/(2f0*a1*a1),
                )
                likelihood_weight=if isinf(a2)
                    1f0
                elseif a2≤0f0
                    dc==0&&dv==0 ? 1f0 : 0f0
                else
                    candidate_expected=max(
                        expected_gpu[neighbor_idx],1f-6,
                    )
                    candidate_likelihood=
                        -candidate_expected+Y*log(candidate_expected)
                    delta=candidate_likelihood-center_likelihood
                    exp(-(delta*delta)/(a2*a2))
                end
                weight=spatial*likelihood_weight
                weight_sum+=weight
                iodine_sum+=weight*I_gpu[neighbor_idx]
                water_sum+=weight*W_gpu[neighbor_idx]
            end
            inverse_weight=1f0/max(weight_sum,eps(Float32))
            output_I[idx]=iodine_sum*inverse_weight
            output_W[idx]=water_sum*inverse_weight
        end
        (
            sino_iodine=Array(output_I),
            sino_water=Array(output_W),
        )
    finally
        BS.release_backend!((
            I_gpu,W_gpu,expected_gpu,measured_gpu,output_I,output_W,
        ))
    end
end

function reconstruct_material_pair(pair)
    (
        water=nchannel_common_fbp_slice(pair.sino_water),
        iodine=nchannel_common_fbp_slice(pair.sino_iodine),
    )
end

function synthesize_vmi_stack(images,energies)
    nx,ny,_=size(images.water)
    stack=Array{Float32}(undef,nx,ny,length(energies))
    iodine_mg_mL=images.iodine.*1000f0
    for (index,energy) in enumerate(energies)
        stack[:,:,index].=BS.synth_vmi_2basis(
            images.water,iodine_mg_mL;energy_keV=energy,
        )[:,:,1]
    end
    stack
end

function circular_roi(image_size;radius=28)
    nx,ny=image_size
    cx,cy=(nx+1)/2,(ny+1)/2
    [
        (i-cx)^2+(j-cy)^2≤radius^2
        for i in 1:nx,j in 1:ny
    ]
end

function circular_object_roi(image_size;radius=225)
    nx,ny=image_size
    cx,cy=(nx+1)/2,(ny+1)/2
    [
        (i-cx)^2+(j-cy)^2≤radius^2
        for i in 1:nx,j in 1:ny
    ]
end

function edge_width_pixels(image)
    nx,ny,_=size(image)
    profile=Float64.(image[:,ny÷2+1,1])
    center=nx÷2+1
    search=(center+round(Int,0.55center)):(nx-4)
    edge=search[argmax(abs.(diff(profile[search])))]
    inside=median(profile[max(center,edge-30):max(center,edge-10)])
    outside=median(profile[min(nx,edge+10):min(nx,edge+30)])
    low=outside+0.1*(inside-outside)
    high=outside+0.9*(inside-outside)
    segment=max(center,edge-20):min(nx,edge+20)
    function crossing(level)
        local_index=argmin(abs.(profile[segment].-level))
        first(segment)+local_index-1
    end
    (
        width=abs(crossing(low)-crossing(high)),
        edge,profile,inside,outside,
    )
end
end

# ╔═╡ c35105bb-7bcb-4905-8c82-c5d695e7c779
tlbf_primary_inputs = let
    reference_sinograms=(
        sino_iodine=matched_noiseless_pipeline.cong.sino_iodine,
        sino_water=matched_noiseless_pipeline.cong.sino_water,
    )
    noisy_sinograms=(
        sino_iodine=sino_basis_nchannel_slab.sino_iodine,
        sino_water=sino_basis_nchannel_slab.sino_water,
    )
    reference_images=reconstruct_material_pair(reference_sinograms)
    noisy_images=(
        water=nchannel_slab_common_fbp.vol_water,
        iodine=nchannel_slab_common_fbp.vol_iodine,
    )
    reference_vmis=synthesize_vmi_stack(
        reference_images,nchannel_validation_energies,
    )
    noisy_vmis=synthesize_vmi_stack(
        noisy_images,nchannel_validation_energies,
    )
    scale=nchannel_slab_counts.nrows
    Φ_total=scale.*vec(sum(Float64.(nchannel_basis.Φ);dims=2))
    reference_measured=total_measured_counts(
        matched_noiseless_pipeline.cong.slab_bins,
        nchannel_basis.I0,scale,
    )
    noisy_measured=total_measured_counts(
        nchannel_slab_counts.bins,nchannel_basis.I0,scale,
    )
    reference_expected=total_expected_counts(
        reference_sinograms.sino_iodine,
        reference_sinograms.sino_water,
        Φ_total,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    )
    noisy_expected=total_expected_counts(
        noisy_sinograms.sino_iodine,
        noisy_sinograms.sino_water,
        Φ_total,nchannel_basis.μρ_I,nchannel_basis.μρ_W,
    )
    homogeneous_roi=circular_roi(size(reference_images.water)[1:2])
    reference_water_mean=mean(
        reference_images.water[:,:,1][homogeneous_roi],
    )
    reference_iodine_mean=mean(
        reference_images.iodine[:,:,1][homogeneous_roi],
    )
    minimal_checks=(
        four_bins=length(nchannel_slab_counts.bins)==4,
        bin_order=collect(scanner.energy_thresholds)==
            [20.0,35.0,55.0,70.0],
        basis_order=(:iodine,:water),
        finite_noisy=all(isfinite,noisy_sinograms.sino_iodine)&&
            all(isfinite,noisy_sinograms.sino_water),
        finite_noiseless=all(isfinite,reference_sinograms.sino_iodine)&&
            all(isfinite,reference_sinograms.sino_water),
        matched_geometry=matched_noiseless_pipeline.simulated.geom.angles==
            sim_bins.geom.angles,
        common_fbp_kernel=:SoftFilter,
        reference_water_mean,
        reference_iodine_mean,
        background_truth_close=
            abs(reference_water_mean-1)<0.2&&
            abs(reference_iodine_mean)<0.01,
        vmi_algebra_check=maximum(abs.(
            reference_vmis[:,:,1].-
            BS.synth_vmi_2basis(
                reference_images.water,
                reference_images.iodine.*1000f0;
                energy_keV=first(nchannel_validation_energies),
            )[:,:,1]
        ))<1f-5,
    )
    (
        reference_noiseless_cong=(
            sinograms=reference_sinograms,
            images=reference_images,
            vmis=reference_vmis,
        ),
        baseline_noisy_cong=(
            sinograms=noisy_sinograms,
            images=noisy_images,
            vmis=noisy_vmis,
        ),
        reference_measured,noisy_measured,
        reference_expected,noisy_expected,
        Φ_total,homogeneous_roi,minimal_checks,
    )
end

# ╔═╡ f4f046cb-fd81-4683-9533-92d1e732dc24
tlbf_controls = let
    delta_sample=tlbf_delta_sample(
        tlbf_primary_inputs.noisy_expected,
        tlbf_primary_inputs.noisy_measured,
    )
    positive=filter(>(100eps(Float32)),delta_sample)
    isempty(positive)&&error("T-LBF likelihood differences are degenerate.")
    low=max(0.1quantile(positive,0.10),100eps(Float32))
    high=max(10quantile(positive,0.95),10low)
    finite_values=exp.(range(log(low),log(high);length=8))
    (
        window=(5,5),alpha1=0.9,
        alpha2=vcat(0.0,finite_values,Inf),
        delta_quantiles=(
            q10=quantile(positive,0.10),
            q50=quantile(positive,0.50),
            q95=quantile(positive,0.95),
            maximum=maximum(positive),
        ),
        angle_boundary=:circular,
        detector_boundary=:nonwrapping,
    )
end

# ╔═╡ ec10ffc2-7ba2-4a05-9346-930a958a6660
tlbf_sinogram_library = let
    noisy=NamedTuple[]
    noiseless=NamedTuple[]
    elapsed=@elapsed for alpha2 in tlbf_controls.alpha2
        push!(noisy,merge(
            (alpha2=alpha2,),
            tlbf_filter_pair(
                tlbf_primary_inputs.baseline_noisy_cong.sinograms.sino_iodine,
                tlbf_primary_inputs.baseline_noisy_cong.sinograms.sino_water,
                tlbf_primary_inputs.noisy_expected,
                tlbf_primary_inputs.noisy_measured,
                alpha2;alpha1=tlbf_controls.alpha1,
            ),
        ))
        push!(noiseless,merge(
            (alpha2=alpha2,),
            tlbf_filter_pair(
                tlbf_primary_inputs.reference_noiseless_cong.sinograms.sino_iodine,
                tlbf_primary_inputs.reference_noiseless_cong.sinograms.sino_water,
                tlbf_primary_inputs.reference_expected,
                tlbf_primary_inputs.reference_measured,
                alpha2;alpha1=tlbf_controls.alpha1,
            ),
        ))
    end
    (
        noisy,noiseless,elapsed_s=elapsed,
        exact_joint_weights=true,
    )
end

# ╔═╡ c2c7b7ef-296e-4005-a578-bd38ec818be3
tlbf_reconstruction_library = let
    entries=NamedTuple[]
    elapsed=@elapsed for index in eachindex(tlbf_controls.alpha2)
        noisy_images=reconstruct_material_pair(
            tlbf_sinogram_library.noisy[index],
        )
        noiseless_images=reconstruct_material_pair(
            tlbf_sinogram_library.noiseless[index],
        )
        push!(entries,(
            alpha2=tlbf_controls.alpha2[index],
            noisy_images,
            noiseless_images,
            noisy_vmis=synthesize_vmi_stack(
                noisy_images,nchannel_validation_energies,
            ),
            noiseless_vmis=synthesize_vmi_stack(
                noiseless_images,nchannel_validation_energies,
            ),
        ))
    end
    (entries,elapsed_s=elapsed)
end

# ╔═╡ 307532ad-9599-4e8d-ba1a-2681fcf2f110
tlbf_metrics = let
    reference=tlbf_primary_inputs.reference_noiseless_cong
    roi=tlbf_primary_inputs.homogeneous_roi
    object_roi=circular_object_roi(size(reference.images.water)[1:2])
    reference_edge=edge_width_pixels(reference.images.water)
    entries=map(tlbf_reconstruction_library.entries) do entry
        noisy_W=entry.noisy_images.water[:,:,1]
        noisy_I=entry.noisy_images.iodine[:,:,1]
        noiseless_W=entry.noiseless_images.water[:,:,1]
        noiseless_I=entry.noiseless_images.iodine[:,:,1]
        ref_W=reference.images.water[:,:,1]
        ref_I=reference.images.iodine[:,:,1]
        edge=edge_width_pixels(entry.noiseless_images.water)
        vmi_noise=[
            std(entry.noisy_vmis[:,:,energy][roi])
            for energy in axes(entry.noisy_vmis,3)
        ]
        (
            alpha2=entry.alpha2,
            water_sd=std(noisy_W[roi]),
            iodine_sd=std(noisy_I[roi]),
            water_rmse=sqrt(mean(abs2,noisy_W[object_roi].-ref_W[object_roi])),
            iodine_rmse=sqrt(mean(abs2,noisy_I[object_roi].-ref_I[object_roi])),
            water_bias=mean(noisy_W[roi].-ref_W[roi]),
            iodine_bias=mean(noisy_I[roi].-ref_I[roi]),
            vmi_noise,
            vmi_rmse=[
                sqrt(mean(abs2,
                    entry.noisy_vmis[:,:,energy][object_roi].-
                    reference.vmis[:,:,energy][object_roi],
                ))
                for energy in axes(entry.noisy_vmis,3)
            ],
            deterministic_water_rmse=sqrt(mean(
                abs2,noiseless_W[object_roi].-ref_W[object_roi],
            )),
            deterministic_iodine_rmse=sqrt(mean(
                abs2,noiseless_I[object_roi].-ref_I[object_roi],
            )),
            edge_width_pixels=edge.width,
            edge_width_ratio=edge.width/max(reference_edge.width,1),
            edge_profile=edge.profile,
        )
    end
    (
        entries,
        reference_edge_width_pixels=reference_edge.width,
        reference_edge_profile=reference_edge.profile,
        object_roi,
    )
end

# ╔═╡ 1ea2dcbe-8cd8-4b0c-85f1-d59bf5d37a78
tlbf_selection = let
    n=length(tlbf_metrics.entries)
    light=clamp(3,2,n-1)
    moderate=clamp(round(Int,(n+1)/2),2,n-1)
    strong=clamp(n-1,2,n-1)
    energy70=argmin(abs.(nchannel_validation_energies.-70))
    candidates=2:n
    best=candidates[argmin([
        tlbf_metrics.entries[index].vmi_rmse[energy70]
        for index in candidates
    ])]
    (
        light,moderate,strong,best,
        representative=(light,moderate,strong),
        best_alpha2=tlbf_controls.alpha2[best],
        selection_rule=:minimum_70keV_RMSE_with_reported_resolution,
    )
end

# ╔═╡ 07f4d2b6-6c29-47f3-88dd-98fc915009bc
tlbf_noise_matching = let
    roi=tlbf_primary_inputs.homogeneous_roi
    library=tlbf_reconstruction_library.entries
    water_index=tlbf_selection.best
    water=library[water_index].noisy_images.water[:,:,1]
    sigma_W=std(water[roi])
    common_stack=library[water_index].noisy_vmis
    matched_stack=similar(common_stack)
    selections=NamedTuple[]
    for (energy_index,energy) in enumerate(nchannel_validation_energies)
        phi_W=BS.compute_mass_μ_at_energy(BS.XA.Materials.water,energy)
        phi_I=BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine,energy)
        scores=Float64[]
        correlations=Float64[]
        for iodine_entry in library
            iodine=iodine_entry.noisy_images.iodine[:,:,1]
            sigma_I=std(iodine[roi])
            rho=cor(water[roi],iodine[roi])
            push!(correlations,rho)
            push!(scores,abs(
                sigma_I*phi_I+rho*sigma_W*phi_W,
            ))
        end
        iodine_index=argmin(scores)
        iodine=library[iodine_index].noisy_images.iodine
        matched_stack[:,:,energy_index].=
            BS.synth_vmi_2basis(
                library[water_index].noisy_images.water,
                iodine.*1000f0;energy_keV=energy,
            )[:,:,1]
        push!(selections,(;
            energy,water_index,iodine_index,
            water_alpha2=library[water_index].alpha2,
            iodine_alpha2=library[iodine_index].alpha2,
            rho=correlations[iodine_index],
            matching_error=scores[iodine_index],
        ))
    end
    (
        tlbf_common_pair_vmis=common_stack,
        tlbf_noise_matched_vmis=matched_stack,
        selections,
        common_noise=[
            std(common_stack[:,:,index][roi])
            for index in axes(common_stack,3)
        ],
        matched_noise=[
            std(matched_stack[:,:,index][roi])
            for index in axes(matched_stack,3)
        ],
    )
end

# ╔═╡ 4f210d68-19cd-4b67-adce-6d59498f1366
tlbf_feasibility_summary = let
    baseline=tlbf_metrics.entries[1]
    best=tlbf_metrics.entries[tlbf_selection.best]
    energy70=argmin(abs.(nchannel_validation_energies.-70))
    common=tlbf_noise_matching.common_noise
    matched=tlbf_noise_matching.matched_noise
    monotonic_nonincreasing(curve)=all(diff(curve).≤0)
    (
        implementation=:Lee_2025_T_LBF,
        deviations=(
            :corrected_fractional_count_equivalents,
            :single_noisy_realization_feasibility_pass,
        ),
        best_alpha2=tlbf_selection.best_alpha2,
        water_noise_reduction_percent=
            100*(1-best.water_sd/baseline.water_sd),
        iodine_noise_reduction_percent=
            100*(1-best.iodine_sd/baseline.iodine_sd),
        vmi70_noise_reduction_percent=
            100*(1-best.vmi_noise[energy70]/baseline.vmi_noise[energy70]),
        water_roi_bias=best.water_bias,
        iodine_roi_bias=best.iodine_bias,
        deterministic_water_rmse=best.deterministic_water_rmse,
        deterministic_iodine_rmse=best.deterministic_iodine_rmse,
        reference_edge_width_pixels=
            tlbf_metrics.reference_edge_width_pixels,
        filtered_edge_width_pixels=best.edge_width_pixels,
        common_pair_monotonic=monotonic_nonincreasing(common),
        noise_matched_monotonic=monotonic_nonincreasing(matched),
        noise_matching_best_improvement_percent=
            100maximum((common.-matched)./common),
        noise_matching_worst_change_percent=
            100minimum((common.-matched)./common),
        decision=:promising_but_needs_refinement,
    )
end

# ╔═╡ d544993a-e959-48c7-b1da-2888edc7623a
let
    best=tlbf_reconstruction_library.entries[tlbf_selection.best]
    reference=tlbf_primary_inputs.reference_noiseless_cong
    baseline=tlbf_primary_inputs.baseline_noisy_cong
    energy_index=argmin(abs.(nchannel_validation_energies.-70))
    images=[
        reference.vmis[:,:,energy_index],
        best.noiseless_vmis[:,:,energy_index],
        baseline.vmis[:,:,energy_index],
        best.noisy_vmis[:,:,energy_index],
    ]
    titles=[
        "Noiseless Cong","Noiseless + T-LBF",
        "Noisy Cong","Noisy + T-LBF",
    ]
    HU_window=(-200,500)
    fig=Mke.Figure(size=(1000,850))
    for index in eachindex(images)
        row,col=Tuple(CartesianIndices((2,2))[index])
        axis=Mke.Axis(
            fig[row,col];title=titles[index],aspect=Mke.DataAspect(),
        )
        Mke.heatmap!(
            axis,images[index];colormap=:grays,colorrange=HU_window,
        )
        Mke.hidedecorations!(axis)
    end
    Mke.Label(
        fig[0,:],
        "Matched 70 keV comparison — α₂=$(round(tlbf_selection.best_alpha2,digits=3))",
        fontsize=22,
    )
    Mke.Colorbar(
        fig[1:2,3];colormap=:grays,colorrange=HU_window,
        label="HU",width=16,labelsize=22,ticklabelsize=18,
    )
    fig
end

# ╔═╡ 881c8fac-c00a-46c3-9bbf-802227d9c4d0
let
    best=tlbf_reconstruction_library.entries[tlbf_selection.best]
    baseline=tlbf_primary_inputs.baseline_noisy_cong
    energies=[40.0,70.0,100.0,140.0]
    indices=[
        argmin(abs.(nchannel_validation_energies.-energy))
        for energy in energies
    ]
    fig=Mke.Figure(size=(1450,700))
    for (column,(energy,index)) in enumerate(zip(energies,indices))
        pair=[
            baseline.vmis[:,:,index],
            best.noisy_vmis[:,:,index],
        ]
        HU_window=(-200,500)
        for row in 1:2
            axis=Mke.Axis(
                fig[row,column];
                title=row==1 ? "$(Int(energy)) keV — Cong" :
                    "$(Int(energy)) keV — T-LBF",
                aspect=Mke.DataAspect(),
            )
            Mke.heatmap!(
                axis,pair[row];colormap=:grays,colorrange=HU_window,
            )
            Mke.hidedecorations!(axis)
        end
    end
    Mke.Label(
        fig[0,:],
        "Noisy VMIs before and after common-pair T-LBF",
        fontsize=22,
    )
    Mke.Colorbar(
        fig[1:2,5];colormap=:grays,colorrange=(-200,500),
        label="HU",width=16,labelsize=22,ticklabelsize=18,
    )
    fig
end

# ╔═╡ c0fcdf63-513d-489e-b105-0454b18c3f83
let
    alpha_labels=[
        isinf(value) ? "LPF" : string(round(value,sigdigits=3))
        for value in tlbf_controls.alpha2
    ]
    baseline=tlbf_metrics.entries[1]
    fig=Mke.Figure(size=(1100,450))
    noise_axis=Mke.Axis(
        fig[1,1];xlabel="α₂ setting",ylabel="70 keV noise / baseline",
        xticks=(eachindex(alpha_labels),alpha_labels),
    )
    energy70=argmin(abs.(nchannel_validation_energies.-70))
    Mke.scatterlines!(
        noise_axis,eachindex(alpha_labels),
        [entry.vmi_noise[energy70]/baseline.vmi_noise[energy70]
         for entry in tlbf_metrics.entries];
        marker=:circle,label="noise",
    )
    Mke.scatterlines!(
        noise_axis,eachindex(alpha_labels),
        [entry.vmi_rmse[energy70]/baseline.vmi_rmse[energy70]
         for entry in tlbf_metrics.entries];
        marker=:rect,label="RMSE",
    )
    Mke.axislegend(noise_axis)
    edge_axis=Mke.Axis(
        fig[1,2];xlabel="pixel",ylabel="water value",
        title="Noiseless edge profiles",
    )
    Mke.lines!(
        edge_axis,tlbf_metrics.reference_edge_profile;
        label="unfiltered",
    )
    for index in tlbf_selection.representative
        Mke.lines!(
            edge_axis,tlbf_metrics.entries[index].edge_profile;
            label="α₂=$(alpha_labels[index])",
        )
    end
    Mke.axislegend(edge_axis;position=:lb)
    fig
end

# ╔═╡ 24db3af8-9e58-4499-8de0-5966c9fe0394
let
    baseline=tlbf_metrics.entries[1].vmi_noise
    fig=Mke.Figure(size=(850,480))
    axis=Mke.Axis(
        fig[1,1];xlabel="VMI energy (keV)",ylabel="ROI SD",
        title="VMI noise-versus-energy feasibility",
    )
    Mke.scatterlines!(
        axis,nchannel_validation_energies,baseline;
        marker=:circle,label="unfiltered Cong",
    )
    Mke.scatterlines!(
        axis,nchannel_validation_energies,tlbf_noise_matching.common_noise;
        marker=:rect,label="common-pair T-LBF",
    )
    Mke.scatterlines!(
        axis,nchannel_validation_energies,tlbf_noise_matching.matched_noise;
        marker=:utriangle,label="noise-matched T-LBF",
    )
    Mke.axislegend(axis)
    fig
end

# ╔═╡ e788e4ab-50f1-4bad-af31-f31ff877e9de
let
    roi=tlbf_primary_inputs.homogeneous_roi
    reference=tlbf_primary_inputs.reference_noiseless_cong.vmis
    baseline=tlbf_primary_inputs.baseline_noisy_cong.vmis
    common=tlbf_noise_matching.tlbf_common_pair_vmis
    matched=tlbf_noise_matching.tlbf_noise_matched_vmis
    roi_mean(stack)=[
        mean(stack[:,:,index][roi]) for index in axes(stack,3)
    ]
    fig=Mke.Figure(size=(850,480))
    axis=Mke.Axis(
        fig[1,1];xlabel="VMI energy (keV)",ylabel="Water-region mean (HU)",
        title="VMI mean HU versus energy",
    )
    Mke.lines!(
        axis,nchannel_validation_energies,roi_mean(reference);
        color=:black,linestyle=:dash,linewidth=2,label="noiseless Cong",
    )
    Mke.scatterlines!(
        axis,nchannel_validation_energies,roi_mean(baseline);
        label="noisy Cong",
    )
    Mke.scatterlines!(
        axis,nchannel_validation_energies,roi_mean(common);
        label="common-pair T-LBF",
    )
    Mke.scatterlines!(
        axis,nchannel_validation_energies,roi_mean(matched);
        label="noise-matched T-LBF",
    )
    Mke.axislegend(axis;position=:rb)
    fig
end

# ╔═╡ a702126e-7e5a-47c5-a45e-b8038ae096c8
let
    best=tlbf_reconstruction_library.entries[tlbf_selection.best]
    reference=tlbf_primary_inputs.reference_noiseless_cong
    baseline=tlbf_primary_inputs.baseline_noisy_cong
    energy_index=argmin(abs.(nchannel_validation_energies.-70))
    differences=[
        baseline.vmis[:,:,energy_index].-reference.vmis[:,:,energy_index],
        best.noisy_vmis[:,:,energy_index].-reference.vmis[:,:,energy_index],
        best.noiseless_vmis[:,:,energy_index].-reference.vmis[:,:,energy_index],
    ]
    titles=[
        "Noisy Cong − reference",
        "Noisy T-LBF − reference",
        "Noiseless T-LBF − reference",
    ]
    limit=quantile(abs.(vcat(vec.(differences)...)),0.98)
    fig=Mke.Figure(size=(1350,430))
    for index in eachindex(differences)
        axis=Mke.Axis(
            fig[1,index];title=titles[index],aspect=Mke.DataAspect(),
        )
        Mke.heatmap!(
            axis,differences[index];
            colormap=:balance,colorrange=(-limit,limit),
        )
        Mke.hidedecorations!(axis)
    end
    Mke.Colorbar(
        fig[1,4];colormap=:balance,colorrange=(-limit,limit),
        label="HU difference",width=16,
    )
    fig
end

# ╔═╡ 71344579-fce7-46a1-b55d-1029c9c20229
tlbf_original_style_results = let
    energies=[40.0,70.0,100.0,140.0]
    energy_indices=[argmin(abs.(nchannel_validation_energies.-E)) for E in energies]
    vmis=Dict(E=>tlbf_noise_matching.tlbf_common_pair_vmis[:,:,i]
              for (E,i) in zip(energies,energy_indices))
    mask_2d=phantom_cpu.mask[:,:,size(phantom_cpu.mask,3)÷2]
    water_mask=collect(BS.erode_mask_2d(
        mask_2d.==UInt8(BS.REGION_SOLID_WATER);erode_px=12.0,
    ))
    labels=(
        Ca=UInt8.((10,11,12,13,14,15,16)),
        I=UInt8.((20,21,22,23,24,25,26)),
    )
    names=(
        Ca=("50 mg/mL","100 mg/mL","200 mg/mL","300 mg/mL",
            "400 mg/mL","500 mg/mL","600 mg/mL"),
        I=("2.0 mg/mL","2.5 mg/mL","5.0 mg/mL","7.5 mg/mL",
           "10.0 mg/mL","15.0 mg/mL","20.0 mg/mL"),
    )
    function rod_roi(label)
        pixels=findall(==(label),mask_2d)
        cx=mean(p->Float64(p[1]),pixels)
        cy=mean(p->Float64(p[2]),pixels)
        [CartesianIndex(i,j)
         for j in max(1,floor(Int,cy-8)):min(size(mask_2d,2),ceil(Int,cy+8))
         for i in max(1,floor(Int,cx-8)):min(size(mask_2d,1),ceil(Int,cx+8))
         if (i-cx)^2+(j-cy)^2≤64]
    end
    μwater=Dict(E=>BS.compute_μ_at_energy(BS.XA.Materials.water,E) for E in energies)
    rod_data=Dict{Symbol,NamedTuple}()
    for group in (:Ca,:I)
        measured=zeros(length(labels[group]),length(energies))
        theoretical=similar(measured)
        for (row,label) in pairs(labels[group])
            roi=rod_roi(label)
            material=phantom_cpu.materials[Int(label)+1]
            for (column,E) in pairs(energies)
                measured[row,column]=mean(vmis[E][roi])
                μ=BS.compute_μ_at_energy(material,E)
                theoretical[row,column]=1000*(μ-μwater[E])/μwater[E]
            end
        end
        rod_data[group]=(names=names[group],measured,theoretical)
    end
    (
        energies,vmis,water_mask,rod_data,
        water_mean=[mean(vmis[E][water_mask]) for E in energies],
        water_noise=[std(vmis[E][water_mask]) for E in energies],
    )
end

# ╔═╡ ab1319af-8619-4ae8-853a-b5e63cb134f3
let
    result=tlbf_original_style_results
    overlay=Float32[value ? 1f0 : NaN32 for value in result.water_mask]
    image70=result.vmis[70.0]
    n=length(result.energies)
    colors=[Mke.cgrad(:plasma,n;categorical=true)[i] for i in 1:n]
    fig=Mke.Figure(size=(1180,1180))
    ax1=Mke.Axis(fig[1,1];title="Eroded Water Region",
        subtitle="Overlaid on 70 keV T-LBF VMI",aspect=Mke.DataAspect(),
        titlesize=32,subtitlesize=24)
    Mke.heatmap!(ax1,image70;colormap=:grays,colorrange=(-200,500))
    Mke.heatmap!(ax1,overlay;colormap=:reds,alpha=0.5,nan_color=(:white,0.0))
    Mke.hidedecorations!(ax1)
    ax2=Mke.Axis(fig[1,2];title="Water Region Mean HU",
        subtitle="Selected T-LBF · Per VMI Energy",
        xlabel="VMI Energy (keV)",ylabel="HU",
        xticks=(1:n,string.(Int.(result.energies))),
        titlesize=32,subtitlesize=24)
    Mke.barplot!(ax2,1:n,result.water_mean;color=colors,
        strokecolor=:black,strokewidth=1)
    Mke.hlines!(ax2,[0.0];color=:black,linestyle=:dash)
    for (index,value) in pairs(result.water_mean)
        Mke.text!(ax2,index,value;text="$(round(value,digits=1)) HU",
            align=(:center,value≥0 ? :bottom : :top),
            offset=(0,value≥0 ? 4 : -4))
    end
    limit=max(15.0,1.2maximum(abs,result.water_mean))
    Mke.ylims!(ax2,-limit,limit)
    ax3=Mke.Axis(fig[2,1];title="Eroded Water Region",
        subtitle="Canonical noise ROI",aspect=Mke.DataAspect(),
        titlesize=32,subtitlesize=24)
    Mke.heatmap!(ax3,image70;colormap=:grays,colorrange=(-200,500))
    Mke.heatmap!(ax3,overlay;colormap=:reds,alpha=0.5,nan_color=(:white,0.0))
    Mke.hidedecorations!(ax3)
    ax4=Mke.Axis(fig[2,2];title="Water-Region Noise vs Energy",
        subtitle="Selected T-LBF",xlabel="VMI Energy (keV)",
        ylabel="Noise σ (HU)",xticks=(1:n,string.(Int.(result.energies))),
        titlesize=32,subtitlesize=24)
    Mke.barplot!(ax4,1:n,result.water_noise;color=:tomato,
        strokecolor=:black,strokewidth=1)
    for index in 1:n
        Mke.text!(ax4,index,result.water_noise[index];
            text="σ=$(round(result.water_noise[index],digits=1))\n"*
                 "⟨HU⟩=$(round(result.water_mean[index],digits=1))",
            align=(:center,:bottom),offset=(0,8))
    end
    Mke.ylims!(ax4,0,1.25maximum(result.water_noise))
    fig
end

# ╔═╡ d95a694b-dac0-4b28-bac6-2dadc94fe355
let
    result=tlbf_original_style_results
    fig=Mke.Figure(size=(1180,580))
    panels=(
        (group=:Ca,title="Calcium rods",subtitle="50–600 mg/mL",
         cmap=Mke.cgrad(:Oranges,7;categorical=true),ylim=(0,5500)),
        (group=:I,title="Iodine rods",subtitle="2–20 mg/mL",
         cmap=Mke.cgrad(:GnBu,7;categorical=true),ylim=(0,2500)),
    )
    for (column,panel) in pairs(panels)
        axis=Mke.Axis(fig[1,column];title=panel.title,subtitle=panel.subtitle,
            xlabel="VMI energy (keV)",ylabel="HU",xticks=result.energies,
            titlesize=32,subtitlesize=24)
        Mke.ylims!(axis,panel.ylim...)
        data=result.rod_data[panel.group]
        rod_lines=Any[]
        for index in eachindex(data.names)
            color=panel.cmap[index]
            Mke.scatterlines!(axis,result.energies,vec(data.measured[index,:]);
                color,linewidth=2.5,markersize=9)
            Mke.lines!(axis,result.energies,vec(data.theoretical[index,:]);
                color,linewidth=1.6,linestyle=:dash)
            push!(rod_lines,Mke.LineElement(;color,linewidth=2.5))
        end
        Mke.axislegend(axis,
            vcat([Mke.MarkerElement(color=:black,marker=:circle),
                  Mke.LineElement(color=:black,linestyle=:dash)],rod_lines),
            vcat(["Measured","Theoretical"],collect(data.names));
            position=:rt,framevisible=true,labelsize=18,rowgap=1)
    end
    fig
end

# ╔═╡ a6694310-43ec-4bf6-bf22-374e2fcf6b00
let
    result=tlbf_original_style_results
    colors=Dict(40.0=>Mke.RGBf(0.85,0.27,0.1),
        70.0=>Mke.RGBf(0.95,0.65,0.13),
        100.0=>Mke.RGBf(0.13,0.59,0.85),
        140.0=>Mke.RGBf(0.1,0.27,0.65))
    fig=Mke.Figure(size=(1000,1200))
    for (row,group) in enumerate((:Ca,:I))
        data=result.rod_data[group]
        axis=Mke.Axis(fig[row,1];
            title=group==:Ca ? "Calcium regression" : "Iodine regression",
            xlabel="Theoretical HU",ylabel="Measured T-LBF HU",
            aspect=Mke.DataAspect(),titlesize=30)
        low=min(0.0,minimum(data.measured),minimum(data.theoretical))
        high=1.05max(maximum(data.measured),maximum(data.theoretical))
        Mke.lines!(axis,[low,high],[low,high];color=:black,
            linestyle=:dash,linewidth=2,label="Unity (y=x)")
        for (column,E) in pairs(result.energies)
            x=vec(data.theoretical[:,column]); y=vec(data.measured[:,column])
            slope=sum((x.-mean(x)).*(y.-mean(y)))/sum(abs2,x.-mean(x))
            intercept=mean(y)-slope*mean(x)
            prediction=intercept.+slope.*x
            r2=1-sum(abs2,y.-prediction)/sum(abs2,y.-mean(y))
            endpoints=collect(extrema(x)); color=colors[E]
            Mke.scatter!(axis,x,y;color,markersize=11)
            Mke.lines!(axis,endpoints,intercept.+slope.*endpoints;
                color,linewidth=2,
                label="$(Int(E)) keV: slope=$(round(slope,digits=2)), "*
                      "R²=$(round(r2,digits=3))")
        end
        Mke.axislegend(axis;position=:rb,labelsize=16)
    end
    fig
end

# ╔═╡ 12497ebd-c1b4-4789-836a-fc49c5a06762
begin
function nchannel_fbp_slice_filter(sino_one_row,filter)
    spectrum=BS.FFTW.fft(Float64.(sino_one_row),3)
    antialiased=Float32.(real.(BS.FFTW.ifft(
        spectrum.*reshape(
            nchannel_fbp_angular_response,1,1,nchannel_fbp_nview,
        ),3,
    )))
    repeated=repeat(antialiased,1,sim_bins.geom.n_rows,1)
    sino_gpu=to_gpu(repeated)
    workspace=BS.create_fdk_recon_workspace(
        sino_gpu,sim_bins.geom,nchannel_fbp_matrix_size;filter,
    )
    try
        Float32.(Array(BS.reconstruct!(workspace,sino_gpu,sim_bins.geom)))
    finally
        BS.release_backend!(workspace)
    end
end
end

# ╔═╡ e6d42b22-8fe5-48a2-982e-0db460fd9797
basis_fbp_filter_sweep = let
    filters=[
        (name="Soft baseline",filter=BS.SoftFilter()),
        (name="Light iodine",filter=BS.CustomFilter(
            (0.0,0.25,0.5,0.75,1.0),
            (1.0,0.65,0.30,0.10,0.001),
        )),
        (name="Original 04 iodine",filter=BS.CustomFilter(
            (0.0,0.25,0.5,0.75,1.0),
            (1.0,0.40,0.12,0.03,0.001),
        )),
        (name="Strong iodine",filter=BS.CustomFilter(
            (0.0,0.25,0.5,0.75,1.0),
            (1.0,0.25,0.05,0.005,0.0),
        )),
    ]
    best=tlbf_selection.best
    noisy_sino=tlbf_sinogram_library.noisy[best]
    noiseless_sino=tlbf_sinogram_library.noiseless[best]
    base=tlbf_reconstruction_library.entries[best]
    entries=NamedTuple[]
    elapsed=@elapsed for candidate in filters
        noisy_iodine=nchannel_fbp_slice_filter(
            noisy_sino.sino_iodine,candidate.filter,
        )
        noiseless_iodine=nchannel_fbp_slice_filter(
            noiseless_sino.sino_iodine,candidate.filter,
        )
        noisy_images=(water=base.noisy_images.water,iodine=noisy_iodine)
        noiseless_images=(
            water=base.noiseless_images.water,iodine=noiseless_iodine,
        )
        push!(entries,(
            name=candidate.name,filter=candidate.filter,
            noisy_images,noiseless_images,
            noisy_vmis=synthesize_vmi_stack(
                noisy_images,nchannel_validation_energies,
            ),
            noiseless_vmis=synthesize_vmi_stack(
                noiseless_images,nchannel_validation_energies,
            ),
        ))
    end
    (entries,elapsed_s=elapsed,water_filter=:SoftFilter)
end

# ╔═╡ 07ed190a-58e8-4d0a-8802-137ceabbb67e
basis_fbp_filter_metrics = let
    roi=tlbf_original_style_results.water_mask
    reference=tlbf_primary_inputs.reference_noiseless_cong.vmis
    object_roi=tlbf_metrics.object_roi
    entries=map(basis_fbp_filter_sweep.entries) do entry
        noise=[
            std(entry.noisy_vmis[:,:,index][roi])
            for index in axes(entry.noisy_vmis,3)
        ]
        means=[
            mean(entry.noisy_vmis[:,:,index][roi])
            for index in axes(entry.noisy_vmis,3)
        ]
        rmse=[
            sqrt(mean(abs2,
                entry.noisy_vmis[:,:,index][object_roi].-
                reference[:,:,index][object_roi],
            ))
            for index in axes(entry.noisy_vmis,3)
        ]
        deterministic_rmse=[
            sqrt(mean(abs2,
                entry.noiseless_vmis[:,:,index][object_roi].-
                reference[:,:,index][object_roi],
            ))
            for index in axes(entry.noiseless_vmis,3)
        ]
        (
            name=entry.name,noise,means,rmse,deterministic_rmse,
            monotonic=all(diff(noise).≤0),
            forty_noise=noise[1],
            forty_edge=edge_width_pixels(
                reshape(entry.noiseless_vmis[:,:,1],512,512,1),
            ).width,
        )
    end
    baseline=entries[1]
    (
        entries,
        best_40_index=argmin(getfield.(entries,:forty_noise)),
        forty_noise_reduction_percent=[
            100*(1-entry.forty_noise/baseline.forty_noise)
            for entry in entries
        ],
    )
end

# ╔═╡ 082a7479-95d7-4973-8b18-a1d639542b25
let
    fig=Mke.Figure(size=(1100,500))
    axis=Mke.Axis(
        fig[1,1];title="Per-basis FBP apodization",
        subtitle="Water fixed at SoftFilter; iodine filter varied",
        xlabel="VMI energy (keV)",ylabel="Water-region noise σ (HU)",
        titlesize=30,subtitlesize=20,
    )
    for entry in basis_fbp_filter_metrics.entries
        Mke.scatterlines!(
            axis,nchannel_validation_energies,entry.noise;
            label=entry.name,linewidth=2,markersize=8,
        )
    end
    Mke.axislegend(axis)
    fig
end

# ╔═╡ d7266214-0e21-411f-90f1-226b7ef1a742
let
    candidates=basis_fbp_filter_sweep.entries
    energy_index=1
    HU_window=(-200,500)
    fig=Mke.Figure(size=(1180,1180))
    for index in eachindex(candidates)
        row=((index-1)÷2)+1
        column=((index-1)%2)+1
        axis=Mke.Axis(
            fig[row,column];title="$(candidates[index].name) · 40 keV",
            aspect=Mke.DataAspect(),titlesize=26,
        )
        Mke.heatmap!(
            axis,candidates[index].noisy_vmis[:,:,energy_index];
            colormap=:grays,colorrange=HU_window,
        )
        Mke.hidedecorations!(axis)
    end
    Mke.Colorbar(
        fig[1:2,3];colormap=:grays,colorrange=HU_window,
        label="HU",width=16,
    )
    fig
end

# ╔═╡ 712d508b-95b5-43f3-b305-d0259134cd74
water_fbp_filter_sweep = let
    filters=[
        (name="Soft water",filter=BS.SoftFilter()),
        (name="Light-soft water",filter=BS.CustomFilter(
            (0.0,0.25,0.5,0.75,1.0),
            (1.0,0.70,0.35,0.12,0.001),
        )),
        (name="Medium-soft water",filter=BS.CustomFilter(
            (0.0,0.25,0.5,0.75,1.0),
            (1.0,0.50,0.18,0.05,0.001),
        )),
        (name="Strong-soft water",filter=BS.CustomFilter(
            (0.0,0.25,0.5,0.75,1.0),
            (1.0,0.30,0.07,0.01,0.0),
        )),
    ]
    best=tlbf_selection.best
    noisy_sino=tlbf_sinogram_library.noisy[best]
    noiseless_sino=tlbf_sinogram_library.noiseless[best]
    fixed_iodine=basis_fbp_filter_sweep.entries[3]
    entries=NamedTuple[]
    elapsed=@elapsed for candidate in filters
        noisy_water=nchannel_fbp_slice_filter(
            noisy_sino.sino_water,candidate.filter,
        )
        noiseless_water=nchannel_fbp_slice_filter(
            noiseless_sino.sino_water,candidate.filter,
        )
        noisy_images=(
            water=noisy_water,
            iodine=fixed_iodine.noisy_images.iodine,
        )
        noiseless_images=(
            water=noiseless_water,
            iodine=fixed_iodine.noiseless_images.iodine,
        )
        push!(entries,(
            name=candidate.name,filter=candidate.filter,
            noisy_images,noiseless_images,
            noisy_vmis=synthesize_vmi_stack(
                noisy_images,nchannel_validation_energies,
            ),
            noiseless_vmis=synthesize_vmi_stack(
                noiseless_images,nchannel_validation_energies,
            ),
        ))
    end
    (
        entries,elapsed_s=elapsed,
        iodine_filter="Original 04 iodine",
    )
end

# ╔═╡ fd6ab3d2-88e1-4862-a574-62b57f265279
water_fbp_filter_metrics = let
    roi=tlbf_original_style_results.water_mask
    object_roi=tlbf_metrics.object_roi
    reference=tlbf_primary_inputs.reference_noiseless_cong.vmis
    entries=map(water_fbp_filter_sweep.entries) do entry
        noise=[
            std(entry.noisy_vmis[:,:,index][roi])
            for index in axes(entry.noisy_vmis,3)
        ]
        deterministic_rmse=[
            sqrt(mean(abs2,
                entry.noiseless_vmis[:,:,index][object_roi].-
                reference[:,:,index][object_roi],
            ))
            for index in axes(entry.noiseless_vmis,3)
        ]
        (
            name=entry.name,noise,deterministic_rmse,
            monotonic=all(diff(noise).≤0),
            upward_variation=sum(max.(diff(noise),0)),
            forty_noise=noise[1],
            oneforty_noise=noise[
                argmin(abs.(nchannel_validation_energies.-140))
            ],
            forty_edge=edge_width_pixels(
                reshape(entry.noiseless_vmis[:,:,1],512,512,1),
            ).width,
        )
    end
    monotonic_indices=findall(getfield.(entries,:monotonic))
    selected=if isempty(monotonic_indices)
        argmin(getfield.(entries,:upward_variation))
    else
        monotonic_indices[argmin([
            mean(entries[index].deterministic_rmse)
            for index in monotonic_indices
        ])]
    end
    (entries,selected,any_monotonic=!isempty(monotonic_indices))
end

# ╔═╡ 4e9a9a3c-f9f2-41b8-b85b-155c0a6d1ff0
let
    fig=Mke.Figure(size=(1100,500))
    axis=Mke.Axis(
        fig[1,1];title="Water FBP kernel sweep",
        subtitle="Iodine fixed at original 04 kernel",
        xlabel="VMI energy (keV)",ylabel="Water-region noise σ (HU)",
        titlesize=30,subtitlesize=20,
    )
    for entry in water_fbp_filter_metrics.entries
        Mke.scatterlines!(
            axis,nchannel_validation_energies,entry.noise;
            label="$(entry.name)$(entry.monotonic ? " ✓ monotonic" : "")",
            linewidth=2,markersize=8,
        )
    end
    Mke.axislegend(axis)
    fig
end

# ╔═╡ 6fabf137-a705-436c-bdf0-0697f5d5bd7b
let
    candidates=water_fbp_filter_sweep.entries
    energy_indices=[
        argmin(abs.(nchannel_validation_energies.-energy))
        for energy in (40.0,70.0,140.0)
    ]
    selected=water_fbp_filter_metrics.selected
    HU_window=(-200,500)
    fig=Mke.Figure(size=(1400,850))
    for (row,index) in enumerate((1,selected))
        for (column,energy_index) in enumerate(energy_indices)
            energy=nchannel_validation_energies[energy_index]
            axis=Mke.Axis(
                fig[row,column];
                title="$(candidates[index].name) · $(Int(energy)) keV",
                aspect=Mke.DataAspect(),titlesize=23,
            )
            Mke.heatmap!(
                axis,candidates[index].noisy_vmis[:,:,energy_index];
                colormap=:grays,colorrange=HU_window,
            )
            Mke.hidedecorations!(axis)
        end
    end
    Mke.Colorbar(
        fig[1:2,4];colormap=:grays,colorrange=HU_window,
        label="HU",width=16,
    )
    fig
end

# ╔═╡ b1d0f9d7-0e3e-4b65-b587-d9efcfac5a97
md"""
## Feasibility conclusion

**Decision: promising but needs refinement.**

The Lee 2025 T-LBF was implemented as a joint filter of the completed water
and iodine Cong sinograms. At the provisional best setting
(`α₂ = $(round(tlbf_feasibility_summary.best_alpha2,digits=3))`), water,
iodine, and 70 keV noise fell by approximately
$(round(tlbf_feasibility_summary.water_noise_reduction_percent,digits=1))%,
$(round(tlbf_feasibility_summary.iodine_noise_reduction_percent,digits=1))%,
and $(round(tlbf_feasibility_summary.vmi70_noise_reduction_percent,digits=1))%,
respectively. The sampled 10–90% edge width remained
$(tlbf_feasibility_summary.filtered_edge_width_pixels) pixel versus
$(tlbf_feasibility_summary.reference_edge_width_pixels) pixel unfiltered,
although deterministic difference images and RMSE show nonzero smoothing.

The common-pair VMI curve is smoother and substantially lower but remains
U-shaped, not monotonic. Lee noise matching improves the high-keV tail but is
not uniformly beneficial and does not make the full curve monotonic. It should
therefore be refined before a five-seed confirmation.

This is a one-realization feasibility result. Corrected fractional count
equivalents were used in the total-likelihood weights; the four-bin Cong
decomposition itself was unchanged. No `BasisSimulator.jl` source file was
modified.
"""

# ╔═╡ Cell order:
# ╠═171294a2-26bd-49e2-ac92-9df48ae5444f
# ╠═69358294-97f2-4782-94d7-c29c747c45f4
# ╠═9ae27110-5c47-442b-a98e-d137599570f2
# ╠═27f065e9-f97c-4392-a83f-e3638682152d
# ╠═a8fc42d0-ee5b-44ea-b4dc-8de368209e44
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─f2798d62-3509-4cc4-a24f-39ace8bb5a9e
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
# ╠═492bb299-678d-4e6f-8c21-1e9178cc2beb
# ╠═9f8d5cd4-147e-4359-95bc-cc096a53f0e7
# ╠═2ff539c9-a678-403c-b629-8068a332a0e9
# ╠═320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
# ╠═86c52e9e-7987-4504-93e6-128017f5e703
# ╟─551f84fe-d7b4-48f9-a475-0c63178a6ede
# ╟─59a5079b-a711-4f28-b3d6-665f0d91fb72
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╠═5b50d9e8-3d67-4f0c-a9e8-92b7021c6cc5
# ╟─a9d9212b-782c-4552-8d23-dcf74c052826
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╠═f9c0af7a-addd-4249-96fb-b9078765fbd1
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╠═dffbbf26-cb0d-45c2-b4ba-621e52b40108
# ╠═67547db5-68e4-4648-badc-610ec40fef9f
# ╟─11fd3c9c-392b-4ee4-adcb-e757021401f7
# ╟─41c4d6c2-69cc-453e-9ad5-35026e027a93
# ╟─a27649c0-e623-473e-bfbe-60fc8dd6d976
# ╠═4985581f-616d-4bb7-ab9b-967d7250b28b
# ╠═73371177-0498-4eda-897b-651c94f43e83
# ╟─b3e1d768-eb02-4c2a-9363-c84077fedc32
# ╠═50ed35bb-df61-4862-a99b-ea37380f30d8
# ╠═b86a9c50-cb10-44b2-af2e-06bdd50943b7
# ╠═30f13da5-f7e2-4f5f-9494-d264a4e4a355
# ╟─f9ca428e-9acf-494e-8f77-34502d68e0ad
# ╠═92692d46-1b57-4daa-b78d-c5aef609ed86
# ╠═ddfde8bb-ddff-44bb-8b70-b397725402cf
# ╟─2d447ccd-b408-42a6-8a1b-af770e3277e4
# ╠═1ff5e801-ef54-45ff-b0a8-e780e2e6cb63
# ╠═65a40efc-c9ff-41d4-8f42-46d6737119d5
# ╟─7707308d-e855-4111-9367-ef4ffc66da8b
# ╠═9161e7bf-eaa9-4d48-9d93-ab743ff3a0c2
# ╠═5cc1f9ba-4a5f-4a78-86dd-e0d844a09a28
# ╠═707aec65-d9d6-4462-a662-c669beb8525b
# ╠═067e491c-b305-46d0-abf9-bad0431f7b5e
# ╟─8a23843f-3414-4641-a4c0-8800ec52cd7c
# ╠═f3de45a6-4818-4ee1-ad56-c65797119dee
# ╠═c35105bb-7bcb-4905-8c82-c5d695e7c779
# ╠═f4f046cb-fd81-4683-9533-92d1e732dc24
# ╠═ec10ffc2-7ba2-4a05-9346-930a958a6660
# ╠═c2c7b7ef-296e-4005-a578-bd38ec818be3
# ╠═307532ad-9599-4e8d-ba1a-2681fcf2f110
# ╠═1ea2dcbe-8cd8-4b0c-85f1-d59bf5d37a78
# ╠═07f4d2b6-6c29-47f3-88dd-98fc915009bc
# ╠═4f210d68-19cd-4b67-adce-6d59498f1366
# ╟─d544993a-e959-48c7-b1da-2888edc7623a
# ╟─881c8fac-c00a-46c3-9bbf-802227d9c4d0
# ╟─c0fcdf63-513d-489e-b105-0454b18c3f83
# ╟─24db3af8-9e58-4499-8de0-5966c9fe0394
# ╟─e788e4ab-50f1-4bad-af31-f31ff877e9de
# ╟─a702126e-7e5a-47c5-a45e-b8038ae096c8
# ╠═71344579-fce7-46a1-b55d-1029c9c20229
# ╟─ab1319af-8619-4ae8-853a-b5e63cb134f3
# ╟─d95a694b-dac0-4b28-bac6-2dadc94fe355
# ╟─a6694310-43ec-4bf6-bf22-374e2fcf6b00
# ╠═12497ebd-c1b4-4789-836a-fc49c5a06762
# ╠═e6d42b22-8fe5-48a2-982e-0db460fd9797
# ╠═07ed190a-58e8-4d0a-8802-137ceabbb67e
# ╟─082a7479-95d7-4973-8b18-a1d639542b25
# ╟─d7266214-0e21-411f-90f1-226b7ef1a742
# ╠═712d508b-95b5-43f3-b305-d0259134cd74
# ╠═fd6ab3d2-88e1-4862-a574-62b57f265279
# ╟─4e9a9a3c-f9f2-41b8-b85b-155c0a6d1ff0
# ╟─6fabf137-a705-436c-bdf0-0697f5d5bd7b
# ╟─b1d0f9d7-0e3e-4b65-b587-d9efcfac5a97
