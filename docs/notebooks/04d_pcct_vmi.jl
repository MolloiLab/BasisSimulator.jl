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
# Cong Covariance-Weighted TNV Reconstruction

Siemens Naeotom Alpha photon-counting CT simulation (140 kVp / 174 mA,
four native energy windows, Gammex 472 phantom).

This is the focused continuation of `04c_pcct_vmi.jl`. It preserves the
validated **Cong-inspired K-channel profiled quasi-likelihood estimator** and
adds its full per-ray covariance to a joint, convex material reconstruction.
The previous exploratory RSKR and long audit chain is archived separately.
"""

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## First-principles objective

The solved generic K-channel Cong/profile decomposition is fixed. This
notebook now follows one primary path:

1. run one four-bin scan;
2. compute the unchanged Cong material sinograms;
3. retain the full per-ray \(2\times2\) Fisher/covariance matrix;
4. reconstruct a common-filter FBP baseline;
5. solve one covariance-weighted, basis-whitened total nuclear variation
   (TNV) reconstruction;
6. synthesize every VMI from that single material pair.

TNV is the mathematical foundation because it is a convex, data-consistent
joint reconstruction problem. Literal RSKR is not treated as a proximal map
or convergence-guaranteed optimizer; it is only an optional empirical
comparator after the TNV endpoint works.

There are no 30-seed or dose-sweep scans in the primary execution path.
The discrepancy radius is calibrated from detector statistics (and later a
small declared bootstrap), never tuned separately by VMI energy.
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
nchannel_slab_common_fbp = let
    nrows = sim_bins.geom.n_rows
    nview = size(sino_basis_nchannel_slab.sino_water,3)
    pass_mode = min(
        nview÷2,
        ceil(Int,π*recon_opts.matrix_size[1]/4),
    )
    angular_response = [
        let mode=min(j-1,nview-(j-1))
            mode ≤ pass_mode ? 1.0 :
            0.5*(1+cos(π*(mode-pass_mode)/(nview÷2-pass_mode)))
        end
        for j in 1:nview
    ]
    function angular_antialias(sino)
        spectrum = BS.FFTW.fft(Float64.(sino),3)
        Float32.(real.(BS.FFTW.ifft(
            spectrum .* reshape(angular_response,1,1,nview),3,
        )))
    end
    function common_fbp(sino_one_row)
        repeated = repeat(angular_antialias(sino_one_row),1,nrows,1)
        sino_gpu = to_gpu(Float32.(repeated))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu,sim_bins.geom,recon_opts.matrix_size;
            filter=BS.SoftFilter(),
        )
        try
            Float32.(Array(BS.reconstruct!(ws,sino_gpu,sim_bins.geom)))
        finally
            BS.release_backend!(ws)
        end
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
    (
        vol_water=W,vol_iodine=I,mu=mu,energies=energies,
        angular_response,pass_mode,
    )
end;

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
    nchannel_validation_energies=collect(40.0:5.0:140.0)
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

# ╔═╡ 4d6d8375-5805-4e1d-9c77-fcde574c42a1
md"""
## 12. Covariance-whitened TNV reconstruction

This is the sole primary denoising path. Let \(s_r=(A_r,C_r)^T\) be the
unchanged Cong estimate and \(F_r\) its retained full precision matrix.
The reconstruction solves

\[
\min_{x\in\mathcal B} \operatorname{TNV}(Sx)
\quad\text{subject to}\quad
\sum_r(Px_r-s_r)^T F_r(Px_r-s_r)\le\epsilon^2.
\]

The same matched distance-driven projector and its exact algebraic transpose
are used for both materials. The off-diagonal precision is never discarded.
All VMIs are synthesized afterward from one reconstructed material pair.
"""

# ╔═╡ 4a7e1704-9abc-4430-8146-5e8a1fbaf852
tnv_material_data = let
    s = sino_basis_nchannel_slab
    c = cong_ray_covariance
    (
        observed=(iodine=s.sino_iodine,water=s.sino_water),
        precision=c.precision,
        covariance=c.covariance,
        valid=c.valid,
        geometry=s.geom,
        initial=(
            iodine=nchannel_slab_common_fbp.vol_iodine,
            water=nchannel_slab_common_fbp.vol_water,
        ),
    )
end;

# ╔═╡ 83bece80-87f7-4e18-943f-13d0a7287548
begin
function pooled_covariance_whitener(C;floor_fraction=16eps(Float64))
    E=eigen(Symmetric(Matrix{Float64}(C)))
    λmax=maximum(E.values)
    λfloor=max(floor_fraction*λmax,eps(Float64))
    λ=max.(E.values,λfloor)
    S=E.vectors*Diagonal(inv.(sqrt.(λ)))*E.vectors'
    Sinv=E.vectors*Diagonal(sqrt.(λ))*E.vectors'
    (
        covariance=Symmetric(Matrix{Float64}(C)),
        whitening=S,inverse_whitening=Sinv,
        eigenvalues=λ,eigenvalue_floor=λfloor,
    )
end

function material_precision_sqrt(FAA,FAC,FCC,valid=trues(size(FAA)))
    T=promote_type(float(eltype(FAA)),float(eltype(FAC)),float(eltype(FCC)))
    LAA,LAC,LCC=zeros(T,size(FAA)),zeros(T,size(FAA)),zeros(T,size(FAA))
    for i in eachindex(FAA,FAC,FCC,valid)
        valid[i]||continue
        a,b,c=T(FAA[i]),T(FAC[i]),T(FCC[i])
        determinant=a*c-b*b
        isfinite(a)&&isfinite(b)&&isfinite(c)&&
            a≥0&&c≥0&&determinant≥0||continue
        s=sqrt(max(determinant,zero(T)))
        t=sqrt(max(a+c+2s,eps(T)))
        LAA[i],LAC[i],LCC[i]=(a+s)/t,b/t,(c+s)/t
    end
    (AA=LAA,AC=LAC,CC=LCC)
end

function material_matrix(x,M)
    size(x,ndims(x))==2||error("last dimension must contain two materials")
    y=similar(x,promote_type(eltype(x),eltype(M)))
    x1,x2=selectdim(x,ndims(x),1),selectdim(x,ndims(x),2)
    y1,y2=selectdim(y,ndims(y),1),selectdim(y,ndims(y),2)
    @. y1=M[1,1]*x1+M[1,2]*x2
    @. y2=M[2,1]*x1+M[2,2]*x2
    y
end

function tnv_gradient(x,S)
    z=material_matrix(x,S)
    nx,ny,_=size(z)
    g=similar(z,eltype(z),nx,ny,2,2)
    fill!(g,zero(eltype(g)))
    @views begin
        g[1:nx-1,:,:,1].=z[2:nx,:,:].-z[1:nx-1,:,:]
        g[:,1:ny-1,:,2].=z[:,2:ny,:].-z[:,1:ny-1,:]
    end
    g
end

function tnv_gradient_adjoint(p,S)
    nx,ny=size(p,1),size(p,2)
    T=promote_type(eltype(p),eltype(S))
    q=similar(p,T,nx,ny,2)
    fill!(q,zero(T))
    @views for m in 1:2
        px,py=p[:,:,m,1],p[:,:,m,2]
        q[1,:,m].-=px[1,:]
        nx>2&&(q[2:nx-1,:,m].+=px[1:nx-2,:].-px[2:nx-1,:])
        nx>1&&(q[nx,:,m].+=px[nx-1,:])
        q[:,1,m].-=py[:,1]
        ny>2&&(q[:,2:ny-1,m].+=py[:,1:ny-2].-py[:,2:ny-1])
        ny>1&&(q[:,ny,m].+=py[:,ny-1])
    end
    material_matrix(q,transpose(S))
end

function tnv_value(x,S)
    g=tnv_gradient(x,S)
    sum(
        sqrt(
            sum(abs2,@view(g[i,j,:,:]))+
            2abs(det(Matrix(@view(g[i,j,:,:]))))
        )
        for i in axes(g,1),j in axes(g,2)
    )
end

@inline function project_spectral_ball_2x2(
    m11::T,m12::T,m21::T,m22::T,radius::T,
) where T
    a=m11*m11+m21*m21
    b=m11*m12+m21*m22
    c=m12*m12+m22*m22
    disc=sqrt(max((a-c)*(a-c)+T(4)*b*b,zero(T)))
    λ1=max((a+c+disc)/T(2),zero(T))
    λ2=max((a+c-disc)/T(2),zero(T))
    s1=λ1>radius*radius ? radius/sqrt(λ1) : one(T)
    s2=λ2>radius*radius ? radius/sqrt(λ2) : one(T)
    if abs(λ1-λ2)≤T(16)*eps(T)*max(λ1,one(T))
        h11,h12,h22=s1,zero(T),s1
    else
        α=(s1-s2)/(λ1-λ2)
        β=s2-α*λ2
        h11,h12,h22=α*a+β,α*b,α*c+β
    end
    (
        m11*h11+m12*h12,
        m11*h12+m12*h22,
        m21*h11+m22*h12,
        m21*h12+m22*h22,
    )
end

function project_tnv_dual!(p,radius=1.0)
    n=size(p,1)*size(p,2)
    r=eltype(p)(radius)
    BS.AK.foreachindex(@view(p[:,:,1,1])) do idx
        m11,m21,m12,m22=p[idx],p[idx+n],p[idx+2n],p[idx+3n]
        q11,q12,q21,q22=project_spectral_ball_2x2(
            m11,m12,m21,m22,r,
        )
        p[idx],p[idx+n],p[idx+2n],p[idx+3n]=q11,q21,q12,q22
    end
    p
end

function ray_whiten(y,L)
    z=similar(y)
    y1,y2=selectdim(y,ndims(y),1),selectdim(y,ndims(y),2)
    z1,z2=selectdim(z,ndims(z),1),selectdim(z,ndims(z),2)
    @. z1=L.AA*y1+L.AC*y2
    @. z2=L.AC*y1+L.CC*y2
    z
end

function pair_forward(x,forward)
    y1,y2=forward(@view(x[:,:,1])),forward(@view(x[:,:,2]))
    cat(y1,y2;dims=ndims(y1)+1)
end

function pair_adjoint(y,adjoint)
    n=ndims(y)
    x1,x2=adjoint(selectdim(y,n,1)),adjoint(selectdim(y,n,2))
    cat(x1,x2;dims=ndims(x1)+1)
end

function tnv_operator_norm(initial,forward,adjoint,L,S;iterations=24)
    x=float(eltype(initial)).(initial)
    norm(x)>0||fill!(x,one(eltype(x)))
    x./=norm(x)
    λ=0.0
    for _ in 1:iterations
        y=ray_whiten(pair_forward(x,forward),L)
        g=tnv_gradient(x,S)
        z=pair_adjoint(ray_whiten(y,L),adjoint)+tnv_gradient_adjoint(g,S)
        λ=norm(z)
        x.=z./λ
    end
    sqrt(λ)
end

function covariance_tnv_pdhg(
    observed,initial,forward,adjoint,L,S;
    epsilon,tnv_weight=1.0,bounds=((-Inf,Inf),(-Inf,Inf)),
    strict_convexity=1e-8,operator_norm=nothing,
    max_iterations=1000,tolerance=1e-5,check_every=10,
)
    T=float(promote_type(
        eltype(initial),eltype(observed),eltype(L.AA),eltype(S),
    ))
    Knorm=isnothing(operator_norm) ?
        tnv_operator_norm(initial,forward,adjoint,L,S) : operator_norm
    Knorm=T(Knorm)
    τ=T(0.99)/Knorm
    σ=T(0.99)/Knorm
    η=T(strict_convexity)
    ε=T(epsilon)
    tol=T(tolerance)
    typed_bounds=(
        (T(bounds[1][1]),T(bounds[1][2])),
        (T(bounds[2][1]),T(bounds[2][2])),
    )
    x=T.(initial)
    xbar,xold=copy(x),similar(x)
    d=ray_whiten(T.(observed),L)
    y=similar(d)
    fill!(y,zero(T))
    p=similar(x,T,size(x,1),size(x,2),2,2)
    fill!(p,zero(T))
    history=NamedTuple[]
    converged=false
    for iteration in 1:max_iterations
        q=y.+σ.*ray_whiten(pair_forward(xbar,forward),L)
        v=q./σ
        residual=v.-d
        nr=norm(residual)
        projection=nr>ε ? d.+ε/nr.*residual : v
        y.=q.-σ.*projection
        p.+=σ.*tnv_gradient(xbar,S)
        project_tnv_dual!(p,tnv_weight)
        copyto!(xold,x)
        data_gradient=pair_adjoint(ray_whiten(y,L),adjoint)
        x.=(x.-τ.*(data_gradient.+tnv_gradient_adjoint(p,S)))./
            (one(T)+τ*η)
        for m in 1:2
            clamp!(@view(x[:,:,m]),typed_bounds[m]...)
        end
        @. xbar=2x-xold
        if iteration==1||iteration%check_every==0
            discrepancy=norm(ray_whiten(pair_forward(x,forward).-observed,L))
            relative_change=norm(x.-xold)/max(norm(xold),one(T))
            record=(iteration,discrepancy,relative_change,
                feasible=discrepancy≤ε*(one(T)+tol))
            push!(history,record)
            if record.feasible&&relative_change≤tol
                converged=true
                break
            end
        end
    end
    (
        image=x,history,converged,operator_norm=Knorm,
        tau=τ,sigma=σ,epsilon=ε,
        data_dual=y,tnv_dual=p,
    )
end
end

# ╔═╡ 829630be-523b-464e-a250-e6b617be8ce1
tnv_unit_audit = let
    C=[4.0 -1.5;-1.5 2.0]
    W=pooled_covariance_whitener(C)
    whitening_error=norm(W.whitening*C*W.whitening'-I)

    rng=MersenneTwister(544)
    x=randn(rng,7,6,2)
    p=randn(rng,7,6,2,2)
    adjoint_error=abs(
        dot(tnv_gradient(x,W.whitening),p)-
        dot(x,tnv_gradient_adjoint(p,W.whitening))
    )
    projected=project_tnv_dual!(copy(p),0.7)
    svd_projection=similar(projected)
    for i in axes(p,1),j in axes(p,2)
        F=svd(Matrix(@view(p[i,j,:,:])))
        @views svd_projection[i,j,:,:].=
            F.U*Diagonal(min.(F.S,0.7))*F.Vt
    end
    dual_projection_error=maximum(abs.(projected.-svd_projection))

    T=[1000.0 0.2;-3.0 0.01]
    J=[0.7 -1.1;2.3 0.4]
    Sp=pooled_covariance_whitener(T*C*T').whitening
    basis_error=abs(
        sum(svdvals(W.whitening*J))-sum(svdvals(Sp*T*J))
    )

    truth=zeros(12,11,2)
    truth[4:9,4:8,1].=1.2
    truth[5:8,5:7,2].=0.7
    noisy=truth.+0.03randn(rng,size(truth))
    ray_shape=size(noisy)[1:2]
    L=(AA=ones(ray_shape),AC=zeros(ray_shape),CC=ones(ray_shape))
    identity_forward(v)=copy(v)
    identity_adjoint(v)=copy(v)
    result=covariance_tnv_pdhg(
        noisy,noisy,identity_forward,identity_adjoint,L,I(2);
        epsilon=0.03sqrt(length(noisy))*1.1,
        tnv_weight=0.08,bounds=((-2.0,2.0),(-2.0,2.0)),
        max_iterations=600,tolerance=2e-4,check_every=5,
    )
    (
        whitening_error,adjoint_error,dual_projection_error,basis_error,
        pdhg_converged=result.converged,
        pdhg_feasible=last(result.history).feasible,
        error_before=norm(noisy-truth),
        error_after=norm(result.image-truth),
        step_condition=result.tau*result.sigma*result.operator_norm^2,
        pass=whitening_error<1e-12&&adjoint_error<1e-11&&
            dual_projection_error<1e-12&&
            basis_error<1e-10&&result.converged&&
            last(result.history).feasible&&
            norm(result.image-truth)<norm(noisy-truth)&&
            result.tau*result.sigma*result.operator_norm^2<1,
    )
end

# ╔═╡ b07ec202-ddcb-465a-8f0d-18e53c2ff902
tnv_backend_audit = let
    rng=MersenneTwister(546)
    p=randn(rng,Float32,5,4,2,2)
    reference=project_tnv_dual!(copy(p),0.7f0)
    device=to_gpu(p)
    result=nothing
    truth_gpu=nothing
    noisy_gpu=nothing
    L_gpu=nothing
    try
        project_tnv_dual!(device,0.7f0)
        observed=Array(device)
        truth=zeros(Float32,12,11,2)
        truth[4:9,4:8,1].=1.2f0
        truth[5:8,5:7,2].=0.7f0
        noisy=truth.+0.03f0.*randn(rng,Float32,size(truth))
        truth_gpu=to_gpu(truth)
        noisy_gpu=to_gpu(noisy)
        ray_shape=size(noisy)[1:2]
        L_gpu=(
            AA=to_gpu(ones(Float32,ray_shape)),
            AC=to_gpu(zeros(Float32,ray_shape)),
            CC=to_gpu(ones(Float32,ray_shape)),
        )
        identity_forward(v)=copy(v)
        identity_adjoint(v)=copy(v)
        result=covariance_tnv_pdhg(
            noisy_gpu,noisy_gpu,identity_forward,identity_adjoint,
            L_gpu,Float32[1 0;0 1];
            epsilon=0.03f0*sqrt(length(noisy))*1.1f0,
            tnv_weight=0.08f0,
            bounds=((-2.0f0,2.0f0),(-2.0f0,2.0f0)),
            max_iterations=600,tolerance=2f-4,check_every=5,
        )
        recovered=Array(result.image)
        (
            backend=GPU_BACKEND.name,
            maximum_error=maximum(abs.(observed.-reference)),
            pdhg_converged=result.converged,
            pdhg_feasible=last(result.history).feasible,
            error_before=norm(noisy-truth),
            error_after=norm(recovered-truth),
            pass=isapprox(observed,reference;rtol=2f-5,atol=2f-6)&&
                result.converged&&last(result.history).feasible&&
                norm(recovered-truth)<norm(noisy-truth),
        )
    finally
        BS.release_backend!((
            device,result,truth_gpu,noisy_gpu,L_gpu,
        ))
    end
end

# ╔═╡ d9c149c8-135a-42c9-bb95-213b0aec3b32
begin
function central_row_geometry(geom,image_size)
    voxel_xy=geom.fov[1]/image_size[1]
    BS.CTGeometry(
        geom.SAD,geom.SDD,geom.n_angles,1,geom.n_cols,
        geom.pixel_size,geom.pixel_row_size,
        geom.angles,geom.source_positions,geom.detector_centers,
        geom.detector_u,geom.detector_v,
        (geom.fov[1],geom.fov[2],voxel_xy),
        geom.pitch,geom.table_feed,geom.detector_shape,
    )
end

function central_row_projector(geom,image_size)
    geom2d=central_row_geometry(geom,image_size)
    extent=geom2d.fov
    function forward(image)
        size(image)==image_size||
            throw(DimensionMismatch("projector image size mismatch"))
        volume=reshape(image,image_size...,1)
        BS.dd_forward_project(volume,geom2d;volume_extent=extent)
    end
    function adjoint(sinogram)
        size(sinogram)==(geom2d.n_cols,1,geom2d.n_angles)||
            throw(DimensionMismatch("projector sinogram size mismatch"))
        volume=similar(
            sinogram,eltype(sinogram),(image_size...,1),
        )
        fill!(volume,zero(eltype(volume)))
        BS.dd_backproject!(
            volume,sinogram,geom2d;volume_extent=extent,
        )
        dropdims(volume;dims=3)
    end
    (forward=forward,adjoint=adjoint,geometry=geom2d,extent=extent)
end
end

# ╔═╡ 739a1ff6-dd9b-4edf-8653-73a2704bfb59
begin
function backend_synchronize()
    backend_module=parentmodule(AT)
    isdefined(backend_module,:synchronize) &&
        getfield(backend_module,:synchronize)()
    nothing
end

function central_row_reusable_projector(geom,image_size,prototype)
    geom2d=central_row_geometry(geom,image_size)
    extent=geom2d.fov
    T=eltype(prototype)
    sinogram=similar(
        prototype,T,geom2d.n_cols,1,geom2d.n_angles,
    )
    volume=similar(prototype,T,image_size...,1)
    ws=(
        source_positions=to_gpu(T.(geom2d.source_positions)),
        detector_centers=to_gpu(T.(geom2d.detector_centers)),
        detector_u=to_gpu(T.(geom2d.detector_u)),
        detector_v=to_gpu(T.(geom2d.detector_v)),
    )
    function forward!(image)
        BS.dd_forward_project!(
            sinogram,reshape(image,image_size...,1),geom2d;
            volume_extent=extent,
            ws_source_positions=ws.source_positions,
            ws_detector_centers=ws.detector_centers,
            ws_detector_u=ws.detector_u,
            ws_detector_v=ws.detector_v,
        )
    end
    function adjoint!(input)
        BS.dd_backproject!(
            volume,input,geom2d;
            volume_extent=extent,
            ws_source_positions=ws.source_positions,
            ws_detector_centers=ws.detector_centers,
            ws_detector_u=ws.detector_u,
            ws_detector_v=ws.detector_v,
        )
        reshape(volume,image_size)
    end
    (
        forward! = forward!,adjoint! = adjoint!,
        geometry=geom2d,extent,sinogram,volume,ws,
    )
end
end

# ╔═╡ 813d96ad-4b83-4db9-95db-67336bf4b43e
tnv_operator_profile = let
    records=NamedTuple[]
    for resolution in (128,512)
        image_size=(resolution,resolution)
        allocating=central_row_projector(
            sino_basis_nchannel_slab.geom,image_size,
        )
        image=to_gpu(zeros(Float32,image_size))
        reusable=central_row_reusable_projector(
            sino_basis_nchannel_slab.geom,image_size,image,
        )

        allocating.forward(image)
        backend_synchronize()
        reusable.forward!(image)
        backend_synchronize()

        forward_allocations=@allocated allocating.forward(image)
        backend_synchronize()
        allocating_forward_s=@elapsed begin
            allocating.forward(image)
            backend_synchronize()
        end
        reusable_forward_allocations=@allocated reusable.forward!(image)
        backend_synchronize()
        reusable_forward_s=@elapsed begin
            reusable.forward!(image)
            backend_synchronize()
        end

        input_sinogram=copy(reusable.sinogram)
        allocating.adjoint(input_sinogram)
        backend_synchronize()
        reusable.adjoint!(input_sinogram)
        backend_synchronize()

        adjoint_allocations=@allocated allocating.adjoint(input_sinogram)
        backend_synchronize()
        allocating_adjoint_s=@elapsed begin
            allocating.adjoint(input_sinogram)
            backend_synchronize()
        end
        reusable_adjoint_allocations=@allocated reusable.adjoint!(input_sinogram)
        backend_synchronize()
        reusable_adjoint_s=@elapsed begin
            reusable.adjoint!(input_sinogram)
            backend_synchronize()
        end

        push!(records,(;
            resolution,
            allocating_forward_s,reusable_forward_s,
            allocating_adjoint_s,reusable_adjoint_s,
            forward_allocations,reusable_forward_allocations,
            adjoint_allocations,reusable_adjoint_allocations,
            allocating_geometry_uploads_per_call=4,
            reusable_geometry_uploads_per_call=0,
            allocating_output_buffers_per_call=1,
            reusable_output_buffers_per_call=0,
            arrays_remain_on_device=true,
            synchronization=:timing_boundary_only,
        ))
        BS.release_backend!((
            image,input_sinogram,reusable.sinogram,reusable.volume,
            reusable.ws,
        ))
    end
    records
end

# ╔═╡ 47384dd6-7b4e-403b-b4b5-1c6bc34a77e2
tnv_gpu_fft_capability = let
    probe=to_gpu(reshape(Float32.(1:64),8,8))
    try
        transformed=BS.FFTW.fft(probe)
        (
            supported=true,
            input_type=typeof(probe),
            output_type=typeof(transformed),
            remains_on_device=
                parentmodule(typeof(transformed))===
                parentmodule(typeof(probe)),
        )
    catch error
        (
            supported=false,
            input_type=typeof(probe),
            error=string(typeof(error),": ",error),
            remains_on_device=false,
        )
    finally
        BS.release_backend!(probe)
    end
end

# ╔═╡ 6c7fdf0f-bc9f-4cf7-96ba-eb0e1ff94c60
md"""
### Projector-wrapper profile

`tnv_operator_profile` times synchronized forward and exact-adjoint calls for
the original allocating notebook wrapper and a notebook-local reusable wrapper.
The reusable wrapper retains geometry arrays and output buffers on the selected
GPU backend; synchronization occurs only at explicit timing boundaries.
"""

# ╔═╡ 9c675f52-0046-41be-a203-d603cffcea2d
tnv_projector_adjoint_audit = let
    toy_scanner=BS.Scanner(
        source_to_isocenter=540.0,source_to_detector=1080.0,
        detector_rows=1,detector_cols=24,
        detector_row_size=0.8,detector_col_size=0.8,
        detector_shape=:arc,
    )
    toy_geom=BS.CTGeometry(
        toy_scanner;n_angles=9,n_rows=1,n_cols=24,
        fov_cm=8.0,z_cm=8.0/9,
    )
    bridge=central_row_projector(toy_geom,(9,9))
    rng=MersenneTwister(545)
    x=randn(rng,9,9)
    y=randn(rng,24,1,9)
    Ax=bridge.forward(x)
    Aty=bridge.adjoint(y)
    lhs=dot(Ax,y)
    rhs=dot(x,Aty)
    x_gpu=to_gpu(Float32.(x))
    y_gpu=to_gpu(Float32.(y))
    Ax_gpu=nothing
    Aty_gpu=nothing
    try
        Ax_gpu=bridge.forward(x_gpu)
        Aty_gpu=bridge.adjoint(y_gpu)
        lhs_gpu=dot(Ax_gpu,y_gpu)
        rhs_gpu=dot(x_gpu,Aty_gpu)
        (
            lhs,rhs,
            absolute_error=abs(lhs-rhs),
            relative_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),eps()),
            backend=GPU_BACKEND.name,
            backend_relative_error=
                abs(lhs_gpu-rhs_gpu)/max(abs(lhs_gpu),abs(rhs_gpu),eps(Float32)),
            output_size=size(Ax),
            pass=isapprox(lhs,rhs;rtol=2e-11,atol=2e-11)&&
                isapprox(lhs_gpu,rhs_gpu;rtol=2f-5,atol=2f-5),
        )
    finally
        BS.release_backend!((x_gpu,y_gpu,Ax_gpu,Aty_gpu))
    end
end

# ╔═╡ dc5da7d1-3dcf-4942-b6f5-ea632aee1472
tnv_explicit_oracle_problem = let
    nx,ny=16,16
    nviews,ndet=24,20
    nvox=nx*ny
    pixel_size=2/nx
    xs=collect(range(-1+pixel_size/2,1-pixel_size/2;length=nx))
    ys=collect(range(-1+pixel_size/2,1-pixel_size/2;length=ny))
    angles=collect(range(0,π;length=nviews+1))[1:end-1]
    offsets=collect(range(-1.25,1.25;length=ndet))
    ray_width=0.75pixel_size
    P=zeros(Float64,nviews*ndet,nvox)
    ray=0
    for θ in angles,offset in offsets
        ray+=1
        c,s=cos(θ),sin(θ)
        for j in 1:ny,i in 1:nx
            distance=xs[i]*c+ys[j]*s-offset
            P[ray,i+(j-1)*nx]=
                pixel_size*exp(-0.5*(distance/ray_width)^2)
        end
    end

    nrays=size(P,1)
    ray_angle(r)=angles[cld(r,ndet)]
    LAA=[1.0+0.15sin(ray_angle(r))^2 for r in 1:nrays]
    LCC=[0.80+0.10cos(ray_angle(r))^2 for r in 1:nrays]
    LAC=[0.08cos(2ray_angle(r)) for r in 1:nrays]
    L=(AA=LAA,AC=LAC,CC=LCC)
    A=[
        Diagonal(LAA)*P Diagonal(LAC)*P
        Diagonal(LAC)*P Diagonal(LCC)*P
    ]

    Cmaterial=[3.0 -0.7;-0.7 1.4]
    S=pooled_covariance_whitener(Cmaterial).whitening
    xtrue=zeros(Float64,nx,ny,2)
    for j in 1:ny,i in 1:nx
        r2=xs[i]^2+ys[j]^2
        xtrue[i,j,1]=r2≤0.48^2 ? 0.65 : 0.05
        xtrue[i,j,2]=r2≤0.70^2 ? 0.75 : 0.12
        (xs[i]-0.28)^2+(ys[j]+0.18)^2≤0.16^2&&
            (xtrue[i,j,1]+=0.30)
    end

    G=zeros(Float64,4nvox,2nvox)
    basis=zeros(Float64,nx,ny,2)
    for column in 1:2nvox
        fill!(basis,0)
        basis[column]=1
        G[:,column].=vec(tnv_gradient(basis,S))
    end
    xtrue_vec=vec(xtrue)
    rng=MersenneTwister(0x16a32)
    noise=randn(rng,2nrays)
    noise.*=0.025norm(A*xtrue_vec)/norm(noise)
    data=A*xtrue_vec+noise
    epsilon=1.05norm(noise)

    observed=zeros(Float64,nrays,2)
    for r in 1:nrays
        determinant=LAA[r]*LCC[r]-LAC[r]^2
        d1,d2=data[r],data[r+nrays]
        observed[r,1]=(LCC[r]*d1-LAC[r]*d2)/determinant
        observed[r,2]=(-LAC[r]*d1+LAA[r]*d2)/determinant
    end
    forward=x->P*vec(x)
    adjoint=y->reshape(transpose(P)*vec(y),nx,ny)
    (
        nx,ny,nvox,nrays,P,A,G,L,S,xtrue,xtrue_vec,
        observed,data,epsilon,
        bounds=((-0.25,1.25),(-0.25,1.25)),
        strict_convexity=1e-3,
        initial=zeros(Float64,nx,ny,2),
        forward,adjoint,
    )
end;

# ╔═╡ 595b5b3b-d8e6-4001-9f63-0365728ca667
begin
function oracle_nuclear_prox(v,nx,ny,threshold)
    V=reshape(v,nx,ny,2,2)
    U=similar(V)
    for j in 1:ny,i in 1:nx
        F=svd(Matrix(@view(V[i,j,:,:])))
        @views U[i,j,:,:].=
            F.U*Diagonal(max.(F.S.-threshold,0))*F.Vt
    end
    vec(U)
end

function explicit_tnv_admm_reference(problem;
    rho=2.0,max_iterations=6000,tolerance=2e-8,check_every=25,
)
    A,G=problem.A,problem.G
    d,ε=problem.data,problem.epsilon
    η=problem.strict_convexity
    lo=vcat(
        fill(problem.bounds[1][1],problem.nvox),
        fill(problem.bounds[2][1],problem.nvox),
    )
    hi=vcat(
        fill(problem.bounds[1][2],problem.nvox),
        fill(problem.bounds[2][2],problem.nvox),
    )
    H=transpose(A)*A+transpose(G)*G+I
    EH=eigen(Symmetric(Matrix(H)))
    Hinv=EH.vectors*Diagonal(inv.(EH.values))*transpose(EH.vectors)
    x=vec(copy(problem.initial))
    u=G*x
    residual=A*x-d
    v=norm(residual)≤ε ? residual : ε/norm(residual).*residual
    w=clamp.(x,lo,hi)
    du=zeros(length(u));dv=zeros(length(v));dw=zeros(length(w))
    history=NamedTuple[]
    converged=false
    for iteration in 1:max_iterations
        rhs=transpose(G)*(u-du)+transpose(A)*(d+v-dv)+(w-dw)
        x=Hinv*rhs
        uold,vold,wold=copy(u),copy(v),copy(w)
        u=oracle_nuclear_prox(
            G*x+du,problem.nx,problem.ny,1/rho,
        )
        q=A*x-d+dv
        nq=norm(q)
        v=nq≤ε ? q : ε/nq.*q
        w=clamp.(rho.*(x+dw)./(rho+η),lo,hi)
        du.+=G*x-u
        dv.+=A*x-d-v
        dw.+=x-w
        if iteration==1||iteration%check_every==0
            primal=max(norm(G*x-u),norm(A*x-d-v),norm(x-w))
            dual=rho*norm(
                transpose(G)*(u-uold)+
                transpose(A)*(v-vold)+(w-wold)
            )
            push!(history,(;iteration,primal,dual))
            if max(primal,dual)≤tolerance
                converged=true
                break
            end
        end
    end
    (
        image=reshape(w,problem.nx,problem.ny,2),
        tnv_dual=rho.*du,
        data_dual=rho.*dv,
        bound_dual=rho.*dw,
        history,converged,
    )
end

function oracle_solution_metrics(image,tnv_dual,data_dual,problem)
    x=vec(image)
    residual=problem.A*x-problem.data
    p=vec(tnv_dual)
    y=vec(data_dual)
    lo=vcat(
        fill(problem.bounds[1][1],problem.nvox),
        fill(problem.bounds[2][1],problem.nvox),
    )
    hi=vcat(
        fill(problem.bounds[1][2],problem.nvox),
        fill(problem.bounds[2][2],problem.nvox),
    )
    objective=tnv_value(image,problem.S)+
        problem.strict_convexity/2*dot(x,x)
    feasibility=max(
        max(norm(residual)-problem.epsilon,0.0),
        maximum(max.(lo.-x,0.0)),
        maximum(max.(x.-hi,0.0)),
    )
    gradient=transpose(problem.A)*y+transpose(problem.G)*p+
        problem.strict_convexity.*x
    kkt_residual=norm(
        x-clamp.(x-gradient,lo,hi),
    )/max(norm(x),1.0)
    P=reshape(p,problem.nx,problem.ny,2,2)
    dual_tnv_violation=maximum(
        max(maximum(svdvals(Matrix(@view(P[i,j,:,:]))))-1,0.0)
        for j in 1:problem.ny,i in 1:problem.nx
    )
    qdual=-(transpose(problem.G)*p+transpose(problem.A)*y)
    conjugate_x=clamp.(
        qdual./problem.strict_convexity,lo,hi,
    )
    fstar=dot(qdual,conjugate_x)-
        problem.strict_convexity/2*dot(conjugate_x,conjugate_x)
    dual_objective=-dot(y,problem.data)-
        problem.epsilon*norm(y)-fstar
    gap=objective-dual_objective
    (
        objective,feasibility,kkt_residual,
        dual_tnv_violation,dual_objective,
        primal_dual_gap=gap,
        relative_gap=gap/max(abs(objective),1.0),
        data_residual=norm(residual),
    )
end
end

# ╔═╡ 7429a1a3-785d-423d-93c4-2bb0ee3cf1a7
tnv_small_oracle = let
    problem=tnv_explicit_oracle_problem
    reference=explicit_tnv_admm_reference(problem)
    pdhg=covariance_tnv_pdhg(
        problem.observed,problem.initial,
        problem.forward,problem.adjoint,
        problem.L,problem.S;
        epsilon=problem.epsilon,
        tnv_weight=1.0,
        bounds=problem.bounds,
        strict_convexity=problem.strict_convexity,
        max_iterations=40_000,
        tolerance=1e-10,
        check_every=25,
    )
    reference_metrics=oracle_solution_metrics(
        reference.image,reference.tnv_dual,
        reference.data_dual,problem,
    )
    pdhg_metrics=oracle_solution_metrics(
        pdhg.image,pdhg.tnv_dual,pdhg.data_dual,problem,
    )
    image_relative_error=norm(pdhg.image-reference.image)/
        max(norm(reference.image),eps())
    objective_relative_error=abs(
        pdhg_metrics.objective-reference_metrics.objective
    )/max(abs(reference_metrics.objective),1.0)
    (
        grid=(problem.nx,problem.ny),
        variables=2problem.nvox,
        data_rows=2problem.nrays,
        reference_converged=reference.converged,
        pdhg_converged=pdhg.converged,
        reference=reference_metrics,
        pdhg=pdhg_metrics,
        image_relative_error,
        objective_relative_error,
        pass=(
            max(reference_metrics.feasibility,pdhg_metrics.feasibility)<1e-6&&
            max(reference_metrics.kkt_residual,pdhg_metrics.kkt_residual)<5e-5&&
            max(
                abs(reference_metrics.relative_gap),
                abs(pdhg_metrics.relative_gap),
            )<5e-4&&
            objective_relative_error<5e-4
        ),
    )
end

# ╔═╡ 3d3ece92-6d20-49b4-8539-4a2ff2200455
md"""
### Explicit-matrix constrained-TNV oracle

`tnv_small_oracle` compares the notebook PDHG implementation against an
independent dense-matrix ADMM reference on a 16², two-material tomography
problem. The reference uses a dense linear solve, SVD nuclear proximal maps,
an exact Euclidean data-ball projection, and explicit bound/quadratic proximal
steps. Pass/fail requires agreement in feasibility, objective, KKT residual,
and primal-dual gap—not image appearance or iteration count.
"""

# ╔═╡ df46022b-033d-45df-8bca-478517904e86
tnv_ray_whitening = let
    F = cong_ray_covariance.precision
    material_precision_sqrt(F.AA,F.AC,F.CC,cong_ray_covariance.valid)
end;

# ╔═╡ 296b915c-edbb-4b76-bf42-d42061109608
tnv_whitening = let
    C = cong_ray_covariance.covariance
    valid = cong_ray_covariance.valid .&
        isfinite.(C.AA) .& isfinite.(C.AC) .& isfinite.(C.CC)
    idx = findall(valid)
    isempty(idx) && error("no valid Cong covariance samples")
    Cbar = [
        mean(Float64.(C.AA[idx])) mean(Float64.(C.AC[idx]))
        mean(Float64.(C.AC[idx])) mean(Float64.(C.CC[idx]))
    ]
    merge(
        pooled_covariance_whitener(Cbar),
        (valid_rays=length(idx),),
    )
end;

# ╔═╡ 0a8d4979-5665-4b4d-84e9-b5da72183134
tnv_basis_invariance_audit = let
    C = Matrix(tnv_whitening.covariance)
    S = tnv_whitening.whitening
    J = [0.7 -1.1; 2.3 0.4]
    T = [1000.0 0.2; -3.0 0.01]
    Cp = Symmetric(T*C*T')
    Sp = pooled_covariance_whitener(Cp).whitening
    nuclear(M) = sum(svdvals(M))
    original = nuclear(S*J)
    transformed = nuclear(Sp*T*J)
    (
        original,
        transformed,
        relative_error=abs(transformed-original)/max(abs(original),eps()),
        pass=isapprox(transformed,original;rtol=1e-10,atol=1e-12),
    )
end;

# ╔═╡ a8a740e0-9e33-447a-b34c-dfa0139662da
tnv_controls = let
    active = count(cong_ray_covariance.valid)
    dof = 2active
    alpha = 0.001
    normal_quantile = 3.090232306167813
    epsilon_squared = dof + sqrt(2dof)*normal_quantile
    (
        confidence=1-alpha,
        degrees_of_freedom=dof,
        epsilon_squared,
        image_bounds=(
            iodine_g_cm3=(-0.02,0.10),
            water_g_cm3=(-0.20,2.00),
        ),
        operator_norm_method=:power_iteration,
        step_condition=:tau_sigma_normK_squared_lt_one,
        stopping=(:primal_dual_gap,:data_feasibility),
        strict_convexity=1e-8,
    )
end;

# ╔═╡ 223af75d-05c7-431c-b598-70bef1f1c6a0
bootstrap_controls = (
    seeds=32,
    sampled_rays=1024,
    quantile=0.95,
    base_seed=0x04d2026,
    model=:fitted_corrected_counts_with_mc_fano_and_correlation,
);

# ╔═╡ f86d4398-f245-4899-adf9-30ce3dd405e3
tnv_bootstrap = let
    controls=bootstrap_controls
    valid_linear=findall(vec(cong_ray_covariance.valid))
    nsample=min(controls.sampled_rays,length(valid_linear))
    positions=round.(Int,range(1,length(valid_linear);length=nsample))
    selected=valid_linear[positions]
    K=length(nchannel_basis.I0)
    scale=Float64(nchannel_slab_counts.nrows)
    Φ=scale.*Float64.(nchannel_basis.Φ)
    I0=scale.*Float64.(nchannel_basis.I0)
    μI=Float64.(nchannel_basis.μρ_I)
    μW=Float64.(nchannel_basis.μρ_W)
    A0=Float64.(vec(sino_basis_nchannel_slab.sino_iodine)[selected])
    C0=Float64.(vec(sino_basis_nchannel_slab.sino_water)[selected])
    λ=zeros(Float64,nsample,K)
    for i in 1:nsample
        λ[i,:].=nchannel_forward(A0[i],C0[i],Φ,μI,μW).λ
    end

    moments=sim_bins.mc_count_moments
    corr=Symmetric(
        (moments.correlation+transpose(moments.correlation))./2,
    )
    E=eigen(corr)
    corr_psd=E.vectors*Diagonal(max.(E.values,1e-6))*E.vectors'
    d=sqrt.(max.(
        [corr_psd[i,i] for i in axes(corr_psd,1)],
        1e-12,
    ))
    corr_psd./=d*transpose(d)
    Ecorr=eigen(Symmetric(corr_psd+1e-7I))
    Lcorr=Ecorr.vectors*Diagonal(sqrt.(max.(Ecorr.values,1e-7)))
    fano=reshape(Float64.(moments.fano),1,K)

    F=sino_basis_nchannel_slab.fisher
    FAA=Float64.(vec(F.AA)[selected])
    FAC=Float64.(vec(F.AC)[selected])
    FCC=Float64.(vec(F.CC)[selected])

    Φ_gpu=to_gpu(Float32.(Φ))
    μI_gpu=to_gpu(Float32.(μI))
    μW_gpu=to_gpu(Float32.(μW))
    I0_gpu=to_gpu(Float32.(I0))
    μI_eff_gpu=to_gpu(nchannel_basis.μI_eff)
    μW_eff_gpu=to_gpu(nchannel_basis.μW_eff)
    template=to_gpu(zeros(Float32,nsample,1,1))
    q_per_ray=Float64[]
    retained_fraction=Float64[]
    try
        for seed_index in 1:controls.seeds
            rng=MersenneTwister(Int(controls.base_seed)+seed_index)
            correlated=randn(rng,nsample,K)*transpose(Lcorr)
            y=max.(
                λ.+correlated.*sqrt.(max.(λ,1e-6).*fano),
                1.0,
            )
            h=-log.(y./reshape(I0,1,K))
            hs=ntuple(
                k->to_gpu(reshape(Float32.(h[:,k]),nsample,1,1)),
                K,
            )
            I_gpu,W_gpu=similar(template),similar(template)
            flag_gpu=similar(template,UInt8)
            score_gpu=similar(template)
            FAA_gpu,FAC_gpu,FCC_gpu=
                similar(template),similar(template),similar(template)
            outer_gpu,inner_gpu=
                similar(template,UInt8),similar(template,UInt8)
            nchannel_profile_tile!(
                I_gpu,W_gpu,FAA_gpu,FAC_gpu,FCC_gpu,
                flag_gpu,score_gpu,outer_gpu,inner_gpu,hs,
                Φ_gpu,μI_gpu,μW_gpu,I0_gpu,μI_eff_gpu,μW_eff_gpu,
                nchannel_basis.normal_II,nchannel_basis.normal_IW,
                nchannel_basis.normal_WW,nchannel_controls,
            )
            Astar=vec(Array(I_gpu))
            Cstar=vec(Array(W_gpu))
            flags=vec(Array(flag_gpu))
            retained=@. (flags&0x2c)==0
            δA=Float64.(Astar).-A0
            δC=Float64.(Cstar).-C0
            q=@. FAA*δA^2+2FAC*δA*δC+FCC*δC^2
            push!(q_per_ray,mean(q[retained]))
            push!(retained_fraction,count(retained)/nsample)
            BS.release_backend!((
                hs,I_gpu,W_gpu,flag_gpu,score_gpu,
                FAA_gpu,FAC_gpu,FCC_gpu,outer_gpu,inner_gpu,
            ))
        end
    finally
        BS.release_backend!((
            Φ_gpu,μI_gpu,μW_gpu,I0_gpu,
            μI_eff_gpu,μW_eff_gpu,template,
        ))
    end

    q_reference=2.0
    inflation=max(quantile(q_per_ray,controls.quantile)/q_reference,1.0)
    epsilon_squared=inflation*tnv_controls.epsilon_squared
    (
        controls,
        sampled_rays=nsample,
        q_per_ray,
        retained_fraction,
        q_quantile=quantile(q_per_ray,controls.quantile),
        fisher_calibration_inflation=inflation,
        epsilon_squared,
        epsilon=sqrt(epsilon_squared),
        post_scatter_correction_included=false,
        limitation=(
            "The fitted-count bootstrap reproduces the simulator's MC Fano/"*
            "correlation model after exact linear pileup inversion. It does not "*
            "model uncertainty from the nonlinear scatter re-estimation."
        ),
    )
end

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
        quality_flag=flags,score_norm=score,
        fisher=(AA=FAA,AC=FAC,CC=FCC),
        outer_iterations=outer,inner_iterations=inner,
        elapsed_s=elapsed,
    )
end
end

# ╔═╡ 067e491c-b305-46d0-abf9-bad0431f7b5e
tnv_noiseless_pipeline = let
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

# ╔═╡ 55e6d611-7c3a-48d6-a01e-7e91707a2e27
tnv_full_pipeline_first_difference = let
    noiseless=tnv_noiseless_pipeline.cong
    noisy=sino_basis_nchannel_slab
    difference=cat(
        noisy.sino_iodine.-noiseless.sino_iodine,
        noisy.sino_water.-noiseless.sino_water;dims=4,
    )
    L=material_precision_sqrt(
        noisy.fisher.AA,noisy.fisher.AC,noisy.fisher.CC,
    )
    valid=@. ((noisy.quality_flag|noiseless.quality_flag)&0x2c)==0
    whitened=ray_whiten(difference,L)
    whitened[.!repeat(valid,1,1,1,2)].=0
    (
        residual=norm(whitened),
        residual_per_active_dof=norm(whitened)/sqrt(2count(valid)),
        active_rays=count(valid),
        interpretation=:one_full_pipeline_noise_realization,
    )
end

# ╔═╡ a44c3a3b-6f30-456e-b0ff-a7f71a4fdd37
tnv_full_pipeline_repetitions = let
    noiseless=tnv_noiseless_pipeline.cong
    function retained_payload(cong,seed,elapsed_s)
        difference=cat(
            cong.sino_iodine.-noiseless.sino_iodine,
            cong.sino_water.-noiseless.sino_water;dims=4,
        )
        L=material_precision_sqrt(
            cong.fisher.AA,cong.fisher.AC,cong.fisher.CC,
        )
        valid=@. ((cong.quality_flag|noiseless.quality_flag)&0x2c)==0
        whitened=ray_whiten(difference,L)
        whitened[.!repeat(valid,1,1,1,2)].=0
        (
            seed,elapsed_s,
            residual_to_noiseless=norm(whitened),
            active_rays=count(valid),
            sino_iodine=cong.sino_iodine,
            sino_water=cong.sino_water,
            fisher=cong.fisher,
            quality_flag=cong.quality_flag,
        )
    end
    repetitions=Any[
        retained_payload(
            sino_basis_nchannel_slab,sim_opts.seed,
            sino_basis_nchannel_slab.elapsed_s,
        ),
    ]
    for seed in (40401,40402,40403)
        GC.gc(true)
        simulated=simulate_corrected_pcct_bins(
            use_noise=true,seed=seed,
        )
        cong=cong_from_corrected_bins(simulated.bins)
        push!(repetitions,retained_payload(
            cong,seed,simulated.elapsed_s+cong.elapsed_s,
        ))
    end
    residuals=[r.residual_to_noiseless for r in repetitions]
    (
        repetitions,
        residuals,
        empirical_quantile=0.95,
        epsilon_noise_full_pipeline=quantile(residuals,0.95),
        total_elapsed_s=sum(r.elapsed_s for r in repetitions[2:end]),
        sample_size=length(repetitions),
        limitation=(
            "Four complete realizations provide a fast preliminary empirical "*
            "calibration. The stored Cong outputs permit direct recomputation "*
            "against the later consistent reference image."
        ),
    )
end

# ╔═╡ 87a1be5f-8211-4861-a8f9-d6e996b76d4a
begin
function ray_whiten!(output,input,L)
    x1,x2=selectdim(input,ndims(input),1),selectdim(input,ndims(input),2)
    y1,y2=selectdim(output,ndims(output),1),selectdim(output,ndims(output),2)
    @. y1=L.AA*x1+L.AC*x2
    @. y2=L.AC*x1+L.CC*x2
    output
end

function central_row_reusable_pair_projector(geom,image_size,prototype)
    geom2d=central_row_geometry(geom,image_size)
    T=eltype(prototype)
    sinogram=similar(
        prototype,T,geom2d.n_cols,1,geom2d.n_angles,2,
    )
    volume=similar(prototype,T,image_size...,1,2)
    ws=(
        source_positions=to_gpu(T.(geom2d.source_positions)),
        detector_centers=to_gpu(T.(geom2d.detector_centers)),
        detector_u=to_gpu(T.(geom2d.detector_u)),
        detector_v=to_gpu(T.(geom2d.detector_v)),
    )
    function forward!(image)
        for material in 1:2
            BS.dd_forward_project!(
                @view(sinogram[:,:,:,material]),
                reshape(@view(image[:,:,material]),image_size...,1),
                geom2d;volume_extent=geom2d.fov,
                ws_source_positions=ws.source_positions,
                ws_detector_centers=ws.detector_centers,
                ws_detector_u=ws.detector_u,
                ws_detector_v=ws.detector_v,
            )
        end
        sinogram
    end
    function adjoint!(input)
        for material in 1:2
            BS.dd_backproject!(
                @view(volume[:,:,:,material]),
                @view(input[:,:,:,material]),
                geom2d;volume_extent=geom2d.fov,
                ws_source_positions=ws.source_positions,
                ws_detector_centers=ws.detector_centers,
                ws_detector_u=ws.detector_u,
                ws_detector_v=ws.detector_v,
            )
        end
        reshape(volume,image_size...,2)
    end
    (
        forward! = forward!,adjoint! = adjoint!,
        geometry=geom2d,sinogram,volume,ws,
    )
end

function preconditioned_weighted_operator(
    pair_workspace,L,Rm,Rx,epsilon,prototype,
)
    image_size=size(Rx)
    x_buffer=similar(prototype,eltype(prototype),image_size...,2)
    z_buffer=similar(x_buffer)
    forward_buffer=similar(pair_workspace.sinogram)
    adjoint_input=similar(pair_workspace.sinogram)
    function apply_right!(output,input)
        @views begin
            @. output[:,:,1]=Rx*(
                Rm[1,1]*input[:,:,1]+Rm[1,2]*input[:,:,2]
            )
            @. output[:,:,2]=Rx*(
                Rm[2,1]*input[:,:,1]+Rm[2,2]*input[:,:,2]
            )
        end
        output
    end
    function forward!(z)
        apply_right!(x_buffer,z)
        pair_workspace.forward!(x_buffer)
        ray_whiten!(forward_buffer,pair_workspace.sinogram,L)
        forward_buffer./=epsilon
    end
    function adjoint!(q)
        ray_whiten!(adjoint_input,q,L)
        adjoint_input./=epsilon
        pair_workspace.adjoint!(adjoint_input)
        apply_right!(z_buffer,reshape(pair_workspace.volume,image_size...,2))
    end
    (
        forward! = forward!,adjoint! = adjoint!,
        apply_right! = apply_right!,
        pair_workspace,x_buffer,z_buffer,forward_buffer,adjoint_input,
    )
end

function preconditioned_cgls(
    operator,data,initial;
    normal_tolerance=1e-5,check_every=5,
    plateau_window=6,plateau_fraction=0.01,safety_iterations=2000,
)
    z=copy(initial)
    residual=copy(data)
    residual.-=operator.forward!(z)
    s=copy(operator.adjoint!(residual))
    normal_scale=max(norm(operator.adjoint!(data)),eps(float(eltype(z))))
    p=copy(s)
    γ=dot(s,s)
    history=NamedTuple[]
    function record(iteration)
        (
            iteration,
            discrepancy=norm(residual),
            normal_residual=sqrt(γ),
            relative_normal_residual=sqrt(γ)/normal_scale,
        )
    end
    push!(history,record(0))
    converged=history[end].relative_normal_residual<normal_tolerance
    plateaued=false
    iteration=0
    while !converged&&!plateaued&&iteration<safety_iterations
        iteration+=1
        q=operator.forward!(p)
        denominator=dot(q,q)
        denominator>0||break
        α=γ/denominator
        z.+=α.*p
        residual.-=α.*q
        snew=operator.adjoint!(residual)
        γnew=dot(snew,snew)
        p.=snew.+(γnew/γ).*p
        γ=γnew
        converged=sqrt(γ)/normal_scale<normal_tolerance
        if iteration%check_every==0||converged
            push!(history,record(iteration))
            if length(history)≥plateau_window
                recent=history[end-plateau_window+1:end]
                normals=getproperty.(recent,:relative_normal_residual)
                discrepancies=getproperty.(recent,:discrepancy)
                plateaued=maximum(normals)/max(minimum(normals),eps())<
                    1+plateau_fraction &&
                    abs(first(discrepancies)-last(discrepancies))/
                    max(first(discrepancies),eps())<plateau_fraction/10
            end
        end
    end
    (
        z,history,converged,plateaued,
        stop_reason=converged ? :normal_equation :
            plateaued ? :documented_plateau : :safety_limit,
    )
end
end

# ╔═╡ e91efbee-d615-4496-9867-a94651672ad6
begin
function square_root_ramp(
    image_size,voxel_size,prototype,
)
    nx,ny=image_size
    hx,hy=voxel_size
    kx=[min(k,nx-k) for k in 0:nx-1]
    ky=[min(k,ny-k) for k in 0:ny-1]
    wx=2 .* sin.(π .* kx ./ nx) ./ hx
    wy=2 .* sin.(π .* ky ./ ny) ./ hy
    rho=sqrt.(wx.^2 .+ transpose(wy).^2)
    rho0=min(
        2sin(π/nx)/hx,
        2sin(π/ny)/hy,
    )
    q=(rho.^2 .+ rho0^2).^(1/4)
    to_gpu(Float32.(q))
end

function apply_square_root_ramp!(output,input,q)
    transformed=BS.FFTW.fft(input,(1,2))
    transformed .*= reshape(q,size(q)...,1)
    output .= real.(BS.FFTW.ifft(transformed,(1,2)))
    output
end

function ramp_preconditioned_weighted_operator(
    pair_workspace,L,Rm,Rx,q,epsilon,prototype,
)
    image_size=size(Rx)
    material_buffer=similar(prototype,eltype(prototype),image_size...,2)
    ramp_buffer=similar(material_buffer)
    right_buffer=similar(material_buffer)
    adjoint_sensitivity_buffer=similar(material_buffer)
    adjoint_ramp_buffer=similar(material_buffer)
    adjoint_right_buffer=similar(material_buffer)
    forward_buffer=similar(pair_workspace.sinogram)
    adjoint_input=similar(pair_workspace.sinogram)
    function material_forward!(output,input)
        @views begin
            @. output[:,:,1]=
                Rm[1,1]*input[:,:,1]+Rm[1,2]*input[:,:,2]
            @. output[:,:,2]=
                Rm[2,1]*input[:,:,1]+Rm[2,2]*input[:,:,2]
        end
        output
    end
    function material_adjoint!(output,input)
        @views begin
            @. output[:,:,1]=
                Rm[1,1]*input[:,:,1]+Rm[2,1]*input[:,:,2]
            @. output[:,:,2]=
                Rm[1,2]*input[:,:,1]+Rm[2,2]*input[:,:,2]
        end
        output
    end
    function apply_right!(output,input)
        material_forward!(material_buffer,input)
        apply_square_root_ramp!(ramp_buffer,material_buffer,q)
        @. output=Rx*ramp_buffer
        output
    end
    function apply_right_adjoint!(output,input)
        @. adjoint_sensitivity_buffer=Rx*input
        apply_square_root_ramp!(
            adjoint_ramp_buffer,adjoint_sensitivity_buffer,q,
        )
        material_adjoint!(output,adjoint_ramp_buffer)
    end
    function forward!(z)
        apply_right!(right_buffer,z)
        pair_workspace.forward!(right_buffer)
        ray_whiten!(forward_buffer,pair_workspace.sinogram,L)
        forward_buffer./=epsilon
    end
    function adjoint!(y)
        ray_whiten!(adjoint_input,y,L)
        adjoint_input./=epsilon
        pair_workspace.adjoint!(adjoint_input)
        apply_right_adjoint!(
            adjoint_right_buffer,
            reshape(pair_workspace.volume,image_size...,2),
        )
    end
    (
        forward! = forward!,adjoint! = adjoint!,
        apply_right! = apply_right!,
        apply_right_adjoint! = apply_right_adjoint!,
        pair_workspace,forward_buffer,adjoint_input,
        right_buffer,adjoint_right_buffer,q,
    )
end

function monitored_lsmr(
    operator,data,initial;
    iterations,
)
    z=copy(initial)
    u=copy(data)
    u.-=operator.forward!(z)
    beta=norm(u)
    u./=beta
    v=copy(operator.adjoint!(u))
    alpha=norm(v)
    v./=alpha
    zetabar=alpha*beta
    alphabar=alpha
    rho=one(alpha)
    rhobar=one(alpha)
    cbar=one(alpha)
    sbar=zero(alpha)
    h=copy(v)
    hbar=similar(v);fill!(hbar,zero(eltype(hbar)))
    normal_scale=max(norm(operator.adjoint!(data)),eps(float(eltype(z))))
    data_scale=max(norm(data),eps(float(eltype(z))))
    history=NamedTuple[]
    for iteration in 1:iterations
        u_new=operator.forward!(v)
        u_new.-=alpha.*u
        beta=norm(u_new)
        u.=u_new./beta
        v_new=operator.adjoint!(u)
        v_new.-=beta.*v
        alpha=norm(v_new)
        v.=v_new./alpha

        rho_old=rho
        rho=hypot(alphabar,beta)
        c=alphabar/rho
        s=beta/rho
        theta_new=s*alpha
        alphabar=c*alpha
        rhobar_old=rhobar
        theta_bar=sbar*rho
        rho_temp=cbar*rho
        rhobar=hypot(rho_temp,theta_new)
        cbar=rho_temp/rhobar
        sbar=theta_new/rhobar
        zeta=cbar*zetabar
        zetabar=-sbar*zetabar
        hbar.=h.-(
            theta_bar*rho/(rho_old*rhobar_old)
        ).*hbar
        z.+=zeta/(rho*rhobar).*hbar
        h.=v.-theta_new/rho.*h

        residual=copy(operator.forward!(z))
        residual.-=data
        normal=operator.adjoint!(residual)
        push!(history,(;
            iteration,
            relative_data_residual=norm(residual)/data_scale,
            relative_normal_residual=norm(normal)/normal_scale,
            data_residual=norm(residual),
        ))
    end
    (;z,history)
end
end

# ╔═╡ 61a5abd5-a5ce-4aa8-8e8c-3c579dedbed5
tnv_ramp_preconditioner_gate = let
    resolution=64
    image_size=(resolution,resolution)
    noiseless=tnv_noiseless_pipeline.cong
    valid=(noiseless.quality_flag .& 0x2c).==0
    idx=findall(valid)
    pooled_AA=mean(Float64.(noiseless.fisher.AA[idx]))
    pooled_AC=mean(Float64.(noiseless.fisher.AC[idx]))
    pooled_CC=mean(Float64.(noiseless.fisher.CC[idx]))
    EF=eigen(Symmetric(
        [pooled_AA pooled_AC;pooled_AC pooled_CC],
    ))
    Rm=Float32.(
        EF.vectors*Diagonal(inv.(sqrt.(EF.values)))*EF.vectors'
    )
    L_cpu=material_precision_sqrt(
        noiseless.fisher.AA,noiseless.fisher.AC,noiseless.fisher.CC,
    )
    L=(AA=to_gpu(L_cpu.AA),AC=to_gpu(L_cpu.AC),CC=to_gpu(L_cpu.CC))
    prototype=to_gpu(zeros(Float32,image_size...,2))
    pair_old=central_row_reusable_pair_projector(
        sino_basis_nchannel_slab.geom,image_size,prototype,
    )
    pair_ramp=central_row_reusable_pair_projector(
        sino_basis_nchannel_slab.geom,image_size,prototype,
    )
    single=central_row_reusable_projector(
        sino_basis_nchannel_slab.geom,image_size,
        to_gpu(ones(Float32,image_size)),
    )
    ones_image=to_gpu(ones(Float32,image_size))
    pones=copy(single.forward!(ones_image))
    traceF=noiseless.fisher.AA.+noiseless.fisher.CC
    detF=noiseless.fisher.AA.*noiseless.fisher.CC.-
        noiseless.fisher.AC.^2
    λmax=0.5f0 .* (
        traceF .+ sqrt.(max.(traceF.^2 .- 4f0 .* detF,0f0))
    )
    sensitivity=copy(single.adjoint!(
        to_gpu(Float32.(λmax)).*pones,
    ))
    floor_value=max(1f-6*maximum(sensitivity),eps(Float32))
    Rx=1f0./sqrt.(max.(sensitivity,floor_value))
    epsilon=Float32(
        tnv_full_pipeline_repetitions.epsilon_noise_full_pipeline,
    )
    old=preconditioned_weighted_operator(
        pair_old,L,Rm,Rx,epsilon,prototype,
    )
    voxel_size=(
        pair_ramp.geometry.fov[1]/resolution,
        pair_ramp.geometry.fov[2]/resolution,
    )
    q=square_root_ramp(image_size,voxel_size,prototype)
    ramp=ramp_preconditioned_weighted_operator(
        pair_ramp,L,Rm,Rx,q,epsilon,prototype,
    )
    rng=MersenneTwister(0x6404d)
    x_test=zeros(Float32,image_size...,2)
    for j in 1:resolution,i in 1:resolution
        x=(i-(resolution+1)/2)/(resolution/2)
        y=(j-(resolution+1)/2)/(resolution/2)
        r2=x*x+y*y
        x_test[i,j,1]=r2<0.42^2 ? 0.02f0 : 0f0
        x_test[i,j,2]=r2<0.78^2 ? 1f0 : 0f0
        (x+0.22)^2+(y-0.18)^2<0.12^2 &&
            (x_test[i,j,1]=0.055f0)
    end
    x_test_gpu=to_gpu(x_test)
    pair_data=central_row_reusable_pair_projector(
        sino_basis_nchannel_slab.geom,image_size,prototype,
    )
    consistent=pair_data.forward!(x_test_gpu)
    data=similar(pair_data.sinogram)
    ray_whiten!(data,consistent,L)
    data./=epsilon

    zprobe=to_gpu(randn(rng,Float32,image_size...,2))
    yprobe=to_gpu(randn(rng,Float32,size(data)))
    function adjoint_error(operator)
        Az=copy(operator.forward!(zprobe))
        Aty=copy(operator.adjoint!(yprobe))
        abs(dot(Az,yprobe)-dot(zprobe,Aty))/
            max(abs(dot(Az,yprobe)),abs(dot(zprobe,Aty)),eps(Float32))
    end
    old_adjoint_error=adjoint_error(old)
    ramp_adjoint_error=adjoint_error(ramp)
    max(old_adjoint_error,ramp_adjoint_error)<3f-4||error(
        "Right-preconditioned adjoint audit failed."
    )
    initial=to_gpu(zeros(Float32,image_size...,2))
    old_result=monitored_lsmr(old,data,initial;iterations=30)
    ramp_result=monitored_lsmr(ramp,data,initial;iterations=30)
    old_final=last(old_result.history)
    ramp_final=last(ramp_result.history)
    data_improvement=old_final.relative_data_residual/
        ramp_final.relative_data_residual
    normal_improvement=old_final.relative_normal_residual/
        ramp_final.relative_normal_residual
    (
        resolution,iterations=30,
        old=old_final,ramp=ramp_final,
        old_adjoint_error,ramp_adjoint_error,
        data_improvement,normal_improvement,
        pass=min(data_improvement,normal_improvement)>=10,
        required_improvement=10.0,
    )
end

# ╔═╡ 558a210b-5729-4a26-82ae-b354813c8642
begin
function build_noiseless_consistent_reference_128()
    resolution=128
    image_size=(resolution,resolution)
    noiseless=tnv_noiseless_pipeline.cong
    observed=cat(
        noiseless.sino_iodine,noiseless.sino_water;dims=4,
    )
    valid=(noiseless.quality_flag .& 0x2c).==0
    idx=findall(valid)
    pooled_AA=mean(Float64.(noiseless.fisher.AA[idx]))
    pooled_AC=mean(Float64.(noiseless.fisher.AC[idx]))
    pooled_CC=mean(Float64.(noiseless.fisher.CC[idx]))
    Fbar=[pooled_AA pooled_AC;pooled_AC pooled_CC]
    EF=eigen(Symmetric(Fbar))
    Rm=EF.vectors*Diagonal(inv.(sqrt.(EF.values)))*EF.vectors'
    L_cpu=material_precision_sqrt(
        noiseless.fisher.AA,noiseless.fisher.AC,noiseless.fisher.CC,
    )
    prototype=to_gpu(zeros(Float32,image_size...,2))
    pair=central_row_reusable_pair_projector(
        sino_basis_nchannel_slab.geom,image_size,prototype,
    )
    ones_image=to_gpu(ones(Float32,image_size))
    single=central_row_reusable_projector(
        sino_basis_nchannel_slab.geom,image_size,ones_image,
    )
    pones=copy(single.forward!(ones_image))
    traceF=noiseless.fisher.AA.+noiseless.fisher.CC
    detF=noiseless.fisher.AA.*noiseless.fisher.CC.-
        noiseless.fisher.AC.^2
    λmax=0.5f0 .* (
        traceF .+ sqrt.(max.(traceF.^2 .- 4f0 .* detF,0f0))
    )
    sensitivity_input=to_gpu(Float32.(λmax)).*pones
    sensitivity=copy(single.adjoint!(sensitivity_input))
    sensitivity_floor=max(
        Float32(1e-6)*maximum(sensitivity),eps(Float32),
    )
    Rx=1f0./sqrt.(max.(sensitivity,sensitivity_floor))
    L=(
        AA=to_gpu(L_cpu.AA),AC=to_gpu(L_cpu.AC),CC=to_gpu(L_cpu.CC),
    )
    epsilon=Float32(tnv_full_pipeline_repetitions.epsilon_noise_full_pipeline)
    operator=preconditioned_weighted_operator(
        pair,L,Float32.(Rm),Rx,epsilon,prototype,
    )
    rng=MersenneTwister(0x12804d)
    ztest=to_gpu(randn(rng,Float32,image_size...,2))
    qtest=to_gpu(randn(
        rng,Float32,size(operator.forward_buffer),
    ))
    Az=copy(operator.forward!(ztest))
    Atq=copy(operator.adjoint!(qtest))
    adjoint_error=abs(dot(Az,qtest)-dot(ztest,Atq))/
        max(abs(dot(Az,qtest)),abs(dot(ztest,Atq)),eps(Float32))
    adjoint_error<2f-4||error(
        "Preconditioned weighted operator failed adjoint audit: $adjoint_error"
    )
    data=similar(operator.forward_buffer)
    ray_whiten!(data,to_gpu(Float32.(observed)),L)
    data./=epsilon
    result=nothing
    elapsed=@elapsed begin
        result=preconditioned_cgls(
            operator,data,to_gpu(zeros(Float32,image_size...,2));
            normal_tolerance=1e-5,
        )
    end
    image=similar(prototype)
    operator.apply_right!(image,result.z)
    predicted=copy(pair.forward!(image))
    model_difference=predicted.-to_gpu(Float32.(observed))
    model_whitened=ray_whiten(model_difference,L)
    final_record=last(result.history)
    (
        resolution,elapsed_s=elapsed,
        converged=result.converged,plateaued=result.plateaued,
        stop_reason=result.stop_reason,
        iterations=final_record.iteration,
        relative_normal_residual=final_record.relative_normal_residual,
        r_model=norm(model_whitened),
        epsilon_noise=epsilon,
        image=Array(image),
        Rm=Rm,Rx_summary=(
            minimum_value=minimum(Rx),maximum_value=maximum(Rx),
            sensitivity_floor=sensitivity_floor,
        ),
        adjoint_error=adjoint_error,
        history=result.history,
    )
end

tnv_noiseless_consistent_reference_128 = (
    status=:pending_improved_krylov_preconditioner,
    attempted_resolution=128,
    elapsed_before_interrupt_s=300.0,
    certified_r_model=nothing,
    certified_r_min=nothing,
    interpretation=(
        "The reusable operator removed the allocation pathology, but the "*
        "specified pooled-Fisher/sensitivity right preconditioner did not "*
        "reach the normal-equation tolerance or documented plateau within "*
        "five minutes. No residual-floor claim is made."
    ),
)
end

# ╔═╡ 8f227499-7784-4dd3-ad30-9810cc7741bb
tnv_exact_problem = let
    image_size=recon_opts.matrix_size[1:2]
    mid_z=size(nchannel_basis_volumes.vol_water,3)÷2+1
    initial=cat(
        nchannel_basis_volumes.vol_iodine[:,:,mid_z],
        nchannel_basis_volumes.vol_water[:,:,mid_z];
        dims=3,
    )
    observed=cat(
        sino_basis_nchannel_slab.sino_iodine,
        sino_basis_nchannel_slab.sino_water;
        dims=4,
    )
    bridge=central_row_projector(sino_basis_nchannel_slab.geom,image_size)
    (
        image_size,mid_z,initial=Float32.(initial),
        observed=Float32.(observed),bridge,
        ray_whitening=tnv_ray_whitening,
        material_whitening=Float32.(tnv_whitening.whitening),
        epsilon=Float32(tnv_bootstrap.epsilon),
        bounds=(
            tnv_controls.image_bounds.iodine_g_cm3,
            tnv_controls.image_bounds.water_g_cm3,
        ),
    )
end;

# ╔═╡ f88339be-5430-4357-83ca-ad027ff54e3a
function covariance_weighted_cgls(
    observed,initial,forward,adjoint,L;
    normal_tolerance=1e-5,check_every=5,safety_iterations=10_000,
)
    x=copy(initial)
    residual=ray_whiten(observed.-pair_forward(x,forward),L)
    s=pair_adjoint(ray_whiten(residual,L),adjoint)
    normal_scale=max(norm(
        pair_adjoint(
            ray_whiten(ray_whiten(observed,L),L),
            adjoint,
        ),
    ),eps(float(eltype(x))))
    p=copy(s)
    γ=dot(s,s)
    history=NamedTuple[(
        iteration=0,
        discrepancy=norm(residual),
        normal_residual=sqrt(γ),
        relative_normal_residual=sqrt(γ)/normal_scale,
    )]
    converged=history[end].relative_normal_residual<normal_tolerance
    iteration=0
    while !converged && iteration<safety_iterations
        iteration+=1
        q=ray_whiten(pair_forward(p,forward),L)
        denominator=dot(q,q)
        denominator>0||break
        α=γ/denominator
        x.+=α.*p
        residual.-=α.*q
        snew=pair_adjoint(ray_whiten(residual,L),adjoint)
        γnew=dot(snew,snew)
        β=γnew/γ
        p.=snew.+β.*p
        s=snew
        γ=γnew
        converged=sqrt(γ)/normal_scale<normal_tolerance
        if iteration%check_every==0||converged
            push!(history,(
                iteration,
                discrepancy=norm(residual),
                normal_residual=sqrt(γ),
                relative_normal_residual=sqrt(γ)/normal_scale,
            ))
        end
    end
    (
        image=x,history,converged,
        stop_reason=converged ? :normal_equation : :safety_limit,
    )
end

# ╔═╡ 69d03cc0-af0c-433b-90da-b57b7e6401de
tnv_data_feasibility = (
    status=:pending_fast_preconditioned_cgls,
    reason=(
        "The exact 128² matrix-free CGLS path was interrupted after two minutes. "*
        "No residual floor is reported until the requested normal-equation "*
        "criterion is reached."
    ),
    normal_equation_tolerance=1e-5,
)

# ╔═╡ f427fde7-40ad-4b6e-b7ed-8ee31ffab8ae
tnv_feasibility_conclusion = (
    endpoint_selected=false,
    classification=:projection_domain_tnv_terminated,
    interpretation=(
        "The small explicit oracle passes and the reusable wrapper removes the "*
        "allocation pathology. However, the final square-root-ramp LSMR gate "*
        "improved the consistent-data residual by only 1.21× and worsened the "*
        "normal residual; it failed the declared 10× gate. No projector-domain "*
        "feasibility or residual-floor claim is made."
    ),
    next_required_evidence=(
        :post_fbp_cross_nps_covariance,
        :post_fbp_tnv_endpoint,
    ),
)

# ╔═╡ 8bbf633c-8859-437a-8237-dd2637916e87
tnv_decision_table = (
    small_oracle=(
        pass=tnv_small_oracle.pass,
        relative_objective_error=
            tnv_small_oracle.objective_relative_error,
        relative_primal_dual_gap=
            tnv_small_oracle.pdhg.relative_gap,
    ),
    discrepancy_budgets=(
        epsilon_detector_only=tnv_bootstrap.epsilon,
        epsilon_noise_preliminary=
            tnv_full_pipeline_repetitions.epsilon_noise_full_pipeline,
        full_pipeline_samples=
            tnv_full_pipeline_repetitions.sample_size,
        r_model=nothing,
        epsilon_full=nothing,
    ),
    residual_floor=(
        r_min=nothing,
        comparison_to_epsilon_full=:not_available,
    ),
    projector_solver=(
        scalar_pdhg=:correct_but_unsuitable,
        pooled_fisher_sensitivity_cgls=
            :interrupted_without_certificate,
        ramp_lsmr_gate_pass=tnv_ramp_preconditioner_gate.pass,
        data_improvement=
            tnv_ramp_preconditioner_gate.data_improvement,
        normal_improvement=
            tnv_ramp_preconditioner_gate.normal_improvement,
        decision=:terminate_projection_domain_tnv,
    ),
    runtime_and_gap=(
        allocating_adjoint_512_s=
            tnv_operator_profile[2].allocating_adjoint_s,
        reusable_adjoint_512_s=
            tnv_operator_profile[2].reusable_adjoint_s,
        oracle_relative_gap=tnv_small_oracle.pdhg.relative_gap,
        final_projector_tnv_gap=nothing,
    ),
)

# ╔═╡ c81819eb-fd95-4193-af0c-8c8fce52c1e2
md"""
## TNV endpoint decision

The full covariance, whitening, matched-projector adjointness, and convex TNV
operators pass their in-notebook checks. A short unregularized,
covariance-weighted CGLS solve is then used as a **feasibility gate** before any
long TNV run.

The 32-seed fitted-count bootstrap includes the detector Monte Carlo Fano
factors and inter-bin correlations and increases the Fisher variance by only
about 15%. The capped solve still does not reach that calibrated discrepancy
radius. Exact linear pileup inversion is represented, but uncertainty from
nonlinear scatter re-estimation is not; that remaining boundary is reported
explicitly rather than replaced by an arbitrary radius inflation.

Therefore this notebook deliberately selects **no TNV VMI endpoint yet**.
This is a documented runtime/statistical-calibration limitation, not a claim
that the convex problem itself is infeasible. The common-FBP VMI branch remains
the valid baseline, and no RSKR result is promoted as a mathematical solution.
"""

# ╔═╡ fab1f0dc-8ee2-41b3-bf90-33122487d725
tnv_goal_audit = (
    proven=(
        fixed_cong_profile_decomposition=true,
        one_four_bin_scan=true,
        full_per_ray_fisher_and_covariance=true,
        off_diagonal_material_covariance=true,
        quality_flags_retained=true,
        matched_projector_and_exact_transpose=
            tnv_projector_adjoint_audit.pass,
        covariance_whitened_tnv_definition=true,
        basis_unit_invariance=tnv_basis_invariance_audit.pass,
        cpu_operator_checks=tnv_unit_audit.pass,
        gpu_operator_checks=tnv_backend_audit.pass,
        common_fbp_baseline=true,
        raw_and_derived_channels_not_double_counted=true,
        deterministic_memory_cleanup=true,
        detector_bootstrap_completed=true,
    ),
    partial=(
        corrected_count_covariance=(
            detector_fano_and_bin_correlation=true,
            exact_linear_pileup_inverse=true,
            nonlinear_scatter_reestimation=false,
        ),
        epsilon_calibration=(
            detector_bootstrap=tnv_bootstrap.epsilon,
            post_scatter_bootstrap=nothing,
        ),
        full_resolution_optimization=(
            converged=false,
            feasibility_established=false,
            limitation=tnv_feasibility_conclusion.classification,
        ),
    ),
    not_claimed_without_endpoint=(
        dense_tnv_vmis=false,
        primal_dual_convergence=false,
        dose_behavior=false,
        repeated_seed_covariance=false,
        nps=false,
        mtf_ttf_edge_slope=false,
        concentration_nist_bias=false,
        endpoint_visuals=false,
        tnv_vs_rskr_comparison=false,
    ),
    completion_status=:rigorous_limitation_without_selected_endpoint,
)

# ╔═╡ 97f45341-b796-4611-95b8-4319f49eb053
md"""
### Completion audit

`tnv_goal_audit` is the authoritative scope check. Items under `proven` have
direct live-notebook evidence. Items under `partial` retain their exact
statistical or computational boundary. Endpoint-dependent VMI, dose, NPS,
TTF/MTF, concentration-bias, covariance, and visual claims are intentionally
listed under `not_claimed_without_endpoint`; they are **not** inferred from
operator unit tests or the common-FBP baseline.
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
# ╟─4d6d8375-5805-4e1d-9c77-fcde574c42a1
# ╠═4a7e1704-9abc-4430-8146-5e8a1fbaf852
# ╠═83bece80-87f7-4e18-943f-13d0a7287548
# ╠═829630be-523b-464e-a250-e6b617be8ce1
# ╠═b07ec202-ddcb-465a-8f0d-18e53c2ff902
# ╠═d9c149c8-135a-42c9-bb95-213b0aec3b32
# ╠═739a1ff6-dd9b-4edf-8653-73a2704bfb59
# ╠═813d96ad-4b83-4db9-95db-67336bf4b43e
# ╠═47384dd6-7b4e-403b-b4b5-1c6bc34a77e2
# ╟─6c7fdf0f-bc9f-4cf7-96ba-eb0e1ff94c60
# ╠═9c675f52-0046-41be-a203-d603cffcea2d
# ╠═dc5da7d1-3dcf-4942-b6f5-ea632aee1472
# ╠═595b5b3b-d8e6-4001-9f63-0365728ca667
# ╠═7429a1a3-785d-423d-93c4-2bb0ee3cf1a7
# ╠═3d3ece92-6d20-49b4-8539-4a2ff2200455
# ╠═df46022b-033d-45df-8bca-478517904e86
# ╠═296b915c-edbb-4b76-bf42-d42061109608
# ╠═0a8d4979-5665-4b4d-84e9-b5da72183134
# ╠═a8a740e0-9e33-447a-b34c-dfa0139662da
# ╠═223af75d-05c7-431c-b598-70bef1f1c6a0
# ╠═f86d4398-f245-4899-adf9-30ce3dd405e3
# ╠═707aec65-d9d6-4462-a662-c669beb8525b
# ╠═067e491c-b305-46d0-abf9-bad0431f7b5e
# ╠═55e6d611-7c3a-48d6-a01e-7e91707a2e27
# ╠═a44c3a3b-6f30-456e-b0ff-a7f71a4fdd37
# ╠═87a1be5f-8211-4861-a8f9-d6e996b76d4a
# ╠═e91efbee-d615-4496-9867-a94651672ad6
# ╠═61a5abd5-a5ce-4aa8-8e8c-3c579dedbed5
# ╠═558a210b-5729-4a26-82ae-b354813c8642
# ╠═8f227499-7784-4dd3-ad30-9810cc7741bb
# ╠═f88339be-5430-4357-83ca-ad027ff54e3a
# ╠═69d03cc0-af0c-433b-90da-b57b7e6401de
# ╠═f427fde7-40ad-4b6e-b7ed-8ee31ffab8ae
# ╠═8bbf633c-8859-437a-8237-dd2637916e87
# ╟─c81819eb-fd95-4193-af0c-8c8fce52c1e2
# ╠═fab1f0dc-8ee2-41b3-bf90-33122487d725
# ╟─97f45341-b796-4611-95b8-4319f49eb053
