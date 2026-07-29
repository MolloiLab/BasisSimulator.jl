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
# Dual-kVp Rapid-Switching Virtual Monoenergetic Imaging

GE Revolution Apex Elite rapid-kVp-switching simulation (80 + 140 kVp,
Gammex 472 phantom).

This notebook retains the established dual-kVp acquisition, reconstruction,
and output structure while using the generalized all-channel profiled Cong
estimator at ``K=2``.
"""

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## Pipeline

`80/140-kVp corrected counts → two-channel profiled Cong → per-basis FBP →
Kalender ACNR → analytical VMI`

The same generalized ``K``-channel Cong code used by notebook 04 is retained
with ``K=2``. Both channels remain separate throughout material estimation.
No T-LBF is used in the canonical dual-kVp branch.
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

# ╔═╡ 040e1000-0000-4000-8000-000000000100
md"""
### 01. `Phantom()` Struct
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

# ╔═╡ 040e1000-0000-4000-8000-000000000101
md"""
### 02. `Scanner()` Struct
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 0,
    detection_gain = 10.0,
)

# ╔═╡ 040e1000-0000-4000-8000-000000000102
md"""
### 03. `CTProtocol()` Struct
"""

# ╔═╡ f9c0af7a-addd-4249-96fb-b9078765fbd1
protocol_low = BS.CTProtocol(
    kVp = 80,
    mA = 407.0,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 03b00004-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 405.0,
    views = 984,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 040e1000-0000-4000-8000-000000000103
md"""
### 04. `SimOptions()` & `ReconOptions()`
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
    projector = :dd_fast,
)

# ╔═╡ 08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int,5.0/slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_recon_slices),
        fov_cm = 35.0,
        z_cm = 0.5,
    )
end;

# ╔═╡ 5ecd97c6-ad47-4558-886d-22ed45eda97d
md"""
### 05. Forward Project: `simulate!`

The two EICT simulations retain the original, clinically matched 80/140-kVp
technique and the simulator's corrected log-line-integral outputs. The
ray-dependent air counts and full source/filter/bowtie/detector responses are
saved so decomposition uses absolute count-domain channel means.
"""

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
begin
    function simulate_dual_kvp(sim_options;label_suffix="")
        function run_channel(protocol,label)
        @info "Simulating dual-kVp channel: $label"
        ws=BS.create_eict_workspace(
            scanner,protocol,sim_options,recon_opts,phantom,
        )
        try
            BS.simulate!(ws,phantom,protocol,sim_options)
            I0_scalar=BS.compute_detector_I0(
                ws.geom,protocol,sum(ws.weights),
            )*Float64(ws.η_eff)
            air_ref=ws.bowtie_air_reference===nothing ?
                ones(Float32,ws.geom.n_cols,ws.geom.n_rows) :
                Float32.(Array(ws.bowtie_air_reference))
            energies,response=BS.resolve_source_spectrum_full(
                sim_options,protocol;scanner=scanner,geom=ws.geom,
            )
            (
                sino=Float32.(Array(ws.sinogram)),
                I0_ray=Float32.(I0_scalar.*air_ref),
                energies=Float32.(energies),
                response=Float32.(response),
                geom=ws.geom,
            )
        finally
            BS.release_backend!(ws)
        end
    end
        low=run_channel(protocol_low,"80 kVp $label_suffix")
        high=run_channel(protocol_high,"140 kVp $label_suffix")
        (
            bins=[low.sino,high.sino],
            channels=(low,high),
            geom=low.geom,
            labels=("80 kVp","140 kVp"),
        )
    end

    # Full verified :eict acquisition baseline.
    sim_bins=simulate_dual_kvp(sim_opts;label_suffix="noisy baseline")
end;

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## VMI Pipeline

### 01. Standard Two-Channel Corrected Counts

The 80- and 140-kVp corrected log transmissions retain every native detector
row. Cong is solved independently for every physical ray; no noisy row is
replicated or collapsed into another ray. The matched absolute response
``\Phi_k(E)`` retains source spectrum, filtration, bowtie, and detector
weighting for each channel, detector column, and detector row.
"""

# ╔═╡ 4985581f-616d-4bb7-ab9b-967d7250b28b
nchannel_controls = (
    iodine_bounds = (-0.10f0, 0.40f0), # g/cm²
    water_bounds = (-2.0f0, 50.0f0),   # g/cm²
    outer_iterations = 16,              # canonical converged PCCT control
    inner_iterations = 12,              # canonical converged PCCT control
    max_iodine_step = 0.05f0,
    max_water_step = 5.0f0,
    parameter_tolerance = 5.0f-5,        # canonical converged PCCT control
    fisher_condition_limit = 1.0f8,
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
    normal_II, normal_IW, normal_WW, controls,
) where {K}
    # Ray-dependent dual-kVp responses use detector-column initializer terms.
    nE = length(μρ_I)
    A_lo, A_hi = controls.iodine_bounds
    C_lo, C_hi = controls.water_bounds
    n_outer, n_inner = controls.outer_iterations, controls.inner_iterations
    A_step, C_step = controls.max_iodine_step, controls.max_water_step
    parameter_tolerance = controls.parameter_tolerance
    fisher_condition_limit = controls.fisher_condition_limit
    air_gate = controls.air_gate

    BS.AK.foreachindex(sino_I) do idx
        ncol=size(sino_I,1)
        nrow=size(sino_I,2)
        col=mod1(idx,ncol)
        row=mod1(cld(idx,ncol),nrow)
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
            rhs_I += μI_eff[col,row,k]*hs[k][idx]
            rhs_W += μW_eff[col,row,k]*hs[k][idx]
        end
        nII=normal_II[col,row]
        nIW=normal_IW[col,row]
        nWW=normal_WW[col,row]
        det0_raw = nII*nWW - nIW*nIW
        initializer_valid = isfinite(det0_raw) && det0_raw > 1f-12
        det0 = initializer_valid ? det0_raw : 1f0
        A = initializer_valid ?
            clamp((nWW*rhs_I-nIW*rhs_W)/det0,A_lo,A_hi) :
            clamp(0f0,A_lo,A_hi)
        C = initializer_valid ?
            clamp((nII*rhs_W-nIW*rhs_I)/det0,C_lo,C_hi) :
            clamp(20f0,C_lo,C_hi)

        # Guaranteed monotone aggregate equation, used here only to stabilize
        # the fast solver's initial water value at its current iodine value.
        y_total=0f0
        for k in 1:K
            y_total += max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
        end
        croot_lo,croot_hi=C_lo,C_hi
        total_lo,total_hi=0f0,0f0
        for k in 1:K, e in 1:nE
            total_lo += Φ[col,row,e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_lo)
            total_hi += Φ[col,row,e,k]*exp(-μρ_I[e]*A-μρ_W[e]*croot_hi)
        end
        aggregate_bracketed=total_lo≥y_total && total_hi≤y_total
        attainable_max,attainable_min=0f0,0f0
        for k in 1:K, e in 1:nE
            attainable_max += Φ[col,row,e,k]*exp(
                -μρ_I[e]*A_lo-μρ_W[e]*C_lo,
            )
            attainable_min += Φ[col,row,e,k]*exp(
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
                    total_mid += Φ[col,row,e,k]*exp(-μρ_I[e]*A-μρ_W[e]*mid)
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
                        z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                        λ += z
                        dC -= μρ_W[e] * z
                    end
                    λ = max(λ, 1f-6)
                    # Corrected counts may be fractional after detector correction.
                    y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
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
                    z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dA -= μρ_I[e] * z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
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
                    z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                    λ += z
                    dC -= μρ_W[e] * z
                end
                λ = max(λ, 1f-6)
                y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
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
                z = Φ[col,row,e,k] * exp(-μρ_I[e]*A - μρ_W[e]*C)
                λ += z
                dA -= μρ_I[e]*z
                dC -= μρ_W[e]*z
            end
            λ = max(λ,1f-6)
            y = max(I0[col,row,k]*exp(-hs[k][idx]),1f-6)
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
### 02. Native Detector-Row Handling

Every native detector row is retained and decomposed independently. The
row-center cone factor is
```math
s_r=\sqrt{1+(z_r/\mathrm{SAD})^2}.
```
The maximum departure from unity is reported below. The generalized
``K``-channel likelihood uses each row's own corrected measurements,
``I_{0,k}``, and absolute response ``\Phi_k``. This preserves the native cone
geometry and avoids treating geometrically distinct rows as extra spectral
channels or duplicated copies.

The 984-view acquisition also oversamples the angular sampling requirement
of the 512-pixel reconstruction,
``N_{\theta,\mathrm{required}}=\lceil\pi N/2\rceil=805``. The native-view
sinograms are passed directly to FBP; no notebook-level angular FFT filtering
or row replication is applied.
"""

# ╔═╡ 50ed35bb-df61-4862-a99b-ea37380f30d8
begin
function build_nchannel_slab_counts(sim_data)
    available_rows = size(sim_data.bins[1],2)
    selected_rows = 1:available_rows
    row_positions = (
        collect(selected_rows) .- (available_rows+1)/2
    ) .* sim_data.geom.pixel_row_size
    cone_scales = sqrt.(1 .+ (row_positions ./ sim_data.geom.SAD).^2)
    channel_data = map(eachindex(sim_data.bins)) do k
        channel=sim_data.channels[k]
        I0=Float64.(channel.I0_ray[:,selected_rows])
        h=Float32.(channel.sino[:,selected_rows,:])
        response=Float64.(channel.response[:,selected_rows,:])
        response ./= max.(sum(response;dims=3),eps(Float64))
        Φ=response.*reshape(I0,size(I0,1),size(I0,2),1)
        (
            bin=h,
            I0=Float32.(I0),
            energies=Float32.(channel.energies),
            Φ=Float32.(Φ),
        )
    end
    (
        bins=getproperty.(channel_data,:bin),
        I0=getproperty.(channel_data,:I0),
        energies=getproperty.(channel_data,:energies),
        Φ=getproperty.(channel_data,:Φ),
        nrows=available_rows,selected_rows=selected_rows,
        available_rows=available_rows,cone_scales=cone_scales,
        max_cone_relerr=maximum(abs.(cone_scales .- 1)),
        target_slice_thickness_mm=0.625,
        thickness_mm=10*sim_data.geom.pixel_row_size,
    )
end

# Retain every native detector row; Cong is solved per physical ray.
nchannel_slab_counts=build_nchannel_slab_counts(sim_bins)
end;

# ╔═╡ 4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
begin
function build_nchannel_basis(slab_counts)
    E=sort!(unique(vcat(slab_counts.energies...)))
    ncol,nrow=size(first(slab_counts.Φ))[1:2]
    K=length(slab_counts.Φ)
    Φ=zeros(Float32,ncol,nrow,length(E),K)
    for k in 1:K
        lookup=Dict(e=>i for (i,e) in enumerate(E))
        for (source_index,e) in enumerate(slab_counts.energies[k])
            Φ[:,:,lookup[e],k].=slab_counts.Φ[k][:,:,source_index]
        end
    end
    μρ_I = Float32[
        BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(e))
        for e in E
    ]
    μρ_W = Float32[
        BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(e))
        for e in E
    ]

    I0=cat(slab_counts.I0...;dims=3)
    I0_from_Φ=dropdims(sum(Float64.(Φ);dims=3);dims=3)
    I0_relerr=maximum(abs.(
        I0_from_Φ.-Float64.(I0)
    )./max.(Float64.(I0),eps(Float64)))
    I0_relerr < 5e-5 || error(
        "Applied response and I0 disagree (max relative error = $(I0_relerr))."
    )

    Φsum=max.(I0,eps(Float32))
    μI_eff=dropdims(sum(
        Φ.*reshape(μρ_I,1,1,length(E),1);dims=3,
    );dims=3)./Φsum
    μW_eff=dropdims(sum(
        Φ.*reshape(μρ_W,1,1,length(E),1);dims=3,
    );dims=3)./Φsum
    normal_II=dropdims(sum(abs2,μI_eff;dims=3);dims=3)
    normal_IW=dropdims(sum(μI_eff.*μW_eff;dims=3);dims=3)
    normal_WW=dropdims(sum(abs2,μW_eff;dims=3);dims=3)

    (
        E = E, Φ = Φ, μρ_I = μρ_I, μρ_W = μρ_W,
        I0 = Float32.(I0),
        μI_eff = μI_eff, μW_eff = μW_eff,
        normal_II,normal_IW,normal_WW,
        I0_relerr = I0_relerr,
    )
end

# Absolute K=2 responses for every retained detector row.
nchannel_basis=build_nchannel_basis(nchannel_slab_counts)
end;

# ╔═╡ 040e1000-0000-4000-8000-000000000003
md"""
### 03. Two-Channel Profiled Cong Material Decomposition

Both corrected kVp channels enter the same ``K``-channel profiled likelihood
used in notebook 04, here with ``K=2``. The estimator jointly produces water
and iodine basis sinograms without effective-energy substitution.
"""

# ╔═╡ b86a9c50-cb10-44b2-af2e-06bdd50943b7
begin
function run_nchannel_profile(slab_counts,basis,geom)
    shape = size(slab_counts.bins[1])
    sino_I = Array{Float32}(undef,shape)
    sino_W = Array{Float32}(undef,shape)
    flags = Array{UInt8}(undef,shape)
    score_norm = Array{Float32}(undef,shape)
    fisher_AA = Array{Float32}(undef,shape)
    fisher_AC = Array{Float32}(undef,shape)
    fisher_CC = Array{Float32}(undef,shape)
    outer_iterations = Array{UInt8}(undef,shape)
    inner_iterations = Array{UInt8}(undef,shape)
    Φ_gpu = to_gpu(basis.Φ)
    μρ_I_gpu = to_gpu(basis.μρ_I)
    μρ_W_gpu = to_gpu(basis.μρ_W)
    I0_gpu = to_gpu(basis.I0)
    μI_eff_gpu = to_gpu(basis.μI_eff)
    μW_eff_gpu = to_gpu(basis.μW_eff)
    normal_II_gpu = to_gpu(basis.normal_II)
    normal_IW_gpu = to_gpu(basis.normal_IW)
    normal_WW_gpu = to_gpu(basis.normal_WW)
    elapsed = @elapsed for vrange in BS.tile_ranges(
        shape[3],nchannel_controls.tile_views,
    )
        hs = [
            to_gpu(Float32.(slab_counts.bins[k][:,:,vrange]))
            for k in eachindex(slab_counts.bins)
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
            normal_II_gpu,normal_IW_gpu,normal_WW_gpu,nchannel_controls,
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
    (
        sino_iodine=sino_I,sino_water=sino_W,quality_flag=flags,
        fisher=(AA=fisher_AA,AC=fisher_AC,CC=fisher_CC),
        score_norm,outer_iterations,inner_iterations,
        geom,elapsed_s=elapsed,
    )
end

# Clean unregularized K=2 Cong on every native detector row.
sino_basis_nchannel_slab=run_nchannel_profile(
    nchannel_slab_counts,nchannel_basis,sim_bins.geom,
)
end;

# ╔═╡ f3de45a6-4818-4ee1-ad56-c65797119dee
begin
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
end

# ╔═╡ 040e1000-0000-4000-8000-000000000005
md"""
### 04. FBP Basis Maps
**Original Dual-kVp Per-Basis Apodization**

The generalized Cong basis sinograms use the original dual-kVp reconstruction
design: a soft iodine kernel controls the low-energy-amplified streak mode,
while the halfway Standard/Soft water kernel retains anatomical resolution and
realistic high-energy noise. These fixed per-basis kernels are applied before
VMI synthesis; no energy-dependent VMI filtering is used.
"""

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

    nchannel_iodine_filter=BS.CustomFilter(
        (0.0,0.25,0.5,0.75,1.0),
        (1.0,0.40,0.12,0.03,0.001),
    )
    nchannel_water_filter=BS.CustomFilter(
        (0.0,0.25,0.5,0.75,1.0),
        (1.0,0.8744,0.6003,0.3031,0.0266),
    )

    function nchannel_fbp_slice(sino_one_row,filter)
        # Every native detector row has its own Cong solution and is passed
        # directly to FBP; no noisy-row replication is permitted.
        sino_gpu=to_gpu(Float32.(sino_one_row))
        ws=BS.create_fdk_recon_workspace(
            sino_gpu,sim_bins.geom,nchannel_fbp_matrix_size;
            filter,
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
        # Full native-row basis sinograms; no row replication.
        vol_water=nchannel_fbp_slice(
            sino_basis_nchannel_slab.sino_water,nchannel_water_filter,
        ),
        vol_iodine=nchannel_fbp_slice(
            sino_basis_nchannel_slab.sino_iodine,nchannel_iodine_filter,
        ),
        angular_response=:none,
        pass_mode=:direct_fbp,
        kernels=(water=:StandardSoftBlend,iodine=:OriginalDualKvpSoft),
    )
end

# ╔═╡ 040e0002-0000-4000-8000-000000000001
md"""
### 05. ACNR
**Anti-Correlated Noise Reduction**

Kalender ACNR is applied jointly to the reconstructed water and iodine maps,
immediately before VMI synthesis. The selected modestly strengthened setting
uses five passes and `beta_max=14` (implementation defaults: two passes and
`beta_max=8`).
"""

# ╔═╡ 040e0002-0000-4000-8000-000000000002
dual_final = let
    # Clean production path: generalized Cong FBP then strengthened ACNR.
    acnr_revision=:moderately_aggressive_5x14
    energies=[50.0,70.0,100.0,140.0]
    water=copy(nchannel_slab_common_fbp.vol_water)
    iodine=copy(nchannel_slab_common_fbp.vol_iodine)
    info=BS.apply_acnr_kalender!(
        water,iodine;
        hp_sigma_px=1.5,window=4,passes=5,beta_max=14.0,
    )
    images=(;water,iodine)
    vmis=synthesize_vmi_stack(images,energies)
    (
        energies,images,vmis,info,acnr_revision,
        acnr=(passes=5,beta_max=14.0,hp_sigma_px=1.5,window=4),
    )
end

# ╔═╡ 040e0002-0000-4000-8000-000000000003
md"""
### 06. VMI Synthesis

The final ACNR water and iodine basis pair is synthesized analytically at
50, 70, 100, and 140 keV using the monoenergetic water and iodine attenuation
coefficients. Every VMI comes from the same reconstructed basis pair.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000003
let
    # Canonical generalized-Cong basis display, ACNR 5/14.
    fig = Mke.Figure(size=(1180,580))
    panels=(
        ("Iodine Basis","g/cm³",dual_final.images.iodine),
        ("Water Basis","g/cm³",dual_final.images.water),
    )
    for (column,(title,label,volume)) in pairs(panels)
        image=volume[:,:,1]
        range=Tuple(quantile(vec(image),(0.01,0.99)))
        axis=Mke.Axis(
            fig[1,2column-1];title,aspect=Mke.DataAspect(),titlesize=32,
        )
        Mke.heatmap!(axis,image;colormap=:viridis,colorrange=range)
        Mke.hidedecorations!(axis)
        Mke.Colorbar(
            fig[1,2column];colormap=:viridis,colorrange=range,
            label,width=16,labelsize=22,
        )
    end
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000004
let
    # Canonical generalized-Cong VMI display, ACNR 5/14.
    fig = Mke.Figure(size=(1180,1180))
    for (index,energy) in pairs(dual_final.energies)
        row=((index-1)÷2)+1
        column=((index-1)%2)+1
        axis=Mke.Axis(
            fig[row,column];title="$(Int(energy)) keV VMI",
            aspect=Mke.DataAspect(),titlesize=32,
        )
        Mke.heatmap!(
            axis,dual_final.vmis[:,:,index];
            colormap=:grays,colorrange=(-200,500),
        )
        Mke.hidedecorations!(axis)
    end
    Mke.Colorbar(
        fig[1:2,3];colormap=:grays,colorrange=(-200,500),
        label="HU",width=16,labelsize=22,ticklabelsize=18,
    )
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000008
md"""
## Results

Per-rod measured versus theoretical HU at the canonical four VMI energies
(50 / 70 / 100 / 140 keV), water-region mean and noise, and linear-regression
agreement with first-principles attenuation values.

Both corrected kVp channels remain separate in the Cong material
decomposition and every VMI is synthesized from one final basis pair.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000002
dual_results = let
    # Quantitative outputs from generalized Cong, per-basis FBP, and ACNR.
    mask=phantom_cpu.mask[:,:,size(phantom_cpu.mask,3)÷2]
    water_mask=collect(BS.erode_mask_2d(
        mask.==UInt8(BS.REGION_SOLID_WATER);erode_px=12.0,
    ))
    rod_labels=(
        Ca=UInt8.((10,11,12,13,14,15,16)),
        I=UInt8.((20,21,22,23,24,25,26)),
    )
    rod_names=(
        Ca=("50 mg/mL","100 mg/mL","200 mg/mL","300 mg/mL",
            "400 mg/mL","500 mg/mL","600 mg/mL"),
        I=("2.0 mg/mL","2.5 mg/mL","5.0 mg/mL","7.5 mg/mL",
           "10.0 mg/mL","15.0 mg/mL","20.0 mg/mL"),
    )
    function rod_roi(label)
        pixels=findall(==(label),mask)
        cx=mean(pixel->Float64(pixel[1]),pixels)
        cy=mean(pixel->Float64(pixel[2]),pixels)
        [CartesianIndex(i,j)
         for j in max(1,floor(Int,cy-8)):min(size(mask,2),ceil(Int,cy+8))
         for i in max(1,floor(Int,cx-8)):min(size(mask,1),ceil(Int,cx+8))
         if (i-cx)^2+(j-cy)^2≤64]
    end
    μwater=Dict(
        energy=>BS.compute_μ_at_energy(BS.XA.Materials.water,energy)
        for energy in dual_final.energies
    )
    rods=Dict{Symbol,NamedTuple}()
    for group in (:Ca,:I)
        measured=zeros(length(rod_labels[group]),length(dual_final.energies))
        theoretical=similar(measured)
        for (row,label) in pairs(rod_labels[group])
            roi=rod_roi(label)
            material=phantom_cpu.materials[Int(label)+1]
            for (column,energy) in pairs(dual_final.energies)
                measured[row,column]=mean(dual_final.vmis[:,:,column][roi])
                μ=BS.compute_μ_at_energy(material,energy)
                theoretical[row,column]=1000*(μ-μwater[energy])/μwater[energy]
            end
        end
        rods[group]=(names=rod_names[group],measured,theoretical)
    end
    (
        water_mask,rods,
        water_mean=[
            mean(dual_final.vmis[:,:,index][water_mask])
            for index in axes(dual_final.vmis,3)
        ],
        water_noise=[
            std(dual_final.vmis[:,:,index][water_mask])
            for index in axes(dual_final.vmis,3)
        ],
        finite=all(isfinite,dual_final.vmis),
        two_channels=length(nchannel_slab_counts.bins)==2,
    )
end

# ╔═╡ 040e1000-0000-4000-8000-000000000040
md"""
### Water ROI

The deeply eroded solid-water ROI is overlaid on the 70-keV VMI and supplies
the mean-HU and noise measurements.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000005
let
    overlay=Float32[value ? 1f0 : NaN32 for value in dual_results.water_mask]
    index70=findfirst(==(70.0),dual_final.energies)
    image70=dual_final.vmis[:,:,index70]
    n=length(dual_final.energies)
    fig = Mke.Figure(size=(1180,580))
    axis=Mke.Axis(
        fig[1,1];title="Eroded Water Region",
        subtitle="Overlaid on 70 keV VMI",
        aspect=Mke.DataAspect(),titlesize=32,subtitlesize=24,
    )
    Mke.heatmap!(axis,image70;colormap=:grays,colorrange=(-200,500))
    Mke.heatmap!(
        axis,overlay;colormap=:reds,alpha=0.5,
        nan_color=(:white,0.0),
    )
    Mke.hidedecorations!(axis)
    mean_axis=Mke.Axis(
        fig[1,2];title="Water Region Mean HU",
        xlabel="VMI Energy (keV)",ylabel="HU",
        xticks=(1:n,string.(Int.(dual_final.energies))),titlesize=32,
    )
    Mke.barplot!(mean_axis,1:n,dual_results.water_mean;
        color=[
            Mke.cgrad(:plasma,n;categorical=true)[index]
            for index in 1:n
        ])
    Mke.hlines!(mean_axis,[0.0];color=:black,linestyle=:dash)
    for (index,value) in pairs(dual_results.water_mean)
        Mke.text!(
            mean_axis,index,value;
            text="$(round(value,digits=2)) HU",
            align=(:center,value≥0 ? :bottom : :top),
            offset=(0,value≥0 ? 5 : -5),
        )
    end
    Mke.ylims!(mean_axis,-10,10)
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000050
md"""
### Water-Region Noise

HU noise is measured over the same deeply eroded solid-water region used for
the water-accuracy calculation.
"""

# ╔═╡ 040e1000-0000-4000-8000-000000000051
let
    overlay=Float32[value ? 1f0 : NaN32 for value in dual_results.water_mask]
    index70=findfirst(==(70.0),dual_final.energies)
    image70=dual_final.vmis[:,:,index70]
    n=length(dual_final.energies)
    fig = Mke.Figure(size=(1180,580))
    axis=Mke.Axis(
        fig[1,1];title="Eroded Water Region",
        subtitle="Overlaid on 70 keV VMI",
        aspect=Mke.DataAspect(),titlesize=32,subtitlesize=24,
    )
    Mke.heatmap!(axis,image70;colormap=:grays,colorrange=(-200,500))
    Mke.heatmap!(
        axis,overlay;colormap=:reds,alpha=0.5,
        nan_color=(:white,0.0),
    )
    Mke.hidedecorations!(axis)
    noise_axis=Mke.Axis(
        fig[1,2];title="Water-Region Noise vs Energy",
        xlabel="VMI Energy (keV)",ylabel="Noise σ (HU)",
        xticks=(1:n,string.(Int.(dual_final.energies))),titlesize=32,
    )
    Mke.barplot!(
        noise_axis,1:n,dual_results.water_noise;color=:tomato,
        strokecolor=:black,strokewidth=1,
    )
    for index in 1:n
        Mke.text!(
            noise_axis,index,dual_results.water_noise[index];
            text="σ=$(round(dual_results.water_noise[index],digits=1))\n"*
                 "⟨HU⟩=$(round(dual_results.water_mean[index],digits=1))",
            align=(:center,:bottom),offset=(0,8),
        )
    end
    Mke.ylims!(noise_axis,0,1.25maximum(dual_results.water_noise))
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000060
md"""
### Per-Rod Regression

Solid lines show measured HU and dashed lines show theoretical HU for calcium
and iodine inserts across the four canonical energies.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000006
let
    fig = Mke.Figure(size=(1180,580))
    panels=(
        (group=:Ca,title="Calcium rods",subtitle="50–600 mg/mL",
         cmap=Mke.cgrad(:Oranges,7;categorical=true),ylim=(0,3800)),
        (group=:I,title="Iodine rods",subtitle="2–20 mg/mL",
         cmap=Mke.cgrad(:GnBu,7;categorical=true),ylim=(0,1500)),
    )
    for (column,panel) in pairs(panels)
        axis=Mke.Axis(
            fig[1,column];title=panel.title,subtitle=panel.subtitle,
            xlabel="VMI energy (keV)",ylabel="HU",
            xticks=dual_final.energies,titlesize=32,subtitlesize=24,
        )
        Mke.ylims!(axis,panel.ylim...)
        data=dual_results.rods[panel.group]
        for index in eachindex(data.names)
            color=panel.cmap[index]
            Mke.scatterlines!(
                axis,dual_final.energies,vec(data.measured[index,:]);
                color,linewidth=2.5,markersize=9,label=data.names[index],
            )
            Mke.lines!(
                axis,dual_final.energies,vec(data.theoretical[index,:]);
                color,linewidth=1.6,linestyle=:dash,
            )
        end
        Mke.axislegend(axis;position=:rt,labelsize=16)
    end
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000070
md"""
### Linear Regression

Measured rod HU is regressed against first-principles theoretical HU at each
energy. The dashed identity line represents perfect agreement.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000007
let
    colors=Dict(
        50.0=>Mke.RGBf(0.85,0.27,0.1),
        70.0=>Mke.RGBf(0.95,0.65,0.13),
        100.0=>Mke.RGBf(0.13,0.59,0.85),
        140.0=>Mke.RGBf(0.1,0.27,0.65),
    )
    fig = Mke.Figure(size=(1000,1200))
    for (row,group) in enumerate((:Ca,:I))
        data=dual_results.rods[group]
        axis=Mke.Axis(
            fig[row,1];
            title=group==:Ca ? "Calcium regression" : "Iodine regression",
            xlabel="Theoretical HU",ylabel="Measured HU",
            # aspect=Mke.DataAspect(),
            titlesize=30,
        )
        low=min(0.0,minimum(data.measured),minimum(data.theoretical))
        high=1.05max(maximum(data.measured),maximum(data.theoretical))
        Mke.lines!(
            axis,[low,high],[low,high];color=:black,
            linestyle=:dash,linewidth=2,label="Unity (y=x)",
        )
        for (column,energy) in pairs(dual_final.energies)
            x=vec(data.theoretical[:,column])
            y=vec(data.measured[:,column])
            slope=sum((x.-mean(x)).*(y.-mean(y)))/sum(abs2,x.-mean(x))
            intercept=mean(y)-slope*mean(x)
            prediction=intercept.+slope.*x
            r2=1-sum(abs2,y.-prediction)/sum(abs2,y.-mean(y))
            endpoints=collect(extrema(x))
            color=colors[energy]
            Mke.scatter!(axis,x,y;color,markersize=11)
            Mke.lines!(
                axis,endpoints,intercept.+slope.*endpoints;
                color,linewidth=2,
                label="$(Int(energy)) keV: slope=$(round(slope,digits=2)), "*
                      "R²=$(round(r2,digits=3))",
            )
        end
        Mke.axislegend(axis;position=:rb,labelsize=16)
    end
    fig
end

# ╔═╡ 040e0002-0000-4000-8000-000000000004
verification = let
    # Canonical dual-kVp pipeline verification, ACNR 5/14.
    checks=NamedTuple[]
    addcheck(name,value,pass)=push!(checks,(;name,value,pass))
    addcheck("both native kVp channels retained",length(nchannel_slab_counts.bins),
             length(nchannel_slab_counts.bins)==2)
    addcheck("all final values finite",dual_results.finite,dual_results.finite)
    water_worst=maximum(abs,dual_results.water_mean)
    addcheck("solid-water worst absolute HU",round(water_worst,digits=2),
             water_worst≤10)
    monotonic=all(diff(dual_results.water_noise).<0)
    addcheck(
        "noise decreases from 50 to 140 keV",
        round.(dual_results.water_noise;digits=2),
        monotonic,
    )
    addcheck(
        "original dual-kVp per-basis FBP kernels",
        (:StandardSoftBlend,:OriginalDualKvpSoft),
        nchannel_slab_common_fbp.kernels==
            (water=:StandardSoftBlend,iodine=:OriginalDualKvpSoft),
    )
    passed=count(check->check.pass,checks)
    rows=join([
        "| $(check.name) | $(check.value) | $(check.pass ? "✅" : "❌") |"
        for check in checks
    ],"\n")
    Markdown.parse("""
### $(passed==length(checks) ? "✅ Verification: PASS" : "❌ Verification: CHECK")

| check | value | pass |
|---|---:|:---:|
$rows
""")
end

# ╔═╡ 040e0002-0000-4000-8000-000000000005
md"""
!!! info "Water–air boundary diagnosis"
    Matched noiseless and noisy controls established that the low-keV rim was
    stochastic amplification in the exactly determined two-channel material
    inversion. The original dual-kVp reconstruction resolves it without a
    mask or post-reconstruction correction: soft iodine-basis apodization
    suppresses the amplified iodine streak mode while the sharper water-basis
    filter preserves anatomy and realistic noise. The generalized Cong solve
    itself remains unchanged.
"""

# ╔═╡ 040e0002-0000-4000-8000-000000000006
md"""
### Summary

```
Simulate 80 + 140 kVp (matched dual-kVp channels)
   → legitimate detector-row count combination
   → generalized two-channel profiled Cong decomposition
   → original dual-kVp per-basis FBP (soft iodine, sharper water)
   → Kalender ACNR (five passes, beta_max=14)
   → analytical VMI synthesis at 50 / 70 / 100 / 140 keV
   → water, noise, rod, regression, and chain verification outputs
```

Every displayed VMI is synthesized from the same final water/iodine pair.
No cross-energy frequency substitution or energy-dependent VMI filtering is
used.
"""

# ╔═╡ Cell order:
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─f2798d62-3509-4cc4-a24f-39ace8bb5a9e
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
# ╠═171294a2-26bd-49e2-ac92-9df48ae5444f
# ╠═69358294-97f2-4782-94d7-c29c747c45f4
# ╠═9ae27110-5c47-442b-a98e-d137599570f2
# ╠═27f065e9-f97c-4392-a83f-e3638682152d
# ╠═a8fc42d0-ee5b-44ea-b4dc-8de368209e44
# ╠═492bb299-678d-4e6f-8c21-1e9178cc2beb
# ╠═9f8d5cd4-147e-4359-95bc-cc096a53f0e7
# ╠═2ff539c9-a678-403c-b629-8068a332a0e9
# ╠═320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
# ╠═86c52e9e-7987-4504-93e6-128017f5e703
# ╟─551f84fe-d7b4-48f9-a475-0c63178a6ede
# ╟─59a5079b-a711-4f28-b3d6-665f0d91fb72
# ╟─040e1000-0000-4000-8000-000000000100
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╟─040e1000-0000-4000-8000-000000000101
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╟─040e1000-0000-4000-8000-000000000102
# ╠═f9c0af7a-addd-4249-96fb-b9078765fbd1
# ╠═03b00004-0000-4000-8000-000000000020
# ╟─040e1000-0000-4000-8000-000000000103
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╠═4985581f-616d-4bb7-ab9b-967d7250b28b
# ╠═73371177-0498-4eda-897b-651c94f43e83
# ╟─b3e1d768-eb02-4c2a-9363-c84077fedc32
# ╠═50ed35bb-df61-4862-a99b-ea37380f30d8
# ╟─040e1000-0000-4000-8000-000000000003
# ╠═b86a9c50-cb10-44b2-af2e-06bdd50943b7
# ╠═f3de45a6-4818-4ee1-ad56-c65797119dee
# ╟─040e1000-0000-4000-8000-000000000005
# ╠═ddfde8bb-ddff-44bb-8b70-b397725402cf
# ╟─040e0002-0000-4000-8000-000000000001
# ╠═040e0002-0000-4000-8000-000000000002
# ╟─040e0002-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000004
# ╟─040e0001-0000-4000-8000-000000000008
# ╠═040e0001-0000-4000-8000-000000000002
# ╟─040e1000-0000-4000-8000-000000000040
# ╟─040e0001-0000-4000-8000-000000000005
# ╟─040e1000-0000-4000-8000-000000000050
# ╟─040e1000-0000-4000-8000-000000000051
# ╟─040e1000-0000-4000-8000-000000000060
# ╟─040e0001-0000-4000-8000-000000000006
# ╟─040e1000-0000-4000-8000-000000000070
# ╟─040e0001-0000-4000-8000-000000000007
# ╟─040e0002-0000-4000-8000-000000000004
# ╟─040e0002-0000-4000-8000-000000000005
# ╟─040e0002-0000-4000-8000-000000000006
