# =============================================================================
# End-to-end functional pipelines behind the five-struct API
# =============================================================================
#
# `EICTPipeline` bundles the host plans for the notebook-01 path
#
#     material fractions → per-material path lengths (DD) → EICT chain (+ water BHC)
#                        → FDK → HU
#
# into one immutable object built from the same five structs the notebooks use
# (`Phantom`, `Scanner`, `CTProtocol`, `SimOptions`, `ReconOptions`).  Every
# stage is pure, so the whole forward map `eict_forward(fractions, pipe, ε, ε_e)`
# is a function of the material-fraction tensor and the noise tensors: it traces
# under `Reactant.@compile` and differentiates under Enzyme end to end.
#
# Oracle: the legacy chain `simulate!` → `apply_bhc_water` → `reconstruct!(FDK)`
# → `to_hounsfield` (see test/functional/test_pipeline.jl).

"""
    EICTPipeline{T}

Host-side bundle of plans for the EICT phantom → HU pipeline. Fields:
`geom`, `vol_shape` (phantom mask shape), `n_mat` (rows of the μ table = phantom
materials, in `phantom.materials` order), `volume_extent` (phantom extent, cm),
`eict::EICTPlan`, `fbp::FBPPlan`, `μ_water` (HU reference), `recon_shape`.
"""
struct EICTPipeline{T <: AbstractFloat, EP, FP}
    geom::BS.CTGeometry
    vol_shape::NTuple{3, Int}
    n_mat::Int
    volume_extent::NTuple{3, Float64}
    eict::EP
    fbp::FP
    μ_water::T
    recon_shape::NTuple{3, Int}
    batching::NamedTuple{(:dd, :spectral, :fdk), Tuple{Int, Int, Int}}   # views per compiled loop iteration, per stage
    unrolled::Bool                                                       # true: legacy per-view unrolled programs (view_batch = 1)
end

Base.eltype(::EICTPipeline{T}) where {T} = T

"""
    eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts;
                  bhc = :water, filter = StandardFilter(), cutoff = 1.0, T = Float32,
                  spectrum_override = nothing) -> EICTPipeline{T}

Build the pipeline from the five structs. Host-only bridge: the legacy
`create_eict_workspace` is used once to harvest the config tensors (spectrum, μ
table, bowtie×heel weights, air reference, η, I0, noise constants) and
`calibrate_bhc_water` builds the knobless water BHC (`bhc = :water`, the
notebook-01 default; `nothing` disables BHC and uses NIST water at 70 keV as the
HU reference; a `WaterBHC` may be passed directly). `filter`/`cutoff` mirror
`create_fdk_recon_workspace` (default `StandardFilter()`).

`view_batch = :auto` (default) sizes the compiled view loops of every stage from
`batch_budget_mb` (see [`_auto_batching`](@ref)), so any number of views and any
phantom grid compile to a program of bounded size; an integer forces that many
views per loop iteration in all stages, and `1` selects the legacy per-view
unrolled programs (bit-identical summation order; toy sizes only).
"""
function eict_pipeline(
        phantom::BS.Phantom, scanner::BS.EICTScanner, protocol, sim_opts, recon_opts;
        bhc = :water,
        filter = BS.StandardFilter(),
        cutoff::Real = 1.0,
        T::Type{<:AbstractFloat} = Float32,
        spectrum_override = nothing,
        view_batch::Union{Symbol, Integer} = :auto,
        batch_budget_mb::Real = 512,
    )
    (view_batch === :auto || (view_batch isa Integer && view_batch >= 1)) ||
        throw(ArgumentError("view_batch must be :auto or an integer ≥ 1, got $view_batch"))
    batch_budget_mb > 0 || throw(ArgumentError("batch_budget_mb must be positive"))
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T, spectrum_override)
    geom = ws.geom
    model = bhc === :water ? BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom) : bhc
    eplan = eict_plan(ws, protocol, sim_opts; bhc = model)
    fplan = fbp_plan(geom, recon_opts.matrix_size; filter, cutoff, T)
    μw = model === nothing ? BS.get_reference_μ_water(70.0) : model.μ_water_ref
    n_mat = size(eplan.μ_tbl, 1)
    vol_shape = size(phantom.mask)
    batching = if view_batch === :auto
        _auto_batching(eplan.sino_shape, vol_shape, length(eplan.wη), n_mat, recon_opts.matrix_size, batch_budget_mb)
    else
        (dd = Int(view_batch), spectral = Int(view_batch), fdk = Int(view_batch))
    end
    return EICTPipeline{T, typeof(eplan), typeof(fplan)}(
        geom, vol_shape, n_mat, phantom.extent, eplan, fplan, T(μw), recon_opts.matrix_size,
        batching, view_batch === 1)
end

"""
    _auto_batching(sino_shape, vol_shape, n_E, n_mat, recon_shape, budget_mb) -> (dd, spectral, fdk)

Views per compiled loop iteration for each stage so that the stage's per-batch
transient stays within `budget_mb` (host estimate of the live Float32 tensors:
DD gather chain `n_col·n_row·n_long·(6 + n_mat)`, spectral sum
`n_col·n_row·n_E·3`, FDK `nx·ny·nz·10` per view).  The program size is then
set by the budget, not by the scan (any number of views, any phantom grid).
"""
function _auto_batching(sino_shape::NTuple{3, Int}, vol_shape::NTuple{3, Int}, n_E::Int, n_mat::Int,
        recon_shape::NTuple{3, Int}, budget_mb::Real)
    n_col, n_row, n_view = sino_shape
    n_long = max(vol_shape[1], vol_shape[2])
    nx, ny, nz = recon_shape
    bytes = budget_mb * 2^20
    per_view(x) = Int(clamp(bytes ÷ (4 * x), 1, n_view))
    return (dd = per_view(n_col * n_row * n_long * (6 + n_mat)),
            spectral = per_view(n_col * n_row * n_E * 3),
            fdk = per_view(nx * ny * nz * 10))
end

"""
    onehot_fractions(mask, n_mat; T = Float32) -> Array{T,4}

HOST helper: 0-based material label mask `(nx, ny, nz)` → material-fraction
tensor `(nx, ny, nz, n_mat)` with `fractions[:, :, :, m] = (mask .== m-1)`.
This is the differentiable parameterization of the phantom (labeled phantoms are
its one-hot special case; XCIST/XCAT volume-fraction phantoms map onto it directly).
"""
function onehot_fractions(mask::AbstractArray{<:Integer, 3}, n_mat::Integer; T::Type{<:AbstractFloat} = Float32)
    mask_h = Array(mask)
    out = Array{T}(undef, size(mask_h)..., n_mat)
    for m in 1:n_mat
        out[:, :, :, m] = T.(mask_h .== (m - 1))
    end
    return out
end

# stack N-D arrays along a new trailing axis with `cat` (traceable; `stack` is not).
_cat_new_axis(xs) = cat(map(x -> reshape(x, size(x)..., 1), xs)...; dims = ndims(xs[1]) + 1)

"""
    PCCTPipeline{T, PP, FP}

Photon-counting analogue of [`EICTPipeline`](@ref): the five-struct build of
the PCCT chain (spectral bins → per-bin log sinograms → optional pile-up →
bin combine) and the FDK plan, with the same batching contract.  Fields as
[`EICTPipeline`](@ref) plus `pcct::PCCTPlan`, `bt` (per-pixel spectral bowtie
or `nothing`), `G`/`I0_groups` (bin-combine matrix and per-group air counts)
and `groups` (host description of the combine).
"""
struct PCCTPipeline{T <: AbstractFloat, PP, FP}
    geom::BS.CTGeometry
    vol_shape::NTuple{3, Int}
    n_mat::Int
    volume_extent::NTuple{3, Float64}
    pcct::PP
    bt::Union{Nothing, Array{T, 3}}
    G::Matrix{T}
    I0_groups::Vector{T}
    groups::Vector{Vector{Int}}
    fbp::FP
    recon_shape::NTuple{3, Int}
    batching::NamedTuple{(:dd, :spectral, :fdk), Tuple{Int, Int, Int}}
    unrolled::Bool
end
Base.eltype(::PCCTPipeline{T}) where {T} = T

"""
    pcct_pipeline(phantom, scanner, protocol, sim_opts, recon_opts;
                  groups = [[b] for b in 1:n_bins], filter = StandardFilter(), cutoff = 1.0, T = Float32,
                  view_batch = :auto, batch_budget_mb = 512) -> PCCTPipeline{T}

Build the photon-counting pipeline from the five structs (a photon-counting
`scanner`).  The legacy `create_workspace` is used once to harvest the
config tensors (μ table, energy→bin weights, per-bin air counts, MC pile-up
matrix, per-pixel spectral bowtie); `groups` lists the bins summed into each
output channel (default: every bin its own channel).  Batching as in
[`eict_pipeline`](@ref).
"""
function pcct_pipeline(
        phantom::BS.Phantom, scanner::BS.PCCTScanner, protocol, sim_opts, recon_opts;
        groups::Union{Nothing, Vector{Vector{Int}}} = nothing,
        filter = BS.StandardFilter(),
        cutoff::Real = 1.0,
        T::Type{<:AbstractFloat} = Float32,
        view_batch::Union{Symbol, Integer} = :auto,
        batch_budget_mb::Real = 512,
    )
    (view_batch === :auto || (view_batch isa Integer && view_batch >= 1)) ||
        throw(ArgumentError("view_batch must be :auto or an integer ≥ 1, got $view_batch"))
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    geom = ws.geom
    n_E = length(ws.energies)
    vol_shape = size(phantom.mask)
    n_view = geom.n_angles
    n_bins = length(ws.I0_bins)
    groups = groups === nothing ? [[b] for b in 1:n_bins] : groups
    batching = if view_batch === :auto
        _auto_batching((geom.n_cols, geom.n_rows, n_view), vol_shape, n_E, size(ws.μ_table, 1), recon_opts.matrix_size, batch_budget_mb)
    else
        (dd = Int(view_batch), spectral = Int(view_batch), fdk = Int(view_batch))
    end
    unrolled = view_batch === 1
    pplan = pcct_plan(ws; view_batch = unrolled ? 0 : batching.spectral, T)
    bt = nothing            # the PCCT workspace folds the centre-pixel bowtie into W (no per-pixel spectral bowtie)
    G, I0g = combine_matrix(Vector{Float64}(ws.I0_bins), groups, T)
    fplan = fbp_plan(geom, recon_opts.matrix_size; filter, cutoff, T)
    return PCCTPipeline{T, typeof(pplan), typeof(fplan)}(
        geom, vol_shape, size(ws.μ_table, 1), phantom.extent, pplan, bt, G, I0g, groups, fplan,
        recon_opts.matrix_size, batching, unrolled)
end

function _pcct_on_device(p::PCCTPlan{T}, ref) where {T}
    μ = _on_device(p.μ_table, ref); W = _on_device(p.W, ref); I0 = _on_device(p.I0_bins, ref)
    St = _on_device(p.pileup_St, ref); Si = _on_device(p.pileup_Sinv_t, ref)
    return PCCTPlan{T, typeof(μ), typeof(W), typeof(I0), typeof(p.I0_bins_f64), typeof(St), typeof(Si)}(
        μ, W, I0, p.I0_bins_f64, St, Si, p.eps, p.noise_reduction, p.view_chunks, p.view_batch)
end

"""
    pcct_forward(fractions, pipe::PCCTPipeline, N_input = nothing) -> μ volumes (nx, ny, nz, n_groups)

The complete pure PCCT forward model: fractions → path lengths → spectral bins
→ log sinograms (→ pile-up) → bin combine → FDK per combined channel.  `N_input`
optionally supplies externally drawn per-bin counts ([`draw_pcct_counts`](@ref))
for the exact-Poisson path; the surrogate path is [`pcct_forward_surrogate`](@ref).
Convert to HU with `to_hu(μ, μ_water)` for the channel's reference water μ.
"""
function pcct_forward(fractions::AbstractArray{<:Any, 4}, pipe::PCCTPipeline{T}, N_input = nothing) where {T}
    P = material_paths(fractions, pipe)
    plan = _pcct_on_device(pipe.pcct, P)
    bt = _on_device(pipe.bt, P)
    sino = pcct_chain_combined(P, plan, _on_device(pipe.G, P), _on_device(pipe.I0_groups, P), N_input; bt)
    return _fdk_channels(sino, pipe)
end

"""
    pcct_forward_surrogate(fractions, pipe::PCCTPipeline, ε) -> μ volumes

[`pcct_forward`](@ref) with the differentiable Gaussian count surrogate driven
by the N(0,1) input tensor `ε` (per-bin sinogram shape).
"""
function pcct_forward_surrogate(fractions::AbstractArray{<:Any, 4}, pipe::PCCTPipeline{T}, ε) where {T}
    P = material_paths(fractions, pipe)
    plan = _pcct_on_device(pipe.pcct, P)
    bt = _on_device(pipe.bt, P)
    bins = pcct_chain_surrogate(P, plan, ε; bt).bins
    sino = combine_bins(bins, _on_device(pipe.G, P), _on_device(pipe.I0_groups, P), plan.eps)
    return _fdk_channels(sino, pipe)
end

function _fdk_channels(sino::AbstractArray{<:Any, 4}, pipe::PCCTPipeline{T}) where {T}
    fplan = _fbp_on_device(pipe.fbp, sino)
    vb = pipe.unrolled ? 1 : pipe.batching.fdk
    vols = [fdk(sino[:, :, :, g], fplan; view_batch = vb) for g in 1:size(sino, 4)]
    return _cat_new_axis(vols)
end

"""
    to_hu(μ, μ_water::Real)

`1000 (μ − μ_water) / μ_water` for an explicit reference water μ.
"""
to_hu(μ::AbstractArray, μ_water::Real) = (oftype(μ_water, 1000) .* (μ .- μ_water)) ./ μ_water

"""
    material_paths(fractions, pipe) -> (n_col, n_row, n_view, n_mat)

Per-material path lengths `P[:, :, :, m] = A(fractions[:, :, :, m])` with the
functional DD projector (one static-tap projection per material per view; the
material axis is the batch axis of the linear operator). Differentiable in
`fractions`.
"""
function material_paths(fractions::AbstractArray{<:Any, 4}, pipe::Union{EICTPipeline{T}, PCCTPipeline{T}}) where {T}
    geom = pipe.geom
    if !pipe.unrolled
        runs = dd_run_plans(geom, pipe.vol_shape; view_batch = pipe.batching.dd,
            volume_extent = pipe.volume_extent, eltype = T)
        return reduce((a, b) -> cat(a, b; dims = 3), [dd_project_run(fractions, r) for r in runs])
    end
    plans = [dd_view_plan(geom, v, pipe.vol_shape; volume_extent = pipe.volume_extent, eltype = T)
             for v in 1:geom.n_angles]
    per_mat = [_cat_new_axis([dd_project_view(fractions[:, :, :, m], p) for p in plans])
               for m in 1:size(fractions, 4)]
    return _cat_new_axis(per_mat)
end


# -----------------------------------------------------------------------------
# Device hooks (overridden by ext/BasisSimulatorReactantExt.jl inside a trace)
# -----------------------------------------------------------------------------

"""
    _on_device(x, ref)

Return `x` in the array world of `ref`. Identity for plain arrays; the Reactant
extension lifts host plan tensors into the traced graph as constants when `ref`
is a traced array (a host `Matrix * traced` otherwise degrades to a scalar
fallback with one op per multiply-add).
"""
_on_device(x, ref) = x

_on_device(::Nothing, ref) = nothing

function _eict_on_device(p::EICTPlan{T}, ref) where {T}
    μ = _on_device(p.μ_tbl, ref); w = _on_device(p.wη, ref); b = _on_device(p.bt, ref)
    a = _on_device(p.air_ref, ref); hc = _on_device(p.scatter_Hc, ref); hr = _on_device(p.scatter_Hr, ref)
    bc = _on_device(p.bhc_coeffs, ref)
    return EICTPlan{T, typeof(μ), typeof(w), typeof(b), typeof(a), typeof(p.ff_log), typeof(hc), typeof(hr), typeof(bc)}(
        μ, w, b, a, p.I0, p.σ_e, p.use_noise, p.use_enoise, p.ff_log, hc, hr, p.scatter_C, p.scatter_sw,
        bc, p.eps, p.sino_shape, p.energies)
end

_fbp_on_device(p::FBPPlan, ref) = FBPPlan(p; tensors = map(x -> _on_device(x, ref), p.tensors))

"""
    simulate_sino(fractions, pipe, ε = nothing, ε_e = nothing) -> (n_col, n_row, n_view)

Pure equivalent of `simulate!(EICTWorkspace)` followed by `apply_bhc_water`
(when the pipeline carries a BHC): the BHC-corrected log sinogram. `ε`, `ε_e`
are the quantum / electronic N(0,1) noise tensors (flat or 3-D); see
[`draw_eict_noise`](@ref).
"""
function simulate_sino(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline, ε = nothing, ε_e = nothing)
    P = material_paths(fractions, pipe)
    return eict_chain(P, _eict_on_device(pipe.eict, P), ε, ε_e; view_batch = pipe.unrolled ? 0 : pipe.batching.spectral)
end

"""
    reconstruct_μ(sino, pipe) -> (nx, ny, nz)

Pure equivalent of `reconstruct!(create_fdk_recon_workspace(sino, geom, matrix))`.
"""
reconstruct_μ(sino::AbstractArray{<:Any, 3}, pipe::EICTPipeline) =
    fdk(sino, _fbp_on_device(pipe.fbp, sino); view_batch = pipe.unrolled ? 1 : pipe.batching.fdk)

"""
    to_hu(μ, pipe)

`1000 (μ − μ_water) / μ_water` with the pipeline's calibrated `μ_water`
(legacy `to_hounsfield(μ; μ_water)`).
"""
to_hu(μ::AbstractArray, pipe::EICTPipeline{T}) where {T} = (T(1000) .* (μ .- pipe.μ_water)) ./ pipe.μ_water

"""
    eict_forward(fractions, pipe, ε = nothing, ε_e = nothing) -> HU volume

The complete pure forward model: fractions → path lengths → EICT chain (+BHC)
→ FDK → HU. Compile with `Reactant.@compile`; differentiate with
`Enzyme.gradient(Reverse, loss ∘ eict_forward, fractions)`.
"""
function eict_forward(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline, ε = nothing, ε_e = nothing)
    return to_hu(reconstruct_μ(simulate_sino(fractions, pipe, ε, ε_e), pipe), pipe)
end

"""
    draw_eict_noise(pipe; seed) -> (ε, ε_e)

HOST helper reproducing the legacy noise draws of `simulate!` for a given
`SimOptions.seed`: `MersenneTwister(seed)` → `randn` quantum tensor, then (if the
detector has electronic noise) the electronic tensor from the same stream.
"""
function draw_eict_noise(pipe::EICTPipeline{T}; seed::Integer) where {T}
    n = prod(pipe.eict.sino_shape)
    rng = Random.MersenneTwister(seed)
    ε = randn(rng, T, n)
    ε_e = pipe.eict.use_enoise ? randn(rng, T, n) : nothing
    return ε, ε_e
end

# -----------------------------------------------------------------------------
# One entry point behind the five structs: dispatch on the scanner family
# -----------------------------------------------------------------------------

"""
    pipeline(phantom, scanner, protocol, sim_opts, recon_opts; kwargs...)

The functional pipeline for the scanner family: [`eict_pipeline`](@ref) for an
[`EICTScanner`](@ref), [`pcct_pipeline`](@ref) for a [`PCCTScanner`](@ref)
(keyword arguments are forwarded).  Run it with [`forward`](@ref).
"""
pipeline(phantom::BS.Phantom, scanner::BS.EICTScanner, protocol, sim_opts, recon_opts; kwargs...) =
    eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; kwargs...)
pipeline(phantom::BS.Phantom, scanner::BS.PCCTScanner, protocol, sim_opts, recon_opts; kwargs...) =
    pcct_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; kwargs...)

"""
    forward(fractions, pipe, noise...) 

The pure forward model of the pipeline: HU image for an [`EICTPipeline`](@ref)
([`eict_forward`](@ref)), per-channel μ volumes for a [`PCCTPipeline`](@ref)
([`pcct_forward`](@ref)).  Compile with `Reactant.@compile`, differentiate with
`Enzyme.gradient(Reverse, …)`.
"""
forward(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline, ε = nothing, ε_e = nothing) = eict_forward(fractions, pipe, ε, ε_e)
forward(fractions::AbstractArray{<:Any, 4}, pipe::PCCTPipeline, N_input = nothing) = pcct_forward(fractions, pipe, N_input)
