# =============================================================================
# Photon-counting pipeline behind the five structs:
#   fractions → DD path lengths → spectral bins → log sinograms (→ pile-up)
#   → bin combine → FDK per channel (μ volumes).
# =============================================================================

"""
    PCCTPipeline{T, PP, FP}

Photon-counting analogue of [`EICTPipeline`](@ref): the five-struct build of
the PCCT chain (spectral bins → per-bin log sinograms → optional pile-up →
bin combine) and the FDK plan, with the same batching contract.  Fields as
[`EICTPipeline`](@ref) plus `pcct::PCCTPlan`, `bt` (per-pixel spectral bowtie
or `nothing`), `G`/`I0_groups` (bin-combine matrix and per-group air counts)
and `groups` (host description of the combine).
"""
struct PCCTPipeline{T <: AbstractFloat, PP, FP} <: AbstractPipeline{T}
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

