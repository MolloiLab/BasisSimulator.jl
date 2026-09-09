# =============================================================================
# Energy-integrating pipeline behind the five structs:
#   fractions → DD path lengths → EICT chain (+ water BHC) → FDK → HU.
# =============================================================================

"""
    EICTPipeline{T}

Host-side bundle of plans for the EICT phantom → HU pipeline. Fields:
`geom`, `vol_shape` (phantom mask shape), `n_mat` (rows of the μ table = phantom
materials, in `phantom.materials` order), `volume_extent` (phantom extent, cm),
`eict::EICTPlan`, `fbp::FBPPlan`, `μ_water` (HU reference), `recon_shape`.
"""
struct EICTPipeline{T <: AbstractFloat, EP, FP} <: AbstractPipeline{T}
    geom::BS.CTGeometry
    vol_shape::NTuple{3, Int}
    n_mat::Int
    volume_extent::NTuple{3, Float64}
    eict::EP
    fbp::FP
    μ_water::T
    recon_shape::NTuple{3, Int}
    batching::NamedTuple{(:dd, :spectral, :fdk, :dense, :col_block, :slab_block, :mat), Tuple{Int, Int, Int, Bool, Int, Int, Int}}   # views per batch, per stage
    unrolled::Bool                                                       # true: legacy per-view unrolled programs (view_batch = 1)
    loop::Bool                                                           # true: batches run inside compiled while loops; false: batches unrolled
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
views per batch in all stages, and `1` selects the legacy per-view
unrolled programs (bit-identical summation order; toy sizes only). `loop = false` unrolls the
batches instead of looping (program size ∝ number of batches; the faster execution on XLA CPU).
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
        loop::Bool = true,
        projector::Symbol = :dense,      # :dense (batched contractions — THE projector) | :gather (static-tap gathers; oracle/tests only)
    )
    projector in (:dense, :gather) || throw(ArgumentError("projector must be :dense or :gather"))
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
        _auto_batching(eplan.sino_shape, vol_shape, length(eplan.wη), n_mat, recon_opts.matrix_size, batch_budget_mb; projector, geom, volume_extent = phantom.extent)
    else
        (dd = Int(view_batch), spectral = Int(view_batch), fdk = Int(view_batch), dense = projector === :dense,
         col_block = geom.n_cols, slab_block = max(vol_shape[1], vol_shape[2]), mat = n_mat)
    end
    return EICTPipeline{T, typeof(eplan), typeof(fplan)}(
        geom, vol_shape, n_mat, phantom.extent, eplan, fplan, T(μw), recon_opts.matrix_size,
        batching, view_batch === 1, loop)
end

"""
    simulate_sino(fractions, pipe, ε = nothing, ε_e = nothing) -> (n_col, n_row, n_view)

Pure equivalent of `simulate!(EICTWorkspace)` followed by `apply_bhc_water`
(when the pipeline carries a BHC): the BHC-corrected log sinogram. `ε`, `ε_e`
are the quantum / electronic N(0,1) noise tensors (flat or 3-D); see
[`draw_eict_noise`](@ref).
"""
function simulate_sino(fractions::PipelineInput, pipe::EICTPipeline, ε = nothing, ε_e = nothing)
    P = material_paths(fractions, pipe)
    return eict_chain(P, _eict_on_device(pipe.eict, P), ε, ε_e; view_batch = pipe.unrolled ? 0 : pipe.batching.spectral, loop = pipe.loop)
end

"""
    reconstruct_μ(sino, pipe) -> (nx, ny, nz)

Pure equivalent of `reconstruct!(create_fdk_recon_workspace(sino, geom, matrix))`.
"""
reconstruct_μ(sino::AbstractArray{<:Any, 3}, pipe::EICTPipeline) =
    fdk(sino, _fbp_on_device(pipe.fbp, sino); view_batch = pipe.unrolled ? 1 : pipe.batching.fdk, loop = pipe.loop)

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
function eict_forward(fractions::PipelineInput, pipe::EICTPipeline, ε = nothing, ε_e = nothing)
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
# View batches as DATA — the host-composed gradient
# -----------------------------------------------------------------------------
#
# The reconstructed volume is a SUM over view batches of independent contributions
# (DD → chain → ramp filter → weighted backprojection are all per view), so
#
#     vol = Σ_b eict_batch_vol(fractions, b)      and      ∂⟨v̄, vol⟩/∂fractions = Σ_b ∂⟨v̄, vol_b⟩/∂fractions.
#
# Each term is one FIXED-SIZE program without loops: compiled once per (orientation, batch length)
# and called from the host for every batch — no while-loop reverse, no checkpoint recomputation,
# the memory of one batch. A batch's per-view tables are passed as DATA (`batch_data`) so the
# compiled program is reused across batches. Measured in PROBES §9.

"""
    EICTBatch — one view batch of an `EICTPipeline` (host description: the views, the DD run plan
    built for them, the window blocks of the windowed projector). The per-view arrays live in
    [`batch_data`](@ref).
"""
struct EICTBatch{R <: DDRunPlan}
    views::UnitRange{Int}
    run::R                                   # its `table` is not used by the batch program (data instead)
    windowed::Bool
    widths::Vector{Int}
    col_blocks::Vector{UnitRange{Int}}
    slab_blocks::Vector{UnitRange{Int}}
    starts::Array{Int32, 3}                  # (n_col_blocks, n_slab_blocks, B) window starts (windowed only)
end

"""
    eict_batches(pipe::EICTPipeline) -> Vector{EICTBatch}

The pipeline's view batches (`batching.dd` views each, split at orientation changes of the DD
runs; the windowed projector's block widths are shared across a run so its batches compile to
the same program).
"""
function eict_batches(pipe::EICTPipeline{T}) where {T}
    pipe.batching.dense || throw(ArgumentError("eict_batches: the batch programs use the dense projector (pipeline built with projector = :gather)"))
    geom = pipe.geom
    B = pipe.batching.dd
    cb, sb = pipe.batching.col_block, pipe.batching.slab_block
    windowed = cb < geom.n_cols || sb < max(pipe.vol_shape[1], pipe.vol_shape[2])
    out = EICTBatch[]
    for run in dd_run_plans(geom, pipe.vol_shape; view_batch = B, volume_extent = pipe.volume_extent, eltype = T)
        starts_run, widths, col_blocks, slab_blocks = windowed ?
            dd_windows(geom, pipe.vol_shape, collect(run.views), cb, sb; volume_extent = pipe.volume_extent, eltype = T) :
            (zeros(Int32, 0, 0, 0), Int[], UnitRange{Int}[], UnitRange{Int}[])
        for lo in first(run.views):B:last(run.views)
            views = lo:min(lo + B - 1, last(run.views))
            r = only(dd_run_plans(geom, pipe.vol_shape; view_batch = length(views), views, volume_extent = pipe.volume_extent, eltype = T))
            loc = (first(views) - first(run.views) + 1):(last(views) - first(run.views) + 1)
            push!(out, EICTBatch(views, r, windowed, widths, col_blocks, slab_blocks, windowed ? starts_run[:, :, loc] : starts_run))
        end
    end
    return out
end

"""
    batch_data(b::EICTBatch, pipe) -> NamedTuple (table, starts, gt, off)

The batch's per-view arrays: the DD view table `(18, B)`, the window starts `(n_blocks, B)`
Int32, the FDK geometry columns `(12, B)`, the view offsets `(1,1,1,B)` Int32. Host arrays;
under Reactant pass `Reactant.to_rarray(batch_data(b, pipe))` to the compiled program.
"""
function batch_data(b::EICTBatch, pipe::EICTPipeline)
    B = length(b.views)
    block = pipe.fbp.n_col * pipe.fbp.n_row
    return (table = b.run.table,
            starts = b.windowed ? reshape(b.starts, :, B) : zeros(Int32, 0, B),
            gt = _geom_table(pipe.fbp.tensors)[:, b.views],
            off = reshape(Int32[(k - 1) * block for k in 1:B], 1, 1, 1, B))
end

"""
    eict_batch_vol(fractions, pipe, b::EICTBatch, d, ε_b = nothing, ε_e_b = nothing) -> (nx, ny, nz)

The contribution of view batch `b` (arrays `d = batch_data(b, pipe)`) to the backprojected μ
volume before the FOV mask: DD → spectral chain (+BHC) → ramp filter → FDK-weighted
backprojection, no loops. `ε_b`, `ε_e_b` are the batch's slices of the noise tensors.
`Σ_b eict_batch_vol(...)` then [`eict_vol_to_hu`](@ref) reproduces [`eict_forward`](@ref).
"""
function eict_batch_vol(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline{T}, b::EICTBatch, d, ε_b = nothing, ε_e_b = nothing) where {T}
    P = b.windowed ?
        dd_project_dense_windowed_run(fractions, b.run, pipe.geom; col_block = pipe.batching.col_block, slab_block = pipe.batching.slab_block,
            volume_extent = pipe.volume_extent, unroll = true, table = d.table, windows = (d.starts, b.widths, b.col_blocks, b.slab_blocks)) :
        dd_project_dense_run(fractions, b.run; unroll = true, table = d.table)
    s = eict_chain(P, _eict_on_device(pipe.eict, P), ε_b, ε_e_b; view_batch = 0)
    fplan = _fbp_on_device(pipe.fbp, s)
    filt = filter_views(s, fplan)
    vol = _bp_chunk(vec(filt), fplan, d.gt, d.off, true)                     # (nx, ny, nz, 1)
    return dropdims(vol; dims = 4) .* fplan.pi_over_angles
end

"""
    eict_vol_to_hu(vol, pipe) -> HU

FOV mask (sentinel outside) and HU conversion of a summed batch volume.
"""
eict_vol_to_hu(vol::AbstractArray{<:Any, 3}, pipe::EICTPipeline) = to_hu(fov_mask(vol, _fbp_on_device(pipe.fbp, vol)), pipe)

"""
    eict_forward_batched(fractions, pipe, ε = nothing, ε_e = nothing) -> HU

Host reference of the batch composition (`Σ_b eict_batch_vol` → `eict_vol_to_hu`); equals
[`eict_forward`](@ref) up to summation order. Under Reactant compile `eict_batch_vol` once per
(orientation, batch length) and loop on the host (see `design/reactant/probes/bench_host_grad.jl`).
"""
function eict_forward_batched(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline{T}, ε = nothing, ε_e = nothing) where {T}
    bs = eict_batches(pipe)
    sl(x, v) = x === nothing ? nothing : reshape(x, pipe.eict.sino_shape)[:, :, v]      # noise tensors may be flat
    vol = sum(eict_batch_vol(fractions, pipe, b, batch_data(b, pipe), sl(ε, b.views), sl(ε_e, b.views)) for b in bs)
    return eict_vol_to_hu(vol, pipe)
end

# -----------------------------------------------------------------------------
# The compiled driver (implemented by the Reactant extension)
# -----------------------------------------------------------------------------
#
# One compiled forward and one compiled pullback program per (orientation, batch length), each
# taking its ACCUMULATOR as an argument (a `.+` on concrete device arrays outside a program falls
# to element-wise host execution), called from the host over the view batches. Measured on XLA:CPU
# (PROBES §9): 256 grid / 984 views forward 14 s (legacy 33 s), gradient step 57 s.

"""
    CompiledEICT — an `EICTPipeline` compiled for a device (`compile_pipeline`): the batch
    programs keyed by (orientation, batch length), the per-batch data on the device, the HU map
    and its pullback.
"""
struct CompiledEICT{P <: EICTPipeline, B, D, F, V, Z, H, VB}
    pipe::P
    batches::Vector{B}
    data::Vector{D}
    fwd::Dict{Tuple{Bool, Int}, F}        # (acc, x, d) -> acc .+ eict_batch_vol(x, b, d)
    vjp::Dict{Tuple{Bool, Int}, V}        # (acc, x, v̄, d) -> acc .+ ∂⟨v̄, vol_b⟩/∂x
    zero_vol::Z                           # fresh device zeros for the volume / the gradient
    zero_grad::Z
    tohu::H                               # vol -> HU
    vbar::VB                              # (vol, hū) -> ∂⟨hū, HU(vol)⟩/∂vol
end

"""
    compile_pipeline(pipe::EICTPipeline, fractions) -> CompiledEICT

Compile the pipeline's batch programs for the device of `fractions` (a `Reactant` array; the
noise-free chain). Requires the Reactant extension.
"""
function compile_pipeline end

"""
    forward(cp::CompiledEICT, fractions) -> HU

The compiled forward: `Σ_b` batch volumes on the device, then the HU map.
"""
function forward(cp::CompiledEICT, x)
    vol = _batch_volume(cp, x)
    return cp.tohu(vol, cp.pipe)
end

"""
    pullback(cp::CompiledEICT, fractions) -> (HU, back)

The compiled forward and its pullback: `back(hu_bar)` returns `∂⟨hu_bar, HU⟩/∂fractions` — the
sum over view batches of the compiled per-batch pullbacks, reusing the forward's volume.
"""
function pullback(cp::CompiledEICT, x)
    vol = _batch_volume(cp, x)
    hu = cp.tohu(vol, cp.pipe)
    back = hu_bar -> begin
        vb = cp.vbar(vol, hu_bar)
        g = cp.zero_grad()
        for (b, d) in zip(cp.batches, cp.data)
            g = cp.vjp[_batch_key(b)](g, x, vb, d)
        end
        g
    end
    return hu, back
end

_batch_key(b::EICTBatch) = (b.run.vertical, length(b.views))

function _batch_volume(cp::CompiledEICT, x)
    vol = cp.zero_vol()
    for (b, d) in zip(cp.batches, cp.data)
        vol = cp.fwd[_batch_key(b)](vol, x, d)
    end
    return vol
end
