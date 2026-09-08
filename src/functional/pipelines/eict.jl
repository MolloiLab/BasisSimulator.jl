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
    batching::NamedTuple{(:dd, :spectral, :fdk, :dense, :col_block, :slab_block), Tuple{Int, Int, Int, Bool, Int, Int}}   # views per batch, per stage
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
         col_block = geom.n_cols, slab_block = max(vol_shape[1], vol_shape[2]))
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
function simulate_sino(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline, ε = nothing, ε_e = nothing)
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

