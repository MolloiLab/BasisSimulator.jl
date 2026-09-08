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
"""
function eict_pipeline(
        phantom::BS.Phantom, scanner, protocol, sim_opts, recon_opts;
        bhc = :water,
        filter = BS.StandardFilter(),
        cutoff::Real = 1.0,
        T::Type{<:AbstractFloat} = Float32,
        spectrum_override = nothing,
    )
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom; T, spectrum_override)
    geom = ws.geom
    model = bhc === :water ? BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom) : bhc
    eplan = eict_plan(ws, protocol, sim_opts; bhc = model)
    fplan = fbp_plan(geom, recon_opts.matrix_size; filter, cutoff, T)
    μw = model === nothing ? BS.get_reference_μ_water(70.0) : model.μ_water_ref
    n_mat = size(eplan.μ_tbl, 1)
    return EICTPipeline{T, typeof(eplan), typeof(fplan)}(
        geom, size(phantom.mask), n_mat, phantom.extent, eplan, fplan, T(μw), recon_opts.matrix_size)
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
    material_paths(fractions, pipe) -> (n_col, n_row, n_view, n_mat)

Per-material path lengths `P[:, :, :, m] = A(fractions[:, :, :, m])` with the
functional DD projector (one static-tap projection per material per view; the
material axis is the batch axis of the linear operator). Differentiable in
`fractions`.
"""
function material_paths(fractions::AbstractArray{<:Any, 4}, pipe::EICTPipeline{T}) where {T}
    geom = pipe.geom
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
        bc, p.eps, p.sino_shape)
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
    return eict_chain(P, _eict_on_device(pipe.eict, P), ε, ε_e)
end

"""
    reconstruct_μ(sino, pipe) -> (nx, ny, nz)

Pure equivalent of `reconstruct!(create_fdk_recon_workspace(sino, geom, matrix))`.
"""
reconstruct_μ(sino::AbstractArray{<:Any, 3}, pipe::EICTPipeline) = fdk(sino, _fbp_on_device(pipe.fbp, sino))

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
