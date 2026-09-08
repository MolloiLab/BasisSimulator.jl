# =============================================================================
# Dual-kVp (n-channel) VMI behind the five structs — the notebook-03 chain as one
# pure program: per-kVp EICT pipelines (no BHC) → channel log-transmissions →
# published n-channel estimator → per-basis FDK (looped) → ACNR → VMI synthesis.
# =============================================================================

"""
    VMIPipeline{T}

One [`EICTPipeline`](@ref) per kVp channel (built without BHC), the
[`NChannelPlan`](@ref) derived from their applied spectra (per-ray tables
`I0·wη·bt`), the per-basis FDK plans (iodine / water kernels), the ACNR plan
(or `nothing`), the VMI energies and their synthesis weights.
"""
struct VMIPipeline{T <: AbstractFloat, EP, NP, FP, AP}
    pipes::Vector{EP}
    nchannel::NP
    fbp_iodine::FP
    fbp_water::FP
    acnr::AP
    energies::Vector{Float64}
    alphas::Vector{T}
    recon_shape::NTuple{3, Int}
end
Base.eltype(::VMIPipeline{T}) where {T} = T

# nb03 §04 per-basis apodization kernels
const VMI_IODINE_FILTER = BS.CustomFilter((0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 0.40, 0.12, 0.03, 0.001))
const VMI_WATER_FILTER = BS.CustomFilter((0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 0.8744, 0.6003, 0.3031, 0.0266))

"""
    vmi_pipeline(phantom, scanner::EICTScanner, protocols, sim_opts, recon_opts;
                 energies = [50.0, 70.0, 100.0, 140.0], iodine_filter = VMI_IODINE_FILTER,
                 water_filter = VMI_WATER_FILTER, acnr = ACNRKalenderPlan(T; hp_sigma_px = 1.5, window = 4, passes = 5, beta_max = 14.0),
                 T = Float32, view_batch = :auto, batch_budget_mb = 512, nchannel_kwargs...) -> VMIPipeline

The dual-kVp (or any `K`-channel) VMI chain of notebook 03 behind the five
structs: one `CTProtocol` per channel.  Channel tables are the pipelines' own
applied spectra (per-ray `Φ_k = I0_k · wη_k · bt_k`, air `I0_k = Σ_E Φ_k`),
merged on the union energy grid ([`nchannel_merge_channels`](@ref)).  Run with
[`vmi_forward`](@ref).
"""
function vmi_pipeline(
        phantom::BS.Phantom, scanner::BS.EICTScanner, protocols::AbstractVector, sim_opts, recon_opts;
        energies::AbstractVector{<:Real} = [50.0, 70.0, 100.0, 140.0],
        iodine_filter = VMI_IODINE_FILTER, water_filter = VMI_WATER_FILTER,
        acnr = :default,
        T::Type{<:AbstractFloat} = Float32,
        view_batch::Union{Symbol, Integer} = :auto, batch_budget_mb::Real = 512, loop::Bool = true,
        nchannel_kwargs...,
    )
    length(protocols) >= 2 || throw(ArgumentError("vmi_pipeline: at least two channels (protocols) are required"))
    pipes = [eict_pipeline(phantom, scanner, pr, sim_opts, recon_opts; bhc = nothing, T, view_batch, batch_budget_mb, loop) for pr in protocols]
    geom = pipes[1].geom
    all(p -> p.geom.n_angles == geom.n_angles && p.eict.sino_shape == pipes[1].eict.sino_shape, pipes) ||
        throw(ArgumentError("vmi_pipeline: every channel must share the geometry (views, detector)"))
    Φs = [applied_spectrum(p.eict; T) for p in pipes]                     # per-ray (n_col, n_row, nE_k)
    Ee, Φe = nchannel_merge_channels([p.eict.energies for p in pipes], Φs)
    I0 = cat((dropdims(sum(Φ; dims = 3); dims = 3) for Φ in Φs)...; dims = 3)   # (n_col, n_row, K)
    nplan = nchannel_plan(Φe, Ee, I0; T, nchannel_kwargs...)
    fbp_I = fbp_plan(geom, recon_opts.matrix_size; filter = iodine_filter, T)
    fbp_W = fbp_plan(geom, recon_opts.matrix_size; filter = water_filter, T)
    aplan = acnr === :default ? ACNRKalenderPlan(T; hp_sigma_px = 1.5, window = 4, passes = 5, beta_max = 14.0) : acnr
    es = Vector{Float64}(energies)
    return VMIPipeline{T, eltype(pipes), typeof(nplan), typeof(fbp_I), typeof(aplan)}(
        pipes, nplan, fbp_I, fbp_W, aplan, es, nchannel_vmi_alphas(es; T), recon_opts.matrix_size)
end

"""
    vmi_forward(fractions, vpipe::VMIPipeline, noises = nothing) -> (; vmis, vol_water, vol_iodine, sino_iodine, sino_water, h)

The complete pure VMI model: fractions → per-channel log-transmissions (no
BHC) → n-channel basis sinograms → per-basis FDK → ACNR → VMI HU stack
`(nx, ny, nz, length(energies))`.  `noises`, when given, is a vector of
`(ε, ε_e)` per channel (see [`draw_eict_noise`](@ref)).
"""
function vmi_forward(fractions::AbstractArray{<:Any, 4}, vp::VMIPipeline{T}, noises = nothing) where {T}
    hs = map(enumerate(vp.pipes)) do (k, p)
        ε, ε_e = noises === nothing ? (nothing, nothing) : noises[k]
        simulate_sino(fractions, p, ε, ε_e)
    end
    h = reduce((a, b) -> cat(a, b; dims = 4), map(x -> reshape(x, size(x)..., 1), hs))   # (n_col, n_row, n_view, K)
    vb_I = vp.pipes[1].unrolled ? 1 : vp.pipes[1].batching.fdk
    # rays whose estimate did not converge (non-finite) carry zero basis density into the FDK,
    # like the notebooks' quality-flagged rays; the FDK would otherwise spread one NaN everywhere
    finite0(x) = (y = _plain(x); ifelse.(isfinite.(y), y, zero(T)))
    lp = vp.pipes[1].loop
    fbp_iodine = s -> fdk(finite0(s), _fbp_on_device(vp.fbp_iodine, s); view_batch = vb_I, loop = lp)
    fbp_water = s -> fdk(finite0(s), _fbp_on_device(vp.fbp_water, s); view_batch = vb_I, loop = lp)
    acnr = vp.acnr === nothing ? nothing : ((W, I) -> (r = acnr_kalender(W, I, vp.acnr); (; water = r[1], iodine = r[2])))
    synth = (W, I, es) -> nchannel_synth_vmi(W, I, _on_device(vp.alphas, W), T)
    c = nchannel_vmi_chain(h, vp.nchannel; fbp_iodine, fbp_water, acnr, synth, energies = vp.energies)
    return (; vmis = c.vmis, vol_water = c.vol_water, vol_iodine = c.vol_iodine,
        sino_iodine = c.sino_iodine, sino_water = c.sino_water, h)
end
