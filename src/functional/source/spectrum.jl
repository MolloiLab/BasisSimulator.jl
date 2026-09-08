# =============================================================================
# Source side of the chain as the detector sees it: the APPLIED spectrum of a
# plan — incident photons per ray × source spectrum × detector efficiency ×
# (per-pixel bowtie / heel transmission) on the plan's energy grid.  These
# per-channel tables are what the n-channel VMI estimator (vmi/nchannel.jl)
# takes as its forward model, exactly as the notebooks build them.
# =============================================================================

"""
    applied_spectrum(plan::EICTPlan; T = eltype(plan)) -> Φ :: (n_col, n_row, n_E)

Per-ray applied spectrum `Φ[c, r, e] = I0 · wη[e] · bt[c, r, e]` (`bt = 1` when
the scanner has no bowtie/heel weighting) on `plan.energies`; the air value of
a ray is `Σ_e Φ[c, r, e]`.
"""
function applied_spectrum(plan::EICTPlan; T::Type{<:AbstractFloat} = _scalar_type(plan.wη))
    isempty(plan.energies) && throw(ArgumentError("applied_spectrum: the EICT plan carries no energy grid"))
    n_col, n_row, _ = plan.sino_shape
    w = reshape(Vector{T}(plan.wη), 1, 1, :)
    b = plan.bt === nothing ? ones(T, n_col, n_row, 1) : Array{T, 3}(plan.bt)
    return Array{T, 3}(T(plan.I0) .* w .* b)
end

"""
    applied_spectrum(plan::PCCTPlan) -> Φ :: (n_E, n_bins)

Energy → bin applied weights of a photon-counting plan (`I0 · w(E) · η(E) · R(E, b)`,
centre-pixel bowtie folded in), i.e. the global tables of the nb04 estimator.
"""
applied_spectrum(plan::PCCTPlan) = Matrix(plan.W)
