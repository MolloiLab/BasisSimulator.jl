# =============================================================================
# Projector selection — Distance-Driven (default) vs Siddon
# =============================================================================
#
# The `dd_*` and `siddon_*` forward projectors share byte-identical signatures
# (the DD port preserved the Siddon API exactly), so a single run-level
# `projector::Symbol` knob can pick between them at every forward-projection
# site without any other code change.
#
#   :dd      — distance-driven (DD3).  DEFAULT.  Anti-aliased footprint
#              integration; robust in severe beam-hardened regions.
#   :siddon  — Siddon exact ray tracing.  ~3.5-5.5x faster on GPU, but
#              point-samples one voxel per step → ALIASES in severe
#              beam-hardened regions.  Use only when speed > accuracy.
#
# Consistency contract: the forward simulation, the iterative-recon system
# matrix (A·x and W = 1/(A·1)), and the BHC correction all read the SAME
# `projector` symbol (same `:dd` default), so the model is self-consistent —
# the recon inverts the operator that generated the data.  The voxel-driven
# back-projector (`backproject!`) has no DD/Siddon variant and is unchanged.
#
# These helpers branch on a runtime Symbol (one cheap comparison) rather than
# `Val` dispatch: they are called per-energy / per-tile / per-subset, never in
# the per-voxel hot loop, so the branch cost is negligible and call sites stay
# readable.
# =============================================================================

"""
    _validate_projector(p::Symbol) -> Symbol

Throw an `ArgumentError` unless `p` is `:dd` or `:siddon`; return `p` unchanged.
Call at every public entry point that accepts a projector so an invalid symbol
fails loudly instead of silently falling back to `:dd` in the shims below.
"""
@inline function _validate_projector(p::Symbol)
    (p === :dd || p === :siddon) ||
        throw(ArgumentError("projector must be :dd or :siddon, got :$p"))
    return p
end

# In-place monochromatic forward projection.
@inline _project_mono!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_forward_project!(args...; kw...) :
                       dd_forward_project!(args...; kw...)

# Allocating monochromatic forward projection.
@inline _project_mono(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_forward_project(args...; kw...) :
                       dd_forward_project(args...; kw...)

# Fused polychromatic (energy-integrating EI).
@inline _project_fused_poly!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_fused_poly_project!(args...; kw...) :
                       dd_fused_poly_project!(args...; kw...)

# Fused spectral (photon-counting PCCT, tiled).
@inline _project_fused_spectral!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_fused_spectral_project!(args...; kw...) :
                       dd_fused_spectral_project!(args...; kw...)
