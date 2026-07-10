# =============================================================================
# Projector selection — Distance-Driven (default) vs Siddon
# =============================================================================
#
# The `dd_*` and `siddon_*` forward projectors share byte-identical signatures
# (the DD port preserved the Siddon API exactly), so a single run-level
# `projector::Symbol` knob can pick between them at every forward-projection
# site without any other code change.
#
#   :dd      — reference distance-driven (DD3) implementation.  Anti-aliased
#              footprint integration; deprecated in favor of `:dd_fast`.
#   :dd_fast — DEFAULT.  Same DD3 model, single-pass per-material path-length fused
#              kernels: identical footprint/overlap weights (results agree with
#              :dd to float ordering), but the full spectrum is produced in ONE
#              volume walk instead of the K=16 tiled re-walks.  Measured 47x
#              faster on the 234-bin polychromatic forward path (M4 Metal).
#              Mono projection is byte-identical to :dd (same kernel).
#              Requires ≤ 32 materials (falls back to :dd kernels above that).
#   :siddon  — Siddon exact ray tracing retained for comparison and
#              compatibility. It point-samples one voxel per step, can ALIAS
#              in severe beam-hardened regions, and is slower than :dd_fast
#              for full polychromatic/spectral simulations.
#
# Consistency contract: the forward simulation, the iterative-recon system
# matrix (A·x and W = 1/(A·1)), and the BHC correction all read the SAME
# `projector` symbol (same `:dd_fast` default), so the model is self-consistent —
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

Throw an `ArgumentError` unless `p` is `:dd`, `:dd_fast`, or `:siddon`; return
`p` unchanged.  Call at every public entry point that accepts a projector so an
invalid symbol fails loudly instead of silently falling back to `:dd` in the
shims below.
"""
function _validate_projector(p::Symbol)
    (p === :dd || p === :dd_fast || p === :siddon) ||
        throw(ArgumentError("projector must be :dd, :dd_fast, or :siddon, got :$p"))
    p === :dd && @warn(
        "projector = :dd is DEPRECATED; use :dd_fast (the new default) — same distance-driven " *
        "physics, results agree to floating-point ordering, ~47× faster polychromatic forward. " *
        ":dd remains as the reference kernel but may be removed in a future release.",
        maxlog = 1,
    )
    return p
end

# In-place monochromatic forward projection.  :dd_fast has no mono variant —
# mono has no energy loop, so it IS the :dd kernel.
@inline _project_mono!(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_forward_project!(args...; kw...) :
                       dd_forward_project!(args...; kw...)

# Allocating monochromatic forward projection.
@inline _project_mono(proj::Symbol, args...; kw...) =
    proj === :siddon ? siddon_forward_project(args...; kw...) :
                       dd_forward_project(args...; kw...)

# Fused polychromatic (energy-integrating EI).
@inline _project_fused_poly!(proj::Symbol, args...; kw...) =
    proj === :siddon  ? siddon_fused_poly_project!(args...; kw...) :
    proj === :dd_fast ? dd_fast_fused_poly_project!(args...; kw...) :
                        dd_fused_poly_project!(args...; kw...)

# Fused spectral (photon-counting PCCT, tiled).
@inline _project_fused_spectral!(proj::Symbol, args...; kw...) =
    proj === :siddon  ? siddon_fused_spectral_project!(args...; kw...) :
    proj === :dd_fast ? dd_fast_fused_spectral_project!(args...; kw...) :
                        dd_fused_spectral_project!(args...; kw...)
