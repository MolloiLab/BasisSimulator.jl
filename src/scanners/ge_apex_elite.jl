"""
Scanner constants for the GE Revolution Apex Elite dual-energy CT.

Exposes:
- `GE_APEX_ELITE_FILTERS` — Dict of named FBP apodization filters, each a
  `CustomFilter` with piecewise-linear control points on normalized frequency
  `[0, 1]`.  Notebooks pick via `BS.GE_APEX_ELITE_FILTERS[:standard]` etc.

Control points were hand-tuned in §10 of
`verification/notebooks/06_ge_apex_elite_clinical.jl` to approximate the GE
STANDARD reconstruction kernel's MTF.  Full MTF calibration against clinical
DICOMs is a separate task — these are starting points good enough for
simulation work.

To add a new GE kernel, append an entry to the Dict below with the kernel
name as the Symbol key (lowercase, e.g. `:soft`, `:detail`, `:lung`,
`:bone_plus`, `:asirv50`) and a matching `CustomFilter`.
"""

const GE_APEX_ELITE_FILTERS = Dict{Symbol, CustomFilter}(
    # GE STANDARD kernel — balanced soft-tissue recon.  Control points chosen
    # to roughly match the STANDARD MTF (soft roll-off through mid frequency,
    # aggressive cutoff past 0.75×Nyquist).  Used as the default baseline in
    # 06 throughout §5 (basis FBP), §6 (VMI FBP), and §7 (scatter recon).
    :standard => CustomFilter(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (1.0, 0.85, 0.6,  0.15, 0.001),
    ),
)

export GE_APEX_ELITE_FILTERS
