"""
    BasisSimulator.Functional

Pure, mutation-free, array-generic tensor-program core of BasisSimulator.

Every stage in this module is a *function of tensors*: no in-place mutation of
inputs, no scalar indexing inside the per-view kernels, no data-dependent loops.
Index arithmetic is broadcast (`floor.`, `clamp.`, `ifelse.`) and memory access
is a gather (`vec(A)[idx]`) with a *static* number of taps derived from the
geometry on the host, so the same source runs on plain `Array`s, on GPU arrays
via broadcasting, and — the point — traces under `Reactant.@compile` into one
XLA program that `Enzyme` differentiates end to end.

The legacy AcceleratedKernels kernels in `src/projection/`, `src/reconstruction/`
etc. remain the numerical **oracle**: each functional stage ships with a parity
test against them (see `test/functional/`) so the physics is provably unchanged.

Nothing is exported; call `BasisSimulator.Functional.f` (alias `BSF` in tests).

Stages present (see design/reactant/README.md for the plan and status board):
  * `dd_view_plan`, `dd_project_view`, `dd_transpose_view`, `dd_project`,
    `dd_transpose` — distance-driven (DD3) mono forward projector and its exact
    transpose as static-tap box-overlap resampling (`dd_projector.jl`).
  * `fbp_plan`, `filter_views`, `backproject`, `fov_mask`, `fdk` — FBP/FDK (`fbp.jl`).
  * `EICTPlan`/`eict_plan`, `poly_log_sinogram`, `eict_noise`, `bhc_apply`,
    `eict_chain` (+ hand-derived VJPs) — energy-integrating detector chain (`eict.jl`).
  * `PCCTPlan`/`pcct_plan`, `spectral_bin_intensities`, `pcct_chain` — photon-counting
    chain (`pcct.jl`).
  * `HIRPlan`/`hir_plan`, `huber_gradient`, `hir_reconstruct` — OS-PWLS hybrid IR (`hir.jl`).
  * `sino_svd_denoise_bilateral`, `acnr_kalender`, `median_z`, `sfjsd_denoise` (`denoise.jl`).
  * `cong_decompose`, `cmv_decompose`, `synth_vmi_2basis` — projection-domain VMI (`vmi.jl`).
  * `nchannel_*` — the published n-channel profiled-likelihood estimator (`nchannel.jl`).
"""
module Functional

using ..BasisSimulator: CTGeometry, is_arc, is_helical
import ..BasisSimulator as BS
using LinearAlgebra: LinearAlgebra
using Random: Random

# Operators
include("resample.jl")
include("dd_projector.jl")
include("fbp.jl")
# Detector physics chains (per-material path lengths → log sinograms)
include("eict.jl")
include("pcct.jl")
# Reconstruction
include("hir.jl")
# Denoising / ACNR
include("denoise.jl")
# Projection-domain VMI (Cong, CMV, synthesis)
include("vmi.jl")
# The published n-channel profiled-likelihood VMI estimator (ported from the notebooks)
include("nchannel.jl")
# End-to-end pipelines behind the five-struct API
include("pipeline.jl")

end # module Functional
