"""
    BasisSimulator.Functional

Pure, mutation-free, array-generic tensor-program core of BasisSimulator, laid
out along the imaging chain.  Every stage is a *function of tensors*: no in-place
mutation, no scalar indexing in the per-view kernels, no data-dependent loops;
index arithmetic is broadcast and memory access is a static-tap gather, so the
same source runs on plain `Array`s and traces under `Reactant.@compile` into one
XLA program that `Enzyme` differentiates end to end.  The legacy kernels in
`src/projection/`, `src/reconstruction/`, … remain the numerical **oracle**:
every stage ships with a parity test (`test/functional/`).

Layout (one folder per part of the chain; files are included in this order):

    core/            loop.jl        compiled view loops (`_batched_loop`, `_dslice`, `_dupdate`, `_zeros`)
                     resample.jl    overlap / gather / window helpers shared by the projector and FDK
                     device.jl      backend hooks (`_iota`, `_on_device`, plan lifting)
    source/          spectrum.jl    applied-spectrum tables of a plan (I0 · w(E) · η(E) · bowtie): the
                                    per-channel Φ that the n-channel decomposition (reconstruction/vmi) consumes
    projection/      dd_projector.jl  distance-driven forward projector + exact transpose (per view)
                     dd_runs.jl       view runs: batched + compiled-loop projection, ordered subsets, HIR operators
                     dd_dense.jl      the same DD physics as two batched contractions (the fast path on CPU and GPU)
    detector/        eict.jl        energy-integrating chain (spectral sum, fill factor, scatter, noise,
                                    air normalisation, −log, water BHC) + hand VJPs
                     pcct.jl        photon-counting chain (spectral bins, DRM/LUT weights, Poisson counts and
                                    surrogate, pile-up, bin combine) + hand VJPs
    reconstruction/  fbp.jl         FDK (cosine weights, ramp filter, backprojection, FOV mask)
                     fbp_dense.jl   the FDK backprojection as dense windowed contractions (the fast path: a contraction's reverse is a contraction)
                     hir.jl         OS-PWLS hybrid iterative reconstruction (operators passed in)
                     denoising.jl   ACNR (Kalender), sinogram SVD-bilateral, median-z, SF-JSD
                     vmi/           virtual monoenergetic imaging = material decomposition + synthesis
                       nchannel.jl    THE decomposition path: the published n-channel profiled-likelihood
                                      estimator, T-LBF, angular apodization, per-basis FDK, synthesis
                       cong_cmv.jl    projection-domain two-material alternatives (Cong, CMV), for comparison
    pipelines/       common.jl      fractions → path lengths, memory-budget batching, plan lifting
                     eict.jl        `eict_pipeline` / `eict_forward`  (five structs → HU)
                     pcct.jl        `pcct_pipeline` / `pcct_forward`  (five structs → per-channel μ)
                     vmi.jl         `vmi_pipeline` / `vmi_forward`    (five structs, K protocols → VMI stack)
                     api.jl         `pipeline(phantom, scanner, …)` / `forward(fractions, pipe)` dispatch

Nothing is exported; call `BasisSimulator.Functional.f` (alias `BSF` in tests and
notebooks).  Plan and status board: `design/reactant/README.md`; how to run:
`design/reactant/HOWTO.md`.
"""
module Functional

using ..BasisSimulator: CTGeometry, is_arc, is_helical
import ..BasisSimulator as BS
using LinearAlgebra: LinearAlgebra
using Random: Random

include("core/loop.jl")
include("core/resample.jl")
include("core/device.jl")
include("projection/dd_projector.jl")
include("projection/dd_runs.jl")
include("projection/dd_dense.jl")
include("detector/eict.jl")
include("detector/pcct.jl")
include("source/spectrum.jl")
include("reconstruction/fbp.jl")
include("reconstruction/fbp_dense.jl")
include("reconstruction/hir.jl")
include("reconstruction/denoising.jl")
include("reconstruction/vmi/cong_cmv.jl")
include("reconstruction/vmi/nchannel.jl")
include("pipelines/common.jl")
include("pipelines/eict.jl")
include("pipelines/pcct.jl")
include("pipelines/vmi.jl")
include("pipelines/api.jl")

end # module Functional
