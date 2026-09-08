# Prior Reactant.jl / Enzyme.jl / TIGRE attempts in BasisSimulator.jl

Survey of `origin/tigre-reactant-rewrite`, `origin/GPU-optimize`, `origin/speed/fused-projection`,
current `origin/main`, and the local Julia depot. Read-only git archaeology.

---

## Two separate attempts, not one

The branch name is misleading. Reactant and Enzyme were tried at different times, in different
lineages, and only one of them ever produced working code.

The repository has **two unrelated root commits**. `git merge-base origin/main
origin/tigre-reactant-rewrite` exits 1. Roots are `e5120be` (2025-12-20, the tigre branch) and
`4c7a541` (2025-12-20, main). So no `origin/main...origin/tigre-reactant-rewrite` diff is possible.
The work was later re-committed into main's lineage under new hashes, which is why commit messages
appear in duplicate pairs, for example `e5baef7` and `3b86676`.

### Attempt A — Reactant / XLA

Lived on `origin/tigre-reactant-rewrite` (107 commits, 2025-12-20 to 2026-01-15) and never reached
main.

| Event | Commit | Date |
|---|---|---|
| Reactant added to Project.toml | `56ce5e2` | 2026-01-12 |
| Rewritten to avoid allowscalar | `1172b04` | 2026-01-12 |
| XLA FDK landed | `53faa3d` | 2026-01-12 |
| Reactant removed, KernelAbstractions in | `e5baef7` | 2026-01-14 |
| AcceleratedKernels replaces KA | `abd18f8` | 2026-01-14 |

Reactant survived two days. Compat was pinned `Reactant = "0.2"` at `527f087:Project.toml`.

### Attempt B — Enzyme

This is the real one, and it was on main's own lineage. `ext/BasisSimulatorEnzymeExt.jl` was added
by `bca8f51` (2026-01-17, "Phase 4-8 complete: Helical, PCCT, Iterative, Differentiable") and
deleted by `77fc7fb` (2026-03-18). It ran 2139 lines with `Enzyme = "0.13"` and a proper
`[extensions] BasisSimulatorEnzymeExt = "Enzyme"` weakdep.

Recover it with:

```
git show 77fc7fb^:ext/BasisSimulatorEnzymeExt.jl
```

It is byte-identical to the copy still living on `origin/GPU-optimize` and
`origin/speed/fused-projection`.

---

## What the Reactant code actually did

There is almost no Reactant code at the branch tip, only design comments. The peak was `527f087`.
The approach was to split every operation into a non-traced precompute pass and a traced gather
pass, because of this, from `1172b04`:

> Key insight: XLA cannot handle scalar array indexing with dynamic indices. Solution: Pre-compute
> all indices before tracing, then use linear indexing with the pre-computed index arrays during
> traced computation.

That pushed the whole cost into memory. `527f087:CLAUDE.md` states the full precomputed geometry for
a clinical scan reaches 96 trillion elements for projection and 335 billion for backprojection,
described verbatim as "impossible". Two mitigations were tried: angle-batched processing (one kernel
launch per angle, roughly 1.5 GB per angle) and a ray-marching reformulation storing only ray
origins and directions.

The only real `@trace` in the tree is `527f087:src/Forward/RayMarching.jl:662`, a loop over energy
bins inside `compute_polychromatic_transmission`. Note it allocates
`zeros(ET, n_cols, n_rows, n_angles)` inside the traced loop body, which is exactly the allocation
pattern that later became a memory problem in Pluto.

No benchmark numbers for Reactant appear anywhere in the branch. The pivot commit `abd18f8`
immediately reports concrete ones: forward projection about 2 ms after warmup, backprojection about
1.5 ms, full FDK correlation 0.968.

Two useful process notes survive in the pivot commits:

- `e5baef7`: "AcceleratedKernels.jl has compatibility issues with Metal closures, so using
  KernelAbstractions.jl @kernel macros directly."
- `abd18f8`: "Key insight: AK.foreachindex works on Metal when: 1. Code is inside typed functions
  2. All captured dimensions use Int32 (not Int64)."

---

## What the Enzyme extension actually did

It never differentiated through the ray-tracing kernels. It registered custom `EnzymeRules` that
substitute the analytic adjoint. From the file header:

```
# Key insight: Forward projection and backprojection are mathematical adjoints.
# - Gradient of forward_project w.r.t. volume = backproject
# - Gradient of backproject w.r.t. sinogram = forward_project
#
# Physics Effects Differentiability:
# - Scatter: Differentiable via convolution adjoint
# - Crosstalk: Differentiable via convolution adjoint
# - BHC: Differentiable (polynomial)
# - Filter operations: Differentiable via convolution adjoint
# - Detector noise: NOT differentiable (stochastic)
```

The reverse rule for `siddon_forward_project!` is four lines of real work. It calls
`backproject!(volume.dval, sinogram.dval, saved_geom; weighted=false)`. The `weighted=false` flag is
load-bearing: it selects the matched adjoint rather than the FDK-weighted backprojector.

### Rule inventory

The extension registered exactly four `EnzymeRules` methods, two `augmented_primal` plus two
`reverse`, at lines 256, 272, 306 and 322 of `ext/BasisSimulatorEnzymeExt.jl`. They cover only
`siddon_forward_project!` and `backproject!`. Everything else in the 2139 lines is hand-written
gradient functions that Enzyme never sees, exported under a uniform convention documented in the
file as `∂L_∂input = gradient_<effect>(∂L_∂output, input, model)`.

| Section (line) | Adjoint substituted |
|---|---|
| Forward projection rule (243) | backprojection, `weighted=false` |
| Backprojection rule (298) | forward projection |
| Scatter (664), scatter correction (786) | convolution adjoint, plus chain rule |
| Crosstalk (886), optical crosstalk (991) | convolution adjoint |
| BHC (1068) | analytic polynomial derivative |
| Ramp filter (1135), filter sinogram (1569) | convolution adjoint |
| Cosine weight (1494) | elementwise derivative |
| FDK (1670), SIRT iteration (1759) | composed from the above |

### Noise handling

Noise was declared out of scope rather than approximated. `NON_DIFFERENTIABLE_EFFECTS` lists:

| Effect | Reason given |
|---|---|
| Quantum noise | Stochastic operation (Poisson sampling) |
| Electronic noise | Stochastic operation (Gaussian sampling) |
| Detector blur | Could be differentiable but not implemented |

The documented workaround was to train on the noiseless expected-value forward model and add noise
only at inference. The reparameterization trick is named and explicitly not implemented.

### Verification helpers

`verify_gradient_scatter`, `verify_gradient_crosstalk`, `verify_gradient_bhc`,
`verify_gradient_filter`, `verify_gradient_fdk_reconstruct`, plus finite-difference sections at
lines 415 and 2047. Two container types existed, `DifferentiableCT` (line 347) and
`DifferentiableFDK` / `DifferentiableSIRT` (line 1955).

### What was actually tested, and what had rotted

The archived suite at `origin/main:test/archived/runtests.jl:5343` onward shows the state at
deletion. Only three testsets were still live: extension-loaded, the differentiability documentation
dictionaries, and the scatter gradient. Eight were commented out behind a bare `# TODO: fix`,
including forward and backward gradients, both in-place variants, adjoint consistency, gradient
chain rule, and the finite-difference verifications. One carries a specific cause:
`# TODO: fix — create_aquilion_one has been deleted`. So the tests died from unrelated API churn,
not from an Enzyme defect.

### Why `77fc7fb` deleted it

Housekeeping, not a technical verdict. The commit is titled "CLEANUP: Delete metrics, scanners,
dual_energy, Enzyme; trim deps" and removes 7019 lines across thirteen files. The Enzyme line reads
only "DELETE ext/BasisSimulatorEnzymeExt.jl (Enzyme autodiff extension)", sitting between deletions
of MTF/NPS/PSF metrics and the scanner factories. The stated rationale for the batch is "not needed
for core simulation", and the last bullet is "Comment out all broken tests (925 pass, 0 fail)". The
same commit dropped `Enzyme`, `CairoMakie`, `ArgParse`, `JSON` and `NPZ` from `[deps]` and deleted
the `[extensions]` block. No note anywhere says the adjoint rules were wrong or slow.

---

## Cross-branch grep

Every branch descending from the old root still carries the extension: `GPU-optimize`,
`speed/fused-projection`, `andy_dev`, `phantom-loading`, `phantom-loading-local`, `Reconstruction`,
and the four `hamidreza/*` branches. The copy on `GPU-optimize` is byte-identical to `77fc7fb^`.

Neither `GPU-optimize` nor `speed/fused-projection` shares a merge-base with main, so cherry-pick or
file copy is the only path.

Reactant appears on no branch except `tigre-reactant-rewrite`, and Enzyme appears nowhere inside
that branch except as aspiration in `extras/paper/manuscript.md` and the README line "Built for
Reactant/Enzyme compatibility". Enzyme was never in that branch's Project.toml.

---

## Current main and the local depot

Current main lists neither package in `Project.toml`. Three mentions remain:

- `src/geometry/scanner.jl:384`, a stale comment saying positions are precomputed "to enable
  Reactant/XLA compilation (no runtime trig)". The `CTGeometry` design is still XLA-friendly by
  accident.
- `test/archived/runtests.jl:5340`, "Enzyme.jl Differentiable CT Tests — REMOVED (Enzyme extension
  deleted)", with the whole block commented out.
- `docs/Manifest.toml`, Enzyme only as conditional extensions of ADTypes, DifferentiationInterface
  and QuadGK.

Both adjoint endpoints survive with compatible signatures:

- `siddon_forward_project!` at `src/projection/siddon.jl:457`
- `backproject!` at `src/reconstruction/core/backprojection.jl:522`, with the matched-adjoint
  contract documented at lines 484 and 520

Depot contents, cached but referenced by no environment:

| Package | Versions present | Slugs |
|---|---|---|
| Reactant | 0.2.41 to 0.2.285 | 11 |
| Enzyme | 0.13.30 to 0.13.201 | 15 |

Also present: `Reactant_jll`, `ReactantCore`, `Enzyme_jll`, `EnzymeCore`. No
`~/.julia/environments/*/Project.toml` names either package.

---

## Lessons learned / blockers to avoid

- **Never precompute indices for XLA.** Clinical scale needs 96 trillion elements forward, 335
  billion back. `527f087:CLAUDE.md` calls it "impossible". That single design choice is what killed
  the Reactant attempt; the two escape hatches (angle-batching, ray marching) each gave back the
  thing XLA was supposed to buy.
- **Recover, do not rewrite.** `git show 77fc7fb^:ext/BasisSimulatorEnzymeExt.jl` gives 2139 working
  lines. Budget the effort for re-validating the physics gradients instead, since scatter,
  crosstalk and BHC have all changed since March 2026.
- **Keep AD outside the ray tracer.** Four `EnzymeRules` methods total; everything else was
  hand-written adjoints. Both attempts converged on substituting analytic adjoints at the operator
  boundary.
- **`weighted=false` is the correctness hinge.** The FDK-weighted backprojector gives
  plausible-looking wrong gradients.
- **Stochastic surface is bigger now.** The old extension excluded Poisson and Gaussian noise; main
  injects exact integer Poisson counts by default, so the noiseless-training pattern matters more
  than it did.
- **AcceleratedKernels on Metal needs typed functions and `Int32` captures**, per `abd18f8`. Missing
  that caused a false abandonment in `e5baef7`.
- **Demand a benchmark first.** Reactant produced none in two days. Its replacement reported 2 ms
  forward, 1.5 ms back, 0.968 FDK correlation immediately.
- **No allocations inside traced loops.** `527f087:src/Forward/RayMarching.jl:662` allocates a
  sinogram-shaped array per energy bin, colliding with this project's standing memory budget rule.
- **Test rot killed this, not the math.** Gradients were disabled because `create_aquilion_one` was
  deleted underneath them. Any revival should pin the extension's tests to stable API surface.
- **No merge path.** Unrelated roots; cherry-pick or copy only.
