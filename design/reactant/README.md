# BasisSimulator → Reactant/Enzyme overhaul — master plan

Branch: `feat/reactant-autodiff`. Started 2026-09-07 from `main` at `d9b8183` (v0.14.0).

**Branch policy:** everything in this effort lives on this branch (and PRs *into* it). Nothing merges
to `main` without Dale's direct involvement. The branch itself must stay fully functional at every
commit: `Pkg.test()` green on CPU, every functional stage backed by an oracle test against the legacy
code, and the Reactant/Enzyme smoke tests runnable from `envs/reactant`.

---

## Handoff (2026-09-08)

Branch `feat/reactant-autodiff` is the library: `src/functional/` (laid out along the imaging
chain — the module docstring is the map), the Reactant extension, the five-struct pipelines
(`Functional.pipeline` / `forward`, `vmi_pipeline` / `vmi_forward`), compiled view loops sized
from a memory budget, the strict `EICTScanner` / `PCCTScanner` API, oracle tests
(`test/functional/`, 1149 assertions) and Reactant smokes (`test/functional/reactant/`). Never
merge to `main` without Dale. The material-decomposition work that USES this branch lives in the
private repo `MolloiLab/basis-autodiff-mmd` (see its `HANDOFF.md`). Performance (PROBES §8): the gather-based projector was the CPU bottleneck (XLA:CPU gathers); the
dense-separable projector (`projection/dd_dense.jl`, the default) makes the compiled forward 5×
faster than the legacy CPU kernels at 64/100 (0.27–0.37 s vs 1.46 s). The GRADIENT is the open
performance item (PROBES §9): 2.8–3.8 s per step at 64/100 (7–10× the forward; the unrolled
program sits at 4×, the ideal one-shot at ≈2×), 392 s at 256/984. Established by measurement, not
guesswork: the projector's reverse is cheap (1.6× its forward); the dense tiled FDK built against
the "scatter adjoint" hypothesis LOST to the gather FDK on forward and gradient in a same-process
A/B and was deleted; the checkpointed while-loop reverse costs ~2× the unrolled one; the per-pixel
bowtie spectral sum as a batched `dot_general` made the forward 3× slower for nothing (reverted).
At 256/984 the while-loop program's gradient step is 819 s (forward 30 s, legacy 33 s). What helps:
the HOST-COMPOSED gradient (`eict_batches` / `batch_data` / `eict_batch_vol` / `eict_vol_to_hu`,
commit 511a39e): one loop-free program per (orientation, batch length), reused across batches with
the per-view tables as data, volume and gradient as sums over batches — 256/984 gradient step
378 s → with the accumulator INSIDE each program (a `.+` on concrete arrays outside a program is
element-wise host execution) **57 s**, forward **14 s** (legacy 33 s); 64/100 0.32 s / 1.26 s. This
is the shipped driver: `compile_pipeline(pipe, x)` → `forward(cp, x)` / `pullback(cp, x)` (ext),
smoke `test/functional/reactant/smoke_compiled.jl`. Inputs: pipelines take either dense
fractions `(nx,ny,nz,n_mat)` (differentiable) or the integer label volume, whose one-hot chunks are
formed on device `batching.mat` materials at a time (large phantoms with many materials never
materialize all fractions). Open items: the PCCT/VMI pipelines on the same batch driver, M6 driver switch
(`simulate!`/`reconstruct!` on the functional core), CUDA validation on the lab box, the
CT-realistic gallery numbers (`design/reactant/probes/gallery_parity.jl`) into PROBES §8.

## 0. Where things stand (status board)

| Item | State |
|---|---|
| Sync with origin, CI fix for the docs deploy | done (PR #65, separate) |
| Surveys: pipeline map, tests/oracles, AD hazards, prior attempts, Reactant capabilities | done → `surveys/` |
| Reactant 0.2.285 + Enzyme 0.13.201 on Julia 1.12.7 / macOS arm64 | installs, compiles, differentiates (`probes/`, `PROBES.md`) |
| **M1 operators** — `dd_project`/`dd_transpose` (flat+arc, axial+helical), `fbp` (filter, weighted/matched backprojection, FOV mask, 8 kernels) | **done.** DD: Float64 parity 1.8e-16 vs `dd_forward_project`, brute-force `Aᵀ` 5.7e-14, adjoint 3.6e-16, Float32 2e-6; FBP: backprojection bit-identical, filter 1.8e-7, FDK 1.8e-7; both compile + Enzyme gradient (DD 7.8e-6 vs transpose, FDK 3e-13 vs FD) |
| **M2 EICT chain** — `eict_chain` (spectral conversion, fill factor, scatter, reparameterized noise, air/log, BHC) + hand VJPs | **done.** ≤3.8e-6 abs vs `simulate!` with the captured noise draws (all four noise/scatter variants), BHC bit-identical, VJP 1e-9 vs FD, compiled chain 4.8e-16 (F64) / 2.7e-7 (F32), Enzyme = VJP to 6e-16 |
| **M2 end to end** — `eict_pipeline` / `eict_forward`: fractions → DD → EICT(+BHC) → FDK → HU behind the five structs | **done.** HU 1.1e-6 rel vs the nb01 chain (`simulate!`→`apply_bhc_water`→`reconstruct!`→`to_hounsfield`), noise on and off, legacy RNG stream reproduced; **one compiled XLA program: parity 3.3e-6, 1.0 ms vs 21.6 ms plain Julia; Enzyme gradient w.r.t. fractions through the whole chain = FD to 1.6e-11** (PROBES.md §5) |
| **M3 PCCT** — `pcct_chain` (spectral bins, per-bin log, host-drawn exact Poisson + Gaussian surrogate, pile-up, combine) | **done.** bins 8.6e-8 vs the fused spectral kernel, full chain 3.8e-6 vs `simulate!(PCCTWorkspace)` (pile-up on), legacy counts reproduced bit-for-bit from the legacy bins, Enzyme = closed-form VJP to 2e-16 |
| **M3 HIR** — `hir_reconstruct` (OS-PWLS, Huber, padded equal subsets, operators passed in) | **done.** bit-identical to `reconstruct!` at strengths 0/60/100 (legacy operators), 4.4e-7 with the functional DD operators, `W_proj` 2e-6, Enzyme gradient 1.2e-10 vs FD |
| **M4 VMI** — `cong_decompose` (fixed-iteration vectorized) + IFT VJP, `cmv_decompose`, `synth_vmi_2basis` | **done.** Cong 4.3e-5 (F32, shared ŵ) / 6.9e-5 (per-ray ŵ) vs `apply_cong!`, F64 residual 3e-16; CMV + synth bit-exact; Enzyme (unrolled) = IFT VJP to 6e-14 |
| **M4 denoisers** — `acnr_kalender`, `sino_svd_denoise_bilateral`, `median_z`, `sfjsd_denoise` | **done.** ACNR bit-exact (nb03/nb04 kwargs), SVD-bilateral 4.6e-5, median-z exact (`:shrink`), SF-JSD 2e-6 with captured constants; compiled ≤2.2e-7; found a legacy SF-JSD thread race (§7) |
| **M4 n-channel estimator** — `nchannel_estimate` (profiled Poisson quasi-likelihood, fixed outer/inner iterations), `nchannel_tlbf` (Lee-2025 joint filter), angular apodization; `nchannel_NOTES.md` maps the notebook | **done.** vs the verbatim nb04 cell code: basis sinograms ≤7e-6, Fisher terms ≤1.4e-6, quality flags / iteration counts identical, T-LBF and apodization ≤3e-7; IFT/FD smoothness 5e-8 (124 tests); compiled 2.7e-15 vs plain arrays, Enzyme gradient through the solver = FD to 1.7e-11 |
| `ext/BasisSimulatorReactantExt.jl` (Reactant weakdep): lifts plan tensors to in-graph constants; in-graph `Ops.iota` for the projector index vectors | done |
| Integrated CPU suite (`Pkg.test()`) | all eight stages + pipeline wired (`test/functional/`, ~1050 functional assertions); see the last commit message for the final count |
| `design/reactant/HOWTO.md` — how to run tests/smokes and drive the pipeline + Reactant/Enzyme from a REPL | done |
| **M5 view batching + compiled view loops** — `dd_run_plans`/`dd_project_run`/`dd_transpose_run`, `poly_log_sinogram_looped`, looped `backproject`: every view-dependent stage runs its batches inside ONE StableHLO `while` loop (hooks in `core/loop.jl`, traced versions in the extension); `eict_pipeline(…; view_batch = :auto, batch_budget_mb)` sizes the loops from a memory budget — five-struct API unchanged | **done (host).** forward bit-identical to per-view (all batch sizes, remainders, multi-channel), transpose ≤4e-16 (F64) / 3e-7 (F32), pipeline HU ≤2e-3; Enzyme reverse through the loop ≡ unrolled ≡ FD (probe); Reactant smoke + scaling numbers → `PROBES.md` §7 |
| **Module layout** — `src/functional/` = core / source / projection / detector / reconstruction (fbp, hir, denoising, vmi/{nchannel = go-to, cong_cmv}) / pipelines; the module docstring is the map | **done.** functional suite 1149/1149 after the move |
| **API split** — `Scanner{T}` abstract; `EICTScanner` / `PCCTScanner` over `ScannerGeometry`; `SimOptions` = common physics only (no `fidelity`, PCCT toggles on the scanner); `Functional.pipeline` / `forward` dispatch on the family; no compatibility shims | **done.** every constructor site in src/tests/notebooks migrated; `pcct_pipeline` behind the five structs |
| **HIR on the looped operators** — `hir_operators(geom, vol_shape; view_batch)` gives `hir_reconstruct` its `A`/`At` pair on `dd_project_views`/`dd_transpose_views` (any ordered view subset, padded subsets included) | **done (host).** forward bit-identical, recon within 1e-4 of the per-view operators |
| **VMI behind the five structs** — `vmi_pipeline(phantom, EICTScanner, protocols, …)` / `vmi_forward`: per-kVp EICT pipelines (no BHC) → per-ray applied-spectrum tables → published n-channel estimator → per-basis looped FDK → ACNR → VMI stack, one pure program | **done (host).** toy dual-kVp: water body ≈ 1 g/cm³, VMI HU near 0; non-finite estimator rays zeroed before the FDK |
| Driver switch (`simulate!`/`reconstruct!` → functional core) | not started (M6) |
| CUDA validation | not started (needs a lab NVIDIA box) |

---

## 1. Goal and non-negotiables

The current pipeline (five-struct API, `simulate!` → corrections → FBP/HIR → VMI chains, notebooks
01–12) is the specification and the numerical **oracle**. The overhaul re-implements the *internals*
as a pure tensor program that Reactant.jl compiles to XLA (CPU today, CUDA on the lab machines) and
Enzyme.jl differentiates from phantom to HU image, with these non-negotiables:

1. **Identical outputs.** Every stage reproduces the legacy result to a stated tolerance
   (Float64 machine precision where the math is exact; ≤1e-4 in Float32 where the legacy fused
   kernels themselves only agree to that). Notebook PASS/FAIL cells must still pass.
2. **Fully differentiable end to end.** Every stage is a pure function of tensors; reverse-mode
   through the whole chain works (Enzyme inside Reactant), including through iterative solvers.
3. **Fast and memory-lean on CPU and GPU via XLA.** Static shapes, gather-only kernels in both
   directions (no atomics, no scatter), per-view/per-chunk graphs, energy axis materialized only per
   chunk, buffer donation, checkpointing for loops.
4. **Oracle at every step.** Nothing lands without a parity test against the legacy code, an adjoint
   or finite-difference gradient test, and a benchmark line. If we know what the goal is, we know
   how to measure it.
5. **Public surface unchanged.** The five structs, `create_*_workspace`, `simulate!`,
   `reconstruct!`, `apply_acnr_kalender!`, `synth_vmi_2basis`, … keep their names and semantics;
   they become thin wrappers over the functional core (mutation is load-bearing at ~90 notebook
   call sites — see `surveys/pipeline_map.md` §6.1).

---

## 2. Facts that shaped the design (from the surveys)

**Codebase (surveys/pipeline_map.md, surveys/ad_hazards.md).**
- 26k lines, 61 files. All 92 device kernels are `AK.foreachindex` closures; **no atomics anywhere**,
  every kernel is an output-stationary gather. One calling convention to replace.
- Five hard walls for tracing: (1) data-dependent `while` ray marches in every projector,
  (2) exact integer Poisson sampling with rejection loops and CPU staging, (3) Brent root-finding
  + Newton-with-break inside the per-ray Cong kernel, (4) per-row SVD / median / bilateral
  denoisers, (5) `try/catch` OOM retry that changes tile shapes at runtime.
- The EICT noise kernel is *already* reparameterized (`λ + √max(λ,1)·ε + σ_e·ε₂`): with ε as an
  input tensor it is differentiable without algorithm change. Only PCCT integer Poisson and the
  pile-up Monte Carlo need genuine treatment.
- FBP filtering is a spatial-domain tap convolution (not an FFT ramp); the ramp/window construction
  is host-only. Units are cm throughout.
- The production VMI estimator (n-channel profiled Poisson quasi-likelihood,
  `nchannel_profile_tile!`) exists **only in notebook cells** (nb04 canonical, mirrored in nb03/12);
  `src`'s `apply_cong!` reproduces nb09 only. The only denoiser any notebook calls is
  `apply_acnr_kalender!`.
- Precomputed tables live on workspaces, not on the five structs; spectra/bowtie/DRM files are
  re-read on every construction (uncached).

**Tests (surveys/tests_oracles.md).**
- `Pkg.test()` and all three CI legs run **CPU-only** (`HAS_GPU=false`): 13 testsets covering the
  entire `simulate!` path never execute in CI. The oracle capture for those paths must be done on
  the Metal machine.
- Twelve numerical contracts are sharp enough to reuse verbatim; the sharpest is the brute-force
  matrix identity for the DD operator (`test/projection.jl:81-94`, 2e-12 in Float64) plus the
  adjoint dot product (2e-11). `:dd_fast ≡ :dd` at 1e-4 abs (Float32) pins the fused kernels.
- Seven notebooks carry coded PASS/FAIL cells (01, 03, 04, 07, 08, 12; 05 prints machine-precision
  numbers without a threshold). nb12 has 21 gates and is the strongest end-to-end oracle.
- `docs/verify_notebook_exports.py` hashes **every file under `src/`** into all 12 export
  fingerprints. Any `src/` change invalidates the committed notebook HTML until a full Metal
  re-render (see §8 fingerprint policy).

**Reactant/Enzyme (surveys/reactant_capabilities.md, PROBES.md).**
- Latest registered: Reactant 0.2.285, Enzyme 0.13.201; Julia 1.12 supported; installed and smoke-
  tested here on the M4.
- **macOS arm64 is XLA-CPU only.** There is no working Metal PJRT backend (Dale's pure-Julia
  MPSGraph backend, EnzymeAD/Reactant.jl#2489, is still open). The same thunks run on NVIDIA
  Linux with `Reactant.set_default_backend("gpu")`. Metal stays the *legacy* AK path (the oracle).
- Static shapes only; scalars freeze at trace time unless `track_numbers=true`.
- **Custom Enzyme rules are not honored inside a trace** (#487 open). Enzyme-MLIR differentiates
  the StableHLO; any hand-written adjoint must be composed at *stage boundaries* on the host, or
  the tensor program must be written so generic AD is already efficient.
- No Poisson sampler; `ReactantRNG` provides `rand/randn` (PHILOX/THREE_FRY deterministic across
  backends).
- KernelAbstractions kernels compile only via "raising" and only with CUDA.jl loaded (even on the
  Mac); AcceleratedKernels is incompatible today. → Clean tensor formulation, not kernel raising.
- Compile time/memory is the main risk on 16 GB: minutes and several GB per graph. → Small
  per-stage thunks, `@trace` loops, no unrolled view loops.

**Prior attempts (surveys/prior_reactant.md).**
- The 2026-01 XLA attempt died from *precomputing gather indices* (96 trillion elements at clinical
  scale), not from XLA. Never precompute index tables; compute indices in-graph from per-view
  scalars.
- The 2026-01→03 Enzyme extension (recovered at
  `surveys/BasisSimulatorEnzymeExt_2026-03_recovered.jl`) worked: forward projection ↔ matched
  backprojection (`weighted=false` is the correctness hinge), convolution adjoints, polynomial BHC
  derivative, finite-difference verifiers. It was deleted as housekeeping after its tests rotted.

---

## 3. Architecture

### 3.1 `BasisSimulator.Functional` — the pure tensor-program core

`src/functional/` is a submodule of BasisSimulator (no new package, no new registry entry). Every
stage is a **pure function of arrays plus an immutable host-side plan struct**:

```
stage(x::AbstractArray{T}, plan::StagePlan) -> AbstractArray{T}
```

- **Array-generic.** Written against `AbstractArray` with broadcasting, `reshape`, `permutedims`,
  `sum(; dims)`, `cumsum`, matrix products and `vec(x)[idx]` gathers with `Int32` index arrays
  computed by broadcasting. Runs unchanged on `Array` (CPU, the CI path), on `CuArray`, and on
  Reactant traced arrays. No `AK.foreachindex`, no scalar indexing, no mutation of inputs.
- **Static shapes.** Sizes derive from the plan, never from data. Subsets padded to equal size,
  kernel radii and iteration counts fixed on the host, bracketed solvers with fixed trip counts and
  masks.
- **Plans, not workspaces.** A plan holds geometry-only tensors (cosine weights, filter taps,
  per-view scalars), spectral tables (μ, wη, W, bowtie ŵ, air reference, I0, BHC coefficients),
  and constants (tap counts, iteration counts, clamps). Plans are built on the host once; legacy
  host helpers (`resolve_source_spectrum_full`, `calibrate_bhc_water`, `create_spatial_kernel`,
  `compute_mc_drm`, …) are reused for that — they are pure host math and not part of the traced
  program.
- **Noise as input.** Every stochastic stage takes its realization as a tensor (`ε ~ N(0,1)` from
  `ReactantRNG` or the host; exact integer Poisson counts from the host sampler). The traced program
  is deterministic given its inputs.
- **Legacy = oracle.** The AK kernels are not modified. They remain the Metal path and the reference
  for every parity test until the driver switch (M6) and beyond.

### 3.2 Dependencies

- `Reactant` and `Enzyme` become `[weakdeps]` with an extension `ext/BasisSimulatorReactantExt.jl`
  (compile helpers, `@trace` loop shims, `ReactantRNG` plumbing). The functional core itself needs
  neither: plain Julia arrays exercise every stage in `Pkg.test()`.
- `envs/reactant/Project.toml` is the local development environment (Reactant + Enzyme +
  BasisSimulator via path). Reactant-backed tests and benchmarks run from it:
  `julia --project=envs/reactant test/functional/reactant/<smoke>.jl`.
- CI stays CPU-only and does not install Reactant (470 MB binary). A Reactant CI leg is a later
  decision (M5).

### 3.3 Stage graph and interfaces

```
phantom labels (UInt8 mask) or material fractions f[nx,ny,nz,n_mat]
        │  dd_project (per view, static taps, separable resampling)
        ▼
P[n_col,n_row,n_view,n_mat]   per-material path lengths (cm)        ← projector/detector interface
        │  EICT: poly_log_sinogram → noise(ε) → air/log → BHC          PCCT: spectral_bin_intensities → per-bin log → counts(N) → pile-up → combine
        ▼
log sinogram(s) [n_col,n_row,n_view(,n_bins)]
        │  fbp: filter_views → backproject(weighted) → fov_mask        hir: OS-PWLS(A, Aᵀ, Huber), fixed epochs/subsets
        ▼
μ volume → HU
        │  VMI: nchannel_estimate (tile kernel, fixed iters) / cong_decompose → per-basis fbp → acnr_kalender → synth_vmi_2basis
        ▼
VMI images at 50/70/100/140 keV
```

Key interface decisions:
- **Per-material path lengths `P`** are the contract between projection and detector physics. This
  is exactly the `:dd_fast` reassociation (`L_e = Σ_m μ[m,e]·P_m`), so the spectral conversion is a
  matrix product over energies, and the projector is *linear* in the material maps.
- **Labeled masks** (today's phantoms) are handled by gathering the material id along with the
  overlap weight and select-accumulating into `n_mat` channels (no one-hot volume). Dense material
  fractions (XCIST/XCAT style, the differentiable parameterization) use the same operator batched
  over the material axis.
- Sinograms stay `(n_col, n_row, n_view)`; bowtie ŵ is `(n_col, n_row, n_E)`.

### 3.4 The projector (the crux) — static-tap separable resampling

For one view and one longitudinal slab the legacy DD3 kernel maps voxel boundaries to the detector
affinely (`t = s_tran + (vmin_t + (it−1)·v_t − s_tran)·mag_fac`), and the detector cell interval is
per (view, col) — independent of the slab. So the overlap weight separates:

```
w(col,row,slab,it,ip) = ox(col,slab,it) · oz(row,slab,ip) · norm(col,row)
```

and each stage is a 1-D **box-overlap resampling** between two uniform grids. A cell overlaps at
most `K = ceil(max span)+1` voxels, so the projection is `K_x·K_z` gathers with weights computed in
closed form (`max(0, min(hi,i) − max(lo,i−1))`) — the *same arithmetic* as the kernel, with no
data-dependent loop, no cumulative sum (no Float32 cancellation) and no scatter. The transpose is
the same construction with cells and voxels swapped (each voxel overlaps ≤ K' cells), so
**both directions are gathers**. Enzyme's generic reverse of the forward is already a scatter-free-
enough program in practice (46 ms vs 84 ms forward on CPU, see `PROBES.md`); the explicit gather
transpose is kept for the adjoint test and as a stage-boundary VJP option.

The per-view `vertical` flag (which axis is the slab axis) is a host constant per view: views are
processed in two orientation groups with the volume permuted once per group.

### 3.5 Differentiation strategy

1. **Default: Enzyme inside Reactant, generic.** Every stage is written so that the StableHLO is
   cheap to differentiate (gathers, matmuls, elementwise, fixed loops). The probe shows this is
   fast for the projector.
2. **Stage-boundary VJP composition** for memory control and for closed-form adjoints that beat
   generic AD: the spectral conversion VJP (`∂/∂P_m = Σ_e w_e T_e μ_me / Σ_e w_e T_e`, never
   materializes the energy axis in reverse), the DD gather transpose, the implicit-function
   VJP for Cong / the n-channel solver (`−J_x⁻¹ J_p` at the root), the polynomial BHC derivative.
   Each is a pure function tested against finite differences and against Enzyme's own gradient.
   Composition happens on the host (ChainRules-style), since Reactant cannot register rules in a
   trace.
3. **Noise.** EICT: pathwise through the reparameterized kernel with ε fixed. PCCT: straight-through
   — the *value* uses exact integer counts drawn on the host with the legacy sampler (bit-for-bit
   legacy output), the *gradient* flows through the Gaussian surrogate `λ + √λ·ε`. Documented as a
   design choice, testable (means/variances) and swappable.
4. **Iterative solvers** (HIR OS-PWLS, Cong, n-channel): fixed trip counts, masked convergence,
   `@trace for` with `checkpointing`/`mincut` for reverse mode; IFT VJPs available at the boundary.

### 3.6 Performance and memory strategy

- **Per-view (or view-chunk) graphs** for projection/backprojection; the energy axis is materialized
  only per chunk (`n_cells_per_chunk × n_E`, 11 MB per view at clinical size).
- **Gather-only in both directions**, static tap counts, indices computed in-graph from a handful of
  per-view scalars (never precomputed tables).
- **CPU (this Mac) is XLA-CPU**: expect ~25× slower than the Metal fused kernel for the projector
  (84 ms/view vs the 2.4 s/720-view oracle); that is the CPU-vs-GPU gap, not a formulation problem.
  Batching views per graph did not help on CPU. **CUDA is the speed target**; the same thunks run
  there unchanged.
- Buffer donation (`donated_args=:auto`), `sync=true` only for timing, hold compiled thunks (no
  automatic cache), `GC.gc()` between large calls.
- Float32 hot path; Float64 only for the exactness oracles (XLA CPU/CUDA run f64 fine).

---

## 4. The oracle harness

Every stage ships with tests in `test/functional/` (plain arrays, runs in `Pkg.test()`, CPU) and a
Reactant smoke in `test/functional/reactant/` (runs from `envs/reactant`). Shared helpers live in
`test/functional/harness.jl`.

| Tier | What | Tolerance rule |
|---|---|---|
| Exactness | Float64 parity vs the legacy kernel on the same inputs | ≤1e-12 rel where the math is a reassociation; report the achieved number |
| Operator identity | adjoint dot product `⟨Ax,y⟩ = ⟨x,Aᵀy⟩`; brute-force matrix `Aᵀ` on tiny fixtures | 2e-11 / 2e-12 (lifted from `test/projection.jl`) |
| Float32 parity | vs the legacy Float32 path and vs the fused kernels | ≤1e-4 abs (the legacy `:dd_fast ≡ :dd` tolerance), mean rel ≤1e-5 |
| Gradient | finite differences (Float64) vs closed-form VJP vs Enzyme-in-Reactant | ≤1e-4 (FD vs VJP), ≤1e-3 (Enzyme vs FD) |
| Statistical | noise stages: mean/variance, dose scaling `σ ∝ 1/√mA` | 2 % / 15 % as in `test/api.jl` |
| End-to-end | notebook PASS/FAIL cells (01: 5 gates; 03/04: 5; 07: 6; 08: 8; 12: 21) run on the functional path | same gates, no tolerance change |
| Performance | per-stage timing at three sizes (CI toy / dev / clinical), CPU-XLA and CUDA, vs the legacy Metal number | reported, regression-gated later |
| Memory | peak RSS / device bytes per stage | reported |

Oracle capture for GPU-only legacy paths (e.g. the PCCT spectral forward, which JIT-compiles for
minutes on CPU) is done once on the Metal machine and stored as small fixtures with SHA-256 pins —
to be added when the first such path needs it.

---

## 5. Roadmap and acceptance gates

| Milestone | Deliverable | Gate |
|---|---|---|
| **M0 Foundation** (this session) | branch, surveys, probes, this plan, `envs/reactant`, harness skeleton, memory notes | Reactant+Enzyme smoke green on the Mac |
| **M1 Operators** | `dd_project` / `dd_transpose` (flat+arc, axial+helical), `fbp` (filter, backproject weighted/matched, FOV mask) | brute-force matrix 2e-12, adjoint 2e-11, Float32 parity 1e-4 vs `dd_forward_project`/`dd_backproject!`/`fdk_reconstruct`; Reactant compile + Enzyme gradient smoke |
| **M2 EICT end to end** | `poly_log_sinogram`, noise(ε), air/log, BHC; `Functional.simulate_eict` | ≤1e-4 vs `simulate!(EICTWorkspace)` with captured ε; nb01 gates PASS on the functional path |
| **M3 PCCT + HIR** | `spectral_bin_intensities`, counts (host draw + surrogate), pile-up, combine; OS-PWLS with Huber | ≤1e-4 vs `simulate!(PCCTWorkspace)` (bit-for-bit with host-drawn counts); HIR parity at strengths 0/60/100 incl. `W_proj` 2e-6 |
| **M4 VMI chains** | `nchannel_estimate` (+ T-LBF, angular apodization), `cong_decompose` (+ IFT VJP), `acnr_kalender`, `synth_vmi_2basis` | parity vs notebook cells ≤1e-4; nb03/04/12 gates PASS |
| **M5 Reactant end to end** | one compiled forward (phantom → HU) and one compiled reverse (loss → ∂/∂fractions, ∂/∂spectrum), `@trace` view loops, checkpointing, benchmarks JSON | gradient vs FD ≤1e-3; compile time and peak memory recorded; CPU timings vs legacy |
| **M6 Driver switch** | `SimOptions.backend = :legacy \| :functional` (default stays `:legacy` until Dale flips it); thin wrappers keep all names | full `Pkg.test()` green in both modes; all 12 notebooks re-rendered; fingerprint policy applied |
| **M7 CUDA** | run the same thunks on the lab NVIDIA box; hybrid raised-kernel overlays only if a stage is >2× slower than the AK Metal number | speed table vs Metal legacy |

---

## 6. Decision log

1. **Submodule, not a new package.** `BasisSimulator.Functional` inside the repo; Reactant/Enzyme as
   weakdeps. Rationale: keeps one API, one test suite, one release train; CI stays light.
2. **Legacy code is frozen as the oracle** (bit-level) until M6; no refactors of AK kernels.
3. **Static-tap separable DD**, not integral images (Float32 cancellation, 10× slower reverse) and
   not kernel raising (needs CUDA.jl, AK incompatible, branchy kernels are the failure class).
4. **Gather-only transpose** kept alongside Enzyme's generic reverse; the adjoint test is the
   contract, the faster of the two wins per backend.
5. **Per-material path lengths as the interface**; labeled masks via select-accumulate, fractions
   via a batched material axis.
6. **Noise as input tensors; PCCT straight-through** (exact counts for value, Gaussian surrogate for
   gradient). Reproducibility: seeds → host draws; PHILOX for in-graph `randn`.
7. **Iterative solvers with fixed trip counts and masks**; IFT VJPs at stage boundaries.
8. **Notebook-only algorithms move into `src/functional/`** (n-channel estimator, T-LBF, angular
   apodization, debias) with the notebook cell code as the oracle test.
9. **Fingerprint policy** (proposal, Dale to confirm): exclude `src/functional/` from
   `verify_notebook_exports.py` while the legacy path is the default (it cannot change notebook
   outputs); include it again at M6 together with the full re-render. Until then, no docs deploy
   from this branch.
10. **CPU on the Mac is for correctness and development, not the speed claim**; the speed claim is
    made on CUDA, with Metal-legacy as the yardstick.

---

## 7. Risks and open questions

- **Compile time / memory on 16 GB** for full-scale graphs (Reactant issues #3051, discourse 132655).
  Mitigation: per-stage thunks, `@trace` loops, toy sizes in tests, clinical sizes only in
  benchmarks.
- **Reverse-mode memory through view loops**: rely on `@trace checkpointing`/`mincut`; fall back to
  host-composed per-chunk VJPs (gradient of a sum over views is the sum of per-view gradients).
- **Forward-mode through `@trace` loops is currently broken** (#2361) — reverse only for now.
- **Arc detector + helical z-ramp** in the array projector: same affine structure, but the exact
  edge handling (`_dd_arc_row_bounds`, `active_z` tiling) must be pinned by the brute-force oracle.
- **Cong / n-channel fixed iteration counts**: must be proven sufficient on every fixture ray;
  worst-ray residuals reported in the stage tests.
- **Metal**: no Reactant backend; if Dale's PJRT PR lands, the same thunks should run there —
  until then Metal = legacy.
- **Legacy bug found (unfixed in `src/`, oracle untouched):** `apply_sino_sfjsd_denoise` captures
  `slice_lo`/`slice_hi` across `Threads.@threads` → rows mixed nondeterministically with
  `nthreads > 1` (0.5–15 % max-rel), bit-exact with one thread. Two-line rename fixes it; the
  functional tests use a sequential replica (`Functional.sfjsd_capture`) as the oracle.
- **Reactant tracing gotchas collected by the builders** (all worked around in the stages):
  (a) `TracedRArray{T} <: AbstractArray{TracedRNumber{T}}` — never bind `T` from an array;
  (b) a host `Matrix * traced` falls back to generic `Matrix{TracedRNumber}` and recurses on
  slicing — pass tables as traced inputs or use broadcast-and-reduce; (c) `min.(x, T(1e30))` with
  `T` taken from a host matrix's eltype sends the interpreter into unbounded recursion even in an
  unexecuted branch; (d) `ifelse.(host_Bool, Float32, traced)` infers a `Union` eltype — lift the
  scalar with `oftype`; (e) compiled `Enzyme.gradient` thunks are called with the mode first
  (`g(Reverse, loss, x, Const(p))`); (f) `stack` is not traceable, `cat` is; (g) keep plans
  host-side and move only their tensors with `to_rarray` so per-view index tensors are built
  in-graph (a host-constant plan bakes them in as literals); (h) nested fused broadcasts cost
  ~10 s of Julia compile each on 1.12 — a `@noinline` broadcast barrier halves it.
  (i) the projector's index vectors must be in-graph (`_iota` → `Ops.iota` in the extension):
  with the plain-array fallback every derived geometry tensor is host-evaluated and embedded as a
  literal per view, so trace time scales with detector columns × slabs (>20 min for a 834-column
  arc; 0.6 s per view with the override); (j) Reactant 0.2.28x caps same-named elementwise helper
  functions at 10 000 per module (`__lookup_unique_name_in_module` probes `name_1, name_2, …`
  with a fresh symbol table each call) — one looped pipeline exceeds it; the extension swaps in a
  per-name counter at load time.
- **Compile-time scaling of unrolled view loops** — solved (M5): view batches are a trailing
  tensor axis AND the batches run inside a compiled loop (`_batched_loop` → `@trace for` with
  `track_numbers = false`; per-batch constants by `Ops.dynamic_slice` of an in-graph table,
  outputs by `Ops.dynamic_update_slice`). Program size is set by the memory budget, not by the
  number of views or the phantom grid. Gotchas: (k) inside `@trace for`, host integers are
  promoted to traced numbers unless `track_numbers = false` — static shapes then break
  (`collect(Int, slice_sizes)` on traced ints); (l) never convert the traced loop index to `Int`
  (`Int32(::TracedRNumber)` has no method) — keep chunk-relative offsets static and slice the data
  tensor instead; (m) keep the loop-carried state ONE array.
- **Test-suite runtime**: the functional testsets add ~4 min of mostly one-time Julia compile
  (`gram_eigen` 6 specializations, DD kernels 4); a PrecompileTools workload would remove it.
- **Float32 vs Float64 drift** in denoisers/ACNR (legacy runs some FFTs in Float64): quantify, and
  decide per stage whether to match the Float64 path or accept a documented difference.

---

## 8. Working on this branch

```
design/reactant/README.md        this plan (keep the status board current)
design/reactant/PROBES.md        empirical numbers from the Reactant/DD probes
design/reactant/probes/          the probe scripts (run from envs/reactant)
design/reactant/surveys/         the five survey reports + the recovered Enzyme extension
src/functional/Functional.jl     module root (include list); one file per stage
test/functional/                 plain-Array oracle tests (in Pkg.test()); harness.jl helpers
test/functional/reactant/        Reactant/Enzyme smokes: julia --project=envs/reactant <file>
envs/reactant/Project.toml       local Reactant + Enzyme + BasisSimulator env (Manifest ignored)
```

Rules for agents and humans working here:
- Read `surveys/pipeline_map.md` §3 for the stage you touch and `surveys/ad_hazards.md` for its
  hazard rows before writing code.
- Pure, static-shape, array-generic, no scalar indexing, no RNG, noise as input, plans as immutable
  structs, host-only precompute allowed and labeled.
- **Never bind the scalar type from an array.** `Reactant.TracedRArray{T,N} <: AbstractArray{TracedRNumber{T},N}`,
  so `f(x::AbstractArray{T,3}, p::Plan{T}) where {T<:AbstractFloat}` throws `MethodError` inside
  `@compile`. Type arrays as `AbstractArray{<:Any,N}` (or leave them untyped) and take `T` only from
  the plan (`p::Plan{T}`) or an explicit `::Type{T}`; never `T(...)` with `T = eltype(x)`; avoid
  `similar(x, T, …)`/`zeros(eltype(x), …)` assumptions (found by the HIR builder, 2026-09-07).
- Every stage lands with: parity test(s) vs legacy with the tolerances in §4, an adjoint or
  finite-difference gradient test, a Reactant smoke, and a line in the status board.
- Do not modify legacy kernels. Do not re-run whole notebooks on the Metal machine without need.
- Never merge to `main` without Dale.
