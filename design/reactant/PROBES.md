# Probe results (2026-09-07, Apple M4, 10 cores / 16 GB, Julia 1.12.7, Reactant 0.2.285, Enzyme 0.13.201)

All runs on the XLA **CPU** PJRT client (no Metal backend exists). Scripts in `probes/`, runnable from
`envs/reactant`. Timings are second calls with `sync=true`.

## 1. Runtime smoke (`probes/01_reactant_smoke.jl`)

| Primitive | Result |
|---|---|
| `@compile` cumsum/broadcast/sum | PASS (11.8 s incl. XLA init) |
| Enzyme reverse gradient inside `@compile` | PASS |
| gather `x[traced idx]` and its scatter adjoint | PASS |
| 3-D gather via `CartesianIndex` of traced ints | FAIL — use linear indices (`vec(A)[lin]`) |
| `fft`/`rfft` along dim 1 | PASS |
| `@trace for` (traced Int promoted by `x .* i`; `Float32(i)` on a traced Int fails) and its gradient | PASS |
| `@trace while` with traced condition | PASS (only traced state is carried) |
| `sort`, min/max 3-tap median, small `svd`, `ReactantRNG` `rand` | PASS |
| 34 M-element fused exp/log kernel | 50.9 ms |

## 2. DD projector, integral-image form (`probes/02_dd_integral_image.jl`)

512²×64 volume → 736×16 detector, one view, per-slab affine maps, 2-D cumulative sum + bilinear
corner sampling.

| Metric | Value |
|---|---|
| forward | 147 ms/view (≈106 s per 720 views) |
| Enzyme reverse | 1457 ms/view — scatter adjoint of the cumsum/gather chain |
| parity vs the same formula in plain Julia | 2e-4 max rel — Float32 cancellation on large cumulative sums |

Rejected: cancellation and a 10× slower reverse pass.

## 3. DD projector, static-tap separable resampling (`probes/03_dd_static_taps.jl`)

Same problem. z-stage `Z[it,row,slab] = Σ_{d<KZ} oz·vol[it,k0+d,slab]`, x-stage
`P[col,row] = Σ_slab Σ_{d<KX} ox·Z[i0+d,row,slab]·norm`, weights in closed form, `k0/i0 = floor(lo)+1`,
static taps from geometry (`KX=2, KZ=3` here; transpose `KXT=3, KZT=3`).

| Metric | Value |
|---|---|
| forward parity vs Float64 direct overlap sum | **max rel 4.4e-6, mean rel 7.2e-7** |
| forward | **84 ms/view** (≈60 s per 720 views, CPU XLA) |
| gather-form transpose | 66 ms/view |
| adjoint dot-product test ⟨Av,s⟩ vs ⟨v,Aᵀs⟩ (Float32) | rel diff 6.3e-6 |
| Enzyme generic reverse of the forward (no custom rule) | **45.5 ms/view**, matches the gather transpose to 6.3e-5 max rel |
| batching 8 views per graph | 89 ms/view — no gain on CPU |
| compile time | 30 s forward, 15 s transpose, 5 s gradient |
| peak RSS of the probe process | 2.4 GB |

Conclusions: the static-tap form is oracle-grade in Float32, differentiable by generic Enzyme without
a custom rule, and gather-only in both directions. On this CPU it is ~25× slower than the legacy
fused Metal kernel (2.4 s for the 234-bin polychromatic 720-view forward); CUDA is the speed target.

## 4. Reference legacy numbers (from `src/projection/dd_fast.jl` comments, M4 Metal)

| Case | `:dd` tiled | `:dd_fast` |
|---|---|---|
| 512²×64 vol, 736×16×720 sino, 234 bins | 113.8 s | 2.46 s |
| 1024²×32 UHR slab | 109.3 s | 5.3 s |
| `:siddon` tiled, 234 bins, 512 case | 10.6 s | — |

These are not reproducible from any committed script; the benchmark harness (M5) rebuilds them.

## 5. End-to-end pipeline under Reactant + Enzyme (`test/functional/reactant/smoke_pipeline.jl`, 2026-09-08)

`Functional.eict_forward`: material fractions → static-tap DD (per material, per view) → EICT chain
(spectral conversion, fill factor, reparameterized noise, air/log, water BHC) → FDK → HU, compiled as
ONE XLA program. Toy: 16×16×2 phantom (compacted Gammex, 15 materials), 32×4×8 sinogram, 16×16×2 recon.

| Quantity | Value |
|---|---|
| Float32 forward: compile | 59.8 s |
| Float32 forward: run (XLA CPU) vs plain Julia | **1.0 ms vs 21.6 ms** |
| Float32 forward parity vs plain arrays | 3.3e-6 |
| Float64 `Enzyme.gradient(Reverse)` of a random HU loss w.r.t. the fractions: compile / run | 52.1 s / 4.4 ms |
| ⟨∇, d⟩ AD vs central finite differences | −1946.95821065 vs −1946.95821068, **rel 1.6e-11** |
| whole smoke wall time | 2 m 19 s |

Two lessons from getting this to run:
1. **Plan tensors must be lifted into the graph.** With the plans' host `Matrix` tables (μ table,
   Toeplitz filter) used directly in `traced * host` products, Reactant fell back to the generic
   scalar `Matrix{TracedRNumber}` product and traced for >20 min at 100 % CPU with no output.
   `ext/BasisSimulatorReactantExt.jl` overrides `Functional._on_device(x, ::TracedRArray)` with
   `Reactant.Ops.constant`, and the pipeline lifts every plan tensor through it. Large data tensors
   should be thunk arguments, not constants.
2. **Every material costs `n_views` unrolled projection programs** (labeled masks are projected
   one-hot per material). The toy uses `compact_materials`; the M5 fix is select-accumulate over the
   gathered material id (one volume walk) and `@trace for` over views.

## 6. Per-stage Reactant/Enzyme smokes (all pass; run `bash test/functional/reactant/run_all.sh`)

| Stage | compiled vs plain arrays | Enzyme reverse vs reference |
|---|---|---|
| DD projector (`smoke_dd`) | 7.7e-7 fwd / 1.9e-5 transpose (F32) | gradient = gather transpose 7.8e-6; ⟨Av,s⟩ vs ⟨v,∇⟩ 3.1e-8 |
| FBP/FDK (`smoke_fbp`) | 2.7e-14 (F64), ~1e-5 (F32, XLA atan2 sub-pixel) | 3.0e-13 vs FD |
| EICT chain (`smoke_eict`) | 4.8e-16 (F64), 2.7e-7 (F32) | = hand VJP 5.9e-16, vs FD 2.8e-9 |
| PCCT chain (`smoke_pcct`) | 2.7e-16 (F64), 7.3e-8 (F32) | vs FD 2.8e-10 (straight-through surrogate 2.4e-10) |
| HIR (`smoke_hir`, dense operators) | 1.0e-15 | vs FD 1.2e-10 |
| Denoisers (`smoke_denoise`) | ≤2.2e-7 | SVD-bilateral vs FD 1.3e-8 |
| Cong VMI (`smoke_vmi`) | 1.2e-13 (F64) | unrolled solver = IFT VJP 6.1e-14, vs FD 2.3e-9 |
| n-channel estimator (`smoke_nchannel`) | 2.7e-15 (F64), 1.3e-6 (F32) | vs FD 1.7e-11 directional, ≤3.6e-9 per entry |
| End-to-end pipeline (`smoke_pipeline`) | 3.3e-6 (F32) | vs FD 1.6e-11 |

## 7. Compiled view loops (M5): toy pipeline under Reactant/Enzyme

Toy fixture (`test/functional/reactant/smoke_view_batch.jl` geometry: 32 cols × 4 rows, 8 views,
16³×2 Gammex, 15 materials), `view_batch = 1` → every stage runs a StableHLO `while` loop over
one view per iteration (DD runs of 2 views, spectral sum, FDK). `design/reactant/probes` staged
timing, XLA CPU, Julia 1.12.7, Reactant 0.2.285:

| stage compiled | compile | note |
|---|---|---|
| DD path lengths (looped) | 554 s | 15 channels share one index computation per batch |
| DD + EICT chain (looped spectral sum) | 400 s | |
| full forward (DD + chain + looped FDK) | 346 s | forward vs host looped path: max 2.1e-2 HU |
| Enzyme reverse of the full forward | 297 s | gradient runs through all three while loops |

Compile time no longer depends on the number of views (the body is traced once); it is set by
the batch size and the number of runs. Tracer rules learned here: loop bodies may only read
traced arrays that are loop arguments (carried in the state tuple), never closure captures; wrapper
arrays (`reshape`/`dropdims` of traced arrays) must be materialized before `dynamic_*_slice`;
the tap accumulator must be a tuple (a `Vector{Any}` of traced arrays turns the final
`sum(; dims)` into a dynamic call that recurses in `mapreducedim!`); a variable assigned inside
the loop closure and elsewhere in the enclosing scope is boxed and hides the traced index.

## 8. Where the CPU time goes, and the dense projector (`probes/bench_cpu.jl`, `probes/stage_times.jl`)

64 grid / 100 views / 64 recon, GE Revolution arc, 16 materials, Apple M-series, 4 threads:

| | forward | gradient step | compile |
|---|---|---|---|
| legacy kernels on CPU arrays | 1.46 s (simulate! 0.74 + recon 0.72) | — | — |
| compiled, gather projector, unrolled batches | 18.1 s | 42.7 s | 180 s / 133 s |
| compiled, gather projector, while loops | 14.9 s | 37.3 s | 34 s / 34 s |
| compiled, **dense projector**, while loops | **0.27 s** | **1.86 s** | 30 s / 22 s |

Stage times of the gather program: DD path lengths 15.6 s (1.06 s per material), spectral chain
0.11 s, FDK ≈ 0 — the projector was everything; XLA:CPU executes gathers as scalar loops. The dense
formulation (`projection/dd_dense.jl`: overlap weights for every voxel, two batched `dot_general`
contractions per view batch, all channels in one matmul) is the same DD3 physics to 1e-11 (F64) /
5e-6 (F32), 5× faster than the legacy CPU kernels and 55× faster than the gather program. The
pipelines use it by default; a view whose dense weights do not fit `batch_budget_mb` raises an
error with the numbers (no silent fallback).
