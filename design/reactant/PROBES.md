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
