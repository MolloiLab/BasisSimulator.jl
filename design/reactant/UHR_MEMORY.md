# Making the differentiable twin work at full UHR

**Requirement, not a question.** The compiled twin must run — forward *and* gradient — on the full
ultra-high-resolution object grid. UHR is the input the decomposition is built around: it is what
makes "one voxel, one material" true, and it is what the reference scan is generated from. Any
claim that the twin can skip the object grid is wrong and should be deleted on sight.

## 0. The hardware this actually runs on

Every budget number in this repo's design notes predates the current machine. `HOWTO.md:22` says
"this 16 GB machine"; `PROBES.md` timings are XLA:CPU. **hpc3-gpu-n54-02 carries an NVIDIA RTX PRO
6000 Blackwell with 97 887 MiB (95.6 GiB) of VRAM**, and `Reactant.set_default_backend("gpu")`
(HOWTO.md:4) targets it. `PROBES.md:52` already names CUDA the speed target.

The notebook's `batch_budget_mb = 2048` is therefore **~47× smaller than available memory**. Before
any redesign is justified, the budget has to be set to the device that exists.

## 1. Measured per-view cost (library's own `dd_dense_view_bytes`, notebook's SCANNER/GEOM)

Object grid `1125² × 16`, `M = 16`; detector **834 × 12** — `n_rows` is the *active* cone-guarded
row count (`scanner.jl:463-472, 610-620`), **not** `scanner.detector_rows = 256`, which is the
physical panel. Confirmed by the cached run: `sino_shape = (834, 12, 36)`.

| term | object `1125²×16`, M=16 | recon `512²×8`, M=6 |
|---|---|---|
| `Wx` transverse weights | **3.932 GiB** ← largest raw term; **this is what windowing removes** | 0.814 GiB |
| `Wz` axial weights | 0.671 GiB | 0.153 GiB |
| `A` accumulator | **0.895 GiB** | 0.076 GiB |
| `V` permuted (one-time, ×2 copies) | 1.207 GiB | 0.047 GiB |
| **`dd_dense_view_bytes`** | **6.705 GiB** | **1.090 GiB** |

## 2. The real defect: a block-independent floor

After windowing removes `Wx`, `Wz + A = 1.566 GiB` remains and **does not respond to the knobs**,
because both are built at full `n_long` outside the block loop (`dd_dense.jl:235-236`):

| `col_block` × `slab_block` | bytes/view | tiles |
|---|---|---|
| 256 × 256 | 2.069 GiB | 20 |
| 128 × 128 | 1.704 GiB | 63 |
| 64 × 64 | 1.603 GiB | 252 |
| **floor (`Wz` + `A`)** | **1.566 GiB** | — |

Shrinking the knobs buys **6 % of memory for 4× the tiles**; the whole tunable range is 32 %. That
is why the planner lands at `dd = 1` and throughput collapses.

## 3. The fix

**F0 — Set the budget to the device.** On CUDA with 95.6 GiB, the *unwindowed* 6.705 GiB/view
already fits ~14 views/batch. This alone may make UHR work; it must be measured before anything
else is built.

**F1 — Two cost-model bugs (~2 lines, do first).**
- `dd_dense_view_bytes` (`dd_dense.jl:73-74`) charges `V` **per view**; it is built once
  (`dd_dense.jl:29-31`). Inflates 5.50 → 6.705 GiB/view.
- `dd_dense_windowed_view_bytes` (`:187-191`) **sums** the `Wx` term over all column blocks while
  its own comment and the code process one at a time. Over-charges 7×.
Both make the planner window harder than the program needs.

**F2 — Slab-stream `Wz` and `A` (one knob).** Make `l` outermost for both stages; never allocate
anything of extent `n_long`. `_bmm_zl` contracts over `(z, l)`, so partitioning `l` and
accumulating into `P` is exact (partition of a sum). `A` becomes a slab-block transient and the
full-`n_long` accumulator disappears. Measured widths give **134 MB/view** (object) and
**47.8 MB/view** (recon).
- Use `_sum_over_batches` (`loop.jl:83`) — it already does `state .+ chunk` and is production-
  differentiated by `dd_transpose_dense_run`.
- **No `row_block`.** `n_rows = 12`; the axis cannot be blocked.
- **Add a z-window guard for `nz ≳ 32`.** `nz` is small *here* only because this notebook uses 5 mm
  collimation. The library default is `(512,512,64)` (`options.jl:132`) and a committed notebook
  uses `nz = 256`, where `Wz` is 104 GiB/view and slab-blocking alone still needs 3456 tiles.

**F3 — Eliminate the two full `V` copies.** `dd_dense.jl:205-207` materializes `Vl` *and* `V`
before any windowing. Slice windows from the native `(nx,ny,nz,M)` layout instead; transpose only
the `(w, nz·mb, blen)` window.

**F4 — Make the label path reach the compiled driver.** Two defects, both required for UHR:
- `PipelineInput` (`common.jl:27`) pins the 3-D branch to `<:Integer`, which
  `TracedRArray{UInt8,3}` fails. Discriminate on **arity**; guard a 3-D float input with a
  **throw** (use unwrapped-eltype — `_scalar_type` returns `Float64` for `TracedRNumber{UInt8}`
  and cannot make this distinction).
- Compare in the label's own type: `labels .== unwrapped_eltype(labels)(m-1)`, else the whole
  volume promotes to i64.
- `eict_batch_vol` (`eict.jl:222`) is typed to 4-D fractions and **bypasses `material_paths`**, so
  the chunked path never reaches the compiled driver. It needs a label branch.

Gain: UHR input 1.21 GiB → 20 MB, with one-hot chunks formed per window on device.

## 4. Out of scope, with reasons
- **`row_block`** — dead at `n_rows = 12`.
- **Material chunking of `material_paths`** — ceiling 11 % (`Wx` is material-independent), and it
  recomputes `Wz`/`Wx`/`V` `M/mb` times. If material chunking is wanted, it belongs *inside* the
  projector, fused with the window extraction.
- **Spectral fusion** — `P` is 640 KB, not 13 MiB. Saves nothing.

## 5. Correctness invariants
1. `eict_forward` on plain host arrays stays the reference oracle.
2. **The gradient is `Enzyme.Reverse` through `eict_batch_vol`** (`ext:138-139`). `eict_chain_vjp`
   and `poly_log_sinogram_vjp` are **test oracles with no production caller**;
   `dd_transpose_dense_run` serves `hir_operators`. None is on the gradient path.
3. Program shapes stay static; only window starts are traced data.
4. FP: blocked-64 reassociation measured at ≤ 4.1e-7 relative, ~50× inside the repo's own 2e-5
   gate (`test_pipeline.jl:104`). All terms are nonnegative, so blocking is a partial pairwise
   tree — no worse than the flat reduction.

## 6. Open, and must be measured not argued
- **The reverse tape.** `eict_batch_vol` passes `unroll = true`, which bypasses the
  `checkpointing = true` on `_batched_loop` (`ext:49-58`). Unrolled, F2's per-block `Wz_r` may all
  be taped. PROBES §9 measured the traced-loop alternative at **819 s vs 57 s on CPU** — re-measure
  both **on CUDA**, where the whole balance differs.
- **`_bp_chunk`** (`fbp.jl:507+`) broadcasts ~16 tensors at `(nx,ny,nz,B)` and `eict_batch_vol`
  runs it at the DD batch size, ignoring `batching.fdk`. Next constraint after F2.
- **`_batch_key`** (`eict.jl:319`) is `(vertical, n_views)` but `widths` is per-run and baked in as
  a static shape — two runs can silently share the wrong program. **Pre-existing bug.**
- **Zero traced coverage.** No Reactant smoke compiles the windowed projector
  (`smoke_compiled.jl` uses `view_batch = 8` ⇒ `windowed = false`). Write that test first.

## 7. RESOLVED (2026-09-14): the compiled chain's −31.7 HU offset was TF32, not memory

With everything above in place the compiled driver ran the UHR object unwindowed on the GPU — and
read a **constant −31.7 HU** against the host oracle at every grid, view count and slice, while the
host oracle matched the legacy kernels to 0.011 HU. Bisected (`probes/bisect_*.jl`):

| path | GPU | CPU |
|---|---|---|
| `eict_forward_batched` (host composition) | 1e-4 HU | 1e-4 HU |
| `@compile eict_forward` (one program) | **−31.2 HU** | +0.08 HU |
| `compile_pipeline` (batch driver) | **−31.0 HU** | +0.08 HU |
| clean `HEAD`, same driver | **same −3.6 % per batch** | — |

Stage-by-stage on GPU (each compiled stage fed the host's input): projector 4.6e-4 rel, spectral
matmul 4.2e-4, **ramp filter 9.3e-3 with a −1.2e-3 bias**, backprojection 5e-5, HU map 7e-8.

**Cause.** XLA:GPU lowers f32 `stablehlo.dot_general` with `precision_config = DEFAULT` to **TF32**
(10-bit mantissa, ~1e-3 relative). Three stages are matmuls; the ramp filter is `H * rows` against a
Toeplitz matrix whose dynamic range (one large central tap, long small tails) turns truncation into
a bias, and a biased filtered projection integrates to a constant image offset. XLA:CPU has no TF32
path, so every CPU smoke passed.

**Fix.** Trace under `Reactant.PrecisionConfig.HIGHEST`: `compile_pipeline` now does so by default
(`precision = :highest`; `:default` reproduces the bias for measurement), and direct
`@compile eict_forward` calls go through `BSF.with_full_precision`. Measured: 32² −0.0012 HU,
512² +0.0004 HU (rms 0.009), forward 1.8 ms/view at 512²; `test/functional/reactant/smoke_gpu_parity.jl`
asserts it on the default backend. `DotGeneralAlgorithmPreset.TF32_TF32_F32_X3` is not usable in
Reactant 0.2.285 (its type check rejects `TF32`).

**Lesson for the memory work above:** none of it was wrong, and all of it was unnecessary on this
device — the object grid fits unwindowed at `dd = 11` views/batch once `batch_budget_mb` is sized to
the card. The blocking design (§3 F2/F3) stays on file for CPU-only machines and for `nz ≳ 32`.
