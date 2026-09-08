# The published n-channel VMI estimator — map of the notebook code

Source of truth: `docs/notebooks/04_pcct_vmi.jl` (canonical, K = 4 PCCT bins,
global response tables) mirrored in `docs/notebooks/03_dual_kvp_switching_vmi.jl`
(K = 2 dual-kVp, per-ray response tables) and
`docs/notebooks/12_siemens_flash_ufc.jl` (K = 2 Flash dual-source, per-ray
tables, same kernel as nb03).  `grep -rn nchannel src/` had zero hits before this
port; the legacy `apply_cong!` in `src/reconstruction/vmi/cong.jl` is a
different (water-first univariate) estimator and is NOT what the published
results use.

Pure port: `src/functional/nchannel.jl`.  Oracle test:
`test/functional/test_nchannel.jl` (runs the ORIGINAL notebook cells, extracted
verbatim by cell UUID with `include_string`, side by side).

## 1. Cells and symbols (nb04 line numbers; identical UUIDs reused by nb03)

| Cell UUID | nb04 lines | Symbol(s) | Role |
|---|---|---|---|
| `4ca28c64-…` | 321-360 | `nchannel_basis` | host config: `E`, `Φ[e,k]`, `μρ_I[e]`, `μρ_W[e]`, `I0[k]`, `μI_eff[k]`, `μW_eff[k]`, `normal_II/IW/WW` (scalars); asserts `I0_relerr < 5e-5` |
| `4985581f-…` | 363-377 | `nchannel_controls` | the ten host constants (below) |
| `73371177-…` | 379-792 | `nchannel_forward`, `nchannel_golden_minimize`, `nchannel_scalar_global`, `nchannel_solve_total_C`, `nchannel_cong_constrained_reference`, `nchannel_poisson_quasi_nll`, `nchannel_profile_reference`, **`nchannel_profile_tile!`** | Float64 host reference solvers (certification only) + the Float32 production `AK.foreachindex` kernel |
| `50ed35bb-…` | 834-861 | `nchannel_slab_counts` | nb04 only: sums `exp(-h)` over the central `nrows` detector rows per bin, re-logs → `(n_col, 1, n_view)` per bin; nb03/nb12 instead keep every native row (`build_nchannel_slab_counts`) |
| `b86a9c50-…` | 874-953 | `sino_basis_nchannel_slab` | tile driver: `BS.tile_ranges(n_view, 8)`, per tile `to_gpu`, call kernel, copy back. nb04 additionally **recomputes the feasibility bit 0x10 on the host** in Float64 from the global attainable count range; nb03/nb12 keep the kernel's bit |
| `ddfde8bb-…` | 979-1030 | `nchannel_common_fbp_slice` | nb04: angular raised-cosine anti-alias (FFT over views) + `SoftFilter` FDK for BOTH bases; nb03/nb12: direct FDK with the two `CustomFilter` per-basis kernels (nb03 still defines the angular response but never uses it — dead code) |
| `f3de45a6-…` | 1035-1160 | `total_measured_counts`, `total_expected_counts`, `tlbf_filter_pair`, `reconstruct_material_pair`, `synthesize_vmi_stack` | nb04 only: Lee-2025 total-likelihood bilateral filter (T-LBF) between decomposition and FBP; `synthesize_vmi_stack` shared |
| `040e0002-…-0002` | 1203-1218 | `pcct_final` / `dual_final` | `apply_acnr_kalender!` then `synth_vmi_2basis` at 50/70/100/140 keV |

### Controls (identical in nb03, nb04, nb12)

```
iodine_bounds = (-0.10, 0.40) g/cm²    water_bounds = (-2.0, 50.0) g/cm²
outer_iterations = 16                  inner_iterations = 12
max_iodine_step = 0.05                 max_water_step = 5.0
parameter_tolerance = 5e-5             fisher_condition_limit = 1e8
air_gate = 0.0 (disabled)              tile_views = 8
```
Hidden constants inside the kernel: 28 bisection steps, count floor `1f-6`,
curvature/determinant floor `1f-12`, initializer fallback `(A, C) = (0, 20)`,
bound-contact tolerance `2f-4`.

## 2. Shapes

Global tables (nb04): `Φ::(nE, K)`, `I0::(K,)`, `μI_eff/μW_eff::(K,)`,
`normal_*::Float32`.  Per-ray tables (nb03/nb12): `Φ::(n_col, n_row, nE, K)`,
`I0::(n_col, n_row, K)`, `μ*_eff::(n_col, n_row, K)`, `normal_*::(n_col, n_row)`.
`μρ_I, μρ_W::(nE,)` always.  Channel data `hs::NTuple{K}` of `(n_col, n_row,
n_view_tile)` Float32 log transmissions `h_k = -log(y_k / I0_k)`.
Outputs, all `(n_col, n_row, n_view_tile)`: `sino_I` (A, iodine g/cm²),
`sino_W` (C, water g/cm²), `fisher_AA/AC/CC`, `score_norm`, `quality_flag::UInt8`,
`outer_count::UInt8`, `inner_count::UInt8`.

## 3. The per-ray algorithm (`nchannel_profile_tile!`)

Model: `λ_k(A,C) = Σ_e Φ[e,k]·exp(-μρ_I[e]·A − μρ_W[e]·C)`, data
`y_k = max(I0_k·exp(-h_k), 1e-6)`, objective `L = Σ_k (λ_k − y_k·log λ_k)`
(Poisson quasi-likelihood).  Analytic `∂λ/∂A = −Σ μρ_I Φ e^{…}`,
`∂λ/∂C = −Σ μρ_W Φ e^{…}`; the Fisher blocks are `Σ_k (∂λ_k)²/λ_k` with `λ`
floored at `1e-6`.

1. **Air gate**: `max_k |h_k| < air_gate` → all outputs zero, flag 0. Never
   fires with `air_gate = 0`.
2. **Linear initializer** (K-channel effective-energy least squares):
   `rhs_I = Σ μI_eff[k] h_k`, `rhs_W = Σ μW_eff[k] h_k`,
   `det0 = nII·nWW − nIW²`; valid iff finite and `> 1e-12`;
   `A0 = clamp((nWW·rhs_I − nIW·rhs_W)/det0)`, `C0 = clamp((nII·rhs_W − nIW·rhs_I)/det0)`,
   else `(A0, C0) = (clamp(0), clamp(20))`.
3. **Aggregate bracket + 28-step bisection** on `total(C) = Σ_k λ_k(A0, C)`
   against `y_total = Σ_k y_k`: if `total(C_lo) ≥ y_total ≥ total(C_hi)` run 28
   bisections (`total(mid) > y_total ⇒ lo = mid`) and set `C = (lo+hi)/2`;
   otherwise keep `C0`.  Also records `aggregate_feasible =
   total(A_lo, C_lo) ≥ y_total ≥ total(A_hi, C_hi)` (flag bit 16).
4. **Profile Newton, outer ≤ 16** (`break` on convergence):
   - inner ≤ 12 (`break`): `gC = Σ (1 − y/λ)·dC`, `FCC = Σ dC²/λ`;
     `C ← clamp(C − clamp(gC/max(FCC,1e-12), ±5), C_lo, C_hi)`;
     done when `|ΔC| ≤ 5e-5·(1+|C|)` (the step IS applied before the break).
   - envelope step: `gA, FAA, FAC, FCC` at `(A, C)`;
     `Hprof = max(FAA − FAC²/max(FCC,1e-12), 1e-12)`;
     `A ← clamp(A − clamp(gA/Hprof, ±0.05), A_lo, A_hi)`;
     converged when `|ΔA| ≤ 5e-5·(1+|A|)`.
5. **Re-profile water** at the final `A`: inner ≤ 12 again; `converged &= c_converged`.
6. **Final score / Fisher** at `(A, C)`: `score_norm = ‖(gA,gC)‖/√max(FAA+FCC,1e-12)`;
   2×2 eigen-ratio `> 1e8` ⇒ ill-conditioned.
7. **Flags**: `1` A at bound (±2e-4), `2` C at bound, `4` not converged,
   `8` ill-conditioned or invalid initializer, `16` aggregate infeasible,
   `32` non-finite.  `outer_count` = outer iterations executed,
   `inner_count` = ALL inner iterations executed (including the re-profile).

## 4. Where the 2-channel dual-kVp and the 4-bin PCCT cases differ

Only in the TABLES, never in the algorithm:

| | nb04 (PCCT, K=4) | nb03 / nb12 (dual-kVp, K=2) |
|---|---|---|
| response | `W_applied[e,k]` from the workspace, global; scaled by `nrows` (slab) | per-ray `response[col,row,e]·I0_ray[col,row]` per channel, energy grids of the two channels merged (`sort(unique(vcat(...)))`, zeros where a channel has no support) |
| `μ_eff` normaliser | `Φsum = sum(Φ; dims=1)` | `Φsum = max.(I0, eps(Float32))` |
| kernel indexing | `Φ[e,k]`, `I0[k]`, `μ*_eff[k]`, scalar normals | `Φ[col,row,e,k]`, `I0[col,row,k]`, `μ*_eff[col,row,k]`, `normal_*[col,row]` (`col, row` decoded from the linear index) |
| rows | central `nrows` rows summed in the count domain → 1 row | every native row solved independently |
| feasibility bit | recomputed on host (Float64, global range) | kernel value |
| downstream | T-LBF (5×5, α1 = 0.9, α2 = 24.635648571666497) → angular anti-alias + `SoftFilter` FDK both bases → ACNR (hp 1.5 px, window 4, 4 passes, β_max 20) | direct FDK, iodine `CustomFilter((0,.25,.5,.75,1),(1,.40,.12,.03,.001))`, water `CustomFilter(…,(1,.8744,.6003,.3031,.0266))` → ACNR (hp 1.5 px, window 4, 5 passes, β_max 14; nb12 uses hp 4.0 px) |
| VMI synth | `synth_vmi_2basis(water, iodine·1000; energy_keV)` at 50/70/100/140 | same |

## 5. Mirror-drift report (automated in the test as well)

- nb03 `nchannel_profile_tile!` ≡ nb12 `nchannel_profile_tile!` byte-for-byte
  modulo indentation (`diff -w` empty).
- nb04 kernel vs nb03 kernel: differs ONLY in the table-indexing lines listed
  in §4 (`::Float32` annotations on the normals, the `col/row` decode, and
  `[col,row,…]` subscripts).  After normalising those substitutions the two
  bodies are identical (the test asserts this).
- Reference solvers (`nchannel_forward` … `nchannel_profile_reference`): nb03 ≡
  nb04 byte-for-byte.  nb12 carries no reference solvers.
- nb03 `build_nchannel_slab_counts` / `build_nchannel_basis` /
  `run_nchannel_profile` ≡ nb12 except two metadata fields
  (`target_slice_thickness_mm`, `thickness_mm`) and an `@info` line.
- Controls identical in all three notebooks (comments differ).
- nb03 defines `nchannel_fbp_angular_response` (nb04's anti-alias window) but
  never applies it — dead code, harmless.

## 6. What the pure port does differently (and why it is still the same estimator)

- **`break` ⇒ freeze masks.**  Every loop runs its fixed ceiling
  (28 / 16×12 / 12) over the whole tile; a per-ray `done` mask freezes the
  iterate once the notebook would have `break`-ed.  Because the frozen value is
  exactly the value the notebook keeps, the fixed-count program computes the
  same function (differences are Float32 reduction-order rounding, ~1e-7).
  Iteration counters are accumulated from the same masks.
- **Energy contraction as a matrix product.**  The moment table
  `[Φ | μρ_I⊙Φ | μρ_W⊙Φ]` (`nE × 3K`, or `(n_col,n_row,nE,3K)` per-ray) turns
  the per-ray energy loop into one `exp` broadcast and one product per
  evaluation; `Σ_k λ_k` uses the row-summed `Φ_total`.
- **Attainable bounds precomputed on the host** (config-only; the notebook
  recomputes them per ray inside the kernel).
- **Flags and counters are returned as `T`-valued integers** (exactly
  representable) so the traced program never converts float→integer; a host
  helper `nchannel_flags_u8` gives the notebook's `UInt8`.
- **Channels are one `(n_col, n_row, n_view, K)` tensor** instead of an
  `NTuple{K}` of 3-D arrays (`nchannel_stack_channels` converts).
- The Float64 host reference solvers (golden-section / grid scans) are not
  ported: they are certification tools, not part of the estimator.
- T-LBF (nb04 only) IS ported (`nchannel_tlbf`) as a 25-tap static gather
  because it reuses the estimator's forward model; it assumes what the
  notebook assumes (each detector row filtered independently; nb04 runs it on
  a single row).
- nb04's angular anti-alias FFT window IS ported (`nchannel_angular_plan` /
  `nchannel_angular_apodize`) as a circulant `(n_view × n_view)` matrix
  product along the view axis (kernel `c[m] = (1/N) Σ_j H_j cos(2πjm/N)`,
  host-precomputed) — no FFT inside the traced program.
- **Reactant typing rule** (from the HIR builder): `TracedRArray{T,N} <:
  AbstractArray{TracedRNumber{T},N}`, so no signature binds `T` from an
  array eltype.  Arrays are `AbstractArray{<:Any,N}`; `T` comes only from the
  plan (`NChannelPlan{T}`, `NChannelTLBFPlan{T}`) or an explicit `::Type{T}`
  argument (`nchannel_combine_rows`, `nchannel_synth_vmi`); the plan structs'
  array parameters are unconstrained `<: AbstractArray` so
  `Reactant.to_rarray(plan)` can rebuild them with traced fields.

## 7. API of `src/functional/nchannel.jl`

| Function | Role |
|---|---|
| `nchannel_plan(Φ, energies, I0; T, scale, controls…)` | host constructor (global `(nE,K)` or per-ray `(n_col,n_row,nE,K)` tables) |
| `nchannel_merge_channels(energies_k, Φ_k)` | nb03's energy-grid merge |
| `nchannel_stack_channels(hs)` | `NTuple{K}` of 3-D → `(…, K)` |
| `nchannel_combine_rows(h, rows, T)` | nb04 slab row combination |
| `nchannel_moments(A, C, plan)` | `λ, ∂λ/∂A, ∂λ/∂C` (= `nchannel_forward`, vectorised) |
| `nchannel_total_counts(A, C, plan)` | `Σ_k λ_k` |
| **`nchannel_estimate_tile(h, plan)`** | the estimator on one tile |
| `nchannel_estimate(h, plan; tile_views)` | view-tiled driver |
| `nchannel_flags_u8`, `nchannel_counts_u8` | host `T`→`UInt8` diagnostics |
| `nchannel_tlbf_plan`, `nchannel_tlbf`, `nchannel_total_measured`, `nchannel_total_expected` | nb04 T-LBF |
| `nchannel_angular_plan`, `nchannel_angular_apodize` | nb04 angular anti-alias |
| `nchannel_vmi_alphas`, `nchannel_synth_vmi` | pure `synthesize_vmi_stack` |
| `nchannel_vmi_chain(h, plan; fbp_iodine, fbp_water, synth, energies, sino_filter, acnr)` | the chain with injected downstream stages |

## 8. Parity evidence (`test/functional/test_nchannel.jl`, plain Arrays, Float32)

Fixtures: real xspect spectra on a 2-keV grid (61 energies); nb04-style
K = 4 PCCT windows of 120 kVp (global tables) and nb03-style K = 2 (80/140
kVp) with a column-dependent water bowtie (per-ray tables); 64-ray tiles on an
8 × 8 (A, C) truth grid, noise-free and with Gaussian-Poisson noise; the
verbatim kernels run through `BS.AK.foreachindex` on the CPU.

| case | sino_iodine rel | sino_water rel | Fisher rel | flags | outer / inner used (max) |
|---|---|---|---|---|---|
| K=4 noise-free | 1.4e-6 | 2.4e-7 | ≤1.1e-7 | 0 mismatches | 6 / 22 |
| K=4 noisy | 1.0e-6 | 2.5e-7 | ≤1.1e-7 | 0 | 13 / 42 |
| K=4 slab ×3 | 1.3e-6 | 2.5e-7 | ≤1.0e-7 | 0 | 13 / 42 |
| K=2 per-ray noise-free | 4.2e-6 | 6.7e-7 | ≤1.4e-6 | 0 | 5 / 14 |
| K=2 per-ray noisy | 2.5e-6 | 7.5e-7 | ≤2.6e-7 | 0 | 5 / 14 |
| end-to-end (64×2×48 rays, 6 tiles) | 7.1e-6 | 3.2e-6 | — | — | 4 / 13 |

Iteration counters agree ray-for-ray except one knife-edge inner count in the
K = 4 noise-free tile.  End-to-end VMI HU images through the legacy
`fdk_reconstruct` (nb03 per-basis `CustomFilter`s) + `apply_acnr_kalender!`
(5 × 14, hp 1.5 px) + `synth_vmi_2basis`: 1.0e-6 rel.  T-LBF vs
`tlbf_filter_pair`: 4.9e-7 (α2 = 24.64), exact for α2 ∈ {∞, 0}.  Angular
apodization vs the FFT window: ≤1e-5 (Float32), ≤1e-6 (Float64).
Float64 smoothness (tol 1e-11): central differences at ε = 1e-4 / 2e-4 agree
to 2.3e-8, and match the implicit-function sensitivity `-F⁻¹∇λ_k` to 5.1e-8.
The production ceilings (16 / 12) were never reached on any fixture ray
(worst 13 outer, 42 total inner).
