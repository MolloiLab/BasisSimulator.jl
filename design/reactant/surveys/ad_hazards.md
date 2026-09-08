# BasisSimulator.jl — Reactant.jl / Enzyme.jl hazard inventory

Exhaustive audit of `src/` (61 files, 26,247 lines). Every file was read in full; greps were used
to cross-check each hazard class for completeness.

## Bottom line

The package is unusually close to traceable in one respect: there are **no hand-written `@kernel`
macros, no KernelAbstractions kernels, and no `@metal` / `@cuda` launches anywhere in `src/`**. All
92 device kernels go through `AK.foreachindex`, dispatched by array type. That is a single rewrite
target with a single calling convention.

But every one of those kernels is written as a *scalar-indexed imperative loop nest*, which is
exactly the form Reactant cannot ingest. The five hard walls are:

1. **Data-dependent `while` ray marches** in every projector (Siddon DDA, DD triple-nested
   footprint loops, DD transpose gather loops).
2. **Exact integer Poisson sampling** (`_poisson_sample`, `Poisson_approx`) with `while true`
   rejection loops and CPU staging round-trips.
3. **Brent root-finding + Newton-with-break inside a per-ray GPU kernel** (`apply_cong!` →
   `brent_solve`), including bit-level `reinterpret` midpoint arithmetic.
4. **Per-row SVD, median filters, and bilateral filters** across the whole denoising stack.
5. **`try/catch` OOM retry that changes tile shapes at runtime** (`with_oom_retry`).

---

# Stage-by-stage hazard inventory

## Forward projection — Siddon (`src/projection/siddon.jl`)

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| siddon.jl:302 | `siddon_trace_ray` | `while t_current < t_exit && iter < max_iter` — data-dependent DDA trip count | Fixed-trip loop over `max_iter = nx+ny+nz+10` with an active mask; `@trace` for Reactant |
| siddon.jl:306 | `siddon_trace_ray` | `break` on out-of-bounds voxel index | Replace with `active &= inbounds` predicate, multiply contribution by `active` |
| siddon.jl:256-258 | `siddon_trace_ray` | `unsafe_trunc(Int32, floor(...))` entry-voxel index | Keep (index math, zero-gradient); for Reactant emit as an integer op, not a Julia `floor` |
| siddon.jl:261-263 | `siddon_trace_ray` | `clamp(ix, 0, nx-1)` | Fine (`clamp` is a subdifferentiable min/max chain) |
| siddon.jl:266-268 | `siddon_trace_ray` | `ray_x >= 0 ? Int32(1) : Int32(-1)` step direction | `ifelse` / `sign` — value is piecewise-constant, gradient legitimately zero |
| siddon.jl:214-216 | `siddon_trace_ray` | epsilon-padding ternaries on ray direction | `ifelse(abs(r) < eps, copysign(eps, r), r)` |
| siddon.jl:228-236 | `siddon_trace_ray` | swap-if for t-interval sorting (3 blocks) | `minmax()` pairs, branchless |
| siddon.jl:243-245 | `siddon_trace_ray` | early `return zero(T)` on ray miss | Multiply result by a `hit` mask instead |
| siddon.jl:248 | `siddon_trace_ray` | `t_enter = max(t_enter, zero(T))` | Fine |
| siddon.jl:271-273 | `siddon_trace_ray` | `dt_x = abs(vsx / ray_x)` — division by an epsilon-guarded value | Fine given the guard |
| siddon.jl:276-292 | `siddon_trace_ray` | 3 sign branches computing `t_next_{x,y,z}` | `ifelse` |
| siddon.jl:311 | `siddon_trace_ray` | `t_next = min(t_next_x, t_next_y, t_next_z, t_exit)` | Fine |
| siddon.jl:316-320 | `siddon_trace_ray` | `if path_length > eps` gating the voxel gather | `ok = path_length > eps; accum += ifelse(ok, μ*pl, 0)` |
| siddon.jl:318 | `siddon_trace_ray` | `volume[ix+1, iy+1, iz+1]` scalar index on a traced array | Reformulate as a `gather` over precomputed index tensors |
| siddon.jl:323-331 | `siddon_trace_ray` | branchless DDA step via `Int32(cmp)*Int32(cmp)` | Already branchless; keep |
| siddon.jl:471-476 | `siddon_forward_project!` | `Int32(size(...))` shape extraction | Static shapes required |
| siddon.jl:482 | `siddon_forward_project!` | `vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov` | Host branch on a `Union{Nothing,...}` |
| siddon.jl:505-532 | `siddon_forward_project!` | 4 × `ws_X !== nothing ? ws_X : (similar + copyto!)` | Resolve workspace buffers before tracing |
| siddon.jl:535 | `siddon_forward_project!` | `AK.foreachindex` over all rays | Array-level `Reactant.@jit`-able map |
| siddon.jl:538-542 | `siddon_forward_project!` | linear→(col,row,angle) via `%` and `÷` | Reshape instead |
| siddon.jl:565-579 | `siddon_forward_project!` | `if arc_det` arc/flat detector branch; `sin`/`cos`/`sqrt` | Host `Bool`, resolves at trace time — fine if hoisted to a type parameter |
| siddon.jl:717-719 | `_fused_accum_energies` | `@generated` + `@inbounds(μ_tbl[mat, i])` — **LUT gather by data index** | Gradient wrt `mat` is zero (mask is discrete); gradient wrt `μ_tbl` **is needed** and is a scatter-add — declare `μ_table_gpu` Duplicated |
| siddon.jl:728-738 | `_fused_beer_lambert` | `@generated` unrolled `Σ w·exp(-L)` | Differentiable; replace `@generated` with `ntuple`+`Val` or an array reduction for Reactant |
| siddon.jl:747-757 | `_fused_beer_lambert_bt` | `@generated` unrolled sum with bowtie linear-index gather `bt[bt_base + (e-1)*ncnr]` | Same; make the bowtie a broadcast over an energy axis |
| siddon.jl:867 | `siddon_fused_poly_project!` | `_bt = has_bowtie ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)` — **dummy array to dodge `Nothing` in the GPU closure** | Dynamic-shape dummy; use a zero-sized static alternative or two traced variants |
| siddon.jl:870-878 | `siddon_fused_poly_project!` | `let` block capturing 25 variables | Fine |
| siddon.jl:880 | `siddon_fused_poly_project!` | `AK.foreachindex` | Array-level map |
| siddon.jl:951-961 | `siddon_fused_poly_project!` | early `return` inside the kernel body (air-ray path) | Compute both branches, select with `ifelse` |
| siddon.jl:952, 1007, 1340 | fused kernels | `ntuple(_ -> zero(T), Val(N_E))` accumulators | Fine (compile-time N) |
| siddon.jl:959, 1059 | `siddon_fused_poly_project!` | `-log(max(I, 1e-10))` | Non-smooth at the clamp; use a log-space floor or accept the subgradient |
| siddon.jl:970-975 | `siddon_fused_poly_project!` | `unsafe_trunc(Int32, floor(...))` + `clamp` ×3 | As line 256 |
| siddon.jl:1016 | `siddon_fused_poly_project!` | `while` DDA (second copy) | As line 302 |
| siddon.jl:1020-1022 | `siddon_fused_poly_project!` | bounds `break` | Mask |
| siddon.jl:1029 | `siddon_fused_poly_project!` | `mat = Int32(mask[...]) + 1` — integer material LUT index | Discrete; gradient flows only into `μ_tbl` values |
| siddon.jl:1032 | `siddon_fused_poly_project!` | `_fused_accum_energies` call | See line 717 |
| siddon.jl:1035-1044 | `siddon_fused_poly_project!` | branchless DDA step | Keep |
| siddon.jl:1098-1103 | `_tiled_accum_energies` | `@generated` K-energy tile accumulate with `μ_tbl[mat, ts+i-1]` | Same LUT hazard |
| siddon.jl:1113-1122 | `_tiled_beer_lambert_col` | `@generated` per-bin partial sum `W[ts+i-1, b]` | Differentiable |
| siddon.jl:1130-1140 | `_tiled_beer_lambert_col_bt` | **`min(exp(-accums), T(1.0e30))` Float32 overflow guard** | Kills gradient above the cap; clamp in log space instead |
| siddon.jl:1255 | `siddon_fused_spectral_project!` | dummy `_bt` array again | As line 867 |
| siddon.jl:1269 | `siddon_fused_spectral_project!` | `AK.foreachindex` on a *pilot* array used only for iteration count | Reactant needs an explicit shape, not a pilot |
| siddon.jl:1345 | `siddon_fused_spectral_project!` | `if t_enter < t_exit && t_exit > zero(T)` wrapping the whole march | Mask instead of branch |
| siddon.jl:1352-1357 | `siddon_fused_spectral_project!` | `unsafe_trunc(Int32, floor(...))` + `clamp` ×3 | As line 256 |
| siddon.jl:1387 | `siddon_fused_spectral_project!` | `while` DDA (third copy) | As line 302 |
| siddon.jl:1390-1392 | `siddon_fused_spectral_project!` | bounds `break` | Mask |
| siddon.jl:1398 | `siddon_fused_spectral_project!` | mask LUT index | Discrete |
| siddon.jl:1421 | `siddon_fused_spectral_project!` | `if hbt` branch selecting two whole loop bodies | Two traced variants |
| siddon.jl:1423, 1428 | `siddon_fused_spectral_project!` | `for b in Int32(1):nb` with runtime `nb` | Lift `n_bins` to `Val` |
| siddon.jl:1425, 1430 | `siddon_fused_spectral_project!` | `oflat[idx + (b-1)*ne] += I_partial` — **in-place accumulate into a caller buffer across host tile calls** | Return per-tile arrays and reduce, or mark `outputs_flat` Duplicated with a zeroed shadow |

## Forward projection — Distance-Driven (`src/projection/dd.jl`, `dd_fast.jl`)

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| dd.jl:83 | `_dd_overlap` | `hi > lo ? hi - lo : zero(T)` | `max(hi-lo, 0)` — ReLU, subdifferentiable, keep |
| dd.jl:96-97 | `_dd_bounds` | two divisions by `vstep` after an `inv_mf` multiply | Fine |
| dd.jl:98-99 | `_dd_bounds` | `clamp(c, 0, n+1)` | Fine |
| dd.jl:100-101 | `_dd_bounds` | `unsafe_trunc(Int32, floor(c_lo))` / `ceil(c_hi)` — **loop bounds derived from traced geometry** | Core dynamic-shape blocker: bounds depend on data. Use a fixed max footprint + overlap mask |
| dd.jl:124-139 | `_dd_col_setup` | `if arc_det` branch producing 4 boundary coordinates; `sin`/`cos`/`sqrt` | Host `Bool` |
| dd.jl:142 | `_dd_col_setup` | `vertical = abs(sy) >= abs(sx)` — **axis swap chosen from a traced source position** | Hoist per-view: compute both orientations, `ifelse`-select; or precompute a per-angle host flag |
| dd.jl:143-153 | `_dd_col_setup` | 10 ternaries selecting axis-role variables off `vertical` | `ifelse` |
| dd.jl:156-157 | `_dd_col_setup` | `scaleL = s_long / (s_long - dlL)` — division that can blow up | Guard |
| dd.jl:160-163 | `_dd_col_setup` | `min`/`max` of the two projected boundaries | Fine |
| dd.jl:165 | `_dd_col_setup` | `detXstep > T(1.0e-12)` validity flag returned in a tuple | `ifelse` |
| dd.jl:186-189 | `_dd_row_setup` | `valid = detXstep > 1e-12 && detZstep > 1e-12`; `norm = valid ? invCos/(detXstep*detZstep) : 0` | `ifelse` + `norm / max(step, eps)` |
| dd.jl:211-214 | `_dd_cell_setup` | `valid_x && valid_z`; `valid ? norm : zero(T)` | `ifelse` |
| dd.jl:237 | `_dd_trace_cell` | `valid \|\| return zero(T)` early exit | Mask |
| dd.jl:241, 250, 258 | `_dd_trace_cell` | **triple nested `while`** over `il`/`it`/`ip`, all data-dependent | Fixed-trip over `n_long` (static) × max footprint (bounded by magnification), masked |
| dd.jl:243-245 | `_dd_trace_cell` | `mag_fac = s_long/(s_long - lp)`; `inv_mf = 1/mag_fac` | Guard the divisions |
| dd.jl:254, 262 | `_dd_trace_cell` | `if ox > zero(T)` / `if oz > zero(T)` gating the gather | Multiply by overlap (already zero when non-overlapping) |
| dd.jl:255-256 | `_dd_trace_cell` | `ixv = vertical ? it : il`; `iyv = vertical ? il : it` — index chosen by data | `ifelse` on the index, then gather |
| dd.jl:263 | `_dd_trace_cell` | `volume[ixv, iyv, ip]` scalar gather | Gather op |
| dd.jl:279-376 | `_dd_trace_rows4` | 4 replicated inner `while` loops each behind `valid1..4` predicates | Vectorize the row lane as a leading array axis |
| dd.jl:293-308 | `_dd_trace_rows4` | `use1..use4 = rowN <= n_rows` tail predication; 4 × `valid_x && validN && useN` | Mask |
| dd.jl:330, 340, 350, 360 | `_dd_trace_rows4` | 4 × `if validN` around identical inner loops | Predicate |
| dd.jl:336, 346, 356, 366 | `_dd_trace_rows4` | 4 × `oz > zero(T) && (accN += ...)` short-circuit accumulate | `ifelse` |
| dd.jl:381-386 | `_dd_geom_array` | `ws !== nothing && return ws` | Host |
| dd.jl:412 | `dd_forward_project!` | `volume_extent !== nothing ? ... : geom.fov` | Host branch |
| dd.jl:415, 466, 561, 687 | all entries | `_dd_check_isotropy(vx, vy)` | See dd.jl:521 |
| dd.jl:429, 479, 586, 713 | forward entry points | `AK.foreachindex` ×4 | Array-level ops |
| dd.jl:430-434 | `dd_forward_project!` | `%`/`÷` index decomposition | Reshape |
| dd.jl:462 | `_dd_forward_project_arc_rowtile4!` | `ntiles = (nr + 3) ÷ 4` | Host |
| dd.jl:485 | `_dd_forward_project_arc_rowtile4!` | `slot >= ntiles && return` early kernel exit | Mask |
| dd.jl:496-499 | same | `row2 <= nr && (sino[col, row2, angle] = a2)` conditional store ×3 | Predicated store / masked scatter |
| dd.jl:521-527 | `_dd_check_isotropy` | `if abs(vx-vy) > 1e-4*max(vx,vy)` then `@warn ... maxlog=1` | Move out of any traced path |
| dd.jl:576-577, 702-703 | fused entries | `has_bowtie` dummy-array trick | As siddon.jl:867 |
| dd.jl:597-600, 724-727 | fused kernels | `_dd_cell_setup` returning a 15-element mixed tuple incl. a `Bool` | Fine for Enzyme; Reactant needs the Bool as a mask |
| dd.jl:603, 730 | fused kernels | `if valid` wrapping the whole march | Mask |
| dd.jl:605-634, 732-761 | fused poly/spectral | nested `while` (copies 2 and 3) | As dd.jl:241 |
| dd.jl:625, 752 | fused kernels | `mat = Int32(mask[ixv, iyv, ip]) + 1` LUT index | Discrete |
| dd.jl:626, 753 | fused kernels | `_fused_accum_energies` / `_tiled_accum_energies` | See siddon.jl:717 |
| dd.jl:637-643 | `dd_fused_poly_project!` | `if hbt` branch; `-log(max(I_total, 1e-10))` | Mask + log floor |
| dd.jl:764-773 | `dd_fused_spectral_project!` | `if hbt`; `for b in 1:nb`; `oflat[...] += ...` cross-call accumulate | As siddon.jl:1425 |
| dd_fast.jl:56-59 | `_plen_accum` | `@generated` **M-way `ifelse` select chain** on `mat` | Already the Enzyme-friendly form; for Reactant emit as a one-hot matmul `P += onehot(mat) * w` |
| dd_fast.jl:62-68 | `_plen_line_integral` | `@generated` `Σ_m P[m]·μ_tbl[m,e]` | This is a **matvec** — rewrite as `μ_tblᵀ * P`, fully traceable and gives the μ-table gradient for free |
| dd_fast.jl:71 | — | `const _PLEN_MAX_MATERIALS = 64` | Register-budget constant |
| dd_fast.jl:73-81 | `_warn_dd_fast_fallback` | `@warn ... maxlog=1` | Host-only |
| dd_fast.jl:103 | `_dd_fused_poly_plen!` | `n_E = Int32(length(wη_gpu))` — runtime loop bound | Lift to `Val` |
| dd_fast.jl:112, 207 | `_dd_fused_*_plen!` | `AK.foreachindex` ×2 | Array-level |
| dd_fast.jl:128, 223 | both | `plens = ntuple(_ -> zero(T), Val(M))` | Fine |
| dd_fast.jl:129, 224 | both | `if valid` wrapping the march | Mask |
| dd_fast.jl:131-159, 226-255 | both | nested `while` ×2 (same DD march) | As dd.jl:241 |
| dd_fast.jl:141, 236 | both | `ox > zero(T)` gate | Mask |
| dd_fast.jl:150, 245 | both | `oz > zero(T)` gate | Mask |
| dd_fast.jl:151, 246 | both | `mat = Int32(mask[ixv,iyv,ip]) + 1` LUT index | Discrete |
| dd_fast.jl:166-174 | `_dd_fused_poly_plen!` | `while e <= nE` energy loop with `if hbt` and `bt[bt_base + (e-1)*ncnr]` linear-index gather | Batch over energy as an array axis; bowtie becomes a broadcast |
| dd_fast.jl:175 | same | `-log(max(I_total, 1e-10))` | Log floor |
| dd_fast.jl:260-276 | `_dd_fused_spectral_plen!` | `while b <= nb` / `while i < kk` nested runtime loops | Lift to `Val`; make bins an array axis |
| dd_fast.jl:267-270 | same | `if hbt` inside the innermost loop; `min(trans, T(1.0e30))` Float32 overflow guard | Hoist the branch; clamp in log space |
| dd_fast.jl:274 | same | `oflat[idx + (b-1)*ne] += acc` | As siddon.jl:1425 |
| dd_fast.jl:325, 399 | public wrappers | host `if size(μ_table_gpu,1) > 64` picks a **different kernel** | Two trace variants; pick before tracing |
| dd_fast.jl:359-360, 435-436 | public wrappers | `has_bowtie` dummy-array trick | As siddon.jl:867 |
| dd_fast.jl:362, 438 | public wrappers | `Val(size(μ_table_gpu, 1))` — **type parameter from a runtime size** | Dynamic specialization; freeze `n_materials` |

## Backprojection / adjoint (`dd_transpose.jl`, `reconstruction/core/backprojection.jl`)

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| dd_transpose.jl:16-20 | `_dd_arc_row_bounds` | 4 divisions by `rho_min`/`rho_max`, then `min`/`max` of 4 values | Guard |
| dd_transpose.jl:22-23 | `_dd_arc_row_bounds` | `unsafe_trunc(Int32, ceil/floor)` row envelope from data | Fixed max row window + mask |
| dd_transpose.jl:42-44 | `_dd_project_point` | `if sd_xy <= 1e-12 \|\| sv_xy <= 1e-12; return (Inf, Inf)` | `ifelse` to a large finite value plus a validity mask |
| dd_transpose.jl:47 | `_dd_project_point` | `γ = atan(across, along)` | Differentiable; watch the branch cut |
| dd_transpose.jl:53 | `_dd_project_point` | `abs(denom) <= 1e-12 && return (Inf, Inf)` | Same |
| dd_transpose.jl:79 | `_dd_backproject_arc_tile4!` | `ntiles = (active_nz + 3) ÷ 4` | Host |
| dd_transpose.jl:90 | same | `view(volume, :, :, z_first:(z_first+ntiles-1))` — **dynamic slice extents** | Static shapes; pass `active_z` as a mask |
| dd_transpose.jl:98, 336 | backproject kernels | `AK.foreachindex` ×2 | Array-level |
| dd_transpose.jl:99-104 | tile4 | `%`/`÷` index decomposition + 4-lane z expansion | Reshape |
| dd_transpose.jl:105 | tile4 | `use2/use3/use4 = izN <= z_last` lane predication | Mask |
| dd_transpose.jl:110-116 | tile4 | circular-support test then **early `return` after writing zeros** | Compute then `ifelse`-mask |
| dd_transpose.jl:120, 352 | both | `while angle <= na` — full view loop **inside** the voxel kernel | This is a reduction over views; express as a batched contraction |
| dd_transpose.jl:126, 363 | both | `vertical = abs(sy) >= abs(sx)` per view | As dd.jl:142 |
| dd_transpose.jl:127-134, 364-371 | both | 8 ternaries selecting axis roles | `ifelse` |
| dd_transpose.jl:136, 373 | both | `mf = s_long/(s_long - lp)` | Guard |
| dd_transpose.jl:146-158, 388-400 | both | corner-extrema `while bx/by` loops with `atan` and `min`/`max` accumulate | Statically unrollable (4 corners) |
| dd_transpose.jl:159-160, 467-468 | both | `unsafe_trunc(Int32, ceil/floor)` col candidate bounds | Dynamic loop bounds — main blocker |
| dd_transpose.jl:164, 407-408 | both | `qx = sx - clamp(sx, x0, x1)` closest-point-on-box | Fine |
| dd_transpose.jl:165, 422 | both | `max(rho_min, T(1.0e-12))` divide guard | Fine |
| dd_transpose.jl:167-178, 411-421 | both | second corner loop pair computing `rho_max_sq` | Unroll |
| dd_transpose.jl:190-197 | tile4 | 4 × `_dd_arc_row_bounds` (one per z lane) | Vectorize the lane |
| dd_transpose.jl:200-261 | tile4 | `while col <= c1` with `if valid_x`, `if ox > 0`, then 4 × (`if useN` + `while row <= rN`) | Fixed footprint + overlap weights (already zero outside) |
| dd_transpose.jl:215, 227, 240, 253 | tile4 | `oz > zero(T) && (accN += sino[col,row,angle] * ox * oz * norm)` | `ifelse` |
| dd_transpose.jl:264-267 | tile4 | conditional stores `use2 && (vol[ix,iy,iz2] = acc2)` ×3 | Predicated store |
| dd_transpose.jl:298-300 | `dd_backproject!` | `throw(ArgumentError)` on a bad `active_z` | Host-side validation, keep out of trace |
| dd_transpose.jl:305 | `dd_backproject!` | `_dd_check_isotropy` `@warn` | Host |
| dd_transpose.jl:312-315 | `dd_backproject!` | `active_z === nothing ? ... : ...`; `view(volume, :, :, z_first:z_last)` | Dynamic shape |
| dd_transpose.jl:323 | `dd_backproject!` | host branch `arc_det && active_z !== nothing` → different kernel | Two variants |
| dd_transpose.jl:343-346 | generic kernel | circular support → `active_volume[idx] = 0; return` | Mask |
| dd_transpose.jl:383 | generic kernel | `if arc_det` selecting two whole extrema-computation blocks | Host `Bool` |
| dd_transpose.jl:424-431 | generic kernel | `while bz` axial corner loop with 2 divisions by `rho_min`/`rho_max` | Unroll |
| dd_transpose.jl:436-457 | generic kernel | triple `while bx/by/bz` (8 corners) calling `_dd_project_point` | Unroll |
| dd_transpose.jl:467-470 | generic kernel | 4 × `unsafe_trunc(Int32, ceil/floor)` col/row bounds | Dynamic loop bounds |
| dd_transpose.jl:473-503 | generic kernel | `while col` / `while row` nested with `if valid_x`, `ox > 0`, `if valid_z`, `oz > 0` | Fixed footprint |
| dd_transpose.jl:495 | generic kernel | `acc += sino[col, row, angle] * ox * oz * norm` scalar gather | Gather |
| backprojection.jl:90-92, 258-260, 427-429 | all three voxel kernels | `if abs(sv_dot_sd) < T(1e-10); continue` | `ifelse` + mask |
| backprojection.jl:94, 262, 431 | all three | `t = sd_len_sq / sv_dot_sd` | Guarded above |
| backprojection.jl:109-121, 272-283, 446-458 | all three | arc/flat branch; `atan(a_u, a_c)`; `sqrt` | Host `Bool`; fine |
| backprojection.jl:126-127 | `backproject_voxel` | `dist_sq_g = arc_det ? (sv_x²+sv_y²) : (…+sv_z²)`; `weight_g = SAD²/dist_sq` | Host `Bool` |
| backprojection.jl:133-159 | `backproject_voxel` | in-bounds test gating **both** the bilinear read and the `w_acc` accumulator | Mask both; keeps the renormalizer differentiable |
| backprojection.jl:136-139, 337-340, 463-466 | all three | `unsafe_trunc(Int32, col_f)` — **truncation, not floor** (only safe because of the ≥0.5 guard) | Latent bug if the guard is ever relaxed |
| backprojection.jl:146-149, 345-348, 473-476 | all three | 4 × `clamp` on interpolation indices | Fine |
| backprojection.jl:152-155, 350-353, 479-482 | all three | 4-point bilinear scalar gather | `gather` + weight tensor |
| backprojection.jl:162 | `backproject_voxel` | `w_acc > zero(T) ? acc*pi_over_angles : zero(T)` | `acc * π/n / max(w_acc, eps)` style |
| backprojection.jl:177-187 | `_wq_aperture` | 3-way piecewise `cos²` taper | C¹; differentiable, keep as an `ifelse` chain |
| backprojection.jl:229 | `backproject_voxel_helical` | `for angle in Int32(1):n_angles` full view loop in-kernel | Batched contraction |
| backprojection.jl:285-287 | same | `if !(col_f in bounds); continue` | Mask |
| backprojection.jl:289-293 | same | `Wq = _wq_aperture(q̂, q_plateau)`; `if Wq <= 0; continue` | Mask |
| backprojection.jl:303-305 | same | `ℓ = (…)/max(dlen2d, 1e-6)`; `γ = asin(clamp(ℓ/SAD, -1, 1))` | `asin` gradient blows up at ±1 — the clamp saturates it |
| backprojection.jl:309 | same | `βc = atan(sinβ, cosβ) + Δβc` | Branch cut |
| backprojection.jl:318-320 | same | `abs(denomc) > 1e-6 ? … : T(2)` **sentinel** for the conjugate ray | Sentinel `2` deliberately falls outside the aperture — encode as a mask, not a magic value |
| backprojection.jl:324-333 | same | `for k in (-n_turns_k):n_turns_k` with `if zk >= lo && zk <= hi` existence test ×2 | `n_turns_k` from `ceil(1/pitch)`; hoist to a host constant, mask the existence test |
| backprojection.jl:334 | same | `norm = max(norm, Wq)` guard | Fine |
| backprojection.jl:358 | same | `acc += (Wq/norm) * w_fdk * val` | Guarded by line 334 |
| backprojection.jl:461 | `backproject_voxel_matched` | in-bounds `if` gating the whole contribution | Mask |
| backprojection.jl:567-594 | `backproject!` | 4 × `ws_X !== nothing ? … : (similar + copyto!)` | Resolve before tracing |
| backprojection.jl:600 | `backproject!` | `if weighted && is_helical(geom)` — 3-way host dispatch | Host |
| backprojection.jl:604 | `backproject!` | `Δθ = length(geom.angles) > 1 ? angles[2]-angles[1] : 2π` | Host |
| backprojection.jl:609-610 | `backproject!` | `pitch_eff = geom.pitch > 0 ? geom.pitch : 1.0`; `n_turns_k = Int32(ceil(Int, 1/pitch_eff)+1)` | Host constant |
| backprojection.jl:611-612 | `backproject!` | `minimum`/`maximum(@view geom.source_positions[3,:])` | Host reductions; precompute |
| backprojection.jl:614, 640, 667 | `backproject!` | `AK.foreachindex` ×3 | Array-level |
| backprojection.jl:721-722 | `backproject` | `similar` + `fill!` allocation | Fine |

## FBP / FDK filtering (`reconstruction/core/filtering.jl`, `fbp/fdk.jl`, `fbp/wfbp_helical.jl`)

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| filtering.jl:120-155 | `create_spatial_kernel` | `if k == 0 / elseif k % 2 == 0 / else` **parity branch** building the ramp taps; `-1/(π²k²Δ)` | Host-only kernel construction; precompute once |
| filtering.jl:158-210 | `apply_spatial_window!` (Ramp/SL/Cos/Hamming/Hann) | 5 methods, each a host loop with `if k != 0`; `sinc`, `cos` | Host |
| filtering.jl:228-243 | `_catsim_apodization_window` | **linear scan for the interval** (`findfirst`-shaped) with `\|\| i == n-1` fallback; `clamp(f, 0, 1)` | Precompute the window on a fixed grid |
| filtering.jl:254-290 | `_apply_catsim_freq_window!` | `fft`/`ifft` on `Complex{T}`; `mod(i-center, n)+1` fftshift ×2; `k <= n÷2 ? k : n-k` | Host-only; fine |
| filtering.jl:292-315 | Standard/Soft/Bone/Custom windows | 4 methods delegating to the FFT window | Host |
| filtering.jl:329-339 | `filter_from_symbol` | 8-way `Symbol` ternary chain + `error` | Host |
| filtering.jl:371-383 | `equiangular_kernel_scale!` | `if γ != 0 && abs(γ) < π*0.999` then `(γ/sin(γ))²` | Host; guard against `sin γ → 0` |
| filtering.jl:404-405 | `cosine_weight!` | `arc_det = is_arc(geom)`; `dγ = pixel_size/SAD` | Host |
| filtering.jl:408 | `cosine_weight!` | `AK.foreachindex` | Broadcast over a precomputed weight tensor |
| filtering.jl:409-413 | `cosine_weight!` | `%`/`÷` index decomposition | Reshape |
| filtering.jl:418-426 | `cosine_weight!` | `if arc_det` branch; `cos(γ)*SDD/sqrt(SDD²+v²)` vs `SDD/sqrt(SDD²+u²+v²)` | Host `Bool` |
| filtering.jl:478 | `filter_sinogram!` | `apply_cosine && cosine_weight!(...)` | Host |
| filtering.jl:481 | `filter_sinogram!` | `pixel_size = ray_spacing === nothing ? geom.pixel_size : ray_spacing` | Host `Union{Nothing,...}` |
| filtering.jl:488-489 | `filter_sinogram!` | `max(Int(ceil(2·n_cols·cutoff)), 64)`; `min(raw + (1 - raw%2), 2n-1)` — **kernel length from a Float64 cutoff, with a parity fix-up** | Host; but it makes the conv shape data-derived. Fix `kernel_size` at workspace creation |
| filtering.jl:492-504 | `filter_sinogram!` | `ws_filter_kernel !== nothing ? … : (create + arc-scale + copyto!)` | Resolve before tracing |
| filtering.jl:498 | `filter_sinogram!` | `if is_arc(geom) && ray_spacing === nothing` | Host |
| filtering.jl:514 | `filter_sinogram!` | `AK.foreachindex` | Array-level |
| filtering.jl:525-534 | `filter_sinogram!` | `for k in Int32(1):kernel_size` with `if src_col >= 1 && src_col <= n_cols` zero-pad | Replace with an FFT conv or a fixed-shape `conv1d` with explicit padding |
| filtering.jl:540 | `filter_sinogram!` | `copyto!(sinogram, filtered)` — **in-place overwrite of the input** | Return a new array for reverse mode |
| filtering.jl:571 | `filter_sinogram` | `copy(sinogram)` | Fine |
| fdk.jl:332, 385, 430, 465 | `fdk_reconstruct` methods | `cutoff::Float64 = 1.0` kwarg threading a Float64 through | Type-stable but widening |
| fdk.jl:434-441 | `fdk_reconstruct(…, fov)` | rebuilds `CTGeometry` with a new FOV inside the call | Host |
| fdk.jl:463-464 | `apply_fov_mask!` | `sentinel_μ::Real = T(-0.04)` default computed from `T` | Fine |
| fdk.jl:476 | `apply_fov_mask!` | `AK.foreachindex` | Array-level |
| fdk.jl:477-478 | `apply_fov_mask!` | **`CartesianIndices(volume)[idx]` constructed inside the kernel**, then `Tuple(ci)` | Use the mod/div decomposition like every other kernel |
| fdk.jl:484-486 | `apply_fov_mask!` | `if x²+y² > radius_sq; volume[idx] = sentinel` — hard mask, in-place | Make the mask a precomputed `Bool` tensor; `ifelse` |
| wfbp_helical.jl:71, 158 | rebin/backproject | `Δβ = geom.angles[2] - geom.angles[1]` | Host |
| wfbp_helical.jl:83 | `_wfbp_rebin!` | `fill!(reb, zero(T))` then conditional partial fill | Untouched entries stay 0 — encode as a mask |
| wfbp_helical.jl:89 | `_wfbp_rebin!` | `AK.foreachindex` | Array-level |
| wfbp_helical.jl:98-99 | `_wfbp_rebin!` | `if s > T(-0.999) && s < T(0.999)`; `γ = asin(s)` | Mask; `asin` derivative singular at ±1 |
| wfbp_helical.jl:103-107 | `_wfbp_rebin!` | `if arc_det` branch; `tan(γ)` in the flat path | Host `Bool`; `tan` singular at ±π/2 |
| wfbp_helical.jl:109 | `_wfbp_rebin!` | in-bounds `if` gating the whole write | Mask |
| wfbp_helical.jl:110-113 | `_wfbp_rebin!` | `unsafe_trunc(Int32, jf/col_f)`; `min(j_lo+1, nv)`, `min(c_lo+1, nc)` | Precompute the rebin as a fixed sparse gather matrix (geometry-only, no data dependence) |
| wfbp_helical.jl:116-118 | `_wfbp_rebin!` | 4-point bilinear gather across two view indices | Gather |
| wfbp_helical.jl:159 | `_wfbp_backproject!` | `n_half = Int32(max(1, round(Int, π/Δβ)))` — **loop count from geometry** | Host constant |
| wfbp_helical.jl:161 | `_wfbp_backproject!` | `z_start = T(geom.source_positions[3, 1])` scalar read | Host |
| wfbp_helical.jl:171 | `_wfbp_backproject!` | `AK.foreachindex` | Array-level |
| wfbp_helical.jl:184, 189 | `_wfbp_backproject!` | `while fam <= n_half` / `while j <= n_views` **strided** family loop (`j += n_half`) | Reshape views to `(n_half, n_per_family)` and reduce along an axis |
| wfbp_helical.jl:191-197 | same | `sin`/`cos` per view; `if -0.999 < s < 0.999`; `asin(s)` | Mask |
| wfbp_helical.jl:203 | same | `if denom > T(1e-3)` | Mask |
| wfbp_helical.jl:205-207 | same | `v = (z-z_s)*(SDD/cosγ)/denom/prm`; `_wq_aperture(q̂, q_plat)` | Guarded divisions |
| wfbp_helical.jl:208 | same | `if Wq > zero(T)` | Mask |
| wfbp_helical.jl:211-212 | same | in-bounds test gating the sample AND `sumW` | Mask both |
| wfbp_helical.jl:213-218 | same | `floor(t_f)`, `floor(row_f)`, `unsafe_trunc` ×2, `clamp` ×4 | Gather |
| wfbp_helical.jl:219-220 | same | 4-point bilinear | Gather |
| wfbp_helical.jl:230-232 | same | `if sumW > T(1e-8); acc += sumWP/sumW` | `sumWP / max(sumW, eps)` |
| wfbp_helical.jl:260-265 | `wfbp_helical_reconstruct` | allocating rebin + filter + backproject chain | Fine |

## Spectrum / source (`source/spectrum.jl`, `bowtie_filter.jl`, `heel_effect.jl`, `focal_spot.jl`, `protocol.jl`)

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| spectrum.jl:167-169 | `downsample_spectrum` | `if n_bins >= n_original; return copy(...)` | Host |
| spectrum.jl:178-180 | `downsample_spectrum` | `round(Int, (i-1)·bin_size)+1`, `round(Int, i·bin_size)`, `min(end_idx, n)` — **bin edges by rounding** | Host; precompute a fixed binning matrix |
| spectrum.jl:183-195 | `downsample_spectrum` | `weights[start:end]` dynamic slices; `if total_weight > 0` else `mean(bin_energies)` | Host |
| spectrum.jl:238-245 | `get_filter_mu` | `Dict` lookup by `String`; `error` on miss with a `sort(collect(keys(...)))` message | Host; precompute μ tables |
| spectrum.jl:263-291 | `load_spectrum_unfiltered` | `readlines`, `parse(Float64, ...)`, then `findfirst(mask)` / `findlast(mask)` on the nonzero-flux mask — **array length from data** | Host; do once, freeze the length |
| spectrum.jl:205-215 | `GD2O2S` const | `XA.Material` with Unitful quantities | Host |
| bowtie_filter.jl:350-398 | `compute_bowtie_attenuation_spectral` | triple host loop building `[n_col, n_row, n_E]` in **Float64**; `atan(u_offset/SDD)`; `cos(cone_angle)`; `interpolate_thickness`; `exp(-μt_total)` | Host precompute; the result *is* the tensor the kernels want |
| bowtie_filter.jl:379 | same | `if is_arc(geom)` per column inside the loop | Hoist |
| bowtie_filter.jl:387 | same | `thickness_vec ./ cos_alpha` — division by a cosine that → 0 at extreme cone angles | Guard |
| heel_effect.jl:155-157 | `apply_heel_effect!` | `if !heel.enabled \|\| effective_thickness_mm <= 0; return` | Host |
| heel_effect.jl:163 | same | `get_target_attenuation(heel.target_material)` `Dict{Symbol,Float64}` + default `85.0` | Host |
| heel_effect.jl:175 | same | `fan_angle_max = atan(fov[1]/2/SAD)` | Host |
| heel_effect.jl:179-182 | same | `sin_ref = max(sin(θ_anode), T(0.01))`; `I_ref = exp(clamp(exp_ref, T(-700), T(700)))` | **Float64 exponent range at type `T`** |
| heel_effect.jl:186 | same | `θ_min = θ_anode / T(3)` | Fine |
| heel_effect.jl:188 | same | `AK.foreachindex` | Broadcast |
| heel_effect.jl:189-193 | same | `%`/`÷` index decomposition | Reshape |
| heel_effect.jl:205 | same | `θ_effective = max(θ_effective, θ_min)` hard floor | Subgradient |
| heel_effect.jl:209 | same | `sin_effective = max(sin(θ_eff), T(0.01))` hard floor | Subgradient |
| heel_effect.jl:212 | same | **`exp(clamp(exp_term, T(-700), T(700)))`** | With `T = Float32` the clamp never binds and `exp` overflows to `Inf` at ~88. Retune to ±88 |
| heel_effect.jl:215 | same | `intensity[idx] *= attenuation / I_ref` in-place | Out-of-place broadcast |
| heel_effect.jl:245-253 | `get_target_attenuation` | `Dict{Symbol,Float64}` + `get(..., 85.0)` | Host |
| focal_spot.jl:319-320 | `create_focal_spot_kernel_spatial` | `sigma = fwhm/(2√(2 ln 2))` | Host |
| focal_spot.jl:323-325 | same | `min(15÷2, max(1, ceil(Int, 3σ)))` ×2, then `max(extent_x, extent_y)` — **kernel shape from data** | Fix the kernel size, zero-pad |
| focal_spot.jl:332-366 | same | 3-way `if fs.shape == :gaussian / :uniform / :bimodal`; inner `if σ > 0` sub-branches | Host |
| focal_spot.jl:347-348 | same | `min(extent, ceil(Int, blur_fwhm[i]/2))` for the uniform shape | Fixed |
| focal_spot.jl:369-374 | same | `if total > 0; kernel ./= total else kernel[center,center] = 1` | Host |
| focal_spot.jl:405-407 | `apply_focal_spot_blur!` | `if fs.width <= 0 && fs.length <= 0; return` | Host |
| focal_spot.jl:410-412 | same | `if object_distance === nothing; = geom.SAD` | Host `Union{Nothing,...}` |
| focal_spot.jl:421-423 | same | `if blur_fwhm[1] < 0.1 && blur_fwhm[2] < 0.1; return` | Host |
| focal_spot.jl:426-434 | same | `ws_kernel !== nothing ? … : (create + copyto!)` | Resolve before tracing |
| focal_spot.jl:443 | same | `AK.foreachindex` | Array-level conv |
| focal_spot.jl:444-448 | same | `%`/`÷` index decomposition | Reshape |
| focal_spot.jl:452-463 | same | fixed 2D conv with `clamp(col+di, 1, n_cols)` / `clamp(row+dj, 1, n_rows)` edge-replicate | Explicit replicate-pad conv |
| focal_spot.jl:469 | same | `copyto!(sinogram, output)` in-place overwrite | Return a new array |
| protocol.jl:229-232 | `print_protocol` | `round(..., digits=…)` display only | Host |

## Detector response, PCCT DRM, pile-up

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| detector_efficiency.jl:470-487 | `get_scintillator_mu` | `clamp(E, e[1], e[end])`; `log.(energies)`, `log.(mus)` allocated per call; `for i … if E in [e_i, e_{i+1}]; break` linear bracket search | `searchsortedfirst` on host, or a differentiable interpolation |
| detector_efficiency.jl:509, 544, 573 | `get_*_mc_efficiency` ×3 | **`if E == floor(E) && 1.0 <= E <= 140.0`** — exact float equality picking a table index | Always interpolate; the equality branch is a discontinuity in the derivative |
| detector_efficiency.jl:513-522 | `get_gemstone_mc_efficiency` | linear bracket search with `break`, then linear interp | Vectorize |
| detector_efficiency.jl:548, 577 | `get_ufc*_mc_efficiency` ×2 | `idx = clamp(floor(Int, E), 1, 139)` LUT index | Gradient wrt `E` is zero through the index but nonzero through `t`; keep linear interp |
| detector_efficiency.jl:597-607 | `compute_eid_efficiency_vector` | `model.mode == MC_LUT && model.material in ("Gemstone", …)` **string membership dispatch** ×3, then a Float64 comprehension | Host; freeze η as a tensor |
| detector_efficiency.jl:606-607 | same | Beer-Lambert fallback `1 - exp(-μ·d)` per energy | Broadcast |
| mc_response.jl:200-221 | `compute_mc_drm` | overlap fraction with `if isinf(mc_width) && isinf(T_high)` / `elseif isinf(mc_width)` / `else` | Host |
| mc_response.jl:230-243 | same | `if E <= mc_E[1] / elseif E >= mc_E[end] / else` interpolation branches | Host |
| mc_response.jl:236 | same | `clamp(Int(floor(E - mc_E[1])) + 1, 1, n_mc_E - 1)` LUT bracket | Host |
| mc_response.jl:247-255 | same | `if row_sum > 1.0; D[i,:] ./= row_sum`; `max(D[i,b], 0.0)` | Host |
| mc_response.jl:282-289 | `compute_mc_count_moments` | 4 validation `throw`s | Host |
| mc_response.jl:300-311 | same | `if t <= first / elseif t >= last / else searchsortedfirst` threshold interpolation | Host |
| mc_response.jl:306, 344 | same | **`searchsortedfirst`** ×2 | Host |
| mc_response.jl:321 | same | `kVp = Float64(maximum(energies))` | Host reduction |
| mc_response.jl:322-324 | same | `R === nothing ? compute_mc_drm(...) : Float64.(R)`; `η === nothing ? … : η` | Host `Union{Nothing,...}` |
| mc_response.jl:337 | same | `w == 0 && continue` | Host |
| mc_response.jl:339-347 | same | 3-way `if E <= first / elseif E >= last / else` with `searchsortedfirst` | Host |
| mc_response.jl:355 | same | `clamp(round(Int, (E-1)/(kVp-1)·(n_R-1)) + 1, 1, n_R)` — **DRM row index by rounding** (one of 5 copies across `src/`) | Replace with linear interpolation between adjacent DRM rows if ∂/∂E is ever needed |
| mc_response.jl:357 | same | `[m_raw[b] > eps ? d[b]/m_raw[b] : 0.0 for b in 1:n_bins]` | `d ./ max(m_raw, eps)` |
| mc_response.jl:358-359 | same | `Diagonal(scale)`; `C = Dscale * C_raw * Dscale` | Fine |
| mc_response.jl:369-377 | same | symmetrization `(X + Xᵀ)/2`; `mean > eps ? … : 1.0` fano; `denom > eps ? … : (i==j)`; `clamp.(correlation, -1, 1)` | Fine |
| mc_response.jl:389-411 | `mc_drm_summary` | `println` diagnostics with `round`, `minimum`, `maximum` | Host |
| photon_counting.jl:252-253 | `EnergyResolvedSinogram` ctor | `@assert` ×2 | Host |
| photon_counting.jl:280-305 | `spatial_bin!` | `AK.foreachindex` + fixed `bf×bf` sum loop | Reshape + sum along two axes — trivially traceable |
| photon_counting.jl:431 | `pcct_forward_project` | `kVp = maximum(energies)` | Host reduction |
| photon_counting.jl:441-446 | same | 3 × `ws_X !== nothing ? … : compute` | Resolve before tracing |
| photon_counting.jl:453-460 | same | `ws_bins !== nothing ? … : [similar(...) for _ in 1:n_bins]`; `fill!` each | Host |
| photon_counting.jl:466 | same | host branch on 3 workspace buffers being non-`nothing` | Pick the path before tracing |
| photon_counting.jl:467, 504, 515-516, 616 | same | `@info … maxlog=1` ×4 | Host |
| photon_counting.jl:471 | same | `n_tiles = n_energies_padded ÷ TILE_K` | Host |
| photon_counting.jl:474-481 | same | 6 × `use_native ? native : binned` selections | Host |
| photon_counting.jl:488-492 | same | `_pilot = use_native ? … : …` pilot array for iteration count | Reactant needs an explicit shape |
| photon_counting.jl:503-513, 548 | same | `@goto tiles_done` / `@label tiles_done` | Host control flow; restructure |
| photon_counting.jl:520-546 | same | tile loop with `copyto!(μ_sub, @view tbl[:, ts:te])`, `copyto!(W_sub, @view W[ts:te, :])`, optional `bt_sub` — **dynamic slices** | With `:dd_fast` the whole tile loop is skipped; prefer that path |
| photon_counting.jl:562-564, 575-579 | same | flat→3D unpack `AK.foreachindex` ×2 with an `off` offset | `reshape` |
| photon_counting.jl:568, 709 | same | `spatial_bin!(bins[b], proj_bins[b], bf)` per bin | Batch |
| photon_counting.jl:586-591, 717-722 | same | `ws_I0_bins_norm !== nothing ? … : [_compute_bin_I0(...) for b in 1:n_bins]` | Host |
| photon_counting.jl:592-598, 723-729 | same | `AK.foreachindex`; `-log(max(ba[idx], eps)/I0_bin_T)` | Subgradient at the floor |
| photon_counting.jl:601-608, 731-739 | same | `ws_thresholds_T !== nothing ? (loop-fill) : T.(...)` | Host |
| photon_counting.jl:619-628 | same | `use_native ? (alloc + fill!) : nothing`; `accum_bins = use_native ? proj_bins : bins` | Host |
| photon_counting.jl:631-644 | same | 3-way and 2-way workspace-buffer selection chains | Host |
| photon_counting.jl:657 | sequential fallback | `if E_float < thresholds[1]; continue` | Mask, or prune the energy grid on host |
| photon_counting.jl:663 | same | `if w < 1e-12; continue` | Mask |
| photon_counting.jl:691, 766 | same, `_compute_bin_I0` | `clamp(round(Int, (E-1)/(kVp-1)·(n_R-1))+1, 1, n_R)` DRM row | As mc_response.jl:355 |
| photon_counting.jl:695-697 | same | `if R_val < T(1e-10); continue` | Mask |
| photon_counting.jl:699-701 | same | `AK.foreachindex`; `ba[idx] += wt*exp(-sino_buf[idx])` accumulate over the host energy loop | Stack energies and reduce |
| photon_counting.jl:762-764 | `_compute_bin_I0` | `if w < 1e-12; continue` | Host |
| photon_counting.jl:770 | `_compute_bin_I0` | `max(I0_bin, 1.0)` | Fine |
| mc_pileup.jl:117-121 | `simulate_pulse_train` | `if w_sum > 0; w ./= w_sum else w .= 1/length(w)` | Host |
| mc_pileup.jl:125 | same | `cdf = cumsum(w)` | Host |
| mc_pileup.jl:129 | same | `n_photons = Poisson_approx(expected_count)` — **integer draw sizes the arrays** | Dynamic shape; host-only MC |
| mc_pileup.jl:131-136 | same | `if n_photons == 0; return` early | Host |
| mc_pileup.jl:140-143 | same | `Vector{Float64}(undef, n_photons)`; `rand(rng)*T_obs` | Host |
| mc_pileup.jl:144 | same | **`sort!(arrival_times)`** | Non-differentiable |
| mc_pileup.jl:148-153 | same | `rand(rng)` + **`searchsortedfirst(cdf, u)`** + `clamp` inverse-CDF energy sampling | Host |
| mc_pileup.jl:163-197 | same | dead-time `while it <= n_photons` with two inner `while jt <= n_photons && arrival_times[jt] < dead_end` loops and `push!` to growing vectors — **dynamic shapes** | Host-only |
| mc_pileup.jl:169, 185 | same | `if model == :seminonparalyzable` else | Host |
| mc_pileup.jl:177 | same | `if rand(rng) < f_retrigger` **Bernoulli** | Host-only |
| mc_pileup.jl:203-215 | same | histogram accumulation into `true_bins` / `rec_bins` via `_find_threshold_bin` | Discrete |
| mc_pileup.jl:231-242 | `_find_threshold_bin` | `if E < thresholds[1]; return 0`; reverse linear scan `for b in n:-1:1` returning an `Int` | Discrete |
| mc_pileup.jl:250-269 | `Poisson_approx` | `if λ <= 0 / elseif λ < 30` **`while true` Knuth inversion with `rand()`** / else `max(0, round(Int, λ + √λ·randn()))` — **global unseeded RNG** | Non-differentiable and non-traceable. Precompute `S` on host once; treat it as a fixed tensor. (Also a reproducibility bug independent of AD.) |
| mc_pileup.jl:336 | `compute_mc_pileup_matrix` | `MersenneTwister(seed)` | Host precompute, run once |
| mc_pileup.jl:348-361 | same | `if expected > 1e6` shrinks `observation_time_s` + `@warn maxlog=1` | Host |
| mc_pileup.jl:369-380 | same | `for trial in 1:n_trials` host MC loop; per-bin accumulation | Host |

## Noise — the hardest wall

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| photon_counting.jl:831-834 | `apply_pcct_noise!` | 2 × `throw(DimensionMismatch)` | Host |
| photon_counting.jl:836-841 | same | `Random.seed!(ws_rng, seed)` / `MersenneTwister(seed)` / `Random.default_rng()` | Reactant RNG, or an externally supplied noise tensor |
| photon_counting.jl:843 | same | `cpu_buf = ws_noise_staging !== nothing ? … : Array(sino.bins[1])` — **GPU→CPU** | Forces a host round-trip; blocks tracing entirely |
| photon_counting.jl:850, 861, 869 | same | `copyto!` GPU→CPU, CPU→`raw_out`, CPU→GPU per bin | Same |
| photon_counting.jl:853-859 | same | `@inbounds for idx in eachindex(cpu_buf)`; `λ = I0_bin·exp(-p)`; **`N = Float64(_poisson_sample(rng, λ))`** | Reparameterize: (a) Gaussian approx `N = λ + √λ·ε` with ε a fixed input tensor, or (b) treat the drawn counts as a *fixed input* and differentiate only the deterministic map λ(x) → −log(N/I0) |
| photon_counting.jl:856-858 | same | `if nr_scale != 1.0; N = λ + nr_scale·(N - λ)` variance-scaling blend | Fine once N is an input |
| photon_counting.jl:864-866 | same | `max(cpu_buf[idx], one(T))` **count floor** before `-log` | Hard gate; zero gradient below 1 count |
| photon_counting.jl:875-884 | `_log_factorial` | `_LOG_FACTORIAL_TABLE` const; `k <= 20` table lookup vs a Stirling–de Moivre series | Host-only |
| photon_counting.jl:894-931 | `_poisson_sample` | `if λ < 1e-10 / elseif λ < 10` **`while true` Knuth inversion** / else **`while true` PTRS rejection**; `rand(rng)` ×3; `floor(Int, …)`; `continue`; log-space rejection test | Fundamentally non-traceable and non-differentiable. Must be replaced or lifted out |
| driver.jl:522 | `simulate!(::EICTWorkspace)` | `if sim_opts.use_noise` host branch | Host |
| driver.jl:527-531 | same | `Random.seed!(ws.rng, seed)`; `randn!(ws.rng, noise_rand_cpu)`; `copyto!(noise_rand_gpu, noise_rand_cpu)` | Pre-generate the noise tensor outside the traced region |
| driver.jl:535-537 | same | `if σ_e_photon > T(0)` selecting two whole kernel variants; second `randn!` + `copyto!` | Two traced variants |
| driver.jl:542, 557, 572 | same | `AK.foreachindex` ×3 (noise+electronic, noise-only, scatter-only) | Array-level |
| driver.jl:543-551, 558-565 | same | `λ_total = I0·exp(-p)`; `λ_noisy = λ_total + sqrt(max(λ_total, one(T)))·ε`; `+ σ_e·ε₂`; `max(λ_noisy - I0·sf·sw, one(T))`; `-log(λ_primary/I0)` | **This half is differentiable once ε is an input** — the EICT path needs no algorithm change |
| driver.jl:544, 559 | same | `sqrt(max(λ_total, one(T)))` | ∂√/∂λ blows up near 0; the `max(…,1)` floor saves it but zeroes the gradient |
| driver.jl:546-550, 560-564 | same | `if do_sc` inside the kernel body | Hoist to two variants |
| driver.jl:520 | same | `sf_kernel = has_scatter ? scatter_field_gpu : ws.physics_output` — **dummy buffer to dodge `Nothing` in the Metal closure** | Two variants |
| driver.jl:569-578 | same | `elseif has_scatter` third kernel for noise-free scatter subtraction | Host |
| driver.jl:734-739 | `add_system_noise_floor!` | `sigma_hu <= 0 && return`; `Random.default_rng()` / `MersenneTwister(seed+7919)`; `vol .+= σ .* randn(rng, T, size(vol))` in place | Fixed noise tensor input; out-of-place |
| driver.jl:243-245 | `simulate!(::PCCTWorkspace)` | `raw_from_noise = capture && use_noise && !pileup ? [similar(bin) …] : nothing` | Host |
| driver.jl:246-255 | same | `if sim_opts.use_noise; apply_pcct_noise!(...)` | Host |
| driver.jl:284-287 | same | `if ws.use_pcct_pileup && ws.pileup_S !== nothing`; `n_bins == 4 \|\| error(...)` | Host |
| driver.jl:290-298 | same | 10 scalar reads `S[i,j]` unrolled into `let` bindings | Express `S` as a matrix |
| driver.jl:299 | same | `AK.foreachindex(b1)` iterating one bin to drive all four | Stack bins into a 4-axis array |
| driver.jl:300-316 | same | 4 × `I0·exp(-b)`; hardcoded lower-triangular `S × counts`; 4 × `-log(max(r, eps)/I0)` | Express as a `(4,4) × (4,N)` matmul — traceable and differentiable in `S` |
| driver.jl:326-335 | same | 3-way `raw_from_noise !== nothing / capture_raw_counts / else nothing` | Host |
| driver.jl:342-344 | same | `if pileup && correction; apply_pcct_pileup_correction!` | Host |
| driver.jl:397-402 | same | `capture_raw_counts ? merge(result, (; raw_counts)) : result` — **return type depends on a flag** | Host |

## Scatter (`detector/scatter.jl`)

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| scatter.jl:76, 132 | kernel builders | `sigma = kernel_fwhm/(2√(2 ln 2))` | Host |
| scatter.jl:79, 133 | kernel builders | `min(63÷2, ceil(Int, 3σ))` — **shape from `kernel_fwhm`** | Fix the kernel size at workspace creation |
| scatter.jl:85-102 | `create_scatter_kernel_spatial` | `if kernel_type == :gaussian / elseif :exponential / else error(...)`; two double loops | Host |
| scatter.jl:105-111 | same | `if total > 0; kernel ./= total else kernel[center,center] = 1.0` | Host |
| scatter.jl:130 | `create_scatter_kernel_1d` | `model.kernel_type == :gaussian \|\| return nothing` — **`Nothing` in a union return** | Host dispatch |
| scatter.jl:143 | same | `kernel_1d ./= sum(kernel_1d)` | Host |
| scatter.jl:162, 194 | `_convolve_separable_{h,v}!` | `AK.foreachindex` ×2 | `conv1d` with replicate padding |
| scatter.jl:163-167, 195-199 | same | `%`/`÷` index decomposition ×2 | Reshape |
| scatter.jl:170-173, 202-205 | same | `for di in -hk:hk` with `clamp(col+di, 1, nc)` replicate; `@inbounds` gather | Replicate-pad conv |
| scatter.jl:324 | `inject_scatter!` | `AK.foreachindex` | Broadcast |
| scatter.jl:326-330 | same | `min(proj, T(20))` **saturating clamp**; `exp(-clamped)`; `max(scatter_intensity, 0)`; `-log(max(total, 1e-10))` | Gradient dies above 20 |
| scatter.jl:360-366 | `inject_scatter_bins!` | `eps = T(1)` **DAS 1-count floor**; `sgn = subtract ? -1 : +1` | Host flag |
| scatter.jl:370-375 | same | `AK.foreachindex` per bin; `N_primary = I0b·exp(-p)`; `max(N_scatter, 0)`; `-log(max(N_total, eps)/I0b)` | Stack bins; zero gradient below the floor |
| scatter.jl:403-430 | `compute_scatter_bin_weights` | host DRM-weighted sum + normalize | Host |
| scatter.jl:274-296 | `compute_scatter_energy_weights` | analytical Compton fraction `1/(1 + (20/E)³)` per energy | Differentiable; host |
| scatter.jl:543-620 | geometry/size scaling | host arithmetic on reference constants | Host |
| scatter.jl:712-721 | `estimate_scatter_field!` | `C = T(scatter_coefficient · scale_factor)` | Host |
| scatter.jl:724-733 | same | `if model.kernel_type == :gaussian` selecting the whole separable path; `ws_kernel_1d !== nothing ? … : (create + copyto!)` | Host |
| scatter.jl:734 | same | `ws_scatter_temp !== nothing ? … : similar(sinogram)` | Resolve before tracing |
| scatter.jl:738-742 | same | `AK.foreachindex`; `min(proj, T(20))`; `exp(-clamped)·proj·C` | Saturating clamp |
| scatter.jl:746-747 | same | two separable conv passes | conv |
| scatter.jl:753-757 | same | 2D fallback: `create_scatter_kernel_spatial` + `copyto!` | Host |
| scatter.jl:761 | same | `AK.foreachindex` (2D fallback) | Array-level |
| scatter.jl:762-766 | same | `%`/`÷` index decomposition | Reshape |
| scatter.jl:769-780 | same | double `for dj/di` with `clamp` ×2; `min(max(src_prep, 1e-10), T(20))` **double clamp** | Replicate-pad conv; saturating |

## Other detector effects

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| fill_factor.jl:94-96 | `apply_fill_factor!` | `if ff ≈ 1.0; return` — **approximate float comparison** | Host |
| fill_factor.jl:98 | same | `offset = T(-log(ff))` | Host |
| fill_factor.jl:100-102 | same | `AK.foreachindex`; `sinogram[idx] += offset` in place | Broadcast add, out-of-place |
| detector_lag.jl:107-109 | `apply_lag!` | `if isempty(model.amplitudes); return` | Host |
| detector_lag.jl:115 | same | `n_frames = min(n_history, n_angles)` — **conv length from data** | Host |
| detector_lag.jl:116-122 | same | `ws_coeffs !== nothing ? … : (compute + copyto!)` | Resolve before tracing |
| detector_lag.jl:126-128 | same | `AK.foreachindex`; `intensity[idx] = exp(-sinogram[idx])` | Broadcast |
| detector_lag.jl:134 | same | `AK.foreachindex` | Array-level |
| detector_lag.jl:135-139 | same | `%`/`÷` index decomposition | Reshape |
| detector_lag.jl:142-149 | same | `for k in 0:(n_frames-1)` with `if prev_angle >= 1` else clamp-to-first-view | Causal conv along the view axis with replicate padding |
| detector_lag.jl:151 | same | `-log(max(weighted_sum, T(1e-10)))` | Log floor |
| detector_lag.jl:155 | same | `copyto!(sinogram, output)` in place | Return a new array |
| optical_crosstalk.jl:72-74 | `apply_optical_crosstalk!` | `if row_coeff ≈ 0 && col_coeff ≈ 0; return` | Host |
| optical_crosstalk.jl:79-85 | same | `ws_kernel !== nothing ? … : (create + copyto!)` | Resolve before tracing |
| optical_crosstalk.jl:90 | same | `AK.foreachindex` | Array-level |
| optical_crosstalk.jl:91-95 | same | `%`/`÷` index decomposition | Reshape |
| optical_crosstalk.jl:98-109 | same | fixed 3×3 conv with `clamp` ×2; `exp(-sinogram[...])` per tap | `conv2d`, replicate pad |
| optical_crosstalk.jl:111 | same | `-log(max(acc, T(1e-10)))` | Log floor |
| optical_crosstalk.jl:115 | same | `copyto!(sinogram, output)` in place | Return a new array |
| physics_pipeline.jl:33-86 | `PhysicsConfig` | `Union{Nothing, Int}` seed field; per-effect `Union{Nothing, Model}` fields | Every effect is gated by a `!== nothing` host branch |

## Calibration / air scan

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| calibration.jl:43-47 | `low_signal_correction_gpu!` | `AK.foreachindex`; **`if prep[idx] <= zero(T); prep[idx] = eps`** — hard threshold, in place | `prep = max(prep, eps)` broadcast (same semantics, differentiable) |
| driver.jl:48-56 | `compute_detector_I0` | Float64 mm conversions; `spectrum_flux_sum · mA · time_per_view · pixel_area` | Host |
| driver.jl:72-77 | `_capture_pcct_raw_counts` | `AK.foreachindex`; `out[idx] = I0b·exp(-input[idx])`; `[similar(bin) for bin in bins]` | Broadcast |
| driver.jl:175-183 | `simulate!(::PCCTWorkspace)` | `if config.focal_spot !== nothing` per-bin blur loop | Host |
| driver.jl:199 | same | `if config.scatter !== nothing && sim_opts.use_pcct_scatter` | Host |
| driver.jl:203-214 | same | `AK.foreachindex` ×2; `comb += I0b·exp(-bs)`; `-log(max(comb, eps)/I0t)` | Stack bins; `sum` along the bin axis |
| driver.jl:352-381 | same | scatter-correction repeat of the combine + estimate + subtract chain | Same |
| driver.jl:437-455 | `simulate!(::EICTWorkspace)` | `fill!` + `_forward_project_poly!` with 15 workspace kwargs | Resolve before tracing |
| driver.jl:463-474 | same | `_apply_physics_no_noise!` with 8 workspace kwargs | Host |
| driver.jl:482-497 | same | `has_scatter = config.scatter !== nothing`; `scatter_total_weight = sum(...)/max(sum(...), 1e-30)` | Host |
| driver.jl:511-512 | same | `I0_raw = compute_detector_I0(...)`; `I0_T = T(I0_raw) * ws.η_eff` | Host |
| driver.jl:586-590 | same, step 4 | `AK.foreachindex`; **`exp(-clamp(sino[idx], T(-1), T(15)))`** hard two-sided clamp | Gradient dies outside `[-1, 15]`; widen or use a soft clamp |
| driver.jl:593-604 | same | `fill!(air_scan, one(T))`; `if bowtie_air_reference !== nothing` then `AK.foreachindex` with `ref[col + (row-1)*nc]` linear-index gather | Broadcast against a 2D array with `reshape` |
| driver.jl:606-611 | same | `AK.foreachindex`; `sino / max(air, eps)` | Fine |
| driver.jl:614 | same | `low_signal_correction_gpu!(ws.sinogram)` | See calibration.jl:43 |
| driver.jl:617-621 | same | `AK.foreachindex`; `-log(max(sino[idx], eps))` | Subgradient |
| driver.jl:632-642 | same | `if config.fill_factor !== nothing`; `if !(ff_eff ≈ one(T))`; `AK.foreachindex` `+= ff_log` | Host branches |
| workspace.jl:176-186 | `create_pcct_workspace` | spectrum resolve; `kVp = Float64(maximum(energies))`; `compute_mc_drm`; `collect(range(...))` | Host |
| workspace.jl:200-201 | same | `_I0_anchor = compute_detector_I0(...) / max(sum(weights_vec), 1e-30)` | Host |
| workspace.jl:215, 743 | both workspaces | `MersenneTwister(0)` stored in the workspace | Re-seeded per call |
| workspace.jl:222-226 | same | double loop filling `μ_table[r, e]` from `compute_μ_at_energy` (Float64) | Host precompute |
| workspace.jl:241-281 | same | `if bf > 1` building a whole native-resolution geometry + buffers, else 7 × `nothing` | Two workspace variants |
| workspace.jl:288-292 | same | `n_energies_padded = cld(n_energies, 16)*16`; `fill!` + `copyto!(view(...))` | Host |
| workspace.jl:299-309 | same | `if w < 1e-12; continue`; `clamp(round(Int, (E-1)/(kVp-1)·(n_R-1))+1, 1, n_R)` DRM row | Host |
| workspace.jl:317-329 | same | `if bowtie !== nothing && name != "none"`; `center_col = sino_shape[1] ÷ 2` bowtie centre fold | Host |
| workspace.jl:336-338 | same | `append!(I0_bins_norm_vec, [sum(...) for b in 1:n_bins])` — **air-calibration by construction** | Host |
| workspace.jl:344-349 | same | `_native_outputs_flat = bf > 1 ? allocate(...) : nothing` | Host |
| workspace.jl:461-463 | `EICTWorkspace` | `noise_rand_cpu::Vector{T}`, `enoise_rand_cpu::Vector{T}` staging buffers | CPU RNG staging |
| workspace.jl:829, 909, 1040 | workspace ctors | `max(Int(ceil(2·n_cols·cutoff)), 64)`; `max(nz, ceil(Int, fov[3]·cone_ratio/dz))`; repeat | Host; freeze shapes |
| workspace.jl:1119 | HIR workspace | `max_subset_size = maximum(length(s) for s in subsets)` — buffer size from data | Host |

## Beam-hardening correction

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| bhc_sinogram.jl:184-205 | `generate_water_calibration_curve` | host loop over paths; `-log(max(I_poly, 1e-10))` | Host precompute; differentiable if ever needed |
| bhc_sinogram.jl:218-225 | `fit_polynomial` | **normal equations `(V'V) \ (V'y)`** — least-squares solve | Enzyme handles `\` via the implicit-function rule; Reactant needs an XLA `triangular_solve`/QR |
| bhc_sinogram.jl:285-286 | `apply_bhc!` (BHC wrapper) | `[b.polynomial for b in bhcs_per_col]` comprehension | Host |
| bhc_sinogram.jl:303-309 | `apply_bhc!` | `length(...) == n_col \|\| error`; per-column order-mismatch `error` in a loop | Host validation |
| bhc_sinogram.jl:311-322 | same | `zeros(T, order+1, n_col)` + `ws_coeffs_gpu !== nothing ? … : (similar + copyto!)` | Resolve before tracing |
| bhc_sinogram.jl:329 | same | `AK.foreachindex` | Array-level |
| bhc_sinogram.jl:330 | same | `col = ((idx-1) % n_col) + 1` | Reshape |
| bhc_sinogram.jl:332-338 | same | `for i in 1:order` power accumulation with a **runtime `order`**; `coeffs[i+1, col]` gather | Lift `order` to `Val`; or evaluate as a polynomial matvec |
| bhc_sinogram.jl:392, 433, 714 | two-material entries | `Base.depwarn` ×3 | Host |
| bhc_sinogram.jl:398-408 | `calibrate_bhc_two_material` | `resolve_source_spectrum_full`; `ref_E === nothing ? (weighted mean) : Float64(...)` | Host |
| bhc_sinogram.jl:438-448 | same | per-column loop calling `calibrate_bhc` (each does a `\` solve) | Host |
| bhc_sinogram.jl:540-543 | `compute_polychromatic_μ_water` | `detector_type === :photon_counting && throw(ArgumentError)` | Host |
| bhc_sinogram.jl:551-554 | same | `mid_c = size(w_per_col, 2) ÷ 2 + 1`; Beer-Lambert pre-hardening; fluence-weighted mean | Host |
| bhc_sinogram.jl:591-608 | `calibrate_bhc_water` | `if ndims(w_col)==1; repeat(...)`; `ref_E === nothing ? … : …`; per-column `calibrate_bhc` loop | Host |
| bhc_sinogram.jl:716-719 | `apply_bhc_two_material` | `_validate_projector`; `length(...) == n_col \|\| error` | Host |
| bhc_sinogram.jl:722-725 | same | `similar` + `copyto!`; `apply_bhc!`; **`fdk_reconstruct` inside the correction** | Full FDK enters the tape |
| bhc_sinogram.jl:738 | same | `AK.foreachindex` | Array-level |
| bhc_sinogram.jl:740-750 | same | `hu_val = 1000(μ-μ_w)/μ_w`; **3-way bone-fraction smoothstep** `t²(3-2t)` on `[hu_low, hu_high]` | C¹ — differentiable; keep as an `ifelse` chain |
| bhc_sinogram.jl:753 | same | `_project_mono(projector, bone_μ_gpu, geom)` — **forward projection inside the correction** | Full projector cost enters the tape |
| bhc_sinogram.jl:758-760 | same | `AK.foreachindex`; `p_s = p_s - p_b` | Broadcast |
| bhc_sinogram.jl:766-771 | same | 3 × `similar` + `copyto!` staging of μ/w tables | Host |
| bhc_sinogram.jl:779 | same | `AK.foreachindex` | Array-level |
| bhc_sinogram.jl:780-791 | same | `col = ((idx-1) % n_col)+1`; `max(p/μ_ref, 0)` ReLU ×2; `if Lw > 1e-6 \|\| Lb > 1e-6`; `for e in 1:n_e` energy loop; `I_poly > 0 ? -log(I_poly) : 0` | Mask + energy-axis reduction |
| bhc_sinogram.jl:795 | same | `return Array(sino_out)` — **forced GPU→CPU** | Return the device array |
| bhc_image_domain.jl:84-92 | `apply_bhc_image_domain` | `Base.depwarn` (deprecated stage) | Excluded from the AD path anyway |
| bhc_image_domain.jl:104 | same | `AK.foreachindex` | Array-level |
| bhc_image_domain.jl:106-118 | same | 3-way linear weight (C⁰, **kinks at both thresholds**); `wf·(μ - μ_w_ref)` | Smoothstep like bhc_sinogram.jl:741 |
| bhc_image_domain.jl:126, 129 | same | `_project_mono` + `fdk_reconstruct` round trip inside the correction | Very expensive tape |
| bhc_image_domain.jl:134-136 | same | `AK.foreachindex`; in-place `recon_μ -= sf·error_image` | Broadcast, out-of-place |

## Cupping / capping QA

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| radial_cupping.jl:48-50 | `measure_radial_cupping` | comprehension filter `-300 <= v <= 300`; `length(rough) < 100 && continue`; **`sort!(rough)[(n+1)÷2]`** median | Order statistic; non-differentiable, dynamic shape |
| radial_cupping.jl:52-58 | same | `if abs(v - centre) <= half_window` `push!` to growing `radii`/`vals` | **Dynamic shape from data** — untraceable as written |
| radial_cupping.jl:59 | same | `length(radii) < 100 && continue` | Host |
| radial_cupping.jl:60-64 | same | Vandermonde build + **`A \ vals`** with a data-dependent row count | Fixed-size weighted LSQ with a soft membership weight |
| radial_cupping.jl:65-66 | same | `r_max = maximum(radii)`; `abs(sum(coeffs[p+1]·r_max^(2p) …))` | `maximum` has a subgradient |
| radial_cupping.jl:67-72 | same | `if cup > worst_cup` running argmax over slices | Discrete selection |
| radial_cupping.jl:110-116 | `apply_radial_cupping_correction!` | `Base.depwarn` (deprecated) | Excluded |
| radial_cupping.jl:128-136 | same | `if hu_lo <= v <= hu_hi` `push!` — dynamic shape | Soft weights |
| radial_cupping.jl:137 | same | `length(radii) < 10 && continue` | Host |
| radial_cupping.jl:141-147 | same | Vandermonde + `A \ vals` | Weighted LSQ |
| radial_cupping.jl:153-161 | same | `maximum(radii)`; `if cup_mag > 5 \|\| dc_mag > 5` `@warn maxlog=3` | Host |
| radial_cupping.jl:165-169 | same | per-voxel `sum(coeffs[p+1]·r^(2p) …)` subtract, in place | Broadcast |
| radial_capping_basis.jl:60-73 | `apply_radial_capping_basis!` | closure `correct_basis!`; `push!` into `in_fov`; `isempty && continue`; **`quantile(in_fov, q_lo)` / `quantile(in_fov, q_hi)`** | Order statistics — non-differentiable; use a soft quantile or fixed thresholds |
| radial_capping_basis.jl:77-87 | same | `if lo <= v <= hi` `push!` to `radii`/`vals` — dynamic shape | Soft weights |
| radial_capping_basis.jl:87 | same | `length(radii) < 10 && continue` | Host |
| radial_capping_basis.jl:90-94 | same | Vandermonde + **`A \ vals`** | Weighted LSQ |
| radial_capping_basis.jl:99-103 | same | per-voxel `offset = sum(coeffs[p+1]·r^(2p))`; `slice[i,j] -= Float32(offset - target)` in place | Broadcast |
| radial_capping_basis.jl:114-117 | same | `poly_order_i >= 1 ? mean(coeffs[2,:]) : 0.0` diagnostics | Host |

## Hybrid IR / OS-PWLS reconstruction

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| driver.jl:1293-1294 | `_hir_seed_work!` | `work === init && return work` identity check | Host |
| driver.jl:1300 | same | `AK.foreachindex(work, backend)` | Array-level |
| driver.jl:1301-1306 | same | `%`/`÷` decomposition; `clamp(kw - z0 + 1, 1, nz)` halo gather | `pad(:replicate)` |
| driver.jl:1311-1325 | `_hir_extract_output!` | `output === work && return`; `AK.foreachindex`; slice gather | `view` / `slice` |
| driver.jl:1417-1458 | `reconstruct!(::HIRReconWorkspace)` | host branches: `init_volume === nothing`, `is_helical(model_geom)`; `size(init_volume) == size(ws.volume) \|\| throw` | Pick before tracing |
| driver.jl:1420-1428 | same | helical path: `_wfbp_rebin!` + `filter_sinogram!` + `_wfbp_backproject!` | Host |
| driver.jl:1466 | same | `λ = params.lambda · (ws.projector === :siddon ? one(T) : T(0.1))` | Host |
| driver.jl:1469 | same | `backend = AK.get_backend(model_volume)` | Host |
| driver.jl:1477-1481 | same | `if nepochs == 0`: extract + mask + **early return (plain FBP)** | Host |
| driver.jl:1487 | same | `apply_fov_mask!(model_volume, model_geom; sentinel_μ = zero(T))` | Precomputed mask |
| driver.jl:1491 | same | `if air_reference !== nothing` selecting two whole weight kernels | Two variants |
| driver.jl:1494, 1508 | same | `AK.foreachindex` ×2 | Array-level |
| driver.jl:1495-1503, 1509-1513 | same | `%`/`÷`; `ref[col + (row-1)·nc]` gather; **`clamp(y_val, T(0), T(10))`** then `exp(-y)+ε` | Hard two-sided clamp on the statistical weight; gradient dies outside |
| driver.jl:1522-1526 | same | `AK.foreachindex`; `dw[idx] = wp[idx]·dw[idx]` **in place** | Broadcast, out-of-place |
| driver.jl:1528-1632 | same | `for epoch in 1:nepochs` × `for (s, angle_indices) in enumerate(ws.subsets)` — **unrolls the whole OS-PWLS tape** | For Enzyme: checkpoint per epoch, or use an implicit-diff fixed-point rule |
| driver.jl:1531, 1538 | same | `ws.projector === :siddon ?` chooses *when* to recompute the Huber gradient | Host |
| driver.jl:1548-1549 | same | `ax_view = n_sub == size(buf,3) ? buf : view(buf,:,:,1:n_sub)` — **dynamic shape for ragged final subsets** | Pad subsets to equal size |
| driver.jl:1551-1558 | same | `_project_mono_hir!` with 4 per-subset geometry buffers | Host |
| driver.jl:1568 | same | `AK.foreachindex(ax, backend)` | Array-level |
| driver.jl:1569-1575 | same | `%`/`÷`; **`a = aidx[k]` integer index-array gather**, then `sino[col,row,a]` and `dw[col,row,a]` | `gather` along the view axis; ∂/∂aidx = 0 |
| driver.jl:1576 | same | `ax[idx] = dw[...]·residual` in place | Out-of-place |
| driver.jl:1580-1598 | same | `fill!(ws.correction, 0)`; host branch selecting backprojector kwargs | Host |
| driver.jl:1607 | same | `reg_views_scale = T(length(geom.angles))/T(1000)` | Host |
| driver.jl:1616 | same | `AK.foreachindex(vol, backend)` | Array-level |
| driver.jl:1617-1628 | same | `%`/`÷`; circular-support test → `vol[idx] = 0` else `vol[idx] += data_update - reg_update` — **in-place mutation of the iterate** | `ifelse` + out-of-place update |
| driver.jl:1636-1637 | same | `_hir_extract_output!` + `apply_fov_mask!` | Host |
| ir/utils.jl:44-51 | `_huber` | `if abs_t ≤ δ` quadratic else linear | C¹; fine |
| ir/utils.jl:53-60 | `_huber_deriv` | `δ * sign(t)` — **`sign` is non-differentiable at 0** | Second derivative is a delta; fine for first-order, hazardous for Hessian-vector products |
| ir/utils.jl:77 | `compute_huber_penalty` | `AK.foreachindex(penalty_vals, backend)` | Array-level |
| ir/utils.jl:78-80 | same | `mod1`/`div` index decomposition | Reshape |
| ir/utils.jl:85-96 | same | 3 boundary `if i < nx / j < ny / k < nz` branches | Shifted-array differences |
| ir/utils.jl:101 | same | `AK.mapreduce(identity, +, penalty_vals; init=zero(T))` | `sum` |
| ir/utils.jl:131 | `compute_huber_gradient!` | `AK.foreachindex(grad, backend)` | Array-level |
| ir/utils.jl:132-134 | same | `mod1`/`div` decomposition | Reshape |
| ir/utils.jl:139-163 | same | **6 boundary branches** (`i<nx`, `j<ny`, `k<nz`, `i>1`, `j>1`, `k>1`) | Shifted arrays with zero padding |
| ir/utils.jl:181-183 | `_ones_like` | `::Nothing` vs `::AbstractArray` dispatch | Host |
| ir/utils.jl:202 | `compute_projection_weights` | `circular_support && apply_fov_mask!(...)` | Host |
| ir/utils.jl:205-208 | same | `AK.foreachindex`; `val > eps ? one(T)/val : zero(T)` — hard reciprocal gate, **in place** | `1 ./ max(val, eps)`; note ∂ = 0 in the dead zone |
| ir/utils.jl:231-236 | `compute_image_weights` | `if projector === :siddon` selecting two backprojector calls | Host |
| ir/utils.jl:238-241 | same | `AK.foreachindex`; same reciprocal gate | Same |
| ir/utils.jl:254-261 | `create_ordered_subsets` | `push!` into `Vector{Vector{Int}}`; `mod1(i, n_subsets)` | Host; precompute |
| ir/utils.jl:268-280 | `create_subset_geometry` | `geom.angles[angle_indices]` fancy indexing ×5 | Host |
| ir/utils.jl:287-297 | `extract_subset_sinogram` | per-index slice copy loop | `gather` |
| hybrid_ir.jl:127-136 | `get_hir_params` | `if !(0 ≤ strength ≤ 100) \|\| strength % 10 != 0` → `throw(ArgumentError)` with a conditional hint | Host |
| hybrid_ir.jl:140-142 | same | `for a in _HIR_ANCHORS; a[1] == strength && return` exact-match scan | Host |
| hybrid_ir.jl:144 | same | **`findlast(a -> a[1] < strength, _HIR_ANCHORS)::Int`** | Host |
| hybrid_ir.jl:149-158 | same | `lerp` then `max(1, round(Int, …))` for `n_iter` and `round(Int, …)` ×2 for the window — **iteration count from an interpolated float** | Host; freeze per configuration |

## VMI — Cong decomposition and the Brent kernel

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| cong.jl:102-108 | `create_cong_workspace` | eltype checks with `error`; `ndims` equality check | Host |
| cong.jl:166-171 | `apply_cong!` | 2 validation `error`s | Host |
| cong.jl:175-177 | same | `per_ray = ndims(ŵ_L) == 3`; `nE_L = per_ray ? size(ŵ_L,3) : length(ŵ_L)` | **Two kernel variants selected by array rank** |
| cong.jl:181 | same | `p_L_min = Float32(minimum(p_L))` | Host reduction |
| cong.jl:184 | same | `minimum(Float32.(Array(p_L)).*a_w .+ Float32.(Array(q_L)).*c_w)` — **device→host `Array()`** | Host precompute |
| cong.jl:195 | same | `AK.foreachindex(sino_low)` over rays | Array-level |
| cong.jl:196-198 | same | `%`/`÷` index decomposition | Reshape |
| cong.jl:206-210 | same | **symmetric air gate** `abs(p_L)<5f-3 && abs(p_H)<5f-3` → write 0 and `return` | Compute then `ifelse`; the gate zeroes the gradient on air rays |
| cong.jl:211-212 | same | `T_L_meas = exp(-p_L_meas)`; `T_H_meas = exp(-p_H_meas)` | Fine |
| cong.jl:215-227 | same | `water_T_L` **closure** with `if per_ray` and `for i in 1:nE_L`, per-ray `ŵ_L[col,row,i]` gather | Batch the energy axis |
| cong.jl:233 | same | `L_hi = max(60f0, 1.5f0·p_L_meas/max(μ_w_min, 1f-4))` — **data-dependent bracket** | Fixed generous bracket |
| cong.jl:234 | same | **`brent_solve(water_T_L, -1f0, L_hi)` — root-find inside the GPU kernel** | Implicit-function-theorem custom rule: `∂L/∂θ = −(∂F/∂θ)/(∂F/∂L)` at the root; or a fixed-count Newton |
| cong.jl:235-239 | same | `if !ok_L` → write zeros and `return` | `ifelse` |
| cong.jl:242-247 | same | `y_max = min(y_fac·p_L/max(p_L_min, eps), y_cap)`; `if y_max <= 0` → fallback and `return` | `ifelse` |
| cong.jl:250-291 | `solve_quintic` (closure) | builds P0..P5 Taylor moments over the energy loop, then Newton `for _ in 1:nm_iter` with **two data-dependent `break`s** (`abs(dF) < 1f-30`, `abs(Δ) < n_tol`) | Fixed 12 iterations, no break; apply IFT for the gradient |
| cong.jl:294-306 | `T_H_pred` (closure) | `if per_ray` + energy loop | Batch |
| cong.jl:308-311 | `G` (closure) | calls `solve_quintic` then `T_H_pred` — **nested solver** | IFT on the outer only |
| cong.jl:321-333 | `apply_cong!` | `G0 = G(0f0)`; 3-way branch on `sign(G0)`; **`while isfinite(G_hi) && G_hi < 0f0 && y_hi < y_max && n_exp < Int32(24)`** geometric bracket expansion | The `isfinite` test alone is untraceable. Replace with a fixed 24-step expansion, masked |
| cong.jl:334-338 | same | `if isfinite(G_hi) && G_hi >= 0` → `brent_solve(G, 0, y_hi)` else `(0, false)` | Second root-find |
| cong.jl:343-348 | same | `Glo = G(-0.1f0)`; `if isfinite(Glo) && Glo < 0` → `brent_solve(G, -0.1, 0)` | Third root-find |
| cong.jl:350-354 | same | `if !ok_y` → water-only fallback and `return` | `ifelse` |
| cong.jl:355 | same | `x_final = solve_quintic(y_opt, c̄)` — fourth solver call | IFT |
| cong.jl:361-362 | same | `clamp(y_opt, -5f0, 1f4)`; `clamp(c̄ + x_final, -5f0, 1f4)` | Saturating |
| roots_kernels.jl:35-47 | `__middle_bits` (2 methods) | **`reinterpret(UInt64/UInt32, abs(x))` + `(xint+yint) >> 1` + `reinterpret` back** | Bit manipulation on floats — not differentiable, not expressible in StableHLO. Use arithmetic bisection `(a+b)/2` |
| roots_kernels.jl:51-59 | `_middle_gpu` | `isinf(x) ? nextfloat(x) : x`; `isinf(y) ? prevfloat(y) : y`; `sign(a)*sign(b) < 0` | Same class |
| roots_kernels.jl:65 | `secant_step_gpu` | `a - fa·(b-a)/(fb-fa)` — division by a difference that can be 0 | Guard |
| roots_kernels.jl:67-73 | `inverse_quadratic_step_gpu` | 6 chained divisions by pairwise differences | Guard |
| roots_kernels.jl:104-110 | `brent_solve` | `fa = f(a)`; `fb = f(b)`; `iszero(fa) && return`; `iszero(fb) && return` | Mask |
| roots_kernels.jl:112-115 | same | conditional swap `if abs(fa) < abs(fb)` | `ifelse` pair |
| roots_kernels.jl:118-121 | same | `if sign(fa)*sign(fb) > 0; return (best, false)` invalid-bracket exit | Mask |
| roots_kernels.jl:127-191 | same | `for _ in 1:maxiters` with an **`mflag` state machine**, `isnan(s) \|\| isinf(s)` fallback, a 5-clause `force_bisect` predicate, `iszero(fs) && return`, `isnan(fs) \|\| isinf(fs) && return`, and **`nextfloat(lo) == hi` bit-exact termination** | Wholesale replacement: fixed-count bisection+Newton, or lift the solve out of the kernel entirely |
| roots_kernels.jl:144 | same | `tol = max(xabstol, max(abs(b),abs(c),abs(d))·xreltol)` | Fine |
| roots_kernels.jl:170-183 | same | bracket update with a sign-tiebreak branch and a second conditional swap | `ifelse` |

## VMI — CMV, PWLS, RWLS, polynomial calibration, basis tables

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| cmv.jl:48-54 | `apply_cmv!` | `per_ray ? Float64.(Array(ŵ_raw)[mc, mr, :]) : Float64.(Array(ŵ_raw))` — **device→host** | Host precompute |
| cmv.jl:55-56 | same | `ŵ ./= sum(ŵ)` ×2 | Host |
| cmv.jl:58-66 | same | 4 × `Float64.(Array(basis.X))`; 4 spectrum-weighted means | Host |
| cmv.jl:68-72 | same | `abs(det_M) < eps(Float64)*1e3 && error(...)` | Host validation |
| cmv.jl:74-78 | same | `inv_det = Float32(1.0/det_M)` and 4 × `Float32(...)` narrowing | Host |
| cmv.jl:80 | same | `AK.foreachindex(sino_low)` | Array-level |
| cmv.jl:85-86 | same | **`ifelse(a_w > 0f0, a_w, 0f0)`** ReLU projection ×2 | Already the right form; broadcast it |
| pwls.jl:95-102 | `create_pwls_workspace` | `with_oom_retry` closure allocating 4 tile buffers | See memory_budget.jl:228 |
| pwls.jl:112, 125 | `_pwls_lapl_{x,y}!` | `AK.foreachindex` ×2 | Shifted arrays with replicate pad |
| pwls.jl:113-118, 126-131 | same | `%`/`÷` decomposition ×2 | Reshape |
| pwls.jl:117-118, 130-131 | same | `c == 1 ? c : c-1`; `c == n_col ? c : c+1` Neumann BC ×4 | Replicate padding |
| pwls.jl:133 | `_pwls_lapl_y!` | `out[...] = accum ? out[...] + val : val` — **accumulate flag as a runtime `Bool`** | Two variants |
| pwls.jl:138-144 | `_pwls_apply_CtC!` | 4 kernel launches per call | Fuse |
| pwls.jl:212-230 | `apply_pwls!` | 8 validation `error`s; `per_ray = ndims(...) == 3` | Host |
| pwls.jl:234-239 | same | 3 shape-mismatch `error`s | Host |
| pwls.jl:243-259 | `_normalize_ŵ` | host per-ray loop with `if s > 0f0` | Broadcast `./ max(sum, eps)` |
| pwls.jl:262-267 | same | 6 × `_match_backend` staging | Host |
| pwls.jl:271-274 | same | 2 length-mismatch `error`s | Host |
| pwls.jl:288-292 | same | `I0_L_raw > 0 && I0_H_raw > 0 \|\| error`; `I0_avg` normalization | Host |
| pwls.jl:295 | same | `n_tiles = cld(n_view, ws.tile_size)` | Host |
| pwls.jl:299-305 | same | `for iter in 1:n_it` × `for tile_range in tile_ranges(...)` + 8 `view`s — **tiling changes shapes** | Fix a single shape |
| pwls.jl:317-318 | same | `_pwls_apply_CtC!` ×2 (8 kernel launches) | Fuse |
| pwls.jl:320 | same | `AK.foreachindex(sI_t)` fused SQS kernel | Array-level |
| pwls.jl:324-326 | same | `%`/`÷` decomposition | Reshape |
| pwls.jl:330-340 | same | `if per_ray` ×2 selecting whole energy loops; `for kk in 1:nE_L` — **runtime-length loops** with per-ray `ŵ[col,row,kk]` gathers | Energy as an array axis; this is a batched matvec |
| pwls.jl:341, 343 | same | `invZ_L = 1f0/max(Z_L, 1f-20)`; `f_L = -log(max(Z_L, 1f-20))` | Subgradient at the floor |
| pwls.jl:347-360 | same | same pattern for the high channel | Same |
| pwls.jl:369-370 | same | `wL = wL_scale·exp(-h_Lv)`; `wH = …` Poisson weights | Differentiable |
| pwls.jl:388-391 | same | `det_m = m_II·m_WW - m_IW²`; `inv_det = 1f0/max(det_m, 1f-20)`; explicit 2×2 inverse | Use an `ifelse` guard, not a branch |
| pwls.jl:393-394 | same | **`max(Iv - relax·ΔI, 0f0)`** non-negativity projection ×2, **in place** | Piecewise; gradient zero when clipped |
| pwls.jl:397-398 | same | `sum(cost_t)`; `sum(sI_t .* reg_I_t) + sum(sW_t .* reg_W_t)` reductions | Fine |
| pwls.jl:402-404 | same | `if cost_history[iter] > cost_history[iter-1]` `@warn` | Host |
| rwls.jl:154-181 | `_apply_I_plus_αβL!` | `AK.foreachindex`; `÷` decomposition; **periodic wrap** `i+1 == N1_ ? 0 : i+1` ×4 | `circshift` |
| rwls.jl:185-221 | `_rwls_cg_solve!` | `copyto!(x, b)`; `@.` residual; **CG `for k in 1:max_iter` with `if rr_new < tol_sq; break`** — data-dependent iteration count | Fixed 20 iterations; or differentiate the linear solve via IFT (it's SPD, so `Aᵀ = A`) |
| rwls.jl:200-207, 211 | same | `sum(r .* r)`; `max(sum(b.*b), 1f-30)`; `max(sum(p.*Ap), 1f-30)` | Guarded |
| rwls.jl:220 | same | `sqrt(max(rr, 0f0)/b_norm_sq)` | Fine |
| rwls.jl:224-243 | `_rwls_cg_prior!` | two sequential CG solves reusing buffers; `if verbose @info` | Host |
| rwls.jl:255-301 | `_rwls_fused_step_n3!` | `AK.foreachindex`; `@inbounds for k in 1:nE` runtime-length energy loop; **hardcoded 3-bin unroll** | Energy axis + bin axis as array dims |
| rwls.jl:290-292 | same | **`wt = 1f0/max(F, 1f0)`** Poisson weight gate ×3 | The `max(F,1)` floor zeroes gradients on starved rays |
| rwls.jl:380-404 | `apply_rwls!` | 9 validation `error`s incl. workspace-shape checks | Host |
| rwls.jl:409-416 | same | host staging loop; `s > 0 \|\| error`; `copyto!` ×5 | Host |
| rwls.jl:428 | same | `do_prior = (α_bw > 0.0) \|\| (α_bI > 0.0)` | Host branch |
| rwls.jl:431-445 | same | `for tile_range in tile_ranges(...)` + 12 `view`s | Fix shapes |
| rwls.jl:447 | same | `for iter in 1:Int(n_iter)` unrolls the tape | Checkpoint |
| rwls.jl:461-469 | same | `@. begin` block: `max(abs(H11·H22 - H12²), 1f-30)`; `clamp(δ_W, -step_W, step_W)`; `clamp(δ_I, -step_I, step_I)`; `max(sW + relax·δ_W, 0f0)`; `max(sI + relax·δ_I, 0f0)` — **buffers reused as transient storage** | Saturating step limits + projection; aliasing hazard for Enzyme |
| rwls.jl:472-477 | same | `if do_prior` → CG prior on both channels | Host |
| pcct_calibration.jl:72-77 | `calibrate_pcct_vmi_poly` | spectrum resolve; `kVp = Float64(maximum(e_full))`; `compute_mc_drm` | Host |
| pcct_calibration.jl:81 | same | `drm_row(E) = clamp(round(Int, (E-1)/(kVp-1)·(n_R-1))+1, 1, n_R)` | As mc_response.jl:355 |
| pcct_calibration.jl:84 | same | `high_bins === nothing ? ((last(low_bins)+1):n_bins) : high_bins` | Host |
| pcct_calibration.jl:88-95 | same | comprehensions with `sum(R_mat[drm_row(e[i]), b] for b in group)`; normalize | Host |
| pcct_calibration.jl:102-104 | same | Chebyshev grid `xmax/2·(1 - cos((2m-1)/(2n)·π))`; `vcat(0.0, …)` | Host |
| pcct_calibration.jl:114-122 | same | double host loop over `(tI, tw)`; `-log(max(tr, 1e-30))` ×2 | Vectorize |
| pcct_calibration.jl:125-126 | same | `terms = [(i,j) for i in 0:order for j in 0:(order-i)]`; `hcat([p_low.^i .* p_high.^j …]...)` | Host |
| pcct_calibration.jl:127-128 | same | **`A_mat \ t_water` / `A_mat \ t_iodine`** — bivariate polynomial LSQ | IFT / XLA QR |
| pcct_calibration.jl:132-136 | same | RMS diagnostics; effective mean energies | Host |
| pcct_calibration.jl:175-182 | `apply_pcct_vmi_poly!` | `_eval_poly` closure with `@inbounds for k in eachindex(coeffs)`; `i, j = terms[k]` tuple unpack | Precompute the monomial basis as a tensor contraction |
| pcct_calibration.jl:184 | same | **`@inbounds Threads.@threads for idx in eachindex(sino_low)`** | Batch |
| pcct_calibration.jl:185-190 | same | `Float64(sino_low[idx])` widening; **`pl^i * ph^j` with runtime integer exponents**; `T(max(aw, 0.0))` clip | Tensor contraction; the clip is a hard gate |
| pcct_calibration.jl:274, 345 | kVp lookups | `Int(round(kVp))` / `round(Int, kVp)` as a `Dict` key | Host |
| clinical_calibrations.jl:313-314, 386-387 | calibration lookups | `round(Int, low_kVp)` / `round(Int, high_kVp)` as `Dict` keys | Host |
| pcct_basis.jl:39-45 | `pcct_effective_spectrum` | spectrum resolve; `maximum(e_full)`; `drm_row` rounding | Host |
| pcct_basis.jl:50-58 | same | comprehension with `sum(R_mat[drm_row(e[i]), b] for b in grp)`; `total > 0 \|\| error`; normalize | Host |
| pcct_basis.jl:85-87 | `pcct_rwls_basis` | 2 comprehensions calling `compute_mass_μ_at_energy` (Float64) | Host |
| pcct_basis.jl:137-145 | `combine_pcct_bin_counts!` | 3 validation `error`s | Host |
| pcct_basis.jl:148 | same | `chunk = chunk_size === nothing ? n_view : Int(chunk_size)` | Host |
| pcct_basis.jl:154 | same | **`typeof(first(raw_bins)).name === typeof(template).name`** backend sniffing | Explicit backend argument |
| pcct_basis.jl:155-156 | same | `staging = same_backend ? nothing : similar(...)` | Two variants |
| pcct_basis.jl:161-166 | same | `for b in grp`; `@. out_k = out_k + I0b·exp(-raw_b)` **in-place accumulate** | Out-of-place `sum` over the bin axis |
| pcct_basis.jl:167-171 | same | else-branch streaming with chunked `copyto!` | Host |
| basis.jl:28-32 | `p_photoelectric` | **`sqrt(32/ε^7)`** — with ε ≈ 0.1, ε⁷ ≈ 1e-7 and the ratio ≈ 3e8 | Float32 overflow/precision hazard; compute in Float64 or reparameterize in log space |
| basis.jl:36-47 | `q_compton` | Klein-Nishina with `log(1+2ε)` ×2 and `1/ε`, `1/(2ε)` factors | ε → 0 singularity; guard |
| basis.jl:57-70 | `water_basis_constants` | closed-form composition arithmetic; `Float32(...)` narrowing | Host |
| basis.jl:88 | `compute_photo_compton_basis` | `Base.depwarn` (deprecated basis) | Host |
| image_domain_decomp.jl:65-67 | `fit_ding_coeffs` | length-mismatch `error` | Host |
| image_domain_decomp.jl:74-77 | same | `A = hcat(ones(n), HU_low, HU_high)`; **`A \ c_f`** LSQ; RMS | IFT |
| image_domain_decomp.jl:82-85 | same | **`findall(c -> c > 0, c_f)`** — dynamic-shape index set; then `Σc²` and two slope fits | Precompute on host |
| image_domain_decomp.jl:118-122 | `apply_ding_decomp!` | shape `error`s; `@. c_iodine = a0 + a1·HU_low + a2·HU_high` | Already traceable |
| image_domain_decomp.jl:185-192 | `synth_vmi_image_domain!` | shape check via a helper that `error`s; `@. HU_E = HU_low + c_iodine·Δα` | Traceable |
| image_domain_decomp.jl:235+ | `eval_cal` | `form::Symbol` runtime dispatch between two calibration forms | Two variants |

## VMI synthesis, Mono+, masking

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| vmi_synth.jl:46-47 | `synth_vmi_hu` | `mask = fov_mask_radius_frac !== nothing`; `r_fov_sq = mask ? … : -1.0` | Host |
| vmi_synth.jl:53-57 | same | `p_photoelectric` / `q_compton` per energy; `@. p_E·a + q_E·c`; `to_hounsfield` | Traceable |
| vmi_synth.jl:58-64 | same | `if mask` then a triple loop with `if (i-cx)² + (j-cy)² > r_fov_sq; hu[...] = -1000f0` | Precomputed `Bool` mask + `ifelse` |
| vmi_synth.jl:70-76 | same | `if size ≥ 384`; `quantile(vec(inner), 0.01)` / `0.99` in the verbose log | Order statistic; logging only |
| vmi_synth.jl:143-144 | `synth_vmi_sino_domain` | length-mismatch `error` | Host |
| vmi_synth.jl:148-149 | same | `first(values(sino_a_by_E))` from a `Dict`; `fbp_workspace_builder(...)` **function-valued callback** | Dynamic dispatch through a `Function` field blocks tracing; specialize |
| vmi_synth.jl:165-167 | same | `Dict` lookup by `Float64` key ×2; `@. μρ_a·sino_a + μρ_b·sino_b` | Host lookup, traceable arithmetic |
| vmi_synth.jl:169-171 | same | `fbp_recon!(ws, sino_E, geom, matrix_size)` **callback**; `to_hounsfield(Array(μ_vol); …)` — **device→host** | Specialize; keep on device |
| vmi_synth.jl:173-179 | same | same hard FOV mask → `-1000f0` | Precomputed mask |
| phantom_mask.jl:65-72 | `resample_phantom_mask_to_recon` | 4 × `origin === nothing ? … : …` | Host |
| phantom_mask.jl:74-87 | `_sample_2d` closure | double loop; `round(Int, cx_p + x_cm/px)` ×2; `if 1 ≤ ip ≤ pnx && 1 ≤ jp ≤ pny`; `slice2d[ip,jp] > 0` threshold | Precompute the gather map |
| phantom_mask.jl:90-103 | same | `if broadcast_in_z` selecting two whole paths; `clamp(round(Int, (k-0.5)·pz_per_rz + 0.5), 1, pnz)` | Host |
| phantom_mask.jl:123-131 | `erode_mask_2d` | `erode_px > 0 \|\| return copy`; FFT blur in **Float64**; **`blurred .≥ 0.999` hard threshold** | Discrete output; precompute on host |
| phantom_mask.jl:148-166 | `erode_mask_3d` | `if broadcast_in_z` two paths; per-slice erosion | Host |
| mono_plus.jl:97-103 | `create_mono_plus_workspace` | `mem_budget_GB === nothing ? floor(Int, Sys.free_memory()·0.6) : floor(Int, GB·2^30)`; `@error` when short | Host |
| mono_plus.jl:169-182, 362-375 | both apply fns | 6 validation `error`s each | Host |
| mono_plus.jl:184, 377 | both | **`findfirst(==(Float64(E_noise_opt)), Float64.(energies))`** — exact float equality search + `error` | Pass the index directly |
| mono_plus.jl:188-194, 381-387 | both | `σ_lp_px isa Real ? fill(...) : (length check + Float64.(...))` | Host |
| mono_plus.jl:198-199, 391-392 | both | frequency grids `fx`, `fy` built in Float64 | Host |
| mono_plus.jl:202-207, 395-400 | both | `_kernel_for(σ)` with **`get!(ws.kernel_cache, σ)` — `Dict` keyed by `Float64`** | Precompute kernels |
| mono_plus.jl:210-217, 401-408 | both | `_gaussian_lp!`: `Threads.@threads for k in 1:nz`; `Float64.(@view img[:,:,k])`; `FFTW.fft`/`ifft`; `Float32.(real.(...))` | Batched FFT; avoid the width round trip |
| mono_plus.jl:222-231, 409-416 | both | `_apply_mask!`: `if phantom_mask !== nothing`; `if !phantom_mask[j]; out[j] = src[j]` | `ifelse` broadcast |
| mono_plus.jl:234, 460 | both | **`all(σ -> σ == σ_vec[1], σ_vec)`** float equality | Host |
| mono_plus.jl:236-271 | `apply_mono_plus!` | 3-level nesting: `if σ_uniform` → `if σ0 == 0.0` → `for i` with `if i == i_opt` | Host |
| mono_plus.jl:245, 251, 267 | same | `@. hp_opt = vmi_opt - lp_opt`; `@. out = lp_buf + hp_opt`; `@. out = lp_buf + vmi_opt - lp_opt` | Traceable |
| mono_plus.jl:423-431 | `_hf_regress!` | global HF-energy accumulation; `λ = sqrt(sEE/max(sOO, 1e-30))`; `βmax = Float32(min(beta_max, λ))` | Fine |
| mono_plus.jl:433-454 | same | **`Threads.SpinLock`** + `Threads.@threads for k in 1:nz` with a locked reduction | Box-filter (cumulative-sum) the window sums |
| mono_plus.jl:438-446 | same | double `for dj/di` window with 2 bounds `continue`s; `sOOw`, `sEOw` accumulation | Box filter |
| mono_plus.jl:447-448 | same | **`clamp(sEOw/max(sOOw, 1f-20), 0.0f0, βmax)`** one-sided clamp; `out = lpE + β·hOpt` | Hard gate at 0 |
| mono_plus.jl:449 | same | `(β > 0) && (local_active += 1)` diagnostic counter | Discrete |
| mono_plus.jl:463-495 | `apply_mono_plus_regression!` | same 3-level host nesting | Host |

## Denoising / ACNR

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| acnr.jl:68-69 | `apply_acnr!` | `(σ === nothing) ⊻ (λ === nothing) \|\| error` | Host |
| acnr.jl:73-74 | same | `c_sq = c_a² + c_b²`; `c_sq > 0 \|\| error` | Host |
| acnr.jl:78 | same | `s_orth = @. Float64(-c_b·sino_a + c_a·sino_b)` — **built in Float64** | Width round trip |
| acnr.jl:82-96 | same | `if σ !== nothing` selecting Gaussian vs Tikhonov transfer function; both built as host comprehensions | Precompute |
| acnr.jl:100-103 | same | **`Threads.@threads for k in 1:n3`** + `FFTW.fft`/`ifft` per z-slice in `Complex{Float64}` | Batched FFT; XLA has one |
| acnr.jl:108-113 | same | `n_orth = Float32.(s_orth .- s_smooth)`; `@. sino_a += α_a·n_orth`; `@. sino_b -= α_b·n_orth` **in place** | Traceable; make out-of-place |
| acnr.jl:115-117 | same | `std()` ×3 diagnostics | Fine |
| acnr.jl:188-189 | `apply_image_acnr!` | shape `error` | Host |
| acnr.jl:195-202 | same | `_nstd` closure: triple loop adjacent-difference noise estimate; `max(_nstd(V), 1e-8)` ×2 | Traceable (shifted-array difference) |
| acnr.jl:205-212 | same | `mW`, `mI` means; covariance triple accumulation; `ρ = bb/sqrt(max(a,1e-30)·max(c,1e-30))` | Traceable |
| acnr.jl:214-217 | same | `if γ <= 0` → early return with diagnostics | Host |
| acnr.jl:219-220 | same | **`θ = 0.5·atan(2bb, a - c)`** — eigen-rotation angle learned from the data; `cos(θ)`, `sin(θ)` | Differentiable, but the branch cut and the degenerate case `a ≈ c` are hazards |
| acnr.jl:223-229 | same | `Array{T}(undef, ...)` ×2; rotation into (signal, noise) axes | Traceable |
| acnr.jl:232-236 | same | `denW`, `denI` range scales; spatial weight matrix `sw` comprehension | Precompute |
| acnr.jl:239-255 | same | quadruple loop joint bilateral with **data-dependent weights** `exp(-(dW²/denW + dI²/denI))`; 2 bounds `continue`s; `acc/wsum` | Differentiable (weights are smooth) but O(r²) scalar loop — vectorize as shifted-array sums |
| acnr.jl:259-264 | same | reconstitute: `pn = p_noise + γ(p_noise_s - p_noise)`; rotate back; restore means, **in place** | Out-of-place |
| acnr.jl:333-340 | `apply_acnr_kalender!` | `if passes > 1` **recursion** over `apply_acnr_kalender!` | Unroll to a fixed pass count |
| acnr.jl:344-370 | same, `G` closure | `r = max(2, ceil(Int, 3σ))` **shape from σ**; kernel `k ./= sum(k)`; two separable passes with `clamp(i+t, 1, nx)` / `clamp(j+t, 1, ny)` replicate | Fixed kernel + replicate-pad conv |
| acnr.jl:372 | same | `Wc = Array(W); Ic = Array(I)` — **device→host** | Keep on device |
| acnr.jl:373-374 | same | `hW = Wc .- G(Wc)`; `hI = Ic .- G(Ic)` | Traceable |
| acnr.jl:381-384 | same | `λ_I = sqrt(sum(abs2,hI)/max(sum(abs2,hW), 1e-30))`; `βmaxI = min(beta_max, λ_I)` (and the symmetric pair) | Fine |
| acnr.jl:386-403 | same | triple loop with a `(2w+1)²` window; 1 bounds `continue`; **`clamp(sWI/max(sWW, 1e-20), -βmaxI, 0)`** one-sided clamp ×2 | Box-filter the sums; hard gate at 0 |
| acnr.jl:404 | same | `copyto!(W, outW); copyto!(I, outI)` in place | Out-of-place |
| acnr.jl:406 | same | `ρ = Σ(hW·hI)/sqrt(max(ΣhW²·ΣhI², 1e-30))` | Fine |
| sino_svd.jl:101-110 | `apply_sino_svd_denoise!` | 4 validation `error`s | Host |
| sino_svd.jl:115-120 | same | `if σ ≤ 0` passthrough `copyto!` | Host |
| sino_svd.jl:123-124 | same | `radius = max(1, ceil(Int, 3σ))` **shape from σ**; `ks ./= sum(ks)` | Fixed |
| sino_svd.jl:127, 295 | both variants | **`Threads.@threads for r in 1:n_row`** | Batch the row axis |
| sino_svd.jl:129-130, 296-297 | both | `[Float32.(@view channels[b][:,r,:]) for b in 1:n_ch]`; `hcat([vec(s) …]...)` per row | Reshape/permute once |
| sino_svd.jl:133, 299 | both | **`LinearAlgebra.svd(M; full=false)` per detector row** | Enzyme: differentiable but blows up when singular values collide (`1/(σᵢ²-σⱼ²)`). Reactant/XLA has no batched SVD on Metal. Use an eigendecomposition of the small `MᵀM` (2×2 or 4×4 — closed form) |
| sino_svd.jl:137-141, 302-311 | both | `U_d[:,1] .= U[:,1]` keep rank-1; denoise components 2..N | Traceable given a differentiable SVD |
| sino_svd.jl:144, 313 | both | `M_d = U_d · Diagonal(Σ) · V'` | Traceable |
| sino_svd.jl:146, 315 | both | `out[b][:,r,:] .= reshape(view(M_d,:,b), n_col, n_view)` in place | Out-of-place |
| sino_svd.jl:150, 319 | both | `@info` per call | Host |
| sino_svd.jl:195-212 | `_separable_gauss_2d` | two passes with `(1 <= c2 <= nc) \|\| continue` and **weight renormalization `s/w`** | Replicate-pad conv |
| sino_svd.jl:284-289 | bilateral variant | `if bilat_range_k ≤ 0` passthrough | Host |
| sino_svd.jl:305-309 | same | `σg = kf·_mad_scale_2d(guide)`; `σt = kf·_mad_scale_2d(tgt)` per component | Order statistics |
| sino_svd.jl:345-350 | `_mad_scale_2d` | **`Statistics.median` ×2** (MAD scale); `max(1.4826·mad, 1f-12)` | Order statistic — subgradient only |
| sino_svd.jl:365-390 | `_joint_bilateral_2d` | quadruple loop; 2 bounds `continue`s; `zg = dg/σg`, `zt = dt/σt`; `exp(-dist²/σs² - 0.5zg² - 0.5zt²)`; `acc/wsum` | Differentiable but scalar-looped |
| sino_sfjsd.jl:63-66 | module constants | `_SFJSD_α`, `_SFJSD_lavg`, `_SFJSD_σ_ref_px`, `_SFJSD_σ_cap` hidden knobs | Host |
| sino_sfjsd.jl:82-84 | `_sfjsd_sep_gauss_3d` | `σ ≤ 0 && return copy`; `radius = max(1, ceil(Int, 3σ))` | Fixed |
| sino_sfjsd.jl:88, 168, 506 | three sites | **`Threads.@threads`** ×3 | Batch |
| sino_sfjsd.jl:101-127 | `_sfjsd_gauss_2d` | two passes with boundary `continue` + `s/w` renormalization | Replicate-pad |
| sino_sfjsd.jl:136-140 | `_sfjsd_mad_scale` | **`median` ×2** | Order statistic |
| sino_sfjsd.jl:162 | `_sfjsd_pass!` | `radius = max(1, ceil(Int, 3σ_sp))` shape from σ | Fixed |
| sino_sfjsd.jl:163-165 | same | 3 × `1/(2σ² + 1e-30)` guarded reciprocals | Fine |
| sino_sfjsd.jl:171-176 | same | **`for dv in -radius:stride:radius`** (stride from data) with 2 bounds `continue`s | Fix the stride; vectorize |
| sino_sfjsd.jl:180-191 | same | inner 5×5 local-average loop with 2 bounds `continue`s; `if cnt > 0` normalize | Box filter |
| sino_sfjsd.jl:194-202 | same | `log_w` sum; `exp(log_w)`; `sum_v/max(wtot, 1f-30)` | Differentiable |
| sino_sfjsd.jl:219, 511 | `_sfjsd_apply_D`, main loop | **`svd(M; full=false)` per row** ×2 | Here `M` is `(n·v)×2`, so `MᵀM` is 2×2 with a closed-form eigendecomposition |
| sino_sfjsd.jl:222-223, 514-515 | both | `σ1 = min(σ0, σ_cap)`; **`σ2 = min(σ0·sqrt(Σ[1]/max(Σ[2],1e-12)), σ_cap)`** — bandwidth from singular values | Differentiable but singular-value-ratio-driven |
| sino_sfjsd.jl:227-228, 520-521 | both | `max(_sfjsd_mad_scale(Λ), eps(Float32))` ×2 | Order statistic |
| sino_sfjsd.jl:250-254 | `_sfjsd_sure` | **`MersenneTwister(42)` + `randn`** Hutchinson divergence probe; `δ = max(1e-3·std(M), 1e-8)`; finite-difference divergence | RNG + finite-difference derivative — replace with the exact AD divergence |
| sino_sfjsd.jl:274-293 | `_sfjsd_sure_optimize` | **golden-section `while abs(b-a) > tol`** — data-dependent hyperparameter search with `if fc < fd` | Fixed iteration count, or hoist σ★ out as a constant |
| sino_sfjsd.jl:306-349 | `_sfjsd_corr_length` | patch extraction; Gaussian smooth with `continue`; autocovariance over 5 lags; **linear scan for the half-max crossing** with `if r2 ≤ 0.5` returning an interpolated Float | Host precompute |
| sino_sfjsd.jl:357-358 | `_sfjsd_pick_stride` | `corr_len < 1.5 ? 1 : (corr_len < 2.5 ? 2 : 3)` — **stride from data** | Changes the *program shape*; must be a host constant |
| sino_sfjsd.jl:367 | `_sfjsd_pick_n_iter` | `min_N ≥ 100 ? 1 : 2` — **iteration count from data** | Host constant |
| sino_sfjsd.jl:439-449 | `apply_sino_sfjsd_denoise` | 3 validation `error`s | Host |
| sino_sfjsd.jl:457-459 | same | `N = I0·exp(-p)` ×2; `min(minimum(N_lo), minimum(N_hi))` | Subgradient |
| sino_sfjsd.jl:462-464 | same | `max(_sfjsd_corr_length(p_lo), _sfjsd_corr_length(p_hi))`; stride and n_iter picks | Host |
| sino_sfjsd.jl:479-482 | same | `w = sqrt.(max.(N, 1.0f0))` whitening; `ξ = w .* (p - p_ref)` | Hard floor |
| sino_sfjsd.jl:484-486 | same | `p_lo = nothing` rebinding + **`GC.gc(true)`** mid-function | Host-only |
| sino_sfjsd.jl:490-500 | same | `if σ₀_user > 0` else run SURE on the mid row | Host |
| sino_sfjsd.jl:505-536 | same | `for t in 0:(n_iter-1)` with a threaded per-row SVD body; `σ0 *= α` decay | Batch; fix n_iter |
| sino_sfjsd.jl:539-540 | same | inverse whiten `ξ ./ w .+ p_ref` | Traceable |
| median_z.jl:47-49 | `apply_median_z!` | 2 validation `error`s; `adjacent_slices == 0 && (copyto!; return)` | Host |
| median_z.jl:52 | same | **`Threads.@threads for k in 1:nz`** | Batch |
| median_z.jl:53-54 | same | `klo = max(1, k-a)`, `khi = min(nz, k+a)`, `n = khi-klo+1` — **window shrinks at slice boundaries** | Replicate-pad to a fixed window |
| median_z.jl:56-61 | same | per-voxel buffer fill, **`sort!(view(buf, 1:n))`**, `buf[(n+1)÷2]` selection | Non-differentiable (subgradient = one-hot at the selected element). For a fixed window a sorting network is traceable; the gradient is a permutation gather |
| rskr.jl:41-53 | `mad_haar_σ` | mid-z slice; Haar HH difference over `(n_col-1)(n_row-1)` pairs; **`median(absvec)`** | Order statistic |
| rskr.jl:77-80 | `joint_bf_2ch_gpu!` | 2 × `1/(2(hσ)² + 1e-30)`; `Int32(radius²)` | Fine |
| rskr.jl:82, 153 | `joint_bf_{2,4}ch_gpu!` | `AK.foreachindex` ×2 | Array-level |
| rskr.jl:83-86, 154-157 | both | `%`/`÷` index decomposition ×2 | Reshape |
| rskr.jl:91-116, 164-199 | both | triple `for dz/dr/dc` with `continue` on bounds **and** on `dist² > radius²` (spherical mask) | Precomputed offset table + mask |
| rskr.jl:109-110, 187-191 | both | `log_w` sum then `w = exp(log_w)` — data-dependent bilateral weights | Differentiable |
| rskr.jl:118-121, 201-206 | both | `inv_w = 1/max(wtot, 1e-30)`; scaled writes | Fine |
| rskr.jl:268-273 | `apply_rskr` | `nch in (2,4) \|\| error`; per-volume shape checks | Host |
| rskr.jl:276-279 | same | `Matrix{Float32}(undef, n_vox, nch)`; `V_mat[:,b] .= vec(vols[b])` | Reshape |
| rskr.jl:283 | same | `for iter in 1:n_iter` re-SVD each iteration | Unrolls the tape |
| rskr.jl:285 | same | **`svd(V_mat; full=false)` on the full `(n_vox × nch)` matrix** | Small right factor — use the `VᵀV` eigendecomposition (2×2 or 4×4 closed form) |
| rskr.jl:290-294 | same | `[reshape(U[:,e], sz) …]`; `mad_haar_σ` per component; **`h = h₀·(Σ[1]/max(Σ[e],1e-12))^γ`** | Bandwidth from singular values; differentiable but singular |
| rskr.jl:302-328 | same | host `if nch == 2` else branch; `gpu_arr_type(...)` uploads; `Array(out)` downloads; `nothing` rebinding | Host |
| rskr.jl:330 | same | `V_mat = U_denoised · Diagonal(Σ) · V₀'` | Traceable |
| rskr.jl:338 | same | `[reshape(V_mat[:,b], sz) for b in 1:nch]` | Reshape |

## Phantom / object / geometry

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| attenuation.jl:32-52 | `compute_μ_at_energy`, `compute_mass_μ_at_energy` | `XA.linear_attenuation_coeff(material, E·u"keV")` + `ustrip` → **Float64** | The single widest entry point into the μ tables; precompute tensors |
| attenuation.jl:79-108 | `μ_to_HU`, `HU_to_μ` | `1000(μ-μ_w)/μ_w` scalar and array forms in Float64 | Traceable |
| materials.jl:44-72 | material registry | `Dict` lookup by `Symbol` with an `XA.Materials` fallback | Host |
| phantom.jl:262-269 | `Phantom` ctor | `origin === nothing ? (computed) : Float64.(origin)` | Host |
| phantom.jl:278-280 | same | **`maximum(labeled_array)` device→host sync**; `max_label > typemax(UInt8) ? UInt16 : UInt8`; `eltype === U ? mask : U.(mask)` — **element type chosen from data** | Host; fix the mask type |
| phantom.jl:339-350 | `create_gammex_472` | `n_slices !== nothing ? … : max(1, round(Int, n_voxels·z_cm/fov_cm))` — **shape from a float ratio** | Host |
| phantom.jl:377-419 | same | triple host loop; `if r <= body_radius`; two insert loops each with `break`; `if mask == SOLID_WATER` guard | Host; the phantom is a constant input |
| phantom.jl:545-549 | `compact_materials` | `Array(phantom.mask)` device→host; **`sort!(Int.(unique(mask_cpu)))`**; `isempty && return`; `throw(ArgumentError)` | Host preprocessing step |
| phantom.jl:553 | same | `labels == collect(0:(n-1)) && return phantom` identity fast path | Host |
| phantom.jl:555-563 | same | `U = length(labels)-1 > typemax(UInt8) ? UInt16 : UInt8` — **type from data**; remap table; per-element remap loop | Host |
| phantom.jl:564-571 | same | `phantom.mask isa Array ? … : (similar + copyto!)`; `phantom.materials[labels .+ 1]` fancy index | Host |
| phantom.jl:578-592 | `build_materials_vector` | **`maximum(keys(materials_dict))`** sizes the vector; `mat isa Symbol ? get_material(mat) : mat` | Host |
| affine.jl:144-170 | `resample_to_recon` `:nearest` | triple loop; `round(Int, pzi)+1` ×3; `(pk < 1 \|\| pk > pnz) && continue` ×3 | Precomputed gather + mask |
| affine.jl:172-229 | same `:linear` | triple loop; `floor(Int, ...)` ×3; `continue` ×3; **6 `clamp`s + 8 scalar gathers** trilinear | `gather` with a fixed index tensor |
| scanner.jl:304 | scanner ctor | `error("energy_thresholds must be sorted ascending")` | Host |
| scanner.jl:493 | `_rows_for_support` | `ceil(Int, support_mm / detector_row_size)` | Host |
| scanner.jl:600-632 | collimation resolution | `round(Int, collimation_mm/row_size)`; a boxed `error` banner when it exceeds the physical max; `nominal_rows` recompute | Host |
| scanner.jl:685 | geometry ctor | **`n_views_total = helical ? round(Int, n_angles·n_rotations) : n_angles`** — sinogram shape from float arithmetic | Host; freeze the shape |
| xcat_artifacts.jl:72 | `xcat_phantoms` | `sort!(collect(keys(XCAT_REGISTRY)))` | Host |
| xcat_artifacts.jl:151-165 | `download_xcat` | registry `Dict` lookup; **`try`** around the download | Host |
| xcat_artifacts.jl:196-199 | progress callback | `round(Int, 100·now/total)`; `print(stderr, ...)` | Host |
| xcat_artifacts.jl:280-304 | phantom assembly | `minimum(lo_c)` ×3; **`round(Int, maximum(lo + n - 1) - g0) + 1`** ×3 grid bounds; `round(Int, lo - g0)` ×3 offsets — **volume shape from float offsets** | Host |
| memory_budget.jl:85 | PCCT device budget | `floor(Int, _PCCT_DEVICE_SAFETY_FACTOR · available)` | Host |
| memory_budget.jl:199-207 | `suggest_tile_size` | 2 validation `error`s; `mem_budget_GB === nothing ? floor(Int, Sys.free_memory()·0.6) : floor(Int, GB·2^30)`; `clamp(avail ÷ per_view, 1, n_view)` | Host |
| memory_budget.jl:217-218 | `tile_ranges` | generator producing a **ragged final range** | Fix a uniform tile size |
| memory_budget.jl:228-250 | `with_oom_retry` | **`try`/`catch` on OOM, `_is_oom(e) \|\| rethrow()`, `GC.gc(true)`, halve `tile_size`, retry up to 4×** — the workspace shape is decided by whether an allocation failed | Fundamentally untraceable. Decide tile sizes before tracing |
| memory_budget.jl:254-255 | `_is_oom` | **`occursin("OutOfGPUMemoryError", string(typeof(e)))`** exception-type-name string matching | Host |

## Spectral / PCCT analysis utilities

| file:line | function | construct | Suggested fix |
|---|---|---|---|
| pcct_spectral.jl:366-368 | `compute_kedge_enhancement` | `haskey(K_EDGE_ENERGIES, element) \|\| error` | Host |
| pcct_spectral.jl:375-380 | same | **`findfirst(t -> t > k_edge, thresholds)`**; `if bin_above === nothing \|\| == 1` `@warn` + fallback | Host |
| pcct_spectral.jl:384-385 | same | `sino.bins[min(bin_above, n_bins)]` clamped bin selection | Host |
| pcct_spectral.jl:387-398 | same | `if method == :subtraction / elseif :ratio / else error`; `AK.foreachindex` ×2 | Broadcast |
| pcct_spectral.jl:420-444 | `get_kedge_sensitivity` | `findfirst`; `bracketed` flag; `min(dist_above, dist_below)` else `Inf`; **4-way `if` producing a `Symbol` rating** | Host |
| pcct_spectral.jl:509 | `compute_effective_z` | `compute_pcct_bin_energies(...)` | Host |
| pcct_spectral.jl:514-544 | same `:dual_ratio` | `AK.foreachindex`; `if μ_high < T(1e-10)` → constant `Z_water`; **`(R/R_water)^α`** fractional power | Guard the power's base |
| pcct_spectral.jl:546-574 | same `:fit` | `AK.foreachindex`; **`μ_vals = [sino.bins[i][idx] for i in 1:n_bins]` allocates a `Vector` inside the kernel**; `log.` ×4; linear regression; `clamp(Z_eff, 1, 100)` | Already a GPU hazard today; stack bins into a 4-D array and reduce |
| pcct_spectral.jl:576 | same | `else error("Unknown effective Z method")` | Host |
| pcct_spectral.jl:588-597 | `get_supported_kedge_elements` | `push!` into a `Vector{Symbol}` over a `Dict` iteration | Host |
| pcct_spectral.jl:610-618 | `synthesize_vmi` | `@assert n_mats > 0`; `ws_μ_values !== nothing ? … : Vector{T}(undef, n_mats)`; `output === nothing ? similar(...) : output` | Host |
| pcct_spectral.jl:620-633 | same | `if n_mats == 2` → `spectral_vmi!` else `fill!` + host material loop with `AK.foreachindex` per material; `result[idx] += mat_i[idx]·μ_i` | `sum` over a stacked material axis |

---

# Compute hot spots: all 92 kernel launches

There are **no** hand-written `@kernel` macros, no KernelAbstractions kernels, and no `@metal` /
`@cuda` launches anywhere in `src/`. Every device kernel is `AK.foreachindex`, dispatched by array
type. That is a real asset: one rewrite target, one calling convention. The only literal
`MtlArray(...)` construction in the tree is inside a docstring example (`object/phantom.jl:126`).

| file | lines with `AK.foreachindex` | count |
|---|---|---|
| `api/driver.jl` | 72, 205, 211, 299, 357, 363, 542, 557, 572, 587, 596, 607, 618, 637, 1300, 1317, 1494, 1508, 1523, 1568, 1616 | 21 |
| `projection/polychromatic.jl` | 232, 259, 568, 576, 633, 644, 653 | 7 |
| `detector/photon_counting.jl` | 285, 562, 576, 594, 699, 725 | 6 |
| `detector/scatter.jl` | 162, 194, 324, 370, 738, 761 | 6 |
| `spectral/pcct_spectral.jl` | 388, 393, 531, 551, 629 | 5 |
| `projection/dd.jl` | 429, 479, 586, 713 | 4 |
| `reconstruction/ir/utils.jl` | 77, 131, 205, 238 | 4 |
| `correction/bhc_sinogram.jl` | 329, 738, 758, 779 | 4 |
| `projection/siddon.jl` | 535, 880, 1269 | 3 |
| `reconstruction/core/backprojection.jl` | 614, 640, 667 | 3 |
| `reconstruction/vmi/pwls.jl` | 112, 125, 320 | 3 |
| `projection/dd_fast.jl` | 112, 207 | 2 |
| `projection/dd_transpose.jl` | 98, 336 | 2 |
| `reconstruction/core/filtering.jl` | 408, 514 | 2 |
| `reconstruction/vmi/rwls.jl` | 161, 269 | 2 |
| `reconstruction/fbp/wfbp_helical.jl` | 89, 171 | 2 |
| `correction/bhc_image_domain.jl` | 104, 134 | 2 |
| `denoising/rskr.jl` | 82, 153 | 2 |
| `detector/detector_lag.jl` | 126, 134 | 2 |
| `reconstruction/fbp/fdk.jl` | 476 | 1 |
| `reconstruction/vmi/cong.jl` | 195 | 1 |
| `reconstruction/vmi/cmv.jl` | 80 | 1 |
| `correction/calibration.jl` | 43 | 1 |
| `correction/pcct_pileup_correction.jl` | 105 | 1 |
| `detector/optical_crosstalk.jl` | 90 | 1 |
| `detector/fill_factor.jl` | 100 | 1 |
| `source/focal_spot.jl` | 443 | 1 |
| `source/heel_effect.jl` | 188 | 1 |
| **Total** | | **92** |

The four kernels that dominate runtime and must become array-level or `@jit`-able first:

1. `dd_fast.jl:112` / `dd_fast.jl:207` — the fused DD projector (the default `:dd_fast` path).
2. `dd_transpose.jl:98` / `dd_transpose.jl:336` — the exact DD adjoint.
3. `backprojection.jl:614` / `640` / `667` — FDK / matched backprojection.
4. `cong.jl:195` — the per-ray root-finding VMI decomposition.

Secondary but still hot: `filtering.jl:514` (ramp convolution), `siddon.jl:880` / `1269`
(the retained Siddon fused paths), and `rskr.jl:82` / `153` (the 3-D joint bilateral).

---

# RNG inventory

All randomness is CPU-side. There is no `CUDA.rand`, no Metal RNG, and no device-side RNG anywhere
in `src/`. The PCCT noise path explicitly stages GPU→CPU→GPU to get it.

| file:line | construct | Notes |
|---|---|---|
| `detector/photon_counting.jl:836-841` | `Random.seed!(ws_rng, isnothing(seed) ? 0 : seed)` / `MersenneTwister(seed)` / `Random.default_rng()` | PCCT noise entry point |
| `detector/photon_counting.jl:855` | `_poisson_sample(rng, λ)` per element | **Exact integer sampling** |
| `detector/photon_counting.jl:903` | `p *= rand(rng)` in the Knuth inversion `while true` | λ < 10 branch |
| `detector/photon_counting.jl:915-916` | `U = rand(rng) - 0.5`; `V = rand(rng)` in the PTRS rejection `while true` | λ ≥ 10 branch |
| `api/driver.jl:527-528` | `if sim_opts.seed !== nothing; Random.seed!(ws.rng, sim_opts.seed)` | EICT reproducibility |
| `api/driver.jl:530` | `randn!(ws.rng, ws.noise_rand_cpu)` | Quantum noise, CPU staging |
| `api/driver.jl:536` | `randn!(ws.rng, ws.enoise_rand_cpu)` | Electronic noise, CPU staging |
| `api/driver.jl:736` | `isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed + 7919)` | HU-domain noise floor |
| `api/driver.jl:737` | `vol .+= T(sigma_hu) .* randn(rng, T, size(vol))` | Allocates a full-volume draw |
| `api/workspace.jl:215` | `rng = MersenneTwister(0)` stored in `PCCTWorkspace` | Re-seeded per call |
| `api/workspace.jl:743` | `rng = MersenneTwister(0)` stored in `EICTWorkspace` | Re-seeded per call |
| `api/workspace.jl:56, 487` | `rng::MersenneTwister` workspace field ×2 | Struct fields |
| `detector/pcct/mc_pileup.jl:107` | `rng::AbstractRNG = Random.default_rng()` default argument | |
| `detector/pcct/mc_pileup.jl:142` | `arrival_times[i] = rand(rng) * T_obs` | Uniform arrival times |
| `detector/pcct/mc_pileup.jl:149` | `u = rand(rng)` then `searchsortedfirst(cdf, u)` | Inverse-CDF energy sampling |
| `detector/pcct/mc_pileup.jl:177` | `if rand(rng) < f_retrigger` | Bernoulli retrigger |
| `detector/pcct/mc_pileup.jl:260` | `p *= rand()` — **global unseeded RNG** inside `Poisson_approx` | Reproducibility bug independent of AD |
| `detector/pcct/mc_pileup.jl:267` | `max(0, round(Int, λ + sqrt(λ) * randn()))` — **global unseeded RNG** | Same |
| `detector/pcct/mc_pileup.jl:336` | `rng = MersenneTwister(seed)` for the MC pile-up matrix | Host precompute, run once |
| `denoising/sino_sfjsd.jl:250-251` | `rng = MersenneTwister(42)`; `b = randn(rng, Float32, size(M))` | Hutchinson divergence probe in SURE |
| `api/options.jl:49, 87, 129, 198` | `seed::Union{Int, Nothing} = 42` plumbing through `SimOptions` | Config |
| `detector/physics_pipeline.jl:33, 44, 64, 75` | `noise_seed::Union{Nothing, Int}` plumbing | Config |
| `detector/photon_counting.jl:95, 142, 163, 189, 197` | `seed::Union{Nothing, Int}` plumbing on the detector struct | Config |

**Good news for Enzyme:** the EICT noise kernel (`driver.jl:542-566`) is *already* in the
reparameterized form `λ_noisy = λ + √max(λ,1)·ε + σ_e·ε₂`. Passing `ε` in as a tensor makes that
whole kernel differentiable with no algorithm change. Only the PCCT integer-Poisson path
(`photon_counting.jl:894`) and the pile-up MC (`mc_pileup.jl:250`) need genuine replacement.

---

# Float64 inventory

`Float64` appears **596 times** across 50 files. Per-file counts (descending):

| file | count | file | count |
|---|---|---|---|
| `geometry/scanner.jl` | 55 | `reconstruction/vmi/basis.jl` | 8 |
| `correction/bhc_sinogram.jl` | 51 | `detector/detector_lag.jl` | 7 |
| `detector/pcct/mc_response.jl` | 50 | `reconstruction/vmi/roots_kernels.jl` | 6 |
| `detector/photon_counting.jl` | 44 | `reconstruction/vmi/clinical_calibrations.jl` | 6 |
| `api/workspace.jl` | 41 | `reconstruction/fbp/fdk.jl` | 6 |
| `source/spectrum.jl` | 31 | `projection/dd.jl` | 5 |
| `api/driver.jl` | 29 | `reconstruction/vmi/rwls.jl` | 4 |
| `detector/detector_efficiency.jl` | 28 | `reconstruction/core/filtering.jl` | 4 |
| `detector/scatter.jl` | 25 | `projection/siddon.jl` | 4 |
| `detector/pcct/mc_pileup.jl` | 25 | `projection/polychromatic.jl` | 4 |
| `reconstruction/vmi/mono_plus.jl` | 24 | `phantoms/xcat_artifacts.jl` | 4 |
| `source/protocol.jl` | 20 | `geometry/affine.jl` | 4 |
| `spectral/pcct_spectral.jl` | 16 | `detector/optical_crosstalk.jl` | 4 |
| `correction/radial_cupping.jl` | 16 | `detector/fill_factor.jl` | 4 |
| `source/focal_spot.jl` | 15 | `reconstruction/vmi/pwls.jl` | 3 |
| `reconstruction/vmi/vmi_synth.jl` | 14 | `reconstruction/workspace/memory_budget.jl` | 2 |
| `object/phantom.jl` | 14 | `projection/dd_transpose.jl` | 2 |
| `object/attenuation.jl` | 14 | `projection/dd_fast.jl` | 2 |
| `denoising/acnr.jl` | 12 | `detector/physics_pipeline.jl` | 2 |
| `denoising/sino_sfjsd.jl` | 11 | `reconstruction/fbp/wfbp_helical.jl` | 1 |
| `source/bowtie_filter.jl` | 10 | `denoising/sino_svd.jl` | 1 |
| `reconstruction/vmi/pcct_calibration.jl` | 10 | | |
| `correction/radial_capping_basis.jl` | 10 | | |
| `reconstruction/vmi/phantom_mask.jl` | 9 | | |
| `reconstruction/vmi/image_domain_decomp.jl` | 9 | | |
| `api/options.jl` | 9 | | |
| `source/heel_effect.jl` | 8 | | |
| `reconstruction/vmi/pcct_basis.jl` | 8 | | |
| `reconstruction/vmi/cmv.jl` | 8 | | |

Most are host-side spectral tables and are harmless. These are the ones that would actually fight a
Float32 trace:

| file:line | construct | Why it matters |
|---|---|---|
| `denoising/acnr.jl:78, 101-102` | `s_orth` built as Float64; per-slice `FFTW.fft`/`ifft` in **`Complex{Float64}`** | The whole ACNR smoother runs at double precision, then narrows back at line 108 |
| `reconstruction/vmi/mono_plus.jl:213-214, 404-405` | `slice = Float64.(@view img[:,:,k])`; FFT; `Float32.(real.(...))` | Same round trip, per slice, per energy |
| `reconstruction/vmi/mono_plus.jl:198-199, 391-392, 205, 398` | frequency grids and Gaussian kernels built in Float64, cached in a **`Dict{Float64, Matrix}`** | The cache key is a Float64 σ |
| `reconstruction/vmi/phantom_mask.jl:127-130` | `fx`/`fy` in Float64; `FFTW.fft(Float64.(mask2d))` erosion | Then hard-thresholded at 0.999 |
| `reconstruction/core/filtering.jl:259, 266, 281` | `Complex{T}` FFT for the CatSim window — `T` follows the kernel, but `_catsim_apodization_window` interpolates Float64 control points | Host |
| `reconstruction/vmi/pcct_calibration.jl:185-190` | `pl = Float64(sino_low[idx])`; `ph = Float64(...)`; `pl^i * ph^j`; then `T(max(aw, 0.0))` | **Per-element widening in the hot polynomial apply** |
| `reconstruction/vmi/cong.jl:184` | `Float32.(Array(p_L)) .* a_w .+ Float32.(Array(q_L)) .* c_w` then `minimum` | Device→host, mixed width |
| `reconstruction/vmi/cmv.jl:50-78` | `Float64.(Array(ŵ))` ×2; `Float64.(Array(basis.X))` ×4; all four moments and the determinant in Float64; then `Float32(...)` ×5 | Full round trip for a 2×2 solve |
| `reconstruction/vmi/rwls.jl:410-413` | `ŵ64 = Float64.(basis.ŵ_bins[b])`; normalize; `Float32.(ŵ64 ./ s)` | Staging widening |
| `reconstruction/vmi/pwls.jl:244` | `ŵ = Float32.(Array(ŵ_raw))` device→host normalize | Host round trip |
| `reconstruction/vmi/basis.jl:29-47` | `p_photoelectric` / `q_compton` return Float64; **`sqrt(32/ε^7)` ≈ 3e8** | Float32 would lose precision *and* risk overflow |
| `source/heel_effect.jl:182, 212` | `exp(clamp(x, T(-700), T(700)))` — a **Float64 exponent range** applied at type `T` | With `T = Float32` the clamp never binds and `exp` overflows to `Inf` at ~88 |
| `detector/detector_efficiency.jl:470-487, 596-607` | every η lookup takes and returns `Float64`; `compute_eid_efficiency_vector` builds a `Vector{Float64}`; `log.(energies)`/`log.(mus)` allocated per call | Narrowed later at the kernel boundary |
| `detector/pcct/mc_response.jl` (50 sites) | the entire DRM / covariance pipeline in Float64 (`zeros(Float64, ...)`, `Diagonal`, `transpose`, `Matrix{Float64}(I, ...)`) | Host precompute — fine, but it is the widest surface |
| `detector/pcct/mc_pileup.jl` (25 sites) | `Vector{Float64}` arrival times and energies (dynamic length), `Float64` transition counts | Host MC |
| `source/bowtie_filter.jl:361-392` | `μ_matrix = zeros(n_materials, n_energies)`; **`transmission = zeros(Float64, n_cols, n_rows, n_energies)`** | The full `[n_col, n_row, n_E]` bowtie tensor is built in Float64 before narrowing |
| `correction/bhc_sinogram.jl` (51 sites) | calibration curves; `w_norm_per_col::Matrix{Float64}`; `μ_water_E::Vector{Float64}`; `energies::Vector{Float64}` struct fields | Host |
| `correction/radial_cupping.jl:44-71`, `radial_capping_basis.jl:60-103` | all fitting in Float64 with `A \ vals` and `quantile` | Host |
| `denoising/sino_sfjsd.jl:81, 137-139, 162, 252, 306-348` | σ handling, MAD, and correlation length computed in Float64 | Mixed with Float32 arrays |
| `denoising/rskr.jl:53` | `Float32(1.4826) * Float32(median(absvec))` | Mixed |
| `api/driver.jl:48-56` | `compute_detector_I0` entirely Float64, with mm conversions | Host |
| `object/attenuation.jl:32-52` | every `compute_μ_at_energy` / `compute_mass_μ_at_energy` returns Float64 via a Unitful `ustrip` | **The single widest entry point into the μ tables** |
| `geometry/scanner.jl` (55 sites) | geometry stored in Float64; `NTuple{3,Float64}` FOV threaded through **every** projector signature | See below |

## The `NTuple{3,Float64}` volume-extent thread

The `volume_extent::Union{Nothing, NTuple{3, Float64}}` kwarg appears in nine projector signatures:

- `siddon.jl:467, 681, 795, 1184`
- `dd.jl:406, 454, 514, 547, 672`
- `dd_fast.jl:317, 391`
- `dd_transpose.jl:286` (plus `bounds::NTuple{3,Float64}` at `dd_transpose.jl:71`)
- `polychromatic.jl:445`

Each call site does `vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov`, then
`T(vol_bounds[i]/2)` at kernel entry. Two separate problems:

1. The `Nothing` half of the union forces a **host-side branch at every projector entry**, and the
   same `Union{Nothing, ...}` pattern governs roughly forty workspace kwargs across the driver and
   workspace constructors.
2. The Float64 tuple is the source of the voxel-size and volume-bound constants that every kernel
   narrows to `T`. Resolving both to concrete typed values before tracing is a prerequisite,
   independent of the numeric-width question.
