# The bowtie across the photon-counting fan: per-ray I0

**Defect.** The PCCT path builds one spectrum `W[e, b] = I0·w[e]·η[e]·R[e, b]` for every ray and folds
the bowtie in at the fan centre only (`workspace.jl`, "fold center-pixel bowtie into W"). Every
ray therefore gets the central flux: measured on the semmd NAEOTOM model, air-scan counts at the
fan edge / centre = 1.00 where the bowtie's transmission is 0.10. Peripheral rays carry up to 10×
too many photons, peripheral noise is too low, the decomposition's per-ray statistics are wrong,
and the dose report describes a bowtie the beam does not have. The EICT path applies the bowtie
per ray (`ws.bowtie_spectral`, `bt[col, row, e]`, consumed by the fused spectral kernel) — the
kernel hook already exists; the PCCT path never passes it. Test: `test/pcct_bowtie.jl`.

**Design.** One representation change, no second path:

- `I0` is per ray: `I0[col, row, b] = Σ_e W[e, b] · bt[col, row, e]`, the kernel's own air response
  with the bowtie, on the backend. The bins stay log-transmissions against their own ray's air
  response, so `p_air ≡ 0` holds per ray (the existing air-calibration invariant).
- The fused spectral projector receives `ws_bowtie_spectral = bt` (as EICT does); its air output
  is `I0[col, row, b]` by construction, which is what normalises it.
- Every consumer that formed counts as `I0_b · exp(-p)` uses `I0[col, row, b]`: scatter's combined
  primary and `inject_scatter_bins!` (scatter counts scale with the local total flux, not a global
  `I0_total`), `apply_pcct_noise!`, the pile-up mixing and `apply_pcct_pileup_correction!`,
  `_capture_pcct_raw_counts`, `combine_pcct_bin_counts!` (the VMI chain's bin grouping).
- `simulate!` returns `I0_bins` as the per-ray array `[n_cols, n_rows, n_bins]`; downstream
  (semmd `hypr_lr`, `spectral_basis`) index it per ray. The name is kept, the shape is documented;
  a scalar-per-bin `I0` no longer exists anywhere.
- A scanner with `bowtie_filter = :none` gives a flat `bt ≡ 1`, so its `I0[col, row, b]` is constant
  across the fan and every result is bit-identical to today (the test pins this).
- The bowtie table is built once per workspace (it depends on geometry and spectrum only).

**Sound from all angles — what the tests pin.** Air-scan fan profile = the bowtie's transmission
per column (3 %); `p_air ≡ 0` in every column and bin; `:none` flat; noise σ per ray ∝ 1/√I0(col)
(a peripheral ray at 0.1 flux has √10 the relative noise); pile-up correction reproduces the
noiseless counts per ray with the bowtie present; scatter injection conserves the per-ray
counts; the semmd chain's water bias with HYPR-LR unchanged at the fan centre.

**Pile-up must follow the ray's own count rate.** `_pileup_S` is one Monte-Carlo migration matrix
computed at the air count rate of the fan centre (`workspace.jl`, `_count_rate_per_dexel`) and
applied as the same linear mixing to every ray. Pile-up is a count-rate effect: a ray through
35 cm of patient carries ~1/1000 of the air flux and piles up ~1/1000 as much, and with the
bowtie the air rate itself falls 10× across the fan. The sound model: compute `S(rate)` on a
log-spaced grid of count rates once per workspace (the MC is ~1 s per rate; 12–16 rates) and
apply, per ray, the matrix for that ray's truth count rate (linear interpolation in log rate);
the correction inverts the same per-ray matrix. Test: pile-up loss on an air ray at the fan
centre equals the MC value at the air rate; on a ray at 1 % of the air flux it is ~1 % of that;
at the fan edge with the bowtie it is that of 0.1× the air rate; correction reproduces the
noiseless counts per ray.

**Implementation order (each site + its test before the next):**
1. `workspace.jl`: PCCT bowtie table `bt[col,row,e]` (as EICT), `I0` per ray `[cols, rows, bins]`
   on the backend from `Σ_e W[e,b]·bt[col,row,e]` (native path: binned from the native table),
   remove the centre fold; `S` on a rate grid.
2. `photon_counting.jl` `pcct_forward_project`: pass `bt` to the fused kernel; normalise by the
   per-ray `I0`.
3. `driver.jl` `simulate!`: scatter combine / inject, noise, pile-up mixing (per-ray `S`), raw
   capture, correction — per ray.  `detector/scatter.jl inject_scatter_bins!`,
   `detector/photon_counting.jl apply_pcct_noise!`, `correction/pcct_pileup_correction.jl`,
   `reconstruction/vmi/pcct_basis.jl combine_pcct_bin_counts!`: per-ray `I0`.
4. Return `I0_bins` as the per-ray array; semmd `hypr_lr(channels, I0)` and
   `spectral_basis(ws; I0_bins)` index per ray.
5. Tests: `test/pcct_bowtie.jl` (flux profile, air ≡ 0, `:none` flat), noise ∝ 1/√I0(col),
   pile-up per rate, correction round trip, scatter conservation; `:none` + no pile-up must be
   bit-identical to main.
