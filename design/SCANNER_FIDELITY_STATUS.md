# fix/scanner-fidelity — status (2026-09-21, end of session 1)

Branch of `main` (dac1380, v0.17.2). One PR when done; multiple adversarial rounds; failing test
first for every item. Sister spec: `semmd-bayesian/docs/scanners/` (source-cited scanner models).

## Committed (54ebba2, b1d4411) — full suite 3559/3559 at 54ebba2
1. Quarter-detector offset honoured everywhere (`column_offset`, `column_center`), tests
   `test/detector_offset.jl` (32) incl. label/affine alignment.
2. HIR support extends in plane to the scan circle (`_hir_support`), tests `test/hir_support.jl` (11).
3. `test/pcct_bowtie.jl` + `design/PCCT_BOWTIE_PER_RAY.md` (failing test and design).

## Round-1 adversarial review of 1–2 (all to fix before merge)
- C1 warm start (`init_volume`) leaves the in-plane ring unwritten → +32.8 HU edge; a reused
  workspace keeps the previous body in the ring (276 HU). Fix: FDK-seed the ring from the
  sinogram (or zero it) in `_hir_seed_work!`; fix the driver.jl:1470 docstring.
- C2 every downstream scanner sets `detector_col_offset = col_iso / 2` in MILLIMETRES (semmd
  scan.jl:49; docs/notebooks/04, 08; test/memory_stability_pcct_representative.jl:48): now a
  0.15-column offset. Fix those to 0.25 (columns) in the same commit; tell semmd.
- C3 no version bump / CHANGELOG → semmd's reuse guard would not remake the cohort. Needs 0.18.0
  and a CHANGELOG entry (offset live by default; HIR work grid to the scan circle, ~2.5× HIR
  time, work buffers ~2.5×; CTGeometry 18th field; new exports); docs/src/routes/api/index.jl:149.
- C4 HIR 2.5× slower on the semmd protocols (714²×56 EICT, 712²×91 PCCT work grids); record
  a GPU A/B of one rod set.
- P1 HIR's data term is nearly inert at strength 60 (gain 0.6 % per sub-iteration on the wide
  grid); "PWLS" in the strength docs does not describe what runs — pre-existing, document.
- P2 nz == 1 and helical still skip the in-plane support (single-row `reconstruct_basis_image`).
- P3 heel_effect.jl centred and on the wrong axis (see below); `detector_row_offset` stored, never
  read; workspace.jl centre_col = n ÷ 2 (0.75 col off, negligible).
- P4 PCCT native-grid offset rescale: use `bf`, not the pixel ratio.
- P5 `scan_circle_diameter` for a flat panel over-estimates (use SAD·h/√(SAD²+h²)).
- P6 hir_support.jl's `reference()` pads to the same work grid as the request → cannot catch a
  gain bug; nothing exercises the offset through siddon/dd_fast/rowtile4/subset geometries
  (the reviewer's adjoint.jl does — fold it into the tests).

## In progress (UNCOMMITTED WIP in the working tree — commit as WIP at session end)
Per-ray `I0` refactor (design/PCCT_BOWTIE_PER_RAY.md): workspace.jl (I0 [cols,rows,bins], I0_all,
I0_cpu, I0_native, bowtie_spectral, native_bowtie_spectral, pileup_S on a 16-point log rate grid,
pileup_rates, pileup_rate_air), photon_counting.jl (`_normalize_bins_per_ray!`, per-ray source
transmission on the per-energy path, `apply_pcct_noise!(sino, I0)`), driver.jl (every count step
per ray, `apply_pcct_pileup!`), pcct_pileup_correction.jl (`apply_pcct_pileup!`,
`apply_pcct_pileup_correction!(bins, I0, S, rates, rate_air)`), scatter.jl
(`inject_scatter_bins!(bins, field, I0, I0_all, w)`), pcct_basis.jl (`combine_pcct_bin_counts!`
per ray, returns per-group matrices), nchannel.jl (`spectral_basis(ws)` ray-resolved through
`ws.bowtie_spectral`).
- `test/pcct_bowtie.jl` bowtie testset: 14/14 GREEN. Noise per ray: 3/3. Scatter: 4/4.
- Pile-up testset FAILS on the TEST's regime: 100 mA / 60 views puts the centre at 99.5 % loss
  (rate·τ ≈ 2.4), so both edge and centre saturate and the correction (near-singular S) blows up.
  Fix the test: choose dead time so rate_air·τ ≈ 0.1 (build ws once with τ = 1 ns, read
  `ws.pileup_rate_air`, rebuild with τ = 0.1 / rate_air). Then verify loss_edge ≈ 0.1× centre
  and the correction round trip (< 5e-3).
- Still to update for the new signatures: test/correction.jl (apply_pcct_pileup_correction!),
  test/api.jl:326 (_capture_pcct_raw_counts), any test using `result.I0_bins[b]` or
  `combine_pcct_bin_counts(…, I0_bins::Vector, …)`, docs notebooks 04/08; semmd `hypr_lr(channels,
  I0_bins)` → per ray `I0[col,row,b]` (hypr.jl uses `I0_k·exp(-h_k)` per row: index per column
  too) and `scan_pcct` (`Array(result.I0_bins)` is now 3-D), `spectral_basis(ws; I0_bins=…)` kw
  renamed `I0`.

## Found on the way, not yet started
- Item 10: heel effect modelled along the FAN (heel_effect.jl:310 `θ_eff = θ_anode + γ`), 5×
  across 48 cm; in a CT the anode axis is z → gradient along rows, tens of percent across the
  cone. On by default. Failing test first.
- Pile-up was one S at the central air rate applied to every ray (now rate-grid per ray in WIP).

## Remaining items of the PR (from the reviews and the scanner specs)
FBP window: keep the grid cut (B) and calibrate per-scanner kernels to measured MTF (Br36f f50
2.84 / f10 4.68; Revolution STANDARD f50 3.0–3.25 lp/cm; control points in basis-verification);
EICT electronic noise floor (real σ 1.5× over 6× mAs vs 2.2× simulated); fitted bowtie profiles per
scanner, explicit, never inherited; GPU FFT via AbstractFFTs (acnr.jl:102, nchannel.jl:874,
mono_plus.jl, phantom_mask.jl); resampler backend/field agreement check, no Float64 on Metal;
the vacuous resampler test and the loosened tolerances in test/api.jl, test/nchannel.jl;
CHANGELOG/API doc corrections; calibrate both scanner models to the Gammex scans (dose/mAs,
effective energy 76.0 keV at 140 kVp, rod HU, noise-vs-dose, Alpha thresholds against the
106 HU VMI/0 noise); then semmd scan.jl to the calibrated models, cohort remake, one registration.
