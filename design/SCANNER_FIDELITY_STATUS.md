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

## Done since (committed on the branch, 5211c7a … )
- Per-ray `I0` refactor complete across the PCCT path; bowtie / noise / scatter / pile-up tests
  green (`test/pcct_bowtie.jl`, 29 assertions). Pile-up grid LINEAR in rate (S(0) = I, MC at
  ¼ ½ ¾ 1 × air rate): the measured loss is linear in rate·τ (0.092 at 0.10), log spacing was
  wrong and 16× the cost.
- Round-1 C1 (warm start seeds the whole support), C2 (offsets in columns in notebooks/test),
  C3 (0.18.0 + CHANGELOG), P4 (`bf` rescale), P5 (flat scan circle), P6 (adjoint / point-object
  tests through dd, ddᵀ, rowtile4, siddon at 4 offsets × 2 shapes; HIR reference at a request
  larger than the scan circle so it pads nothing), P2 (single-slice request gets the in-plane
  extension; helical still excluded and documented).
- Heel effect on the row axis (item 10): `heel_effect.jl` both routines, anode on +z, test
  `test/heel_axis.jl` (flat along the fan, monotonic along rows, few % at 15 mm, tens of % at 160 mm).
- semmd: `scan.jl` offset 0.25 columns; `hypr_lr` / `spectral_basis` / poly image on per-ray I0;
  `test/hypr.jl`; Project/Manifest on the branch (0.18.0).
- Full suite at 5beb27b+: 3580 pass, 9 test-side indexing mistakes fixed after.

## Round-2 adversarial review (commits 54ebba2..76c0320) — findings and what was done
- C1 **binned path: per-ray I0 was bf² × physical** (noise at 4× the photons on the NAEOTOM model).
  Fixed: the kernel's matrix is per native dexel (`W / bf²`), the basis multiplies back and uses
  the MEAN native transmission; test asserts the binned air count is below the incident photons
  and equals the sum of its dexels. (My earlier "bf²" basis fix had hard-coded the defect.)
- C2 `spectral_basis(ws)` threw for bf > 1 with a bowtie — fixed by the same change.
- C3 the calibrated heel (1 µm) was unreachable: `simulate!` built `HeelEffect(angle, :tungsten,
  0.01, true)` directly — now `default_heel_effect(anode_angle_deg = angle)`.
- C4 GPU-gated api tests asserted the removed API — updated (pileup_S 3-D, `ws.I0`).
- C5 CHANGELOG/README/docs notebooks stale — updated (linear 5-point grid, heel axis, physical
  binned counts, `I0` keyword, per-ray `I0_bins` in notebooks 04/08).
- C6 the pile-up grid's top was the unfiltered central rate; now the brightest ray's incident
  rate (spectrum-weighted transmission from the same table), and every ray's rate is relative to
  the brightest ray's air counts (so heel-bright cathode rows are not clamped).
- C7 negative takeoff angles were clamped: an anode angle ≤ half-cone now throws
  (`_heel_geometry_valid`); the 160 mm test uses a 12° anode.
- Plausible, recorded in the CHANGELOG as known: scatter behind a bowtie scaled by the receiving
  ray's flux; `estimate_pcct_workspace_bytes` omits the new tables; the pile-up test is slow
  (two rate grids at the toy's high rate).
- Sound (measured by the reviewer): air ≡ I0 on all five projection paths with bowtie + heel +
  offset (≤ 8e-6); indexing column-major at all seven sites; S(0) = I continuous (2e-4 MC
  floor), linear blend error ≤ 3.5e-4; correction residual < 5e-4; semmd `hypr_lr`/`scan_pcct`
  correct.

## Found on the way, not yet started
(both done above)

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
