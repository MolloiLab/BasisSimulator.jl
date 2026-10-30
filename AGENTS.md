# AGENTS.md

How to work on BasisSimulator.jl: for contributors, and for coding agents, whose one instruction
file this is (there is no CLAUDE.md, and `.gitignore` keeps one out). Keep it true: a change that
makes a line here wrong fixes the line in the same pull request.

## What this is

A GPU-portable CT and photon-counting CT simulator in Julia: polychromatic forward projection
(distance-driven `:dd_fast`, or Siddon), energy-integrating and photon-counting detector physics,
FDK / helical WFBP / hybrid iterative reconstruction, dose (CTDIvol, DLP), and spectral imaging
(the K-channel profile-likelihood decomposition and VMI chain, `vmi_pipeline`, with the
`SpectralHYPR` denoiser). Every kernel runs on the CPU, CUDA, Metal, AMDGPU or oneAPI through
AcceleratedKernels.jl; the package never loads a GPU backend itself.

The public API is five structs — `EICTScanner` / `PCCTScanner` (over `ScannerGeometry`),
`CTProtocol`, `SimOptions`, `ReconOptions`, `Phantom` — and the functions around them:
`create_workspace` (either scanner family), `simulate!`, `reconstruct!`,
`dose_report`, `spectral_basis*`, `vmi_pipeline`.

## Layout

```
src/BasisSimulator.jl   module: includes and exports, in dependency order
src/api/                the five structs' options, workspaces, simulate! / reconstruct!
src/geometry/           ScannerGeometry, the scanner structs, CTGeometry, the recon-grid affine
src/source/             spectra, bowtie, heel effect, focal spot, protocol, dose
src/object/             materials, attenuation tables, Phantom
src/phantoms/           the XCAT artifact loaders
src/projection/         :dd_fast (default), dd, dd transpose, Siddon, polychromatic kernels
src/detector/           EICT physics chain, photon-counting (pcct/: MC response, pile-up), scatter
src/correction/         water BHC (sinogram and image domain), pile-up correction, radial cupping
src/reconstruction/     core (filtering, backprojection), fbp (FDK, WFBP), hybrid_ir, ir, vmi, workspace
src/denoising/          SpectralHYPR (hypr.jl), ACNR, T-LBF, sinogram SVD / SF-JSD, median-z, RSKR
src/spectrum/, src/bowtie/   data tables read at run time
test/                   runtests.jl includes one file per area; test/docs.jl checks the API docs
docs/                   the documentation site (below)
lib/BasisMonteCarlo/    BasisMonteCarlo.jl, the Geant4 detector and scatter Monte Carlo: its own
                        package (src/, test/, tables/) and notebooks (docs/notebooks/)
```

## Working on the code

- **Julia 1.12 or later** (`julia = "1.12"`; the bundled MC response is in 1.12's serialization
  format, which 1.11 cannot read). The suite passes on 1.12 and 1.13.
  `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
- **Tests:** `julia --project=. -t 8 -e 'using Pkg; Pkg.test()'` (about 6 minutes on 8 CPU threads).
  Tests that need a GPU skip without one ("Skipping … no GPU backend"); run the ones your change
  touches on a GPU and say so in the pull request. `test/nchannel_nb04_reference.jl` is a frozen
  oracle (the published estimator); never edit it.
- **Every exported name has a docstring**, and every source file that defines one is mapped to a
  section of the API page in `docs/src/api_sections.jl`; `test/docs.jl` enforces both. A new file
  or export therefore needs a docstring and, for a new file, a line in that map.
- **Numbers are the contract.** A change to what the simulator or a reconstruction returns is
  breaking unless the caller asked for it: it needs a test that pins the new numbers and a
  `Breaking` / `Changed` line in `CHANGELOG.md` saying what moved and by how much.
- **Never bump `version`** in a feature pull request, and add every user-visible change under
  `## [Unreleased]` in `CHANGELOG.md`. Releases are one pull request: see `RELEASING.md`. CI's
  `version` job fails if `version` is not the last registered version or a valid next one.
- **Backend-agnostic always:** kernels are `AK.foreachindex` / broadcasts over `AbstractArray`; no
  CUDA- or Metal-only code, no scalar indexing of device arrays, no `Array(...)` round trips in
  hot paths.

## API design rules

The API is a few structs (`Phantom`, `EICTScanner` / `PCCTScanner`, `CTProtocol`, `SimOptions`,
`ReconOptions`) and the functions that take them. Keep it that small.

- **One struct per concept, no presets.** A real scanner is a documented `const` instance of its
  struct (`SOMATOM_FORCE`, `NAEOTOM_ALPHA`, `src/geometry/clinical_scanners.jl`), every value
  sourced in its docstring. A variant is the copy constructor: `PCCTScanner(NAEOTOM_ALPHA;
  detector_rows = 96)`, `CTProtocol(p; mA = 200)`. Never a preset function, a preset or mode
  keyword, or a registry of presets.
- **Fields are single-source and specific.** Each field is one physical quantity in its own units
  (a dexel pitch, a binning factor, a threshold dispersion in keV), set in one place and read where
  it applies. Never a generic switch that implies other values, such as `mode = :standard` /
  `:uhr`: a mode is a combination of specific field values, and if it is common it is its own
  `const` (`NAEOTOM_ALPHA_UHR = PCCTScanner(NAEOTOM_ALPHA; …)`). A field nothing reads is removed.
- **No bloat.** New physics goes into the existing structs' fields, not parallel types, helper
  layers or option bags; an addition replaces something or is essential. Fields may be added and
  dropped across a breaking release when that makes the structs simpler.
- **Several acquisitions are several ordinary acquisitions.** Dual source (`SOMATOM_FORCE` +
  `SOMATOM_FORCE_B`, `SOMATOM_DEFINITION_FLASH` + `_B`) is one (scanner, protocol) pair per tube;
  rapid kVp switching is two protocols on one scanner, each with its share of the view period.
  `simulate!` runs once per acquisition. No dual-energy type or switch.
- **Defects and their corrections come in pairs.** A modelled detector imperfection is a field of
  the scanner (its per-dexel maps drawn reproducibly from a seed); its correction is a
  `calibrate_x(...)` → `apply_x(...)` pair (like `calibrate_bhc` → `apply_bhc!`) that sees only
  simulated calibration acquisitions a real scanner could make (air scans, water, flux sweeps),
  never the true defect. So corrections are realistic, not perfect, and each has a test of its
  residual against the ideal detector. The ideal detector is a copy with the defect set to zero.
- **BasisSimulator never depends on BasisMonteCarlo.** `lib/BasisMonteCarlo` (Geant4) produces
  tables and kernels; BasisSimulator reads only their plain data (arrays, numbers, strings), copied
  into its own `src/` or fetched as an artifact. No dependency, weak dependency or extension.

## The documentation site

A Therapy.jl static site: `docs/app.jl`, pages in `docs/src/routes/`, components in
`docs/src/components/`, Tailwind (`docs/input.css`, `docs/tailwind.config.js`).

- **API reference** (`/api/`) is generated from the docstrings at build time
  (`docs/src/routes/api/index.jl`, sections in `docs/src/api_sections.jl`). Fix the docstring, not
  the page.
- **Examples** are the Pluto notebooks in `docs/notebooks/*.jl` (plain Julia files; edit them
  directly — cells are `# ╔═╡ <uuid>` blocks and the `# ╔═╡ Cell order:` block at the end decides
  order and fold state). They are rendered on a GPU into `docs/notebooks-static/` (committed; the
  site ships these) and write their gallery figures into `docs/assets/`. CI never renders.
- **Build the site locally:** `docs/build.sh` (verifies the exports, instantiates the docs env,
  Tailwind, then `app.jl build` into `docs/dist/`), or
  `BASISSIM_SKIP_NB_EXPORT=1 julia --project=docs docs/app.jl build` to skip the export check while
  iterating on pages. Serve with `python3 -m http.server -d docs/dist` (islands need HTTP, not
  `file://`).
- **Render the notebooks:** `docs/render_notebooks.sh` re-renders every stale export, one exporter
  per GPU (NVIDIA: one lane per `nvidia-smi` GPU, or `BASISSIM_GPUS="0 1"`; Mac/CPU: one lane),
  then runs `docs/verify_notebook_exports.py` (`BASISSIM_LANES_PER_GPU=3` runs three per GPU on the lab's 96 GB cards; a lane that dies is retried once). `docs/render_notebooks.sh 04_pcct_vmi` renders the
  named ones. Logs: `docs/render-logs/`. On the lab's two-GPU node all 12 take about 17 minutes (about 15 per GPU).
- **Why an export goes stale:** its fingerprint hashes the notebook, all of `src/`, `Project.toml`
  (not its `version` line), both docs lockfiles, `docs/extract_all.jl` and
  `docs/notebooks/DATA_PROVENANCE.sha256`. Any code change makes every export stale; that is by
  design: the docs deploy refuses stale exports, so re-render after a code change (in that pull
  request or a follow-up). The site deploys from `main`; no release is needed. (The
  BasisMonteCarlo notebooks have their own fingerprint: next item.)
- **BasisMonteCarlo notebooks** (`lib/BasisMonteCarlo/docs/notebooks/<stem>.jl`) are the
  Examples' "Detector physics (BasisMonteCarlo.jl)" category, slug `mc-<stem>`
  (`/examples/mc-04_siemens_force/`). They run on the CPU in their own environment,
  `lib/BasisMonteCarlo/docs/` (each starts 8 Distributed workers), and `docs/extract_montecarlo.jl`
  renders them into the same `docs/notebooks-static/`, saving each one's first figure as its card,
  `docs/assets/mc-<stem>.webp`. Render them by slug: `docs/render_notebooks.sh mc-04_siemens_force`
  (5–15 min each; `mc-10_scatter_library` about 3.6 h). With no arguments the script renders every
  stale notebook of both kinds, these on a CPU lane beside the GPU lanes (`BASISSIM_MC_LANES=2` for
  two); `BASISSIM_SKIP_NOTEBOOKS=mc-10_scatter_library` leaves that one out. A render reruns the
  simulations, so it rewrites what the notebook saves (`lib/BasisMonteCarlo/tables/`). Their
  fingerprint hashes the notebook, `lib/BasisMonteCarlo/src/`, `lib/BasisMonteCarlo/Project.toml`
  (not its `version` line), `lib/BasisMonteCarlo/docs/Project.toml` and its committed
  `Manifest.toml`, `lib/BasisMonteCarlo/docs/reference_data/`, and the exporter
  (`docs/build_env/Manifest.toml`, `docs/extract_all.jl`, `docs/extract_montecarlo.jl`); not
  `docs/Manifest.toml` or the root `src/`, so a BasisSimulator change does not re-render them.
  Because `docs/extract_all.jl` is in every BasisSimulator export's fingerprint, a change that only
  concerns these notebooks goes in `docs/extract_montecarlo.jl`. The kV sliders of 04–08 are forced
  fallbacks (`MC_FORCE_FALLBACK_BONDS`, mirrored in the verifier). A notebook not rendered yet has
  no page and an unlinked card, and the verifier (so `docs/build.sh` and the deploy) fails until it
  is rendered; `BASISSIM_SKIP_NOTEBOOKS='mc-*' docs/build.sh` builds without them (a trailing `*`
  matches a prefix, in the verifier and the site).
- **Data:** notebook 05 reads the XCAT male chest export from `BASISSIM_XCAT_DIR`
  (the lab's copy: `/share/crsp/lab/symolloi/share/XCAT Anatomies/vmale_50/vmale_50_chest_with_plaque`);
  without it the notebook renders its no-data notice. `DATA_PROVENANCE.sha256` holds the checksum
  of every file a render reads, and a render refuses a changed file: after an intended change to
  the data, update its line. Notebook 02 downloads its XCAT slab as a Julia artifact; every other
  notebook is self-contained (07/08 build the QRM-Thorax phantom analytically).
- **Snapshot islands:** the exporter (Snapshot.jl, `docs/build_env/`) compiles a `@bind` group to
  WebAssembly when everything reactive downstream of the slider is plain Julia arithmetic over
  scalars and tuples (upstream scalars are baked in as constants; arrays, structs, package calls and
  CairoMakie are not compilable). The pattern that works (notebook 11's helical-scan calculator):
  an Int `PlutoUI.Slider`; one hidden cell (ending in `;`) that returns a plain tuple; a `md"""…"""`
  readout that only interpolates `$(t[1])`, `$(t[2])`, … in ASCII text. Sliders that drive GPU
  volumes are forced to an honest static fallback (`FORCE_FALLBACK_BONDS` in `docs/extract_all.jl`,
  mirrored in `docs/verify_notebook_exports.py`: 01 `z_slice`, 05 `z_helical`, 11 `z_idx`);
  renaming such a bond means updating both lists. The verifier accepts compiled islands and exactly
  those fallbacks; it rejects a partial island or any other fallback. Check a new island in a real
  browser (`python3 -m http.server -d docs/dist`, move the slider).
- **Julia Markdown trap:** a paragraph containing `($(` (an opening parenthesis right before an
  interpolation) loses every interpolation in it — the `$` signs are read as LaTeX. Write
  `, $(x) s` rather than `($(x) s)`. Likewise an underscore inside a word outside backticks
  (`mu_water`) becomes italics: put identifiers in backticks.
- **Notebook rules:** use the package, never a re-implementation of a package stage; keep seeds
  fixed; print no machine paths (the verifier rejects `/home/…`, `/Users/…`, `/tmp/…`); state only
  numbers the notebook computes; pick the GPU with `GPUSelect.Storage()`.
- **VMI in notebooks and examples** follows basis-spectral-denoising's final version: view
  integration on every scanner (`SimOptions(; view_samples = 5)`, `view_arc` the duty cycle for
  rapid kVp switching), one projection per exposure shared by its draws (`keep_projection` /
  `projection`), and `vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PairFilter(…),
  pair_basis, composite_energy)` with the scanner's fitted window pair (one window where no pair
  was fitted, as for the Force in notebook 09) and the pair's basis fixed from a separate
  calibration draw.

## CI and deployment (`.github/workflows/`)

- `CI.yml`: the `version` check, the test suite on Linux, Windows and macOS (Julia 1.12), and
  `lib/BasisMonteCarlo`'s test suite on Linux (`basismontecarlo`; Geant4.jl, plus `g++` for its
  cross-section shim).
- `TagBot.yml`: tags `v<version>` and writes the GitHub Release after the General registry merges a
  registration.
- `docs.yml` (GitHub Pages) and `snapshot.yml` (Snapshot, managed by Snapshot with a marked local
  edit) build `main` on every push that changes `docs/`, `src/`, `Project.toml` or
  `lib/BasisMonteCarlo/`, or any commit by hand from the Actions tab.

## Around this repository

- The lab's repositories that use the package — basis-vmi, basis-spectral-denoising,
  basis-verification, semmd-bayesian, basis-autodiff-mmd, the pimmd-* repositories — are the
  users; grep them before removing or renaming an export. semmd-bayesian stores
  `"<version> <depot slug>"` with every cached scan, so any version change invalidates its cache.
- `feat/reactant-autodiff` (the Reactant/Enzyme functional core) is parked on purpose: it is kept
  merged up with `main` but is not developed; do not port main's changes into it unless asked.
