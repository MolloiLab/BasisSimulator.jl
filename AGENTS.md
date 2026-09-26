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
```

## Working on the code

- **Julia 1.12 only** (`julia = "~1.12"`: XrayAttenuation does not resolve on 1.13, and the bundled
  MC response is in 1.12's serialization format).
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
  design, and the exports are re-rendered once per release (RELEASING.md, step 6), because the
  published site is built from the release tag.
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
  pair_basis, composite_energy)` with the scanner's fitted window pair and the pair's basis fixed
  from a separate calibration draw.

## CI and deployment (`.github/workflows/`)

- `CI.yml`: the `version` check, and the test suite on Linux, Windows and macOS (Julia 1.12).
- `TagBot.yml`: tags `v<version>` and writes the GitHub Release after the General registry merges a
  registration.
- `docs.yml` (GitHub Pages) and `snapshot.yml` (Snapshot, managed by Snapshot with a marked local
  edit) build the newest `v*` tag when TagBot completes, or by hand from the Actions tab.

## Around this repository

- The lab's repositories that use the package — basis-vmi, basis-spectral-denoising,
  basis-verification, semmd-bayesian, basis-autodiff-mmd, the pimmd-* repositories — are the
  users; grep them before removing or renaming an export. semmd-bayesian stores
  `"<version> <depot slug>"` with every cached scan, so any version change invalidates its cache.
- `feat/reactant-autodiff` (the Reactant/Enzyme functional core) is parked on purpose: it is kept
  merged up with `main` but is not developed; do not port main's changes into it unless asked.
