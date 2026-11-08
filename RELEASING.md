# Releasing BasisSimulator.jl

A release is one pull request and one registry comment. The docs are not part of it: the site
deploys from `main` on its own (see the end of this file).

## Rules that keep `main` releasable

- **`version` in `Project.toml` stays at the last registered version** until a release pull request
  changes it. Feature pull requests never touch it. The registry only accepts the next version
  after the last registered one, and the `version` job in `.github/workflows/CI.yml` fails any
  commit whose `version` is neither that nor a valid next one.
- **Every user-visible change adds a line under `## [Unreleased]`** in `CHANGELOG.md`, in the same
  pull request: what a reader has to do differently, with the measured number (and hardware) for
  anything claimed to be faster or more accurate.

## Cutting a release

1. **Choose the version.** Below 1.0, a breaking change bumps the minor version (0.15 → 0.16) and
   anything else the patch (0.15.0 → 0.15.1). Breaking means removing or renaming an export,
   changing what an argument means, or changing the numbers a caller gets back unasked.
2. **Branch** `release/<version>` from an up-to-date `main`.
3. **Bump** `version` in `Project.toml` and `version:` in `CITATION.cff`.
4. **Date the changelog:** rename `## [Unreleased]` to
   `## [<version>](https://github.com/MolloiLab/BasisSimulator.jl/compare/v<previous>...v<version>) (<YYYY-MM-DD>)`
   and add an empty `## [Unreleased]` above it. Group entries as `Breaking` / `Added` / `Changed` /
   `Fixed` / `Performance` / `Documented`, skipping empty groups.
5. **Scatter kernels (when the library changed).** Split BasisMonteCarlo's library per scanner and
   zip it, `julia lib/BasisMonteCarlo/scripts/split_scatter_library.jl scatter_library_v2.toml out/ --archive`
   (in an environment with `p7zip_jll`); attach the six `.zip`s to a GitHub release of their own,
   tag `scatter-kernels-v2` (a new tag for a new library), and paste the printed SHA-256 and tree
   hash of each into `SCATTER_KERNEL_ARTIFACTS` (`src/detector/scatter_kernels.jl`). Check one
   download: `BS.scatter_kernels("SOMATOM Force")` in a fresh depot.
6. **Test:** `julia --project=. -t 8 -e 'using Pkg; Pkg.test()'`, the GPU paths on a GPU (CI has
   none; the recipe is in the guide's Contributing section), and loading on the newest released
   Julia (`julia +<newest> --project=. -e 'using Pkg; Pkg.test()'`), because the registry's
   AutoMerge loads the package there whatever the `julia` compat says. Say in the pull request what
   passed on which hardware.
7. **Merge the pull request, then comment on its merge commit** on GitHub:

   ```
   @JuliaRegistrator register
   ```

   A breaking release must add release notes, or AutoMerge blocks it:

   ```
   @JuliaRegistrator register

   Release notes:

   ## Breaking changes
   - <the Breaking entries of the changelog section, one line each>
   ```

   AutoMerge takes about 15 minutes. When it merges, TagBot creates the `v<version>` tag and the
   GitHub Release.

## If something goes wrong

- **"Not a valid version increment":** `version` skipped a number. Set it to the next valid one
  (the `version` CI job prints the allowed set) and comment again.
- **AutoMerge fails to load the package:** a dependency does not resolve on the newest Julia. Fix
  the compat bound (or release the dependency) and comment again.
- **TagBot did not tag:** run it from the Actions tab (`TagBot`, "Run workflow").

## The docs site

`docs.yml` (GitHub Pages) and `snapshot.yml` (Snapshot) build `main` whenever a push changes
`docs/`, `src/`, `Project.toml` or `lib/BasisMonteCarlo/`, and can be run from the Actions tab for
any commit. A docs-only
change is live a few minutes after it merges.

The build refuses notebook pages whose fingerprint is stale. The fingerprint covers what decides
a page: the notebook, its data declarations, the docs environment (the build and docs lockfiles,
without this repository's own `version` lines) and the export contract. It does not cover the
package's `src/` or `Project.toml`, so a code change or a version bump never blocks the deploy and
never forces every notebook to re-render. When a change alters what notebooks show (a release that
moves results), re-render the notebooks it affects on a GPU machine, by name
(`docs/render_notebooks.sh 04_pcct_vmi 08_qrm_thorax_pure_material_pcct`), or all of them with
`BASISSIM_FORCE_NB_REBUILD=1 docs/render_notebooks.sh`, and merge the new `docs/notebooks-static/`.
Editing a notebook makes its own page stale until it is re-rendered. The BasisMonteCarlo notebooks
(`mc-*`) work the same way and render on the CPU (`docs/render_notebooks.sh mc-<stem>`; AGENTS.md,
"The documentation site").
