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
5. **Test:** `julia --project=. -t 8 -e 'using Pkg; Pkg.test()'`, the GPU paths on a GPU (CI has
   none; the recipe is in the guide's Contributing section), and loading on the newest released
   Julia (`julia +<newest> --project=. -e 'using Pkg; Pkg.test()'`), because the registry's
   AutoMerge loads the package there whatever the `julia` compat says. Say in the pull request what
   passed on which hardware.
6. **Merge the pull request, then comment on its merge commit** on GitHub:

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
`docs/`, `src/` or `Project.toml`, and can be run from the Actions tab for any commit. A docs-only
change is live a few minutes after it merges.

The build refuses notebook pages whose fingerprint is stale. The fingerprint covers the notebook,
all of `src/`, `Project.toml` (not its `version` line) and the docs lockfiles, so after a code
change the deploy fails until the notebooks are re-rendered on a GPU machine
(`docs/render_notebooks.sh`, then `python3 docs/verify_notebook_exports.py`) and the new
`docs/notebooks-static/` is merged. Re-render in the pull request that changes the code, or in a
follow-up one.
