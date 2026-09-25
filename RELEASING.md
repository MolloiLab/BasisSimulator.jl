# Releasing BasisSimulator.jl

A release is one pull request and one registry comment. Nothing picks a version, writes the
changelog or opens a release on its own; `.github/workflows/TagBot.yml` only tags a version after
the General registry has accepted it.

## The rules that keep `main` releasable

- **`version` in `Project.toml` is the last registered version until the release pull request
  changes it.** A feature pull request never touches it. (0.15.0 exists because this rule was not
  kept: `main` walked from 0.14.0 to 0.18.0 without registering anything, and the registry only
  accepts the version that follows the last registered one.) The `version` job in
  `.github/workflows/CI.yml` fails any commit whose `version` is neither the last registered
  version nor a valid next one.
- **Every user-visible change adds a line under `## [Unreleased]` at the top of `CHANGELOG.md`**
  in the same pull request: what a reader has to do differently, with the measured number for
  anything claimed to be faster or more accurate (and the hardware it was measured on).
- **The published docs describe the latest release.** They are deployed from the newest `v*` tag
  after TagBot creates it (`docs.yml`, and Snapshot via `snapshot.yml`), never from `main`, so a
  code change on `main` cannot break the site. The notebook exports are re-rendered once, in the
  release pull request.

## Cutting a release

1. **Choose the version.** Below 1.0, a breaking change bumps the minor version (0.15 → 0.16)
   and anything else bumps the patch (0.15.0 → 0.15.1). A change is breaking if it removes or
   renames something exported, changes the meaning of an argument, or changes the numbers a
   caller gets back without asking for it. The `version` check accepts exactly the last
   registered version, its next patch, its next minor, and the next major.
2. **Branch** `release/<version>` from `main`.
3. **Bump `version`** in `Project.toml`.
4. **Date the changelog.** Rename `## [Unreleased]` to
   `## [<version>](https://github.com/MolloiLab/BasisSimulator.jl/compare/v<previous>...v<version>) (<YYYY-MM-DD>)`
   and add a fresh, empty `## [Unreleased]` above it. Group the entries as `Breaking` / `Added` /
   `Changed` / `Fixed` / `Performance` / `Documented`, skipping empty groups.
5. **Run the tests:** `julia --project=. -t 4 -e 'using Pkg; Pkg.test()'`. CI has no GPU, so run
   anything that touches a GPU path on one and say in the pull request what passed on what
   hardware.
6. **Re-render the notebook exports** on a GPU machine (see `AGENTS.md`, "Docs"):
   `docs/render_notebooks.sh` renders every stale notebook, split across the machine's GPUs, and
   `python3 docs/verify_notebook_exports.py` must end with `verified <n> notebook source/export
   pairs`. Commit `docs/notebooks-static/`. The export fingerprint covers all of `src/`,
   `Project.toml` (not its `version` line), the docs lockfiles and the exporter, so any code change
   since the last render makes every export stale; that is by design.
7. **Check the site locally:** `docs/build.sh`, then serve `docs/dist/` (for example
   `python3 -m http.server -d docs/dist`) and open the landing page, the API reference and a
   notebook.
8. **Open the pull request, merge it, then register** by commenting on the merge commit on GitHub:

   ```
   @JuliaRegistrator register
   ```

   The registry opens a pull request against General (AutoMerge takes about 15 minutes when the
   version and compat bounds are valid). When it merges, TagBot creates the `v<version>` tag and
   the GitHub Release from the changelog section, and the docs workflows deploy that tag.

## If something goes wrong

- **Registration refused as "not a valid version increment":** `version` skipped a number. Set it
  to the next valid version (the `version` CI job prints the allowed set) and comment again.
- **`docs.yml` / `snapshot.yml` fail with `stale export`:** the tag's exports were not re-rendered
  after its last code change. Re-render them on a GPU machine (step 6), release a patch version,
  and the docs deploy from that tag.
- **TagBot did not tag:** run it by hand from the Actions tab (`TagBot`, "Run workflow").
