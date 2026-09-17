# Releasing BasisSimulator.jl

Releases are cut by hand. Nothing in this repository picks a version number, writes a changelog
entry, or opens a release pull request on its own; the only automation left is
`.github/workflows/TagBot.yml`, which creates the git tag and the GitHub Release *after* the
Julia registry has accepted a registration request.

## Steps

1. **Decide the version.** While the package is below 1.0, a breaking change bumps the minor
   version (0.14 → 0.15) and everything else bumps the patch version. A change is breaking if it
   removes or renames something exported, changes the meaning of an argument, or changes the
   numbers a caller gets back without them asking for it.

2. **Run the tests.** `julia --project=. -e 'using Pkg; Pkg.test()'`, and say in the pull
   request what passed and on what hardware. The GPU paths are not exercised by CI, so run
   anything that touches them yourself and report the numbers.

3. **Write the changelog entry.** A new section at the top of `CHANGELOG.md`, dated, with the
   version compared against the previous tag. Group by `Added` / `Changed` / `Fixed` /
   `Performance` / `Breaking`. Say what a reader has to do differently, and give the measured
   number for anything claimed to be faster or more accurate. One line per change is enough if
   the line is specific; "various improvements" is not an entry.

4. **Bump `version` in `Project.toml`** to match.

5. **Merge to `main`.**

6. **Register.** Comment on the release commit:

   ```
   @JuliaRegistrator register
   ```

   The registry opens a pull request against `General`. When it merges, TagBot tags this
   repository and publishes the GitHub Release from the changelog section.

## What was removed, and why it is not coming back by accident

`release-please` used to read Conventional Commit subjects, choose the next version, write the
changelog, open a release pull request, tag, and post the registration comment. It has been
removed: `.github/workflows/release-please.yml`, `.release-please-config.json` and
`.release-please-manifest.json` are gone. Commit subjects no longer decide anything, so the
`feat:` / `fix:` / `perf:` prefixes are now a convention for readers rather than an instruction
to a robot. Keep using them; they make the history easy to scan.
