# BasisSimulator.jl — Testing, Benchmarking, and Oracle Infrastructure Survey

Prepared for a follow-up team building an oracle-driven test harness for a Reactant.jl + Enzyme.jl
rewrite, where the current code is the numerical oracle.

Repo root: `/Users/daleblack/Documents/dev/MolloiLab/BasisSimulator.jl`
Version surveyed: 0.14.0, branch `main`, HEAD `d9b8183` (chore(main): release 0.14.0, PR #64).
Working tree clean at time of survey. **No files were modified.** One test-suite run was executed.

---

## 1. Test suite layout

### 1.1 Tree

```
test/
├── runtests.jl                                 42 lines   flat include list
├── api.jl                                    1922 lines
├── bowtie.jl                                  273 lines
├── correction.jl                              590 lines
├── denoising.jl                               382 lines
├── detector.jl                                893 lines
├── geometry.jl                                563 lines
├── memory_lifecycle.jl                         54 lines
├── object.jl                                  259 lines
├── phantoms.jl                                108 lines
├── projection.jl                              559 lines
├── source.jl                                  599 lines
├── memory_stability_pcct.jl                    86 lines   NOT in runtests.jl
├── memory_stability_pcct_representative.jl    161 lines   NOT in runtests.jl
└── archived/
    ├── runtests.jl                           6996 lines   pre-refactor suite
    ├── test_mc_pileup.jl                      202 lines
    ├── test_mc_response.jl                    130 lines
    ├── validate_pcct_physics.jl               674 lines
    ├── vmi_brent_parity.jl                    134 lines
    └── references/README.md                              planned clinical ref library
```

`test/archived/` is not referenced from `test/runtests.jl` and is not run by anything.

### 1.2 `test/runtests.jl` structure

`test/runtests.jl:1-42`. Preamble is `using Test, BasisSimulator, Random, LinearAlgebra`,
`using Statistics: mean, std`, and `const BS = BasisSimulator` at `:6`. Then one outer
`@testset "BasisSimulator.jl"` at `:8` containing eleven `@testset` wrappers, each a bare
`include`, in this order:

| Order | Wrapper name | Include | Line |
|---|---|---|---|
| 1 | `api/` | `api.jl` | `test/runtests.jl:10` |
| 2 | `bowtie/` | `bowtie.jl` | `:13` |
| 3 | `correction/` | `correction.jl` | `:16` |
| 4 | `denoising/` | `denoising.jl` | `:19` |
| 5 | `detector/` | `detector.jl` | `:22` |
| 6 | `geometry/` | `geometry.jl` | `:25` |
| 7 | `memory lifecycle/` | `memory_lifecycle.jl` | `:28` |
| 8 | `object/` | `object.jl` | `:31` |
| 9 | `phantoms/` | `phantoms.jl` | `:34` |
| 10 | `projection/` | `projection.jl` | `:37` |
| 11 | `source/` | `source.jl` | `:40` |

### 1.3 Test group selection

**There is none.** No `ARGS` parsing, no env-var gating, no tag system, no `--group` convention
anywhere in `test/`. Every include runs on every invocation. The only env var read in the whole
test tree is `PCCT_MEMORY_CYCLES`, default `"3"`, at
`test/memory_stability_pcct_representative.jl:13`, and that file is not part of `runtests.jl`.

The two soak scripts self-document their invocation:

- `test/memory_stability_pcct.jl:1-6`: `julia --project=docs test/memory_stability_pcct.jl`.
  Header says it is "intentionally not included in runtests.jl: it compiles and executes the
  complete PCCT spectral projector several times." Runs 5 create/simulate/release cycles.
- `test/memory_stability_pcct_representative.jl:1-6`: `julia --project=docs
  test/memory_stability_pcct_representative.jl`. "Exact 04d scan-geometry unified-memory soak
  test." Runs `N_CYCLES` cycles at clinical scale: 512x512x16 Gammex, 1195-column PCCT detector,
  144 rows, 1200 views, recon 512x512x12.

### 1.4 Assertion counts

| File | `@test` lines | `@testset` blocks |
|---|---|---|
| `test/api.jl` | 449 | see below |
| `test/detector.jl` | 314 | |
| `test/source.jl` | 235 | |
| `test/geometry.jl` | 192 | |
| `test/correction.jl` | 119 | |
| `test/denoising.jl` | 112 | |
| `test/object.jl` | 99 | |
| `test/projection.jl` | 92 | |
| `test/bowtie.jl` | 66 | |
| `test/phantoms.jl` | 24 | |
| `test/memory_lifecycle.jl` | 14 | |
| `test/runtests.jl` | 12 | |
| **Total** | **1728** | **436 testsets** |

Executed assertion count is higher because many are inside loops. Measured: **3156 passing
assertions**.

### 1.5 GPU backend selection

This is the single most important operational fact for the rewrite.

There is **no `GPUSelect`** and **no `BS.backend`** in the test path. `README.md` advertises the
`GPUSelect.Storage()` pattern and `docs/Project.toml:4` depends on `GPUSelect`, but the tests do
not use it.

Detection lives at `test/api.jl:18-41`:

```julia
const GPU_BACKEND = let
    candidates = [
        (:Metal,  "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
        (:CUDA,   "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
        (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
    ]
    detected = (name = "CPU", to_gpu = identity)
    for (pkg, uuid, ctor) in candidates
        pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
        Base.locate_package(pkg_id) === nothing && continue
        try
            m = Base.require(pkg_id)
            if Base.invokelatest(getfield(m, :functional))
                detected = (name = string(pkg), to_gpu = getfield(m, ctor))
                break
            end
        catch
        end
    end
    detected
end

const HAS_GPU = GPU_BACKEND.name != "CPU"
to_gpu(x) = GPU_BACKEND.to_gpu(x)
@info "[api.jl] GPU backend = $(GPU_BACKEND.name)"
```

`test/api.jl:39` defines `HAS_GPU`; `test/api.jl:40` defines the `to_gpu` helper. `test/api.jl:41`
logs the backend. Because `api.jl` is included first by `runtests.jl`, `HAS_GPU` and `to_gpu` leak
into the `Main` scope used by every later file; `test/correction.jl:403` and `:433` consume them
without redefining.

The rationale comment at `test/api.jl:11-16` says the PCCT spectral forward project is GPU-only in
practice because "on CPU, the K=16 tiled path JIT-compiles for 14+ min on even a 32^3 phantom."

**Metal and CUDA are weakdeps, not test deps.** `Project.toml:22-24` lists them under
`[weakdeps]`; `Project.toml:45-49` sets `[extras] Test` and `[targets] test = ["Test"]`. Therefore
`Pkg.test()` resolves an environment with no GPU package and `HAS_GPU` is `false`. Confirmed by my
run: the log's first line is `[ Info: [api.jl] GPU backend = CPU`.

Metal appears only in `docs/Project.toml:8`. The GPU path is reachable via
`julia --project=docs test/runtests.jl`, which is the same environment the soak scripts name.

### 1.6 GPU-gated testsets that skip on CPU

Eleven in `test/api.jl`, two in `test/correction.jl`. Line numbers of the `if !HAS_GPU` guard:

| File:line | Testset skipped |
|---|---|
| `test/api.jl:362` | `simulate!(PCCTWorkspace)` return contract, incl. exact-Poisson raw counts |
| `test/api.jl:429` | `simulate!(PCCTWorkspace)` MC-LUT pileup wiring |
| `test/api.jl:617` | `simulate!(EICTWorkspace)` return contract |
| `test/api.jl:643` | `simulate!(EICTWorkspace)` noise on/off + seed reproducibility |
| `test/api.jl:1520` | `create_workspace` (PCCT) field invariants |
| `test/api.jl:1568` | `create_workspace` (PCCT) `use_pcct_pileup=false` skips `pileup_S` |
| `test/api.jl:1579` | `create_eict_workspace` field invariants |
| `test/api.jl:1609` | `create_eict_workspace` `spectrum_override` path |
| `test/api.jl:1778` | `simulate!(EICTWorkspace)` scatter on/off |
| `test/api.jl:1822` | `simulate!(PCCTWorkspace)` focal-spot blur on/off |
| `test/api.jl:1859` | `simulate!(EICTWorkspace)` quantum noise scales as 1/sqrt(mA) |
| `test/correction.jl:403` | `apply_bhc_two_material` on GPU arrays |
| `test/correction.jl:433` | `apply_bhc_image_domain` on GPU arrays |

**Consequence: the entire `simulate!` end-to-end path and every workspace-constructor invariant is
unverified by `Pkg.test()` and by CI on all three runners.** What CI actually verifies is the
projector kernels on CPU, the reconstruction kernels on CPU, the detector and source physics
models, geometry, corrections on CPU arrays, denoising, and the option/struct contracts.

### 1.7 Measured runtime

Command: `julia --project=. -e 'using Pkg; Pkg.test()'` on macOS arm64, Apple M4.

| Phase | Time |
|---|---|
| Dependency precompile, 50 packages | 47 s |
| `Test Summary: BasisSimulator.jl` | **3m04.0s**, 3156/3156 pass |
| Total wallclock incl. precompile and setup | **3:57.72**, exit 0 |
| `test/api.jl` alone, from its internal `_ts` logger | 99.0 s |

Within `api.jl` the dominant cost is the two CPU Monte Carlo pileup testsets: `test/api.jl:486`
(`compute_mc_pileup_matrix` physical-shape contract) and `test/api.jl:538` (pile-up math round
trip), each calling `compute_mc_pileup_matrix(...; n_trials = 2000, seed = 1)` at
`count_rate = 1.0e8`, `dead_time_ns = 5.0`. Together roughly 70 s of the 99 s. Second cost is
`reconstruct!(HIRReconWorkspace)` from `test/api.jl:1220`, about 15 s, which runs 11 full HIR
reconstructions in the strength sweep at `test/api.jl:1509-1514`.

Progress instrumentation exists only in `api.jl`: `_ts(label)` at `test/api.jl:46-47` prints to
`stderr` with an explicit flush, motivated by the comment at `:43-45` that "Pluto/Pkg.test buffers
stdout until @testset exits."

With a GPU present, expect substantially longer: the 13 skipped testsets include multiple
`create_workspace` + `simulate!` cycles on a 32x32x4 Gammex with a 64x8 detector, plus a two-point
mA dose sweep at `test/api.jl:1908-1913` that runs four full simulations.

### 1.8 CI matrix

`/Users/daleblack/Documents/dev/MolloiLab/BasisSimulator.jl/.github/workflows/CI.yml`

| Field | Value | Line |
|---|---|---|
| Triggers | push to `main`, tags `*`, all pull_request, workflow_dispatch | `:3-9` |
| Concurrency | group per workflow+ref, cancel-in-progress only for PRs | `:11-13` |
| Timeout | 60 minutes | `:19` |
| `fail-fast` | false | `:21` |
| Julia versions | `'1'` only, auto-tracks newest stable | `:23-24` |
| OS | ubuntu-latest, windows-latest, macos-latest | `:25-28` |
| Arch | x64, with macos-latest x64 excluded and macos-latest arm64 added | `:29-38` |
| Effective legs | ubuntu x64, windows x64, macos arm64 | |
| Env | `JULIA_NUM_THREADS: 2` | `:53-54` |
| Steps | checkout v4, setup-julia v2, cache v2, buildpkg v1, runtest v1, processcoverage v1, codecov v4 | `:41-61` |
| Codecov | `fail_ci_if_error: false` | `:61` |

No GPU runner. No nightly. No LTS leg. `Project.toml:42` requires `julia = "1.11"`.

Other workflows:

- `.github/workflows/docs.yml` — builds and deploys GitHub Pages on push to `main`. Julia 1.12.
  Runs `docs/build.sh`. Comment at `:32-37` explains `BASISSIM_SKIP_NB_EXPORT` short-circuits the
  Pluto render because "CI can't run our Metal/CUDA kernels"; notebooks are rendered locally on
  Apple Silicon and the HTML is committed under `docs/notebooks-static/`.
- `.github/workflows/snapshot.yml` — Snapshot build and publish, pinned SHAs, `SNAPSHOT_OPTIMIZE:
  size`, `SNAPSHOT_ORACLE_SAMPLES: "5"` at `:31-32`. Driven by `snapshot.toml`, which sets
  `type = "build"` so the default notebook-to-WASM collection pipeline is overridden and
  `docs/build.sh` runs instead.
- `.github/workflows/release-please.yml`, `.github/workflows/TagBot.yml` — release automation.

---

## 2. Existing parity and oracle-style tests

This is the richest existing material for the rewrite. Each row is a self-contained numerical
contract you can lift directly.

### 2.1 Projector adjoint and transpose oracles

`test/projection.jl:31-96`, testset `"distance-driven exact transpose"`. Loops
`for shape in (:flat, :arc)`.

Fixture: `BS.Scanner(source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 7,
detector_cols = 24, detector_row_size = 0.8, detector_col_size = 0.8, detector_shape = shape)`;
`BS.CTGeometry(scanner; n_angles = 9, n_rows = 7, n_cols = 24, fov_cm = 8.0, z_cm = 1.4)`.
RNG `MersenneTwister(0xDD3 + (shape === :arc))`. `x = randn(Float64, 9, 9, 5)`,
`y = randn(Float64, 24, 7, 9)`. All Float64.

| Check | Compared | Tolerance | Line |
|---|---|---|---|
| Arc row-tiled forward equals scalar | `_dd_forward_project_arc_rowtile4!` vs `dd_forward_project` | `==`, bit-identical | `:49-50` |
| Adjoint dot product | `sum(Ax .* y)` vs `sum(x .* Aty)` | `rtol = 2.0e-11`, `atol = 2.0e-11` | `:56` |
| Backprojection determinism | second `dd_backproject!` call | `==`, bit-identical | `:60-61` |
| Arc `active_z` tiling | `active_z = 2:4` slice vs full-range slice | `==`, plus `all(isnan)` outside the range | `:68-71` |
| `active_z` bounds validation | `0:2` and `4:6` | `@test_throws ArgumentError` | `:74-75` |
| **Brute-force matrix oracle** | explicitly form every column of `A` by forward-projecting 27 unit-basis volumes, then compare optimized gather against `transpose(A) * vec(y)` | `rtol = 2.0e-12`, `atol = 2.0e-12` | `:81-94` |

The brute-force oracle at `:77-80` carries the rationale: "This catches omitted edge candidates
even when a global dot-product check happens to be numerically forgiving." Oracle volume shape is
`(3, 3, 3)`, 27 columns, against `24 * 7 * 9 = 1512` rays.

**This is the sharpest instrument in the repo.** It is a complete linear-operator identity at
2e-12 in Float64 and it will detect any semantic change in the DD footprint under a rewrite.

### 2.2 DD versus Siddon agreement

`test/projection.jl:157-203`, testset `"dd_forward_project (distance-driven)"`.

Fixture: `_toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)`; volume 64x64x8;
centred cylinder radius `0.6 * 32` voxels at mu = 0.2 cm^-1. Mask is
`sino_siddon .> 1.0f-4`.

| Check | Tolerance | Line |
|---|---|---|
| Object actually illuminated | `count(mask) > 1000` | `:188` |
| Mean relative deviation | `< 0.01` | `:189` |
| Worst-ray relative deviation | `< 0.05` | `:190` |
| In-place equals allocating | max abs `< 1.0e-5` | `:196` |
| Linearity in mu, 2x scale | max abs `< 1.0e-4` | `:201` |
| Zero volume gives zero sinogram | `all(sino .== 0)`, `eltype == Float32` | `:167-168` |

Siddon standalone, `test/projection.jl:101-149`: max line integral bounded by
`0.2 * 5.0 * sqrt(2) + 1.0e-3` at `:124`; linearity max abs `< 1.0e-4` at `:132`; in-place equals
allocating max abs `< 1.0e-4` at `:148`. Fixture `_toy_proj_geom()` defaults: 16 cols, 4 rows,
4 angles, fov 5 cm, volume 16x16x4.

### 2.3 Fused-kernel parity, DD versus Siddon

`test/projection.jl:211-265`, testset `"dd fused projectors vs Siddon fused"`.

Fixture: same 64-col geometry; `N_E = 8`; 2-material `UInt16` mask, water cylinder; `mu_table`
2x8 with water ramping `0.30f0` down to `0.18f0`; `wη = fill(Float32(1/N_E), N_E)`.

| Check | Compared | Tolerance | Line |
|---|---|---|---|
| Fused poly | `dd_fused_poly_project!` vs `siddon_fused_poly_project!` | mean rel `< 0.01`, max rel `< 0.05`, `count > 1000` | `:236-239` |
| Fused spectral, per bin | `dd_fused_spectral_project!` vs `siddon_fused_spectral_project!`, 2 bins, `K = 8`, single tile | bin mean rel `< 0.02` | `:262` |

Spectral fixture builds `W` as an 8x2 energy-to-bin matrix with linear ramps, `:245-249`; output
buffers are flat `Vector{Float32}` of length `n_elements * 2`.

### 2.4 `:dd_fast` identical-physics proof

`test/projection.jl:273-375`, testset
`"dd_fast fused projectors ≡ dd fused (path-length reassociation)"`. This is the exact contract
your rewrite must preserve.

Fixture: same 64-col geometry; **3 materials** so the per-material accumulator is exercised, water
cylinder plus a radius-3 rod at `(45.5, 32.5)`; `mu_table` 3x8, water `0.30` to `0.18`,
rod `1.50` to `0.60`.

| Sub-testset | Compared | Tolerance | Line |
|---|---|---|---|
| `"fused poly: dd_fast ≡ dd (float-ordering only)"` | `dd_fast_fused_poly_project!` vs `dd_fused_poly_project!` | max abs `< 1.0f-4` | `:301` |
| `"fused poly: dd_fast tracks Siddon exactly as legacy dd does"` | `maximum(relΔ_f) <= maximum(relΔ_d) + 1.0f-4`, plus mean rel `< 0.01`, `count > 1000` | relative envelope | `:317-319` |
| `"fused spectral: dd_fast ≡ dd (single tile)"` | flat output vectors, 2 bins | max abs `< 1.0f-4` | `:336` |
| `"helical arc geometry + spectral bowtie remain equivalent"` | arc 32x4 detector, `n_angles = 16`, `pitch = 1.0`, `n_rotations = 2.0`, `volume_extent = (10.0, 10.0, 1.0)`, spectral bowtie 32x4x8 with column and energy ramps | max abs `< 1.0f-4` | `:361` |
| `"M > 64 materials warns and falls back to the legacy dd kernel"` | `mu_big` 65x8; `@test_logs (:warn, r"DD_FAST SINGLE-PASS DISABLED.*65"s)`; then `sino_f == sino_d` | **bit-identical** | `:371-373` |

The Siddon-envelope framing at `:305-307` is worth quoting because it tells you what is and is not
a real invariant: "On this hard-edged rod phantom Siddon-vs-DD legitimately disagree on grazing
rays (the aliasing DD exists to fix), so the invariant is: dd_fast deviates from Siddon NO MORE
than legacy dd does."

### 2.5 Projector-selection routing

`test/projection.jl:427-462`.

| Check | Tolerance | Line |
|---|---|---|
| `_validate_projector(:dd) === :dd`, `(:siddon) === :siddon` | exact | `:432-433` |
| `_validate_projector(:bogus)` | `@test_throws ArgumentError` | `:434` |
| Allocating shim `_project_mono(:dd, ...)` vs `dd_forward_project` | `==`, bit-identical | `:438` |
| Allocating shim `_project_mono(:siddon, ...)` vs `siddon_forward_project` | `==`, bit-identical | `:439` |
| In-place shim, both projectors | `==`, bit-identical | `:446, :451` |
| `:dd` and `:siddon` are genuinely different code paths | `s_dd != s_si`, then `isapprox(...; rtol = 0.05)` on the mid ray | `:458-460` |

Note `test/projection.jl:432-434` only validates `:dd`, `:siddon`, and `:bogus`. The actual
implementation at `src/projection/select_projector.jl:44-46` accepts `:dd`, `:dd_fast`, and
`:siddon`. `:dd_fast` acceptance is covered instead by `test/api.jl:152-158`.

### 2.6 Arc detector resample oracle, MATLAB fanbeam style

`test/projection.jl:474-558`, testset `"arc detector — projectors"`. The header comment at
`:466-472` states the design: "a FLAT sinogram resampled onto the arc's column grid
(col -> gamma -> u = SDD*tan gamma) must match the NATIVE arc sinogram (this is the 'simple
interpolation' conversion, here used as a test oracle rather than as the implementation)."

Fixture: two scanners identical except `detector_shape = :flat` and `:arc`, both SID 540, SDD
1080, 8 rows, 128 cols, 1.0 x 1.0 mm; `n_angles = 8`, `fov_cm = 20.0`. Volume 64x64x8, water
cylinder at 0.2 plus a radius-3 rod at 1.0.

The oracle loop at `:511-517`:

```julia
dγ = gf.pixel_size / SAD
pm = gf.pixel_size * (SDD / SAD)
γ    = (i - cc) * dγ
colf = SDD * tan(γ) / pm + cc
c0   = clamp(floor(Int, colf), 1, nc - 1)
w    = clamp(colf - c0, 0.0, 1.0)
resampled[i, r, a] = (1 - w) * sino_flat[c0, r, a] + w * sino_flat[c0 + 1, r, a]
```

| Check | Tolerance | Line |
|---|---|---|
| Interior chord mask non-trivial | `count(m) > 3000` where `m = (sino_arc .> 0.5f0) .& (resampled .> 0.5f0)` | `:524` |
| Mean relative error | `< 0.005` | `:525` |
| p99 relative error | `< 0.05` | `:526` |
| Max relative error | `< 0.10` | `:527` |
| Central columns 57:72, abs gamma under 0.03 rad, arc vs flat | max rel `< 0.01` | `:534` |
| Arc behaves like flat under Siddon | arc mean rel `<= 1.3 *` flat mean rel `+ 0.002` | `:547` |
| `dd_fast` equals `dd` on the arc, 3 materials, `N_E = 4` | max abs `< 1.0f-4` | `:557` |

The comment at `:518-521` records the measured values behind the asserted bounds: "Measured
interior: mean 0.25%, p99 2.3%, max 4%." The asserted bounds are therefore about 2x looser than
measurement, deliberately, because "tangent (grazing) rays have near-singular gradients where ANY
resampling shows large relative error — that is interpolation physics, not a geometry defect."

### 2.7 Reconstruction oracles

`test/geometry.jl:478-563`.

**Helical FDK round trip**, `test/geometry.jl:478-520`. Fixture: SID 540, SDD 1080, 16 rows,
128 cols, 1.0 x 1.0 mm; `CTGeometry(scanner; n_angles = 96, fov_cm = 12.8, pitch = 1.0,
n_rotations = 3.0)`, feed 1.6 cm/rot, travel 4.8 cm. Volume 64x64x48 with
`vol_extent = (12.8, 12.8, 6.4)`, cylinder radius `0.35 * 64` at mu = 0.2. Recon 64x64x24.

| Check | Tolerance | Line |
|---|---|---|
| Mean mu in ROI `rec[25:40, 25:40, 12]` vs 0.2 | `abs(mu_bar - 0.2) < 0.01`, comment says "within 5%" | `:507` |
| z-uniformity, 22 slice means k in 2:23 | `max - min < 0.02`, comment "< 10% of water mu" | `:511` |
| Axial parity against a circular scan, same phantom | `abs(mu_bar - axial mean) < 0.005` | `:519` |

**Arc reconstruction round trips**, `test/geometry.jl:525-563`. Same scanner with
`detector_shape = :arc`, volume 64x64x48, `volume_extent = (12.8, 12.8, 9.6)`.

| Sub-testset | Check | Tolerance | Line |
|---|---|---|---|
| axial equiangular FDK | centre ROI `rec[29:36, 29:36, 4]` vs 0.2 | `< 0.01` | `:548` |
| axial equiangular FDK | off-centre ROI `rec[43:50, 29:36, 4]` vs centre, radial flatness | `< 0.008` | `:550` |
| helical WFBP on arc | mean of 22 z-means vs 0.2 | `< 0.012` | `:560` |
| helical WFBP on arc | z-mean spread, no banding | `< 0.015` | `:561` |

**FDK contract tests**, `test/api.jl:1094-1218`. CPU-only, no GPU gate. Fixture
`_toy_fdk_setup`: SID 540, SDD 1080, 8 rows, 32 cols, `n_angles = 16`, `fov_cm = 20.0`,
`z_cm = 5.0`, matrix 16x16x4.

| Check | Tolerance | Line |
|---|---|---|
| Return identity, shape, eltype | `ret === ws.volume`, `Float32` | `:1112-1114` |
| Zero sinogram gives 0 inside FOV or exactly the `-0.04` sentinel | `v == 0.0f0 \|\| v ≈ -0.04f0` | `:1125` |
| Centre voxel exactly zero | `== 0.0f0` | `:1128` |
| **Determinism** | two workspaces, same random sinogram, `s1.ws.volume == s2.ws.volume` | **bit-identical** | `:1140` |
| Non-trivial recon has variance | `std(ws.volume) > 0` | `:1155` |
| FOV mask corners get sentinel, all z | `≈ -0.04f0` | `:1169-1172` |
| Five filter kernels run finite | `:ram_lak, :shepp_logan, :cosine, :hamming, :hann` | `:1181-1191` |
| **No accumulation** | pollute buffer with `999.0f0`, rerun, `ws.volume == v_clean` | **bit-identical** | `:1216` |

**HIR contract tests**, `test/api.jl:1220-1517`. Fixture `_toy_hir_setup`: same scanner,
`n_angles = 24` chosen divisible by `n_subsets = 12`, matrix 16x16x4, `strength = 60`.

| Check | Tolerance | Line |
|---|---|---|
| **strength = 0 equals FDK init exactly** | `ws.volume == ws_fdk.volume` | **bit-identical** | `:1305` |
| `work_volume === volume` and `work_geom === geom` at strength 0 | identity | `:1298-1300` |
| **`W_proj` against closed form** | `ifelse(ray_sum > 1.0f-8, inv(ray_sum), 0.0f0)` on the FOV-masked circular support | `rtol = 2.0f-6`, `atol = 2.0f-6` | `:1349` |
| Axial ROI, terminal slices equal each other | `means[1] ≈ means[end]` | `rtol = 2.0e-5` | `:1354` |
| Axial ROI, terminal mean vs mid | `rtol = 0.01` | `:1355` |
| Retains DC signal | `0.01 < means[6] < 0.03` for a 0.02 cm^-1 cylinder | `:1353` |
| FDK axial parity, same geometry | `rtol = 0.01` | `:1360` |
| HIR damps high-pass texture vs FDK | `mean(hir_hp) < 0.9 * mean(fbp_hp)` | `:1382` |
| No terminal mesh artifact | `max(hir_hp[1], hir_hp[end]) / hir_hp[6] < 1.25` | `:1383` |
| Finite object stays finite | `short_means[6] > 2 * max(terminals)` | `:1395` |
| Short-object terminal symmetry | `rtol = 0.1`, `atol = 5.0e-4` | `:1396` |
| Helical HIR exact-DD smoke | pitch 0.5, 2 rotations, 12 angles, matrix 16x16x5, `all(isfinite)` | `:1400-1417` |
| Huber gradient zero on z-uniform field | `iszero(grad)` on `fill(0.25f0, 9, 7, 5)`, delta `0.06f0` | exact | `:1421-1423` |
| `Ax` field removed from struct | `!(:Ax in fieldnames(BS.HIRReconWorkspace))` | `:1428` |
| **Determinism** | two workspaces, `s1.ws.volume == s2.ws.volume` | **bit-identical** | `:1446` |
| Warm start differs from FDK init | `!=` | `:1461` |
| Strength monotonicity, every 10% step 10 to 100 | `lambda` up, `huber_delta` down, `nepochs` up, `relaxation` down, `target_noise_reduction[1]` up | `:1502-1508` |
| Every strength 0 to 100 runs finite | 11 full reconstructions | `:1510-1514` |

The geometry-level affine and resample oracles, `test/geometry.jl`:

| Check | Tolerance | Line |
|---|---|---|
| `phantom_to_world_affine` diagonal and translation vs closed form | default `≈` | `:340-346` |
| `recon_to_world_affine` | same pattern | `:333-348` |
| `resample_to_recon` identity grid, `:nearest` preserves labels | exact | `:363-369` |
| `resample_to_recon` identity grid, `:linear` | max abs `< 1.0e-5` | `:376` |
| Uniform angle spacing | `max abs(delta - 2pi/n) < 1.0e-12` | `:200` |
| Source on a circle of radius SAD | `atol = 1.0e-10` | `:207` |
| Detector centre opposite source | `atol = 1.0e-10` | `:217` |
| u-axis unit and perpendicular to v | `< 1.0e-12` each | `:230-231` |
| Helical last angle | `atol = 1e-9` | `:432` |
| Helical last z | `atol = 1e-9` | `:436` |
| Helical uniform dz | `max abs(dz - dz[1]) < 1e-12` | `:441` |

### 2.8 Pile-up and count-domain round trips

**Pile-up correction round trip**, `test/correction.jl:119-149`. Fixture: hand-built
lower-triangular 4x4 `S` with diagonals 0.92, 0.88, 0.85, 0.80 at `test/correction.jl:91-96`;
`I0_bins = [1.0e6, 8.0e5, 5.0e5, 3.0e5]`; sinograms 8x4x6; `Random.seed!(2026)`. Forward: build
`bins[b] = -log(max(sum_c S[b,c] * I0[c] * exp(-t_truth[c]), 1e-12) / I0[b])`, then correct in
place and recover.

| Check | Tolerance | Line |
|---|---|---|
| Round trip recovers truth | max abs `< 1.0e-3` | `:147` |
| Identity matrix is a no-op | max abs `< 1.0e-5` | `:158` |
| 3-bin input errors | `@test_throws ErrorException` | `:110` |
| Wrong `S` shape errors | `@test_throws ErrorException` | `:114` |
| Wrong `I0_bins` length errors | `@test_throws ErrorException` | `:116` |

**Pile-up I0 renormalization**, `test/api.jl:538-575`. Pure math, no phantom, no GPU. Fixture:
`I0_truth = [1.3e12, 1.0e12, 4.2e11, 2.1e11]`; `S` from `compute_mc_pileup_matrix(thresholds,
weights, energies, 1.0e8, 5.0; n_trials = 2000, seed = 1)`; flat spectrum 20 to 140 keV;
thresholds `[20.0, 35.0, 55.0, 70.0]`.

| Check | Tolerance | Line |
|---|---|---|
| `I0_truth[i] * exp(-bins_air[i])` equals recorded counts | `rtol = 1.0e-12` | `:561` |
| Air-ray offset equals `log(I0_truth / I0_recorded)` | `rtol = 1.0e-12` | `:568` |

**Pile-up matrix physical shape**, `test/api.jl:486-526`. Same `S`.

| Check | Tolerance | Line |
|---|---|---|
| Shape, finiteness, non-negativity | `(4,4)`, `all(isfinite)`, `all(>=(0))` | `:501-503` |
| Columns not degenerate | `max_col_diff > 0.05`; comment notes "original PR's broken matrix gave 0 here" | `:510` |
| Column sums are count-conserving | `0.0 < s <= 1.0 + 1.0e-6` | `:515` |
| Upward-only migration | `S[i, j] < 0.05` for all `i < j` | `:521` |
| Real pile-up present at high a-tau | `S[2, 1] > 0.01` | `:525` |

**Raw-count snapshot helper**, `test/api.jl:344-359`. CPU-only. `logged[b] = fill(-log(0.25b),
2, 3, 4)`, `I0 = [100, 200, 300, 400]`.

| Check | Tolerance | Line |
|---|---|---|
| `raw[b] == I0[b] .* exp.(-logged[b])` | `rtol = 8eps(Float32)` | `:352` |
| Snapshots are independent copies | `raw[b] !== logged[b]`, mutate-and-compare | `:349, :356-357` |
| Length mismatch errors | `@test_throws DimensionMismatch` | `:358` |

GPU-gated extensions of the same contract at `test/api.jl:396` use
`rtol = 8eps(Float32) atol = 1.0e-6`; the exact-integer-Poisson checks at `test/api.jl:415-423`
assert `all(isinteger)`, `all(>=(0))`, and that log bins encode `max(N, 1)` with abs `< 0.1`.

### 2.9 CatSim and XCIST cross-validation

`test/bowtie.jl` header at `:1-11` names the two upstream tests it ports:
`gecatsim/tests/test_catsim/test_Xray_Filter.py` and
`test_Resample_Spectrum_Bowtie_FlatFilter.py`. CatSim is BSD 3-Clause, GE Precision HealthCare.

**Data pinning**, `test/bowtie.jl:54-77`. `_CATSIM_FIRST_ROW` at `:26-31` holds five reference
values per body size, verified byte-identical against upstream on 2026-05-11 per the comment at
`:23-25`.

| Body | angle_rad | t_Al_cm | t_graphite | t_Cu | t_Ti |
|---|---|---|---|---|---|
| large | -0.479582 | 3.711864 | 0.0 | 0.0 | 0.0 |
| medium | -0.479582 | 3.537235 | 0.0 | 0.0 | 0.0 |
| small | -0.479852 | 3.547008 | 0.0 | 0.0 | 0.0 |

All five compared at `atol = 1.0e-6`, `test/bowtie.jl:71-75`. Also asserted: file has exactly 891
lines, 3 header plus 888 data, `:65-66`; first line contains "GE Precision HealthCare" and
"xcist", `:60-61`.

**Shape parity with `test_Xray_Filter.py`**, `test/bowtie.jl:180-221`. Clinical-scale scanner,
900 cols, 16 rows, about 25 degree fan; 20-bin energy grid 20 to 120 keV.

| Check | Tolerance | Line |
|---|---|---|
| `size(trans) == (900, 16, 20)` | exact | `:189` |
| Transmittance is a probability | `all(>(0.0))`, `all(<=(1.0))`, `all(isfinite)` | `:192-194` |
| Max near but below 1 | `0.85 < maximum(trans) < 1.0` | `:197` |
| Edge attenuation, about 3.5 cm Al | `minimum(trans) < 0.1` | `:199` |
| Energy monotonicity at centre pixel | `issorted(center_E_profile)` | `:206` |
| Cone-angle correction, `t / cos(alpha)` | edge rows below middle row | `:214-215` |
| Fan modulation | centre col above both edge cols | `:219-220` |
| `:none` collapses to unity | `all(trans .≈ 1.0)` | `:231` |
| Three bundled profiles distinct and physical | `0.7 < t < 1.0` each, all three `!=` | `:265-272` |

`test/correction.jl` header at `:1-12` names `test_Prep_BHC_Accurate.py` and `test_PrepView.py`
as its upstream models.

`test/detector.jl:288-293` documents the CatSim `Xray_Filter` port relationship for the detector
side. `src/projection/dd.jl:5-15` names the exact upstream file being ported:
`clib_build/src/DD3Proj_roi_notrans_mm.cpp` from gecatsim, with the De Man and Basu 2004 reference.
`src/projection/siddon.jl:1-11` names TIGRE as its upstream.

### 2.10 Detector Monte Carlo LUT provenance oracles

`test/detector.jl:286-440`. These are verbatim published-value pins, the closest thing in the repo
to golden reference data.

| LUT | Index | Value | Line |
|---|---|---|---|
| `UFC_MC_EFFICIENCY_LUT` (Force) | 1 | 9.90863305e-01 | `test/detector.jl:303` |
| | 50 | 9.69026725e-01 | `:304` |
| | 51 | 7.41154644e-01 | `:305` |
| | 140 | 8.15971281e-01 | `:306` |
| `UFC_FLASH_MC_EFFICIENCY_LUT` | 1 | 9.90863305e-01 | `:386` |
| | 50 | 9.24938847e-01 | `:387` |
| | 51 | 7.39476107e-01 | `:388` |
| | 140 | 5.87915529e-01 | `:389` |

Sources named in the comments: Khodajou-Chokami `efficiency_results.csv` and
`flash_efficiency_results.csv`. Both LUTs are 140 points, asserted at `:301-302` and `:384-385`.

Derived physics assertions:

| Check | Tolerance | Line |
|---|---|---|
| Gd K-edge drop at 50.24 keV, Force | `eta50 > 0.95`, `eta51 < 0.78`, `eta50 - eta51 > 0.20` | `:311-317` |
| Gd K-edge drop, Flash | `eta50 > 0.90`, `eta51 < 0.78`, `eta50 - eta51 > 0.15` | `:394-400` |
| Gd L-edge dip near 8 keV | `eta(8) < eta(7)` | `:321`, `:404` |
| Interpolation bracketed | `lo - 1e-12 < eta(60.5) < hi + 1e-12` | `:325-329`, `:408-412` |
| Clamping below 1 and above 140 keV | `==` LUT endpoints | `:331-332`, `:414-415` |
| **Force and Flash are distinct detectors** | `flash(140) < 0.65`, `force(140) > 0.80`, `flash(100) < force(100)` | `:427-429` |
| Beer-Lambert path contradicts MC at the K-edge, by design | `eta[3] > eta[2]` | `:352`, `:437` |
| Vector API agrees with scalar | `eta ≈ [get_*(E) for E in energies]` | `:272`, `:339`, `:430` |

Detector material properties are pinned too: CdTe density 5.85, K-edges `[26.711, 31.814]` at
`:661-662`; CZT 5.78 at `:667`; Si 2.33 at `:671`.

The archived Brent parity test, `test/archived/vmi_brent_parity.jl:1-11`, states the philosophy
you should copy: "Roots.jl is the authoritative reference — our port must match its trajectory
bit-exactly up to tolerance." Its tolerance is `4 * eps(Float64) * max(1.0, abs(root))`, about 4
ULP, at `:21` and `:31`, plus residual `abs(f(r)) <= 1e-10` at `:34`. It is not in `runtests.jl`
because it needs `Roots` as a dependency.

### 2.11 Other exactness contracts

| Check | Tolerance | Line |
|---|---|---|
| `create_mu_volume!` eltype respected, Float32 vs Float64 agree | `Float32(mu64) ≈ mu32` | `test/projection.jl:420` |
| EICT noise seed reproducibility | identical seeds give identical sinograms | `test/api.jl:694-704` |
| Noise-floor RNG isolation | does not disturb `default_rng` | `test/api.jl:719-736` |
| `_log_factorial` vs `sum(log, 1:n)` | `rtol = 1.0e-12` at n = 21 and 150 | `test/detector.jl:766-767` |
| Poisson sampler mean and variance | `rtol = 0.02` and `0.05` | `test/detector.jl:773-774` |
| Poisson third moment vs `1/sqrt(lambda)` | `atol = 0.03` | `test/detector.jl:781` |
| PCCT noise sampler mean and variance | `rtol = 0.01` and `0.15` | `test/detector.jl:802-803` |
| Spatial binning, 2x2 dexels | `all(dst .≈ 4.0f0)` | `test/detector.jl:681` |
| PCCT covariance symmetry, correlation diagonal | `≈ transpose`, `diag ≈ ones(4)` | `test/detector.jl:743-745` |
| Denoising in-place equals allocating | max abs `< 1.0e-5` | `test/denoising.jl:230-231`, `:355-356` |
| Median-z identity at `adjacent_slices = 0` | `out ≈ vol` | `test/denoising.jl:149` |
| SVD passthrough at `sigma_px = 0` | copy | `test/denoising.jl:188-193` |
| ACNR identity at `gamma = 0` | `test/denoising.jl:349-357` |
| SF-JSD edge preservation, flat ROI mean | abs `< 0.05` | `test/denoising.jl:104-105` |
| Memory release counts and budget guard | `release_backend!` returns 2, budget throws | `test/memory_lifecycle.jl:37-53` |

### 2.12 Notebook PASS/FAIL certification cells

A second oracle layer lives in `/Users/daleblack/Documents/dev/MolloiLab/BasisSimulator.jl/docs/notebooks/`.
Twelve notebooks; seven carry coded gates.

| NB | Gate cell | Gates | Structure | Recorded verdict |
|---|---|---|---|---|
| 01 | `01_five_struct_api.jl:896-1013` | 5 | `checks::Vector{NamedTuple}` + `addcheck(name, val, lo, hi)` + Markdown table | PASS 5/5, 14/14 rods |
| 02 | none | 0 | descriptive sigma table only | |
| 03 | `03_dual_kvp_switching_vmi.jl:1320-1352` | 5 | `checks = NamedTuple[]` + `addcheck(name, value, pass)` | PASS 5/5 |
| 04 | `04_pcct_vmi.jl:1537-1562` | 5 | same idiom | PASS 5/5 |
| 05 | `05_xcat_grid_to_recon.jl:1374-1424` and `:1425-1467` | 0 coded | two `let` blocks printing numeric tables | see below |
| 06 | none | 0 | qualitative mosaic + runtime table | runtime only |
| 07 | `07_qrm_thorax_pure_material_vmi.jl:2141-2191` | 6 | literal `checks = [...]` array | PASS 6/6 |
| 08 | `08_qrm_thorax_pure_material_pcct.jl:2372-2439` | 8 | literal `checks = [...]` array | PASS 8/8 |
| 09 | none | 0 | regression figure legends only | |
| 10 | none | 0 | 3-row HU table + prose scope | |
| 11 | none | 0 | prose "Verification and Scope" + z-profile figure | |
| 12 | `12_siemens_flash_ufc.jl:2172-2250` | **21** | `checks::Vector{Tuple{String,Bool,String}}`, `n_pass == length(checks)` | PASS 21/21 |

**Notebook 01 gates**, `01_five_struct_api.jl`:

| Gate | Quantity | Bound | Line | Measured |
|---|---|---|---|---|
| 1 | water mean HU, corrected, central slice, 1-px eroded mask | `[-6.0, 6.0]` | `:934-935` | 1.7 |
| 2 | noise ratio low/std, water ROI, 200 mA vs 50 mA | `[1.7, 2.3]`, theory 2.0 | `:936-937` | 2.1 |
| 3 | radial cupping `cup_hu`, worst slice | `[0.0, 12.0]` | `:944-945` | 9.9 |
| 4 | DC offset `abs(dc_hu)`, worst slice | `[0.0, 6.0]` | `:946` | 5.7 |
| 5 | rods passing both sub-gates | exactly 14 of 14 | `:987` | 14 |

Per-rod sub-gates at `:977-978`: theory match
`abs(meas - theory) <= max(15.0, 0.15 * abs(theory))`, dose invariance
`abs(meas - meas_low) <= max(10.0, 0.03 * abs(theory))`. Theory is mono HU at the BHC reference
energy, 69.7 keV in the certified run. Rods with fewer than 20 eroded pixels are skipped at `:963`.
Worst iodine rod measured -13.8 percent against a 15 percent gate, and DC offset 5.7 against 6.0.
Those are the two thinnest margins in the gallery.

Fixture: Gammex 512x512x16, fov 35 cm, z 1.0; scanner SID 625.6, SDD 1100.0, 256 rows x 834 cols,
0.625 x 0.6 mm, Lumex 3.0 mm, GE large bowtie, electronic noise 3500 e-; 120 kVp, 200 mA and
50 mA, 500 views, 1.0 s, 5 mm collimation; recon 512x512x8, fov 35, z 0.5.

**Notebook 12 gates**, `12_siemens_flash_ufc.jl:2172-2250`, 21 total:

| Gates | Quantity | Bound | Line | Measured |
|---|---|---|---|---|
| 1-3 | regular water tube A, tube B, combined mean HU | `abs <= 5.0` | `:2176-2183` | 1.12, 1.07, 1.09 |
| 4 | dual-power sigma ratio, ideal 0.707 | `[0.62, 0.80]` | `:2185-2189` | 0.706 |
| 5-7 | DE poly water 100 kVp, Sn140 kVp, mixed | `abs <= 5.0` | `:2191-2199` | 1.25, 0.15, 0.7 |
| 8-11 | VMI water at 50, 70, 100, 140 keV | `abs <= 10.0` | `:2201-2207` | 2.01, 0.3, -0.58, -0.95 |
| 12 | VMI noise monotonic decrease | `all(diff(sigmas) .< 0)`, **hard, no tolerance** | `:2208-2214` | 88.6 > 77.3 > 73.8 > 73.0 |
| 13 | per-basis FBP kernels | exact tuple equality | `:2216-2221` | matched |
| 14-21 | Ca and I regression at 4 keV each | `0.85 <= slope <= 1.15` and `r2 >= 0.99` | `:2223-2241` | slopes 0.953 to 0.995, r2 0.9942 to 1.0 |

Gate 4 exists to catch shared RNG seeds between the two tubes: tube A `seed = 1234` at `:391`,
tube B `seed = 4321` at `:402`. Tightest margin is I regression at 140 keV, r2 = 0.9942 against a
0.99 gate.

Fixture: Gammex 512x512x16 fov 35 z 1.0; Siemens Flash 64 rows x 736 cols, `:ufc_flash` MC LUT;
regular 120 kVp 420 mA 1152 views run twice; DE 100 kVp 460 mA and 140 kVp 356 mA with 0.4 mm Sn;
recon 512x512x5 fov 35 z 0.30; rod ROI radius 8 px.

**Notebooks 03 and 04 gates:** bin count exact, `all(isfinite)`, solid-water worst absolute HU
`<= 10` over a 12-px eroded mask, hard monotonic noise decrease, exact FBP kernel symbol equality.
Measured worst water HU 3.14 in nb03 and 3.27 in nb04; nb03 noise
`Float32[54.28, 36.37, 32.3, 32.23]`.

**Notebooks 07 and 08 gates:** `I0_relerr < 5e-5` at `07:2153-2157` and `08:2384-2388`, measured
1.33e-8 and 0.0; basis calibration `0.8 <= c_water <= 1.2` and `abs(c_iodine) <= 0.01` at
`08:2408-2417`, measured `(0.9999, -1.0e-5)`; canonical VMI energies exactly
`[50.0, 70.0, 100.0, 140.0]`. Notebook 07 has **no water-HU gate and no monotonic-noise gate**, a
coverage gap relative to its sibling 08.

**Notebook 05, the affine 1-to-1 claim.** `05_xcat_grid_to_recon.jl:1374-1424` runs
`affine_audit(geom, ms)` comparing `(A*v)[ax]` against the backprojector's own rule
`-fov[ax]/2 + (idx - 0.5) * (fov[ax]/ms[ax])`, reported in micrometres, and
`abs((Ainv*(A*v))[ax] - k)` in voxel index units, over three indices per axis for both axial and
helical geometries.

| Geometry | Grid agreement | Round trip |
|---|---|---|
| axial | 8.9e-12 micrometres | 0.0 voxel |
| helical | 8.9e-12 micrometres | 0.0 voxel |

Registration cell at `:1425-1467` cross-correlates a Sobel edge map against a label-boundary map
over integer shifts in -5:5; both geometries return best shift `(0, 0)` with score 1.0.
**There is no coded threshold.** The "machine precision" verdict is prose at `:1417-1420`.
8.9e-12 micrometres is 8.9e-16 cm, roughly 1 ulp of Float64 at cm scale.

**Notebook 06, CatSim versus BasisSim.** **There is no numeric agreement gate anywhere in this
notebook.** No `checks`, no PASS/FAIL, no RMSE, no correlation, no tolerance. The comparison is
three mid-slice images at a shared HU window `colorrange = (-200, 600)` at `06:908, 923, 937`, plus
a wallclock table. The notebook title itself says "Qualitative Match and Runtime" at `06:32`.

What is matched to make the comparison fair, which is where the rigor actually lives: noise
disabled in all three runs, `use_noise = false` at `06:359-361` and `electronic_noise = 0` at
`06:325`; a shared BHC via `bhc_120` at `06:722-729` with `mu_water_ref` passed into CatSim's own
recon config at `06:466`; detector-geometry convention conversion by magnification at `06:301-303`;
a `REGION_TO_CATSIM` label-to-material map at `06:560`.

**Notebook 11, helical.** No coded gates. The "Verification and Scope" section at `11:437-467` is
a prose claim list: `:dd_fast` needed zero changes; recon is rebinned WFBP after Stierstorfer 2004
with cos-squared aperture weighting and plateau Q = 0.7; exposure matched by beam-width times
current product; honest limits stated as 128 or fewer active rows and pitch 1.5 or less. Empirical
evidence is a z-profile figure at `11:381-405` over a fixed ROI `xr, yr = 25:36, 73:88`, 150
slices, station boundaries marked at z = plus and minus 5.0 cm. **The cached output is stale**: it
says 2 stations where `11:271` uses `station_zs = [-10.0, 0.0, 10.0]`, 3 stations.

**Notebook 10, titanium implant.** No gates by design. `10:246-259` states the bundled BHC is a
water polynomial and "will not remove these bands", and "no MAR algorithm ships with this release".
Its 3-row measurement table at `10:228-241` is explicitly framed at `10:270-271` as "the
quantitative table is the regression target to preserve when a future MAR method is added".

Cross-cutting: slope and r-squared are computed in notebooks 03, 04, 07, 08, and 09 but **gated
only in 12**. Elsewhere they appear only in Makie plot legends, e.g. `03:1300-1311` and
`09:1465-1481`, so a regression drift there fails nothing.

---

## 3. Fixtures and reference data

**There are no saved reference arrays, no JLD2 files, no `.mat` files, and no golden-output files
in the test tree.** Every fixture is constructed in-process from scanner and protocol structs plus
`create_gammex_472`. This is the single biggest gap for an oracle harness: the oracle has to be
captured, it cannot be loaded.

What does exist:

### 3.1 `docs/notebooks/DATA_PROVENANCE.sha256`

12 SHA-256 entries. Header at `:1-6` explains: "The raw files are intentionally not committed; this
declaration is committed so local exports and CI compute the same export fingerprint. When any raw
dataset is present locally, the exporter verifies every present file against this list before
rendering."

| Entry | Path |
|---|---|
| `44760c2e...` | `qrm_thorax/qrm_thorax_1600x1100_rot_uint8.raw` |
| `e27367ab...` | `qrm_thorax/qrm_thorax_3200x2200_rot_uint8.raw` |
| `f9952b0a...` | `qrm_thorax/qrm_thorax_md4_1100x1600_uint8.raw` |
| `6b37bb28...` | `qrm_thorax/qrm_thorax_md4_1500x1500_uint8.raw` |
| `4d3337fe...` | `ufc_flash_mc_efficiency_v1.PROVENANCE.txt` |
| `627cb101...` | `ufc_flash_mc_efficiency_v1.csv` |
| `b2586ede...` | `ufc_mc_efficiency_v1.PROVENANCE.txt` |
| `a450dde4...` | `ufc_mc_efficiency_v1.csv` |
| `39cd8fe3...` | `xcat/Material_Spreadsheets/vmale_50_materials_heart_high_contrast.xlsx` |
| `18504e6f...` | `..._heart_low_contrast.xlsx` |
| `90eda603...` | `..._heart_non_contrast.xlsx` |
| `b80446af...` | `xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin` |

`docs/notebooks/data/` is gitignored at `.gitignore:47`. All 12 files are present locally.

### 3.2 Binary and text assets under `src/`

| Asset | Size | Loader |
|---|---|---|
| `src/detector/pcct/cdte_response_v4.jls` | 7.55 MB | `default_mc_drm_path()` at `src/detector/pcct/mc_response.jl:36`, read by `load_mc_response` at `:86` |
| `src/bowtie/{large,medium,small}.txt` | 888 data rows each | `load_catsim_bowtie`, `load_builtin_bowtie` |
| `src/spectrum/tungsten_tar{7.0,10.0}_{70..140}_filt.dat` | 16 files | `load_spectrum(kVp; source = :xspect, target_angle = ...)` |
| `src/spectrum/Anode{8,10}/{80,100,120,140}.TXT` | 8 files | same |
| `src/spectrum/xcist_kVp{80,100,120,140}_tar7_bin1.dat` | 4 files | `load_spectrum(kVp; source = :xcist)` |
| `src/spectrum/{convert.py, XCISTspectrum.m}` | | generation scripts, not runtime |

### 3.3 XCAT phantom artifacts

`src/phantoms/xcat_artifacts.jl`. Four phantoms in `XCAT_REGISTRY` at `:39`:
`:female_chest, :female_slab, :male_chest, :male_slab`, enumerated and asserted at
`test/phantoms.jl:12`.

Each `XCATPhantomEntry` carries `sha256` for download integrity and `tree_hash` as the artifact
store key, `:33-34`. Flow at `:149-181`: check registry, check
`Artifacts.artifact_exists(SHA1(tree_hash))` and return early if present, otherwise download from
`_XCAT_BASE_URL`, verify SHA-256 and `error` on mismatch at `:172-173`, extract into a staging
dir under `~/.julia/artifacts/`, and handle the concurrent-download race at `:179-180`. Citation
`@info` block prints Segars 2010 Medical Physics 37(9):4902-4915 and Wu 2022 PMB 67, source
github.com/xcist/phantoms-voxelized, BSD-3-Clause.

`test/phantoms.jl:22-25` asserts every registry entry has `length(sha256) == 64` and
`length(tree_hash) == 40`. `test/phantoms.jl:29-60` parses a hand-computed synthetic XCIST
voxelized phantom offline: two 2x2x1 material boxes offset by one voxel, asserting the assembled
3x3x1 grid has 3 water voxels, 4 muscle, 2 air. No network access in the test.

### 3.4 Gammex 472

Generated, not stored. `create_gammex_472(n_voxels, n_slices, fov_cm, z_cm)`. 14 rods: 7 calcium
`Ca_50` through `Ca_600` and 7 iodine `I_2_0` through `I_20_0`, plus `solid_water`, enumerated at
`test/object.jl:16-19`. `get_region_materials()` returns exactly 27 entries, asserted at
`test/object.jl:38`; index 1 is air, 3 is water, 4 is Gammex solid water.

### 3.5 Scanner dossiers

`SCANNERS.md`, 4778 bytes, covers GE Revolution Apex Elite with FDA K213715, Siemens NAEOTOM Alpha
K201501 in both standard 2x2 and UHR 1x1 modes, Canon Aquilion ONE as a legacy reference, and
Siemens SOMATOM Definition Flash K082220 with Stellar variant K113342. Header at `:3-4` says these
"were previously implemented as factory functions and have been moved here for reference."

`docs/scanner_dossiers/somatom_definition_flash.md`, 23257 bytes, is the only deep dossier:
geometry, tubes, filtration, protocols, and documented-assumption gaps. `SCANNERS.md:112-114`
records the key discriminator your tests pin: the Flash LUT is distinct from the Force UFC LUT,
same Gd2O2S material, thinner crystal, efficiency 28 percent lower at 140 keV.

### 3.6 The clinical reference library that does not exist

`test/archived/references/README.md` describes a planned structure: `catsim/` as an XCIST baseline
modeling GE Revolution Apex geometry, plus five clinical scanner folders, each with
`rod_measurements.csv` of 16 rows with `mean_hu, std_hu, cnr, n_pixels, cx, cy`, plus `nps.csv`,
`mtf.csv`, `metadata.txt`, and a master `summary.csv`.

**Every row in its status table at `:62-69` says "Planned". Nothing has been captured.** The
directory contains only the README.

The worked validation example at `:127-153` proposes the tolerances a future harness would use:
per-rod `abs(basis.mean_hu - ref.mean_hu) < 10.0` HU at `:141`, and background noise
`abs(basis - ref) / ref < 0.2`, 20 percent, at `:145`.

---

## 4. Benchmark scripts

**`scripts/` exists and is empty.** There is no BenchmarkTools dependency in `Project.toml`,
`docs/Project.toml`, or any Manifest. There is no `@btime`, `@benchmark`, or `@belapsed` call in
any tracked file. All timing is `@elapsed` inside Pluto notebooks.

### 4.1 Every timing site

| Location | What is timed |
|---|---|
| `docs/notebooks/06_catsim_vs_basissim.jl:752` | CatSim Python run, end to end |
| `docs/notebooks/06_catsim_vs_basissim.jl:796` | BasisSim CPU, forward + BHC + FDK |
| `docs/notebooks/06_catsim_vs_basissim.jl:841` | BasisSim GPU (MtlArray), same |
| `docs/notebooks/11_helical_scanning.jl:269, 276` | helical simulate + corrected recon |
| `docs/notebooks/11_helical_scanning.jl:311, 318` | per-station axial simulate + recon, accumulated |
| `docs/notebooks/03_dual_kvp_switching_vmi.jl:838` | per-tile spectral forward loop |
| `docs/notebooks/04_pcct_vmi.jl:892` | same |
| `docs/notebooks/07_qrm_thorax_pure_material_vmi.jl:1185` | same |
| `docs/notebooks/08_qrm_thorax_pure_material_pcct.jl:1222` | same |
| `docs/notebooks/12_siemens_flash_ufc.jl:1391` | same |
| `test/memory_stability_pcct_representative.jl:117` | per-cycle create + simulate + release |

### 4.2 Reported numbers

**Notebook 06 committed cache**, `06_catsim_vs_basissim.jl.pluto-cache.toml`, result hash
`3c6a3371e468835e`. Table built at `06:1046-1072`.

| Pipeline | Wallclock | Speedup vs CatSim |
|---|---|---|
| CatSim, Python, voxelized projector | 76.60 s | 1.00x reference |
| BasisSim CPU | 6.95 s | 11.02x |
| BasisSim GPU, MtlArray | 0.35 s | 217.67x |

Caveat text in the notebook: "Both BasisSim runs were JIT-warmed once before the timing pass" and
"reported is steady-state hot-cache wallclock, comparable to CatSim's C-kernel runtime".
Fixture: Gammex 128x128x8 at about 2.7 mm voxels, 120 kVp, 200 mA, 500 views, 4.0 mm collimation,
recon 256x256x6, fov 35 cm.

**Published paper**, `paper/main.tex:263` and the figure caption at `:268`. Same benchmark, same
machine class.

| Pipeline | Wallclock | Speedup |
|---|---|---|
| CatSim, Python | 73.85 s | baseline |
| BasisSimulator CPU, multi-threaded | 7.71 s | 9.6x |
| BasisSimulator Apple Metal | 1.18 s | 62.6x |

Machine stated in the caption: MacBook Pro, Apple M4, 10-core CPU with 4 performance and 6
efficiency cores, 10-core GPU, 16 GB unified memory, macOS 26.3, Julia 1.12.4, Metal.jl 1.9.3.
The caption also disclaims the GPU figure: "this figure conflates GPU-versus-CPU, parallelism, and
Julia-versus-Python language overhead and is not a single-axis speedup."

**The cache and the paper disagree by a factor of 3.4 on the Metal number, 0.35 s versus 1.18 s,
and by 3.5x on the speedup, 217.67x versus 62.6x.** Treat neither as a stable baseline.

**Notebook 11 cache**: "helical (5760 views, full EICT physics + corrections) about 14.4 s; volume
axial (2 stations x 360 views) about 6.2 s." Stale, see section 2.12.

### 4.3 The dd_fast numbers your team cares about

These are documented in source comments and a commit body. **They are not reproducible from any
committed script.** You will have to rebuild the harness.

`src/projection/dd_fast.jl:22-26`:

> Measured (M4 Metal, 234-bin polychromatic forward, 512^2 x 64 vol, 736 x 16 x 720 sino):
> 113.8 s (:dd tiled hosts) -> 2.4 s (:dd_fast single-pass), agreement mean_rel ~ 5e-7, and
> 4.4x faster than Siddon's tiled 234-bin path. UHR slab (1024^2 x 32): 109.3 s -> 5.3 s.

Commit `8d072da` body adds the Siddon absolute number and the UHR speedup:

| Case | `:dd` tiled | `:dd_fast` | Speedup |
|---|---|---|---|
| 512^2 x 64 vol, 736 x 16 x 720 sino, 234 bins | 113.8 s | 2.46 s | about 47x |
| 1024^2 x 32 UHR slab | 109.3 s | 5.3 s | about 21x |
| `:siddon` tiled, 234 bins, same 512 case | 10.6 s | | `:dd_fast` is 4.4x faster |

Agreement stated as mean relative about 5e-7.

`src/projection/select_projector.jl:15-16` repeats "Measured 47x faster on the 234-bin
polychromatic forward path (M4 Metal)". The runtime deprecation warning at `:47-52` repeats
"~47x faster polychromatic forward" and fires with `maxlog = 1` whenever `:dd` is selected.

The mechanism is documented at `src/projection/dd_fast.jl:6-19`. Legacy `:dd` accumulates an
`NTuple{N_E}` of per-energy sums, costing `N_E` FMAs, `N_E` table reads, and `N_E` live registers
per overlapped voxel, which spills at clinical `N_E ≈ 234` and forced the host-side K = 16 energy
tiling that re-walks the volume `n_tiles` times. `:dd_fast` reassociates by linearity:
`L_e = sum_m mu[m, e] * P_m` where `P_m = sum over voxels of material m of w_v`, so one volume walk
accumulates per-material path lengths in at most 64 registers and every energy converts once per
detector cell. Register-budget cap is `_PLEN_MAX_MATERIALS = 64`. The K = 16 tile constant still
lives at `src/projection/polychromatic.jl:527`.

The HIR speedup entry in `CHANGELOG.md:169` records "32x faster HIR, percentage strength kwarg,
counts-domain DAS noise", commit `87e9564`. No script backs it either.

---

## 5. Numerical tolerances used throughout

Grouped by precision tier so you can read off the accepted noise floor per stage.

### 5.1 Machine-precision tier, 1e-12 and tighter

| Tolerance | Check | Location |
|---|---|---|
| `rtol = 2.0e-12`, `atol = 2.0e-12` | brute-force transpose matrix oracle | `test/projection.jl:94` |
| `rtol = 2.0e-11`, `atol = 2.0e-11` | DD adjoint dot product, Float64 | `test/projection.jl:56` |
| `rtol = 1.0e-12` | pileup count round trip, `I0*exp(-bin)` | `test/api.jl:561` |
| `rtol = 1.0e-12` | pileup air offset equals `log(I0_truth/I0_recorded)` | `test/api.jl:568` |
| `rtol = 1.0e-12` | `_log_factorial(21)` and `(150)` vs `sum(log, 1:n)` | `test/detector.jl:766-767` |
| `< 1.0e-12` | max deviation of angle spacing from `2pi/n` | `test/geometry.jl:200` |
| `< 1.0e-12` | u-axis unit norm | `test/geometry.jl:230` |
| `< 1.0e-12` | u perpendicular to v | `test/geometry.jl:231` |
| `< 1e-12` | helical uniform dz | `test/geometry.jl:441` |
| `atol = 1.0e-12` | scatter reference coefficient | `test/detector.jl:79` |
| `atol = 1.0e-12` | scatter kernel sums to 1, two kernels | `test/detector.jl:101, 111` |
| `atol = 1.0e-12` | bin weights sum to 1 | `test/detector.jl:173` |
| `atol = 1.0e-12` | Lumex mu equals Gemstone mu at 60 keV | `test/detector.jl:481` |
| `atol = 1.0e-12` | unknown material falls back to Gemstone | `test/detector.jl:485` |
| `atol = 1.0e-12` | effective fill factor, three cases | `test/detector.jl:494, 501, 506` |
| `atol = 1.0e-12` | crosstalk kernel sums to 1 and centre value | `test/detector.jl:544-545` |
| `atol = 1.0e-12` | lag amplitudes sum to 0.015 | `test/detector.jl:574` |
| `atol = 1.0e-12` | lag coefficients sum to 1 | `test/detector.jl:581` |
| `atol = 1.0e-12` | quantum efficiency vector vs scalar | `test/detector.jl:655` |
| `atol = 1.0e-12` | focal spot kernel symmetry, both axes | `test/source.jl:358-359` |
| `atol = 1.0e-12` | focal spot weights sum to 1 | `test/source.jl:443` |
| `atol = 1.0e-12 * w[i]` | filtered spectrum Beer-Lambert per bin | `test/source.jl:136` |
| `± 1.0e-12` bracket | MC LUT interpolation containment, three LUTs | `test/detector.jl:257-259, 327, 410` |
| `atol = 1.0e-10` | mu-to-HU at water and offsets | `test/object.jl:74, 80` |
| `atol = 1.0e-10` | source on circle of radius SAD | `test/geometry.jl:207` |
| `atol = 1.0e-10` | detector centre distance | `test/geometry.jl:217` |
| `atol = 1e-9` | helical last angle | `test/geometry.jl:432` |
| `atol = 1e-9` | helical last source z | `test/geometry.jl:436` |
| `atol = 1.0e-9` | BHC per-column weight normalization sums to 1 | `test/correction.jl:385` |
| `atol = 1.0e-9` | water calibration curve first point is 0 | `test/correction.jl:187` |
| `< 1.0e-9` | single-energy calibration, measured equals true | `test/correction.jl:193` |
| `< 1.0e-8` | cubic polynomial coefficient recovery | `test/correction.jl:172` |

### 5.2 Float32 machine tier, 1e-7 to 1e-4

| Tolerance | Check | Location |
|---|---|---|
| `rtol = 8eps(Float32)` | raw-count snapshot round trip, CPU | `test/api.jl:352` |
| `rtol = 8eps(Float32)`, `atol = 1.0e-6` | raw-count round trip, GPU path | `test/api.jl:396` |
| `rtol = 2.0f-6`, `atol = 2.0f-6` | HIR `W_proj` vs closed form | `test/api.jl:1349` |
| `rtol = 2.0e-5` | HIR terminal-slice symmetry | `test/api.jl:1354` |
| `rtol = 1.0e-5` | spectrum sums to 1, four sites | `test/api.jl:801, 825, 916, 1598` |
| `rtol = 1.0e-4` | per-ray bowtie spectrum sums to 1 | `test/api.jl:840` |
| `atol = 1.0e-6 * sum(w)` | spectrum downsample preserves total | `test/source.jl:107, 117` |
| `< 1.0e-6` | BHC `Vector{BHCPolynomial}` vs `Vector{BeamHardeningCorrection}` dispatch | `test/correction.jl:257` |
| `< 1.0e-5` | pileup identity matrix is a no-op | `test/correction.jl:158` |
| `< 1.0e-5` | image-domain BHC in-place contract | `test/correction.jl:449` |
| `< 1.0e-5` | DD in-place equals allocating | `test/projection.jl:196` |
| `< 1.0e-5` | resample identity grid, linear method | `test/geometry.jl:376` |
| `< 1.0e-5` | SVD denoise in-place equals allocating, 2 channels | `test/denoising.jl:230-231` |
| `< 1.0e-5` | ACNR gamma = 0 identity, 2 channels | `test/denoising.jl:355-356` |
| `< 1.0f-4` | **`:dd_fast` equals `:dd`, fused poly** | `test/projection.jl:301` |
| `< 1.0f-4` | **`:dd_fast` equals `:dd`, fused spectral** | `test/projection.jl:336` |
| `< 1.0f-4` | **`:dd_fast` equals `:dd`, helical arc + bowtie** | `test/projection.jl:361` |
| `< 1.0f-4` | **`:dd_fast` equals `:dd` on the arc** | `test/projection.jl:557` |
| `+ 1.0f-4` slack | `:dd_fast` Siddon envelope vs `:dd` | `test/projection.jl:319` |
| `< 1.0e-4` | Siddon linearity in mu | `test/projection.jl:132` |
| `< 1.0e-4` | Siddon in-place equals allocating | `test/projection.jl:148` |
| `< 1.0e-4` | DD linearity in mu | `test/projection.jl:201` |
| `+ 1.0e-3` slack | Siddon max line-integral bound | `test/projection.jl:124` |
| `atol = 1.0e-4` | HU conversion array path, four values | `test/object.jl:98-101` |
| `atol = 1.0e-4` | HU conversion with explicit calibration | `test/object.jl:116` |
| `atol = 1.0e-4` | BHC constant term on a zero sinogram | `test/correction.jl:247` |
| `atol = 1.0e-4` | focal-spot convolution preserves mass | `test/source.jl:405` |
| `atol = 1.0e-3` | HU conversion Float32 vs Float64 | `test/object.jl:110` |
| `< 1.0e-3` | pileup correction round trip | `test/correction.jl:147` |
| `atol = 1.0e-6` | fill factor applied to a flat sinogram | `test/detector.jl:513` |
| `atol = 2f-5` | PCCT bins after noise, low-rate limit | `test/detector.jl:816` |

### 5.3 Model-agreement tier, percent-level

| Tolerance | Check | Location |
|---|---|---|
| `< 0.005` mean | arc resample oracle, mean relative | `test/projection.jl:525` |
| `< 0.05` p99 | arc resample oracle, 99th percentile | `test/projection.jl:526` |
| `< 0.10` max | arc resample oracle, worst | `test/projection.jl:527` |
| `< 0.01` mean | DD vs Siddon, mono | `test/projection.jl:189` |
| `< 0.05` max | DD vs Siddon, worst ray | `test/projection.jl:190` |
| `< 0.01` mean | DD vs Siddon, fused poly | `test/projection.jl:238` |
| `< 0.05` max | DD vs Siddon, fused poly worst | `test/projection.jl:239` |
| `< 0.02` mean | DD vs Siddon, fused spectral per bin | `test/projection.jl:262` |
| `< 0.01` mean | `:dd_fast` vs Siddon | `test/projection.jl:318` |
| `< 0.01` max | arc vs flat, central columns | `test/projection.jl:534` |
| `1.3x + 0.002` | arc Siddon envelope vs flat | `test/projection.jl:547` |
| `rtol = 0.05` | `:dd` vs `:siddon` mid-ray closeness | `test/projection.jl:460` |
| `< 0.05` | polynomial BHC residual on the water curve | `test/correction.jl:216` |
| `rtol = 0.01` | HIR terminal vs mid slice | `test/api.jl:1355` |
| `rtol = 0.01` | FDK terminal vs mid slice | `test/api.jl:1360` |
| `rtol = 0.1`, `atol = 5.0e-4` | HIR short-object terminal symmetry | `test/api.jl:1396` |
| `< 0.9x` | HIR damps high-pass texture vs FDK | `test/api.jl:1382` |
| `< 1.25` | HIR terminal-to-mid texture ratio | `test/api.jl:1383` |
| `< 0.01` | helical FDK mu accuracy vs 0.2 | `test/geometry.jl:507` |
| `< 0.02` | helical FDK z-uniformity spread | `test/geometry.jl:511` |
| `< 0.005` | helical vs axial parity | `test/geometry.jl:519` |
| `< 0.01` | arc axial FDK mu accuracy | `test/geometry.jl:548` |
| `< 0.008` | arc radial flatness centre vs off-centre | `test/geometry.jl:550` |
| `< 0.012` | arc helical mean mu accuracy | `test/geometry.jl:560` |
| `< 0.015` | arc helical banding spread | `test/geometry.jl:561` |
| `< 0.05` | SF-JSD flat-ROI mean preservation, 2 channels | `test/denoising.jl:104-105` |

### 5.4 Statistical tier

| Tolerance | Check | Location |
|---|---|---|
| `rtol = 0.01` | PCCT noise sampler mean | `test/detector.jl:802` |
| `rtol = 0.02` | Poisson sampler mean | `test/detector.jl:773` |
| `rtol = 2.0e-2` | PCCT bin means, noise on vs off | `test/api.jl:1849` |
| `atol = 0.03` | Poisson third moment vs `1/sqrt(lambda)` | `test/detector.jl:781` |
| `rtol = 0.05` | Poisson sampler variance | `test/detector.jl:774` |
| `rtol = 5.0e-2` | Gaussian noise-floor empirical std, 64^3 | `test/api.jl:689` |
| `atol = 1.0` | Gaussian noise-floor mean is zero | `test/api.jl:690, 714` |
| `rtol = 1.0e-1` | noise-floor std on a smaller volume | `test/api.jl:713` |
| `rtol = 0.15` | PCCT noise sampler variance | `test/detector.jl:803` |
| `rtol = 0.15` | **dose sweep, `sigma_lo/sigma_hi ≈ sqrt(800/50)`** | `test/api.jl:1920` |

### 5.5 Physical-bound tier, one-sided

| Bound | Check | Location |
|---|---|---|
| `count > 1000` | object illuminated, three sites | `test/projection.jl:188, 237, 317` |
| `count > 3000` | arc oracle interior mask | `test/projection.jl:524` |
| `> 0.5` keV | bowtie hardening, edge minus centre mean E; measured 16.3 | `test/api.jl:880` |
| `0.85 < max < 1.0` | bowtie transmission upper bound | `test/bowtie.jl:197` |
| `< 0.1` | bowtie transmission at the fan edge | `test/bowtie.jl:199` |
| `0.7 < t < 1.0` | centre transmission, all three bodies | `test/bowtie.jl:265-267` |
| `< 0.2`, `> 3.0` | medium bowtie Al thickness range in cm | `test/bowtie.jl:108-109` |
| `> 0.05` | pileup matrix column non-degeneracy | `test/api.jl:510` |
| `0 < s <= 1 + 1e-6` | pileup column sums | `test/api.jl:515` |
| `< 0.05` | pileup upward-only, `S[i,j]` for `i < j` | `test/api.jl:521` |
| `> 0.01` | pileup low-to-high leak `S[2,1]` | `test/api.jl:525` |
| `0.15 < mu < 0.3` | polychromatic water mu at three path lengths | `test/correction.jl:337-339` |
| `55.0 < E < 75.0` | BHC reference energy | `test/correction.jl:396` |
| `< 1.0` | radial cupping flattens to target HU | `test/correction.jl:499` |
| `0 < eta <= 1` | detector efficiency, two sites | `test/api.jl:624, 1594` |
| `32 <= len <= 2*n_cols - 1` | FBP filter kernel length | `test/api.jl:1664, 1684` |
| `> 5 * 2^30` | PCCT workspace GPU byte estimate at clinical scale | `test/memory_lifecycle.jl:44` |
| `> 1.0e-6` | pileup, scatter, focal-spot toggles measurably change output | `test/api.jl:662, 1794, 1843` |
| 16 MiB | Metal allocation drift, small soak | `test/memory_stability_pcct.jl:77` |
| 64 MiB | Metal allocation drift, clinical soak | `test/memory_stability_pcct_representative.jl:14` |

### 5.6 Exact-equality contracts

These are the strongest and the ones a rewrite most easily breaks.

| Check | Location |
|---|---|
| Arc row-tiled forward equals scalar kernel | `test/projection.jl:50` |
| Backprojection is deterministic across calls | `test/projection.jl:61` |
| Arc `active_z` tile equals full-range slice | `test/projection.jl:69` |
| Above-64-material fallback is bit-identical to `:dd` | `test/projection.jl:373` |
| `_project_mono` shim routes bit-identically, both projectors | `test/projection.jl:438-439` |
| `_project_mono!` shim routes bit-identically, both projectors | `test/projection.jl:446, 451` |
| FDK determinism across workspaces | `test/api.jl:1140` |
| FDK second call overwrites, no accumulation | `test/api.jl:1216` |
| HIR determinism across workspaces | `test/api.jl:1446` |
| HIR strength 0 equals the FDK workspace result | `test/api.jl:1305` |
| `resolve_source_spectrum_with_bowtie` equals manual composition | `test/api.jl:903-904` |
| Huber gradient is exactly zero on a z-uniform field | `test/api.jl:1423` |
| `:nearest` resample preserves integer labels | `test/geometry.jl:363-369` |
| Median-z in-place equals allocating | `test/denoising.jl:153-160` |
| Notebook 12 FBP kernel tuple equality | `12_siemens_flash_ufc.jl:2216-2221` |
| Notebook 03 FBP kernel tuple equality | `03_dual_kvp_switching_vmi.jl:1336-1341` |

### 5.7 Bare `≈` with Julia defaults

Roughly 60 further assertions use bare `≈`, which is `rtol = sqrt(eps(Float64))`, about 1.49e-8 for
Float64 and `sqrt(eps(Float32))` about 3.45e-4 for Float32. They cluster in closed-form scaling
laws: `compute_detector_I0` linearity in mA, flux, views, rotation time, magnification squared, and
pixel area at `test/api.jl:233-286`; CTDIvol and DLP scaling at `test/source.jl:241-259`; spectrum
mean energy at `test/source.jl:67-75`; `interpolate_thickness` at `test/bowtie.jl:156-167`; affine
matrix entries at `test/geometry.jl:340-346`.

**Watch the Float32 cases.** `test/api.jl:1125` uses `v ≈ -0.04f0` for the FOV sentinel and
`test/api.jl:1169-1172` uses bare `≈` for corner voxels; at Float32 default that is a 3.45e-4
relative window, far looser than the other sentinel checks.

---

## 6. Dev docs

### 6.1 Agent-facing docs contain no testing conventions

`AGENTS.md` at the repo root, `CLAUDE.md` at the repo root, and `../CLAUDE.md` one level up are
**byte-identical** and consist entirely of the SpaceStation-managed Pluto notebook block, wrapped
in `<!-- SPACESTATION:BEGIN (managed ...) -->` at `AGENTS.md:1` and `<!-- SPACESTATION:END -->` at
`:82`. Content: cell UUID markers, the authoritative `# ╔═╡ Cell order:` block, fold-state
prefixes, `pluto-collab status|run --stale|restart`, and live-concurrency guarantees.

**There is no documented testing convention, no benchmarking convention, no tolerance policy, and
no instruction about running the suite with a GPU, in any of the three files.** A follow-up team
would have to reverse-engineer the `--project=docs` requirement from
`test/memory_stability_pcct.jl:2`.

### 6.2 `README.md`

Badges for CI, docs, SoftwareX DOI 10.1016/j.softx.2026.102910, Zenodo DOI 10.5281/zenodo.20262003.
Quick example at `:33-58` shows the `GPUSelect.Storage()` pattern, which the tests do not use.
Install instructions name Metal, CUDA, AMDGPU, and oneAPI as optional backends.

### 6.3 `SCANNERS.md`

See section 3.5. Note `SCANNERS.md:74-79` documents a deliberate spec-over-geometry choice worth
preserving: NAEOTOM row pitch uses the published 0.4 mm rather than the geometrically derived
0.352 mm, a 13 percent difference, so `detector_rows * detector_row_size` matches the 57.6 mm spec.

### 6.4 `docs/WINDOWS_SETUP.md`

CUDA setup for Windows. Backend table at `:60-64` maps macOS to `MtlArray`, Windows to `CuArray`,
Linux to `CuArray` or `ROCArray`, and asserts "The source code is identical — AcceleratedKernels.jl
handles GPU dispatch automatically."

**Contains a stale snippet.** `:41-47` passes `phantom_cpu.fov` as the fifth `Phantom` constructor
argument. Every current call site, for example `test/api.jl:339` and
`test/memory_stability_pcct.jl:47`, passes `phantom_cpu.extent`. The field was renamed and the doc
was not updated.

### 6.5 `src/reconstruction/README.md`

**Substantially stale.** Its folder diagram at `:7-24` documents `ir/sirt.jl`, `ir/cgls.jl`,
`mbir/mbir.jl`, `regularization/tv_regularization.jl`, and `statistical_ir.jl`. **None of those
files exist in the current `src/` tree.** The actual tree is `core/{backprojection,filtering}.jl`,
`fbp/{fdk,wfbp_helical}.jl`, `hybrid_ir/hybrid_ir.jl`, `ir/utils.jl`, `vmi/` with 13 files, and
`workspace/memory_budget.jl`. The clinical-terminology mapping table at `:28-32` and the strength
levels 1 to 5 at `:56-60` also predate the current percentage strength kwarg, which runs 0 to 100
in steps of 10 per `test/api.jl:1502-1514`.

### 6.6 `CHANGELOG.md` entries relevant to testing and benchmarking

| Line | Entry |
|---|---|
| `:169` | `perf(hir)!: 32x faster HIR, percentage strength kwarg, counts-domain DAS noise`, `87e9564` |
| `:197, :202` | `feat(geometry)!: equiangular arc detector — implemented, validated, DEFAULT`, `7b3440d` |
| `:234` | `feat(projection): :dd_fast — single-pass path-length DD (47x polychromatic forward)`, `7637bd0` |
| `:328` | "`test/detector.jl` — 155-assertion behavioral test suite covering every exported symbol from `src/detector/`" — now 314 assertions |
| `:329` | "`test/api.jl` — regression test ensuring `build_physics_config` skips the EICT scintillator branch for PCCT scanners" |
| `:361` | notebook inventory, describes notebook 06 as a "CatSim vs BasisSim parity benchmark" — see section 2.12 for why "parity" overstates it |
| `:69, :95-102` | a cluster of `test:` and `validate:` commits from the PCCT denoising research phase |

### 6.7 The real CI regression gate is Python, not Julia

`docs/verify_notebook_exports.py`, 11719 bytes. Docstring at `:2`: "Reject broken, stale, or
machine-specific committed notebook exports."

`export_fingerprint(path, snapshot_tree)` at `:101-109` computes a SHA-256 over a null-joined
payload of seven components:

1. `source_hash(notebook.jl)`
2. `simulator_input_hash()` at `:72-84`, itself a SHA-256 over `DATA_PROVENANCE.sha256` plus
   **every file under `src/`**, each as `relpath:sha256`, sorted
3. `snapshot_tree_hash()` at `:87-98`, the locked Snapshot git-tree-sha1 from
   `docs/build_env/Manifest.toml`
4. `source_hash(docs/build_env/Manifest.toml)`
5. `source_hash(docs/Manifest.toml)`
6. `source_hash(docs/extract_all.jl)`
7. `EXPORT_CONTRACT` at `:17-20`, the literal string
   `basissim-lean-v4|therapy=true|fragment=true|islands=true|verify=true|optimize=size|forced-fallback=01:z_slice,05:z_helical,11:z_idx`

`main()` at `:123` compares that digest against a fingerprint embedded in each committed HTML,
read by `recorded_hash` at `:112-121` from either a `<meta name="basissim-export-fingerprint">` tag
or an HTML comment for fragments. Any mismatch is a failure. It additionally rejects rendered Pluto
errors, `<jlerror` and `plain_error` at `:21-24`, and any leaked absolute path matching
`/Users/`, `/home/`, `/tmp/`, or `/Volumes/` at `:25-36`. Forced-fallback island assets are
whitelisted per notebook at `:38-53`.

`docs/build.sh:26-40` runs it **twice**, before and after `Pkg.instantiate()`, and aborts if
instantiate rewrote `docs/Manifest.toml`: "build exactly what was verified, and fail if
instantiation unexpectedly rewrites the lock."

**Operational consequence for the rewrite: any change to any file under `src/` invalidates the
export fingerprint of all 12 committed notebooks, and the docs deploy fails until every notebook is
re-rendered locally on a machine with Metal.** There is no way to bypass it short of editing the
verifier. Budget for this.

`docs/extract_all.jl:212-228` provides one escape hatch for local iteration:
`BASISSIM_SKIP_NOTEBOOKS=06_catsim_vs_basissim julia --project=docs docs/app.jl build`, a
comma-separated skip list. It does not suppress the verifier.

---

## 7. Assessment for the Reactant + Enzyme oracle harness

**What you can lift directly.** Twelve numerical contracts are already sharp enough to serve as
oracle assertions with no modification: the brute-force transpose matrix oracle at
`test/projection.jl:81-94` at 2e-12, the adjoint dot product at `test/projection.jl:56` at 2e-11,
the four `:dd_fast` equals `:dd` checks at 1e-4 absolute, the above-64-material bit-identical
fallback, the two projector-shim bit-identical routings, the FDK and HIR determinism pairs, the
HIR `W_proj` closed form at 2e-6, and the pileup count round trip at 1e-12. Together these pin the
linear operator, the fused-kernel reassociation, and the reconstruction determinism, which is most
of what an Enzyme rewrite can silently break.

**What you have to build.** There is no stored reference data of any kind. Every fixture is
constructed in-process. To make the current code an oracle you need a capture step: run the
existing fixtures, serialize the outputs, and assert against them. The fixture constructors are
already isolated and cheap, so this is mechanical: `_toy_proj_geom` at `test/projection.jl:19-29`,
`_toy_scan` at `test/api.jl:52-77`, `_toy_fdk_setup` at `test/api.jl:1098-1108`, `_toy_hir_setup`
at `test/api.jl:1221-1232`, `_toy_pcct_setup` at `test/api.jl:302-341`, `_toy_eict_setup` at
`test/api.jl:577-611`, `_toy_bhc_scanner` at `test/correction.jl:26-40`, and
`_clinical_scanner_with_bowtie` at `test/bowtie.jl:35`.

**The GPU blind spot is the biggest risk.** `Pkg.test()` and all three CI legs run with
`HAS_GPU = false`, so 13 testsets covering the entire `simulate!` path and every workspace
constructor are skipped. Nothing in CI has ever executed `simulate!(PCCTWorkspace, ...)` or
`simulate!(EICTWorkspace, ...)`. Before you change anything, establish a baseline by running
`julia --project=docs test/runtests.jl` on a Metal machine and record what that adds. Otherwise you
will be comparing a rewrite against an oracle that was never itself exercised on the path you care
about.

**Three published claims are weaker than the memory index suggests.** The helical affine 1-to-1
proof at `05_xcat_grid_to_recon.jl:1374-1424` prints machine-precision numbers but has no coded
threshold. Notebook 06 has no numeric CatSim agreement tolerance at all, only a shared-window image
mosaic. And the notebook 06 cache disagrees with `paper/main.tex` by 3.5x on the Metal speedup.
None of these will fail if a rewrite regresses them.

**Two docs are stale enough to mislead.** `src/reconstruction/README.md` documents five source
files that do not exist. `docs/WINDOWS_SETUP.md:46` passes a renamed `Phantom` field.

**Budget for the export fingerprint.** `docs/verify_notebook_exports.py` hashes every file under
`src/` into all 12 notebook export fingerprints. The first `src/` change breaks the docs deploy
until every notebook is re-rendered locally on Metal.

---

## Appendix: measured test-suite output

```
     Testing Running tests...
[ Info: [api.jl] GPU backend = CPU
[api.jl t=0.0s]  entering SimOptions testset
[api.jl t=99.0s] entering EICT quantum noise dose-sweep testset
...
Test Summary:     | Pass  Total     Time
BasisSimulator.jl | 3156   3156  3m04.0s
     Testing BasisSimulator tests passed
julia --project=. -e 'using Pkg; Pkg.test()'  315.73s user 15.92s system 139% cpu 3:57.72 total
EXIT=0
```

Machine: Apple M4, macOS 25.3.0 (Darwin), arm64. Julia project environment, CPU-only, 13 GPU-gated
testsets skipped. Log retained at
`/private/tmp/claude-501/-Users-daleblack-Documents-dev-MolloiLab-BasisSimulator-jl/8c064cb4-7c17-4a8f-b484-ef8ee30b0ec5/scratchpad/testrun.log`.
