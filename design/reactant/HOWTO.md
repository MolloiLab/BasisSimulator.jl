# How to test the functional core (branch `feat/reactant-autodiff`)

Everything below runs on this Mac (XLA CPU). The same thunks run on NVIDIA Linux with
`Reactant.set_default_backend("gpu")`. Metal has no Reactant backend; the legacy AK path stays
the Metal path and is the oracle for every test here.

## 0. One-time setup

```bash
julia --project=envs/reactant -e 'using Pkg; Pkg.instantiate()'   # Reactant 0.2.285 + Enzyme 0.13.201 + this repo
```

## 1. Run the oracle tests (plain arrays, no Reactant) and the Reactant smokes

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                        # whole suite incl. test/functional/ (~4.5 min)
julia --project=. -t 2 test/functional/test_pipeline.jl              # one stage standalone (each test_*.jl works this way)
bash test/functional/reactant/run_all.sh                             # all Reactant/Enzyme smokes, serialized behind a lock
bash test/functional/reactant/run_all.sh pipeline                    # just the end-to-end one (~2.5 min)
```

Memory rule for this 16 GB machine: never run two Reactant compiles at once, never run a Reactant
smoke while `Pkg.test()` is running. `run_all.sh` takes `/tmp/bs_reactant.lock`; if a run is killed,
`rmdir /tmp/bs_reactant.lock`.

## 2. The functional pipeline behind the five structs (REPL or notebook)

```julia
using BasisSimulator; const BS = BasisSimulator; const BSF = BS.Functional

phantom  = BS.compact_materials(BS.create_gammex_472(n_voxels = 64, n_slices = 4, fov_cm = 20.0, z_cm = 2.0))
scanner  = BS.Scanner(source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 8,
                      detector_cols = 128, detector_row_size = 1.0, detector_col_size = 1.0,
                      detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 32, rotation_time = 0.5)
sim_opts = BS.SimOptions(fidelity = :eict, seed = 42, use_noise = true, use_scatter = false,
                         use_lag = false, use_focal_spot = false, use_optical_crosstalk = false)
recon    = BS.ReconOptions(matrix_size = (64, 64, 4), fov_cm = 20.0)

pipe = BSF.eict_pipeline(phantom, scanner, protocol, sim_opts, recon)   # plans: DD, EICT(+water BHC), FDK, μ_water
fr   = BSF.onehot_fractions(phantom.mask, pipe.n_mat)                 # the differentiable phantom parameterization
ε, ε_e = BSF.draw_eict_noise(pipe; seed = 42)                         # same draws simulate! would make

sino = BSF.simulate_sino(fr, pipe, ε, ε_e)   # == simulate! + apply_bhc_water   (1e-6 rel)
μ    = BSF.reconstruct_μ(sino, pipe)         # == reconstruct!(FDK workspace)   (1e-6 rel)
hu   = BSF.eict_forward(fr, pipe, ε, ε_e)    # == to_hounsfield(μ; μ_water)     (1e-6 rel)
```

Stage-level entry points (each with its own plan struct and oracle test): `dd_project`/`dd_transpose`,
`fdk`, `eict_chain`, `pcct_chain`, `hir_reconstruct`, `acnr_kalender`, `sino_svd_denoise_bilateral`,
`cong_decompose`, `synth_vmi_2basis`. Nothing is exported; see the docstrings in `src/functional/`.

## 3. Compile it with Reactant and differentiate it with Enzyme

```julia
using Reactant, Enzyme                      # from envs/reactant; loads ext/BasisSimulatorReactantExt.jl

fr_r, ε_r, εe_r = Reactant.to_rarray.((fr, ε, ε_e))
fwd   = (fr, ε, ε_e) -> BSF.eict_forward(fr, pipe, ε, ε_e)
thunk = @compile sync = true fwd(fr_r, ε_r, εe_r)       # one XLA program: DD → EICT → BHC → FDK → HU
hu_c  = Array(thunk(fr_r, ε_r, εe_r))

w      = randn(Float32, pipe.recon_shape)
loss(fr) = sum(w .* BSF.eict_forward(fr, pipe))          # noise-free for a clean gradient
grad   = @compile sync = true (fr -> Enzyme.gradient(Reverse, loss, fr)[1])(fr_r)
∂hu_∂fr = Array(grad(fr_r))                              # (nx, ny, nz, n_mat): d loss / d material fraction
```

`Enzyme.gradient` must be called with the mode first when you compile a wrapper of it. Keep plans
host-side (they are); `eict_forward` lifts the plan tables into the graph as constants
(`Functional._on_device`). Measured on the toy in `smoke_pipeline.jl`: compile ~60 s, 1.0 ms per
forward (21.6 ms plain Julia), gradient = finite differences to 1.6e-11.

## 4. What is still unrolled (the M5 work)

Views and materials are unrolled in the traced graph today (every material is projected one-hot,
every view is its own program), so compile time grows with `n_views × n_mat` and the toy sizes above
are what compile in a minute. The M5 tasks in `README.md` — `@trace for` over views with batched
per-view plans, and select-accumulate over the gathered material id — turn that into one loop body.
Until then, use `compact_materials` and tens of views for compiled experiments; the plain-array path
has no such limit.
