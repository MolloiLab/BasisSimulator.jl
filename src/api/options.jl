"""
    Simulation/Options.jl

Simulation and reconstruction options shared by every detector family.
"""

export SimOptions, ReconOptions

"""
    SimOptions

Resolved boolean toggles + numeric knobs for one simulation run.  Each
`use_*` field is `Bool` (`true` = effect ON, `false` = effect OFF); every field
is set by the keyword constructor.

# Fields
- `use_fill_factor::Bool`: Enable detector fill factor.
- `use_detector_efficiency::Bool`: Enable energy-dependent detector efficiency.
- `use_scatter::Bool`: Enable scatter simulation (correction is decoupled to notebook level).
- `use_optical_crosstalk::Bool`: Enable optical crosstalk.
- `use_focal_spot::Bool`: Enable focal spot blur.
- `use_noise::Bool`: Enable noise.  PCCT: exact per-bin integer Poisson count
  realizations of the MC-detector expected counts (no electronic noise — counting
  thresholds eliminate it).  EICT: Gaussian quantum + electronic noise on counts.
- `use_lag::Bool`: Enable detector lag (afterglow).
- `use_heel_effect::Bool`: Enable anode heel effect.
  vendor reconstruction (e.g., Siemens QIR) by blending sampled Poisson counts toward their
  expectation.  0.0 = raw physics, exact integer Poisson counts (default); 0.7 = 70% noise
  reduction (~QIR-3).  ANY nonzero value leaves the strict Poisson count model (a scaled Poisson
  deviate is not Poisson) — statistical-model validation must run at 0.0.  Only affects PCCT
  sinogram noise; EICT noise is unaffected.  VALIDATION DOCTRINE: HU-accuracy claims must hold at
  0.0.  Use nonzero only for noise-magnitude studies.
- `seed::Union{Int, Nothing}`: Random seed for reproducibility.  Default 42.
- `detector_efficiency_mode::Symbol`: Override detector efficiency calculation mode.
  `:auto` (default) = let driver decide; `:mc_lut` = force MC LUT; `:beer_lambert` = force analytical.
- `projector::Symbol`: Forward-projection ray tracer.  `:dd_fast` (default) = distance-driven,
  anti-aliased footprint integration with single-pass per-material path-length fused kernels —
  the full spectrum runs in ONE volume walk (measured 47x faster than `:dd` on a 234-bin
  polychromatic forward on M4 Metal), results agree with `:dd` to floating-point ordering;
  supports ≤ 64 materials (emits a prominent warning and falls back to the `:dd` kernels above
  that); call `compact_materials` to remove inactive table entries. Mono projection is
  the `:dd` kernel unchanged.  `:dd` = the original per-energy distance-driven kernel —
  **DEPRECATED** (kept as the numerical reference; emits a warning and may be removed in a
  future release; use `:dd_fast`).  `:siddon` = exact point-sampled ray tracing retained for
  comparison and compatibility; it is slower than `:dd_fast` for full polychromatic/spectral
  simulations and can ALIAS in severe beam-hardened regions.  NOTE: to keep
  the iterative-recon system matrix consistent with the data, pass the SAME projector to
  `create_hir_recon_workspace(; projector=…)` (both default `:dd_fast`).
"""
struct SimOptions
    # --- Physics pipeline ---
    use_fill_factor::Bool
    use_detector_efficiency::Bool
    use_scatter::Bool
    use_optical_crosstalk::Bool
    use_focal_spot::Bool
    use_noise::Bool
    use_lag::Bool
    # --- Signal chain ---
    use_heel_effect::Bool
    # --- General ---
    seed::Union{Int, Nothing}
    detector_efficiency_mode::Symbol   # :auto, :mc_lut, :beer_lambert
    projector::Symbol                  # :dd_fast (default), :dd (DEPRECATED reference), :siddon (comparison)
end

"""
    SimOptions(; kwargs...)

Physics toggles common to every detector family (all `true` except
`use_optical_crosstalk`), the noise `seed`, the detector-efficiency mode and the
projector.  Detector-specific physics lives on the scanner: pile-up, pile-up
correction, scatter correction and the count-noise blend are fields of
[`PCCTScanner`](@ref); the scintillator lag model applies to [`EICTScanner`](@ref)
only.

- `use_fill_factor`, `use_detector_efficiency`, `use_scatter`, `use_optical_crosstalk`,
  `use_focal_spot`, `use_noise`, `use_lag`, `use_heel_effect::Bool`
- `seed::Union{Int, Nothing} = 42` — noise RNG seed (`nothing` = unseeded)
- `detector_efficiency_mode::Symbol = :auto` — `:auto`, `:mc_lut`, `:beer_lambert`
- `projector::Symbol = :dd_fast` — `:dd_fast`, `:dd` (deprecated reference), `:siddon`
"""
function SimOptions(;
        use_fill_factor::Bool = true,
        use_detector_efficiency::Bool = true,
        use_scatter::Bool = true,
        use_optical_crosstalk::Bool = false,
        use_focal_spot::Bool = true,
        use_noise::Bool = true,
        use_lag::Bool = true,
        use_heel_effect::Bool = true,
        seed::Union{Int, Nothing} = 42,
        detector_efficiency_mode::Symbol = :auto,
        projector::Symbol = :dd_fast,
    )
    _validate_projector(projector)
    return SimOptions(use_fill_factor, use_detector_efficiency, use_scatter, use_optical_crosstalk,
        use_focal_spot, use_noise, use_lag, use_heel_effect, seed, detector_efficiency_mode, projector)
end

"""
    ReconOptions

Reconstruction-grid configuration consumed by the workspace constructors.
Three fields, each with one specific consumer.

# Fields
- `matrix_size::NTuple{3,Int}`: Output volume size `(nx, ny, nz)`.  Notebooks
  pull this and pass it explicitly into `create_fdk_recon_workspace` /
  `create_hir_recon_workspace` (those ctors take `volume_size` as a
  positional arg).
- `fov_cm::Float64`: XY field of view in cm.  Read by both PCCT and EICT
  workspace ctors via `CTGeometry(scanner; fov_cm = recon_opts.fov_cm, ...)`.
- `z_cm::Union{Float64,Nothing}`: Z extent in cm.  `nothing` → auto-compute
  from detector coverage; set explicitly to `sliceCount * sliceThickness / 10`
  for clinical slice-thickness control.  Read by the same `CTGeometry` call.
"""
struct ReconOptions
    matrix_size::NTuple{3, Int}
    fov_cm::Float64
    z_cm::Union{Float64, Nothing}
end

"""
    ReconOptions(; matrix_size=(512,512,64), fov_cm=35.0, z_cm=nothing)

Construct `ReconOptions`.  All three kwargs have sane defaults.

# Examples
```julia
# Standard 512² × 64 recon at 35 cm FOV
ReconOptions(matrix_size = (512, 512, 64), fov_cm = 35.0)

# Clinical slice-thickness control (5 cm Z extent at 0.625 mm slice)
ReconOptions(matrix_size = (512, 512, 80), fov_cm = 50.0, z_cm = 5.0)
```
"""
function ReconOptions(;
        matrix_size::Union{NTuple{3, Int}, Nothing} = nothing,
        fov_cm::Real = 35.0,
        z_cm::Union{Real, Nothing} = nothing,
    )
    _size = isnothing(matrix_size) ? (512, 512, 64) : matrix_size
    _z_cm = isnothing(z_cm) ? nothing : Float64(z_cm)
    return ReconOptions(_size, Float64(fov_cm), _z_cm)
end
