"""
    Simulation/Options.jl

High-level options for controlling simulation fidelity and reconstruction.
"""

export SimOptions, ReconOptions

"""
    SimOptions

Resolved boolean toggles + numeric knobs for one simulation run.  Each
`use_*` field is `Bool` (`true` = effect ON, `false` = effect OFF) and is
resolved from a `fidelity` preset (see ctor) plus optional per-effect kwarg
overrides.  The `fidelity` symbol itself is consumed only inside the ctor
for preset lookup — it is not stored on the struct.

# Fields
- `use_fill_factor::Bool`: Enable detector fill factor.
- `use_detector_efficiency::Bool`: Enable energy-dependent detector efficiency.
- `use_scatter::Bool`: Enable scatter simulation (correction is decoupled to notebook level).
- `use_optical_crosstalk::Bool`: Enable optical crosstalk.
- `use_focal_spot::Bool`: Enable focal spot blur.
- `use_noise::Bool`: Enable quantum/electronic noise.
- `use_lag::Bool`: Enable detector lag (afterglow).
- `use_heel_effect::Bool`: Enable anode heel effect.
- `use_pcct_pileup::Bool`: Apply MC pulse pileup at simulate-time (PCCT only).
  Default `true` for `:pcct`, `false` for `:eict`.  When `false`, the workspace
  skips the (expensive) MC pileup matrix calibration entirely and the simulator
  produces noise-free count rates.  The MC-LUT detector response matrix is
  always on; pileup is the only PCCT physics knob exposed here.
- `use_pcct_pileup_correction::Bool`: invert the MC pileup matrix inside `simulate!()`
  (model-based un-pileup) after the forward pileup pass.  Default `false`; enable to fold the
  decoupled notebook-level `apply_pcct_pileup_correction!` into `simulate!()`.
- `use_pcct_scatter::Bool`: PCCT-specific patient scatter injection, distinct from the
  EICT `use_scatter`.  Default `true` for `:pcct`, `false` for `:eict`.
- `use_pcct_scatter_correction::Bool`: model-based PCCT scatter correction applied inside
  `simulate!()` after pileup (re-estimates the scatter field from the current bins and
  subtracts it).  Default `false`; enable to fold the notebook-level correction into `simulate!()`.
- `pcct_noise_reduction::Float64`: PCCT noise reduction factor (0.0–1.0).  Approximates clinical
  vendor reconstruction (e.g., Siemens QIR).  0.0 = raw physics (default), 0.7 = 70% noise reduction
  (~QIR-3).  Only affects PCCT sinogram noise; EICT noise is unaffected.
- `seed::Union{Int, Nothing}`: Random seed for reproducibility.  Default 42.
- `detector_efficiency_mode::Symbol`: Override detector efficiency calculation mode.
  `:auto` (default) = let driver decide; `:mc_lut` = force MC LUT; `:beer_lambert` = force analytical.
- `projector::Symbol`: Forward-projection ray tracer.  `:dd` (default) = distance-driven,
  anti-aliased footprint integration (robust in severe beam-hardened regions).  `:siddon` =
  exact ray tracing, ~3.5-5.5x faster on GPU but ALIASES in severe beam-hardened regions —
  use only when speed outranks accuracy.  NOTE: to keep the iterative-recon system matrix
  consistent with the data, pass the SAME projector to `create_hir_recon_workspace(; projector=…)`
  (both default `:dd`).
"""
struct SimOptions
    # --- Physics Pipeline (7 effects) ---
    use_fill_factor::Bool
    use_detector_efficiency::Bool
    use_scatter::Bool
    use_optical_crosstalk::Bool
    use_focal_spot::Bool
    use_noise::Bool
    use_lag::Bool

    # --- Signal Chain ---
    use_heel_effect::Bool

    # --- PCCT physics ---
    use_pcct_pileup::Bool
    use_pcct_pileup_correction::Bool
    use_pcct_scatter::Bool
    use_pcct_scatter_correction::Bool
    pcct_noise_reduction::Float64

    # --- General ---
    seed::Union{Int, Nothing}
    detector_efficiency_mode::Symbol   # :auto, :mc_lut, :beer_lambert
    projector::Symbol                  # :dd (default, anti-aliased) or :siddon (fast)
end

"""
    SimOptions(; kwargs...)

Create simulation options with fidelity presets and per-effect overrides.

# Presets (via `fidelity`)
- `:eict`: All EICT effects ON (polychromatic, full physics).
- `:pcct`: Same as :eict but with PCCT detector corrections enabled.

# Keyword Overrides
Pass `use_*=true/false` to override the fidelity preset for individual effects.
- `nothing` (default) = use preset
- `true` = force ON
- `false` = force OFF

# Examples
```julia
SimOptions(fidelity=:eict)                         # Full physics
SimOptions(fidelity=:eict, use_scatter=false)       # Everything except scatter
SimOptions(fidelity=:pcct)                          # PCCT mode
```
"""
function SimOptions(;
        fidelity::Symbol = :eict,
        use_fill_factor::Union{Bool, Nothing} = nothing,
        use_detector_efficiency::Union{Bool, Nothing} = nothing,
        use_scatter::Union{Bool, Nothing} = nothing,
        use_optical_crosstalk::Union{Bool, Nothing} = nothing,
        use_focal_spot::Union{Bool, Nothing} = nothing,
        use_noise::Union{Bool, Nothing} = nothing,
        use_lag::Union{Bool, Nothing} = nothing,
        use_heel_effect::Union{Bool, Nothing} = nothing,
        use_pcct_pileup::Union{Bool, Nothing} = nothing,
        use_pcct_pileup_correction::Union{Bool, Nothing} = nothing,
        use_pcct_scatter::Union{Bool, Nothing} = nothing,
        use_pcct_scatter_correction::Union{Bool, Nothing} = nothing,
        pcct_noise_reduction::Float64 = 0.0,
        seed::Union{Int, Nothing} = 42,
        detector_efficiency_mode::Symbol = :auto,
        projector::Symbol = :dd
    )
    _validate_projector(projector)
    # Fidelity preset defaults
    # :eict = all EICT effects ON; :pcct = :eict + MC pile-up degradation.
    # Pile-up correction (the inverse) is decoupled — apply it post-simulate
    # via `apply_pcct_pileup_correction!` if needed.
    # Note: `optical_crosstalk` defaults to FALSE.  `apply_optical_crosstalk!`
    # is a physically-accurate nonlinear `−log(K · exp(−x))` blur in intensity
    # domain, but BS does not yet ship a numerically stable inverse for the
    # deeply-attenuated pixels behind dense rods (Van Cittert deconvolution
    # produces FBP streaks).  Users who want to *see* the un-corrected blur
    # in diagnostic sinograms can opt in via `use_optical_crosstalk = true`;
    # downstream basis-decomposition accuracy will degrade accordingly.
    # :pcct preset notes:
    # - `focal_spot = false`: the tube-side focal-spot blur IS wired into the
    #   PCCT path (per-bin, before scatter/noise/pile-up) but ships disabled
    #   by default in this release; opt in with `use_focal_spot = true`.
    # - `lag = false`: the shipped lag model is scintillator (Gd₂O₂S)
    #   afterglow, which direct-conversion PCCT detectors do not exhibit, so
    #   lag is not applied on the PCCT path.
    defaults = if fidelity == :pcct
        (
            fill_factor = true, detector_efficiency = true,
            scatter = true, optical_crosstalk = false,
            focal_spot = false, noise = true, lag = false,
            heel_effect = true,
            pcct_pileup = true, pcct_pileup_correction = false,
            pcct_scatter = true, pcct_scatter_correction = false,
        )
    elseif fidelity == :eict
        (
            fill_factor = true, detector_efficiency = true,
            scatter = true, optical_crosstalk = false,
            focal_spot = true, noise = true, lag = true,
            heel_effect = true,
            pcct_pileup = false, pcct_pileup_correction = false,
            pcct_scatter = false, pcct_scatter_correction = false,
        )
    else
        error("Unknown fidelity preset: $fidelity. Use :eict or :pcct.")
    end

    # Resolve each toggle: user override wins, otherwise use preset default
    _fill_factor = isnothing(use_fill_factor) ? defaults.fill_factor : use_fill_factor
    _detector_efficiency = isnothing(use_detector_efficiency) ? defaults.detector_efficiency : use_detector_efficiency
    _scatter = isnothing(use_scatter) ? defaults.scatter : use_scatter
    _optical_crosstalk = isnothing(use_optical_crosstalk) ? defaults.optical_crosstalk : use_optical_crosstalk
    _focal_spot = isnothing(use_focal_spot) ? defaults.focal_spot : use_focal_spot
    _noise = isnothing(use_noise) ? defaults.noise : use_noise
    _lag = isnothing(use_lag) ? defaults.lag : use_lag
    _heel_effect = isnothing(use_heel_effect) ? defaults.heel_effect : use_heel_effect
    _pcct_pileup = isnothing(use_pcct_pileup) ? defaults.pcct_pileup : use_pcct_pileup
    _pcct_pileup_correction = isnothing(use_pcct_pileup_correction) ? defaults.pcct_pileup_correction : use_pcct_pileup_correction
    _pcct_scatter = isnothing(use_pcct_scatter) ? defaults.pcct_scatter : use_pcct_scatter
    _pcct_scatter_correction = isnothing(use_pcct_scatter_correction) ? defaults.pcct_scatter_correction : use_pcct_scatter_correction

    return SimOptions(
        _fill_factor, _detector_efficiency,
        _scatter, _optical_crosstalk,
        _focal_spot, _noise, _lag,
        _heel_effect,
        _pcct_pileup,
        _pcct_pileup_correction,
        _pcct_scatter,
        _pcct_scatter_correction,
        clamp(pcct_noise_reduction, 0.0, 1.0),
        seed,
        detector_efficiency_mode,
        projector
    )
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
