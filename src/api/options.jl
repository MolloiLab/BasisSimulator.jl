"""
    Simulation/Options.jl

Simulation and reconstruction options shared by every detector family.
"""

export SimOptions, ReconOptions

"""
    SimOptions

Resolved boolean toggles for one simulation run.  Each
`use_*` field is `Bool` (`true` = effect ON, `false` = effect OFF); every field
is set by the keyword constructor.

# Fields
- `use_fill_factor::Bool`: Enable detector fill factor.
- `use_detector_efficiency::Bool`: Enable energy-dependent detector efficiency.
- `use_scatter::Bool`: Superpose the scanner's Geant4 scatter kernels on the patient
  ([`scatter_kernels`](@ref); the scanner's `scatter_kernels` field names them). On by default,
  as on a real scanner: the clinical constants name their kernels (a download of about 250 MB
  per scanner, cached), and a scanner that names none throws unless this is `false`. The
  scanner's `scatter_correction` decides whether its detector subtracts what it estimates.
- `use_optical_crosstalk::Bool`: Enable optical crosstalk.
- `use_focal_spot::Bool`: Enable focal spot blur.
- `use_noise::Bool`: Enable noise.  PCCT: exact per-bin integer Poisson count
  realizations of the MC-detector expected counts (no electronic noise — counting
  thresholds eliminate it).  EICT: Gaussian quantum + electronic noise on counts.
- `use_lag::Bool`: Enable detector lag (afterglow).
- `use_heel_effect::Bool`: Enable anode heel effect.
- `seed::Union{Int, Nothing}`: Random seed for reproducibility.  Default 42.
- `detector_efficiency_mode::Symbol`: an energy integrator's response, `:mc_lut` (default, its Geant4
  table) or `:beer_lambert` (analytic absorption).
- `projector::Symbol`: Forward-projection ray tracer.  `:dd_fast` (default) = distance-driven,
  anti-aliased footprint integration with single-pass per-material path-length fused kernels —
  the full spectrum runs in ONE volume walk (measured 47x faster than the per-energy tiled
  distance-driven kernels on a 234-bin polychromatic forward on M4 Metal, agreeing with them
  to floating-point ordering); supports ≤ 64 materials (emits a prominent warning and falls
  back to the per-energy tiled kernels above that); call `compact_materials` to remove
  inactive table entries.  `:siddon` = exact point-sampled ray tracing retained for
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
    detector_efficiency_mode::Symbol   # :mc_lut, :beer_lambert
    projector::Symbol                  # :dd_fast (default), :siddon (comparison)
    # --- Acquisition ---
    view_samples::Int                  # angular samples per view across its integration arc (1 = instantaneous)
    view_arc::Float64                  # the arc a view integrates over, as a fraction of the view spacing
end

"""
    SimOptions(; kwargs...)

Physics toggles common to every detector family (all `true` except `use_optical_crosstalk`),
the noise `seed`, the detector-efficiency mode and the projector.  Detector-specific physics lives
on the scanner: its scatter kernels and scatter correction are fields of both
[`EICTScanner`](@ref) and [`PCCTScanner`](@ref), pile-up and the count-noise blend of
[`PCCTScanner`](@ref); the scintillator lag model applies to [`EICTScanner`](@ref) only.
A variant is the copy constructor, `SimOptions(o; use_noise = false)`.

- `use_fill_factor`, `use_detector_efficiency`, `use_scatter`, `use_optical_crosstalk`,
  `use_focal_spot`, `use_noise`, `use_lag`, `use_heel_effect::Bool`
- `seed::Union{Int, Nothing} = 42` — noise RNG seed. `nothing` leaves the energy-integrating
  path unseeded; the photon-counting path draws a fresh seed (`pcct_count!`).
- `detector_efficiency_mode::Symbol = :mc_lut` — `:mc_lut`, `:beer_lambert`
- `projector::Symbol = :dd_fast` — `:dd_fast` or `:siddon`; anything else throws an `ArgumentError`
- `view_samples::Int = 1` — gantry rotation during a view: the detector integrates while the gantry
  turns through the view's arc, so a view records the transmitted intensity averaged over that arc,
  a blur of the object along the direction of rotation that grows with the distance from the
  isocentre and that the noise, counted once per view, does not share. `view_samples` points across
  the arc (the midpoint rule) sample the average; `1` is the instantaneous view (no rotation blur).
  The blur is the projection of an arc of length `r · view_arc · Δθ` at radius `r`.
- `view_arc::Real = 1.0` — the arc a view integrates over, as a fraction of the view spacing
  `Δθ`: 1 for a detector that integrates for the whole view period, the duty cycle of one
  energy for rapid kVp switching.
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
        detector_efficiency_mode::Symbol = :mc_lut,
        projector::Symbol = :dd_fast,
        view_samples::Integer = 1,
        view_arc::Real = 1.0,
    )
    _validate_projector(projector)
    detector_efficiency_mode in (:mc_lut, :beer_lambert) || throw(
        ArgumentError("detector_efficiency_mode must be :mc_lut or :beer_lambert, got :$(detector_efficiency_mode)")
    )
    view_samples >= 1 || throw(ArgumentError("view_samples must be at least 1, got $(view_samples)"))
    0 < view_arc <= 1 || throw(ArgumentError("view_arc must be in (0, 1], got $(view_arc)"))
    return SimOptions(
        use_fill_factor, use_detector_efficiency, use_scatter, use_optical_crosstalk,
        use_focal_spot, use_noise, use_lag, use_heel_effect, seed, detector_efficiency_mode, projector,
        Int(view_samples), Float64(view_arc)
    )
end

# A variant of `o`: every option copied, the keywords given replaced.
SimOptions(o::SimOptions; kwargs...) = SimOptions(;
    merge(NamedTuple{fieldnames(SimOptions)}(getfield.(Ref(o), fieldnames(SimOptions))), values(kwargs))...,
)

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
