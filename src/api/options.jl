"""
    Simulation/Options.jl

High-level options for controlling simulation fidelity and reconstruction.
"""

export SimOptions, ReconOptions

"""
    SimOptions

Controls the fidelity and physics realism of the simulation.
Each `use_*` field is `Bool`: `true` = effect ON, `false` = effect OFF.
These are resolved from fidelity presets and user overrides at construction time.

# Spectrum Mode (IMPORTANT)

There are two spectrum modes, controlled by `use_real_spectrum`:

## `use_real_spectrum=false` (DEFAULT) — CatSim Pre-Filtered Spectra
  - Loads pre-filtered spectra from `tungsten_tar*_filt.dat` files (already filtered by CatSim).
  - `flat_filter` can be ON/OFF to add projection-domain filtering on top.
  - Simple, fast, backward-compatible with existing CatSim workflows.

## `use_real_spectrum=true` — Real IPEM Unfiltered Spectra + User-Defined Filtering
  - Loads raw, unfiltered spectra from IPEM Anode data (`Anode8/`, `Anode10/` folders).
  - User defines filters via `CTProtocol(additional_filters=...)`:
    - Materials (e.g., "Al", "Cu", "Sn") and thicknesses in mm.
    - Filtering applied in spectrum domain using Beer-Lambert law: T(E) = exp(-Σ μᵢ(E)×tᵢ).
  - `flat_filter` is **automatically disabled** (mutually exclusive with `additional_filters`).
  - Physically correct: energy-dependent filtering before simulation.

To activate real spectra, set **both** SimOptions and CTProtocol:
```julia
sim_opts = SimOptions(fidelity=:high, use_real_spectrum=true)
protocol = CTProtocol(kVp=120, mA=300,
    anode_angle=10,                                # which anode spectrum (8° or 10°)
    additional_filters=[("Al", 2.5), ("Cu", 0.1)]) # filter materials + thicknesses (mm)
```

# Fields
- `fidelity::Symbol`: Preset level (:pcct, :high_plus, :high, :medium, :low, :ideal). Default :high.
- `use_fill_factor::Bool`: Enable detector fill factor.
- `use_flat_filter::Bool`: Enable flat (inherent) filtration in projection domain.
  Automatically disabled when `use_real_spectrum=true` (mutually exclusive).
- `use_bowtie_filter::Bool`: Enable bowtie filter.
- `use_detector_efficiency::Bool`: Enable energy-dependent detector efficiency.
- `use_scatter::Bool`: Enable scatter simulation.
- `use_scatter_correction::Bool`: Enable scatter correction in signal chain.
- `use_crosstalk::Bool`: Enable electronic detector crosstalk.
- `use_optical_crosstalk::Bool`: Enable optical crosstalk.
- `use_focal_spot::Bool`: Enable focal spot blur.
- `use_noise::Bool`: Enable quantum/electronic noise (via sim_detect).
- `use_lag::Bool`: Enable detector lag (afterglow).
- `use_heel_effect::Bool`: Enable anode heel effect.
- `use_das::Bool`: Enable DAS model (BROKEN — always false).
- `use_bhc::Bool`: Enable beam hardening correction.
- `use_pcct_corrections::Bool`: Enable PCCT detector corrections (inverse pileup, inverse charge sharing).
- `use_real_spectrum::Bool`: Enable real IPEM unfiltered spectra with user-defined filtering.
  When false (default), uses CatSim pre-filtered spectra. See "Spectrum Mode" above.
- `pcct_noise_reduction::Float64`: PCCT noise reduction factor (0.0–1.0). Approximates clinical
  vendor reconstruction (e.g., Siemens QIR). 0.0 = raw physics (default), 0.7 = 70% noise reduction
  (~QIR-3). Only affects PCCT sinogram noise; EICT noise is unaffected.
- `n_energy_bins::Int`: Number of spectrum bins for polychromatic mode. Default 30.
- `seed::Union{Int, Nothing}`: Random seed for reproducibility. Default 42.
"""
struct SimOptions
    fidelity::Symbol

    # --- Physics Pipeline (10 effects) ---
    use_fill_factor::Bool
    use_flat_filter::Bool
    use_bowtie_filter::Bool
    use_detector_efficiency::Bool
    use_scatter::Bool
    use_scatter_correction::Bool
    use_crosstalk::Bool
    use_optical_crosstalk::Bool
    use_focal_spot::Bool
    use_noise::Bool
    use_lag::Bool

    # --- Signal Chain (3 effects) ---
    use_heel_effect::Bool
    use_das::Bool
    use_bhc::Bool

    # --- PCCT Corrections ---
    use_pcct_corrections::Bool

    # --- Spectrum Source ---
    use_real_spectrum::Bool

    # --- General ---
    pcct_noise_reduction::Float64
    seed::Union{Int,Nothing}
    n_energy_bins::Int
end

"""
    SimOptions(; kwargs...)

Create simulation options with fidelity presets and per-effect overrides.

# Presets (via `fidelity`)
- `:ideal`: All effects OFF — geometric ray tracing only.
- `:low`: Noise only.
- `:medium`: Noise + focal_spot + crosstalk + flat_filter + bhc (polychromatic).
- `:high`: All effects ON except DAS (BROKEN). Uses CatSim pre-filtered spectra by default.
- `:high_plus`: Same as :high but with real IPEM spectra, spectrum-domain filtering,
  and MC-based detector efficiency. Requires `CTProtocol(anode_angle=, additional_filters=...)`.
- `:pcct`: Same as :high but with PCCT detector corrections enabled.

# Keyword Overrides
Pass `use_*=true/false` to override the fidelity preset for individual effects.
- `nothing` (default) = use preset
- `true` = force ON
- `false` = force OFF

# Examples
```julia
# Default: CatSim pre-filtered spectra (backward compatible)
SimOptions(fidelity=:high)

# High Plus: Real IPEM spectra + spectrum-domain filtering + MC detector efficiency
# NOTE: also set CTProtocol(anode_angle=10, additional_filters=[("Al", 2.5)])
SimOptions(fidelity=:high_plus)

# Or manually activate real spectra on any preset
SimOptions(fidelity=:high, use_real_spectrum=true)

# Override individual effects
SimOptions(fidelity=:high, use_scatter=false)       # Everything except scatter
SimOptions(fidelity=:ideal, use_noise=true)         # Noise only
SimOptions(fidelity=:medium, use_bowtie_filter=true) # Medium + bowtie
```
"""
function SimOptions(;
    fidelity::Symbol=:high,
    use_fill_factor::Union{Bool,Nothing}=nothing,
    use_flat_filter::Union{Bool,Nothing}=nothing,
    use_bowtie_filter::Union{Bool,Nothing}=nothing,
    use_detector_efficiency::Union{Bool,Nothing}=nothing,
    use_scatter::Union{Bool,Nothing}=nothing,
    use_scatter_correction::Union{Bool,Nothing}=nothing,
    use_crosstalk::Union{Bool,Nothing}=nothing,
    use_optical_crosstalk::Union{Bool,Nothing}=nothing,
    use_focal_spot::Union{Bool,Nothing}=nothing,
    use_noise::Union{Bool,Nothing}=nothing,
    use_lag::Union{Bool,Nothing}=nothing,
    use_heel_effect::Union{Bool,Nothing}=nothing,
    use_das::Union{Bool,Nothing}=nothing,
    use_bhc::Union{Bool,Nothing}=nothing,
    use_pcct_corrections::Union{Bool,Nothing}=nothing,
    use_real_spectrum::Union{Bool,Nothing}=nothing,
    pcct_noise_reduction::Float64=0.0,
    n_energy_bins::Int=30,
    seed::Union{Int,Nothing}=42
)
    # Fidelity preset defaults for all 15 effects
    # :ideal = all OFF; :low = noise only; :medium = polychromatic subset; :high = all ON except DAS; :pcct = :high + corrections
    defaults = if fidelity == :pcct
        # Same as :high but with PCCT corrections enabled
        (fill_factor=true, flat_filter=true, bowtie_filter=true, detector_efficiency=true,
            scatter=true, scatter_correction=true, crosstalk=true, optical_crosstalk=true,
            focal_spot=true, noise=true, lag=true,
            heel_effect=true, das=false, bhc=true, pcct_corrections=true, real_spectrum=false)
    elseif fidelity == :high_plus
        # Real IPEM spectra + spectrum-domain filtering + MC detector efficiency
        # flat_filter=false: filtering done via CTProtocol(additional_filters=...) in spectrum domain
        # real_spectrum=true: loads unfiltered IPEM Anode spectra instead of CatSim pre-filtered
        # Requires: CTProtocol(anode_angle=8|10, additional_filters=[(...)])
        (fill_factor=true, flat_filter=false, bowtie_filter=true, detector_efficiency=true,
            scatter=true, scatter_correction=true, crosstalk=true, optical_crosstalk=true,
            focal_spot=true, noise=true, lag=true,
            heel_effect=true, das=false, bhc=true, pcct_corrections=false, real_spectrum=true)
    elseif fidelity == :high
        (fill_factor=true, flat_filter=true, bowtie_filter=true, detector_efficiency=true,
            scatter=true, scatter_correction=true, crosstalk=true, optical_crosstalk=true,
            focal_spot=true, noise=true, lag=true,
            heel_effect=true, das=false, bhc=true, pcct_corrections=false, real_spectrum=false)  # das=false: DAS model is BROKEN
    elseif fidelity == :medium
        # NOTE: flat_filter=true here for backward compat with pre-filtered CatSim spectra
        (fill_factor=false, flat_filter=true, bowtie_filter=false, detector_efficiency=false,
            scatter=false, scatter_correction=false, crosstalk=true, optical_crosstalk=false,
            focal_spot=true, noise=true, lag=false,
            heel_effect=false, das=false, bhc=true, pcct_corrections=false, real_spectrum=false)
    elseif fidelity == :low
        (fill_factor=false, flat_filter=false, bowtie_filter=false, detector_efficiency=false,
            scatter=false, scatter_correction=false, crosstalk=false, optical_crosstalk=false,
            focal_spot=false, noise=true, lag=false,
            heel_effect=false, das=false, bhc=false, pcct_corrections=false, real_spectrum=false)
    elseif fidelity == :ideal
        (fill_factor=false, flat_filter=false, bowtie_filter=false, detector_efficiency=false,
            scatter=false, scatter_correction=false, crosstalk=false, optical_crosstalk=false,
            focal_spot=false, noise=false, lag=false,
            heel_effect=false, das=false, bhc=false, pcct_corrections=false, real_spectrum=false)
    else
        error("Unknown fidelity preset: $fidelity. Use :pcct, :high_plus, :high, :medium, :low, or :ideal.")
    end

    # Resolve each toggle: user override wins, otherwise use preset default
    _fill_factor = isnothing(use_fill_factor) ? defaults.fill_factor : use_fill_factor
    _flat_filter = isnothing(use_flat_filter) ? defaults.flat_filter : use_flat_filter
    _bowtie_filter = isnothing(use_bowtie_filter) ? defaults.bowtie_filter : use_bowtie_filter
    _detector_efficiency = isnothing(use_detector_efficiency) ? defaults.detector_efficiency : use_detector_efficiency
    _scatter = isnothing(use_scatter) ? defaults.scatter : use_scatter
    _scatter_correction = isnothing(use_scatter_correction) ? defaults.scatter_correction : use_scatter_correction
    _crosstalk = isnothing(use_crosstalk) ? defaults.crosstalk : use_crosstalk
    _optical_crosstalk = isnothing(use_optical_crosstalk) ? defaults.optical_crosstalk : use_optical_crosstalk
    _focal_spot = isnothing(use_focal_spot) ? defaults.focal_spot : use_focal_spot
    _noise = isnothing(use_noise) ? defaults.noise : use_noise
    _lag = isnothing(use_lag) ? defaults.lag : use_lag
    _heel_effect = isnothing(use_heel_effect) ? defaults.heel_effect : use_heel_effect
    _das = isnothing(use_das) ? defaults.das : use_das
    _bhc = isnothing(use_bhc) ? defaults.bhc : use_bhc
    _pcct_corrections = isnothing(use_pcct_corrections) ? defaults.pcct_corrections : use_pcct_corrections
    _real_spectrum = isnothing(use_real_spectrum) ? defaults.real_spectrum : use_real_spectrum

    # === Mutual exclusion: use_real_spectrum=true → flat_filter must be OFF ===
    # When using physics-based unfiltered spectra with additional_filters (spectrum domain),
    # the projection-domain flat_filter is redundant and mutually exclusive.
    # Automatically disable flat_filter to prevent double-filtering.
    if _real_spectrum && _flat_filter
        _flat_filter = false
        @info "use_real_spectrum=true → flat_filter automatically disabled (mutually exclusive)." *
              " Filtering is handled by CTProtocol(additional_filters=...) in spectrum domain."
    end

    return SimOptions(
        fidelity,
        _fill_factor, _flat_filter, _bowtie_filter, _detector_efficiency,
        _scatter, _scatter_correction, _crosstalk, _optical_crosstalk,
        _focal_spot, _noise, _lag,
        _heel_effect, _das, _bhc,
        _pcct_corrections,
        _real_spectrum,
        clamp(pcct_noise_reduction, 0.0, 1.0),
        seed, n_energy_bins
    )
end

"""
    ReconOptions

Standardized options for image reconstruction.

Supports all reconstruction algorithms with algorithm-specific parameters.
Parameters irrelevant to the chosen algorithm are silently ignored.

# Core Fields
- `algorithm::Symbol`: Reconstruction algorithm (see below)
- `matrix_size::NTuple{3, Int}`: Output volume size (nx, ny, nz)
- `fov_cm::Float64`: XY field of view in cm
- `z_cm::Union{Float64, Nothing}`: Z extent in cm. If nothing, auto-computed from detector coverage.
  Set this to `sliceCount * sliceThickness / 10` for clinical-style slice thickness control.
- `filter::Symbol`: FDK filter kernel (:ram_lak, :shepp_logan, :cosine, :hamming, :hann, :standard, :soft, :bone)
- `iterations::Int`: Number of iterations (iterative methods)

# Iterative Parameters
- `lambda::Float64`: Relaxation/step size (SIRT, MBIR, ASIR)
- `tv_weight::Float64`: TV regularization strength (TV-SIRT, TV-CGLS)
- `n_subsets::Int`: Ordered subsets count (MBIR)
- `penalty::Symbol`: Regularizer type (:none, :quadratic, :huber, :hyperbola)
- `penalty_delta::Float64`: Huber/hyperbola delta parameter
- `use_edge_weights::Bool`: Edge-preserving weights (MBIR)
- `blend_percent::Float64`: FDK/iterative blend percentage (ASIR)

# VMI Parameters
- `vmi_energies::Vector{Float64}`: VMI energies to reconstruct (keV)
- `vmi_basis::Vector{Symbol}`: Material basis for decomposition (2+ materials; accepts Tuple for backward compat)

# Initialization
- `warm_start::Union{Nothing, AbstractArray}`: Initial estimate for iterative methods

# Supported Algorithms
| Symbol | Function | Key Params |
|--------|----------|-----------|
| :fdk | FDK (filtered backprojection) | filter |
| :sirt | SIRT (iterative) | iterations, lambda |
| :cgls | CGLS (conjugate gradient) | iterations |
| :tv_sirt | TV-SIRT (TV regularized) | iterations, lambda, tv_weight |
| :tv_cgls | TV-CGLS | iterations, tv_weight |
| :asir | ASIR-style blend | iterations, lambda, blend_percent |
| :mbir | Model-based IR | iterations, lambda, n_subsets, penalty |
"""
struct ReconOptions
    # Core fields
    algorithm::Symbol
    matrix_size::NTuple{3,Int}
    fov_cm::Float64
    z_cm::Union{Float64,Nothing}
    filter::Symbol
    iterations::Int
    # Iterative parameters
    lambda::Float64
    tv_weight::Float64
    n_subsets::Int
    penalty::Symbol
    penalty_delta::Float64
    use_edge_weights::Bool
    blend_percent::Float64
    # VMI parameters
    vmi_energies::Vector{Float64}
    vmi_basis::Vector{Symbol}
    # Initialization
    warm_start::Union{Nothing,AbstractArray}
    cascade_warm_start::Bool
end

"""
    ReconOptions(; kwargs...)

Create reconstruction options. All new fields have backward-compatible defaults.

# Examples
```julia
# Simple FDK (unchanged from before)
ReconOptions(algorithm=:fdk, matrix_size=(512, 512, 64), fov_cm=35.0)

# SIRT with custom lambda
ReconOptions(algorithm=:sirt, iterations=50, lambda=0.5)

# TV-SIRT with regularization
ReconOptions(algorithm=:tv_sirt, iterations=50, lambda=1.0, tv_weight=0.01)

# MBIR with penalty
ReconOptions(algorithm=:mbir, iterations=30, n_subsets=12, penalty=:hyperbola)

# VMI reconstruction request
ReconOptions(algorithm=:fdk, vmi_energies=[40.0, 50.0, 70.0, 100.0], vmi_basis=(:water, :iodine))
```
"""
function ReconOptions(;
    algorithm::Symbol=:fdk,
    matrix_size::Union{NTuple{3,Int},Nothing}=nothing,
    fov_cm::Real=35.0,
    z_cm::Union{Real,Nothing}=nothing,
    filter::Symbol=:standard,
    iterations::Int=10,
    # Iterative parameters
    lambda::Real=0.01,
    tv_weight::Real=0.0,
    n_subsets::Int=1,
    penalty::Symbol=:none,
    penalty_delta::Real=0.01,
    use_edge_weights::Bool=false,
    blend_percent::Real=50.0,
    # VMI parameters
    vmi_energies::Vector{Float64}=Float64[],
    vmi_basis::Union{Tuple{Symbol,Symbol},Vector{Symbol}}=(:water, :iodine),
    # Initialization
    warm_start::Union{Nothing,AbstractArray}=nothing,
    cascade_warm_start::Bool=false
)
    # Default to 512x512x64 if not specified
    _size = isnothing(matrix_size) ? (512, 512, 64) : matrix_size

    # Convert Tuple to Vector for backward compatibility
    _vmi_basis = vmi_basis isa Tuple ? collect(Symbol, vmi_basis) : Vector{Symbol}(vmi_basis)

    _z_cm = isnothing(z_cm) ? nothing : Float64(z_cm)

    return ReconOptions(
        algorithm, _size, Float64(fov_cm), _z_cm, filter, iterations,
        Float64(lambda), Float64(tv_weight), n_subsets,
        penalty, Float64(penalty_delta), use_edge_weights, Float64(blend_percent),
        vmi_energies, _vmi_basis,
        warm_start, cascade_warm_start
    )
end
