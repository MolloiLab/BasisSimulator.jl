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

# Fields
- `fidelity::Symbol`: Preset level (:high, :medium, :low, :ideal). Default :high.
- `use_fill_factor::Bool`: Enable detector fill factor.
- `use_flat_filter::Bool`: Enable flat (inherent) filtration.
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

    # --- General ---
    seed::Union{Int, Nothing}
    n_energy_bins::Int
end

"""
    SimOptions(; kwargs...)

Create simulation options with fidelity presets and per-effect overrides.

# Presets (via `fidelity`)
- `:ideal`: All effects OFF — geometric ray tracing only.
- `:low`: Noise only.
- `:medium`: Noise + focal_spot + crosstalk + flat_filter + bhc (polychromatic).
- `:high`: All effects ON except DAS (BROKEN).

# Keyword Overrides
Pass `use_*=true/false` to override the fidelity preset for individual effects.
- `nothing` (default) = use preset
- `true` = force ON
- `false` = force OFF

# Examples
```julia
SimOptions(fidelity=:high)                         # Full physics
SimOptions(fidelity=:high, use_scatter=false)       # Everything except scatter
SimOptions(fidelity=:ideal, use_noise=true)         # Noise only (like old :low)
SimOptions(fidelity=:medium, use_bowtie_filter=true) # Medium + bowtie
```
"""
function SimOptions(;
    fidelity::Symbol = :high,
    use_fill_factor::Union{Bool, Nothing} = nothing,
    use_flat_filter::Union{Bool, Nothing} = nothing,
    use_bowtie_filter::Union{Bool, Nothing} = nothing,
    use_detector_efficiency::Union{Bool, Nothing} = nothing,
    use_scatter::Union{Bool, Nothing} = nothing,
    use_scatter_correction::Union{Bool, Nothing} = nothing,
    use_crosstalk::Union{Bool, Nothing} = nothing,
    use_optical_crosstalk::Union{Bool, Nothing} = nothing,
    use_focal_spot::Union{Bool, Nothing} = nothing,
    use_noise::Union{Bool, Nothing} = nothing,
    use_lag::Union{Bool, Nothing} = nothing,
    use_heel_effect::Union{Bool, Nothing} = nothing,
    use_das::Union{Bool, Nothing} = nothing,
    use_bhc::Union{Bool, Nothing} = nothing,
    n_energy_bins::Int = 30,
    seed::Union{Int, Nothing} = 42,
    # Deprecated kwarg — ignored but accepted for backwards compatibility
    use_beam_hardening::Union{Bool, Nothing} = nothing
)
    # Fidelity preset defaults for all 14 effects
    # :ideal = all OFF; :low = noise only; :medium = polychromatic subset; :high = all ON except DAS
    defaults = if fidelity == :high
        (fill_factor=true, flat_filter=true, bowtie_filter=true, detector_efficiency=true,
         scatter=true, scatter_correction=true, crosstalk=true, optical_crosstalk=true,
         focal_spot=true, noise=true, lag=true,
         heel_effect=true, das=false, bhc=true)  # das=false: DAS model is BROKEN
    elseif fidelity == :medium
        (fill_factor=false, flat_filter=true, bowtie_filter=false, detector_efficiency=false,
         scatter=false, scatter_correction=false, crosstalk=true, optical_crosstalk=false,
         focal_spot=true, noise=true, lag=false,
         heel_effect=false, das=false, bhc=true)
    elseif fidelity == :low
        (fill_factor=false, flat_filter=false, bowtie_filter=false, detector_efficiency=false,
         scatter=false, scatter_correction=false, crosstalk=false, optical_crosstalk=false,
         focal_spot=false, noise=true, lag=false,
         heel_effect=false, das=false, bhc=false)
    elseif fidelity == :ideal
        (fill_factor=false, flat_filter=false, bowtie_filter=false, detector_efficiency=false,
         scatter=false, scatter_correction=false, crosstalk=false, optical_crosstalk=false,
         focal_spot=false, noise=false, lag=false,
         heel_effect=false, das=false, bhc=false)
    else
        error("Unknown fidelity preset: $fidelity. Use :high, :medium, :low, or :ideal.")
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

    return SimOptions(
        fidelity,
        _fill_factor, _flat_filter, _bowtie_filter, _detector_efficiency,
        _scatter, _scatter_correction, _crosstalk, _optical_crosstalk,
        _focal_spot, _noise, _lag,
        _heel_effect, _das, _bhc,
        seed, n_energy_bins
    )
end

"""
    ReconOptions

Standardized options for image reconstruction.

# Fields
- `algorithm::Symbol`: :fdk, :sirt, :cgls. Default :fdk.
- `matrix_size::NTuple{3, Int}`: Output volume size (nx, ny, nz).
- `fov_cm::Float64`: Field of view in cm.
- `filter::Symbol`: FDK filter kernel (:ram_lak, :shepp_logan). Default :ram_lak.
- `iterations::Int`: Number of iterations (for SIRT/CGLS). Default 10.
"""
struct ReconOptions
    algorithm::Symbol
    matrix_size::NTuple{3, Int}
    fov_cm::Float64
    filter::Symbol
    iterations::Int
end

"""
    ReconOptions(; kwargs...)

Create reconstruction options. Defaults to standard 512x512x64 clinical volume.
"""
function ReconOptions(;
    algorithm::Symbol = :fdk,
    matrix_size::Union{NTuple{3, Int}, Nothing} = nothing,
    fov_cm::Real = 35.0,
    filter::Symbol = :ram_lak,
    iterations::Int = 10
)
    # Default to 512x512x64 if not specified
    _size = isnothing(matrix_size) ? (512, 512, 64) : matrix_size
    
    return ReconOptions(algorithm, _size, Float64(fov_cm), filter, iterations)
end
