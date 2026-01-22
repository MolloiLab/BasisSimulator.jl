"""
    Simulation/Options.jl

High-level options for controlling simulation fidelity and reconstruction.
"""

export SimOptions, ReconOptions

"""
    SimOptions

Controls the fidelity and physics realism of the simulation.
Defaults are chosen to align with CatSim clinical accuracy.

# Fields
- `fidelity::Symbol`: Preset level (:high, :medium, :low, :ideal). Default :high.
- `use_scatter::Bool`: Enable scatter simulation (expensive). Default true.
- `use_noise::Bool`: Enable quantum/electronic noise. Default true.
- `use_beam_hardening::Bool`: Enable polychromatic physics. Default true.
- `use_crosstalk::Bool`: Enable detector crosstalk. Default true.
- `use_focal_spot::Bool`: Enable focal spot blur. Default true.
- `seed::Union{Int, Nothing}`: Random seed for reproducibility. Default 42.
"""
struct SimOptions
    fidelity::Symbol
    use_scatter::Bool
    use_noise::Bool
    use_beam_hardening::Bool
    use_crosstalk::Bool
    use_focal_spot::Bool
    seed::Union{Int, Nothing}
end

"""
    SimOptions(; kwargs...)

Create simulation options.

# Presets (via `fidelity`)
- `:high` (Default): Full CatSim realism (Scatter, Noise, Poly, Blur, Crosstalk)
- `:medium`: No scatter (faster), but includes Noise, Poly, Blur.
- `:low`: Monochromatic, Noise only.
- `:ideal`: Geometric ray tracing only (no noise, no physics blur).

# Keyword Overrides
You can pass specific flags to override the preset defaults.
Example: `SimOptions(fidelity=:high, use_scatter=false)`
"""
function SimOptions(;
    fidelity::Symbol = :high,
    use_scatter::Union{Bool, Nothing} = nothing,
    use_noise::Union{Bool, Nothing} = nothing,
    use_beam_hardening::Union{Bool, Nothing} = nothing,
    use_crosstalk::Union{Bool, Nothing} = nothing,
    use_focal_spot::Union{Bool, Nothing} = nothing,
    seed::Union{Int, Nothing} = 42
)
    # 1. Set base defaults based on fidelity preset
    defaults = if fidelity == :high
        (scatter=true, noise=true, poly=true, cross=true, blur=true)
    elseif fidelity == :medium
        (scatter=false, noise=true, poly=true, cross=true, blur=true)
    elseif fidelity == :low
        (scatter=false, noise=true, poly=false, cross=false, blur=false)
    elseif fidelity == :ideal
        (scatter=false, noise=false, poly=false, cross=false, blur=false)
    else
        error("Unknown fidelity preset: $fidelity. Use :high, :medium, :low, or :ideal.")
    end

    # 2. Apply overrides (if provided) or use defaults
    _scatter = isnothing(use_scatter) ? defaults.scatter : use_scatter
    _noise = isnothing(use_noise) ? defaults.noise : use_noise
    _poly = isnothing(use_beam_hardening) ? defaults.poly : use_beam_hardening
    _cross = isnothing(use_crosstalk) ? defaults.cross : use_crosstalk
    _blur = isnothing(use_focal_spot) ? defaults.blur : use_focal_spot

    return SimOptions(
        fidelity,
        _scatter,
        _noise,
        _poly,
        _cross,
        _blur,
        seed
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
