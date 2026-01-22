"""
    Forward/Protocol.jl

Protocol definitions for CT simulation.
"""

"""
    CTProtocol

Scan protocol parameters for physical simulation.

# Fields
- `mA`: Tube current (milliamperes)
- `kVp`: Tube peak voltage (kV)
- `views`: Number of projections per rotation
- `rotation_time`: Gantry rotation time in seconds
- `flux_density`: Reference photon flux density at 1m (photons/mm²/s)
- `spectrum_path`: Optional path to spectrum file
"""
struct CTProtocol
    mA::Float64            # Tube current
    kVp::Float64           # Tube voltage
    views::Int             # Number of projections
    rotation_time::Float64 # Rotation time
    flux_density::Float64  # Reference flux
    spectrum_path::Union{String, Nothing}
end

"""
    CTProtocol(; mA=nothing, mAs=nothing, kVp=120.0, views=984, rotation_time=1.0, flux_density=2.0e6)

Create a CT protocol. You must provide either `mA` OR `mAs`.

# Arguments
- `mA`: Tube current (e.g., 200.0)
- `mAs`: Total mAs (e.g., 200.0). If provided, `mA` is calculated as `mAs / rotation_time`.
"""
function CTProtocol(; 
    mA=nothing,
    mAs=nothing,
    kVp=120.0, 
    views=984, 
    rotation_time=1.0, 
    flux_density=2.0e6,
    spectrum_path=nothing
)
    # Handle mA / mAs exclusivity
    final_mA = if !isnothing(mA)
        Float64(mA)
    elseif !isnothing(mAs)
        Float64(mAs) / Float64(rotation_time)
    else
        # Default fallback if neither provided
        200.0
    end

    return CTProtocol(
        final_mA, 
        Float64(kVp), 
        Int(views), 
        Float64(rotation_time), 
        Float64(flux_density), 
        spectrum_path
    )
end

export CTProtocol