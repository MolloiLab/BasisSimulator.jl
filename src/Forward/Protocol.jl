"""
    Forward/Protocol.jl

Protocol definitions for CT simulation.
"""

"""
    CTProtocol

Scan protocol parameters for physical simulation.

Supports four scan modes via `scan_mode` and `dual_energy`:
- Axial single-kVp: `scan_mode=:axial, dual_energy=false` (default)
- Axial dual-kVp: `scan_mode=:axial, dual_energy=true`
- Helical single-kVp: `scan_mode=:helical, dual_energy=false`
- Helical dual-kVp: `scan_mode=:helical, dual_energy=true`

For dual-energy, `kVp` and `mA` are the HIGH energy settings.
`kVp_low` and `mA_low` are the LOW energy settings.

# Fields
- `mA`: Tube current (milliamperes) — high energy for dual-kVp
- `kVp`: Tube peak voltage (kV) — high energy for dual-kVp
- `views`: Number of projections per rotation
- `rotation_time`: Gantry rotation time in seconds
- `flux_density`: Reference photon flux density at 1m (photons/mm²/s)
- `spectrum_path`: Optional path to spectrum file
- `scan_mode`: `:axial` or `:helical`
- `pitch`: Table pitch for helical (0.0 for axial)
- `n_rotations`: Number of gantry rotations
- `dual_energy`: Whether this is a dual-kVp scan
- `kVp_low`: Low tube voltage for dual-energy (0.0 if single)
- `mA_low`: Low tube current for dual-energy (0.0 if single)
- `integration_fraction`: Fraction of views at low kVp (0.5 default)
"""
struct CTProtocol
    # Existing fields (unchanged)
    mA::Float64            # Tube current (high energy for DE)
    kVp::Float64           # Tube voltage (high energy for DE)
    views::Int             # Number of projections per rotation
    rotation_time::Float64 # Rotation time
    flux_density::Float64  # Reference flux
    spectrum_path::Union{String, Nothing}
    # New fields for 4-mode support
    scan_mode::Symbol      # :axial or :helical
    pitch::Float64         # Helical pitch (0.0 for axial)
    n_rotations::Float64   # Number of gantry rotations
    dual_energy::Bool      # Dual-kVp scan flag
    kVp_low::Float64       # Low kVp for dual-energy
    mA_low::Float64        # Low mA for dual-energy
    integration_fraction::Float64  # Fraction of views at low kVp
end

"""
    CTProtocol(; mA=nothing, mAs=nothing, kVp=120.0, views=984, rotation_time=1.0, ...)

Create a CT protocol. You must provide either `mA` OR `mAs`.

# Arguments
- `mA`: Tube current (e.g., 200.0) — high energy for dual-kVp
- `mAs`: Total mAs (e.g., 200.0). If provided, `mA` is calculated as `mAs / rotation_time`.
- `kVp`: Tube voltage (default: 120.0) — high energy for dual-kVp
- `views`: Projections per rotation (default: 984)
- `rotation_time`: Rotation time in seconds (default: 1.0)
- `flux_density`: Reference flux density (default: 2.0e6)
- `spectrum_path`: Custom spectrum file (default: nothing)
- `scan_mode`: `:axial` or `:helical` (default: :axial)
- `pitch`: Helical pitch factor (default: 0.0, required > 0 for helical)
- `n_rotations`: Number of gantry rotations (default: 1.0)
- `dual_energy`: Enable dual-kVp mode (default: false)
- `kVp_low`: Low tube voltage for DE (default: 0.0, required > 0 when dual_energy=true)
- `mA_low`: Low tube current for DE (default: 0.0)
- `integration_fraction`: Fraction of views at low kVp (default: 0.5)

# Examples
```julia
# Simple axial (backward compatible)
CTProtocol(kVp=120, mA=200, views=984)

# Helical
CTProtocol(scan_mode=:helical, kVp=120, mA=200, views=984, pitch=0.984, n_rotations=10.0)

# Dual-energy axial
CTProtocol(dual_energy=true, kVp=140, mA=200, kVp_low=80, mA_low=350, views=984)

# Dual-energy helical
CTProtocol(scan_mode=:helical, dual_energy=true, kVp=140, mA=200, kVp_low=80, mA_low=350, pitch=0.531, n_rotations=5.0)
```
"""
function CTProtocol(;
    mA=nothing,
    mAs=nothing,
    kVp=120.0,
    views=984,
    rotation_time=1.0,
    flux_density=2.0e6,
    spectrum_path=nothing,
    # New fields for 4-mode support
    scan_mode::Symbol=:axial,
    pitch::Real=0.0,
    n_rotations::Real=1.0,
    dual_energy::Bool=false,
    kVp_low::Real=0.0,
    mA_low::Real=0.0,
    integration_fraction::Real=0.5
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

    # Validate scan_mode
    if scan_mode ∉ (:axial, :helical)
        error("scan_mode must be :axial or :helical (got :$scan_mode)")
    end

    # Validate helical requirements
    if scan_mode == :helical && pitch <= 0.0
        error("Helical mode requires pitch > 0 (got $pitch)")
    end

    # Validate dual-energy requirements
    if dual_energy && kVp_low <= 0.0
        error("Dual-energy mode requires kVp_low > 0 (got $kVp_low)")
    end

    return CTProtocol(
        final_mA,
        Float64(kVp),
        Int(views),
        Float64(rotation_time),
        Float64(flux_density),
        spectrum_path,
        scan_mode,
        Float64(pitch),
        Float64(n_rotations),
        dual_energy,
        Float64(kVp_low),
        Float64(mA_low),
        Float64(integration_fraction)
    )
end

export CTProtocol