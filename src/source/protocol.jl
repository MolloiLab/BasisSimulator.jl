"""
    src/source/protocol.jl

CT scan protocol definitions + validation + protocol
transformers.

  * `CTProtocol` struct — kVp / mA(s) / views / rotation_time / etc.
    Used by every notebook simulation.
  * `validate_protocol` — sanity-checks a protocol against scanner
    physical limits (kVp range, mA range, collimation vs scanner max).
  * Protocol transformers — `constant_dose_protocol`,
    `constant_noise_protocol` — for view-count sweeps where you want
    to hold either total dose or per-view noise fixed.

Dose (CTDI100 / CTDIw / CTDIvol / DLP) lives in `source/dose.jl`, computed from the simulated beam.
"""

"""
    CTProtocol

Scan protocol parameters for physical simulation.

# Fields
- `mA`: Tube current (milliamperes)
- `kVp`: Tube peak voltage (kV)
- `views`: Number of projections per rotation
- `rotation_time`: Gantry rotation time in seconds
- `n_rotations`: Number of gantry rotations
- `collimation_mm`: Detector z-collimation in mm (nothing = use full detector)
- `anode_angle`: IPEM anode angle in degrees (8 or 10)
- `additional_filters`: Extra filter layers `[(material, thickness_mm), ...]` applied
  on top of the scanner's built-in flat filter in the spectrum domain.
- `pitch`: Helical pitch (IEC: table feed per rotation ÷ collimation); `nothing` = axial.
"""
struct CTProtocol
    mA::Float64            # Tube current
    kVp::Float64           # Tube voltage
    views::Int             # Number of projections per rotation
    rotation_time::Float64 # Rotation time
    n_rotations::Float64   # Number of gantry rotations
    collimation_mm::Union{Float64, Nothing}  # Detector z-collimation (mm), nothing = full detector
    anode_angle::Int       # IPEM anode angle (8 or 10 degrees)
    additional_filters::Vector{Tuple{String, Float64}}  # Extra filter layers [(material, thickness_mm)]
    pitch::Union{Float64, Nothing}  # Helical pitch (IEC: table feed per rotation / collimation); nothing = axial
end

"""
    CTProtocol(; mA=nothing, mAs=nothing, kVp=120.0, views=984, rotation_time=1.0, ...)

Create a CT protocol. You must provide either `mA` OR `mAs`.

# Arguments
- `mA`: Tube current (e.g., 200.0)
- `mAs`: Total mAs (e.g., 200.0). If provided, `mA` is calculated as `mAs / rotation_time`.
- `kVp`: Tube voltage (default: 120.0)
- `views`: Projections per rotation (default: 984)
- `rotation_time`: Rotation time in seconds (default: 1.0)
- `n_rotations`: Number of gantry rotations (default: 1.0)
- `collimation_mm`: Detector z-collimation in mm (default: nothing = full detector)
- `anode_angle`: IPEM anode angle, 8 or 10 degrees (default: 10)
- `additional_filters`: Extra filter layers `[(material, thickness_mm), ...]` (default: empty)
- `pitch`: Helical pitch (default: nothing = axial scan).  IEC 60601-2-44 definition:
  table feed per rotation ÷ total active collimation width.  When set, the
  gantry traces a helix of `n_rotations` turns centred on isocentre — the table
  travel is `pitch × collimation × n_rotations`.  Typical clinical range
  0.5–1.5 (dual-source Flash up to 3.2).

# Examples
```julia
# Simple axial
CTProtocol(kVp=120, mA=200, views=984)

# With collimation (128×0.625mm = 80mm)
CTProtocol(kVp=120, mA=200, views=984, collimation_mm=80.0)

# Helical: pitch 1.0, 8 rotations, 40 mm collimation → 320 mm table travel
CTProtocol(kVp=120, mA=200, views=984, collimation_mm=40.0, pitch=1.0, n_rotations=8)

# Extra filtration
CTProtocol(kVp=120, mA=200, additional_filters=[("Al", 4.5)])
```
"""
function CTProtocol(;
        mA = nothing,
        mAs = nothing,
        kVp = 120.0,
        views = 984,
        rotation_time = 1.0,
        n_rotations::Real = 1.0,
        collimation_mm::Union{Real, Nothing} = nothing,
        anode_angle::Int = 10,
        additional_filters::Vector{Tuple{String, Float64}} = Tuple{String, Float64}[],
        pitch::Union{Real, Nothing} = nothing,
    )
    # Handle mA / mAs exclusivity
    final_mA = if !isnothing(mA)
        Float64(mA)
    elseif !isnothing(mAs)
        Float64(mAs) / Float64(rotation_time)
    else
        200.0
    end

    if pitch !== nothing
        pitch > 0 || throw(ArgumentError("pitch must be positive, got $pitch"))
        n_rotations >= 1 || throw(ArgumentError(
            "helical scans need n_rotations ≥ 1, got $n_rotations"))
    end

    return CTProtocol(
        final_mA,
        Float64(kVp),
        Int(views),
        Float64(rotation_time),
        Float64(n_rotations),
        collimation_mm === nothing ? nothing : Float64(collimation_mm),
        anode_angle,
        additional_filters,
        pitch === nothing ? nothing : Float64(pitch),
    )
end

export CTProtocol

# =============================================================================
# Protocol Validation
# =============================================================================

"""
    validate_protocol(protocol::CTProtocol, scanner::Scanner) -> (valid::Bool, messages::Vector{String})

Validate CT protocol parameters against physical constraints and scanner limits.
"""
function validate_protocol(protocol::CTProtocol, scanner::Scanner)
    messages = String[]
    valid = true

    if !(70.0 ≤ protocol.kVp ≤ 150.0)
        push!(messages, "ERROR: kVp must be in [70, 150] (got $(protocol.kVp))")
        valid = false
    end

    if !(10.0 ≤ protocol.mA ≤ 1000.0)
        push!(messages, "ERROR: mA must be in [10, 1000] (got $(protocol.mA))")
        valid = false
    end

    if !(0.2 ≤ protocol.rotation_time ≤ 5.0)
        push!(messages, "ERROR: rotation_time must be in [0.2, 5.0] s (got $(protocol.rotation_time))")
        valid = false
    end

    if !(100 ≤ protocol.views ≤ 5000)
        push!(messages, "ERROR: views must be in [100, 5000] (got $(protocol.views))")
        valid = false
    end

    if protocol.collimation_mm !== nothing
        if protocol.collimation_mm <= 0
            push!(messages, "ERROR: collimation_mm must be positive (got $(protocol.collimation_mm))")
            valid = false
        end
        max_mm = scanner.detector_rows * scanner.detector_row_size
        if protocol.collimation_mm > max_mm
            push!(messages, "ERROR: collimation_mm ($(protocol.collimation_mm)) exceeds scanner max ($max_mm mm)")
            valid = false
        end
    end

    return valid, messages
end

# Dose (CTDI100, CTDIw, CTDIvol, DLP) is computed from the simulated beam in `source/dose.jl`.

export validate_protocol

# =============================================================================
# Constant-Dose / Constant-Noise Protocol Helpers
# =============================================================================

"""
    constant_dose_protocol(base::CTProtocol, new_views::Int) -> CTProtocol

Create a new protocol with same mA (same dose) but different view count.
"""
function constant_dose_protocol(base::CTProtocol, new_views::Int)
    return CTProtocol(
        base.mA, base.kVp, new_views, base.rotation_time,
        base.n_rotations, base.collimation_mm, base.anode_angle,
        base.additional_filters, base.pitch
    )
end

"""
    constant_noise_protocol(base::CTProtocol, new_views::Int) -> CTProtocol

Create a new protocol with adjusted mA to maintain constant noise per view.
new_mA = base_mA × (new_views / base_views).
"""
function constant_noise_protocol(base::CTProtocol, new_views::Int)
    new_mA = base.mA * (new_views / base.views)
    return CTProtocol(
        new_mA, base.kVp, new_views, base.rotation_time,
        base.n_rotations, base.collimation_mm, base.anode_angle,
        base.additional_filters, base.pitch
    )
end

export constant_dose_protocol, constant_noise_protocol
