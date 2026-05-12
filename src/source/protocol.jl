"""
    src/source/protocol.jl

CT scan protocol definitions + validation + dose helpers + protocol
transformers.

  * `CTProtocol` struct — kVp / mA(s) / views / rotation_time / etc.
    Used by every notebook simulation.
  * `validate_protocol` — sanity-checks a protocol against scanner
    physical limits (kVp range, mA range, collimation vs scanner max).
  * Dose helpers — `compute_ctdi_vol`, `compute_dlp`, `dose_report`.
    Implements CTDIvol and DLP per the IEC 60601-2-44 / AAPM TG-200
    definitions, scaled by a generic-scanner calibration constant.
    Not currently consumed by any notebook but kept as the standard
    clinical dose-reporting surface for future protocol design.
  * Protocol transformers — `constant_dose_protocol`,
    `constant_noise_protocol` — for view-count sweeps where you want
    to hold either total dose or per-view noise fixed.

BasisSim-original — no upstream port for the dose formulas (different
scanners use different vendor-specific CTDI calibration tables; the
generic `_CTDI_CAL_CONSTANT` here is a starting point, not a clinical
ground truth).
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

# Examples
```julia
# Simple axial
CTProtocol(kVp=120, mA=200, views=984)

# With collimation (128×0.625mm = 80mm)
CTProtocol(kVp=120, mA=200, views=984, collimation_mm=80.0)

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
        additional_filters::Vector{Tuple{String, Float64}} = Tuple{String, Float64}[]
    )
    # Handle mA / mAs exclusivity
    final_mA = if !isnothing(mA)
        Float64(mA)
    elseif !isnothing(mAs)
        Float64(mAs) / Float64(rotation_time)
    else
        200.0
    end

    return CTProtocol(
        final_mA,
        Float64(kVp),
        Int(views),
        Float64(rotation_time),
        Float64(n_rotations),
        collimation_mm === nothing ? nothing : Float64(collimation_mm),
        anode_angle,
        additional_filters
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

# =============================================================================
# Dose Estimation (CTDI / DLP)
# =============================================================================

const _CTDI_CAL_CONSTANT = 0.05  # mGy/mAs at 120 kVp (generic scanner)

"""
    compute_ctdi_vol(protocol::CTProtocol; phantom_diameter::Real=320.0) -> Float64

Estimate CTDIvol (Volume CT Dose Index) in mGy.
"""
function compute_ctdi_vol(protocol::CTProtocol; phantom_diameter::Real = 320.0)
    mAs = protocol.mA * protocol.rotation_time
    kvp_factor = (protocol.kVp / 120.0)^2.5
    size_factor = (320.0 / phantom_diameter)^2
    return _CTDI_CAL_CONSTANT * mAs * kvp_factor * size_factor
end

"""
    compute_dlp(protocol::CTProtocol, scan_length_cm::Real; phantom_diameter::Real=320.0) -> Float64

Compute Dose-Length Product (DLP) in mGy·cm.
"""
function compute_dlp(protocol::CTProtocol, scan_length_cm::Real; phantom_diameter::Real = 320.0)
    ctdi = compute_ctdi_vol(protocol; phantom_diameter)
    return ctdi * scan_length_cm * protocol.n_rotations
end

"""
    dose_report(protocol::CTProtocol, geom::CTGeometry, spectrum_flux_sum::Float64; ...) -> NamedTuple

Generate a dose report for the given protocol and geometry.
"""
function dose_report(
        protocol::CTProtocol, geom::CTGeometry, spectrum_flux_sum::Float64;
        phantom_diameter::Real = 320.0,
        scan_length_cm::Union{Real, Nothing} = nothing
    )
    I0 = compute_detector_I0(geom, protocol, spectrum_flux_sum)
    mAs = protocol.mA * protocol.rotation_time
    ctdi = compute_ctdi_vol(protocol; phantom_diameter)
    sl = scan_length_cm !== nothing ? Float64(scan_length_cm) : geom.fov[3]
    dlp = compute_dlp(protocol, sl; phantom_diameter)
    total_photons = I0 * geom.n_cols * geom.n_rows * protocol.views

    println("="^50)
    println("CT Dose Report")
    println("="^50)
    println("  kVp: $(protocol.kVp), mA: $(protocol.mA), mAs: $(round(mAs, digits = 1))")
    println("  Views: $(protocol.views), Rotation: $(protocol.rotation_time) s")
    println("  CTDIvol: $(round(ctdi, digits = 2)) mGy, DLP: $(round(dlp, digits = 2)) mGy·cm")
    println("  I₀/pixel/view: $(round(I0, sigdigits = 4))")
    println("="^50)

    return (
        ctdi_vol = ctdi, dlp = dlp, I0_per_view = I0, total_photons = total_photons,
        mAs = mAs, kVp = protocol.kVp, views = protocol.views,
    )
end

export validate_protocol, compute_ctdi_vol, compute_dlp, dose_report

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
        base.additional_filters
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
        base.additional_filters
    )
end

export constant_dose_protocol, constant_noise_protocol
