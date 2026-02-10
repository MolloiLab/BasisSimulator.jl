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

# =============================================================================
# Protocol Validation
# =============================================================================

"""
    validate_protocol(protocol::CTProtocol, scanner::Scanner) -> (valid::Bool, messages::Vector{String})

Validate CT protocol parameters against physical constraints and scanner limits.

# Checks
- kVp in valid range (70-150 kVp)
- mA in valid range (10-1000 mA)
- rotation_time in valid range (0.2-5.0 s)
- views in valid range (100-5000)
- Warnings for unusual values (views < 500, views > 3000)

# Example
```julia
protocol = CTProtocol(kVp=120, mA=200, views=984)
scanner = Scanner()
valid, msgs = validate_protocol(protocol, scanner)
```
"""
function validate_protocol(protocol::CTProtocol, scanner::Scanner)
    messages = String[]
    valid = true

    # kVp range
    if !(70.0 ≤ protocol.kVp ≤ 150.0)
        push!(messages, "ERROR: kVp must be in [70, 150] (got $(protocol.kVp))")
        valid = false
    end

    # mA range
    if !(10.0 ≤ protocol.mA ≤ 1000.0)
        push!(messages, "ERROR: mA must be in [10, 1000] (got $(protocol.mA))")
        valid = false
    end

    # rotation_time range
    if !(0.2 ≤ protocol.rotation_time ≤ 5.0)
        push!(messages, "ERROR: rotation_time must be in [0.2, 5.0] s (got $(protocol.rotation_time))")
        valid = false
    end

    # views range
    if !(100 ≤ protocol.views ≤ 5000)
        push!(messages, "ERROR: views must be in [100, 5000] (got $(protocol.views))")
        valid = false
    end

    # Warnings for unusual views
    if 100 ≤ protocol.views < 500
        push!(messages, "WARNING: views=$(protocol.views) may be undersampled (< 500)")
    end
    if protocol.views > 3000
        push!(messages, "WARNING: views=$(protocol.views) is unusually high (> 3000)")
    end

    # Dual-energy low kVp validation
    if protocol.dual_energy
        if !(40.0 ≤ protocol.kVp_low ≤ 100.0)
            push!(messages, "ERROR: kVp_low must be in [40, 100] for dual-energy (got $(protocol.kVp_low))")
            valid = false
        end
        if protocol.kVp_low ≥ protocol.kVp
            push!(messages, "ERROR: kVp_low ($(protocol.kVp_low)) must be < kVp ($(protocol.kVp))")
            valid = false
        end
        if !(10.0 ≤ protocol.mA_low ≤ 1000.0)
            push!(messages, "ERROR: mA_low must be in [10, 1000] for dual-energy (got $(protocol.mA_low))")
            valid = false
        end
    end

    # flux_density sanity check
    if protocol.flux_density ≤ 0.0
        push!(messages, "ERROR: flux_density must be positive (got $(protocol.flux_density))")
        valid = false
    end

    return valid, messages
end

# =============================================================================
# Dose Estimation (CTDI / DLP)
# =============================================================================

# Scanner-specific CTDI calibration constants (mGy per mAs at 120 kVp, 32cm phantom)
# These are approximate values for estimation purposes.
const _CTDI_CAL_CONSTANT = 0.05  # mGy/mAs at 120 kVp (generic scanner)

"""
    compute_ctdi_vol(protocol::CTProtocol; phantom_diameter::Real=320.0) -> Float64

Estimate CTDIvol (Volume CT Dose Index) in mGy.

Uses the empirical formula:
    CTDIvol = C × mAs × (kVp/120)^2.5 / pitch

where C is a scanner-specific calibration constant (default: generic research scanner).

# Arguments
- `protocol`: CT protocol with mA, kVp, rotation_time, scan_mode, pitch

# Keyword Arguments
- `phantom_diameter`: Phantom diameter in mm (320 for body, 160 for head). Default: 320.

# Returns
CTDIvol estimate in mGy.

# Example
```julia
protocol = CTProtocol(kVp=120, mA=200, views=984, rotation_time=1.0)
ctdi = compute_ctdi_vol(protocol)  # ~10 mGy
```
"""
function compute_ctdi_vol(protocol::CTProtocol; phantom_diameter::Real=320.0)
    mAs = protocol.mA * protocol.rotation_time

    # kVp scaling: dose scales approximately as (kVp/120)^2.5
    kvp_factor = (protocol.kVp / 120.0)^2.5

    # Phantom size correction: smaller phantom → higher dose per mAs
    # Body (320mm) = reference, Head (160mm) ≈ 2× body CTDIvol
    size_factor = (320.0 / phantom_diameter)^2

    # Pitch correction (helical only)
    pitch_factor = if protocol.scan_mode == :helical && protocol.pitch > 0
        1.0 / protocol.pitch
    else
        1.0
    end

    return _CTDI_CAL_CONSTANT * mAs * kvp_factor * size_factor * pitch_factor
end

"""
    compute_dlp(protocol::CTProtocol, scan_length_cm::Real; phantom_diameter::Real=320.0) -> Float64

Compute Dose-Length Product (DLP) in mGy·cm.

    DLP = CTDIvol × scan_length × n_rotations

# Arguments
- `protocol`: CT protocol
- `scan_length_cm`: Scan length in cm

# Keyword Arguments
- `phantom_diameter`: Phantom diameter in mm (320 body, 160 head). Default: 320.

# Returns
DLP in mGy·cm.

# Example
```julia
protocol = CTProtocol(kVp=120, mA=200, rotation_time=1.0)
dlp = compute_dlp(protocol, 30.0)  # 30 cm scan
```
"""
function compute_dlp(protocol::CTProtocol, scan_length_cm::Real; phantom_diameter::Real=320.0)
    ctdi = compute_ctdi_vol(protocol; phantom_diameter)
    return ctdi * scan_length_cm * protocol.n_rotations
end

"""
    dose_report(protocol::CTProtocol, geom::CTGeometry; phantom_diameter::Real=320.0, scan_length_cm::Union{Real,Nothing}=nothing) -> NamedTuple

Generate a dose report for the given protocol and geometry.

# Arguments
- `protocol`: CT protocol
- `geom`: Scanner geometry (for I0 calculation)

# Keyword Arguments
- `phantom_diameter`: Phantom diameter in mm (default: 320)
- `scan_length_cm`: Scan length in cm (default: computed from geometry z-coverage)

# Returns
Named tuple with fields: `ctdi_vol`, `dlp`, `I0_per_view`, `total_photons`, `mAs`, `kVp`, `views`

Also prints a formatted summary.

# Example
```julia
protocol = CTProtocol(kVp=120, mA=200, views=984, rotation_time=1.0)
geom = create_aquilion_one(n_angles=984)
report = dose_report(protocol, geom)
```
"""
function dose_report(protocol::CTProtocol, geom::CTGeometry;
    phantom_diameter::Real=320.0,
    scan_length_cm::Union{Real,Nothing}=nothing
)
    # Compute I0 per pixel per view
    I0 = compute_detector_I0(geom, protocol)

    # mAs
    mAs = protocol.mA * protocol.rotation_time

    # Dose metrics
    ctdi = compute_ctdi_vol(protocol; phantom_diameter)

    # Scan length from z-coverage if not provided
    sl = scan_length_cm !== nothing ? Float64(scan_length_cm) : geom.fov[3]
    dlp = compute_dlp(protocol, sl; phantom_diameter)

    # Total photons (across all pixels, all views)
    total_photons = I0 * geom.n_cols * geom.n_rows * protocol.views

    # Print formatted report
    println("=" ^ 50)
    println("CT Dose Report")
    println("=" ^ 50)
    println("Protocol:")
    println("  kVp:            $(protocol.kVp)")
    println("  mA:             $(protocol.mA)")
    println("  mAs:            $(round(mAs, digits=1))")
    println("  Views:          $(protocol.views)")
    println("  Rotation time:  $(protocol.rotation_time) s")
    println("  Scan mode:      $(protocol.scan_mode)")
    if protocol.scan_mode == :helical
        println("  Pitch:          $(protocol.pitch)")
    end
    if protocol.dual_energy
        println("  Dual-energy:    kVp_low=$(protocol.kVp_low), mA_low=$(protocol.mA_low)")
    end
    println()
    println("Dose Estimates:")
    println("  CTDIvol:        $(round(ctdi, digits=2)) mGy")
    println("  DLP:            $(round(dlp, digits=2)) mGy·cm")
    println("  (phantom:       $(Int(phantom_diameter)) mm, scan: $(round(sl, digits=1)) cm)")
    println()
    println("Photon Statistics:")
    println("  I₀/pixel/view:  $(round(I0, sigdigits=4))")
    println("  Total photons:  $(round(total_photons, sigdigits=4))")
    println("=" ^ 50)

    return (
        ctdi_vol = ctdi,
        dlp = dlp,
        I0_per_view = I0,
        total_photons = total_photons,
        mAs = mAs,
        kVp = protocol.kVp,
        views = protocol.views
    )
end

export validate_protocol, compute_ctdi_vol, compute_dlp, dose_report