"""
    src/geometry/scanner.jl

CT scanner geometry definitions and pre-computed trajectory positions.

This file provides two complementary abstractions:

1. `Scanner{T}` — Generic scanner definition struct with all physical
   parameters.  Accepts kwargs for flexible scanner configuration.
   Parameter naming and units follow CatSim/XCIST conventions (see the
   CatSim Parameter Mapping table in the `Scanner` docstring).

2. `CTGeometry` — Pre-computed source/detector trajectory positions for
   simulation.  All positions computed at construction time so the
   per-view ray geometry is JIT-friendly (no runtime trig).

# Workflow
```julia
# Define a scanner with physical parameters
scanner = Scanner(
    source_to_isocenter = 541.0,  # mm
    source_to_detector = 949.0,   # mm
    detector_rows = 64,
    detector_cols = 900,
)

# Create computed geometry for simulation
geom = CTGeometry(scanner; n_angles = 360, fov_cm = 35.0)
```
"""

# =============================================================================
# Scanner Definition Struct (Physical Parameters)
# =============================================================================

"""
    DetectorShape

Enumeration of detector array geometries.
"""
@enum DetectorShape begin
    CURVED_DETECTOR    # Third-generation CT: curved detector array
    FLAT_DETECTOR      # Flat panel detectors (CBCT, interventional)
end

"""
    Scanner{T<:AbstractFloat}

Generic CT scanner definition with all physical parameters.

This struct defines the physical scanner configuration. Use `CTGeometry` for
simulation with pre-computed trajectories. All distances in mm for consistency
with CatSim and medical imaging conventions.

# Geometry Parameters (Required)
- `source_to_isocenter::T`: Source-to-isocenter distance (mm), aka SID/SOD
- `source_to_detector::T`: Source-to-detector distance (mm), aka SDD

# Detector Parameters
- `detector_rows::Int`: Number of detector rows (z-direction)
- `detector_cols::Int`: Number of detector columns (fan direction)
- `detector_row_size::T`: Detector element size in z (mm) at isocenter
- `detector_col_size::T`: Detector element size in fan direction (mm) at isocenter
- `detector_shape::DetectorShape`: Detector geometry (:curved or :flat)
- `detector_row_offset::T`: Row offset from centered position (rows)
- `detector_col_offset::T`: Column offset (quarter-detector offset for aliasing)

# Source Parameters
- `focal_spot_width::T`: Focal spot width (mm)
- `focal_spot_length::T`: Focal spot length (mm)
- `target_angle::T`: Anode target angle (degrees)

# Gantry Parameters
- `gantry_rotation_time::T`: Gantry rotation time (seconds)
- `scan_diameter::T`: Maximum scan diameter (mm)
- `gantry_aperture::T`: Gantry bore diameter (mm)

# Filter Parameters
- `flat_filter_material::Symbol`: Flat filter material (:aluminum, :copper, :titanium)
- `flat_filter_thickness::T`: Flat filter thickness (mm)
- `bowtie_filter::Symbol`: Bowtie filter name (:large_body, :medium_body, :small_body, :head, :none)

# Detection Parameters
- `detector_material::Symbol`: Detector scintillator/sensor material
- `detector_depth::T`: Detector sensor depth (mm)
- `fill_factor_row::T`: Active area fraction (row direction, 0-1)
- `fill_factor_col::T`: Active area fraction (column direction, 0-1)
- `detection_gain::T`: Conversion gain (electrons/keV)
- `electronic_noise::T`: Electronic noise std dev (electrons)

# Constructor
```julia
Scanner(;
    source_to_isocenter = 540.0,
    source_to_detector = 950.0,
    detector_rows = 64,
    detector_cols = 900,
    # ... other kwargs with defaults
)
```

# CatSim Parameter Mapping
| Scanner Field | CatSim Parameter |
|---------------|------------------|
| source_to_isocenter | scanner.sid |
| source_to_detector | scanner.sdd |
| detector_rows | scanner.detectorRowCount |
| detector_cols | scanner.detectorColCount |
| detector_row_size | scanner.detectorRowSize |
| detector_col_size | scanner.detectorColSize |
| detector_row_offset | scanner.detectorRowOffset |
| detector_col_offset | scanner.detectorColOffset |
| target_angle | scanner.targetAngle |
| focal_spot_width | scanner.focalspotWidth |
| focal_spot_length | scanner.focalspotLength |
| detector_material | scanner.detectorMaterial |
| detector_depth | scanner.detectorDepth |
| fill_factor_row | scanner.detectorRowFillFraction |
| fill_factor_col | scanner.detectorColFillFraction |
| detection_gain | scanner.detectionGain |
| electronic_noise | scanner.eNoise |

# References
- CatSim scanner configuration: cfg/Scanner_Default.cfg
- AAPM TG-233: CT Image Quality Standards
"""
struct Scanner{T <: AbstractFloat}
    # Geometry (required)
    source_to_isocenter::T      # mm (SID/SOD)
    source_to_detector::T       # mm (SDD)

    # Detector array
    detector_rows::Int          # number of rows
    detector_cols::Int          # number of columns
    detector_row_size::T        # mm at isocenter
    detector_col_size::T        # mm at isocenter
    detector_shape::DetectorShape
    detector_row_offset::T      # rows
    detector_col_offset::T      # columns (quarter-detector offset)

    # Source/focal spot
    focal_spot_width::T         # mm
    focal_spot_length::T        # mm
    target_angle::T             # degrees

    # Gantry
    gantry_rotation_time::T     # seconds
    scan_diameter::T             # mm
    gantry_aperture::T          # mm

    # Filters
    flat_filter_material::Symbol
    flat_filter_thickness::T    # mm
    bowtie_filter::Symbol       # :large_body, :medium_body, :small_body, :head, :none, etc.

    # Detection
    detector_material::Symbol
    detector_depth::T           # mm
    fill_factor_row::T          # 0-1
    fill_factor_col::T          # 0-1
    detection_gain::T           # electrons/keV
    electronic_noise::T         # electrons

    # === PCCT Fields (flat kwargs, defaults = conventional EID behavior) ===
    # Ignored when detector_type == :energy_integrating
    detector_type::Symbol       # :energy_integrating (default) or :photon_counting
    n_energy_bins::Int          # 1 (EID) or 2-8 (PCCT)
    energy_thresholds::Vector{T}  # Energy thresholds keV (empty for EID)
    energy_resolution::T        # Detector FWHM keV (0.0 for EID)
    charge_sharing_fwhm::T      # Charge cloud FWHM mm (0.0 for EID)
    dead_time_ns::T             # Pulse dead time ns (0.0 for EID)
    pixel_mode::Symbol          # :standard, :uhr, :macro

    # Native dexel parameters (PCCT only — EID ignores these)
    native_dexel_col_mm::T      # native dexel col size at detector face (mm); 0 = infer
    native_dexel_row_mm::T      # native dexel row size at detector face (mm); 0 = infer
    binning_factor::Int         # spatial binning (1=unbinned, 2=2×2 standard)
end

"""
    Scanner(; kwargs...)

Construct a Scanner with configurable parameters via kwargs.

All distances are in mm. Default values match a generic research CT scanner
(similar to CatSim defaults).

# Keyword Arguments (with defaults)
- `source_to_isocenter::Real = 540.0`: Source-to-isocenter distance (mm)
- `source_to_detector::Real = 950.0`: Source-to-detector distance (mm)
- `detector_rows::Int = 64`: Number of detector rows
- `detector_cols::Int = 900`: Number of detector columns
- `detector_row_size::Real = 1.0`: Detector row pitch (mm)
- `detector_col_size::Real = 1.0`: Detector column pitch (mm)
- `detector_shape::DetectorShape = CURVED_DETECTOR`: Detector geometry
- `detector_row_offset::Real = 0.0`: Row offset (rows)
- `detector_col_offset::Real = 0.25`: Column offset for quarter-detector shift
- `focal_spot_width::Real = 1.0`: Focal spot width (mm)
- `focal_spot_length::Real = 1.0`: Focal spot length (mm)
- `target_angle::Real = 7.0`: Anode target angle (degrees)
- `gantry_rotation_time::Real = 0.5`: Rotation time (seconds)
- `scan_diameter::Real = 500.0`: Maximum scan diameter (mm)
- `gantry_aperture::Real = 700.0`: Gantry bore diameter (mm)
- `flat_filter_material::Symbol = :aluminum`: Flat filter material
- `flat_filter_thickness::Real = 2.0`: Flat filter thickness (mm)
- `bowtie_filter::Symbol = :large_body`: Bowtie filter (`:large_body`, `:medium_body`, `:small_body`, `:head`, `:none`)
- `detector_material::Symbol = :lumex`: Detector scintillator
- `detector_depth::Real = 3.0`: Detector depth (mm)
- `fill_factor_row::Real = 0.9`: Row fill factor (0-1)
- `fill_factor_col::Real = 0.9`: Column fill factor (0-1)
- `detection_gain::Real = 15.0`: Detection gain (electrons/keV)
- `electronic_noise::Real = 5000.0`: Electronic noise (electrons)

# Example
```julia
# Generic research scanner (defaults)
scanner = Scanner()

# Custom scanner with specific geometry
scanner = Scanner(
    source_to_isocenter = 626.0,  # GE Revolution-like
    source_to_detector = 1097.0,
    detector_rows = 256,
    detector_cols = 832,
    detector_row_size = 0.625,
    target_angle = 10.0
)

# Flat panel scanner
scanner = Scanner(
    detector_shape = FLAT_DETECTOR,
    detector_rows = 512,
    detector_cols = 512,
    detector_row_size = 0.15,
    detector_col_size = 0.15
)
```
"""
function Scanner(;
        # Geometry (CatSim defaults)
        source_to_isocenter::Real = 540.0,
        source_to_detector::Real = 950.0,

        # Detector array
        detector_rows::Int = 64,
        detector_cols::Int = 900,
        detector_row_size::Real = 1.0,
        detector_col_size::Real = 1.0,
        detector_shape::DetectorShape = CURVED_DETECTOR,
        detector_row_offset::Real = 0.0,
        detector_col_offset::Real = 0.25,

        # Source/focal spot
        focal_spot_width::Real = 1.0,
        focal_spot_length::Real = 1.0,
        target_angle::Real = 7.0,

        # Gantry
        gantry_rotation_time::Real = 0.5,
        scan_diameter::Real = 500.0,
        gantry_aperture::Real = 700.0,

        # Filters
        flat_filter_material::Symbol = :aluminum,
        flat_filter_thickness::Real = 2.0,
        bowtie_filter::Symbol = :large_body,

        # Detection
        detector_material::Symbol = :lumex,
        detector_depth::Real = 3.0,
        fill_factor_row::Real = 0.9,
        fill_factor_col::Real = 0.9,
        detection_gain::Real = 15.0,
        electronic_noise::Real = 5000.0,

        # PCCT fields (flat kwargs — ignored when detector_type == :energy_integrating)
        detector_type::Symbol = :energy_integrating,
        n_energy_bins::Int = 1,
        energy_thresholds::Vector{<:Real} = Float64[],
        energy_resolution::Real = 0.0,
        charge_sharing_fwhm::Real = 0.0,
        dead_time_ns::Real = 0.0,
        pixel_mode::Symbol = :standard,

        # Native dexel parameters (PCCT only)
        native_dexel_col_mm::Real = 0.0,
        native_dexel_row_mm::Real = 0.0,
        binning_factor::Int = 1
    )
    T = Float64

    # PCCT validation
    if detector_type == :photon_counting
        if isempty(energy_thresholds)
            error("PCCT scanner requires energy_thresholds (got empty vector)")
        end
        if n_energy_bins != length(energy_thresholds)
            error("n_energy_bins ($n_energy_bins) must equal length(energy_thresholds) ($(length(energy_thresholds)))")
        end
        if !issorted(energy_thresholds)
            error("energy_thresholds must be sorted ascending (got $energy_thresholds)")
        end
        if !(pixel_mode in (:standard, :uhr, :macro))
            error("pixel_mode must be :standard, :uhr, or :macro (got :$pixel_mode)")
        end
    elseif detector_type != :energy_integrating
        error("detector_type must be :energy_integrating or :photon_counting (got :$detector_type)")
    end

    # Binning factor validation
    if binning_factor < 1
        error("binning_factor must be >= 1 (got $binning_factor)")
    end

    # Infer native dexel from binned pixel size × magnification when 0.0
    magnification_val = Float64(source_to_detector) / Float64(source_to_isocenter)
    _native_col = if native_dexel_col_mm > 0.0
        native_dexel_col_mm
    else
        # Infer: binned pixel at iso × magnification / binning = native dexel at detector
        detector_col_size * magnification_val / binning_factor
    end
    _native_row = if native_dexel_row_mm > 0.0
        native_dexel_row_mm
    else
        detector_row_size * magnification_val / binning_factor
    end

    return Scanner{T}(
        T(source_to_isocenter),
        T(source_to_detector),
        detector_rows,
        detector_cols,
        T(detector_row_size),
        T(detector_col_size),
        detector_shape,
        T(detector_row_offset),
        T(detector_col_offset),
        T(focal_spot_width),
        T(focal_spot_length),
        T(target_angle),
        T(gantry_rotation_time),
        T(scan_diameter),
        T(gantry_aperture),
        flat_filter_material,
        T(flat_filter_thickness),
        bowtie_filter,
        detector_material,
        T(detector_depth),
        T(fill_factor_row),
        T(fill_factor_col),
        T(detection_gain),
        T(electronic_noise),
        detector_type,
        n_energy_bins,
        T.(energy_thresholds),
        T(energy_resolution),
        T(charge_sharing_fwhm),
        T(dead_time_ns),
        pixel_mode,
        T(_native_col),
        T(_native_row),
        binning_factor
    )
end

# =============================================================================
# CTGeometry - Pre-computed Trajectory Positions
# =============================================================================

"""
    CTGeometry

CT scanner geometry with pre-computed trajectories.

All source and detector positions are pre-computed at construction time
to enable Reactant/XLA compilation (no runtime trig).

# Fields
- `SAD::Float64`: Source-to-axis distance (cm)
- `SDD::Float64`: Source-to-detector distance (cm)
- `n_angles::Int`: Number of projection angles
- `n_rows::Int`: Detector rows (z direction)
- `n_cols::Int`: Detector columns (fan direction)
- `pixel_size::Float64`: Detector pixel size (cm) at isocenter
- `angles::Vector{Float64}`: Projection angles (radians)
- `source_positions::Matrix{Float64}`: [3, n_angles] source XYZ positions
- `detector_centers::Matrix{Float64}`: [3, n_angles] detector center XYZ
- `detector_u::Matrix{Float64}`: [3, n_angles] detector u-axis (column direction)
- `detector_v::Matrix{Float64}`: [3, n_angles] detector v-axis (row direction)
- `fov::NTuple{3, Float64}`: (fov_x, fov_y, fov_z) volume FOV in cm

# Coordinate System
- X: left-right (increasing right)
- Y: anterior-posterior (source starts at -SAD on Y-axis)
- Z: inferior-superior (increasing superior)
- Rotation around Z-axis, counter-clockwise when viewed from above
"""
struct CTGeometry
    SAD::Float64
    SDD::Float64
    n_angles::Int
    n_rows::Int
    n_cols::Int
    pixel_size::Float64
    pixel_row_size::Float64    # row-based pixel size at isocenter (cm)
    angles::Vector{Float64}
    source_positions::Matrix{Float64}
    detector_centers::Matrix{Float64}
    detector_u::Matrix{Float64}
    detector_v::Matrix{Float64}
    fov::NTuple{3, Float64}  # (fov_x, fov_y, fov_z) in cm
end

"""
    CTGeometry(scanner::Scanner; n_angles=360, fov_cm=nothing, z_cm=nothing, n_rows=nothing, n_cols=nothing, collimation_mm=nothing)

Create a CTGeometry from a Scanner definition.

This constructor converts the physical Scanner parameters into pre-computed
trajectory positions suitable for simulation.

# Arguments
- `scanner::Scanner`: Scanner definition with physical parameters (in mm)

# Keyword Arguments
- `n_angles::Int = 360`: Number of projection angles
- `fov_cm::Union{Float64,Nothing} = nothing`: Reconstruction XY FOV in cm. If nothing, uses full detector coverage at isocenter.
- `z_cm::Union{Float64,Nothing} = nothing`: Reconstruction Z extent in cm. If nothing, computes from detector coverage.
- `n_rows::Union{Int,Nothing} = nothing`: Override detector rows. If nothing, uses scanner.detector_rows.
- `n_cols::Union{Int,Nothing} = nothing`: Override detector columns. If nothing, uses scanner.detector_cols.
- `collimation_mm::Union{Float64,Nothing} = nothing`: Detector z-collimation in mm.
  Derives n_rows automatically. Errors if it exceeds scanner physical max or if n_rows is also specified.
- `extended_collimation::Bool = false`: **Simulator-only escape hatch** that bypasses
  the `collimation_mm > scanner_max` check with a loud `@warn`.  Use to approximate
  Siemens NAEOTOM Alpha's high-pitch (3.2) Flash helical mode as a single wider
  axial rotation — the Alpha really only has 57.6 mm of detector but its Flash
  mode covers ~120 mm in ~160 ms via table translation, not via a wider detector.
  This kwarg renders that effective coverage as a single axial scan.

  Physical caveats when extended:
  * The detector array becomes artificially taller (`_n_rows` exceeds
    `scanner.detector_rows`).  Forward sim still works because every downstream
    function treats `n_rows` as the geometry's source of truth.
  * Per-voxel projection density is HIGHER than real Flash (full rotation vs.
    a ~64% sweep) — image quality is *better* than the actual hardware would
    produce, which is fine for training-data generation but wrong for
    hardware-fidelity claims.
  * Dual-source 66 ms temporal resolution is NOT modeled.  Treat the result
    as a 0.25 s (or longer) single-rotation scan.

# Returns
`CTGeometry` with pre-computed source/detector positions.

# Example
```julia
# Create scanner and geometry
scanner = Scanner(
    source_to_isocenter = 541.0,
    source_to_detector = 949.0,
    detector_rows = 64,
    detector_cols = 900
)
geom = CTGeometry(scanner; n_angles=360, fov_cm=35.0)

# Or with reduced detector for fast testing
geom_fast = CTGeometry(scanner; n_angles=90, n_rows=16, n_cols=128, fov_cm=35.0)

# With collimation (derives n_rows automatically)
geom_coll = CTGeometry(scanner; n_angles=360, collimation_mm=80.0)
```
"""
function CTGeometry(
        scanner::Scanner{T};
        n_angles::Int = 360,
        fov_cm::Union{Float64, Nothing} = nothing,
        z_cm::Union{Float64, Nothing} = nothing,
        n_rows::Union{Int, Nothing} = nothing,
        n_cols::Union{Int, Nothing} = nothing,
        collimation_mm::Union{Float64, Nothing} = nothing,
        extended_collimation::Bool = false,
    ) where {T}

    # Determine active detector rows from collimation or explicit override
    if collimation_mm !== nothing
        if n_rows !== nothing
            error("Cannot specify both collimation_mm and n_rows")
        end
        max_collimation = scanner.detector_rows * scanner.detector_row_size
        if collimation_mm > max_collimation
            if extended_collimation
                _virt_rows = round(Int, collimation_mm / scanner.detector_row_size)
                @warn """
                ╔══════════════════════════════════════════════════════════════════════╗
                ║       ⚠  EXTENDED-COLLIMATION MODE — SIMULATOR-ONLY APPROX  ⚠       ║
                ╠══════════════════════════════════════════════════════════════════════╣
                ║ Requested collimation $(round(collimation_mm; digits=2)) mm exceeds  ║
                ║ scanner physical max $(round(max_collimation; digits=2)) mm          ║
                ║ ($(scanner.detector_rows) rows × $(scanner.detector_row_size) mm).   ║
                ║                                                                      ║
                ║ Treating this as a single axial rotation through an artificially     ║
                ║ widened detector ($(_virt_rows) virtual rows).  This approximates    ║
                ║ Siemens NAEOTOM Alpha's high-pitch (3.2) Flash helical mode, which   ║
                ║ scans ~120 mm of Z in ~160 ms via table translation — NOT via a      ║
                ║ wider detector.                                                      ║
                ║                                                                      ║
                ║ • Per-voxel projection density is higher than real Flash             ║
                ║   (full rotation vs ~64% sweep) → IQ better than real hardware.      ║
                ║ • Dual-source 66 ms temporal resolution is NOT modeled.              ║
                ║ • Do NOT use this output for NAEOTOM-Alpha-specific image-quality    ║
                ║   or temporal-behavior claims.                                       ║
                ╚══════════════════════════════════════════════════════════════════════╝
                """ collimation_mm max_collimation virtual_rows=_virt_rows
            else
                error("""
                collimation_mm ($collimation_mm mm) exceeds scanner physical maximum
                ($(scanner.detector_rows) × $(scanner.detector_row_size) = $max_collimation mm).

                Set `extended_collimation = true` on this CTGeometry call to bypass
                the check with a simulator-only approximation (see docstring).
                """)
            end
        end
        _n_rows = round(Int, collimation_mm / scanner.detector_row_size)
    else
        _n_rows = n_rows !== nothing ? n_rows : scanner.detector_rows
    end
    _n_cols = n_cols !== nothing ? n_cols : scanner.detector_cols

    # Convert mm to cm (BasisSimulator internal unit)
    SAD = scanner.source_to_isocenter / 10.0
    SDD = scanner.source_to_detector / 10.0

    # Pixel size at isocenter (detector sizes are already at isocenter, just convert mm→cm)
    pixel_size = scanner.detector_col_size / 10.0
    pixel_row_size = scanner.detector_row_size / 10.0

    # FOV is independent: it controls the reconstruction grid, not the detector geometry
    if fov_cm !== nothing
        fov_xy = fov_cm
    else
        # Default FOV = full detector coverage at isocenter
        fov_xy = _n_cols * pixel_size
    end

    # Z FOV — uses active rows (from collimation or override), not full scanner
    if z_cm !== nothing
        fov_z = z_cm
    else
        z_coverage_mm = _n_rows * scanner.detector_row_size
        fov_z = z_coverage_mm / 10.0  # mm → cm
    end

    # Generate angles (full 360° rotation)
    angles = collect(range(0.0, 2π - 2π / n_angles, length = n_angles))

    # Pre-compute all positions
    source_positions = Matrix{Float64}(undef, 3, n_angles)
    detector_centers = Matrix{Float64}(undef, 3, n_angles)
    detector_u = Matrix{Float64}(undef, 3, n_angles)
    detector_v = Matrix{Float64}(undef, 3, n_angles)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Source position: starts at (0, -SAD, 0), rotates around Z
        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = 0.0

        # Detector center: opposite side of source
        det_dist = SDD - SAD
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = 0.0

        # Detector u-axis (column direction)
        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        # Detector v-axis (row direction): always +Z
        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    fov = (fov_xy, fov_xy, fov_z)

    return CTGeometry(
        SAD, SDD, n_angles, _n_rows, _n_cols, pixel_size, pixel_row_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov
    )
end

# =============================================================================
# PCCT Scanner Helpers (PCCT-SCANNER-BRIDGE)
# =============================================================================

"""
    _build_pcct_detector(scanner::Scanner) -> PhotonCountingDetector

Internal: construct a PhotonCountingDetector from Scanner's flat PCCT kwargs.

This bridges the user-facing flat kwargs API to the internal physics struct.
Used by the simulation driver, nb04, and the PCCT calibration/basis helpers
in `src/reconstruction/vmi/`.
"""
function _build_pcct_detector(scanner::Scanner{T}) where {T}
    @assert scanner.detector_type == :photon_counting "_build_pcct_detector called on non-PCCT scanner"

    # Map detector_material Symbol to DetectorMaterialPCCT enum
    material = _infer_pcct_material(scanner.detector_material)

    # Use native dexel size directly (at detector face) for PCCT physics
    return PhotonCountingDetector(
        material = material,
        thickness_mm = scanner.detector_depth,
        pixel_size_mm = (scanner.native_dexel_row_mm, scanner.native_dexel_col_mm),
        energy_thresholds_keV = Float64.(scanner.energy_thresholds),
        energy_resolution_keV = scanner.energy_resolution,
        charge_sharing_fwhm_mm = scanner.charge_sharing_fwhm,
        enable_charge_sharing = scanner.charge_sharing_fwhm > 0.0,
        dead_time_ns = scanner.dead_time_ns,
        enable_pile_up = scanner.dead_time_ns > 0.0,
        enable_anti_coincidence = scanner.charge_sharing_fwhm > 0.0,
        coincidence_window_ns = scanner.dead_time_ns,
        electronic_noise_keV = 0.0,  # PCCT eliminates electronic noise via thresholding
        binning_factor = scanner.binning_factor
    )
end

"""
    _infer_pcct_material(material_symbol::Symbol) -> DetectorMaterialPCCT

Map a Symbol to DetectorMaterialPCCT enum for internal physics dispatch.
"""
function _infer_pcct_material(material_symbol::Symbol)
    if material_symbol in (:cdte, :CdTe, :CDTE)
        return CDTE_MATERIAL
    elseif material_symbol in (:czt, :CZT, :CdZnTe)
        return CZT_MATERIAL
    elseif material_symbol in (:si, :Si, :silicon, :Silicon)
        return SI_MATERIAL
    else
        @warn "Unknown PCCT detector material :$material_symbol, defaulting to CdTe"
        return CDTE_MATERIAL
    end
end

# =============================================================================
# Exports
# =============================================================================

# Scanner definition
export Scanner, DetectorShape, CURVED_DETECTOR, FLAT_DETECTOR

# CTGeometry (computed positions for simulation)
export CTGeometry
