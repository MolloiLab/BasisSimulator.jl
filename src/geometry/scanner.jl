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
- `detector_row_offset::T`: Row offset from centered position (rows)

The detector is an equiangular ARC (cylindrical, centred on the focal spot —
clinical third-generation MDCT geometry) by DEFAULT; `detector_shape = :flat`
selects a planar panel instead.  Both shapes are supported end-to-end by all
projectors (Siddon / DD / dd_fast), the FDK weighting (Kak-Slaney equiangular
cosγ pre-weight + (γ/sinγ)² kernel for :arc), the helical WFBP rebinning, and
the bowtie model (whose thickness tables are natively fan-angle-parameterised).
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
abstract type Scanner{T <: AbstractFloat} end

"""
    ScannerGeometry{T}

The detector-agnostic core shared by [`EICTScanner`](@ref) and
[`PCCTScanner`](@ref): source/detector distances, detector array, focal spot,
gantry, filtration and detector shape.  Built by the scanner constructors; its
fields are reachable directly on the scanner (`scanner.source_to_isocenter`).
"""
struct ScannerGeometry{T <: AbstractFloat}
    source_to_isocenter::T      # mm (SID/SOD)
    source_to_detector::T       # mm (SDD)
    detector_rows::Int
    detector_cols::Int
    detector_row_size::T        # mm at isocenter
    detector_col_size::T        # mm at isocenter
    detector_row_offset::T      # rows
    detector_col_offset::T      # columns (quarter-detector offset)
    focal_spot_width::T         # mm
    focal_spot_length::T        # mm
    target_angle::T             # degrees
    gantry_rotation_time::T     # seconds
    scan_diameter::T            # mm
    gantry_aperture::T          # mm
    flat_filter_material::Symbol
    flat_filter_thickness::T    # mm
    bowtie_filter::Symbol
    detector_shape::Symbol      # :arc or :flat
end

"""
    EICTScanner{T} <: Scanner{T}

An energy-integrating (scintillator) CT scanner: the shared [`ScannerGeometry`](@ref)
plus the scintillator detection model (material, depth, fill factors, detection
gain, electronic noise).  Construct with [`EICTScanner(; kwargs...)`](@ref).
"""
struct EICTScanner{T <: AbstractFloat} <: Scanner{T}
    geometry::ScannerGeometry{T}
    detector_material::Symbol   # scintillator (:lumex, :gos, …)
    detector_depth::T           # mm
    fill_factor_row::T          # 0-1
    fill_factor_col::T          # 0-1
    detection_gain::T           # electrons/keV
    electronic_noise::T         # electrons
end

"""
    PCCTScanner{T} <: Scanner{T}

A photon-counting CT scanner: the shared [`ScannerGeometry`](@ref) plus the
direct-conversion detector model (sensor material and depth, fill factors,
energy bins and thresholds, energy resolution, charge sharing, dead time, pixel
mode, native dexel size and binning).  Construct with
[`PCCTScanner(; kwargs...)`](@ref).
"""
struct PCCTScanner{T <: AbstractFloat} <: Scanner{T}
    geometry::ScannerGeometry{T}
    detector_material::Symbol   # sensor (:CdTe, :CZT, :Si)
    detector_depth::T           # mm
    fill_factor_row::T          # 0-1
    fill_factor_col::T          # 0-1
    n_energy_bins::Int
    energy_thresholds::Vector{T}  # keV, ascending
    energy_resolution::T        # detector FWHM keV
    charge_sharing_fwhm::T      # charge cloud FWHM mm
    dead_time_ns::T             # pulse dead time ns
    pixel_mode::Symbol          # :standard, :uhr, :macro
    native_dexel_col_mm::T      # native dexel col size at the detector face (mm)
    native_dexel_row_mm::T      # native dexel row size at the detector face (mm)
    binning_factor::Int         # spatial binning (1 = unbinned)
    # detector model toggles (simulation-side physics of THIS detector)
    pileup::Bool                # MC pulse pile-up degradation (needs dead_time_ns > 0)
    pileup_correction::Bool     # model-based un-pile-up of the recorded bins
    scatter_correction::Bool    # model-based scatter re-estimate-and-subtract on the bins
    noise_reduction::T          # count-noise blend, 0 = exact Poisson counts … 1 = expected counts
end

# Geometry fields are reachable directly on the scanner (`scanner.detector_rows`):
# the geometry core is a composition detail, not part of the vocabulary.
const _GEOMETRY_FIELDS = fieldnames(ScannerGeometry)
function Base.getproperty(s::Scanner, name::Symbol)
    name in _GEOMETRY_FIELDS && return getfield(getfield(s, :geometry), name)
    return getfield(s, name)
end
Base.propertynames(s::Scanner) = (_GEOMETRY_FIELDS..., fieldnames(typeof(s))[2:end]...)

function _scanner_geometry(T;
        source_to_isocenter, source_to_detector, detector_rows, detector_cols, detector_row_size, detector_col_size,
        detector_row_offset, detector_col_offset, focal_spot_width, focal_spot_length, target_angle,
        gantry_rotation_time, scan_diameter, gantry_aperture, flat_filter_material, flat_filter_thickness,
        bowtie_filter, detector_shape)
    detector_shape in (:flat, :arc) || error("detector_shape must be :flat or :arc (got :$detector_shape)")
    return ScannerGeometry{T}(T(source_to_isocenter), T(source_to_detector), detector_rows, detector_cols,
        T(detector_row_size), T(detector_col_size), T(detector_row_offset), T(detector_col_offset),
        T(focal_spot_width), T(focal_spot_length), T(target_angle), T(gantry_rotation_time), T(scan_diameter),
        T(gantry_aperture), flat_filter_material, T(flat_filter_thickness), bowtie_filter, detector_shape)
end

# Geometry keyword defaults shared by every constructor (CatSim-like research scanner).
const _GEOMETRY_DEFAULTS = (
    source_to_isocenter = 540.0, source_to_detector = 950.0, detector_rows = 64, detector_cols = 900,
    detector_row_size = 1.0, detector_col_size = 1.0, detector_row_offset = 0.0, detector_col_offset = 0.25,
    focal_spot_width = 1.0, focal_spot_length = 1.0, target_angle = 7.0, gantry_rotation_time = 0.5,
    scan_diameter = 500.0, gantry_aperture = 700.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.0,
    bowtie_filter = :large_body, detector_shape = :arc,
)
_split_geometry_kwargs(kw) = (; (k => v for (k, v) in pairs(merge(_GEOMETRY_DEFAULTS, (; (k => v for (k, v) in pairs(kw) if k in keys(_GEOMETRY_DEFAULTS))...))))...)
_other_kwargs(kw) = (; (k => v for (k, v) in pairs(kw) if !(k in keys(_GEOMETRY_DEFAULTS)))...)

"""
    EICTScanner(; kwargs...)

Energy-integrating scanner.  Geometry keywords as [`Scanner`](@ref) (distances in
mm, detector array, focal spot, gantry, filtration, `detector_shape`) plus the
scintillator model: `detector_material = :lumex`, `detector_depth = 3.0`,
`fill_factor_row = 0.9`, `fill_factor_col = 0.9`, `detection_gain = 15.0`
(electrons/keV), `electronic_noise = 5000.0` (electrons).
"""
function EICTScanner(; detector_material::Symbol = :lumex, detector_depth::Real = 3.0,
        fill_factor_row::Real = 0.9, fill_factor_col::Real = 0.9, detection_gain::Real = 15.0,
        electronic_noise::Real = 5000.0, kwargs...)
    T = Float64
    extra = _other_kwargs(kwargs)
    isempty(extra) || throw(ArgumentError("EICTScanner: unknown keyword(s) $(collect(keys(extra))) — photon-counting parameters belong to PCCTScanner"))
    geometry = _scanner_geometry(T; _split_geometry_kwargs(kwargs)...)
    return EICTScanner{T}(geometry, detector_material, T(detector_depth), T(fill_factor_row), T(fill_factor_col),
        T(detection_gain), T(electronic_noise))
end

"""
    PCCTScanner(; energy_thresholds, kwargs...)

Photon-counting scanner.  Geometry keywords as [`Scanner`](@ref) plus the
direct-conversion detector model: `detector_material = :CdTe`, `detector_depth = 1.6`,
`fill_factor_row = 0.9`, `fill_factor_col = 0.9`, `energy_thresholds` (keV,
ascending; required), `n_energy_bins = length(energy_thresholds)`,
`energy_resolution = 0.0`, `charge_sharing_fwhm = 0.0`, `dead_time_ns = 0.0`,
`pixel_mode = :standard`, `native_dexel_col_mm = 0` / `native_dexel_row_mm = 0`
(0 = infer from the binned pixel and magnification), `binning_factor = 1`; and the
detector-model toggles `pileup = true`, `pileup_correction = false`,
`scatter_correction = false`, `noise_reduction = 0.0`.
"""
function PCCTScanner(; detector_material::Symbol = :CdTe, detector_depth::Real = 1.6,
        fill_factor_row::Real = 0.9, fill_factor_col::Real = 0.9,
        energy_thresholds::AbstractVector{<:Real} = Float64[], n_energy_bins::Int = length(energy_thresholds),
        energy_resolution::Real = 0.0, charge_sharing_fwhm::Real = 0.0, dead_time_ns::Real = 0.0,
        pixel_mode::Symbol = :standard, native_dexel_col_mm::Real = 0.0, native_dexel_row_mm::Real = 0.0,
        binning_factor::Int = 1, pileup::Bool = true, pileup_correction::Bool = false,
        scatter_correction::Bool = false, noise_reduction::Real = 0.0, kwargs...)
    T = Float64
    0 <= noise_reduction <= 1 || error("noise_reduction must lie in [0, 1] (got $noise_reduction)")
    extra = _other_kwargs(kwargs)
    isempty(extra) || throw(ArgumentError("PCCTScanner: unknown keyword(s) $(collect(keys(extra))) — scintillator parameters belong to EICTScanner"))
    isempty(energy_thresholds) && error("PCCT scanner requires energy_thresholds (got empty vector)")
    n_energy_bins == length(energy_thresholds) ||
        error("n_energy_bins ($n_energy_bins) must equal length(energy_thresholds) ($(length(energy_thresholds)))")
    issorted(energy_thresholds) || error("energy_thresholds must be sorted ascending (got $energy_thresholds)")
    pixel_mode in (:standard, :uhr, :macro) || error("pixel_mode must be :standard, :uhr, or :macro (got :$pixel_mode)")
    binning_factor >= 1 || error("binning_factor must be >= 1 (got $binning_factor)")
    geometry = _scanner_geometry(T; _split_geometry_kwargs(kwargs)...)
    magnification_val = Float64(geometry.source_to_detector) / Float64(geometry.source_to_isocenter)
    _native_col = native_dexel_col_mm > 0.0 ? native_dexel_col_mm : geometry.detector_col_size * magnification_val / binning_factor
    _native_row = native_dexel_row_mm > 0.0 ? native_dexel_row_mm : geometry.detector_row_size * magnification_val / binning_factor
    return PCCTScanner{T}(geometry, detector_material, T(detector_depth), T(fill_factor_row), T(fill_factor_col),
        n_energy_bins, T.(collect(energy_thresholds)), T(energy_resolution), T(charge_sharing_fwhm), T(dead_time_ns),
        pixel_mode, T(_native_col), T(_native_row), binning_factor,
        pileup, pileup_correction, scatter_correction, T(noise_reduction))
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

# Flat-panel-style scanner (the detector is always modeled as planar)
scanner = Scanner(
    detector_rows = 512,
    detector_cols = 512,
    detector_row_size = 0.15,
    detector_col_size = 0.15
)
```
"""

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
    # ── Helical trajectory metadata (0.0/0.0 = axial circular orbit) ─────────
    pitch::Float64           # IEC pitch: table feed per rotation ÷ collimation
    table_feed::Float64      # table feed per rotation (cm); 0.0 = axial
    # ── Detector shape: :arc (equiangular cylindrical) or :flat (planar) ─────
    detector_shape::Symbol
end

# Backward-compatible positional constructor: the pre-helical 14-field form
# (subset extraction, FOV overrides, native-PCCT geometry) defaults to axial.
function CTGeometry(
        SAD::Float64, SDD::Float64, n_angles::Int, n_rows::Int, n_cols::Int,
        pixel_size::Float64, pixel_row_size::Float64,
        angles::Vector{Float64},
        source_positions::Matrix{Float64}, detector_centers::Matrix{Float64},
        detector_u::Matrix{Float64}, detector_v::Matrix{Float64},
        fov::NTuple{3, Float64},
    )
    return CTGeometry(
        SAD, SDD, n_angles, n_rows, n_cols, pixel_size, pixel_row_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov, 0.0, 0.0, :flat)
end

# 16-field compat (helical metadata, pre-arc): defaults to a flat panel.
function CTGeometry(
        SAD::Float64, SDD::Float64, n_angles::Int, n_rows::Int, n_cols::Int,
        pixel_size::Float64, pixel_row_size::Float64,
        angles::Vector{Float64},
        source_positions::Matrix{Float64}, detector_centers::Matrix{Float64},
        detector_u::Matrix{Float64}, detector_v::Matrix{Float64},
        fov::NTuple{3, Float64}, pitch::Float64, table_feed::Float64,
    )
    return CTGeometry(
        SAD, SDD, n_angles, n_rows, n_cols, pixel_size, pixel_row_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov, pitch, table_feed, :flat)
end

"""
    is_helical(geom::CTGeometry) -> Bool

`true` when the geometry carries a helical trajectory (non-zero table feed).
"""
is_helical(geom::CTGeometry) = geom.table_feed != 0.0

export is_helical

"""
    is_arc(geom::CTGeometry) -> Bool

`true` for an equiangular (cylindrical, source-centred) detector.  The
column angular pitch is `Δγ = geom.pixel_size / geom.SAD` (the column pitch
at isocentre interpreted as arc length at the isocentre radius).
"""
is_arc(geom::CTGeometry) = geom.detector_shape === :arc

export is_arc

"""
    required_axial_detector_rows(scanner; fov_cm, z_cm) -> Int

Minimum symmetric detector-row count needed to support the complete axial
reconstruction cylinder. Cone divergence magnifies the terminal z coordinate
at the near edge of the reconstruction FOV by `SAD / (SAD - radius)`.
"""
function required_axial_detector_rows(
        scanner::Scanner; fov_cm::Real, z_cm::Real,
    )
    radius_mm = Float64(fov_cm) * 5.0
    sad_mm = Float64(scanner.source_to_isocenter)
    radius_mm < sad_mm || throw(ArgumentError(
        "reconstruction radius $radius_mm mm must be smaller than SAD $sad_mm mm"))
    support_mm = Float64(z_cm) * 10.0 * sad_mm / (sad_mm - radius_mm)
    return ceil(Int, support_mm / Float64(scanner.detector_row_size))
end

export required_axial_detector_rows

"""
    CTGeometry(scanner::Scanner; n_angles=360, fov_cm=nothing, z_cm=nothing, n_rows=nothing, n_cols=nothing, collimation_mm=nothing)

Create a CTGeometry from a Scanner definition.

This constructor converts the physical Scanner parameters into pre-computed
trajectory positions suitable for simulation.

# Arguments
- `scanner::Scanner`: Scanner definition with physical parameters (in mm)

# Keyword Arguments
- `n_angles::Int = 360`: Number of projection angles PER ROTATION
- `fov_cm::Union{Float64,Nothing} = nothing`: Reconstruction XY FOV in cm. If nothing, uses full detector coverage at isocenter.
- `z_cm::Union{Float64,Nothing} = nothing`: Reconstruction Z extent in cm. If nothing, computes
  from detector coverage (axial) or `table_travel − collimation` (helical).
- `pitch::Union{Float64,Nothing} = nothing`: Helical pitch (IEC: table feed per rotation ÷
  active collimation).  `nothing` = axial circular orbit.  When set, source AND detector
  translate along z over `n_rotations` turns, helix centred on isocentre; total views
  = `n_angles × n_rotations`.
- `n_rotations::Real = 1.0`: Number of gantry rotations (helical only; ignored for axial).
- `n_rows::Union{Int,Nothing} = nothing`: Override detector rows. If nothing, uses scanner.detector_rows.
- `n_cols::Union{Int,Nothing} = nothing`: Override detector columns. If nothing, uses scanner.detector_cols.
- `collimation_mm::Union{Float64,Nothing} = nothing`: Detector z-collimation in mm.
  Derives the nominal row count automatically. For axial scans with explicit
  `fov_cm` and `z_cm`, symmetric cone-guard rows are added as needed to support
  the complete reconstruction cylinder. Errors if the guarded count exceeds
  the physical detector or if `n_rows` is also specified.
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
        pitch::Union{Float64, Nothing} = nothing,
        n_rotations::Real = 1.0,
    ) where {T}

    n_angles >= 2 || throw(ArgumentError("n_angles must be at least 2, got $n_angles"))
    n_rotations > 0 || throw(ArgumentError("n_rotations must be positive, got $n_rotations"))
    if pitch !== nothing
        pitch > 0 || throw(ArgumentError("pitch must be positive, got $pitch"))
        n_rotations >= 1 || throw(ArgumentError(
            "helical scans need n_rotations ≥ 1, got $n_rotations"))
    end
    collimation_mm === nothing || collimation_mm > 0 ||
        throw(ArgumentError("collimation_mm must be positive, got $collimation_mm"))
    n_rows === nothing || n_rows > 0 ||
        throw(ArgumentError("n_rows must be positive, got $n_rows"))
    n_cols === nothing || n_cols > 0 ||
        throw(ArgumentError("n_cols must be positive, got $n_cols"))

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
        nominal_rows = round(Int, collimation_mm / scanner.detector_row_size)
        _n_rows = nominal_rows
        if pitch === nothing && fov_cm !== nothing && z_cm !== nothing
            required_rows = required_axial_detector_rows(scanner; fov_cm, z_cm)
            # Preserve detector centering parity so guards are symmetric.
            if isodd(required_rows - nominal_rows)
                required_rows += 1
            end
            _n_rows = max(nominal_rows, required_rows)
            if _n_rows > scanner.detector_rows && !extended_collimation
                error("""
                Full-FOV axial support requires $(_n_rows) detector rows, but the
                scanner has only $(scanner.detector_rows). Reduce fov_cm/z_cm or
                use a scanner with sufficient physical row coverage.
                """)
            end
            if _n_rows > nominal_rows
                @info "axial cone guards added" nominal_rows guarded_rows=_n_rows
            end
        end
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

    # ── Helical trajectory parameters ────────────────────────────────────────
    # IEC pitch = table feed per rotation ÷ total active collimation.  The
    # collimation used here is the ACTIVE detector z-coverage at isocentre
    # (_n_rows × pixel_row_size), which equals `collimation_mm` when given.
    # The helix runs `n_rotations` turns, centred on isocentre (z = 0 at the
    # scan midpoint), translating BOTH source and detector (gantry ↔ table).
    helical = pitch !== nothing
    collim_iso_cm = _n_rows * pixel_row_size
    table_feed = helical ? pitch * collim_iso_cm : 0.0
    table_travel = table_feed * (helical ? Float64(n_rotations) : 0.0)

    # Total view count and angular span
    n_views_total = helical ? round(Int, n_angles * n_rotations) : n_angles

    # Z FOV — uses active rows (from collimation or override), not full scanner
    if z_cm !== nothing
        fov_z = z_cm
    elseif helical
        # Default recon z-extent = planned range: table travel minus one
        # collimation width (the first/last half-collimation is over-ranging —
        # rays exist but z-sampling is incomplete there).  Clamped so very
        # short helices still get one collimation width.
        fov_z = max(table_travel - collim_iso_cm, collim_iso_cm)
    else
        z_coverage_mm = _n_rows * scanner.detector_row_size
        fov_z = z_coverage_mm / 10.0  # mm → cm
    end

    # Generate angles: axial = one full 360° rotation; helical = n_rotations
    # turns at the same per-rotation angular sampling (Δθ = 2π/n_angles).
    angles = if helical
        collect(range(0.0, step = 2π / n_angles, length = n_views_total))
    else
        collect(range(0.0, 2π - 2π / n_angles, length = n_angles))
    end

    # Pre-compute all positions
    source_positions = Matrix{Float64}(undef, 3, n_views_total)
    detector_centers = Matrix{Float64}(undef, 3, n_views_total)
    detector_u = Matrix{Float64}(undef, 3, n_views_total)
    detector_v = Matrix{Float64}(undef, 3, n_views_total)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Gantry z: helix centred on isocentre (z=0 at the scan midpoint)
        z_i = helical ? (-table_travel / 2 + table_feed * θ / (2π)) : 0.0

        # Source position: starts at (0, -SAD, z_start), rotates around Z
        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = z_i

        # Detector center: opposite side of source, same gantry z
        det_dist = SDD - SAD
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = z_i

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
        SAD, SDD, n_views_total, _n_rows, _n_cols, pixel_size, pixel_row_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov, helical ? Float64(pitch) : 0.0, table_feed,
        scanner.detector_shape,
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
function _build_pcct_detector(scanner::PCCTScanner{T}) where {T}

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
export Scanner, EICTScanner, PCCTScanner, ScannerGeometry

# CTGeometry (computed positions for simulation)
export CTGeometry
