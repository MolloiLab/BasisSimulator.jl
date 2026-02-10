"""
    Scanners/Scanners.jl

Clinical CT scanner configurations with publication-ready documentation.

This module provides detailed scanner specifications based on FDA 510(k) filings,
manufacturer technical data, and peer-reviewed publications. Each parameter
includes source citations for reproducibility.

# Supported Scanners
- GE Revolution Apex Elite (K213715)

# Usage
```julia
# Get scanner specification
spec = GERevolutionApexElite()

# Create geometry for simulation
geom = create_geometry(spec; n_angles=984, n_rows=64)

# Use protocol presets
geom_chest = create_geometry(spec, ChestProtocol())
```

# Adding New Scanners
See CLAUDE.md "Clinical Scanner Configurations" section for the design pattern.
"""

# =============================================================================
# Base Types for Scanner Specifications
# =============================================================================

"""
    ScannerManufacturer

Enumeration of supported CT scanner manufacturers.
"""
@enum ScannerManufacturer begin
    GE_HEALTHCARE
    SIEMENS_HEALTHINEERS
    CANON_MEDICAL
    PHILIPS_HEALTHCARE
end

"""
    DetectorType

Enumeration of CT detector technologies.
"""
@enum DetectorType begin
    ENERGY_INTEGRATING    # Traditional scintillator + photodiode
    PHOTON_COUNTING       # Direct conversion (CdTe, CZT)
end

"""
    DetectorMaterial

Enumeration of scintillator/detector materials.
"""
@enum DetectorMaterial begin
    GOS         # Gadolinium Oxysulfide (Gd2O2S:Pr)
    LUMEX       # GE proprietary garnet scintillator
    CSI         # Cesium Iodide
    CDTE        # Cadmium Telluride (photon counting)
    CZT         # Cadmium Zinc Telluride (photon counting)
end

"""
    SourceCitation

Documentation of parameter source for publication reproducibility.

# Fields
- `value`: The parameter value
- `source`: Source type (:fda_510k, :manufacturer, :publication, :derived)
- `url`: URL to source document (if available)
- `note`: Additional notes on derivation or interpretation
"""
struct SourceCitation{T}
    value::T
    source::Symbol
    url::String
    note::String
end

# Convenience constructors
SourceCitation(value; source=:unknown, url="", note="") =
    SourceCitation(value, source, url, note)

# Extract just the value
Base.getindex(sc::SourceCitation) = sc.value

"""
    DetectorSpecification

Detailed detector array specification.

# Fields
All fields are `SourceCitation` to track provenance.

- `material`: Scintillator/sensor material
- `n_rows`: Number of detector rows (z-direction)
- `n_cols`: Number of detector columns (fan direction)
- `row_size_mm`: Detector element size in z (mm)
- `col_size_mm`: Detector element size in fan direction (mm)
- `depth_mm`: Sensor depth/thickness (mm)
- `detector_type`: Energy-integrating or photon-counting
- `z_coverage_mm`: Total z-axis coverage (mm)
- `fill_factor_row`: Active area fraction (row direction)
- `fill_factor_col`: Active area fraction (column direction)
"""
struct DetectorSpecification
    material::SourceCitation{DetectorMaterial}
    n_rows::SourceCitation{Int}
    n_cols::SourceCitation{Int}
    row_size_mm::SourceCitation{Float64}
    col_size_mm::SourceCitation{Float64}
    depth_mm::SourceCitation{Float64}
    detector_type::SourceCitation{DetectorType}
    z_coverage_mm::SourceCitation{Float64}
    fill_factor_row::SourceCitation{Float64}
    fill_factor_col::SourceCitation{Float64}
end

"""
    TubeSpecification

X-ray tube specification.

# Fields
- `model_name`: Tube model identifier
- `max_power_kw`: Maximum generator power (kW)
- `target_angle_deg`: Anode target angle (degrees)
- `focal_spot_small_mm`: Small focal spot size (width × length) in mm
- `focal_spot_large_mm`: Large focal spot size (width × length) in mm
- `kvp_options`: Available kVp settings
- `max_ma`: Maximum tube current (mA)
- `has_flying_focal_spot`: Whether flying focal spot is supported
"""
struct TubeSpecification
    model_name::SourceCitation{String}
    max_power_kw::SourceCitation{Float64}
    target_angle_deg::SourceCitation{Float64}
    focal_spot_small_mm::SourceCitation{Tuple{Float64,Float64}}
    focal_spot_large_mm::SourceCitation{Tuple{Float64,Float64}}
    kvp_options::SourceCitation{Vector{Int}}
    max_ma::SourceCitation{Int}
    has_flying_focal_spot::SourceCitation{Bool}
end

"""
    GeometrySpecification

Scanner geometry specification.

# Fields
- `sid_mm`: Source-to-isocenter distance (mm)
- `sdd_mm`: Source-to-detector distance (mm)
- `gantry_aperture_mm`: Gantry bore diameter (mm)
- `max_sfov_mm`: Maximum scan field of view (mm)
- `detector_curve_radius_mm`: Detector array curve radius (mm), 0 for flat
"""
struct GeometrySpecification
    sid_mm::SourceCitation{Float64}
    sdd_mm::SourceCitation{Float64}
    gantry_aperture_mm::SourceCitation{Float64}
    max_sfov_mm::SourceCitation{Float64}
    detector_curve_radius_mm::SourceCitation{Float64}
end

"""
    AcquisitionSpecification

Acquisition capability specification.

# Fields
- `min_rotation_time_s`: Fastest rotation time (seconds)
- `max_rotation_time_s`: Slowest rotation time (seconds)
- `rotation_time_options_s`: Available rotation times (seconds)
- `max_views_per_rotation`: Maximum views per 360° rotation
"""
struct AcquisitionSpecification
    min_rotation_time_s::SourceCitation{Float64}
    max_rotation_time_s::SourceCitation{Float64}
    rotation_time_options_s::SourceCitation{Vector{Float64}}
    max_views_per_rotation::SourceCitation{Int}
end

"""
    AbstractScannerSpec

Abstract base type for scanner specifications.

All concrete scanner specs should subtype this and implement:
- `manufacturer(spec)` - Return ScannerManufacturer
- `model_name(spec)` - Return model name string
- `fda_510k(spec)` - Return FDA 510(k) number
- `detector(spec)` - Return DetectorSpecification
- `tube(spec)` - Return TubeSpecification
- `geometry(spec)` - Return GeometrySpecification
- `acquisition(spec)` - Return AcquisitionSpecification
"""
abstract type AbstractScannerSpec end

# Required interface methods (to be implemented by concrete types)
manufacturer(::AbstractScannerSpec) = error("Not implemented")
model_name(::AbstractScannerSpec) = error("Not implemented")
fda_510k(::AbstractScannerSpec) = error("Not implemented")
detector(::AbstractScannerSpec) = error("Not implemented")
tube(::AbstractScannerSpec) = error("Not implemented")
geometry(::AbstractScannerSpec) = error("Not implemented")
acquisition(::AbstractScannerSpec) = error("Not implemented")

# =============================================================================
# Protocol Types
# =============================================================================

"""
    AbstractProtocol

Abstract base type for scan protocols.
"""
abstract type AbstractProtocol end

"""
    AxialProtocol

Axial (step-and-shoot) scan protocol.

# Fields
- `kvp`: Tube voltage (kV)
- `ma`: Tube current (mA)
- `rotation_time_s`: Rotation time (seconds)
- `n_angles`: Number of projection angles per rotation
- `slice_thickness_mm`: Reconstructed slice thickness (mm)
"""
struct AxialProtocol <: AbstractProtocol
    kvp::Int
    ma::Int
    rotation_time_s::Float64
    n_angles::Int
    slice_thickness_mm::Float64
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
    create_geometry(spec::AbstractScannerSpec; kwargs...)

Create a CTGeometry from a scanner specification.

# Arguments
- `spec`: Scanner specification

# Keyword Arguments
- `n_angles::Int`: Number of projection angles (default: 984, typical clinical)
- `n_rows::Int`: Number of detector rows to use (default: 64 for faster simulation)
- `n_cols::Int`: Number of detector columns to use (default: from spec)
- `fov_cm::Float64`: Field of view in cm (default: from spec max SFOV)

# Returns
`CTGeometry` configured for the specified scanner.
"""
function create_geometry(spec::AbstractScannerSpec;
                        n_angles::Int=984,
                        n_rows::Int=64,
                        n_cols::Int=0,
                        fov_cm::Float64=0.0)

    det = detector(spec)
    geom_spec = geometry(spec)

    # Use defaults from spec if not provided
    _n_cols = n_cols > 0 ? n_cols : det.n_cols[]
    _fov_cm = fov_cm > 0.0 ? fov_cm : geom_spec.max_sfov_mm[] / 10.0

    # Convert mm to cm for BasisSimulator
    sid_cm = geom_spec.sid_mm[] / 10.0
    sdd_cm = geom_spec.sdd_mm[] / 10.0

    # Compute pixel size to cover FOV
    # Add margin to ensure full coverage
    pixel_size_cm = (_fov_cm * 1.1) / _n_cols

    # Generate angles (full 360° rotation)
    angles = collect(range(0.0, 2π - 2π/n_angles, length=n_angles))

    # Pre-compute all positions
    source_positions = Matrix{Float64}(undef, 3, n_angles)
    detector_centers = Matrix{Float64}(undef, 3, n_angles)
    detector_u = Matrix{Float64}(undef, 3, n_angles)
    detector_v = Matrix{Float64}(undef, 3, n_angles)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Source position: starts at (0, -SAD, 0), rotates around Z
        source_positions[1, i] = -sid_cm * sinθ
        source_positions[2, i] = -sid_cm * cosθ
        source_positions[3, i] = 0.0

        # Detector center: opposite side of source
        det_dist = sdd_cm - sid_cm  # Distance from isocenter to detector
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = 0.0

        # Detector u-axis (column direction): perpendicular to source-detector line
        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        # Detector v-axis (row direction): always points in +Z
        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    # Compute z FOV from detector coverage
    magnification = sdd_cm / sid_cm
    pixel_row_size_cm = (det.row_size_mm[] / 10.0) / magnification
    z_coverage_cm = n_rows * pixel_row_size_cm
    fov = (_fov_cm, _fov_cm, z_coverage_cm)

    return CTGeometry(
        sid_cm, sdd_cm, n_angles, n_rows, _n_cols, pixel_size_cm, pixel_row_size_cm,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov
    )
end

"""
    create_geometry(spec::AbstractScannerSpec, protocol::AxialProtocol; n_rows::Int=64)

Create an axial CTGeometry from scanner spec and protocol.

# Arguments
- `spec`: Scanner specification
- `protocol`: Axial protocol with rotation time, angles, etc.
- `n_rows`: Number of detector rows to simulate (default: 64 for speed)
"""
function create_geometry(spec::AbstractScannerSpec, protocol::AxialProtocol; n_rows::Int=64)
    return create_geometry(spec;
        n_angles=protocol.n_angles,
        n_rows=n_rows
    )
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    print_scanner_info(spec::AbstractScannerSpec)

Print detailed scanner specification with source citations.
"""
function print_scanner_info(spec::AbstractScannerSpec)
    println("=" ^ 80)
    println("SCANNER SPECIFICATION: $(model_name(spec))")
    println("=" ^ 80)
    println("Manufacturer: $(manufacturer(spec))")
    println("FDA 510(k):   $(fda_510k(spec))")
    println()

    # Detector
    det = detector(spec)
    println("DETECTOR ($(det.material[]))")
    println("-" ^ 40)
    println("  Array:        $(det.n_cols[]) × $(det.n_rows[]) elements")
    println("  Element Size: $(det.col_size_mm[]) × $(det.row_size_mm[]) mm")
    println("  Z-Coverage:   $(det.z_coverage_mm[]) mm")
    println("  Depth:        $(det.depth_mm[]) mm")
    println("  Type:         $(det.detector_type[])")
    println()

    # Tube
    tb = tube(spec)
    println("X-RAY TUBE ($(tb.model_name[]))")
    println("-" ^ 40)
    println("  Power:        $(tb.max_power_kw[]) kW")
    println("  Target Angle: $(tb.target_angle_deg[])°")
    println("  Focal Spots:  $(tb.focal_spot_small_mm[]) mm (small)")
    println("                $(tb.focal_spot_large_mm[]) mm (large)")
    println("  kVp Options:  $(tb.kvp_options[])")
    println("  Max mA:       $(tb.max_ma[]) mA")
    println("  Flying FS:    $(tb.has_flying_focal_spot[])")
    println()

    # Geometry
    gm = geometry(spec)
    println("GEOMETRY")
    println("-" ^ 40)
    println("  SID:          $(gm.sid_mm[]) mm")
    println("  SDD:          $(gm.sdd_mm[]) mm")
    println("  Aperture:     $(gm.gantry_aperture_mm[]) mm")
    println("  Max SFOV:     $(gm.max_sfov_mm[]) mm")
    println()

    # Acquisition
    acq = acquisition(spec)
    println("ACQUISITION")
    println("-" ^ 40)
    println("  Rotation:     $(acq.min_rotation_time_s[]) - $(acq.max_rotation_time_s[]) s")
    println("  Max Views:    $(acq.max_views_per_rotation[]) per rotation")
    println("=" ^ 80)
end

"""
    get_source_citations(spec::AbstractScannerSpec)

Get all source citations for a scanner specification.
Returns a Dict mapping parameter names to SourceCitation objects.
"""
function get_source_citations(spec::AbstractScannerSpec)
    citations = Dict{String, SourceCitation}()

    det = detector(spec)
    citations["detector.material"] = det.material
    citations["detector.n_rows"] = det.n_rows
    citations["detector.n_cols"] = det.n_cols
    citations["detector.row_size_mm"] = det.row_size_mm
    citations["detector.col_size_mm"] = det.col_size_mm
    citations["detector.depth_mm"] = det.depth_mm
    citations["detector.z_coverage_mm"] = det.z_coverage_mm

    gm = geometry(spec)
    citations["geometry.sid_mm"] = gm.sid_mm
    citations["geometry.sdd_mm"] = gm.sdd_mm
    citations["geometry.gantry_aperture_mm"] = gm.gantry_aperture_mm
    citations["geometry.max_sfov_mm"] = gm.max_sfov_mm

    tb = tube(spec)
    citations["tube.model_name"] = tb.model_name
    citations["tube.max_power_kw"] = tb.max_power_kw
    citations["tube.target_angle_deg"] = tb.target_angle_deg
    citations["tube.max_ma"] = tb.max_ma

    return citations
end

# =============================================================================
# Include Manufacturer-Specific Files
# =============================================================================

include("general_electric.jl")
include("siemens.jl")

# =============================================================================
# Exports
# =============================================================================

export ScannerManufacturer, GE_HEALTHCARE, SIEMENS_HEALTHINEERS, CANON_MEDICAL, PHILIPS_HEALTHCARE
export DetectorType, ENERGY_INTEGRATING, PHOTON_COUNTING
export DetectorMaterial, GOS, LUMEX, CSI, CDTE, CZT
export SourceCitation
export DetectorSpecification, TubeSpecification, GeometrySpecification, AcquisitionSpecification
export AbstractScannerSpec, AbstractProtocol
export AxialProtocol
export manufacturer, model_name, fda_510k, detector, tube, geometry, acquisition
export create_geometry, print_scanner_info, get_source_citations
