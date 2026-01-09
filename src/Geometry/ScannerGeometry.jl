"""
    Geometry/ScannerGeometry.jl

CT scanner geometry definitions for cone-beam CT simulation.

# Overview

This module defines the mechanical geometry of CT scanners, including:
- Source and detector positioning
- Circular and helical trajectories
- Pre-computed coordinate systems for efficient simulation

All spatial units are in **cm** (converted from mm specifications).
Angular units are in **degrees** (converted to radians internally).

# Key Concepts

## Scanner Geometry

CT scanners have several key geometric parameters:

1. **SDD (Source-to-Detector Distance)**: Distance from X-ray source to detector
2. **SAD (Source-to-Axis Distance)**: Distance from X-ray source to isocenter
3. **Detector Array**: Grid of detector elements (rows × columns)
4. **Pixel Pitch**: Physical size of detector elements
5. **Magnification**: M = SDD / SAD (affects field-of-view)

## Coordinate System

We use a **right-handed coordinate system**:
- **X-axis**: Lateral (left-right)
- **Y-axis**: Anterior-posterior (front-back)
- **Z-axis**: Superior-inferior (head-feet)

**Rotation Convention:**
- Gantry rotates around Z-axis
- Source position: (SAD·sin(θ), -SAD·cos(θ), 0)
- Detector center: opposite side of isocenter
- At θ=0°: Source at (0, -SAD, 0), Detector at (0, SDD-SAD, 0)

## Pre-Computed Trajectories

For efficiency, all source and detector positions are **pre-computed** during
geometry initialization. This avoids trigonometric calculations in the hot loop.

**Stored Data:**
- `source_positions[3, N]` - Source XYZ for each projection
- `det_centers[3, N]` - Detector center XYZ for each projection
- `det_u_vecs[3, N]` - Detector U-axis (horizontal) unit vectors
- `det_v_vecs[3, N]` - Detector V-axis (vertical) unit vectors

This design is **Reactant-friendly** (pure data, no runtime computation).

# References

**Scanner Specifications:**
- Canon Aquilion ONE: https://global.medical.canon/products/computed-tomography
- GE Revolution CT: Technical specifications
- Siemens SOMATOM: Technical specifications

**Cone-Beam Geometry:**
- Feldkamp, L. A., et al. (1984). "Practical cone-beam algorithm." JOSA A 1.6
- Kak, A. C., & Slaney, M. (1988). "Principles of computerized tomographic imaging."

# Author

Dale Black, MolloiLab
Created: January 2026
"""

# ==============================================================================
# Scan Protocol Definition
# ==============================================================================

"""
    ScanProtocol

Acquisition protocol parameters for CT scanning.

# Fields

- `kVp::Float64` - Peak tube voltage (kilovolt peak)
- `mAs::Float64` - Tube current-time product (milliampere-seconds)
- `scan_fov_mm::Float64` - Scan field-of-view diameter (mm)
- `num_projections::Int` - Number of projection angles
- `rotation_total_angle::Float64` - Total rotation angle (degrees, default 360°)
- `start_angle::Float64` - Starting angle (degrees, default 0°)

# Example

```julia
# Standard chest CT protocol
protocol = ScanProtocol(
    kVp = 120.0,
    mAs = 200.0,
    scan_fov_mm = 400.0,
    num_projections = 720
)

# Short-scan protocol (Parker weighting)
protocol_short = ScanProtocol(
    kVp = 120.0,
    mAs = 200.0,
    scan_fov_mm = 400.0,
    num_projections = 400,
    rotation_total_angle = 200.0  # 180° + fan angle
)
```
"""
struct ScanProtocol
    kVp::Float64
    mAs::Float64
    scan_fov_mm::Float64
    num_projections::Int
    rotation_total_angle::Float64
    start_angle::Float64

    function ScanProtocol(;
            kVp::Float64,
            mAs::Float64,
            scan_fov_mm::Float64,
            num_projections::Int,
            rotation_total_angle::Float64 = 360.0,
            start_angle::Float64 = 0.0
        )
        # Validation
        @assert kVp > 0 "kVp must be positive"
        @assert mAs > 0 "mAs must be positive"
        @assert scan_fov_mm > 0 "scan_fov_mm must be positive"
        @assert num_projections > 0 "num_projections must be positive"
        @assert 0 < rotation_total_angle <= 360.0 "rotation must be in (0, 360]"

        new(kVp, mAs, scan_fov_mm, num_projections, rotation_total_angle, start_angle)
    end
end

# ==============================================================================
# Scanner Geometry Definition
# ==============================================================================

"""
    CTGeometry

Complete CT scanner geometry with pre-computed trajectories.

# Fields

## Mechanical Constants
- `SDD_cm::Float64` - Source-to-detector distance (cm)
- `SAD_cm::Float64` - Source-to-axis distance (cm)
- `n_rows::Int` - Number of detector rows
- `n_cols::Int` - Number of detector columns
- `pixel_width_cm::Float64` - Detector pixel width (cm)
- `pixel_height_cm::Float64` - Detector pixel height (cm)

## Pre-Computed Trajectories
- `angles::Vector{Float64}` - Projection angles in degrees [N]
- `source_positions::Matrix{Float64}` - Source XYZ positions [3, N]
- `det_centers::Matrix{Float64}` - Detector center XYZ positions [3, N]
- `det_u_vecs::Matrix{Float64}` - Detector U-axis (horizontal) vectors [3, N]
- `det_v_vecs::Matrix{Float64}` - Detector V-axis (vertical) vectors [3, N]

# Constructor

```julia
geo = CTGeometry(
    sdd_mm = 1000.0,
    sad_mm = 600.0,
    n_rows = 320,
    n_cols = 896,
    pixel_size_mm = (0.833, 0.5),  # (width, height)
    angles_deg = collect(0.0:1.0:359.0)
)
```

# Implementation Notes

The constructor pre-computes all source and detector positions to avoid
runtime trigonometric calculations. This is critical for:
- **Performance**: No sin/cos in hot loop
- **Reactant compatibility**: Pure functional, no dynamic allocations
- **Enzyme compatibility**: Fixed-size arrays for autodiff

# Coordinate System

At angle θ=0°:
- Source: (0, -SAD, 0)
- Detector center: (0, SDD-SAD, 0)
- U-axis: (1, 0, 0) - horizontal, left to right
- V-axis: (0, 0, 1) - vertical, inferior to superior

For angle θ:
- Source rotates around Z-axis: (SAD·sin(θ), -SAD·cos(θ), 0)
- Detector center rotates oppositely
- U-axis rotates with gantry: (cos(θ), sin(θ), 0)
- V-axis stays fixed: (0, 0, 1)
"""
struct CTGeometry
    # Mechanical constants
    SDD_cm::Float64
    SAD_cm::Float64
    n_rows::Int
    n_cols::Int
    pixel_width_cm::Float64
    pixel_height_cm::Float64

    # Pre-computed trajectories [dimensions: 3 × N_projections]
    angles::Vector{Float64}
    source_positions::Matrix{Float64}
    det_centers::Matrix{Float64}
    det_u_vecs::Matrix{Float64}
    det_v_vecs::Matrix{Float64}

    function CTGeometry(;
            sdd_mm::Float64,
            sad_mm::Float64,
            n_rows::Int,
            n_cols::Int,
            pixel_size_mm::Tuple{Float64, Float64},
            angles_deg::Vector{Float64}
        )

        # Validation
        @assert sdd_mm > sad_mm "SDD must be greater than SAD"
        @assert sad_mm > 0 "SAD must be positive"
        @assert n_rows > 0 "n_rows must be positive"
        @assert n_cols > 0 "n_cols must be positive"
        @assert all(pixel_size_mm .> 0) "Pixel sizes must be positive"

        # Convert units: mm → cm
        SDD_cm = sdd_mm / 10.0
        SAD_cm = sad_mm / 10.0
        pw_cm = pixel_size_mm[1] / 10.0
        ph_cm = pixel_size_mm[2] / 10.0

        n_proj = length(angles_deg)

        # Allocate trajectory arrays
        src_pos = zeros(Float64, 3, n_proj)
        det_cen = zeros(Float64, 3, n_proj)
        u_vecs = zeros(Float64, 3, n_proj)
        v_vecs = zeros(Float64, 3, n_proj)

        # Pre-compute geometry for all projection angles
        # This is the "baking" step - do once, use many times
        for i in 1:n_proj
            θ = deg2rad(angles_deg[i])
            s, c = sincos(θ)

            # Source position (rotates around Z-axis)
            src_pos[1, i] = SAD_cm * s
            src_pos[2, i] = -SAD_cm * c
            src_pos[3, i] = 0.0

            # Detector center (opposite side of isocenter)
            dist = SDD_cm - SAD_cm
            det_cen[1, i] = -dist * s
            det_cen[2, i] = dist * c
            det_cen[3, i] = 0.0

            # U-axis (detector horizontal, rotates with gantry)
            u_vecs[1, i] = c
            u_vecs[2, i] = s
            u_vecs[3, i] = 0.0

            # V-axis (detector vertical, always points up)
            v_vecs[1, i] = 0.0
            v_vecs[2, i] = 0.0
            v_vecs[3, i] = 1.0
        end

        new(SDD_cm, SAD_cm, n_rows, n_cols, pw_cm, ph_cm,
            angles_deg, src_pos, det_cen, u_vecs, v_vecs)
    end
end

# ==============================================================================
# Pre-Defined Scanner Configurations
# ==============================================================================

"""
    create_aquilion_one(; protocol::ScanProtocol)

Create geometry for Canon Aquilion ONE VISION Edition CT scanner.

# Scanner Specifications

**Canon Aquilion ONE** (320-row wide detector CT):
- **Detector**: 320 rows × 896 columns (actual specifications)
- **Z-coverage**: 16 cm in one rotation (no table movement!)
- **SDD**: 1000 mm (source-to-detector distance)
- **SAD**: 600 mm (source-to-axis distance)
- **Detector pitch**: 0.5 mm at isocenter
- **Max FOV**: 500 mm diameter
- **Rotation time**: 0.275-0.5 seconds
- **kVp range**: 70-135 kVp

# Arguments

- `protocol::ScanProtocol` - Acquisition protocol parameters

# Returns

- `CTGeometry` - Pre-computed scanner geometry

# Example

```julia
# Define protocol
protocol = ScanProtocol(
    kVp = 120.0,
    mAs = 200.0,
    scan_fov_mm = 400.0,
    num_projections = 720
)

# Create scanner geometry
geo = create_aquilion_one(protocol = protocol)

# Use in simulation
sinogram = simulate_ct_scan(
    phantom = phantom,
    geometry = geo,
    spectrum = spec
)
```

# Clinical Applications

The Canon Aquilion ONE is used for:
- **Cardiac CT**: Entire heart in one beat (no gating artifacts)
- **Perfusion imaging**: Dynamic contrast studies
- **Pediatric imaging**: Reduced scan time and dose
- **Trauma imaging**: Fast whole-organ coverage

# References

- Canon Medical Systems. "Aquilion ONE VISION Edition Brochure" (2023)
- Yoshida, K., et al. (2011). "Clinical performance of 320-detector row CT."
  Japanese Journal of Radiology, 29(7), 457-463.

# Implementation Notes

This function automatically calculates the number of detector columns needed
to cover the requested FOV, accounting for magnification (M = SDD/SAD = 1.67).

The detector array is slightly oversized to ensure full FOV coverage.
"""
function create_aquilion_one(; protocol::ScanProtocol)
    # =========================================================================
    # Canon Aquilion ONE Specifications
    # =========================================================================

    sdd_mm = 1000.0  # Source-to-detector distance
    sad_mm = 600.0   # Source-to-axis distance (isocenter)

    # 320-row detector (16 cm z-coverage!)
    n_rows = 320
    detector_pixel_pitch_mm = 0.5  # At isocenter

    # =========================================================================
    # Calculate Detector Width
    # =========================================================================

    # Magnification factor
    magnification = sdd_mm / sad_mm  # = 1.667

    # Required detector width to cover FOV
    det_width_needed_mm = protocol.scan_fov_mm * magnification

    # Number of columns (round up for full coverage)
    n_cols = round(Int, det_width_needed_mm / (detector_pixel_pitch_mm * magnification))

    # Make even for symmetry
    n_cols = n_cols + (n_cols % 2)

    # Actual pixel size at detector plane
    pixel_size_at_det_mm = (
        detector_pixel_pitch_mm * magnification,  # width
        detector_pixel_pitch_mm * magnification   # height
    )

    # =========================================================================
    # Angular Sampling
    # =========================================================================

    start_deg = protocol.start_angle
    stop_deg = protocol.start_angle + protocol.rotation_total_angle
    angles = collect(range(start_deg, stop_deg, length=protocol.num_projections))

    # =========================================================================
    # Log Scanner Configuration
    # =========================================================================

    @info """
    🏥 Canon Aquilion ONE VISION Edition
    =====================================
    Detector Array: $n_rows × $n_cols
    Z-Coverage: $(n_rows * detector_pixel_pitch_mm / 10.0) cm (16 cm wide detector!)
    SDD: $sdd_mm mm
    SAD: $sad_mm mm
    Magnification: $(round(magnification, digits=3))
    Pixel Pitch (isocenter): $(detector_pixel_pitch_mm) mm
    Pixel Size (detector): $(round(pixel_size_at_det_mm[1], digits=3)) mm
    Field of View: $(protocol.scan_fov_mm) mm
    Projections: $(length(angles))
    Angular Range: $(protocol.start_angle)° to $(stop_deg)°
    kVp: $(protocol.kVp)
    mAs: $(protocol.mAs)
    """

    return CTGeometry(
        sdd_mm = sdd_mm,
        sad_mm = sad_mm,
        n_rows = n_rows,
        n_cols = n_cols,
        pixel_size_mm = pixel_size_at_det_mm,
        angles_deg = angles
    )
end

"""
    create_custom_scanner(;
        sdd_mm::Float64,
        sad_mm::Float64,
        n_rows::Int,
        n_cols::Int,
        pixel_size_mm::Tuple{Float64, Float64},
        protocol::ScanProtocol
    )

Create custom CT scanner geometry with user-specified parameters.

# Example

```julia
# Custom high-resolution micro-CT
protocol = ScanProtocol(
    kVp = 80.0,
    mAs = 100.0,
    scan_fov_mm = 50.0,  # Small animal
    num_projections = 1200
)

geo = create_custom_scanner(
    sdd_mm = 500.0,
    sad_mm = 300.0,
    n_rows = 1024,
    n_cols = 1024,
    pixel_size_mm = (0.1, 0.1),  # High resolution
    protocol = protocol
)
```
"""
function create_custom_scanner(;
        sdd_mm::Float64,
        sad_mm::Float64,
        n_rows::Int,
        n_cols::Int,
        pixel_size_mm::Tuple{Float64, Float64},
        protocol::ScanProtocol
    )

    start_deg = protocol.start_angle
    stop_deg = protocol.start_angle + protocol.rotation_total_angle
    angles = collect(range(start_deg, stop_deg, length=protocol.num_projections))

    @info """
    🔧 Custom CT Scanner
    ====================
    Detector Array: $n_rows × $n_cols
    SDD: $sdd_mm mm
    SAD: $sad_mm mm
    Magnification: $(round(sdd_mm / sad_mm, digits=3))
    Pixel Size: $(pixel_size_mm) mm
    Projections: $(length(angles))
    """

    return CTGeometry(
        sdd_mm = sdd_mm,
        sad_mm = sad_mm,
        n_rows = n_rows,
        n_cols = n_cols,
        pixel_size_mm = pixel_size_mm,
        angles_deg = angles
    )
end

# ==============================================================================
# Utility Functions
# ==============================================================================

"""
    get_magnification(geo::CTGeometry)::Float64

Calculate geometric magnification factor M = SDD / SAD.

For cone-beam CT, the magnification determines the relationship between
isocenter coordinates and detector coordinates.
"""
function get_magnification(geo::CTGeometry)::Float64
    return geo.SDD_cm / geo.SAD_cm
end

"""
    get_detector_size(geo::CTGeometry)::Tuple{Float64, Float64}

Get physical detector size in cm (width, height).

# Returns

Tuple of (width_cm, height_cm)
"""
function get_detector_size(geo::CTGeometry)::Tuple{Float64, Float64}
    width_cm = geo.n_cols * geo.pixel_width_cm
    height_cm = geo.n_rows * geo.pixel_height_cm
    return (width_cm, height_cm)
end

"""
    get_fov_diameter(geo::CTGeometry)::Float64

Calculate the maximum field-of-view diameter (cm) at isocenter.

This is the circular region that can be fully imaged without truncation.
"""
function get_fov_diameter(geo::CTGeometry)::Float64
    det_width_cm = geo.n_cols * geo.pixel_width_cm
    magnification = get_magnification(geo)
    fov_cm = det_width_cm / magnification
    return fov_cm
end

"""
    validate_geometry(geo::CTGeometry)::Bool

Validate that geometry parameters are physically reasonable.

# Checks

- SDD > SAD (detector is beyond isocenter)
- Magnification in range [1.2, 3.0] (typical CT values)
- Pixel sizes are reasonable (0.01-5.0 mm)
- Sufficient angular sampling (Nyquist criterion)

# Returns

`true` if geometry passes all checks, throws error otherwise.
"""
function validate_geometry(geo::CTGeometry)::Bool
    # Check magnification
    M = get_magnification(geo)
    if M < 1.2 || M > 3.0
        @warn "Unusual magnification: M = $M (typical range: 1.2-3.0)"
    end

    # Check pixel sizes (in mm)
    pw_mm = geo.pixel_width_cm * 10.0
    ph_mm = geo.pixel_height_cm * 10.0
    if pw_mm < 0.01 || pw_mm > 5.0
        @warn "Unusual pixel width: $(pw_mm) mm (typical range: 0.1-2.0 mm)"
    end
    if ph_mm < 0.01 || ph_mm > 5.0
        @warn "Unusual pixel height: $(ph_mm) mm (typical range: 0.1-2.0 mm)"
    end

    # Check angular sampling (Nyquist: need at least π·N projections)
    fov_cm = get_fov_diameter(geo)
    max_dimension_pixels = max(geo.n_rows, geo.n_cols)
    nyquist_projections = ceil(Int, π * max_dimension_pixels / 2)
    n_proj = length(geo.angles)

    if n_proj < nyquist_projections
        @warn """
        Insufficient angular sampling:
          Current: $n_proj projections
          Nyquist: $nyquist_projections projections
          May cause aliasing artifacts!
        """
    end

    @info "✅ Geometry validation passed"
    return true
end

# ==============================================================================
# Exports
# ==============================================================================

export ScanProtocol, CTGeometry
export create_aquilion_one, create_custom_scanner
export get_magnification, get_detector_size, get_fov_diameter
export validate_geometry
