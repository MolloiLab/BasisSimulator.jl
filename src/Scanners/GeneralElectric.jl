"""
    Scanners/GeneralElectric.jl

GE Healthcare CT scanner configurations.

# Supported Scanners
- GE Revolution Apex Elite (FDA 510(k): K213715)

# References
- FDA 510(k) K213715: https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf
- FDA 510(k) K133705: https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf
- PMC10332658: https://pmc.ncbi.nlm.nih.gov/articles/PMC10332658/
- GE Technical Data: https://www.gehealthcare.com/
"""

# =============================================================================
# GE Revolution Apex Elite
# =============================================================================

# URL constants for source citations
const GE_APEX_FDA_K213715 = "https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf"
const GE_APEX_FDA_K133705 = "https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf"
const GE_APEX_PMC10332658 = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10332658/"
const GE_APEX_SELLSHEET = "https://www.gehealthcare.com/-/jssmedia/gehc/us/images/products/goldseal/goldseal-ct-redesign/sell-sheet-goldseal-revolution-ct-ex160-us-jb02387xx_v2.pdf"
const GE_APEX_QUANTIX_WP = "https://www.gehealthcare.com/-/jssmedia/global/products/images/revolution-apex-platform/quantix_whitepaper_jb78157xx.pdf"
const GE_APEX_AJR_2018 = "https://ajronline.org/doi/pdf/10.2214/AJR.18.19851"

"""
    GERevolutionApexElite <: AbstractScannerSpec

GE Revolution Apex Elite CT scanner specification.

This is GE's flagship energy-integrating detector (EID) scanner with
wide-coverage volumetric imaging capability.

# FDA 510(k)
- Primary: K213715
- Predicate: K133705 (Revolution CT)

# Key Features
- 256-row Gemstone Clarity detector
- 160 mm z-axis coverage
- Quantix 160 X-ray tube (108 kW)
- 0.23 second minimum rotation
- 2,496 views per rotation maximum

# Usage
```julia
spec = GERevolutionApexElite()
print_scanner_info(spec)

# Create geometry
geom = create_geometry(spec; n_angles=984, n_rows=64)

# Create with protocol
protocol = HelicalProtocol(120, 400, 0.5, 0.992, 3.0, 984, 0.625)
geom = create_geometry(spec, protocol)
```

# References
All parameters are sourced from FDA 510(k) summaries, manufacturer
technical specifications, and peer-reviewed publications. See individual
field citations for specific sources.
"""
struct GERevolutionApexElite <: AbstractScannerSpec
    detector_spec::DetectorSpecification
    tube_spec::TubeSpecification
    geometry_spec::GeometrySpecification
    acquisition_spec::AcquisitionSpecification
end

"""
    GERevolutionApexElite()

Construct GE Revolution Apex Elite scanner specification with all
documented parameters and source citations.
"""
function GERevolutionApexElite()
    # =========================================================================
    # DETECTOR SPECIFICATION
    # =========================================================================
    # Gemstone Clarity detector array
    # Source: FDA K133705, GE sell sheet

    detector_spec = DetectorSpecification(
        # Material: Gemstone Clarity (proprietary garnet scintillator, similar to Lumex)
        SourceCitation(LUMEX;
            source=:fda_510k,
            url=GE_APEX_FDA_K133705,
            note="Gemstone Clarity is GE proprietary garnet-based scintillator"),

        # 256 detector rows
        SourceCitation(256;
            source=:fda_510k,
            url=GE_APEX_FDA_K133705,
            note="256 detector rows for 160mm z-coverage"),

        # 832 detector columns
        # Derived: 212,992 total cells / 256 rows = 832 columns
        SourceCitation(832;
            source=:derived,
            url=GE_APEX_SELLSHEET,
            note="Derived from 212,992 total cells / 256 rows"),

        # Row size: 0.625 mm
        SourceCitation(0.625;
            source=:fda_510k,
            url=GE_APEX_FDA_K133705,
            note="Native detector row size at isocenter"),

        # Column size: ~1.053 mm (derived to cover 500mm SFOV)
        # At isocenter: 500mm / (832 * (626/1097)) ≈ 1.053 mm
        SourceCitation(1.053;
            source=:derived,
            url=GE_APEX_SELLSHEET,
            note="Derived from SFOV and magnification: 500/(832*(626/1097))"),

        # Detector depth: estimated 3.0 mm for garnet scintillator
        SourceCitation(3.0;
            source=:estimate,
            url="",
            note="Typical garnet scintillator depth, manufacturer data not public"),

        # Energy-integrating detector
        SourceCitation(ENERGY_INTEGRATING;
            source=:fda_510k,
            url=GE_APEX_FDA_K213715,
            note="Traditional energy-integrating detector"),

        # Z-coverage: 160 mm
        SourceCitation(160.0;
            source=:fda_510k,
            url=GE_APEX_FDA_K133705,
            note="256 rows × 0.625 mm = 160 mm"),

        # Fill factors: estimated 0.9 typical for modern CT detectors
        SourceCitation(0.90;
            source=:estimate,
            url="",
            note="Typical fill factor, exact value proprietary"),

        SourceCitation(0.90;
            source=:estimate,
            url="",
            note="Typical fill factor, exact value proprietary")
    )

    # =========================================================================
    # TUBE SPECIFICATION
    # =========================================================================
    # Quantix 160 X-ray tube
    # Source: PMC10332658, GE Quantix whitepaper

    tube_spec = TubeSpecification(
        # Model name
        SourceCitation("Quantix 160";
            source=:publication,
            url=GE_APEX_PMC10332658,
            note="GE Quantix 160 high-power X-ray tube"),

        # Max power: 108 kW
        SourceCitation(108.0;
            source=:publication,
            url=GE_APEX_PMC10332658,
            note="Maximum generator power"),

        # Target angle: 10 degrees
        SourceCitation(10.0;
            source=:publication,
            url=GE_APEX_PMC10332658,
            note="Anode target angle"),

        # Small focal spot: 1.0 × 0.7 mm
        SourceCitation((1.0, 0.7);
            source=:manufacturer,
            url=GE_APEX_SELLSHEET,
            note="Small focal spot (width × length)"),

        # Large focal spot: 1.6 × 1.2 mm
        SourceCitation((1.6, 1.2);
            source=:manufacturer,
            url=GE_APEX_SELLSHEET,
            note="Large focal spot (width × length)"),

        # kVp options: 70, 80, 100, 120, 140
        SourceCitation([70, 80, 100, 120, 140];
            source=:manufacturer,
            url=GE_APEX_SELLSHEET,
            note="Available tube voltage settings"),

        # Max mA: 1300 mA at 70/80 kV
        SourceCitation(1300;
            source=:manufacturer,
            url=GE_APEX_QUANTIX_WP,
            note="Maximum tube current at 70/80 kVp"),

        # Flying focal spot: yes (magnetic wobble)
        SourceCitation(true;
            source=:manufacturer,
            url=GE_APEX_QUANTIX_WP,
            note="Magnetic focal spot deflection supported")
    )

    # =========================================================================
    # GEOMETRY SPECIFICATION
    # =========================================================================
    # Source: FDA K133705, GE sell sheet

    geometry_spec = GeometrySpecification(
        # SID: 626.0 mm
        SourceCitation(626.0;
            source=:manufacturer,
            url=GE_APEX_SELLSHEET,
            note="Source-to-isocenter distance"),

        # SDD: 1097.0 mm
        SourceCitation(1097.0;
            source=:manufacturer,
            url=GE_APEX_SELLSHEET,
            note="Source-to-detector distance"),

        # Gantry aperture: 800 mm (80 cm)
        SourceCitation(800.0;
            source=:fda_510k,
            url=GE_APEX_FDA_K133705,
            note="Gantry bore diameter"),

        # Max SFOV: 500 mm (50 cm)
        SourceCitation(500.0;
            source=:manufacturer,
            url=GE_APEX_SELLSHEET,
            note="Maximum scan field of view"),

        # Detector curve radius: equal to SDD for third-gen geometry
        SourceCitation(1097.0;
            source=:derived,
            url="",
            note="Curved detector array, radius = SDD")
    )

    # =========================================================================
    # ACQUISITION SPECIFICATION
    # =========================================================================
    # Source: FDA K213715, AJR 2018

    acquisition_spec = AcquisitionSpecification(
        # Min rotation time: 0.23 seconds
        SourceCitation(0.23;
            source=:fda_510k,
            url=GE_APEX_FDA_K213715,
            note="Fastest rotation time"),

        # Max rotation time: 1.0 seconds
        SourceCitation(1.0;
            source=:fda_510k,
            url=GE_APEX_FDA_K213715,
            note="Slowest rotation time"),

        # Rotation time options
        SourceCitation([0.23, 0.28, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
            source=:fda_510k,
            url=GE_APEX_FDA_K213715,
            note="Available rotation times"),

        # Max views per rotation: 2496
        SourceCitation(2496;
            source=:manufacturer,
            url="https://info.ncdhhs.gov/dhsr/coneed/reviews/2020/dec/3455-Cabarrus-Carolinas-HealthCare-System-Imaging-Kannapolis-061206-Exemption.pdf",
            note="Maximum projection views per rotation"),

        # Helical pitch options
        SourceCitation([0.5, 0.531, 0.969, 0.992, 1.375, 1.531];
            source=:publication,
            url=GE_APEX_AJR_2018,
            note="Documented helical pitch values")
    )

    return GERevolutionApexElite(detector_spec, tube_spec, geometry_spec, acquisition_spec)
end

# Implement AbstractScannerSpec interface
manufacturer(::GERevolutionApexElite) = GE_HEALTHCARE
model_name(::GERevolutionApexElite) = "Revolution Apex Elite"
fda_510k(::GERevolutionApexElite) = "K213715"
detector(spec::GERevolutionApexElite) = spec.detector_spec
tube(spec::GERevolutionApexElite) = spec.tube_spec
geometry(spec::GERevolutionApexElite) = spec.geometry_spec
acquisition(spec::GERevolutionApexElite) = spec.acquisition_spec

# =============================================================================
# GE Revolution Apex Elite - Protocol Presets
# =============================================================================

"""
    GEApexChestHelical(; dose_level=:standard)

Standard chest helical protocol for GE Revolution Apex Elite.

# Keyword Arguments
- `dose_level`: :low, :standard, or :high

# Protocol Details
- 120 kVp
- 0.5 s rotation
- 0.992 pitch
- 0.625 mm slice thickness
"""
function GEApexChestHelical(; dose_level::Symbol=:standard)
    ma = dose_level == :low ? 200 :
         dose_level == :standard ? 400 :
         dose_level == :high ? 600 : 400

    return HelicalProtocol(
        120,        # kVp
        ma,         # mA
        0.5,        # rotation time
        0.992,      # pitch
        3.0,        # rotations
        984,        # angles per rotation
        0.625       # slice thickness
    )
end

"""
    GEApexHeadAxial(; dose_level=:standard)

Standard head axial protocol for GE Revolution Apex Elite.

# Keyword Arguments
- `dose_level`: :low, :standard, or :high

# Protocol Details
- 120 kVp
- 1.0 s rotation
- Full 360° coverage
- 0.625 mm slice thickness
"""
function GEApexHeadAxial(; dose_level::Symbol=:standard)
    ma = dose_level == :low ? 150 :
         dose_level == :standard ? 300 :
         dose_level == :high ? 450 : 300

    return AxialProtocol(
        120,        # kVp
        ma,         # mA
        1.0,        # rotation time
        984,        # angles
        0.625       # slice thickness
    )
end

"""
    GEApexCardiacHelical(; dose_level=:standard)

Cardiac helical protocol for GE Revolution Apex Elite.

Uses fast rotation and low pitch for cardiac imaging.

# Protocol Details
- 120 kVp
- 0.28 s rotation (fastest practical for cardiac)
- 0.5 pitch (good temporal resolution)
- 0.625 mm slice thickness
"""
function GEApexCardiacHelical(; dose_level::Symbol=:standard)
    ma = dose_level == :low ? 400 :
         dose_level == :standard ? 600 :
         dose_level == :high ? 800 : 600

    return HelicalProtocol(
        120,        # kVp
        ma,         # mA
        0.28,       # rotation time (fast)
        0.5,        # pitch (low for cardiac)
        5.0,        # rotations
        984,        # angles per rotation
        0.625       # slice thickness
    )
end

"""
    GEApexAbdomenHelical(; dose_level=:standard)

Standard abdomen/pelvis helical protocol for GE Revolution Apex Elite.

# Protocol Details
- 120 kVp
- 0.5 s rotation
- 0.992 pitch
- 1.25 mm slice thickness
"""
function GEApexAbdomenHelical(; dose_level::Symbol=:standard)
    ma = dose_level == :low ? 250 :
         dose_level == :standard ? 450 :
         dose_level == :high ? 650 : 450

    return HelicalProtocol(
        120,        # kVp
        ma,         # mA
        0.5,        # rotation time
        0.992,      # pitch
        4.0,        # rotations
        984,        # angles per rotation
        1.25        # slice thickness
    )
end

"""
    GEApexPediatricHelical(; age_group=:child)

Pediatric helical protocol for GE Revolution Apex Elite.

Uses reduced dose parameters appropriate for pediatric imaging.

# Keyword Arguments
- `age_group`: :infant, :child, or :adolescent

# Protocol Details
- 80-100 kVp (based on age)
- Reduced mA
- 0.5 s rotation
- 0.992 pitch
"""
function GEApexPediatricHelical(; age_group::Symbol=:child)
    kvp = age_group == :infant ? 80 :
          age_group == :child ? 100 :
          age_group == :adolescent ? 120 : 100

    ma = age_group == :infant ? 80 :
         age_group == :child ? 150 :
         age_group == :adolescent ? 250 : 150

    return HelicalProtocol(
        kvp,        # kVp
        ma,         # mA
        0.5,        # rotation time
        0.992,      # pitch
        2.0,        # rotations
        984,        # angles per rotation
        0.625       # slice thickness
    )
end

# =============================================================================
# Exports
# =============================================================================

export GERevolutionApexElite
export GEApexChestHelical, GEApexHeadAxial, GEApexCardiacHelical
export GEApexAbdomenHelical, GEApexPediatricHelical
