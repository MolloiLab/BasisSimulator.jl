let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
        # ── shared styles ──────────────────────────────────────────────────
        card_cls = "border border-warm-200 dark:border-warm-800 rounded-xl p-6 space-y-3 bg-warm-50 dark:bg-warm-900/40"
        code_cls = "mt-2 bg-warm-900 dark:bg-warm-950 text-warm-200 p-3 rounded text-xs font-mono overflow-x-auto"
        sig_cls = "font-mono font-semibold text-warm-900 dark:text-warm-100 text-sm"
        prose_cls = "text-sm text-warm-600 dark:text-warm-400 leading-relaxed"
        note_cls = "text-xs text-warm-500 dark:text-warm-400 italic"
        h2_cls = "text-2xl font-semibold text-warm-800 dark:text-warm-200"
        h3_cls = "text-lg font-semibold text-warm-700 dark:text-warm-300"
        inline = "text-accent-500 font-mono"
        table_cls = "w-full text-xs text-left border-collapse"
        th_cls = "py-1.5 pr-4 font-mono text-warm-700 dark:text-warm-300 border-b border-warm-200 dark:border-warm-800"
        td_cls = "py-1.5 pr-4 text-warm-600 dark:text-warm-400 border-b border-warm-100 dark:border-warm-900 align-top"
        td_mono = "py-1.5 pr-4 font-mono text-accent-600 dark:text-accent-400 border-b border-warm-100 dark:border-warm-900 align-top"

        # ── ToC sections — top-level H2 anchors ────────────────────────────
        sections = [
            ("overview", "Overview"),
            ("five-structs", "The five-struct API"),
            ("phantom", "Phantom & materials"),
            ("geometry", "Geometry & coordinate mapping"),
            ("spectrum", "Spectrum & source"),
            ("forward", "Forward projection"),
            ("dose", "Dose & CTDI"),
            ("reconstruction", "Reconstruction"),
            ("corrections", "Corrections & calibration"),
            ("detector", "Detector physics"),
            ("constants", "Constants & calibrations"),
        ]

        PageWithTOC(
            sections, Div(
                :class => "max-w-3xl mx-auto space-y-12",

                # ════════════════════════════════════════════════════════════════
                # Page header
                # ════════════════════════════════════════════════════════════════
                H1(
                    :class => "text-3xl font-serif font-bold text-warm-900 dark:text-warm-100",
                    "API Reference"
                ),
                P(
                    :class => "text-warm-600 dark:text-warm-400 leading-relaxed",
                    "BasisSimulator.jl exposes a small core surface — five structs that define ",
                    "the simulation, two entry points (",
                    Code(:class => inline, "simulate!"), " and ",
                    Code(:class => inline, "reconstruct!"),
                    "), and a stable set of helpers organized by workflow stage. ",
                    "This page documents the public API in the order you actually use it; ",
                    "the worked examples on ",
                    A(:href => "$(BASE)/examples/", :class => "text-accent-500 hover:text-accent-600 underline no-underline", "/examples/"),
                    " show every entry below in a real pipeline."
                ),

                # ════════════════════════════════════════════════════════════════
                # § 1. Overview
                # ════════════════════════════════════════════════════════════════
                H2(:id => "overview", :class => h2_cls, "Overview"),
                P(
                    :class => prose_cls,
                    "Every simulation walks the same five-stage pipeline:"
                ),
                Pre(
                    :class => code_cls, Code(
                        :class => "language-text", """Phantom   ─┐
                        EICT/PCCTScanner ─┤
                        Protocol  ─┼─▶  create_*_workspace → simulate!  →  reconstruct!  →  to_hounsfield → recon (HU)
                        SimOpts   ─┤
                        ReconOpts ─┘"""
                    )
                ),
                P(
                    :class => prose_cls,
                    "All five stages are pure Julia structs.  GPU vs CPU is selected by where the ",
                    "phantom mask lives — wrap with ",
                    Code(:class => inline, "MtlArray"), " / ",
                    Code(:class => inline, "CuArray"), " / ",
                    Code(:class => inline, "ROCArray"),
                    " for GPU, leave as plain ", Code(:class => inline, "Array"),
                    " for CPU.  The same code path runs everywhere."
                ),

                # ════════════════════════════════════════════════════════════════
                # § 2. The five-struct API
                # ════════════════════════════════════════════════════════════════
                H2(:id => "five-structs", :class => h2_cls, "The five-struct API"),
                P(
                    :class => prose_cls,
                    "These are the load-bearing types — every public entry point in the package ",
                    "takes some combination of them.  Four are concrete structs with named fields; ",
                    "the scanner is the one abstraction, an abstract ", Code(:class => inline, "Scanner"),
                    " with a concrete family per detector type, so that a photon-counting parameter ",
                    "cannot be set on a scintillator and the other way round.  Construct, pass, done."
                ),

                # ── 2a. Phantom ──────────────────────────────────────────────
                H3(:id => "phantom-struct", :class => h3_cls, "Phantom"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "Phantom(mask, materials, voxel_size [, origin, extent])"),
                    P(
                        :class => prose_cls,
                        "The thing being scanned.  A labeled voxel mask + a vector of ",
                        Code(:class => inline, "XA.Material"),
                        " (one per label) + the physical voxel size in cm.  The mask's array type drives the ",
                        "GPU/CPU choice for the rest of the pipeline."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "field"), Th(:class => th_cls, "type"), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "mask"), Td(:class => td_cls, "AbstractArray{<:Unsigned, 3}"), Td(:class => td_cls, "Per-voxel region label.  Array type selects the backend.")),
                        Tr(Td(:class => td_mono, "materials"), Td(:class => td_cls, "Vector{XA.Material}"), Td(:class => td_cls, "One material per region label; index = label + 1.")),
                        Tr(Td(:class => td_mono, "voxel_size"), Td(:class => td_cls, "NTuple{3,Float64}"), Td(:class => td_cls, "(dx, dy, dz) in cm.")),
                        Tr(Td(:class => td_mono, "origin"), Td(:class => td_cls, "NTuple{3,Float64}"), Td(:class => td_cls, "Center of voxel (0,0,0) in world coords (cm).  Defaults to centered-at-iso.")),
                        Tr(Td(:class => td_mono, "extent"), Td(:class => td_cls, "NTuple{3,Float64}"), Td(:class => td_cls, "Physical extent (cm) = size(mask) .* voxel_size.")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """import GPUSelect
                            AT = GPUSelect.Storage()
                            to_gpu(x) = AT(x)
                            phantom_cpu = BS.create_gammex_472(n_voxels = 512)
                            phantom = BS.Phantom(to_gpu(phantom_cpu.mask),   # mask on the device
                                                 phantom_cpu.materials,     # materials on the host
                                                 phantom_cpu.voxel_size,
                                                 phantom_cpu.origin,
                                                 phantom_cpu.extent)"""
                        )
                    ),
                    P(:class => note_cls, "see: notebooks 01, 02, 05"),
                ),

                # ── 2b. Scanner ──────────────────────────────────────────────
                H3(:id => "scanner-struct", :class => h3_cls, "EICTScanner / PCCTScanner"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "EICTScanner(; source_to_isocenter, source_to_detector, detector_rows, ...)  |  PCCTScanner(; energy_thresholds, ...)"),
                    P(
                        :class => prose_cls,
                        "The hardware — geometry, source, detector, filtration.  All distances in mm; ",
                        "detector pitch is at isocenter (not detector face — divide by magnification ",
                        "if you're porting from a face-pitch convention)."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "kwarg group"), Th(:class => th_cls, ""), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "Geometry"), Td(:class => td_cls, ""), Td(:class => td_cls, "source_to_isocenter, source_to_detector (mm); scan_diameter; gantry_aperture")),
                        Tr(Td(:class => td_mono, "Detector array"), Td(:class => td_cls, ""), Td(:class => td_cls, "detector_rows, detector_cols, detector_row_size, detector_col_size, detector_row_offset, detector_col_offset, detector_shape (:flat | :arc)")),
                        Tr(Td(:class => td_mono, "Source"), Td(:class => td_cls, ""), Td(:class => td_cls, "focal_spot_width, focal_spot_length, target_angle")),
                        Tr(Td(:class => td_mono, "Filtration"), Td(:class => td_cls, ""), Td(:class => td_cls, "flat_filter_material, flat_filter_thickness, bowtie_filter (Symbol or struct)")),
                        Tr(Td(:class => td_mono, "EICTScanner"), Td(:class => td_cls, ""), Td(:class => td_cls, "detector_material, detector_depth, fill_factor_row, fill_factor_col, detection_gain, electronic_noise")),
                        Tr(Td(:class => td_mono, "PCCTScanner"), Td(:class => td_cls, ""), Td(:class => td_cls, "n_energy_bins, energy_thresholds, energy_resolution, charge_sharing_fwhm, dead_time_ns, pixel_mode, native_dexel_col_mm, native_dexel_row_mm, binning_factor, pileup (modelled only when dead_time_ns > 0), pileup_correction, scatter_correction, noise_reduction")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """# GE Revolution Apex Elite (EICT, polychromatic)
                            scanner = BS.EICTScanner(
                                source_to_isocenter = 625.6,
                                source_to_detector  = 1100.0,
                                detector_rows       = 256,
                                detector_cols       = 834,
                                detector_row_size   = 0.625,
                                detector_col_size   = 0.6,
                                bowtie_filter       = :ge_revolution_large,
                                detector_material   = :lumex,
                                detector_depth      = 3.0,
                            )"""
                        )
                    ),
                    P(:class => note_cls, "see: notebooks 01, 02, 03, 04, 05, 06"),
                ),

                # ── 2c. CTProtocol ───────────────────────────────────────────
                H3(:id => "ctprotocol-struct", :class => h3_cls, "CTProtocol"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "CTProtocol(; kVp, mA, views, rotation_time, collimation_mm, additional_filters)"),
                    P(
                        :class => prose_cls,
                        "The acquisition — what tube voltage, how much current, how many views, ",
                        "how thick a slab.  ", Code(:class => inline, "additional_filters"),
                        " is a list of extra material/thickness pairs layered on top of the scanner's flat filter."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """protocol = BS.CTProtocol(
                                kVp = 120,
                                mA  = 200.0,
                                views          = 500,
                                rotation_time  = 1.0,
                                collimation_mm = 5.0,
                                additional_filters = [("Al", 4.5), ("Ti", 0.9)],   # extra Al + Ti on top of Scanner.flat_filter_*
                            )"""
                        )
                    ),
                    P(:class => note_cls, "Active detector rows are derived from collimation_mm via geom.n_rows = round(collimation_mm / scanner.detector_row_size)."),
                ),

                # ── 2d. SimOptions ───────────────────────────────────────────
                H3(:id => "simoptions-struct", :class => h3_cls, "SimOptions"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "SimOptions(; seed, use_*..., detector_efficiency_mode, projector)"),
                    P(
                        :class => prose_cls,
                        "The physics common to both detector families, one toggle each.  What belongs to a ",
                        "particular detector — pile-up, its correction, the scatter correction, the DAS noise ",
                        "reduction — lives on ", Code(:class => inline, "PCCTScanner"), " instead, so a scan ",
                        "cannot be configured with physics its detector does not have."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "field"), Th(:class => th_cls, "default"), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "seed"), Td(:class => td_cls, "42"), Td(:class => td_cls, "Reproducibility for noise / scatter sampling.")),
                        Tr(Td(:class => td_mono, "use_noise"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Quantum (Poisson) noise.")),
                        Tr(Td(:class => td_mono, "use_scatter"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Ohnesorge spatial scatter model.")),
                        Tr(Td(:class => td_mono, "use_focal_spot"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Focal-spot blur.")),
                        Tr(Td(:class => td_mono, "use_heel_effect"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Anode heel-effect intensity gradient.")),
                        Tr(Td(:class => td_mono, "use_lag"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Scintillator afterglow (EICT only).")),
                        Tr(Td(:class => td_mono, "use_optical_crosstalk"), Td(:class => td_cls, "false"), Td(:class => td_cls, "EICT scintillator pixel crosstalk.")),
                        Tr(Td(:class => td_mono, "use_fill_factor"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Sub-pixel detector fill factor.")),
                        Tr(Td(:class => td_mono, "use_detector_efficiency"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Energy-dependent quantum efficiency.")),
                        Tr(Td(:class => td_mono, "detector_efficiency_mode"), Td(:class => td_cls, ":auto"), Td(:class => td_cls, ":auto | :mc_lut | :beer_lambert")),
                        Tr(Td(:class => td_mono, "projector"), Td(:class => td_cls, ":dd_fast"), Td(:class => td_cls, ":dd_fast | :dd | :siddon")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """sim_opts = BS.SimOptions(seed = 1234)
                            # photon-counting detector physics belongs to the scanner:
                            scanner = BS.PCCTScanner(energy_thresholds = [20.0, 35.0, 55.0, 70.0],
                                                     dead_time_ns = 5.0,      # pile-up needs one
                                                     pileup_correction = true,
                                                     noise_reduction = 0.3)"""
                        )
                    ),
                ),

                # ── 2e. ReconOptions ─────────────────────────────────────────
                H3(:id => "reconoptions-struct", :class => h3_cls, "ReconOptions"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "ReconOptions(; matrix_size, fov_cm, z_cm = nothing)"),
                    P(
                        :class => prose_cls,
                        "The output grid. Recon is ", Strong("always centered at isocenter"),
                        " — there's no off-center FOV parameter (see notebook 05 for the SFOV-equivalent ",
                        "phantom-cropping pattern)."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "field"), Th(:class => th_cls, ""), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "matrix_size"), Td(:class => td_cls, ""), Td(:class => td_cls, "(nx, ny, nz) recon volume shape")),
                        Tr(Td(:class => td_mono, "fov_cm"), Td(:class => td_cls, ""), Td(:class => td_cls, "In-plane FOV diameter (cm)")),
                        Tr(Td(:class => td_mono, "z_cm"), Td(:class => td_cls, ""), Td(:class => td_cls, "Recon slab thickness (cm); usually = collimation_mm/10")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """recon_opts = BS.ReconOptions(
                                matrix_size = (512, 512, 8),
                                fov_cm      = 35.0,
                                z_cm        = 0.5,
                            )"""
                        )
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 3. Phantom & materials
                # ════════════════════════════════════════════════════════════════
                H2(:id => "phantom", :class => h2_cls, "Phantom & materials"),

                # ── 3a. Phantom factories ─────────────────────────────────────
                H3(:id => "phantom-factories", :class => h3_cls, "Phantom factories"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_gammex_472(; n_voxels = 64, n_slices = nothing, fov_cm = 35.0, z_cm = 4.0) → Phantom"),
                    P(
                        :class => prose_cls,
                        "The Gammex Model 472 calibration phantom — 33 cm solid-water cylinder + 7 calcium ",
                        "rods (50–600 mg/mL) + 7 iodine rods (2–20 mg/mL) at 28 mm rod diameter.  ",
                        "Returns a ", Code(:class => inline, "Phantom"),
                        " with semantic mask labels (10–16 = Ca, 20–26 = I)."
                    ),
                    Pre(:class => code_cls, Code(:class => "language-julia", """phantom = BS.create_gammex_472(n_voxels = 256, n_slices = 8, fov_cm = 35.0, z_cm = 0.5)""")),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_phantom_from_mask(labeled_array, materials_dict, voxel_size_cm; origin = nothing) → Phantom"),
                    P(
                        :class => prose_cls,
                        "Wrap any labeled UInt8 voxel grid + a label → material dict into a ",
                        Code(:class => inline, "Phantom"), ".  Used by notebooks 02 / 05 for XCAT loading."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """xcat_mask = load_xcat_bin("vmale_50.bin"; cols = 1600, rows = 1400, slices = 500)
                            phantom = BS.create_phantom_from_mask(xcat_mask, materials_dict, (0.04, 0.04, 0.04))"""
                        )
                    ),
                ),

                # ── 3b. Attenuation helpers ───────────────────────────────────
                H3(:id => "attenuation-helpers", :class => h3_cls, "Attenuation helpers"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_μ(phantom, energy_keV) → Array{Float32, 3}"),
                    P(
                        :class => prose_cls,
                        "Per-voxel linear attenuation coefficient (cm⁻¹) at a given energy, ",
                        "derived by indexing each voxel's material's μ at ", Code(:class => inline, "energy_keV"),
                        ".  Useful for monochromatic forward projection or theoretical reference volumes."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_μ_at_energy(material::XA.Material, energy_keV) → Float64"),
                    P(
                        :class => prose_cls,
                        "Linear attenuation μ (cm⁻¹) of a single material at a single energy.  ",
                        "Backed by NIST XCOM data through XrayAttenuation.jl."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_mass_μ_at_energy(material, energy_keV) → Float64"),
                    P(
                        :class => prose_cls,
                        "Mass attenuation μ/ρ (cm²/g).  Numerically equal to ",
                        Code(:class => inline, "compute_μ_at_energy"),
                        " for water (ρ ≈ 1).  Used internally by the M-matrix μ_water computation."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "get_reference_μ_water(energy_keV) → Float64"),
                    P(
                        :class => prose_cls,
                        "Convenience: μ_water at a single energy.  For polychromatic scans, prefer ",
                        Code(:class => inline, "compute_polychromatic_μ_water"),
                        " (§5) which integrates over the spectrum + phantom hardening."
                    ),
                ),

                # ── 3c. Materials via BS.XA ───────────────────────────────────
                H3(:id => "materials-xa", :class => h3_cls, "Materials via ", Code(:class => inline, "BS.XA")),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "BS.XA   # re-exported XrayAttenuation submodule"),
                    P(
                        :class => prose_cls,
                        "XrayAttenuation.jl is re-exported as ", Code(:class => inline, "BS.XA"),
                        " — you don't need to add XrayAttenuation as a separate dependency.  ",
                        "Use it for the canonical Material constructor and the prebuilt NCAT/XCAT tissue library."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """# Prebuilt tissues (full NCAT/XCAT library)
                            water  = BS.XA.Materials.water
                            muscle = BS.XA.Materials.ncat_muscle
                            blood  = BS.XA.Materials.ncat_blood

                            # Custom material from elemental mass fractions
                            iodine_blood = BS.XA.Material(
                                "iodine_blood_5mgml",
                                z_a_ratio,
                                mean_excitation_energy * BS.XA.u"eV",
                                1.06 * BS.XA.u"g/cm^3",
                                Dict(1 => 0.105, 6 => 0.110, 7 => 0.033, 8 => 0.745, ...),  # Z → mass_frac
                            )"""
                        )
                    ),
                    P(:class => note_cls, "see notebook 02 for a full custom-material walkthrough"),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 4. Geometry & coordinate mapping
                # ════════════════════════════════════════════════════════════════
                H2(:id => "geometry", :class => h2_cls, "Geometry & coordinate mapping"),

                # ── 4a. CTGeometry & detector shape ───────────────────────────
                H3(:id => "ct-geometry", :class => h3_cls, "CTGeometry"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "CTGeometry(scanner; n_angles, fov_cm, z_cm, collimation_mm) → CTGeometry"),
                    P(
                        :class => prose_cls,
                        "Pre-computes 3-D source/detector positions for every view angle.  ",
                        "Built automatically inside ", Code(:class => inline, "create_*_workspace"),
                        " calls; you only need to construct one explicitly when you want to ",
                        "inspect the recon grid before running ", Code(:class => inline, "simulate!"),
                        " (e.g. the affine round-trip in notebook 05)."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """geom = BS.CTGeometry(scanner;
                                n_angles       = protocol.views,
                                fov_cm         = recon_opts.fov_cm,
                                z_cm           = recon_opts.z_cm,
                                collimation_mm = protocol.collimation_mm,
                            )
                            # geom.n_cols, geom.n_rows, geom.n_angles, geom.fov, geom.angles, ..."""
                        )
                    )
                ),
                # ── 4b. Affine round-trip ─────────────────────────────────────
                H3(:id => "affine", :class => h3_cls, "Affine round-trip"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "phantom_to_world_affine(phantom) → Matrix{Float64}  (4×4)"),
                    P(
                        :class => prose_cls,
                        "Maps 0-indexed phantom voxel ",
                        Code(:class => inline, "(i, j, k)"), " → world ",
                        Code(:class => inline, "(x, y, z)"),
                        " in cm.  Pure scale + translate; encodes ",
                        Code(:class => inline, "phantom.voxel_size"), " and ",
                        Code(:class => inline, "phantom.origin"), "."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "recon_to_world_affine(geom, matrix_size) → Matrix{Float64}  (4×4)"),
                    P(
                        :class => prose_cls,
                        "Same shape as above but for the reconstruction grid.  Origin is hard-locked to ",
                        "isocenter-centered (",
                        Code(:class => inline, "tx = -fov/2 + voxel_size/2"),
                        ")."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "resample_to_recon(phantom, geom, matrix_size; method = :nearest | :linear)"),
                    P(
                        :class => prose_cls,
                        "Resample the phantom mask onto the recon grid using the two affines above.  ",
                        Code(:class => inline, ":nearest"), " preserves integer labels (returns ",
                        Code(:class => inline, "UInt8"), "); ",
                        Code(:class => inline, ":linear"), " does trilinear (returns ",
                        Code(:class => inline, "Float32"),
                        ").  Use the linear path on a binary mask if you want true partial-volume fractions."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """gt_nn  = BS.resample_to_recon(phantom_cpu, geom, recon_opts.matrix_size; method = :nearest)
                            gt_lin = BS.resample_to_recon(phantom_cpu, geom, recon_opts.matrix_size; method = :linear)
                            # Both have shape == recon volume shape — voxel-aligned for ROI extraction."""
                        )
                    ),
                    P(:class => note_cls, "see notebook 05 for the full round-trip + custom-interpolator pattern"),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 5. Spectrum & source
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "resample_field_to_recon(field, voxel_size, origin, geom, matrix_size; outside = 0, to_backend = identity) → Array{Float32,3}"),
                    P(
                        :class => prose_cls,
                        "The exact box average of a continuous field — a material fraction, a density — onto the ",
                        "reconstruction grid, from the axis-aligned overlap of the two grids: one banded kernel per axis ",
                        "(an output voxel overlaps a few consecutive source voxels), the axis that shrinks most first.  ",
                        "Every output voxel is the mean of the field over its own footprint, ",
                        "so a 0.2 mm truth on 0.625 mm slices carries the partial volume a reconstruction sees.  ",
                        Code(:class => inline, "resample_to_recon(…; method = :linear)"),
                        " point-samples at the voxel centre and does not.  ", Code(:class => inline, "outside"),
                        " fills the part of a voxel beyond the field's grid — 1 for an air fraction — so fractions ",
                        "that sum to one still do.  With a device array as ", Code(:class => inline, "field"),
                        " and its constructor as ", Code(:class => inline, "to_backend"),
                        " (a 1600 × 1400 × 200 anatomy window as a ", Code(:class => inline, "CuArray"),
                        "), the average runs on the device — AcceleratedKernels, no BLAS — and only the result comes back: ",
                        "0.13 s per field against 4.6 s on the host, bit-identical."
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                H2(:id => "spectrum", :class => h2_cls, "Spectrum & source"),
                P(
                    :class => prose_cls,
                    "The X-ray spectrum is resolved once per simulation from the scanner's flat filter, ",
                    "the protocol's additional filters, and (optionally) the bowtie.  ",
                    "These helpers are typically called inside ", Code(:class => inline, "simulate!"),
                    ", but exposed publicly because BHC and μ_water calibration need to share the ",
                    "same spectrum the simulator uses."
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "load_spectrum(kVp::Int) → (energies, weights)"),
                    P(
                        :class => prose_cls,
                        "Raw tube spectrum at the given kVp before any flat / additional filtration.  ",
                        "Backed by IPEM Report 78 tables.  ",
                        Code(:class => inline, "energies"), " in keV, ",
                        Code(:class => inline, "weights"),
                        " are normalized photon-count fractions per energy bin."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner) → (e, w)"),
                    P(
                        :class => prose_cls,
                        "The post-filtration spectrum the simulator actually uses, EXCLUDING bowtie. ",
                        "This is what you want for BHC calibration + scatter-bin-weight derivation ",
                        "(both are bowtie-agnostic by construction)."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "resolve_source_spectrum_with_bowtie(sim_opts, protocol; scanner, geom, include_bowtie = true) → (e, ŵ)"),
                    P(
                        :class => prose_cls,
                        "Same as above but with the bowtie attenuation folded in per detector column.  ",
                        Code(:class => inline, "ŵ"),
                        " is 3D ", Code(:class => inline, "(n_col, n_row, n_E)"),
                        " when bowtie is present, 1D otherwise."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_polychromatic_μ_water(sim_opts, protocol; scanner, geom, water_path_cm) → Float64"),
                    P(
                        :class => prose_cls,
                        "Spectrum-weighted μ_water (cm⁻¹) after Beer-Lambert hardening through ",
                        Code(:class => inline, "water_path_cm"), " of water.  ",
                        "Pass the phantom diameter (full chord, not radius) — e.g. 33.0 for a Gammex 472 body. ",
                        "This is the value to use as ", Code(:class => inline, "to_hounsfield(...; μ_water = ...)"),
                        " for clean HU baselines."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """μ_water = BS.compute_polychromatic_μ_water(
                                sim_opts, protocol;
                                scanner = scanner, geom = geom, water_path_cm = 33.0,
                            )"""
                        )
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 6. Forward projection
                # ════════════════════════════════════════════════════════════════
                H2(:id => "forward", :class => h2_cls, "Forward projection"),

                # ── 6a. Simulation workspaces ─────────────────────────────────
                H3(:id => "sim-workspaces", :class => h3_cls, "Simulation workspaces"),
                P(
                    :class => prose_cls,
                    "Allocate a workspace once, reuse it across ", Code(:class => inline, "simulate!"),
                    " calls.  After JIT warm-up the hot path is zero-allocation."
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom) → EICTWorkspace"),
                    P(
                        :class => prose_cls,
                        "EICT (single-kVp polychromatic) workspace.  Pre-allocates per-energy intensity ",
                        "buffers, bowtie spectral table, scatter / focal-spot / lag kernels, noise staging, ",
                        "and the geometry struct.  Backend is inferred from ", Code(:class => inline, "phantom.mask"), "."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_workspace(scanner, protocol, sim_opts, recon_opts, phantom) → PCCTWorkspace"),
                    P(
                        :class => prose_cls,
                        "PCCT (multi-bin spectral) workspace.  Adds per-energy DRM tile buffers, ",
                        "native-dexel-resolution buffers (when ", Code(:class => inline, "binning_factor > 1"),
                        "), pileup matrix, charge-sharing kernel, anti-coincidence matrix.  Significantly ",
                        "larger than the EICT workspace; sized to your protocol's view count."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """ws = scanner isa BS.PCCTScanner ?
                            BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom) :
                            BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)"""
                        )
                    ),
                ),

                # ── 6b. simulate! ────────────────────────────────────────────
                H3(:id => "simulate", :class => h3_cls, Code(:class => inline, "simulate!")),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "simulate!(ws, phantom, protocol, sim_opts = SimOptions(); capture_raw_counts=true)"),
                    P(
                        :class => prose_cls,
                        "The forward-projection entry point.  Dispatches on workspace type — ",
                        Code(:class => inline, "EICTWorkspace"),
                        " runs the selected polychromatic projector (", Code(:class => inline, ":dd_fast"),
                        " by default) → noise → optional corrections path; ",
                        Code(:class => inline, "PCCTWorkspace"),
                        " runs spectral ray-tracing with detector-response convolution + per-bin Poisson noise."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "ws type"), Th(:class => th_cls, "writes to"), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "EICTWorkspace"), Td(:class => td_mono, "ws.sinogram"), Td(:class => td_cls, "Single Float32 sinogram (n_cols, n_rows, n_views)")),
                        Tr(Td(:class => td_mono, "PCCTWorkspace"), Td(:class => td_mono, "result.pcct_sino.bins"), Td(:class => td_cls, "Vector of n_bins sinograms; also returns pre-correction detector counts as result.raw_counts (on by default)")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
                            BS.simulate!(ws, phantom, protocol, sim_opts)
                            sino = Array(ws.sinogram)   # pull off GPU"""
                        )
                    ),
                    P(
                        :class => prose_cls,
                        "The ", Code(:class => inline, "raw_counts"),
                        " arrays (captured by default) are independent snapshots of the detector counts after enabled acquisition physics and immediately before pile-up/scatter correction — with noise on and pile-up off, bit-exact integer Poisson realizations with true zeros preserved (the floor at 1 lives only in the log-domain sinograms). The capture allocates one full sinogram per energy bin; pass ",
                        Code(:class => inline, "capture_raw_counts=false"),
                        " only when memory-constrained."
                    ),
                    P(
                        :class => prose_cls,
                        "Both methods return a NamedTuple carrying ", Code(:class => inline, "dose"),
                        " — a ", Code(:class => inline, "DoseReport"), " for the acquisition just simulated, or ",
                        Code(:class => inline, "nothing"), " when the workspace was built with a ",
                        Code(:class => inline, "spectrum_override"), " and so has no beam in absolute units.  ",
                        "Pass ", Code(:class => inline, "report_dose = false"), " to skip the Monte Carlo, or ",
                        Code(:class => inline, "dose_kwargs"), " to reach the keywords of ",
                        Code(:class => inline, "compute_dose"), "."
                    ),
                    P(:class => note_cls, "see notebooks 01–11 for canonical end-to-end call patterns"),
                ),

                # ── 6c. Cached per-material path lengths ─────────────────────
                H3(:id => "paths-cache", :class => h3_cls, "Cached per-material path lengths"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "material_paths(ws, phantom) → Array   |   material_paths!(paths, ws, phantom)"),
                    P(
                        :class => prose_cls,
                        "Walks the phantom once and returns the path length of every ray through every ",
                        "material, in cm, shaped ", Code(:class => inline, "(n_materials, n_cols, n_rows, n_views)"),
                        " on the phantom's backend.  The walk is about 95 % of a polychromatic forward ",
                        "projection and depends only on the geometry and the material map — not on kVp, ",
                        "filtration, bowtie or detector — so one cache serves every spectrum measured ",
                        "through that geometry.  Pass it back as ", Code(:class => inline, "paths"),
                        " and each further acquisition costs the spectral conversion alone, bit-identically."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """paths = BS.material_paths(ws, phantom)              # one walk
                            for kvp in (80.0, 100.0, 120.0, 140.0)
                                protocol_kvp = BS.CTProtocol(kVp = kvp, mA = 200.0, views = 984)
                                ws_kvp = BS.create_eict_workspace(scanner, protocol_kvp, sim_opts, rec_opts, phantom)
                                BS.simulate!(ws_kvp, phantom, protocol_kvp, sim_opts; paths)
                            end"""
                        )
                    ),
                    P(
                        :class => note_cls,
                        "The cache belongs to the phantom and geometry it was walked for and nothing ",
                        "downstream can tell that it does not; its size is the other cost (16 materials on an ",
                        "834 × 34 × 1000 sinogram is 1.7 GiB), so run compact_materials(phantom) first."
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 6½. Dose & CTDI
                # ════════════════════════════════════════════════════════════════
                H2(:id => "dose", :class => h2_cls, "Dose & CTDI"),
                P(
                    :class => prose_cls,
                    "CTDI is measured the way IEC 60601-2-44 defines it: the beam is transported by ",
                    "Monte Carlo through a PMMA cylinder — 32 cm body or 16 cm head — and the air kerma ",
                    "is integrated over a 100 mm pencil chamber at the centre and at the four peripheral ",
                    "holes.  Nothing about it is a lookup table, so it follows the spectrum, the flat ",
                    "filter, the added filters and the bowtie of the scanner that is actually being simulated."
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_dose(scanner, protocol; phantom = :body32, overbeam_mm = 0.0, n_tubes = 1, dose_calibration = 1.0, n_histories = 1_000_000, seed = 1) → DoseReport"),
                    P(
                        :class => prose_cls,
                        "CTDIvol and DLP of one acquisition.  Helical protocols divide CTDIw by the pitch ",
                        "and take the scan length from it; axial ones use the table increment, which ",
                        "defaults to a contiguous N·T step.  ",
                        Code(:class => inline, "overbeam_mm"),
                        " is how much wider than N·T the real collimator runs (the simulator's beam is ",
                        "exactly the collimation, so it defaults to 0); ",
                        Code(:class => inline, "dose_calibration"),
                        " multiplies the result, to pin a scanner to a measured CTDIvol."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "DoseReport field"), Th(:class => th_cls, "unit"), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "ctdi_vol_mGy"), Td(:class => td_cls, "mGy"), Td(:class => td_cls, "CTDIw scaled by mAs and pitch (or increment).")),
                        Tr(Td(:class => td_mono, "dlp_mGy_cm"), Td(:class => td_cls, "mGy·cm"), Td(:class => td_cls, "CTDIvol × scan length.")),
                        Tr(Td(:class => td_mono, "ctdi_w_mGy_per_100mAs"), Td(:class => td_cls, "mGy/100 mAs"), Td(:class => td_cls, "⅓ centre + ⅔ periphery, per 100 mAs.")),
                        Tr(Td(:class => td_mono, "ctdi100_center_… / …_periphery_…"), Td(:class => td_cls, "mGy/100 mAs"), Td(:class => td_cls, "The two chamber integrals the weighting combines.")),
                        Tr(Td(:class => td_mono, "air_kerma_free_in_air_…"), Td(:class => td_cls, "mGy/100 mAs"), Td(:class => td_cls, "Centre ray behind the bowtie, at isocentre.")),
                        Tr(Td(:class => td_mono, "rel_stat_uncertainty"), Td(:class => td_cls, "—"), Td(:class => td_cls, "Statistical error of the Monte Carlo at the chosen n_histories.")),
                        Tr(Td(:class => td_mono, "wide_beam_reference_mm"), Td(:class => td_cls, "mm"), Td(:class => td_cls, "Set when the IEC N·T > 40 mm reference-beam rule was applied.")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """report = BS.compute_dose(scanner, protocol)      # → DoseReport, prints a full summary
                            report.ctdi_vol_mGy, report.dlp_mGy_cm

                            # or straight off a simulation
                            result = BS.simulate!(ws, phantom, protocol, sim_opts)
                            result.dose.ctdi_vol_mGy"""
                        )
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "dose_source(scanner, protocol) → DoseSource   |   ctdi100(src; phantom, beam_width_mm, n_histories, seed)"),
                    P(
                        :class => prose_cls,
                        "The pieces underneath, for when the beam is wanted without an acquisition around it.  ",
                        Code(:class => inline, "DoseSource"),
                        " is the absolute spectrum with its filtration and bowtie; ",
                        Code(:class => inline, "ctdi100"),
                        " returns the two chamber integrals per mAs and their weighted combination.  ",
                        "Both results are deterministic for a given seed whatever the thread count, and ",
                        Code(:class => inline, "ctdi100"),
                        " is cached on the beam it was run for — clear it with ",
                        Code(:class => inline, "empty_ctdi_cache!()"), "."
                    ),
                    P(
                        :class => note_cls,
                        "Also exported: air_kerma_free_in_air, compute_ctdi_vol, compute_dlp, dose_report, muen_rho_air."
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 7. Reconstruction
                # ════════════════════════════════════════════════════════════════
                H2(:id => "reconstruction", :class => h2_cls, "Reconstruction"),
                P(
                    :class => prose_cls,
                    "Analytic FDK for a circular orbit, weighted FBP for a helical one, penalized ",
                    "iterative HIR (Hybrid IR), and two material-decomposition routes — K-channel in the ",
                    "projection domain and Ding-calibrated in the image domain.  The volume reconstructors ",
                    "share a single ", Code(:class => inline, "reconstruct!"),
                    " entry point that dispatches on workspace type."
                ),

                # ── 7a. FDK ──────────────────────────────────────────────────
                H3(:id => "recon-fdk", :class => h3_cls, "FDK — filtered back-projection"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_fdk_recon_workspace(sino, geom, matrix_size; filter = :standard) → FDKReconWorkspace"),
                    P(
                        :class => prose_cls,
                        "Pre-allocates the FBP-domain filter kernel + back-projection buffers.  ",
                        Code(:class => inline, "filter"),
                        " takes a Symbol from the preset table below or a ", Code(:class => inline, "CustomFilter"), "."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "grid_bandlimit(geom, matrix_size; ray_spacing = geom.pixel_size) → Float64   |   frequency_window(filter, f) → window at f ∈ [0, 1]"),
                    P(
                        :class => prose_cls,
                        "Since 0.17 every FBP kernel is bandlimited to the reconstruction grid.  ",
                        Code(:class => inline, "grid_bandlimit"), " is ", Code(:class => inline, "min(1, Δ_ray / Δ_grid)"),
                        " — the fraction of the ray sampling's Nyquist that a grid of ", Code(:class => inline, "matrix_size"),
                        " over the geometry's field of view can represent — and ", Code(:class => inline, "fdk_reconstruct"),
                        ", the helical chain and both reconstruction workspaces derive it themselves.  The apodization ",
                        "window (", Code(:class => inline, "frequency_window"),
                        ": Ram-Lak, Shepp-Logan, cosine, Hamming, Hann, and the CatSim standard / soft / bone control points) ",
                        "is stretched over that band and the response is zero above it, so a named kernel means the same ",
                        "resolution on any detector, and a detector finer than the grid — a 0.30 mm photon-counting column on ",
                        "a 0.68 mm grid passed 2.3× the grid's Nyquist before — no longer folds that band back into the image ",
                        "as fine grain.  On a detector no finer than the grid nothing changes.  ",
                        Code(:class => inline, "create_spatial_kernel(n, filter, Δ; bandlimit)"), " and ",
                        Code(:class => inline, "filter_sinogram!(…; bandlimit)"), " take it explicitly for callers who build their own chain."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "reconstruct!(ws::FDKReconWorkspace, sino, geom) → ws.volume"),
                    P(
                        :class => prose_cls,
                        "Runs filtered back-projection in-place.  Returns the μ-domain volume; ",
                        "convert to HU via ", Code(:class => inline, "to_hounsfield"),
                        " (§8).  Filter, kernel size, and output volume size are locked at workspace creation."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """ws_fdk  = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size; filter = :standard)
                            recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom))
                            recon_HU = BS.to_hounsfield(recon_μ; μ_water = μ_water_120)"""
                        )
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "CustomFilter(control_x, control_y) <: FilterType"),
                    P(
                        :class => prose_cls,
                        "User-defined frequency-domain apodization.  Two same-length tuples of (normalized ",
                        "frequency, gain).  Linear interpolation between control points.  Use it when ",
                        "the named presets don't quite match your scanner's vendor filter."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """fdk_filter = BS.CustomFilter(
                                (0.0, 0.25, 0.5,  0.75, 1.0),
                                (1.0, 0.75, 0.6,  0.2,  0.001),
                            )
                            ws = BS.create_fdk_recon_workspace(sino, geom, matrix_size; filter = fdk_filter)"""
                        )
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "Filter symbol presets"),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "symbol"), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, ":ram_lak"), Td(:class => td_cls, "Pure ramp — sharpest, noisiest.")),
                        Tr(Td(:class => td_mono, ":shepp_logan"), Td(:class => td_cls, "Ramp × sinc; classic medical default.")),
                        Tr(Td(:class => td_mono, ":cosine"), Td(:class => td_cls, "Mild low-pass.")),
                        Tr(Td(:class => td_mono, ":standard"), Td(:class => td_cls, "Standard apodization (close to clinical).")),
                        Tr(Td(:class => td_mono, ":bone"), Td(:class => td_cls, "Sharper kernel for bone / high-contrast tasks.")),
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_fov_mask!(volume, geom; sentinel_μ = -1024)"),
                    P(
                        :class => prose_cls,
                        "Zeros out voxels outside the recon's inscribed circular FOV (replaces them with ",
                        Code(:class => inline, "sentinel_μ"),
                        " — typically −1024 HU = air).  Run after FBP to clean up corner ringing."
                    ),
                ),

                # ── 7a′. Helical WFBP ────────────────────────────────────────
                H3(:id => "recon-helical", :class => h3_cls, "Helical WFBP"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "wfbp_helical_reconstruct(sino, geom, matrix_size; filter, helical_q = 0.7, coverage = nothing, mask_fov = true)"),
                    P(
                        :class => prose_cls,
                        "Weighted filtered back-projection for a helical trajectory (Stierstorfer 2004).  ",
                        "A geometry built with a ", Code(:class => inline, "pitch"), " is helical, and ",
                        Code(:class => inline, "fdk_reconstruct"), " routes it here on its own — the entry ",
                        "point is only needed to reach the helical keywords directly.  Arc detectors get the ",
                        "cylindrical row mapping and flat ones the planar 1/cos γ form, so a panel's shape ",
                        "is honoured rather than assumed."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "keyword"), Th(:class => th_cls, "default"), Th(:class => th_cls, "")),
                        Tr(Td(:class => td_mono, "helical_q"), Td(:class => td_cls, "0.7"), Td(:class => td_cls, "Width of the Stierstorfer view weighting, in half-turns.")),
                        Tr(Td(:class => td_mono, "coverage"), Td(:class => td_cls, "nothing"), Td(:class => td_cls, "Pre-allocated volume filled with the fraction of conjugate families that found data, 0–1. A voxel below 1 was reconstructed from an incomplete helix.")),
                        Tr(Td(:class => td_mono, "mask_fov"), Td(:class => td_cls, "true"), Td(:class => td_cls, "Replace voxels outside the inscribed circle with the air sentinel, as the axial path does.")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """geom = BS.CTGeometry(scanner; n_angles = 2000, fov_cm = 35.0, z_cm = 10.0, pitch = 1.0)
                            cover = zeros(Float32, matrix_size)
                            vol   = BS.fdk_reconstruct(sino, geom, matrix_size; coverage = cover)
                            all(≈(1), cover) || @warn "the requested volume is longer than the helix covers""""
                        )
                    ),
                    P(:class => note_cls, "An odd number of views per rotation misaligns the conjugate families and is warned about."),
                ),

                # ── 7b. Hybrid IR ────────────────────────────────────────────
                H3(:id => "recon-hir", :class => h3_cls, "Hybrid IR — penalized iterative"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_hir_recon_workspace(sino, geom, matrix_size; strength = 60, filter = :standard) → HIRReconWorkspace"),
                    P(
                        :class => prose_cls,
                        "Penalized weighted least-squares iterative recon (Huber prior + ordered subsets).  ",
                        Code(:class => inline, "strength"),
                        " is a percentage in 10 % steps: 0 = pure FDK pass-through, 60 = standard clinical, ",
                        "100 = heaviest regularization (it reads like the GE ASIR-V dial).  ",
                        "Workspace warm-starts from FDK, then runs the iterations."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "reconstruct!(ws::HIRReconWorkspace, sino, geom; init_volume=nothing, air_reference=nothing, weights=:transmission) → ws.volume"),
                    P(
                        :class => prose_cls,
                        "Same dispatch shape as FDK.  Converges in 2–5 iterations on clinical data; ",
                        "costs a few FDK passes — the workspace is built once and reused.  ",
                        Code(:class => inline, "weights"), " says what the sinogram is: ",
                        Code(:class => inline, ":transmission"), " (default) is the ", Code(:class => inline, "exp(−y)"),
                        " Poisson heuristic, right for any log-transmission — energy-integrating, photon-counting summed ",
                        "bins, either tube voltage of a dual-energy pair; ", Code(:class => inline, ":uniform"),
                        " leaves only the geometric ray-length normalisation; an array is the caller's per-ray statistical ",
                        "weight, which is what a material-basis sinogram needs (its inverse variance from the decomposition), ",
                        "since g/cm² is not a transmission and exp(−y) of it means nothing."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """ws_hir  = BS.create_hir_recon_workspace(sino, geom, matrix_size; strength = 60, filter = :standard)
                            recon_μ = Array(BS.reconstruct!(ws_hir, sino, geom))"""
                        )
                    ),
                    P(:class => note_cls, "see notebook 02 for FDK vs HIR side-by-side on XCAT"),
                ),

                # ── 7c. VMI ──────────────────────────────────────────────────
                H3(:id => "recon-nchannel", :class => h3_cls, "K-channel decomposition — projection domain"),
                P(
                    :class => prose_cls,
                    "The projection-domain route: decompose the measured channels into iodine and water ",
                    "line integrals (g/cm²) under the Poisson likelihood of the forward model that produced ",
                    "them, then reconstruct those and synthesize monoenergetic images from the pair.  ",
                    "Because the decomposition happens before reconstruction it is free of the beam ",
                    "hardening the image-domain route has to calibrate around, and it takes any number of ",
                    "channels — photon-counting bins, several kVp acquisitions, or both."
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "vmi_pipeline(; channels, basis, geom, to_backend = identity, kwargs...)"),
                    P(
                        :class => prose_cls,
                        "The whole chain, one call: ",
                        Code(:class => inline, "merge_channels → reduce_detector_rows → decompose → tlbf_denoise → FBP → ACNR → VMI"),
                        ".  Every stage's settings are keywords, so nothing is decided that the caller ",
                        "cannot see and override — which is what makes it the inner call of an ablation sweep.  ",
                        "Returns ", Code(:class => inline, "(vmis, energies, images = (water, iodine), quality, elapsed_s, settings)"), "."
                    ),
                    Table(
                        :class => table_cls,
                        Tr(Th(:class => th_cls, "stage"), Th(:class => th_cls, "keywords")),
                        Tr(Td(:class => td_mono, "decomposition"), Td(:class => td_cls, "method (:nchannel | :cong), controls::NChannelControls, merge_groups, tile_views")),
                        Tr(Td(:class => td_mono, "detector rows"), Td(:class => td_cls, "reduce_rows, rows — summed in counts for a z-invariant object; otherwise every row is kept and reconstructed slice by slice")),
                        Tr(Td(:class => td_mono, "T-LBF"), Td(:class => td_cls, "use_tlbf, tlbf_alpha1, tlbf_alpha2, tlbf_radius")),
                        Tr(Td(:class => td_mono, "ACNR"), Td(:class => td_cls, "use_acnr, acnr_passes, acnr_beta_max, acnr_hp_sigma_px, acnr_window")),
                        Tr(Td(:class => td_mono, "reconstruction"), Td(:class => td_cls, "matrix_size, fbp_filter, antialias, recon_rows, recon_method (:fbp | :hir), hir_strength, recon_projector, hir_reference_kev")),
                        Tr(Td(:class => td_mono, "synthesis"), Td(:class => td_cls, "vmi_energies = (40, 70, 100, 140)")),
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """basis  = BS.spectral_basis(ws_pcct)                 # response and air counts, from the workspace
                            result = BS.vmi_pipeline(;
                                channels = bin_sinograms, basis, geom,
                                to_backend = CuArray,
                                reduce_rows = true, use_tlbf = true,          # the published PCCT configuration
                                matrix_size = (512, 512, 1),
                            )
                            vmi_70 = result.vmis[:, :, 1, 2]              # (nx, ny, nz, energy)"""
                        )
                    ),
                    P(
                        :class => note_cls,
                        "matrix_size defaults to 512² by one slice whatever the workspace's ReconOptions says — this function never sees them."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "spectral_basis(; energies, response, I0)   |   spectral_basis(ws::PCCTWorkspace)   |   spectral_basis_from_bins   |   spectral_basis_from_acquisitions"),
                    P(
                        :class => prose_cls,
                        "The absolute per-channel response ", Code(:class => inline, "Φ(nc, nr, nE, K)"),
                        " and air counts ", Code(:class => inline, "I0(nc, nr, K)"),
                        " the estimator inverts, plus the iodine and water mass attenuations at those ",
                        "energies.  The constructor checks that Φ sums to I0 and refuses a basis that does not.  ",
                        "The ", Code(:class => inline, "PCCTWorkspace"),
                        " method reads a simulated detector's own response; ",
                        Code(:class => inline, "spectral_basis_from_acquisitions"),
                        " stacks several kVp acquisitions onto one shared energy grid."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "decompose_nchannel(; channels, basis, controls, to_backend, tile_views, keep_diagnostics = false)"),
                    P(
                        :class => prose_cls,
                        "The K-channel estimator over a full sinogram (K ≥ 2): a Newton solve per ray on the ",
                        "Poisson log-likelihood, with the Fisher information kept for the quality map.  ",
                        "Views are tiled so the backend never holds a whole sinogram of workspace.  ",
                        Code(:class => inline, "decompose_cong"),
                        " is the two-channel closed-form alternative, for comparison.  ",
                        "The returned ", Code(:class => inline, "quality"),
                        " names the fraction of rays that hit a bound, failed to converge, or were infeasible."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "tlbf_denoise(sino_iodine, sino_water, expected, measured; alpha1 = 0.9, alpha2 = 24.64, radius = 2)"),
                    P(
                        :class => prose_cls,
                        "Lee (2025) total-likelihood bilateral filter on the decomposed pair.  Each ",
                        "neighbour in the (column, view) window is weighted by how well it explains the ",
                        "centre ray's summed measured counts under the Poisson model, and one shared, ",
                        "normalised weight is applied to iodine and water alike, so the pair stays coherent.  ",
                        "Feed it ", Code(:class => inline, "total_expected_counts"), " and ",
                        Code(:class => inline, "total_measured_counts"),
                        ".  It needs counts, so it is photon-counting only, and a single detector row — ",
                        Code(:class => inline, "reduce_detector_rows"), " first."
                    ),
                    P(
                        :class => note_cls,
                        "alpha2 = Inf keeps the spatial weights alone and alpha2 ≤ 0 is the identity; both are the ablation endpoints."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "reconstruct_basis_slice(sino, geom, matrix_size; n_rows, antialias = true, method = :fbp, hir_weights, scale)   |   synthesize_vmi_stack(water, iodine, energies)"),
                    P(
                        :class => prose_cls,
                        "The two ends of the chain on their own, for when the pipeline is being assembled ",
                        "by hand.  ", Code(:class => inline, "antialias"),
                        " applies the deterministic angular response (",
                        Code(:class => inline, "angular_antialias_response"),
                        ") that keeps a sparse-view basis sinogram from streaking; the synthesis is the ",
                        "two-basis monoenergetic sum in HU.  ",
                        Code(:class => inline, "method = :hir"),
                        " reconstructs the pair with the penalized iterative reconstructor instead.  Three things ",
                        "make that sound on a basis sinogram: HIR's default ", Code(:class => inline, "exp(−y)"),
                        " weighting assumes a log-transmission, so each material is weighted by its own ",
                        "inverse variance from the decomposition's Fisher information (",
                        Code(:class => inline, "reconstruct!(…; weights)"), "); each is reconstructed in ",
                        "μ-equivalent units at ", Code(:class => inline, "hir_reference_kev"),
                        " so the Huber threshold and strength dial mean what they were tuned to; and a ",
                        "reduced-row sinogram goes through a one-row geometry rather than being repeated.  ",
                        "The same arm serves a dual-kVp pair decomposed through ",
                        Code(:class => inline, "spectral_basis_from_acquisitions"), "."
                    ),
                ),

                # ── 7d. Image-domain VMI ─────────────────────────────────────
                H3(:id => "recon-vmi", :class => h3_cls, "VMI — virtual monoenergetic + material decomposition (image domain)"),
                P(
                    :class => prose_cls,
                    "Image-domain dual-energy / spectral pipeline.  Pair of low/high reconstructed volumes → ",
                    "joint denoise → calibrated decomposition → per-energy VMI synthesis → noise-shaping post-processing."
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_rskr(volumes; n_iter = 2, h_param, radius = 2, γ = 0.5, gpu_arr_type) → Vector{Array}"),
                    P(
                        :class => prose_cls,
                        "Joint SVD + bilateral denoising on a μ-domain volume pair (Clark & Badea 2023).  ",
                        "Run before HU conversion — that's where the iodine/water noise is maximally ",
                        "anti-correlated.  ",
                        Code(:class => inline, "h_param"),
                        " controls the bilateral filter strength (lower = less smoothing)."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "fit_ding_coeffs(HU_low_vec, HU_high_vec, c_iodine_vec) → (coeffs, α_low, α_high, rms, pred_c)"),
                    P(
                        :class => prose_cls,
                        "Least-squares fit of the Ding (2012) image-domain decomposition coefficients ",
                        Code(:class => inline, "(a₀, a₁, a₂)"), " from a calibration table of measured rod HUs ",
                        "and known iodine concentrations.  Returns the coefficients plus per-bin α sensitivities ",
                        "(HU per mg/mL) and the RMS calibration error."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_ding_decomp(vol_low_HU, vol_high_HU, coeffs) → c_iodine"),
                    P(
                        :class => prose_cls,
                        "Per-voxel ", Code(:class => inline, "c_iodine = a₀ + a₁·HU_low + a₂·HU_high"),
                        " (mg/mL).  Pair with a 3-slice axial median filter (",
                        Code(:class => inline, "apply_median_z"),
                        ") to wipe single-voxel speckle without in-plane blur."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "synth_vmi_image_domain(HU_low, c_iodine; energy_keV, α_iod_low_cal) → HU_E"),
                    P(
                        :class => prose_cls,
                        "Synthesize a virtual monoenergetic image at any keV from the low-energy HU + ",
                        "iodine concentration map.  Implements ",
                        Code(:class => inline, "HU_E = HU_low + c_iodine·(α_E_phys − α_low_cal)"),
                        " — per-energy α from XrayAttenuation physics, low-bin α from the Ding fit."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "create_mono_plus_workspace(vol_template; n_energies) → MonoPlusWorkspace"),
                    P(
                        :class => prose_cls,
                        "Pre-allocates buffers for the Mono+ post-processing — a per-energy frequency-domain ",
                        "filter that anchors at the noise-quietest VMI energy and applies low-pass shaping ",
                        "to the others (FBP-equivalent noise behavior across the full keV sweep)."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_mono_plus!(ws, volumes, energies; E_noise_opt, σ_lp_px)"),
                    P(
                        :class => prose_cls,
                        "In-place Mono+ post-processing on a vector of synthesized VMIs.  ",
                        Code(:class => inline, "E_noise_opt"),
                        " is the anchor energy (passed through unchanged); ",
                        Code(:class => inline, "σ_lp_px"),
                        " is a per-energy vector of low-pass σ in pixels."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """vols_in = [vmi_dict[E] for E in energies]
                            σ_vec = Float64[2.0, 0.0, 2.0, 2.0]   # paired with energies
                            ws  = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(energies))
                            res = BS.apply_mono_plus!(ws, vols_in, energies; E_noise_opt = 70.0, σ_lp_px = σ_vec)"""
                        )
                    ),
                    P(:class => note_cls, "see notebooks 03 (dual-kVp) and 04 (PCCT) for full VMI pipelines"),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_median_z(volume; radius = 1) → Array"),
                    P(
                        :class => prose_cls,
                        "Axial-only median filter with the given z-radius (radius=1 → 3-slice window).  ",
                        "Wipes single-voxel xy impulses without any in-plane blur — perfect for cleaning up ",
                        "speckle in iodine-concentration maps."
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 8. Corrections & calibration
                # ════════════════════════════════════════════════════════════════
                H2(:id => "corrections", :class => h2_cls, "Corrections & calibration"),

                # ── 8a. HU conversion ────────────────────────────────────────
                H3(:id => "hu-conversion", :class => h3_cls, "HU conversion"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "to_hounsfield(μ_volume; μ_water) → Array{Float32, 3}"),
                    P(
                        :class => prose_cls,
                        "Per-voxel ", Code(:class => inline, "1000·(μ − μ_water)/μ_water"),
                        ".  Pass the polychromatic μ_water from §5 — using a single-energy μ_water ",
                        "with a polychromatic recon will land water HU off by tens of HU."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "μ_to_HU(μ, μ_water) → Float64    HU_to_μ(HU, μ_water) → Float64"),
                    P(
                        :class => prose_cls,
                        "Scalar conversions for one-off calculations (e.g. theoretical HU at a given energy ",
                        "from a material's μ).  No allocation, no array machinery."
                    ),
                ),

                # ── 8b. Noise floor & cupping ─────────────────────────────────
                H3(:id => "noise-cupping", :class => h3_cls, "Noise and cupping QA"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "add_system_noise_floor!(hu_volume, σ_HU; seed = 1234)"),
                    P(
                        :class => prose_cls,
                        "Legacy HU-domain additive Gaussian helper. Ordinary detector noise belongs in ",
                        "simulate!, which injects quantum and electronic/DAS noise in counts before the log. ",
                        "Reserve this helper for an explicitly modeled post-reconstruction residual."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "measure_radial_cupping(hu_volume; fov_cm) → QA metrics"),
                    P(
                        :class => prose_cls,
                        "Non-mutating radial-bias measurement on water-like voxels. Large residual cup or ",
                        "DC offset is evidence to fix the upstream spectrum, BHC, or coverage model."
                    ),
                ),

                # ── 8c. Beam hardening correction ─────────────────────────────
                H3(:id => "bhc", :class => h3_cls, "Beam-hardening correction"),
                P(
                    :class => prose_cls,
                    "The production path is knobless detected-spectrum water BHC: calibrate a per-column ",
                    "poly→mono model, apply it once in the sinogram domain, reconstruct, then convert with ",
                    "the calibration's μ_water_ref. The thresholded two-material/image-domain path is deprecated."
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "calibrate_bhc_water(sim_opts, protocol; scanner, geom) → WaterBHC"),
                    P(
                        :class => prose_cls,
                        "One-time per-column fit from the full detected spectrum (tube, filters, bowtie, heel, ",
                        "and detector efficiency). Use its ", Code(:class => inline, "μ_water_ref"),
                        " for the downstream HU conversion so calibration stays self-consistent."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_bhc_water(sino, model) → corrected sinogram"),
                    P(
                        :class => prose_cls,
                        "Sinogram-domain water-stage correction using the calibrated polynomial.  ",
                        "Run BEFORE FDK; removes the bulk polychromatic bias from the line integrals."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "Canonical water-BHC reconstruction pipeline"),
                    P(
                        :class => prose_cls,
                        "The older calibrate_bhc_two_material/apply_bhc_image_domain functions remain for ",
                        "compatibility only: thresholded bone segmentation can misclassify dense iodine, and ",
                        "the image-domain scaled self-subtraction deflates dense-material HU."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """# Calibrate once from the same detected spectrum as simulation
                            bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner = scanner, geom = geom)

                            # Apply once before reconstruction
                            sino_bhc = BS.apply_bhc_water(sino, bhc)
                            ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, geom, recon_opts.matrix_size)
                            recon_μ = Array(BS.reconstruct!(ws_fdk, sino_bhc, geom))
                            recon_HU = BS.to_hounsfield(recon_μ; μ_water = bhc.μ_water_ref)
                            qa = BS.measure_radial_cupping(recon_HU; fov_cm = recon_opts.fov_cm)"""
                        )
                    ),
                    P(:class => note_cls, "see notebook 02 for the full BHC pipeline on XCAT"),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 9. Detector physics
                # ════════════════════════════════════════════════════════════════
                H2(:id => "detector", :class => h2_cls, "Detector physics"),
                P(
                    :class => prose_cls,
                    "Both EICT (energy-integrating) and PCCT (photon-counting) detectors use Monte-Carlo-derived ",
                    "response models.  The EICT path uses lookup tables (DQE, quantum efficiency, fill factor), ",
                    "while the PCCT path uses the full per-energy detector response matrix (DRM) and pileup model."
                ),

                # ── 9a. EICT detector response ───────────────────────────────
                H3(:id => "detector-eict", :class => h3_cls, "EICT detector response"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "DetectorEfficiency, DetectorEfficiencyMode, BEER_LAMBERT, MC_LUT"),
                    P(
                        :class => prose_cls,
                        "Two efficiency modes — analytic Beer-Lambert through the scintillator depth, ",
                        "or MC-derived lookup tables for the GE GEMSTONE / standard GOS / CsI / CdTe ",
                        "scintillators.  Sim-time selection via ",
                        Code(:class => inline, "sim_opts.detector_efficiency_mode"), "."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "GEMSTONE_MC_EFFICIENCY_LUT  ::  Matrix{Float64}"),
                    P(
                        :class => prose_cls,
                        "Pre-computed MC efficiency per (energy, depth) for the GE GEMSTONE detector.  ",
                        "Used by ", Code(:class => inline, "MC_LUT"), " mode."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_dqe(detector_eff, energies, weights) → Float64"),
                    P(
                        :class => prose_cls,
                        "Detective quantum efficiency at the spectrum's mean — 0 = perfect, 1 = no detection.  ",
                        "Useful for theoretical noise-budget calculations against vendor spec sheets."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "quantum_efficiency(material, depth_mm, energies) → Vector{Float64}"),
                    P(
                        :class => prose_cls,
                        "Per-energy quantum efficiency ∈ [0, 1] for a given scintillator material + depth.  ",
                        "EICT analog of ", Code(:class => inline, "quantum_efficiency_vector"), " (PCCT)."
                    ),
                ),

                # ── 9b. PCCT detector response ───────────────────────────────
                H3(:id => "detector-pcct", :class => h3_cls, "PCCT detector response"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "_build_pcct_detector(scanner) → PhotonCountingDetector"),
                    P(
                        :class => prose_cls,
                        "Construct the PCCT detector struct directly from the scanner's PCCT fields ",
                        "(", Code(:class => inline, "energy_thresholds"), ", ",
                        Code(:class => inline, "energy_resolution"), ", ",
                        Code(:class => inline, "charge_sharing_fwhm"), ", ",
                        Code(:class => inline, "dead_time_ns"),
                        ").  Internal helper exposed because notebook 04's μ_water + scatter-fraction ",
                        "derivations need it."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_mc_drm(detector, kVp) → Matrix{Float64}  (n_E, n_bins)"),
                    P(
                        :class => prose_cls,
                        "Monte-Carlo-derived detector response matrix.  ",
                        Code(:class => inline, "R[E, b]"),
                        " = probability that a photon of energy ", Code(:class => inline, "E"),
                        " is recorded in bin ", Code(:class => inline, "b"),
                        " — captures charge sharing, fluorescence escape, anti-coincidence."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "quantum_efficiency_vector(material, thickness_mm, energies) → Vector{Float64}"),
                    P(
                        :class => prose_cls,
                        "Per-energy detection efficiency for the CdTe / CZT / Si crystals — Beer-Lambert ",
                        "through the active depth, includes K-edge and Fano weighting."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "apply_pcct_noise!(pcct_sino, detector, protocol; seed, I0, energies, weights, ...)"),
                    P(
                        :class => prose_cls,
                        "Per-bin Poisson noise applied AFTER scatter injection.  Uses the detector ",
                        "DRM to compute the right per-bin I0 from the source spectrum."
                    ),
                ),

                # ── 9c. Pileup ───────────────────────────────────────────────
                H3(:id => "detector-pileup", :class => h3_cls, "Pileup"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_mc_pileup_matrix(thresholds_keV, w_norm, energies, count_rate, τ_ns; n_trials, seed) → Matrix"),
                    P(
                        :class => prose_cls,
                        "Spectral migration matrix S[b_in, b_out] from MC simulation of dead-time pileup at ",
                        "the detector's per-dexel count rate.  Computed once at workspace creation; ",
                        "used during PCCT noise application to migrate counts between bins."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "seminonparalyzable_count_factor(aτ; f_retrigger = 0.3) → Float64"),
                    P(
                        :class => prose_cls,
                        "Closed-form count-loss factor for the semi-non-paralyzable dead-time model.  ",
                        "Matches the MC pileup result to within ~1% at ", Code(:class => inline, "aτ < 5"),
                        ".  ", Code(:class => inline, "aτ"), " is per-dexel count rate × dead-time."
                    ),
                ),

                # ── 9d. Scatter ──────────────────────────────────────────────
                H3(:id => "detector-scatter", :class => h3_cls, "Scatter"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "estimate_phantom_diameter_cm(mask, voxel_size_mm) → Float64"),
                    P(
                        :class => prose_cls,
                        "Equivalent body diameter of the scanned object — drives the scatter kernel ",
                        "amplitude (Ohnesorge size scaling, exponent 1.5).  Pass ",
                        Code(:class => inline, "phantom.voxel_size .* 10.0"),
                        " for the mm units."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "geometry_aware_scatter_model(scanner; phantom_diameter_cm) → ScatterModel"),
                    P(
                        :class => prose_cls,
                        "Builds the Ohnesorge spatial scatter convolution kernel with geometry scaling — ",
                        "accounts for SID/SDD, air gap, pixel pitch, and phantom size.  Pass to ",
                        Code(:class => inline, "estimate_scatter_field!"), "."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "estimate_scatter_field!(scatter_field, combined_sino, scatter_model)"),
                    P(
                        :class => prose_cls,
                        "In-place: predicts the scatter sinogram from the recombined primary line integral.  ",
                        "Output is the scatter field (counts), to subtract from the measurement after scaling ",
                        "by per-bin scatter fractions."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_scatter_energy_weights(energies) → Vector{Float64}"),
                    P(
                        :class => prose_cls,
                        "Per-energy scatter-to-primary spectral ratio.  Used by ",
                        Code(:class => inline, "compute_scatter_bin_weights"),
                        " to derive PCCT bin-specific scatter fractions."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "compute_scatter_bin_weights(energies, weights, ew, η_vec, R_mat, kVp) → Vector{Float64}"),
                    P(
                        :class => prose_cls,
                        "Per-bin fraction of total scatter that falls into each PCCT energy bin (",
                        Code(:class => inline, "Σ_b frac_b = 1"),
                        ").  Subtract ", Code(:class => inline, "scatter_field × I0_total × frac_b"),
                        " from each bin's measured counts to do exact scatter correction."
                    ),
                    P(:class => note_cls, "see notebook 04 §6 for the canonical PCCT scatter-correction flow"),
                ),

                # ════════════════════════════════════════════════════════════════
                # § 10. Constants & calibrations
                # ════════════════════════════════════════════════════════════════
                H2(:id => "constants", :class => h2_cls, "Constants & calibrations"),

                # ── 10a. Clinical calibration tables ──────────────────────────
                H3(:id => "clinical-cal", :class => h3_cls, "Clinical calibration tables"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "GE_REVOLUTION_APEX_ELITE_DE_CAL  ::  Dict{String, NamedTuple}"),
                    P(
                        :class => prose_cls,
                        "Pre-computed Gammex 472 rod HU table from a real GE Revolution Apex Elite ",
                        "dual-kVp Gemstone Spectral Imaging (GSI) acquisition (post-RSKR, post-cupping).  ",
                        "Feed into ", Code(:class => inline, "fit_ding_coeffs"), " for self-calibrated decomposition."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """cal = BS.GE_REVOLUTION_APEX_ELITE_DE_CAL
                            # cal["I 5.0 mg/mL"] = (material = :iodine, mg_per_mL = 5.0, HU_low = ..., HU_high = ...)"""
                        )
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "SIEMENS_NAEOTOM_ALPHA_140KVP_CAL  ::  Dict{String, NamedTuple}"),
                    P(
                        :class => prose_cls,
                        "Same shape but for the Siemens Naeotom Alpha PCCT scanner at 140 kVp / 174 mA / ",
                        "10 mGy CTDIvol.  Used in notebook 04 for self-calibrated Ding decomposition."
                    ),
                ),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "SIEMENS_NAEOTOM_ALPHA_120KVP_CAL  ::  Dict{String, NamedTuple}"),
                    P(
                        :class => prose_cls,
                        "Same shape, 120 kVp variant of the Naeotom Alpha calibration."
                    ),
                ),

                # ── 10b. Region labels ────────────────────────────────────────
                H3(:id => "region-labels", :class => h3_cls, "Region labels"),
                Div(
                    :class => card_cls,
                    Div(:class => sig_cls, "REGION_BACKGROUND, REGION_AIR, REGION_WATER, REGION_SOLID_WATER, REGION_CA_*, REGION_I_*"),
                    P(
                        :class => prose_cls,
                        "UInt8 enum constants used by ", Code(:class => inline, "create_gammex_472"),
                        " to label voxels in the mask.  Calcium inserts: ",
                        Code(:class => inline, "REGION_CA_50"), " through ",
                        Code(:class => inline, "REGION_CA_600"),
                        " (10 → 16); iodine inserts: ",
                        Code(:class => inline, "REGION_I_2_0"), " through ",
                        Code(:class => inline, "REGION_I_20_0"),
                        " (20 → 26).  Use ", Code(:class => inline, "BS.get_material(:Ca_100)"),
                        " etc. to resolve a Symbol to its `XA.Material` for forward projection."
                    ),
                    Pre(
                        :class => code_cls, Code(
                            :class => "language-julia", """# Filter for all calcium voxels
                            ca_mask = (UInt8(BS.REGION_CA_50)  .≤ phantom.mask) .&
                                      (phantom.mask .≤ UInt8(BS.REGION_CA_600))"""
                        )
                    ),
                ),

                # ════════════════════════════════════════════════════════════════
                # Footer pointer
                # ════════════════════════════════════════════════════════════════
                Div(
                    :class => "pt-8 border-t border-warm-200 dark:border-warm-800",
                    P(
                        :class => prose_cls,
                        "This page covers the surface that's actually used by the example notebooks.  ",
                        "The full export list lives in ",
                        Code(:class => inline, "src/BasisSimulator.jl"),
                        "; lower-level entry points (distance-driven and Siddon projectors, individual physics-effect ",
                        "constructors, internal solver workspaces) are documented in their respective ",
                        "source files."
                    ),
                ),
            )
        )
    end
end
