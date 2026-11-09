# Scanners — what a scanner is, one section per clinical constant (src/geometry/clinical_scanners.jl),
# dual-source scans, varying a scanner, the constructor defaults and catalog, and how to model a new
# scanner. Every keyword table is read from the constant (or constructor) when the page is built, so
# the page and the package cannot disagree. Prose strings put `backticked` text in code.

import BasisSimulator

let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
        BS = BasisSimulator

        prose = "text-warm-600 dark:text-warm-400 leading-relaxed text-[17px]"
        small = "text-[15px] text-warm-600 dark:text-warm-400 leading-relaxed"
        h2_cls = "text-3xl font-serif font-semibold text-warm-900 dark:text-warm-100 scroll-mt-24 pt-8"
        h3_cls = "no-rule text-xl font-serif font-semibold text-warm-900 dark:text-warm-100 pt-2 scroll-mt-24"
        inline = "font-mono text-[0.85em] px-1 py-0.5 rounded bg-warm-200/70 dark:bg-warm-800/70 text-accent-700 dark:text-accent-300"
        linkc = "text-accent-600 dark:text-accent-400 hover:underline"

        c(x) = Code(:class => inline, x)
        b(x) = Strong(:class => "text-warm-900 dark:text-warm-100", x)
        link(href, x) = A(:href => href, :class => linkc, x)
        # a string's `backticked` spans become code; other pieces pass through
        rich(x) = x isa AbstractString ?
            [isodd(i) ? s : c(s) for (i, s) in enumerate(split(x, '`'))] :
            [x]
        p(xs...) = P(:class => prose, collect(Iterators.flatten(rich.(xs)))...)
        ps(xs...) = P(:class => small, collect(Iterators.flatten(rich.(xs)))...)
        julia(code) = CodeBlock(code)
        steps(items...) = Ol(
            :class => "bs-steps space-y-3 " * prose,
            [Li(collect(Iterators.flatten(rich.(item)))...) for item in items]...,
        )
        table(head, rows) = Div(
            :class => "bs-table-wrap",
            Table(
                :class => "bs-table",
                Thead(Tr([Th(h) for h in head]...)),
                Tbody([Tr([Td(rich(x)...) for x in row]...) for row in rows]...),
            ),
        )
        nb(slug, label) = link("$(BASE)/examples/$(slug)/", label)
        function notebooks(items...)
            xs = Any[b("Notebooks. ")]
            for (i, (slug, label)) in enumerate(items)
                i > 1 && push!(xs, ", ")
                push!(xs, nb(slug, label))
            end
            return P(:class => prose, xs..., ".")
        end

        # A scanner's keywords, read from the scanner itself (the one source of truth)
        shown(x) = repr(x; context = :compact => true)
        param_table(s) = table(
            ["Keyword", "Value"],
            [[c(String(f)), c(shown(getproperty(s, f)))] for f in propertynames(s)],
        )
        # The keywords in which the variant `v` differs from the constant `s`
        diff_table(s, v, name_s, name_v) = table(
            ["Keyword", name_s, name_v],
            [
                [c(String(f)), c(shown(getproperty(s, f))), c(shown(getproperty(v, f)))]
                for f in propertynames(s) if getproperty(s, f) != getproperty(v, f)
            ],
        )
        # A constructor's defaults, read from a scanner built from them; each row names its keywords
        default_value(s, f) = f === :energy_thresholds ? "required" : shown(getproperty(s, f))
        default_table(s, rows) = table(
            ["Keyword", "Default", "Meaning"],
            [
                [c(join(String.(fs), " / ")), c(join([default_value(s, f) for f in fs], " / ")), meaning]
                for (fs, meaning) in rows
            ],
        )

        eict_defaults = BS.EICTScanner()
        pcct_defaults = BS.PCCTScanner(; energy_thresholds = [20.0])
        geometry_rows = [
            ((:source_to_isocenter, :source_to_detector), "distances, mm"),
            ((:detector_rows, :detector_cols), "the physical detector array"),
            ((:detector_row_size, :detector_col_size), "element size at the isocentre, mm"),
            ((:detector_row_offset, :detector_col_offset), "offsets, rows and columns (0.25: quarter-detector)"),
            ((:detector_shape,), "`:arc` (equiangular) or `:flat`"),
            ((:focal_spot_width, :focal_spot_length), "focal spot, mm"),
            ((:target_angle,), "anode, degrees: heel effect and spectrum (IPEM table at 8° or 10°)"),
            ((:gantry_rotation_time, :scan_diameter, :gantry_aperture), "descriptive: the protocol sets the rotation time"),
            ((:flat_filter_material, :flat_filter_thickness), "`:aluminum`, `:copper` or `:titanium`; mm"),
            ((:bowtie_filter,), "from the catalog below"),
        ]
        eict_rows = [
            ((:detector_material, :detector_depth), "scintillator and depth, mm: its Geant4 table's"),
            ((:fill_factor_row, :fill_factor_col), "active-area fraction, in (0, 1]"),
            ((:detection_gain, :electronic_noise), "electrons per keV; DAS noise σ, electrons"),
            ((:use_fill_factor, :use_detector_efficiency), "the fill factor; the scintillator's energy response (off: every photon absorbed whole)"),
            ((:detector_efficiency_mode,), "that response: `:mc_lut` (the Geant4 table) or `:beer_lambert` (analytic, any depth)"),
            ((:use_optical_crosstalk, :use_lag), "light spread to the neighbouring dexels; the scintillator's afterglow from view to view"),
            ((:gain_spread, :offset_spread_keV, :defect_seed), "dexel defects; corrected by `calibrate`'s flat field"),
            ((:scatter_kernels, :scatter_correction), "scatter library entry (`\"\"` = none); subtract the estimate"),
        ]
        pcct_rows = [
            ((:energy_thresholds,), "keV, ascending, among the response table's 16"),
            ((:detector_material, :detector_depth), "`:CdTe`, `:CZT` or `:Si`; mm"),
            ((:dead_time_ns, :pileup), "per native dexel; corrected by `calibrate`'s pile-up correction"),
            ((:binning_factor,), "native dexels summed per pixel side (2 standard, 1 UHR)"),
            ((:gain_spread, :threshold_spread_keV, :defect_seed), "native-dexel defects; corrected by `calibrate`'s flat field"),
            ((:scatter_kernels, :scatter_correction), "as for `EICTScanner`"),
            ((:noise_reduction,), "0 = Poisson counts … 1 = expected counts; leave at 0"),
        ]
        # every keyword of each family is in its tables: a new field fails the build until it is
        for (s, rows) in ((eict_defaults, [geometry_rows; eict_rows]), (pcct_defaults, [geometry_rows; pcct_rows]))
            missing_keywords = setdiff(propertynames(s), Iterators.flatten(first.(rows)))
            isempty(missing_keywords) || error("scanners.jl: the defaults tables miss $(missing_keywords)")
        end

        sections = [
            ("overview", "What a Scanner Is"),
            ("ge-revolution-apex-elite", "1 · GE Revolution Apex Elite"),
            ("siemens-naeotom-alpha", "2 · Siemens NAEOTOM Alpha"),
            ("somatom-force", "3 · Siemens SOMATOM Force"),
            ("somatom-definition-flash", "4 · Siemens SOMATOM Definition Flash"),
            ("dual-source", "Dual-Source Scans"),
            ("varying", "Varying a Scanner"),
            ("defaults", "Constructor Defaults"),
            ("catalog", "Detectors, Bowties and Filters"),
            ("new-scanner", "Modelling a New Scanner"),
        ]

        PageWithTOC(
            sections,
            Div(
                :class => "max-w-3xl mx-auto space-y-6",
                Div(
                    :class => "space-y-5",
                    Div(
                        :class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500",
                        RawHtml("""<span style="color:var(--color-accent-500)">●</span>&nbsp; Hardware reference"""),
                    ),
                    H1(
                        :class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl leading-[1.05] tracking-tight",
                        "Scanners",
                    ),
                ),

                # ── What a scanner is ────────────────────────────────────────────
                H2(:id => "overview", :class => h2_cls, "What a Scanner Is"),
                p("""
                  A scanner is one struct: `EICTScanner` (energy-integrating, a scintillator) or `PCCTScanner` \
                  (photon-counting, a direct-conversion sensor). The clinical systems are documented constants of \
                  them, each value sourced in the docstring (`?BS.SOMATOM_FORCE`). Start from one and vary it with \
                  the copy constructor, which copies every parameter and replaces the keywords you give:"""),
                julia(raw"""import BasisSimulator as BS

scanner = BS.SOMATOM_FORCE # a clinical scanner, an EICTScanner
scanner.detector_rows # 96: every keyword reads straight off the scanner
scanner_narrow = BS.EICTScanner(BS.SOMATOM_FORCE; detector_rows = 48) # a variant"""),
                p("""
                  There are no presets and no mode keywords: a mode (the Alpha's UHR) or a second tube is its own \
                  constant, and a dual-source or dual-kVp scan is two ordinary (scanner, protocol) acquisitions. \
                  kVp, mA, views, collimation, pitch and added filters belong to the `CTProtocol`."""),

                # ── 1 · GE Revolution Apex Elite ─────────────────────────────────
                H2(:id => "ge-revolution-apex-elite", :class => h2_cls, "1 · GE Revolution Apex Elite"),
                p("""
                  GE's 256-row wide-cone energy-integrating scanner, with a curved Gemstone garnet detector. The \
                  rows (160 mm) are FDA K133705 and K213715; the columns, distances and tube are unpublished, the \
                  notebooks' values. The 3.0 mm garnet (its Geant4 table `GEMSTONE_MC_EFFICIENCY_LUT`), fill factor \
                  and gain are assumptions; the 3500 e⁻ electronic noise is the clinical DAS readout, which the VMI \
                  notebooks turn off. Rapid kVp switching is two protocols on it (notebook 03)."""),
                julia(raw"""scanner = BS.REVOLUTION_APEX
scanner = BS.EICTScanner(BS.REVOLUTION_APEX; electronic_noise = 0.0) # VMI notebooks 03, 06, 07"""),
                param_table(BS.REVOLUTION_APEX),
                notebooks(
                    ("01_five_struct_api", "01"),
                    ("02_xcat_custom_materials", "02"),
                    ("03_dual_kvp_switching_vmi", "03"),
                    ("05_xcat_grid_to_recon", "05"),
                    ("06_catsim_vs_basissim", "06"),
                    ("07_qrm_thorax_pure_material_vmi", "07"),
                    ("10_titanium_implant", "10"),
                    ("11_helical_scanning", "11"),
                    ("mc-06_ge_revolution_apex", "MC 06 (its detector in Geant4)"),
                ),

                # ── 2 · Siemens NAEOTOM Alpha ────────────────────────────────────
                H2(:id => "siemens-naeotom-alpha", :class => h2_cls, "2 · Siemens NAEOTOM Alpha"),
                p(
                    """
                    Siemens's photon-counting scanner, detector A in its standard mode: native CdTe dexels summed \
                    2 × 2 after counting. Geometry: Li 2025, Hennemann 2026, FDA K243523; thresholds: Sartoretti \
                    2023; the CdTe response and 27 ns dead time: Geant4, BasisMonteCarlo notebook 03. It is \
                    realistic: its dexels' gains spread by 2 % and thresholds by 0.35 keV (an assumption), corrected \
                    as the scanner does: `calibrate`, passed to `create_workspace`, with which `simulate!` \
                    corrects pile-up, flat field, then scatter (""",
                    link("$(BASE)/getting-started/#spectral", "Getting Started"),
                    ").",
                ),
                julia(raw"""scanner = BS.NAEOTOM_ALPHA"""),
                param_table(BS.NAEOTOM_ALPHA),
                notebooks(
                    ("04_pcct_vmi", "04"),
                    ("08_qrm_thorax_pure_material_pcct", "08"),
                    ("mc-03_alpha_detector", "MC 03 (its detector in Geant4)"),
                ),
                H3(:id => "naeotom-alpha-uhr", :class => h3_cls, "NAEOTOM_ALPHA_UHR"),
                p("""
                  The ultra-high-resolution mode, its own constant: every native dexel read on its own, at one \
                  threshold. Every other keyword, defects and kernels included, is the standard mode's. No notebook \
                  runs it yet."""),
                julia(raw"""scanner = BS.NAEOTOM_ALPHA_UHR"""),
                diff_table(BS.NAEOTOM_ALPHA, BS.NAEOTOM_ALPHA_UHR, "NAEOTOM_ALPHA", "NAEOTOM_ALPHA_UHR"),

                # ── 3 · Siemens SOMATOM Force ────────────────────────────────────
                H2(:id => "somatom-force", :class => h2_cls, "3 · Siemens SOMATOM Force"),
                p("""
                  Siemens's third-generation dual source, tube/detector A, with a UFC Gd₂O₂S detector (Geant4 table \
                  `UFC_MC_EFFICIENCY_LUT`, BasisMonteCarlo notebook 04). Rows and channels: Force white paper §1.2, \
                  FDA K133589. The distances, tube, bowtie, 1.4 mm crystal and zero electronic noise are notebook \
                  09's assumptions; the protocol adds the tube's 0.9 mm titanium window. The spectra stop at 140 \
                  kVp, so the clinical x/Sn150 pair runs as 100/Sn140; cross-scatter is not modelled."""),
                julia(raw"""scanner = BS.SOMATOM_FORCE"""),
                param_table(BS.SOMATOM_FORCE),
                notebooks(
                    ("09_siemens_force_ufc_dual_source_vmi", "09"),
                    ("mc-04_siemens_force", "MC 04 (its detector in Geant4)"),
                ),
                H3(:id => "somatom-force-b", :class => h3_cls, "SOMATOM_FORCE_B"),
                p("""
                  Tube/detector B: 640 channels and a 35.5 cm field (Force white paper; Wang et al. 2021). Tube B's \
                  95° and detector B's 0.88 mm z offset are not scanner parameters: each tube is its own acquisition."""),
                julia(raw"""scanner = BS.SOMATOM_FORCE_B"""),
                diff_table(BS.SOMATOM_FORCE, BS.SOMATOM_FORCE_B, "SOMATOM_FORCE", "SOMATOM_FORCE_B"),

                # ── 4 · Siemens SOMATOM Definition Flash ─────────────────────────
                H2(:id => "somatom-definition-flash", :class => h2_cls, "4 · Siemens SOMATOM Definition Flash"),
                p("""
                  Siemens's second-generation dual source (2008), tube/detector A, with a UFC detector of its own \
                  Geant4 table (`:ufc_flash`, BasisMonteCarlo notebook 05; within 0.5 % of the Force's). Rows and \
                  channels: FDA K133589, Siemens datasheet (2010); distances: the DICOM-CT-PD geometry of the AAPM \
                  low-dose CT projection data; anode, filtration and bore: the datasheet. Assumptions: the bowtie, \
                  the filter as pure aluminium, the 1.4 mm crystal and fill factor, 1500 e⁻ of electronic noise (the \
                  Stellar detector) and no tube-B z offset."""),
                julia(raw"""scanner = BS.SOMATOM_DEFINITION_FLASH"""),
                param_table(BS.SOMATOM_DEFINITION_FLASH),
                notebooks(
                    ("12_siemens_flash_ufc", "12"),
                    ("mc-05_siemens_flash", "MC 05 (its detector in Geant4)"),
                ),
                H3(:id => "somatom-definition-flash-b", :class => h3_cls, "SOMATOM_DEFINITION_FLASH_B"),
                p("""
                  Tube/detector B: 480 channels and a 33.4 cm field (Siemens datasheet), 95° from tube A (Petersilka \
                  et al. 2008), behind the 0.4 mm tin Selective Photon Shield (Primak et al. 2010) on its protocol."""),
                julia(raw"""scanner = BS.SOMATOM_DEFINITION_FLASH_B"""),
                diff_table(
                    BS.SOMATOM_DEFINITION_FLASH,
                    BS.SOMATOM_DEFINITION_FLASH_B,
                    "SOMATOM_DEFINITION_FLASH",
                    "SOMATOM_DEFINITION_FLASH_B",
                ),
                ps("""
                   Sources: FDA 510(k) K082220, K113342, K121072, K122471, K133589, K173630, K230421 · Siemens \
                   datasheet (Dec 2010) and brochure (2016) · NHS Supply Chain CT scanner attributes · DICOM-CT-PD \
                   User Manual v3, TCIA LDCT-and-Projection-data · Petersilka et al., Eur J Radiol 2008;68:362 · \
                   Flohr et al., Med Phys 2009;36:5641 · Primak et al., AJR 2010 · Schardt et al., Med Phys \
                   2004;31:2699."""),

                # ── Dual-source scans ────────────────────────────────────────────
                H2(:id => "dual-source", :class => h2_cls, "Dual-Source Scans"),
                p(
                    """
                    A dual-energy scan is two ordinary acquisitions, one per tube, each with its own seed: tube A \
                    with `SOMATOM_FORCE`, tube B with `SOMATOM_FORCE_B`. With the `phantom` and `recon` of """,
                    link("$(BASE)/getting-started/#first-scan", "Your First Scan"),
                    ":",
                ),
                julia(raw"""protocol_low = BS.CTProtocol(
    kVp = 100,
    mA = 380.0,
    views = 1160,
    rotation_time = 0.5,
    collimation_mm = 4.8,
    additional_filters = [("Ti", 0.9)],
)
protocol_high = BS.CTProtocol(protocol_low; kVp = 140, mA = 190.0, additional_filters = [("Ti", 0.9), ("Sn", 0.6)])

tubes = [
    (scanner = BS.SOMATOM_FORCE, protocol = protocol_low, sim_opts = BS.SimOptions(seed = 1)),
    (scanner = BS.SOMATOM_FORCE_B, protocol = protocol_high, sim_opts = BS.SimOptions(seed = 2)),
]
sinograms = map(tubes) do tube
    ws = BS.create_workspace(tube.scanner, tube.protocol, tube.sim_opts, recon, phantom)
    BS.simulate!(ws, phantom, tube.protocol, tube.sim_opts)
    sinogram = Array(ws.sinogram)
    BS.release_backend!(ws)
    return sinogram
end"""),
                p("""
                  `spectral_basis_from_acquisitions` combines them (notebook 09). The Flash is the same with \
                  `SOMATOM_DEFINITION_FLASH` and `SOMATOM_DEFINITION_FLASH_B` behind `[("Sn", 0.4)]` (notebook 12). \
                  To co-register the pair ray by ray, both notebooks run tube B on tube A's arc, with the phantom \
                  inside detector B's field."""),

                # ── Varying a scanner ────────────────────────────────────────────
                H2(:id => "varying", :class => h2_cls, "Varying a Scanner"),
                julia(raw"""scanner = BS.EICTScanner(BS.SOMATOM_FORCE; detector_rows = 16) # narrower, for speed
scanner = BS.PCCTScanner(BS.NAEOTOM_ALPHA; gain_spread = 0, threshold_spread_keV = 0) # an ideal Alpha
scanner = BS.PCCTScanner(BS.NAEOTOM_ALPHA; pileup = false) # pile-up off
scanner = BS.EICTScanner(BS.SOMATOM_FORCE; scatter_correction = false) # scatter left in the data"""),
                p("""
                  An ideal Alpha is a variant, not a mode: the same constant with its defects at zero. A variant \
                  keeps the constant's scatter kernels, which are made for its two distances, so keep those (notebooks \
                  10 and 11 change only the Apex's detector). To simulate no scatter at all, use \
                  `SimOptions(use_scatter = false)`."""),

                # ── Constructor defaults ─────────────────────────────────────────
                H2(:id => "defaults", :class => h2_cls, "Constructor Defaults"),
                p("""
                  The defaults are a CatSim-like research scanner, for building a scanner no constant models; the \
                  notebooks start from constants. `PCCTScanner` also needs `energy_thresholds`. A scanner built from \
                  the defaults has `scatter_kernels = ""`, so it needs `SimOptions(use_scatter = false)` or a library \
                  entry (a key of `SCATTER_KERNEL_ARTIFACTS`)."""),
                H3(:class => h3_cls, "Geometry, Both Families"),
                default_table(eict_defaults, geometry_rows),
                H3(:class => h3_cls, "EICTScanner"),
                default_table(eict_defaults, eict_rows),
                H3(:class => h3_cls, "PCCTScanner"),
                default_table(pcct_defaults, pcct_rows),

                # ── Catalog ──────────────────────────────────────────────────────
                H2(:id => "catalog", :class => h2_cls, "Detectors, Bowties and Filters"),
                table(
                    ["detector_material", "Scintillator", "Geant4 table", "Depth"],
                    [
                        ["`:lumex`", "GE Gemstone garnet", "`GEMSTONE_MC_EFFICIENCY_LUT`", "3.0 mm"],
                        ["`:ufc`", "Siemens UFC Gd₂O₂S (Force)", "`UFC_MC_EFFICIENCY_LUT`", "1.4 mm"],
                        ["`:ufc_flash`", "Siemens UFC Gd₂O₂S (Flash)", "`UFC_FLASH_MC_EFFICIENCY_LUT`", "1.4 mm"],
                    ],
                ),
                ps("""
                   Each table is made at one depth; `EICTScanner(s; detector_efficiency_mode = :beer_lambert)` models any \
                   depth analytically. Photon counters (`:CdTe`, `:CZT`, `:Si`) read the Alpha's Geant4 CdTe response \
                   at the native dexels. Bowties: `:large_body`, `:medium_body`, `:small_body` (CatSim/XCIST; also \
                   `:ge_revolution_large`, …), `:head`, `:none`. Per-scan filters go on the protocol, \
                   `CTProtocol(additional_filters = [("Ti", 0.9), ("Sn", 0.6)])`, in Al, Cu, Sn, Ti, C or W."""),

                # ── Modelling a new scanner ──────────────────────────────────────
                H2(:id => "new-scanner", :class => h2_cls, "Modelling a New Scanner"),
                steps(
                    (b("Start from the nearest constant "), "of the same family, ideally at the same distances."),
                    (
                        b("Name its kernels. "),
                        "Keep the constant's if the distances match, name another entry of ",
                        "`SCATTER_KERNEL_ARTIFACTS`, or make one with BasisMonteCarlo's ",
                        nb("mc-10_scatter_library", "scatter library"),
                        "; until then, `use_scatter = false`.",
                    ),
                    (
                        b("Pick its detector table: "),
                        "a `detector_material` above at its depth, or a new Geant4 run; a photon counter's ",
                        "thresholds must be among the CdTe table's.",
                    ),
                    (
                        b("Write it as a documented const with sources, "),
                        "each unpublished value marked as an assumption, in `src/geometry/clinical_scanners.jl`; ",
                        "a second tube or mode is a second const.",
                    ),
                ),
                p("""
                  Then check `detector_cols × detector_col_size` covers the body, and use \
                  `validate_protocol(protocol, scanner)` and `required_axial_detector_rows(scanner; fov_cm, z_cm)`."""),
            ),
        )
    end
end
