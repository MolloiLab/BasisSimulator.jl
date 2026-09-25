# Getting started — install, the five structs, reconstruction, dose, helical, spectral/VMI.
# Every listing on this page runs as written against the current package (CPU, small sizes).

let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
        prose  = "text-warm-600 dark:text-warm-400 leading-relaxed"
        small  = "text-sm text-warm-600 dark:text-warm-400 leading-relaxed"
        h2_cls = "text-2xl font-serif font-semibold text-warm-900 dark:text-warm-100 scroll-mt-24"
        inline = "font-mono text-[0.85em] px-1 py-0.5 rounded bg-warm-200/70 dark:bg-warm-800/70 text-accent-700 dark:text-accent-300"
        link   = "text-accent-600 dark:text-accent-400 hover:underline"
        c(x) = Code(:class => inline, x)

        struct_row(n, name, what) = Div(:class => "flex gap-4 items-baseline py-3 border-b last:border-b-0 border-warm-200 dark:border-warm-800",
            Span(:class => "font-serif text-2xl text-accent-500 w-8 shrink-0 leading-none", n),
            Div(Code(:class => "font-mono text-sm font-semibold text-warm-900 dark:text-warm-100", name),
                P(:class => "text-sm text-warm-600 dark:text-warm-400 mt-0.5", what)))

        sections = [
            ("install",          "Install"),
            ("five-structs",     "The five structs"),
            ("first-simulation", "First simulation"),
            ("reconstruct",      "Reconstruct"),
            ("dose",             "Dose"),
            ("helical",          "Helical and multi-kVp"),
            ("photon-counting",  "Photon counting → VMI"),
            ("dual-energy",      "Dual-kVp and dual-source"),
            ("spectral-hypr",    "SpectralHYPR denoising"),
            ("whats-next",       "What's next"),
        ]

        PageWithTOC(sections, Div(:class => "max-w-3xl mx-auto space-y-8",
            Div(:class => "space-y-4",
                Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500",
                    RawHtml("""<span style="color:var(--color-accent-500)">●</span>&nbsp; Julia 1.12 · BasisSimulator 0.15""")),
                H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl leading-[1.05] tracking-tight",
                    "Getting started"),
                P(:class => prose,
                    "From an empty environment to a reconstructed image, a dose report and a set of virtual ",
                    "monoenergetic images. The listings build on one another and run as written, on a GPU or ",
                    "on the CPU; the sizes are small enough for a laptop.")
            ),

            # ── Install ────────────────────────────────────────────────────────
            H2(:id => "install", :class => h2_cls, "Install"),
            P(:class => prose, "BasisSimulator.jl requires Julia 1.12. From the REPL:"),
            CodeBlock("""using Pkg
Pkg.add("BasisSimulator")
Pkg.add("GPUSelect")   # picks the device array type for you"""),
            P(:class => prose,
                "Add the package for your GPU, or none to run on the CPU. The kernels are written once against ",
                A(:href => "https://github.com/JuliaGPU/AcceleratedKernels.jl", :target => "_blank", :class => link, "AcceleratedKernels.jl"),
                "; the example notebooks run on CUDA and Metal."),
            CodeBlock("""Pkg.add("CUDA")      # NVIDIA
Pkg.add("Metal")     # Apple silicon
Pkg.add("AMDGPU")    # AMD
Pkg.add("oneAPI")    # Intel"""),

            # ── Five structs ───────────────────────────────────────────────────
            H2(:id => "five-structs", :class => h2_cls, "The five structs"),
            P(:class => prose, "A simulation is specified by five values, each with one job:"),
            Div(:class => "rounded-xl border border-warm-200 dark:border-warm-800 bg-warm-50 dark:bg-warm-900/40 px-5",
                struct_row("1", "Phantom", "what is scanned: a label mask on the device, one XrayAttenuation material per label, the voxel size"),
                struct_row("2", "EICTScanner / PCCTScanner", "the hardware: geometry, focal spot, filtration, bowtie, and the detector model of one family"),
                struct_row("3", "CTProtocol", "the acquisition: kVp, mA, views, rotation time, collimation, pitch, added filters"),
                struct_row("4", "SimOptions", "the physics common to both families: which effects are on, the noise seed, the projector"),
                struct_row("5", "ReconOptions", "the output grid: matrix size, field of view, z extent")),
            P(:class => small,
                "Parameter sets for the clinical systems the notebooks model are on the ",
                A(:href => "$(BASE)/scanners/", :class => link, "Scanners"), " page."),

            # ── First simulation ───────────────────────────────────────────────
            H2(:id => "first-simulation", :class => h2_cls, "First simulation"),
            P(:class => prose,
                "A Gammex 472 phantom on a model of the GE Revolution Apex Elite. The workspace allocates every ",
                "buffer once; ", c("simulate!"), " writes the log line-integral sinogram into ", c("ws.sinogram"),
                " and returns the acquisition's dose report."),
            CodeBlock("""import BasisSimulator as BS
import GPUSelect

AT = GPUSelect.Storage()      # CuArray, MtlArray, ROCArray or oneArray; Array on the CPU
to_gpu(x) = AT(x)

cpu = BS.create_gammex_472(n_voxels = 256, n_slices = 4, z_cm = 1.0)
phantom = BS.Phantom(to_gpu(cpu.mask), cpu.materials, cpu.voxel_size, cpu.origin, cpu.extent)

scanner  = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0,
                          detector_rows = 32, detector_cols = 834,
                          detector_row_size = 0.625, detector_col_size = 0.6)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = 500, collimation_mm = 5.0)
sim_opts = BS.SimOptions(seed = 42)
rec_opts = BS.ReconOptions(matrix_size = (512, 512, 4), fov_cm = 35.0, z_cm = 0.5)

ws     = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
result = BS.simulate!(ws, phantom, protocol, sim_opts)"""),
            P(:class => small,
                "The phantom's mask lives on the device and its materials on the host. Quantum and electronic ",
                "noise are drawn in the counts domain, before the log, where a real detector produces them. ",
                "A photon-counting scanner's workspace comes from ", c("create_workspace"), " (below)."),

            # ── Reconstruct ────────────────────────────────────────────────────
            H2(:id => "reconstruct", :class => h2_cls, "Reconstruct"),
            P(:class => prose,
                "The water beam-hardening correction is calibrated from the full detected spectrum of this ",
                "acquisition (tube, filters, bowtie, heel effect, detector efficiency) and has no tunable ",
                "parameters. Then FDK, and μ to Hounsfield units against the calibration's water reference:"),
            CodeBlock("""bhc  = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom)
sino = to_gpu(BS.apply_bhc_water(ws.sinogram, bhc))

fdk = BS.create_fdk_recon_workspace(sino, ws.geom, rec_opts.matrix_size)
hu  = BS.to_hounsfield(Array(BS.reconstruct!(fdk, sino, ws.geom)); μ_water = bhc.μ_water_ref)"""),
            P(:class => prose,
                "Hybrid iterative reconstruction takes the same sinogram. Its one dial is ", c("strength"),
                ", 0–100 in steps of 10: 0 is plain FBP, 60 the standard clinical setting."),
            CodeBlock("""hir    = BS.create_hir_recon_workspace(sino, ws.geom, rec_opts.matrix_size; strength = 60)
hu_hir = BS.to_hounsfield(Array(BS.reconstruct!(hir, sino, ws.geom)); μ_water = bhc.μ_water_ref)"""),
            P(:class => small,
                "Keep ", c("create_hir_recon_workspace(; projector)"), " equal to ", c("SimOptions(; projector)"),
                " (both default to ", c(":dd_fast"), ") so the reconstruction inverts the operator that made the data."),

            # ── Dose ───────────────────────────────────────────────────────────
            H2(:id => "dose", :class => h2_cls, "Dose"),
            P(:class => prose,
                "Every ", c("simulate!"), " result carries a ", c("DoseReport"), " computed by Monte Carlo from the ",
                "simulated beam, in the 32 cm body CTDI phantom by default:"),
            CodeBlock("""result.dose                  # CTDIvol, DLP, CTDIw, CTDI100 centre and periphery
result.dose.ctdi_vol_mGy     # 21.7 mGy for the scan above
result.dose.dlp_mGy_cm

# a head protocol, or two tubes: keywords reach compute_dose
BS.simulate!(ws, phantom, protocol, sim_opts; dose_kwargs = (; phantom = :head16))"""),

            # ── Helical / multi-kVp ────────────────────────────────────────────
            H2(:id => "helical", :class => h2_cls, "Helical and multi-kVp"),
            P(:class => prose,
                "One keyword, ", c("pitch"), ", makes a protocol helical; ", c("reconstruct!"), " then dispatches ",
                "to rebinned WFBP automatically."),
            CodeBlock("""
spiral = BS.CTProtocol(kVp = 120, mA = 200.0, views = 360,
                       collimation_mm = 10.0, pitch = 1.0, n_rotations = 2)"""),
            P(:class => prose,
                "The volume walk depends on the geometry and the material map, not on the spectrum, so a study of ",
                "one phantom at several tube voltages walks once and reuses the path lengths, bit-identically:"),
            CodeBlock("""paths = BS.material_paths(ws, phantom)
for kvp in (80, 100, 120, 140)
    p = BS.CTProtocol(kVp = kvp, mA = 200.0, views = 500, collimation_mm = 5.0)
    w = BS.create_eict_workspace(scanner, p, sim_opts, rec_opts, phantom)
    BS.simulate!(w, phantom, p, sim_opts; paths)
end"""),

            # ── PCCT ───────────────────────────────────────────────────────────
            H2(:id => "photon-counting", :class => h2_cls, "Photon counting → VMI"),
            P(:class => prose,
                "A ", c("PCCTScanner"), " adds the counting detector: thresholds, energy resolution, charge ",
                "sharing, dead time, and the pile-up and scatter corrections applied inside ", c("simulate!"),
                ". The result holds one corrected log sinogram per bin and the air counts of every ray."),
            CodeBlock("""scanner_pc = BS.PCCTScanner(
    source_to_isocenter = 610.0, source_to_detector = 1113.0,
    detector_rows = 32, detector_cols = 1200,          # 1200 × 0.3 mm covers the 33 cm phantom
    detector_row_size = 0.35, detector_col_size = 0.3,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0],     # four counting bins, keV
    energy_resolution = 10.0, charge_sharing_fwhm = 0.08,
    dead_time_ns = 5.0,                               # pile-up is modelled once there is a dead time
    pileup_correction = true, scatter_correction = true,
)
ws_pc  = BS.create_workspace(scanner_pc, protocol, sim_opts, rec_opts, phantom)
res_pc = BS.simulate!(ws_pc, phantom, protocol, sim_opts)"""),
            P(:class => small,
                "A dead time switches on the Monte Carlo pile-up model, which ", c("create_workspace"),
                " runs once, on the CPU, for the rates the scan will see: several minutes, after which ",
                c("simulate!"), " takes seconds. ", c("dead_time_ns = 0"), " skips pile-up while iterating."),
            P(:class => prose,
                c("spectral_basis"), " takes the detected response straight from the workspace, the model that ",
                "generated the counts, so the decomposition needs no calibration scan. ", c("vmi_pipeline"),
                " runs the n-channel profile-likelihood decomposition into water and iodine, reconstructs the basis ",
                "pair and synthesises virtual monoenergetic images in HU:"),
            CodeBlock("""channels = [Array(b) for b in res_pc.pcct_sino.bins]
basis    = BS.spectral_basis(ws_pc; I0 = res_pc.I0_bins)

vmi = BS.vmi_pipeline(; channels, basis, geom = ws_pc.geom, to_backend = to_gpu,
                      matrix_size = rec_opts.matrix_size)
vmi.vmis        # (512, 512, 4, 4): 40, 70, 100 and 140 keV
vmi.images      # the reconstructed (water, iodine) basis pair"""),
            P(:class => small,
                "Every stage is a keyword: ", c("method = :cong"), " for the Cong estimator; ", c("reduce_rows = true"),
                " with ", c("use_tlbf = true"), " for a z-invariant phantom's photon-counting configuration; ",
                c("recon_method = :hir"), " to reconstruct the basis pair iteratively; ", c("vmi_energies"),
                " for other energies. See ", c("?BS.vmi_pipeline"), "."),

            # ── Dual energy ────────────────────────────────────────────────────
            H2(:id => "dual-energy", :class => h2_cls, "Dual-kVp and dual-source"),
            P(:class => prose,
                "The same estimator takes any number of separate acquisitions. Rapid kVp switching and two ",
                "tubes both deliver a low and a high channel, each with the absolute response its own forward ",
                "model applied; ", c("spectral_basis_from_acquisitions"), " assembles them on the union of their ",
                "energy grids."),
            CodeBlock("""function acquire(protocol)
    w = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
    BS.simulate!(w, phantom, protocol, sim_opts)
    air = w.bowtie_air_reference === nothing ? ones(w.geom.n_cols, w.geom.n_rows) :
          Array(w.bowtie_air_reference)
    I0 = BS.compute_detector_I0(w.geom, protocol, sum(w.weights)) * w.η_eff
    energies, response = BS.resolve_source_spectrum_full(sim_opts, protocol; scanner, geom = w.geom)
    (sino = Array(w.sinogram), geom = w.geom, energies, response, I0_ray = I0 .* air)
end

# rapid kVp switching; for two tubes, add e.g. additional_filters = [("Sn", 0.4)] on the high one
low  = acquire(BS.CTProtocol(kVp = 80,  mA = 407 * 0.65, views = 500, collimation_mm = 5.0))
high = acquire(BS.CTProtocol(kVp = 140, mA = 405 * 0.35, views = 500, collimation_mm = 5.0))

basis_de = BS.spectral_basis_from_acquisitions(acquisitions = [
    (; low.energies,  low.response,  low.I0_ray),
    (; high.energies, high.response, high.I0_ray)])
vmi_de = BS.vmi_pipeline(; channels = [low.sino, high.sino], basis = basis_de, geom = low.geom,
                         to_backend = to_gpu, matrix_size = rec_opts.matrix_size)"""),

            # ── SpectralHYPR ───────────────────────────────────────────────────
            H2(:id => "spectral-hypr", :class => h2_cls, "SpectralHYPR denoising"),
            P(:class => prose,
                c("SpectralHYPR"), " is a generalized HYPR-LR denoiser with two instances: one on the counts of ",
                "each detector row before the decomposition, one on the reconstructed basis pair. Pass it as ",
                c("denoiser"), " to either call above; this is the chain the basis-spectral-denoising study runs:"),
            CodeBlock("""chain = BS.SpectralHYPR(
    projection = BS.ProjectionHYPR(kernel = BS.HYPRKernel((3, 3), BS.BoxProfile())),
    image = BS.ImageHYPR(composite  = BS.HYPRKernel((3, 3, 7), BS.BoxProfile(); linear = false),
                         complement = BS.HYPRKernel((15, 15, 7), BS.BoxProfile(); linear = false)),
)
vmi_denoised = BS.vmi_pipeline(; channels, basis, geom = ws_pc.geom, to_backend = to_gpu,
                               matrix_size = rec_opts.matrix_size,
                               denoiser = chain, use_acnr = true)"""),
            P(:class => small,
                c("BS.SpectralHYPR()"), " with no arguments uses the package defaults. ACNR (anti-correlated noise ",
                "reduction) is on by default only when there is no denoiser, so ", c("use_acnr = true"), " asks for both."),

            # ── Next ───────────────────────────────────────────────────────────
            H2(:id => "whats-next", :class => h2_cls, "What's next"),
            Div(:class => "grid sm:grid-cols-3 gap-4",
                [A(:href => href, :class => "block group no-underline rounded-xl border border-warm-200 dark:border-warm-800 bg-warm-50 dark:bg-warm-900/40 hover:border-accent-400 dark:hover:border-accent-600 p-5 transition-colors",
                    Div(:class => "font-serif text-lg text-warm-900 dark:text-warm-100 group-hover:text-accent-600 dark:group-hover:text-accent-400", t),
                    P(:class => "text-sm text-warm-600 dark:text-warm-400 mt-1", d))
                 for (href, t, d) in [
                    ("$(BASE)/examples/", "Examples →", "Twelve notebooks, from the five-struct walkthrough to dual-source VMI."),
                    ("$(BASE)/scanners/", "Scanners →", "Parameter sets for the Apex Elite, NAEOTOM Alpha, Force and Definition Flash."),
                    ("$(BASE)/api/", "API →", "Every exported function and type, from the docstrings."),
                 ]]...)
        ))
    end
end
