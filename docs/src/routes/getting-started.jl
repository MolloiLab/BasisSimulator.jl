# Getting started — the guide: what a simulated scan is, install, a first scan, then the four steps
# (phantom, scan, reconstruct, spectral) with every rough edge where it bites, running real studies,
# contributing, and a context block for coding agents. Every listing runs as written, in order, on a
# GPU (photon counting) or on the CPU too (everything else); the versions are read from Project.toml.

let BASE = get(ENV, "BASISSIM_BASE", ""),
    PROJECT = read(joinpath(@__DIR__, "..", "..", "..", "Project.toml"), String),
    VERSION_LINE = let v = match(r"^version\s*=\s*\"(.+?)\""m, PROJECT)[1],
                       j = match(r"^julia\s*=\s*\"(.+?)\""m, PROJECT)[1]
        "BasisSimulator $(v) · Julia $(j) or later"
    end

    () -> begin
        prose  = "text-warm-600 dark:text-warm-400 leading-relaxed text-[17px]"
        small  = "text-[15px] text-warm-600 dark:text-warm-400 leading-relaxed"
        h2_cls = "text-3xl font-serif font-semibold text-warm-900 dark:text-warm-100 scroll-mt-24 pt-8"
        h3_cls = "no-rule text-xl font-serif font-semibold text-warm-900 dark:text-warm-100 pt-2"
        inline = "font-mono text-[0.85em] px-1 py-0.5 rounded bg-warm-200/70 dark:bg-warm-800/70 text-accent-700 dark:text-accent-300"
        linkc  = "text-accent-600 dark:text-accent-400 hover:underline"

        p(xs...) = P(:class => prose, xs...)
        c(x) = Code(:class => inline, x)
        b(x) = Strong(:class => "text-warm-900 dark:text-warm-100", x)
        link(href, x) = A(:href => href, :class => linkc, x)
        ext(href, x) = A(:href => href, :target => "_blank", :class => linkc, x)
        julia(code) = CodeBlock(code)
        shell(code) = CodeBlock(code; language = "bash")
        parts(x) = x isa Tuple ? x : (x,) # an item is one string or a tuple of pieces
        bullets(items...) = Ul(:class => "list-disc pl-6 space-y-2 " * prose, [Li(parts(x)...) for x in items]...)
        steps(items...) = Ol(:class => "bs-steps space-y-3 " * prose, [Li(parts(x)...) for x in items]...)
        note(title, xs...) = Details(:class => "bs-note", Summary(title),
            Div(:class => "bs-note-body space-y-3 " * small, xs...))
        # a rough edge: what you see, why it happens, what to do
        edge(symptom, why, fix) = Details(:class => "bs-note bs-edge", Summary(symptom),
            Div(:class => "bs-note-body space-y-2 " * small,
                P(b("Why. "), parts(why)...), P(b("What to do. "), parts(fix)...)))
        edges(xs...) = Div(:class => "space-y-2", xs...)
        table(head, rows) = Div(:class => "bs-table-wrap",
            Table(:class => "bs-table",
                Thead(Tr([Th(h) for h in head]...)),
                Tbody([Tr([Td(x) for x in row]...) for row in rows]...)))
        figure(src, alt, caption) = Figure(:class => "bs-figure",
            Img(:src => "$(BASE)/assets/$(src)", :alt => alt, :loading => "lazy"), Figcaption(caption))

        sections = [
            ("what",          "What a Simulated Scan Is"),
            ("install",       "Install"),
            ("first-scan",    "Your First Scan"),
            ("phantom",       "1 · The Phantom"),
            ("scan",          "2 · The Scan"),
            ("reconstruct",   "3 · Reconstruction"),
            ("spectral",      "4 · Spectral Imaging"),
            ("studies",       "Running Real Studies"),
            ("rough-edges",   "Rough Edges"),
            ("contributing",  "Contributing"),
            ("next",          "Where to Go Next"),
            ("agents",        "For Your Agent"),
        ]

        PageWithTOC(sections, Div(:class => "max-w-3xl mx-auto space-y-6",

            # ── Header ───────────────────────────────────────────────────────
            Div(:class => "space-y-5",
                Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500",
                    RawHtml("""<span style="color:var(--color-accent-500)">●</span>&nbsp; $(VERSION_LINE)""")),
                H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl leading-[1.05] tracking-tight",
                    "Getting Started"),
                p("A CT scanner turns a patient into numbers: an X-ray spectrum goes through the body, a detector ",
                  "counts what comes out, and a reconstruction turns the counts into an image. BasisSimulator.jl does the ",
                  "same thing to a digital patient whose every voxel you know. That is the point of it. A simulated scan has ",
                  "a ground truth, so you can ask questions a real scan cannot answer: how much of this error is noise, ",
                  "what does this protocol do to iodine, how low can the dose go before a measurement breaks."),
                p("By the end of this page you will have scanned a calibration phantom and an anatomical chest, with and ",
                  "without contrast, on a model of a clinical scanner. You will have Hounsfield-unit images, the dose of each ",
                  "scan, and photon-counting virtual monoenergetic images. Along the way the page explains why each setting ",
                  "is what it is, and it collects every rough edge we and other users have hit, with what to do about it."),
                p("Read the plain text first. The boxes that open are for when you need the detail.")
            ),

            # ── What a simulated scan is ─────────────────────────────────────
            H2(:id => "what", :class => h2_cls, "What a Simulated Scan Is"),
            p("Every study is the same four steps, and the rest of this page follows them in order:"),
            steps(
                (b("The phantom. "), "What is scanned: a 3-D map of labels, and one material for each label (water, bone, ",
                 "iodinated blood). A calibration phantom, an anatomical model, or your own segmentation."),
                (b("The scan. "), "A model of a real scanner (its geometry, filters, bowtie and detector) runs a protocol ",
                 "(kVp, mA, views, collimation). The spectrum is traced through the phantom, the detector records it with ",
                 "its noise, and the dose of the scan is reported."),
                (b("Reconstruction. "), "The measured line integrals are corrected for beam hardening and reconstructed by ",
                 "filtered back-projection, helical reconstruction or hybrid iterative reconstruction, then converted to ",
                 "Hounsfield units."),
                (b("Spectral imaging. "), "A photon-counting detector, or two tube voltages, measure the same patient at ",
                 "several energies. That separates water from iodine and gives virtual monoenergetic images."),
            ),
            p("In code, a scan is five values, each with one job, and two calls:"),
            table(["Value", "Its job"], [
                [c("Phantom"), "what is scanned: the label map on the device, one material per label, the voxel size"],
                [Span(c("EICTScanner"), " or ", c("PCCTScanner")), "the hardware: geometry, filters, bowtie and the detector (energy-integrating or photon-counting)"],
                [c("CTProtocol"), "the acquisition: kVp, mA, views, rotation time, collimation, pitch, added filters"],
                [c("SimOptions"), "the physics switches, the noise seed and the projector"],
                [c("ReconOptions"), "the output grid: matrix size, field of view, z extent"],
            ]),
            p(c("create_workspace"), " allocates every buffer once, and ", c("simulate!"), " runs the scan into it. The ",
              "workspace can be reused for scan after scan without allocating again."),

            # ── Install ──────────────────────────────────────────────────────
            H2(:id => "install", :class => h2_cls, "Install"),
            p("BasisSimulator needs Julia 1.12 or later. Install Julia with ", ext("https://julialang.org/install/", "juliaup"),
              ", then start it with one thread per core, which the CPU path needs:"),
            shell("julia -t auto"),
            p("Add the package, GPUSelect (which picks your GPU), and the package for your GPU, all in the same environment:"),
            julia(raw"""using Pkg
Pkg.add(["BasisSimulator", "GPUSelect"])
Pkg.add("Metal") # Apple silicon; "CUDA" for NVIDIA; nothing for the CPU"""),
            p("The first install and precompile take about two minutes. The first scan in a new Julia session also ",
              "compiles, so it takes a few seconds longer than every scan after it. Nothing is stuck."),
            note("Which GPUs are tested",
                P("The test suite and the example notebooks run on NVIDIA (CUDA) and Apple (Metal) GPUs, and the suite runs on ",
                  "the CPU on Linux, Windows and macOS. The kernels are written once against AcceleratedKernels.jl, so AMD ",
                  "(", c("AMDGPU"), ") and Intel (", c("oneAPI"), ") GPUs should work, but they are not tested. Apple GPUs ",
                  "have no 64-bit floats; BasisSimulator computes in 32-bit floats everywhere by default, so that is fine."),
                P("Photon counting is GPU-only in practice: on a CPU its first scan takes many minutes just to compile.")),
            note("Using it next to other packages",
                P("BasisSimulator 0.15.1 and later share an environment with packages that need AcceleratedKernels 0.5 ",
                  "(earlier versions could not). If ", c("Pkg.add"), " reports unsatisfiable requirements, run ",
                  c("Pkg.status()"), " and check which version of BasisSimulator it can install."),
                P("Pin the version you use in your project's ", c("[compat]"), " section. Below version 1.0 a minor ",
                  "release can change the API: read the ", ext("https://github.com/MolloiLab/BasisSimulator.jl/blob/main/CHANGELOG.md", "changelog"),
                  " before you upgrade, and restart Julia after any ", c("Pkg.add"), " or ", c("Pkg.update"),
                  ", because a running session keeps the version it loaded.")),

            # ── First scan ───────────────────────────────────────────────────
            H2(:id => "first-scan", :class => h2_cls, "Your First Scan"),
            p("The Gammex 472 is the phantom every CT physicist knows: a 33 cm water body with rods of calcium and ",
              "iodine at known concentrations. Here it is scanned on a model of the GE Revolution Apex Elite at 120 kVp ",
              "and 200 mA. The scanner is copied keyword for keyword from the ", link("$(BASE)/scanners/", "Scanners"),
              " page, which holds every clinical parameter set the examples use."),
            julia(raw"""import BasisSimulator as BS
import GPUSelect

device = GPUSelect.Storage() # a GPU array type (MtlArray, CuArray), or Array

gammex = BS.create_gammex_472(n_voxels = 512, n_slices = 4, z_cm = 1.0)
mask = device(gammex.mask) # the label map, on the GPU
phantom = BS.Phantom(mask, gammex.materials, gammex.voxel_size, gammex.origin, gammex.extent)

# GE Revolution Apex Elite, copied from the Scanners page
scanner = BS.EICTScanner(
    source_to_isocenter = 625.6, source_to_detector = 1100.0,
    detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6,
    target_angle = 10.0,
    flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0,
    electronic_noise = 3500.0, detection_gain = 10.0,
)
protocol = BS.CTProtocol(
    kVp = 120, mA = 200.0, views = 500, rotation_time = 1.0,
    collimation_mm = 5.0, additional_filters = [("Al", 4.5)],
)
sim_opts = BS.SimOptions(seed = 1)
recon = BS.ReconOptions(matrix_size = (512, 512, 4), fov_cm = 35.0, z_cm = 0.5)

ws = BS.create_workspace(scanner, protocol, sim_opts, recon, phantom)
result = BS.simulate!(ws, phantom, protocol, sim_opts)
result.dose.ctdi_vol_mGy"""),
            p("Three things in it matter more than they look:"),
            bullets(
                (b("The mask's array type picks the device. "), c("device(gammex.mask)"), " puts the label map on your GPU, ",
                 "and every buffer the workspace allocates follows it. Pass a CPU mask and the whole scan runs on the CPU, ",
                 "with no message. ", c("typeof(phantom.mask)"), " tells you which you have."),
                (b("The phantom voxel is no larger than the image voxel. "), "512 voxels over the 35 cm field are 0.68 mm, the ",
                 "same as the reconstruction's. A coarser phantom puts staircase edges on every rod, and they reconstruct as ",
                 "dotted rings."),
                (b("The seed is set. "), "Noise is drawn from ", c("seed"), ", the same way every time. Two scans with the ",
                 "same seed get the same noise, which is what you want for a comparison and wrong for anything that must be ",
                 "independent (see ", link("#scan", "The Scan"), ")."),
            ),
            p(c("simulate!"), " writes the corrected log sinogram, the line integrals, into ", c("ws.sinogram"),
              ", laid out as (detector columns, detector rows, views), and returns the scan's dose report. This scan is ",
              "about 14 mGy CTDIvol in the 32 cm body phantom."),
            p("Now reconstruct it. The beam-hardening correction is calibrated from this scan's own detected spectrum: ",
              "the tube, filters, bowtie and detector, with no parameters to tune. Then filtered back-projection, and the ",
              "attenuation is converted to Hounsfield units against the calibration's water:"),
            julia(raw"""bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom)
sino = BS.apply_bhc_water(ws.sinogram, bhc)

fbp = BS.create_fdk_recon_workspace(sino, ws.geom, recon.matrix_size)
volume = Array(BS.reconstruct!(fbp, sino, ws.geom))
hu = BS.to_hounsfield(volume; μ_water = bhc.μ_water_ref)"""),
            p(c("hu"), " is a 512 × 512 × 4 array of Hounsfield units. To look at it, add CairoMakie (",
              c("Pkg.add(\"CairoMakie\")"), "):"),
            julia(raw"""import CairoMakie as CM

fig = CM.Figure(size = (620, 520))
ax = CM.Axis(fig[1, 1]; title = "Gammex 472, FBP", aspect = CM.DataAspect())
CM.hidedecorations!(ax)
hm = CM.heatmap!(ax, hu[:, :, 2]; colormap = :grays, colorrange = (-160, 240))
CM.Colorbar(fig[1, 2], hm; label = "HU")
CM.save("gammex_fbp.png", fig)"""),
            p("Hybrid iterative reconstruction takes the same sinogram. Its one dial is ", c("strength"), ", from 0 ",
              "(plain filtered back-projection) to 100 in steps of 10. Give it the scan's air reference, so its ",
              "statistical weights know where the bowtie thins the beam:"),
            julia(raw"""hir = BS.create_hir_recon_workspace(sino, ws.geom, recon.matrix_size; strength = 60)
volume = BS.reconstruct!(hir, sino, ws.geom; air_reference = ws.bowtie_air_reference)
hu_hir = BS.to_hounsfield(Array(volume); μ_water = bhc.μ_water_ref)"""),
            figure("gs_gammex_fbp_hir.png", "Gammex 472 reconstructed by FBP and by hybrid IR at strength 60",
                "The scan above: filtered back-projection (left) and hybrid iterative reconstruction at strength 60 (right). The rods are calcium (bright) and iodine; the window is 400 HU wide around 40 HU."),
            p("There are no image files yet: results are Julia arrays. Save them with any Julia format. With JLD2 (",
              c("Pkg.add(\"JLD2\")"), "), keep the voxel size and what made the scan next to the images:"),
            julia(raw"""using JLD2

voxel_mm = 10 .* (recon.fov_cm, recon.fov_cm, recon.z_cm) ./ recon.matrix_size
jldsave(
    "gammex_scan.jld2"; hu, hu_hir, voxel_mm, dose = result.dose,
    seed = sim_opts.seed, version = string(pkgversion(BS)),
)"""),
            p("For DICOM or NIfTI, write the arrays with DICOM.jl or NIfTI.jl. The package itself writes no image files."),
            note("How long this takes",
                P("Measured on an Apple M4 laptop (Metal), after the first run's compilation:"),
                table(["Step", "Apple M4 GPU", "4-thread CPU"], [
                    [c("simulate!"), "0.6 s", "18 s"],
                    ["beam-hardening calibration", "0.2 s", "0.2 s"],
                    ["filtered back-projection", "0.3 s", "3 s"],
                    ["hybrid IR, strength 60", "13 s", "2.5 min"],
                ]),
                P("The first scan of a session adds 5–15 s of compilation. A full clinical volume (hundreds of slices, ",
                  "view integration on) takes seconds to minutes on a data-centre GPU, and needs one.")),
            note("Is water really 0 HU?",
                P("Not at the centre of this phantom, and that is physics, not a bug. The Gammex rods hold up to 600 mg/mL of ",
                  "calcium. They harden the beam more than water does, and a water-only beam-hardening correction cannot ",
                  "undo that, so the water between the rods reads about +20 HU at the centre and close to 0 at the edge. ",
                  "The same scan of a water cylinder reads 0 HU across the whole body (", link("#phantom", "The Phantom"),
                  " builds one). Judge water on a water phantom, or over the whole body, never in a small central ROI of the Gammex."),
                P("If water reads about +19 HU everywhere, ", c("to_hounsfield"), " was called without ",
                  c("μ_water = bhc.μ_water_ref"), " and fell back to tabulated water at 70 keV.")),

            # ── 1 · The phantom ──────────────────────────────────────────────
            H2(:id => "phantom", :class => h2_cls, "1 · The Phantom"),
            p("A phantom is a label array and a dictionary from each label to a material. Materials come from ",
              ext("https://github.com/MolloiLab/XrayAttenuation.jl", "XrayAttenuation.jl"), ", which BasisSimulator ",
              "re-exports as ", c("BS.XA"), ": ", c("BS.XA.Materials.water"), ", ", c("BS.XA.Materials.ncat_blood"),
              ", ", c("BS.XA.Elements.Iodine"), ". Reach them through ", c("BS.XA"), " rather than installing ",
              "XrayAttenuation yourself, so the two can never disagree on its version."),
            p("Your own segmentation works the same way. Here is a 30 cm water cylinder built from nothing, the phantom to ",
              "check water on:"),
            julia(raw"""voxel_cm = 0.05
xy = [hypot(i - 400.5, j - 400.5) * voxel_cm for i in 1:800, j in 1:800]
labels = repeat(UInt8.(xy .<= 15.0), 1, 1, 20) # a 30 cm water cylinder, 1 cm long

materials = Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water)
cylinder = BS.Phantom(device(labels), materials, (voxel_cm, voxel_cm, voxel_cm))"""),
            p("Scanned and reconstructed exactly as above, it reads water within noise of 0 HU from the centre to the ",
              "edge (−3.5, +1.4 and −1.5 HU at 0, 7 and 13 cm from the centre), and air at −996 HU."),
            table(["Rule", "Why"], [
                ["Voxel sizes are in centimetres, as a tuple (x, y, z).", "The phantom and the reconstruction grid are in cm; the scanner is in mm (see the units table below)."],
                ["Every label in the array needs a material.", "A label missing from the dictionary is scanned as air, without a warning."],
                ["Keep labels as UInt8 (up to 255) or UInt16.", "Converting more than 255 labels to UInt8 wraps them around."],
                ["The phantom is centred on the rotation axis.", "The volume is always placed at the isocentre. Crop or pad the array to move anatomy; origin does not move it."],
                ["Keep the whole body in the volume.", "Photons cross everything in the volume. A phantom cut to the heart scans as if the rest of the chest were air."],
                ["Leave a margin in z beyond the slices you want.", "The cone beam reaches past the reconstructed slab; anatomy that stops at the volume's edge spoils the end slices."],
                ["Use 64 materials or fewer.", "The fast projector holds 64 materials in one pass. More falls back to a path tens of times slower, with a warning."],
            ]),

            H3(:class => h3_cls, "Contrast and Anatomy"),
            p("Contrast is a material too: a tissue with iodine added. This function makes one from any material, keeping ",
              "the tissue's density and adding iodine by mass (it needs ", c("Pkg.add(\"Unitful\")"), "):"),
            julia(raw"""using Unitful

"`base` with `mg_per_mL` of iodine added, at the same density."
function with_iodine(base, mg_per_mL)
    f = mg_per_mL / (1000 * ustrip(u"g/cm^3", base.density)) # iodine mass fraction
    composition = Dict(Z => w * (1 - f) for (Z, w) in base.composition)
    composition[53] = get(composition, 53, 0.0) + f # Z = 53 is iodine
    name = "$(base.name) + $(mg_per_mL) mg/mL iodine"
    return BS.XA.Material(name, base.ZA_ratio, base.I, base.density, composition)
end"""),
            p("For anatomy, BasisSimulator downloads the published XCAT adult chest phantoms on first use and checks ",
              "their checksums. ", c("BS.xcat_default_materials()"), " maps every XCAT tissue name to a material, so ",
              "contrast is one line: replace the blood. Then scan the chest twice, once plain and once enhanced, each ",
              "with its own seed. This is the whole chain of ", link("#first-scan", "Your First Scan"), " in one function:"),
            julia(raw"""function scan(phantom, seed)
    opts = BS.SimOptions(; seed)
    ws = BS.create_workspace(scanner, protocol, opts, recon, phantom)
    BS.simulate!(ws, phantom, protocol, opts)
    bhc = BS.calibrate_bhc_water(opts, protocol; scanner, geom = ws.geom)
    sino = BS.apply_bhc_water(ws.sinogram, bhc)
    fbp = BS.create_fdk_recon_workspace(sino, ws.geom, recon.matrix_size)
    volume = Array(BS.reconstruct!(fbp, sino, ws.geom))
    BS.release_backend!(ws) # free the GPU memory before the next scan
    return BS.to_hounsfield(volume; μ_water = bhc.μ_water_ref)
end

materials = BS.xcat_default_materials()
materials["ncat_blood"] = with_iodine(materials["ncat_blood"], 5.0)

plain = BS.load_xcat_male_slab()
enhanced = BS.load_xcat_male_slab(; materials)

to_phantom(x) = BS.Phantom(device(x.mask), x.materials, x.voxel_size_cm)
before = scan(to_phantom(plain), 1)
after = scan(to_phantom(enhanced), 2)
enhancement = after .- before"""),
            figure("gs_xcat_contrast.png", "XCAT chest scanned without and with 5 mg/mL iodine in the blood, and the difference",
                "The XCAT adult male chest on the same scanner, without contrast (left) and with 5 mg/mL of iodine in the blood (middle). The difference (right) is the enhancement: the heart's chambers and the great vessels, plus the noise of two independent scans."),
            p("The slab is one 5 cm thick section at 0.25 mm in-plane, small enough to download in a second. ",
              c("BS.load_xcat_male_chest()"), " and ", c("BS.load_xcat_female_chest()"), " are the full volumes (about 65 MB ",
              "each); ", c("BS.xcat_phantoms()"), " lists them all."),
            note("The units, in one place",
                table(["Quantity", "Unit", "Where"], [
                    ["scanner distances, detector element sizes, collimation, filter thickness", "mm", Span(c("EICTScanner"), ", ", c("PCCTScanner"), ", ", c("CTProtocol"))],
                    ["detector element size", "mm at the isocentre, not at the detector face", Span(c("detector_col_size"), ", ", c("detector_row_size"))],
                    ["detector offset", "detector columns", c("detector_col_offset")],
                    ["phantom voxel size, reconstruction field of view and z extent", "cm", Span(c("Phantom"), ", ", c("create_gammex_472"), ", ", c("ReconOptions"))],
                    ["tube voltage", "kVp: 80, 100, 120 or 140", c("CTProtocol(kVp)")],
                    ["tube current", "mA, or mAs per rotation", c("CTProtocol(mA, mAs)")],
                    ["rotation time", "s", c("CTProtocol(rotation_time)")],
                    ["energies and thresholds", "keV", Span(c("PCCTScanner"), ", ", c("vmi_pipeline"))],
                    ["reconstructed images", "μ in cm⁻¹; HU after to_hounsfield", c("reconstruct!")],
                    ["dose", "mGy (CTDIvol), mGy·cm (DLP)", c("result.dose")],
                ]),
                P("Arrays are indexed (x, y, z) for volumes and (detector column, detector row, view) for sinograms. ",
                  "x runs left to right and y anterior to posterior; check a known landmark the first time you display your own data.")),

            # ── 2 · The scan ─────────────────────────────────────────────────
            H2(:id => "scan", :class => h2_cls, "2 · The Scan"),
            p("A scanner is a struct, not a name: there are no presets to look up. The ", link("$(BASE)/scanners/", "Scanners"),
              " page has the parameter sets of the GE Revolution Apex Elite, the Siemens NAEOTOM Alpha, SOMATOM Force and ",
              "SOMATOM Definition Flash as the examples run them. Copy them; do not retype them from a datasheet."),
            p("The one trap in a datasheet is the detector size. ", c("detector_col_size"), " and ", c("detector_row_size"),
              " are millimetres at the isocentre, the size a detector element projects to at the centre of rotation. A ",
              "vendor's detector pitch is at the detector face, larger by the magnification (about 1.8). A quick check: ",
              c("detector_cols × detector_col_size"), " is the scan field, about 50 cm for a clinical scanner, and it must ",
              "cover the body."),
            p("The protocol is what the technologist sets:"),
            bullets(
                (c("kVp"), " is 80, 100, 120 or 140, the tube voltages with measured spectra. Anything else is an error. ",
                 "Siemens's Sn150 is out of reach; the examples use Sn140."),
                (c("mA"), " and ", c("rotation_time"), " set the photons per view and the dose; ", c("mAs"),
                 " per rotation works too."),
                (c("views"), " per rotation: clinical scanners use about 1000. Use an even number for helical scans."),
                (c("collimation_mm"), " is the beam width at the isocentre, and ", c("pitch"), " makes the scan helical."),
                (c("additional_filters"), " are extra filters in the beam, written ", c("[(\"Al\", 4.5)]"), " or ",
                 c("[(\"Sn\", 0.4)]"), " in millimetres, with a decimal point."),
            ),
            p(c("SimOptions"), " holds the physics. Everything a clinical acquisition does is on by default: the detector's ",
              "efficiency and fill factor, scatter, the focal spot, the heel effect, lag, and quantum and electronic noise, ",
              "drawn in the counts before the log, where a real detector produces them. Three settings are yours to choose:"),
            bullets(
                (b(c("seed")), ": every independent acquisition needs its own. ", c("simulate!"), " restarts the noise from ",
                 c("seed"), " each time, so two tubes, two kVp, a before and after, or a hundred repeat scans with one seed ",
                 "all get the same noise, and it cancels in every difference. ", c("BS.SimOptions(; seed = k)"), " for scan ",
                 c("k"), " is enough."),
                (b(c("view_samples")), ": a real detector integrates while the gantry turns, which blurs the image slightly ",
                 "at the edge of the field. ", c("view_samples = 5"), " models it, at five times the projection cost. Every ",
                 "example uses it; leave it at 1 while you iterate."),
                (b(c("projector")), ": ", c(":dd_fast"), " (the default, distance-driven) or ", c(":siddon"), ". Leave it."),
            ),
            p("The physics switches are read when the workspace is built. To turn scatter off, say, build a new workspace ",
              "with the new options; changing them before ", c("simulate!"), " does nothing."),
            H3(:class => h3_cls, "Dose"),
            p("Every scan reports its CTDIvol and DLP, from a Monte Carlo of the simulated beam in the 32 cm body CTDI phantom:"),
            julia(raw"""result.dose # CTDIvol, DLP, CTDIw, and CTDI100 at the centre and the edge
result.dose.ctdi_vol_mGy
result.dose.dlp_mGy_cm

# a head protocol uses the 16 cm phantom; a dual-source scan has two tubes
head = (; phantom = :head16, n_tubes = 2)
BS.simulate!(ws, phantom, protocol, sim_opts; dose_kwargs = head)"""),
            p("At the same mA, the simulated CTDIvol is about 0.85 of what the scanner console reports, because the ",
              "tube output is not calibrated to a vendor. To match a clinical scan, scale ", c("mA"), " until the CTDIvol ",
              "matches the console's."),
            note("Helical scans and several tube voltages",
                P("One keyword, ", c("pitch"), ", makes a protocol helical, and ", c("reconstruct!"), " then uses rebinned ",
                  "helical reconstruction by itself:"),
                julia(raw"""helical = BS.CTProtocol(kVp = 120, mA = 200.0, views = 1000, rotation_time = 0.5,
                      collimation_mm = 40.0, pitch = 1.0, n_rotations = 3)"""),
                P("About one collimation width at each end of a helix has incomplete data. Pass ", c("coverage"), ", an array ",
                  "like the volume, to ", c("reconstruct!"), " to learn which voxels had data."),
                P("For several tube voltages on one phantom, the walk through the volume does not depend on the spectrum, ",
                  "so it can be done once:"),
                julia(raw"""paths = BS.material_paths(ws, phantom)
for (k, kVp) in enumerate((80, 100, 120, 140))
    p = BS.CTProtocol(; kVp, mA = 200.0, views = 500, collimation_mm = 5.0,
                      additional_filters = [("Al", 4.5)])
    opts = BS.SimOptions(seed = k)
    w = BS.create_workspace(scanner, p, opts, recon, phantom)
    BS.simulate!(w, phantom, p, opts; paths)
end"""),
                P("The cache holds point views, so it needs ", c("view_samples = 1"), ". With view integration, reuse a ",
                  "scan's noise-free part instead: ", c("simulate!(…; keep_projection = true)"), " once, then ",
                  c("projection = …"), " for every further noise draw.")),

            # ── 3 · Reconstruction ───────────────────────────────────────────
            H2(:id => "reconstruct", :class => h2_cls, "3 · Reconstruction"),
            p("Reconstruction is always the same three calls: calibrate the beam-hardening correction, reconstruct, convert ",
              "to Hounsfield units. Three rules keep the numbers right:"),
            bullets(
                (b("Calibrate with the scan's own options. "), c("calibrate_bhc_water"), " rebuilds the detected spectrum from ",
                 "every effect that is switched on, so give it the same ", c("SimOptions"), ", scanner and geometry as the scan."),
                (b("Convert with the calibration's water. "), c("to_hounsfield(…; μ_water = bhc.μ_water_ref)"),
                 ". Without it, water is about 19 HU off."),
                (b("Keep the projector. "), "Hybrid IR inverts the projector that made the data; ",
                 c("create_hir_recon_workspace(…; projector)"), " and ", c("SimOptions(; projector)"), " both default to ",
                 c(":dd_fast"), "."),
            ),
            p("The water correction leaves what water cannot explain: dense bone and calcium (the +20 HU at the Gammex ",
              "centre) and dense iodine, which cups by a few tens of HU in blood at 15 mg/mL. ",
              c("apply_bhc_two_material"), " corrects bone as well. HU per mg/mL of iodine also depends on the size of the ",
              "body (about 36 in a 30 cm thorax and 44 in a 10 cm phantom at 100 kVp), so calibrate iodine on the body you scan."),
            p("Images are square fields centred on the rotation axis. The corners outside the inscribed circle hold no data; ",
              "measure inside it. An off-centre organ needs a field large enough to include it."),
            note("Reconstruction kernels",
                P(c("create_fdk_recon_workspace(…; filter, cutoff)"), " takes the filter: ", c("StandardFilter()"), " (the ",
                  "default), ", c("SoftFilter()"), ", or a ", c("CustomFilter"), " whose knots are fractions of the image ",
                  "grid's Nyquist frequency. The same knots on a different field of view or matrix are a different ",
                  "kernel, so fit a kernel to the grid you will use."),
                P("A kernel's shape transfers from one dose to another; its noise does not, because a vendor's iterative ",
                  "reconstruction adapts to the dose. Hybrid IR strength 60 here is close to filtered back-projection plus ",
                  "edge-preserving smoothing, not a vendor's product.")),

            # ── 4 · Spectral ─────────────────────────────────────────────────
            H2(:id => "spectral", :class => h2_cls, "4 · Spectral Imaging"),
            p("A photon-counting detector sorts every photon by its energy into bins. ", c("PCCTScanner"), " adds that ",
              "detector: its energy thresholds and dead time, with the pile-up and scatter corrections applied inside ",
              c("simulate!"), ". This one counts in four bins, with a detector narrowed to the phantom to save memory:"),
            julia(raw"""scanner_pc = BS.PCCTScanner(
    source_to_isocenter = 610.0, source_to_detector = 1113.0,
    detector_rows = 32, detector_cols = 1200, # 1200 × 0.3 mm = 36 cm
    detector_row_size = 0.35, detector_col_size = 0.3,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0], # four counting bins, keV
    dead_time_ns = 5.0, # pile-up is modelled once there is a dead time
    pileup_correction = true, scatter_correction = true,
)
ws_pc = BS.create_workspace(scanner_pc, protocol, sim_opts, recon, phantom)
res_pc = BS.simulate!(ws_pc, phantom, protocol, sim_opts)"""),
            p("The first photon-counting workspace of a session takes about two minutes: the pile-up model is a Monte ",
              "Carlo run on the CPU for the count rates this scan will see. Every scan after that takes seconds. Set ",
              c("dead_time_ns = 0"), " to skip pile-up while you iterate."),
            p("The decomposition needs the detector's response, and the workspace has it: it is the model that made ",
              "the counts, so no calibration scan is needed. ", c("vmi_pipeline"), " separates water from iodine in every ",
              "ray, reconstructs both and makes virtual monoenergetic images in HU:"),
            julia(raw"""channels = [Array(b) for b in res_pc.pcct_sino.bins]
basis = BS.spectral_basis(ws_pc; I0 = res_pc.I0_bins)

vmi = BS.vmi_pipeline(; channels, basis, geom = ws_pc.geom, to_backend = device,
                      matrix_size = recon.matrix_size)
vmi.vmis   # (512, 512, 4, 4): 40, 70, 100 and 140 keV
vmi.images # the water and iodine images"""),
            p("Dual-energy acquisitions, rapid kVp switching and two tubes, go through the same estimator with ",
              c("spectral_basis_from_acquisitions"), ": one acquisition per tube voltage, ", b("each with its own seed"),
              ". The ", link("$(BASE)/examples/03_dual_kvp_switching_vmi/", "dual-kVp"), " and ",
              link("$(BASE)/examples/09_siemens_force_ufc_dual_source_vmi/", "dual-source"), " examples build them in full, ",
              "with the SpectralHYPR denoiser and each scanner's reconstruction windows."),
            note("What the photon-counting model covers",
                P("The detector response is a Monte Carlo model of cadmium telluride, with charge sharing and K-fluorescence. ",
                  c("energy_resolution"), " and ", c("charge_sharing_fwhm"), " are recorded on the scanner but do not change ",
                  "the physics, and ", c("detector_material = :CZT"), " or ", c(":Si"), " changes only the detection ",
                  "efficiency. Pile-up is modelled for four bins; with another number of bins, set ", c("dead_time_ns = 0"), "."),
                P("Counts are integer Poisson draws, as a real counter's are. ", c("use_noise = false"), " gives the expected ",
                  "counts, for a noise-free reference. Leave ", c("noise_reduction"), " at 0: it scales the noise and is not physics."),
                P("Two materials, water and iodine, cannot represent calcium exactly. At 40 keV, bone and calcium are less ",
                  "accurate than water and iodine in every VMI, whatever the number of bins.")),

            # ── Running real studies ─────────────────────────────────────────
            H2(:id => "studies", :class => h2_cls, "Running Real Studies"),
            H3(:class => h3_cls, "Memory"),
            p("GPU memory runs out before anything else. A full chest or head volume needs 10–11 GB at its peak for one ",
              "scan; a 16 GB laptop can hold one such scan at a time, not two. Four habits keep it in check:"),
            bullets(
                ("Copy what you need off the device (", c("Array(…)"), ") and call ", c("BS.release_backend!(ws)"),
                 " when a workspace is done, in a ", c("finally"), " block for long loops. Inside a loop over tiles, ",
                 c("release_backend!(ws; collect = false)"), " skips the full garbage collection."),
                ("Run one heavy scan at a time, and reduce sweeps to numbers as they run instead of keeping every volume."),
                ("Photon counting keeps its raw counts unless you pass ", c("capture_raw_counts = false"), " to ",
                 c("simulate!"), "; ", c("BS.estimate_pcct_workspace_bytes"), " tells you the size first."),
                (c("release_backend!"), " frees every device array it can reach from what you give it. Do not share a device ",
                 "array between a workspace you release and a phantom you still need."),
            ),
            H3(:class => h3_cls, "Notebooks"),
            p("The ", link("$(BASE)/examples/", "Examples"), " are Pluto notebooks, and the best way to learn a new kind of ",
              "study is to open the closest one and change it. To run one yourself, clone the repository and instantiate ",
              "the docs environment, which holds every package the notebooks use:"),
            shell(raw"""git clone https://github.com/MolloiLab/BasisSimulator.jl
cd BasisSimulator.jl
julia --project=docs -t auto -e 'using Pkg; Pkg.instantiate()'
julia --project=docs -t auto -e 'using Pluto; Pluto.run()'"""),
            p("Then open a notebook from ", c("docs/notebooks/"), ". Two things about Pluto: ", c("simulate!"),
              " changes the workspace in place, which Pluto cannot see, so return the values you plot from the cell that ",
              "computes them; and restart the notebook after any package update."),
            H3(:class => h3_cls, "On a Cluster"),
            p("A long study runs best as a plain script: ", c("julia --project=. -t auto study.jl"), ". Pin the ",
              "BasisSimulator version in the project, instantiate it once on a compute node, and run one GPU-heavy process ",
              "per GPU. Check the GPU works before a long job (", c("Array(device([1f0, 2f0])) == [1, 2]"), "), because ",
              "a broken driver can still report a device. Save each finished scan to disk as it completes."),
            H3(:class => h3_cls, "Reproducible Results"),
            p("A simulated scan is deterministic: the same version, inputs and seed give the same numbers. Save the seed, ",
              "the scanner, the protocol, the options and ", c("pkgversion(BasisSimulator)"), " with every output, and make ",
              "outputs again when any of them changes. Below 1.0, a release can change returned numbers on purpose; the ",
              "changelog says when, and by how much. Never present a scan shrunk for speed (fewer views, a coarser ",
              "phantom) as the real one without saying so."),

            # ── Rough edges ──────────────────────────────────────────────────
            H2(:id => "rough-edges", :class => h2_cls, "Rough Edges"),
            p("Each of these has cost someone hours. They are written as what you will see, why it happens and what to do. ",
              "Open the ones that match what you are looking at."),

            H3(:class => h3_cls, "Installing and Running"),
            edges(
                edge("Pkg.add reports unsatisfiable requirements",
                    ("Another package in the environment needs a different version of a shared dependency. Before 0.15.1, ",
                     "BasisSimulator could not sit next to packages that need AcceleratedKernels 0.5."),
                    ("Install BasisSimulator 0.15.1 or later. ", c("Pkg.status()"), " shows which versions resolved.")),
                edge("Code from another project fails: Scanner, SimOptions(fidelity = …), apply_cong!",
                    ("Below version 1.0, minor releases change the API, and old names are removed rather than kept as aliases."),
                    ("Read the changelog's Breaking sections between the two versions. ", c("EICTScanner"), " and ",
                     c("PCCTScanner"), " replaced ", c("Scanner"), " in 0.15; ", c("vmi_pipeline"), " replaced the Cong calls.")),
                edge("A package update has no effect",
                    ("A running Julia session or notebook keeps the version it loaded. ", c("Pkg.status()"),
                     " shows the installed and the loaded version when they differ."),
                    ("Restart Julia, or the notebook, after every ", c("Pkg.add"), ", ", c("Pkg.update"), " or ", c("Pkg.develop"), ".")),
                edge("Everything runs on the CPU, or GPUSelect warns that it fell back",
                    ("GPUSelect finds your GPU only through its package (Metal, CUDA) in the same environment, and the scan ",
                     "runs wherever the phantom's mask is."),
                    (c("Pkg.add(\"Metal\")"), " or ", c("Pkg.add(\"CUDA\")"), " in the active environment, restart, and make the ",
                     "phantom from ", c("device(mask)"), ". ", c("typeof(phantom.mask)"), " should be a GPU array.")),
                edge("The CPU path is very slow",
                    ("Julia starts with one thread unless told otherwise."),
                    ("Start Julia with ", c("julia -t auto"), ". Use a GPU for photon counting and for full volumes.")),
                edge("\"Cannot read stream serialized with a newer version of Julia\"",
                    ("The photon-counting detector response ships in a file written by Julia 1.12."),
                    ("Use Julia 1.12 or later.")),
                edge("Pkg.add succeeds, then using fails with \"not found in current path\"",
                    ("A launcher or shell profile set ", c("JULIA_LOAD_PATH"), " without the active project in it."),
                    ("Unset it (", c("env -u JULIA_LOAD_PATH julia"), "), or include ", c("@"), " in it: ",
                     c("JULIA_LOAD_PATH=\"@:@v#.#:@stdlib\""), ".")),
                edge("TypeError: in keyword argument fov_cm, expected Float64",
                    ("A few keywords are typed strictly: ", c("create_gammex_472(fov_cm, z_cm)"), " and the thicknesses in ",
                     c("additional_filters"), "."),
                    ("Write ", c("35.0"), ", not ", c("35"), ", and ", c("[(\"Al\", 4.0)]"), ", not ", c("[(\"Al\", 4)]"), ".")),
            ),

            H3(:class => h3_cls, "The Phantom"),
            edges(
                edge("MethodError: no method matching Phantom(…, ::Vector{Material}, …)",
                    ("Materials are a dictionary from each label value to its material, not a list."),
                    (c("BS.Phantom(labels, Dict(0 => air, 1 => water), (dx, dy, dz))"), ", with sizes in cm.")),
                edge("A tissue comes out as air",
                    ("A label with no entry in the materials dictionary is scanned as air, without a warning."),
                    ("Check ", c("setdiff(unique(labels), keys(materials))"), " is empty before scanning.")),
                edge("Labels above 255 turn into other labels",
                    ("Converting an array with more than 255 labels to UInt8 wraps the values around."),
                    ("Keep UInt16 labels; ", c("Phantom"), " picks the smallest type that holds them.")),
                edge("The anatomy is not where the origin says",
                    ("The projectors place the volume at the isocentre. ", c("origin"), " is used only to map labels onto the ",
                     "reconstruction grid."),
                    ("Crop or pad the label array so the anatomy you want sits at the volume's centre.")),
                edge("Wrong attenuation, or streaks at the body's edge",
                    ("The phantom was cut to the organ of interest, so the rest of the body is missing from every ray; or the ",
                     "body is wider than the detector covers."),
                    ("Scan the whole cross-section and reconstruct a smaller field if you want one. Check ",
                     c("detector_cols × detector_col_size"), " (mm) is larger than the body.")),
                edge("The first and last slices are wrong",
                    ("The cone beam reaches beyond the slices you reconstruct, and the anatomy stops at the volume's edge."),
                    ("Make the phantom longer in z than the slab you reconstruct, by about half the collimation plus a few mm.")),
                edge("Dotted rings or streaks around small objects",
                    ("Phantom voxels larger than the image's turn every round edge into a staircase."),
                    ("Use phantom voxels no larger than the reconstruction voxel, and a phantom field wider than the image's.")),
                edge("\"DD_FAST SINGLE-PASS DISABLED\", and the scan is much slower",
                    ("The fast projector holds 64 materials in one pass. A phantom with many iodine levels quickly has more."),
                    ("Run ", c("BS.compact_materials(phantom)"), " and merge labels with the same material. To quantize ",
                     "iodine to 64 levels, round each voxel up or down at random in proportion to its distance from each level, ",
                     "which keeps every region's mean exact; plain rounding biases region means by several percent. Never ",
                     "merge labels across regions you will measure.")),
                edge("A coarser copy of the phantom has different material volumes",
                    ("Resampling labels by nearest neighbour drops and duplicates voxels."),
                    ("Keep one high-resolution phantom as the source and derive coarser ones from it, never from each other.")),
                edge("The image is mirrored relative to the clinical scan",
                    ("Display conventions differ between tools, and a segmentation may come with its axes flipped."),
                    ("Fix the orientation once, on the label array, and check it against a known landmark, such as a ",
                     "labelled rod, not by eye.")),
            ),

            H3(:class => h3_cls, "Scanner, Protocol and Noise"),
            edges(
                edge("The fan is 80° wide, or the scan field is 80 cm",
                    ("Detector element sizes were entered at the detector face. BasisSimulator takes them at the isocentre."),
                    ("Divide the face pitch by the magnification (", c("source_to_detector / source_to_isocenter"), "), or copy ",
                     "the scanner from the Scanners page.")),
                edge("\"kVp must be one of (80, 100, 120, 140)\"",
                    ("Spectra exist for those four tube voltages."),
                    ("Use the nearest of them; 150 kVp is not available, so Sn150 protocols use Sn140.")),
                edge("\"anode_angle must be one of (8, 10)\", or a heel-effect error about the half-cone angle",
                    ("There are two angles. ", c("CTProtocol(anode_angle)"), " (8° or 10°) picks the spectrum; the scanner's ",
                     c("target_angle"), " (default 7°) drives the heel effect, and it must exceed the beam's half-cone angle."),
                    ("Set ", c("target_angle"), " to the scanner's anode angle (10° on the Scanners page), or collimate the beam.")),
                edge("Changing the scanner's rotation time changes nothing",
                    (c("gantry_rotation_time"), ", ", c("scan_diameter"), " and ", c("gantry_aperture"), " on the scanner are ",
                     "descriptive."),
                    ("Set ", c("rotation_time"), " on the ", c("CTProtocol"), ".")),
                edge("A physics switch has no effect",
                    ("The physics switches are read when the workspace is built."),
                    ("Build a new workspace with the new ", c("SimOptions"), ".")),
                edge("Two scans have exactly the same noise; a difference image or VMI is far too clean",
                    (c("simulate!"), " restarts the noise from ", c("seed"), " every call, and the default is 42."),
                    ("Give every independent acquisition its own seed: each tube, each kVp, each time point, each repeat.")),
                edge("The CTDIvol is lower than the scanner console's",
                    ("The tube output is not calibrated to a vendor; at the same mA the simulated CTDIvol is about 0.85 of the console's."),
                    ("Scale ", c("mA"), " until the CTDIvol matches. Use ", c("dose_kwargs = (; phantom = :head16)"),
                     " for head protocols, and compute dose at the clinical collimation if you simulate a narrower one.")),
                edge("Resolution and noise texture do not both match a real scanner",
                    ("A real detector integrates each view while the gantry turns; the default samples a point."),
                    (c("SimOptions(view_samples = 5)"), ", as every example uses.")),
                edge("\"collimation_mm exceeds scanner physical maximum\"",
                    ("The beam is wider than the detector."),
                    ("Use more detector rows, or ", c("create_workspace(…; extended_collimation = true)"), ", a simulator-only ",
                     "wide beam whose image quality is better than a real scanner's.")),
                edge("\"MC pile-up application is currently specialized to 4 bins\"",
                    ("Pile-up is modelled for four energy bins. The error comes at the scan, after the workspace build."),
                    ("Use four thresholds, or ", c("dead_time_ns = 0"), ".")),
            ),

            H3(:class => h3_cls, "Reconstruction and Spectral"),
            edges(
                edge("Water is about +19 HU everywhere",
                    (c("to_hounsfield"), " without ", c("μ_water"), " uses tabulated water at 70 keV."),
                    ("Pass ", c("μ_water = bhc.μ_water_ref"), ".")),
                edge("Water is about +20 HU at the centre of the Gammex",
                    ("The calcium rods harden the beam beyond what a water correction can undo."),
                    ("Check water on a water phantom, or over the whole body. ", c("apply_bhc_two_material"), " corrects bone too.")),
                edge("Water shifts after a change to SimOptions",
                    ("The calibration rebuilds the detected spectrum from the options it is given."),
                    ("Calibrate with the same ", c("SimOptions"), ", scanner and ", c("ws.geom"), " as the scan.")),
                edge("Hybrid IR is noisier at the body's edge",
                    ("Without the air reference, its statistical weights ignore the bowtie."),
                    (c("reconstruct!(hir, sino, geom; air_reference = ws.bowtie_air_reference)"), ".")),
                edge("\"HIR strength must be 0-100 in steps of 10\"",
                    ("Strength is a percentage; the old levels 1 to 5 were replaced."),
                    ("Level n is strength 20n.")),
                edge("MethodError: no method matching reconstruct!(…FDKReconWorkspace{Float32}…, ::Array{Float64, 3})",
                    ("A reconstruction workspace takes the element type of the sinogram it was made from."),
                    ("Make the reconstruction workspace from the sinogram you pass to it.")),
                edge("An image changes after the next reconstruction",
                    (c("reconstruct!"), " returns the workspace's own buffer, and the next call overwrites it."),
                    ("Copy it: ", c("Array(reconstruct!(…))"), " or ", c("copy(…)"), ".")),
                edge("Air reads about −1200 HU in an image corner",
                    ("The corners lie outside the circle the scan covers."),
                    ("Measure inside the inscribed circle.")),
                edge("A helical scan is biased by a few HU everywhere",
                    ("An odd number of views per rotation biases rebinned helical reconstruction."),
                    ("Use an even number of views per rotation.")),
                edge("A reconstruction kernel looks different on another field of view",
                    (c("CustomFilter"), " knots are fractions of the image grid's Nyquist frequency, which changes with the field and matrix."),
                    ("Fit the kernel on the grid you use, and refit it when the grid changes.")),
                edge("BoundsError indexing the detected spectrum",
                    (c("resolve_source_spectrum_full"), " returns one spectrum per detector element, a (columns, rows, energies) array."),
                    ("Take the central ray: ", c("w[end ÷ 2 + 1, end ÷ 2 + 1, :]"), ".")),
                edge("VMI noise goes down with energy and never up, or is far below the scanner's",
                    ("The acquisitions shared a seed, so their noise cancelled in the decomposition."),
                    ("One seed per tube or per kVp.")),
                edge("VMI noise is wrong on a new body",
                    ("The SpectralHYPR window pair (", c("pair_basis"), ", ", c("composite_energy"), ") is fitted to an object."),
                    ("Fit it on a noise draw of a body like the one you scan.")),
                edge("vmi_pipeline refuses recon_method = :hir with a PairFilter or the image denoiser",
                    ("The window pair is defined for filtered back-projection."),
                    ("Use filtered back-projection with the pair, or iterative reconstruction without it.")),
                edge("BoundsError on vmi.vmis[:, :, k]",
                    (c("vmis"), " is four-dimensional: x, y, z and energy."),
                    (c("vmi.vmis[:, :, z, e]"), ".")),
            ),

            H3(:class => h3_cls, "Memory and Notebooks"),
            edges(
                edge("The notebook's worker dies, or the GPU runs out of memory, hours in",
                    ("Sinogram-sized arrays pile up: workspaces that were never released, raw counts, kept volumes."),
                    ("Follow the four habits under ", link("#studies", "Memory"), ". Cache each finished scan to disk, so ",
                     "a crash loses one scan, not all of them.")),
                edge("A plot does not update after simulate!",
                    (c("simulate!"), " changes the workspace in place; Pluto does not see that, so a cell that plots ",
                     c("ws.sinogram"), " is not rerun."),
                    ("Return the values you plot from the cell that computes them.")),
                edge("A notebook from the Examples does not run on its own",
                    ("Every notebook activates the repository's docs environment."),
                    ("Clone the repository and instantiate ", c("docs/"), " first (", link("#studies", "Notebooks"), ").")),
                edge("UndefVarError when a notebook runs as a script",
                    ("A notebook is a set of cells that Pluto runs by dependency."),
                    ("Run notebooks in Pluto; for a script, move the code into a plain ", c(".jl"), " file.")),
                edge("Results changed after an upgrade",
                    ("Below 1.0, a release may change returned numbers on purpose; the changelog records each change and its size."),
                    ("Record ", c("pkgversion(BasisSimulator)"), " with every output, and make outputs again after upgrading.")),
                edge("An old result came back after changing an input",
                    ("A cache was keyed on less than everything that changes the scan."),
                    ("Key cached scans on the phantom, materials, scanner, protocol, options, seed and version.")),
            ),

            # ── Contributing ─────────────────────────────────────────────────
            H2(:id => "contributing", :class => h2_cls, "Contributing"),
            p("Bug reports, scanners, examples and code are all welcome. ", ext("https://github.com/MolloiLab/BasisSimulator.jl/issues", "Open an issue"),
              " with the shortest code that shows the problem and the output of ", c("Pkg.status()"), "."),
            p("For code, clone the repository and run the tests:"),
            shell(raw"""git clone https://github.com/MolloiLab/BasisSimulator.jl
cd BasisSimulator.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t 8 -e 'using Pkg; Pkg.test()'"""),
            p("The suite takes a few minutes on 8 threads. ", c("AGENTS.md"), " in the repository holds the conventions ",
              "in full. The ones that matter most:"),
            bullets(
                (b("One code path for every backend. "), "Kernels are written once with AcceleratedKernels (",
                 c("foreachindex"), ") or broadcasts over ", c("AbstractArray"), ": no CUDA- or Metal-only code, no scalar ",
                 "indexing of device arrays, and workspaces that allocate once."),
                (b("Numbers are the contract. "), "A change to a returned number is a breaking change: it gets a test that ",
                 "pins the new value and a changelog line that says how much it moved, measured, with the hardware."),
                (b("Every user-visible change "), "gets a line under ", c("## [Unreleased]"), " in ", c("CHANGELOG.md"),
                 ". Never change ", c("version"), " in a feature pull request; releases do that."),
                (b("Every exported name "), "has a docstring and its file a section in ", c("docs/src/api_sections.jl"),
                 "; ", c("test/docs.jl"), " checks both."),
                (b("A strict API. "), "Below 1.0, a better design replaces the old one; there are no deprecation layers. ",
                 "Update every call site, the examples and the docs in the same pull request, and delete what is superseded."),
            ),
            note("GPU tests",
                P("The CI machines have no GPU, and ", c("Pkg.test()"), " cannot load CUDA or Metal from your global ",
                  "environment, so GPU tests print \"Skipping … no GPU backend\" even on a GPU machine. Run the suite ",
                  "from an environment that has the package, its test imports and your GPU package (", c("\"CUDA\""),
                  " instead of ", c("\"Metal\""), " on NVIDIA), made once:"),
                shell(raw"""julia -e 'using Pkg; Pkg.activate("bs-gpu-test"; shared = true); Pkg.develop(path = ".");
    Pkg.add(["Metal", "JSON", "FFTW", "AcceleratedKernels", "XrayAttenuation"])'
julia --project=@bs-gpu-test -t 8 test/runtests.jl"""),
                P("Say in the pull request what passed on which GPU. A speed claim needs a measurement on that GPU; a ",
                  "speedup measured on the CPU says nothing about the GPU.")),
            note("The documentation site",
                P("The site is built with ", ext("https://github.com/GroupTherapyOrg/Therapy.jl", "Therapy.jl"), " from ",
                  c("docs/"), ". The example notebooks need a GPU, so they are rendered on a GPU machine and committed as ",
                  "HTML; the site build only mounts them."),
                shell(raw"""# fast local build, reusing the committed notebook pages
BASISSIM_SKIP_NB_EXPORT=1 julia --project=docs docs/app.jl build
python3 -m http.server -d docs/dist

# re-render stale notebooks (bash 4 or later), then check them
docs/render_notebooks.sh
python3 docs/verify_notebook_exports.py"""),
                P("Each rendered page is fingerprinted against its notebook, all of ", c("src/"), ", ", c("Project.toml"),
                  " and the docs lock files. Any change to them makes every page stale, and the docs deploy refuses stale ",
                  "pages, so a pull request that changes ", c("src/"), " re-renders the notebooks. Stop any live Pluto ",
                  "server before rendering: a server that saves the notebook again afterwards makes it stale. The verifier ",
                  "also rejects machine paths and errors in a page's output."),
                P("The site deploys from release tags, so a docs change reaches the site with the next release.")),
            note("Releases",
                P(c("RELEASING.md"), " is the runbook. In short: a ", c("release/<version>"), " branch bumps ", c("version"),
                  " in ", c("Project.toml"), " and ", c("CITATION.cff"), " and dates the changelog; the tests pass, including ",
                  "the GPU paths on a GPU and the newest Julia; the notebooks are re-rendered; the pull request is merged; a ",
                  c("@JuliaRegistrator register"), " comment on the merge commit registers it, with release notes that say ",
                  "what breaks when a minor version does. When the registry merges it, TagBot tags the release and the docs deploy.")),

            # ── Next ─────────────────────────────────────────────────────────
            H2(:id => "next", :class => h2_cls, "Where to Go Next"),
            Div(:class => "grid sm:grid-cols-3 gap-4",
                [A(:href => href, :class => "block group no-underline rounded-xl border border-warm-200 dark:border-warm-800 bg-warm-50 dark:bg-warm-900/40 hover:border-accent-400 dark:hover:border-accent-600 p-5 transition-colors",
                    Div(:class => "font-serif text-lg text-warm-900 dark:text-warm-100 group-hover:text-accent-600 dark:group-hover:text-accent-400", t),
                    P(:class => "text-sm text-warm-600 dark:text-warm-400 mt-1", d))
                 for (href, t, d) in [
                    ("$(BASE)/examples/", "Examples →", "Twelve notebooks, from the five structs to dual-source VMI, each checked against theory."),
                    ("$(BASE)/scanners/", "Scanners →", "The clinical parameter sets, and how to add a scanner."),
                    ("$(BASE)/api/", "API →", "Every exported function and type, from its docstring."),
                 ]]...),
            P(:class => small,
                "If BasisSimulator.jl is useful in your work, please cite it: ",
                ext("https://doi.org/10.1016/j.softx.2026.102910", "SoftwareX 2026;35:102910"), "."),

            # ── For your agent ───────────────────────────────────────────────
            H2(:id => "agents", :class => h2_cls, "For Your Agent"),
            p("Paste this into a coding agent before it writes BasisSimulator code. It holds the calls, the rules and the ",
              "rough edges above in the form an agent needs."),
            CodeBlock(raw"""You are writing Julia code with BasisSimulator.jl (github.com/MolloiLab/BasisSimulator.jl; docs: molloilab.github.io/BasisSimulator.jl), a GPU-portable simulator of energy-integrating (EICT) and photon-counting (PCCT) CT: spectrum -> phantom -> detector counts with noise -> beam-hardening correction -> FBP / helical WFBP / hybrid IR -> HU, plus water/iodine decomposition and VMI, with the dose of every scan.

SETUP
- Julia >= 1.12. Pkg.add(["BasisSimulator", "GPUSelect"]) plus "Metal" or "CUDA" in the SAME environment. Start Julia with -t auto. Restart Julia after any Pkg change.
- import BasisSimulator as BS; materials via BS.XA (never import XrayAttenuation separately).
- device = GPUSelect.Storage(); the phantom mask's array type picks the device for the whole scan: BS.Phantom(device(mask), ...). Float32 throughout (Metal has no Float64).

THE CHAIN
- phantom = BS.Phantom(device(labels), Dict(label => BS.XA.Materials.x, ...), (dx_cm, dy_cm, dz_cm))
- scanner = BS.EICTScanner(...) or BS.PCCTScanner(...; energy_thresholds) — copy parameter sets from the docs Scanners page; there are no presets.
- protocol = BS.CTProtocol(kVp = 80|100|120|140, mA, views, rotation_time, collimation_mm, additional_filters = [("Al", 4.5)], pitch)
- sim_opts = BS.SimOptions(seed = k, view_samples = 1|5); recon = BS.ReconOptions(matrix_size, fov_cm, z_cm)
- ws = BS.create_workspace(scanner, protocol, sim_opts, recon, phantom); result = BS.simulate!(ws, phantom, protocol, sim_opts) -> ws.sinogram (cols, rows, views), result.dose.ctdi_vol_mGy
- bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom); sino = BS.apply_bhc_water(ws.sinogram, bhc)
- fbp = BS.create_fdk_recon_workspace(sino, ws.geom, recon.matrix_size); hu = BS.to_hounsfield(Array(BS.reconstruct!(fbp, sino, ws.geom)); μ_water = bhc.μ_water_ref)
- hir = BS.create_hir_recon_workspace(sino, ws.geom, recon.matrix_size; strength = 0:10:100); BS.reconstruct!(hir, sino, ws.geom; air_reference = ws.bowtie_air_reference)
- PCCT: res = BS.simulate!(ws_pc, ...); basis = BS.spectral_basis(ws_pc; I0 = res.I0_bins); vmi = BS.vmi_pipeline(; channels = [Array(b) for b in res.pcct_sino.bins], basis, geom = ws_pc.geom, to_backend = device, matrix_size) -> vmi.vmis (x, y, z, energy) in HU
- Anatomy: BS.load_xcat_male_slab / _chest(; materials = BS.xcat_default_materials() with edits); contrast = base material with iodine (Z = 53) added by mass fraction mg_per_mL / (1000 * density_g_cm3).
- Free GPU memory: copy results with Array(...), then BS.release_backend!(ws).

RULES
- Units: scanner and collimation in mm, detector element sizes in mm AT THE ISOCENTRE; phantom voxels and recon FOV/z in cm; detector_col_offset in columns; kVp only 80/100/120/140; float literals for fov_cm, z_cm and filter thicknesses.
- Every independent acquisition (tube, kVp, time point, repeat) gets its own seed; simulate! reseeds from seed every call.
- Physics switches are fixed at create_workspace; rebuild the workspace to change them.
- BHC uses the scan's own sim_opts, scanner and ws.geom; to_hounsfield always gets μ_water = bhc.μ_water_ref; HIR projector equals SimOptions projector.
- Every label needs a material (missing -> air silently). Phantom voxel <= recon voxel. Whole body cross-section in the volume, z margin beyond the recon slab; the volume is centred at the isocentre (origin does not move it). <= 64 materials for the fast projector (compact_materials; dithered iodine quantization; never merge across measured regions).
- Detector must cover the body: detector_cols * detector_col_size (mm) > body width. Even views per rotation for helical. Measure inside the inscribed FOV circle.
- Water on a Gammex centre ROI reads about +20 HU (calcium hardening); judge water on a water phantom. HU per mg/mL iodine depends on body size; calibrate on the scanned body.
- PCCT: GPU only in practice; first workspace ~2 min (CPU pile-up Monte Carlo); pile-up needs exactly 4 bins; energy_resolution/charge_sharing_fwhm are informational; noise_reduction stays 0.
- reconstruct! returns the workspace's buffer: copy before the next call. vmi_pipeline(recon_method = :hir) cannot use PairFilter or image HYPR.
- Pluto: simulate! mutates in place (return values from cells); restart after package updates; one heavy scan at a time (a full chest/head scan peaks at 10-11 GB).
- Record seed, scanner, protocol, options and pkgversion(BasisSimulator) with every output. Never present a downscaled run (fewer views, coarser phantom) as the real scan.

CONTRIBUTING
- Backend-agnostic kernels only (AcceleratedKernels foreachindex / broadcasts), no scalar indexing of device arrays, workspaces preallocate.
- Numbers are the contract: changed outputs need a pinning test and a CHANGELOG [Unreleased] line with the measured change and hardware. Never edit `version` in a feature PR.
- Every export needs a docstring and an entry in docs/src/api_sections.jl (test/docs.jl). No deprecation shims: update all call sites, examples and docs together; delete superseded code.
- Tests: julia --project=. -t 8 -e 'using Pkg; Pkg.test()' (GPU tests skip there). For GPU tests: a shared env with Pkg.develop(path = ".") plus Metal or CUDA, JSON, FFTW, AcceleratedKernels, XrayAttenuation, then julia --project=@bs-gpu-test -t 8 test/runtests.jl; report the hardware.
- Docs notebooks render on a GPU (docs/render_notebooks.sh, python3 docs/verify_notebook_exports.py); any src/ change stales every page; no machine paths in outputs.""";
                language = "text"),
        ))
    end
end
