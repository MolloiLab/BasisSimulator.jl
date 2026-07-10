let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
    code_block = "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 rounded-lg overflow-x-auto border border-warm-800"

    sections = [
        ("install",         "Install"),
        ("first-simulation", "First simulation"),
        ("whats-next",       "What's next"),
    ]

    PageWithTOC(sections, Div(:class => "max-w-3xl mx-auto space-y-10",
        H1(:class => "text-3xl font-serif font-bold text-warm-900 dark:text-warm-100",
            "Getting Started"),
        P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed",
            "BasisSimulator.jl runs on Julia 1.11+ and is GPU-backend-agnostic. Pick the backend that matches your hardware ",
            "(Metal for Apple Silicon, CUDA for NVIDIA, AMDGPU for AMD, oneAPI for Intel), or omit it entirely to run on CPU."
        ),

        H2(:id => "install", :class => "text-xl font-semibold text-warm-800 dark:text-warm-200",
            "Install"),
        P(:class => "text-warm-600 dark:text-warm-400", "From the Julia REPL:"),
        Pre(:class => code_block,
            Code(:class => "language-julia text-sm font-mono", """using Pkg
Pkg.add(url=\"https://github.com/MolloiLab/BasisSimulator.jl\")""")
        ),
        P(:class => "text-warm-600 dark:text-warm-400", "Then add GPUSelect and your GPU backend (skip the backend for CPU-only):"),
        Pre(:class => code_block,
            Code(:class => "language-julia text-sm font-mono", """Pkg.add(\"GPUSelect\")
Pkg.add(\"Metal\")     # Apple Silicon
Pkg.add(\"CUDA\")      # NVIDIA
Pkg.add(\"AMDGPU\")    # AMD
Pkg.add(\"oneAPI\")    # Intel""")
        ),

        H2(:id => "first-simulation", :class => "text-xl font-semibold text-warm-800 dark:text-warm-200",
            "First simulation"),
        P(:class => "text-warm-600 dark:text-warm-400",
            "The five-struct API maps to the five things every CT simulation needs to specify: what to scan, the scanner, the acquisition, simulation fidelity, and the reconstruction output."),
        Pre(:class => code_block,
            Code(:class => "language-julia text-sm font-mono", """import BasisSimulator as BS
import GPUSelect

AT = GPUSelect.Storage()  # MtlArray / CuArray / ROCArray / oneArray / Array
to_gpu(x) = AT(x)

# 1. Phantom — labeled mask + materials dict + voxel size
phantom_cpu = BS.create_gammex_472(n_voxels=256)
phantom = BS.Phantom(to_gpu(phantom_cpu.mask),
                     phantom_cpu.materials,
                     phantom_cpu.voxel_size)

# 2. Scanner — geometry, source, detector, filtration
scanner = BS.Scanner(
    source_to_isocenter = 626.0,
    source_to_detector  = 1097.0,
    detector_rows       = 64,
    detector_cols       = 832,
    detector_row_size   = 0.625,
    detector_col_size   = 1.053,
)

# 3. Protocol — kVp, mA, views, rotation time
protocol = BS.CTProtocol(kVp=120.0, mA=200.0, views=984, rotation_time=0.5)

# 4. SimOptions — physics fidelity preset (:eict | :pcct)
sim_opts = BS.SimOptions(fidelity=:eict, seed=42)

# 5. ReconOptions — output matrix and physical grid extent
rec_opts = BS.ReconOptions(matrix_size=(512, 512, 64), fov_cm=35.0)""")
        ),
        P(:class => "text-warm-600 dark:text-warm-400",
          "Allocate a workspace once and reuse it on subsequent calls — ",
          Code(:class => "text-accent-500 font-mono", "simulate!"),
          " writes into pre-allocated buffers and runs zero-allocation after JIT warm-up."),
        Pre(:class => code_block,
            Code(:class => "language-julia text-sm font-mono", """ws = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
BS.simulate!(ws, phantom, protocol, sim_opts)

bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner=scanner, geom=ws.geom)
sino_bhc = to_gpu(BS.apply_bhc_water(ws.sinogram, bhc))
ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, ws.geom, rec_opts.matrix_size)
hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, sino_bhc, ws.geom));
    μ_water = bhc.μ_water_ref,
)""")
        ),

        H2(:id => "whats-next", :class => "text-xl font-semibold text-warm-800 dark:text-warm-200",
            "What's next"),
        P(:class => "text-warm-600 dark:text-warm-400",
          "Worked examples covering the five-struct API, dual-kVp VMI, PCCT, XCAT phantoms, and a CatSim runtime comparison live on the ",
          A(:href => "$(BASE)/examples/",
            :class => "text-accent-500 hover:text-accent-600 underline no-underline", "Examples"),
          " page. The full API reference is at ",
          A(:href => "$(BASE)/api/",
            :class => "text-accent-500 hover:text-accent-600 underline no-underline", "API"),
          "."
        )
    ))
    end
end
