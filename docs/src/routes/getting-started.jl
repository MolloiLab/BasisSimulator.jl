() -> begin
    Div(:class => "max-w-3xl mx-auto space-y-8",
        H1("Getting Started"),
        P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed",
            "BasisSimulator.jl runs on Julia 1.11+ and is GPU-backend-agnostic. Pick the backend that matches your hardware ",
            "(Metal for Apple Silicon, CUDA for NVIDIA, AMDGPU for AMD), or omit it entirely to run on CPU."
        ),

        H2("Install"),
        P("From the Julia REPL:"),
        Pre(:class => "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 rounded-lg overflow-x-auto border border-warm-800",
            Code(:class => "language-julia text-sm font-mono", """using Pkg
Pkg.add(url=\"https://github.com/MolloiLab/BasisSimulator.jl\")""")
        ),
        P("Then add your GPU backend (skip for CPU-only):"),
        Pre(:class => "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 rounded-lg overflow-x-auto border border-warm-800",
            Code(:class => "language-julia text-sm font-mono", """Pkg.add(\"Metal\")     # Apple Silicon
Pkg.add(\"CUDA\")      # NVIDIA
Pkg.add(\"AMDGPU\")    # AMD""")
        ),

        H2("First simulation"),
        P("The five-struct API maps to the five things every CT simulation needs to specify: what to scan, the scanner, the acquisition, simulation fidelity, and the reconstruction output."),
        Pre(:class => "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 rounded-lg overflow-x-auto border border-warm-800",
            Code(:class => "language-julia text-sm font-mono", """import BasisSimulator as BS
using Metal  # or CUDA / AMDGPU; omit for CPU

# 1. Phantom — labeled mask + materials dict + voxel size
phantom_cpu = BS.create_gammex_472(n_voxels=256)
phantom = BS.Phantom(MtlArray(phantom_cpu.mask),
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

# 5. ReconOptions — output matrix, FOV, algorithm
rec_opts = BS.ReconOptions(matrix_size=(512, 512, 64), fov_cm=35.0)""")
        ),
        P("Allocate a workspace once and reuse it on subsequent calls — ", Code(:class => "text-accent-500 font-mono", "simulate!"),
          " writes into pre-allocated buffers and runs zero-allocation after JIT warm-up."),
        Pre(:class => "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 rounded-lg overflow-x-auto border border-warm-800",
            Code(:class => "language-julia text-sm font-mono", """ws = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
BS.simulate!(ws, phantom, scanner, protocol, sim_opts, rec_opts)

ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, rec_opts.matrix_size)
hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, rec_opts.matrix_size));
    μ_water = BS.get_reference_μ_water(70.0),
)""")
        ),

        H2("What's next"),
        P("Worked examples — including the four notebooks that produce the figures in the SoftwareX paper — live on the ",
          A(:href => "/BasisSimulator.jl/examples/",
            :class => "text-accent-500 hover:text-accent-600 underline no-underline", "Examples"),
          " page. The full API reference is at ",
          A(:href => "/BasisSimulator.jl/api/",
            :class => "text-accent-500 hover:text-accent-600 underline no-underline", "API"),
          "."
        ),
        P("Verification notebooks against CatSim and clinical-data benchmarks live in the companion repo ",
          A(:href => "https://github.com/MolloiLab/basis-verification", :target => "_blank",
            :class => "text-accent-500 hover:text-accent-600 underline no-underline", "MolloiLab/basis-verification"),
          "."
        )
    )
end
