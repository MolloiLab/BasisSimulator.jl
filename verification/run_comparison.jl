#!/usr/bin/env julia
# =============================================================================
# BasisSimulator vs CatSim Verification Pipeline
# =============================================================================
#
# Main entry point for running comparisons between BasisSimulator and CatSim.
#
# Usage:
#   julia --project verification/run_comparison.jl --scale=dev
#   julia --project verification/run_comparison.jl --scale=integration --phantom=gammex
#   julia --project verification/run_comparison.jl --scale=dev --catsim=./catsim_results
#
# The pipeline:
#   1. Runs BasisSimulator at the specified scale
#   2. Optionally loads CatSim results (if --catsim provided)
#   3. Compares both against ground truth HU values (XrayAttenuation.jl)
#   4. Generates comparison report
#
# Ground truth is the tiebreaker: if CatSim disagrees with ground truth,
# we investigate CatSim, not BasisSimulator.
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse

# Parse command line arguments
function parse_args()
    s = ArgParseSettings(
        description = "BasisSimulator vs CatSim verification pipeline"
    )

    @add_arg_table! s begin
        "--scale"
            help = "Simulation scale: dev, integration, verification, publication"
            arg_type = String
            default = "dev"
        "--phantom"
            help = "Phantom type: water, gammex"
            arg_type = String
            default = "water"
        "--kvp"
            help = "Tube voltage (kVp)"
            arg_type = Int
            default = 120
        "--catsim"
            help = "Path to CatSim results directory (optional)"
            arg_type = String
            default = nothing
        "--output"
            help = "Output directory for results"
            arg_type = String
            default = "verification_results"
        "--no-gpu"
            help = "Disable GPU acceleration"
            action = :store_true
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 42
        "--run-catsim"
            help = "Also run CatSim (requires Python)"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end

# Main function
function main()
    args = parse_args()

    println()
    println("=" ^ 80)
    println("BasisSimulator vs CatSim Verification Pipeline")
    println("=" ^ 80)
    println()

    # Parse scale
    scale = Symbol(args["scale"])
    if scale ∉ [:dev, :integration, :verification, :publication]
        error("Invalid scale: $(args["scale"]). Must be one of: dev, integration, verification, publication")
    end

    # Parse phantom type
    phantom_type = Symbol(args["phantom"])
    if phantom_type ∉ [:water, :gammex]
        error("Invalid phantom: $(args["phantom"]). Must be one of: water, gammex")
    end

    println("Configuration:")
    println("  Scale: $scale")
    println("  Phantom: $phantom_type")
    println("  kVp: $(args["kvp"])")
    println("  GPU: $(args["no-gpu"] ? "disabled" : "enabled")")
    println("  Output: $(args["output"])")
    println()

    # Run CatSim first if requested
    catsim_dir = args["catsim"]
    if args["run-catsim"]
        println("Running CatSim...")
        catsim_dir = joinpath(args["output"], "catsim_results")

        # Call Python script
        catsim_script = joinpath(@__DIR__, "run_catsim.py")
        cmd = `python3 $catsim_script --phantom $(args["phantom"]) --scale $(args["scale"]) --kvp $(args["kvp"]) --output $catsim_dir`

        try
            run(cmd)
        catch e
            @warn "CatSim run failed: $e"
            @warn "Continuing with BasisSimulator-only comparison"
            catsim_dir = nothing
        end
    end

    # Include comparison module
    include(joinpath(@__DIR__, "compare_simulators.jl"))
    using .CompareSimulators

    # Create configuration
    config = ComparisonConfig(
        scale = scale,
        kvp = args["kvp"],
        phantom_type = phantom_type,
        output_dir = args["output"],
        catsim_results_dir = catsim_dir,
        use_gpu = !args["no-gpu"],
        noise_seed = args["seed"]
    )

    # Run comparison
    result = run_comparison(config)

    # Return exit code based on pass/fail
    if result.passed
        println("\n✓ Verification PASSED\n")
        return 0
    else
        println("\n✗ Verification FAILED\n")
        return 1
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

# Also export for interactive use
export main
