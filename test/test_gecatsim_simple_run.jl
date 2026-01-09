"""
Simple GECATSIM simulation test.

Try to run GECATSIM with our custom configurations and see if it works.
"""

using PythonCall

println("\n" * "="^70)
println("GECATSIM SIMPLE SIMULATION TEST")
println("="^70)

# Import GECATSIM
gecatsim = pyimport("gecatsim")

# Get the test/cfg_gecatsim directory (absolute path)
cfg_dir = joinpath(@__DIR__, "cfg_gecatsim")
println("\nConfiguration directory: $cfg_dir")

# Check that config files exist
println("\nChecking configuration files...")
config_files = [
    "Scanner_Aquilion.cfg",
    "Protocol_Simple.cfg",
    "Phantom_WaterCylinder.cfg",
    "Physics_Simple.cfg",
    "water_cylinder.ppm"
]

for file in config_files
    path = joinpath(cfg_dir, file)
    if isfile(path)
        println("   ✅ $file")
    else
        println("   ❌ $file NOT FOUND")
        error("Missing configuration file: $file")
    end
end

# Try to create CatSim object
println("\n1. Creating CatSim object...")
try
    # Need to provide paths without .cfg extension
    phantom_cfg = joinpath(cfg_dir, "Phantom_WaterCylinder")
    scanner_cfg = joinpath(cfg_dir, "Scanner_Aquilion")
    protocol_cfg = joinpath(cfg_dir, "Protocol_Simple")

    println("   Phantom config: $phantom_cfg")
    println("   Scanner config: $scanner_cfg")
    println("   Protocol config: $protocol_cfg")

    global ct = gecatsim.CatSim(phantom_cfg, scanner_cfg, protocol_cfg)
    println("   ✅ CatSim object created")
catch e
    println("   ❌ Failed to create CatSim object")
    println("   Error: $e")
    rethrow(e)
end

# Load physics configuration
println("\n2. Loading physics configuration...")
try
    physics_cfg = joinpath(cfg_dir, "Physics_Simple")
    println("   Physics config: $physics_cfg")

    ct.load_cfg(physics_cfg)
    println("   ✅ Physics configuration loaded")
catch e
    println("   ❌ Failed to load physics configuration")
    println("   Error: $e")
    # Physics config might be optional, continue anyway
end

# Inspect configuration
println("\n3. Inspecting loaded configuration...")
try
    println("   Scanner:")
    println("      SID: $(ct.scanner.sid) mm")
    println("      SDD: $(ct.scanner.sdd) mm")
    println("      Detector: $(ct.scanner.detectorRowCount) x $(ct.scanner.detectorColCount)")
    println("      Pixel size: $(ct.scanner.detectorColSize) x $(ct.scanner.detectorRowSize) mm")

    println("\n   Protocol:")
    println("      Views: $(ct.protocol.viewCount)")
    println("      mA: $(ct.protocol.mA)")

    # Check if phantom loaded
    if pyhasattr(ct, "phantom")
        println("\n   Phantom:")
        println("      Callback: $(ct.phantom.callback)")
    end

    println("\n   ✅ Configuration inspection successful")
catch e
    println("   ⚠️  Could not inspect all configuration")
    println("   Error: $e")
end

# Set output name
println("\n4. Setting output configuration...")
try
    ct.resultsName = joinpath(cfg_dir, "test_output")
    println("   Results will be saved to: $(ct.resultsName)")
    println("   ✅ Output configuration set")
catch e
    println("   ❌ Failed to set output configuration")
    println("   Error: $e")
end

# Try to run simulation (this might fail, but we'll learn from the error)
println("\n5. Attempting to run simulation...")
println("   ⚠️  This may fail - we're learning GECATSIM's requirements")

try
    # Run minimal scan (just phantom, skip air/offset)
    ct.run_all()
    println("   ✅ Simulation completed!")

    # Check if output files were created
    println("\n6. Checking output files...")
    output_base = ct.resultsName
    for ext in [".prep", ".raw", ".offset", ".air"]
        file = output_base * ext
        if isfile(file)
            size_mb = filesize(file) / 1024^2
            println("   ✅ $file ($(round(size_mb, digits=2)) MB)")
        end
    end
catch e
    println("   ❌ Simulation failed (expected, need to debug)")
    println("\n   Error message:")
    println("   $(e)")

    # Print more debugging info
    println("\n   Debugging info:")
    println("   This is likely due to:")
    println("   1. Phantom file format issues")
    println("   2. Missing required configuration parameters")
    println("   3. Incompatible spectrum file path")
    println("   4. Python/MATLAB syntax in .ppm file")
    println("\n   Next steps:")
    println("   - Check GECATSIM error message above")
    println("   - Compare our configs to working GECATSIM examples")
    println("   - Adjust phantom file format")
    println("   - Add missing required parameters")
end

println("\n" * "="^70)
println("GECATSIM SIMPLE TEST COMPLETE")
println("="^70 * "\n")
