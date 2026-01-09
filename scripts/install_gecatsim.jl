"""
Install GECATSIM from MolloiLab fork for cross-validation.

This script sets up the Python environment and installs GECATSIM
from https://github.com/MolloiLab/main

Usage:
    julia --project=. scripts/install_gecatsim.jl
"""

using CondaPkg
using PythonCall

println("\n" * "="^70)
println("GECATSIM INSTALLATION")
println("="^70)

# Step 1: Ensure CondaPkg environment is set up
println("\n1. Setting up CondaPkg environment...")
try
    CondaPkg.status()
    println("   ✅ CondaPkg environment ready")
catch e
    println("   ⚠️  Setting up CondaPkg for first time...")
    CondaPkg.resolve()
    println("   ✅ CondaPkg environment created")
end

# Step 2: Install GECATSIM from MolloiLab fork
println("\n2. Installing GECATSIM from MolloiLab/main...")

# Modern CondaPkg uses uv for pip packages
uv = CondaPkg.which("uv")
python = CondaPkg.which("python")

if uv === nothing && python === nothing
    error("Could not find uv or python in CondaPkg environment")
end

# Install GECATSIM from GitHub
gecatsim_url = "git+https://github.com/MolloiLab/main.git"
println("   Installing from: $gecatsim_url")

# Try using uv first (modern approach), fallback to python -m pip
if uv !== nothing
    println("   Using uv: $uv")
    try
        run(`$uv pip install $gecatsim_url`)
        println("   ✅ GECATSIM installed successfully")
    catch e
        println("   ❌ uv installation failed: $e")
        println("\n   Trying with --upgrade flag...")
        try
            run(`$uv pip install --upgrade $gecatsim_url`)
            println("   ✅ GECATSIM installed successfully (with --upgrade)")
        catch e2
            println("   ❌ Installation failed: $e2")
            error("Could not install GECATSIM. Please check the error messages above.")
        end
    end
elseif python !== nothing
    println("   Using python: $python")
    try
        run(`$python -m pip install $gecatsim_url`)
        println("   ✅ GECATSIM installed successfully")
    catch e
        println("   ❌ pip installation failed: $e")
        println("\n   Trying with --upgrade flag...")
        try
            run(`$python -m pip install --upgrade $gecatsim_url`)
            println("   ✅ GECATSIM installed successfully (with --upgrade)")
        catch e2
            println("   ❌ Installation failed: $e2")
            error("Could not install GECATSIM. Please check the error messages above.")
        end
    end
end

# Step 3: Verify installation
println("\n3. Verifying GECATSIM installation...")

try
    # Import gecatsim to verify it's available
    gecatsim = pyimport("gecatsim")
    println("   ✅ GECATSIM imported successfully")

    # Try to get version if available
    try
        version = gecatsim.__version__
        println("   Version: $version")
    catch
        println("   Version: Unknown (no __version__ attribute)")
    end

    # List available modules
    println("\n   Available GECATSIM modules:")
    for name in pylist(pybuiltins.dir(gecatsim))
        name_str = pyconvert(String, name)
        if !startswith(name_str, "_")
            println("      - $name_str")
        end
    end

catch e
    println("   ❌ Could not import GECATSIM: $e")
    println("\n   Troubleshooting:")
    println("   1. Check that the installation completed without errors")
    println("   2. Try restarting Julia")
    println("   3. Check GitHub repository is accessible: https://github.com/MolloiLab/main")
    error("GECATSIM verification failed")
end

println("\n" * "="^70)
println("✅ GECATSIM INSTALLATION COMPLETE")
println("="^70)
println("\nYou can now use GECATSIM for cross-validation:")
println("  julia> using PythonCall")
println("  julia> gecatsim = pyimport(\"gecatsim\")")
println("\n")
