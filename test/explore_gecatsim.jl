"""
Explore GECATSIM API and configuration structure.

This script introspects the GECATSIM package to understand:
1. Available modules and functions
2. Configuration structure
3. Example usage patterns
"""

using PythonCall

println("\n" * "="^70)
println("GECATSIM API EXPLORATION")
println("="^70)

# Import GECATSIM
gecatsim = pyimport("gecatsim")

println("\n1. GECATSIM Package Structure")
println("-"^70)

# List all modules and top-level items
println("Available modules/attributes:")
for item in pylist(pybuiltins.dir(gecatsim))
    item_str = pyconvert(String, item)
    if !startswith(item_str, "_")
        println("  - $item_str")
    end
end

# ==============================================================================
# 2. Explore CFG Module
# ==============================================================================
println("\n2. CFG Module (Configuration)")
println("-"^70)

cfg_module = gecatsim.CFG
println("CFG type: ", pytype(cfg_module))
println("\nCFG attributes:")
for item in pylist(pybuiltins.dir(cfg_module))
    item_str = pyconvert(String, item)
    if !startswith(item_str, "_")
        println("  - $item_str")
    end
end

# Try to get documentation
try
    cfg_doc = pybuiltins.getattr(cfg_module, "__doc__", "No documentation")
    println("\nCFG documentation:")
    println(pyconvert(String, cfg_doc))
catch e
    println("\nCould not get CFG documentation")
end

# ==============================================================================
# 3. Explore CatSim Module
# ==============================================================================
println("\n3. CatSim Module (Simulation)")
println("-"^70)

catsim = gecatsim.CatSim
println("CatSim type: ", pytype(catsim))
println("\nCatSim attributes:")
for item in pylist(pybuiltins.dir(catsim))
    item_str = pyconvert(String, item)
    if !startswith(item_str, "_") && !startswith(item_str, "np") && !startswith(item_str, "os")
        println("  - $item_str")
    end
end

# ==============================================================================
# 4. Explore source_cfg
# ==============================================================================
println("\n4. source_cfg (Example Configuration)")
println("-"^70)

try
    source_cfg = gecatsim.source_cfg
    println("source_cfg type: ", pytype(source_cfg))

    # If it's a module, explore it
    if pyhasattr(source_cfg, "__file__")
        file_path = pyconvert(String, source_cfg.__file__)
        println("source_cfg file: $file_path")

        println("\nsource_cfg attributes:")
        for item in pylist(pybuiltins.dir(source_cfg))
            item_str = pyconvert(String, item)
            if !startswith(item_str, "_")
                println("  - $item_str")
            end
        end
    end
catch e
    println("Could not explore source_cfg: $e")
end

# ==============================================================================
# 5. Try to create a minimal configuration
# ==============================================================================
println("\n5. Attempting Minimal Configuration Creation")
println("-"^70)

try
    # Try to import cfg_from_file if it exists
    if pyhasattr(gecatsim, "cfg_from_file")
        println("cfg_from_file function available")
    end

    # Try to create a config dict
    println("\nCreating empty configuration dict...")
    cfg = pydict()
    println("Empty cfg created: ", pytype(cfg))

    # Try to see if there's a default config
    if pyhasattr(source_cfg, "cfg")
        println("\nsource_cfg.cfg exists!")
        default_cfg = source_cfg.cfg
        println("Type: ", pytype(default_cfg))

        # List keys if it's a dict
        if pyhasattr(default_cfg, "keys")
            println("\nConfiguration sections:")
            for key in pylist(default_cfg.keys())
                key_str = pyconvert(String, key)
                println("  - $key_str")
            end
        end
    end
catch e
    println("Error during configuration exploration: $e")
end

# ==============================================================================
# 6. Explore GetMu (Attenuation)
# ==============================================================================
println("\n6. GetMu Module (Attenuation Coefficients)")
println("-"^70)

getmu = gecatsim.GetMu
println("GetMu type: ", pytype(getmu))
println("\nGetMu attributes:")
for item in pylist(pybuiltins.dir(getmu))
    item_str = pyconvert(String, item)
    if !startswith(item_str, "_") && !startswith(item_str, "np") && !startswith(item_str, "os")
        println("  - $item_str")
    end
end

# ==============================================================================
# 7. Check for example files
# ==============================================================================
println("\n7. Looking for Example Files")
println("-"^70)

try
    import_path = pyconvert(String, gecatsim.__file__)
    println("GECATSIM installed at: $import_path")

    # Try to find example directory
    gecatsim_dir = dirname(import_path)
    println("Package directory: $gecatsim_dir")
catch e
    println("Could not determine package location: $e")
end

println("\n" * "="^70)
println("GECATSIM EXPLORATION COMPLETE")
println("="^70 * "\n")
