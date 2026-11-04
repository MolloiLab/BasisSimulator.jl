# =============================================================================
# Geant4's own photon cross sections
# =============================================================================
# The attenuation a simulation actually samples is Geant4's, not XCOM's. Geant4.jl does not
# wrap G4EmCalculator, so a ten-line C++ shim (src/cpp/em_calculator.cpp) is compiled on first
# use against the Geant4 that Geant4.jl ships, into a Scratch.jl directory, and rebuilt whenever
# the shim or the Geant4 build changes. It needs a C++ compiler (`c++`, `g++` or `clang++`).
# =============================================================================

const EM_CALCULATOR = Ref{Ptr{Cvoid}}(C_NULL)

const G4_PROCESS_NAMES = (photoelectric = "phot", compton = "compt", rayleigh = "Rayl")

function em_calculator_symbol()
    EM_CALCULATOR[] == C_NULL || return EM_CALCULATOR[]
    source = joinpath(@__DIR__, "cpp", "em_calculator.cpp")
    prefix = Geant4.Geant4_jll.artifact_dir
    dir = @get_scratch!("em_calculator")
    tag = string(hash((read(source, String), prefix)); base = 16)
    library = joinpath(dir, "em_calculator_$(tag).$(Libdl.dlext)")
    if !isfile(library)
        compiler = something(
            Sys.which("c++"),
            Sys.which("g++"),
            Sys.which("clang++"),
            Some(nothing),
        )
        compiler === nothing && error(
            """
            Geant4's cross sections need a C++ compiler (c++, g++ or clang++) on the PATH,
            to build the small G4EmCalculator shim once. Install one, or compare with the XCOM
            values of `linear_attenuation` instead."""
        )
        rpath = Sys.islinux() ? ["-Wl,-rpath,$(prefix)/lib", "-Wl,--no-as-needed"] :
            ["-Wl,-rpath,$(prefix)/lib"]
        libs = [
            "-lG4processes_core", "-lG4particles", "-lG4materials", "-lG4global", "-lG4clhep",
            "-lG4run", "-lG4track", "-lG4intercoms",
        ]
        cmd = `$compiler -O2 -shared -fPIC -std=c++17 -I$(prefix)/include/Geant4 $rpath
               -L$(prefix)/lib $libs -o $library $source`
        run(cmd)
    end
    EM_CALCULATOR[] = Libdl.dlsym(Libdl.dlopen(library), :basismc_cross_section_per_volume)
    return EM_CALCULATOR[]
end

"""
    geant4_attenuation(material, energy_keV; kind = :total, physics = PhysicsSettings()) -> μ in 1/mm

The linear attenuation coefficient Geant4 itself uses for `material` at `energy_keV`, from the
same models every simulation samples (`G4EmCalculator`). `kind` is `:total` (the sum of the
three below; pair production is zero under 1 MeV), `:photoelectric`, `:compton` or `:rayleigh`.

This is the attenuation the Monte Carlo is built on; [`linear_attenuation`](@ref) gives NIST
XCOM's for comparison. The two agree to about 1 % for photoelectric absorption and Compton
scattering; Geant4's Livermore Rayleigh cross sections differ from XCOM's by up to ~20 % at
some energies. The first call builds a small C++ library (a few seconds, once per Geant4
version) and, if no simulation has run yet, starts Geant4.
"""
function geant4_attenuation(
        m::DetectorMaterial, energy_keV::Real; kind::Symbol = :total,
        physics::PhysicsSettings = PhysicsSettings()
    )
    kind === :total && return sum(
        geant4_attenuation(m, energy_keV; kind = k, physics)
            for k in keys(G4_PROCESS_NAMES)
    )
    haskey(G4_PROCESS_NAMES, kind) ||
        throw(ArgumentError("kind must be :total, :photoelectric, :compton or :rayleigh"))
    e = ENGINE[]
    if e === nothing || e.detector.material.name != m.name
        thickness = e !== nothing && e.detector isa SlabDetector ? e.detector.thickness_mm : 1.0
        engine(SlabDetector(m, thickness), physics)    # the material must be in the geometry
    elseif e.physics != physics
        engine(e.detector, physics)                     # throws: physics is fixed per process
    end
    f = em_calculator_symbol()
    μ = ccall(
        f, Float64, (Float64, Cstring, Ptr{Cvoid}),
        energy_keV * keV, G4_PROCESS_NAMES[kind], g4_material(m).cpp_object
    )
    return μ * mm
end

geant4_attenuation(key::Symbol, energy_keV::Real; kw...) =
    geant4_attenuation(DetectorMaterial(key), energy_keV; kw...)
