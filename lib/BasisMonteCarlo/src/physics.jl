# =============================================================================
# Physics settings
# =============================================================================
# Geant4 fixes its physics when the run manager is first initialised; it cannot be changed
# afterwards in the same process. BasisMonteCarlo therefore takes the settings once, when the
# engine is created, and refuses a different set later instead of silently ignoring it.
# =============================================================================

"""
    PhysicsSettings(;
        em = :option4,
        fluorescence = true,
        auger = false,
        pixe = false,
        production_cut_um = 1.0,
        electron_step_um = 1.0,
    )

The Geant4 electromagnetic physics for every run in this Julia process.

- `em = :option4`: `G4EmStandardPhysics_option4`, Geant4's most accurate electromagnetic
  constructor and the one the Geant4 collaboration recommends for medical physics. Below a few
  hundred keV it uses the Livermore/EPICS2014 photoelectric effect with shell-resolved
  cross sections, the Monash Compton model (electron binding and Doppler broadening) and
  Livermore Rayleigh scattering, with Penelope/Livermore electron ionisation. `:standard` is
  the faster high-energy default and is kept only for comparison.
- `fluorescence = true`: the characteristic x rays an atom emits when it refills an inner-shell
  vacancy, the K-fluorescence that makes a detector's efficiency drop above a K-edge. Switching
  it off is how a notebook shows what it does.
- `auger = false`: the Auger electrons the atom can emit instead, followed as particles through
  the full cascade. Off by default: they carry a few keV at most and stop within nanometres, so
  their energy is deposited where the vacancy was either way. Following them costs 3.4× the run
  time (CdTe, 60 keV: 21 000 against 71 000 photons per second) and moves the energy kept by
  about 0.2 % (36 keV: 0.8745 with, 0.8770 without), the few that escape through the front face.
- `pixe = false`: fluorescence after electron-impact ionisation (an electron knocking out a
  K-shell electron) rather than after photon absorption. A small second-order effect.
- `production_cut_um`: the range below which secondaries are not produced but deposited on the
  spot. 1 µm sits under Geant4's 990 eV floor in every detector material, so every secondary
  above about 1 keV is followed. De-excitation ignores the cut, so no fluorescence is lost to it.
- `electron_step_um`: the shortest step an electron is allowed to take in one go. Geant4's
  default (10 µm) lets an electron whose remaining range is below that cover all of it in a
  single straight step; in CdTe that is every electron a CT photon makes, so each one would be
  one hop whose length is only its net displacement. 1 µm makes electrons take several steps
  along a realistic path. It does not change how much energy stays in the crystal, but it sets
  *where*, which matters once the crystal is divided into dexels (charge sharing).

These are fixed when the first simulation runs; a different set in the same process throws an
error that says to restart Julia.
"""
Base.@kwdef struct PhysicsSettings
    em::Symbol = :option4
    fluorescence::Bool = true
    auger::Bool = false
    pixe::Bool = false
    production_cut_um::Float64 = 1.0
    electron_step_um::Float64 = 1.0
end

# Geant4.jl builds the physics list from a type with a one-argument (verbosity) constructor.
struct EMOption4Physics <: G4VUserPhysicsList
    function EMOption4Physics(verbose)
        pl = G4VModularPhysicsList()
        RegisterPhysics(pl, move!(G4EmStandardPhysics_option4(verbose)))
        return pl
    end
end

struct EMStandardPhysics <: G4VUserPhysicsList
    function EMStandardPhysics(verbose)
        pl = G4VModularPhysicsList()
        RegisterPhysics(pl, move!(G4EmStandardPhysics(verbose)))
        return pl
    end
end

function physics_type(p::PhysicsSettings)
    p.em === :option4 && return EMOption4Physics
    p.em === :standard && return EMStandardPhysics
    throw(ArgumentError("em must be :option4 or :standard"))
end

"""UI commands that apply `p`; they must run before the run manager initialises."""
ui_commands(p::PhysicsSettings) = [
    "/process/em/fluo $(p.fluorescence)",
    "/process/em/auger $(p.auger)",
    "/process/em/augerCascade $(p.auger)",
    "/process/em/pixe $(p.pixe)",
    "/process/em/deexcitationIgnoreCut true",
    "/run/setCut $(p.production_cut_um) um",
    "/process/eLoss/StepFunction 0.2 $(p.electron_step_um) um",
]
