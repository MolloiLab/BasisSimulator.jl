# =============================================================================
# The Geant4 engine
# =============================================================================
# Geant4 allows one run manager per process. BasisMonteCarlo keeps one application for the life
# of the Julia session, created on first use; a new geometry re-initialises it, a new energy only
# changes the gun. All the bookkeeping is done in a stepping action rather than with sensitive
# detectors, because Geant4.jl does not re-attach sensitive detectors after a geometry change.
# =============================================================================

# What left the detector, by kind (the `kind` column of the escape tally)
const ESCAPE_TRANSMITTED = 0x01   # the primary photon, never interacted
const ESCAPE_SCATTERED = 0x02     # the primary photon after Compton or Rayleigh scattering
const ESCAPE_FLUORESCENCE = 0x03  # a characteristic x ray (K or L fluorescence)
const ESCAPE_BREMSSTRAHLUNG = 0x04
const ESCAPE_ELECTRON = 0x05      # an electron (photo-, Compton or Auger) that got out
const ESCAPE_KINDS = (:transmitted, :scattered, :fluorescence, :bremsstrahlung, :electron)

# The processes that count as the primary photon interacting
const PHOTON_INTERACTIONS = ("phot", "compt", "Rayl", "conv")

"""
    TrackStep

One step of one particle, recorded by [`record_tracks`](@ref): event number, track and parent
ids, what the particle is (`:primary`, `:fluorescence`, `:bremsstrahlung`, `:electron`), the
start and end points in mm (the frame of [`SlabDetector`](@ref)), the process that ended the
step (`"phot"`, `"compt"`, `"Rayl"`, `"eIoni"`, `"Transportation"`, …), the energy deposited in
the step and the kinetic energy at its start, in keV.
"""
struct TrackStep
    event::Int
    track::Int
    parent::Int
    particle::Symbol
    start::NTuple{3, Float64}
    stop::NTuple{3, Float64}
    process::String
    edep_keV::Float64
    energy_keV::Float64
end

"""Where each bit of energy was left in the detector, one entry per Geant4 step that deposited."""
struct DepositLog
    event::Vector{Int32}
    x_mm::Vector{Float32}
    y_mm::Vector{Float32}
    z_mm::Vector{Float32}
    energy_keV::Vector{Float32}
end

DepositLog() = DepositLog(Int32[], Float32[], Float32[], Float32[], Float32[])

Base.length(d::DepositLog) = length(d.event)

function Base.empty!(d::DepositLog)
    foreach(empty!, (d.event, d.x_mm, d.y_mm, d.z_mm, d.energy_keV))
    return d
end

Base.copy(d::DepositLog) = DepositLog(copy(d.event), copy(d.x_mm), copy(d.y_mm), copy(d.z_mm), copy(d.energy_keV))

# The photon scatterings logged by `VertexLog` (its `process` column)
const VERTEX_COMPTON = 0x01
const VERTEX_RAYLEIGH = 0x02

"""
Every Compton and Rayleigh scattering of a photon in the detector (or phantom), one entry each:
the photon, the process (`VERTEX_COMPTON` or `VERTEX_RAYLEIGH`), whether it is the incident photon's
first (`first`), where it happened (mm), and the photon's energy (keV) and direction just before
it. Forced detection
([`ForcedScatterReadout`](@ref)) scores every one of them toward the detectors.
"""
struct VertexLog
    event::Vector{Int32}
    process::Vector{UInt8}
    first::BitVector
    position_mm::Vector{NTuple{3, Float32}}
    energy_keV::Vector{Float32}
    direction::Vector{NTuple{3, Float32}}
end

VertexLog() = VertexLog(Int32[], UInt8[], BitVector(), NTuple{3, Float32}[], Float32[], NTuple{3, Float32}[])

Base.length(v::VertexLog) = length(v.event)

function Base.empty!(v::VertexLog)
    foreach(empty!, (v.event, v.process, v.first, v.position_mm, v.energy_keV, v.direction))
    return v
end

Base.copy(v::VertexLog) = VertexLog(
    copy(v.event),
    copy(v.process),
    copy(v.first),
    copy(v.position_mm),
    copy(v.energy_keV),
    copy(v.direction),
)

mutable struct RunData <: G4JLSimulationData
    # The current event
    edep::Float64
    interacted::Bool
    scattered::Bool
    event::Int

    # The run: one entry per photon
    edeps::Vector{Float32}
    interacted_flags::BitVector

    # The run: one entry per particle that escaped
    escape_event::Vector{Int32}
    escape_kind::Vector{UInt8}
    escape_energy::Vector{Float32}
    escape_backward::BitVector
    escape_position::Vector{NTuple{3, Float32}}
    escape_direction::Vector{NTuple{3, Float32}}

    # Optional: every energy deposit (record_deposits = true)
    recording_deposits::Bool
    deposits::DepositLog

    # Optional: every photon scattering (record_vertices = true)
    recording_vertices::Bool
    vertices::VertexLog

    # Photon energies, when they are drawn from a spectrum
    photon_energies::Vector{Float32}

    # Optional: every step (record_tracks only)
    recording::Bool
    steps::Vector{TrackStep}

    # Cached Geant4 pointers
    gamma::Ptr{Cvoid}
    electron::Ptr{Cvoid}
end

RunData() = RunData(
    0.0,
    false,
    false,
    0,
    Float32[],
    BitVector(),
    Int32[],
    UInt8[],
    Float32[],
    BitVector(),
    NTuple{3, Float32}[],
    NTuple{3, Float32}[],
    false,
    DepositLog(),
    false,
    VertexLog(),
    Float32[],
    false,
    TrackStep[],
    C_NULL,
    C_NULL,
)

function reset!(
        d::RunData;
        recording::Bool = false,
        recording_deposits::Bool = false,
        recording_vertices::Bool = false,
    )
    d.event = 0
    empty!(d.edeps)
    empty!(d.interacted_flags)
    empty!(d.escape_event)
    empty!(d.escape_kind)
    empty!(d.escape_energy)
    empty!(d.escape_backward)
    empty!(d.escape_position)
    empty!(d.escape_direction)
    d.recording_deposits = recording_deposits
    empty!(d.deposits)
    d.recording_vertices = recording_vertices
    empty!(d.vertices)
    empty!(d.photon_energies)
    d.recording = recording
    empty!(d.steps)
    return d
end

# -----------------------------------------------------------------------------
# Primary generator: one photon per event, energy and entry point set between runs
# -----------------------------------------------------------------------------

mutable struct BeamData <: G4JLGeneratorData
    energy::Float64                    # Geant4 units; the last photon's, when drawn from a spectrum
    beam::Beam
    rng::Xoshiro                       # entry points and spectrum draws; reseeded for every run
    spectrum_keV::Vector{Float64}      # bin centres of the spectrum (empty: monoenergetic)
    spectrum_cdf::Vector{Float64}
    spectrum_bin_keV::Float64
    gamma::CxxPtr{G4ParticleDefinition}
end

function beam_generator()
    data = BeamData(
        60keV,
        PencilBeam(),
        Xoshiro(1),
        Float64[],
        Float64[],
        0.0,
        CxxPtr{G4ParticleDefinition}(C_NULL),
    )

    function init(data::BeamData, ::G4JLApplication)
        data.gamma = FindParticle("gamma")
        return nothing
    end

    function generate(evt::G4Event, data::BeamData)::Nothing
        if !isempty(data.spectrum_cdf)
            k = min(
                searchsortedfirst(data.spectrum_cdf, rand(data.rng)),
                length(data.spectrum_keV),
            )
            data.energy = (data.spectrum_keV[k] + (rand(data.rng) - 0.5) * data.spectrum_bin_keV) * keV
        end
        (x, y, z), u = beam_ray(data.beam, data.rng)
        # momentum = E along the beam direction
        primary = G4PrimaryParticle(
            data.gamma,
            data.energy * u[1],
            data.energy * u[2],
            data.energy * u[3],
        )
        start = G4ThreeVector(x * mm, y * mm, z * mm)
        vertex = G4PrimaryVertex(start, 0ns)
        SetPrimary(vertex, move!(primary))
        AddPrimaryVertex(evt, move!(vertex))
        return nothing
    end

    return G4JLPrimaryGenerator(
        "BasisMonteCarloBeam",
        data;
        init_method = init,
        generate_method = generate,
    )
end

# -----------------------------------------------------------------------------
# User actions
# -----------------------------------------------------------------------------

process_name(p) = p == C_NULL ? "" : String(GetProcessName(p))

xyz(v) = (x(v), y(v), z(v))

function in_detector(point)
    volume = GetVolume(GetTouchable(point))
    return volume != C_NULL && GetCopyNo(volume) == CRYSTAL_COPY
end

function particle_kind(d::RunData, track)
    GetTrackID(track) == 1 && return :primary
    def = GetDefinition(track).cpp_object
    def == d.electron && return :electron
    def == d.gamma || return :other
    return process_name(GetCreatorProcess(track)) == "eBrem" ? :bremsstrahlung : :fluorescence
end

function begin_event(::G4Event, app::G4JLApplication)::Nothing
    d = getSIMdata(app)
    d.edep = 0.0
    d.interacted = false
    d.scattered = false
    d.event += 1
    return nothing
end

function end_event(::G4Event, app::G4JLApplication)::Nothing
    d = getSIMdata(app)
    push!(d.edeps, d.edep / keV)
    push!(d.interacted_flags, d.interacted)
    gen = app.generator.data
    isempty(gen.spectrum_cdf) || push!(d.photon_energies, gen.energy / keV)
    return nothing
end

function stepping(step::G4Step, app::G4JLApplication)::Nothing
    d = getSIMdata(app)
    pre = GetPreStepPoint(step)
    post = GetPostStepPoint(step)
    edep = GetTotalEnergyDeposit(step)
    track = GetTrack(step)
    tid = GetTrackID(track)
    in_crystal = in_detector(pre)

    # Only energy left in the crystal is signal (not in a collimator wall)
    in_crystal && (d.edep += edep)

    # Did the primary photon interact in the detector?
    if tid == 1 && !d.interacted && in_crystal
        if process_name(GetProcessDefinedStep(post)) in PHOTON_INTERACTIONS
            d.interacted = true
        end
    end

    # Leaving the detector: tally what escaped
    if in_crystal && !in_detector(post)
        code = escape_code(particle_kind(d, track), d.interacted)
        if code != 0x00
            push!(d.escape_event, d.event)
            push!(d.escape_kind, code)
            push!(d.escape_energy, GetKineticEnergy(post) / keV)
            direction = xyz(GetMomentumDirection(post))
            push!(d.escape_backward, direction[3] < 0)
            push!(d.escape_position, Float32.(xyz(GetPosition(post)) ./ mm))
            push!(d.escape_direction, Float32.(direction))
        end
    end

    # Every deposit, when the dexel readout needs positions (at the step's end: steps are ≤ 1 µm)
    if d.recording_deposits && edep > 0 && in_crystal
        px, py, pz = xyz(GetPosition(post))
        push!(d.deposits.event, d.event)
        push!(d.deposits.x_mm, px / mm)
        push!(d.deposits.y_mm, py / mm)
        push!(d.deposits.z_mm, pz / mm)
        push!(d.deposits.energy_keV, edep / keV)
    end

    # Every photon scattering, for forced detection (the energy and direction it arrived with)
    if d.recording_vertices && in_crystal && GetDefinition(track).cpp_object == d.gamma
        process = process_name(GetProcessDefinedStep(post))
        code = process == "compt" ? VERTEX_COMPTON : process == "Rayl" ? VERTEX_RAYLEIGH : 0x00
        if code != 0x00
            push!(d.vertices.event, d.event)
            push!(d.vertices.process, code)
            push!(d.vertices.first, tid == 1 && !d.scattered)
            tid == 1 && (d.scattered = true)
            push!(d.vertices.position_mm, Float32.(xyz(GetPosition(post)) ./ mm))
            push!(d.vertices.energy_keV, GetKineticEnergy(pre) / keV)
            push!(d.vertices.direction, Float32.(xyz(GetMomentumDirection(pre))))
        end
    end

    # Every step, when drawing
    if d.recording
        step_record = TrackStep(
            d.event,
            tid,
            GetParentID(track),
            particle_kind(d, track),
            xyz(GetPosition(pre)) ./ mm,
            xyz(GetPosition(post)) ./ mm,
            process_name(GetProcessDefinedStep(post)),
            edep / keV,
            GetKineticEnergy(pre) / keV,
        )
        push!(d.steps, step_record)
    end

    return nothing
end

function escape_code(kind::Symbol, primary_interacted::Bool)
    kind === :primary && return primary_interacted ? ESCAPE_SCATTERED : ESCAPE_TRANSMITTED
    kind === :fluorescence && return ESCAPE_FLUORESCENCE
    kind === :bremsstrahlung && return ESCAPE_BREMSSTRAHLUNG
    kind === :electron && return ESCAPE_ELECTRON
    return 0x00
end

# -----------------------------------------------------------------------------
# The session engine
# -----------------------------------------------------------------------------

mutable struct Engine
    app::Any
    physics::PhysicsSettings
    detector::AbstractDetector
end

const ENGINE = Ref{Union{Nothing, Engine}}(nothing)

same_geometry(a::AbstractDetector, b::AbstractDetector) = false
same_geometry(a::SlabDetector, b::SlabDetector) = (
    a.material.name == b.material.name &&
        a.material.density == b.material.density &&
        a.material.mass_fractions == b.material.mass_fractions &&
        a.thickness_mm == b.thickness_mm &&
        a.width_mm == b.width_mm &&
        a.shape === b.shape &&
        a.collimator == b.collimator &&
        a.septa == b.septa
)

function ui(cmd::AbstractString)
    status = ApplyCommand(G4UImanager!GetUIpointer(), String(cmd))
    status == 0 || error("Geant4 rejected the command '$cmd'")
    return nothing
end

const QUIET_COMMANDS = (
    "/control/verbose 0",
    "/run/verbose 0",
    "/event/verbose 0",
    "/tracking/verbose 0",
    "/process/em/verbose 0",
    "/process/had/verbose 0",
)

function engine(detector::AbstractDetector, physics::PhysicsSettings)
    e = ENGINE[]

    # First use in this process: build the application and fix the physics
    if e === nothing
        app = G4JLApplication(;
            detector = G4Geometry(detector),
            simdata = RunData(),
            generator = beam_generator(),
            nthreads = 0,
            verbose = 0,
            physics_type = physics_type(physics),
            stepaction_method = stepping,
            begineventaction_method = begin_event,
            endeventaction_method = end_event,
        )
        configure(app)
        foreach(ui, QUIET_COMMANDS)
        foreach(ui, ui_commands(physics))
        initialize(app)

        d = getSIMdata(app)
        d.gamma = FindParticle("gamma").cpp_object
        d.electron = FindParticle("e-").cpp_object

        # A run with no photons builds the physics tables
        beamOn(app, 0)
        e = ENGINE[] = Engine(app, physics, detector)

        # Different physics: impossible without a new process
    elseif e.physics != physics
        error(
            """
            Geant4 physics is fixed for the life of the process; this session runs
            $(e.physics). Restart Julia to use $(physics).
            """
        )

        # Different geometry: re-initialise and rebuild the tables for the new material
    elseif !same_geometry(e.detector, detector)
        reinitialize(e.app, G4Geometry(detector))

        # Rebuild every model's tables: a material that left the geometry and came back would
        # otherwise reach the electron multiple scattering (Goudsmit–Saunderson, with its Mott
        # correction) without its data, and crash
        ui("/run/physicsModified")
        beamOn(e.app, 0)
        e.detector = detector
    end

    return e
end

function set_beam!(e::Engine, energy_keV::Real, beam::Beam, seed::Integer)
    gen = e.app.generator.data
    gen.energy = energy_keV * keV
    gen.beam = beam
    empty!(gen.spectrum_keV)
    empty!(gen.spectrum_cdf)

    # The same (seed, energy) always gives the same photons: Geant4 takes two 31-bit seeds,
    # the beam's entry points their own generator
    h = hash((seed, Float64(energy_keV)))
    ui("/random/setSeeds $(1 + h % 2147483646) $(1 + (h >> 32) % 2147483646)")
    Random.seed!(gen.rng, h)
    return e
end
