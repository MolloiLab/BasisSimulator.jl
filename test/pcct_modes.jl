# The NAEOTOM Alpha's two modes against BasisMonteCarlo's tables: standard (2 × 2 digitally binned)
# and UHR (native dexels). Air scans: the counts, their Fano factors and the correlation of
# neighbours. CPU only. The binned table (lib/BasisMonteCarlo/tables) is reference data for the
# test; BasisSimulator itself reads only its native table.

using Serialization: deserialize
using Statistics: var

const _BINNED_TABLE = joinpath(pkgdir(BS), "lib", "BasisMonteCarlo", "tables", "naeotom_alpha_cdte_binned2x2_v1.jls")

function _alpha_air_draws(scanner)
    air = BS.Phantom(zeros(UInt8, 8, 8, 2), Dict{Int, Any}(0 => BS.XA.Materials.air), (0.1, 0.1, 0.1))
    opts(seed) = BS.SimOptions(; use_scatter = false, use_heel_effect = false, use_focal_spot = false, seed)
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 20.0, views = 40)
    recon = BS.ReconOptions(; matrix_size = (64, 64, 1), fov_cm = 10.0, z_cm = 0.2)
    ws = BS.create_workspace(scanner, protocol, opts(1), recon, air)
    a = [copy(x) for x in BS.simulate!(ws, air, protocol, opts(1); report_dose = false).raw_counts]
    b = BS.simulate!(ws, air, protocol, opts(2); report_dose = false).raw_counts
    return (; ws, a = sum(Float64.(x) for x in a), b = sum(Float64.(x) for x in b), protocol)
end

_quiet(s; kw...) = BS.PCCTScanner(
    s; bowtie_filter = :none, pileup = false,
    scatter_correction = false, energy_thresholds = [20.0, 35.0, 55.0, 70.0], kw...
)

@testset "NAEOTOM Alpha: standard and UHR modes" begin
    std_mode = _alpha_air_draws(_quiet(BS.NAEOTOM_ALPHA; detector_cols = 48, detector_rows = 12))
    uhr_mode = _alpha_air_draws(_quiet(BS.NAEOTOM_ALPHA_UHR; detector_cols = 96, detector_rows = 24))

    # a standard pixel is four native dexels: four times the counts in air
    @test sum(std_mode.a) / length(std_mode.a) ≈ 4 * sum(uhr_mode.a) / length(uhr_mode.a) rtol = 2.0e-3

    # UHR: each native dexel is Poisson; neighbours across the fan covary by their shared photons
    d = (uhr_mode.a .- uhr_mode.b) ./ sqrt(2)
    m = sum(uhr_mode.a) / length(uhr_mode.a)
    @test sum(abs2, d) / length(d) / m ≈ 1 atol = 0.02
    classes = uhr_mode.ws.noise_classes
    λ = [sum(uhr_mode.ws.I0_cpu[:, :, b]) for b in 1:4]
    predicted = sum(p.c1 * λ[p.b] for p in classes.pairs if (p.dx, p.dy) == (1, 0)) / sum(λ)
    corr = sum(d[1:(end - 1), :, :] .* d[2:end, :, :]) / sum(abs2, d[1:(end - 1), :, :])
    @test corr ≈ predicted atol = 0.01
    @test predicted > 0.03

    # standard: shared photons count twice in the digital sum, as the binned table says
    ds = (std_mode.a .- std_mode.b) ./ sqrt(2)
    fano = sum(abs2, ds) / length(ds) / (sum(std_mode.a) / length(std_mode.a))
    t = deserialize(_BINNED_TABLE)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        BS.SimOptions(), std_mode.protocol; scanner = BS.NAEOTOM_ALPHA,
    )
    wt = BS._table_weights(Float64.(t["energies_keV"]), e, w)
    types = [findfirst(==(s), t["type_labels"]) for s in t["type_map"]]
    second = sum(wt[k] * sum(t["pair"][ty, k, 2, 2, 1, 1] for ty in types) for k in eachindex(wt))
    mean_ = sum(wt[k] * sum(t["mean"][ty, k, 1] for ty in types) for k in eachindex(wt))
    @test fano > 1.05
    @test fano ≈ second / mean_ rtol = 0.03
end
