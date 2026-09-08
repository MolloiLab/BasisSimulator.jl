# FDK stage under Reactant: gather backprojection vs the dense tiled one (tiles 8/16/32),
# forward and Enzyme gradient, plus parity against the host gather.
#   NV=64 VIEWS=100 VB=25 julia --project=envs/reactant -t 4 --heap-size-hint=5G design/reactant/probes/bench_fdk.jl
using Reactant, Enzyme, BasisSimulator, Random, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); VB = envi("VB", 25); NZ = envi("NZ", 2)
TILES = parse.(Int, split(get(ENV, "TILES", "8,16,32"), ","))
GATHER = envi("GATHER", 1) == 1
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc)
geom = BS.CTGeometry(scanner; n_angles = VIEWS, fov_cm = 35.0, z_cm = 0.5, collimation_mm = 5.0)
plan = BSF.fbp_plan(geom, (NV, NV, NZ); T = Float32)
sino = rand(MersenneTwister(1), Float32, geom.n_cols, geom.n_rows, VIEWS); sino_r = Reactant.to_rarray(sino)
ref = BSF.fdk(sino, plan; view_batch = 1, dense = false)
say(@sprintf("NV=%d VIEWS=%d VB=%d rows=%d cols=%d recon z=%d", NV, VIEWS, VB, geom.n_rows, geom.n_cols, NZ))
for t in TILES
    st, w, tx, ty = BSF.fdk_tile_windows(plan, t)
    say(@sprintf("tile %d: %d×%d tiles, window %d of %d columns", t, length(tx), length(ty), w, geom.n_cols))
end
cases = Any[]
GATHER && push!(cases, ("gather", false, 16))
for t in TILES; push!(cases, ("dense t$t", true, t)); end
for (name, dense, tile) in cases
    fwd = s -> BSF.fdk(s, BSF._fbp_on_device(plan, s); view_batch = VB, dense = dense, tile = tile)
    loss = s -> sum(fwd(s) .^ 2)
    g = s -> Enzyme.gradient(Reverse, loss, s)
    tc = @elapsed cf = @compile sync = true fwd(sino_r); cf(sino_r); tf = @elapsed out = cf(sino_r)
    rel = maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref))
    tc2 = @elapsed cg = @compile sync = true g(sino_r); cg(sino_r); tg = @elapsed cg(sino_r)
    say(@sprintf("%-10s forward %.3f s (compile %.0f s) | gradient %.3f s (compile %.0f s) → ratio %.1f | parity vs host gather %.1e", name, tf, tc, tg, tc2, tg / tf, rel))
end
say("FDK_BENCH_DONE")
