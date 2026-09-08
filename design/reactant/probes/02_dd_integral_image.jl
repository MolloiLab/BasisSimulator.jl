using Reactant, Enzyme, Statistics
using Reactant: @trace
probe(name, f) = try
    t = @elapsed r = f(); println("PASS  ", rpad(name, 40), " ", round(t; digits=2), "s  ", r)
catch e; println("FAIL  ", rpad(name, 40), " ", sprint(showerror, e)[1:min(end, 300)]); end

# --- fix-ups for the two earlier failures -----------------------------------
x = Reactant.to_rarray(rand(Float32, 4096))
function tl(x)
    acc = x .* 0f0
    @trace for i in 1:10
        acc = acc .+ x .* i        # promotion handles traced Int
    end
    sum(acc)
end
probe("@trace for (promotion, no Float32())", () -> (@compile tl(x))(x))
gtl(x) = Enzyme.gradient(Reverse, tl, x)[1]
probe("gradient through @trace for", () -> Array((@compile gtl(x))(x))[1])
A = Reactant.to_rarray(rand(Float32, 64, 64, 8))
i3 = rand(1:64, 500); j3 = rand(1:64, 500); k3 = rand(1:8, 500)
lin = Reactant.to_rarray(i3 .+ (j3 .- 1) .* 64 .+ (k3 .- 1) .* 64 * 64)
h3(A, lin) = sum(vec(A)[lin])
probe("3D gather via linear index", () -> (@compile h3(A, lin))(A, lin))
cf(x) = sum(floor.(Int32, x .* 3f0))
probe("floor.(Int32, traced)", () -> (@compile cf(x))(x))

# --- per-view cumsum distance-driven projection probe ------------------------
# Integral-image formulation: per slab, P += norm * [II(x_hi,z_hi) - II(x_lo,z_hi) - II(x_hi,z_lo) + II(x_lo,z_lo)]
# with II = 2-D cumulative sum and bilinear sampling at the projected cell corners. Same weights as
# the overlap sum in dd_fast.jl (piecewise-constant integral == difference of the cumulative integral).
const NX, NZ, NSLAB = 512, 64, 512
const NCOL, NROW = 736, 16

# separable fractional sampling: x-gather over (col,slab), then z-gather over (row,slab)
function sample_corner(II, cx, cz, nx, nz, ncol, nrow, nslab)
    ix0 = clamp.(floor.(Int32, cx), Int32(1), Int32(nx - 1))          # ncol × nslab
    fx  = cx .- ix0
    iz0 = clamp.(floor.(Int32, cz), Int32(1), Int32(nz - 1))          # nrow × nslab
    fz  = cz .- iz0
    kofs = reshape(Int32.((0:nz-1) .* nx), 1, nz, 1)
    sofs = reshape(Int32.((0:nslab-1) .* (nx * nz)), 1, 1, nslab)
    L0 = reshape(ix0, ncol, 1, nslab) .+ kofs .+ sofs                 # ncol × nz × nslab
    IIv = vec(II)
    G = reshape(IIv[vec(L0)], ncol, nz, nslab) .* reshape(1f0 .- fx, ncol, 1, nslab) .+
        reshape(IIv[vec(L0 .+ Int32(1))], ncol, nz, nslab) .* reshape(fx, ncol, 1, nslab)
    cofs = reshape(Int32.(0:ncol-1), ncol, 1, 1)
    sofs2 = reshape(Int32.((0:nslab-1) .* (ncol * nz)), 1, 1, nslab)
    L1 = cofs .+ reshape((iz0 .- Int32(1)) .* Int32(ncol), 1, nrow, nslab) .+ sofs2 .+ Int32(1)  # ncol × nrow × nslab
    Gv = vec(G)
    H = reshape(Gv[vec(L1)], ncol, nrow, nslab) .* reshape(1f0 .- fz, 1, nrow, nslab) .+
        reshape(Gv[vec(L1 .+ Int32(ncol))], ncol, nrow, nslab) .* reshape(fz, 1, nrow, nslab)
    return H
end

function dd_view(vol, ax, bx, az, bz, ulo, uhi, vlo, vhi, norm)
    nx, nz, nslab = size(vol); ncol = length(ulo); nrow = length(vlo)
    II = cumsum(cumsum(vol; dims = 1); dims = 2)
    cx_lo = reshape(ax, 1, :) .+ reshape(bx, 1, :) .* reshape(ulo, :, 1)   # ncol × nslab (in-graph geometry)
    cx_hi = reshape(ax, 1, :) .+ reshape(bx, 1, :) .* reshape(uhi, :, 1)
    cz_lo = reshape(az, 1, :) .+ reshape(bz, 1, :) .* reshape(vlo, :, 1)   # nrow × nslab
    cz_hi = reshape(az, 1, :) .+ reshape(bz, 1, :) .* reshape(vhi, :, 1)
    H = sample_corner(II, cx_hi, cz_hi, nx, nz, ncol, nrow, nslab) .-
        sample_corner(II, cx_lo, cz_hi, nx, nz, ncol, nrow, nslab) .-
        sample_corner(II, cx_hi, cz_lo, nx, nz, ncol, nrow, nslab) .+
        sample_corner(II, cx_lo, cz_lo, nx, nz, ncol, nrow, nslab)
    return dropdims(sum(H .* reshape(norm, 1, 1, nslab); dims = 3); dims = 3)      # ncol × nrow
end

vol  = rand(Float32, NX, NZ, NSLAB)
ax = 8f0 .+ 4f0 .* rand(Float32, NSLAB);  bx = 0.6f0 .+ 0.3f0 .* rand(Float32, NSLAB)
az = 2f0 .+ 2f0 .* rand(Float32, NSLAB);  bz = 0.6f0 .+ 0.3f0 .* rand(Float32, NSLAB)
ulo = collect(Float32, 0:NCOL-1) .* 0.65f0; uhi = ulo .+ 0.65f0
vlo = collect(Float32, 0:NROW-1) .* 0.9f0;  vhi = vlo .+ 0.9f0
norm = 0.05f0 .+ 0.01f0 .* rand(Float32, NSLAB)
args = (vol, ax, bx, az, bz, ulo, uhi, vlo, vhi, norm)
rargs = map(Reactant.to_rarray, args)

# CPU Julia reference of the same formula (plain arrays) for a parity check
ref = dd_view(args...)
println("ref stats: mean=", mean(ref), " max=", maximum(ref))

fwd = nothing
probe("compile dd_view (per view, 512²×64 vol → 736×16)", () -> (global fwd = @compile sync=true dd_view(rargs...); "ok"))
if fwd !== nothing
    out = fwd(rargs...); out = fwd(rargs...)
    t = @elapsed fwd(rargs...)
    println("PERF  forward per view: ", round(t*1000; digits=1), " ms  →  ×720 views ≈ ", round(t*720; digits=1), " s (CPU XLA)")
    println("PARITY forward vs plain-Julia same formula: max rel = ", maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref)))
end
loss(vol, rest...) = sum(dd_view(vol, rest...))
gradf(vol, rest...) = Enzyme.gradient(Reverse, Const(loss), vol, map(Const, rest)...)[1]
gf = nothing
probe("compile gradient wrt volume (Enzyme reverse)", () -> (global gf = @compile sync=true gradf(rargs...); "ok"))
if gf !== nothing
    g = gf(rargs...); g = gf(rargs...)
    t = @elapsed gf(rargs...)
    println("PERF  gradient per view: ", round(t*1000; digits=1), " ms  →  ×720 ≈ ", round(t*720; digits=1), " s (CPU XLA)")
    println("grad stats: sum=", sum(Array(g)), " nnz=", count(!=(0f0), Array(g)))
end
