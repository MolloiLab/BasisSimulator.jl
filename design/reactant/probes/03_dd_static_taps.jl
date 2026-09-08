using Reactant, Enzyme, Statistics, LinearAlgebra
probe(name, f) = try
    t = @elapsed r = f(); println("PASS  ", rpad(name, 44), " ", round(t; digits=2), "s  ", r)
catch e; println("FAIL  ", rpad(name, 44), " ", sprint(showerror, e)[1:min(end, 400)]); end

const NX, NZ, NSLAB = 512, 64, 512
const NCOL, NROW = 736, 16
KX = 0; KZ = 0               # static tap counts, set from geometry below (ceil(max span)+1)

# overlap of voxel i (spanning [i-1, i]) with interval [lo, hi], zero outside 1..n
@inline ovl(lo, hi, i, fi, n) = ifelse((i >= 1) & (i <= n), max(0f0, min(hi, fi) - max(lo, fi - 1f0)), 0f0)

# ---------- forward: A · vol  (per view) ----------
# transverse map per (col, slab): c = ax[slab] + bx[slab]*u   ; longitudinal-z map per (row, slab): c = az[slab] + bz[slab]*v
function dd_forward(vol, ax, bx, az, bz, ulo, uhi, vlo, vhi, norm)
    nx, nz, nslab = size(vol); ncol = length(ulo); nrow = length(vlo)
    # --- z stage: Z[it, row, slab] = Σ_dz oz · vol[it, k0+dz, slab]
    czlo = reshape(az, 1, :) .+ reshape(bz, 1, :) .* reshape(vlo, :, 1)        # nrow × nslab
    czhi = reshape(az, 1, :) .+ reshape(bz, 1, :) .* reshape(vhi, :, 1)
    k0   = floor.(Int32, czlo) .+ Int32(1)                                      # first overlapping voxel
    volv = vec(vol)
    xofs = reshape(Int32.(0:nx-1), nx, 1, 1)
    sofs = reshape(Int32.((0:nslab-1) .* (nx * nz)), 1, 1, nslab)
    Z = nothing
    for dz in 0:KZ-1
        k  = k0 .+ Int32(dz)                                                    # nrow × nslab
        w  = ovl.(czlo, czhi, k, k .* 1f0, nz)
        kc = clamp.(k, Int32(1), Int32(nz))
        L  = xofs .+ reshape((kc .- Int32(1)) .* Int32(nx), 1, nrow, nslab) .+ sofs .+ Int32(1)   # nx × nrow × nslab
        term = reshape(volv[vec(L)], nx, nrow, nslab) .* reshape(w, 1, nrow, nslab)
        Z = Z === nothing ? term : Z .+ term
    end
    # --- x stage: P[col,row] = Σ_slab norm[slab] Σ_dx ox · Z[i0+dx, row, slab]
    cxlo = reshape(ax, 1, :) .+ reshape(bx, 1, :) .* reshape(ulo, :, 1)        # ncol × nslab
    cxhi = reshape(ax, 1, :) .+ reshape(bx, 1, :) .* reshape(uhi, :, 1)
    i0   = floor.(Int32, cxlo) .+ Int32(1)
    Zv   = vec(Z)
    rofs = reshape(Int32.((0:nrow-1) .* nx), 1, nrow, 1)
    sofs2 = reshape(Int32.((0:nslab-1) .* (nx * nrow)), 1, 1, nslab)
    P = nothing
    for dx in 0:KX-1
        i  = i0 .+ Int32(dx)                                                    # ncol × nslab
        w  = ovl.(cxlo, cxhi, i, i .* 1f0, nx) .* reshape(norm, 1, :)
        ic = clamp.(i, Int32(1), Int32(nx))
        L  = reshape(ic, ncol, 1, nslab) .+ rofs .+ sofs2                       # ncol × nrow × nslab
        term = reshape(Zv[vec(L)], ncol, nrow, nslab) .* reshape(w, ncol, 1, nslab)
        P = P === nothing ? term : P .+ term
    end
    return dropdims(sum(P; dims = 3); dims = 3)
end

# ---------- transpose: Aᵀ · sino  (per view), gather form, no scatter ----------
# voxel i (transverse) overlaps cells col with [cxlo,cxhi] ∩ [i-1,i] ≠ ∅; cells are uniform in u so
# col0 = floor((i-1 - ax)/(bx*du)) - 1 ... use KX' = KX taps (scale ≈ 1 ⇒ ≤ 3 cells per voxel; 5 is safe)
function dd_transpose(sino, ax, bx, az, bz, ulo, uhi, vlo, vhi, norm, nx, nz, du, dv, u0, v0)
    ncol, nrow = size(sino); nslab = length(ax)
    # --- x stage: X[it, row, slab] = Σ_dc ox(col) · sino[col, row]
    it = reshape(collect(Float32, 1:nx), nx, 1)
    col0 = floor.(Int32, ((it .- 1f0) .- reshape(ax, 1, :)) ./ (reshape(bx, 1, :) .* du) .- (u0 / du)) .+ Int32(1)   # nx × nslab
    sv = vec(sino)
    rofs = reshape(Int32.((0:nrow-1) .* ncol), 1, nrow, 1)
    X = nothing
    for dc in 0:KXT-1
        col = col0 .+ Int32(dc)                                                # nx × nslab
        colf = col .* 1f0
        clo = reshape(ax, 1, :) .+ reshape(bx, 1, :) .* (u0 .+ (colf .- 1f0) .* du)
        chi = reshape(ax, 1, :) .+ reshape(bx, 1, :) .* (u0 .+ colf .* du)
        w  = ifelse.((col .>= 1) .& (col .<= ncol), max.(0f0, min.(chi, it) .- max.(clo, it .- 1f0)), 0f0) .* reshape(norm, 1, :)
        cc = clamp.(col, Int32(1), Int32(ncol))
        L  = reshape(cc, nx, 1, nslab) .+ rofs                                  # nx × nrow × nslab (sino index independent of slab)
        term = reshape(sv[vec(L)], nx, nrow, nslab) .* reshape(w, nx, 1, nslab)
        X = X === nothing ? term : X .+ term
    end
    # --- z stage: B[it, ip, slab] = Σ_dr oz(row) · X[it, row, slab]
    ip = reshape(collect(Float32, 1:nz), nz, 1)
    row0 = floor.(Int32, ((ip .- 1f0) .- reshape(az, 1, :)) ./ (reshape(bz, 1, :) .* dv) .- (v0 / dv)) .+ Int32(1)  # nz × nslab
    Xv = vec(X)
    xofs = reshape(Int32.(0:nx-1), nx, 1, 1)
    sofs = reshape(Int32.((0:nslab-1) .* (nx * nrow)), 1, 1, nslab)
    B = nothing
    for dr in 0:KZT-1
        row = row0 .+ Int32(dr)                                                # nz × nslab
        rowf = row .* 1f0
        rlo = reshape(az, 1, :) .+ reshape(bz, 1, :) .* (v0 .+ (rowf .- 1f0) .* dv)
        rhi = reshape(az, 1, :) .+ reshape(bz, 1, :) .* (v0 .+ rowf .* dv)
        w  = ifelse.((row .>= 1) .& (row .<= nrow), max.(0f0, min.(rhi, ip) .- max.(rlo, ip .- 1f0)), 0f0)
        rc = clamp.(row, Int32(1), Int32(nrow))
        L  = xofs .+ reshape((rc .- Int32(1)) .* Int32(nx), 1, nz, nslab) .+ sofs .+ Int32(1)   # nx × nz × nslab
        term = reshape(Xv[vec(L)], nx, nz, nslab) .* reshape(w, 1, nz, nslab)
        B = B === nothing ? term : B .+ term
    end
    return B
end

# ---------- exact Float64 reference: direct overlap double loop (what the oracle kernel does) ----------
function dd_forward_ref(vol, ax, bx, az, bz, ulo, uhi, vlo, vhi, norm)
    nx, nz, nslab = size(vol); ncol = length(ulo); nrow = length(vlo)
    P = zeros(Float64, ncol, nrow)
    for s in 1:nslab, row in 1:nrow, col in 1:ncol
        clo = ax[s] + bx[s]*ulo[col]; chi = ax[s] + bx[s]*uhi[col]
        zlo = az[s] + bz[s]*vlo[row]; zhi = az[s] + bz[s]*vhi[row]
        acc = 0.0
        for i in max(1, floor(Int, clo)):min(nx, ceil(Int, chi))
            ox = max(0.0, min(chi, i) - max(clo, i-1)); ox == 0 && continue
            for k in max(1, floor(Int, zlo)):min(nz, ceil(Int, zhi))
                oz = max(0.0, min(zhi, k) - max(zlo, k-1)); oz == 0 && continue
                acc += ox*oz*Float64(vol[i, k, s])
            end
        end
        P[col, row] += norm[s]*acc
    end
    P
end

# geometry chosen so a cell spans ≤ ~2 voxels transversally (scale 0.6–0.9 voxel/cell-unit × 0.65 units)
vol  = rand(Float32, NX, NZ, NSLAB)
ax = 8f0 .+ 4f0 .* rand(Float32, NSLAB);  bx = 0.9f0 .+ 0.4f0 .* rand(Float32, NSLAB)
az = 2f0 .+ 2f0 .* rand(Float32, NSLAB);  bz = 0.9f0 .+ 0.4f0 .* rand(Float32, NSLAB)
ulo = collect(Float32, 0:NCOL-1) .* 0.65f0; uhi = ulo .+ 0.65f0
vlo = collect(Float32, 0:NROW-1) .* 0.9f0;  vhi = vlo .+ 0.9f0
norm = 0.05f0 .+ 0.01f0 .* rand(Float32, NSLAB)
global KX = ceil(Int, maximum(bx) * 0.65f0) + 1; global KZ = ceil(Int, maximum(bz) * 0.9f0) + 1
global KXT = ceil(Int, 1 / (minimum(bx) * 0.65f0)) + 1; global KZT = ceil(Int, 1 / (minimum(bz) * 0.9f0)) + 1
println("TAPS   forward KX=", KX, " KZ=", KZ, "   transpose KXT=", KXT, " KZT=", KZT)
args = (vol, ax, bx, az, bz, ulo, uhi, vlo, vhi, norm)
rargs = map(Reactant.to_rarray, args)

t = @elapsed ref = dd_forward_ref(args...); println("ref (Float64 direct overlap loops) took ", round(t; digits=1), "s; mean=", mean(ref))
fwd = nothing
probe("compile static-tap forward", () -> (global fwd = @compile sync=true dd_forward(rargs...); "ok"))
out = fwd(rargs...); out = fwd(rargs...); t = @elapsed fwd(rargs...)
println("PERF  forward per view: ", round(t*1000; digits=1), " ms  → ×720 ≈ ", round(t*720; digits=1), " s (CPU XLA)")
println("PARITY forward vs Float64 direct-overlap reference: max rel = ", maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref)), "  mean rel = ", mean(abs.(Array(out) .- ref)) / mean(abs.(ref)))
println("TAPS   max voxels/cell transverse = ", maximum(ceil.(Int, bx .* 0.65f0) .+ 1), ", z = ", maximum(ceil.(Int, bz .* 0.9f0) .+ 1))

sino = rand(Float32, NCOL, NROW); rsino = Reactant.to_rarray(sino)
dd_transpose_c(s, a...) = dd_transpose(s, a..., NX, NZ, uhi[1]-ulo[1], vhi[1]-vlo[1], ulo[1], vlo[1])
tr = nothing
probe("compile static-tap transpose (gather form)", () -> (global tr = @compile sync=true dd_transpose_c(rsino, rargs[2:end]...); "ok"))
bp = tr(rsino, rargs[2:end]...); bp = tr(rsino, rargs[2:end]...); t = @elapsed tr(rsino, rargs[2:end]...)
println("PERF  transpose per view: ", round(t*1000; digits=1), " ms  → ×720 ≈ ", round(t*720; digits=1), " s (CPU XLA)")
lhs = dot(Float64.(Array(out)), Float64.(sino)); rhs = dot(Float64.(vol), Float64.(Array(bp)))
println("ADJOINT dot-product test <A v, s> = ", lhs, "  <v, Aᵀ s> = ", rhs, "  rel diff = ", abs(lhs - rhs) / abs(lhs))

# Enzyme reverse through the static-tap forward (no custom rule) for comparison
loss(vol, rest...) = sum(dd_forward(vol, rest...))
gradf(vol, rest...) = Enzyme.gradient(Reverse, Const(loss), vol, map(Const, rest)...)[1]
gf = nothing
probe("compile Enzyme reverse of static-tap forward", () -> (global gf = @compile sync=true gradf(rargs...); "ok"))
if gf !== nothing
    g = gf(rargs...); g = gf(rargs...); t = @elapsed gf(rargs...)
    println("PERF  Enzyme-generic gradient per view: ", round(t*1000; digits=1), " ms (scatter adjoint) vs gather-form transpose above")
    ones_s = Reactant.to_rarray(ones(Float32, NCOL, NROW))
    bp1 = Array(tr(ones_s, rargs[2:end]...))
    println("CHECK  Enzyme gradient == gather-form transpose of ones: max rel = ", maximum(abs.(Array(g) .- bp1)) / maximum(abs.(bp1)))
end

# ---------- batched views: vectorize the per-view graph over B views via mapslices-free broadcasting ----------
const NB = 8
function dd_forward_batched(vol, axB, bxB, azB, bzB, ulo, uhi, vlo, vhi, normB)   # geometry arrays: nslab × NB
    outs = ntuple(b -> dd_forward(vol, axB[:, b], bxB[:, b], azB[:, b], bzB[:, b], ulo, uhi, vlo, vhi, normB[:, b]), NB)
    return cat(outs...; dims = 3)
end
axB = repeat(ax, 1, NB); bxB = repeat(bx, 1, NB); azB = repeat(az, 1, NB); bzB = repeat(bz, 1, NB); normB = repeat(norm, 1, NB)
rB = map(Reactant.to_rarray, (vol, axB, bxB, azB, bzB, ulo, uhi, vlo, vhi, normB))
fb = nothing
probe("compile batched forward (8 views)", () -> (global fb = @compile sync=true dd_forward_batched(rB...); "ok"))
if fb !== nothing
    ob = fb(rB...); ob = fb(rB...); t = @elapsed fb(rB...)
    println("PERF  batched forward: ", round(t*1000/NB; digits=1), " ms/view  → ×720 ≈ ", round(t/NB*720; digits=1), " s (CPU XLA)")
end
