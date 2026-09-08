using Pkg; Pkg.add("AbstractFFTs"; io=devnull)
using Reactant, Enzyme, AbstractFFTs
using Reactant: @trace
println("Reactant ", pkgversion(Reactant), "  Enzyme ", pkgversion(Enzyme), "  Julia ", VERSION)
println("backend devices: ", Reactant.devices())
probe(name, f) = try
    t = @elapsed r = f()
    println("PASS  ", rpad(name, 34), " ", round(t; digits=2), "s  ", r)
catch e
    println("FAIL  ", rpad(name, 34), " ", sprint(showerror, e)[1:min(end, 220)])
end
x = Reactant.to_rarray(rand(Float32, 4096))
f1(x) = sum(cumsum(x .^ 2))
probe("compile+run cumsum/broadcast/sum", () -> (@compile f1(x))(x))
g1(x) = Enzyme.gradient(Reverse, f1, x)[1]
probe("Enzyme reverse gradient in @compile", () -> begin g = (@compile g1(x))(x); (length(g), Array(g)[1:2]) end)
idx = Reactant.to_rarray(rand(1:4096, 2048))
h1(x, idx) = sum(x[idx])
probe("gather x[traced idx]", () -> (@compile h1(x, idx))(x, idx))
gh(x, idx) = Enzyme.gradient(Reverse, Const(h1), x, Const(idx))[1]
probe("gradient through gather (scatter)", () -> sum(Array((@compile gh(x, idx))(x, idx))))
A = Reactant.to_rarray(rand(Float32, 64, 64, 8))
i3 = Reactant.to_rarray(rand(1:64, 500)); j3 = Reactant.to_rarray(rand(1:64, 500)); k3 = Reactant.to_rarray(rand(1:8, 500))
h3(A, i, j, k) = sum(A[CartesianIndex.(i, j, k)])
probe("3D gather w/ traced CartesianIndex", () -> (@compile h3(A, i3, j3, k3))(A, i3, j3, k3))
S = Reactant.to_rarray(rand(Float32, 512, 16, 90))
ff(S) = real(sum(abs2, fft(S, 1)))
probe("fft along dim 1 (real->complex)", () -> (@compile ff(S))(S))
rf(S) = sum(abs2, rfft(S, 1))
probe("rfft along dim 1", () -> (@compile rf(S))(S))
function tl(x)
    acc = similar(x); acc .= 0
    @trace for i in 1:10
        acc = acc .+ x .* Float32(i)
    end
    sum(acc)
end
probe("@trace for loop (10 iters)", () -> (@compile tl(x))(x))
gtl(x) = Enzyme.gradient(Reverse, tl, x)[1]
probe("gradient through @trace for", () -> Array((@compile gtl(x))(x))[1])
function tw(x)
    n = Reactant.to_rarray(0) # placeholder to keep traced
    s = sum(x)
    i = 0
    @trace while s < 1f6
        s = s * 2f0
        i += 1
    end
    (s, i)
end
probe("@trace while (traced cond)", () -> (@compile tw(x))(x))
srt(x) = sum(sort(x)[1:10])
probe("sort", () -> (@compile srt(x))(x))
med(A) = sum(min.(max.(min.(A[:, :, 1:end-2], A[:, :, 2:end-1]), A[:, :, 3:end]), max.(A[:, :, 1:end-2], A[:, :, 2:end-1])))
probe("3-tap median via min/max network", () -> (@compile med(A))(A))
using LinearAlgebra
M = Reactant.to_rarray(rand(Float32, 2, 736))
gram(M) = sum(svd(M * M').S)
probe("svd of small matrix", () -> (@compile gram(M))(M))
rng = Reactant.ReactantRNG(Reactant.to_rarray(UInt64[1, 2]))
rr(rng) = sum(rand(rng, Float32, 100_000))
probe("Reactant RNG rand", () -> (@compile rr(rng))(rng))
# throughput sanity: fused elementwise+reduce on 34M elements (clinical sinogram size)
B = Reactant.to_rarray(rand(Float32, 736, 16, 720 * 4))
poly(B) = sum(-log.(max.(exp.(-B) .* 0.5f0 .+ exp.(-2f0 .* B) .* 0.5f0, 1f-10)))
pc = @compile sync=true poly(B)
pc(B); t = @elapsed pc(B)
println("PERF  34M-elem fused exp/log kernel (CPU XLA, $(Threads.nthreads()) julia threads): ", round(t*1000; digits=1), " ms")
