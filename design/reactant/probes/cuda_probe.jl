using Reactant
println("CUDA_VISIBLE_DEVICES=", get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>"))
try
    ds = Reactant.devices(); println("devices: ", length(ds))
    for d in ds; try; println("  kind=", Reactant.XLA.device_kind(d)); catch; end; end
    x = Reactant.to_rarray(ones(Float32, 2048, 2048)); f = Reactant.@compile sync=true (a -> a * a)(x); y = Array(f(x))
    println("CUDA_PROBE_OK sum=", sum(y))
catch e
    println("CUDA_PROBE_FAIL: ", first(sprint(showerror, e), 300))
end
