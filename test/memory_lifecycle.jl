module FakeMemoryBackend

mutable struct FakeDeviceArray{T} <: AbstractVector{T}
    data::Vector{T}
    freed::Bool
end

Base.size(a::FakeDeviceArray) = size(a.data)
Base.getindex(a::FakeDeviceArray, i::Int) = a.data[i]

function unsafe_free!(a::FakeDeviceArray)
    a.freed = true
    nothing
end

struct FakeDevice
    currentAllocatedSize::Int
    recommendedMaxWorkingSetSize::Int
end

device(::FakeDeviceArray) = FakeDevice(1024, 4096)

end

@testset "Explicit backend-memory lifecycle" begin
    cpu = zeros(Float32, 4)
    @test BS.release_backend!((cpu = cpu,)) == 0
    @test length(cpu) == 4

    a = FakeMemoryBackend.FakeDeviceArray(zeros(Float32, 8), false)
    b = FakeMemoryBackend.FakeDeviceArray(zeros(Float32, 4), false)
    nested = (buffers = [a, b], alias = a, cpu = cpu)
    snap = BS.backend_memory_snapshot(a)
    @test snap.backend == :FakeMemoryBackend
    @test snap.device_allocated_bytes == 1024
    @test snap.device_working_set_bytes == 4096
    @test BS.release_backend!(nested; collect = false) == 2
    @test a.freed
    @test b.freed

    estimate = BS.estimate_pcct_workspace_bytes(
        (1200, 20, 1200), (512, 512, 16), 4, 2,
    )
    @test estimate.gpu_bytes > 5 * 2^30
    @test estimate.host_bytes > 0
    tiny = BS.estimate_pcct_workspace_bytes((8, 2, 4), (4, 4, 2), 4, 1)
    checked = BS.check_pcct_workspace_budget(cpu, tiny)
    @test checked.gpu_bytes == tiny.gpu_bytes
    @test checked.host_bytes == tiny.host_bytes
    constrained = FakeMemoryBackend.FakeDeviceArray(zeros(Float32, 1), false)
    @test_throws ErrorException BS.check_pcct_workspace_budget(
        constrained, tiny,
    )
end
