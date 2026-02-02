#####
##### System metadata
#####

struct BenchmarkMetadata
    julia_version::String
    oceananigans_version::String
    breeze_version::String
    architecture::String
    gpu_name::Union{String, Nothing}
    cuda_version::Union{String, Nothing}
    cpu_model::String
    num_threads::Int
    hostname::String
    timestamp::DateTime
end

function BenchmarkMetadata(arch)
    gpu_name = nothing
    cuda_version = nothing

    if arch isa Oceananigans.Architectures.GPU{CUDABackend}
        try
            gpu_name = CUDA.name(CUDA.device())
            cuda_version = string(CUDA.runtime_version())
        catch
            gpu_name = "Unknown GPU"
            cuda_version = "Unknown"
        end
    end

    # Get CPU model
    cpu_model = "$(Sys.cpu_info()[1].model) ($(Sys.CPU_NAME))"

    return BenchmarkMetadata(
        string(VERSION),
        string(pkgversion(Oceananigans)),
        string(pkgversion(Breeze)),
        string(typeof(arch)),
        gpu_name,
        cuda_version,
        cpu_model,
        Threads.nthreads(),
        gethostname(),
        now()
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::BenchmarkMetadata)
    println(io, "BenchmarkMetadata")
    println(io, "├── julia_version: ", m.julia_version)
    println(io, "├── oceananigans_version: ", m.oceananigans_version)
    println(io, "├── breeze_version: ", m.breeze_version)
    println(io, "├── architecture: ", m.architecture)
    if !isnothing(m.gpu_name)
        println(io, "├── gpu_name: ", m.gpu_name)
        println(io, "├── cuda_version: ", m.cuda_version)
    end
    println(io, "├── cpu_model: ", m.cpu_model)
    println(io, "├── num_threads: ", m.num_threads)
    println(io, "├── hostname: ", m.hostname)
    print(io,   "└── timestamp: ", m.timestamp)
end
