#####
##### ecCKD versus RRTMGP radiation-update benchmark
#####
##### Run on a CUDA node from the repository root:
#####
#####   julia +1.12 --project=test benchmarking/radiation_update_benchmark.jl
#####
##### Optional positional filters are backend (`ecckd` or `rrtmgp`), float type (`f32` or `f64`),
##### and horizontal size (`32` or `128`), in any order.
#####

using Breeze
using Breeze.AtmosphereModels: _update_radiation!, compute_auxiliary_variables!
using CUDA
using ClimaComms
using CloudMicrophysics
using Dates: DateTime
using NCDatasets
using NumericalRadiation: NumericalRadiation
using Oceananigans
using Oceananigans.Units
using Printf
using RRTMGP
using Statistics: median

const CloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .CloudMicrophysicsExt: OneMomentCloudMicrophysics

const Nz = 64
const TOP = 16kilometers
const CLOUD_BOTTOM = 1kilometer
const CLOUD_TOP = 1.5kilometers
const NREPEAT = 7

const BACKENDS = (ecckd = EcCKDOptics(clouds = CloudScatteringTables()),
                  rrtmgp = AllSkyOptics())

function selected_cases(args)
    backends = :ecckd in Symbol.(args) ? (:ecckd,) : :rrtmgp in Symbol.(args) ? (:rrtmgp,) : keys(BACKENDS)
    float_types = "f32" in args ? (Float32,) : "f64" in args ? (Float64,) : (Float32, Float64)
    horizontal_sizes = "32" in args ? (32,) : "128" in args ? (128,) : (32, 128)
    return Iterators.product(backends, float_types, horizontal_sizes)
end

function radiation_model(grid, backend)
    optics = BACKENDS[backend]
    common = (; background_atmosphere = BackgroundAtmosphere(; CO₂ = 420e-6),
              surface_temperature = 300,
              surface_emissivity = 0.98,
              surface_albedo = 0.1,
              solar_constant = 1361,
              solar_position = FixedCosineZenith(0.5))

    if backend === :ecckd
        return RadiativeTransferModel(grid, optics, ThermodynamicConstants(); common..., column_extension = nothing)
    else
        return RadiativeTransferModel(grid, optics, ThermodynamicConstants(); common...)
    end
end

held_liquid_microphysics(FT) =
    OneMomentCloudMicrophysics(FT;
        cloud_formation = NonEquilibriumCloudFormation(ConstantRateCondensateFormation(zero(FT))))

function benchmark_model(FT, Nxy, backend)
    grid = RectilinearGrid(GPU(), FT;
                           size = (Nxy, Nxy, Nz),
                           x = (0, Nxy * 250),
                           y = (0, Nxy * 250),
                           z = (0, TOP),
                           topology = (Periodic, Periodic, Bounded))

    radiation = radiation_model(grid, backend)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = held_liquid_microphysics(FT)
    clock = Clock(time = DateTime(2024, 6, 21, 12))
    model = AtmosphereModel(grid; clock, dynamics, microphysics,
                            formulation = :LiquidIcePotentialTemperature, radiation)

    θ(x, y, z) = 300 + 5e-3 * z
    qᵗ(x, y, z) = 0.015 * exp(-z / 2500)
    qᶜˡ(x, y, z) = ifelse(CLOUD_BOTTOM < z < CLOUD_TOP, 0.5e-3, 0)
    set!(model; θ, qᵗ, qᶜˡ)
    compute_auxiliary_variables!(model)

    return model, radiation
end

# Count unique CUDA allocations reachable from the radiation model. Array wrappers such as
# Oceananigans Fields and views are reduced to their parent CuArray, so shared storage is counted once.
function cuda_storage(array)
    array isa CuArray && return array
    array isa AbstractArray || return nothing
    storage = parent(array)
    storage === array && return nothing
    return cuda_storage(storage)
end

function gpu_array_bytes(object, seen = IdDict{Any, Nothing}())
    storage = cuda_storage(object)
    if !isnothing(storage)
        haskey(seen, storage) && return 0
        seen[storage] = nothing
        return sizeof(eltype(storage)) * length(storage)
    elseif object isa AbstractArray
        isbitstype(eltype(object)) && return 0
        return sum(x -> gpu_array_bytes(x, seen), object; init = 0)
    elseif object isa Number || object isa Symbol || object isa AbstractString || object isa Function ||
           object isa Module || object isa Type || isnothing(object)
        return 0
    elseif object isa Tuple || object isa NamedTuple
        return sum(x -> gpu_array_bytes(x, seen), object; init = 0)
    end

    T = typeof(object)
    isstructtype(T) || return 0
    if ismutabletype(T)
        haskey(seen, object) && return 0
        seen[object] = nothing
    end
    return sum(name -> gpu_array_bytes(getfield(object, name), seen), fieldnames(T); init = 0)
end

function benchmark_update(backend, FT, Nxy; nrepeat = NREPEAT)
    CUDA.reclaim()
    model, radiation = benchmark_model(FT, Nxy, backend)
    columns = Nxy * Nxy
    owned_gpu_bytes = gpu_array_bytes(radiation)

    CUDA.@sync _update_radiation!(radiation, model)
    samples = Vector{Float64}(undef, nrepeat)
    for n in eachindex(samples)
        samples[n] = @elapsed CUDA.@sync _update_radiation!(radiation, model)
    end

    seconds = median(samples)
    result = (; backend, grid = "$(Nxy)x$(Nxy)x$(Nz)", FT = string(FT),
              milliseconds = 1e3 * seconds,
              milliseconds_per_column = 1e3 * seconds / columns,
              radiation_gpu_bytes = owned_gpu_bytes,
              samples_ms = 1e3 .* samples)

    model = radiation = nothing
    GC.gc(true)
    CUDA.reclaim()
    return result
end

function print_header()
    @printf("%-8s %-12s %-8s %14s %16s %16s\n",
            "backend", "grid", "FT", "median ms", "ms / column", "radiation MiB")
end

function print_result(result)
    @printf("%-8s %-12s %-8s %14.3f %16.8f %16.2f\n",
            result.backend, result.grid, result.FT, result.milliseconds,
            result.milliseconds_per_column, result.radiation_gpu_bytes / 2^20)
    @info "radiation update samples" backend=result.backend grid=result.grid FT=result.FT samples_ms=result.samples_ms
end

function main(args = ARGS)
    CUDA.functional() || error("CUDA is not functional; run this benchmark on a CUDA GPU node")
    @info "radiation update benchmark" device=CUDA.name(CUDA.device()) repeats=NREPEAT
    print_header()
    for (backend, FT, Nxy) in selected_cases(args)
        print_result(benchmark_update(backend, FT, Nxy))
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
