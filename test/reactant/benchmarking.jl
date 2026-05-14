#####
##### Reactant forward forecast benchmark
#####
##### Standalone script for benchmarking a fixed-step Breeze compressible
##### forecast. Microphysics is omitted for now while the Reactant lowering
##### issue in the CloudMicrophysics path is isolated.
#####
##### Run from the repository root with the test environment:
#####
#####   julia --project=test test/reactant/benchmarking.jl
#####
##### Useful options:
#####   --backend=cpu                 # default: gpu
#####   --size=128x128x64             # default: 128x128x64
#####   --steps=10                    # number of compiled forecast steps
#####   --nrepeat=3                   # profiling repeats
#####   --warmup=1                    # profiling warmup repeats
#####   --profile                     # print kernel/op profiling tables
#####   --profile_dir=/path/to/traces # save Reactant profiler output
#####

using Reactant
using Reactant: @trace

# Load CUDA after Reactant so ReactantCUDAExt can hook CUDA compilation.
using CUDA: CUDA

using Breeze
using Breeze: CompressibleDynamics, ExplicitTimeStepping
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Printf: @printf

option(name, default) = begin
    prefix = "--" * name * "="
    index = findfirst(arg -> startswith(arg, prefix), ARGS)
    index === nothing ? default : split(ARGS[index], "="; limit = 2)[2]
end

has_flag(name) = "--" * name in ARGS

function parse_size(value)
    parts = split(lowercase(value), "x")
    length(parts) == 3 || error("Invalid --size=$value. Use Nx x Ny x Nz, for example 128x128x64.")
    return Tuple(parse(Int, part) for part in parts)
end

const grid_size = parse_size(option("size", "128x128x64"))
const nsteps = parse(Int, option("steps", "10"))
const nrepeat = parse(Int, option("nrepeat", "3"))
const warmup = parse(Int, option("warmup", "1"))
const backend = lowercase(option("backend", "gpu"))
const dt_value = parse(Float64, option("dt", "0.05"))
const profile_dir = option("profile_dir", nothing)
const detailed_profile = has_flag("profile")

backend in ("cpu", "gpu") || error("Invalid --backend=$backend. Use cpu or gpu.")

struct PrecisionCase
    name :: String
    FT :: DataType
    compile_options :: Reactant.CompileOptions
end

function reactant_compile_options(; multifloat = nothing)
    return Reactant.CompileOptions(;
        sync = true,
        raise = false,
        raise_first = false,
        multifloat,
    )
end

const precision_cases = (
    PrecisionCase("Float64", Float64, reactant_compile_options()),
    PrecisionCase("Float32", Float32, reactant_compile_options()),
    PrecisionCase("MultiFloat_BF16x2", Float32, reactant_compile_options(;
        multifloat = Reactant.MultiFloatOptions(source = "f32", target = "bf16", limbs = 2))),
)

function compressible_forecast_model(::Type{FT}; size = grid_size) where FT
    Oceananigans.defaults.FloatType = FT

    Lx = FT(12_000)
    Ly = FT(12_000)
    Lz = FT(3_000)

    grid = RectilinearGrid(ReactantState();
        size,
        x = (zero(FT), Lx),
        y = (zero(FT), Ly),
        z = (zero(FT), Lz),
        halo = (5, 5, 5),
        topology = (Periodic, Periodic, Bounded),
    )

    dynamics = CompressibleDynamics(ExplicitTimeStepping();
        surface_pressure = FT(101_325),
        reference_potential_temperature = FT(300),
    )

    advection = WENO(FT; order = 5)

    model = AtmosphereModel(grid; dynamics, advection)

    xc = Lx / 2
    yc = Ly / 2
    zc = Lz / 3
    horizontal_radius_squared = (Lx / 8)^2
    vertical_radius_squared = (Lz / 8)^2

    bubble(x, y, z) = exp(-((x - xc)^2 + (y - yc)^2) / horizontal_radius_squared -
                          (z - zc)^2 / vertical_radius_squared)

    θᵢ(x, y, z) = FT(300) + FT(0.5) * bubble(x, y, z)

    set!(model;
        ρ = FT(1),
        θ = θᵢ,
        u = FT(5),
        v = zero(FT),
        w = zero(FT),
    )

    return model
end

function forward_forecast!(model, Δt, steps)
    @trace mincut = true checkpointing = false track_numbers = false for _ in 1:steps
        time_step!(model, Δt)
    end
    return nothing
end

runtime_seconds(result) = hasproperty(result, :runtime) ? result.runtime : result.runtime_ns / 1e9

function compile_forward_forecast!(case::PrecisionCase, model, Δt)
    compile_start = time_ns()
    compiled_forecast! = Reactant.@compile compile_options = case.compile_options forward_forecast!(
        model, Δt, nsteps)
    compile_time = (time_ns() - compile_start) / 1e9
    return compiled_forecast!, compile_time
end

function run_case(case::PrecisionCase)
    model = compressible_forecast_model(case.FT)
    Δt = case.FT(dt_value)

    println()
    println("-"^82)
    println("Precision: ", case.name)
    println("Float type: ", case.FT)
    println("Compile options: raise=false, multifloat=", case.compile_options.multifloat)
    println("Grid: ", grid_size, " PPB, steps: ", nsteps, ", Δt: ", Δt, ", backend: ", backend)
    println("-"^82)

    compiled_forecast!, compile_time = compile_forward_forecast!(case, model, Δt)
    @printf("Ahead-of-time compile: %.6e s\n", compile_time)

    if detailed_profile
        Reactant.@profile nrepeat = nrepeat warmup = warmup profile_dir = profile_dir compiled_forecast!(
            model, Δt, nsteps)
        return nothing
    else
        result = Reactant.@timed nrepeat = nrepeat warmup = warmup profile_dir = profile_dir compiled_forecast!(
            model, Δt, nsteps)
        println(result)
        return (; profiling = result, compile_time)
    end
end

function main()
    Reactant.set_default_backend(backend)

    println("Breeze Reactant forward forecast benchmark")
    println("Dry compressible dynamics; microphysics temporarily omitted.")
    println("Reactant raising disabled; the multifloat case converts f32 -> bf16 with 2 limbs.")

    results = Pair{String, Any}[]

    for case in precision_cases
        result = run_case(case)
        result === nothing || push!(results, case.name => result)
    end

    if !isempty(results)
        println()
        println("="^82)
        println("Summary")
        println("="^82)
        @printf("%-24s %14s %14s %14s\n", "precision", "runtime/step", "runtime", "compile")
        for (name, result) in results
            runtime = runtime_seconds(result.profiling)
            @printf("%-24s %14.6e %14.6e %14.6e\n",
                    name, runtime / nsteps, runtime, result.compile_time)
        end
    end

    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
