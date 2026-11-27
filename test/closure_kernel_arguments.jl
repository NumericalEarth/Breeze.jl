include(joinpath(@__DIR__, "setup.jl"))

#####
##### The arguments of a GPU kernel share a parameter memory of about 32 KiB (31.996 KiB on sm_90). A kernel of the
##### turbulence closures whose arguments grow past it fails to launch on the GPU only, so the size of its arguments
##### on the device is measured here on the CPU: every array as a CUDA device array, the size every kernel of
##### `Breeze.TurbulenceClosures` receives during one time step of the fullest configurations.
#####

using Adapt: Adapt, adapt
using Breeze
using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, AllProcesses
using Oceananigans
using Oceananigans.Architectures: convert_to_device
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Grids: ColumnEnsembleSize
using Test

const TC = Breeze.TurbulenceClosures

# A CUDA device array, `CuDeviceArray{T, N}`: its pointer, its size in bytes, its dimensions and its length
struct DeviceArrayStandIn{T, N} <: DenseArray{T, N}
    pointer :: UInt64
    maxsize :: Int
    dims :: NTuple{N, Int}
    len :: Int
end

Base.size(a::DeviceArrayStandIn) = a.dims

struct DeviceArrayAdaptor end
Adapt.adapt_storage(::DeviceArrayAdaptor, a::Array{T, N}) where {T, N} = DeviceArrayStandIn{T, N}(0, sizeof(a), size(a), length(a))

device_arguments(::CPU, args) = adapt(DeviceArrayAdaptor(), args)
device_arguments(arch, args) = convert_to_device(arch, args)

# The kernels of a module and of its submodules: the functions for which `@kernel` defined a CPU kernel
function module_kernels(mod, kernels = Function[])
    for name in names(mod; all = true)
        isdefined(mod, name) || continue
        f = getfield(mod, name)
        if f isa Module && parentmodule(f) === mod && f !== mod
            module_kernels(f, kernels)
        elseif f isa Function && isdefined(mod, Symbol(:cpu_, name))
            push!(kernels, f)
        end
    end
    return kernels
end

const ClosureKernel = Union{map(typeof, module_kernels(TC))...}
const RECORDING = Ref(false)
const ARGUMENT_SIZES = Dict{Symbol, Int}()

function record_argument_size!(arch, kernel!, args)
    adapted = device_arguments(arch, args)
    @test isbits(adapted)
    name = nameof(kernel!)
    ARGUMENT_SIZES[name] = max(get(ARGUMENT_SIZES, name, 0), sizeof(adapted))
    return nothing
end

# Measures the arguments of the closure kernels while `RECORDING[]`, then launches them as Oceananigans does
function Oceananigans.Utils._launch!(arch, grid, workspec, kernel!::ClosureKernel, kernel_args::Vararg{Any, N}; kw...) where N
    RECORDING[] && record_argument_size!(arch, kernel!, kernel_args)
    return invoke(Oceananigans.Utils._launch!, Tuple{Any, Any, Any, Any, Vararg{Any, N}}, arch, grid, workspec, kernel!,
                  kernel_args...; kw...)
end

function measured_closure_kernels(closure)
    grid = RectilinearGrid(default_arch; size = ColumnEnsembleSize(Nz = 16, ensemble = (3, 1), Hz = 3), z = (0, 1600),
                           topology = (Flat, Flat, Bounded))
    closure = closure isa Function ? closure(grid) : closure
    condensation = SubgridCondensation(Breeze.TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation, negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure = 101325, potential_temperature = 290)
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics, closure)
    set!(model; θ = z -> 290 + 3e-3 * max(z - 800, 0), qᵗ = z -> 0.012 - 2e-6 * z, u = 5, ρe = z -> 0.36 * exp(-z / 800))
    empty!(ARGUMENT_SIZES)
    RECORDING[] = true
    try
        time_step!(model, 10)
    finally
        RECORDING[] = false
    end
    return copy(ARGUMENT_SIZES)
end

# Every optional component of the closure and of its distribution, with the per-column coefficients of an ensemble of
# columns, and with each of the two saturated fractions of the static stability that read the subgrid statistics
function fullest_tke_closure(static_stability, solver)
    distribution = DoubleGaussian(; third_moment = TC.TKEBudgetThirdMoment(; location = Face()),
                                  assumed_pdf = TC.SkewedScalarDoubleGaussianPDF(),
                                  variance_transport = TC.ScalarVarianceTransport(; component_third_moment = false),
                                  liquid_flux = TC.LiquidFluxVarianceRelation())
    return TKEBasedTurbulenceClosure(; subgrid_distribution = distribution, static_stability, solver,
                                     heating_budget_flux = HeatingBudgetFlux(), total_water_budget_flux = TC.TotalWaterBudgetFlux())
end

per_column(template) = grid -> per_column_closure(grid, reshape([with_parameters(template; ℂⁱ = 1.5 + n / 100) for n in 1:3], 3, 1))

# The default leaves a margin of a quarter below the limit, for the context of the kernel and for growth
const maximum_argument_size = 24 * 1024

@testset "Device arguments of the closure kernels below $(maximum_argument_size ÷ 1024) KiB" begin
    @test sizeof(DeviceArrayStandIn{Float64, 3}) == sizeof(CUDA.CuDeviceArray{Float64, 3, CUDA.AS.Global})
    configurations = (
        "default TKE closure" => TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian()),
        "fullest, condensate flux fraction, bracketed solve" =>
            per_column(fullest_tke_closure(SaturationWeightedStaticStability(CondensateFluxFraction()), BracketedFractionSolve(2))),
        "fullest, microphysics saturated fraction" =>
            per_column(fullest_tke_closure(SaturationWeightedStaticStability(MicrophysicsSaturatedFraction()), SinglePass())),
        "SimplifiedHigherOrderClosure" => SimplifiedHigherOrderClosure())
    for (label, closure) in configurations
        @testset "$label" begin
            sizes = measured_closure_kernels(closure)
            @test !isempty(sizes)
            largest = argmax(sizes)
            @info "$label: the largest closure kernel arguments, $(largest) with $(sizes[largest]) B"
            for (name, size) in sizes
                @test size < maximum_argument_size
                size < maximum_argument_size || @info "$name: $size B"
            end
        end
    end
end
