include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using NumericalRadiation: NumericalRadiation
using Adapt: Adapt, adapt
using Oceananigans
using Test

const NumericalRadiationExt = Base.get_extension(Breeze, :BreezeNumericalRadiationExt)

# Reactant kernel arrays encode their dimensions in the type. Reproduce that property without
# requiring a compiler or a device for this storage test.
struct SizeTypedArray{T, D, S} <: AbstractArray{T, D}
    data :: Array{T, D}
end

SizeTypedArray(data::Array{T, D}) where {T, D} = SizeTypedArray{T, D, size(data)}(data)
Base.size(::SizeTypedArray{T, D, S}) where {T, D, S} = S
@inline Base.getindex(array::SizeTypedArray, i::Int...) = array.data[i...]
Adapt.adapt_storage(::Val{:sizes_in_types}, array::Array) = SizeTypedArray(array)

@testset "Radiation storage with sizes in array types [$FT]" for FT in (Float32, Float64)
    grid = RectilinearGrid(CPU(), FT; size = 4, x = 0, y = 0, z = (0, 3000), topology = (Flat, Flat, Bounded))
    extension = NumericalRadiationExt.materialize_column_extension(ColumnExtension(FT; layers = 3), grid, BackgroundAtmosphere())
    gas_model = NumericalRadiation.EcCKDTabulatedGasOpticsModel(names = (:h2o, :co2),
                                                                pressure_grid = FT[10_000, 100_000],
                                                                temperature_grid = FT[220, 300],
                                                                longwave_absorption = ones(FT, 2, 2, 2, 2),
                                                                shortwave_absorption = ones(FT, 2, 2, 2, 2))
    # Two batch rows of one column, 7 layers, an all-sky workspace (any cloud but `nothing`)
    workspace = NumericalRadiationExt.ColumnWorkspace(CPU(), FT, 1, 2, 4, 7, gas_model, :cloud, extension)
    workspace = adapt(Val(:sizes_in_types), workspace)

    @test eltype(workspace) === FT
    @test all(isconcretetype, fieldtypes(typeof(workspace)))
    @test size(workspace.reflectance) == (2, 7)
    @test size(workspace.stack_albedo) == (2, 8)
    @test size(workspace.interface_temperatures) == (2, 8)
    @test size(workspace.liquid_water_path) == (2, 7)
    @test size(workspace.stencil.pressure_index) == (2, 7)
    @test eltype(workspace.stencil.pressure_index) === Int32
    @test size(workspace.extension_pressure) == (2, 4)
    @test size(workspace.cos_zenith) == (4,)
    @test typeof(workspace.reflectance) !== typeof(workspace.stack_albedo)

    extension = workspace.extension
    @test all(isconcretetype, fieldtypes(typeof(extension)))
    @test size(extension.T) == (3,)
    @test size(extension.Δzᵃᵃᶜ) == (3,)
end
