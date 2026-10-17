include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using NumericalRadiation: NumericalRadiation
using Adapt: Adapt, adapt
using Test

const RadiationStorageExt = Base.get_extension(Breeze, :BreezeNumericalRadiationExt)

# Reactant kernel arrays encode their dimensions in the type. Reproduce that
# property without requiring a compiler or a device for this storage test.
struct SizeTypedColumnMatrix{T, N, M} <: AbstractMatrix{T}
    data :: Matrix{T}
end

SizeTypedColumnMatrix(data::Matrix{T}) where T = SizeTypedColumnMatrix{T, size(data, 1), size(data, 2)}(data)
Base.size(::SizeTypedColumnMatrix{T, N, M}) where {T, N, M} = (N, M)
@inline Base.getindex(array::SizeTypedColumnMatrix, i::Int, j::Int) = array.data[i, j]
Adapt.adapt_storage(::Val{:column_sizes}, array::Matrix) = SizeTypedColumnMatrix(array)

@testset "Radiation storage with sizes in array types" for FT in (Float32, Float64)
    original = RadiationStorageExt.SpectralColumns(CPU(), FT, 2, 4, nothing)
    columns = adapt(Val(:column_sizes), original)
    @test eltype(columns) === FT
    @test all(isconcretetype, fieldtypes(typeof(columns)))
    @test size(columns.pressure_layers) == (2, 4)
    @test size(columns.pressure_interfaces) == (2, 5)
    @test typeof(columns.pressure_layers) !== typeof(columns.pressure_interfaces)
    @test typeof(columns.pressure_index) !== typeof(columns.source_index)
    # ShortwaveColumnScratch's separate layer/interface types are covered by
    # NumericalRadiation's own tests; this test needs no unreleased API there.
end
