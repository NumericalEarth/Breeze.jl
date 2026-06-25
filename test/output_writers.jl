using Test
using Breeze
using Oceananigans
using Oceananigans.Architectures: on_architecture, CPU
using Oceananigans.OutputWriters: jldopen
using Logging: Warn

# Regression test for https://github.com/NumericalEarth/Breeze.jl/issues/643.
# Before the fix, Breeze's `default_included_properties` override included `:grid`,
# which collided with Oceananigans' automatic grid serialization and produced a
# "grid is already present within this group" warning during JLD2Writer init.
@testset "JLD2Writer initializes without warnings [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), extent=(100, 100, 100))
    model = AtmosphereModel(grid)

    filepath = tempname() * ".jld2"
    outputs = (; ρu = model.momentum.ρu)

    writer = JLD2Writer(model, outputs;
                        filename = filepath,
                        schedule = IterationInterval(1),
                        overwrite_existing = true)

    try
        @test_logs min_level=Warn Oceananigans.initialize!(writer, model)
        @test isfile(writer.filepath)

        jldopen(writer.filepath, "r") do file
            @test haskey(file, "serialized/grid")
            @test haskey(file, "serialized/thermodynamic_constants")
            @test file["serialized/grid"] == on_architecture(CPU(), grid)
        end
    finally
        rm(writer.filepath; force=true)
    end
end
