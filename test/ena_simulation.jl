include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "..", "examples", "setups", "ena_simulation.jl"))
include(joinpath(@__DIR__, "..", "examples", "setups", "ena_diagnostics.jl"))

using Test
using Metal: Metal, MetalBackend

ena_test_architectures = Metal.functional() ? (default_arch, GPU(MetalBackend())) : (default_arch,)

@testset "ENA forcing lifecycle [$arch]" for arch in ena_test_architectures
    Oceananigans.defaults.FloatType = Float32
    mktempdir() do directory
        # Synthetic SAM-format inputs keep this test independent of network access.
        open(joinpath(directory, "snd"), "w") do io
            println(io, "z p theta q u v")
            for day in (199.25, 199.75)
                println(io, "$day 4 1019.3")
                println(io, "-9999 1040 292.2 11.2 5 -6")
                println(io, "-9999 950 295 8 5 -6")
                println(io, "-9999 800 305 2 5 -6")
                println(io, "-9999 600 330 0.1 5 -6")
            end
        end
        open(joinpath(directory, "lsf_time_varying"), "w") do io
            println(io, "z p tls qls u v w")
            for day in (199.25, 199.75)
                println(io, "$day 4 1019.3")
                for pressure in (1040, 950, 800, 600)
                    println(io, "-9999 $pressure -0.00001 -0.00000001 5 -6 -0.001")
                end
            end
        end
        write(joinpath(directory, "sfc_time_varying"),
              "day sst H LE stress\n199.25 294.9 10 100 0.0625\n199.75 294.9 10 100 0.0625\n")
        data = covert_ena_case(directory)
        open(joinpath(directory, "lsf"), "w") do io
            println(io, "z p tls qls u v w ug vg")
            for day in (199.0, 200.0)
                println(io, "$day, 2, 1019.3 day,levels,pres0")
                for height in (0, 4000)
                    println(io, "$height -999.9 0.00002 -0.00000002 3 4 -0.002 7 8")
                end
            end
        end
        write(joinpath(directory, "sfc"),
              "day sst H LE stress\n199.0 295 12 110 -999.9\n200.0 295 12 110 -999.9\n")
        lasso = lasso_ena_case("era5"; directory, common=data)
        @test lasso.profiles.u(1000, 0) == 7 # geostrophic, not nudging-target wind
        @test lasso.profiles.v(1000, 0) == 8
        @test lasso.profiles.moisture_tendency(1000, 0) == -2e-8
        @test lasso.profiles.subsidence(1000, 0) == -0.002
        @test lasso.surface.kinematic_stress(0, 0) == 0.0625
        @test lasso.surface.latent_heat_flux(0, 0) == 110
        radiation = TimeHeightProfile([0.0, 3000], [0.0, 2.0], [-0.02 -0.04; -0.02 -0.04])
        simulation = ena_simulation(arch; directory, size=(16, 16, 16), stop_time=2, radiation)
        @test simulation.model.microphysics isa P3Microphysics
        @test haskey(simulation.model.microphysical_fields, :ρnʳ)
        @test haskey(simulation.model.microphysical_fields, :ρqⁱ)
        drivers = simulation.callbacks[:ena_drivers].func
        @test_throws ErrorException drivers(simulation)
        wrong_date = ENACase(data.name, Date(2017, 7, 19), data.source, data.profiles, data.surface)
        @test_throws ArgumentError set!(simulation, wrong_date)
        @test !drivers.initialized[]
        storage = drivers.profiles.moisture_tendency.values
        set!(simulation, data)
        @test drivers.initialized[]
        @test simulation.model.radiation === nothing
        @test !haskey(simulation.callbacks, :ena_radiation)
        @test all(isapprox.(-0.02f0, Array(interior(drivers.radiative_heating))))
        ena_update_prescribed_radiation!(drivers.radiative_heating, drivers.prescribed_radiation, 1f0)
        @test all(isapprox.(-0.03f0, Array(interior(drivers.radiative_heating))))
        @test drivers.profiles.moisture_tendency.values === storage
        @test Array(storage) ≈ fill(-1f-8, size(storage))
        initial_temperature = Array(interior(simulation.model.temperature))
        set!(simulation, lasso)
        @test Array(storage) ≈ fill(-2f-8, size(storage))
        @test Array(interior(simulation.model.temperature)) == initial_temperature
        @test minimum(Oceananigans.Grids.znodes(simulation.model.grid, Center(), Center(), Center())) > 0
        output_directory = joinpath(directory, "diagnostics")
        diagnostics = add_ena_diagnostics!(simulation, output_directory; interval=1)
        run!(simulation)
        @test isfile(joinpath(output_directory, "turbulence_profiles.csv"))
        @test isfile(joinpath(output_directory, "radiative_heating.csv"))
        @test all(isfinite, Array(interior(diagnostics.turbulence.variance)))
        @test simulation.model.clock.time == 2
        @test isfinite(maximum(simulation.model.temperature))
        @test all(field -> all(isfinite, Array(interior(field))), values(Oceananigans.prognostic_fields(simulation.model)))
        # A horizontally uniform nonzero w has zero variance, not mean(w²).
        set!(simulation.model.velocities.w, 2)
        diagnostics(simulation)
        @test maximum(diagnostics.turbulence.w) > 0
        @test maximum(abs, Array(interior(diagnostics.turbulence.variance))) < 1e-6
        @test_throws ArgumentError set!(simulation, data)
    end
end
