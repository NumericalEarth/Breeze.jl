include(joinpath(@__DIR__, "setup.jl"))

using Test
using Breeze
using Breeze.AtmosphereModels: total_density
using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets, P3ProcessRates,
                                                       compute_p3_process_rates,
                                                       p3_grid_process_inputs
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Fields: interior, location

function p3_process_rate_test_model(FT)
    grid = RectilinearGrid(default_arch, FT; size = (2, 2, 8), x = (0, 200), y = (0, 200), z = (0, 800))
    microphysics = P3Microphysics(FT; cloud = CloudDroplets(FT; number_concentration = 75e6))
    model = AtmosphereModel(grid; microphysics)

    # A cloud layer in the upper half of a subsaturated domain, rain throughout, and an
    # updraft that vanishes at the impenetrable top and bottom.
    cloud(x, y, z) = ifelse(z > 400, FT(1e-3), zero(FT))
    updraft(x, y, z) = sinpi(z / 800) / 2
    set!(model; θ = 285, qᵛ = 0.005, qᶜˡ = cloud, qʳ = 1e-4, nʳ = 1e5, w = updraft,
         enforce_mass_conservation = false)

    return model
end

@testset "P3 process-rate diagnostics [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    model = p3_process_rate_test_model(FT)
    grid = model.grid
    Nz = size(grid, 3)

    autoconversion = MicrophysicalProcessRate(model, :autoconversion)
    accretion = MicrophysicalProcessRate(model, :accretion)
    condensation = MicrophysicalProcessRate(model, :condensation)
    rain_evaporation = MicrophysicalProcessRate(model, :rain_evaporation)

    for op in (autoconversion, accretion, condensation, rain_evaporation)
        @test op isa Oceananigans.AbstractOperations.KernelFunctionOperation
        @test location(op) == (Center, Center, Center)
    end

    @test_throws ArgumentError MicrophysicalProcessRate(model, :not_a_process)
    @test_throws ArgumentError MicrophysicalProcessRate(AtmosphereModel(grid), :autoconversion)

    Sᵃᶜⁿᵛ = Field(autoconversion)
    Sᵃᶜᶜ = Field(accretion)
    Sᶜᵒⁿᵈ = Field(condensation)
    Sᵉᵛᵃᵖ = Field(rain_evaporation)
    foreach(compute!, (Sᵃᶜⁿᵛ, Sᵃᶜᶜ, Sᶜᵒⁿᵈ, Sᵉᵛᵃᵖ))

    @testset "Rates match compute_p3_process_rates" begin
        μ = model.microphysical_fields
        ρ = total_density(model.dynamics)
        constants = model.thermodynamic_constants
        for k in (2, Nz - 1)
            rates = @allowscalar begin
                inputs = p3_grid_process_inputs(1, 1, k, grid, model.microphysics, μ,
                                                model.formulation, model.dynamics, constants,
                                                ρ, model.velocities)
                compute_p3_process_rates(model.microphysics, inputs.ρ, inputs.ℳ, inputs.𝒰, constants,
                                         inputs.properties, inputs.surface_temperature,
                                         inputs.temperature_tendency, inputs.pressure_tendency)
            end
            @test rates isa P3ProcessRates{FT}
            @test @allowscalar(Sᵃᶜⁿᵛ[1, 1, k]) ≈ rates.autoconversion
            @test @allowscalar(Sᵃᶜᶜ[1, 1, k]) ≈ rates.accretion
            @test @allowscalar(Sᶜᵒⁿᵈ[1, 1, k]) ≈ rates.condensation
            @test @allowscalar(Sᵉᵛᵃᵖ[1, 1, k]) ≈ rates.rain_evaporation
        end
    end

    @testset "Physical sanity" begin
        cloudy = (Nz÷2 + 1):Nz
        clear = 1:(Nz÷2)
        Sᵃᶜⁿᵛ_data = Array(interior(Sᵃᶜⁿᵛ))
        Sᵃᶜᶜ_data = Array(interior(Sᵃᶜᶜ))
        @test eltype(Sᵃᶜⁿᵛ_data) == FT
        @test all(isfinite, Sᵃᶜⁿᵛ_data)
        @test all(Sᵃᶜⁿᵛ_data .≥ 0)
        @test all(Sᵃᶜⁿᵛ_data[:, :, clear] .== 0)
        @test all(Sᵃᶜⁿᵛ_data[:, :, cloudy] .> 0)
        @test all(Sᵃᶜᶜ_data[:, :, clear] .== 0)
        @test all(Sᵃᶜᶜ_data[:, :, cloudy] .> 0)
        @test all(Array(interior(Sᵉᵛᵃᵖ)) .≥ 0)
    end

    @testset "Precipitation production rate" begin
        P = precipitation_rate(model, :liquid)
        compute!(P)
        @test Array(interior(P)) ≈ Array(interior(Sᵃᶜⁿᵛ)) .+ Array(interior(Sᵃᶜᶜ))
        @test precipitation_rate(model, :ice) === nothing
    end

    @testset "Slices, integrals, and output" begin
        k = Nz - 1
        slice = Field(autoconversion; indices = (:, :, k))
        compute!(slice)
        @test size(slice) == (2, 2, 1)
        @test Array(interior(slice)) ≈ Array(interior(Sᵃᶜⁿᵛ, :, :, k:k))

        ρ = total_density(model.dynamics)
        column_autoconversion = Field(Integral(ρ * autoconversion, dims = 3))
        compute!(column_autoconversion)
        ρ_data = Array(interior(ρ))
        Δz = 800 / Nz
        expected = sum(ρ_data .* Array(interior(Sᵃᶜⁿᵛ)), dims = 3) .* Δz
        @test Array(interior(column_autoconversion)) ≈ expected
        @test all(Array(interior(column_autoconversion)) .> 0)

        filename = tempname() * ".jld2"
        simulation = Simulation(model; Δt = 1e-3, stop_iteration = 1)
        outputs = (; autoconversion = slice, column_autoconversion)
        simulation.output_writers[:rates] = JLD2Writer(model, outputs; filename,
                                                       schedule = IterationInterval(1),
                                                       overwrite_files = true)
        run!(simulation)
        autoconversion_timeseries = FieldTimeSeries(filename, "autoconversion")
        @test length(autoconversion_timeseries.times) == 2
        @test all(isfinite, Array(interior(autoconversion_timeseries[end])))
        rm(filename; force = true)
    end
end

@testset "Precipitation flux profile [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    model = p3_process_rate_test_model(FT)
    Nz = size(model.grid, 3)

    rain_flux = precipitation_flux(model; condensates = (:ρqʳ,))
    @test location(rain_flux) == (Center, Center, Face)

    total_flux = Field(precipitation_flux(model))
    compute!(total_flux)
    rain_flux = Field(rain_flux)
    compute!(rain_flux)

    rain_flux_data = Array(interior(rain_flux))
    total_flux_data = Array(interior(total_flux))
    @test size(rain_flux_data, 3) == Nz + 1
    @test all(rain_flux_data[:, :, 1:Nz] .> 0)
    @test all(rain_flux_data[:, :, Nz+1] .== 0)
    @test all(total_flux_data .≥ rain_flux_data)

    # At the impenetrable bottom the sedimentation flux is the full bottom precipitation flux.
    bottom_flux = bottom_precipitation_flux(model)
    compute!(bottom_flux)
    @test total_flux_data[:, :, 1] ≈ Array(interior(bottom_flux))[:, :, 1]

    # Rain flux ≈ ρ qʳ |wʳ| for this uniform rain field: compare against the upwind estimate.
    k = 3
    ρqʳ = @allowscalar model.microphysical_fields.ρqʳ[1, 1, k]
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, k]
    @test @allowscalar(rain_flux[1, 1, k]) ≈ -ρqʳ * wʳ rtol = 0.05
end
