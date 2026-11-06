include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Microphysics: relaxation_timescale
using CloudMicrophysics
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, one_moment_cloud_microphysics_categories

column(field) = Array(interior(field))[1, 1, :]

function mpne1m_column(formation; Δt, Nz = 32, H = 8000)
    grid = RectilinearGrid(CPU(); size = Nz, z = (0, H), topology = (Flat, Flat, Bounded))
    # The donor limiter requires τⁿᵘᵐ ≥ Δt
    categories = one_moment_cloud_microphysics_categories(; τⁿᵘᵐ = Δt)
    microphysics = OneMomentCloudMicrophysics(; categories,
                                              cloud_formation = NonEquilibriumCloudFormation(formation, formation))
    model = AtmosphereModel(grid; microphysics, advection = nothing)
    # A moist, supersaturated layer between 1 and 5 km
    qᵗ(z) = 0.02 * exp(-z / 2500) * (1 + 0.3 * (1000 < z < 5000))
    set!(model; θ = z -> 290 + 4e-3 * z, qᵗ)
    return model
end

@testset "ExponentialRateCondensateFormation" begin
    τ = 120.0

    @testset "effective timescale limits" begin
        for Δt in (1e-6, 1e-3)
            @test relaxation_timescale(ExponentialRateCondensateFormation(1 / τ, Δt)) ≈ τ rtol = 1e-5
        end
        for Δt in (1e4, 1e5)
            @test relaxation_timescale(ExponentialRateCondensateFormation(1 / τ, Δt)) ≈ Δt rtol = 1e-12
        end
        τₑ = relaxation_timescale(ExponentialRateCondensateFormation(1 / τ, 900.0))
        @test τₑ ≈ 900 / (1 - exp(-900 / τ))
        @test τₑ ≥ 900

        # The default is unchanged
        @test relaxation_timescale(ConstantRateCondensateFormation(1 / τ)) === 1 / (1 / τ)
        @test ExponentialRateCondensateFormation(1 / 120, 900) isa ExponentialRateCondensateFormation{Float64}
    end

    @testset "MPNE1M forward step at Δt = 900 s, τ = 120 s stays bounded" begin
        Δt = 900.0
        model = mpne1m_column(ExponentialRateCondensateFormation(1 / τ, Δt); Δt)
        qᵗ⁰ = sum(column(model.moisture_density) .+ column(model.microphysical_fields.ρqᶜˡ) .+
                  column(model.microphysical_fields.ρqᶜⁱ))
        for _ in 1:4
            time_step!(model, Δt)
        end
        ρqᶜˡ = column(model.microphysical_fields.ρqᶜˡ)
        ρqᶜⁱ = column(model.microphysical_fields.ρqᶜⁱ)
        ρqᵛ  = column(model.moisture_density)
        T = column(model.temperature)
        @test all(isfinite, ρqᶜˡ) && all(isfinite, ρqᶜⁱ) && all(isfinite, T)
        ρqᶜ_max = maximum(ρqᶜˡ .+ ρqᶜⁱ)
        @info "Δt = 900 s, τ = 120 s" minimum(ρqᶜˡ) minimum(ρqᶜⁱ) minimum(ρqᵛ) ρqᶜ_max
        # Positive up to the round-off-sized undershoots of the explicit τⁿᵘᵐ guards
        @test minimum(ρqᶜˡ) ≥ -1e-6 * ρqᶜ_max
        @test minimum(ρqᶜⁱ) ≥ -1e-6 * ρqᶜ_max
        @test all(ρqᵛ .≥ 0)
        @test any(ρqᶜˡ .+ ρqᶜⁱ .> 0)  # condensate forms in the supersaturated layer
        # Condensate stays a fraction of the column's initial water
        @test maximum(ρqᶜˡ .+ ρqᶜⁱ) < 0.05 * qᵗ⁰
        @test all(150 .< T .< 350)
    end
end
