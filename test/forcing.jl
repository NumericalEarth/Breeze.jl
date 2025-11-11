using Breeze
using Oceananigans
using Oceananigans: interior
using Test
using Base: Returns

function setup_forcing_model(grid, forcing)
    model = AtmosphereModel(grid; tracers=:ρc, forcing)
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀)
    return model
end

increment_tolerance(::Type{Float32}) = 1f-5
increment_tolerance(::Type{Float64}) = 1e-10

@testset "AtmosphereModel forcing increments prognostic fields [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    forcings = [
        Returns(one(FT)),
        Forcing(Returns(one(FT)), discrete_form=true),
        Forcing(Returns(one(FT)), field_dependencies=:ρu, discrete_form=true),
        Forcing(Returns(one(FT)), field_dependencies=(:ρe, :ρqᵗ, :ρu), discrete_form=true),
        1,
    ]

    Δt = 3

    @testset "Momentum forcing ($FT, $(typeof(forcing)))" for forcing in forcings
        # x-momentum (ρu)
        u_forcing = (; ρu=forcing)
        model = setup_forcing_model(grid, u_forcing)
        ρu_before = deepcopy(model.momentum.ρu)
        time_step!(model, Δt)

        expected_ρu = ρu_before + Δt
        diff_ρu = maximum(abs, model.momentum.ρu - expected_ρu)
        @test diff_ρu ≤ increment_tolerance(FT)

        # y-momentum (ρv)
        v_forcing = (; ρv=forcing)
        model = setup_forcing_model(grid, v_forcing)
        ρv_before = deepcopy(model.momentum.ρv)
        time_step!(model, Δt)

        expected_ρv = ρv_before + Δt
        diff_ρv = maximum(abs, model.momentum.ρv - expected_ρv)
        @test diff_ρv ≤ increment_tolerance(FT)

        # z-momentum (ρw)
        w_forcing = (; ρw=forcing)
        model = setup_forcing_model(grid, w_forcing)
        ρw_before = deepcopy(model.momentum.ρw)
        time_step!(model, Δt)

        expected_ρw = ρw_before + Δt
        diff_ρw = maximum(abs, model.momentum.ρw - expected_ρw)
        @test diff_ρw ≤ increment_tolerance(FT)
    end

    @testset "Energy forcing ($FT, $(typeof(forcing)))" for forcing in forcings
        e_forcing = (; ρe=forcing)
        model = setup_forcing_model(grid, e_forcing)
        ρe_before = deepcopy(model.energy_density)
        time_step!(model, Δt)

        expected_ρe = ρe_before + Δt
        diff_ρe = maximum(abs, model.energy_density - expected_ρe)
        @test diff_ρe ≤ increment_tolerance(FT)
    end

    @testset "Moisture forcing ($FT, $(typeof(forcing)))" for forcing in forcings
        q_forcing = (; ρqᵗ=forcing)
        model = setup_forcing_model(grid, q_forcing)
        ρq_before = deepcopy(model.moisture_density)
        time_step!(model, Δt)

        expected = ρq_before + Δt
        diff = maximum(abs, model.moisture_density - expected)
        @test diff ≤ increment_tolerance(FT)
    end

    @testset "Scalar forcing ($FT, $(typeof(forcing)))" for forcing in forcings
        c_forcing = (; ρc=forcing)
        model = setup_forcing_model(grid, c_forcing)
        ρc_before = deepcopy(model.tracers.ρc)
        time_step!(model, Δt)

        expected_ρc = ρc_before + Δt
        diff_ρc = maximum(abs, model.tracers.ρc - expected_ρc)
        @test diff_ρc ≤ increment_tolerance(FT)
    end
end
