using Breeze
using Oceananigans
using Test

@testset "Vertically implicit diffusion correctness [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = FT(100)
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))
    vitd = VerticallyImplicitTimeDiscretization()
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()

    # Cosine profile: c(z) = cos(π z / Lz)
    # Analytical solution for diffusion: c(z,t) = cos(π z / Lz) * exp(-κ (π/Lz)² t)
    # This satisfies zero-flux BCs at z=0 and z=Lz
    k = FT(π) / Lz
    cosine(z) = cos(k * z)

    # Analytical decay factor after time t with diffusivity κ
    analytical_decay(κ, t) = exp(-κ * k^2 * t)

    @testset "Implicit scalar diffusion matches analytical solution" begin
        κ = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        set!(model; ρc = (x, y, z) -> cosine(z))
        ρc₀ = sum(interior(model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        # Compare numerical decay to analytical decay
        ρc₁ = sum(interior(model.tracers.ρc) .^ 2)
        numerical_decay = sqrt(ρc₁ / ρc₀)
        expected_decay = analytical_decay(κ, t_final)

        @test isapprox(numerical_decay, expected_decay, rtol=0.05)
    end

    @testset "Implicit and explicit diffusion match analytical solution" begin
        κ = FT(1)
        Δt = FT(0.5)
        Nt = 10
        t_final = Δt * Nt

        implicit_closure = VerticalScalarDiffusivity(vitd; κ)
        explicit_closure = VerticalScalarDiffusivity(etd; κ)

        implicit_model = AtmosphereModel(grid; closure=implicit_closure, advection=nothing, tracers=:ρc)
        explicit_model = AtmosphereModel(grid; closure=explicit_closure, advection=nothing, tracers=:ρc)

        set!(implicit_model; ρc = (x, y, z) -> cosine(z))
        set!(explicit_model; ρc = (x, y, z) -> cosine(z))

        ρc₀_implicit = sum(interior(implicit_model.tracers.ρc) .^ 2)
        ρc₀_explicit = sum(interior(explicit_model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(implicit_model, Δt)
            time_step!(explicit_model, Δt)
        end

        ρc₁_implicit = sum(interior(implicit_model.tracers.ρc) .^ 2)
        ρc₁_explicit = sum(interior(explicit_model.tracers.ρc) .^ 2)

        numerical_decay_implicit = sqrt(ρc₁_implicit / ρc₀_implicit)
        numerical_decay_explicit = sqrt(ρc₁_explicit / ρc₀_explicit)
        expected_decay = analytical_decay(κ, t_final)

        # Both should match analytical solution
        @test isapprox(numerical_decay_implicit, expected_decay, rtol=0.05)
        @test isapprox(numerical_decay_explicit, expected_decay, rtol=0.05)

        # And they should match each other closely
        @test isapprox(numerical_decay_implicit, numerical_decay_explicit, rtol=0.01)
    end

    @testset "Implicit viscosity matches analytical solution" begin
        ν = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; ν)
        model = AtmosphereModel(grid; closure, advection=nothing)

        set!(model; ρu = (x, y, z) -> cosine(z))
        ρu₀ = sum(interior(model.momentum.ρu) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρu₁ = sum(interior(model.momentum.ρu) .^ 2)
        numerical_decay = sqrt(ρu₁ / ρu₀)
        expected_decay = analytical_decay(ν, t_final)

        @test isapprox(numerical_decay, expected_decay, rtol=0.05)
    end

    @testset "Implicit diffusion with both ν and κ matches analytical solutions" begin
        ν = FT(5)
        κ = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; ν, κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        set!(model; ρu = (x, y, z) -> cosine(z), ρc = (x, y, z) -> cosine(z))

        ρu₀ = sum(interior(model.momentum.ρu) .^ 2)
        ρc₀ = sum(interior(model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρu₁ = sum(interior(model.momentum.ρu) .^ 2)
        ρc₁ = sum(interior(model.tracers.ρc) .^ 2)

        numerical_decay_u = sqrt(ρu₁ / ρu₀)
        numerical_decay_c = sqrt(ρc₁ / ρc₀)
        expected_decay_u = analytical_decay(ν, t_final)
        expected_decay_c = analytical_decay(κ, t_final)

        @test isapprox(numerical_decay_u, expected_decay_u, rtol=0.05)
        @test isapprox(numerical_decay_c, expected_decay_c, rtol=0.05)
    end
end
