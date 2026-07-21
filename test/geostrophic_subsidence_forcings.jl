using Breeze
using Breeze: ReferenceState, AnelasticDynamics, LiquidIcePotentialTemperatureFormulation,
              GeostrophicForcing, SpecificForcing
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes, Center
using Statistics: mean
using Test

@testset "GeostrophicForcing smoke test [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    @test haskey(geostrophic, :u)
    @test haskey(geostrophic, :v)

    coriolis = FPlane(f=1e-4)
    model = AtmosphereModel(grid; coriolis, forcing=geostrophic)

    # Geostrophic forcings are now keyed under specific names; the dispatch wraps each
    # in SpecificForcing and stores it under the corresponding ρ-key.
    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)
    @test model.forcing.ρu isa SpecificForcing
    @test model.forcing.ρv isa SpecificForcing
    @test model.forcing.ρu.forcing isa GeostrophicForcing
    @test model.forcing.ρv.forcing isa GeostrophicForcing

    Δt = 1e-6
    time_step!(model, Δt)

    # With constant uᵍ = -10 and vᵍ = 0: Fρv = +f * ρᵣ * (-10) < 0
    @test minimum(model.momentum.ρv) < 0
end

@testset "GeostrophicForcing uses live compressible density [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch;
                           size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 100), y = (0, 100), z = (0, 100),
                           topology = (Periodic, Periodic, Bounded))

    uᵍ(z) = FT(-10)
    vᵍ(z) = FT(0)
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)
    coriolis = FPlane(f = FT(1e-4))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 2,
                                                                    damping = NoDivergenceDamping());
                                    reference_potential_temperature = FT(300),
                                    surface_pressure = FT(1e5),
                                    standard_pressure = FT(1e5))

    model = AtmosphereModel(grid; dynamics, coriolis, forcing = geostrophic)

    set!(model, ρ = (x, y, z) -> FT(1),
                θ = (x, y, z) -> FT(300),
                u = (x, y, z) -> FT(-8),
                v = (x, y, z) -> FT(0))

    Δt = FT(1e-6)
    time_step!(model, Δt)

    # u - uᵍ = 2 m/s, so the geostrophic adjustment tendency for v is negative.
    # If geostrophic momentum is materialized before compressible ρ is set, this
    # sign flips because the geostrophic contribution is silently zero.
    @test sum(model.momentum.ρv) < 0
end

@testset "SubsidenceForcing smoke test [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    wˢ(z) = -0.01
    subsidence = SubsidenceForcing(wˢ)

    model = AtmosphereModel(grid; forcing=(; θ=subsidence))

    @test haskey(model.forcing, :ρθ)
    @test model.forcing.ρθ isa SpecificForcing
    @test model.forcing.ρθ.forcing isa SubsidenceForcing
    @test !isnothing(model.forcing.ρθ.forcing.subsidence_vertical_velocity)

    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "SubsidenceForcing with LiquidIcePotentialTemperature [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 10
    Hz = 1000
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Hz))

    wˢ(z) = FT(-0.01)
    subsidence = SubsidenceForcing(wˢ)

    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, forcing=(; qᵛ=subsidence))

    θ₀ = model.dynamics.reference_state.potential_temperature

    q₀ = FT(0.015)
    Γq = FT(1e-5)
    qᵗ_profile(x, y, z) = q₀ - Γq * z
    set!(model, θ=θ₀, qᵗ=qᵗ_profile)

    @test haskey(model.forcing, :ρqᵛ)
    @test model.forcing.ρqᵛ isa SpecificForcing
    @test model.forcing.ρqᵛ.forcing isa SubsidenceForcing

    ρqᵛ_initial = sum(model.moisture_density)

    # Reduced iterations (from 10 to 3)
    Δt = FT(0.1)
    for _ in 1:3
        time_step!(model, Δt)
    end

    ρqᵛ_final = sum(model.moisture_density)

    @test !isnan(ρqᵛ_final)
    @test ρqᵛ_final < ρqᵛ_initial
end

@testset "θ → e conversion in StaticEnergy model [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy)

    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model, θ=θ₀)

    @test sum(abs, model.formulation.energy_density) > 0

    Δt = 1e-6
    time_step!(model, Δt)
end

@testset "Combined GeostrophicForcing and SubsidenceForcing [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    coriolis = FPlane(f=1e-4)

    uᵍ(z) = -10
    vᵍ(z) = 0
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)

    wˢ(z) = -0.01
    subsidence = SubsidenceForcing(wˢ)

    forcing = (;
        u = (subsidence, geostrophic.u),
        v = (subsidence, geostrophic.v)
    )

    model = AtmosphereModel(grid; coriolis, forcing)

    @test haskey(model.forcing, :ρu)
    @test haskey(model.forcing, :ρv)

    Δt = 1e-6
    time_step!(model, Δt)

    @test maximum(model.momentum.ρv) < 0
end

#####
##### Analytical subsidence forcing tests
#####

@testset "GeostrophicForcing equivalence to manual ρᵣ * vᵍ Forcing reference [$(FT)]" for FT in test_float_types()
    # Path A (new): geostrophic_forcings under specific keys u, v, wrapped by SpecificForcing.
    # Path B (reference): Forcing(field) under ρu, ρv where the field stores the exact output
    # of the old GeostrophicForcing kernel, ±f * ℑxᶠᵃᵃ(ρᵣ * vᵍ) or ±f * ℑyᵃᶠᵃ(ρᵣ * uᵍ).
    # With ρᵣ anelastic (time-invariant) and uᵍ/vᵍ horizontally uniform, the two paths
    # must be bit-for-bit identical at every step. Running for several steps stresses
    # the per-step ρ-multiply in SpecificForcing.
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    coriolis = FPlane(f=FT(1e-4))
    f = FT(coriolis.f)

    uᵍ_func(z) = FT(-10 + FT(0.05) * z)
    vᵍ_func(z) = FT(2 - FT(0.02) * z)

    # Path A: new-style specific keys.
    model_new = AtmosphereModel(grid; coriolis,
                                      forcing = geostrophic_forcings(uᵍ_func, vᵍ_func))

    # Path B: reference forcings built as fields storing the old-kernel output.
    ρᵣ = model_new.dynamics.reference_state.density
    uᵍ_field = Field{Nothing, Nothing, Center}(grid)
    vᵍ_field = Field{Nothing, Nothing, Center}(grid)
    set!(uᵍ_field, z -> uᵍ_func(z))
    set!(vᵍ_field, z -> vᵍ_func(z))

    ρu_ref_field = Field{Face, Center, Center}(grid)
    ρv_ref_field = Field{Center, Face, Center}(grid)
    set!(ρu_ref_field, -f * ρᵣ * vᵍ_field)
    set!(ρv_ref_field, +f * ρᵣ * uᵍ_field)
    model_ref = AtmosphereModel(grid; coriolis,
                                      forcing = (; ρu = Forcing(ρu_ref_field),
                                                   ρv = Forcing(ρv_ref_field)))

    θ₀ = model_new.dynamics.reference_state.potential_temperature
    set!(model_new; θ=θ₀)
    set!(model_ref; θ=θ₀)

    Δt = FT(1e-3)
    N = 10
    for _ in 1:N
        time_step!(model_new, Δt)
        time_step!(model_ref, Δt)
    end

    ρu_new = interior(model_new.momentum.ρu) |> Array
    ρu_ref = interior(model_ref.momentum.ρu) |> Array
    ρv_new = interior(model_new.momentum.ρv) |> Array
    ρv_ref = interior(model_ref.momentum.ρv) |> Array

    @test maximum(abs.(ρu_new .- ρu_ref)) < eps(FT) * 1000 * max(maximum(abs, ρu_new), one(FT))
    @test maximum(abs.(ρv_new .- ρv_ref)) < eps(FT) * 1000 * max(maximum(abs, ρv_new), one(FT))
end

@testset "SubsidenceForcing multi-step linear accumulation [$FT]" for FT in test_float_types()
    # A constant z-gradient profile is preserved under uniform subsidence (the advected
    # gradient is itself the gradient), so Gρϕ = -ρᵣ wˢ Γ is constant in time and ρϕ
    # should change linearly with N. Running N steps catches per-step bugs in the
    # SpecificForcing ρ-multiply that wouldn't show up in a single-step check.
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 10), y=(0, 10), z=(0, 16))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)

    wˢ = 1
    Γ = 1e-2
    ϕᵢ(x, y, z) = Γ * z
    Δt = 1e-2
    N = 5
    Δϕ_per_step = - Δt * wˢ * Γ |> FT
    subsidence = SubsidenceForcing(FT(wˢ))

    for (specific_name, density_name) in ((:θ, :ρθ), (:qᵛ, :ρqᵛ))
        forcing = (; specific_name => subsidence)
        kw = (; advection=nothing, dynamics, formulation=:LiquidIcePotentialTemperature, forcing)
        model = AtmosphereModel(grid; tracers=:ρc, kw...)
        θ₀ = model.dynamics.reference_state.potential_temperature

        ρᵣ = model.dynamics.reference_state.density
        ρϕ = CenterField(grid)
        set!(ρϕ, ϕᵢ)
        set!(ρϕ, ρᵣ * ρϕ)

        kw_set = (; density_name => ρϕ)
        if density_name == :ρθ
            set!(model; kw_set...)
        else
            set!(model; θ=θ₀, kw_set...)
        end

        ρϕ_field = prognostic_fields(model)[density_name]
        ρϕ₀ = interior(ρϕ_field) |> Array
        for _ in 1:N
            time_step!(model, Δt)
        end
        ρϕ₁ = interior(ρϕ_field) |> Array
        ρᵣ_int = interior(ρᵣ) |> Array

        expected_change = N .* ρᵣ_int .* Δϕ_per_step
        actual_change = ρϕ₁ .- ρϕ₀
        # Loose tolerance — anelastic projection and other dynamics introduce small numerical drift.
        @test maximum(abs.(actual_change .- expected_change)) <
              FT(1e-3) * maximum(abs.(expected_change))
    end
end

@testset "Subsidence forcing gradient [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 4), x=(0, 10), y=(0, 10), z=(0, 16))
    reference_state = ReferenceState(grid)
    dynamics = AnelasticDynamics(reference_state)

    wˢ = 1
    Γ = 1e-2
    ϕᵢ(x, y, z) = Γ * z
    Δt = 1e-2
    Δϕ = - Δt * wˢ * Γ |> FT
    subsidence = SubsidenceForcing(FT(wˢ))

    # Test a representative subset of fields. The specific name keys the forcing
    # (e.g. `:θ`), while the ρ-prefixed name keys the prognostic field used to
    # check the predicted tendency.
    @testset "Subsidence forcing with constant gradient [$specific_name, $FT]" for (specific_name, density_name) in ((:u, :ρu), (:θ, :ρθ), (:qᵛ, :ρqᵛ))
        forcing = (; specific_name => subsidence)

        kw = (; advection=nothing, dynamics, formulation=:LiquidIcePotentialTemperature, forcing)
        model = AtmosphereModel(grid; tracers=:ρc, kw...)
        θ₀ = model.dynamics.reference_state.potential_temperature

        ρᵣ = model.dynamics.reference_state.density
        ρϕ = CenterField(grid)
        set!(ρϕ, ϕᵢ)
        set!(ρϕ, ρᵣ * ρϕ)

        kw = (; density_name => ρϕ)
        if density_name == :ρθ
            set!(model; kw...)
        else
            set!(model; θ=θ₀, kw...)
        end

        ρϕ = prognostic_fields(model)[density_name]
        ρϕ₀ = interior(ρϕ) |> Array
        time_step!(model, Δt)
        ρϕ₁ = interior(ρϕ) |> Array
        ρᵣ = interior(ρᵣ) |> Array

        @test ρϕ₁[1, 1, 1] - ρϕ₀[1, 1, 1] ≈ ρᵣ[1, 1, 1] * Δϕ rtol=1e-3
        @test ρϕ₁[1, 1, 4] - ρϕ₀[1, 1, 4] ≈ ρᵣ[1, 1, 4] * Δϕ rtol=1e-3
    end
end
