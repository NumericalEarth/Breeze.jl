include(joinpath(@__DIR__, "setup.jl"))

# The arguments of the stability functions of `TKEBasedTurbulenceClosure`: the `shear` (`GridMeanShear`),
# the turbulent Richardson number of `TurbulentRichardsonStabilityFunction`, and the
# `static_stability` that blends the dry and the saturated buoyancy frequency with a weight from an
# external subgrid cloud model (`SaturationWeightedStaticStability`).

using Breeze
using Breeze.TurbulenceClosures: TurbulenceClosures, TKE_NAME, Riᶜᶜᶠ, Riᶜᶜᶜ, richardson_number,
                                 static_stabilityᶜᶜᶠ, saturated_static_stabilityᶜᶜᶠ,
                                 effective_shear_squaredᶜᶜᶠ, stratification_and_shear_numbersᶜᶜᶠ,
                                 tke_stability_functionᶜᶜᶠ, tracer_stability_functionᶜᶜᶠ, dissipation_stability_functionᶜᶜᶜ,
                                 momentum_stability_function, tracer_stability_function, tke_stability_function,
                                 dissipation_stability_function, local_stability_functions,
                                 stability_coefficient_bounds, minimum_stratification_scale, within_bounds,
                                 bounded_coefficient, unbounded_coefficient,
                                 with_minimum_shear_squared, MicrophysicsSaturatedFraction, microphysics_subgrid_model
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: buoyancy_tracers, buoyancy_force
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: shearᶜᶜᶠ
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Breeze.TurbulenceClosures: TurbulenceClosureStatistics
using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, ProcessRate
using Adapt: adapt
using GPUArraysCore: @allowscalar
using Enzyme: Enzyme
using Test

#####
##### Dummy subgrid cloud models: a constant weight, and one that reads the stored mixing length
#####

struct ConstantWeight{FT} <: Breeze.AtmosphereModels.AbstractSupersaturationStatistics
    w :: FT
end

@inline TurbulenceClosures.saturated_stability_weightᶜᶜᶠ(i, j, k, grid, model::ConstantWeight, buoyancy, tracers, closure_fields) = model.w

# As the statistics of P3's `SubgridCondensation`, the constant weight supplies no spread,
# whose saturated fraction is the weight
@inline Breeze.AtmosphereModels.supersaturation_statistics(model::ConstantWeight, i, j, k, grid, closure, closure_fields, model_fields) =
    (; μ = zero(eltype(grid)), σ = zero(eltype(grid)), skewness = zero(eltype(grid)), f⁺ = convert(eltype(grid), model.w))

struct MixingLengthWeight{FT}
    ℓ⁰ :: FT
end

@inline function TurbulenceClosures.saturated_stability_weightᶜᶜᶠ(i, j, k, grid, model::MixingLengthWeight, buoyancy, tracers, closure_fields)
    ℓ = @inbounds closure_fields.ℓᵐ[i, j, k]
    return ℓ / (ℓ + model.ℓ⁰)
end

#####
##### Columns: dry convective, BOMEX and a RICO-like moist layer with cloud
#####

dry_θ(z) = 300 + 3e-3 * max(0, z - 100)

function bomex_θ(z)
    z ≤ 520  && return 298.7
    z ≤ 1480 && return 298.7 + (z - 520) * (302.4 - 298.7) / (1480 - 520)
    z ≤ 2000 && return 302.4 + (z - 1480) * (308.2 - 302.4) / (2000 - 1480)
    return 308.2 + (z - 2000) * 3.65e-3
end

function bomex_q(z)
    z ≤ 520  && return 1e-3 * (17.0 + z * (16.3 - 17.0) / 520)
    z ≤ 1480 && return 1e-3 * (16.3 + (z - 520) * (10.7 - 16.3) / (1480 - 520))
    z ≤ 2000 && return 1e-3 * (10.7 + (z - 1480) * (4.2 - 10.7) / (2000 - 1480))
    return 1e-3 * max(0, 4.2 - (z - 2000) * 1.2e-3)
end

rico_θ(z) = z ≤ 740 ? 297.9 : 297.9 + (z - 740) * (317.0 - 297.9) / (4000 - 740)
rico_q(z) = z ≤ 740 ? 1e-3 * (17.0 + z * (13.8 - 17.0) / 740) : 1e-3 * max(1.8, 13.8 + (z - 740) * (2.4 - 13.8) / (3260 - 740))

warm_adjustment() = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

const column_cases = (dry   = (Nz = 32, H = 3200, θ = dry_θ,   q = z -> 5e-3, w′θ′ = 6e-2, w′q′ = 0.0,    u = 0.0,   microphysics = nothing),
                      bomex = (Nz = 40, H = 3000, θ = bomex_θ, q = bomex_q,   w′θ′ = 8e-3, w′q′ = 5.2e-5, u = -8.75, microphysics = warm_adjustment()),
                      rico  = (Nz = 40, H = 4000, θ = rico_θ,  q = rico_q,    w′θ′ = 5e-3, w′q′ = 6e-5,   u = -9.9,  microphysics = warm_adjustment()))

function column_model(closure, case; arch = default_arch)
    grid = RectilinearGrid(arch; size = case.Nz, z = (0, case.H), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, base_pressure = 101500, potential_temperature = case.θ(0))
    dynamics = AnelasticDynamics(reference_state)
    ρ₀ = surface_density(reference_state)
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * case.w′θ′)),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * case.w′q′)))
    model = AtmosphereModel(grid; dynamics, closure, case.microphysics, boundary_conditions, advection = nothing)
    set!(model; case.θ, qᵗ = case.q, case.u, ρe = 1e-3)
    set!(model.tracers.ρe, reference_state.density * model.tracers.ρe)
    update_state!(model)
    return model
end

function run_steps!(model; Δt = 60, steps = 40)
    for _ in 1:steps
        time_step!(model, Δt)
    end
    return model
end

column(field) = Array(interior(field, 1, 1, :))

# Every prognostic field and every stored closure field
function state(model)
    prognostic = NamedTuple(name => column(field) for (name, field) in pairs(Oceananigans.prognostic_fields(model)))
    closure_fields = model.closure_fields
    closure = NamedTuple(name => column(getproperty(closure_fields, name)) for name in (:Kᵘ, :Kᶜ, :Kᵉ, :N², :ℓᵐ, :ℓˢ, :Lᵉ))
    return merge(prognostic, closure)
end

specific_tke(model) = model.tracers[TKE_NAME] / model.dynamics.reference_state.density

face_field(model, f, args...) = column(Field(KernelFunctionOperation{Center, Center, Face}(f, model.grid, args...)))
center_field(model, f, args...) = column(Field(KernelFunctionOperation{Center, Center, Center}(f, model.grid, args...)))

#####
##### Shear
#####

@testset "Shear of the TKE closure: construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @test TKEBasedTurbulenceClosure().shear === GridMeanShear{FT}(FT(1e-10))
    @test GridMeanShear(1f-9) isa GridMeanShear{Float32}
    @test isbits(GridMeanShear())

    shear = GridMeanShear(minimum_shear_squared = 1e-9)
    closure = TKEBasedTurbulenceClosure(; shear)
    @test closure.shear === GridMeanShear{FT}(FT(1e-9))
    @test isbits(closure)
    @test adapt(Array, closure).shear === closure.shear

    # The closure's `minimum_shear_squared` keyword replaces the floor of the shear
    @test TKEBasedTurbulenceClosure(minimum_shear_squared = 1e-12).shear === GridMeanShear{FT}(FT(1e-12))
    @test TKEBasedTurbulenceClosure(; shear, minimum_shear_squared = 1e-12).shear === GridMeanShear{FT}(FT(1e-12))
    @test with_minimum_shear_squared(shear, nothing) === shear

    @test_throws ArgumentError TKEBasedTurbulenceClosure(shear = GridMeanShear(0.0))
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; shear, minimum_shear_squared = 0)

    @test sprint(show, GridMeanShear()) == "GridMeanShear(minimum_shear_squared = 1.0e-10)"
    @test occursin("shear: GridMeanShear", sprint(show, TKEBasedTurbulenceClosure()))
end

#####
##### Stability functions of the stratification and shear numbers
#####

const stability_function_kinds = (momentum_stability_function, tracer_stability_function,
                                  tke_stability_function, dissipation_stability_function)

endpoint_triples(r) = ((r.ℂᵘ⁻, r.ℂᵘ⁰, r.ℂᵘ⁺), (r.ℂᶜ⁻, r.ℂᶜ⁰, r.ℂᶜ⁺), (r.ℂᵉ⁻, r.ℂᵉ⁰, r.ℂᵉ⁺), (r.ℂᴰ⁻, r.ℂᴰ⁰, r.ℂᴰ⁺))

const TRSF = TurbulentRichardsonStabilityFunction

# The rational functions of the Richardson number with the values of `sf` and its two transition scales ℂᴿⁱ±
same_values_rational(sf) = RationalStabilityFunction(; (name => getfield(sf, name) for name in fieldnames(RationalStabilityFunction)[1:12])...,
                                                     Ri⁻ = sf.ℂᴿⁱ⁻, Ri⁺ = sf.ℂᴿⁱ⁺)

# v = Riᵗ / √(1 + Riᵗ²) at which the functions are evaluated, from α_N at zero shear
stratification_number_of(sf, v) = sf.ℂᵅ * v / sqrt(1 - v^2)

@testset "TurbulentRichardsonStabilityFunction: construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    sf = TRSF()
    @test sf isa TRSF{FT}
    @test isbits(sf)
    @test adapt(Array, sf) === sf
    @test TKEBasedTurbulenceClosure().stability_functions === sf
    @test sf.ℂᵘ⁰ === FT(0.15900569445804447)
    @test within_bounds(sf)
    @test sf.ℂᵅ ≥ minimum_stratification_scale()
    @test minimum_stratification_scale() ≈ 4.18637 * (1 / 2 + 2 * sqrt(6) / 9) rtol=1e-5
    @test sf.ℂᴿⁱ⁻ === FT(16) && sf.ℂᴿⁱ⁺ === FT(0.33)

    @test_throws ArgumentError TRSF(ℂᵘ⁰ = 0)
    @test_throws ArgumentError TRSF(ℂᵅ = -1)
    @test_throws ArgumentError stability_coefficient_bounds(1e-3, 4)
    @test startswith(sprint(show, sf), "TurbulentRichardsonStabilityFunction{$FT}\n├── Riᵗ → -∞")

    # The bounds: independent intervals, ℂᵅ above the width condition
    bounds = stability_coefficient_bounds()
    @test keys(bounds) == fieldnames(TRSF)[1:15]
    @test bounds.ℂᵘ⁰ == (0.05, 20) && bounds.ℂᴿⁱ⁻ == (0.05, 20) && bounds.ℂᴿⁱ⁺ == (0.05, 20)
    @test bounds.ℂᵅ[1] ≈ 0.6987 * 1.04433 * log(400) rtol=1e-3
    @test stability_coefficient_bounds(0.01, 100).ℂᵅ[1] ≈ 6.4355 * 1.044331 rtol=1e-4
    @test !within_bounds(TRSF(ℂᵅ = 1))
    for θ in (-30.0, -1.0, 0.0, 2.5, 30.0)
        C = bounded_coefficient(θ, 0.05, 20)
        @test 0.05 ≤ C ≤ 20
        abs(θ) < 10 && @test unbounded_coefficient(C, 0.05, 20) ≈ θ
    end

    # The parameters are the fifteen values, and replacing them rebuilds the cached logarithms
    closure = TKEBasedTurbulenceClosure()
    names = keys(closure_parameters(closure))
    @test fieldnames(TRSF)[1:15] ⊆ names
    @test :quadratic_coefficients ∉ names
    changed = with_parameters(closure; ℂᶜ⁻ = 2.0, ℂᴿⁱ⁺ = 0.2, ℂᵅ = 6.0)
    @test changed.stability_functions == TRSF(FT; ℂᶜ⁻ = 2, ℂᴿⁱ⁺ = 0.2, ℂᵅ = 6)
    @test changed.stability_functions.quadratic_coefficients == TRSF(FT; ℂᶜ⁻ = 2).quadratic_coefficients
    @test closure_parameters(with_parameters(closure)) == closure_parameters(closure)
end

@testset "TurbulentRichardsonStabilityFunction: the three values, positivity, smoothness [$(FT)]" for FT in test_float_types()
    counter = Ref(0)
    uniform() = (counter[] += 1; mod(counter[] * 0.6180339887498949 + 0.1, 1))
    lower, upper = 0.05, 20
    for trial in 1:200
        C() = FT(bounded_coefficient(8 * (uniform() - 1/2), lower, upper))
        sf = TRSF(FT; ℂᵘ⁻ = C(), ℂᵘ⁰ = C(), ℂᵘ⁺ = C(), ℂᶜ⁻ = C(), ℂᶜ⁰ = C(), ℂᶜ⁺ = C(), ℂᵉ⁻ = C(), ℂᵉ⁰ = C(), ℂᵉ⁺ = C(),
                  ℂᴰ⁻ = C(), ℂᴰ⁰ = C(), ℂᴰ⁺ = C(), ℂᴿⁱ⁻ = C(), ℂᴿⁱ⁺ = C(),
                  ℂᵅ = FT(bounded_coefficient(uniform(), minimum_stratification_scale(), upper)))
        for (f, (C⁻, C⁰, C⁺)) in zip(stability_function_kinds, endpoint_triples(sf))
            # The values at v = -1, 0, 1: strongly unstable, neutral, strongly stable
            @test f(sf, zero(FT), zero(FT)) ≈ C⁰ rtol=10eps(FT)
            @test f(sf, FT(-1e9), zero(FT)) ≈ C⁻ rtol=100eps(FT)
            @test f(sf, FT(1e9), zero(FT)) ≈ C⁺ rtol=100eps(FT)
            # Positive and finite everywhere, within the box widened by the curvature
            for αᴺ in FT.((-1e30, -1e3, -1, -1e-3, 0, 1e-3, 0.5, 1, 1e30)), αˢ in FT.((0, 1e-3, 1, 1e3, 1e30))
                𝕊 = f(sf, αᴺ, αˢ)
                @test isfinite(𝕊) && 𝕊 > 0
            end
            # At any shear the logarithmic slope in α_N is below the width condition, one
            for αᴺ in FT.((-8, -2, -0.5, -0.1, -1e-3, 0, 1e-3, 0.03, 0.1, 0.5, 1)), αˢ in FT.((0, 0.01, 0.1, 1, 10))
                δ = FT(1e-3)
                slope = (log(f(sf, αᴺ + δ, αˢ)) - log(f(sf, αᴺ - δ, αˢ))) / 2δ
                @test abs(slope) ≤ 1 + sqrt(eps(FT)) * 100
            end
        end
    end
end

@testset "TurbulentRichardsonStabilityFunction: smooth through N² = 0 without shear, Ri with strong shear [$(FT)]" for FT in test_float_types()
    sf = TRSF(FT)
    for (f, (C⁻, C⁰, C⁺)) in zip(stability_function_kinds, endpoint_triples(sf))
        # The derivative at N² = 0 without shear is ½ ln(ℂ⁺/ℂ⁻) / ℂᵅ from either side
        slope = log(C⁺ / C⁻) / 2 / sf.ℂᵅ
        for δ in FT.((1e-2, 1e-3))
            right = (log(f(sf, δ, zero(FT))) - log(C⁰)) / δ
            left = (log(C⁰) - log(f(sf, -δ, zero(FT)))) / δ
            @test right ≈ slope atol=10δ
            @test left ≈ slope atol=10δ
        end
        # The derivative at N² = 0 with shear is the mean of the slopes of the two sides, from either side
        for αˢ in FT.((0.1, 10))
            mean_slope = (1 / (sf.ℂᴿⁱ⁻ * αˢ + sf.ℂᵅ) + 1 / (sf.ℂᴿⁱ⁺ * αˢ + sf.ℂᵅ)) / 2 * log(C⁺ / C⁻) / 2
            for δ in FT.((1e-4, 1e-5))
                right = (log(f(sf, δ, αˢ)) - log(C⁰)) / δ
                left = (log(C⁰) - log(f(sf, -δ, αˢ))) / δ
                @test right ≈ mean_slope atol = 200δ + sqrt(eps(FT))
                @test left ≈ mean_slope atol = 200δ + sqrt(eps(FT))
            end
        end
        # With strong shear a function of Ri = α_N / α_S alone, with v = Riᵗ / √(1 + Riᵗ²), Riᵗ = Ri / ℂᴿⁱ⁻
        # on the unstable side and Ri / ℂᴿⁱ⁺ on the stable side, once the larger |Riᵗ±| ≫ δ = 1/4
        for Ri in FT.((-30, -5, 0.5, 5)), αˢ in FT.((1e4, 1e6))
            u = Ri < 0 ? Ri / sf.ℂᴿⁱ⁻ : Ri / sf.ℂᴿⁱ⁺
            v = u / sqrt(1 + u^2)
            expected = exp(log(C⁰) + log(C⁺ / C⁻) / 2 * v + (log(C⁺ * C⁻) / 2 - log(C⁰)) * v^2)
            @test f(sf, Ri * αˢ, αˢ) ≈ expected rtol = 5e-3
        end
        # The values are reached through v: halfway in ln 𝕊 between C⁰ and C⁺ (the linear part) at v = ½
        @test f(sf, FT(stratification_number_of(sf, 1/2)), zero(FT)) ≈
              exp(log(C⁰) + log(C⁺ / C⁻) / 4 + (log(C⁺ * C⁻) / 2 - log(C⁰)) / 4) rtol=100eps(FT)
    end

    # The stable transition under strong shear: v = 1/2 at Ri ≈ 0.2, v = 0.1 and 0.9 at Ri ≈ 0.045 and 0.69
    αˢ = FT(1e6)
    v(Ri) = TurbulenceClosures.bounded_richardson_coordinate(sf, Ri * αˢ, αˢ)
    @test v(FT(0.198)) ≈ 1/2 rtol=1e-2
    @test v(FT(0.0445)) ≈ 0.1 rtol=1e-2
    @test v(FT(0.69)) ≈ 0.9 rtol=1e-2
    @test v(FT(-9.24)) ≈ -1/2 rtol=1e-2

    # Riᵗ: exactly α_N / ℂᵅ without shear, increasing in α_N, smooth through 0 (equal one-sided slopes,
    # continuous second derivative)
    Riᵗ(αᴺ, αˢ) = TurbulenceClosures.turbulent_richardson_number(sf, αᴺ, αˢ)
    for αᴺ in FT.((-3, -0.1, 0, 0.1, 3))
        @test Riᵗ(αᴺ, zero(FT)) ≈ αᴺ / sf.ℂᵅ rtol=10eps(FT)
    end
    for αˢ in FT.((0, 1e-2, 1, 1e2))
        αs = FT.(range(-2, 2, length=401))
        @test issorted(map(a -> Riᵗ(a, αˢ), αs))
        h = FT(FT == Float32 ? 1e-3 : 1e-5)
        right = (Riᵗ(h, αˢ) - Riᵗ(zero(FT), αˢ)) / h
        left = (Riᵗ(zero(FT), αˢ) - Riᵗ(-h, αˢ)) / h
        @test right ≈ left rtol = FT == Float32 ? 1e-2 : 1e-3
    end

    # The local form at a point: N², S², the floor and τ²
    τ², N², S² = FT(400), FT(-2e-4), FT(1e-5)
    @test collect(local_stability_functions(sf, N², S², FT(1e-10), τ²)) ≈
          [f(sf, τ² * N², τ² * (S² + FT(1e-10))) for f in stability_function_kinds]
end

@testset "TurbulentRichardsonStabilityFunction on the grid [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    case = column_cases.bomex
    Nz = case.Nz
    interior_faces = 2:Nz
    model = run_steps!(column_model(TKEBasedTurbulenceClosure(), case); steps = 20)
    closure, fields, velocities = model.closure, model.closure_fields, model.velocities
    e = compute!(Field(specific_tke(model)))
    Oceananigans.BoundaryConditions.fill_halo_regions!(e)

    αᴺ = face_field(model, (args...) -> stratification_and_shear_numbersᶜᶜᶠ(args...)[1], closure, velocities, fields.N², e, fields.ℓᵐ)
    αˢ = face_field(model, (args...) -> stratification_and_shear_numbersᶜᶜᶠ(args...)[2], closure, velocities, fields.N², e, fields.ℓᵐ)
    N², ℓ = column(fields.N²), column(fields.ℓᵐ)
    S² = face_field(model, shearᶜᶜᶠ, velocities.u, velocities.v)
    eᶜ = max.(column(e), closure.minimum_tke)
    w★ = (sqrt.(eᶜ[1:Nz-1]) .+ sqrt.(eᶜ[2:Nz])) ./ 2
    τ² = (ℓ[interior_faces] ./ w★).^2
    @test αᴺ[interior_faces] ≈ τ² .* N²[interior_faces]
    @test αˢ[interior_faces] ≈ τ² .* (S²[interior_faces] .+ closure.shear.minimum_shear_squared)
    @test αᴺ[1] == 0 && αˢ[1] == 0 # ℓ = 0 at the surface
    # ℓ ≤ √e / N, so α_N ≤ 1
    @test maximum(αᴺ) ≤ 1 + 10 * eps(FT)
    𝕊ᶜ = face_field(model, tracer_stability_functionᶜᶜᶠ, closure, velocities, fields.N², e, fields.ℓᵐ)
    @test 𝕊ᶜ ≈ tracer_stability_function.(Ref(closure.stability_functions), αᴺ, αˢ)
    @test all(isfinite, center_field(model, dissipation_stability_functionᶜᶜᶜ, closure, velocities, fields.N², e, fields.ℓᵐ))

    @testset "a windless column: the functions are continuous where the Richardson number is not" begin
        windless = merge(column_cases.dry, (; u = 0.0))
        default = run_steps!(column_model(TKEBasedTurbulenceClosure(), windless))
        rational = run_steps!(column_model(TKEBasedTurbulenceClosure(stability_functions = RationalStabilityFunction()), windless))
        Ri = face_field(rational, Riᶜᶜᶠ, rational.closure, rational.velocities, rational.closure_fields.N²)
        @test maximum(abs, Ri[2:windless.Nz]) > 1e4 * max(rational.closure.stability_functions.Ri⁻, rational.closure.stability_functions.Ri⁺)
        @test all(a -> all(isfinite, a), state(default))
        @test maximum(column(default.closure_fields.Kᶜ)) > 0
    end
end

# The spike test: a moist column without shear (a BOMEX-like sounding with a moist layer), with the
# subgrid distribution and P3's subgrid condensation, 100 s at Δt = 2 s, then 60 s at Δt = 0.5 s. With
# the functions of the Richardson number the diffusivity grows one-face spikes in the cloud layer, a
# checkerboard measured by cb(K) = Σ |δ²K| / (4 Σ |K|); with the turbulent Richardson number it stays smooth.
checkerboard(K) = sum(abs, diff(diff(K))) / (4 * sum(abs, K) + floatmin(eltype(K)))

spike_θ(z) = z < 520 ? 298.7 : z < 1480 ? 298.7 + (z - 520) * 3.7 / 960 : 302.4 + (z - 1480) * 5.8 / 520
spike_q(z) = 1e-3 * (z < 520 ? 17.0 - z * 0.7 / 520 : z < 1480 ? 16.3 - (z - 520) * 5.6 / 960 :
                     max(4.2, 10.7 - (z - 1480) * 6.5 / 520)) + (500 < z < 1100 ? 2.0e-3 : 0.0)

function spike_column(stability_functions; Nz = 40, H = 2400, subgrid_distribution = DoubleGaussian())
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, H), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid; base_pressure = 101500, potential_temperature = 299)
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation = SubgridCondensation(TurbulenceClosureStatistics()),
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition(),
                                                           process_rates = ProcessRate(Float64; sink_limiting_timescale = 10.0))
    closure = TKEBasedTurbulenceClosure(; minimum_tke = 4e-4, subgrid_distribution, stability_functions,
                                        static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
    ρ₀ = surface_density(reference_state)
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * 8e-3)),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * 5.2e-5)))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics, closure, boundary_conditions)
    set!(model; θ = spike_θ, qᵗ = spike_q, ρe = z -> max(0.6 * exp(-z / 500), 1e-3))
    for _ in 1:50
        time_step!(model, 2)
    end
    for _ in 1:120
        time_step!(model, 0.5)
    end
    return checkerboard(column(model.closure_fields.Kᶜ)[2:Nz]), column(model.closure_fields.Kᶜ)
end

@testset "The spike test: cb(K) in the shear-free moist column" begin
    Oceananigans.defaults.FloatType = Float64
    # The default and the functions of the Richardson number with its values
    default = TRSF()
    cbᵗ, Kᵗ = spike_column(default)
    cbᴿ, Kᴿ = spike_column(same_values_rational(default))
    @info "spike test, default values" cbᵗ cbᴿ
    @test all(isfinite, Kᵗ) && all(isfinite, Kᴿ)
    @test cbᵗ < 0.8 * cbᴿ
    # The rational functions' own coefficients, whose ratios are far from one (outside the bounds)
    rational = RationalStabilityFunction()
    cbᵗ, Kᵗ = spike_column(TRSF(; (name => getfield(rational, name) for name in fieldnames(RationalStabilityFunction)[1:12])...))
    cbᴿ, Kᴿ = spike_column(rational)
    @info "spike test, the rational functions' values" cbᵗ cbᴿ maximum(Kᵗ)
    @test maximum(Kᵗ) > 100
    @test cbᵗ < 0.1
    # The latent source of the third moment and the bound of the skewness that the PDF represents (the defaults of
    # `DoubleGaussian`) smooth both columns, the one with the functions of the Richardson number the most (its cb(K)
    # falls sevenfold), so the two are closer than a factor two; the turbulent Richardson number is still the smoother
    @test cbᵗ < cbᴿ
    # ... and with the distribution without them (no latent source, the skewness bounded at 0.85) it is by more than two
    former_distribution() = DoubleGaussian(third_moment = TKEBudgetThirdMoment(ℂʷˡ = 0), skewness = PolynomialSaturatedSkewness(maximum_skewness = 0.85))
    cbᵗ, Kᵗ = spike_column(TRSF(; (name => getfield(rational, name) for name in fieldnames(RationalStabilityFunction)[1:12])...);
                           subgrid_distribution = former_distribution())
    cbᴿ, Kᴿ = spike_column(rational; subgrid_distribution = former_distribution())
    @info "spike test, the rational functions' values, no latent source and the skewness bounded at 0.85" cbᵗ cbᴿ maximum(Kᵗ)
    @test maximum(Kᵗ) > 100
    @test cbᵗ < 0.1
    @test cbᵗ < cbᴿ / 2
end

#####
##### Static stability with a subgrid cloud weight
#####

@testset "SaturationWeightedStaticStability [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    case = column_cases.rico
    Nz = case.Nz
    interior_faces = 2:Nz

    @test isbits(SaturationWeightedStaticStability(ConstantWeight(FT(0.5))))
    @test startswith(sprint(show, SaturationWeightedStaticStability(ConstantWeight(0.5))), "SaturationWeightedStaticStability(")
    @test endswith(sprint(show, SaturationWeightedStaticStability(ConstantWeight(0.5))), "ConstantWeight{Float64}(0.5))")
    @test sprint(show, SaturationWeightedStaticStability()) == "SaturationWeightedStaticStability(MicrophysicsSaturatedFraction())"
    @test adapt(Array, SaturationWeightedStaticStability(ConstantWeight(FT(0.5)))) === SaturationWeightedStaticStability(ConstantWeight(FT(0.5)))

    with_weight(model; kw...) = TKEBasedTurbulenceClosure(; static_stability = SaturationWeightedStaticStability(model), kw...)

    dry = column_model(TKEBasedTurbulenceClosure(static_stability = DryStaticStability()), case)
    buoyancy, tracers = buoyancy_force(dry), buoyancy_tracers(dry)

    # Breeze has no model of the weight
    @test_throws MethodError TurbulenceClosures.saturated_stability_weightᶜᶜᶠ(1, 1, 2, dry.grid, nothing, buoyancy, tracers, dry.closure_fields)

    N²ᵈ = face_field(dry, ∂z_b, buoyancy, tracers)
    N²ˢ = face_field(dry, saturated_static_stabilityᶜᶜᶠ, buoyancy, tracers.T, tracers.qᵛᵉ)
    @test column(dry.closure_fields.N²) == N²ᵈ
    @test maximum(abs, N²ˢ[interior_faces] .- N²ᵈ[interior_faces]) > 1e-5

    @testset "the subgrid model of the microphysics" begin
        @test SaturationWeightedStaticStability() === SaturationWeightedStaticStability(MicrophysicsSaturatedFraction())
        @test microphysics_subgrid_model(nothing) === nothing
        @test microphysics_subgrid_model(warm_adjustment()) === nothing
        # P3 hands over the statistics of its `SubgridCondensation`: here a constant weight
        w = FT(0.3)
        p3_case = merge(case, (; microphysics = P3Microphysics(FT; condensation = SubgridCondensation(ConstantWeight(w)))))
        from_microphysics = column_model(TKEBasedTurbulenceClosure(static_stability = SaturationWeightedStaticStability()), p3_case)
        own = column_model(with_weight(ConstantWeight(w)), p3_case)
        @test microphysics_subgrid_model(from_microphysics.microphysics) === ConstantWeight(w)
        N² = column(from_microphysics.closure_fields.N²)
        @test all(isfinite, N²)
        @test N² == column(own.closure_fields.N²)
        @test N² != column(column_model(TKEBasedTurbulenceClosure(static_stability = DryStaticStability()), p3_case).closure_fields.N²)
    end

    @testset "the blend" begin
        for w in FT.((0, 0.3, 1))
            model = column_model(with_weight(ConstantWeight(w)), case)
            N² = column(model.closure_fields.N²)
            @test N²[interior_faces] ≈ ((1 - w) .* N²ᵈ .+ w .* N²ˢ)[interior_faces]
            operation = KernelFunctionOperation{Center, Center, Face}(static_stabilityᶜᶜᶠ, model.grid, model.closure.static_stability,
                                                                      buoyancy_force(model), buoyancy_tracers(model), model.closure_fields)
            @test column(Field(operation)) == N²
        end

        # No weight: the dry static stability, bit for bit. (A run differs: with a saturated fraction the
        # closure's buoyancy production is the conversion of its applied fluxes, not -Kᶜ N².)
        a = column_model(TKEBasedTurbulenceClosure(static_stability = DryStaticStability()), case)
        b = column_model(with_weight(ConstantWeight(zero(FT))), case)
        @test isequal(column(a.closure_fields.N²), column(b.closure_fields.N²))
        @test b.closure_fields.subgrid_cloud isa Breeze.TurbulenceClosures.SubgridCloudFields
    end

    @testset "the stored N² reaches the mixing length and the Richardson number" begin
        w = FT(0.6)
        model = column_model(with_weight(ConstantWeight(w); mixing_length = LocalMinimumMixingLength()), case)
        closure, fields = model.closure, model.closure_fields
        N² = column(fields.N²)
        @test N²[interior_faces] ≈ ((1 - w) .* N²ᵈ .+ w .* N²ˢ)[interior_faces]

        e = max.(column(compute!(Field(specific_tke(model)))), closure.minimum_tke)
        w★ = (sqrt.(e[1:Nz-1]) .+ sqrt.(e[2:Nz])) ./ 2
        zᶠ = Array(znodes(model.grid, Face()))[interior_faces]
        ℓᵇ = [N > 0 ? w★[n] / sqrt(N) : FT(Inf) for (n, N) in enumerate(N²[interior_faces])]
        @test column(fields.ℓᵐ)[interior_faces] ≈ min.(closure.mixing_length.ℂˢ .* zᶠ, ℓᵇ)

        S² = face_field(model, shearᶜᶜᶠ, model.velocities.u, model.velocities.v)
        Ri = face_field(model, Riᶜᶜᶠ, closure, model.velocities, fields.N²)
        @test Ri[interior_faces] ≈ N²[interior_faces] ./ (S²[interior_faces] .+ closure.shear.minimum_shear_squared)

        # and it changes them where the two frequencies differ
        dry_local = column_model(TKEBasedTurbulenceClosure(static_stability = DryStaticStability(), mixing_length = LocalMinimumMixingLength()), case)
        @test column(fields.Kᶜ) != column(dry_local.closure_fields.Kᶜ)
        @test column(fields.Lᵉ) != column(dry_local.closure_fields.Lᵉ)
    end

    @testset "a weight that reads the mixing length reads the previous update's" begin
        ℓ⁰ = FT(100)
        model = column_model(with_weight(MixingLengthWeight(ℓ⁰)), case)
        for _ in 1:3
            ℓ⁻ = column(model.closure_fields.ℓᵐ)
            update_state!(model)
            w = ℓ⁻ ./ (ℓ⁻ .+ ℓ⁰)
            @test column(model.closure_fields.N²)[interior_faces] ≈ (N²ᵈ .+ w .* (N²ˢ .- N²ᵈ))[interior_faces]
        end
        @test maximum(column(model.closure_fields.ℓᵐ)) > 0
    end
end

@testset "Stability functions and static stability combine [$(FT), $name]" for FT in test_float_types(), name in (:bomex, :rico)
    Oceananigans.defaults.FloatType = FT
    case = column_cases[name]
    functions = (TRSF(), RationalStabilityFunction())
    stabilities = (MoistStaticStability(), SaturationWeightedStaticStability(MixingLengthWeight(FT(100))))
    states = [state(run_steps!(column_model(TKEBasedTurbulenceClosure(; stability_functions, static_stability), case)))
              for stability_functions in functions, static_stability in stabilities]
    for s in states
        @test all(a -> all(isfinite, a), s)
        @test maximum(s.Kᶜ) > 0
    end
    # Four different closures
    @test allunique([s.ρe for s in states])
end

#####
##### Derivatives: Enzyme against finite differences, pointwise on the CPU
#####

reverse_gradient(f, x) = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.set_strong_zero(Enzyme.Reverse)), Enzyme.Const(f),
                                         Enzyme.Active, Enzyme.Active(x))[1][1]

central_difference(f, x, h) = (f(x * (1 + h)) - f(x * (1 - h))) / (2x * h)

@testset "Enzyme against finite differences: the coefficients of the stability functions and a subgrid cloud weight" begin
    Oceananigans.defaults.FloatType = Float64
    case = column_cases.rico
    template = TKEBasedTurbulenceClosure(static_stability = SaturationWeightedStaticStability(MixingLengthWeight(100.0)))
    model = run_steps!(column_model(template, case; arch = CPU()); steps = 20)
    grid, fields, velocities = model.grid, model.closure_fields, model.velocities
    e = compute!(Field(specific_tke(model)))
    Oceananigans.BoundaryConditions.fill_halo_regions!(e)
    buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)

    with_scalar(name, C) = TKEBasedTurbulenceClosure(stability_functions = TRSF(; (name => C,)...))
    default = TRSF()

    for k in (3, 8, 14, 20, 30)
        # Stable and unstable interfaces, through the TKE diffusivity's stability function and, at
        # the cell center, the dissipation's
        for name in (:ℂᴿⁱ⁻, :ℂᴿⁱ⁺, :ℂᵅ, :ℂᵉ⁻, :ℂᵉ⁰, :ℂᵉ⁺, :ℂᴰ⁻, :ℂᴰ⁰, :ℂᴰ⁺)
            for f in (C -> tke_stability_functionᶜᶜᶠ(1, 1, k, grid, with_scalar(name, C), velocities, fields.N², e, fields.ℓᵐ),
                      C -> dissipation_stability_functionᶜᶜᶜ(1, 1, k, grid, with_scalar(name, C), velocities, fields.N², e, fields.ℓᵐ))
                C₀ = getfield(default, name)
                ad = reverse_gradient(f, C₀)
                fd = (f(C₀ + 1e-6) - f(C₀ - 1e-6)) / 2e-6
                @test isfinite(ad)
                @test ad ≈ fd rtol=1e-5 atol=1e-8 * abs(f(C₀))
            end
        end

        # The weight: ∂N²/∂w = N²ˢ - N²ᵈ, and through the parameter of a weight that reads ℓ
        N²ᵈ = ∂z_b(1, 1, k, grid, buoyancy, tracers)
        N²ˢ = saturated_static_stabilityᶜᶜᶠ(1, 1, k, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
        in_weight = w -> static_stabilityᶜᶜᶠ(1, 1, k, grid, SaturationWeightedStaticStability(ConstantWeight(w)), buoyancy, tracers, fields)
        @test reverse_gradient(in_weight, 0.4) ≈ N²ˢ - N²ᵈ rtol=1e-10
        @test reverse_gradient(in_weight, 0.4) ≈ central_difference(in_weight, 0.4, 1e-4) rtol=1e-6
        in_length = ℓ⁰ -> static_stabilityᶜᶜᶠ(1, 1, k, grid, SaturationWeightedStaticStability(MixingLengthWeight(ℓ⁰)), buoyancy, tracers, fields)
        ad = reverse_gradient(in_length, 100.0)
        @test isfinite(ad)
        @test ad ≈ central_difference(in_length, 100.0, 1e-5) rtol=1e-5 atol=1e-16
    end

    # The surface interface, where ℓ = 0: neutral, so finite and zero
    f = C -> tke_stability_functionᶜᶜᶠ(1, 1, 1, grid, with_scalar(:ℂᵅ, C), velocities, fields.N², e, fields.ℓᵐ)
    @test reverse_gradient(f, 4.4) == 0
end

#####
##### The saturated fraction and the phase relaxation rate of the microphysics
#####

using Breeze.AtmosphereModels: AtmosphereModels, supersaturation_statistics, supersaturation_statistics_model_fields,
                               saturated_fraction
using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets, CondensationTimescaleRelaxation
using Breeze.TurbulenceClosures: applied_flux_state, cell_phase_relaxation_rate, condensing_fraction,
                                 saturated_response_shareᶜᶜᶠ, saturated_fraction_fields, with_saturated_fraction_fields

function closure_statistics_column(static_stability; Nz = 40, H = 2400, microphysics_kw = (;), steps = 25,
                                   subgrid_distribution = DoubleGaussian())
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, H), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid; base_pressure = 101500, potential_temperature = 299)
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation = SubgridCondensation(TurbulenceClosureStatistics()),
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition(),
                                                           process_rates = ProcessRate(Float64; sink_limiting_timescale = 10.0),
                                                           microphysics_kw...)
    closure = TKEBasedTurbulenceClosure(; minimum_tke = 4e-4, subgrid_distribution, static_stability)
    ρ₀ = surface_density(reference_state)
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * 8e-3)),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * 5.2e-5)))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics, closure, boundary_conditions)
    set!(model; θ = spike_θ, qᵗ = spike_q, ρe = z -> max(0.6 * exp(-z / 500), 1e-3))
    for _ in 1:steps
        time_step!(model, 2)
    end
    return model
end

@testset "MicrophysicsSaturatedFraction: f⁺ of the supersaturation statistics" begin
    Oceananigans.defaults.FloatType = Float64
    model = closure_statistics_column(SaturationWeightedStaticStability())
    grid, closure, fields = model.grid, model.closure, model.closure_fields
    Nz = size(grid, 3)
    model_fields = supersaturation_statistics_model_fields(model)
    aᶜ = column(fields.subgrid_cloud.cloud.aᶜ)
    f⁺ = @allowscalar [saturated_fraction(supersaturation_statistics(TurbulenceClosureStatistics(), 1, 1, k, grid,
                                                                       closure, fields, model_fields)) for k in 1:Nz]
    # The saturated fraction is the distribution's own cloud fraction, and there is cloud
    @test f⁺ == aᶜ
    @test maximum(f⁺) > 0.05

    # The static stability blends with f⁺ averaged to the interfaces. N² of the next update reads
    # the distribution this update leaves.
    buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
    static_stability = with_saturated_fraction_fields(closure.static_stability, buoyancy, saturated_fraction_fields(model))
    operation = KernelFunctionOperation{Center, Center, Face}(static_stabilityᶜᶜᶠ, grid, static_stability, buoyancy, tracers, fields)
    N² = column(Field(operation))
    N²ᵈ = face_field(model, ∂z_b, buoyancy, tracers)
    N²ˢ = face_field(model, saturated_static_stabilityᶜᶜᶠ, buoyancy, tracers.T, tracers.qᵛᵉ)
    aᴺ = [(f⁺[max(k - 1, 1)] + f⁺[min(k, Nz)]) / 2 for k in 1:Nz+1]
    @test N²[2:Nz] ≈ (N²ᵈ .+ aᴺ .* (N²ˢ .- N²ᵈ))[2:Nz]
    @test N²[2:Nz] != N²ᵈ[2:Nz]
    update_state!(model)
    @test column(fields.N²) == N²
    @test all(isfinite, column(fields.Kᶜ))
end

@testset "The closure's η sees P3's fixed condensation timescale" begin
    Oceananigans.defaults.FloatType = Float64
    τ = 4.0
    cloud = CloudDroplets(Float64; condensation_timescale = τ)
    model = closure_statistics_column(SaturationWeightedStaticStability(CondensateFluxFraction());
                                      microphysics_kw = (; cloud, cloud_relaxation = CondensationTimescaleRelaxation()), steps = 2)
    s = applied_flux_state(model)
    Nz = size(model.grid, 3)
    @allowscalar begin
        @test all(k -> cell_phase_relaxation_rate(1, 1, k, model.grid, s) == 1 / τ, 1:Nz)
        tₑ = 30.0
        @test saturated_response_shareᶜᶜᶠ(Val(false), tₑ, 1, 1, 5, model.grid, s) == condensing_fraction(tₑ / τ)
    end
    physical = closure_statistics_column(SaturationWeightedStaticStability(CondensateFluxFraction()); microphysics_kw = (; cloud), steps = 2)
    @test @allowscalar cell_phase_relaxation_rate(1, 1, 5, physical.grid, applied_flux_state(physical)) != 1 / τ
end
