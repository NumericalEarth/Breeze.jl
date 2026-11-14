include(joinpath(@__DIR__, "setup.jl"))

# The two components of the Richardson number of `TKEBasedTurbulenceClosure`: the `shear`
# (`GridMeanShear`, `TurbulenceAugmentedShear`) and the `static_stability` that blends the dry and
# the saturated buoyancy frequency with a weight from an external subgrid cloud model
# (`SaturationWeightedStaticStability`).

using Breeze
using Breeze.TurbulenceClosures: TurbulenceClosures, TKE_NAME, Riᶜᶜᶠ, Riᶜᶜᶜ, richardson_number,
                                 static_stabilityᶜᶜᶠ, saturated_static_stabilityᶜᶜᶠ,
                                 turbulence_shear_squaredᶜᶜᶠ, effective_shear_squaredᶜᶜᶠ,
                                 tke_stability_functionᶜᶜᶠ, dissipation_stability_functionᶜᶜᶜ,
                                 with_minimum_shear_squared, MicrophysicsSaturatedFraction, microphysics_subgrid_model
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: buoyancy_tracers, buoyancy_force
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: shearᶜᶜᶠ
using Oceananigans.BuoyancyFormulations: ∂z_b
using Adapt: adapt
using Enzyme: Enzyme
using Test

#####
##### Dummy subgrid cloud models: a constant weight, and one that reads the stored mixing length
#####

struct ConstantWeight{FT}
    w :: FT
end

@inline TurbulenceClosures.saturated_stability_weightᶜᶜᶠ(i, j, k, grid, model::ConstantWeight, buoyancy, tracers, closure_fields) = model.w

# In P3's subgrid supersaturation hook the constant weight is a model without a correlation
const PPP = Breeze.Microphysics.PredictedParticleProperties
PPP.subgrid_supersaturation_fields(::ConstantWeight, model) = nothing
@inline PPP.subgrid_supersaturation_statistics(i, j, k, grid, ::ConstantWeight, fields) = nothing
@inline PPP.subgrid_supersaturation(::ConstantWeight, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) = zero(qᵛ)

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
    @test TurbulenceAugmentedShear() === TurbulenceAugmentedShear(0.0, 1e-10)
    @test TurbulenceAugmentedShear(1, 1e-10) === TurbulenceAugmentedShear(1.0, 1e-10)
    @test isbits(GridMeanShear()) && isbits(TurbulenceAugmentedShear())

    shear = TurbulenceAugmentedShear(turbulence_shear_coefficient = 0.1, minimum_shear_squared = 1e-9)
    closure = TKEBasedTurbulenceClosure(; shear)
    @test closure.shear === TurbulenceAugmentedShear{FT}(FT(0.1), FT(1e-9))
    @test isbits(closure)
    @test adapt(Array, closure).shear === closure.shear

    # The closure's `minimum_shear_squared` keyword replaces the floor of either shear
    @test TKEBasedTurbulenceClosure(minimum_shear_squared = 1e-12).shear === GridMeanShear{FT}(FT(1e-12))
    @test TKEBasedTurbulenceClosure(; shear, minimum_shear_squared = 1e-12).shear === TurbulenceAugmentedShear{FT}(FT(0.1), FT(1e-12))
    @test with_minimum_shear_squared(shear, nothing) === shear

    @test_throws ArgumentError TKEBasedTurbulenceClosure(shear = GridMeanShear(0.0))
    @test_throws ArgumentError TKEBasedTurbulenceClosure(shear = TurbulenceAugmentedShear(0.1, -1e-10))
    @test_throws ArgumentError TKEBasedTurbulenceClosure(shear = TurbulenceAugmentedShear(-0.1, 1e-10))
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; shear, minimum_shear_squared = 0)

    @test sprint(show, GridMeanShear()) == "GridMeanShear(minimum_shear_squared = 1.0e-10)"
    @test occursin("TurbulenceAugmentedShear(turbulence_shear_coefficient = 0.1", sprint(show, closure))
    @test occursin("shear: GridMeanShear", sprint(show, TKEBasedTurbulenceClosure()))
end

@testset "TurbulenceAugmentedShear with a zero coefficient is GridMeanShear, bitwise [$(FT), $name]" for FT in test_float_types(), name in keys(column_cases)
    Oceananigans.defaults.FloatType = FT
    case = column_cases[name]

    for stability_functions in (RationalStabilityFunction(), ConstantStabilityFunctions())
        grid_mean = run_steps!(column_model(TKEBasedTurbulenceClosure(; stability_functions), case))
        augmented = run_steps!(column_model(TKEBasedTurbulenceClosure(; stability_functions, shear = TurbulenceAugmentedShear()), case))
        a, b = state(grid_mean), state(augmented)
        @test all(name -> all(isfinite, a[name]), (:ρe, :ρθ, :Kᶜ, :N²))
        @test maximum(a.Kᶜ) > 0
        for field in keys(a)
            @test isequal(a[field], b[field])
        end
    end
end

@testset "TurbulenceAugmentedShear: the Richardson number [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    ℂˢᵗ = FT(0.5)
    case = column_cases.bomex
    Nz = case.Nz
    interior_faces = 2:Nz

    shear = TurbulenceAugmentedShear(turbulence_shear_coefficient = ℂˢᵗ)
    model = run_steps!(column_model(TKEBasedTurbulenceClosure(; shear), case); steps = 20)
    closure, fields, velocities = model.closure, model.closure_fields, model.velocities
    e = compute!(Field(specific_tke(model)))
    Oceananigans.BoundaryConditions.fill_halo_regions!(e)

    N² = column(fields.N²)
    ℓ = column(fields.ℓᵐ)
    S² = face_field(model, shearᶜᶜᶠ, velocities.u, velocities.v)
    S²ᵗ = face_field(model, turbulence_shear_squaredᶜᶜᶠ, closure, e, fields.ℓᵐ)
    Ri = face_field(model, Riᶜᶜᶠ, closure, velocities, fields.N², e, fields.ℓᵐ)
    S²ᵐⁱⁿ = closure.shear.minimum_shear_squared

    # e floored at the minimum TKE and reconstructed at the face, as in the diffusivity
    eᶜ = max.(column(e), closure.minimum_tke)
    w★ = (sqrt.(eᶜ[1:Nz-1]) .+ sqrt.(eᶜ[2:Nz])) ./ 2
    @test S²ᵗ[interior_faces] ≈ ℂˢᵗ .* (w★ ./ ℓ[interior_faces]).^2
    @test S²ᵗ[1] == 0 # ℓ = 0 at the surface
    @test all(isfinite, S²ᵗ)
    @test all(S²ᵗ[interior_faces] .> 0)
    @test Ri[interior_faces] ≈ N²[interior_faces] ./ (S²[interior_faces] .+ S²ᵗ[interior_faces] .+ S²ᵐⁱⁿ)
    @test face_field(model, effective_shear_squaredᶜᶜᶠ, closure, velocities, e, fields.ℓᵐ) == S² .+ S²ᵗ

    # The grid-mean shear reads neither e nor ℓ, and the short form is the same number
    grid_mean = TKEBasedTurbulenceClosure()
    @test face_field(model, Riᶜᶜᶠ, grid_mean, velocities, fields.N², e, fields.ℓᵐ) ==
          face_field(model, Riᶜᶜᶠ, grid_mean, velocities, fields.N²)
    @test center_field(model, Riᶜᶜᶜ, grid_mean, velocities, fields.N², e, fields.ℓᵐ) ==
          center_field(model, Riᶜᶜᶜ, grid_mean, velocities, fields.N²)
    @test face_field(model, Riᶜᶜᶠ, grid_mean, velocities, fields.N²)[interior_faces] ≈
          N²[interior_faces] ./ (S²[interior_faces] .+ S²ᵐⁱⁿ)

    # Neutral stratification is neutral whatever the turbulence
    @test richardson_number(zero(FT), FT(3e-4), S²ᵐⁱⁿ) == 0

    @testset "a windless convective layer" begin
        # Without wind the grid-mean Richardson number is N² / S²ᵐⁱⁿ, at the limits of the stability
        # functions wherever N² is not exactly zero; with the turbulence term it is of order N² ℓ² / e
        windless = merge(column_cases.dry, (; u = 0.0))
        augmented = run_steps!(column_model(TKEBasedTurbulenceClosure(; shear), windless))
        plain = run_steps!(column_model(TKEBasedTurbulenceClosure(), windless))
        Ri_of(model) = begin
            e = compute!(Field(specific_tke(model)))
            Oceananigans.BoundaryConditions.fill_halo_regions!(e)
            face_field(model, Riᶜᶜᶠ, model.closure, model.velocities, model.closure_fields.N², e, model.closure_fields.ℓᵐ)
        end
        Riᵃ, Riᵖ = Ri_of(augmented), Ri_of(plain)
        sf = plain.closure.stability_functions
        @test maximum(abs, Riᵖ[2:windless.Nz]) > 1e4 * max(sf.Ri⁻, sf.Ri⁺)
        # ℓ ≤ √e / N, so the stable Richardson number is at most 1 / ℂˢᵗ
        @test maximum(Riᵃ[2:windless.Nz]) ≤ (1 + 10 * eps(FT)) / ℂˢᵗ
        @test all(isfinite, Riᵃ)
        @test all(a -> all(isfinite, a), state(augmented))
        @test maximum(column(augmented.closure_fields.Kᵉ)) < maximum(column(plain.closure_fields.Kᵉ))
    end
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
        # P3 hands over the model in its subgrid supersaturation hook: here a constant weight
        w = FT(0.3)
        p3_case = merge(case, (; microphysics = P3Microphysics(FT; subgrid = ConstantWeight(w))))
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

        # No weight: the dry closure, bit for bit, after a run
        a = state(run_steps!(column_model(TKEBasedTurbulenceClosure(static_stability = DryStaticStability()), case)))
        b = state(run_steps!(column_model(with_weight(ConstantWeight(zero(FT))), case)))
        for field in keys(a)
            @test isequal(a[field], b[field])
        end
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

@testset "Shear and static stability combine [$(FT), $name]" for FT in test_float_types(), name in (:bomex, :rico)
    Oceananigans.defaults.FloatType = FT
    case = column_cases[name]
    shears = (GridMeanShear(), TurbulenceAugmentedShear(turbulence_shear_coefficient = 0.5))
    stabilities = (MoistStaticStability(), SaturationWeightedStaticStability(MixingLengthWeight(FT(100))))
    states = [state(run_steps!(column_model(TKEBasedTurbulenceClosure(; shear, static_stability), case)))
              for shear in shears, static_stability in stabilities]
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

@testset "Enzyme against finite differences: the shear coefficient and a subgrid cloud weight" begin
    Oceananigans.defaults.FloatType = Float64
    case = column_cases.rico
    template = TKEBasedTurbulenceClosure(shear = TurbulenceAugmentedShear(turbulence_shear_coefficient = 0.3),
                                         static_stability = SaturationWeightedStaticStability(MixingLengthWeight(100.0)))
    model = run_steps!(column_model(template, case; arch = CPU()); steps = 20)
    grid, fields, velocities = model.grid, model.closure_fields, model.velocities
    e = compute!(Field(specific_tke(model)))
    Oceananigans.BoundaryConditions.fill_halo_regions!(e)
    buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)

    with_coefficient(ℂˢᵗ) = TKEBasedTurbulenceClosure(shear = TurbulenceAugmentedShear(ℂˢᵗ, 1e-10))

    for k in (3, 8, 14, 20, 30)
        # Stable and unstable interfaces, through the TKE diffusivity's stability function and, at
        # the cell center, the dissipation's
        for f in (ℂˢᵗ -> Riᶜᶜᶠ(1, 1, k, grid, with_coefficient(ℂˢᵗ), velocities, fields.N², e, fields.ℓᵐ),
                  ℂˢᵗ -> tke_stability_functionᶜᶜᶠ(1, 1, k, grid, with_coefficient(ℂˢᵗ), velocities, fields.N², e, fields.ℓᵐ),
                  ℂˢᵗ -> dissipation_stability_functionᶜᶜᶜ(1, 1, k, grid, with_coefficient(ℂˢᵗ), velocities, fields.N², e, fields.ℓᵐ))
            for ℂˢᵗ in (0.1, 1.0)
                ad = reverse_gradient(f, ℂˢᵗ)
                fd = central_difference(f, ℂˢᵗ, 1e-5)
                @test isfinite(ad)
                @test ad != 0
                @test ad ≈ fd rtol=1e-6
            end
        end
        # At a zero coefficient the derivative is finite: -N² (e / ℓ²) / (S² + S²ᵐⁱⁿ)²
        f = ℂˢᵗ -> Riᶜᶜᶠ(1, 1, k, grid, with_coefficient(ℂˢᵗ), velocities, fields.N², e, fields.ℓᵐ)
        S² = shearᶜᶜᶠ(1, 1, k, grid, velocities.u, velocities.v)
        e_ℓ² = turbulence_shear_squaredᶜᶜᶠ(1, 1, k, grid, with_coefficient(1.0), e, fields.ℓᵐ)
        @test reverse_gradient(f, 0.0) ≈ -fields.N²[1, 1, k] * e_ℓ² / (S² + 1e-10)^2 rtol=1e-10

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

    # The surface interface, where ℓ = 0: finite and zero
    f = ℂˢᵗ -> turbulence_shear_squaredᶜᶜᶠ(1, 1, 1, grid, with_coefficient(ℂˢᵗ), e, fields.ℓᵐ)
    @test reverse_gradient(f, 0.3) == 0
    in_energy = scale -> turbulence_shear_squaredᶜᶜᶠ(1, 1, 1, grid, TKEBasedTurbulenceClosure(shear = TurbulenceAugmentedShear(0.3, 1e-10), minimum_tke = scale * 1e-6), e, fields.ℓᵐ)
    @test reverse_gradient(in_energy, 1.0) == 0
end
