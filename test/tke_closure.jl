include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME, TKEClosureFields, absorb_stratification_coefficient, dissipationᶜᶜᶜ,
                                 tke_sink_rate
using Oceananigans
using Oceananigans.TimeSteppers: update_state!, time_discretization
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, ExplicitTimeDiscretization,
                                       buoyancy_tracers, buoyancy_force
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Units
using Test
using Adapt: adapt

# The mixing length is stored with the closure fields: its envelope over the column is computed by
# two sweeps, once per stage, so a script reads it as `model.closure_fields.ℓᵐ`.
diagnosed_mixing_length(model) = model.closure_fields.ℓᵐ

# The envelope written out in full, as the test's reference: at every face the minimum over every
# face z′ of the bound there — the wall length or the buoyancy penetration depth, whichever is
# smaller — plus ℂˢ times the distance. The ground, the bottom face, is a zero; above it the wall
# length is ℂˢ times the height.
function brute_force_envelope(grid, ℓᵇ, ℂˢ)
    zf = Array(znodes(grid, Face())) # a stretched grid on the GPU keeps its nodes on the device
    obstacles = min.(ℂˢ .* zf, ℓᵇ)
    obstacles[1] = 0
    return [minimum(obstacles[m] + ℂˢ * abs(zf[k] - zf[m]) for m in eachindex(zf)) for k in eachindex(zf)]
end

#####
##### Construction
#####

@testset "TKEBasedTurbulenceClosure construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    closure = TKEBasedTurbulenceClosure(mixing_length = GradientLimitedMixingLength(),
                                        stability_functions = ConstantStabilityFunctions())
    @test closure isa TKEBasedTurbulenceClosure
    @test time_discretization(closure) isa VerticallyImplicitTimeDiscretization
    @test closure.mixing_length isa GradientLimitedMixingLength{FT}
    @test closure.stability_functions isa ConstantStabilityFunctions{FT}
    @test closure.static_stability isa MoistStaticStability

    @testset "defaults and the constants they imply" begin
        sf = closure.stability_functions
        ℂˢ = closure.mixing_length.ℂˢ
        @test ℂˢ ≈ 1.316
        @test (sf.ℂᵘ, sf.ℂᶜ, sf.ℂᵉ, sf.ℂᴰ) == FT.((0.149, 0.201, 0.298, 0.388))
        @test closure.minimum_tke == FT(1e-6)
        @test closure.minimum_shear_squared == FT(1e-10)
        @test closure.negative_tke_damping_time_scale == FT(60)
        @test isinf(closure.maximum_viscosity)
        @test isinf(closure.maximum_tracer_diffusivity)
        @test isinf(closure.maximum_tke_diffusivity)

        # The neutral log layer: von Kármán constant, surface TKE, Prandtl number; and the
        # stratified steady state: the critical Richardson number
        @test ℂˢ * (sf.ℂᵘ^3 / sf.ℂᴰ)^(1/4) ≈ 0.40 atol=0.005
        @test 1 / sqrt(sf.ℂᵘ * sf.ℂᴰ) ≈ 4.2 atol=0.05
        @test sf.ℂᵘ / sf.ℂᶜ ≈ 0.74 atol=0.005
        @test sf.ℂᵘ / (sf.ℂᶜ + sf.ℂᴰ) ≈ 0.25 atol=0.005
    end

    @testset "keyword arguments, promotion and float type" begin
        closure = TKEBasedTurbulenceClosure(; mixing_length = GradientLimitedMixingLength(ℂˢ = 1),
                                              stability_functions = ConstantStabilityFunctions(ℂᵘ = 0.3, ℂᶜ = 0.3, ℂᵉ = 1, ℂᴰ = 1),
                                              maximum_viscosity = 100,
                                              minimum_tke = 1e-8,
                                              minimum_shear_squared = 1e-12,
                                              negative_tke_damping_time_scale = 10minutes)
        @test closure.mixing_length.ℂˢ === FT(1)
        @test closure.stability_functions.ℂᵘ === FT(0.3)
        @test closure.stability_functions.ℂᵉ === FT(1)
        @test closure.maximum_viscosity === FT(100)
        @test closure.minimum_tke === FT(1e-8)
        @test closure.minimum_shear_squared === FT(1e-12)
        @test closure.negative_tke_damping_time_scale === FT(600)
        @test_throws ArgumentError TKEBasedTurbulenceClosure(FT; minimum_shear_squared = 0)
        @test_throws ArgumentError TKEBasedTurbulenceClosure(FT; minimum_shear_squared = -1e-10)

        explicit = TKEBasedTurbulenceClosure(ExplicitTimeDiscretization())
        @test time_discretization(explicit) isa ExplicitTimeDiscretization

        closure32 = TKEBasedTurbulenceClosure(Float32; stability_functions = ConstantStabilityFunctions())
        @test closure32.mixing_length isa GradientLimitedMixingLength{Float32}
        @test closure32.stability_functions isa ConstantStabilityFunctions{Float32}
        @test closure32.minimum_tke isa Float32
    end

    @testset "isbits and show" begin
        @test isbits(closure)
        @test isbits(GradientLimitedMixingLength())
        @test isbits(LocalMinimumMixingLength())
        @test isbits(IntegralMixingLength())
        @test isbits(ConstantStabilityFunctions())
        @test isbits(DryStaticStability())
        @test summary(closure) == "TKEBasedTurbulenceClosure{VerticallyImplicitTimeDiscretization}"
        str = sprint(show, closure)
        @test occursin("ℂˢ", str)
        @test occursin("ConstantStabilityFunctions", str)
        @test occursin("MoistStaticStability", str)
        @test occursin("DryStaticStability", sprint(show, TKEBasedTurbulenceClosure(static_stability = DryStaticStability())))
        @test occursin("minimum_tke", str)
        @test occursin("minimum_shear_squared", str)
        @test occursin("ℂᴰ", sprint(show, ConstantStabilityFunctions()))
        @test occursin("ℂˢ", sprint(show, GradientLimitedMixingLength()))
        @test occursin("LocalMinimum", sprint(show, LocalMinimumMixingLength()))
        @test occursin("Integral", sprint(show, IntegralMixingLength()))
    end
end

#####
##### In an AtmosphereModel
#####

# Set a uniform specific TKE `e₀` and refresh the closure fields
function set_tke!(model, e₀)
    ρ = model.dynamics.reference_state.density
    set!(model.tracers[TKE_NAME], e₀)
    parent(model.tracers[TKE_NAME]) .*= parent(ρ)
    update_state!(model)
    return nothing
end

column(field) = Array(interior(field, 1, 1, :))

@testset "Calibrated defaults and split mixing length [$(FT)]" for FT in test_float_types()
    closure = TKEBasedTurbulenceClosure(FT)
    sf = closure.stability_functions
    @test adapt(identity, closure) === closure
    @test adapt(identity, closure.mixing_length) === closure.mixing_length
    @test adapt(identity, sf) === sf
    @test sf isa RationalStabilityFunction{FT}
    @test closure.mixing_length.ℂˢ === FT(0.7877315198717979)
    @test closure.mixing_length.ℂⁱ === FT(1.777871783491512)
    @test closure.mixing_length.ℂˢ * (sf.ℂᵘ⁰^3 / sf.ℂᴰ⁰)^(1/4) ≈ 0.4 rtol=1e-6
    @test sf.ℂᵘ⁰ / sf.ℂᶜ⁰ ≈ 0.7999974525668859
    @test tke_surface_flux(FT(1), FT(1), FT(0)) === FT(0.01002182909933591)
    @test tke_surface_flux(FT(1), FT(0), FT(1)) === FT(0.013669219822302736)
    @test tke_surface_flux(FT(2), FT(3), FT(4)) ≈ FT(2) * (FT(3) * FT(0.01002182909933591) + FT(4) * FT(0.013669219822302736))
    @test @inferred(tke_surface_flux(FT(2), FT(3), FT(4), FT(0.5), FT(0.25))) === FT(5)
    @test GradientLimitedMixingLength(2).ℂⁱ == 2
    @test GradientLimitedMixingLength{FT}(2).ℂⁱ === FT(2)
    @test GradientLimitedMixingLength(ℂˢ=2, ℂⁱ=0.5).ℂⁱ === 0.5

    # Independent O(Nz²) face-envelope reference, including unequal slopes in BOTH directions.
    for stretched in (false, true), (Cs, Ci) in ((0.65, 1.56), (1.3, 0.4), (0.8, 0.8))
        faces = stretched ? [0, 20, 50, 100, 180, 300, 460, 650, 850, 1000] : collect(0:100:1000)
        grid = RectilinearGrid(default_arch, FT; size=length(faces)-1, z=FT.(faces),
                              topology=(Flat, Flat, Bounded))
        ml = GradientLimitedMixingLength(ℂˢ=Cs, ℂⁱ=Ci)
        closure = TKEBasedTurbulenceClosure(FT; mixing_length=ml, static_stability=DryStaticStability())
        model = AtmosphereModel(grid; closure, advection=nothing)
        θ(z) = 300 + 0.01 * (min(z, 300) + max(z-700, 0))
        set!(model; θ)
        e₀ = FT(0.5)
        set_tke!(model, e₀)
        zf = FT.(faces)
        N² = column(model.closure_fields.N²)
        depths = [N > 0 ? sqrt(e₀ / N) : FT(Inf) for N in N²]
        envelope = [minimum(depths[m] + Ci * abs(zf[k] - zf[m]) for m in 2:length(zf)) for k in eachindex(zf)]
        expected = min.(Cs .* zf, envelope)
        @test column(model.closure_fields.ℓᵐ) ≈ expected rtol=1e-5
        @test column(model.closure_fields.ℓˢ) ≈ envelope rtol=1e-5
        if Cs == Ci
            @test expected ≈ brute_force_envelope(grid, depths, Cs) rtol=1e-5
        end

        # A neutral column is wall-limited at faces AND centers even if Ci < Cs.
        set!(model; θ=300, u=z -> 0.01z)
        set_tke!(model, e₀)
        @test column(model.closure_fields.ℓᵐ) ≈ Cs .* zf rtol=1e-5
        e = model.tracers.ρe / model.dynamics.reference_state.density
        lengths = column(Field(KernelFunctionOperation{Center, Center, Center}(
            Breeze.TurbulenceClosures.mixing_lengthᶜᶜᶜ, grid, closure, e, model.closure_fields)))
        @test lengths ≈ Cs .* Array(znodes(grid, Center())) rtol=1e-5
    end
end

@testset "TKEBasedTurbulenceClosure in an AtmosphereModel [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = 1000
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    zf = znodes(grid, Face())
    closure = TKEBasedTurbulenceClosure(mixing_length = GradientLimitedMixingLength(),
                                        stability_functions = ConstantStabilityFunctions())

    @testset "the TKE tracer and the closure fields" begin
        model = AtmosphereModel(grid; closure)
        @test TKE_NAME === :ρe
        @test :ρe ∈ keys(model.tracers)
        @test :ρe ∈ keys(Oceananigans.prognostic_fields(model))
        @test model.closure_fields isa TKEClosureFields
        @test keys(model.closure_fields.tupled_tracer_diffusivities) == (:ρθ, :ρqᵛ, :ρe)
        @test model.closure_fields.tupled_tracer_diffusivities.ρe === model.closure_fields.Kᵉ
        @test model.closure_fields.tupled_tracer_diffusivities.ρθ === model.closure_fields.Kᶜ
        @test model.closure_fields.N² isa Field{Center, Center, Face}

        # A user tracer coexists with the closure's, and naming the closure's tracer is harmless
        model = AtmosphereModel(grid; closure, tracers = :ρc)
        @test :ρc ∈ keys(model.tracers)
        @test :ρe ∈ keys(model.tracers)
        model = AtmosphereModel(grid; closure, tracers = :ρe)
        @test count(==(:ρe), keys(Oceananigans.prognostic_fields(model))) == 1

        # Prognostic names must be unique: a tracer cannot take the name of another prognostic
        @test_throws ArgumentError AtmosphereModel(grid; closure, tracers = :ρqᵛ)
    end

    @testset "mixing length and diffusivities in a neutral column" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Kᵉ = column(model.closure_fields.Kᵉ)
        sf = closure.stability_functions
        ℂˢ = closure.mixing_length.ℂˢ
        interior_faces = 2:Nz

        # In neutral air the mixing length is ℂˢ times the height above the surface: the ground is
        # the only obstacle, the surface face is a zero — the diffusivities there are masked
        # regardless — and the top face is unmasked.
        ℓ = column(diagnosed_mixing_length(model))
        Δz = Lz / Nz
        @test ℓ[1] == 0
        @test ℓ[Nz+1] ≈ ℂˢ * Lz
        @test all(ℓ[interior_faces] .≈ ℂˢ .* zf[interior_faces])

        # Kᵘ = ℂᵘ ℓ √e, masked on the boundary faces
        @test Kᵘ[1] == 0
        @test Kᵘ[Nz+1] == 0
        @test all(Kᵘ[interior_faces] .≈ sf.ℂᵘ .* ℓ[interior_faces] .* sqrt(e₀))

        # The ratios Kᶜ/Kᵘ, Kᵉ/Kᵘ are the stability-function ratios
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.ℂᶜ / sf.ℂᵘ)
        @test all(Kᵉ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.ℂᵉ / sf.ℂᵘ)
    end

    @testset "the buoyancy penetration depth" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        set!(model; θ = z -> 300 + Γ * z)
        set_tke!(model, e₀)

        ℓ = column(diagnosed_mixing_length(model))
        θ = column(model.formulation.potential_temperature)
        g = model.thermodynamic_constants.gravitational_acceleration
        Δz = Lz / Nz
        ℂˢ = closure.mixing_length.ℂˢ

        diagnosed_N² = column(model.closure_fields.N²)
        for k in 2:Nz
            N² = g * (log(θ[k]) - log(θ[k-1])) / Δz
            # The density/temperature round trip loses digits in Float32. Check that diagnosis
            # separately, then test the length against the actual diagnosed stability.
            @test diagnosed_N²[k] ≈ N² rtol=(FT === Float32 ? 2e-3 : 1e-5)
            ℓᵇ = sqrt(e₀) / sqrt(diagnosed_N²[k])
            @test ℓ[k] ≈ min(ℂˢ * zf[k], ℓᵇ) rtol=1e-5
        end

        # Stratification limits the length well above the surface
        @test ℓ[Nz] < ℂˢ * zf[Nz] / 2

        # The stored static stability of this dry column is the buoyancy gradient ∂z_b, computed
        # once per stage: the default moist static stability reduces to it in subsaturated air
        N²_stored = column(model.closure_fields.N²)
        N²_direct = column(Field(KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, buoyancy_force(model), buoyancy_tracers(model))))
        @test all(N²_stored[2:Nz] .== N²_direct[2:Nz])
        @test all(N²_stored[2:Nz] .> 0)
    end

    @testset "equivalence with the ℓ = min(z, Cᴺ √e / N) normalization" begin
        # The parameters of the earlier normalization — Nakanishi & Niino's coefficients with
        # Deardorff's Cᴺ on the stratification length — mapped through the one named conversion
        # must reproduce that closure exactly: Kᵘ = ℂᵘ ℓ √e and ω = ℂᴰ √e / ℓ with
        # ℓ = min(z, Cᴺ √e / N), here written out at the faces and centers.
        Cᴺ = FT(0.76)
        legacy = ConstantStabilityFunctions(ℂᵘ = 0.196, ℂᶜ = 0.265, ℂᵉ = 0.392, ℂᴰ = 0.295)
        mixing_length, stability_functions = absorb_stratification_coefficient(Cᴺ, legacy)
        mapped = TKEBasedTurbulenceClosure(; mixing_length, stability_functions)
        @test mapped.mixing_length.ℂˢ ≈ 1 / Cᴺ
        @test mapped.stability_functions.ℂᵘ * mapped.mixing_length.ℂˢ ≈ legacy.ℂᵘ
        @test mapped.stability_functions.ℂᴰ / mapped.mixing_length.ℂˢ ≈ legacy.ℂᴰ

        model = AtmosphereModel(grid; closure = mapped, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        set!(model; θ = z -> 300 + Γ * z)
        set_tke!(model, e₀)

        θ = column(model.formulation.potential_temperature)
        g = model.thermodynamic_constants.gravitational_acceleration
        Δz = Lz / Nz
        # This checks the length normalization, not the separately tested stability diagnosis.
        N² = column(model.closure_fields.N²)[2:Nz]
        ℓᶠ = [min(zf[k], Cᴺ * sqrt(e₀) / sqrt(N²[k-1])) for k in 2:Nz]

        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Kᵉ = column(model.closure_fields.Kᵉ)
        @test all(isapprox.(Kᵘ[2:Nz], legacy.ℂᵘ .* ℓᶠ .* sqrt(e₀); rtol = 1e-5))
        @test all(isapprox.(Kᶜ[2:Nz], legacy.ℂᶜ .* ℓᶠ .* sqrt(e₀); rtol = 1e-5))
        @test all(isapprox.(Kᵉ[2:Nz], legacy.ℂᵉ .* ℓᶠ .* sqrt(e₀); rtol = 1e-5))

        # The mapped closure is the local minimum, so its sink rate at the interior centers is the
        # legacy formula with N² and B = -Kᶜ N² reconstructed from the two adjacent faces, plus the
        # negative buoyancy flux divided by e
        @test mapped.mixing_length isa LocalMinimumMixingLength
        Lᵉ = column(model.closure_fields.Lᵉ)
        zc = znodes(grid, Center())
        for k in 2:Nz-1
            N²ᶜ = (N²[k-1] + N²[k]) / 2
            ℓᶜ = min(zc[k], Cᴺ * sqrt(e₀) / sqrt(N²ᶜ))
            B = -(Kᶜ[k] * N²[k-1] + Kᶜ[k+1] * N²[k]) / 2
            @test Lᵉ[k] ≈ -(legacy.ℂᴰ * sqrt(e₀) / ℓᶜ - B / e₀) rtol=1e-5
        end
    end


    @testset "the envelope: an elevated neutral layer is bounded by the stratified air around it" begin
        # Stable below 300 m, neutral between 300 and 700 m, stable above: min(ℂˢ z, ℓᵇ) gives the
        # wall length ℂˢ z ≈ 400–900 m inside the neutral layer, where nothing is stratified; the
        # envelope is bounded by the stratified faces just below and above the layer.
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        θᵢ(z) = z < 300 ? 300 + Γ * z : z < 700 ? 300 + Γ * 300 : 300 + Γ * (z - 400)
        set!(model; θ = θᵢ)
        set_tke!(model, e₀)
        ℂˢ = closure.mixing_length.ℂˢ
        Δz = Lz / Nz

        ℓ = column(diagnosed_mixing_length(model))
        N² = column(model.closure_fields.N²)
        ℓᵇ = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : Inf for k in eachindex(zf)]

        # Exactly the envelope, at every face
        @test all(isapprox.(ℓ, brute_force_envelope(grid, ℓᵇ, ℂˢ); rtol = 1e-6))

        # Bounded by every obstacle, and by its neighbors through the slope ℂˢ
        @test all(ℓ[2:end] .≤ ℂˢ .* zf[2:end] .+ 1e-6)
        @test all(ℓ .≤ ℓᵇ .+ 1e-6)
        @test all(abs.(diff(ℓ)) .≤ ℂˢ * Δz * (1 + 1e-6))

        # In the neutral layer the mixing length is well below the wall length: the stratified
        # faces above and below bind it through ℂˢ times the distance to them
        neutral = findall(z -> 350 < z < 650, zf)
        @test all(ℓ[neutral] .< 0.75 .* ℂˢ .* zf[neutral])
        # Density/temperature reconstruction in Float32 introduces O(eps*g/Δz) noise.
        g = model.thermodynamic_constants.gravitational_acceleration
        neutral_tolerance = max(1e-10, 8eps(FT) * g / Δz)
        @test all(abs.(N²[neutral]) .< neutral_tolerance)

        # In the stratified layer below, the point itself binds: the familiar min(ℂˢ z, ℓᵇ)
        stratified = findall(z -> 100 < z < 250, zf)
        @test all(isapprox.(ℓ[stratified], min.(ℂˢ .* zf[stratified], ℓᵇ[stratified]); rtol = 1e-5))

        # The dissipation reads the envelope at the centers, halfway between the faces, bounded by
        # the center's own local minimum (its wall length and penetration depth)
        e = model.tracers.ρe / model.dynamics.reference_state.density
        ε = column(Field(KernelFunctionOperation{Center, Center, Center}(dissipationᶜᶜᶜ, grid, closure, e,
                                                                        model.velocities, model.closure_fields)))
        zc = znodes(grid, Center())
        ℓᵇᶜ = [(N²[k] + N²[k+1]) / 2 > 0 ? sqrt(e₀) / sqrt((N²[k] + N²[k+1]) / 2) : Inf for k in 1:Nz]
        ℓᶜ = [min(min(ℓ[k], ℓ[k+1]) + ℂˢ * Δz / 2, ℂˢ * zc[k], ℓᵇᶜ[k]) for k in 1:Nz]
        @test all(isapprox.(ε[2:Nz-1], closure.stability_functions.ℂᴰ .* e₀^(3/2) ./ ℓᶜ[2:Nz-1]; rtol = 1e-6))
    end

    @testset "the three formulations on the elevated neutral layer" begin
        e₀ = FT(0.5)
        Γ = FT(0.005)
        θᵢ(z) = z < 300 ? 300 + Γ * z : z < 700 ? 300 + Γ * 300 : 300 + Γ * (z - 400)
        ℂˢ = FT(1.316)
        Δz = Lz / Nz
        lengths = Dict()
        N² = nothing
        for (name, mixing_length) in (:local => LocalMinimumMixingLength(ℂˢ = ℂˢ), :integral => IntegralMixingLength(ℂˢ = ℂˢ), :envelope => GradientLimitedMixingLength(ℂˢ = ℂˢ))
            model = AtmosphereModel(grid; closure = TKEBasedTurbulenceClosure(; mixing_length), advection = nothing)
            set!(model; θ = θᵢ)
            set_tke!(model, e₀)
            lengths[name] = column(diagnosed_mixing_length(model))
            N² = column(model.closure_fields.N²)
        end
        ℓᵇ = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : Inf for k in eachindex(zf)]
        wall = ℂˢ .* zf
        neutral = findall(z -> 350 < z < 650, zf)
        stratified = findall(z -> 100 < z < 250, zf)

        # The local minimum is the wall length through the neutral layer, where nothing else binds it
        @test all(isapprox.(lengths[:local][2:end], min.(wall, ℓᵇ)[2:end]; rtol = 1e-6))
        @test all(isapprox.(lengths[:local][neutral], wall[neutral]; rtol = 1e-6))

        # Both nonlocal formulations bound it by the stratified air around the layer, and agree with
        # the local minimum where only the ground and the level itself bind
        for name in (:integral, :envelope)
            ℓ = lengths[name]
            @test all(ℓ[neutral] .< 0.75 .* wall[neutral])
            @test all(ℓ[2:end] .≤ wall[2:end] .* (1 + 1e-6))
            @test all(isapprox.(ℓ[stratified], min.(wall, ℓᵇ)[stratified]; rtol = 1e-3))
        end

        # The parcel lengths: in uniform stratification the deficit is N² s² / 2 and the parcel stops
        # at √e / N, so the integral reproduces the penetration depth; a parcel released in the neutral
        # layer stops where it has penetrated the stratified air by about that much, never farther
        # from the nearest stratified face than the penetration depth there
        ℓ = lengths[:integral]
        below = maximum(filter(k -> N²[k] > 0 && zf[k] < 400, eachindex(zf)))
        above = minimum(filter(k -> N²[k] > 0 && zf[k] > 600, eachindex(zf)))
        for k in neutral
            geometric = ℂˢ * min(zf[k] - zf[below], zf[above] - zf[k])
            @test geometric ≤ ℓ[k] ≤ geometric + ℂˢ * max(ℓᵇ[below], ℓᵇ[above]) + ℂˢ * Δz
        end
    end

    @testset "the envelope on a stretched grid" begin
        # Cell heights vary, so the sweeps must add the height of the cell between the faces
        z_stretched = PiecewiseStretchedDiscretization(z = [0, Lz], Δz = [Lz / (2Nz), 2Lz / Nz])
        stretched = RectilinearGrid(default_arch; size = length(z_stretched) - 1, z = z_stretched, topology = (Flat, Flat, Bounded))
        model = AtmosphereModel(stretched; closure, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        θᵢ(z) = z < 300 ? 300 + Γ * z : z < 700 ? 300 + Γ * 300 : 300 + Γ * (z - 400)
        set!(model; θ = θᵢ)
        set_tke!(model, e₀)
        ℂˢ = closure.mixing_length.ℂˢ
        zf_stretched = Array(znodes(stretched, Face()))

        ℓ = column(diagnosed_mixing_length(model))
        N² = column(model.closure_fields.N²)
        ℓᵇ = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : Inf for k in eachindex(zf_stretched)]
        @test all(isapprox.(ℓ, brute_force_envelope(stretched, ℓᵇ, ℂˢ); rtol = 1e-6))
    end
    @testset "diffusivity caps" begin
        capped = TKEBasedTurbulenceClosure(maximum_viscosity = 1e-3, maximum_tracer_diffusivity = 2e-3,
                                           maximum_tke_diffusivity = 3e-3)
        model = AtmosphereModel(grid; closure = capped, advection = nothing)
        set!(model; θ = 300)
        set_tke!(model, FT(1))
        @test maximum(column(model.closure_fields.Kᵘ)) ≈ 1e-3
        @test maximum(column(model.closure_fields.Kᶜ)) ≈ 2e-3
        @test maximum(column(model.closure_fields.Kᵉ)) ≈ 3e-3
    end

    @testset "a sheared, capped boundary layer stays finite and positive" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        θᵢ(z) = 300 + 0.01 * max(0, z - 500)
        uᵢ(z) = 5 * min(1, z / 300)
        set!(model; θ = θᵢ, u = uᵢ)
        set_tke!(model, FT(0.1))

        for _ in 1:20
            time_step!(model, 10)
        end

        ρe = column(model.tracers.ρe)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        @test all(isfinite, ρe)
        @test all(ρe .≥ 0)
        @test all(isfinite, Kᵘ)
        @test maximum(Kᵘ) > 0
        sf = closure.stability_functions
        @test all(Kᶜ[2:Nz] .≈ Kᵘ[2:Nz] .* (sf.ℂᶜ / sf.ℂᵘ))
    end

    @testset "the implicit linear coefficient is the dissipation rate" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        @test model.closure_fields.tupled_implicit_linear_coefficients.ρe === model.closure_fields.Lᵉ
        @test model.closure_fields.tupled_implicit_linear_coefficients.ρθ isa Oceananigans.Fields.ZeroField

        # Neutral air, no buoyancy flux: Lᵉ = - Sᴰ √e / ℓ with ℓ = ℂˢ z at the cell centers
        Lᵉ = column(model.closure_fields.Lᵉ)
        zc = znodes(grid, Center())
        ℂᴰ = closure.stability_functions.ℂᴰ
        ℂˢ = closure.mixing_length.ℂˢ
        @test all(Lᵉ .≈ -ℂᴰ * sqrt(e₀) ./ (ℂˢ .* zc))

        # The dissipation diagnostic is the same rate times e
        e = model.tracers.ρe / model.dynamics.reference_state.density
        ε = column(Field(KernelFunctionOperation{Center, Center, Center}(dissipationᶜᶜᶜ, grid, closure, e,
                                                                        model.velocities, model.closure_fields)))
        @test all(ε .≈ -Lᵉ .* e₀)
        @test all(Lᵉ .< 0)

        # Below the minimum TKE the dissipation rate keeps following √e — the floor applies to
        # the diffusivities, not to the dissipation (as in CATKE)
        e₋ = FT(1e-8)
        @test e₋ < closure.minimum_tke
        set_tke!(model, e₋)
        Lᵉ = column(model.closure_fields.Lᵉ)
        @test all(Lᵉ .≈ -ℂᴰ * sqrt(e₋) ./ (ℂˢ .* zc))
    end

    @testset "the buoyancy sink is continuous at the minimum TKE" begin
        # ωᴮ = -B⁻ / max(e, eᵐⁱⁿ): unchanged above the floor and constant below it, so that the
        # implicit buoyancy sink ωᴮ e is continuous at e = eᵐⁱⁿ and falls linearly with e. The
        # buoyancy flux is passed to the sink rate directly; B = 0 gives the dissipation rate alone.
        model = AtmosphereModel(grid; closure, advection = nothing)
        set!(model; θ = z -> 300 + 0.005z)
        eᵐⁱⁿ = closure.minimum_tke
        τ = closure.negative_tke_damping_time_scale
        B = -FT(1e-7)
        δ = FT === Float64 ? 1e-6 : 1e-3
        rtol = FT === Float64 ? 1e-10 : 1e-4
        function sink_rates(e₀)
            set_tke!(model, e₀)
            e = model.tracers.ρe / model.dynamics.reference_state.density
            rate(b) = column(Field(KernelFunctionOperation{Center, Center, Center}(tke_sink_rate, grid, model.closure, e, b,
                                                                                   model.velocities, model.closure_fields)))
            ω = rate(zero(FT))
            return (; e = column(Field(e)), ω, ωᴮ = rate(B) .- ω)
        end

        # The left and right limits at the floor agree, for the rate and for the sink
        below, at, above = sink_rates(eᵐⁱⁿ * (1 - δ)), sink_rates(eᵐⁱⁿ), sink_rates(eᵐⁱⁿ * (1 + δ))
        @test all(below.e .< eᵐⁱⁿ .< above.e)
        @test isapprox(below.ω .+ below.ωᴮ, above.ω .+ above.ωᴮ; rtol = 4δ)
        @test isapprox(below.ωᴮ .* below.e, above.ωᴮ .* above.e; rtol = 4δ)
        @test all(isapprox.(at.ωᴮ, -B / eᵐⁱⁿ; rtol))

        # Above the floor the rate is -B⁻ / e, as without the floor
        for e₀ in FT.((2eᵐⁱⁿ, 10eᵐⁱⁿ, 1e-4))
            r = sink_rates(e₀)
            @test all(isapprox.(r.ωᴮ, -B ./ r.e; rtol))
        end

        # Below the floor it is -B⁻ / eᵐⁱⁿ, so the sink ωᴮ e is linear in e, and it adds to the
        # damping of negative TKE, which it also drives towards zero
        for e₀ in FT.((eᵐⁱⁿ / 2, eᵐⁱⁿ / 4))
            r = sink_rates(e₀)
            @test all(isapprox.(r.ωᴮ, -B / eᵐⁱⁿ; rtol))
        end
        r = sink_rates(-FT(0.01))
        @test all(isapprox.(r.ω, 1 / τ; rtol))
        @test all(isapprox.(r.ωᴮ, -B / eᵐⁱⁿ; rtol))
    end

    @testset "the sources enter the stage tendency" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        S = FT(0.01)
        set!(model; θ = 300, u = z -> S * z)
        set_tke!(model, e₀)

        Gρe = model.timestepper.Gⁿ.ρe
        fill!(Gρe, 0)
        Breeze.AtmosphereModels.compute_closure_tendencies!(model)

        # Neutral: only shear production, P = Kᵘ S² at the faces averaged to the centers; in the
        # first and last cells the masked boundary face is replaced by its interior neighbour
        ρ = column(model.dynamics.reference_state.density)
        Kᵘ = column(model.closure_fields.Kᵘ)
        P = [S^2 * (Kᵘ[k] + Kᵘ[k+1]) / 2 for k in 1:Nz]
        P[1] = S^2 * Kᵘ[2]
        P[Nz] = S^2 * Kᵘ[Nz]
        @test all(column(Gρe) .≈ ρ .* P)
        @test all(column(Gρe) .> 0)

        passive = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=FT(1), κ=FT(1))
        for closures in ((closure, passive), (passive, closure), (passive, closure, passive))
            tuple_model = AtmosphereModel(grid; closure=closures, advection=nothing)
            set!(tuple_model; θ=300, u=z -> S * z)
            set_tke!(tuple_model, e₀)

            fill!(tuple_model.timestepper.Gⁿ.ρe, 0)
            Breeze.AtmosphereModels.compute_closure_tendencies!(tuple_model)
            @test column(tuple_model.timestepper.Gⁿ.ρe) ≈ column(Gρe)
        end
    end

    @testset "dissipation decays TKE at the analytic rate" begin
        # No TKE diffusion, no wind, no stratification: each cell decays as ∂ₜe = -ℂᴰ e^{3/2}/ℓ
        # with ℓ = ℂˢ z, i.e. e(t) = (e₀^{-1/2} + ℂᴰ t / 2ℓ)^{-2}
        undiffused = TKEBasedTurbulenceClosure(maximum_tke_diffusivity = 0,
                                             stability_functions = ConstantStabilityFunctions())
        model = AtmosphereModel(grid; closure = undiffused, advection = nothing)
        e₀ = FT(1)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        Δt = FT(1)
        Nt = 100
        for _ in 1:Nt
            time_step!(model, Δt)
        end

        e = column(model.tracers.ρe) ./ column(model.dynamics.reference_state.density)
        zc = znodes(grid, Center())
        ℂᴰ = undiffused.stability_functions.ℂᴰ
        ℂˢ = undiffused.mixing_length.ℂˢ
        t = Nt * Δt
        e_analytic = @. (e₀^(-1/2) + ℂᴰ * t / (2 * ℂˢ * zc))^(-2)
        # The sinks are implicit per stage, so the decay is first-order accurate in Δt ω ≈ 0.02
        @test all(e .≥ 0)
        @test all(isapprox.(e, e_analytic; rtol = 0.05))
    end

    @testset "negative TKE is damped on the damping time scale" begin
        τ = FT(60)
        # Without tracer diffusivity the buoyancy flux -Kᶜ N² vanishes exactly. Otherwise the
        # rounding-level N² that the stages leave in a Float32 neutral column would add its
        # buoyancy sink rate -B⁻ / eᵐⁱⁿ, which applies below the minimum TKE, to the damping.
        undiffused = TKEBasedTurbulenceClosure(mixing_length = GradientLimitedMixingLength(),
                                               stability_functions = ConstantStabilityFunctions(),
                                               maximum_tracer_diffusivity = 0)
        model = AtmosphereModel(grid; closure = undiffused, advection = nothing)
        set!(model; θ = 300)
        set_tke!(model, FT(-0.1))

        # No shear, no buoyancy flux and no TKE gradient, so the damping rate 1/τ is the only
        # term. It is applied implicitly in each of the three SSP-RK3 stages with αΔt = Δt, Δt/4,
        # 2Δt/3, and the stage combinations make one step of length τ multiply a uniform e by
        # [1/3 + 2/3 (3/4 + 1/4 (1/2)) / (5/4)] / (5/3) = 0.48
        time_step!(model, τ)
        e = column(model.tracers.ρe) ./ column(model.dynamics.reference_state.density)
        @test all(e .≈ -0.048)
    end

    @testset "explicit time discretization" begin
        explicit = TKEBasedTurbulenceClosure(ExplicitTimeDiscretization())
        model = AtmosphereModel(grid; closure = explicit, advection = nothing)
        set!(model; θ = z -> 300 + 0.003z, u = z -> 5 * min(1, z / 300))
        set_tke!(model, FT(0.1))
        for _ in 1:5
            time_step!(model, 1)
        end
        @test all(isfinite, column(model.tracers.ρe))
        @test all(column(model.tracers.ρe) .≥ 0)
    end

    @testset "compressible dynamics" begin
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = 300)
        model = AtmosphereModel(grid; dynamics, closure)
        @test keys(model.closure_fields.tupled_tracer_diffusivities) == (:ρθ, :ρqᵛ, :ρe)
        set!(model; θ = 300, ρ = model.dynamics.reference_state.density)
        time_step!(model, 1)
        @test model.clock.iteration == 1
        @test all(isfinite, Array(interior(model.dynamics.dry_density)))
        @test all(isfinite, Array(interior(model.tracers.ρe)))
    end
end

#####
##### Moist static stability
#####

using Breeze.TurbulenceClosures: static_stabilityᶜᶜᶠ

@testset "MoistStaticStability [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 40
    Lz = 1000
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    interior_faces = 2:Nz

    @test isbits(MoistStaticStability())
    moist = TKEBasedTurbulenceClosure(static_stability = MoistStaticStability())
    @test moist.static_stability isa MoistStaticStability
    @test occursin("MoistStaticStability", sprint(show, moist))

    # A model with saturation adjustment and its stored N² for a given static stability
    function stored_N²(static_stability, equilibrium; θ, qᵗ)
        microphysics = SaturationAdjustment(; equilibrium)
        closure = TKEBasedTurbulenceClosure(; static_stability)
        model = AtmosphereModel(grid; closure, microphysics, advection = nothing)
        set!(model; θ, qᵗ)
        set_tke!(model, FT(0.1))
        return column(model.closure_fields.N²), model
    end

    @testset "a saturated adiabat is neutral to a saturated displacement" begin
        # Uniform liquid-water potential temperature and total water, saturated throughout: the
        # model's own saturated adiabat. It is stable to a dry displacement — θᵨ increases with
        # height as condensation warms the air — and neutral to a saturated one, up to the
        # approximations of the Durran–Klemp expression (a few percent of the dry value).
        θ₀ = 288
        qᵗ₀ = 13e-3
        N²ᵈ, dry_model = stored_N²(DryStaticStability(), WarmPhaseEquilibrium(); θ = θ₀, qᵗ = qᵗ₀)
        N²ˢ, moist_model = stored_N²(MoistStaticStability(), WarmPhaseEquilibrium(); θ = θ₀, qᵗ = qᵗ₀)

        qˡ = column(moist_model.microphysical_fields.qˡ)
        @test all(qˡ .> 1e-4)
        @test all(N²ᵈ[interior_faces] .> 5e-5)
        @test maximum(abs, N²ˢ[interior_faces]) < 0.1 * maximum(N²ᵈ[interior_faces])
    end

    @testset "the dry branch is selected where the air is subsaturated" begin
        # Subsaturated below mid-depth, saturated above; the face at the step sees the
        # interpolated state
        qᵗ(z) = ifelse(z < Lz / 2, 4e-3, 13e-3)
        N²ᵈ, _ = stored_N²(DryStaticStability(), WarmPhaseEquilibrium(); θ = 288, qᵗ)
        N²ˢ, model = stored_N²(MoistStaticStability(), WarmPhaseEquilibrium(); θ = 288, qᵗ)
        qˡ = column(model.microphysical_fields.qˡ)
        k★ = Nz ÷ 2 + 1 # the first cloudy cell
        @test qˡ[k★-1] == 0
        @test qˡ[k★] > 0

        # Both centers subsaturated: exactly the dry value; both saturated: reduced stability
        @test all(N²ˢ[2:k★-1] .== N²ᵈ[2:k★-1])
        @test all(N²ˢ[k★+1:Nz] .< N²ᵈ[k★+1:Nz])
        @test all(isfinite, N²ˢ)

        # The KernelFunctionOperation form of the same diagnostic agrees with the stored field
        op = KernelFunctionOperation{Center, Center, Face}(static_stabilityᶜᶜᶠ, grid, MoistStaticStability(),
                                                           buoyancy_force(model), buoyancy_tracers(model))
        @test column(Field(op)) == N²ˢ
    end

    @testset "mixed phase" begin
        # Cold and saturated: the equilibrium's liquid fraction interpolates the saturation vapor
        # pressure and the latent heat between liquid and ice; the result is finite and, on a
        # saturated adiabat, much less stable than the dry value
        equilibrium = MixedPhaseEquilibrium()
        N²ᵈ, _ = stored_N²(DryStaticStability(), equilibrium; θ = 250, qᵗ = 1.5e-3)
        N²ˢ, model = stored_N²(MoistStaticStability(), equilibrium; θ = 250, qᵗ = 1.5e-3)
        qⁱ = column(model.microphysical_fields.qⁱ)
        @test all(qⁱ .> 0)
        @test all(isfinite, N²ˢ)
        @test all(N²ᵈ[interior_faces] .> 0)
        @test maximum(abs, N²ˢ[interior_faces]) < 0.2 * maximum(N²ᵈ[interior_faces])
    end
end

#####
##### Richardson-number-dependent stability functions
#####

using Breeze.TurbulenceClosures: rational_stability, richardson_number, Riᶜᶜᶠ,
                                 momentum_stability_function, tracer_stability_function,
                                 tke_stability_function, dissipation_stability_function

#####
##### Rational stability functions
#####

@testset "RationalStabilityFunction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = 1000
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    zf = znodes(grid, Face())
    zc = znodes(grid, Center())
    interior_faces = 2:Nz

    Riᶠ(model) = column(Field(KernelFunctionOperation{Center, Center, Face}(Riᶜᶜᶠ, grid, model.closure,
                                                                           model.velocities,
                                                                           model.closure_fields.N²)))

    @testset "a windless column with stable and unstable layers stays finite" begin
        # No shear, so Ri = N² / S²ᵐⁱⁿ at every interface, large and finite, and the sign changes
        # at mid-depth. At the cell center there Ri is formed from the reconstructed N² and S².
        closure = TKEBasedTurbulenceClosure()
        sf = closure.stability_functions
        model = AtmosphereModel(grid; closure, advection = nothing)
        θᵢ(z) = 300 - 0.005 * min(z, Lz / 2) + 0.01 * max(0, z - Lz / 2)
        set!(model; θ = θᵢ)
        set_tke!(model, FT(0.5))

        Ri = column(Field(KernelFunctionOperation{Center, Center, Face}(Riᶜᶜᶠ, grid, model.closure, model.velocities, model.closure_fields.N²)))
        @test all(isfinite, Ri[interior_faces])
        @test all(abs.(Ri[interior_faces]) .> 1e4 * max(sf.Ri⁻, sf.Ri⁺))
        @test any(Ri[interior_faces] .< 0) && any(Ri[interior_faces] .> 0)

        Lᵉ = column(model.closure_fields.Lᵉ)
        Kᵘ = column(model.closure_fields.Kᵘ)
        @test all(isfinite, Lᵉ)
        @test all(Lᵉ .< 0)
        @test all(isfinite, Kᵘ)

        for _ in 1:10
            time_step!(model, 10)
        end
        ρe = column(model.tracers.ρe)
        @test all(isfinite, ρe)
        @test all(ρe .≥ 0)
    end

    @testset "with the moist static stability" begin
        closure = TKEBasedTurbulenceClosure(; static_stability = MoistStaticStability())
        microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; closure, microphysics, advection = nothing)
        set!(model; θ = 288, qᵗ = z -> ifelse(z < Lz / 2, 4e-3, 13e-3), u = z -> 0.005 * z)
        set_tke!(model, FT(0.1))
        for _ in 1:5
            time_step!(model, 10)
        end
        @test all(isfinite, column(model.tracers.ρe))
        @test all(isfinite, column(model.closure_fields.Kᶜ))
    end
    @testset "construction, defaults and validation" begin
        sf = Breeze.TurbulenceClosures.convert_eltype(FT, RationalStabilityFunction())
        @test isbits(sf) # a GPU kernel can only carry an isbits closure
        @test sf isa RationalStabilityFunction{FT}

        # Calibrated to the Shen et al. (2022) LES library
        @test sf.ℂᵘ⁰ === FT(0.15439310625401925)
        @test sf.ℂᶜ⁰ === FT(0.1929919973602801)
        @test sf.Ri⁻ === FT(6.562979157594204)
        @test sf.Ri⁺ === FT(15.12797320010583)
        # Twelve endpoints and two transition scales
        @test length(fieldnames(RationalStabilityFunction)) == 14

        # Mixed integer and float keyword arguments are promoted, as elsewhere in the closure
        mixed = Breeze.TurbulenceClosures.convert_eltype(FT, RationalStabilityFunction(ℂᵘ⁻ = 1, Ri⁺ = 2))
        @test mixed.ℂᵘ⁻ === FT(1)
        @test mixed.Ri⁺ === FT(2)
        @test mixed.ℂᵘ⁰ === FT(0.15439310625401925)

        closure = TKEBasedTurbulenceClosure(stability_functions = RationalStabilityFunction())
        @test closure.stability_functions isa RationalStabilityFunction{FT}
        @test isbits(closure)
        closure32 = TKEBasedTurbulenceClosure(Float32; stability_functions = RationalStabilityFunction())
        @test closure32.stability_functions isa RationalStabilityFunction{Float32}
        closure64 = TKEBasedTurbulenceClosure(Float64; stability_functions = RationalStabilityFunction())
        @test closure64.stability_functions isa RationalStabilityFunction{Float64}

        str = sprint(show, closure)
        @test occursin("RationalStabilityFunction", str)
        @test occursin("transition scales", str)
        @test occursin("Ri → -∞", sprint(show, sf))

        # Every endpoint and both scales must be positive
        @test_throws ArgumentError RationalStabilityFunction(ℂᵘ⁰ = 0)
        @test_throws ArgumentError RationalStabilityFunction(ℂᴰ⁺ = -1)
        @test_throws ArgumentError RationalStabilityFunction(ℂᵉ⁻ = NaN)
        @test_throws ArgumentError RationalStabilityFunction(Ri⁻ = 0)
        @test_throws ArgumentError RationalStabilityFunction(Ri⁺ = -0.5)
    end

    @testset "the rational shape and its limits" begin
        C⁻, C⁰, C⁺, Ri⁻, Ri⁺ = FT.((1, 2, 5, 0.5, 2))
        S(Ri) = rational_stability(FT(Ri), C⁻, C⁰, C⁺, Ri⁻, Ri⁺)

        # The three endpoints, exactly: neutral at Ri = 0 and the two asymptotes at ±∞
        @test S(0) === C⁰
        @test S(-0.0) === C⁰
        @test S(Inf) === C⁺
        @test S(-Inf) === C⁻
        @test S(0) isa FT

        # Halfway to each asymptote at that branch's transition scale
        @test S(Ri⁺) ≈ (C⁰ + C⁺) / 2
        @test S(-Ri⁻) ≈ (C⁰ + C⁻) / 2
        # and nine tenths of the way at nine times the scale
        @test S(9Ri⁺) ≈ C⁰ + 9 * (C⁺ - C⁰) / 10
        @test S(-9Ri⁻) ≈ C⁰ + 9 * (C⁻ - C⁰) / 10

        # The closed form S = C⁰ + (C - C⁰) |Ri| / (|Ri| + Ri^±)
        for Ri in FT.((0.01, 0.3, 1, 7, 250))
            @test S(Ri) ≈ C⁰ + (C⁺ - C⁰) * Ri / (Ri + Ri⁺)
            @test S(-Ri) ≈ C⁰ + (C⁻ - C⁰) * Ri / (Ri + Ri⁻)
        end

        # The two branches are independent: the stable scale does not touch Ri < 0
        wider = rational_stability(FT(-1), C⁻, C⁰, C⁺, Ri⁻, 100Ri⁺)
        @test wider ≈ S(-1)
        narrower = rational_stability(FT(1), C⁻, C⁰, C⁺, 100Ri⁻, Ri⁺)
        @test narrower ≈ S(1)

        # Monotone in |Ri| on each branch, and bounded by the endpoints it travels between
        stable = [S(Ri) for Ri in FT.((0, 0.1, 0.5, 1, 2, 5, 20, 1000))]
        @test issorted(stable)                          # C⁰ < C⁺ here, so S rises
        @test all(C⁰ .≤ stable .≤ C⁺)
        unstable = [S(-Ri) for Ri in FT.((0, 0.1, 0.5, 1, 2, 5, 20, 1000))]
        @test issorted(unstable, rev = true)            # C⁻ < C⁰ here, so S falls
        @test all(C⁻ .≤ unstable .≤ C⁰)

        # Nothing overflows: huge and tiny finite Richardson numbers, and the float extremes
        extremes = FT.((floatmax(FT), -floatmax(FT), sqrt(floatmax(FT)), -sqrt(floatmax(FT)),
                        floatmin(FT), -floatmin(FT), nextfloat(zero(FT)), -nextfloat(zero(FT)),
                        1e30, -1e30, 1e-30, -1e-30, Inf, -Inf, 0))
        @test all(isfinite, S.(extremes))
        @test !any(isnan, S.(extremes))
        @test all(min(C⁻, C⁰) .≤ S.(extremes) .≤ max(C⁺, C⁰))
        @test S(floatmax(FT)) ≈ C⁺
        @test S(-floatmax(FT)) ≈ C⁻
        @test S(nextfloat(zero(FT))) ≈ C⁰
        @test S(-nextfloat(zero(FT))) ≈ C⁰
    end

    @testset "the shear floor makes the Richardson number continuous" begin
        closure = TKEBasedTurbulenceClosure(FT)
        sf = closure.stability_functions
        S²ᵐⁱⁿ = closure.minimum_shear_squared
        functions = (momentum_stability_function, tracer_stability_function,
                     tke_stability_function, dissipation_stability_function)
        endpoints = ((sf.ℂᵘ⁻, sf.ℂᵘ⁰, sf.ℂᵘ⁺), (sf.ℂᶜ⁻, sf.ℂᶜ⁰, sf.ℂᶜ⁺),
                     (sf.ℂᵉ⁻, sf.ℂᵉ⁰, sf.ℂᵉ⁺), (sf.ℂᴰ⁻, sf.ℂᴰ⁰, sf.ℂᴰ⁺))

        # Neutral air is neutral with or without shear
        @test richardson_number(zero(FT), zero(FT), S²ᵐⁱⁿ) === zero(FT)
        @test richardson_number(zero(FT), FT(1e-4), S²ᵐⁱⁿ) === zero(FT)

        # Without shear, a rounding-level N² of either sign leaves every function at its neutral
        # value, to within |C± - C⁰| |N²| / (Ri± S²ᵐⁱⁿ); the unfloored N² / S² sent it to a limit
        for N² in FT.((-1e-16, 0, 1e-16)), (S, (C⁻, C⁰, C⁺)) in zip(functions, endpoints)
            Ri = richardson_number(N², zero(FT), S²ᵐⁱⁿ)
            Cᵇ, Riᵇ = N² ≥ 0 ? (C⁺, sf.Ri⁺) : (C⁻, sf.Ri⁻)
            @test abs(S(sf, Ri) - C⁰) ≤ abs(Cᵇ - C⁰) * abs(N²) / (Riᵇ * S²ᵐⁱⁿ) + 2eps(C⁰)
            @test abs(S(sf, Ri) - C⁰) ≤ 1e-3 * C⁰
        end

        # Without shear the functions saturate monotonically once |N²| ≫ Ri± S²ᵐⁱⁿ, to within
        # |C⁰ - C±| Ri± S²ᵐⁱⁿ / |N²| of their limits
        for (S, (C⁻, C⁰, C⁺)) in zip(functions, endpoints)
            magnitudes = FT.(10.0 .^ (-16:-6))
            stable = [abs(S(sf, richardson_number(N², zero(FT), S²ᵐⁱⁿ)) - C⁺) for N² in magnitudes]
            unstable = [abs(S(sf, richardson_number(-N², zero(FT), S²ᵐⁱⁿ)) - C⁻) for N² in magnitudes]
            @test issorted(stable; rev = true)
            @test issorted(unstable; rev = true)

            magnitudes = FT.(10.0 .^ (-16:2:0))
            for N² in magnitudes
                @test abs(S(sf, richardson_number(N², zero(FT), S²ᵐⁱⁿ)) - C⁺) ≤
                      abs(C⁰ - C⁺) * sf.Ri⁺ * S²ᵐⁱⁿ / N² + 4eps(max(C⁰, C⁺))
                @test abs(S(sf, richardson_number(-N², zero(FT), S²ᵐⁱⁿ)) - C⁻) ≤
                      abs(C⁰ - C⁻) * sf.Ri⁻ * S²ᵐⁱⁿ / N² + 4eps(max(C⁰, C⁻))
            end
        end

        # With shear well above the floor, Ri is N² / S² to relative order S²ᵐⁱⁿ / S²
        for S² in FT.((1e-6, 1e-4, 1e-2, 1)), N² in FT.((-1e-2, -1e-4, 1e-4, 1e-2))
            Ri = richardson_number(N², S², S²ᵐⁱⁿ)
            @test abs(Ri - N² / S²) ≤ abs(N² / S²) * (S²ᵐⁱⁿ / S² + 2eps(FT))
        end

        # Never NaN, for zero, subnormal, small, huge and infinite static stability of either sign
        # and shear — short of both being infinite at once, a state that has already blown up
        tiny = nextfloat(zero(FT))
        N²s = (zero(FT), -zero(FT), tiny, -tiny, FT(1e-16), FT(-1e-16), one(FT), -one(FT),
               floatmax(FT), -floatmax(FT), FT(Inf), FT(-Inf))
        S²s = (zero(FT), tiny, FT(1e-16), one(FT), floatmax(FT), FT(Inf))
        for N² in N²s, S² in S²s
            isinf(N²) && isinf(S²) && continue
            Ri = richardson_number(N², S², S²ᵐⁱⁿ)
            @test !isnan(Ri)
            @test all(S -> isfinite(S(sf, Ri)), functions)
        end
    end

    # In Float32 the rounding level of N² at θ ≈ 300 K, about 1e-8 s⁻², exceeds Ri± S²ᵐⁱⁿ, so this
    # protection by the default floor holds in Float64
    FT === Float64 && @testset "a shear-free neutral column is insensitive to rounding-level stratification" begin
        # Uniform wind, so S² = 0 at every interface. One-ulp changes of ρθ at alternate levels give
        # N² of either sign at the rounding level, which Ri = N² / (S² + S²ᵐⁱⁿ) maps to nearly
        # neutral, where N² / S² sent every interface to the stable or the unstable limit
        closure = TKEBasedTurbulenceClosure()
        function closure_fields(perturb)
            model = AtmosphereModel(grid; closure, advection = nothing)
            set!(model; θ = 300, u = 5)
            perturb && (interior(model.formulation.potential_temperature_density)[:, :, 1:2:Nz] .*= 1 + eps(FT))
            set_tke!(model, FT(0.5))
            return (N² = column(model.closure_fields.N²), Kᵘ = column(model.closure_fields.Kᵘ),
                    Kᶜ = column(model.closure_fields.Kᶜ), Kᵉ = column(model.closure_fields.Kᵉ),
                    Lᵉ = column(model.closure_fields.Lᵉ))
        end
        neutral = closure_fields(false)
        perturbed = closure_fields(true)
        @test all(neutral.N²[interior_faces] .== 0)
        @test any(perturbed.N²[interior_faces] .> 0) && any(perturbed.N²[interior_faces] .< 0)
        # The residual change grows with the endpoint ratios over Ri±: about 2e-3 in Kᵘ with the
        # calibrated defaults, against O(1) were an interface sent to a limit
        for name in (:Kᵘ, :Kᶜ, :Kᵉ, :Lᵉ)
            @test isapprox(perturbed[name], neutral[name]; rtol = 1e-2)
        end
    end

    @testset "the constant submodel is exact" begin
        # Three coinciding endpoints give that constant at every Richardson number, with no limit
        # to take: the term multiplying the shape factor is identically zero
        C = FT(0.37)
        S(Ri) = rational_stability(FT(Ri), C, C, C, FT(0.764), FT(0.764))
        Ris = FT.((-Inf, -1e30, -3, -0.1, 0, 0.1, 3, 1e30, Inf, floatmax(FT), -floatmax(FT)))
        @test all(S.(Ris) .=== C)

        # and the whole closure then reproduces `ConstantStabilityFunctions` in a model
        constants = (ℂᵘ = 0.149, ℂᶜ = 0.201, ℂᵉ = 0.298, ℂᴰ = 0.388)
        rational = RationalStabilityFunction(ℂᵘ⁻ = constants.ℂᵘ, ℂᵘ⁰ = constants.ℂᵘ, ℂᵘ⁺ = constants.ℂᵘ,
                                             ℂᶜ⁻ = constants.ℂᶜ, ℂᶜ⁰ = constants.ℂᶜ, ℂᶜ⁺ = constants.ℂᶜ,
                                             ℂᵉ⁻ = constants.ℂᵉ, ℂᵉ⁰ = constants.ℂᵉ, ℂᵉ⁺ = constants.ℂᵉ,
                                             ℂᴰ⁻ = constants.ℂᴰ, ℂᴰ⁰ = constants.ℂᴰ, ℂᴰ⁺ = constants.ℂᴰ)
        rational = Breeze.TurbulenceClosures.convert_eltype(FT, rational)

        for Ri in FT.((-Inf, -2, 0, 0.3, 2, Inf))
            @test momentum_stability_function(rational, Ri) === FT(constants.ℂᵘ)
            @test tracer_stability_function(rational, Ri) === FT(constants.ℂᶜ)
            @test tke_stability_function(rational, Ri) === FT(constants.ℂᵉ)
            @test dissipation_stability_function(rational, Ri) === FT(constants.ℂᴰ)
        end

        # A stratified, sheared column, where Ri varies with height, run with each of the two
        θᵢ(z) = 300 + 0.01z
        uᵢ(z) = 0.02z
        fields = map((ConstantStabilityFunctions(; constants...), rational)) do stability_functions
            closure = TKEBasedTurbulenceClosure(; stability_functions)
            model = AtmosphereModel(grid; closure, advection = nothing)
            set!(model; θ = θᵢ, u = uᵢ)
            set_tke!(model, FT(0.5))
            (Kᵘ = column(model.closure_fields.Kᵘ), Kᶜ = column(model.closure_fields.Kᶜ),
             Kᵉ = column(model.closure_fields.Kᵉ), Lᵉ = column(model.closure_fields.Lᵉ))
        end
        @test fields[1].Kᵘ == fields[2].Kᵘ
        @test fields[1].Kᶜ == fields[2].Kᶜ
        @test fields[1].Kᵉ == fields[2].Kᵉ
        @test fields[1].Lᵉ == fields[2].Lᵉ
    end

    @testset "in a model: neutral, stable and unstable columns" begin
        sf = RationalStabilityFunction()
        closure = TKEBasedTurbulenceClosure(; mixing_length = GradientLimitedMixingLength(ℂˢ = 1.131),
                                              stability_functions = sf)
        ℂˢ = closure.mixing_length.ℂˢ
        e₀ = FT(0.5)
        model = AtmosphereModel(grid; closure, advection = nothing)

        # Neutral shear: Ri = 0, so the neutral endpoints exactly, with no plateau needed to get
        # them, and the dissipation ℂᴰ⁰ √e / (ℂˢ z) — Sᴰ multiplies ε, it does not divide it
        set!(model; θ = 300, u = z -> 0.01 * z)
        set_tke!(model, e₀)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Kᵉ = column(model.closure_fields.Kᵉ)
        Lᵉ = column(model.closure_fields.Lᵉ)
        @test all(Riᶠ(model)[interior_faces] .== 0)
        @test all(Kᵘ[interior_faces] .≈ sf.ℂᵘ⁰ .* ℂˢ .* zf[interior_faces] .* sqrt(e₀))
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.ℂᶜ⁰ / sf.ℂᵘ⁰)
        @test all(Kᵉ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.ℂᵉ⁰ / sf.ℂᵘ⁰)
        @test all(Lᵉ .≈ -sf.ℂᴰ⁰ * sqrt(e₀) ./ (ℂˢ .* zc))

        # Stable stratification with weak shear: the ratio Kᶜ/Kᵘ follows the rational functions
        # pointwise, and lies between its neutral and its stable value rather than at either
        set!(model; θ = z -> 300 + 0.03 * z, u = z -> 1e-3 * z)
        set_tke!(model, e₀)
        Ri = Riᶠ(model)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        @test all(Ri[interior_faces] .> 0)
        expected = [tracer_stability_function(sf, r) / momentum_stability_function(sf, r) for r in Ri[interior_faces]]
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ expected)
        # The ratio stays strictly between its neutral and stable endpoints
        lo, hi = minmax(sf.ℂᶜ⁰ / sf.ℂᵘ⁰, sf.ℂᶜ⁺ / sf.ℂᵘ⁺)
        @test all(lo .< expected .< hi)

        # Unstable stratification with shear: the unstable branch, again pointwise, and again
        # short of its asymptote at a finite Ri
        set!(model; θ = z -> 300 - 0.01 * z, u = z -> 0.01 * z)
        set_tke!(model, e₀)
        Ri = Riᶠ(model)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        @test all(Ri[interior_faces] .< 0)
        @test all(isfinite, Ri[interior_faces])
        expected = [tracer_stability_function(sf, r) / momentum_stability_function(sf, r) for r in Ri[interior_faces]]
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ expected)
        # and again strictly between its unstable and neutral endpoints
        lo, hi = minmax(sf.ℂᶜ⁻ / sf.ℂᵘ⁻, sf.ℂᶜ⁰ / sf.ℂᵘ⁰)
        @test all(lo .< expected .< hi)
    end

    @testset "a windless column with stable and unstable layers stays finite" begin
        # No shear, so Ri = N² / S²ᵐⁱⁿ at every interface, finite but far beyond both transition
        # scales: the rational functions are within O(Ri± / |Ri|) of their asymptotes there
        sf = RationalStabilityFunction()
        closure = TKEBasedTurbulenceClosure(stability_functions = sf)
        model = AtmosphereModel(grid; closure, advection = nothing)
        θᵢ(z) = 300 - 0.005 * min(z, Lz / 2) + 0.01 * max(0, z - Lz / 2)
        set!(model; θ = θᵢ)
        set_tke!(model, FT(0.5))

        Ri = Riᶠ(model)
        @test all(isfinite, Ri[interior_faces])
        @test all(abs.(Ri[interior_faces]) .> 1e4 * max(sf.Ri⁻, sf.Ri⁺))
        @test any(Ri[interior_faces] .< 0) && any(Ri[interior_faces] .> 0)

        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Lᵉ = column(model.closure_fields.Lᵉ)
        @test all(isfinite, Kᵘ)
        @test all(isfinite, Kᶜ)
        @test all(isfinite, Lᵉ)
        @test all(Lᵉ .< 0)
        # Where the shear vanishes the ratio follows the rational functions at Ri = N² / S²ᵐⁱⁿ,
        # within the relative distance 1 / (1 + |Ri| / Ri±) ≤ 1e-4 of the asymptotes
        stable_faces = [k for k in interior_faces if Ri[k] > 0]
        unstable_faces = [k for k in interior_faces if Ri[k] < 0]
        expected = [tracer_stability_function(sf, r) / momentum_stability_function(sf, r) for r in Ri]
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ expected[interior_faces])
        @test all(isapprox.(Kᶜ[stable_faces] ./ Kᵘ[stable_faces], sf.ℂᶜ⁺ / sf.ℂᵘ⁺; rtol = 1e-3))
        @test all(isapprox.(Kᶜ[unstable_faces] ./ Kᵘ[unstable_faces], sf.ℂᶜ⁻ / sf.ℂᵘ⁻; rtol = 1e-3))

        for _ in 1:10
            time_step!(model, 10)
        end
        ρe = column(model.tracers.ρe)
        @test all(isfinite, ρe)
        @test all(ρe .≥ 0)
    end

    @testset "with the moist static stability" begin
        closure = TKEBasedTurbulenceClosure(stability_functions = RationalStabilityFunction(),
                                            static_stability = MoistStaticStability())
        microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; closure, microphysics, advection = nothing)
        set!(model; θ = 288, qᵗ = z -> ifelse(z < Lz / 2, 4e-3, 13e-3), u = z -> 0.005 * z)
        set_tke!(model, FT(0.1))
        for _ in 1:5
            time_step!(model, 10)
        end
        @test all(isfinite, column(model.tracers.ρe))
        @test all(isfinite, column(model.closure_fields.Kᶜ))
    end
end
