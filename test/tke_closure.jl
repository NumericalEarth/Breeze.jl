include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME, TKEClosureFields, absorb_stratification_coefficient, dissipationᶜᶜᶜ
using Oceananigans
using Oceananigans.TimeSteppers: update_state!, time_discretization
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, ExplicitTimeDiscretization,
                                       buoyancy_tracers, buoyancy_force
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Units
using Test

# The mixing length is stored with the closure fields: its envelope over the column is computed by
# two sweeps, once per stage, so a script reads it as `model.closure_fields.ℓ`.
diagnosed_mixing_length(model) = model.closure_fields.ℓ

# The envelope written out in full, as the test's reference: at every face the minimum over every
# face z′ of the bound there — the wall length or the buoyancy penetration depth, whichever is
# smaller — plus Cˢ times the distance. The ground, the bottom face, is a zero; above it the wall
# length is Cˢ times the height.
function brute_force_envelope(grid, ℓᵇ, Cˢ)
    zf = znodes(grid, Face())
    obstacles = min.(Cˢ .* zf, ℓᵇ)
    obstacles[1] = 0
    return [minimum(obstacles[m] + Cˢ * abs(zf[k] - zf[m]) for m in eachindex(zf)) for k in eachindex(zf)]
end

#####
##### Construction
#####

@testset "TKEBasedTurbulenceClosure construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    closure = TKEBasedTurbulenceClosure()
    @test closure isa TKEBasedTurbulenceClosure
    @test time_discretization(closure) isa VerticallyImplicitTimeDiscretization
    @test closure.mixing_length isa GradientLimitedMixingLength{FT}
    @test closure.stability_functions isa ConstantStabilityFunctions{FT}
    @test closure.static_stability isa MoistStaticStability

    @testset "defaults and the constants they imply" begin
        sf = closure.stability_functions
        Cˢ = closure.mixing_length.Cˢ
        @test Cˢ ≈ 1.316
        @test (sf.Cᵘ, sf.Cᶜ, sf.Cᵉ, sf.Cᴰ) == FT.((0.149, 0.201, 0.298, 0.388))
        @test closure.minimum_tke == FT(1e-6)
        @test closure.negative_tke_damping_time_scale == FT(60)
        @test isinf(closure.maximum_viscosity)
        @test isinf(closure.maximum_tracer_diffusivity)
        @test isinf(closure.maximum_tke_diffusivity)

        # The neutral log layer: von Kármán constant, surface TKE, Prandtl number; and the
        # stratified steady state: the critical Richardson number
        @test Cˢ * (sf.Cᵘ^3 / sf.Cᴰ)^(1/4) ≈ 0.40 atol=0.005
        @test 1 / sqrt(sf.Cᵘ * sf.Cᴰ) ≈ 4.2 atol=0.05
        @test sf.Cᵘ / sf.Cᶜ ≈ 0.74 atol=0.005
        @test sf.Cᵘ / (sf.Cᶜ + sf.Cᴰ) ≈ 0.25 atol=0.005
    end

    @testset "keyword arguments, promotion and float type" begin
        closure = TKEBasedTurbulenceClosure(; mixing_length = GradientLimitedMixingLength(Cˢ = 1),
                                              stability_functions = ConstantStabilityFunctions(Cᵘ = 0.3, Cᶜ = 0.3, Cᵉ = 1, Cᴰ = 1),
                                              maximum_viscosity = 100,
                                              minimum_tke = 1e-8,
                                              negative_tke_damping_time_scale = 10minutes)
        @test closure.mixing_length.Cˢ === FT(1)
        @test closure.stability_functions.Cᵘ === FT(0.3)
        @test closure.stability_functions.Cᵉ === FT(1)
        @test closure.maximum_viscosity === FT(100)
        @test closure.minimum_tke === FT(1e-8)
        @test closure.negative_tke_damping_time_scale === FT(600)

        explicit = TKEBasedTurbulenceClosure(ExplicitTimeDiscretization())
        @test time_discretization(explicit) isa ExplicitTimeDiscretization

        closure32 = TKEBasedTurbulenceClosure(Float32)
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
        @test occursin("Cˢ", str)
        @test occursin("ConstantStabilityFunctions", str)
        @test occursin("MoistStaticStability", str)
        @test occursin("DryStaticStability", sprint(show, TKEBasedTurbulenceClosure(static_stability = DryStaticStability())))
        @test occursin("minimum_tke", str)
        @test occursin("Cᴰ", sprint(show, ConstantStabilityFunctions()))
        @test occursin("Cˢ", sprint(show, GradientLimitedMixingLength()))
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

@testset "TKEBasedTurbulenceClosure in an AtmosphereModel [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = 1000
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    zf = znodes(grid, Face())
    closure = TKEBasedTurbulenceClosure()

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
        Cˢ = closure.mixing_length.Cˢ
        interior_faces = 2:Nz

        # In neutral air the mixing length is Cˢ times the height above the surface: the ground is
        # the only obstacle, the surface face is a zero — the diffusivities there are masked
        # regardless — and the top face is unmasked.
        ℓ = column(diagnosed_mixing_length(model))
        Δz = Lz / Nz
        @test ℓ[1] == 0
        @test ℓ[Nz+1] ≈ Cˢ * Lz
        @test all(ℓ[interior_faces] .≈ Cˢ .* zf[interior_faces])

        # Kᵘ = Cᵘ ℓ √e, masked on the boundary faces
        @test Kᵘ[1] == 0
        @test Kᵘ[Nz+1] == 0
        @test all(Kᵘ[interior_faces] .≈ sf.Cᵘ .* ℓ[interior_faces] .* sqrt(e₀))

        # The ratios Kᶜ/Kᵘ, Kᵉ/Kᵘ are the stability-function ratios
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᶜ / sf.Cᵘ)
        @test all(Kᵉ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᵉ / sf.Cᵘ)
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
        Cˢ = closure.mixing_length.Cˢ

        for k in 2:Nz
            N² = g * (log(θ[k]) - log(θ[k-1])) / Δz
            ℓᵇ = sqrt(e₀) / sqrt(N²)
            @test ℓ[k] ≈ min(Cˢ * zf[k], ℓᵇ) rtol=1e-5
        end

        # Stratification limits the length well above the surface
        @test ℓ[Nz] < Cˢ * zf[Nz] / 2

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
        # must reproduce that closure exactly: Kᵘ = Cᵘ ℓ √e and ω = Cᴰ √e / ℓ with
        # ℓ = min(z, Cᴺ √e / N), here written out at the faces and centers.
        Cᴺ = FT(0.76)
        legacy = ConstantStabilityFunctions(Cᵘ = 0.196, Cᶜ = 0.265, Cᵉ = 0.392, Cᴰ = 0.295)
        mixing_length, stability_functions = absorb_stratification_coefficient(Cᴺ, legacy)
        mapped = TKEBasedTurbulenceClosure(; mixing_length, stability_functions)
        @test mapped.mixing_length.Cˢ ≈ 1 / Cᴺ
        @test mapped.stability_functions.Cᵘ * mapped.mixing_length.Cˢ ≈ legacy.Cᵘ
        @test mapped.stability_functions.Cᴰ / mapped.mixing_length.Cˢ ≈ legacy.Cᴰ

        model = AtmosphereModel(grid; closure = mapped, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        set!(model; θ = z -> 300 + Γ * z)
        set_tke!(model, e₀)

        θ = column(model.formulation.potential_temperature)
        g = model.thermodynamic_constants.gravitational_acceleration
        Δz = Lz / Nz
        N² = [g * (log(θ[k]) - log(θ[k-1])) / Δz for k in 2:Nz]
        ℓᶠ = [min(zf[k], Cᴺ * sqrt(e₀) / sqrt(N²[k-1])) for k in 2:Nz]

        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Kᵉ = column(model.closure_fields.Kᵉ)
        @test all(isapprox.(Kᵘ[2:Nz], legacy.Cᵘ .* ℓᶠ .* sqrt(e₀); rtol = 1e-5))
        @test all(isapprox.(Kᶜ[2:Nz], legacy.Cᶜ .* ℓᶠ .* sqrt(e₀); rtol = 1e-5))
        @test all(isapprox.(Kᵉ[2:Nz], legacy.Cᵉ .* ℓᶠ .* sqrt(e₀); rtol = 1e-5))

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
            @test Lᵉ[k] ≈ -(legacy.Cᴰ * sqrt(e₀) / ℓᶜ - B / e₀) rtol=1e-5
        end
    end


    @testset "the envelope: an elevated neutral layer is bounded by the stratified air around it" begin
        # Stable below 300 m, neutral between 300 and 700 m, stable above: min(Cˢ z, ℓᵇ) gives the
        # wall length Cˢ z ≈ 400–900 m inside the neutral layer, where nothing is stratified; the
        # envelope is bounded by the stratified faces just below and above the layer.
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        Γ = FT(0.005)
        θᵢ(z) = z < 300 ? 300 + Γ * z : z < 700 ? 300 + Γ * 300 : 300 + Γ * (z - 400)
        set!(model; θ = θᵢ)
        set_tke!(model, e₀)
        Cˢ = closure.mixing_length.Cˢ
        Δz = Lz / Nz

        ℓ = column(diagnosed_mixing_length(model))
        N² = column(model.closure_fields.N²)
        ℓᵇ = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : Inf for k in eachindex(zf)]

        # Exactly the envelope, at every face
        @test all(isapprox.(ℓ, brute_force_envelope(grid, ℓᵇ, Cˢ); rtol = 1e-6))

        # Bounded by every obstacle, and by its neighbors through the slope Cˢ
        @test all(ℓ[2:end] .≤ Cˢ .* zf[2:end] .+ 1e-6)
        @test all(ℓ .≤ ℓᵇ .+ 1e-6)
        @test all(abs.(diff(ℓ)) .≤ Cˢ * Δz * (1 + 1e-6))

        # In the neutral layer the mixing length is well below the wall length: the stratified
        # faces above and below bind it through Cˢ times the distance to them
        neutral = findall(z -> 350 < z < 650, zf)
        @test all(ℓ[neutral] .< 0.75 .* Cˢ .* zf[neutral])
        @test all(abs.(N²[neutral]) .< 1e-10)

        # In the stratified layer below, the point itself binds: the familiar min(Cˢ z, ℓᵇ)
        stratified = findall(z -> 100 < z < 250, zf)
        @test all(isapprox.(ℓ[stratified], min.(Cˢ .* zf[stratified], ℓᵇ[stratified]); rtol = 1e-5))

        # The dissipation reads the envelope at the centers, halfway between the faces, bounded by
        # the center's own local minimum (its wall length and penetration depth)
        e = model.tracers.ρe / model.dynamics.reference_state.density
        ε = column(Field(KernelFunctionOperation{Center, Center, Center}(dissipationᶜᶜᶜ, grid, closure, e,
                                                                        model.velocities, model.closure_fields)))
        zc = znodes(grid, Center())
        ℓᵇᶜ = [(N²[k] + N²[k+1]) / 2 > 0 ? sqrt(e₀) / sqrt((N²[k] + N²[k+1]) / 2) : Inf for k in 1:Nz]
        ℓᶜ = [min(min(ℓ[k], ℓ[k+1]) + Cˢ * Δz / 2, Cˢ * zc[k], ℓᵇᶜ[k]) for k in 1:Nz]
        @test all(isapprox.(ε[2:Nz-1], closure.stability_functions.Cᴰ .* e₀^(3/2) ./ ℓᶜ[2:Nz-1]; rtol = 1e-6))
    end

    @testset "the three formulations on the elevated neutral layer" begin
        e₀ = FT(0.5)
        Γ = FT(0.005)
        θᵢ(z) = z < 300 ? 300 + Γ * z : z < 700 ? 300 + Γ * 300 : 300 + Γ * (z - 400)
        Cˢ = FT(1.316)
        Δz = Lz / Nz
        lengths = Dict()
        N² = nothing
        for (name, mixing_length) in (:local => LocalMinimumMixingLength(Cˢ = Cˢ), :integral => IntegralMixingLength(Cˢ = Cˢ), :envelope => GradientLimitedMixingLength(Cˢ = Cˢ))
            model = AtmosphereModel(grid; closure = TKEBasedTurbulenceClosure(; mixing_length), advection = nothing)
            set!(model; θ = θᵢ)
            set_tke!(model, e₀)
            lengths[name] = column(diagnosed_mixing_length(model))
            N² = column(model.closure_fields.N²)
        end
        ℓᵇ = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : Inf for k in eachindex(zf)]
        wall = Cˢ .* zf
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
            geometric = Cˢ * min(zf[k] - zf[below], zf[above] - zf[k])
            @test geometric ≤ ℓ[k] ≤ geometric + Cˢ * max(ℓᵇ[below], ℓᵇ[above]) + Cˢ * Δz
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
        Cˢ = closure.mixing_length.Cˢ
        zf_stretched = znodes(stretched, Face())

        ℓ = column(diagnosed_mixing_length(model))
        N² = column(model.closure_fields.N²)
        ℓᵇ = [N²[k] > 0 ? sqrt(e₀) / sqrt(N²[k]) : Inf for k in eachindex(zf_stretched)]
        @test all(isapprox.(ℓ, brute_force_envelope(stretched, ℓᵇ, Cˢ); rtol = 1e-6))
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
        @test all(Kᶜ[2:Nz] .≈ Kᵘ[2:Nz] .* (sf.Cᶜ / sf.Cᵘ))
    end

    @testset "the implicit linear coefficient is the dissipation rate" begin
        model = AtmosphereModel(grid; closure, advection = nothing)
        e₀ = FT(0.5)
        set!(model; θ = 300)
        set_tke!(model, e₀)

        @test model.closure_fields.tupled_implicit_linear_coefficients.ρe === model.closure_fields.Lᵉ
        @test model.closure_fields.tupled_implicit_linear_coefficients.ρθ isa Oceananigans.Fields.ZeroField

        # Neutral air, no buoyancy flux: Lᵉ = - Sᴰ √e / ℓ with ℓ = Cˢ z at the cell centers
        Lᵉ = column(model.closure_fields.Lᵉ)
        zc = znodes(grid, Center())
        Cᴰ = closure.stability_functions.Cᴰ
        Cˢ = closure.mixing_length.Cˢ
        @test all(Lᵉ .≈ -Cᴰ * sqrt(e₀) ./ (Cˢ .* zc))

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
        @test all(Lᵉ .≈ -Cᴰ * sqrt(e₋) ./ (Cˢ .* zc))
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
    end

    @testset "dissipation decays TKE at the analytic rate" begin
        # No TKE diffusion, no wind, no stratification: each cell decays as ∂ₜe = -Cᴰ e^{3/2}/ℓ
        # with ℓ = Cˢ z, i.e. e(t) = (e₀^{-1/2} + Cᴰ t / 2ℓ)^{-2}
        undiffused = TKEBasedTurbulenceClosure(maximum_tke_diffusivity = 0)
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
        Cᴰ = undiffused.stability_functions.Cᴰ
        Cˢ = undiffused.mixing_length.Cˢ
        t = Nt * Δt
        e_analytic = @. (e₀^(-1/2) + Cᴰ * t / (2 * Cˢ * zc))^(-2)
        # The sinks are implicit per stage, so the decay is first-order accurate in Δt ω ≈ 0.02
        @test all(e .≥ 0)
        @test all(isapprox.(e, e_analytic; rtol = 0.05))
    end

    @testset "negative TKE is damped on the damping time scale" begin
        τ = FT(60)
        model = AtmosphereModel(grid; closure, advection = nothing)
        set!(model; θ = 300)
        set_tke!(model, FT(-0.1))

        # No shear, no stratification and no TKE gradient, so the damping rate 1/τ is the only
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

using Breeze.TurbulenceClosures: stability_ramp, Riᶜᶜᶠ

@testset "RiDependentStabilityFunctions [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = 1000
    grid = RectilinearGrid(default_arch; size = Nz, z = (0, Lz), topology = (Flat, Flat, Bounded))
    zf = znodes(grid, Face())
    zc = znodes(grid, Center())
    interior_faces = 2:Nz

    @testset "CATKE's values and the constants they imply" begin
        sf = RiDependentStabilityFunctions()
        @test isbits(sf)
        @test sf isa RiDependentStabilityFunctions{FT}
        @test (sf.Cᵘ⁻, sf.Cᵘ⁰, sf.Cᵘ⁺) == FT.((0.370, 0.361, 0.242))
        @test (sf.Cᶜ⁻, sf.Cᶜ⁰, sf.Cᶜ⁺) == FT.((0.572, 0.369, 0.098))
        @test (sf.Cᵉ⁻, sf.Cᵉ⁰, sf.Cᵉ⁺) == FT.((1.447, 7.863, 0.548))
        @test (sf.Cᴰ⁻, sf.Cᴰ⁰, sf.Cᴰ⁺) == FT.((0.923, 1.604, 0.579))
        @test (sf.Ri⁰, sf.Riᵟ) == FT.((0.254, 1.02))

        parameters = catke_parameters()
        @test parameters.mixing_length.Cˢ == 1.131
        @test parameters.stability_functions isa RiDependentStabilityFunctions
        closure = TKEBasedTurbulenceClosure(; parameters...)
        @test closure.mixing_length.Cˢ === FT(1.131)
        @test closure.stability_functions isa RiDependentStabilityFunctions{FT}
        @test isbits(closure)

        # What CATKE's neutral values mean in the atmospheric surface layer: a von Kármán constant
        # 17% above 0.40, a surface TKE a factor three below e/u★² ≈ 4, a Prandtl number near one,
        # a TKE diffusivity 22 times the viscosity, and a critical Richardson number of 0.18
        Cˢ = closure.mixing_length.Cˢ
        @test Cˢ * (sf.Cᵘ⁰^3 / sf.Cᴰ⁰)^(1/4) ≈ 0.468 atol=0.002
        @test 1 / sqrt(sf.Cᵘ⁰ * sf.Cᴰ⁰) ≈ 1.31 atol=0.01
        @test sf.Cᵘ⁰ / sf.Cᶜ⁰ ≈ 0.98 atol=0.01
        @test sf.Cᵉ⁰ / sf.Cᵘ⁰ ≈ 21.8 atol=0.1
        @test sf.Cᵘ⁰ / (sf.Cᶜ⁰ + sf.Cᴰ⁰) ≈ 0.183 atol=0.002
        # and in stable stratification the Prandtl number rises to 2.5
        @test sf.Cᵘ⁺ / sf.Cᶜ⁺ ≈ 2.47 atol=0.01

        # Keyword promotion and float type
        mixed = RiDependentStabilityFunctions(Cᵘ⁻ = 1, Riᵟ = 2)
        @test mixed.Cᵘ⁻ === FT(1)
        @test mixed.Riᵟ === FT(2)
        @test mixed.Cᵘ⁰ === FT(0.361)
        closure32 = TKEBasedTurbulenceClosure(Float32; stability_functions = RiDependentStabilityFunctions())
        @test closure32.stability_functions isa RiDependentStabilityFunctions{Float32}

        str = sprint(show, closure)
        @test occursin("RiDependentStabilityFunctions", str)
        @test occursin("Ri⁰", str)
        @test occursin("Ri → ∞", sprint(show, sf))
    end

    @testset "the piecewise-linear ramp and its limits" begin
        C⁻, C⁰, C⁺, Ri⁰, Riᵟ = FT.((1, 2, 3, 0.25, 1))
        S(Ri) = stability_ramp(FT(Ri), C⁻, C⁰, C⁺, Ri⁰, Riᵟ)
        @test S(-1) == C⁻
        @test S(-Inf) == C⁻
        @test S(0) == C⁰
        @test S(0.25) == C⁰
        @test S(0.75) ≈ (C⁰ + C⁺) / 2
        @test S(1.25) == C⁺
        @test S(10) == C⁺
        @test S(1000) == C⁺
        @test S(Inf) == C⁺
        @test S(0) isa FT
    end

    @testset "in a model: neutral, stable and unstable columns" begin
        closure = TKEBasedTurbulenceClosure(; catke_parameters()...)
        sf = closure.stability_functions
        Cˢ = closure.mixing_length.Cˢ
        e₀ = FT(0.5)

        # Neutral shear: Ri = 0, the neutral endpoints, and dissipation Cᴰ⁰ √e / (Cˢ z)
        model = AtmosphereModel(grid; closure, advection = nothing)
        set!(model; θ = 300, u = z -> 0.01 * z)
        set_tke!(model, e₀)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Kᵉ = column(model.closure_fields.Kᵉ)
        Lᵉ = column(model.closure_fields.Lᵉ)
        @test all(Kᵘ[interior_faces] .≈ sf.Cᵘ⁰ .* Cˢ .* zf[interior_faces] .* sqrt(e₀))
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᶜ⁰ / sf.Cᵘ⁰)
        @test all(Kᵉ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᵉ⁰ / sf.Cᵘ⁰)
        @test all(Lᵉ .≈ -sf.Cᴰ⁰ * sqrt(e₀) ./ (Cˢ .* zc))

        # Strong stratification and weak shear: Ri far beyond the ramp, the stable asymptotes
        set!(model; θ = z -> 300 + 0.03 * z, u = z -> 1e-3 * z)
        set_tke!(model, e₀)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Ri = column(Field(KernelFunctionOperation{Center, Center, Face}(Riᶜᶜᶠ, grid, model.velocities, model.closure_fields.N²)))
        @test all(Ri[interior_faces] .> sf.Ri⁰ + sf.Riᵟ)
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᶜ⁺ / sf.Cᵘ⁺)

        # Unstable stratification with shear: Ri < 0, the unstable endpoints
        set!(model; θ = z -> 300 - 0.01 * z, u = z -> 0.01 * z)
        set_tke!(model, e₀)
        Kᵘ = column(model.closure_fields.Kᵘ)
        Kᶜ = column(model.closure_fields.Kᶜ)
        Ri = column(Field(KernelFunctionOperation{Center, Center, Face}(Riᶜᶜᶠ, grid, model.velocities, model.closure_fields.N²)))
        @test all(Ri[interior_faces] .< 0)
        @test all(Kᶜ[interior_faces] ./ Kᵘ[interior_faces] .≈ sf.Cᶜ⁻ / sf.Cᵘ⁻)
    end

    @testset "a windless column with stable and unstable layers stays finite" begin
        # No shear, so Ri = ±∞ at every interface and the sign changes at mid-depth. At the cell
        # center there Ri is formed from the reconstructed N² and S², a number, where averaging
        # the interfaces' ±∞ would have given NaN and switched the dissipation off.
        closure = TKEBasedTurbulenceClosure(; catke_parameters()...)
        sf = closure.stability_functions
        model = AtmosphereModel(grid; closure, advection = nothing)
        θᵢ(z) = 300 - 0.005 * min(z, Lz / 2) + 0.01 * max(0, z - Lz / 2)
        set!(model; θ = θᵢ)
        set_tke!(model, FT(0.5))

        Ri = column(Field(KernelFunctionOperation{Center, Center, Face}(Riᶜᶜᶠ, grid, model.velocities, model.closure_fields.N²)))
        @test all(isinf, Ri[interior_faces])
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
        closure = TKEBasedTurbulenceClosure(; catke_parameters()..., static_stability = MoistStaticStability())
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
