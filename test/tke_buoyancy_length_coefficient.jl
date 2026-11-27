include(joinpath(@__DIR__, "setup.jl"))

# The coefficient ℂˡᵇ of the buoyancy length ℂˡᵇ √e / N of `GradientLimitedMixingLength`: ℂˡᵇ = 1 is the
# original envelope, to the bit, and the length of stable air scales with ℂˡᵇ.

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME, mixing_lengthᶜᶜᶜ, convert_eltype, turbulent_velocityᶜᶜᶜ, face_valueᶜᶜᶠ
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: trainable_parameter_names
using Oceananigans.TurbulenceClosures: height_above_bottomᶜᶜᶜ
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: ℑbzᵃᵃᶜ
using Oceananigans
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using Test

# A convective layer under an inversion with a stable layer aloft
column_θ(z) = 300 + 8e-3 * max(0, z - 1000) + 2 * (1 + tanh((z - 1000) / 80))

function length_column(mixing_length; FT = Float64, Nz = 40)
    grid = RectilinearGrid(CPU(), FT; size = Nz, z = (0, 2000), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, base_pressure = 101500, potential_temperature = 300)
    ρ₀ = surface_density(reference_state)
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * 5e-2)))
    closure = TKEBasedTurbulenceClosure(FT; mixing_length)
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions, advection = nothing)
    set!(model; θ = column_θ, qᵗ = 4e-3, u = z -> 4 + 2e-3 * z, ρe = z -> 0.3 * exp(-z / 800) + 1e-3)
    set!(model.tracers.ρe, reference_state.density * model.tracers.ρe)
    update_state!(model)
    return model
end

column(field) = Array(interior(field, 1, 1, :))

function specific_tke(model)
    e = compute!(Field(model.tracers[TKE_NAME] / model.dynamics.reference_state.density))
    Oceananigans.BoundaryConditions.fill_halo_regions!(e)
    return e
end

center_lengths(model, e = specific_tke(model)) =
    [mixing_lengthᶜᶜᶜ(1, 1, k, model.grid, model.closure, e.data, model.closure_fields) for k in 1:model.grid.Nz]

# The envelope of the original gradient-limited length, written out from the stored N² and e: the penetration depth
# √e / N, swept up and down at the slope ℂⁱ and capped by the wall length ℂˢ z (the hard minimum)
function original_envelope(model, e = specific_tke(model))
    grid, closure = model.grid, model.closure
    ml = closure.mixing_length
    N² = model.closure_fields.N²
    Nz = grid.Nz
    FT = eltype(grid)
    ℓˢ = fill(FT(Inf), Nz + 1)
    for k in 2:Nz+1
        N²₊ = max(N²[1, 1, k], zero(FT))
        ℓᵇ = ℑzᵃᵃᶠ(1, 1, k, grid, turbulent_velocityᶜᶜᶜ, closure, e.data) / sqrt(N²₊)
        ℓᵇ = ifelse(N²₊ == 0, FT(Inf), ℓᵇ)
        ℓˢ[k] = ifelse(isnan(ℓᵇ), FT(Inf), ℓᵇ)
    end
    for k in 2:Nz+1
        ℓˢ[k] = min(ℓˢ[k], ℓˢ[k-1] + ml.ℂⁱ * Δzᶜᶜᶜ(1, 1, k-1, grid))
    end
    for k in Nz:-1:1
        ℓˢ[k] = min(ℓˢ[k], ℓˢ[k+1] + ml.ℂⁱ * Δzᶜᶜᶜ(1, 1, k, grid))
    end
    zᶠ = znodes(grid, Face())
    ℓ = [k == 1 ? zero(FT) : min(ℓˢ[k], ml.ℂˢ * zᶠ[k]) for k in 1:Nz+1]
    return ℓ, ℓˢ
end

@testset "GradientLimitedMixingLength: the coefficient ℂˡᵇ [$(FT)]" for FT in test_float_types()
    ml = GradientLimitedMixingLength(; ℂˢ = FT(0.79), ℂⁱ = FT(1.78))
    @test ml.ℂˡᵇ === one(FT)
    @test GradientLimitedMixingLength(0.79, 1.78) === GradientLimitedMixingLength(0.79, 1.78, 1.0, HardMinimum())
    @test GradientLimitedMixingLength(0.79, 1.78, HarmonicBlend(4.0)).ℂˡᵇ === 1.0
    @test GradientLimitedMixingLength(; ℂˡᵇ = 20).ℂˡᵇ === 20.0
    @test convert_eltype(Float32, GradientLimitedMixingLength(; ℂˡᵇ = 20)).ℂˡᵇ === 20f0
    @test isbits(GradientLimitedMixingLength(; ℂˡᵇ = 20))
    @test occursin("ℂˡᵇ = 20", sprint(show, GradientLimitedMixingLength(; ℂˡᵇ = 20)))

    # A calibratable coefficient of the closure
    closure = TKEBasedTurbulenceClosure(FT)
    @test keys(closure_parameters(closure))[1:3] == (:ℂˢ, :ℂⁱ, :ℂˡᵇ)
    @test closure_parameters(closure).ℂˡᵇ === one(FT)
    @test :ℂˡᵇ ∈ trainable_parameter_names(closure)
    @test with_parameters(closure; ℂˡᵇ = FT(30)).mixing_length.ℂˡᵇ == 30

    # and a coefficient of each column of an ensemble
    grid = RectilinearGrid(CPU(), FT; size = Oceananigans.Grids.ColumnEnsembleSize(Nz = 4, ensemble = (2, 1)),
                           z = (0, 1000), topology = (Flat, Flat, Bounded))
    closures = reshape([with_parameters(closure; ℂˡᵇ) for ℂˡᵇ in FT.((1, 20))], 2, 1)
    ensemble = per_column_closure(grid, closures)
    @test Array(ensemble.mixing_length.ℂˡᵇ.values) == reshape(FT[1, 20], 2, 1)
end

@testset "ℂˡᵇ = 1 is the original gradient-limited length [$(FT)]" for FT in test_float_types()
    model = length_column(GradientLimitedMixingLength(; ℂˢ = FT(0.79), ℂⁱ = FT(1.78)); FT)
    ℓ, ℓˢ = original_envelope(model)
    # To round-off: the closure forms the buoyancy length through its inverse, N / √e
    roundoff = 8eps(FT)
    @test all(isapprox.(column(model.closure_fields.ℓᵐ), ℓ; rtol = roundoff))
    @test all(isapprox.(column(model.closure_fields.ℓˢ), ℓˢ; rtol = roundoff))

    # The centers: the smaller face envelope carried half a cell, bounded by the center's √e / N (N² from the two
    # faces, as the closure reconstructs it) and ℂˢ z
    e = specific_tke(model)
    grid, closure = model.grid, model.closure
    N² = model.closure_fields.N²
    for k in 1:grid.Nz
        N²₊ = max(ℑbzᵃᵃᶜ(1, 1, k, grid, face_valueᶜᶜᶠ, N²), zero(FT))
        ℓᵇ = turbulent_velocityᶜᶜᶜ(1, 1, k, grid, closure, e.data) / sqrt(N²₊)
        ℓᵇ = ifelse(N²₊ == 0, FT(Inf), ℓᵇ)
        expected = min(min(ℓˢ[k], ℓˢ[k+1]) + closure.mixing_length.ℂⁱ * Δzᶜᶜᶜ(1, 1, k, grid) / 2,
                       min(closure.mixing_length.ℂˢ * height_above_bottomᶜᶜᶜ(1, 1, k, grid), ℓᵇ))
        @test center_lengths(model, e)[k] ≈ expected rtol = roundoff
    end
end

@testset "The length of stable air scales with ℂˡᵇ [$(FT)]" for FT in test_float_types()
    # The buoyancy length alone: no wall, no envelope slope. In stable air ℓ = ℂˡᵇ √e / N at faces and centers.
    buoyancy_length(ℂˡᵇ) = GradientLimitedMixingLength(; ℂˢ = FT(Inf), ℂⁱ = FT(Inf), ℂˡᵇ = FT(ℂˡᵇ))
    one_model = length_column(buoyancy_length(1); FT)
    model = length_column(buoyancy_length(20); FT)
    N² = column(one_model.closure_fields.N²)
    stable = filter(k -> 1 < k ≤ one_model.grid.Nz && N²[k] > 1e-5, eachindex(N²))
    @test !isempty(stable)
    @test column(model.closure_fields.ℓᵐ)[stable] ≈ 20 .* column(one_model.closure_fields.ℓᵐ)[stable] rtol=10eps(FT)
    stable_centers = filter(k -> N²[k] > 1e-5 && N²[k+1] > 1e-5, 1:one_model.grid.Nz)
    @test center_lengths(model)[stable_centers] ≈ 20 .* center_lengths(one_model)[stable_centers] rtol=10eps(FT)

    # With the wall and the envelope slope: the length of the stable layer aloft grows with ℂˡᵇ, while the
    # neutral wall layer, ℂˢ z, does not change
    ℂˢ, ℂⁱ = FT(0.79), FT(1.78)
    reference = length_column(GradientLimitedMixingLength(; ℂˢ, ℂⁱ); FT)
    long = length_column(GradientLimitedMixingLength(; ℂˢ, ℂⁱ, ℂˡᵇ = FT(20)); FT)
    ℓ₁, ℓ₂₀ = column(reference.closure_fields.ℓᵐ), column(long.closure_fields.ℓᵐ)
    @test all(ℓ₂₀ .≥ ℓ₁)
    @test ℓ₂₀[end] > 2ℓ₁[end]
    @test ℓ₂₀[2] == ℓ₁[2]
end
