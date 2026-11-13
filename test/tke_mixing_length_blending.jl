include(joinpath(@__DIR__, "setup.jl"))

# The blending of the lengths that bound the mixing length of `TKEBasedTurbulenceClosure`:
# `HardMinimum` (the default) and `HarmonicBlend`.

using Breeze
using Breeze.TurbulenceClosures: TKE_NAME, blend, length_blending, fill_mixing_length!, mixing_lengthᶜᶜᶜ, convert_eltype
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Adapt: adapt
using Enzyme: Enzyme
using Test

# A convective layer under an inversion with a stable layer aloft: wall-limited near the ground,
# unbounded by the local stratification in the mixed layer, stratification-limited above
blending_θ(z) = 300 + 8e-3 * max(0, z - 1000) + 2 * (1 + tanh((z - 1000) / 80))

function blending_column(mixing_length; arch = default_arch, Nz = 40, stability_functions = RationalStabilityFunction())
    grid = RectilinearGrid(arch; size = Nz, z = (0, 2000), topology = (Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, base_pressure = 101500, potential_temperature = 300)
    ρ₀ = surface_density(reference_state)
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * 5e-2)))
    closure = TKEBasedTurbulenceClosure(; mixing_length, stability_functions)
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions, advection = nothing)
    set!(model; θ = blending_θ, qᵗ = 4e-3, u = z -> 4 + 2e-3 * z, ρe = z -> 0.3 * exp(-z / 800) + 1e-3)
    set!(model.tracers.ρe, reference_state.density * model.tracers.ρe)
    update_state!(model)
    return model
end

function run_steps!(model; Δt = 30, steps = 40)
    for _ in 1:steps
        time_step!(model, Δt)
    end
    return model
end

column(field) = Array(interior(field, 1, 1, :))

function state(model)
    prognostic = NamedTuple(name => column(field) for (name, field) in pairs(Oceananigans.prognostic_fields(model)))
    closure = NamedTuple(name => column(getproperty(model.closure_fields, name)) for name in (:Kᵘ, :Kᶜ, :Kᵉ, :N², :ℓᵐ, :ℓˢ, :Lᵉ))
    return merge(prognostic, closure)
end

@testset "Blending of two lengths [$(FT)]" for FT in all_float_types()
    hard = HardMinimum()
    a, b = FT(30), FT(120)
    @test blend(hard, a, b) === a
    @test isbits(hard) && isbits(HarmonicBlend(FT(4)))

    for p in FT.((1, 2, 4, 16))
        harmonic = HarmonicBlend(p)
        ℓ = blend(harmonic, a, b)
        @test ℓ isa FT
        @test ℓ ≈ (a^-p + b^-p)^(-1 / p)
        @test blend(harmonic, b, a) === ℓ
        @test ℓ ≤ a
        p ≤ 4 && @test ℓ < a
        @test blend(harmonic, a, a) ≈ a * FT(2)^(-1 / p)
        # Scale-free
        @test blend(harmonic, 10a, 10b) ≈ 10ℓ
        # An infinite length is no bound; a zero length is the bound
        @test blend(harmonic, a, FT(Inf)) === a
        @test blend(harmonic, FT(Inf), a) === a
        @test blend(harmonic, FT(Inf), FT(Inf)) === FT(Inf)
        @test blend(harmonic, zero(FT), a) === zero(FT)
        @test blend(harmonic, zero(FT), zero(FT)) === zero(FT)
        @test blend(harmonic, zero(FT), FT(Inf)) === zero(FT)
        # Lengths far apart, and very small or large ones, neither overflow nor underflow
        @test blend(harmonic, FT(1e-20), FT(1e20)) == FT(1e-20)
        @test isfinite(blend(harmonic, floatmax(FT), floatmax(FT)))
        @test blend(harmonic, floatmin(FT), floatmin(FT)) > 0
    end

    # The hard minimum as p → ∞, monotonically
    errors = [abs(blend(HarmonicBlend(FT(p)), a, FT(1.1) * a) - a) for p in (2, 8, 32, 128, 512)]
    @test issorted(errors; rev = true)
    @test errors[end] < 1e-4 * a
    @test blend(HarmonicBlend(FT(1e4)), a, a) ≈ a rtol=1e-3
end

@testset "Mixing-length blending: construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @test GradientLimitedMixingLength().blending isa HardMinimum
    @test LocalMinimumMixingLength().blending isa HardMinimum
    @test TKEBasedTurbulenceClosure().mixing_length.blending isa HardMinimum
    @test length_blending(IntegralMixingLength()) isa HardMinimum
    @test GradientLimitedMixingLength(2, 0.5) === GradientLimitedMixingLength(2.0, 0.5, HardMinimum())
    @test GradientLimitedMixingLength{Float32}(2, 0.5) === GradientLimitedMixingLength(2f0, 0.5f0, HardMinimum())
    @test LocalMinimumMixingLength{Float32}(2) === LocalMinimumMixingLength(2f0, HardMinimum())
    @test LocalMinimumMixingLength(2.0) === LocalMinimumMixingLength(ℂˢ = 2.0)

    blending = HarmonicBlend(exponent = 6)
    @test_throws ArgumentError HarmonicBlend(exponent = 0.5)
    for mixing_length in (GradientLimitedMixingLength(ℂˢ = 0.8, ℂⁱ = 1.8; blending), LocalMinimumMixingLength(; blending))
        closure = TKEBasedTurbulenceClosure(; mixing_length)
        @test closure.mixing_length.blending === HarmonicBlend{FT}(FT(6))
        @test closure.mixing_length.ℂˢ isa FT
        @test isbits(closure)
        @test adapt(Array, closure).mixing_length === closure.mixing_length
        @test occursin("blending = HarmonicBlend(exponent = 6", sprint(show, closure))
    end
    @test !occursin("blending", sprint(show, TKEBasedTurbulenceClosure()))
    @test sprint(show, HardMinimum()) == "HardMinimum()"
    @test convert_eltype(Float32, HarmonicBlend(4)) === HarmonicBlend(4f0)
end

@testset "HarmonicBlend approaches HardMinimum [$(FT), $(nameof(typeof(hard)))]" for FT in test_float_types(),
        hard in (GradientLimitedMixingLength(ℂˢ = 0.79, ℂⁱ = 1.78), LocalMinimumMixingLength(ℂˢ = 0.79))
    Oceananigans.defaults.FloatType = FT
    with_blending(blending) = hard isa GradientLimitedMixingLength ?
        GradientLimitedMixingLength(ℂˢ = hard.ℂˢ, ℂⁱ = hard.ℂⁱ; blending) : LocalMinimumMixingLength(ℂˢ = hard.ℂˢ; blending)

    reference = blending_column(hard)
    ℓ = column(reference.closure_fields.ℓᵐ)
    interior_faces = 2:reference.grid.Nz
    @test all(isfinite, ℓ) && maximum(ℓ) > 100

    # The same state, so only the blending differs
    differences = map((2, 8, 64, 1024)) do p
        model = blending_column(with_blending(HarmonicBlend(exponent = p)))
        ℓᵖ = column(model.closure_fields.ℓᵐ)
        @test all(isfinite, ℓᵖ)
        hard isa GradientLimitedMixingLength && @test ℓᵖ[1] == 0
        # a blend never exceeds the minimum it replaces
        @test all(ℓᵖ[interior_faces] .≤ ℓ[interior_faces] .* (1 + 10 * eps(FT)))
        # the center length too, through the dissipation rate's sink
        @test all(isfinite, column(model.closure_fields.Lᵉ))
        maximum(abs.(ℓᵖ .- ℓ)[interior_faces] ./ ℓ[interior_faces])
    end
    @test issorted(differences; rev = true)
    @test differences[1] > 0.05
    @test differences[end] < 2e-2

    # and a run with a large exponent stays close to the hard minimum's
    a = state(run_steps!(blending_column(hard)))
    b = state(run_steps!(blending_column(with_blending(HarmonicBlend(exponent = 1024)))))
    @test all(x -> all(isfinite, x), b)
    @test b.ρθ ≈ a.ρθ rtol=1e-4
    @test b.ρe ≈ a.ρe rtol=0.1
end

@testset "HardMinimum is the default, bitwise [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    for (default, explicit) in ((GradientLimitedMixingLength(ℂˢ = 0.79, ℂⁱ = 1.78), GradientLimitedMixingLength(ℂˢ = 0.79, ℂⁱ = 1.78, blending = HardMinimum())),
                                (LocalMinimumMixingLength(ℂˢ = 0.79), LocalMinimumMixingLength(ℂˢ = 0.79, blending = HardMinimum())))
        @test default === explicit
        a = state(run_steps!(blending_column(default)))
        b = state(run_steps!(blending_column(explicit)))
        @test all(name -> isequal(a[name], b[name]), keys(a))
    end
end

#####
##### Derivatives of the gradient-limited length: Enzyme against finite differences, on the CPU
#####

reverse_derivative(f, x) = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.set_strong_zero(Enzyme.Reverse)), Enzyme.Const(f),
                                           Enzyme.Active, Enzyme.Active(x))[1][1]
central_difference(f, x, h) = (f(x * (1 + h)) - f(x * (1 - h))) / (2x * h)

@testset "Enzyme against finite differences: the blend exponent and the mixing-length coefficients" begin
    Oceananigans.defaults.FloatType = Float64

    # Pointwise, including the degenerate arguments
    for (a, b) in ((30.0, 120.0), (30.0, 30.0), (30.0, Inf), (Inf, Inf), (0.0, 30.0), (0.0, 0.0))
        in_exponent = p -> blend(HarmonicBlend(p), a, b)
        in_length = x -> blend(HarmonicBlend(4.0), x * a, b)
        @test isfinite(reverse_derivative(in_exponent, 4.0))
        isfinite(a) && a > 0 && @test reverse_derivative(in_length, 1.0) ≈ central_difference(in_length, 1.0, 1e-6) rtol=1e-6
        isfinite(b) && a > 0 && @test reverse_derivative(in_exponent, 4.0) ≈ central_difference(in_exponent, 4.0, 1e-6) rtol=1e-5 atol=1e-12
    end
    @test reverse_derivative(p -> blend(HarmonicBlend(p), 30.0, Inf), 4.0) == 0

    model = run_steps!(blending_column(GradientLimitedMixingLength(ℂˢ = 0.79, ℂⁱ = 1.78, blending = HarmonicBlend(exponent = 4)); arch = CPU()))
    grid, fields = model.grid, model.closure_fields
    Nz = grid.Nz
    specific_tke = compute!(Field(model.tracers[TKE_NAME] / model.dynamics.reference_state.density))
    Oceananigans.BoundaryConditions.fill_halo_regions!(specific_tke)
    # The arrays under the fields, with their halos: what the column sweeps index
    e = specific_tke.data
    N² = fields.N².data

    # The face length at level k, and the center length (the dissipation's), from a fresh pair of sweeps
    function face_length(mixing_length, k)
        closure = TKEBasedTurbulenceClosure(; mixing_length)
        ℓ = zeros(1, 1, Nz + 1)
        ℓˢ = zeros(1, 1, Nz + 1)
        fill_mixing_length!(ℓ, ℓˢ, 1, 1, grid, closure.mixing_length, closure, e, N²)
        return ℓ[1, 1, k]
    end

    function center_length(mixing_length, k)
        closure = TKEBasedTurbulenceClosure(; mixing_length)
        ℓ = zeros(1, 1, Nz + 1)
        ℓˢ = zeros(1, 1, Nz + 1)
        fill_mixing_length!(ℓ, ℓˢ, 1, 1, grid, closure.mixing_length, closure, e, N²)
        return mixing_lengthᶜᶜᶜ(1, 1, k, grid, closure, e, (; ℓˢ, N²))
    end

    gradient_limited(; p = 4.0, ℂˢ = 0.79, ℂⁱ = 1.78) = GradientLimitedMixingLength(ℂˢ, ℂⁱ, HarmonicBlend(p))
    @test face_length(gradient_limited(), 12) == column(fields.ℓᵐ)[12]

    resolved = 0
    for k in (2, 6, 12, 18, 21, 24, 30), length in (face_length, center_length)
        for (f, x) in ((p -> length(gradient_limited(; p), k), 4.0),
                       (ℂˢ -> length(gradient_limited(; ℂˢ), k), 0.79),
                       (ℂⁱ -> length(gradient_limited(; ℂⁱ), k), 1.78))
            ad = reverse_derivative(f, x)
            fd = central_difference(f, x, 1e-6)
            @test isfinite(ad)
            @test ad ≈ fd rtol=1e-5 atol=1e-8
            resolved += abs(ad) > 1e-6
        end
    end
    @test resolved ≥ 20

    # The hard minimum's derivative with respect to ℂⁱ exists too, piecewise
    hard = ℂⁱ -> face_length(GradientLimitedMixingLength(0.79, ℂⁱ), 21)
    @test isfinite(reverse_derivative(hard, 1.78))
end
