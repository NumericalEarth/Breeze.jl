include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Grids: znodes
using Breeze.TurbulenceClosures: mixing_lengthᶜᶜᶜ
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: blended_length, asymptotic_length
using Test

#####
##### The mixing length of SHOC, `BlendedMixingLength`, a mixing length of `TKEBasedTurbulenceClosure`, compared
##### with the expressions of shoc.F90 written out with its literals. The port bounds the length from above by
##### sums of inverse squares (maxlen, the horizontal grid spacing) where shoc.F90 takes a `min`, and does not
##### impose `minlen`: in the regime where `minlen` is inactive the two differ by at most the fraction
##### (ℓ / maxlen)² + (ℓ / Δʰ)².
#####

# `compute_shoc_mix_shoc_length` followed by `check_length_scale_shoc_length` of E3SM `shoc.F90`
function fortran_mixing_length(tke, brunt, zt, l_inf, host_scale)
    FT = typeof(tke)
    length_fac, tscale, vk, maxlen, minlen = FT(0.5), FT(400), FT(0.4), FT(20000), FT(20)
    tkes = sqrt(tke)
    brunt2 = brunt >= 0 ? brunt : zero(FT)
    mix = min(maxlen, (FT(2.8284) * sqrt(1 / ((1 / (tscale * tkes * vk * zt))
              + (1 / (tscale * tkes * l_inf)) + FT(0.01) * (brunt2 / tke)))) / length_fac)
    mix = min(maxlen, mix)
    mix = max(minlen, mix)
    mix = min(host_scale, mix)
    return mix
end

# `compute_l_inf_shoc_length`: 0.1 ∑ √e z dz / ∑ √e dz
fortran_asymptotic_length(tke, zt, dz) = 0.1 * sum(sqrt.(tke) .* zt .* dz) / sum(sqrt.(tke) .* dz)

bound_fraction(ℓ, host_scale) = (ℓ / 20000)^2 + (ℓ / host_scale)^2

@testset "BlendedMixingLength is the length of shoc.F90 [$FT]" for FT in test_float_types()
    ml = BlendedMixingLength(FT)
    @test ml isa AbstractMixingLength
    @test summary(ml) == "BlendedMixingLength{$FT}"
    @test ml.ℂˡ₁ === FT(0.5) && ml.ℂˡ₂ === FT(2.8284) && ml.ℂˡ₅ === FT(400) && ml.maximum_length === FT(20000)
    @test_throws ArgumentError BlendedMixingLength(FT; minlen = 20)

    rtol_round = FT == Float32 ? 1e-5 : 1e-12
    for tke in FT[0.01, 0.1, 0.5, 2], brunt in FT[-1e-4, 0, 1e-5, 3e-4], zt in FT[20, 100, 500, 2000],
        l_inf in FT[100, 300], host_scale in FT[1e5, 3000, Inf]
        ℓᶠ = fortran_mixing_length(tke, brunt, zt, l_inf, host_scale)
        ℓᶠ > 20 || continue # minlen is not imposed by the port
        ℓ = blended_length(ml, zt, sqrt(tke), l_inf, max(brunt, zero(FT)), host_scale^2)
        @test abs(ℓ - ℓᶠ) / ℓᶠ ≤ bound_fraction(ℓᶠ, host_scale) + rtol_round
        @test ℓ ≤ ℓᶠ * (1 + rtol_round)
    end

    # The length vanishes with the height above the surface and with √e, without a floor
    @test blended_length(ml, zero(FT), FT(0.3), FT(200), zero(FT), FT(Inf)) == 0
    @test blended_length(ml, FT(100), FT(1e-6), FT(200), FT(1e-4), FT(Inf)) < FT(1)
end

function bomex_like_model(FT; closure = SimplifiedHigherOrderClosure(FT), Nz = 40)
    grid = RectilinearGrid(default_arch, FT; size = Nz, z = (0, 2000), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    θᵢ(z) = 298.7 + 4e-3 * max(z - 500, 0)
    qᵢ(z) = 0.017 - 3e-6 * z
    reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = θᵢ,
                                     vapor_mass_fraction = qᵢ)
    ρ₀ = FT(1.16)
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * FT(8e-3))),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * FT(5.2e-5))))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), closure, boundary_conditions,
                            microphysics = SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium()))
    set!(model, θ = θᵢ, qᵗ = qᵢ, u = -5)
    e₀ = CenterField(grid)
    set!(e₀, z -> max(0.5 * (1 - z / 1000), 0.01))
    set!(model.tracers.ρe, reference_state.density * e₀)
    return model
end

column(f) = Array{Float64}(interior(f))[1, 1, :]

@testset "BlendedMixingLength in a column [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    model = bomex_like_model(FT)
    for _ in 1:10
        time_step!(model, 30)
    end
    closure = model.closure
    fields = model.closure_fields
    grid = model.grid
    ρ = column(model.dynamics.reference_state.density)
    e = column(model.tracers.ρe) ./ ρ
    eᵐⁱⁿ = closure.minimum_tke
    zc = collect(znodes(grid, Center()))
    zf = collect(znodes(grid, Face()))
    Δz = diff(zf)
    N² = column(fields.N²)
    ℓ = column(fields.ℓᵐ)

    # l∞ of shoc.F90 from the TKE floored at `minimum_tke`, kept in ℓˢ
    l∞ = fortran_asymptotic_length(max.(e, eᵐⁱⁿ), zc, Δz)
    @test all(isapprox.(column(fields.ℓˢ), l∞; rtol = FT == Float32 ? 1e-4 : 1e-10))

    # The faces: √e reconstructed from the centers, the stored N²
    Nz = size(grid, 3)
    rtol = FT == Float32 ? 1e-4 : 1e-10
    for k in 2:Nz
        u = (sqrt(max(e[k-1], eᵐⁱⁿ)) + sqrt(max(e[k], eᵐⁱⁿ))) / 2
        ℓᶠ = fortran_mixing_length(u^2, N²[k], zf[k], l∞, Inf)
        ℓᶠ > 20 || continue
        @test abs(ℓ[k] - ℓᶠ) / ℓᶠ ≤ bound_fraction(ℓᶠ, Inf) + rtol
    end

    # The centers, where the dissipation lives, with N² from the two faces
    e_field = Field(model.tracers.ρe / model.dynamics.reference_state.density)
    compute!(e_field)
    for k in 2:Nz-1
        ℓᶜ = mixing_lengthᶜᶜᶜ(1, 1, k, grid, closure, e_field, fields)
        ℓᶠ = fortran_mixing_length(max(e[k], eᵐⁱⁿ), (N²[k] + N²[k+1]) / 2, zc[k], l∞, Inf)
        ℓᶠ > 20 || continue
        @test abs(ℓᶜ - ℓᶠ) / ℓᶠ ≤ bound_fraction(ℓᶠ, Inf) + rtol
    end
    @test all(isfinite, ℓ)
    @test all(≥(0), ℓ)
end
