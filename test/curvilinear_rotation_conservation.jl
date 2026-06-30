#####
##### Conservation tests for the rotation-type terms (Coriolis + curvature metric)
##### in compressible momentum advection on a LatitudeLongitudeGrid.
#####
##### Both the Coriolis force f×𝐮 and the curvature metric are "rotation" terms:
##### they are fictitious forces perpendicular to the flow. Two discrete
##### invariants must therefore be respected over the whole domain:
#####
#####   1. ENERGY: a rotation does no work, Σ 𝐮·(f×𝐮)·V = 0. This is checked for
#####      the full 3D `SphericalCoriolis` (including the non-traditional
#####      f̃ = 2Ω cosφ vertical coupling), in 2D (Nz=1) and 3D (Nz>1).
#####
#####   2. ENSTROPHY: for nondivergent barotropic flow the absolute enstrophy
#####      ½∫(ζ+f)² is conserved, so the rotation terms inject no spurious
#####      enstrophy: Σ (ζ+f)·∂ₜζ·A = 0, where ∂ₜζ is the discrete curl of the
#####      momentum tendency (vorticity-flux advection — which carries the
#####      hydrostatic curvature metric — plus Coriolis). A streamfunction-
#####      derived velocity makes the flow nondivergent to machine precision.
#####
##### These mirror the curvature-metric energy tests in
##### `test/curvilinear_metric_terms.jl`, extended to the vorticity budget and
##### to validating the 3D spherical Coriolis discretization.

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics, ExplicitTimeStepping
using Breeze.AtmosphereModels: x_momentum_flux_divergence, y_momentum_flux_divergence
using Oceananigans
using Oceananigans: Field, interior, compute!, set!, fill_halo_regions!
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: ζ₃ᶠᶠᶜ, div_xyᶜᶜᶜ, Azᶠᶠᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ,
                              δxᶜᵃᵃ, δyᵃᶜᵃ, Δx⁻¹ᶜᶠᶜ, Δy⁻¹ᶠᶜᶜ
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U
using Oceananigans.TimeSteppers: update_state!
import Oceananigans.Coriolis: fᶠᶠᵃ
using Test

if !@isdefined(test_float_types)
    test_float_types() = (Float64,)
end
if !@isdefined(default_arch)
    using Oceananigans.Architectures: CPU
    default_arch = CPU()
end

const φ₀ = 80

llg(arch, FT, Nz) = LatitudeLongitudeGrid(arch, FT; size = (72, 70, Nz), halo = (6, 6, 6),
                                          longitude = (0, 360), latitude = (-φ₀, φ₀), z = (0, 1e3),
                                          topology = (Periodic, Bounded, Bounded))

base_model(arch, FT, Nz; momentum_advection = WENO()) =
    AtmosphereModel(llg(arch, FT, Nz);
                    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                                    reference_potential_temperature = FT(300),
                                                    surface_pressure = FT(1e5)),
                    coriolis = SphericalCoriolis(),
                    momentum_advection)

#####
##### 1. SphericalCoriolis does no work (energy), in 2D and 3D
#####

# A generic flow; the model enforces impenetrability at the walls.
cuᵢ(λ, φ, z) = 8 + 3 * cosd(2λ) + 2 * sind(φ)
cvᵢ(λ, φ, z) = 5 + 2 * sind(3λ) + cosd(2φ)
cwᵢ(λ, φ, z) = 0.5 + 0.3 * cosd(λ) + 0.2 * sind(2φ) + 0.1 * sind(λ) * cosd(φ)

function coriolis_work(model)
    grid = model.grid; cor = model.coriolis; mom = model.momentum
    Cu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(x_f_cross_U, grid, cor, mom)))
    Cv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(y_f_cross_U, grid, cor, mom)))
    Cw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(z_f_cross_U, grid, cor, mom)))
    Vu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(Vᶠᶜᶜ, grid)))
    Vv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(Vᶜᶠᶜ, grid)))
    Vw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(Vᶜᶜᶠ, grid)))
    u, v, w = model.velocities
    wu = interior(u) .* interior(Cu) .* interior(Vu)
    wv = interior(v) .* interior(Cv) .* interior(Vv)
    ww = interior(w) .* interior(Cw) .* interior(Vw)
    return abs(sum(wu) + sum(wv) + sum(ww)) / (sum(abs, wu) + sum(abs, wv) + sum(abs, ww))
end

@testset "SphericalCoriolis does no work [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    tol = FT == Float64 ? 1e-10 : 1e-4
    for (label, Nz) in (("2D", 1), ("3D", 6))
        model = base_model(default_arch, FT, Nz)
        set!(model; θ = 300, ρ = model.dynamics.reference_state.density, u = cuᵢ, v = cvᵢ, w = cwᵢ)
        update_state!(model)
        @test coriolis_work(model) < tol
        if Nz > 1   # the non-traditional vertical Coriolis must be active in 3D
            Cw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(z_f_cross_U, model.grid, model.coriolis, model.momentum)))
            @test maximum(abs, interior(Cw)) > 0
        end
    end
end

#####
##### 2. No spurious enstrophy source from metric + Coriolis (nondivergent flow)
#####

# Streamfunction at (Face,Face,Center), vanishing at the meridional walls so the
# flow is impenetrable-consistent and confined; ∝ amplitude 5e6 ⇒ u ~ 20 m/s.
Ψfunc(λ, φ, z) = 5e6 * (1 - (φ/φ₀)^2)^2 * cosd(90*φ/φ₀) *
                 (sind(2λ) + 0.6*cosd(5λ) + 0.5*sind(9λ) + 0.4*cosd(13λ))

# Discrete skew-gradient ⇒ exactly nondivergent velocity.
@inline ustream(i, j, k, grid, Ψ) = -δyᵃᶜᵃ(i, j, k, grid, Ψ) * Δy⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline vstream(i, j, k, grid, Ψ) = +δxᶜᵃᵃ(i, j, k, grid, Ψ) * Δx⁻¹ᶜᶠᶜ(i, j, k, grid)

function set_nondivergent_flow!(model)
    grid = model.grid
    Ψ = Field{Face, Face, Center}(grid); set!(Ψ, Ψfunc); fill_halo_regions!(Ψ)
    u, v, w = model.velocities
    set!(u, compute!(Field(KernelFunctionOperation{Face, Center, Center}(ustream, grid, Ψ))))
    set!(v, compute!(Field(KernelFunctionOperation{Center, Face, Center}(vstream, grid, Ψ))))
    set!(w, 0)
    fill_halo_regions!((u, v, w))
    set!(model.momentum.ρu, u); set!(model.momentum.ρv, v); set!(model.momentum.ρw, w)
    fill_halo_regions!((model.momentum.ρu, model.momentum.ρv, model.momentum.ρw))
    return nothing
end

function enstrophy_source(model)
    grid = model.grid; dyn = model.dynamics; cor = model.coriolis
    mom = model.momentum; vel = model.velocities; adv = model.advection.momentum
    Au = compute!(Field(KernelFunctionOperation{Face, Center, Center}(x_momentum_flux_divergence, grid, adv, mom, vel, dyn)))
    Av = compute!(Field(KernelFunctionOperation{Center, Face, Center}(y_momentum_flux_divergence, grid, adv, mom, vel, dyn)))
    Cu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(x_f_cross_U, grid, cor, mom)))
    Cv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(y_f_cross_U, grid, cor, mom)))
    Tu = compute!(Field(-Au - Cu)); Tv = compute!(Field(-Av - Cv))   # ρ=1 velocity tendency (pressure is curl-free)
    fill_halo_regions!((Tu, Tv))
    dζ = compute!(Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, Tu, Tv)))
    ζ  = compute!(Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, vel.u, vel.v)))
    f  = compute!(Field(KernelFunctionOperation{Face, Face, Center}(fᶠᶠᵃ, grid, cor)))
    Az = compute!(Field(KernelFunctionOperation{Face, Face, Center}(Azᶠᶠᶜ, grid)))
    q = interior(ζ) .+ interior(f)
    e = q .* interior(dζ) .* interior(Az)
    rel = abs(sum(e)) / sum(abs, e)
    return rel, maximum(abs, interior(ζ))
end

function max_divergence(model)
    d = compute!(Field(KernelFunctionOperation{Center, Center, Center}(div_xyᶜᶜᶜ, model.grid, model.momentum.ρu, model.momentum.ρv)))
    return maximum(abs, interior(d))
end

@testset "No spurious enstrophy source from metric + Coriolis [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    tol = FT == Float64 ? 1e-10 : 1e-4
    # The vector-invariant scheme carries the hydrostatic metric in its
    # (enstrophy-conserving) vorticity flux; that is what conserves enstrophy.
    for (label, Nz) in (("2D", 1), ("3D", 6))
        model = base_model(default_arch, FT, Nz; momentum_advection = CompressibleVectorInvariant())
        set!(model; θ = 300, ρ = 1)
        set_nondivergent_flow!(model)
        @test max_divergence(model) < tol            # flow is genuinely nondivergent
        rel, ζmax = enstrophy_source(model)
        @test ζmax > 1e-6                            # ... and carries nontrivial vorticity
        @test rel < tol                              # ⇒ no spurious enstrophy source
    end
end
