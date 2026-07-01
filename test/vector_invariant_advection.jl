using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics, ExplicitTimeStepping
using Oceananigans
using Oceananigans.Advection: VectorInvariant, WENOVectorInvariant, Centered, WENO
using Oceananigans.Grids: required_halo_size_x, required_halo_size_y, required_halo_size_z
using Oceananigans.TimeSteppers: update_state!
using Test

#####
##### Helpers
#####

# Halo 6 accommodates the wide WENOVectorInvariant stencil.
vi_grid(FT) = RectilinearGrid(default_arch;
                              size = (16, 16, 8), halo = (6, 6, 6),
                              x = (0, 1e3), y = (0, 1e3), z = (0, 1e3),
                              topology = (Periodic, Periodic, Bounded))

function vi_model(grid, momentum_advection)
    FT = eltype(grid)
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = FT(300),
                                    surface_pressure = FT(1e5),
                                    standard_pressure = FT(1e5))
    return AtmosphereModel(grid; dynamics,
                           formulation = :LiquidIcePotentialTemperature,
                           microphysics = nothing,
                           momentum_advection)
end

# Refresh halos + auxiliary state + tendencies, then return the momentum tendencies.
function momentum_tendencies(model)
    update_state!(model)
    G = model.timestepper.Gⁿ
    return map(f -> Array(interior(f)), (G.ρu, G.ρv, G.ρw))
end

const VI_FLAVORS = (
    ("VI horizontal",   () -> CompressibleVectorInvariant(; divergence = HorizontalDivergence())),
    ("VI 3D",           () -> CompressibleVectorInvariant(; divergence = ThreeDimensionalDivergence())),
    ("WENO VI 3D",      () -> CompressibleWENOVectorInvariant()),
    ("WENO VI horiz",   () -> CompressibleWENOVectorInvariant(; divergence = HorizontalDivergence())),
)

#####
##### Construction & configuration
#####

@testset "CompressibleVectorInvariant construction & API [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @test CompressibleVectorInvariant() isa CompressibleVectorInvariant
    @test CompressibleVectorInvariant().divergence isa HorizontalDivergence
    @test CompressibleVectorInvariant(; divergence = ThreeDimensionalDivergence()).divergence isa ThreeDimensionalDivergence

    # WENOVectorInvariant is a constructor returning a VectorInvariant; the wrapper holds it.
    @test CompressibleWENOVectorInvariant().scheme isa VectorInvariant
    @test CompressibleWENOVectorInvariant().divergence isa ThreeDimensionalDivergence
    @test CompressibleWENOVectorInvariant(; divergence = HorizontalDivergence()).divergence isa HorizontalDivergence

    # Sub-scheme kwargs forward to the underlying VectorInvariant.
    @test CompressibleVectorInvariant(; vorticity_scheme = WENO()).scheme isa VectorInvariant

    # summary/show produce informative strings.
    s = summary(CompressibleVectorInvariant())
    @test occursin("CompressibleVectorInvariant", s)
    @test occursin("HorizontalDivergence", s)
end

#####
##### Halo delegation and order adaptation
#####

@testset "halo delegation & adapt_advection_order [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = vi_grid(FT)

    # Halo requirement must match the underlying scheme (the generic fallback would
    # return 1, under-sizing the wide WENO stencil).
    for (vi, inner) in ((CompressibleVectorInvariant(), VectorInvariant()),
                        (CompressibleWENOVectorInvariant(), WENOVectorInvariant()))
        @test required_halo_size_x(vi) == required_halo_size_x(inner)
        @test required_halo_size_y(vi) == required_halo_size_y(inner)
        @test required_halo_size_z(vi) == required_halo_size_z(inner)
    end

    a = CompressibleVectorInvariant()
    @test Oceananigans.Advection.adapt_advection_order(a, grid) === a
end

#####
##### Model construction, materialization, and stability over several steps
#####

@testset "builds, materializes & steps finite: $label [$FT]" for FT in test_float_types(),
                                                                 (label, make) in VI_FLAVORS
    Oceananigans.defaults.FloatType = FT
    grid = vi_grid(FT)
    model = vi_model(grid, make())

    @test model.advection.momentum isa CompressibleVectorInvariant

    set!(model;
         ρ = (x, y, z) -> 1.2,
         θ = (x, y, z) -> 300.0 + 0.1 * sinpi(2x / 1e3),
         u = (x, y, z) ->  2.0 * sinpi(2x / 1e3) * cospi(2y / 1e3),
         v = (x, y, z) -> -2.0 * cospi(2x / 1e3) * sinpi(2y / 1e3),
         w = (x, y, z) ->  0.2 * sinpi(z / 1e3))

    for _ in 1:3
        time_step!(model, 1e-4)
    end

    @test all(isfinite, interior(model.momentum.ρu))
    @test all(isfinite, interior(model.momentum.ρv))
    @test all(isfinite, interior(model.momentum.ρw))
    @test !any(isnan, parent(model.momentum.ρu))
end

#####
##### Correctness: uniform flow has zero advective tendency, so the vector-invariant
##### momentum tendency must equal the flux-form one (only the advection differs;
##### every other term is identical, and advection of a uniform field is zero).
#####

@testset "uniform flow ⇒ VI advection == flux-form: $label [$FT]" for FT in test_float_types(),
                                                                     (label, make) in VI_FLAVORS
    Oceananigans.defaults.FloatType = FT
    grid = vi_grid(FT)

    uniform!(model) = set!(model; ρ = (x, y, z) -> 1.2, θ = (x, y, z) -> 300.0,
                                  u = (x, y, z) -> 7.0, v = (x, y, z) -> -3.0, w = (x, y, z) -> 0)

    m_vi = vi_model(grid, make());            uniform!(m_vi)
    m_ff = vi_model(grid, Centered(order=2)); uniform!(m_ff)

    Gρu_vi, Gρv_vi, _ = momentum_tendencies(m_vi)
    Gρu_ff, Gρv_ff, _ = momentum_tendencies(m_ff)

    atol = sqrt(eps(FT)) * (1 + maximum(abs, Gρu_ff))
    @test maximum(abs, Gρu_vi .- Gρu_ff) < atol
    @test maximum(abs, Gρv_vi .- Gρv_ff) < atol
end

#####
##### Consistency: with w ≡ 0 the vertical advection term and the vertical part of
##### the divergence correction both vanish, so the horizontal and 3D divergence
##### flavors must produce identical horizontal-momentum tendencies.
#####

@testset "w=0 ⇒ horizontal and 3D flavors agree [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = vi_grid(FT)

    sheared!(model) = set!(model; ρ = (x, y, z) -> 1.2, θ = (x, y, z) -> 300.0,
                                  u = (x, y, z) ->  2.0 * sinpi(2x / 1e3) * cospi(2y / 1e3),
                                  v = (x, y, z) -> -2.0 * cospi(2x / 1e3) * sinpi(2y / 1e3),
                                  w = (x, y, z) -> 0)

    m_h = vi_model(grid, CompressibleVectorInvariant(; divergence = HorizontalDivergence()));     sheared!(m_h)
    m_3 = vi_model(grid, CompressibleVectorInvariant(; divergence = ThreeDimensionalDivergence())); sheared!(m_3)

    Gρu_h, Gρv_h, _ = momentum_tendencies(m_h)
    Gρu_3, Gρv_3, _ = momentum_tendencies(m_3)

    atol = sqrt(eps(FT)) * (1 + maximum(abs, Gρu_h))
    @test maximum(abs, Gρu_h .- Gρu_3) < atol
    @test maximum(abs, Gρv_h .- Gρv_3) < atol
end

#####
##### Exactness: the momentum flux divergence must converge to the analytic
##### ∇·(ρ𝐮⊗𝐮) for a smooth compressible flow with ∇·𝐮 ≠ 0 and variable ρ.
#####
##### This is the "continuous/discrete identity test" of
##### design/compressible_orthogonal_cgrid_weno_vi_with_hollingsworth.md. It
##### catches, in particular, assembling the 3D flavor from Oceananigans'
##### `U_dot_∇u`, whose upwind vertical advection assumes ∇·𝐮 = 0 and therefore
##### carries a spurious O(1) term 𝐮(∇·𝐮) in compressible flow.
#####

using Breeze.AtmosphereModels: x_momentum_flux_divergence, y_momentum_flux_divergence,
                               z_momentum_flux_divergence
using Oceananigans.AbstractOperations: KernelFunctionOperation

# Smooth periodic-in-x,y fields; w and ∂z(u,v) vanish at the z-walls so the
# state is compatible with impenetrable free-slip boundaries.
const Lᵛⁱ = 1e3
uᵛⁱ(x, y, z) = 10 + 3 * sinpi(2x / Lᵛⁱ) * cospi(2y / Lᵛⁱ) + 2 * cospi(z / Lᵛⁱ)
vᵛⁱ(x, y, z) = 5 + 2 * cospi(2x / Lᵛⁱ) + sinpi(2y / Lᵛⁱ) + cospi(z / Lᵛⁱ)
wᵛⁱ(x, y, z) = 0.5 * sinpi(z / Lᵛⁱ) * (1 + 0.3 * cospi(2x / Lᵛⁱ) * sinpi(2y / Lᵛⁱ))
ρᵛⁱ(x, y, z) = 1 + 0.2 * cospi(z / Lᵛⁱ) + 0.1 * sinpi(2x / Lᵛⁱ) * cospi(2y / Lᵛⁱ)

# Analytic momentum fluxes and their divergence via machine-accurate central
# differences (h² ≈ 1e-8 relative; far below the discretization errors probed).
ρuuᵛⁱ(x, y, z) = ρᵛⁱ(x, y, z) * uᵛⁱ(x, y, z) * uᵛⁱ(x, y, z)
ρvuᵛⁱ(x, y, z) = ρᵛⁱ(x, y, z) * vᵛⁱ(x, y, z) * uᵛⁱ(x, y, z)
ρwuᵛⁱ(x, y, z) = ρᵛⁱ(x, y, z) * wᵛⁱ(x, y, z) * uᵛⁱ(x, y, z)
ρvvᵛⁱ(x, y, z) = ρᵛⁱ(x, y, z) * vᵛⁱ(x, y, z) * vᵛⁱ(x, y, z)
ρwvᵛⁱ(x, y, z) = ρᵛⁱ(x, y, z) * wᵛⁱ(x, y, z) * vᵛⁱ(x, y, z)
ρwwᵛⁱ(x, y, z) = ρᵛⁱ(x, y, z) * wᵛⁱ(x, y, z) * wᵛⁱ(x, y, z)

const hᵛⁱ = 1e-1
∂xᵛⁱ(f, x, y, z) = (f(x + hᵛⁱ, y, z) - f(x - hᵛⁱ, y, z)) / 2hᵛⁱ
∂yᵛⁱ(f, x, y, z) = (f(x, y + hᵛⁱ, z) - f(x, y - hᵛⁱ, z)) / 2hᵛⁱ
∂zᵛⁱ(f, x, y, z) = (f(x, y, z + hᵛⁱ) - f(x, y, z - hᵛⁱ)) / 2hᵛⁱ

exact_x_mfd(x, y, z) = ∂xᵛⁱ(ρuuᵛⁱ, x, y, z) + ∂yᵛⁱ(ρvuᵛⁱ, x, y, z) + ∂zᵛⁱ(ρwuᵛⁱ, x, y, z)
exact_y_mfd(x, y, z) = ∂xᵛⁱ(ρvuᵛⁱ, x, y, z) + ∂yᵛⁱ(ρvvᵛⁱ, x, y, z) + ∂zᵛⁱ(ρwvᵛⁱ, x, y, z)
exact_z_mfd(x, y, z) = ∂xᵛⁱ(ρwuᵛⁱ, x, y, z) + ∂yᵛⁱ(ρwvᵛⁱ, x, y, z) + ∂zᵛⁱ(ρwwᵛⁱ, x, y, z)

function vi_exactness_grid(FT, N)
    return RectilinearGrid(default_arch, FT;
                           size = (N, N, N ÷ 2), halo = (6, 6, 6),
                           x = (0, Lᵛⁱ), y = (0, Lᵛⁱ), z = (0, Lᵛⁱ),
                           topology = (Periodic, Periodic, Bounded))
end

# Max relative error of the discrete momentum flux divergence against the exact
# ∇·(ρ𝐮⊗𝐮), measured away from the z-walls.
function vi_exactness_errors(model)
    grid = model.grid
    adv, mom, vel, dyn = model.advection.momentum, model.momentum, model.velocities, model.dynamics
    Ax = Field(KernelFunctionOperation{Face, Center, Center}(x_momentum_flux_divergence, grid, adv, mom, vel, dyn))
    Ay = Field(KernelFunctionOperation{Center, Face, Center}(y_momentum_flux_divergence, grid, adv, mom, vel, dyn))
    Az = Field(KernelFunctionOperation{Center, Center, Face}(z_momentum_flux_divergence, grid, adv, mom, vel, dyn))
    ex = Field{Face, Center, Center}(grid);   set!(ex, exact_x_mfd)
    ey = Field{Center, Face, Center}(grid);   set!(ey, exact_y_mfd)
    ez = Field{Center, Center, Face}(grid);   set!(ez, exact_z_mfd)
    Nz = size(grid, 3)
    r = 4:Nz-3
    err(A, e) = maximum(abs.(interior(A)[:, :, r] .- interior(e)[:, :, r])) / maximum(abs.(interior(e)))
    return err(Ax, ex), err(Ay, ey), err(Az, ez)
end

EXACTNESS_SCHEMES = (VI_FLAVORS..., ("flux-form WENO", () -> WENO()))

@testset "momentum advection converges to exact ∇·(ρ𝐮⊗𝐮): $label" for (label, make) in EXACTNESS_SCHEMES
    Oceananigans.defaults.FloatType = Float64

    errs = map((16, 32)) do N
        model = vi_model(vi_exactness_grid(Float64, N), make())
        set!(model; ρ = ρᵛⁱ, θ = (x, y, z) -> 300.0, u = uᵛⁱ, v = vᵛⁱ, w = wᵛⁱ)
        update_state!(model)
        vi_exactness_errors(model)
    end
    (ex₁, ey₁, ez₁), (ex₂, ey₂, ez₂) = errs

    # Consistent (converging) in all three components; a spurious 𝐮(∇·𝐮) term
    # would leave these stuck at O(1).
    @test ex₁ < 0.5
    @test ey₁ < 0.5
    @test ez₁ < 0.5
    @test ex₂ < ex₁ / 2
    @test ey₂ < ey₁ / 2
    @test ez₂ < ez₁ / 2
end

#####
##### Energy-production audit (design note): the discrete kinetic-energy
##### production of the advective operator, P = Σ 𝐮·[∇·(ρ𝐮⊗𝐮)]·V, must match
##### the continuum value for every scheme — an indefinite or non-converging
##### residual is the Hollingsworth signature.
#####

using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ

function vi_energy_production(model)
    grid = model.grid
    adv, mom, vel, dyn = model.advection.momentum, model.momentum, model.velocities, model.dynamics
    Ax = Field(KernelFunctionOperation{Face, Center, Center}(x_momentum_flux_divergence, grid, adv, mom, vel, dyn))
    Ay = Field(KernelFunctionOperation{Center, Face, Center}(y_momentum_flux_divergence, grid, adv, mom, vel, dyn))
    Az = Field(KernelFunctionOperation{Center, Center, Face}(z_momentum_flux_divergence, grid, adv, mom, vel, dyn))
    ex = Field{Face, Center, Center}(grid);   set!(ex, exact_x_mfd)
    ey = Field{Center, Face, Center}(grid);   set!(ey, exact_y_mfd)
    ez = Field{Center, Center, Face}(grid);   set!(ez, exact_z_mfd)
    Vu = Field(KernelFunctionOperation{Face, Center, Center}(Vᶠᶜᶜ, grid))
    Vv = Field(KernelFunctionOperation{Center, Face, Center}(Vᶜᶠᶜ, grid))
    Vw = Field(KernelFunctionOperation{Center, Center, Face}(Vᶜᶜᶠ, grid))
    production(qx, qy, qz) = sum(interior(vel.u) .* interior(qx) .* interior(Vu)) +
                             sum(interior(vel.v) .* interior(qy) .* interior(Vv)) +
                             sum(interior(vel.w) .* interior(qz) .* interior(Vw))
    return production(Ax, Ay, Az), production(ex, ey, ez)
end

@testset "energy-production audit P_VI: $label" for (label, make) in EXACTNESS_SCHEMES
    Oceananigans.defaults.FloatType = Float64

    audits = map((16, 32)) do N
        model = vi_model(vi_exactness_grid(Float64, N), make())
        set!(model; ρ = ρᵛⁱ, θ = (x, y, z) -> 300.0, u = uᵛⁱ, v = vᵛⁱ, w = wᵛⁱ)
        update_state!(model)
        P, P_exact = vi_energy_production(model)
        abs(P - P_exact) / abs(P_exact)
    end

    # Converging, with an absolute floor: WENO nonlinear weights make the
    # residual non-monotone once it is already ≲ 10⁻³ of the production.
    @test audits[1] < 0.2
    @test audits[2] < max(audits[1] / 1.8, 0.005)
end

#####
##### Full-divergence test (design note): ∇ₕ·(ρ𝐮ₕ) = 0 but ∂z(ρw) ≠ 0, with
##### uniform u. The exact x-flux divergence is u ∂z(ρw), carried entirely by
##### the vertical part of the mass-flux divergence — this catches accidental
##### reuse of horizontal-only divergence logic in the 3D flavor.
#####

@testset "3D flavor carries the vertical mass-flux divergence [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = vi_exactness_grid(FT, 16)

    U₀ = 10
    ρ₀(x, y, z) = 1 + 0.2 * cospi(z / Lᵛⁱ)
    w₀(x, y, z) = 0.5 * sinpi(z / Lᵛⁱ)
    ρw(x, y, z) = ρ₀(x, y, z) * w₀(x, y, z)
    exact_x(x, y, z) = U₀ * ∂zᵛⁱ(ρw, x, y, z)

    for make in (() -> CompressibleVectorInvariant(; divergence = ThreeDimensionalDivergence()),
                 () -> CompressibleWENOVectorInvariant(; divergence = ThreeDimensionalDivergence()))
        model = vi_model(grid, make())
        set!(model; ρ = ρ₀, θ = (x, y, z) -> 300.0, u = (x, y, z) -> U₀, w = w₀)
        update_state!(model)
        adv, mom, vel, dyn = model.advection.momentum, model.momentum, model.velocities, model.dynamics
        Ax = Field(KernelFunctionOperation{Face, Center, Center}(x_momentum_flux_divergence, grid, adv, mom, vel, dyn))
        ex = Field{Face, Center, Center}(grid); set!(ex, exact_x)
        Nz = size(grid, 3)
        r = 4:Nz-3
        scale = maximum(abs.(interior(ex)))
        @test scale > 0   # the divergence term must not vanish
        @test maximum(abs.(interior(Ax)[:, :, r] .- interior(ex)[:, :, r])) / scale < 0.1
    end
end
