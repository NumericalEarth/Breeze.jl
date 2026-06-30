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
