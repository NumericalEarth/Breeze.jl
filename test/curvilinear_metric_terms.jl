#####
##### Conservation tests for the curvilinear (Christoffel) metric terms in
##### compressible momentum advection — exercised over the WHOLE domain,
##### including the boundary treatment (no interior isolation).
#####
##### On a horizontally-curvilinear grid (e.g. LatitudeLongitudeGrid) flux-form
##### momentum advection ∇·(ρ𝐮⊗u) is missing the basis-vector-rotation
##### contribution, supplied separately as the curvature metric terms
##### `U_dot_∇{u,v,w}_metric`: the hydrostatic `uv·tanφ/a` etc. (metric-ratio
##### form) plus the deep-atmosphere `uw/a`, `vw/a`, `(u²+v²)/a` w-coupling. For
##### the vector-invariant schemes the hydrostatic part is folded into the
##### vorticity flux, so only the nonhydrostatic part is added — see
##### `src/AtmosphereModels/vector_invariant_advection.jl` and Oceananigans
##### `src/Advection/curvature_metric_terms.jl`.
#####
##### The curvature force is *fictitious*: it does zero work pointwise in the
##### continuum (u·Gᵤ + v·Gᵥ + w·G_w = 0). Over a closed/impenetrable domain the
##### discrete net work must therefore vanish up to truncation. The wall-normal
##### momentum components carry `Open` (impenetrable) boundary conditions, so a
##### physical state has zero normal velocity at every wall and no boundary
##### energy flux — these tests use a real model so that boundary treatment is
##### genuine, and integrate over the full domain.
#####
#####   1. On a RectilinearGrid the metric terms are identically zero.
#####   2. The model enforces impenetrability (zero normal velocity at all walls).
#####   3. The vector-invariant (nonhydrostatic) curvature metric does exactly
#####      zero net work over the whole domain (machine precision).
#####   4. The flux-form (hydrostatic + nonhydrostatic) metric does no work only
#####      to truncation; the energy error converges at second order.

using Breeze
using Breeze.CompressibleEquations: CompressibleDynamics, ExplicitTimeStepping
using Oceananigans
using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, Field,
                    interior, compute!, set!, fill_halo_regions!
using Oceananigans.AbstractOperations: KernelFunctionOperation, @at
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ
using Oceananigans.Advection: Centered, WENO, materialize_advection,
                              U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric
using Oceananigans.TimeSteppers: update_state!
using Test

# Standalone-run fallbacks (defined in the test runner's init_code otherwise).
if !@isdefined(test_float_types)
    test_float_types() = (Float64,)
end
if !@isdefined(default_arch)
    using Oceananigans.Architectures: CPU
    default_arch = CPU()
end

#####
##### Helpers
#####

# The metric term added by the momentum path: vector-invariant wrappers strip the
# hydrostatic part by dispatching on the inner `scheme`; flux-form schemes use the
# full metric. Mirror exactly what `*_momentum_flux_divergence` passes.
metric_advection(a::CompressibleVectorInvariant) = a.scheme
metric_advection(a) = a

# Generic flow with no special boundary tapering. The model's impenetrable
# (Open) boundary conditions zero the wall-normal components for us.
uᵢ(λ, φ, z) = 8 + 3 * cosd(2λ) + 2 * sind(φ)
vᵢ(λ, φ, z) = 5 + 2 * sind(3λ) + cosd(2φ)
wᵢ(λ, φ, z) = 0.5 + 0.3 * cosd(λ) + 0.2 * sind(2φ)

# Build a compressible model on a LatitudeLongitudeGrid, set a generic flow and
# the hydrostatic reference density, and update_state! so halos/auxiliary fields
# (and impenetrability at the walls) are enforced. Lz is kept modest so the
# constant-θ Exner reference state stays positive.
function metric_test_model(arch, FT; Nx = 48, Ny = 44, Nz = 8)
    grid = LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz), halo = (6, 6, 6),
                                 longitude = (0, 360), latitude = (-80, 80), z = (0, 1e4),
                                 topology = (Periodic, Bounded, Bounded))
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = FT(300),
                                    surface_pressure = FT(1e5))
    model = AtmosphereModel(grid; dynamics, coriolis = SphericalCoriolis(),
                            momentum_advection = WENO())
    set!(model; θ = 300, ρ = model.dynamics.reference_state.density, u = uᵢ, v = vᵢ, w = wᵢ)
    update_state!(model)
    return model
end

function metric_fields(grid, madv, momentum, velocities)
    Mu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(U_dot_∇u_metric, grid, madv, momentum, velocities)))
    Mv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(U_dot_∇v_metric, grid, madv, momentum, velocities)))
    Mw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(U_dot_∇w_metric, grid, madv, momentum, velocities)))
    return Mu, Mv, Mw
end

# Net kinetic-energy work of the curvature force over the whole domain,
# Σ u·Gᵤ·V + v·Gᵥ·V + w·G_w·V, normalized by the sum of absolute contributions.
function whole_domain_curvature_work(grid, madv, momentum, velocities)
    Mu, Mv, Mw = metric_fields(grid, madv, momentum, velocities)
    Vu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(Vᶠᶜᶜ, grid)))
    Vv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(Vᶜᶠᶜ, grid)))
    Vw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(Vᶜᶜᶠ, grid)))
    u, v, w = velocities
    wu = interior(u) .* interior(Mu) .* interior(Vu)
    wv = interior(v) .* interior(Mv) .* interior(Vv)
    ww = interior(w) .* interior(Mw) .* interior(Vw)
    W = sum(wu) + sum(wv) + sum(ww)
    scale = sum(abs, wu) + sum(abs, wv) + sum(abs, ww)
    return abs(W) / scale
end

# Vector-invariant flavors add only the (energy-conserving) nonhydrostatic metric.
vi_schemes() = (
    ("VI horizontal",       CompressibleVectorInvariant(; divergence = HorizontalDivergence())),
    ("VI 3D",               CompressibleVectorInvariant(; divergence = ThreeDimensionalDivergence())),
    ("WENO-VI horizontal",  CompressibleWENOVectorInvariant(; divergence = HorizontalDivergence())),
    ("WENO-VI 3D",          CompressibleWENOVectorInvariant(; divergence = ThreeDimensionalDivergence())),
)

# Flux-form schemes add the full metric (hydrostatic + nonhydrostatic).
flux_schemes() = (("Centered (flux)", Centered()), ("WENO (flux)", WENO()))

all_schemes() = (vi_schemes()..., flux_schemes()...)

#####
##### 1. Metric terms vanish identically on a RectilinearGrid
#####

@testset "Curvature metric terms vanish on RectilinearGrid [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = (16, 16, 8), halo = (6, 6, 6),
                           x = (0, 1e5), y = (0, 1e5), z = (0, 1e4),
                           topology = (Periodic, Periodic, Bounded))
    u = XFaceField(grid); v = YFaceField(grid); w = ZFaceField(grid)
    set!(u, (x, y, z) -> 7 + sin(x / 3e4)); set!(v, (x, y, z) -> 4 + cos(y / 2e4)); set!(w, (x, y, z) -> z / 1e4)
    fill_halo_regions!((u, v, w))
    velocities = (; u, v, w)
    momentum = (; ρu = u, ρv = v, ρw = w)   # ρ = 1; metric is zero regardless

    for (name, scheme) in all_schemes()
        madv = metric_advection(materialize_advection(scheme, grid))
        Mu, Mv, Mw = metric_fields(grid, madv, momentum, velocities)
        @test all(interior(Mu) .== 0)
        @test all(interior(Mv) .== 0)
        @test all(interior(Mw) .== 0)
    end
end

#####
##### 2. The model enforces impenetrability (no normal flow at the walls) and
##### 3. the vector-invariant curvature does exactly zero net work everywhere.
#####

@testset "Vector-invariant curvature conserves energy over whole domain [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    model = metric_test_model(default_arch, FT)
    grid = model.grid
    u, v, w = model.velocities

    # Impenetrability: wall-normal velocity is zero at every bounded wall.
    @test maximum(abs, interior(w)[:, :, 1])   < 1e-12   # bottom
    @test maximum(abs, interior(w)[:, :, end]) < 1e-12   # top
    @test maximum(abs, interior(v)[:, 1, :])   < 1e-12   # south
    @test maximum(abs, interior(v)[:, end, :]) < 1e-12   # north

    # The nonhydrostatic curvature metric does exactly zero net work over the
    # full domain (energy-conserving discretization), with genuine BCs.
    tol = FT == Float64 ? 1e-10 : 1e-4
    for (name, scheme) in vi_schemes()
        madv = metric_advection(materialize_advection(scheme, grid))
        W = whole_domain_curvature_work(grid, madv, model.momentum, model.velocities)
        @test W < tol
    end
end

#####
##### 4. The flux-form (hydrostatic) metric does no work only to truncation:
#####    the whole-domain energy error converges at second order.
#####

@testset "Flux-form curvature energy error converges (2nd order) [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    coarse = metric_test_model(default_arch, FT; Nx = 48, Ny = 44, Nz = 8)
    fine   = metric_test_model(default_arch, FT; Nx = 96, Ny = 88, Nz = 8)

    for (name, scheme) in flux_schemes()
        Wc = whole_domain_curvature_work(coarse.grid, metric_advection(materialize_advection(scheme, coarse.grid)),
                                         coarse.momentum, coarse.velocities)
        Wf = whole_domain_curvature_work(fine.grid, metric_advection(materialize_advection(scheme, fine.grid)),
                                         fine.momentum, fine.velocities)
        # Small and shrinking faster than first order (2nd order ⇒ ~4× per halving).
        @test Wc < 1e-2
        @test Wf < Wc / 2.5
    end
end
