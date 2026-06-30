#####
##### Conservation tests for the curvilinear (Christoffel) metric terms in
##### compressible momentum advection.
#####
##### On a horizontally-curvilinear grid (e.g. LatitudeLongitudeGrid) flux-form
##### momentum advection ∇·(ρ𝐮⊗u) is missing the basis-vector-rotation
##### contribution, which Oceananigans supplies separately as the curvature
##### metric terms `U_dot_∇{u,v,w}_metric` (hydrostatic `uv·tanφ/a` etc. plus the
##### deep-atmosphere `uw/a`, `vw/a`, `(u²+v²)/a` w-coupling). For the
##### vector-invariant schemes the hydrostatic part is folded into the vorticity
##### flux, so only the nonhydrostatic part is added — see
##### `src/AtmosphereModels/vector_invariant_advection.jl` and
##### Oceananigans `src/Advection/curvature_metric_terms.jl`.
#####
##### The curvature force is *fictitious*: it does zero work pointwise in the
##### continuum (u·Gᵤ + v·Gᵥ + w·G_w = 0). These tests verify the discretization
##### inherits that property and introduces no spurious energy source:
#####
#####   1. On a RectilinearGrid the metric terms are identically zero.
#####   2. With constant density the curvature does exactly zero net work (to
#####      machine precision), for every momentum advection scheme.
#####   3. With variable density the (truncation-level) work converges to zero at
#####      second order under refinement — consistent, not a spurious source.
#####
##### All three are checked for flux-form (Centered, WENO) and vector-invariant
##### (`CompressibleVectorInvariant`, `CompressibleWENOVectorInvariant`) momentum
##### advection.

using Breeze
using Oceananigans
using Oceananigans: CenterField, XFaceField, YFaceField, ZFaceField, Field,
                    interior, compute!, set!, fill_halo_regions!
using Oceananigans.AbstractOperations: KernelFunctionOperation, @at
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ
using Oceananigans.Advection: Centered, WENO, materialize_advection,
                              U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric
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

const φ₀ = 80   # latitude half-width of the test grid

# Flow that vanishes at the meridional (φ = ±φ₀) and vertical (z = 0, Lz)
# boundaries so the discrete energy identity is free of boundary-flux terms.
# Longitude is periodic, so no vanishing is required there.
gφ(φ) = 1 - (φ / φ₀)^2
uf(λ, φ, z, Lz) = gφ(φ) * (8 + 3 * cosd(2λ))
vf(λ, φ, z, Lz) = gφ(φ) * (5 + 2 * sind(3λ))
wf(λ, φ, z, Lz) = gφ(φ) * sinpi(z / Lz) * (0.5 + 0.3 * cosd(λ))

# Build velocity + coupled-momentum (ρ𝐮) fields. Momentum is interpolated as the
# model holds it: ρu = ℑ(ρ) u, etc.
function metric_test_fields(grid, ρfunc)
    Lz = grid.Lz
    u = XFaceField(grid); v = YFaceField(grid); w = ZFaceField(grid)
    ρ = CenterField(grid); set!(ρ, ρfunc); fill_halo_regions!(ρ)
    set!(u, (λ, φ, z) -> uf(λ, φ, z, Lz))
    set!(v, (λ, φ, z) -> vf(λ, φ, z, Lz))
    set!(w, (λ, φ, z) -> wf(λ, φ, z, Lz))
    fill_halo_regions!((u, v, w))
    ρu = XFaceField(grid); ρv = YFaceField(grid); ρw = ZFaceField(grid)
    set!(ρu, compute!(Field(@at (Face, Center, Center) ρ * u)))
    set!(ρv, compute!(Field(@at (Center, Face, Center) ρ * v)))
    set!(ρw, compute!(Field(@at (Center, Center, Face) ρ * w)))
    fill_halo_regions!((ρu, ρv, ρw))
    return (; ρu, ρv, ρw), (; u, v, w)
end

function metric_fields(grid, madv, momentum, velocities)
    Mu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(U_dot_∇u_metric, grid, madv, momentum, velocities)))
    Mv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(U_dot_∇v_metric, grid, madv, momentum, velocities)))
    Mw = compute!(Field(KernelFunctionOperation{Center, Center, Face}(U_dot_∇w_metric, grid, madv, momentum, velocities)))
    return Mu, Mv, Mw
end

# Discrete kinetic-energy work of the curvature force, normalized by the sum of
# the absolute per-cell contributions: Σ u·Gᵤ·V over the three momentum points.
function relative_curvature_work(grid, madv, momentum, velocities)
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

make_llg(arch, FT; Nx, Ny, Nz, Lz = 3.0e4) =
    LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz), halo = (6, 6, 6),
                          longitude = (0, 360), latitude = (-φ₀, φ₀), z = (0, Lz),
                          topology = (Periodic, Bounded, Bounded))

# All momentum advection schemes that carry curvature metric terms.
metric_schemes() = (
    ("Centered (flux)",     Centered()),
    ("WENO (flux)",         WENO()),
    ("VI horizontal",       CompressibleVectorInvariant(; divergence = HorizontalDivergence())),
    ("VI 3D",               CompressibleVectorInvariant(; divergence = ThreeDimensionalDivergence())),
    ("WENO-VI horizontal",  CompressibleWENOVectorInvariant(; divergence = HorizontalDivergence())),
    ("WENO-VI 3D",          CompressibleWENOVectorInvariant(; divergence = ThreeDimensionalDivergence())),
)

#####
##### 1. Metric terms vanish identically on a RectilinearGrid
#####

@testset "Curvature metric terms vanish on RectilinearGrid [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = (16, 16, 8), halo = (6, 6, 6),
                           x = (0, 1e5), y = (0, 1e5), z = (0, 3e4),
                           topology = (Periodic, Periodic, Bounded))
    momentum, velocities = metric_test_fields(grid, (x...) -> 1.2)

    for (name, scheme) in metric_schemes()
        madv = metric_advection(materialize_advection(scheme, grid))
        Mu, Mv, Mw = metric_fields(grid, madv, momentum, velocities)
        @test all(interior(Mu) .== 0)
        @test all(interior(Mv) .== 0)
        @test all(interior(Mw) .== 0)
    end
end

#####
##### 2. Curvature does exactly zero net work with constant density
#####

@testset "Curvature does zero net work, constant ρ [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = make_llg(default_arch, FT; Nx = 48, Ny = 44, Nz = 12)
    momentum, velocities = metric_test_fields(grid, (x...) -> 1.2)

    # Machine-precision identity for the energy-conserving curvature discretization.
    tol = FT == Float64 ? 1e-11 : 1e-4
    for (name, scheme) in metric_schemes()
        madv = metric_advection(materialize_advection(scheme, grid))
        W = relative_curvature_work(grid, madv, momentum, velocities)
        @test W < tol
    end
end

#####
##### 3. Curvature work converges at 2nd order with variable density
#####

@testset "Curvature work converges under refinement, variable ρ [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    ρvar(λ, φ, z) = 1.2 * exp(-z / 8000) * (1 + 0.1 * cosd(φ))

    coarse = make_llg(default_arch, FT; Nx = 48, Ny = 32, Nz = 8)
    fine   = make_llg(default_arch, FT; Nx = 96, Ny = 64, Nz = 8)

    for (name, scheme) in metric_schemes()
        mom_c, vel_c = metric_test_fields(coarse, ρvar)
        mom_f, vel_f = metric_test_fields(fine, ρvar)
        Wc = relative_curvature_work(coarse, metric_advection(materialize_advection(scheme, coarse)), mom_c, vel_c)
        Wf = relative_curvature_work(fine,   metric_advection(materialize_advection(scheme, fine)),   mom_f, vel_f)

        # The work is a truncation error, not a spurious source: small and
        # shrinking faster than first order (2nd order ⇒ ~4× per halving).
        @test Wc < 1e-2
        @test Wf < Wc / 3
    end
end
