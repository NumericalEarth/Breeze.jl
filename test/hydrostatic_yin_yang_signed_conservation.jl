include(joinpath(@__DIR__, "setup.jl"))

#####
##### The conservation fixer of HydrostaticDynamics on a YinYangGrid for scalars that are not
##### non-negative: a negative-only tracer, a tracer of both signs, and a tracer that is zero
#####
##### The fixer returns the seam budget of a scalar in proportion to the part of its density that
##### has the sign of most of its content. An earlier version always weighted with the positive
##### part max(0, ρc), whose integral vanishes for a negative-only scalar: its correction was then
##### zero and the seam budget was not returned. The negative-only tracer here is the
##### discriminator: it is conserved only if the weight is defined for it.
#####

using Breeze
using Oceananigans
using Oceananigans.MultiRegion: YinYangGrid
using Test

include(joinpath(@__DIR__, "hydrostatic_yin_yang_utils.jl"))

# The crossing flow of the conservation tests, with five tracers advected by fifth-order WENO:
# `negative` (non-uniform, negative everywhere), `mixed` (a blob minus a constant: both signs,
# with a mean that is not zero), `balanced` (both signs with almost no net content: its positive
# and negative parts are within 1e-3 of each other), `positive` (the blob, the control) and `zero`
#
# Limits, which these tests do not remove. The correction is GLOBAL: it restores the integral
# over the sphere and nothing else. It does not match the fluxes of the two panels at the seam,
# so content that the seam takes in one place is returned everywhere the scalar has the dominant
# sign, and no budget of a region smaller than the sphere closes. Nothing here is a statement of
# local conservation. For a scalar of both signs only the part of the dominant sign is corrected,
# and which part that is switches where the two parts are equal, as they nearly are for
# `balanced`: its integral is restored, its shape is not protected.
function signed_tracer_model(grid; kw...)
    A, B = hybrid_coefficients(size(grid, 3))
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
    names = (:negative, :mixed, :balanced, :positive, :zero)
    model = hydrostatic_model(grid; dynamics, coriolis=HydrostaticSphericalCoriolis(FT; rotation_rate),
                              tracers=names, scalar_advection=WENO(FT; order=5),
                              thermodynamic_constants=reference_model_constants(), kw...)

    set!(model; pˢ = (λ, φ) -> 1e5 + 500 * cosd(2λ) * cosd(φ)^2,
                θ = (λ, φ, z) -> 290 + 60 * z + 2 * sind(λ) * sind(2φ),
                u = (λ, φ, z) -> 20 * (1 + z) * cosd(φ) + 5 * sind(λ),
                v = (λ, φ, z) -> 5 * z * sind(2λ) * cosd(φ))

    ρ = continuity_density(model.dynamics)
    b = horizontal_field(grid, (λ, φ) -> blob(λ, φ, (-60.0, 20.0)))
    wave = horizontal_field(grid, (λ, φ) -> sind(2λ + 20) * cosd(φ) + 1e-3)
    set!(model; negative = ρ * (-0.2 - b), mixed = ρ * (b - 0.3), balanced = ρ * wave, positive = ρ * b)

    return model
end

tracer_contents(model) = map(total, model.tracers)

# The owned integral of the magnitude of a tracer density: the scale of a content that nearly cancels
magnitude(tracer) = total(Field(abs(tracer)))

#####
##### Tolerances, written before these runs
#####
#
# GATES (asserted), with the fixer, after 20 time steps, for SSP RK3 and both split-explicit
# candidates:
#   1. The owned integrals of the negative, the mixed and the positive tracer are within 1e-12 of
#      their initial values, relative (the bound of the conservation tests of the core).
#   2. The negative tracer is still negative everywhere, and the zero tracer exactly zero.
#   3. The owned integral of the balanced tracer, whose net content is about 1e-3 of the integral
#      of its magnitude, changes by less than 1e-12 of the integral of its magnitude; it still
#      takes both signs.
# DISCRIMINATOR (asserted): without the fixer the owned integral of the negative tracer drifts
# by more than 100 times the bound of gate 1, so a fixer that left it alone would fail gate 1.
# MEASUREMENTS (recorded): the drifts without the fixer.

@testset "Signed tracers [$(scheme.name), fixer $(fixer ? "on" : "off")]" for scheme in schemes, fixer in (false, true)
    resolution = first(resolutions)
    grid = yin_yang_grid(resolution; Nz=6)
    cells = sphere_cells(grid)
    Δt = time_step(scheme, resolution)
    steps = 20

    model = signed_tracer_model(grid; scheme.kw...)
    callbacks = fixer ? conservation_fixer_callbacks(model) : []
    C⁰ = tracer_contents(model)
    balanced_magnitude = magnitude(model.tracers.balanced)
    @test abs(C⁰.balanced) < 1e-2 * balanced_magnitude
    @test C⁰.negative < 0 && C⁰.mixed < 0 && C⁰.positive > 0 && C⁰.zero == 0

    for _ in 1:steps
        time_step!(model, Δt; callbacks)
    end

    C = tracer_contents(model)
    case = "$(resolution.name), $(scheme.name), Δt=$Δt, fixer $(fixer ? "on" : "off")"

    for name in (:negative, :mixed, :positive)
        change = C[name] / C⁰[name] - 1
        record("signed conservation", case, "relative change of the $name tracer over $steps steps", change)
        @test isfinite(change)
        fixer && @test abs(change) < 1e-12                        # gate 1
    end

    fixer || @test abs(C.negative / C⁰.negative - 1) > 1e-10      # discriminator

    # Gate 3: the balanced tracer, against the integral of its magnitude
    balanced_change = (C.balanced - C⁰.balanced) / balanced_magnitude
    record("signed conservation", case, "change of the balanced tracer over the integral of its magnitude", balanced_change)
    record("signed conservation", case, "net content of the balanced tracer over the integral of its magnitude", C⁰.balanced / balanced_magnitude)
    @test isfinite(balanced_change)
    fixer && @test abs(balanced_change) < 1e-12
    @test minimum(minimum, panels(model.tracers.balanced)) < 0 < maximum(maximum, panels(model.tracers.balanced))

    # Gate 2
    @test maximum(maximum, panels(model.tracers.negative)) < 0
    @test all(all(iszero, panel) for panel in panels(model.tracers.zero))
    @test C.zero == 0

    # Both signs are still there
    @test minimum(minimum, panels(model.tracers.mixed)) < 0 < maximum(maximum, panels(model.tracers.mixed))
end
