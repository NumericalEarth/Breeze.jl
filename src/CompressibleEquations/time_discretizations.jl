#####
##### Time discretization types for CompressibleDynamics
#####
##### These types determine how the compressible equations are time-stepped:
##### - SplitExplicit: Acoustic substepping (Wicker-Skamarock scheme)
##### - ExplicitTimeStepping: Standard explicit time-stepping (small Δt required)
#####

"""
$(TYPEDEF)

Split-explicit time discretization for compressible dynamics.

Uses acoustic substepping following [Wicker and Skamarock (2002)](@cite WickerSkamarock2002):
- Outer loop: SSP RK3 for slow tendencies (advection, Coriolis, diffusion)
- Inner loop: Acoustic substeps for fast tendencies (pressure gradient, buoyancy)

This allows using advective CFL time steps (~10-20 m/s) instead of acoustic CFL
time steps (~340 m/s), typically enabling ~6× larger time steps.

See also [`ExplicitTimeStepping`](@ref).
"""
struct SplitExplicit end

"""
$(TYPEDEF)

Standard explicit time discretization for compressible dynamics.

All tendencies (including pressure gradient and acoustic modes) are computed
together and time-stepped explicitly. This requires small time steps limited
by the acoustic CFL condition (sound speed ~340 m/s).

Use [`SplitExplicit`](@ref) for more efficient time-stepping with larger Δt.
"""
struct ExplicitTimeStepping end
