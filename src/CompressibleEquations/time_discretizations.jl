#####
##### Time discretization types for CompressibleDynamics
#####
##### These types determine how the compressible equations are time-stepped:
##### - SplitExplicitTimeDiscretization: Acoustic substepping (Wicker-Skamarock scheme)
##### - ExplicitTimeStepping: Standard explicit time-stepping (small Δt required)
#####

"""
$(TYPEDEF)

Split-explicit time discretization for compressible dynamics using the
Exner pressure formulation following CM1 (Bryan 2002).

Uses acoustic substepping following [Wicker and Skamarock (2002)](@cite WickerSkamarock2002):
- Outer loop: WS-RK3 for slow tendencies (advection, Coriolis, diffusion)
- Inner loop: Forward-backward acoustic substeps for fast tendencies (pressure gradient)

The acoustic loop uses velocity (u, v, w) and Exner pressure perturbation (π') as
prognostic variables, with a vertically implicit w-π' coupling and CM1-style
divergence damping.

This allows using advective CFL time steps (~10-20 m/s) instead of acoustic CFL
time steps (~340 m/s), typically enabling ~6x larger time steps.

Fields
======

- `substeps`: Number of acoustic substeps for the **full** time step (stage 3 of WS-RK3). For WS-RK3, earlier stages take fewer substeps (``Nτ = \\mathrm{round}(β N)``), keeping ``Δτ = Δt/N`` constant. Default: `nothing` (automatically computed from the acoustic CFL condition each time step)
- `forward_weight`: Off-centering parameter ω for the vertically implicit solver. ω > 0.5 damps vertical acoustic modes. Default: 0.6 (CM1 default)
- `divergence_damping_coefficient`: Forward-extrapolation filter coefficient ``ϰ^{di}`` applied to the Exner pressure perturbation: ``π̃' = π' + ϰ^{di} (π' - π'_{old})``. Default: 0.10 (CM1 default)
- `acoustic_damping_coefficient`: Klemp (2018) divergence damping ``ϰ^{ac}``. Post-implicit-solve velocity correction: ``u -= ϰ^{ac} c_p θ_v ∂Δπ'/∂x``. Provides constant damping per outer Δt regardless of substep count. Needed by WS-RK3 at large Δt. Default: 0.0

See also [`ExplicitTimeStepping`](@ref).
"""
struct SplitExplicitTimeDiscretization{N, FT}
    substeps :: N
    forward_weight :: FT
    divergence_damping_coefficient :: FT
    acoustic_damping_coefficient :: FT
end

function SplitExplicitTimeDiscretization(; substeps=nothing,
                                           forward_weight=0.6,
                                           divergence_damping_coefficient=0.10,
                                           acoustic_damping_coefficient=0.0)
    return SplitExplicitTimeDiscretization(substeps,
                                           forward_weight,
                                           divergence_damping_coefficient,
                                           acoustic_damping_coefficient)
end

"""
$(TYPEDEF)

Standard explicit time discretization for compressible dynamics.

All tendencies (including pressure gradient and acoustic modes) are computed
together and time-stepped explicitly. This requires small time steps limited
by the acoustic CFL condition (sound speed ~340 m/s).

Use [`SplitExplicitTimeDiscretization`](@ref) for more efficient time-stepping with larger Δt.
"""
struct ExplicitTimeStepping end

#####
##### Vertical time discretization (legacy, kept for compatibility)
#####

"""
$(TYPEDEF)

Vertically implicit time discretization for acoustic substeps.

Note: With the Exner pressure formulation, the vertically implicit solver
is always enabled. This type is kept for API compatibility but is no longer
required as a parameter to [`SplitExplicitTimeDiscretization`](@ref).

Fields
======

- `implicit_weight`: Off-centering parameter (0.5 for Crank-Nicolson, typically 0.5-0.6)
"""
struct VerticallyImplicit{FT}
    implicit_weight :: FT
end
