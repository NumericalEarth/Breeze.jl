using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticleTracking

#####
##### Advancing Lagrangian particles with the time-stepping schemes of Breeze
#####
##### Both SSP RK3 and Wicker-Skamarock RK3 combine every stage with the state at the beginning of
##### the time step. Particle positions X are therefore advanced in terms of the position Xⁿ at the
##### beginning of the time step and the "unwrapped" displacement D from Xⁿ,
#####
#####     Dᵏ = aₖ Dᵏ⁻¹ + bₖ Δt U(X⁽ᵏ⁻¹⁾),    X⁽ᵏ⁾ = Xⁿ + Dᵏ,    D⁰ = 0,
#####
##### where U(X⁽ᵏ⁻¹⁾) is the particle velocity at the beginning of stage k. With aₖ = bₖ = αₖ,
##### this is the Shu-Osher form of SSP RK3, X⁽ᵏ⁾ = (1 - αₖ) Xⁿ + αₖ (X⁽ᵏ⁻¹⁾ + Δt U(X⁽ᵏ⁻¹⁾)).
##### With aₖ = 0 and bₖ = βₖ, this is Wicker-Skamarock RK3, X⁽ᵏ⁾ = Xⁿ + βₖ Δt U(X⁽ᵏ⁻¹⁾).
##### Boundary conditions are applied to X⁽ᵏ⁾ after every stage, but never to Xⁿ + D, so that
##### particles that cross a periodic boundary within a time step are advanced consistently.
#####

"""
    DisplacementParticleStage{FT}

Describes stage `k` of a time-stepping scheme that advances Lagrangian particle positions
with the displacement `Dᵏ = a Dᵏ⁻¹ + b Δt U` from the position at the beginning of the time step.
`first` is `true` for the first stage, which stores the initial position and starts from zero displacement.
"""
struct DisplacementParticleStage{FT}
        a :: FT
        b :: FT
    first :: Bool
end

initial_position_and_displacement(x) = (x⁰ = similar(x, 0), y⁰ = similar(x, 0), z⁰ = similar(x, 0),
                                        Dx = similar(x, 0), Dy = similar(x, 0), Dz = similar(x, 0))

const DisplacementTimeStepper = Union{SSPRungeKutta3, AcousticRungeKutta3}

LagrangianParticleTracking.particle_time_stepper_state(::DisplacementTimeStepper, x) = initial_position_and_displacement(x)
LagrangianParticleTracking.particle_stages(::DisplacementTimeStepper) = 3

function LagrangianParticleTracking.particle_stage(ts::SSPRungeKutta3, stage, restart)
    α = ifelse(stage == 1, ts.α¹, ifelse(stage == 2, ts.α², ts.α³))
    return DisplacementParticleStage(α, α, stage == 1)
end

function LagrangianParticleTracking.particle_stage(ts::AcousticRungeKutta3, stage, restart)
    β = stage_fraction(ts, stage)
    return DisplacementParticleStage(zero(β), β, stage == 1)
end

@inline function LagrangianParticleTracking.advance_particle_position(stage::DisplacementParticleStage,
                                                                     (x, y, z), (ẋ, ẏ, ż), Δt, state, p)
    first = stage.first

    # The first stage starts from the current position with zero displacement.
    # Stored values are not used at the first stage, and may not be initialized.
    @inbounds begin
        x⁰ = ifelse(first, x, state.x⁰[p])
        y⁰ = ifelse(first, y, state.y⁰[p])
        z⁰ = ifelse(first, z, state.z⁰[p])
        Dx = ifelse(first, zero(x), state.Dx[p])
        Dy = ifelse(first, zero(y), state.Dy[p])
        Dz = ifelse(first, zero(z), state.Dz[p])
    end

    a = stage.a
    b = stage.b
    Dx = a * Dx + b * Δt * ẋ
    Dy = a * Dy + b * Δt * ẏ
    Dz = a * Dz + b * Δt * ż

    @inbounds begin
        state.x⁰[p] = x⁰
        state.y⁰[p] = y⁰
        state.z⁰[p] = z⁰
        state.Dx[p] = Dx
        state.Dy[p] = Dy
        state.Dz[p] = Dz
    end

    return (x⁰ + Dx, y⁰ + Dy, z⁰ + Dz)
end
