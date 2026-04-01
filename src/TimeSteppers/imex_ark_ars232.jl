#####
##### IMEX Additive Runge-Kutta ARS(2,3,2)
#####
##### [Ascher, Ruuth, Spiteri (1997)](@cite AscherRuuthSpiteri1997).
##### 3 stages, combined order 2. L-stable SDIRK with constant diagonal γ.
#####
##### KNOWN ISSUE: The current HEVI splitting zeros the explicit vertical
##### PGF+buoyancy but keeps vertical ρθ advection in the explicit tendency.
##### This creates an inconsistency: ρθ is advected vertically by a ρw that
##### has no vertical restoring force. The resulting spurious ρθ perturbation
##### corrupts the pressure field and drives exponential growth of horizontal
##### winds (observed even from a balanced state with zero initial velocity).
#####
##### FIX NEEDED: Also exclude vertical ρθ advection from fᴱ and include
##### it in fᴵ (Gardner et al. 2018, splitting A). The implicit ρθ flux
##### (-w ∂ρθ/∂z) should use the same centered stencil as the Helmholtz RHS.
##### The difference (WENO - centered) stays in fᴱ as a small correction.
#####
##### Stiffly accurate: y_{n+1} = z₃ (no final combination needed).
#####
##### γ = 1 - √2/2 ≈ 0.2929
##### δ = 1/(2(1-γ)) = 1/√2 ≈ 0.7071
#####
##### Explicit tableau:                 Implicit tableau:
#####   0   |                             0   |
#####   2γ  | 2γ                          2γ  | γ     γ
#####   1   | 1-γ    γ                    1   | δ     0     γ
#####       | 1-γ    γ       0                | δ     0     γ
#####
##### Both tableaux are stiffly accurate (last row of a = b).
#####

using Oceananigans: CenterField, Field, Center, Face
using Oceananigans.TimeSteppers: AbstractTimeStepper

const ARS232_γ = 1 - sqrt(2) / 2              # ≈ 0.29289321881345254
const ARS232_δ = 1 / (2 * (1 - ARS232_γ))     # = 1/√2 ≈ 0.70710678118654752

"""
$(TYPEDEF)

IMEX-ARK time stepper using the ARS(2,3,2) method from
[Ascher, Ruuth, and Spiteri (1997)](@cite AscherRuuthSpiteri1997).

Second-order, L-stable, stiffly accurate. The implicit part uses an
SDIRK scheme with constant diagonal γ = 1 - √2/2, which provides
unconditional stability for the vertical acoustic mode.

Fields
======

- `U⁰`: State at the beginning of the time step
- `Gⁿ`: Current tendency storage (reused each stage)
- `fE`: Explicit tendencies at each stage
- `fI`: Implicit tendencies (ρθ, ρw, ρ) at each stage
- `ρw_scratch`: Scratch field for saving ρw before the implicit solve
- `implicit_solver`: Optional implicit diffusion solver
"""
struct IMEXRungeKuttaARS232{U0, TG, FE, FI, RS, TI} <: AbstractTimeStepper
    U⁰              :: U0
    Gⁿ              :: TG
    fE               :: FE
    fI               :: FI
    ρw_scratch       :: RS
    implicit_solver  :: TI
end

function IMEXRungeKuttaARS232(grid, prognostic_fields;
                               dynamics = nothing,
                               implicit_solver = nothing,
                               Gⁿ = map(similar, prognostic_fields))
    U⁰ = map(similar, prognostic_fields)
    fE = ntuple(i -> map(similar, prognostic_fields), 3)
    fI = ntuple(i -> (CenterField(grid), Field{Center, Center, Face}(grid), CenterField(grid)), 3)
    ρw_scratch = Field{Center, Center, Face}(grid)

    return IMEXRungeKuttaARS232(U⁰, Gⁿ, fE, fI, ρw_scratch, implicit_solver)
end

#####
##### Time stepping
#####

function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:IMEXRungeKuttaARS232}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    maybe_prepare_first_time_step!(model, callbacks)

    ts = model.timestepper
    h = Δt
    γ = ARS232_γ
    δ = ARS232_δ

    ## ARS(2,3,2) Butcher tableaux
    aE = ((0, 0, 0),
          (2γ, 0, 0),
          (1 - γ, γ, 0))

    aI = ((0, 0, 0),
          (γ, γ, 0),
          (δ, 0, γ))

    tⁿ⁺¹ = model.clock.time + Δt

    ## Save yₙ
    for (u⁰, u) in zip(ts.U⁰, prognostic_fields(model))
        parent(u⁰) .= parent(u)
    end

    ## ─── Stage 1 (γ₁ = 0, fully explicit) ───
    ## z₁ = yₙ.  Compute fᴱ(z₁); fᴵ(z₁) = 0.

    compute_flux_bc_tendencies!(model)
    for (fE_field, G) in zip(ts.fE[1], ts.Gⁿ)
        parent(fE_field) .= parent(G)
    end

    ## ─── Stage 2 (γ₂ = γ) ───
    ## z*₂ = yₙ + h [2γ fᴱ₁ + γ fᴵ₁] = yₙ + 2γh fᴱ₁

    set_stage_predictor!(model, ts.U⁰, h, ts.fE, ts.fI, aE[2], aI[2], 1)
    zero_boundary_ρw!(model)
    update_state!(model, callbacks; compute_tendencies=false)

    parent(ts.ρw_scratch) .= parent(model.momentum.ρw)
    imex_acoustic_solve!(model, γ * h)
    store_implicit_tendency!(ts.fI[2][1], ts.fI[2][2], ts.fI[2][3], model, γ * h, ts.ρw_scratch)

    update_state!(model, callbacks; compute_tendencies=true)
    compute_flux_bc_tendencies!(model)
    for (fE_field, G) in zip(ts.fE[2], ts.Gⁿ)
        parent(fE_field) .= parent(G)
    end

    ## ─── Stage 3 (γ₃ = γ, stiffly accurate: y_{n+1} = z₃) ───
    ## z*₃ = yₙ + h [(1-γ)fᴱ₁ + γ fᴱ₂ + δ fᴵ₁ + 0·fᴵ₂]
    ##      = yₙ + h [(1-γ)fᴱ₁ + γ fᴱ₂]     (fᴵ₁ = 0, aᴵ₃₂ = 0)

    set_stage_predictor!(model, ts.U⁰, h, ts.fE, ts.fI, aE[3], aI[3], 2)
    zero_boundary_ρw!(model)
    update_state!(model, callbacks; compute_tendencies=false)

    parent(ts.ρw_scratch) .= parent(model.momentum.ρw)
    imex_acoustic_solve!(model, γ * h)

    ## z₃ IS y_{n+1} (stiffly accurate)
    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)
    update_state!(model, callbacks; compute_tendencies=true)

    return nothing
end
