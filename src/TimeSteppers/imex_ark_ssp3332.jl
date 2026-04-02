#####
##### IMEX Additive Runge-Kutta SSP3(3,3,2)
#####
##### Pareschi and Russo (2005), Table 5. 4 stages (3 implicit + final
##### combination), combined order 2. Constant SDIRK diagonal γ = 1-1/√2.
##### NOT stiffly accurate — requires final combination step.
#####
##### γ = 1 - 1/√2 ≈ 0.2929
#####
##### Explicit:                         Implicit:
#####   0     |                           γ     | γ
#####   1     | 1                         1-γ   | 1-2γ    γ
#####   1/2   | 1/4    1/4                1/2   | 1/2-γ   0      γ
#####         | 1/6    1/6    2/3               | 1/6     1/6    2/3
#####
##### Key property: |R(iω)| ≤ 1 for ALL ω — damps oscillatory modes,
##### never amplifies. This makes it suitable for acoustic dynamics in fᴵ.
#####

using Oceananigans: CenterField, Field, Center, Face
using Oceananigans.TimeSteppers: AbstractTimeStepper

const SSP3332_γ = 1 - 1 / sqrt(2)  # ≈ 0.29289321881345254

"""
$(TYPEDEF)

IMEX-ARK time stepper using the SSP3(3,3,2) method from
[Pareschi and Russo (2005)](@cite PareschiRusso2005).

Second-order, L-stable. Damps oscillatory implicit modes (|R(iω)| ≤ 1).
NOT stiffly accurate — uses a final combination step.

Fields
======

- `U⁰`: State at the beginning of the time step
- `Gⁿ`: Current tendency storage
- `fE`: Explicit tendencies at each of the 3 implicit stages
- `fI`: Implicit tendencies (ρθ, ρw, ρ) at each stage
- `ρw_scratch`: Scratch field for saving ρw before implicit solve
- `implicit_solver`: Optional implicit diffusion solver
"""
struct IMEXRungeKuttaSSP3332{U0, TG, FE, FI, RS, TI} <: AbstractTimeStepper
    U⁰              :: U0
    Gⁿ              :: TG
    fE               :: FE
    fI               :: FI
    ρw_scratch       :: RS
    implicit_solver  :: TI
end

function IMEXRungeKuttaSSP3332(grid, prognostic_fields;
                                dynamics = nothing,
                                implicit_solver = nothing,
                                Gⁿ = map(similar, prognostic_fields))
    U⁰ = map(similar, prognostic_fields)
    fE = ntuple(i -> map(similar, prognostic_fields), 3)
    fI = ntuple(i -> (CenterField(grid), Field{Center, Center, Face}(grid), CenterField(grid)), 3)
    ρw_scratch = Field{Center, Center, Face}(grid)

    return IMEXRungeKuttaSSP3332(U⁰, Gⁿ, fE, fI, ρw_scratch, implicit_solver)
end

#####
##### Time stepping
#####

function OceananigansTimeSteppers.time_step!(model::AtmosphereModel{<:Any, <:Any, <:Any, <:IMEXRungeKuttaSSP3332}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    maybe_prepare_first_time_step!(model, callbacks)

    ts = model.timestepper
    h = Δt
    γ = SSP3332_γ

    ## Butcher tableaux (Pareschi-Russo 2005, Table 5)
    aE = ((0, 0, 0),
          (1, 0, 0),
          (1//4, 1//4, 0))

    aI = ((γ, 0, 0),
          (1 - 2γ, γ, 0),
          (1//2 - γ, 0, γ))

    bE = (1//6, 1//6, 2//3)
    bI = (1//6, 1//6, 2//3)

    tⁿ⁺¹ = model.clock.time + Δt

    ## Save yₙ
    for (u⁰, u) in zip(ts.U⁰, prognostic_fields(model))
        parent(u⁰) .= parent(u)
    end

    ## ─── Stage 1 (γ₁ = γ, implicit solve at initial state) ───
    ## z₁* = yₙ  (no previous stages)
    ## z₁ = z₁* + γh fᴵ(z₁)

    ## No predictor needed — state is yₙ
    ## But we need update_state to have valid T, p for the Helmholtz
    ## (already done by maybe_prepare_first_time_step)

    parent(ts.ρw_scratch) .= parent(model.momentum.ρw)
    imex_acoustic_solve!(model, γ * h)
    store_implicit_tendency!(ts.fI[1][1], ts.fI[1][2], ts.fI[1][3], model, γ * h, ts.ρw_scratch)

    update_state!(model, callbacks; compute_tendencies=true)
    compute_flux_bc_tendencies!(model)
    for (fE_field, G) in zip(ts.fE[1], ts.Gⁿ)
        parent(fE_field) .= parent(G)
    end
    remove_vertical_ρθ_flux_from_explicit!(ts.fE[1], ts.fI[1][1], model.grid)

    ## ─── Stage 2 (γ₂ = γ) ───
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
    remove_vertical_ρθ_flux_from_explicit!(ts.fE[2], ts.fI[2][1], model.grid)

    ## ─── Stage 3 (γ₃ = γ) ───
    set_stage_predictor!(model, ts.U⁰, h, ts.fE, ts.fI, aE[3], aI[3], 2)
    zero_boundary_ρw!(model)
    update_state!(model, callbacks; compute_tendencies=false)

    parent(ts.ρw_scratch) .= parent(model.momentum.ρw)
    imex_acoustic_solve!(model, γ * h)
    store_implicit_tendency!(ts.fI[3][1], ts.fI[3][2], ts.fI[3][3], model, γ * h, ts.ρw_scratch)

    update_state!(model, callbacks; compute_tendencies=true)
    compute_flux_bc_tendencies!(model)
    for (fE_field, G) in zip(ts.fE[3], ts.Gⁿ)
        parent(fE_field) .= parent(G)
    end
    remove_vertical_ρθ_flux_from_explicit!(ts.fE[3], ts.fI[3][1], model.grid)

    ## ─── Final combination ───
    ## y_{n+1} = yₙ + h [1/6(fᴱ₁+fᴵ₁) + 1/6(fᴱ₂+fᴵ₂) + 2/3(fᴱ₃+fᴵ₃)]

    apply_final_combination!(model, ts.U⁰, h, ts.fE, ts.fI, bE, bI, 3)

    corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_Δt, Δt)
    update_state!(model, callbacks; compute_tendencies=true)

    return nothing
end
