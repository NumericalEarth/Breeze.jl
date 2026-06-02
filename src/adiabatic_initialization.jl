"""
$(TYPEDSIGNATURES)

Spin up a balanced vertical momentum `ρw` (and the nonhydrostatic pressure
balance) consistent with a `model`'s initial (analysis) state, via FV3-SHiELD-style
adiabatic initialization (`na_init`).

Analyses (ERA5, GFS, …) supply the density, momentum, and thermodynamic state but
cold-start the vertical velocity `w` at zero (hydrostatic), so the vertical /
nonhydrostatic state is out of balance with the rest. For each of `cycles`
iterations the routine runs a symmetric forward/backward dynamics excursion —
which lets `ρw` develop — and then nudges the *initial fields* back toward their
`t = 0` snapshot by the weighted mean

    x ← (x + weight·x₀) / (1 + weight)

(default `weight = 2` → ⅓ dynamics + ⅔ snapshot). The vertical momentum `ρw` is
never snapshotted or nudged, so the balance the excursion imprints on it is
exactly what is kept: with the initial fields held to the analysis and the
vertical field free, `ρw` relaxes into balance with them.

The initial fields depend on the dynamics:

  * `CompressibleDynamics`: `(ρ, ρu, ρv, ρθ, ρqᵉ)`.
  * `AnelasticDynamics`: `(ρu, ρv, ρθ, ρqᵉ)` — density is the fixed anelastic
    reference `ρᵣ(z)`, not prognostic, so it is not nudged. The nudge does not
    re-impose `∇·(ρᵣ u) = 0`; the first production step's pressure solve restores
    the anelastic constraint.

`update_state!` after each nudge rebuilds the diagnostic fields; the clock is
reset to `t = 0` on exit. `Δt` is the forward/backward step size, taken with the
model's own time stepper.

`adiabatic_initialization!` performs *adiabatic* dynamics only. The caller must
pass a model built without physics (`microphysics = nothing`), without an upper
sponge (`sponge = nothing`), and without forcing — these run inside
`update_state!`/`time_step!` and would corrupt the spin-up. Boundary conditions
are not modified; pass a model whose boundaries are time-invariant (e.g. frozen
at the analysis time) so the symmetric excursion stays reversible.
"""
function adiabatic_initialization!(model::AtmosphereModel; Δt, cycles = 1, weight = 2)
    snapshot = snapshot_initial_fields(model)

    for _ in 1:cycles
        # Half-cycle A: 0 → +Δt → 0, then nudge.
        time_step!(model, +Δt)
        time_step!(model, -Δt)
        nudge_initial_fields!(model, snapshot, weight)
        update_state!(model)

        # Half-cycle B: 0 → -Δt → 0, then nudge.
        time_step!(model, -Δt)
        time_step!(model, +Δt)
        nudge_initial_fields!(model, snapshot, weight)
        update_state!(model)
    end

    # Production integration begins from t = 0.
    model.clock.time = zero(model.clock.time)
    model.clock.iteration = 0

    return model
end

# The initial fields, nudged back toward their t = 0 values each cycle, per
# dynamics. `ρw` is deliberately excluded everywhere — it is the free vertical
# field that spins up balance with them.
initial_fields(model::CompressibleModel) =
    (dynamics_density(model.dynamics),          # ρ (prognostic)
     model.momentum.ρu,
     model.momentum.ρv,
     thermodynamic_density(model.formulation),  # ρθ
     model.moisture_density)                    # ρqᵉ

# Anelastic: density is the fixed reference ρᵣ(z), not prognostic, so it is
# omitted from the initial-field set.
initial_fields(model::AnelasticModel) =
    (model.momentum.ρu,
     model.momentum.ρv,
     thermodynamic_density(model.formulation),  # ρθ
     model.moisture_density)                    # ρqᵉ

# Copy the initial fields' full (haloed) parent arrays at t = 0.
snapshot_initial_fields(model::AtmosphereModel) =
    map(f -> copy(parent(f)), initial_fields(model))

# In-place weighted blend of each initial field toward its snapshot:
#   x ← (x + weight·x₀) / (1 + weight)
function nudge_initial_fields!(model::AtmosphereModel, snapshot, weight)
    w = convert(eltype(model.grid), weight)
    for (f, x₀) in zip(initial_fields(model), snapshot)
        p = parent(f)
        @. p = (p + w * x₀) / (1 + w)
        fill_halo_regions!(f)
    end
    return nothing
end
