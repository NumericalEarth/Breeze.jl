"""
$(TYPEDSIGNATURES)

Project a `CompressibleModel`'s initial condition onto the discrete slow
manifold via FV3-SHiELD-style adiabatic initialization (`na_init`), damping
spurious fast-mode (acoustic / Lamb / inertia–gravity) noise and spinning up a
balanced vertical momentum `ρw` before production integration.

For each of `cycles` iterations the routine runs a symmetric forward/backward
dynamics excursion and then nudges the *slow* prognostic fields
`(ρ, ρu, ρv, ρθ, ρqᵉ)` back toward their `t = 0` snapshot by the weighted mean

    x ← (x + weight·x₀) / (1 + weight)

(default `weight = 2` → ⅓ dynamics + ⅔ snapshot). The vertical momentum `ρw`
is never snapshotted or nudged, so the balance the excursion imprints on it
survives. `update_state!` after each nudge rebuilds the diagnostic pressure and
density-consistent fields. The clock is reset to `t = 0` on exit.

`Δt` is the forward/backward step size (typically the acoustic-CFL timestep);
the backward step requires the split-explicit acoustic substepper, which
`CompressibleDynamics` uses by default.

`adiabatic_initialization!` performs *adiabatic* dynamics only. The caller must
pass a model built without physics (`microphysics = nothing`), without an upper
sponge (`sponge = nothing`), and without forcing — these run inside
`update_state!`/`time_step!` and would corrupt the projection. Boundary
conditions are not modified; pass a model whose boundaries are time-invariant
(e.g. frozen at the analysis time) so the symmetric excursion stays reversible.
"""
function adiabatic_initialization!(model::CompressibleModel; Δt, cycles = 1, weight = 2)
    snapshot = snapshot_slow_fields(model)

    for _ in 1:cycles
        # Half-cycle A: 0 → +Δt → 0, then nudge.
        time_step!(model, +Δt)
        time_step!(model, -Δt)
        nudge_slow_fields!(model, snapshot, weight)
        update_state!(model)

        # Half-cycle B: 0 → -Δt → 0, then nudge.
        time_step!(model, -Δt)
        time_step!(model, +Δt)
        nudge_slow_fields!(model, snapshot, weight)
        update_state!(model)
    end

    # Production integration begins from t = 0.
    model.clock.time = zero(model.clock.time)
    model.clock.iteration = 0

    return model
end

# The slow prognostic fields nudged toward the IC. `ρw` is deliberately
# excluded — it is left free to develop balance with the nudged slow fields.
slow_fields(model::CompressibleModel) =
    (dynamics_density(model.dynamics),          # ρ
     model.momentum.ρu,
     model.momentum.ρv,
     thermodynamic_density(model.formulation),  # ρθ
     model.moisture_density)                    # ρqᵉ

# Copy the slow prognostics' full (haloed) parent arrays at t = 0.
snapshot_slow_fields(model::CompressibleModel) =
    map(f -> copy(parent(f)), slow_fields(model))

# In-place weighted blend of each slow field toward its snapshot:
#   x ← (x + weight·x₀) / (1 + weight)
function nudge_slow_fields!(model::CompressibleModel, snapshot, weight)
    w = convert(eltype(model.grid), weight)
    for (f, x₀) in zip(slow_fields(model), snapshot)
        p = parent(f)
        @. p = (p + w * x₀) / (1 + w)
        fill_halo_regions!(f)
    end
    return nothing
end
