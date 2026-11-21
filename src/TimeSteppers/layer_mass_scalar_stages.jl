#####
##### Mass-weighted scalar stages on a pressure-level grid
#####
#
# On a `PressureLevelGrid` the stages advance a prognostic scalar density ρc through the mass of
# the scalar per unit area, M = (Δp / g) c, and recombine M linearly:
#
#   - when the step starts, the scalars of `U⁰` are stored as M (`store_initial_layer_masses!`);
#   - when a stage starts, pˢ⁻ ← pˢ, and the state of the scalar is M = (Δp⁻ / g) ρc / ρ;
#   - a tendency per unit volume G enters as Δz⁻ G, and transport as the divergence of the face
#     fluxes per unit area (`layer_mass_flux_divergence`);
#   - when the stage ends, c = M / (Δp / g) at the new surface pressure and ρc = ρ c, with the
#     density ρ of the grid that the stage started from: `update_state!` re-diagnoses ρ and
#     carries ρc with it at fixed c.
#
# The conversions (`PressureLevelDiscretization.scalar_layer_mass`, `layer_mass_tendency`,
# `scalar_density_from_layer_mass`) are identities on every other grid, and there the launchers
# below call the kernels of the stages as they were, so that the results are bitwise unchanged.

# The prognostic scalars: every prognostic field but the momentum and the dynamics' own (pˢ)
function layer_mass_scalar_names(model)
    excluded = (prognostic_momentum_field_names(model.dynamics)..., prognostic_dynamics_field_names(model.dynamics)...)
    return filter(name -> name ∉ excluded, keys(prognostic_fields(model)))
end

@kernel function _convert_to_layer_mass!(ρc, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds ρc[i, j, k] = scalar_layer_mass(i, j, k, grid, ρc)
end

@kernel function _convert_tendency_to_layer_mass!(G, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = layer_mass_tendency(i, j, k, grid, G[i, j, k])
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

On a [`PressureLevelGrid`](@ref Breeze.PressureLevelDiscretization.PressureLevelGrid), store
the scalars `names` of the initial state `U⁰` of a time step as their masses per unit area,
`M = (Δp / g) ρc / ρ`, at the surface pressure and density of the start of the step. Nothing on
any other grid.
"""
store_initial_layer_masses!(grid, U⁰, names) = nothing

function store_initial_layer_masses!(grid::PressureLevelGrid, U⁰, names)
    store_previous_surface_pressure!(grid)
    for name in names
        launch!(grid.architecture, grid, :xyz, _convert_to_layer_mass!, U⁰[name], grid)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Convert the tendencies per unit volume `G` of the fields `names` to the units of the stages, `Δz⁻ G`
with the depth of the cell at the start of the stage, on a
[`PressureLevelGrid`](@ref Breeze.PressureLevelDiscretization.PressureLevelGrid). Nothing on any
other grid.
"""
convert_tendencies_to_layer_mass!(grid, G, names) = nothing

function convert_tendencies_to_layer_mass!(grid::PressureLevelGrid, G, names)
    for name in names
        launch!(grid.architecture, grid, :xyz, _convert_tendency_to_layer_mass!, G[name], grid)
    end
    return nothing
end

#####
##### The SSP RK3 stage of a scalar
#####

# M = (1 - α) M⁰ + α [M + Δt Δz⁻ G], then ρc = ρ M / (Δp / g) at the new surface pressure, which
# the stage has already set: the surface pressure comes first among the prognostic fields.
@kernel function _layer_mass_ssp_rk3_substep!(u, grid, u⁰, G, Δt, α)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        M = (1 - α) * u⁰[i, j, k] + α * (scalar_layer_mass(i, j, k, grid, u) + Δt * layer_mass_tendency(i, j, k, grid, G[i, j, k]))
        u[i, j, k] = scalar_density_from_layer_mass(i, j, k, grid, M)
    end
end

# On a pressure-level grid the scalars take the mass-weighted stage; the round-off treatments
# of the time stepper are not applied to them (round to nearest).
launch_scalar_ssp_rk3_substep!(grid, scalar, rounding, arch, u, u⁰, G, Δt, α, key) =
    launch_ssp_rk3_substep!(rounding, arch, grid, u, u⁰, G, Δt, α, key)

launch_scalar_ssp_rk3_substep!(grid::PressureLevelGrid, ::Val{true}, rounding, arch, u, u⁰, G, Δt, α, key) =
    launch!(arch, grid, :xyz, _layer_mass_ssp_rk3_substep!, u, grid, u⁰, G, Δt, α)

#####
##### The completion of a stage of the multirate infinitesimal step
#####

@kernel function _set_from_layer_mass_sum!(a, grid, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = scalar_density_from_layer_mass(i, j, k, grid, b[i, j, k] + c[i, j, k])
end

@kernel function _set_from_layer_mass_increment!(a, grid, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = scalar_density_from_layer_mass(i, j, k, grid, b[i, j, k] + layer_mass_tendency(i, j, k, grid, c[i, j, k]))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set the scalar `ρc = U⁰ + δ` from its initial state `U⁰` and the increment `δ` of a stage, both in
the units of the stages: `set_sum!`, or on a pressure-level grid the density of the mass per unit
area `U⁰ + δ` at the current surface pressure.
"""
set_scalar_sum!(ρc, U⁰, δ) = set_scalar_sum!(ρc.grid, ρc, U⁰, δ)
set_scalar_sum!(grid, ρc, U⁰, δ) = set_sum!(ρc, U⁰, δ)

function set_scalar_sum!(grid::PressureLevelGrid, ρc, U⁰, δ)
    launch!(grid.architecture, grid, :xyz, _set_from_layer_mass_sum!, ρc, grid, U⁰, δ)
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set the scalar `ρc = U⁰ + δ` from its initial state `U⁰`, in the units of the stages, and an increment
`δ` per unit volume, which the gravity-wave substeps advance for the potential temperature
density: `set_sum!`, or on a pressure-level grid the density of `U⁰ + Δz⁻ δ`, with `Δz⁻` the depth of
the cell at the start of the stage.

The substeps evolve the increment per unit volume of the cell they are linearized on; reading it
with the depth of the start of the stage is consistent to first order in the change of the
surface pressure over the stage.
"""
set_substepped_scalar_sum!(ρc, U⁰, δ) = set_substepped_scalar_sum!(ρc.grid, ρc, U⁰, δ)
set_substepped_scalar_sum!(grid, ρc, U⁰, δ) = set_sum!(ρc, U⁰, δ)

function set_substepped_scalar_sum!(grid::PressureLevelGrid, ρc, U⁰, δ)
    launch!(grid.architecture, grid, :xyz, _set_from_layer_mass_increment!, ρc, grid, U⁰, δ)
    return nothing
end
