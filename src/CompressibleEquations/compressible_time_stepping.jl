#####
##### Time stepping for CompressibleDynamics
#####

#####
##### Model initialization
#####

# No default initialization for compressible models
AtmosphereModels.initialize_model_thermodynamics!(::CompressibleModel) = nothing

#####
##### Pressure correction (no-op for compressible dynamics)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed diagnostically from the equation of state.
"""
AtmosphereModels.compute_pressure_correction!(::CompressibleModel, Δt) = nothing

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - no pressure projection is needed.
"""
AtmosphereModels.make_pressure_correction!(::CompressibleModel, Δt) = nothing

#####
##### Pressure solver (no-op)
#####

"""
$(TYPEDSIGNATURES)

No-op for `CompressibleDynamics` - pressure is computed from the equation of state, not solved.
"""
solve_for_pressure!(::CompressibleModel) = nothing

#####
##### Total air density diagnostic (ρ = ρᵈ + Σρˣ)
#####

# Diagnose the total air density into `dynamics.total_density` from the prognostic dry density and
# the water (vapor + condensate) densities. Run once per `update_state!`, before the moisture
# recovery / EOS / buoyancy consume it. It is invariant under saturation adjustment (which
# conserves total water and dry mass), so a single evaluation per update is correct.
function AtmosphereModels.compute_total_density!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    ρ = model.dynamics.total_density
    launch!(arch, grid, :xyz, _compute_total_density!,
            ρ, grid, model.dynamics.dry_density, model.microphysics,
            model.moisture_density, model.microphysical_fields)
    fill_halo_regions!(ρ)
    return nothing
end

@kernel function _compute_total_density!(ρ, grid, dry_density, microphysics,
                                         moisture_density, microphysical_fields)
    i, j, k = @index(Global, NTuple)
    @inbounds ρ[i, j, k] =
        AtmosphereModels.total_density(i, j, k, dry_density, microphysics,
                                       moisture_density, microphysical_fields)
end

#####
##### Initial-condition density reconciliation: make ρᵈ and the total density ρ consistent
#####

# `set!` mid-hook (after density + moisture, before the thermodynamic variable and velocities).
# The two density-input modes need different computations, since the moisture partial densities
# ρqˣ = ρ·qˣ themselves depend on the total ρ:
#
#   `:ρ`  (total given) — the value sits in the dry-density field as a placeholder, and the moisture
#         branches already weighted the partial densities by it. Move it into the total-density
#         field and back out ρᵈ = ρ − Σρqˣ.
#   `:ρᵈ` (dry given)   — combine specific inputs and independently supplied partial densities to
#         recover total ρ, then reweight only the specific inputs by ρ. This distinction matters
#         for non-equilibrium schemes whose condensate may be supplied alongside vapor.
#   neither — diagnose ρ = ρᵈ + Σρqˣ from the existing fields.
#
# The thermodynamic variable (ρθ = ρᵈθ) and momentum (ρu = ρᵈu) are set afterwards (phase 2) and so
# need no rescaling here.
function density_reconciliation_fields(model, specific_microphysical_names)
    specific_density_names = map(AtmosphereModels.specific_to_density_weighted,
                                 specific_microphysical_names)
    condensate_names = AtmosphereModels.condensate_field_names(model.microphysics)
    specific_condensate_names = filter(name -> name ∈ specific_density_names,
                                        condensate_names)
    absolute_condensate_names = filter(name -> name ∉ specific_density_names,
                                        condensate_names)

    specific_microphysical_fields = map(name -> model.microphysical_fields[name],
                                         specific_density_names)
    specific_condensate_fields = map(name -> model.microphysical_fields[name],
                                      specific_condensate_names)
    absolute_condensate_fields = map(name -> model.microphysical_fields[name],
                                      absolute_condensate_names)

    return specific_microphysical_fields, specific_condensate_fields, absolute_condensate_fields
end

function AtmosphereModels.establish_densities!(model::CompressibleModel,
                                                total_density_given,
                                                dry_density_given,
                                                moisture_given=false,
                                                specific_moisture_given=false,
                                                total_moisture_given=false,
                                                specific_microphysical_names=())
    grid = model.grid
    arch = grid.architecture
    ρᵈ = model.dynamics.dry_density
    ρ  = model.dynamics.total_density

    if total_density_given
        launch!(arch, grid, :xyz, _split_total_into_dry!,
                ρ, ρᵈ, model.microphysics, model.moisture_density,
                model.microphysical_fields, total_moisture_given)
    elseif dry_density_given || moisture_given || !isempty(specific_microphysical_names)
        qᵛᵉ = specific_prognostic_moisture(model)
        specific_microphysical_fields, specific_condensate_fields, absolute_condensate_fields =
            density_reconciliation_fields(model, specific_microphysical_names)

        launch!(arch, grid, :xyz, _dry_to_total!,
                ρ, ρᵈ, qᵛᵉ, model.moisture_density,
                specific_moisture_given, total_moisture_given,
                specific_microphysical_fields,
                specific_condensate_fields,
                absolute_condensate_fields)
    else
        return AtmosphereModels.compute_total_density!(model)
    end

    fill_halo_regions!(ρ)
    fill_halo_regions!(ρᵈ)
    return nothing
end

# Relative humidity is diagnosed after the preliminary density split because it needs temperature.
# At this point the moisture field contains the requested vapor partial density
# ρqᵛ = ℋ pᵛ⁺ / (Rᵛ T), and specifically supplied microphysical fields have already
# been weighted by the old total density. Preserve the vapor partial density and their specific
# values while reconciling dry and total density.
function AtmosphereModels.establish_relative_humidity_densities!(model::CompressibleModel,
                                                                  total_density_given,
                                                                  specific_microphysical_names=())
    grid = model.grid
    arch = grid.architecture
    ρᵈ = model.dynamics.dry_density
    ρ = model.dynamics.total_density
    qᵛᵉ = specific_prognostic_moisture(model)

    specific_microphysical_fields, specific_condensate_fields, absolute_condensate_fields =
        density_reconciliation_fields(model, specific_microphysical_names)

    launch!(arch, grid, :xyz, _establish_relative_humidity_densities!,
            ρ, ρᵈ, qᵛᵉ, model.moisture_density,
            specific_microphysical_fields,
            specific_condensate_fields,
            absolute_condensate_fields,
            total_density_given)

    fill_halo_regions!(ρ)
    fill_halo_regions!(ρᵈ)
    return nothing
end

# `:ρ` placeholder (= total ρ) is in `ρᵈ`; copy it to the total-density field and back out the dry
# density ρᵈ = ρ − Σρqˣ. The read-before-write at the same point makes the in-place `ρᵈ` aliasing safe.
@kernel function _split_total_into_dry!(ρ, ρᵈ, microphysics, moisture_density,
                                        microphysical_fields, total_moisture_given)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        total = ρᵈ[i, j, k]   # placeholder = total density ρ
        partitioned_water_density =
            AtmosphereModels.total_condensate_density(i, j, k, microphysics,
                                                       moisture_density, microphysical_fields)
        ρqᵗ = ifelse(total_moisture_given,
                     moisture_density[i, j, k],
                     partitioned_water_density)
        ρ[i, j, k]  = total
        ρᵈ[i, j, k] = total - ρqᵗ
    end
end

# Sum and rescale statically-sized tuples of fields without runtime symbol lookup in kernels.
@inline sum_field_tuple(i, j, k, ::Tuple{}) = false

@inline function sum_field_tuple(i, j, k, fields::Tuple)
    field = first(fields)
    value = @inbounds field[i, j, k]
    return value + sum_field_tuple(i, j, k, Base.tail(fields))
end

@inline rescale_field_tuple!(i, j, k, ::Tuple{}, ratio) = nothing

@inline function rescale_field_tuple!(i, j, k, fields::Tuple, ratio)
    field = first(fields)
    @inbounds field[i, j, k] *= ratio
    return rescale_field_tuple!(i, j, k, Base.tail(fields), ratio)
end

# For fixed dry density, solve
#
#     ρ⁺ = (ρᵈ + ρqᵛ + A) / (1 - S),
#
# where the requested relative humidity fixes vapor partial density ρqᵛ, S contains
# condensates supplied as specific values, and A contains condensates supplied as partial
# densities. For fixed total density, retain the supplied ρ and recompute dry density as the
# residual. Specifically supplied P3 moments remain total-density weighted in either mode.
@kernel function _establish_relative_humidity_densities!(ρ, ρᵈ, qᵛᵉ, moisture_density,
                                                          specific_microphysical_fields,
                                                          specific_condensate_fields,
                                                          absolute_condensate_fields,
                                                          total_density_given)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        old_total = ρ[i, j, k]
        dry = ρᵈ[i, j, k]
        vapor_density = moisture_density[i, j, k]
        specific_condensate_density =
            sum_field_tuple(i, j, k, specific_condensate_fields)
        absolute_condensate_density =
            sum_field_tuple(i, j, k, absolute_condensate_fields)
        specific_condensate = specific_condensate_density / old_total

        diagnosed_total =
            (dry + vapor_density + absolute_condensate_density) / (1 - specific_condensate)
        new_total = ifelse(total_density_given, old_total, diagnosed_total)
        ratio = ifelse(total_density_given, 1, new_total / old_total)
        fixed_total_dry = new_total - vapor_density - specific_condensate_density -
                          absolute_condensate_density

        rescale_field_tuple!(i, j, k, specific_microphysical_fields, ratio)

        ρ[i, j, k] = new_total
        qᵛᵉ[i, j, k] = vapor_density / new_total
        ρᵈ[i, j, k] = ifelse(total_density_given, fixed_total_dry, dry)
    end
end

# Given dry density ρᵈ, recover total density from a mixture of specific inputs S
# and absolute partial-density inputs A:
#
#     ρ = (ρᵈ + A) / (1 - S).
#
# If qᵗ was supplied, the moisture slot already represents all water, so independent
# condensate fields must not be counted a second time. Only fields supplied through
# specific kwargs are reweighted from their provisional ρᵈ basis to the recovered ρ.
@kernel function _dry_to_total!(ρ, ρᵈ, qᵛᵉ, moisture_density,
                                specific_moisture_given, total_moisture_given,
                                specific_microphysical_fields,
                                specific_condensate_fields,
                                absolute_condensate_fields)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵈ_ijk = ρᵈ[i, j, k]
        moisture_specific = qᵛᵉ[i, j, k]
        moisture_partial_density = moisture_density[i, j, k]

        specific_water = ifelse(specific_moisture_given, moisture_specific, 0)
        absolute_water = ifelse(specific_moisture_given, 0, moisture_partial_density)

        specific_condensate = sum_field_tuple(i, j, k, specific_condensate_fields) / ρᵈ_ijk
        absolute_condensate = sum_field_tuple(i, j, k, absolute_condensate_fields)

        S = ifelse(total_moisture_given,
                   specific_water,
                   specific_water + specific_condensate)
        A = ifelse(total_moisture_given,
                   absolute_water,
                   absolute_water + absolute_condensate)

        ρ_ijk = (ρᵈ_ijk + A) / (1 - S)
        ρ[i, j, k] = ρ_ijk

        moisture_density[i, j, k] =
            ifelse(specific_moisture_given,
                   ρ_ijk * moisture_specific,
                   moisture_partial_density)
        qᵛᵉ[i, j, k] =
            ifelse(specific_moisture_given,
                   moisture_specific,
                   moisture_partial_density / ρ_ijk)

        rescale_field_tuple!(i, j, k, specific_microphysical_fields, ρ_ijk / ρᵈ_ijk)
    end
end

#####
##### Auxiliary dynamics variables (temperature and pressure for compressible)
#####

"""
$(TYPEDSIGNATURES)

Compute temperature and pressure jointly for `CompressibleModel`.

For compressible dynamics with potential temperature thermodynamics, temperature and
pressure are coupled via the ideal gas law and the potential temperature definition:

```math
θ = T (p₀/p)^κ \\quad \\text{and} \\quad p = ρ R^m T
```

Eliminating the circular dependency gives the direct formula:

```math
T = θ^γ \\left(\\frac{ρ R^m}{p₀}\\right)^{γ-1}
```

where ``γ = c_p / c_v`` is the heat capacity ratio. Once temperature is known,
pressure is computed from the ideal gas law ``p = ρ R^m T``.

This joint computation is necessary because, unlike anelastic dynamics where pressure
comes from a reference state, compressible dynamics requires solving for both
temperature and pressure simultaneously.
"""
function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::CompressibleModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    # Ensure halos are filled (may have been async from update_state!)
    # These fields are needed for pressure computation via equation of state
    fill_halo_regions!(dynamics.dry_density)
    fill_halo_regions!(prognostic_fields(model.formulation))

    launch!(arch, grid, :xyz,
            _compute_temperature_and_pressure!,
            model.temperature,
            dynamics.pressure,
            dynamics.dry_density,
            dynamics.total_density,
            model.formulation,
            dynamics,
            specific_prognostic_moisture(model),
            grid,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    fill_halo_regions!(model.temperature)
    fill_halo_regions!(dynamics.pressure)

    return nothing
end

@kernel function _compute_temperature_and_pressure!(temperature_field, pressure_field,
                                                    dry_density, total_density, formulation, dynamics,
                                                    specific_prognostic_moisture, grid, microphysics,
                                                    microphysical_fields, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρᵈ = dry_density[i, j, k]    # coupling density (recovers θ = ρθ/ρᵈ)
        ρ  = total_density[i, j, k]  # total air density (mass fractions, EOS, inversion)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
    end

    # Compute moisture fractions (mass fractions: divide by total ρ)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    Rᵐ = mixture_gas_constant(q, constants)

    # Compute temperature and pressure jointly
    T, p = temperature_and_pressure(i, j, k, grid, formulation, dynamics, ρᵈ, ρ, Rᵐ, q, constants)

    @inbounds begin
        temperature_field[i, j, k] = T
        pressure_field[i, j, k] = p
    end
end

# Dispatch on formulation type for the coupled temperature-pressure computation

@inline function temperature_and_pressure(i, j, k, grid,
                                          formulation::LiquidIcePotentialTemperatureFormulation,
                                          dynamics, ρᵈ, ρ, Rᵐ, q, constants)
    # Note: potential_temperature_density is ρθ = ρᵈθ (prognostic, dry-coupled);
    # θ is recovered with the coupling density ρᵈ, while the inversion and EOS below use total ρ.
    ρθ = @inbounds formulation.potential_temperature_density[i, j, k]
    θ = ρθ / ρᵈ
    pˢᵗ = standard_pressure(dynamics)

    # Invert θˡⁱ at constant density via LiquidIceDensityState.temperature, which iterates the
    # implicit relation T = (ρRᵐT/pˢᵗ)^κ θ + (ℒˡqˡ+ℒⁱqⁱ)/cᵖᵐ to convergence. This is the same
    # inversion used by the saturation adjustment on the compressible core, so the dynamics and the
    # microphysics carry one self-consistent T (fixes the κ·ΔL split — NumericalEarth/Breeze.jl#765).
    𝒰 = LiquidIceDensityState(θ, q, pˢᵗ, ρ, formulation.temperature_solver)
    T = temperature(𝒰, constants)

    # Ideal gas law: p = ρ Rᵐ T
    p = ρ * Rᵐ * T

    return T, p
end

# Build the density-based thermodynamic state for the compressible core, so that the
# saturation adjustment and the θˡⁱ→T inversion are evaluated at the prognostic density ρ (true
# pressure p = ρRᵐT) rather than a reference pressure. The generic (reference-pressure) method in
# PotentialTemperatureFormulations is retained for anelastic dynamics. See NumericalEarth/Breeze.jl#765.
@inline function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid,
                                                               formulation::LiquidIcePotentialTemperatureFormulation,
                                                               dynamics::CompressibleDynamics, q)
    θ = @inbounds formulation.potential_temperature[i, j, k]
    ρ = @inbounds dynamics.total_density[i, j, k]  # total ρ: density-based inversion / true pressure
    pˢᵗ = standard_pressure(dynamics)
    return LiquidIceDensityState(θ, q, pˢᵗ, ρ, formulation.temperature_solver)
end
