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
#   `:ρᵈ` (dry given)   — recover ρ = ρᵈ/qᵈ with qᵈ = 1 − qᵗ (combining ρᵈ and the moisture), then
#         (re)weight the moisture partial densities by the total. A moist `:ρᵈ` initial condition
#         should provide moisture as a fraction (`qᵗ`/`qᵛ`/`qᵉ`).
#   neither — diagnose ρ = ρᵈ + Σρqˣ from the existing fields.
#
# The thermodynamic variable (ρθ = ρᵈθ) and momentum (ρu = ρᵈu) are set afterwards (phase 2) and so
# need no rescaling here.
function AtmosphereModels.establish_densities!(model::CompressibleModel, total_density_given, dry_density_given)
    grid = model.grid
    arch = grid.architecture
    ρᵈ = model.dynamics.dry_density
    ρ  = model.dynamics.total_density

    if total_density_given
        launch!(arch, grid, :xyz, _split_total_into_dry!,
                ρ, ρᵈ, model.microphysics, model.moisture_density, model.microphysical_fields)
    elseif dry_density_given
        qᵛᵉ = specific_prognostic_moisture(model)
        launch!(arch, grid, :xyz, _dry_to_total!, ρ, ρᵈ, qᵛᵉ, model.moisture_density)
    else
        return AtmosphereModels.compute_total_density!(model)
    end

    fill_halo_regions!(ρ)
    fill_halo_regions!(ρᵈ)
    return nothing
end

# `:ρ` placeholder (= total ρ) is in `ρᵈ`; copy it to the total-density field and back out the dry
# density ρᵈ = ρ − Σρqˣ. The read-before-write at the same point makes the in-place `ρᵈ` aliasing safe.
@kernel function _split_total_into_dry!(ρ, ρᵈ, microphysics, moisture_density, microphysical_fields)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        total = ρᵈ[i, j, k]   # placeholder = total density ρ
        ρqᵗ = AtmosphereModels.total_condensate_density(i, j, k, microphysics,
                                                        moisture_density, microphysical_fields)
        ρ[i, j, k]  = total
        ρᵈ[i, j, k] = total - ρqᵗ
    end
end

# `:ρᵈ` is the dry density; recover ρ = ρᵈ/qᵈ (qᵈ = 1 − qᵛᵉ) and re-weight the moisture density by ρ.
@kernel function _dry_to_total!(ρ, ρᵈ, qᵛᵉ, moisture_density)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        qᵛ = qᵛᵉ[i, j, k]
        qᵈ = 1 - qᵛ                # dry-air mass fraction (vapor/equilibrium moisture at init)
        ρ_ijk = ρᵈ[i, j, k] / qᵈ   # total density ρ = ρᵈ / qᵈ
        ρ[i, j, k] = ρ_ijk
        moisture_density[i, j, k] = ρ_ijk * qᵛ
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
