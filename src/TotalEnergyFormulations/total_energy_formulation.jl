#####
##### TotalEnergyFormulation
#####

"""
$(TYPEDEF)

`TotalEnergyFormulation` evolves the total energy density ``ρE`` as the prognostic thermodynamic
variable, with ``ρ`` the total (dry air plus water) density and

```math
E = 𝓊 + K + g z , \\qquad 𝓊 = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ ,
```

the internal, kinetic and potential energy per unit mass. The internal energy is measured from
liquid water at zero kelvin with constant heat capacities, ``cᵛᵐ = qᵈ cᵛᵈ + qᵛ cᵛᵛ + qˡ cˡ + qⁱ cⁱ``,
and ``ℒˡ₀``, ``ℒⁱ₀`` are the latent heats extrapolated to zero kelvin. The kinetic energy ``K`` is
formed at cell centers from the staggered velocities (see [`kinetic_energyᶜᶜᶜ`](@ref)).

The energy equation is in conservation form,

```math
∂ₜ(ρE) + ∇ ⋅ \\left[ ρᵈ 𝐮 ĥᵈ + \\sum_x ĥˣ 𝐅ˣ + ρᵈ 𝐉ˢ + \\sum_x (hˣ + g z) 𝐉ˣ \\right] = ρQ ,
\\qquad ĥˣ = hˣ + K + g z ,
```

in which each constituent carries its own total enthalpy (the pressure work is inside the flux): the
dry-air momentum ``ρᵈ𝐮`` carries ``ĥᵈ = cᵖᵈ (T - T₀) + K + g z``, and each water species' own advective
mass flux ``𝐅ˣ`` carries its ``ĥˣ``, with ``hᵛ = cᵖᵛ T + ℒˡ₀ - cˡ T₀``, ``hˡ = cˡ (T - T₀)`` and
``hⁱ = cⁱ T + ℒˡ₀ - ℒⁱ₀ - cˡ T₀``. So the latent energy and the origin of the water move with exactly the
water they belong to, whatever scheme and velocity transport it. ``𝐉ˢ`` is the subgrid flux of the static energy
``ŝ = cᵖᵐ T + g z + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ``, ``𝐉ˣ`` the sedimentation flux of condensate ``x``
carrying its enthalpy ``hˣ`` and potential energy, and ``Q`` the radiative and imposed heating.
Phase changes, expansion and dissipation exchange energy within ``E`` and need no source:
``∫ρE`` changes only through the domain boundaries. The temperature is recovered in closed form,
``T = (E - K - g z - ℒˡ₀ qᵛ - (ℒˡ₀ - ℒⁱ₀) qⁱ) / cᵛᵐ``, and the pressure from ``p = ρ Rᵐ T``.
This is the formulation of [Satoh (2003)](@cite Satoh2003).

The origin of the energy is set by `reference_temperature` ``T₀``: liquid water has zero internal
energy and dry air zero enthalpy at ``T₀``, which shifts every water phase uniformly by ``cˡ T₀``
and dry air by ``cᵖᵈ T₀``,

```math
ρE_{T₀} = ρE_{0} - T₀ (ρᵈ cᵖᵈ + ρqᵗ cˡ) .
```

The latent heats are unchanged, and since dry mass and water are conserved so is ``∫ρE`` for any
``T₀``. With ``T₀ ≈ 273 K`` (the convention of ClimaAtmos) ``E`` is several times smaller in
magnitude, which reduces round-off; ``T₀ = 0`` (the default) is the origin of the stored-energy
budget written above.

Energy inputs supplied under the energy key `ρE` (boundary fluxes, forcings) apply directly,
and water crossing a wall adds its enthalpy ``cᵖᵛ T₁ + ℒˡ₀`` per unit mass. The formulation is
supported by `CompressibleDynamics` only. `formulation = :TotalEnergy` (or `:E`, `:ρE`) selects it.

```jldoctest
using Breeze

TotalEnergyFormulation(reference_temperature = 273.16)

# output
TotalEnergyFormulation
└── reference_temperature: 273.16
```
"""
struct TotalEnergyFormulation{E, U, H, Q, S, X, W, T}
    energy_density :: E              # ρE (prognostic)
    internal_energy :: U             # 𝓊 = ρE / ρ - K - g z + T₀ (qᵈ cᵖᵈ + qᵗ cˡ), from liquid water at 0 K (diagnostic)
    dry_enthalpy :: H                # ĥᵈ = cᵖᵈ (T - T₀) + K + g z (diagnostic, advected with the dry air)
    moisture_enthalpy :: Q           # eᵛᵉ, the enthalpy of the prognostic moisture less K + g z (diagnostic)
    static_energy :: S               # ŝ - T₀ (qᵈ cᵖᵈ + qᵗ cˡ) (diagnostic, diffused)
    diffusion_workspace :: X         # ρᵈ ŝ, the variable of the vertically implicit diffusion solve
    water_pressure_perturbation :: W # Σₓ Ĉₓ (ρqˣ)′ at the end of an acoustic loop (see `total_energy_compressible.jl`)
    reference_temperature :: T       # T₀, the energy origin
end

"""
$(TYPEDSIGNATURES)

Return a `TotalEnergyFormulation` whose energy origin is `reference_temperature` ``T₀``: liquid
water has zero internal energy and dry air zero enthalpy at ``T₀`` (default `0`).
"""
TotalEnergyFormulation(; reference_temperature = 0) =
    TotalEnergyFormulation(nothing, nothing, nothing, nothing, nothing, nothing, nothing, reference_temperature)

"""
$(TYPEDSIGNATURES)

Return the temperature ``T₀`` that sets the energy origin of `formulation`: liquid water has zero
internal energy and dry air zero enthalpy at ``T₀``. The stored energy of a `TotalEnergyFormulation`
model is ``ρE = ρE_{0} - T₀ (ρᵈ cᵖᵈ + ρqᵗ cˡ)`` with ``ρE_{0}`` measured from liquid water at zero kelvin.
"""
AtmosphereModels.reference_temperature(formulation::TotalEnergyFormulation) = formulation.reference_temperature
AtmosphereModels.reference_temperature(::Val) = 0

"""
$(TYPEDSIGNATURES)

The energy per unit volume by which the origin ``T₀`` lowers the stored energy,
``T₀ (ρᵈ cᵖᵈ + ρqᵗ cˡ)`` with ``ρqᵗ = ρ - ρᵈ`` the total water density. It depends on the
composition through the total water only, so a phase change leaves it unchanged.
"""
@inline energy_origin_offset(T₀, ρᵈ, ρ, constants) =
    T₀ * (ρᵈ * constants.dry_air.heat_capacity + (ρ - ρᵈ) * constants.liquid.heat_capacity)

Adapt.adapt_structure(to, formulation::TotalEnergyFormulation) =
    TotalEnergyFormulation(adapt(to, formulation.energy_density),
                           adapt(to, formulation.internal_energy),
                           adapt(to, formulation.dry_enthalpy),
                           adapt(to, formulation.moisture_enthalpy),
                           adapt(to, formulation.static_energy),
                           adapt(to, formulation.diffusion_workspace),
                           adapt(to, formulation.water_pressure_perturbation),
                           formulation.reference_temperature)

function BoundaryConditions.fill_halo_regions!(formulation::TotalEnergyFormulation)
    fill_halo_regions!(formulation.dry_enthalpy)
    fill_halo_regions!(formulation.moisture_enthalpy)
    fill_halo_regions!(formulation.static_energy)
    return nothing
end

#####
##### Kinetic energy at cell centers
#####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

# The mean of u² over the two faces of a cell, or u² itself in a Flat direction (no second face).
@inline x_mean_square(i, j, k, grid, u) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u)
@inline x_mean_square(i, j, k, grid::XFlatGrid, u) = ϕ²(i, j, k, grid, u)
@inline y_mean_square(i, j, k, grid, v) = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)
@inline y_mean_square(i, j, k, grid::YFlatGrid, v) = ϕ²(i, j, k, grid, v)
@inline z_mean_square(i, j, k, grid, w) = ℑzᵃᵃᶜ(i, j, k, grid, ϕ², w)

"""
$(TYPEDSIGNATURES)

Return the kinetic energy per unit mass at the center of cell `(i, j, k)` from the staggered
velocity components `u`, `v`, `w`,

```math
K = \\frac{1}{4} \\left[ (u²ᵢ + u²ᵢ₊₁) + (v²ⱼ + v²ⱼ₊₁) + (w²ₖ + w²ₖ₊₁) \\right] ,
```

the mean of each squared component over the two faces of the cell. This is the kinetic energy the
total energy ``E = 𝓊 + K + g z`` of `TotalEnergyFormulation` contains, everywhere it is formed.
"""
@inline kinetic_energyᶜᶜᶜ(i, j, k, grid, u, v, w) =
    (x_mean_square(i, j, k, grid, u) + y_mean_square(i, j, k, grid, v) + z_mean_square(i, j, k, grid, w)) / 2

@inline kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities) =
    kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

#####
##### Field naming interface
#####

const TotalEnergyNames = Union{Val{:TotalEnergy}, Val{:E}, Val{:ρE}}

AtmosphereModels.prognostic_thermodynamic_field_names(::TotalEnergyFormulation) = tuple(:ρE)
AtmosphereModels.additional_thermodynamic_field_names(::TotalEnergyFormulation) = (:𝓊, :ĥᵈ, :eᵛᵉ, :ŝ)
AtmosphereModels.thermodynamic_density_name(::TotalEnergyFormulation) = :ρE
AtmosphereModels.thermodynamic_density(formulation::TotalEnergyFormulation) = formulation.energy_density
AtmosphereModels.with_thermodynamic_density(f::TotalEnergyFormulation, ρE) =
    TotalEnergyFormulation(ρE, f.internal_energy, f.dry_enthalpy, f.moisture_enthalpy, f.static_energy,
                           f.diffusion_workspace, f.water_pressure_perturbation, f.reference_temperature)

AtmosphereModels.prognostic_thermodynamic_field_names(::TotalEnergyNames) = tuple(:ρE)
AtmosphereModels.additional_thermodynamic_field_names(::TotalEnergyNames) = (:𝓊, :ĥᵈ, :eᵛᵉ, :ŝ)
AtmosphereModels.thermodynamic_density_name(::TotalEnergyNames) = :ρE

Oceananigans.fields(formulation::TotalEnergyFormulation) =
    (; 𝓊 = formulation.internal_energy, ĥᵈ = formulation.dry_enthalpy, eᵛᵉ = formulation.moisture_enthalpy,
       ŝ = formulation.static_energy)

Oceananigans.prognostic_fields(formulation::TotalEnergyFormulation) = (; ρE = formulation.energy_density)

# The enthalpy of the dry air is what the advection scheme of ρE reconstructs.
AtmosphereModels.specific_thermodynamic_field(formulation::TotalEnergyFormulation) = formulation.dry_enthalpy

#####
##### Materialization
#####

AtmosphereModels.materialize_formulation(::TotalEnergyNames, dynamics, grid, boundary_conditions) =
    materialize_formulation(TotalEnergyFormulation(), dynamics, grid, boundary_conditions)

AtmosphereModels.materialize_formulation(::TotalEnergyFormulation, dynamics, grid, boundary_conditions) =
    throw(ArgumentError("TotalEnergyFormulation is supported only by CompressibleDynamics, " *
                        "not by $(summary(dynamics)). Use formulation = :StaticEnergy for the " *
                        "conservative energy variable of the anelastic equations."))

# Compressible dynamics without terrain metrics. The terrain-following core diagnoses its
# contravariant velocity alongside temperature and pressure, which this formulation does not yet do.
const FlatCompressibleDynamics = CompressibleDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}

AtmosphereModels.materialize_formulation(::TotalEnergyFormulation, dynamics::CompressibleDynamics, grid, boundary_conditions) =
    throw(ArgumentError("TotalEnergyFormulation is not yet supported on terrain-following grids."))

function AtmosphereModels.materialize_formulation(formulation::TotalEnergyFormulation, dynamics::FlatCompressibleDynamics,
                                                  grid, boundary_conditions)
    energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρE)
    internal_energy = CenterField(grid)
    dry_enthalpy = CenterField(grid)
    moisture_enthalpy = CenterField(grid)
    static_energy = CenterField(grid)
    diffusion_workspace = CenterField(grid)
    water_pressure_perturbation = CenterField(grid)
    T₀ = convert(eltype(grid), formulation.reference_temperature)
    return TotalEnergyFormulation(energy_density, internal_energy, dry_enthalpy, moisture_enthalpy, static_energy,
                                  diffusion_workspace, water_pressure_perturbation, T₀)
end

#####
##### Auxiliary variable computation
#####

# The internal energy needs the velocities, which the per-cell hook does not receive, so it is
# computed by the model-level method below, before the thermodynamic state is diagnosed.
AtmosphereModels.compute_auxiliary_thermodynamic_variables!(::TotalEnergyFormulation, dynamics, i, j, k, grid) = nothing

const TotalEnergyModel = AtmosphereModel{<:Any, <:TotalEnergyFormulation}

function AtmosphereModels.compute_auxiliary_thermodynamic_variables!(model::TotalEnergyModel)
    grid = model.grid
    formulation = model.formulation
    launch!(grid.architecture, grid, :xyz, _compute_internal_energy!,
            formulation.internal_energy, grid, formulation.energy_density, total_density(model.dynamics),
            dynamics_density(model.dynamics), model.velocities, formulation.reference_temperature,
            model.thermodynamic_constants)

    # Then diagnose the temperature (with any saturation adjustment) as for every formulation
    return invoke(AtmosphereModels.compute_auxiliary_thermodynamic_variables!, Tuple{AtmosphereModel}, model)
end

@kernel function _compute_internal_energy!(𝓊, grid, ρE, ρ, ρᵈ, velocities, T₀, constants)
    i, j, k = @index(Global, NTuple)
    K = kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities)
    z = znode(i, j, k, grid, c, c, c)
    g = constants.gravitational_acceleration
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρE₀ = ρE[i, j, k] + energy_origin_offset(T₀, ρᵈ[i, j, k], ρᵢ, constants)
        𝓊[i, j, k] = ρE₀ / ρᵢ - K - g * z
    end
end

#####
##### Thermodynamic state diagnosis
#####

"""
$(TYPEDSIGNATURES)

Build an `InternalEnergyDensityState` at grid point `(i, j, k)` from the internal energy of the
`formulation` (measured from liquid water at zero kelvin, whatever the formulation's origin),
the total density of the `dynamics`, and the moisture mass fractions `q`.
"""
@inline function AtmosphereModels.diagnose_thermodynamic_state(i, j, k, grid, formulation::TotalEnergyFormulation,
                                                               dynamics, q)
    𝓊 = @inbounds formulation.internal_energy[i, j, k]
    ρ = @inbounds total_density(dynamics)[i, j, k]
    return InternalEnergyDensityState(𝓊, q, ρ)
end

#####
##### Prognostic field collection
#####

function AtmosphereModels.collect_prognostic_fields(formulation::TotalEnergyFormulation,
                                                    dynamics,
                                                    momentum,
                                                    moisture_density,
                                                    moisture_name,
                                                    microphysical_fields,
                                                    tracers)
    ρE = formulation.energy_density
    thermodynamic_variables = merge((ρE=ρE,), NamedTuple{(moisture_name,)}((moisture_density,)))
    dynamics_fields = dynamics_prognostic_fields(dynamics)
    return merge(dynamics_fields, momentum, thermodynamic_variables, microphysical_fields, tracers)
end

#####
##### Saturation adjustment at fixed internal energy and density
#####

@inline Microphysics.saturation_adjustment_specific_humidity(T, 𝒰::InternalEnergyDensityState, constants, equilibrium) =
    saturation_specific_humidity(T, 𝒰.density, constants, equilibrium)

@inline Microphysics.saturation_adjustment_residual(T, 𝒰::InternalEnergyDensityState, constants) =
    T - temperature(𝒰, constants)

#####
##### Show methods
#####

Base.summary(::TotalEnergyFormulation) = "TotalEnergyFormulation"

function Base.show(io::IO, formulation::TotalEnergyFormulation)
    print(io, summary(formulation), '\n')
    if formulation.energy_density !== nothing
        print(io, "├── energy_density: ", prettysummary(formulation.energy_density), '\n')
        print(io, "├── internal_energy: ", prettysummary(formulation.internal_energy), '\n')
        print(io, "├── dry_enthalpy: ", prettysummary(formulation.dry_enthalpy), '\n')
        print(io, "├── moisture_enthalpy: ", prettysummary(formulation.moisture_enthalpy), '\n')
        print(io, "├── static_energy: ", prettysummary(formulation.static_energy), '\n')
    end
    print(io, "└── reference_temperature: ", prettysummary(formulation.reference_temperature))
end
