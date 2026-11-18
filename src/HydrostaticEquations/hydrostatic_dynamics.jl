#####
##### HydrostaticDynamics definition
#####

struct HydrostaticDynamics{A, B, PS, GS, D, P, E, EF, G, W, HD, VD, FT, M}
    pressure_coefficient :: A          # A at z-faces, in Pa: p = A + B pˢ
    surface_pressure_coefficient :: B  # B at z-faces: p = A + B pˢ
    surface_pressure :: PS             # pˢ (prognostic, two-dimensional)
    surface_geopotential :: GS         # Φˢ (two-dimensional)
    density :: D                       # ρ = Δp / (g Δz), the layer mass per unit area and unit of z
    pressure :: P                      # layer pressure p = pˢᵗ Π^(1/κ)
    exner_function :: E                # layer Exner function Π: the mass-weighted mean of (p / pˢᵗ)^κ
    interface_exner_function :: EF     # Πᶠ = (p / pˢᵗ)^κ at z-faces
    geopotential :: G                  # Φ at cell centers
    vertical_mass_flux :: W            # ρw at z-faces: the mass flux through coordinate surfaces
    hyperdiffusion :: HD               # GridScaleHyperdiffusion of u, v and θ along coordinate surfaces, or nothing
    vertical_diffusion :: VD           # VerticalDiffusion of u, v and θ in the column, or nothing
    top_pressure :: FT                 # pressure of the model top, a coordinate surface
    standard_pressure :: FT            # pˢᵗ, the reference pressure of the potential temperature
    moisture :: M                      # HydrostaticMoisture: the diagnostics of the moist column, or nothing (dry air)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `HydrostaticDynamics` representing hydrostatic dynamics on a hybrid sigma-pressure
coordinate: of dry air, or of moist air when the model has a microphysics scheme.

The grid's vertical coordinate labels coordinate surfaces rather than heights: the pressure on
the z-face `k` of a column with surface pressure `pˢ` is `p = A[k] + B[k] * pˢ`, so the lowest face
is the ground (`A = 0`, `B = 1`) and the highest is the model top, a surface of constant pressure
(`B = 0`). The surface pressure `pˢ`, the horizontal velocities `u` and `v`, and the potential
temperature density `ρθ` are prognostic, with `ρ = Δp / (g Δz)` the mass of a layer per unit area
and unit of the grid's vertical coordinate. The mass flux through the coordinate surfaces is
diagnosed from continuity and the geopotential from hydrostatic balance; `model.velocities.w` is
the velocity through the coordinate surfaces, in units of the grid's vertical coordinate per
second.

The layer Exner function is the mass-weighted layer mean of `(p / pˢᵗ)^κ` and the hydrostatic
integral is the exact one for a layer of uniform potential temperature, so that the
pressure-gradient force of a resting isentropic atmosphere vanishes over any terrain.

On a `YinYangGrid` each panel integrates its own cells, and the halos of the surface pressure,
of the scalars and of the diagnostics of the column are interpolated from the other panel, those
of the velocities as a rotated pair. Nothing matches the fluxes of the two panels where they
meet, so the integrals over the sphere of the mass and of the scalars are not conserved unless
`add_conservation_fixer!` adds the global
[`HydrostaticConservationFixer`](@ref Breeze.HydrostaticEquations.HydrostaticConservationFixer)
to a simulation.

Momentum advection is `VectorInvariant` with a kinetic energy gradient that is `EnergyConserving`,
which with second-order `Centered` advection of the potential temperature is the configuration
whose pressure-gradient work cancels the conversion of internal and potential energy; or a
`WENOVectorInvariant`; or a centered or upwind-biased scheme such as `WENO(order=5)`, with which
the momentum equations are advanced in flux form. An upwind-biased scalar advection scheme
reconstructs the potential temperature on the faces in all three directions. The upwind-biased
schemes dissipate energy rather than conserve it. The thermodynamic formulation must be
`LiquidIcePotentialTemperatureFormulation`. The time stepper is `:SSPRungeKutta3`, which
integrates the equations unsplit, or `:GravityWaveMultirateInfinitesimalStep`, which substeps
the gravity waves.

Not supported, and rejected with an `ArgumentError`:

- Turbulence closures. Dissipation is not a `closure` of the model but part of the dynamics: see
  `hyperdiffusion` and `vertical_diffusion` below.
- Moisture in a model without microphysics: its equations are those of dry air, so the moisture
  density must remain zero and may not be forced.
- Every microphysics scheme other than the warm-phase one-moment scheme with saturation
  adjustment (see "Moisture" below), and with it: forcings and boundary fluxes of water, fluxes of
  energy through a boundary, radiation, `hyperdiffusion`, `vertical_diffusion`, a negative
  moisture correction, the conservation fixer, an Exner exponent other than `DryExponent()`, a
  saturation vapor pressure other than `ClausiusClapeyron`, and the time stepper
  `:GravityWaveMultirateInfinitesimalStep` unless it is built with
  `moist_gravity_wave_operator = true`.
- Radiation on a multi-region grid, and radiative transfer models that do not take the pressures
  of the coordinate surfaces from the dynamics.
- A Coriolis force with a horizontal component of the rotation: only `FPlane`, `BetaPlane` and
  `HydrostaticSphericalCoriolis` are accepted. The vertical velocity of this model is the
  velocity through the coordinate surfaces, which the non-traditional terms must not be given.
- Forcings of the velocities other than functions of the coordinates and time (without
  parameters or field dependencies), supplied under `u` or `v` as an acceleration or under `ρu`
  or `ρv` as a force per unit volume, and
  [`held_suarez_forcings`](@ref Breeze.Forcings.held_suarez_forcings).

A radiative transfer model heats the air through its flux divergence, which enters `ρθ` divided
by `cᵖᵈ Π`, and is given the exact pressures `A + B pˢ` of the coordinate surfaces as its level
pressures. A flux of energy through a boundary, supplied under `ρE`, enters `ρθ` divided by
`cᵖᵈ Π` likewise.

# Moisture

A model built with `microphysics = OneMomentCloudMicrophysics(cloud_formation =
SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()))` and `formulation =
LiquidIcePotentialTemperatureFormulation(exner_exponent = DryExponent())` carries vapor, cloud
liquid and rain. The coordinate, continuity and every prognostic weight remain those of dry
air: the prognostic surface pressure, which keeps the name `pˢ` among the prognostic fields, is
then the dry-mass surface pressure `πˢ`, and `ρ` the mass of dry air. The pressure is the weight
of the dry air and of all the water above, so the total surface pressure
`pˢ = πˢ + Σ g Δz ρqˣ` is diagnosed (`surface_pressure(dynamics)`); `set!(model; pˢ=...)` takes
the total surface pressure and finds the dry one. Water is carried as a mixing ratio per unit
dry mass by the mass fluxes that carry `θ`. The pressure-gradient force and the hydrostatic
integral use the density potential temperature `θᵨ = T Rᵐ / (Rᵈ Π)` in place of `θ`, an energy
source enters `ρθ` divided by `(cᵖᵈ + rᵗ cᵖᵛ) Π` with `rᵗ` the mixing ratio of all water, and the
tendency of `ρθ` includes the expansion work that the first law requires of `θ` with the dry
exponent. Cloud forms by saturation adjustment in every stage. Rain forms, evaporates and falls
once per time step after the dynamics (first-order operator splitting), with rates and a fall
speed evaluated at the density of the air in kg m⁻³ and a sedimentation that is implicit in the
physical layer depths. Falling rain carries its enthalpy from layer to layer and through the
ground, and the potential energy it releases heats the air it falls through; what leaves through
the ground is accumulated in `dynamics.moisture.ledger`.

Every layer must have a positive mass, that is a positive pressure thickness
`Δp = ΔA + ΔB pˢ`, in every column. This depends on the surface pressure, so it is checked each
time the layer mass is diagnosed: a surface pressure that leaves a layer without mass is
rejected with an `ArgumentError` when it is set, and stops a time step with a `DomainError`.

!!! warning "Known gate: vorticity and Coriolis schemes on the sphere"
    The vorticity flux `(ζ + f) ẑ × u` is that of Oceananigans' `VectorInvariant` scheme for a
    layer of uniform thickness. With the variable layer mass of this coordinate it conserves
    neither the mass-weighted kinetic energy nor the potential vorticity `(ζ + f) / ρ`: both
    need the mass fluxes and the layer mass in the vorticity flux. This is not verified to be
    harmless on the sphere, and it must be before the core is used for climate integrations.

Keyword arguments
=================

- `surface_pressure`: initial surface pressure `pˢ` in Pa: a number, a function of the horizontal
  coordinates, or an array.
- `top_pressure`: pressure of the model top in Pa, used to build the default coefficients.
- `standard_pressure`: reference pressure `pˢᵗ` of the potential temperature, in Pa.
- `surface_geopotential`: geopotential `Φˢ` of the ground in m² s⁻²: a number, a function of the
  horizontal coordinates, or an array.
- `pressure_coefficient`, `surface_pressure_coefficient`: the coefficients `A` (in Pa) and `B` on
  the z-faces, ordered from the ground to the model top. By default the coordinate is pure sigma,
  uniform in the grid's vertical coordinate: with `σ` the fraction of the vertical extent above
  a z-face, `A = (1 - σ) * top_pressure` and `B = σ`.
- `hyperdiffusion`: `nothing` (default) or a
  [`GridScaleHyperdiffusion`](@ref Breeze.HydrostaticEquations.GridScaleHyperdiffusion) of `u`,
  `v` and `θ` along the coordinate surfaces.
- `vertical_diffusion`: `nothing` (default) or a
  [`VerticalDiffusion`](@ref Breeze.HydrostaticEquations.VerticalDiffusion) of `u`, `v` and `θ` in
  the column, solved implicitly at the end of every time step.

```jldoctest
using Breeze

HydrostaticDynamics(top_pressure=200)

# output
HydrostaticDynamics
├── surface_pressure: 101325
├── top_pressure: 200.0
├── standard_pressure: 100000.0
└── surface_geopotential: 0
```
"""
function HydrostaticDynamics(; surface_pressure = 101325,
                               top_pressure = 0,
                               standard_pressure = 1e5,
                               surface_geopotential = 0,
                               pressure_coefficient = nothing,
                               surface_pressure_coefficient = nothing,
                               hyperdiffusion = nothing,
                               vertical_diffusion = nothing)

    isnothing(pressure_coefficient) == isnothing(surface_pressure_coefficient) ||
        throw(ArgumentError("pressure_coefficient and surface_pressure_coefficient must be given together."))

    top_pressure, standard_pressure = promote(top_pressure, standard_pressure)

    return HydrostaticDynamics(pressure_coefficient, surface_pressure_coefficient,
                               surface_pressure, surface_geopotential,
                               nothing, nothing, nothing, nothing, nothing, nothing,
                               hyperdiffusion, vertical_diffusion,
                               top_pressure, standard_pressure, nothing)
end

Adapt.adapt_structure(to, dynamics::HydrostaticDynamics) =
    HydrostaticDynamics(adapt(to, dynamics.pressure_coefficient),
                        adapt(to, dynamics.surface_pressure_coefficient),
                        adapt(to, dynamics.surface_pressure),
                        adapt(to, dynamics.surface_geopotential),
                        adapt(to, dynamics.density),
                        adapt(to, dynamics.pressure),
                        adapt(to, dynamics.exner_function),
                        adapt(to, dynamics.interface_exner_function),
                        adapt(to, dynamics.geopotential),
                        adapt(to, dynamics.vertical_mass_flux),
                        adapt(to, dynamics.hyperdiffusion),
                        adapt(to, dynamics.vertical_diffusion),
                        dynamics.top_pressure,
                        dynamics.standard_pressure,
                        adapt(to, dynamics.moisture))

@regional HydrostaticDynamics

#####
##### Hybrid coefficients
#####

# Pure sigma, uniform in the grid's vertical coordinate: σ = 1 at the ground and 0 at the top
function hybrid_coefficients(grid, ::Nothing, ::Nothing, top_pressure)
    z = on_architecture(CPU(), znodes(grid, Face()))
    z = collect(z)
    σ = (z[end] .- z) ./ (z[end] - z[1])
    A = (1 .- σ) .* top_pressure
    B = σ
    return A, B
end

hybrid_coefficients(grid, A, B, top_pressure) = collect(A), collect(B)

function validate_hybrid_coefficients(grid, A, B, standard_pressure)
    Nz = size(grid, 3)

    (length(A) == Nz + 1 && length(B) == Nz + 1) ||
        throw(ArgumentError("The hybrid coefficients must have one value per z-face, Nz + 1 = $(Nz + 1), " *
                            "but have $(length(A)) and $(length(B))."))

    (A[1] == 0 && B[1] == 1) ||
        throw(ArgumentError("The lowest z-face must be the ground, where the pressure is the surface " *
                            "pressure: pressure_coefficient = 0 and surface_pressure_coefficient = 1."))

    B[end] == 0 ||
        throw(ArgumentError("The model top must be a surface of constant pressure: " *
                            "surface_pressure_coefficient = 0 on the highest z-face."))

    (all(A .≥ 0) && all(diff(B) .≤ 0)) ||
        throw(ArgumentError("The hybrid coefficients are ordered from the ground to the model top, so " *
                            "pressure_coefficient may not be negative and surface_pressure_coefficient may not increase."))

    all(diff(A .+ B .* standard_pressure) .< 0) ||
        throw(ArgumentError("The pressure A + B pˢ must decrease from each z-face to the next at pˢ = $standard_pressure Pa."))

    return nothing
end

# A column of z-face values, readable in kernels at any (i, j)
function z_face_column_field(grid, values)
    column = Field{Nothing, Nothing, Face}(grid)
    FT = eltype(grid)
    data = on_architecture(architecture(grid), reshape(convert(Vector{FT}, values), 1, 1, :))
    @apply_regionally copyto!(interior(column), data)
    return column
end

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Materialize a stub `HydrostaticDynamics` into a full dynamics object: the hybrid coefficients
on the z-faces of `grid`, the prognostic surface pressure, and the diagnostic fields of the
hydrostatic column.
"""
function AtmosphereModels.materialize_dynamics(dynamics::HydrostaticDynamics, grid, boundary_conditions, constants)
    topology(grid, 3) === Flat &&
        throw(ArgumentError("HydrostaticDynamics requires a grid with a vertical dimension."))

    FT = eltype(grid)

    A, B = hybrid_coefficients(grid, dynamics.pressure_coefficient,
                               dynamics.surface_pressure_coefficient, dynamics.top_pressure)
    standard_pressure = convert(FT, dynamics.standard_pressure)
    validate_hybrid_coefficients(grid, A, B, standard_pressure)
    top_pressure = convert(FT, A[end])

    pressure_coefficient = z_face_column_field(grid, A)
    surface_pressure_coefficient = z_face_column_field(grid, B)

    surface_pressure = Field{Center, Center, Nothing}(grid)
    set!(surface_pressure, dynamics.surface_pressure)
    fill_halo_regions!(surface_pressure)

    surface_geopotential = Field{Center, Center, Nothing}(grid)
    set!(surface_geopotential, dynamics.surface_geopotential)
    fill_halo_regions!(surface_geopotential)

    density = CenterField(grid)
    pressure = CenterField(grid)
    exner_function = CenterField(grid)
    interface_exner_function = ZFaceField(grid)
    geopotential = CenterField(grid)
    vertical_mass_flux = ZFaceField(grid)

    return HydrostaticDynamics(pressure_coefficient, surface_pressure_coefficient,
                               surface_pressure, surface_geopotential,
                               density, pressure, exner_function, interface_exner_function,
                               geopotential, vertical_mass_flux,
                               materialize_hyperdiffusion(dynamics.hyperdiffusion, grid),
                               materialize_vertical_diffusion(dynamics.vertical_diffusion, grid),
                               top_pressure, standard_pressure, nothing)
end

# The horizontal velocities are prognostic; `w` is the diagnosed velocity through the coordinate surfaces.
function AtmosphereModels.materialize_momentum_and_velocities(::HydrostaticDynamics, grid, boundary_conditions)
    u = XFaceField(grid, boundary_conditions=boundary_conditions.u)
    v = YFaceField(grid, boundary_conditions=boundary_conditions.v)
    w = ZFaceField(grid)
    return (; u, v), (; u, v, w)
end

#####
##### Dynamics interface
#####

AtmosphereModels.prognostic_momentum_field_names(::HydrostaticDynamics) = (:u, :v)
AtmosphereModels.prognostic_dynamics_field_names(::HydrostaticDynamics) = tuple(:pˢ)
AtmosphereModels.additional_dynamics_field_names(::HydrostaticDynamics) = ()
AtmosphereModels.dynamics_prognostic_fields(dynamics::HydrostaticDynamics) = (; pˢ=dynamics.surface_pressure)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the continuity density of `HydrostaticDynamics`: the mass of a layer per unit area and
unit of the grid's vertical coordinate, ``ρ = Δp / (g Δz)``, diagnosed from the surface pressure.
"""
AtmosphereModels.continuity_density(dynamics::HydrostaticDynamics) = dynamics.density

# The air is dry and the layer mass is diagnosed, not prognostic: an energy flux through a boundary
# enters ρθ undiluted, as 𝒬 / (cᵖᵈ Π).
AtmosphereModels.energy_flux_density(::HydrostaticDynamics) = AtmosphereModels.DryAirDensity()

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the layer pressure of `HydrostaticDynamics`, in Pa: the pressure ``pˢᵗ Π^{1/κ}`` at which
the Exner function equals the mass-weighted layer mean ``Π``, so that ``T = θ Π``.
"""
AtmosphereModels.dynamics_pressure(dynamics::HydrostaticDynamics) = dynamics.pressure

AtmosphereModels.pressure_anomaly(::HydrostaticDynamics) = ZeroField()
AtmosphereModels.total_pressure(dynamics::HydrostaticDynamics) = dynamics.pressure
AtmosphereModels.surface_pressure(dynamics::HydrostaticDynamics) = dynamics.surface_pressure
AtmosphereModels.standard_pressure(dynamics::HydrostaticDynamics) = dynamics.standard_pressure
AtmosphereModels.dynamics_pressure_solver(::HydrostaticDynamics, grid) = nothing

# The layer mass is not the density of the local state, which is the equation of state at the
# layer pressure: ρ = p / (Rᵐ T).
@inline AtmosphereModels.thermodynamic_state_density(i, j, k, dynamics::HydrostaticDynamics, T, q, constants) =
    fixed_pressure_state_density(i, j, k, dynamics, T, q, constants)

# The horizontal velocities are prognostic and carry boundary conditions; `w` is diagnosed.
function AtmosphereModels.validate_velocity_boundary_conditions(dynamics::HydrostaticDynamics, boundary_conditions)
    :w ∈ keys(boundary_conditions) &&
        throw(ArgumentError("Boundary conditions for w are not supported for $(summary(dynamics)), " *
                            "whose vertical velocity is diagnosed from continuity."))
    return nothing
end

AtmosphereModels.validate_timestepper(::Val{:SSPRungeKutta3}, ::HydrostaticDynamics) = nothing

AtmosphereModels.validate_timestepper(::Val{:GravityWaveMultirateInfinitesimalStep}, ::HydrostaticDynamics) = nothing
AtmosphereModels.timestepper_uses_dynamics(::Val{:GravityWaveMultirateInfinitesimalStep}) = true

AtmosphereModels.validate_timestepper(::Val{name}, ::HydrostaticDynamics) where name =
    throw(ArgumentError("HydrostaticDynamics is integrated by :SSPRungeKutta3 or " *
                        ":GravityWaveMultirateInfinitesimalStep, got :$name."))

#####
##### Show methods
#####

Base.summary(dynamics::HydrostaticDynamics) =
    string("HydrostaticDynamics(top_pressure=", prettysummary(dynamics.top_pressure),
           ", pˢᵗ=", prettysummary(dynamics.standard_pressure), ")")

function Base.show(io::IO, dynamics::HydrostaticDynamics)
    print(io, "HydrostaticDynamics\n")
    print(io, "├── surface_pressure: ", prettysummary(dynamics.surface_pressure), '\n')
    print(io, "├── top_pressure: ", prettysummary(dynamics.top_pressure), '\n')
    print(io, "├── standard_pressure: ", prettysummary(dynamics.standard_pressure), '\n')
    print(io, "└── surface_geopotential: ", prettysummary(dynamics.surface_geopotential))
end
