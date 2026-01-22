#####
##### Boundary condition wrappers for thermodynamic variable conversions
#####
##### These functions allow users to specify boundary conditions in terms of one
##### thermodynamic variable (e.g., energy flux) and have them automatically converted
##### to the equivalent BC for another variable (e.g., potential temperature flux).
#####

#####
##### EnergyFluxBoundaryConditionFunction: converts energy flux → potential temperature flux
##### Used when: user specifies ρe BCs but prognostic variable is ρθ
#####

"""
    EnergyFluxBoundaryConditionFunction

A wrapper for boundary conditions that converts energy flux to potential temperature flux.

When using `LiquidIcePotentialTemperatureFormulation`, the prognostic thermodynamic variable
is `ρθ` (potential temperature density). This wrapper allows users to specify energy fluxes
(e.g., sensible heat flux in W/m²) which are converted to potential temperature fluxes by
dividing by the local mixture heat capacity `cᵖᵐ`.

The relationship is:
```math
Jᶿ = 𝒬 / cᵖᵐ
```

where `𝒬` is the energy flux and `Jᶿ` is the potential temperature flux.

The mixture heat capacity is computed using moisture fractions from the microphysics scheme,
which correctly accounts for liquid and ice condensate when present.
"""
struct EnergyFluxBoundaryConditionFunction{C, S, M, TC, D}
    condition :: C
    side :: S
    microphysics :: M
    thermodynamic_constants :: TC
    density :: D
end

function Adapt.adapt_structure(to, ef::EnergyFluxBoundaryConditionFunction)
    return EnergyFluxBoundaryConditionFunction(Adapt.adapt(to, ef.condition),
                                               Adapt.adapt(to, ef.side),
                                               Adapt.adapt(to, ef.microphysics),
                                               Adapt.adapt(to, ef.thermodynamic_constants),
                                               Adapt.adapt(to, ef.density))
end

function Base.summary(ef::EnergyFluxBoundaryConditionFunction)
    cond = ef.condition
    cond_str = cond isa Number ? string(cond) : summary(cond)
    return string("EnergyFluxBoundaryConditionFunction(", cond_str, ")")
end

# Type aliases for dispatch on boundary side
const BottomEnergyFluxBC = EnergyFluxBoundaryConditionFunction{<:Any, <:Bottom}
const TopEnergyFluxBC    = EnergyFluxBoundaryConditionFunction{<:Any, <:Top}
const WestEnergyFluxBC   = EnergyFluxBoundaryConditionFunction{<:Any, <:West}
const EastEnergyFluxBC   = EnergyFluxBoundaryConditionFunction{<:Any, <:East}
const SouthEnergyFluxBC  = EnergyFluxBoundaryConditionFunction{<:Any, <:South}
const NorthEnergyFluxBC  = EnergyFluxBoundaryConditionFunction{<:Any, <:North}

# Convert energy flux to potential temperature flux: Jᶿ = 𝒬 / cᵖᵐ
@inline function 𝒬_to_Jᶿ(i, j, k, grid, ef, 𝒬, fields)
    qᵗ = @inbounds fields.qᵗ[i, j, k]
    ρ = @inbounds ef.density[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, ef.microphysics, ρ, qᵗ, fields)
    cᵖᵐ = mixture_heat_capacity(q, ef.thermodynamic_constants)
    return 𝒬 / cᵖᵐ
end

# getbc for bottom boundary (k = 1)
@inline function OceananigansBC.getbc(ef::BottomEnergyFluxBC, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    𝒬 = OceananigansBC.getbc(ef.condition, i, j, grid, clock, fields)
    return 𝒬_to_Jᶿ(i, j, 1, grid, ef, 𝒬, fields)
end

# getbc for top boundary (k = Nz)
@inline function OceananigansBC.getbc(ef::TopEnergyFluxBC, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    𝒬 = OceananigansBC.getbc(ef.condition, i, j, grid, clock, fields)
    return 𝒬_to_Jᶿ(i, j, grid.Nz, grid, ef, 𝒬, fields)
end

# getbc for west boundary (i = 1)
@inline function OceananigansBC.getbc(ef::WestEnergyFluxBC, j::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    𝒬 = OceananigansBC.getbc(ef.condition, j, k, grid, clock, fields)
    return 𝒬_to_Jᶿ(1, j, k, grid, ef, 𝒬, fields)
end

# getbc for east boundary (i = Nx)
@inline function OceananigansBC.getbc(ef::EastEnergyFluxBC, j::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    𝒬 = OceananigansBC.getbc(ef.condition, j, k, grid, clock, fields)
    return 𝒬_to_Jᶿ(grid.Nx, j, k, grid, ef, 𝒬, fields)
end

# getbc for south boundary (j = 1)
@inline function OceananigansBC.getbc(ef::SouthEnergyFluxBC, i::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    𝒬 = OceananigansBC.getbc(ef.condition, i, k, grid, clock, fields)
    return 𝒬_to_Jᶿ(i, 1, k, grid, ef, 𝒬, fields)
end

# getbc for north boundary (j = Ny)
@inline function OceananigansBC.getbc(ef::NorthEnergyFluxBC, i::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    𝒬 = OceananigansBC.getbc(ef.condition, i, k, grid, clock, fields)
    return 𝒬_to_Jᶿ(i, grid.Ny, k, grid, ef, 𝒬, fields)
end

const EnergyFluxBCType = BoundaryCondition{<:Flux, <:EnergyFluxBoundaryConditionFunction}

"""
    EnergyFluxBoundaryCondition(flux)

Create a boundary condition that wraps an energy flux and converts it to a potential
temperature flux for use with `LiquidIcePotentialTemperatureFormulation`.

The energy flux is divided by the local mixture heat capacity `cᵖᵐ` to obtain the
potential temperature flux: `Jᶿ = 𝒬 / cᵖᵐ`.
"""
function EnergyFluxBoundaryCondition(flux)
    ef = EnergyFluxBoundaryConditionFunction(flux, nothing, nothing, nothing, nothing)
    return BoundaryCondition(Flux(), ef)
end

#####
##### ThetaFluxBoundaryConditionFunction: converts potential temperature flux → energy flux
##### Used when: user wants energy flux diagnostics but prognostic variable is ρθ
#####

"""
    ThetaFluxBoundaryConditionFunction

A wrapper for boundary conditions that converts potential temperature flux to energy flux.

When building a diagnostic `energy_density` field from a `PotentialTemperatureFormulation`,
the boundary conditions on `ρθ` (potential temperature density) must be converted to
energy flux boundary conditions by multiplying by the local mixture heat capacity `cᵖᵐ`.

The relationship is:
```math
𝒬 = Jᶿ × cᵖᵐ
```

where `𝒬` is the energy flux and `Jᶿ` is the potential temperature flux.
"""
struct ThetaFluxBoundaryConditionFunction{C, S, M, TC, D}
    condition :: C
    side :: S
    microphysics :: M
    thermodynamic_constants :: TC
    density :: D
end

ThetaFluxBoundaryConditionFunction(condition, side, microphysics, thermodynamic_constants) =
    ThetaFluxBoundaryConditionFunction(condition, side, microphysics, thermodynamic_constants, nothing)

function Adapt.adapt_structure(to, tf::ThetaFluxBoundaryConditionFunction)
    return ThetaFluxBoundaryConditionFunction(Adapt.adapt(to, tf.condition),
                                              Adapt.adapt(to, tf.side),
                                              Adapt.adapt(to, tf.microphysics),
                                              Adapt.adapt(to, tf.thermodynamic_constants),
                                              Adapt.adapt(to, tf.density))
end

function Base.summary(tf::ThetaFluxBoundaryConditionFunction)
    cond = tf.condition
    cond_str = cond isa Number ? string(cond) : summary(cond)
    return string("ThetaFluxBoundaryConditionFunction(", cond_str, ")")
end

# Type aliases for dispatch on boundary side
const BottomThetaFluxBC = ThetaFluxBoundaryConditionFunction{<:Any, <:Bottom}
const TopThetaFluxBC    = ThetaFluxBoundaryConditionFunction{<:Any, <:Top}
const WestThetaFluxBC   = ThetaFluxBoundaryConditionFunction{<:Any, <:West}
const EastThetaFluxBC   = ThetaFluxBoundaryConditionFunction{<:Any, <:East}
const SouthThetaFluxBC  = ThetaFluxBoundaryConditionFunction{<:Any, <:South}
const NorthThetaFluxBC  = ThetaFluxBoundaryConditionFunction{<:Any, <:North}

# Convert potential temperature flux to energy flux: 𝒬 = Jᶿ × cᵖᵐ
@inline function Jᶿ_to_𝒬(i, j, k, grid, tf, Jᶿ, fields)
    qᵗ = @inbounds fields.qᵗ[i, j, k]
    ρ = @inbounds tf.density[i, j, k]
    q = grid_moisture_fractions(i, j, k, grid, tf.microphysics, ρ, qᵗ, fields)
    cᵖᵐ = mixture_heat_capacity(q, tf.thermodynamic_constants)
    return Jᶿ * cᵖᵐ
end

# getbc for bottom boundary (k = 1)
@inline function OceananigansBC.getbc(tf::BottomThetaFluxBC, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    Jᶿ = OceananigansBC.getbc(tf.condition, i, j, grid, clock, fields)
    return Jᶿ_to_𝒬(i, j, 1, grid, tf, Jᶿ, fields)
end

# getbc for top boundary (k = Nz)
@inline function OceananigansBC.getbc(tf::TopThetaFluxBC, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    Jᶿ = OceananigansBC.getbc(tf.condition, i, j, grid, clock, fields)
    return Jᶿ_to_𝒬(i, j, grid.Nz, grid, tf, Jᶿ, fields)
end

# getbc for west boundary (i = 1)
@inline function OceananigansBC.getbc(tf::WestThetaFluxBC, j::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    Jᶿ = OceananigansBC.getbc(tf.condition, j, k, grid, clock, fields)
    return Jᶿ_to_𝒬(1, j, k, grid, tf, Jᶿ, fields)
end

# getbc for east boundary (i = Nx)
@inline function OceananigansBC.getbc(tf::EastThetaFluxBC, j::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    Jᶿ = OceananigansBC.getbc(tf.condition, j, k, grid, clock, fields)
    return Jᶿ_to_𝒬(grid.Nx, j, k, grid, tf, Jᶿ, fields)
end

# getbc for south boundary (j = 1)
@inline function OceananigansBC.getbc(tf::SouthThetaFluxBC, i::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    Jᶿ = OceananigansBC.getbc(tf.condition, i, k, grid, clock, fields)
    return Jᶿ_to_𝒬(i, 1, k, grid, tf, Jᶿ, fields)
end

# getbc for north boundary (j = Ny)
@inline function OceananigansBC.getbc(tf::NorthThetaFluxBC, i::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields)
    Jᶿ = OceananigansBC.getbc(tf.condition, i, k, grid, clock, fields)
    return Jᶿ_to_𝒬(i, grid.Ny, k, grid, tf, Jᶿ, fields)
end

const ThetaFluxBCType = BoundaryCondition{<:Flux, <:ThetaFluxBoundaryConditionFunction}

"""
    ThetaFluxBoundaryCondition(flux)

Create a boundary condition that wraps a potential temperature flux and converts it to
an energy flux for use with diagnostic energy density fields.

The potential temperature flux is multiplied by the local mixture heat capacity `cᵖᵐ`
to obtain the energy flux: `𝒬 = Jᶿ × cᵖᵐ`.
"""
function ThetaFluxBoundaryCondition(flux)
    tf = ThetaFluxBoundaryConditionFunction(flux, nothing, nothing, nothing)
    return BoundaryCondition(Flux(), tf)
end

#####
##### Conversion functions: energy ↔ theta boundary conditions
#####

# Convert ρe BCs → ρθ BCs (for LiquidIcePotentialTemperatureFormulation)
energy_to_theta_bc(bc) = bc
energy_to_theta_bc(bc::BulkSensibleHeatFluxBoundaryCondition) = bc  # Already a θ flux
energy_to_theta_bc(bc::BoundaryCondition{<:Flux}) = EnergyFluxBoundaryCondition(bc.condition)

function energy_to_theta_bcs(fbcs::FieldBoundaryConditions)
    return FieldBoundaryConditions(; west     = energy_to_theta_bc(fbcs.west),
                                     east     = energy_to_theta_bc(fbcs.east),
                                     south    = energy_to_theta_bc(fbcs.south),
                                     north    = energy_to_theta_bc(fbcs.north),
                                     bottom   = energy_to_theta_bc(fbcs.bottom),
                                     top      = energy_to_theta_bc(fbcs.top),
                                     immersed = energy_to_theta_bc(fbcs.immersed))
end

# Convert ρθ BCs → ρe BCs (for diagnostic energy_density with PotentialTemperatureFormulation)
theta_to_energy_bc(bc) = bc
# For EnergyFluxBC, extract the original energy flux
theta_to_energy_bc(bc::EnergyFluxBCType) = BoundaryCondition(Flux(), bc.condition.condition)
# For regular flux BCs (actual θ fluxes), wrap to multiply by cᵖᵐ
theta_to_energy_bc(bc::BoundaryCondition{<:Flux}) = ThetaFluxBoundaryCondition(bc.condition)

function theta_to_energy_bcs(fbcs::FieldBoundaryConditions)
    return FieldBoundaryConditions(; west     = theta_to_energy_bc(fbcs.west),
                                     east     = theta_to_energy_bc(fbcs.east),
                                     south    = theta_to_energy_bc(fbcs.south),
                                     north    = theta_to_energy_bc(fbcs.north),
                                     bottom   = theta_to_energy_bc(fbcs.bottom),
                                     top      = theta_to_energy_bc(fbcs.top),
                                     immersed = theta_to_energy_bc(fbcs.immersed))
end

#####
##### Regularization functions for BC wrappers
#####

# Regularize EnergyFluxBoundaryCondition: populate side, microphysics, and thermodynamic_constants
const UnregularizedEnergyFluxBC = BoundaryCondition{<:Flux, <:EnergyFluxBoundaryConditionFunction{<:Any, Nothing}}

function regularize_atmosphere_boundary_condition(bc::UnregularizedEnergyFluxBC,
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants)
    ef = bc.condition
    density = dynamics_density(dynamics)
    new_ef = EnergyFluxBoundaryConditionFunction(ef.condition, side, microphysics, constants, density)
    return BoundaryCondition(Flux(), new_ef)
end

# Regularize ThetaFluxBoundaryCondition: populate side, microphysics, and thermodynamic_constants
const UnregularizedThetaFluxBC = BoundaryCondition{<:Flux, <:ThetaFluxBoundaryConditionFunction{<:Any, Nothing}}

function regularize_atmosphere_boundary_condition(bc::UnregularizedThetaFluxBC,
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants)
    tf = bc.condition
    density = dynamics_density(dynamics)
    new_tf = ThetaFluxBoundaryConditionFunction(tf.condition, side, microphysics, constants, density)
    return BoundaryCondition(Flux(), new_tf)
end
