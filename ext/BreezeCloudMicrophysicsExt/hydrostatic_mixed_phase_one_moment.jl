#####
##### Mixed-phase one-moment microphysics with saturation adjustment (MP1M) on HydrostaticDynamics
#####
#
# The scheme runs on `HydrostaticDynamics` through the generic microphysics interface, as on any
# dynamics: its sources are the stage tendencies of `compute_microphysical_tendencies!` and its
# rain and snow fall with the velocities it declares (`sedimentation_velocity`), which the
# dynamics turns into the sedimentation of its stages and into the mass that leaves through the
# ground (plvd_logs/3c_1M_ENERGY_DERIVATION.txt). Its rates and fall speeds read the density as
# that of the air in kg m⁻³ and Δz as a depth in m, so it requires a `PressureLevelGrid`.
#
# Nothing here adapts the scheme: these methods tell the dynamics which of its fields hold what.

HydrostaticEquations.supports_hydrostatic_dynamics(::MP1M) = true
HydrostaticEquations.requires_pressure_level_grid(::MP1M) = true

# All liquid (cloud and rain) and all ice (cloud and snow) load the air and enter its enthalpy.
# The fall speeds are those the scheme writes: the dynamics computes none (`wʳ = nothing`).
HydrostaticEquations.hydrostatic_water_fields(::MP1M, μ) =
    (; qᵛ = μ.qᵛ, qˡ = μ.qˡ, qⁱ = μ.qⁱ, qᶜˡ = μ.qᶜˡ, ρqʳ = μ.ρqʳ, ρqˢ = μ.ρqˢⁿ, wʳ = nothing)
