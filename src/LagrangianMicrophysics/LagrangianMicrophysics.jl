"""
Lagrangian (particle-based) cloud microphysics: per-droplet condensational growth with
κ-Köhler equilibrium, droplet motion under drag and gravity, and the air transport
properties they require.
"""
module LagrangianMicrophysics

export
    # Parameters and component models
    DropletPhysicsParameters,
    PowerLawVaporDiffusivity,
    LinearThermalConductivity,
    SutherlandViscosity,
    LinearViscosity,
    KineticCorrection,
    NoKineticCorrection,
    BeardPruppacherVentilation,
    NoVentilation,
    KappaKoehlerActivity,
    DiluteSolutionActivity,
    BeardTerminalVelocity,
    RogersYauTerminalVelocity,
    StokesTerminalVelocity,

    # Air properties
    vapor_diffusivity,
    thermal_conductivity,
    dynamic_viscosity,

    # Köhler equilibrium and condensational growth
    curvature_coefficient,
    equilibrium_supersaturation,
    critical_radius,
    critical_supersaturation,
    equilibrium_radius,
    diffusional_growth_rate,
    condensational_growth_step,
    droplet_water_mass,
    ventilation_factor,

    # Droplet motion
    drag_correction,
    droplet_response_time,
    exponential_drag_step,
    droplet_terminal_velocity

include("droplet_physics.jl")

end # module LagrangianMicrophysics
