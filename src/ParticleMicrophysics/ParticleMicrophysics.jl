"""
Lagrangian (particle-based) cloud microphysics: per-droplet condensational growth with
κ-Köhler equilibrium, droplet motion under drag and gravity, and the air transport
properties they require.
"""
module ParticleMicrophysics

export
    # Particle-based microphysics schemes
    LagrangianMicrophysics,
    DiffusiveSelfInteraction,
    Droplet,

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

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index, @atomic

using Oceananigans.Advection: Centered
using Oceananigans.Architectures: Architectures, architecture, on_architecture
using Oceananigans.Fields: CenterField, FractionalIndices
using Oceananigans.Grids: Bounded, Face, Flat, Periodic, RectilinearGrid, halo_size, topology
using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticleTracking, LagrangianParticles, StructArray
using Oceananigans.Operators: Vᶜᶜᶜ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ
using Oceananigans.TimeSteppers: materialize_lagrangian_particles, update_state!
using Oceananigans.TurbulenceClosures: ScalarDiffusivity, ThreeDimensionalFormulation
using Oceananigans.Utils: KernelParameters, interpolator, launch!

using Breeze.AtmosphereModels: AtmosphereModels, dynamics_density, dynamics_pressure,
                                standard_pressure, kernel_time_step
using Breeze.PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation
using Breeze.TimeSteppers: AcousticRungeKutta3, SSPRungeKutta3
using Breeze.Thermodynamics: MoistureMassFractions, mixture_gas_constant, mixture_heat_capacity, supersaturation

include("lagrangian_microphysics.jl")

end # module ParticleMicrophysics
