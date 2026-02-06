"""
Julia package for finite volume GPU and CPU large eddy simulations (LES)
of atmospheric flows. The abstractions, design, and finite volume engine
are based on Oceananigans.
"""
module Breeze

export
    # AtmosphereModel
    MoistAirBuoyancy,
    ThermodynamicConstants,
    ReferenceState,
    surface_density,
    AnelasticDynamics,
    AnelasticModel,
    CompressibleDynamics,
    CompressibleModel,
    PrescribedDensity,
    PrescribedDynamics,
    KinematicModel,
    AtmosphereModel,
    StaticEnergyFormulation,
    LiquidIcePotentialTemperatureFormulation,
    RadiativeTransferModel,
    BackgroundAtmosphere,
    GrayOptics,
    ClearSkyOptics,
    AllSkyOptics,
    ConstantRadiusParticles,
    TemperatureField,
    IdealGas,
    CondensedPhase,
    mixture_gas_constant,
    mixture_heat_capacity,
    dynamics_density,
    dynamics_pressure,

    # Diagnostics
    compute_hydrostatic_pressure!,
    PotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy,
    static_energy_density,
    static_energy,
    total_energy,
    liquid_ice_potential_temperature_density,
    liquid_ice_potential_temperature,
    precipitation_rate,
    surface_precipitation_flux,
    total_pressure,
    specific_humidity,

    # Thermodynamics
    temperature,
    supersaturation,
    saturation_specific_humidity,
    PlanarLiquidSurface,
    PlanarIceSurface,

    # Microphysics
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium,
    SaturationSpecificHumidity,
    SaturationSpecificHumidityField,
    equilibrium_saturation_specific_humidity,
    RelativeHumidity,
    RelativeHumidityField,
    BulkMicrophysics,
    compute_hydrostatic_pressure!,
    NonEquilibriumCloudFormation,

    # BoundaryConditions
    BulkDrag,
    BulkSensibleHeatFlux,
    BulkVaporFlux,

    # Forcing utilities
    geostrophic_forcings,
    SubsidenceForcing,

    # Grid utilities
    PiecewiseStretchedDiscretization,

    # TimeSteppers
    SSPRungeKutta3,

    # ParcelDynamics
    ParcelDynamics,
    ParcelModel,
    ParcelState

using Oceananigans: Oceananigans, @at, AnisotropicMinimumDissipation, Average,
                    AveragedTimeInterval, BackgroundField, BetaPlane, Bounded,
                    CPU, Callback, Center, CenterField, Centered, Checkpointer, Clock,
                    ConstantCartesianCoriolis, Distributed, DynamicSmagorinsky,
                    ExponentialDiscretization, FPlane, Face, Field, FieldBoundaryConditions,
                    FieldDataset, FieldTimeSeries, Flat, FluxBoundaryCondition, Forcing,
                    Relaxation, GaussianMask, GPU,
                    GradientBoundaryCondition, GridFittedBottom,
                    ImmersedBoundaryCondition, ImmersedBoundaryGrid, InMemory,
                    Integral, IterationInterval, JLD2Writer,
                    KernelFunctionOperation, LagrangianParticles, NetCDFWriter,
                    NonTraditionalBetaPlane, OnDisk, OpenBoundaryCondition,
                    PartialCellBottom, Partition, Periodic,
                    PerturbationAdvection, RectilinearGrid, Simulation,
                    SmagorinskyLilly, SpecifiedTimes, TimeInterval,
                    UpwindBiased, ValueBoundaryCondition, WENO, FluxFormAdvection,
                    WallTimeInterval, XFaceField, YFaceField, ZFaceField,
                    add_callback!, compute!, conjure_time_step_wizard!,
                    interior, iteration, minimum_xspacing, minimum_yspacing,
                    minimum_zspacing, nodes, prettytime, regrid!, run!, set!,
                    time_step!, xnodes, xspacings, ynodes, yspacings, znodes,
                    zspacings, ∂x, ∂y, ∂z

using Oceananigans.Grids: znode
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition

export
    CPU, GPU,
    Center, Face, Periodic, Bounded, Flat,
    RectilinearGrid, ExponentialDiscretization, PiecewiseStretchedDiscretization, Clock,
    nodes, xnodes, ynodes, znodes,
    znode,
    xspacings, yspacings, zspacings,
    minimum_xspacing, minimum_yspacing, minimum_zspacing,
    ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom, ImmersedBoundaryCondition,
    Distributed, Partition,
    Centered, UpwindBiased, WENO, FluxFormAdvection,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition, ImpenetrableBoundaryCondition,
    OpenBoundaryCondition, PerturbationAdvection, FieldBoundaryConditions,
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    Average, Integral,
    BackgroundField, interior, set!, compute!, regrid!,
    Forcing, Relaxation, GaussianMask,
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,
    SmagorinskyLilly, AnisotropicMinimumDissipation, DynamicSmagorinsky,
    LagrangianParticles,
    conjure_time_step_wizard!,
    time_step!, Simulation, run!, Callback, add_callback!, iteration,
    NetCDFWriter, JLD2Writer, Checkpointer,
    TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval, SpecifiedTimes,
    FieldTimeSeries, FieldDataset, InMemory, OnDisk,
    ∂x, ∂y, ∂z, @at, KernelFunctionOperation,
    prettytime

include("Thermodynamics/Thermodynamics.jl")
using .Thermodynamics

include("MoistAirBuoyancies.jl")
using .MoistAirBuoyancies

include("AtmosphereModels/AtmosphereModels.jl")
using .AtmosphereModels

# BoundaryConditions is loaded early so formulation modules can use BC conversion utilities
include("BoundaryConditions/BoundaryConditions.jl")
using .BoundaryConditions

# Thermodynamic formulation modules (included after AtmosphereModels so they can dispatch on AtmosphereModel)
include("StaticEnergyFormulations/StaticEnergyFormulations.jl")
using .StaticEnergyFormulations: StaticEnergyFormulation

include("PotentialTemperatureFormulations/PotentialTemperatureFormulations.jl")
using .PotentialTemperatureFormulations: LiquidIcePotentialTemperatureFormulation

# Dynamics modules (included after AtmosphereModels so they can dispatch on AtmosphereModel)
include("AnelasticEquations/AnelasticEquations.jl")
using .AnelasticEquations: AnelasticDynamics, AnelasticModel

include("CompressibleEquations/CompressibleEquations.jl")
using .CompressibleEquations: CompressibleDynamics, CompressibleModel

include("KinematicDriver/KinematicDriver.jl")
using .KinematicDriver: PrescribedDensity, PrescribedDynamics, KinematicModel

include("Microphysics/Microphysics.jl")
using .Microphysics

include("TurbulenceClosures/TurbulenceClosures.jl")
using .TurbulenceClosures

include("Advection.jl")
using .Advection

include("CelestialMechanics/CelestialMechanics.jl")
using .CelestialMechanics

include("Forcings/Forcings.jl")
using .Forcings

include("VerticalGrids.jl")
using .VerticalGrids

include("TimeSteppers/TimeSteppers.jl")
using .TimeSteppers

include("ParcelModels/ParcelModels.jl")
using .ParcelModels

end # module Breeze
