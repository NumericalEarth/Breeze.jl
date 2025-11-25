"""
Julia package for finite volume GPU and CPU large eddy simulations (LES)
of atmospheric flows. The abstractions, design, and finite volume engine
are based on Oceananigans.
"""
module Breeze

export
    MoistAirBuoyancy,
    ThermodynamicConstants,
    ReferenceState,
    AnelasticFormulation,
    AtmosphereModel,
    PotentialTemperature,
    PotentialTemperatureField,
    TemperatureField,
    IdealGas,
    CondensedPhase,
    mixture_gas_constant,
    mixture_heat_capacity,
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium,
    BulkMicrophysics

using Oceananigans: Oceananigans, @at, AnisotropicMinimumDissipation, Average,
                    AveragedTimeInterval, BackgroundField, BetaPlane, Bounded,
                    CPU, Callback, Center, CenterField, Centered, Checkpointer,
                    ConstantCartesianCoriolis, Distributed, FPlane, Face,
                    Field, FieldBoundaryConditions, FieldDataset,
                    FieldTimeSeries, Flat, FluxBoundaryCondition, Forcing, GPU,
                    GradientBoundaryCondition, GridFittedBottom,
                    ImmersedBoundaryCondition, ImmersedBoundaryGrid, InMemory,
                    Integral, IterationInterval, JLD2Writer,
                    KernelFunctionOperation, LagrangianParticles, NetCDFWriter,
                    NonTraditionalBetaPlane, OnDisk, OpenBoundaryCondition,
                    PartialCellBottom, Partition, Periodic,
                    PerturbationAdvection, RectilinearGrid, Simulation,
                    SmagorinskyLilly, SpecifiedTimes, TimeInterval,
                    UpwindBiased, ValueBoundaryCondition, WENO,
                    WallTimeInterval, XFaceField, YFaceField, ZFaceField,
                    add_callback!, compute!, conjure_time_step_wizard!,
                    interior, iteration, minimum_xspacing, minimum_yspacing,
                    minimum_zspacing, nodes, prettytime, regrid!, run!, set!,
                    time_step!, xnodes, xspacings, ynodes, yspacings, znodes,
                    zspacings, ∂x, ∂y, ∂z

using Oceananigans.Grids: znode

export
    CPU, GPU,
    Center, Face, Periodic, Bounded, Flat,
    RectilinearGrid,
    nodes, xnodes, ynodes, znodes,
    znode,
    xspacings, yspacings, zspacings,
    minimum_xspacing, minimum_yspacing, minimum_zspacing,
    ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom, ImmersedBoundaryCondition,
    Distributed, Partition,
    Centered, UpwindBiased, WENO,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    OpenBoundaryCondition, PerturbationAdvection, FieldBoundaryConditions,
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    Average, Integral,
    BackgroundField, interior, set!, compute!, regrid!,
    Forcing,
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,
    SmagorinskyLilly, AnisotropicMinimumDissipation,
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

include("Microphysics/Microphysics.jl")
using .Microphysics

include("TurbulenceClosures/TurbulenceClosures.jl")
using .TurbulenceClosures

end # module Breeze
