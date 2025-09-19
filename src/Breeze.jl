"""
Julia package for finite volume GPU and CPU large eddy simulations (LES)
of atmospheric flows. The abstractions, design, and finite volume engine
are based on Oceananigans.
"""
module Breeze

export
    MoistAirBuoyancy,
    AtmosphereThermodynamics,
    ReferenceStateConstants,
    AnelasticFormulation,
    AtmosphereModel,
    TemperatureField

using Oceananigans

export
    CPU, GPU,
    Center, Face, Periodic, Bounded, Flat,
    RectilinearGrid,
    nodes, xnodes, ynodes, znodes,
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
    SmagorinskyLilly, DynamicSmagorinsky, AnisotropicMinimumDissipation,
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

end # module Breeze
