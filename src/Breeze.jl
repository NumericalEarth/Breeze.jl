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
    TemperatureField,
    IdealGas,
    PhaseTransitionConstants,
    CondensedPhase,
    mixture_gas_constant,
    mixture_heat_capacity

using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.Architectures: array_type, CPU, GPU
using Oceananigans: field

export
    array_type,
    CPU, GPU,
    Center, Face, Periodic, Bounded, Flat,
    RectilinearGrid,
    nodes, xnodes, ynodes, znodes,
    xnode, ynode, znode,
    xspacings, yspacings, zspacings,
    minimum_xspacing, minimum_yspacing, minimum_zspacing,
    ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom, ImmersedBoundaryCondition,
    Distributed, Partition,
    Centered, UpwindBiased, WENO,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    OpenBoundaryCondition, PerturbationAdvection, FieldBoundaryConditions,
    Field, CenterField, XFaceField, YFaceField, ZFaceField,
    field,
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

include("utils_grid.jl")
export 
    ncols

include("Thermodynamics/Thermodynamics.jl")
using .Thermodynamics

include("MoistAirBuoyancies.jl")
using .MoistAirBuoyancies

include("AtmosphereModels/AtmosphereModels.jl")
using .AtmosphereModels

include("Radiation/Radiation.jl")
using .Radiation

export
    AbstractRadiationModel,
    RRTMGPModel,
    initialize_rrtmgp_model,
    compute_vertical_fluxes!,
    flux_results

end # module Breeze
