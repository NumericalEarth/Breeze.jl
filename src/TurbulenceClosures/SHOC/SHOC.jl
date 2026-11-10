"""
    SHOC

The Simplified Higher-Order Closure (SHOC) of Bogenschutz and Krueger (2013), re-expressed from
E3SM's `shoc.F90` (pinned at E3SM `ba53243f`) as allocation-free column functions that run one
thread per column on CPUs and GPUs, and wired into `AtmosphereModel` as
[`SimplifiedHigherOrderClosure`](@ref).
"""
module SHOC

export SimplifiedHigherOrderClosure, SHOCParameters

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index
using SpecialFunctions: erf

using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: Field, CenterField, ZFaceField
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat, znode, peripheral_node
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, getclosure
using Oceananigans.Utils: Utils, launch!, prettysummary

using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS

include("shoc_parameters.jl")
include("shoc_column.jl")
include("simplified_higher_order_closure.jl")

end # module SHOC
