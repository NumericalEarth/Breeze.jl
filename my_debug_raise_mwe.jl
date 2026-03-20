using CUDA, Reactant, Enzyme
using Breeze, Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, convert_to_device
using Oceananigans.Fields: CenterField, Field, instantiated_location
using Reactant: TracedRNumber, ConcreteRNumber
using OffsetArrays: OffsetArray

Reactant.Compiler.DUMP_LLVMIR[] = false

Reactant.set_default_backend("cpu")

# ── Inlined from Oceananigans.BoundaryConditions ──

const NoBCs = Union{Nothing, Missing, Tuple{Vararg{Nothing}}}

@inline fill_halo_event!(c, kernel!, bcs::Tuple{Any, Any}, loc, grid, args...; kwargs...) =
    kernel!(c, bcs[1], bcs[2], loc, grid, Tuple(args))
@inline fill_halo_event!(c, kernel!, bcs::Tuple{Any}, loc, grid, args...; kwargs...) =
    kernel!(c, bcs[1], loc, grid, Tuple(args))
@inline fill_halo_event!(c, ::Nothing, ::NoBCs, loc, grid, args...; kwargs...) = nothing

function fill_halo_regions!(c::OffsetArray, boundary_conditions, indices, loc, grid, args...; kwargs...)
    kernels!, bcs = boundary_conditions.kernels, boundary_conditions.ordered_bcs
    for task = 1:length(kernels!)
        @inbounds fill_halo_event!(c, kernels![task], bcs[task], loc, grid, args...; kwargs...)
    end
    return nothing
end

# ── Setup ──

Nx, Nz = 16, 8
grid = RectilinearGrid(ReactantState(); size=(Nx, Nz),
                       x=(-5000, 5000), z=(0, 10000),
                       topology=(Periodic, Flat, Bounded))

ρ      = CenterField(grid)
clock  = Clock(time=0.0, last_Δt=Inf, last_stage_Δt=Inf, iteration=ConcreteRNumber(0), stage=1)
mf     = ()

function fhr!(field::Field, positional_args...; kwargs...)
    arch = architecture(field.grid)
    args = (field.data,
            field.boundary_conditions,
            field.indices,
            instantiated_location(field),
            field.grid,
            positional_args...)

    GC.@preserve args begin
        converted_args = convert_to_device(arch, args)
        fill_halo_regions!(converted_args...; kwargs...)
    end
    return nothing
end

function loss(ρ, clock, mf)
    fhr!(ρ, clock, mf)
    return 0.0
end

@info "Compiling with $(length(mf)) fields..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(ρ, clock, mf)
