using Oceananigans: ReactantState, CPU
using Oceananigans.Architectures: on_architecture
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Grids: AbstractGrid, Center
using Oceananigans.Fields: Field, interior, ZeroField

using Breeze.Thermodynamics:
    Thermodynamics,
    ThermodynamicConstants,
    ReferenceState,
    ExnerReferenceState

const ReactantArch = Union{ReactantState, Distributed{<:ReactantState}}
const ReactantGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:ReactantArch}

# ── ExnerReferenceState ──────────────────────────────────────────────────
#
# ExnerReferenceState construction calls set!/fill!/launch! on grid fields,
# all of which fail on Reactant distributed grids. Since the computation is
# purely z-column (Field{Nothing,Nothing,Center}, indexing [1,1,k]), we
# build it on a CPU copy of the grid and transfer the result.

function Thermodynamics.ExnerReferenceState(
    grid::ReactantGrid,
    constants=ThermodynamicConstants(eltype(grid));
    kwargs...
)
    cpu_grid = on_architecture(CPU(), grid)
    cpu_ref  = ExnerReferenceState(cpu_grid, constants; kwargs...)

    pᵣ = Field{Nothing, Nothing, Center}(grid)
    ρᵣ = Field{Nothing, Nothing, Center}(grid)
    πᵣ = Field{Nothing, Nothing, Center}(grid)

    copyto!(interior(pᵣ), interior(cpu_ref.pressure))
    copyto!(interior(ρᵣ), interior(cpu_ref.density))
    copyto!(interior(πᵣ), interior(cpu_ref.exner_function))

    return ExnerReferenceState(
        cpu_ref.surface_pressure,
        cpu_ref.surface_potential_temperature,
        cpu_ref.standard_pressure,
        pᵣ, ρᵣ, πᵣ)
end

# ── ReferenceState ───────────────────────────────────────────────────────
#
# Same problem: set!(field, z -> ...) fails on Reactant grids.
# Build on CPU then transfer pressure, density, temperature, and any
# non-trivial moisture fields.

function _transfer_reference_field(cpu_field::ZeroField, grid)
    return ZeroField(eltype(grid))
end

function _transfer_reference_field(cpu_field, grid)
    dev_field = Field{Nothing, Nothing, Center}(grid)
    copyto!(interior(dev_field), interior(cpu_field))
    return dev_field
end

function Thermodynamics.ReferenceState(
    grid::ReactantGrid,
    constants=ThermodynamicConstants(eltype(grid));
    kwargs...
)
    cpu_grid = on_architecture(CPU(), grid)
    cpu_ref  = ReferenceState(cpu_grid, constants; kwargs...)

    pᵣ = _transfer_reference_field(cpu_ref.pressure, grid)
    ρᵣ = _transfer_reference_field(cpu_ref.density, grid)
    Tᵣ = _transfer_reference_field(cpu_ref.temperature, grid)
    qᵛᵣ = _transfer_reference_field(cpu_ref.vapor_mass_fraction, grid)
    qˡᵣ = _transfer_reference_field(cpu_ref.liquid_mass_fraction, grid)
    qⁱᵣ = _transfer_reference_field(cpu_ref.ice_mass_fraction, grid)

    return ReferenceState(
        cpu_ref.surface_pressure,
        cpu_ref.potential_temperature,
        cpu_ref.standard_pressure,
        pᵣ, ρᵣ, Tᵣ, qᵛᵣ, qˡᵣ, qⁱᵣ)
end
