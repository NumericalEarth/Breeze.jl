# MWE: fill_halo_regions! with all Oceananigans types inlined
#
# Reproduces debug_raise_mwe.jl without importing Oceananigans/Breeze.
# Types, structs, and kernels are ported directly from Oceananigans source.
# Goal: reproduce the LLVM operand count mismatch on Julia 1.12.

using CUDA, Reactant, Enzyme, KernelAbstractions, OffsetArrays
using Reactant: ConcreteRNumber

Reactant.set_default_backend("cpu")
CUDA.allowscalar(true)

# ═══════════════════════════════════════════════════════════════════════
# Types (ported from Oceananigans, matching struct layouts exactly)
# ═══════════════════════════════════════════════════════════════════════

# --- Topology singletons (src/Grids/grid_utils.jl) ---
struct Periodic end
struct Flat end
struct Bounded end

# --- Location singletons (src/Grids/grid_utils.jl) ---
struct Center end

# --- Boundary condition classifications (boundary_condition_classifications.jl) ---
abstract type AbstractBoundaryConditionClassification end
struct BCPeriodic <: AbstractBoundaryConditionClassification end
struct Flux      <: AbstractBoundaryConditionClassification end

# --- BoundaryCondition (boundary_condition.jl:8-11) ---
struct BoundaryCondition{C<:AbstractBoundaryConditionClassification, T}
    classification :: C
    condition      :: T
end

const PBC = BoundaryCondition{<:BCPeriodic}
const FBC = BoundaryCondition{<:Flux}

PeriodicBoundaryCondition() = BoundaryCondition(BCPeriodic(), nothing)
NoFluxBoundaryCondition()   = BoundaryCondition(Flux(), nothing)

# --- Clock (src/TimeSteppers/clock.jl:17-23) ---
mutable struct Clock{TT, DT, IT, S}
    time          :: TT
    last_Δt       :: DT
    last_stage_Δt :: DT
    iteration     :: IT
    stage         :: S
end

# --- Vertical discretization (src/Grids/vertical_discretization.jl:25-30) ---
struct StaticVerticalDiscretization{C, D, E, F}
    cᵃᵃᶠ :: C
    cᵃᵃᶜ :: D
    Δᵃᵃᶠ :: E
    Δᵃᵃᶜ :: F
end

# --- RectilinearGrid (src/Grids/rectilinear_grid.jl:3-25) ---
struct RectilinearGrid{FT, TX, TY, TZ, CZ, FX, FY, VX, VY, Arch}
    architecture :: Arch
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    Lx :: FT
    Ly :: FT
    Lz :: FT
    Δxᶠᵃᵃ :: FX
    Δxᶜᵃᵃ :: FX
    xᶠᵃᵃ  :: VX
    xᶜᵃᵃ  :: VX
    Δyᵃᶠᵃ :: FY
    Δyᵃᶜᵃ :: FY
    yᵃᶠᵃ  :: VY
    yᵃᶜᵃ  :: VY
    z      :: CZ
end

# --- FieldBoundaryConditions (field_boundary_conditions.jl:59-69) ---
mutable struct FieldBoundaryConditions{W, E, S, N, B, T, I, K, O}
    west     :: W
    east     :: E
    south    :: S
    north    :: N
    bottom   :: B
    top      :: T
    immersed :: I
    kernels  :: K
    ordered_bcs :: O
end

# ═══════════════════════════════════════════════════════════════════════
# Kernels (exact code from Oceananigans)
# ═══════════════════════════════════════════════════════════════════════

# fill_halo_regions_periodic.jl:5-13
@kernel function _fill_periodic_west_and_east_halo!(c, west_bc, east_bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    H = grid.Hx
    N = grid.Nx
    @inbounds for i = 1:H
        parent(c)[i, j, k]     = parent(c)[N+i, j, k]
        parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k]
    end
end

# fill_halo_regions_flux.jl:15-16
@inline _fill_flux_bottom_halo!(i, j, k, grid, c) = @inbounds c[i, j, 1-k] = c[i, j, k]
@inline _fill_flux_top_halo!(i, j, k, grid, c)    = @inbounds c[i, j, grid.Nz+k] = c[i, j, grid.Nz+1-k]

# fill_halo_regions_flux.jl:26-27
@inline _fill_bottom_halo!(i, j, grid, c, ::FBC, args...) = _fill_flux_bottom_halo!(i, j, 1, grid, c)
@inline _fill_top_halo!(i, j, grid, c, ::FBC, args...)    = _fill_flux_top_halo!(i, j, 1, grid, c)

# fill_halo_regions.jl:78-82
@kernel function _fill_bottom_and_top_halo!(c, bottom_bc, top_bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bottom_bc, loc, args...)
       _fill_top_halo!(i, j, grid, c, top_bc,    loc, args...)
end

# ═══════════════════════════════════════════════════════════════════════
# Fill halo dispatch (fill_halo_regions.jl:34-38)
# ═══════════════════════════════════════════════════════════════════════

const NoBCs = Union{Nothing, Missing, Tuple{Vararg{Nothing}}}

@inline fill_halo_event!(c, kernel!, bcs::Tuple{Any, Any}, loc, grid, args...) =
    kernel!(c, bcs[1], bcs[2], loc, grid, Tuple(args))

@inline fill_halo_event!(c, ::Nothing, ::NoBCs, loc, grid, args...) = nothing

function fill_halo_regions!(c, kernels_list, ordered_bcs, loc, grid, args...)
    for task = 1:length(kernels_list)
        @inbounds fill_halo_event!(c, kernels_list[task], ordered_bcs[task], loc, grid, args...)
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════
# Object construction (matching debug_raise_mwe.jl)
#
# RectilinearGrid(ReactantState(); size=(16, 8),
#     x=(-5000, 5000), z=(0, 10000), topology=(Periodic, Flat, Bounded))
# ═══════════════════════════════════════════════════════════════════════

Nx, Nz = 16, 8
Hx, Hy, Hz = 3, 0, 3   # default halos: 3 for non-Flat, 0 for Flat

Lx, Lz = 10000.0, 10000.0
Δx, Δz = Lx / Nx, Lz / Nz

# Coordinate ranges (regular spacing, wrapped in OffsetArrays with -H offset)
xf = OffsetArray(range(-5000.0, stop=5000.0, length=Nx+1), -Hx)
xc = OffsetArray(range(-5000.0 + Δx/2, step=Δx, length=Nx), -Hx)
zf = OffsetArray(range(0.0, stop=Lz, length=Nz+1), -Hz)
zc = OffsetArray(range(Δz/2, step=Δz, length=Nz), -Hz)

z_disc = StaticVerticalDiscretization(zf, zc, Δz, Δz)

struct ReactantState end

grid = RectilinearGrid{Float64, Periodic, Flat, Bounded,
    typeof(z_disc), Float64, Float64, typeof(xf), typeof(xc), ReactantState}(
    ReactantState(),
    Nx, 1, Nz, Hx, Hy, Hz,
    Lx, 0.0, Lz,
    Δx, Δx, xf, xc,
    0.0, 0.0, nothing, nothing,
    z_disc)

# Field data: OffsetArray{Float64, 3, ConcreteRArray{Float64, 3}}
# total_size = (Nx+2Hx, 1, Nz+2Hz) = (22, 1, 14)
# offset axes = (1-Hx:Nx+Hx, 1:1, 1-Hz:Nz+Hz) = (-2:19, 1:1, -2:11)
raw = Reactant.to_rarray(zeros(Nx + 2Hx, 1, Nz + 2Hz))
c   = OffsetArray(raw, 1-Hx:Nx+Hx, 1:1, 1-Hz:Nz+Hz)

# Boundary conditions for CenterField on (Periodic, Flat, Bounded)
pbc = PeriodicBoundaryCondition()
fbc = NoFluxBoundaryCondition()

bcs = FieldBoundaryConditions(
    pbc,     # west   (Periodic)
    pbc,     # east   (Periodic)
    nothing, # south  (Flat → nothing)
    nothing, # north  (Flat → nothing)
    fbc,     # bottom (Bounded, Center → NoFlux)
    fbc,     # top    (Bounded, Center → NoFlux)
    fbc,     # immersed (default NoFlux)
    nothing, nothing)

# Clock (matching debug_raise_mwe.jl:26)
clock = Clock(0.0, Inf, Inf, ConcreteRNumber(0), 1)
mf = ()

loc = (Center(), Center(), Center())

# Pre-computed kernel ordering from permute_boundary_conditions:
#   1. SouthAndNorth (nothing, nothing) → no-op
#   2. BottomAndTop  (fbc, fbc)         → _fill_bottom_and_top_halo!, ndrange=(Nx, 1)
#   3. WestAndEast   (pbc, pbc)         → _fill_periodic_west_and_east_halo!, ndrange=(1, Nz+2Hz)

ordered_bcs_list = (
    (nothing, nothing),
    (fbc, fbc),
    (pbc, pbc),
)

# ═══════════════════════════════════════════════════════════════════════
# Loss function & Reactant compilation
# ═══════════════════════════════════════════════════════════════════════

function loss(c, clock, mf)
    backend = KernelAbstractions.get_backend(parent(c))
    args = (clock, mf)

    # Task 2: BottomAndTop — flux BC
    _fill_bottom_and_top_halo!(backend)(c, fbc, fbc, loc, grid, args; ndrange=(Nx, 1))
    KernelAbstractions.synchronize(backend)

    # Task 3: WestAndEast — periodic BC
    _fill_periodic_west_and_east_halo!(backend)(c, pbc, pbc, loc, grid, args; ndrange=(1, Nz + 2Hz))
    KernelAbstractions.synchronize(backend)

    return 0.0
end

@info "Compiling..."
@time compiled = Reactant.@compile raise=true raise_first=true loss(c, clock, mf)

@info "Running compiled function..."
compiled(c, clock, mf)
@info "Done."
