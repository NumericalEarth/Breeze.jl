#####
##### Kernel launch parameters for the acoustic substep loop
#####
##### These helpers return `KernelParameters` describing the active launch
##### range for kernels inside the acoustic substep loop. They are the
##### counterpart to the per-side topology dispatch used in
##### `Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces`
##### (see `split_explicit_kernel_size` in
##### `split_explicit_free_surface.jl`).
#####
##### For Phase 1 (serial no-fill), all helpers are called with
##### `halo_width = 0` and return ranges equivalent to `1:N` in each
##### direction — identical to the historical `:xyz` / `:xy` launches.
##### Physical-boundary correctness without per-substep `fill_halo_regions!`
##### is delegated to topology-aware horizontal operators inside the
##### kernels (see `acoustic_operators.jl`).
#####
##### Phase 2 (distributed) will extend the active range into connected
##### halo columns by increasing `halo_width` and dispatching
##### `acoustic_kernel_size` on local-topology types. Physical-bounded
##### sides remain non-augmented and are still handled by the
##### topology-aware operators.
#####

using Oceananigans.Grids: LeftConnected, RightConnected, FullyConnected, topology
using Oceananigans.Utils: KernelParameters

# Active launch range in one horizontal direction.
#
# Fallback covers `Periodic`, `Bounded`, `Flat`, and any unknown
# non-connected topology: `1:N`. Connected-topology methods extend the
# range into the connected halo, matching Oceananigans's
# `split_explicit_kernel_size`. The `H == 0` guard keeps the
# helpers behavior-preserving on local connected topologies during the
# Phase 1 serial work, where `halo_width = 0` and every range should
# collapse to `1:N`.
@inline acoustic_kernel_size(topo, N, halo_width) = 1:N
@inline acoustic_kernel_size(::Type{FullyConnected}, N, H) = H == 0 ? (1:N) : ((-H + 2):(N + H - 1))
@inline acoustic_kernel_size(::Type{LeftConnected},  N, H) = H == 0 ? (1:N) : ((-H + 2):N)
@inline acoustic_kernel_size(::Type{RightConnected}, N, H) = H == 0 ? (1:N) :       (1:(N + H - 1))

"""
$(TYPEDSIGNATURES)

Return `KernelParameters` for an acoustic kernel that writes a
cell-centered field. The active range covers physical interior cells
plus, in distributed mode (`halo_width > 0`), connected horizontal halo
columns.
"""
@inline function acoustic_center_parameters(grid; halo_width = 0)
    TX, TY, _ = topology(grid)
    Nx, Ny, Nz = size(grid)
    return KernelParameters(acoustic_kernel_size(TX, Nx, halo_width),
                            acoustic_kernel_size(TY, Ny, halo_width),
                            1:Nz)
end

"""
$(TYPEDSIGNATURES)

Return `KernelParameters` for an acoustic kernel that writes an
`XFaceField`. The normal direction launches over the center-cell
range `1:Nx` rather than the full face range `1:Nx+1`. This is
intentional and exploits a property of the acoustic substep loop:

- On `Periodic` x, face index `Nx + 1` is identified with face 1, so
  `1:Nx` covers every face exactly once.

- On `Bounded` x, faces 1 and `Nx + 1` are physical walls where the
  perturbation momentum must remain zero by impenetrability. The init
  kernel sets `(ρu)′[1, j, k] = 0` (since `Uᴸ_outer.ρu` and `Uᴸ_stage.ρu`
  are both zero at the wall), and topology-aware horizontal operators
  return zero increments at `i = 1`, so the left wall stays at zero
  without an explicit write. The right wall face `i = Nx + 1` is never
  launched and stays at its zero-initialized value. Stage-end
  `fill_halo_regions!` on the recovered `model.momentum.ρu` re-enforces
  impenetrability for the next stage's slow-tendency computation.

Connected-topology dispatch through `acoustic_kernel_size` extends the
range into connected halo columns when `halo_width > 0` (Phase 2
distributed).
"""
@inline function acoustic_xface_parameters(grid; halo_width = 0)
    TX, TY, _ = topology(grid)
    Nx, Ny, Nz = size(grid)
    return KernelParameters(acoustic_kernel_size(TX, Nx, halo_width),
                            acoustic_kernel_size(TY, Ny, halo_width),
                            1:Nz)
end

"""
$(TYPEDSIGNATURES)

Return `KernelParameters` for an acoustic kernel that writes a
`YFaceField`. See [`acoustic_xface_parameters`](@ref) for the
explanation of why the normal direction uses the center-cell range —
the same argument applies symmetrically in y.
"""
@inline function acoustic_yface_parameters(grid; halo_width = 0)
    TX, TY, _ = topology(grid)
    Nx, Ny, Nz = size(grid)
    return KernelParameters(acoustic_kernel_size(TX, Nx, halo_width),
                            acoustic_kernel_size(TY, Ny, halo_width),
                            1:Nz)
end

"""
$(TYPEDSIGNATURES)

Return `KernelParameters` for an acoustic kernel that writes a
`ZFaceField`. The bottom wall (`k = 1`) and top wall (`k = Nz + 1`)
are handled by `peripheral_node` / `ifelse` guards inside the
relevant kernels (`get_coefficient`, `ℑbzᵃᵃᶠ`, etc.); this helper
launches over the centered z-range `1:Nz`. The vertical tridiagonal
RHS kernel uses [`acoustic_column_parameters`](@ref) and iterates
over `k` internally.
"""
@inline function acoustic_zface_parameters(grid; halo_width = 0)
    TX, TY, _ = topology(grid)
    Nx, Ny, Nz = size(grid)
    return KernelParameters(acoustic_kernel_size(TX, Nx, halo_width),
                            acoustic_kernel_size(TY, Ny, halo_width),
                            1:Nz)
end

"""
$(TYPEDSIGNATURES)

Return `KernelParameters` for an acoustic column kernel (launched over
the horizontal `(i, j)` plane only; the kernel iterates over k
internally). Used for `_build_predictors_and_vertical_rhs!` and any
column-wise vertical solver.
"""
@inline function acoustic_column_parameters(grid; halo_width = 0)
    TX, TY, _ = topology(grid)
    Nx, Ny, _ = size(grid)
    return KernelParameters(acoustic_kernel_size(TX, Nx, halo_width),
                            acoustic_kernel_size(TY, Ny, halo_width))
end
