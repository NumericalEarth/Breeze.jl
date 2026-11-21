"""
    PressureLevelDiscretization

Module implementing hybrid sigma-pressure vertical coordinates via the
[`PressureLevelVerticalDiscretization`](@ref) (PLVD) grid type.

PLVD stores the hybrid coefficients ``A`` and ``B`` at the z-faces, the two-dimensional surface
pressure ``pˢ(x, y, t)`` and the cell density ``ρ``. The pressure on face ``k`` is

```math
p_k = A_k + B_k \\, pˢ ,
```

so face 1 is the ground (``A = 0``, ``B = 1``) and face ``N_z + 1`` the model top, a surface of
constant pressure (``B = 0``). The layer pressure thickness and the physical layer depth are
derived on the fly,

```math
Δp_k = (A_k - A_{k+1}) + (B_k - B_{k+1}) \\, pˢ , \\qquad Δz_k = \\frac{Δp_k}{g \\, ρ_k} ,
```

so the grid stores no three-dimensional geometry: the metrics move with ``pˢ`` and ``ρ``, which
the dynamics owns and updates.

Public API:
  - [`PressureLevelVerticalDiscretization`](@ref) — pass as the `z` argument to
    `RectilinearGrid` / `LatitudeLongitudeGrid` (and, through its panels, to a `YinYangGrid`).
  - [`PressureLevelGrid`](@ref) — the alias of grids with this vertical discretization.
  - [`layer_pressure_thickness`](@ref), [`interface_pressure`](@ref) — the pressure metrics.
"""
module PressureLevelDiscretization

# Not exported from Breeze (D408): use Breeze.PressureLevelDiscretization.PressureLevelVerticalDiscretization

using Adapt: Adapt
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS
using Oceananigans: Oceananigans, Center, Face
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Grids: Grids, AbstractVerticalCoordinate, AbstractUnderlyingGrid, Bounded, new_data
using Oceananigans.Operators: Operators, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxyᶠᶠᵃ

include("pressure_level_vertical_discretization.jl")
include("pressure_level_metrics.jl")

end # module
