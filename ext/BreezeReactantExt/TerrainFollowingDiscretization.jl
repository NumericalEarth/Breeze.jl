using Oceananigans: Oceananigans
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization,
                                             LinearDecay, TwoLevelDecay, TwoLevelBasis

#####
##### Moving a terrain-following LatitudeLongitudeGrid onto ReactantState
#####
##### Oceananigans' Reactant extension builds a `LatitudeLongitudeGrid` on the CPU and then moves
##### it to `ReactantState` field by field through `_to_reactant`, which materializes every array
##### (including `StepRangeLen`s) into a `ConcreteRArray` and rebuilds each vertical coordinate it
##### knows explicitly (see `_to_reactant(::StaticVerticalDiscretization)` in
##### OceananigansReactantExt/Architectures.jl). Follow that pattern for the terrain-following
##### coordinate and its formulations so that `LatitudeLongitudeGrid(ReactantState(), ...; z)`
##### and `on_architecture(ReactantState(), grid)` work on a terrain-following grid.
#####

const OceananigansReactantExt = Base.get_extension(Oceananigans, :OceananigansReactantExt)
using .OceananigansReactantExt.Architectures: _to_reactant

OceananigansReactantExt.Architectures._to_reactant(z::TerrainFollowingVerticalDiscretization) =
    TerrainFollowingVerticalDiscretization(_to_reactant(z.cᵃᵃᶠ),
                                           _to_reactant(z.cᵃᵃᶜ),
                                           _to_reactant(z.Δᵃᵃᶠ),
                                           _to_reactant(z.Δᵃᵃᶜ),
                                           _to_reactant(z.formulation))

OceananigansReactantExt.Architectures._to_reactant(f::LinearDecay) =
    LinearDecay(f.z_top,
                _to_reactant(f.h),
                _to_reactant(f.∂x_h),
                _to_reactant(f.∂y_h))

OceananigansReactantExt.Architectures._to_reactant(f::TwoLevelDecay) =
    TwoLevelDecay(f.z_top, f.large_scale_height, f.small_scale_height,
                  _to_reactant(f.h₁), _to_reactant(f.h₂),
                  _to_reactant(f.∂x_h₁), _to_reactant(f.∂x_h₂),
                  _to_reactant(f.∂y_h₁), _to_reactant(f.∂y_h₂),
                  _to_reactant(f.basis))

OceananigansReactantExt.Architectures._to_reactant(b::TwoLevelBasis) =
    TwoLevelBasis(_to_reactant(b.b₁ᶜ), _to_reactant(b.b₁ᶠ),
                  _to_reactant(b.b₂ᶜ), _to_reactant(b.b₂ᶠ),
                  _to_reactant(b.∂b₁ᶜ), _to_reactant(b.∂b₁ᶠ),
                  _to_reactant(b.∂b₂ᶜ), _to_reactant(b.∂b₂ᶠ))
