#####
##### GPU/architecture support for P3 container structs
#####
##### When ice/rain integrals are tabulated (TabulatedFunction3D / TabulatedFunction1D),
##### the lookup table arrays must be transferred to the GPU. Scalar fields and
##### singleton integral types pass through unchanged.
#####
##### Most container types just walk every field with `Adapt.adapt` and
##### `on_architecture`. The `@walk_fields_for_gpu` macro below generates both
##### methods so we don't repeat the field list twice per type. A field-by-field
##### walk is equivalent to passing scalars through unchanged because both
##### `Adapt.adapt` and `on_architecture` fall back to identity for types without
##### specific methods (`adapt_storage(to, x) = x`, `on_architecture(arch, a) = a`).
#####

using Adapt: Adapt
using Oceananigans.Architectures: on_architecture

"""
    @walk_fields_for_gpu T

Generate `Adapt.adapt_structure` and `Oceananigans.Architectures.on_architecture`
methods for `T` that walk every field of `T` and reconstruct via the positional
constructor. `T` must already be defined when the macro is expanded.
"""
macro walk_fields_for_gpu(T)
    fields = fieldnames(getfield(__module__, T))
    adapt_args = [:(Adapt.adapt(to, x.$f)) for f in fields]
    on_arch_args = [:(on_architecture(arch, x.$f)) for f in fields]
    return esc(quote
        Adapt.adapt_structure(to, x::$T) = $T($(adapt_args...))
        Oceananigans.Architectures.on_architecture(arch, x::$T) = $T($(on_arch_args...))
    end)
end

# TabulatedFunction6D is the one asymmetric case: `Adapt.adapt` must replace
# `func` with `nothing` (Julia functions are not callable on the GPU inside
# kernels), while `on_architecture` preserves it so CPU-side copies remain
# callable. This matches Oceananigans' own TabulatedFunction adapter.
Adapt.adapt_structure(to, x::TabulatedFunction6D) =
    TabulatedFunction6D(nothing, Adapt.adapt(to, x.table), x.range, x.inverse_Δ)

Oceananigans.Architectures.on_architecture(arch, x::TabulatedFunction6D) =
    TabulatedFunction6D(x.func, on_architecture(arch, x.table), x.range, x.inverse_Δ)

@walk_fields_for_gpu FortranTabulatedFunction5D
@walk_fields_for_gpu FortranTabulatedFunction6D
@walk_fields_for_gpu FortranTabulatedFunction3
@walk_fields_for_gpu IceFallSpeed
@walk_fields_for_gpu IceDeposition
@walk_fields_for_gpu IceBulkProperties
@walk_fields_for_gpu IceCollection
@walk_fields_for_gpu IceSixthMoment
@walk_fields_for_gpu IceLambdaLimiter
@walk_fields_for_gpu IceRainCollection
@walk_fields_for_gpu P3IceIntegralsTable
@walk_fields_for_gpu P3RainIceCollectionTable
@walk_fields_for_gpu P3ThreeMomentShapeTable
@walk_fields_for_gpu P3LookupTables
@walk_fields_for_gpu IceProperties
@walk_fields_for_gpu RainProperties
@walk_fields_for_gpu PredictedParticlePropertiesMicrophysics
