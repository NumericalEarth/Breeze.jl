#####
##### GPU/architecture support for P3 container structs
#####
##### The ice integrals (5D/6D Fortran tables) and rain integrals (1D tables
##### built from quadrature) hold lookup arrays that must be transferred to the
##### GPU. Scalar fields and singleton integral types pass through unchanged.
#####
##### Most container types just walk every field with `Adapt.adapt` and
##### `on_architecture`. The `@adapt_architecture` macro below generates both
##### methods so we don't repeat the field list twice per type. A field-by-field
##### walk is equivalent to passing scalars through unchanged because both
##### `Adapt.adapt` and `on_architecture` fall back to identity for types without
##### specific methods (`adapt_storage(to, x) = x`, `on_architecture(arch, a) = a`).
#####

using Adapt: Adapt
using Oceananigans.Architectures: on_architecture

"""
    @adapt_architecture T

Generate `Adapt.adapt_structure` and `Oceananigans.Architectures.on_architecture`
methods for `T` that walk every field of `T` and reconstruct via the positional
constructor. `T` must already be defined when the macro is expanded.
"""
macro adapt_architecture(T)
    fields = fieldnames(getfield(__module__, T))
    adapt_args = [:(Adapt.adapt(to, x.$f)) for f in fields]
    on_arch_args = [:(on_architecture(arch, x.$f)) for f in fields]
    return esc(quote
        Adapt.adapt_structure(to, x::$T) = $T($(adapt_args...))
        Oceananigans.Architectures.on_architecture(arch, x::$T) = $T($(on_arch_args...))
    end)
end

@adapt_architecture TabulatedFunction6D
@adapt_architecture RimeDensityIndexedTable5D
@adapt_architecture RimeDensityIndexedTable6D
@adapt_architecture IceFallSpeed
@adapt_architecture IceDeposition
@adapt_architecture IceBulkProperties
@adapt_architecture IceCollection
@adapt_architecture IceLambdaLimiter
@adapt_architecture IceRainCollection
@adapt_architecture P3IceIntegralsTable
@adapt_architecture P3RainIceCollectionTable
@adapt_architecture P3LookupTables
@adapt_architecture IceProperties
@adapt_architecture RainProperties
@adapt_architecture PredictedParticlePropertiesMicrophysics
