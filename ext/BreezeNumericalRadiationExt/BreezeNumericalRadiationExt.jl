module BreezeNumericalRadiationExt

#####
##### ecCKD radiation for AtmosphereModel through NumericalRadiation.jl
#####
##### The extension stages the grid's columns (plus a column extension above the grid top) into
##### top-down spectral column arrays, solves the longwave and shortwave fluxes of every column
##### with NumericalRadiation's ecCKD gas optics and two-stream solvers, and copies the fluxes back
##### onto the grid's `ZFaceField`s in Breeze's positive-upward convention.
#####

using Breeze

using Breeze.AtmosphereModels: AtmosphereModels, RadiativeTransferModel, SurfaceRadiation,
                               EcCKDOptics, ColumnExtension, BackgroundAtmosphere,
                               ConstantRadiusParticles, AbstractSolarPosition, ApparentSolarPosition,
                               materialize_background_atmosphere, materialize_surface_property,
                               validate_surface_fractions, constant_field_property, resolve_surface_albedos,
                               assert_bound_surface_temperature, maybe_infer_solar_position,
                               initialize_cos_zenith!, update_cos_zenith!,
                               column_index, compute_radiation_flux_divergence!, show_radiation_summary,
                               bottom_face_pressure, top_face_pressure,
                               bottom_face_temperature, top_face_temperature,
                               column_extension_faces,
                               dynamics_pressure, total_density, specific_prognostic_moisture,
                               grid_moisture_fractions
using Breeze.Thermodynamics: ThermodynamicConstants

using NumericalRadiation: EcCKDTabulatedGasOpticsModel, EcCKDGasOpticsModel,
                          read_reference_ecckd_gas_optics, read_ecckd_tabulated_gas_optics,
                          ColumnAtmosphere, RadiativeFluxes, LongwaveOptics, ShortwaveOptics,
                          CloudlessLongwave, CloudlessShortwave,
                          LongwaveBoundaryConditions, ShortwaveBoundaryConditions,
                          optical_properties!, radiative_fluxes!,
                          surface_longwave_emission, longwave_source, source_table_bracket

using Oceananigans.Architectures: architecture, on_architecture, array_type
using Oceananigans.Fields: ZFaceField, CenterField, AbstractField
using Oceananigans.Grids: AbstractGrid, Face, znodes
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.Utils: launch!, IterationInterval, prettysummary

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS

include("spectral_columns.jl")
include("column_extension.jl")
include("ecckd_radiative_transfer_model.jl")
include("column_staging_kernels.jl")
include("host_update.jl")

end # module
