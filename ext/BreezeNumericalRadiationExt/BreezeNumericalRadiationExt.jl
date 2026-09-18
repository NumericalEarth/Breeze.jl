module BreezeNumericalRadiationExt

#####
##### ecCKD radiation for AtmosphereModel through NumericalRadiation.jl
#####
##### The extension stages the grid's columns (plus a column extension above the grid top) into
##### top-down spectral column arrays, solves the longwave and shortwave fluxes of every column in
##### one kernel with NumericalRadiation's scalar ecCKD gas optics and streaming column solvers,
##### and copies the fluxes back onto the grid's `ZFaceField`s in Breeze's positive-upward
##### convention.
#####

using Breeze

using Breeze.AtmosphereModels: AtmosphereModels, RadiativeTransferModel, SurfaceRadiation,
                               EcCKDOptics, CloudScatteringTables, ColumnExtension, BackgroundAtmosphere,
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
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant

using NumericalRadiation: EcCKDTabulatedGasOpticsModel, EcCKDGasOpticsModel,
                          read_reference_ecckd_gas_optics, read_ecckd_tabulated_gas_optics,
                          reference_ecckd_definition_paths, ecrad_data_file,
                          read_cloud_scattering_table, read_ecckd_spectral_mapping,
                          ColumnAtmosphere, PhysicalConstants,
                          GasOpticsStencil, gas_optics_stencil, source_table_bracket, hydrostatic_air_moles,
                          longwave_optical_depth, shortwave_optical_depth, rayleigh_optical_depth,
                          longwave_source, TabulatedSurfaceEmission,
                          SpectralCloudOptics, effective_radius_bracket,
                          add_cloud_scattering_layer, cloud_absorption_optical_depth,
                          streaming_longwave_fluxes!, ShortwaveColumnScratch, streaming_shortwave_fluxes!

using Oceananigans.Architectures: architecture, on_architecture, array_type
using Oceananigans.Fields: ZFaceField, CenterField, AbstractField
using Oceananigans.Grids: AbstractGrid, Face, znodes
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.Utils: launch!, IterationInterval, prettysummary

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

include("spectral_columns.jl")
include("column_extension.jl")
include("cloud_optics.jl")
include("ecckd_radiative_transfer_model.jl")
include("column_staging_kernels.jl")
include("layer_optics.jl")
include("radiative_transfer_kernels.jl")
include("column_atmosphere.jl")

end # module
