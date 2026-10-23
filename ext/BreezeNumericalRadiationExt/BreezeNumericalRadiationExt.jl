module BreezeNumericalRadiationExt

#####
##### ecCKD radiation for AtmosphereModel through NumericalRadiation.jl
#####
##### One kernel solves the longwave and shortwave fluxes of every column (plus an overlying atmosphere
##### above the grid top) with NumericalRadiation's scalar ecCKD gas optics and streaming column
##### solvers. The layer optics read the model's fields directly and the fluxes accumulate directly
##### into the radiation model's `ZFaceField`s: the only per-column storage is the solvers' scratch.
#####

using Breeze

using Breeze.AtmosphereModels: AtmosphereModels, RadiativeTransferModel, SurfaceRadiation, resolve_column_batch_rows,
                               EcCKDOptics, CloudScatteringTables, OverlyingAtmosphereColumn, BackgroundAtmosphere,
                               ConstantRadiusParticles, AbstractSolarPosition, ApparentSolarPosition,
                               materialize_background_atmosphere, materialize_surface_property,
                               validate_surface_fractions, constant_field_property, resolve_surface_albedos,
                               assert_bound_surface_temperature, maybe_infer_solar_position,
                               initialize_cos_zenith!, update_cos_zenith!,
                               column_index, compute_radiation_flux_divergence!, show_radiation_summary,
                               bottom_face_pressure, top_face_pressure,
                               overlying_atmosphere_faces,
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
using Oceananigans.Grids: AbstractGrid, Center, Face, znode, znodes
using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ, ∂zᶜᶜᶠ
using Oceananigans.Utils: launch!, IterationInterval, prettysummary

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

include("overlying_atmosphere.jl")
include("column_workspace.jl")
include("cloud_optics.jl")
include("ecckd_radiative_transfer_model.jl")
include("column_state.jl")
include("layer_optics.jl")
include("radiative_transfer_kernels.jl")
include("column_atmosphere.jl")

end # module
