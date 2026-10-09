#####
##### All-sky (gas + cloud optics) RadiativeTransferModel: full-spectrum RRTMGP radiative transfer model
#####

using Oceananigans.Utils: launch!
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: AbstractGrid

using Breeze.AtmosphereModels: AtmosphereModels, AllSkyOptics, ConstantRadiusParticles,
                               cloud_liquid_effective_radius, cloud_ice_effective_radius,
                               grid_moisture_fractions, specific_prognostic_moisture
using Breeze.Thermodynamics: ThermodynamicConstants

using KernelAbstractions: @kernel, @index

using RRTMGP: AllSkyRadiation
using RRTMGP.AtmosphericStates: CloudState, MaxRandomOverlap

#####
##### Constructor
#####

"""
$(TYPEDSIGNATURES)

Construct an all-sky (gas + cloud) full-spectrum `RadiativeTransferModel` for the given grid.

This constructor requires that `NCDatasets` is loadable in the user environment because
RRTMGP loads lookup tables from netCDF via an extension.

# Keyword Arguments
$(full_spectrum_keywords_docstring)
- `liquid_effective_radius`: Model for cloud liquid effective radius in meters (default: `ConstantRadiusParticles(10e-6)`)
- `ice_effective_radius`: Model for cloud ice effective radius in meters (default: `ConstantRadiusParticles(30e-6)`)
- `ice_roughness`: Ice crystal roughness for cloud optics (1=smooth, 2=medium, 3=rough; default: 2)
"""
function AtmosphereModels.RadiativeTransferModel(grid::AbstractGrid,
                                                 ::AllSkyOptics,
                                                 constants::ThermodynamicConstants;
                                                 liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                                 ice_effective_radius = ConstantRadiusParticles(30e-6),
                                                 ice_roughness = 2,
                                                 keywords...)

    FT = eltype(grid)
    cloud_state = rrtmgp_cloud_state(grid, ice_roughness)
    liquid_effective_radius = materialize_effective_radius(liquid_effective_radius, FT)
    ice_effective_radius = materialize_effective_radius(ice_effective_radius, FT)

    # AllSkyRadiation(aerosol_radiation, reset_rng_seed)
    radiation_method = AllSkyRadiation(false, false)

    return full_spectrum_radiative_transfer_model(grid, radiation_method, constants, cloud_state,
                                                  liquid_effective_radius, ice_effective_radius; keywords...)
end

# Cloud state arrays, sized for the whole domain and zeroed until the first radiation update
function rrtmgp_cloud_state(grid, ice_roughness)
    FT = eltype(grid)
    context = rrtmgp_context(architecture(grid))
    ArrayType = ClimaComms.array_type(context.device)
    Nx, Ny, Nz = size(grid)
    Nc = Nx * Ny

    cloud_liquid_radius = ArrayType{FT}(undef, Nz, Nc)
    cloud_ice_radius = ArrayType{FT}(undef, Nz, Nc)
    cloud_liquid_water_path = ArrayType{FT}(undef, Nz, Nc)
    cloud_ice_water_path = ArrayType{FT}(undef, Nz, Nc)
    cloud_fraction = ArrayType{FT}(undef, Nz, Nc)
    cloud_mask_longwave = ArrayType{Bool}(undef, Nz, Nc)
    cloud_mask_shortwave = ArrayType{Bool}(undef, Nz, Nc)

    fill!(cloud_liquid_radius, zero(FT))
    fill!(cloud_ice_radius, zero(FT))
    fill!(cloud_liquid_water_path, zero(FT))
    fill!(cloud_ice_water_path, zero(FT))
    fill!(cloud_fraction, zero(FT))
    fill!(cloud_mask_longwave, false)
    fill!(cloud_mask_shortwave, false)

    return CloudState(cloud_liquid_radius,
                      cloud_ice_radius,
                      cloud_liquid_water_path,
                      cloud_ice_water_path,
                      cloud_fraction,
                      cloud_mask_longwave,
                      cloud_mask_shortwave,
                      MaxRandomOverlap(),
                      ice_roughness)
end

# Convert constant effective radii to the grid's float type
materialize_effective_radius(radius::ConstantRadiusParticles, FT) = ConstantRadiusParticles(convert(FT, radius.radius))
materialize_effective_radius(radius, FT) = radius

#####
##### Update cloud state
#####

function update_rrtmgp_cloud_state!(cloud_state::CloudState, model, liquid_effective_radius, ice_effective_radius)
    grid = model.grid
    arch = architecture(grid)

    microphysics = model.microphysics
    microphysical_fields = model.microphysical_fields
    qᵛ = specific_prognostic_moisture(model)

    # Total ρ, since the cloud water paths weight total air mass. Passed straight into the launch
    # so no local shadows the `total_density` accessor.
    launch!(arch, grid, :xyz, _update_rrtmgp_cloud_state!,
            cloud_state, grid, total_density(model.dynamics),
            microphysics, microphysical_fields, qᵛ,
            liquid_effective_radius, ice_effective_radius)

    return nothing
end

@kernel function _update_rrtmgp_cloud_state!(cloud_state, grid, total_density, microphysics, microphysical_fields, specific_prognostic_moisture,
                                             liquid_effective_radius, ice_effective_radius)
    i, j, k = @index(Global, NTuple)

    c = rrtmgp_column_index(i, j, grid.Nx)

    FT = eltype(total_density)
    kg_to_g = convert(FT, 1000)

    @inbounds begin
        ρ = total_density[i, j, k]
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]

        # Get moisture fractions from microphysics
        q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)

        # Extract liquid and ice mass fractions
        qˡ = q.liquid
        qⁱ = q.ice

        # Cloud water path in g/m² (RRTMGP convention)
        # Note: cld_path_liq/ice, cld_frac, cld_r_eff_liq/ice are RRTMGP's CloudState field names
        cloud_liquid_water_path = kg_to_g * ρ * qˡ * Δz
        cloud_ice_water_path = kg_to_g * ρ * qⁱ * Δz
        cloud_state.cld_path_liq[k, c] = cloud_liquid_water_path
        cloud_state.cld_path_ice[k, c] = cloud_ice_water_path

        # Binary cloud fraction (1 if any condensate, 0 otherwise)
        has_cloud = (qˡ + qⁱ) > zero(FT)
        cloud_state.cld_frac[k, c] = ifelse(has_cloud, one(FT), zero(FT))

        # Effective radii (convert from meters to μm for RRTMGP)
        m_to_μm = convert(FT, 1e6)
        rˡ = cloud_liquid_effective_radius(i, j, k, grid, liquid_effective_radius)
        rⁱ = cloud_ice_effective_radius(i, j, k, grid, ice_effective_radius)
        cloud_state.cld_r_eff_liq[k, c] = m_to_μm * rˡ
        cloud_state.cld_r_eff_ice[k, c] = m_to_μm * rⁱ
    end
end
