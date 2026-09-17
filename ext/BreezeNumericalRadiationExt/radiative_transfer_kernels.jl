#####
##### Column radiative transfer kernels
#####
##### Kernel C solves the longwave and shortwave fluxes of every staged column with
##### NumericalRadiation's streaming solvers, one column per work item, reading the layer optics
##### through the functors of `layer_optics.jl`. Kernel D copies the solved fluxes back onto the
##### grid's faces with Breeze's sign convention. The update assembles the staging kernels, the
##### two kernels here, and the flux divergence.
#####

# The per-column scratch of the shortwave adding method: views of row `c` of the scratch arrays
@inline column_scratch(shortwave, c) = ShortwaveColumnScratch(view(shortwave.reflectance, c, :),
                                                              view(shortwave.transmittance, c, :),
                                                              view(shortwave.direct_reflectance, c, :),
                                                              view(shortwave.direct_diffuse_transmittance, c, :),
                                                              view(shortwave.direct_transmittance, c, :),
                                                              view(shortwave.stack_albedo, c, :),
                                                              view(shortwave.source, c, :))

"""
$(TYPEDSIGNATURES)

Solve the longwave and shortwave fluxes of every staged column of `rtm` (kernel C): per column,
fill the gas optics stencils and Planck brackets, then stream the g points of the no-scattering
longwave solver and of the two-stream shortwave solver through the column's layer optics,
writing the fluxes into the column rows of `rtm.atmospheric_state`. At night the shortwave
irradiance `S₀ max(μ₀, 0)` is zero and the shortwave fluxes vanish without a branch.
"""
function solve_spectral_columns!(rtm::EcCKDRadiativeTransferModel, grid)
    arch = architecture(grid)
    columns = rtm.atmospheric_state
    longwave = rtm.longwave_solver
    shortwave = rtm.shortwave_solver
    surface = rtm.surface_radiation

    launch!(arch, grid, :xy, _spectral_column_fluxes!,
            columns, longwave.gas_model, longwave.cloud, shortwave.cloud,
            longwave.mole_fractions, shortwave.solar_constant,
            surface.surface_temperature, surface.surface_emissivity,
            surface.direct_surface_albedo, surface.diffuse_surface_albedo,
            rtm.liquid_effective_radius.radius, rtm.ice_effective_radius.radius, grid)

    return nothing
end

@kernel function _spectral_column_fluxes!(columns, gas_model, longwave_cloud, shortwave_cloud,
                                          mole_fractions, solar_constant,
                                          surface_temperature, surface_emissivity,
                                          direct_surface_albedo, diffuse_surface_albedo,
                                          liquid_radius, ice_radius, grid)
    i, j = @index(Global, NTuple)

    FT = eltype(columns)
    c = column_index(i, j, grid.Nx)
    N = number_of_layers(columns)
    Ngˡʷ = length(gas_model.longwave_weights)
    Ngˢʷ = length(gas_model.shortwave_weights)

    stage_column_stencils!(columns, gas_model, c, N)

    @inbounds begin
        T₀ = surface_temperature[i, j, 1]
        ε = surface_emissivity[i, j, 1]
        α_direct = direct_surface_albedo[i, j, 1]
        α_diffuse = diffuse_surface_albedo[i, j, 1]
        μ₀ = columns.cos_zenith[c]
    end

    # Longwave: no downwelling flux enters the top of the atmosphere; the surface emits
    # `ε B(T₀)` per g point and reflects `1 - ε` of the downwelling flux
    longwave_liquid, longwave_ice = cloud_phases(longwave_cloud)
    longwave = LongwaveLayerOptics(gas_model, columns, mole_fractions, longwave_liquid, longwave_ice,
                                   effective_radius_bracket(longwave_liquid, liquid_radius),
                                   effective_radius_bracket(longwave_ice, ice_radius), c)
    surface_emission = TabulatedSurfaceEmission(gas_model, T₀; emissivity = ε)

    streaming_longwave_fluxes!(view(columns.longwave_up, c, :), view(columns.longwave_down, c, :),
                               longwave, surface_emission, 1 - ε, zero(FT),
                               gas_model.longwave_weights, Ngˡʷ, N,
                               view(columns.transmittance, c, :), view(columns.source_up, c, :))

    # Shortwave: the horizontal irradiance at the top of the atmosphere is `S₀ μ₀`, zero at night
    shortwave_liquid, shortwave_ice = cloud_phases(shortwave_cloud)
    shortwave = ShortwaveLayerOptics(gas_model, columns, mole_fractions, shortwave_liquid, shortwave_ice,
                                     effective_radius_bracket(shortwave_liquid, liquid_radius),
                                     effective_radius_bracket(shortwave_ice, ice_radius), c)

    streaming_shortwave_fluxes!(view(columns.shortwave_up, c, :), view(columns.shortwave_down, c, :),
                                shortwave, μ₀, solar_constant * max(μ₀, 0), α_direct, α_diffuse,
                                gas_model.shortwave_weights, Ngˢʷ, N, column_scratch(columns.shortwave, c))
end

"""
$(TYPEDSIGNATURES)

Copy the solved column fluxes onto the four `ZFaceField`s of `rtm` (kernel D): grid face `k` is
column interface `N + 2 - k`, and the downwelling fluxes change sign to Breeze's positive-upward
convention.
"""
function copy_spectral_fluxes!(rtm, grid)
    arch = architecture(grid)
    columns = rtm.atmospheric_state
    Nx, Ny, Nz = size(grid)

    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_spectral_fluxes!,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            columns, grid)

    return nothing
end

@kernel function _copy_spectral_fluxes!(upwelling_longwave, downwelling_longwave, upwelling_shortwave, downwelling_shortwave,
                                        columns, grid)
    i, j, k = @index(Global, NTuple)

    N = number_of_layers(columns)
    c = column_index(i, j, grid.Nx)
    kᶠ = N + 2 - k

    @inbounds begin
        upwelling_longwave[i, j, k] = columns.longwave_up[c, kᶠ]
        downwelling_longwave[i, j, k] = -columns.longwave_down[c, kᶠ]
        upwelling_shortwave[i, j, k] = columns.shortwave_up[c, kᶠ]
        downwelling_shortwave[i, j, k] = -columns.shortwave_down[c, kᶠ]
    end
end

"""
$(TYPEDSIGNATURES)

Update the ecCKD radiative fluxes of `rtm` from the current state of `model`: stage the grid's
columns and their extension, solve every column, copy the fluxes onto the grid and compute the
flux divergence.
"""
function AtmosphereModels._update_radiation!(rtm::EcCKDRadiativeTransferModel, model)
    assert_bound_surface_temperature(rtm)
    grid = model.grid
    columns = rtm.atmospheric_state

    update_cos_zenith!(columns.cos_zenith, rtm.solar_position, grid, model.clock)

    stage_spectral_columns!(columns, model, rtm.background_atmosphere)
    extend_spectral_columns!(columns, columns.extension, model)
    solve_spectral_columns!(rtm, grid)
    copy_spectral_fluxes!(rtm, grid)
    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end
