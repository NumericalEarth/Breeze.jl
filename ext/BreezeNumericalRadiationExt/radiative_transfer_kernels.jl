#####
##### Column radiative transfer
#####
##### The columns are solved in batches of whole rows (`column_workspace.jl`). Per batch, a layer pass
##### stores the per-layer state with one work item per layer (`layer_optics.jl`), and one kernel
##### solves the longwave and shortwave fluxes with NumericalRadiation's streaming solvers, one column
##### per work item, reading the layer optics through the functors of `layer_optics.jl` and
##### accumulating the fluxes directly into the radiation model's flux fields.
#####

"""
$(TYPEDEF)

The fluxes on column `(i, j)` of a `ZFaceField` presented to NumericalRadiation's solvers as a
top-down vector of the `N + 1` column interfaces, each stream positive in its own direction:
column interface `k` is grid face `N + 2 - k`, `sign` converts between NumericalRadiation's
convention and Breeze's positive-upward one, and the `Nₑ` interfaces above the grid top (whose
fluxes only the solvers' sweeps need, and which they never read back) are not stored.
"""
struct FaceFluxColumn{FT, D} <: AbstractVector{FT}
    data :: D
    i :: Int
    j :: Int
    extension_layers :: Int
    layers :: Int
    sign :: FT
end

Base.size(flux::FaceFluxColumn) = (flux.layers + 1,)

@inline grid_face(flux::FaceFluxColumn, k) = clamp(flux.layers + 2 - k, 1, flux.layers + 1 - flux.extension_layers)

@inline Base.getindex(flux::FaceFluxColumn, k::Int) = @inbounds flux.sign * flux.data[flux.i, flux.j, grid_face(flux, k)]

# Above the grid top the write leaves the grid's top face as it is
@inline function Base.setindex!(flux::FaceFluxColumn, value, k::Int)
    kᶠ = grid_face(flux, k)
    @inbounds begin
        stored = flux.data[flux.i, flux.j, kᶠ]
        flux.data[flux.i, flux.j, kᶠ] = ifelse(k > flux.extension_layers, flux.sign * value, stored)
    end
    return value
end

"""
$(TYPEDSIGNATURES)

Solve the longwave and shortwave fluxes of the `batch_size = (Nx, rows)` columns of the batch of
`state`, whose layer state the layer pass has stored in `rtm`'s workspace: per column, stream the g points of the no-scattering longwave solver and of the two-stream
shortwave solver through the column's layer optics, accumulating the fluxes into the four flux
fields of `rtm`. At night the shortwave irradiance `S₀ max(μ₀, 0)` is zero and the shortwave
fluxes vanish without a branch.
"""
function solve_columns!(rtm::EcCKDRadiativeTransferModel, state, batch_size)
    grid = state.grid
    longwave = rtm.longwave_solver
    shortwave = rtm.shortwave_solver
    surface = rtm.surface_radiation

    launch!(architecture(grid), grid, batch_size, _column_radiative_fluxes!,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            rtm.atmospheric_state, state, longwave.gas_model, longwave.cloud, shortwave.cloud,
            longwave.mole_fractions, shortwave.solar_constant,
            surface.surface_temperature, surface.surface_emissivity,
            surface.direct_surface_albedo, surface.diffuse_surface_albedo,
            rtm.liquid_effective_radius.radius, rtm.ice_effective_radius.radius)

    return nothing
end

@kernel function _column_radiative_fluxes!(upwelling_longwave, downwelling_longwave,
                                           upwelling_shortwave, downwelling_shortwave,
                                           workspace, state, gas_model, longwave_cloud, shortwave_cloud,
                                           mole_fractions, solar_constant,
                                           surface_temperature, surface_emissivity,
                                           direct_surface_albedo, diffuse_surface_albedo,
                                           liquid_radius, ice_radius)
    i, jᵇ = @index(Global, NTuple)

    FT = eltype(workspace)
    Nx = state.grid.Nx
    j = jᵇ + state.row_offset
    c = batch_column(i, jᵇ, Nx)
    N = number_of_layers(state)
    Nₑ = number_of_extension_layers(state.extension)

    @inbounds begin
        T₀ = surface_temperature[i, j, 1]
        ε = surface_emissivity[i, j, 1]
        α_direct = direct_surface_albedo[i, j, 1]
        α_diffuse = diffuse_surface_albedo[i, j, 1]
        μ₀ = workspace.cos_zenith[column_index(i, j, Nx)]
    end

    # Longwave: no downwelling flux enters the top of the atmosphere; the surface emits
    # `ε B(T₀)` per g point and reflects `1 - ε` of the downwelling flux
    ℐ_lw_up = FaceFluxColumn(upwelling_longwave, i, j, Nₑ, N, one(FT))
    ℐ_lw_dn = FaceFluxColumn(downwelling_longwave, i, j, Nₑ, N, -one(FT))
    longwave = LayerOptics(Longwave(), gas_model, workspace, mole_fractions, longwave_cloud, liquid_radius, ice_radius, c)
    surface_emission = TabulatedSurfaceEmission(gas_model, T₀; emissivity = ε)
    transmittance, source_up = longwave_scratch(workspace, c)

    streaming_longwave_fluxes!(ℐ_lw_up, ℐ_lw_dn,
                               longwave, surface_emission, 1 - ε, zero(FT),
                               gas_model.longwave_weights, length(gas_model.longwave_weights), N,
                               transmittance, source_up)

    # Shortwave: the horizontal irradiance at the top of the atmosphere is `S₀ μ₀`, zero at night
    ℐ_sw_up = FaceFluxColumn(upwelling_shortwave, i, j, Nₑ, N, one(FT))
    ℐ_sw_dn = FaceFluxColumn(downwelling_shortwave, i, j, Nₑ, N, -one(FT))
    shortwave = LayerOptics(Shortwave(), gas_model, workspace, mole_fractions, shortwave_cloud, liquid_radius, ice_radius, c)

    streaming_shortwave_fluxes!(ℐ_sw_up, ℐ_sw_dn,
                                shortwave, μ₀, solar_constant * max(μ₀, 0), α_direct, α_diffuse,
                                gas_model.shortwave_weights, length(gas_model.shortwave_weights), N,
                                shortwave_scratch(workspace, c))
end

"""
$(TYPEDSIGNATURES)

Update the ecCKD radiative fluxes of `rtm` from the current state of `model`. Batch by batch:
integrate the extension pressures above the columns' tops, store the per-layer state, and solve
the columns into the flux fields. Then compute the flux divergence.
"""
function AtmosphereModels._update_radiation!(rtm::EcCKDRadiativeTransferModel, model)
    assert_bound_surface_temperature(rtm)
    grid = model.grid
    workspace = rtm.atmospheric_state
    gas_model = rtm.longwave_solver.gas_model

    update_cos_zenith!(workspace.cos_zenith, rtm.solar_position, grid, model.clock)

    for (row_offset, batch_size) in column_batches(workspace, grid)
        state = ColumnState(rtm, model, row_offset)
        compute_extension_pressure!(state, batch_size)
        compute_layer_state!(workspace, state, gas_model, batch_size)
        solve_columns!(rtm, state, batch_size)
    end

    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end
