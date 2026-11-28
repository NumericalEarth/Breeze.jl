"""
$(TYPEDEF)

A turbulent flux of θˡⁱ that redistributes the radiative heating over the turbulent layer, with its energy
conversion in the TKE equation of [`TKEBasedTurbulenceClosure`](@ref).
"""
struct HeatingBudgetFlux{FT}
    ℂᑫ :: FT
    ℂᑫᴸ :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`HeatingBudgetFlux`](@ref) that applies the share `ℂᑫ` of the destabilizing flux, with the heating
exchanged over `ℂᑫᴸ` mixing lengths.
"""
function HeatingBudgetFlux(FT::DataType = Oceananigans.defaults.FloatType;
                           ℂᑫ = 0.5, ℂᑫᴸ = 8)
    0 <= ℂᑫ <= 1 || throw(ArgumentError("ℂᑫ must lie in [0, 1], got $(ℂᑫ)."))
    0 < ℂᑫᴸ < Inf || throw(ArgumentError("ℂᑫᴸ must be positive and finite (the levels of still air must not exchange), got $(ℂᑫᴸ)."))
    return HeatingBudgetFlux(convert(FT, ℂᑫ), convert(FT, ℂᑫᴸ))
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The interval of each coefficient of a [`HeatingBudgetFlux`](@ref) for a calibration, as a `NamedTuple` of
`(lower, upper)`.
"""
heating_budget_flux_bounds() = (ℂᑫ = (0.0, 1.0), ℂᑫᴸ = (1.0, 32.0))

convert_eltype(::Type{FT}, hb::HeatingBudgetFlux) where FT =
    HeatingBudgetFlux(convert(FT, hb.ℂᑫ), convert(FT, hb.ℂᑫᴸ))

Adapt.adapt_structure(to, hb::HeatingBudgetFlux) =
    HeatingBudgetFlux(adapt(to, hb.ℂᑫ), adapt(to, hb.ℂᑫᴸ))

Base.summary(hb::HeatingBudgetFlux) =
    string("HeatingBudgetFlux(ℂᑫ = ", prettysummary(hb.ℂᑫ), ", ℂᑫᴸ = ", prettysummary(hb.ℂᑫᴸ), ")")

Base.show(io::IO, hb::HeatingBudgetFlux) = print(io, summary(hb))

heating_budget_flux(closure::TKEBasedTurbulenceClosure) = closure.heating_budget_flux
heating_budget_flux(closure::TKEClosureArray) = first(closure).heating_budget_flux

validate_heating_budget_flux(::Nothing, static_stability, distribution) = nothing

function validate_heating_budget_flux(hb::HeatingBudgetFlux, static_stability, distribution)
    isnothing(distribution) && !(static_stability isa SaturationWeightedStaticStability) &&
        throw(ArgumentError("HeatingBudgetFlux needs the energetics of the applied fluxes: a subgrid_distribution, or " *
                            "static_stability = SaturationWeightedStaticStability(...)."))
    return nothing
end

@inline component_scalars(hb::HeatingBudgetFlux) = (hb.ℂᑫ, hb.ℂᑫᴸ)
component_scalar_names(::HeatingBudgetFlux, name) = (:ℂᑫ, :ℂᑫᴸ)
@inline rebuild_component(hb::HeatingBudgetFlux, scalars, n) =
    HeatingBudgetFlux(promote(scalars[n], scalars[n+1])...), n + 2

heating_budget_flux_fields(grid, ::Nothing) = NamedTuple()

heating_budget_flux_fields(grid, ::HeatingBudgetFlux) =
    (𝒬 = CenterField(grid), sᑫ = CenterField(grid), heating_work = CenterField(grid),
     dissipative_heating_rate = CenterField(grid), Jᑫ = ZFaceField(grid), Gᑫ = CenterField(grid), Gᑫᴸ = CenterField(grid))

#####
##### The heating: the tendency of θˡⁱ of each source, added to 𝒬
#####

@kernel function _add_radiative_heating!(𝒬, flux_divergence, ρ, ĉᵖΠ)
    i, j, k = @index(Global, NTuple)
    @inbounds 𝒬[i, j, k] += flux_divergence[i, j, k] / (ρ[i, j, k] * ĉᵖΠ[i, j, k])
end

function add_radiative_heating!(𝒬, model, fields)
    flux_divergence = AtmosphereModels.radiation_flux_divergence(model.radiation)
    isnothing(flux_divergence) && return nothing
    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _add_radiative_heating!, 𝒬, flux_divergence, total_density(model.dynamics), fields.cloud.ĉᵖΠ)
    return nothing
end

#####
##### The redistributed heating and the flux, column by column
#####

@kernel function _compute_heating_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    hb = closure_ij.heating_budget_flux
    ℓᵐ = closure_fields.ℓᵐ
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    rate = cl.dissipative_heating_rate
    @inbounds for k in 1:Nz
        rate[i, j, k] = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, e, velocities, closure_fields) / cl.ĉᵖΠ[i, j, k]
    end
    exchange_amplitudes!(cl.sᑫ, cl.heating_work, i, j, grid, hb.ℂᑫᴸ, TurbulentShape(e, closure_ij.minimum_tke), cl.𝒬, ρ, ℓᵐ)
    below = zero(FT)
    @inbounds begin
        cl.Jᑫ[i, j, 1] = zero(FT)
        for k in 1:Nz
            ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            below += ρΔz * (cl.𝒬[i, j, k] - cl.sᑫ[i, j, k])
            ρᶠ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ)
            δ = ℓᵐ[i, j, k+1] * (rate[i, j, k] + rate[i, j, min(k + 1, Nz)]) / 2
            cl.Jᑫ[i, j, k+1] = ifelse(k < Nz, hb.ℂᑫ * smooth_positive_part(below / ρᶠ, δ), zero(FT))
        end
    end
end

compute_heating_budget_flux!(fields, ::Nothing, closure_fields, closure, model, tracers) = nothing

function compute_heating_budget_flux!(fields, hb::HeatingBudgetFlux, closure_fields, closure, model, tracers)
    grid = model.grid
    cl = fields.cloud
    fill!(parent(cl.𝒬), 0)
    add_radiative_heating!(cl.𝒬, model, fields)
    launch!(grid.architecture, grid, :xy, _compute_heating_budget_flux!, cl, grid, closure, closure_fields,
            tracers[TKE_NAME], model.velocities, total_density(model.dynamics))
    return nothing
end

#####
##### The conversions of the flux, at the centers
#####

@inline function heating_flux_conversionsᶜᶜᶠ(i, j, k, grid, cl)
    @inbounds begin
        b_θ = (cl.b_θ[i, j, k] + cl.b_θ[i, j, k-1]) / 2
        bᶜᵛ, bⁱᵛ, bˡᵛ = latent_buoyancy_derivativesᶜᶜᶠ(i, j, k, cl)
        latent = -bᶜᵛ * cl.aᴺ[i, j, k] * cl.c_θ[i, j, k] - bⁱᵛ * cl.ηⁱ[i, j, k] * cl.c_θⁱ[i, j, k] -
                 bˡᵛ * cl.ηʷⁱ[i, j, k] * cl.c_θ[i, j, k]
        J = cl.Jᑫ[i, j, k]
    end
    return b_θ * J, latent * J
end

@inline function heating_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    lower = heating_flux_conversionsᶜᶜᶠ(i, j, max(k, 2), grid, cl)
    upper = heating_flux_conversionsᶜᶜᶠ(i, j, min(k + 1, size(grid, 3)), grid, cl)
    return map((l, u) -> half_cell_conversionᶜᶜᶜ(i, j, k, grid, ρ, l, u), lower, upper)
end

@inline store_heating_flux_conversions!(cl, i, j, k, grid, ρ) =
    store_heating_flux_conversions!(has_field(cl, Val(:Jᑫ)), cl, i, j, k, grid, ρ)

@inline store_heating_flux_conversions!(::Val{false}, cl, i, j, k, grid, ρ) = zero(grid_float_type(grid))

@inline function store_heating_flux_conversions!(::Val{true}, cl, i, j, k, grid, ρ)
    dry, latent = heating_flux_conversionsᶜᶜᶜ(i, j, k, grid, ρ, cl)
    @inbounds begin
        cl.Gᑫ[i, j, k] = dry + latent
        cl.Gᑫᴸ[i, j, k] = latent
    end
    return dry + latent
end

@inline heating_flux_conversion(cl, i, j, k) = stored_or_zero(cl, Val(:Gᑫ), i, j, k, eltype(cl.B))
@inline applied_heating_flux(cl, i, j, k) = stored_or_zero(cl, Val(:Jᑫ), i, j, k, eltype(cl.w′²))

@kernel function _add_heating_budget_flux_tendency!(Gρθ, grid, cl, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        divergence = (ℑzᵃᵃᶠ(i, j, k + 1, grid, ρ) * cl.Jᑫ[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k, grid, ρ) * cl.Jᑫ[i, j, k]) / Δzᶜᶜᶜ(i, j, k, grid)
        latent_draw = ρ[i, j, k] * cl.Gᑫᴸ[i, j, k] / cl.ĉᵖΠ[i, j, k]
        Gρθ[i, j, k] -= (divergence + latent_draw) * !inactive_cell(i, j, k, grid)
    end
end

add_heating_budget_flux_tendency!(Gⁿ, fields, ::Nothing, model) = nothing

function add_heating_budget_flux_tendency!(Gⁿ, fields, ::HeatingBudgetFlux, model)
    grid = model.grid
    name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    launch!(grid.architecture, grid, :xyz, _add_heating_budget_flux_tendency!, Gⁿ[name], grid, fields.cloud, total_density(model.dynamics))
    return nothing
end
