# Diagnostics use the closure's own kernels and average instantaneous quantities over the
# same windows as the scored profiles. In particular, mean(K * gradient) is not formed as
# mean(K) * gradient(mean(profile)). Include this file from analysis drivers.
using Oceananigans: Field, Center, Face, interior
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: ∂zᶜᶜᶠ
using Oceananigans.TurbulenceClosures: getclosure
using Breeze.TurbulenceClosures: Riᶜᶜᶠ, dissipationᶜᶜᶜ, shear_productionᶜᶜᶠ, buoyancy_productionᶜᶜᶠ

@inline function column_dissipation(i, j, k, grid, closures, e, velocities, fields)
    return dissipationᶜᶜᶜ(i, j, k, grid, getclosure(i, j, closures), e, velocities, fields)
end

@inline function kinematic_diffusive_flux(i, j, k, grid, diffusivity, scalar)
    return @inbounds -diffusivity[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, scalar)
end

function closure_diagnostic_recorder(number_of_members)
    sums = Dict{Symbol, Array{Float64, 3}}()
    counts = zeros(Int, number_of_members)
    function sample(model, t, active)
        grid, fields = model.grid, model.closure_fields
        ρ = model.dynamics.reference_state.density
        e = Field(model.tracers.ρe / ρ)
        qt = Field((model.microphysical_fields.qᵛ + model.microphysical_fields.qᶜˡ))
        face(f, args...) = Array(interior(Field(KernelFunctionOperation{Center, Center, Face}(f, grid, args...))))
        center(f, args...) = Array(interior(Field(KernelFunctionOperation{Center, Center, Center}(f, grid, args...))))
        Ri = face(Riᶜᶜᶠ, model.velocities, fields.N²)
        # Infinite Ri is physically legitimate in zero shear. Store finite regime occupancy
        # rather than averaging infinities into an uninterpretable mean Richardson number.
        instant = (; e = Array(interior(e)),
                   K_u = Array(interior(fields.Kᵘ)), K_c = Array(interior(fields.Kᶜ)),
                   K_e = Array(interior(fields.Kᵉ)), mixing_length = Array(interior(fields.ℓ)),
                   N2 = Array(interior(fields.N²)),
                   shear_production = face(shear_productionᶜᶜᶠ, fields.Kᵘ, model.velocities.u, model.velocities.v),
                   buoyancy_production = face(buoyancy_productionᶜᶜᶠ, fields.Kᶜ, fields.N²),
                   dissipation = center(column_dissipation, model.closure, e, model.velocities, fields),
                   theta_flux = face(kinematic_diffusive_flux, fields.Kᶜ, model.formulation.potential_temperature),
                   total_water_flux = face(kinematic_diffusive_flux, fields.Kᶜ, qt),
                   unstable_fraction = Float64.(Ri .< 0),
                   strongly_stable_fraction = Float64.(Ri .> 1))
        for (name, values) in pairs(instant)
            accumulator = get!(sums, name) do
                zeros(size(values))
            end
            for j in findall(active)
                accumulator[:, j, :] .+= values[:, j, :]
            end
        end
        counts .+= active
        return nothing
    end
    function result()
        any(iszero, counts) && error("Missing diagnostic samples")
        return (; counts = copy(counts), means = Dict(name => values ./ reshape(counts, 1, :, 1)
                                                       for (name, values) in sums))
    end
    return (; sample, result)
end
