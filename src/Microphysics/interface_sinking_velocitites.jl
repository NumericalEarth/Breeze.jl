using Oceananigans.Fields: ZFaceField, AbstractField, location, Center, Face, compute!
using Oceananigans.Forcings: maybe_constant_field
using Oceananigans.Grids: AbstractGrid

import Adapt: adapt_structure, adapt

# nothings for constant fields 
const valid_sinking_velocity_locations = ((Center, Center, Face), (Nothing, Nothing, Face), (Nothing, Nothing, Nothing)) 

function setup_velocity_fields(drift_speeds, grid::AbstractGrid, open_bottom; smoothing_distance = 2)
    drift_velocities = []
    for (name, w) in pairs(drift_speeds)
        if isa(w, Number)
            w_field = ZFaceField(grid)
            for k=1:grid.Nz 
                @inbounds w_field[:, :, k] .= - w * ifelse(open_bottom, 1.0, (1 - exp((1-k) / smoothing_distance)))
            end
        elseif isa(w, AbstractField)
            location(w) in valid_sinking_velocity_locations ||
                @warn "The location of the sinking velocity field provided for $name is incorrect, it should be (Center, Center, Face)" 

            open_bottom || @warn "The sinking velocity provided for $name is a field and therefore `open_bottom=false` can't be enforced automatically"

            compute!(w)
            w_field = w
        else
            @warn "Sinking speed provided for $name was not a number or field so may be unsiutable"
            w_field = w
        end

        push!(drift_velocities, w_field)
    end

    return NamedTuple{keys(drift_speeds)}(drift_velocities)
end

adapt_structure(to, velocities::NamedTuple{(:u, :v, :w), Tuple{AbstractField, AbstractField, AbstractField}}) = NamedTuple{(:u, :v, :w)}(adapt.(to, values(velocities)))

function show_sinking_velocities(sinking_velocities::NamedTuple{T, V}) where {T, V} 
    str = ""
    if length(T) == 0
        return str
    elseif length(T) == 1
        return "    └── $(T[1]): $(maximum_sinking(sinking_velocities[1])) to $(minimum_sinking(sinking_velocities[1])) m/s"
    else
        for idx in 1:length(T) - 1
            str *= "    ├── $(T[idx]): $(maximum_sinking(sinking_velocities[idx])) to $(minimum_sinking(sinking_velocities[idx])) m/s \n"
        end
        str *= "    └── $(T[end]): $(maximum_sinking(sinking_velocities[end])) to $(minimum_sinking(sinking_velocities[end])) m/s"

        return str
    end
end

maximum_sinking(velocity) = maximum(velocity)
minimum_sinking(velocity) = minimum(velocity)

maximum_sinking(velocity::NamedTuple) = maximum(velocity.w)
minimum_sinking(velocity::NamedTuple) = minimum(velocity.w)
