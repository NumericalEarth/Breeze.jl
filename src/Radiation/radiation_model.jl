abstract type AbstractRadiationModel end

function update_radative_fluxes!(model::AbstractRadiationModel) 
    throw("Not implemented for $(typeof(model)).")
end
