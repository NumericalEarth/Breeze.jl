using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: set!

"""
$(TYPEDSIGNATURES)

Convert the total-water density staged by `set!` in `model.moisture_density` into the
scheme's prognostic moisture density by subtracting the independent condensate densities,
``ρqᵛᵉ = ρqᵗ - Σ ρqᶜ``, then diagnose the specific prognostic moisture ``qᵛᵉ = ρqᵛᵉ / ρ``
from the established total air density. The subtraction involves no density, so the
supplied water budget is preserved exactly.

Reject condensate exceeding total water beyond local round-off tolerance. Retain
round-off-sized residuals so the conversion preserves the supplied water budget. Inside a
compiled program, where the check cannot raise, an invalid state's moisture is NaN instead.
"""
convert_total_moisture!(model) = convert_total_moisture!(model, condensate_field_names(model.microphysics))

# Without independent condensates the staged total water is already the prognostic moisture.
function convert_total_moisture!(model, ::Tuple{})
    ρqᵗ = model.moisture_density
    valid = minimum(ρqᵗ) ≥ 0
    validate_total_moisture(valid)
    set!(specific_prognostic_moisture(model), ρqᵗ / total_density(model.dynamics))
    invalidate_moisture!(model, valid)
    return nothing
end

function convert_total_moisture!(model, condensate_names::Tuple{Symbol, Vararg})
    ρqᵗ = model.moisture_density
    ρqᶜ = sum_properties(model.microphysical_fields, condensate_names)
    moisture = specific_prognostic_moisture(model)

    valid = validate_total_moisture(ρqᵗ, ρqᶜ, moisture)
    set!(ρqᵗ, ρqᵗ - ρqᶜ)
    set!(moisture, ρqᵗ / total_density(model.dynamics))
    invalidate_moisture!(model, valid)
    return nothing
end

# Margin by which the prognostic moisture density clears zero, within a tolerance
# relative to the water amounts in the cell.
@inline function total_moisture_validation_margin(i, j, k, grid, total_moisture_density, condensate_density)
    @inbounds begin
        ρqᵗ = total_moisture_density[i, j, k]
        ρqᶜ = condensate_density[i, j, k]
    end
    tolerance = 10eps(eltype(grid)) * max(abs(ρqᵗ), abs(ρqᶜ))
    return ρqᵗ - ρqᶜ + tolerance
end

function validate_total_moisture(total_moisture_density, condensate_density, validation_field)
    margin = KernelFunctionOperation{Center, Center, Center}(total_moisture_validation_margin,
                                                            validation_field.grid,
                                                            total_moisture_density, condensate_density)
    # Reactant reductions need a stored field. The specific moisture field is free until
    # the conversion overwrites it, and the margin does not read it.
    set!(validation_field, margin)
    valid = minimum(validation_field) ≥ 0
    validate_total_moisture(valid)
    return valid
end

# Traced backends, where the check cannot raise, specialize this to do nothing and
# `invalidate_moisture!` to mark an invalid state instead.
function validate_total_moisture(valid)
    Bool(valid) || throw(ArgumentError("set! received a total moisture qᵗ smaller than the supplied \
                                       condensates, or a non-finite moisture state. Increase qᵗ or \
                                       reduce the condensate inputs."))
    return nothing
end

# An invalid state has already been rejected by `validate_total_moisture`.
invalidate_moisture!(model, valid) = nothing
