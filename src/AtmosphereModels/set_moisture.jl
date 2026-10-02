using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: set!

"""
$(TYPEDSIGNATURES)

Convert the total-water density staged by `set!` in `model.moisture_density` into
the scheme's prognostic moisture, after the air densities have been established.
Independent condensates are subtracted using total-air mass fractions.

Reject condensate exceeding total water beyond local round-off tolerance. Retain
round-off-sized residuals so the conversion preserves the supplied water budget.
"""
function convert_total_moisture!(model)
    ρ = total_density(model.dynamics)
    qᵗ = model.moisture_density / ρ
    qᵛᵉ = specific_prognostic_moisture(model.microphysics, qᵗ, model.microphysical_fields, ρ)
    moisture = specific_prognostic_moisture(model)

    validate_total_moisture(qᵗ, qᵛᵉ, moisture)
    set!(moisture, qᵛᵉ)
    set!(model.moisture_density, ρ * moisture)
    return nothing
end

@inline function total_moisture_validation_margin(i, j, k, grid, total_moisture, prognostic_moisture)
    total_water = @inbounds total_moisture[i, j, k]
    prognostic_water = @inbounds prognostic_moisture[i, j, k]
    condensate = total_water - prognostic_water
    tolerance = 10eps(eltype(grid)) * max(abs(total_water), abs(condensate))
    return prognostic_water + tolerance
end

function validate_total_moisture(total_moisture, prognostic_moisture, validation_field)
    margin = KernelFunctionOperation{Center, Center, Center}(total_moisture_validation_margin,
                                                            validation_field.grid, total_moisture, prognostic_moisture)
    # Reactant reductions need a stored field. Reuse the specific moisture field
    # as scratch; the conversion above restores it before storing the density.
    set!(validation_field, margin)
    validate_total_moisture(minimum(validation_field) ≥ 0)
    return nothing
end

# Traced backends specialize this scalar check to run at execution time.
function validate_total_moisture(valid)
    Bool(valid) || throw(ArgumentError("set! received a total moisture qᵗ smaller than the supplied \
                                       condensates, or a non-finite moisture state. Increase qᵗ or \
                                       reduce the condensate inputs."))
    return nothing
end
