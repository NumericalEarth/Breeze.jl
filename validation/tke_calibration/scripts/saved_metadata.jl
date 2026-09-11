# JLD2 reconstructs unavailable package types in lightweight analysis processes. In particular,
# a NamedTuple containing a vector of such types may become ReconstructedMutable: it has properties
# but no keys/haskey methods, and two separately loaded copies are not equal by value. Normalize
# containers before checking compatibility; retain the saved type name of every reconstructed model.
using JLD2

saved_metadata_value(x) = x
saved_metadata_value(x::NamedTuple) = map(saved_metadata_value, x)
saved_metadata_value(x::Tuple) = map(saved_metadata_value, x)
saved_metadata_value(x::AbstractArray) = map(saved_metadata_value, x)
saved_metadata_value(x::AbstractDict) = Dict(saved_metadata_value(k) => saved_metadata_value(v) for (k, v) in x)

function saved_metadata_value(x::JLD2.AbstractReconstructedType{N}) where N
    names = propertynames(x)
    fields = NamedTuple{names}(Tuple(saved_metadata_value(getproperty(x, name)) for name in names))
    # Restore the structural container, not the missing scientific type. Distinct stability
    # formulations must remain distinct even when both are fieldless singleton structs.
    return startswith(string(N), "NamedTuple{") ? fields : (; saved_type = string(N), fields)
end
