#####
##### Save/load P3 lookup tables to/from JLD2 files
#####

"""
$(TYPEDSIGNATURES)

Save a tabulated P3 microphysics scheme to a JLD2 file.

The entire tabulated scheme is serialized, including all three lookup table
families and tabulated rain properties. Load with [`load_p3_lookup_tables`](@ref).

# Arguments

- `filepath`: Path to the output JLD2 file (e.g., `"p3_lookup_tables.jld2"`)
- `p3`: A tabulated `PredictedParticlePropertiesMicrophysics` (from [`tabulate`](@ref))

# Example

```julia
using Oceananigans
using Breeze.Microphysics.PredictedParticleProperties

p3 = PredictedParticlePropertiesMicrophysics()
p3_tabulated = tabulate(p3, CPU())
save_p3_lookup_tables("p3_lookup_tables.jld2", p3_tabulated)
```
"""
function save_p3_lookup_tables(filepath, p3::PredictedParticlePropertiesMicrophysics)
    jldsave(filepath; p3_tabulated=p3)
end

"""
$(TYPEDSIGNATURES)

Load a tabulated P3 microphysics scheme from a JLD2 file.

Returns a fully tabulated `PredictedParticlePropertiesMicrophysics` ready for use.
The file must have been created by [`save_p3_lookup_tables`](@ref).

# Arguments

- `filepath`: Path to the JLD2 file

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

p3 = load_p3_lookup_tables("p3_lookup_tables.jld2")
p3.ice.lookup_tables  # P3LookupTables with all three table families
```
"""
function load_p3_lookup_tables(filepath)
    file = jldopen(filepath)
    p3 = file["p3_tabulated"]
    close(file)
    return p3
end
