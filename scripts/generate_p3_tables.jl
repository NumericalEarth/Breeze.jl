## Generate P3 lookup tables and save to JLD2
##
## Usage:
##   julia --project scripts/generate_p3_tables.jl [output_path]
##
## Default output: data/p3_lookup_tables.jld2

using Oceananigans
using Breeze.Microphysics.PredictedParticleProperties

output_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "p3_lookup_tables.jld2")

## Create output directory if needed
mkpath(dirname(output_path))

## Create untabulated P3 scheme
@info "Creating P3 scheme..."
p3 = PredictedParticlePropertiesMicrophysics()

## Tabulate with default (full resolution) parameters
@info "Tabulating lookup tables (this may take a few minutes)..."
p3_tabulated = tabulate(p3, CPU())

## Save to JLD2
@info "Saving to $output_path..."
save_p3_lookup_tables(output_path, p3_tabulated)

@info "Done. Lookup tables saved to $output_path"
