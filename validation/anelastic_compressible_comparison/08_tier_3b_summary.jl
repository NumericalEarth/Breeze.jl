#####
##### Tier-3b summary: rico, neutral_abl, cloudy_thermal_bubble (cont.)
#####

include("ABCompare.jl")
using .ABCompare

@info "=========== Tier 3b summary ==========="

# rico: same blockers as bomex (BulkSensibleHeatFlux + ρe-energy forcing).
notes_rico = "Blocked: same anelastic-specific BulkSensibleHeatFlux & forcing infra as bomex"

# neutral_atmospheric_boundary_layer: forcings are ρθ-based (good), drag BCs are
# state-dependent FluxBoundaryConditions (no virtual-T issue). But Δz=10m on 96³ GPU
# grid means substepper CFL limit ≈ 0.04s vs anelastic 0.5s — 12× more iters and
# a real-time GPU run needed. Out of scope for this comparison.
notes_abl = "Not run: substepper Δt limit ≈ 0.04s (12× more iters than anelastic); GPU-heavy 5h LES out of scope"

# cloudy_thermal_bubble: see 07_*. Sat-adjust + substepper NaN'd at iter 100
# regardless of Δt. Likely sat-adjust modifies ρθ in a way that breaks the
# substepper's frozen-linearization assumption.
notes_cloudy = "FAILED: sat-adjust + substepper NaN at iter 100 even at small Δt (latent-heating + frozen linearization conflict)"

isfile(REPORT_PATH) || (open(REPORT_PATH, "w") do io
    write(io, ABCompare.header())
end)
open(REPORT_PATH, "a") do io
    write(io, "| rico | (not run) | (not run) | — | — | — | — | $notes_rico |\n")
    write(io, "| neutral_atmospheric_boundary_layer | (not run) | (not run) | — | — | — | — | $notes_abl |\n")
end
@info "Wrote tier-3b rows to $REPORT_PATH"
