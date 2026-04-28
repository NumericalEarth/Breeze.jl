#####
##### Tier-5a: radiative_convection
#####

include("ABCompare.jl")
using .ABCompare

@info "=========== radiative_convection ==========="

# Two blockers:
# 1. ρe sponge forcing — anelastic-specific (same as bomex)
# 2. set!(model; T = Tᵢ, ℋ = ℋᵢ, ...) — sets temperature and moist static energy,
#    requires StaticEnergyThermodynamics. CompressibleDynamics doesn't support it.
# 3. PiecewiseStretchedDiscretization in z — substepper supports it (verified
#    earlier), but combined with the above two blockers this example needs
#    significant porting work.
notes = "Blocked: ρe sponge forcing + StaticEnergy IC (needs CompressibleDynamics support for StaticEnergy)"

isfile(REPORT_PATH) || (open(REPORT_PATH, "w") do io
    write(io, ABCompare.header())
end)
open(REPORT_PATH, "a") do io
    write(io, "| radiative_convection | (not run) | (not run) | — | — | — | — | $notes |\n")
end
@info "Wrote radiative_convection placeholder row"
