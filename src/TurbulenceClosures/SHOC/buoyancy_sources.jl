#####
##### Slot: buoyancy source of the turbulent kinetic energy
#####
##### A buoyancy source component implements
#####
#####     buoyancy_production(bs, c, wthv_sec_k, tkh_k, brunt_k) -> B
#####
##### from the buoyancy flux of the assumed PDF and the heat diffusivity of the previous step, and
##### the stratification of this one. The buoyancy flux `wthv_sec` of the assumed PDF is diagnosed
##### and stored whatever the choice; the slot selects what the TKE equation consumes.
#####

"""
$(TYPEDEF)

The buoyancy production of E3SM `shoc.F90`: `B = (g / θ₀) w'θᵥ'`, with the buoyancy flux
`w'θᵥ'` that the assumed PDF diagnosed in the previous step, which includes the flux of liquid
water in partly cloudy air.
"""
struct PDFBuoyancyFlux <: AbstractSHOCComponent end

@inline buoyancy_production(::PDFBuoyancyFlux, c, wthv_sec_k, tkh_k, brunt_k) = (c.ggr / c.basetemp) * wthv_sec_k
