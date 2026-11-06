#####
##### Lateral mass balance for a laterally-bounded anelastic domain
#####

"""
$(TYPEDSIGNATURES)

Return `dynamics` holding the boundary transports that
[`enforce_lateral_mass_balance!`](@ref Breeze.AtmosphereModels.enforce_lateral_mass_balance!)
corrects from — a flux reduction and an area for each boundary that opts in, and `nothing` when none
does.
"""
AtmosphereModels.materialize_boundary_transport(dynamics::AnelasticDynamics, momentum) =
    AnelasticDynamics(dynamics.reference_state,
                      dynamics.pressure_anomaly,
                      dynamics.pressure_solver_float_type,
                      initialize_boundary_transport(momentum))

"""
$(TYPEDSIGNATURES)

Remove the net mass flux through the boundaries of an anelastic domain,

    ∮ ρᵣ u⋅n̂ dA = 0 ,

which is the condition for the pressure Poisson problem to be solvable under the homogeneous Neumann
conditions a `Bounded` direction's cosine transform imposes.

The offset subtracted from the normal momentum on each boundary face that opts in is uniform in
momentum, not in velocity, so the velocity increment scales as 1/ρᵣ(z) and the flux integral is
removed exactly whatever ρᵣ does over the column. A boundary opts in with
`NormalFlowBoundaryCondition(value; scheme = BalancedTransport())` or a radiation scheme; a `Periodic`
direction carries no net flux and an impenetrable boundary — the default on a `Bounded` side, and what
an anelastic domain's top and bottom are — is balanced by construction, so a domain with neither is a
no-op.
"""
AtmosphereModels.enforce_lateral_mass_balance!(dynamics::AnelasticDynamics, momentum) =
    enforce_net_zero_transport!(momentum, dynamics.boundary_transport)
