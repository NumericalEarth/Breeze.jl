module RadiativeTransfer

export RadiativeTransferModel

mutable struct RadiativeTransferModel{Sol, F, Z, E, Dir, Dif, SW}
    solver :: Sol
    downwelling_longwave_flux :: F
    downwelling_shortwave_flux :: F
    zenith_angle :: Z
    surface_emissivity :: E
    direct_surface_albedo :: Dir
    diffuse_surface_albedo :: Dif
    incoming_shortwave :: SW
    incoming_longwave :: LW
end

struct RRTMGPSolver end

# include("radiative_transfer_model.jl")
# include("grid_conversion.jl")
# include("atmosphere_model_integration.jl")

end # module

