export P3IceIntegralsTable, P3RainIceCollectionTable, P3LookupTables

struct P3IceIntegralsTable{FS, DP, BP, CL, LL, IR}
    fall_speed :: FS
    deposition :: DP
    bulk_properties :: BP
    collection :: CL
    lambda_limiter :: LL
    ice_rain :: IR
end

struct P3RainIceCollectionTable{M, N}
    mass :: M
    number :: N
end

struct P3LookupTables{LT1, LT2}
    ice_integrals :: LT1
    rain_ice_collection :: LT2
end
