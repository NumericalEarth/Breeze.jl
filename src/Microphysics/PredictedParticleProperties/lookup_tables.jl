export P3IceIntegralsTable, P3RainIceCollectionTable, P3ThreeMomentShapeTable, P3LookupTables

struct P3IceIntegralsTable{FS, DP, BP, CL, M6, LL, IR}
    fall_speed :: FS
    deposition :: DP
    bulk_properties :: BP
    collection :: CL
    sixth_moment :: M6
    lambda_limiter :: LL
    ice_rain :: IR
end

struct P3RainIceCollectionTable{M, N, Z}
    mass :: M
    number :: N
    sixth_moment :: Z
end

struct P3ThreeMomentShapeTable{MU, LAM, RHOM}
    shape :: MU
    slope :: LAM
    mean_density :: RHOM
end

struct P3LookupTables{LT1, LT2, LT3}
    ice_integrals :: LT1
    rain_ice_collection :: LT2
    three_moment_shape :: LT3
end
