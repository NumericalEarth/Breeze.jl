export NullP3LookupTables, P3LookupTable1, P3LookupTable2, P3LookupTable3, P3LookupTables

struct NullP3LookupTables end

struct P3LookupTable1{FS, DP, BP, CL, M6, LL, IR}
    fall_speed :: FS
    deposition :: DP
    bulk_properties :: BP
    collection :: CL
    sixth_moment :: M6
    lambda_limiter :: LL
    ice_rain :: IR
end

struct P3LookupTable2{M, N, Z}
    mass :: M
    number :: N
    sixth_moment :: Z
end

struct P3LookupTable3{MU, LAM, RHOM}
    shape :: MU
    slope :: LAM
    mean_density :: RHOM
end

struct P3LookupTables{LT1, LT2, LT3}
    table_1 :: LT1
    table_2 :: LT2
    table_3 :: LT3
end
