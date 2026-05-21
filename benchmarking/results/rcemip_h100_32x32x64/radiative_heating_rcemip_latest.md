# Radiative Heating RCEMIP-Style Benchmark

- status: final_4x_evidence
- backend: H100
- grid: 32 x 32 x 64
- gas model source: validated_ecCKD
- gas model kind: official_ecCKD_32_lw_32_sw
- gas model accuracy status: passed
- gas values: Dict("o3" => 0.0, "ch4" => 1.8e-6, "cfc11" => 2.3e-10, "n2o" => 3.3e-7, "cfc12" => 5.2e-10, "co2" => 0.00042)
- gas model device support: supported
- gas model device support reason: H100 extension path supports tabulated multi-gas ecCKD with recorded CPU/GPU parity evidence
- gas model device support source: BreezeRadiativeHeatingExt
- RadiativeHeating supported: true
- RRTMGP supported: true
- speedup: 31.335x
- final 4x claim supported: true
- blocking reason: none
