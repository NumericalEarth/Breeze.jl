# Radiative Heating RCEMIP-Style Benchmark

- status: scaffold_not_final_4x_evidence
- backend: CPU
- grid: 1 x 1 x 4
- gas model source: validated_ecCKD
- gas model kind: official_ecCKD_32_lw_32_sw
- gas model accuracy status: passed
- gas values: Dict("o3" => 0.0, "ch4" => 1.8e-6, "cfc11" => 2.3e-10, "n2o" => 3.3e-7, "cfc12" => 5.2e-10, "co2" => 0.00042)
- gas model device support: supported
- gas model device support reason: CPU benchmark path supports package-native gas optics
- RadiativeHeating supported: true
- RRTMGP supported: true
- speedup: 3.932x
- final 4x claim supported: false
- blocking reason: benchmark status is scaffold_not_final_4x_evidence, not final_4x_evidence
