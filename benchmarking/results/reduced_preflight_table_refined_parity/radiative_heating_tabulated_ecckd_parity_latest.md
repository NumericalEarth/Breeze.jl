# Radiative Heating Tabulated ecCKD CPU/GPU Parity

- status: passed
- backend: H100
- gas model source: validated_ecCKD_reduced_preflight_table_refined
- gas model kind: official_ecCKD_reduced_preflight_table_refined_32_lw_16_sw
- grid: 2 x 2 x 8
- tolerance atol: 1.0e-8
- tolerance rtol: 1.0e-8
- passed: true
- reason: CPU/GPU field parity passed
- field errors:
  - downwelling_longwave_flux: max_abs=1.7763568394002505e-15, rmse=5.924263396364183e-16, reference_max_abs=296.0554687087857
  - downwelling_shortwave_flux: max_abs=1.1368683772161603e-13, rmse=6.563712636189231e-14, reference_max_abs=551.0000000000001
  - flux_divergence: max_abs=3.0357660829594124e-17, rmse=2.2119788346190346e-17, reference_max_abs=0.006695599542096549
  - upwelling_longwave_flux: max_abs=5.684341886080802e-14, rmse=3.142138202705193e-14, reference_max_abs=450.11432138022025
