# Radiative Heating Tabulated ecCKD CPU/GPU Parity

- status: passed
- backend: H100
- gas model source: validated_ecCKD
- gas model kind: official_ecCKD_32_lw_32_sw
- grid: 2 x 2 x 8
- tolerance atol: 1.0e-8
- tolerance rtol: 1.0e-8
- passed: true
- reason: CPU/GPU field parity passed
- field errors:
  - downwelling_longwave_flux: max_abs=1.7763568394002505e-15, rmse=5.924263396364183e-16, reference_max_abs=296.0554687087857
  - downwelling_shortwave_flux: max_abs=1.1368683772161603e-13, rmse=5.359248925640619e-14, reference_max_abs=551.0
  - flux_divergence: max_abs=6.063400649625184e-17, rmse=2.4563240447328427e-17, reference_max_abs=0.006867587670208998
  - upwelling_longwave_flux: max_abs=5.684341886080802e-14, rmse=3.142138202705193e-14, reference_max_abs=450.11432138022025
