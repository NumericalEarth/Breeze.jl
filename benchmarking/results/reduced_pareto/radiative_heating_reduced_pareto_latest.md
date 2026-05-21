# Reduced ecCKD Pareto Evidence

Status: h100_runtime_accuracy_failed_threshold

| Label | backend | grid | RH ms | RRTMGP ms | speedup | accuracy | TOA W m^-2 | surface W m^-2 | support |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| synthetic 16x16 H100 term-count scaffold | H100 | 32x32x64 | 20.7811 | 561.531 | 27.0212 | missing | n/a | n/a | missing |
| synthetic 32x16 H100 term-count scaffold | H100 | 32x32x64 | 18.4005 | 547.668 | 29.7638 | missing | n/a | n/a | missing |
| official 32x16 preflight table-refined H100 smoke | H100 | 2x2x8 | 0.590822 | 21.6732 | 36.6831 | failed_threshold | 2.49657297122 | 2.38029831045 | supported |

Accuracy source: `results/reduced_accuracy/radiative_heating_reduced_accuracy_latest.json`.
The official table-refined row is smoke/runtime-path evidence until the reduced hard thresholds pass.
