# Reduced ecCKD Shortwave Subset Search

Status: **failed_threshold**

Search objective: `flux_boundary_and_shortwave_heating_rate`

Selected shortwave g-points: `2, 4, 7, 8, 10, 11, 12, 14, 16, 18, 21, 25, 27, 28, 30, 32`

| Case | Passed | TOA forcing error | Surface forcing error | SW up RMSE | SW down RMSE | Heating RMSE | Heating max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ecckd_clear_sky_tropical_column | false | 4.26800277057 W m^-2 | 2.86429350541 W m^-2 | 3.08993452037 W m^-2 | 2.61440809004 W m^-2 | 1.25042184403 K day^-1 | 5.09552972985 K day^-1 |
| ecckd_rcemip_style_column_subset | false | 7.37779904074 W m^-2 | 7.02576997551 W m^-2 | 2.84554146553 W m^-2 | 2.70673483883 W m^-2 | 0.992148811451 K day^-1 | 5.09552972985 K day^-1 |

## Weighted Subset Search

Selected shortwave g-points: `2, 4, 7, 10, 11, 12, 14, 16, 18, 21, 22, 27, 28, 30, 31, 32`

Boundary weight: `30`

| Case | Passed | TOA forcing error | Surface forcing error | SW up RMSE | SW down RMSE | Heating RMSE | Heating max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ecckd_clear_sky_tropical_column | false | 6.98955437931 W m^-2 | 6.83942930433 W m^-2 | 4.73361021121 W m^-2 | 5.05106822397 W m^-2 | 1.0897481383 K day^-1 | 7.34626394558 K day^-1 |
| ecckd_rcemip_style_column_subset | false | 6.98955437931 W m^-2 | 6.83942930433 W m^-2 | 4.16265216449 W m^-2 | 4.22322538415 W m^-2 | 0.880620143762 K day^-1 | 7.34626394558 K day^-1 |

A failed status means this deterministic subset search improved the current 16-g shortwave reduction but still does not satisfy the hard clean ecCKD thresholds. The current failure is useful evidence that simple subset selection is not enough; a real ecCKD reduction/optimization method is still required.

## Search History

| Pass | Approximate normalized score | Indices |
|---:|---:|---|
| 0 | 151.800780231 | `2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32` |
| 1 | 39.532464365 | `2, 4, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32` |
| 2 | 33.825821147 | `2, 4, 7, 8, 10, 11, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32` |
| 3 | 29.6982463611 | `2, 4, 7, 8, 10, 11, 12, 14, 16, 18, 20, 24, 27, 28, 30, 32` |
| 4 | 26.6002898439 | `2, 4, 7, 8, 10, 11, 12, 14, 16, 18, 19, 20, 27, 28, 30, 32` |
| 5 | 26.2709017087 | `2, 4, 7, 8, 10, 11, 12, 14, 16, 18, 19, 21, 27, 28, 30, 32` |
| 6 | 25.0101657583 | `2, 4, 7, 8, 10, 11, 12, 14, 16, 18, 21, 25, 27, 28, 30, 32` |

## Boundary-Weight Trials

| Boundary weight | Approximate normalized score | Indices |
|---:|---:|---|
| 1 | 27.4792824155 | `2, 4, 7, 8, 10, 12, 14, 16, 18, 21, 26, 27, 28, 29, 30, 32` |
| 3 | 31.6835820958 | `2, 4, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 27, 28, 30, 32` |
| 10 | 24.7084632571 | `2, 4, 7, 8, 9, 10, 12, 14, 16, 18, 21, 27, 28, 30, 31, 32` |
| 30 | 23.299930424 | `2, 4, 7, 10, 11, 12, 14, 16, 18, 21, 22, 27, 28, 30, 31, 32` |

## Full-Fit Pruning Trial

Selected shortwave g-points: `1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 26, 27, 28, 29`

Boundary weight: `3`

| Case | Passed | TOA forcing error | Surface forcing error | SW up RMSE | SW down RMSE | Heating RMSE | Heating max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ecckd_clear_sky_tropical_column | false | 7.07690866492 W m^-2 | 23.7988670186 W m^-2 | 4.90620641996 W m^-2 | 21.7637511995 W m^-2 | 6.80026246917 K day^-1 | 25.4769895646 K day^-1 |
| ecckd_rcemip_style_column_subset | false | 7.80363748854 W m^-2 | 23.7988670186 W m^-2 | 4.59516295733 W m^-2 | 16.7464999266 W m^-2 | 5.4890523116 K day^-1 | 25.4769895646 K day^-1 |

## Hard-Gate Max-Norm Weight Trial

Selected shortwave g-points: `2, 4, 7, 10, 11, 12, 14, 16, 18, 21, 22, 27, 28, 30, 31, 32`

Source topology: `boundary-weighted greedy subset`

Approximate normalized hard-gate objective: `26.8186307771`

| Case | Passed | TOA forcing error | Surface forcing error | SW up RMSE | SW down RMSE | Heating RMSE | Heating max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ecckd_clear_sky_tropical_column | false | 6.98899700819 W m^-2 | 6.83402151419 W m^-2 | 4.7333622194 W m^-2 | 5.04941262423 W m^-2 | 1.09007009803 K day^-1 | 7.34705811981 K day^-1 |
| ecckd_rcemip_style_column_subset | false | 6.98899700819 W m^-2 | 6.83402151419 W m^-2 | 4.1620403098 W m^-2 | 4.22195731788 W m^-2 | 0.880872142382 K day^-1 | 7.34705811981 K day^-1 |
