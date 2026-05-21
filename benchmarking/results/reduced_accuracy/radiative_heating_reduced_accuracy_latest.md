# Reduced ecCKD Accuracy

Status: **failed_threshold**

Reference scope: clean ecCKD cloudless/no-aerosol tropical and RCEMIP-style cases.

| ng_lw | ng_sw | Method | Passed | Worst TOA forcing error | Worst surface forcing error |
|---:|---:|---|---:|---:|---:|
| 32 | 32 | official ecCKD 32x32 baseline without shortwave reduction | true | 0.00806408713606 W m^-2 | 0.0140335470378 W m^-2 |
| 32 | 16 | evenly selected official ecCKD g-points with renormalized weights | false | 15.148939416 W m^-2 | 66.5675113805 W m^-2 |
| 32 | 16 | greedy searched 16 shortwave g-point subset with official weights renormalized | false | 7.17718646856 W m^-2 | 6.84298667546 W m^-2 |
| 32 | 16 | greedy searched 16 shortwave g-point subset with least-squares fitted shortwave weights | false | 31.0262219594 W m^-2 | 66.6731185186 W m^-2 |
| 32 | 16 | greedy searched 16 shortwave g-point subset with projected boundary-weighted fitted shortwave weights | false | 5.55763254463 W m^-2 | 2.74834035515 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with projected simplex weights | false | 5.52030576048 W m^-2 | 3.17852209835 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with projected hard-gate max-norm weights | false | 5.51884380351 W m^-2 | 3.18861438431 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with diagnostic fitted coefficient scales | false | 5.88171009786 W m^-2 | 5.77898204224 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with latest preflight-optimized weights and coefficient scales | false | 2.58865294431 W m^-2 | 2.56622140734 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with latest preflight-optimized weights, coefficient scales, and pressure-band table moves | false | 2.31170988514 W m^-2 | 2.75790205932 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with latest preflight table moves and reduced incoming shortwave spectral weights | false | 2.31170988514 W m^-2 | 2.75790205932 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with boundary-aware post-constrained weight refit | false | 2.32587243275 W m^-2 | 2.75109439529 W m^-2 |
| 32 | 16 | weighted greedy 16 shortwave g-point subset with boundary-aware table, component, structural, objective-probe, surface-probe, capped table, continuation, post-capped weight, post-weight surface-table, bounded weight, four current component-scale refits, selected current gas-pressure component scan refit, gas-pressure continuation refit, weighted gas-pressure continuation refit, and high-weight gas-pressure continuation refit | false | 2.07524167121 W m^-2 | 2.02695668832 W m^-2 |
| 32 | 16 | evenly selected official ecCKD g-points with least-squares fitted shortwave weights | false | 56.8890709074 W m^-2 | 118.958235969 W m^-2 |
| 16 | 16 | evenly selected official ecCKD g-points with renormalized weights | false | 92.3557392308 W m^-2 | 171.674844598 W m^-2 |
| 32 | 16 | adjacent official ecCKD g-point bins with spectral-weighted coefficient averages | false | 66.0484885039 W m^-2 | 164.983235003 W m^-2 |
| 16 | 16 | adjacent official ecCKD g-point bins with spectral-weighted coefficient averages | false | 79.9785618291 W m^-2 | 156.746283354 W m^-2 |
| 32 | 16 | cumulative spectral-weight bins with coefficient averages | false | 81.055508224 W m^-2 | 215.161432856 W m^-2 |
| 16 | 16 | cumulative spectral-weight bins with coefficient averages | false | 94.9855815493 W m^-2 | 206.599801603 W m^-2 |
| 32 | 16 | non-adjacent coefficient-similarity shortwave g-point pairs with spectral-weighted coefficient averages | false | 26.2279763918 W m^-2 | 73.6097384692 W m^-2 |
| 32 | 16 | weighted-greedy anchor g-points plus nearest-neighbor coefficient-similarity bins | false | 40.3200732112 W m^-2 | 123.040442676 W m^-2 |
| 32 | 16 | weighted-greedy anchor g-points plus nearest spectral-order Voronoi bins | false | 40.3601966211 W m^-2 | 125.54176579 W m^-2 |

This is real reduced-model evidence, not a placeholder. A `failed_threshold` status means the selected reduced g-point subset does not yet meet the hard clean ecCKD thresholds.
