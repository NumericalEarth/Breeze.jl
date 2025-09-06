# Notes on MABL modelling

## Surface fluxes

### Momentum flux

| Symbol   | Name    | Unit    |
| -------- | ------- |-------- |
| $C^D = 10^{-3}$   |  drag coefficient  |   |
| $G = 0.01$   |  gustiness friction velocity  | m / s |
| $u^\star = \sqrt(Cᴰ (Δu^2 + Δv^2)) + G$   |  friction velocity  | m / s |
| $u^{\star 2}u $ | $x$ momentum flux  | m³ / s³ |
| $u^{\star 2}v $ | $y$ momentum flux  | m³ / s³ |


### Sensible heat flux

| Symbol   | Name    | Unit    |
| -------- | ------- |-------- |
| $C^H = 10^{-3}$   |  heat transfer coefficient  |   |
| $c_p = 1005$ | dry air heat capacity   | J / (kg K) |
| $\theta\star = C^H/\sqrt{C^D}\Delta\theta$  | friction temperature    | K |
| $J\theta = -u^\star\theta^\star$  | temperature flux   | K m / s |
| $J^{SH} = \rho_0 c_p J\theta$  | heat flux   |  W / m² |


### Latent heat flux

| Symbol   | Name    | Unit    |
| -------- | ------- |-------- |
| $C^v = 10^{-3}$   |  vapor transfer coefficient  |   |
| $L^v = 2.5008 \times 10 ^6$   | latent heat of vaporisation  | J / kg |
| $q^\star = Cᵛ / \sqrt{Cᴰ}  Δq $   |  friction humidity (?)  | kg / kg |
| $Jq = -u^\star q^\star$   |  specific humidity flux  | m / s |
| $J^{LH} = \rho_0 L^v Jq$   |  latent heat flux  | W / m²  |

