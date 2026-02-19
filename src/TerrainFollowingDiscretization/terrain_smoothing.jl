#####
##### Terrain smoothing / decay functions
#####
##### These control how the terrain perturbation decays with height.
##### For BasicTerrainFollowing (Gal-Chen & Somerville 1975), the decay
##### is linear: terrain influence vanishes at the model top.
#####

"""
    BasicTerrainFollowing

Linear decay of terrain influence with height, following
[Gal-Chen and Somerville (1975)](@cite GalChen1975).

The coordinate transformation is

```math
z(x, y, \\zeta) = \\zeta + h(x, y) \\left(1 - \\frac{\\zeta}{z_{top}}\\right)
```

which gives ``\\sigma = (z_{top} - h) / z_{top}`` and ``\\eta = h``.
"""
struct BasicTerrainFollowing end
