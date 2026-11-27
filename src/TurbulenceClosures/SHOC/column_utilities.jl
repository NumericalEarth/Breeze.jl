#####
##### Interpolation between levels and interfaces
#####

@inline linearly_interpolate(x1u, x1l, y1u, y1l, x2) = y1u + (y1l - y1u) * (x2 - x1u) / (x1l - x1u)
