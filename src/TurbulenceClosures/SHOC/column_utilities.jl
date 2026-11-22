#####
##### Interpolation between levels and interfaces
#####

# shoc.F90 `linear_interp` between an upper node (x1u, y1u) and a lower node (x1l, y1l).
@inline linear_interp(x1u, x1l, y1u, y1l, x2) = y1u + (y1l - y1u) * (x2 - x1u) / (x1l - x1u)
