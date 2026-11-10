#####
##### Small utilities
#####

# √x whose value is exactly that of `sqrt` for x ≥ 0, with a finite (zero) derivative at x = 0.
@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))

# Fortran `if (y < minthresh) y = minthresh`, which keeps a negative zero where `max` would not.
@inline clip_below(y, minthresh) = ifelse(y < minthresh, minthresh, y)
