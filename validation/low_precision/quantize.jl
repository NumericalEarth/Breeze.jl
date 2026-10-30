# Emulated low-precision storage: compute in Float64, round the stored value to a format with p significand
# bits (including the implicit bit) and minimum normal exponent emin (gradual underflow below it).
# Round-to-nearest-even (RN) or stochastic rounding (SR, probability proportional to distance).
# Overflow is not modelled; callers scale values so that they stay in range (checked by `in_range`).

struct StorageFormat # isbits, so it can be passed to GPU kernels
    p :: Int
    emin :: Int
    floatmax :: Float64
end

const Float32Format = StorageFormat(24, -126, Float64(floatmax(Float32)))
const BFloat16Format = StorageFormat(8, -126, 3.39e38)
const Float16Format = StorageFormat(11, -14, 65504.0)
const E4M3Format = StorageFormat(4, -6, 448.0)
const E5M2Format = StorageFormat(3, -14, 57344.0)

@inline function quantize(x, f::StorageFormat, stochastic::Bool, u)
    # Unbiased exponent from the bits (GPU-friendly, no throw on zero)
    e = max(Int((reinterpret(UInt64, Float64(x)) >> 52) & 0x7ff) - 1023, f.emin)
    ulp = exp2(e - (f.p - 1))
    y = x / ulp
    q = ifelse(stochastic, floor(y + u), round(y))
    return q * ulp
end

in_range(x, f::StorageFormat) = abs(x) ≤ f.floatmax
