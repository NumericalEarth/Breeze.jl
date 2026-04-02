using Base: @propagate_inbounds

import Adapt: adapt_structure
import Base: getindex

"""
    ProductOfArrays{A, B}

Lazy representation of the element-wise product of two arrays.
Returns `a[i, j, k] * b[i, j, k]` when indexed.
"""
struct ProductOfArrays{A, B}
    a :: A
    b :: B
end

@propagate_inbounds getindex(p::ProductOfArrays, i, j, k) = @inbounds p.a[i, j, k] * p.b[i, j, k]

adapt_structure(to, p::ProductOfArrays) =
    ProductOfArrays(adapt_structure(to, p.a), adapt_structure(to, p.b))
