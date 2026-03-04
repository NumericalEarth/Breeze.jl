#####
##### Systematic probe: getindex on ConcreteRArray inside KA kernels
#####
#
# The LatitudeLongitudeGrid broadcast failure boils down to whether the
# ReactantKA extension can handle getindex on ConcreteRArray / ConcretePJRTArray
# inside kernels — both as direct arguments and nested inside structs.
#
# This file probes the boundary progressively:
#   Test 1: Direct ConcreteRArray as kernel arg, index it
#   Test 2: ConcreteRArray wrapped in a struct, index through the struct
#   Test 3: Two arrays in a struct, only one is used (other is dead weight in type tree)
#   Test 4: Struct containing OffsetVector{Float64, ConcreteRArray} (matches LatLonGrid exactly)
#
# Tests 3 & 4 mimic the real failure: the grid's metric arrays sit in the
# kernel type closure but the identity interpolation operator never touches them.

using Reactant
using KernelAbstractions: @kernel, @index, StaticSize
using OffsetArrays: OffsetVector

const RKAExt  = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const Backend = RKAExt.ReactantBackend

const N = 4

# ─────────────────────────────────────────────────────────────
# Test 1: Direct ConcreteRArray arg
# ─────────────────────────────────────────────────────────────

@kernel function _direct_index_kernel!(out, v)
    i = @index(Global, Linear)
    @inbounds out[i] = v[i]
end

function run_direct!(out, v)
    _direct_index_kernel!(Backend(), StaticSize((N,)), StaticSize((N,)))(out, v)
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Test 2: ConcreteRArray inside a struct, kernel indexes it
# ─────────────────────────────────────────────────────────────

struct Holder{A}
    data::A
end

@kernel function _struct_index_kernel!(out, h)
    i = @index(Global, Linear)
    @inbounds out[i] = h.data[i]
end

function run_struct!(out, h)
    _struct_index_kernel!(Backend(), StaticSize((N,)), StaticSize((N,)))(out, h)
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Test 3: Struct with two arrays, kernel only uses one
#          (dead array in the type tree, like LatLonGrid metrics)
# ─────────────────────────────────────────────────────────────

struct TwoArrays{A, B}
    used::A
    unused::B
end

@kernel function _dead_weight_kernel!(out, s)
    i = @index(Global, Linear)
    @inbounds out[i] = s.used[i]
end

function run_dead_weight!(out, s)
    _dead_weight_kernel!(Backend(), StaticSize((N,)), StaticSize((N,)))(out, s)
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Test 4: OffsetVector wrapping ConcreteRArray (exact LatLonGrid pattern)
# ─────────────────────────────────────────────────────────────

struct GridLike{A, M}
    data::A
    metric::M   # OffsetVector{Float64, ConcreteRArray} — never used by kernel
end

@kernel function _gridlike_kernel!(out, g)
    i = @index(Global, Linear)
    @inbounds out[i] = g.data[i]
end

function run_gridlike!(out, g)
    _gridlike_kernel!(Backend(), StaticSize((N,)), StaticSize((N,)))(out, g)
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

out = Reactant.ConcreteRArray(zeros(N))
v   = Reactant.ConcreteRArray(collect(1.0:Float64(N)))

# Test 1
@info "Test 1: Direct ConcreteRArray arg"
try
    f1! = @compile sync=true run_direct!(out, v)
    f1!(out, v)
    @info "  → PASS" result=Array(out)
catch e
    @info "  → FAIL" exception=e
end

# Test 2
@info "Test 2: ConcreteRArray inside struct"
h = Holder(v)
try
    f2! = @compile sync=true run_struct!(out, h)
    f2!(out, h)
    @info "  → PASS" result=Array(out)
catch e
    @info "  → FAIL" exception=e
end

# Test 3
@info "Test 3: Struct with used + unused ConcreteRArray"
unused = Reactant.ConcreteRArray(collect(100.0:100.0+Float64(N-1)))
s = TwoArrays(v, unused)
try
    f3! = @compile sync=true run_dead_weight!(out, s)
    f3!(out, s)
    @info "  → PASS" result=Array(out)
catch e
    @info "  → FAIL" exception=e
end

# Test 4
@info "Test 4: Struct with OffsetVector{ConcreteRArray} (LatLonGrid pattern)"
metric = OffsetVector(Reactant.ConcreteRArray(collect(1.0:Float64(N+1))), 0:N)
g = GridLike(v, metric)
try
    f4! = @compile sync=true run_gridlike!(out, g)
    f4!(out, g)
    @info "  → PASS" result=Array(out)
catch e
    @info "  → FAIL" exception=e
end
