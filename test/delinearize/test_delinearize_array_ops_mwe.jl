#=
MWE: Progressive array operation tests to isolate the --check-bounds=yes issue
This tests the operations that happen inside Oceananigans' set!(field, field)

Run with: julia --project=test test/delinearize/test_delinearize_array_ops_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_array_ops_mwe.jl
=#

using Reactant
using Enzyme
using Statistics: mean

@info "Julia options" check_bounds=Base.JLOptions().check_bounds
Reactant.set_default_backend("cpu")

#####
##### Test 1: Simple array broadcast (a .= b)
#####

@info "=" ^ 50
@info "Test 1: Array broadcast a .= b"

arr = Reactant.ConcreteRArray(randn(4, 4))
darr = Reactant.ConcreteRArray(zeros(4, 4))

function loss_broadcast(a, b)
    a .= b
    return mean(a.^2)
end

function grad_broadcast(a, da, b, db)
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_broadcast, Enzyme.Active,
        Enzyme.Duplicated(a, da), Enzyme.Duplicated(b, db))
    return da, lv
end

b = Reactant.ConcreteRArray(randn(4, 4))
db = Reactant.ConcreteRArray(zeros(4, 4))

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_broadcast(arr, darr, b, db)
    @time "Running" result = compiled(arr, darr, b, db)
    @info "✅ Test 1 PASSED"
catch e
    @error "❌ Test 1 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 2: View broadcast (view(a) .= view(b))
#####

@info "=" ^ 50
@info "Test 2: View broadcast view(a) .= view(b)"

arr2 = Reactant.ConcreteRArray(randn(6, 6))  # With "halos"
darr2 = Reactant.ConcreteRArray(zeros(6, 6))
b2 = Reactant.ConcreteRArray(randn(6, 6))
db2 = Reactant.ConcreteRArray(zeros(6, 6))

function loss_view_broadcast(a, b)
    # Simulate interior view (indices 2:5 in a 6x6 array with 1-cell halo)
    view(a, 2:5, 2:5) .= view(b, 2:5, 2:5)
    return mean(view(a, 2:5, 2:5).^2)
end

function grad_view_broadcast(a, da, b, db)
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_view_broadcast, Enzyme.Active,
        Enzyme.Duplicated(a, da), Enzyme.Duplicated(b, db))
    return da, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_view_broadcast(arr2, darr2, b2, db2)
    @time "Running" result = compiled(arr2, darr2, b2, db2)
    @info "✅ Test 2 PASSED"
catch e
    @error "❌ Test 2 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 3: OffsetArray-like indexing
#####

@info "=" ^ 50
@info "Test 3: OffsetArray-style parent access"

arr3 = Reactant.ConcreteRArray(randn(10, 10, 8))  # Simulating field with halos
darr3 = Reactant.ConcreteRArray(zeros(10, 10, 8))
b3 = Reactant.ConcreteRArray(randn(10, 10, 8))
db3 = Reactant.ConcreteRArray(zeros(10, 10, 8))

# Simulate interior indices: 1+H:N+H where H=3, N=4 → 4:7
function loss_offset_style(a, b)
    H = 3
    N = 4
    i = (1+H):(N+H)
    j = (1+H):(N+H)
    k = (1+H):(N-2+H)  # Nz=2
    view(a, i, j, k) .= view(b, i, j, k)
    return mean(view(a, i, j, k).^2)
end

function grad_offset_style(a, da, b, db)
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_offset_style, Enzyme.Active,
        Enzyme.Duplicated(a, da), Enzyme.Duplicated(b, db))
    return da, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_offset_style(arr3, darr3, b3, db3)
    @time "Running" result = compiled(arr3, darr3, b3, db3)
    @info "✅ Test 3 PASSED"
catch e
    @error "❌ Test 3 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 4: Mutable struct field assignment (like your test_delinearize.jl)
##### Note: Struct must be parametric for Reactant tracing to work
#####

@info "=" ^ 50
@info "Test 4: Mutable struct field assignment"

mutable struct ArrayWrapper{A}
    data::A
end

function set_wrapper!(w::ArrayWrapper, src)
    w.data .= src
    return nothing
end

function loss_wrapper(w, src)
    set_wrapper!(w, src)
    return mean(w.data.^2)
end

function grad_wrapper(w, dw, src, dsrc)
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_wrapper, Enzyme.Active,
        Enzyme.Duplicated(w, dw), Enzyme.Duplicated(src, dsrc))
    return dw, lv
end

w = ArrayWrapper(Reactant.ConcreteRArray(randn(4, 4)))
dw = ArrayWrapper(Reactant.ConcreteRArray(zeros(4, 4)))
src = Reactant.ConcreteRArray(randn(4, 4))
dsrc = Reactant.ConcreteRArray(zeros(4, 4))

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_wrapper(w, dw, src, dsrc)
    @time "Running" result = compiled(w, dw, src, dsrc)
    @info "✅ Test 4 PASSED"
catch e
    @error "❌ Test 4 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 5: Nested struct with array (closer to Oceananigans model)
##### Note: Structs must be parametric for Reactant tracing to work
#####

@info "=" ^ 50
@info "Test 5: Nested struct (model.tracers.T pattern)"

mutable struct Tracer{A}
    data::A
end

mutable struct Tracers{T}
    T::T
end

mutable struct SimpleModel{TR}
    tracers::TR
end

function set_tracer!(model::SimpleModel, src)
    model.tracers.T.data .= src
    return nothing
end

function loss_model(model, src)
    set_tracer!(model, src)
    return mean(model.tracers.T.data.^2)
end

function grad_model(model, dmodel, src, dsrc)
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_model, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(src, dsrc))
    return dmodel, lv
end

model = SimpleModel(Tracers(Tracer(Reactant.ConcreteRArray(randn(4, 4, 2)))))
dmodel = Enzyme.make_zero(model)
src5 = Reactant.ConcreteRArray(randn(4, 4, 2))
dsrc5 = Reactant.ConcreteRArray(zeros(4, 4, 2))

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_model(model, dmodel, src5, dsrc5)
    @time "Running" result = compiled(model, dmodel, src5, dsrc5)
    @info "✅ Test 5 PASSED"
catch e
    @error "❌ Test 5 FAILED" exception=(e, catch_backtrace())
end

@info "=" ^ 50
@info "All tests completed"
