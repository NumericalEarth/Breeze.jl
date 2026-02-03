#=
Investigation: DelinearizingIndexPassing Segmentation Fault (B.6.3)
Status: ACTIVE INVESTIGATION
Purpose: Near-MWE testing periodic indexing patterns WITHOUT Oceananigans

This test removes Oceananigans entirely and tests whether the issue is in:
- Reactant's handling of modular/periodic indexing patterns
- The delinearizing pass with certain array access patterns

If this fails with --check-bounds=yes, the issue is purely in Reactant.
If this passes, the issue is in how Oceananigans constructs its halo operations.

Run with: julia --project=. test/delinearize/test_delinearize_periodic_indexing_mwe.jl
Run with bounds checking: julia --project=. --check-bounds=yes test/delinearize/test_delinearize_periodic_indexing_mwe.jl
=#

using Reactant
using Enzyme
using Statistics: mean

# Log versions and options
@info "Package versions" Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
@info "Julia options" check_bounds=Base.JLOptions().check_bounds

Reactant.set_default_backend("cpu")

#####
##### Test 1: Simple array operations (baseline)
#####

@info "=" ^ 60
@info "Test 1: Simple array operations (baseline)"
@info "=" ^ 60

@time "Creating ConcreteRArray" arr = Reactant.ConcreteRArray(randn(6, 6))

function simple_loss(a)
    return mean(a.^2)
end

function grad_simple(a, da)
    parent(da) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        simple_loss, Enzyme.Active,
        Enzyme.Duplicated(a, da)
    )
    return da, loss_value
end

@time "Creating shadow array" darr = Reactant.ConcreteRArray(zeros(6, 6))

@info "Testing simple array autodiff..."
try
    @time "Compiling grad_simple" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_simple(arr, darr)
    @time "Running grad_simple" da_result, loss_val = compiled(arr, darr)
    @info "✅ Simple array SUCCESS" loss_value=loss_val
catch e
    @error "❌ Simple array FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 2: Periodic boundary copy (manual halo filling)
#####

@info "=" ^ 60
@info "Test 2: Manual periodic boundary copy"
@info "=" ^ 60

# Simulate a field with halos: interior is [2:5, 2:5], halos are [1,6]
# Total size 6x6, interior 4x4, halo width 1

@time "Creating array with halos" arr2 = Reactant.ConcreteRArray(randn(6, 6))
@time "Creating shadow array" darr2 = Reactant.ConcreteRArray(zeros(6, 6))

# Manual periodic halo filling (what fill_halo_regions! does for periodic BCs)
function periodic_halo_fill!(a)
    # For a 6x6 array with 1-cell halos around a 4x4 interior:
    # Interior indices: 2:5 in both dimensions
    # Halo indices: 1 and 6
    
    # Copy right edge to left halo (periodic in x)
    @inbounds for j in 1:6
        a[1, j] = a[5, j]  # left halo = right interior edge
        a[6, j] = a[2, j]  # right halo = left interior edge
    end
    
    # Copy top edge to bottom halo (periodic in y)
    @inbounds for i in 1:6
        a[i, 1] = a[i, 5]  # bottom halo = top interior edge
        a[i, 6] = a[i, 2]  # top halo = bottom interior edge
    end
    
    return nothing
end

function loss_with_periodic_halo(a)
    periodic_halo_fill!(a)
    # Use interior only
    interior_sum = zero(eltype(a))
    @inbounds for i in 2:5, j in 2:5
        interior_sum += a[i, j]^2
    end
    return interior_sum / 16  # mean over 4x4 interior
end

function grad_periodic_halo(a, da)
    parent(da) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_with_periodic_halo, Enzyme.Active,
        Enzyme.Duplicated(a, da)
    )
    return da, loss_value
end

@info "Testing periodic halo autodiff..."
try
    @time "Compiling grad_periodic_halo" compiled2 = Reactant.@compile raise_first=true raise=true sync=true grad_periodic_halo(arr2, darr2)
    @time "Running grad_periodic_halo" da_result, loss_val = compiled2(arr2, darr2)
    @info "✅ Periodic halo SUCCESS" loss_value=loss_val
catch e
    @error "❌ Periodic halo FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 3: Periodic halo with modular arithmetic
#####

@info "=" ^ 60
@info "Test 3: Periodic halo with mod() indexing"
@info "=" ^ 60

@time "Creating array for mod test" arr3 = Reactant.ConcreteRArray(randn(6, 6))
@time "Creating shadow array" darr3 = Reactant.ConcreteRArray(zeros(6, 6))

# Use mod for periodic indexing (more like what KernelAbstractions might generate)
function periodic_halo_fill_mod!(a)
    n = 4  # interior size
    h = 1  # halo width
    
    # Fill halos using modular arithmetic
    @inbounds for i in 1:6, j in 1:6
        # Only update halo cells
        if i == 1 || i == 6 || j == 1 || j == 6
            # Map to periodic interior index
            ii = mod1(i - h, n) + h
            jj = mod1(j - h, n) + h
            if ii != i || jj != j
                a[i, j] = a[ii, jj]
            end
        end
    end
    return nothing
end

function loss_with_mod_halo(a)
    periodic_halo_fill_mod!(a)
    interior_sum = zero(eltype(a))
    @inbounds for i in 2:5, j in 2:5
        interior_sum += a[i, j]^2
    end
    return interior_sum / 16
end

function grad_mod_halo(a, da)
    parent(da) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_with_mod_halo, Enzyme.Active,
        Enzyme.Duplicated(a, da)
    )
    return da, loss_value
end

@info "Testing mod-based periodic halo autodiff..."
try
    @time "Compiling grad_mod_halo" compiled3 = Reactant.@compile raise_first=true raise=true sync=true grad_mod_halo(arr3, darr3)
    @time "Running grad_mod_halo" da_result, loss_val = compiled3(arr3, darr3)
    @info "✅ Mod halo SUCCESS" loss_value=loss_val
catch e
    @error "❌ Mod halo FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 4: Loop with periodic operations (like time stepping)
#####

@info "=" ^ 60
@info "Test 4: Loop with periodic operations"
@info "=" ^ 60

@time "Creating array for loop test" arr4 = Reactant.ConcreteRArray(randn(6, 6))
@time "Creating shadow array" darr4 = Reactant.ConcreteRArray(zeros(6, 6))

function loss_with_loop_halos(a, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for step in 1:nsteps
        # Modify interior
        @inbounds for i in 2:5, j in 2:5
            a[i, j] = a[i, j] * 0.99
        end
        # Fill halos
        periodic_halo_fill!(a)
    end
    
    interior_sum = zero(eltype(a))
    @inbounds for i in 2:5, j in 2:5
        interior_sum += a[i, j]^2
    end
    return interior_sum / 16
end

function grad_loop_halos(a, da, nsteps)
    parent(da) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_with_loop_halos, Enzyme.Active,
        Enzyme.Duplicated(a, da),
        Enzyme.Const(nsteps)
    )
    return da, loss_value
end

nsteps = 4
@info "Testing loop with periodic halos (nsteps=$nsteps)..."
try
    @time "Compiling grad_loop_halos" compiled4 = Reactant.@compile raise_first=true raise=true sync=true grad_loop_halos(arr4, darr4, nsteps)
    @time "Running grad_loop_halos" da_result, loss_val = compiled4(arr4, darr4, nsteps)
    @info "✅ Loop halos SUCCESS" loss_value=loss_val
catch e
    @error "❌ Loop halos FAILED" exception=(e, catch_backtrace())
end

@info "=" ^ 60
@info "All tests completed"
@info "=" ^ 60
