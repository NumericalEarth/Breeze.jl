using Reactant
using Reactant: @trace
using Enzyme
using CUDA # Required by Reactant's kernel raising, including its CPU backend.

function ena_prediction!(workspace, controls, baseline, sampling, Δt, steps)
    prediction = ena_forecast_begin!(workspace, controls, baseline, sampling, Δt)
    @trace mincut=true checkpointing=true track_numbers=false for step in 2:steps
        ena_forecast_step!(prediction, workspace, sampling, Δt, step)
    end
    return prediction
end

function ena_objective!(workspace, controls, baseline, target, sampling, Δt, steps, regularization)
    prediction = ena_prediction!(workspace, controls, baseline, sampling, Δt, steps)
    return ena_fit_loss(prediction, target, controls, regularization)
end

function ena_gradient!(workspace, dworkspace, controls, dcontrols,
                        baseline, target, sampling, Δt, steps, regularization)
    # This shadow was created with make_zero: its immutable scalar fields are
    # already zero. Reset mutable storage without trying to replace those fields.
    Enzyme.remake_zero!(dworkspace)
    dcontrols .= 0
    _, value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                              ena_objective!, Enzyme.Active,
                              Enzyme.Duplicated(workspace, dworkspace),
                              Enzyme.Duplicated(controls, dcontrols),
                              Enzyme.Const(baseline), Enzyme.Const(target), Enzyme.Const(sampling),
                              Enzyme.Const(Δt), Enzyme.Const(steps), Enzyme.Const(regularization))
    return value, dcontrols
end

# P3 kernel inference needs more stack than Julia's default Task stack on macOS.
function ena_with_stack(function_to_run)
    task = Task(function_to_run, 64 * 1024^2)
    schedule(task)
    return fetch(task)
end

function ena_compile_inverse(problem, target; Δt=1, steps=2, regularization=0.01)
    workspace, baseline, controls = problem.workspace, problem.baseline, problem.controls
    arch = architecture(workspace.model)
    FT = eltype(controls)
    device(array) = on_architecture(arch, FT.(array))
    sampling = device(ena_sampling_weights(target, Δt, steps))
    target = (values=device(target.values), weights=device(target.weights))
    dworkspace = Enzyme.make_zero(workspace)
    dcontrols = zero(controls)
    Δt, regularization = FT(Δt), FT(regularization)
    @info "Compiling ENA P3 objective" steps Δt
    forward = ena_with_stack() do
        Reactant.@compile raise=true raise_first=true sync=true ena_objective!(
            workspace, controls, baseline, target, sampling, Δt, steps, regularization)
    end
    @info "Compiling ENA P3 reverse derivative" steps Δt
    backward = ena_with_stack() do
        Reactant.@compile raise=true raise_first=true sync=true ena_gradient!(
            workspace, dworkspace, controls, dcontrols, baseline, target, sampling, Δt, steps, regularization)
    end
    value = function (parameters)
        copyto!(controls, parameters)
        return Float64(ena_with_stack(() -> forward(workspace, controls, baseline, target, sampling, Δt, steps, regularization)))
    end
    value_gradient = function (parameters)
        copyto!(controls, parameters)
        loss, gradient = ena_with_stack() do
            backward(workspace, dworkspace, controls, dcontrols, baseline, target, sampling, Δt, steps, regularization)
        end
        return Float64(loss), Array(gradient)
    end
    return (; value, value_gradient)
end
