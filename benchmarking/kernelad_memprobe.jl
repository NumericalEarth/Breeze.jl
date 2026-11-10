#####
##### Memory probe for the kernel AD `jit` pipeline.
#####
##### Parses a `pre_kernelad_binary_pm.mlir` dump (the module Reactant hands to the binary
##### pipeline: ~1600 gpu.modules + their jit_calls) and runs that pipeline one pass at a
##### time, printing the process RSS after each, so the memory can be attributed to a pass.
#####
#####     julia --project=<env with Reactant pb/kernel-ad> kernelad_memprobe.jl <dump.mlir>
#####
##### Environment:
#####   PROBE_CHIP     sm_XY for nvvm-attach-target (default sm_89, the L4). On a CUDA machine the
#####                  default is taken from the device instead.
#####   PROBE_PASSES   comma-separated subset/prefix of the pipeline to run (default: all; the
#####                  CUDA-only passes are skipped automatically without a CUDA toolkit).
#####   PROBE_PRINT    "1": also print the module after each pass to a string and report its
#####                  size (how big the embedded binaries are). Costs memory itself.
#####
##### Without CUDA (e.g. macOS) the probe covers parse, convert-polygeist-to-llvm,
##### strip-gpu-info and nvvm-attach-target, and attempts gpu-module-to-binary with
##### format=isa (PTX only, no ptxas); lower-jit needs the CUDA driver.

using Reactant
using Reactant: MLIR, XLA
using Dates

const file = ARGS[1]

function rss_mib()
    # The CI image has no `ps`; /proc has the same number on Linux
    if Sys.islinux()
        for line in eachline("/proc/self/status")
            startswith(line, "VmRSS:") && return parse(Int, split(line)[2]) ÷ 1024
        end
        error("no VmRSS in /proc/self/status")
    end
    return parse(Int, strip(read(`ps -o rss= -p $(getpid())`, String))) ÷ 1024
end
let last = Ref(0)
    global function report(label)
        GC.gc()
        now = rss_mib()
        println(Dates.format(Dates.now(UTC), "HH:MM:SS"), "  ", rpad(label, 64), lpad(now, 7), " MiB   Δ ", lpad(now - last[], 6), " MiB   maxrss ", Sys.maxrss() ÷ 2^20, " MiB")
        flush(stdout)
        last[] = now
    end
end

has_cuda = Sys.islinux() && !isempty(get(ENV, "CUDA_VISIBLE_DEVICES", "x")) && try
    lowercase(XLA.platform_name(XLA.default_backend())) == "cuda"
catch
    false
end

chip = get(ENV, "PROBE_CHIP", "sm_89")
if has_cuda
    properties = XLA.device_properties(XLA.default_device(XLA.default_backend()))
    chip = "sm_$(properties.major)$(properties.minor)"
end
features = Reactant.Compiler.cubinFeatures()
optlevel = Reactant.Compiler.cuOptLevel[]
cuda_get_kernel = try
    div(MLIR.API.ReactantHermeticCudaGetVersion(), 1000) == 12
catch
    false
end
toolkit = has_cuda ? XLA.CUDA_DATA_DIR[] : ""
format = has_cuda ? Reactant.Compiler.cubinFormat[] : "isa"

pipeline = [
    "convert-polygeist-to-llvm{backend=cuda preserve-xla-funcs=true cuda-get-kernel=$(cuda_get_kernel)}",
    "strip-gpu-info",
    "nvvm-attach-target{chip=$(chip) features=$(features) O=$(optlevel)}",
    "enzymexla-gpu-module-to-binary{format=$(format)" * (isempty(toolkit) ? "}" : " toolkit=$(toolkit)}"),
    "lower-jit",
    "symbol-dce",
]
if haskey(ENV, "PROBE_PASSES")
    wanted = strip.(split(ENV["PROBE_PASSES"], ","))
    pipeline = filter(p -> any(w -> startswith(p, w), wanted), pipeline)
end
!has_cuda && filter!(p -> !startswith(p, "lower-jit"), pipeline)

println("file: ", file, " (", round(filesize(file) / 2^20; digits = 1), " MiB)")
println("cuda: ", has_cuda, "  chip: ", chip, "  features: ", features, "  format: ", format)
println("passes:"); foreach(p -> println("  ", p), pipeline); println()
report("start")

Reactant.initialize_dialect()
MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
MLIR.IR.@with_context ctx begin
    report("MLIR context")
    source = read(file, String)
    ngpu = count("gpu.module @", source)
    # A GEP without indices prints as `%p[]`, which the parser rejects; `%p[0]` over i8 is the
    # same address
    source = replace(source, r"(llvm\.getelementptr %[A-Za-z0-9_]+)\[\] : \(([^)]*)\) -> ([^,]+), [^ ]+" => s"\1[0] : (\2) -> \3, i8")
    report("read file")
    mod = parse(MLIR.IR.Module, source)
    source = nothing
    GC.gc()
    report("parse module")
    println("gpu.modules: ", ngpu)

    for pass in pipeline
        short = String(first(split(pass, "{")))
        println(Dates.format(Dates.now(UTC), "HH:MM:SS"), "  running ", short, " ...")
        flush(stdout)
        t = @elapsed begin
            try
                Reactant.Compiler.run_pass_pipeline!(mod, pass, short; enable_verifier = false)
            catch err
                println("pass ", short, " FAILED: ", sprint(showerror, err)[1:min(end, 400)])
                report("after $short (failed)")
                break
            end
        end
        report("after $short ($(round(t; digits = 1)) s)")
        if get(ENV, "PROBE_PRINT", "0") == "1"
            s = string(mod)
            println("    module text: ", round(sizeof(s) / 2^20; digits = 1), " MiB")
            s = nothing
            report("after printing $short")
        end
    end
end
end
report("context disposed")
