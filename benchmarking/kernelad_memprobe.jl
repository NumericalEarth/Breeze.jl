#####
##### Memory probe for the kernel AD `jit` pipeline.
#####
##### Parses a `pre_kernelad_binary_pm.mlir` dump (the module Reactant hands to the binary
##### pipeline: ~1600 gpu.modules + their jit_calls) and runs that pipeline one pass at a
##### time, printing the process RSS after each, so the memory can be attributed to a pass.
#####
#####     julia --project=<env with Reactant pb/kernel-ad> kernelad_memprobe.jl <dump.mlir>
#####
##### Each experiment re-parses the dump in a fresh MLIR context and runs the pipeline with a
##### binary format and a lower-jit `jit` option (true compiles and loads each call's host
##### module, false only builds and prints them), so the memory lower-jit retains can be split
##### between its submodules and what the JIT and the CUDA module loads keep. On CUDA the
##### default experiments are, in order (the memory the JIT retains is never freed, so the
##### lean ones go first):
#####
#####     fatbin:false   build and print the submodules, compile nothing
#####     bin:true       cubin only: a driver that loads SASS has nothing to JIT
#####     isa:true       PTX only: the driver JITs every module
#####
##### Environment:
#####   PROBE_EXPERIMENTS   comma-separated `format:jit` pairs overriding the defaults
#####   PROBE_CHIP          sm_XY for nvvm-attach-target (default sm_89, the L4; on a CUDA
#####                       machine the device's own value is used)
#####   PROBE_PRINT         "1": print the module after each pass to a string and report its
#####                       size (how big the embedded binaries are). Costs memory itself.
#####   PROBE_STRIP_CTORS   "1": drop the llvm.mlir.global_ctors/dtors ops before the pipeline, so
#####                       lower-jit still JIT-compiles every call's host module but no
#####                       constructor runs: no CUDA module is loaded. Splits the JIT's own
#####                       retention from the driver's.
#####
##### Without CUDA (e.g. macOS) the probe covers parse, convert-polygeist-to-llvm,
##### strip-gpu-info and nvvm-attach-target, and gpu-module-to-binary with format=isa (PTX
##### only, no ptxas); lower-jit needs the CUDA driver for jit=true and is then skipped, while
##### jit=false (build and print the submodules only) runs anywhere.

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

stamp() = Dates.format(Dates.now(UTC), "HH:MM:SS")

let last = Ref(0)
    global function report(label)
        GC.gc()
        now = rss_mib()
        println(stamp(), "  ", rpad(label, 64), lpad(now, 7), " MiB   Δ ", lpad(now - last[], 6), " MiB   maxrss ", Sys.maxrss() ÷ 2^20, " MiB")
        flush(stdout)
        last[] = now
    end
end

has_cuda = Sys.islinux() && try
    lowercase(XLA.platform_name(XLA.default_backend())) == "cuda"
catch
    false
end

chip = get(ENV, "PROBE_CHIP", "sm_89")
if has_cuda
    properties = XLA.device_properties(XLA.default_device())
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

experiments = if haskey(ENV, "PROBE_EXPERIMENTS")
    [(String(f), parse(Bool, j)) for (f, j) in split.(split(ENV["PROBE_EXPERIMENTS"], ","), ":")]
elseif has_cuda
    [("fatbin", false), ("bin", true), ("isa", true)]
else
    [("isa", false)]
end

function pipeline(format, jit)
    passes = [
        "convert-polygeist-to-llvm{backend=cuda preserve-xla-funcs=true cuda-get-kernel=$(cuda_get_kernel)}",
        "strip-gpu-info",
        "nvvm-attach-target{chip=$(chip) features=$(features) O=$(optlevel)}",
        "enzymexla-gpu-module-to-binary{format=$(format)" * (isempty(toolkit) ? "}" : " toolkit=$(toolkit)}"),
        "lower-jit{jit=$(jit)}",
        "symbol-dce",
    ]
    has_cuda || !jit || filter!(p -> !startswith(p, "lower-jit"), passes)
    return passes
end

println("file: ", file, " (", round(filesize(file) / 2^20; digits = 1), " MiB)")
println("cuda: ", has_cuda, "  chip: ", chip, "  features: ", features, "  toolkit: ", toolkit)
println("experiments: ", join(("$f:$j" for (f, j) in experiments), ", "))
println()
report("start")

Reactant.initialize_dialect()

source = read(file, String)
ngpu = count("gpu.module @", source)
# A GEP without indices prints as `%p[]`, which the parser rejects; `%p[0]` over i8 is the
# same address
source = replace(source, r"(llvm\.getelementptr %[A-Za-z0-9_]+)\[\] : \(([^)]*)\) -> ([^,]+), [^ ]+" => s"\1[0] : (\2) -> \3, i8")
if get(ENV, "PROBE_STRIP_CTORS", "0") == "1"
    nctors = count("llvm.mlir.global_ctors", source)
    source = replace(source, r"\n  llvm\.mlir\.global_[cd]tors [^\n]*" => "")
    println("stripped ", nctors, " global_ctors (and the dtors): no CUDA module will be loaded")
end
report("read file ($ngpu gpu.modules)")

for (format, jit) in experiments
    println("\n===== experiment format=$format jit=$jit =====")
    passes = pipeline(format, jit)
    foreach(p -> println("  ", p), passes)
    report("before experiment")
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
    MLIR.IR.@with_context ctx begin
        mod = parse(MLIR.IR.Module, source)
        report("parse module")
        for pass in passes
            short = String(first(split(pass, "{")))
            println(stamp(), "  running ", pass, " ...")
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
end
