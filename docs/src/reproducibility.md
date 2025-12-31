# [Reproducibility of Breeze.jl models](@id reproducibility)

`Breeze.jl` cannot guarantee _bitwise_ reproducibility of atmospheric simulations across Julia versions and machine architectures.
For [chaotic](https://en.wikipedia.org/wiki/Chaos_theory) simulations, bitwise differences between two initially identical solutions can amplify, leading to qualitatively different trajectories after long simulation times.
In this page we analyze some sources of non-reproducibility, and how to control them.

## Sources of non-reproducibility

When comparing the result of two different runs of the same code, possible sources of non-reproducibility of `Breeze.jl` models include

* some special functions in Julia `Base` may have different rounding errors on different CPU architectures (e.g. `aarch64` vs `x86-64`), even though they are usually consistent within very few [ULPs](https://en.wikipedia.org/wiki/Unit_in_the_last_place)
* in general, the compiler can generate different code on different CPUs, even within the same architecture, and sometimes even on the same CPU but across different versions of Julia (if, for example, a newer version of LLVM introduced different optimizations).
  This is particularly evident when using aggressive optimization levels (like `-O2`, which is the default in Julia), which lead to different vectorization optimizations.
* multi-threaded `for` loops can further cause differences, when the order of the loops is important to exactly reproduce the same results
* using randomly generated numbers within the simulations, if not fixing the seed, will result in different output.
  Running multi-threaded simulations can lead to different results even when fixing the random-number generator (RNG) seed if the random numbers are used inside threaded loops, because the scheduler may reorder the loops differently, falling in the point above
* when a fast Fourier transform (FFT) is involved, certain FFTW flags may not produce consistent results (see discussion in [CliMA/Oceananigans.jl#2790](https://github.com/CliMA/Oceananigans.jl/discussions/2790))
* running simulations on completely different devices (e.g. CPU vs GPU) will also produce very different results because of the all the points above.

## Controlling reproducibility

There are multiple strategies that can help reproducibility of `Breeze.jl` simulations, based on the previous section, the more you can adopt, the better.
Here are some of them:

* the easiest way to reproduce the results of another simulation, is to use a similar system (same Julia and `Breeze.jl` version, same CPU, same accelerator, if any, etc.).
  Of course this may not always be possible, especially when it comes to hardware variations, but in general you can't expect to be able to reproduce on CPU a simulation run on GPU, and viceversa, as the code generation for these targets is extremely different.
  Similarly, when targeting different CPU architectures (e.g. `x86-64` vs `aarch64`) small numerical differences are unavoidable due to Julia's `Base` numerical functions
*  when running simulations on different CPUs of the same architecture, using lower optimization levels (e.g. `-O0`) can reduce numerical differences and increase reproducibility, but it also generates very slow code, so this may not always be a practical solution
* if using randomly generated numbers, setting the seed (e.g. [`Random.seed!`](https://docs.julialang.org/en/v1/stdlib/Random/#Random.seed!) for CPU code, and `CUDA.seed!` for CUDA code) at the beginning of the simulation should help reproducibility
* running CPU code on a single thread reduces variations due to different scheduling.
  Also in this case, the code will be running slower than if multi-threaded and so this may not necessarily a practical solution, but it could be an option if comparing results between different runs is important
* if your model is using FFT-based solvers from Oceananigans, passing the `FFTW.ESTIMATE` flag should help reproducibility
* it may also be worth pointing out that chaotic divergences should emerge after a few timesteps of a simulation, while the first few tens or hundreds should still be consistent within a reasonable tolerance.
