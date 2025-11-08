# Contributors guide

## Developing `Breeze.jl` locally

The easiest way to develop `Breeze.jl` locally is to run the following command in the Julia REPL:

```julia
import Pkg
Pkg.dev("https://github.com/NumericalEarth/Breeze.jl")
```

This will automatically clone the repository to `~/.julia/dev/Breeze`, where you can then hack its source code.

## [Running the tests](@id running-tests)

To run the tests of `Breeze.jl`, in an interactive Julia REPL, after having activated the environment in you which you installed `Breeze.jl`, you can type `]` to enter the Pkg mode and then run

```
test Breeze
```

or, in the normal REPL mode (not Pkg) run the commands

```julia
import Pkg
Pkg.test("Breeze")
```

`Breeze.jl` uses [`ParallelTestRunner.jl`](https://github.com/JuliaTesting/ParallelTestRunner.jl) for distributing the tests and running them in parallel.
Read the documentation of `ParallelTestRunner.jl` for more information about it, but interesting arguments are

* `--jobs N` to use `N` jobs for running the tests
* `--verbose` to print more information while the tests are running (e.g. when a test job starts, duration of each job, etc.)
* the list of tests to run, excluding all others, this can be useful for quickly running only a subset of the whole tests.

You can pass the arguments with the `test_args` keyword argument to `Pkg.test`, for example

```julia
import Pkg
Pkg.test("Breeze"; test_args=`--verbose --jobs 2 moist_air atmosphere`)
```

!!! note "List of tests"

    The names of the test jobs are the file names under the `test` directory, without the `.jl` extension, excluding the `runtests.jl` file.
    Filtering test names is done by matching the provided arguments with [`startswith`](https://docs.julialang.org/en/v1/base/strings/#Base.startswith), so you can use the first few letters of the test names.
    Be sure not to catch also other tests you want to skip.
    To see the full list of available tests you can use the `--list` option:

    ```julia
    import Pkg
    Pkg.test("Breeze"; test_args=`--list`)
    ```

## Coding style

### Explicitly imported packages

The `Breeze.jl` community doesn't currently enforce a strict coding style, but it uses the package [`ExplicitImports.jl`](https://github.com/JuliaTesting/ExplicitImports.jl) to ensure consistency of loaded modules and accessed functions and variables.
This is checked during the [tests](@ref running-tests), so you may get test failures if you don't follow the prescribed package importing style, the test error message will contain information to suggest you how to fix the issues, read it carefully.
See [`ExplicitImports.jl` documentation](https://juliatesting.github.io/ExplicitImports.jl/) for the motivation of this style.

## Building the documentation locally

`Breeze.jl` [documentation](https://numericalearth.github.io/BreezeDocumentation) is generated using [`Documenter.jl`](https://github.com/JuliaDocs/Documenter.jl).
You can preview how the documentation will look like with your changes by building the documentation locally. 
From the top-level directory of your local repository run

```sh
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

to instantiate the documentation environment and then

```sh
julia --project=docs/ docs/make.jl
```

to build the documentation.
If you want to quickly build a draft copy of the documentation (i.e. without running all the examples or running the doctests), modify the [call to the `makedocs`](https://github.com/NumericalEarth/Breeze.jl/blob/073f16e7819b310f0ef68e1f41187965323fc1a0/docs/make.jl#L14-L30) function in `docs/make.jl` to add the keyword argument `draft=true` and run again the `docs/make.jl` script.
When you submit a pull request to `Breeze.jl`, if the documentation building job is successfull a copy of the build will be uploaded as an artifact, which you can retrieve by looking at the summary page of the documentation job.
