# Contributors guide

## Developing `Breeze.jl` locally

To develop `Breeze.jl` locally, git clone the repo:

```sh
git clone https://github.com/NumericalEarth/Breeze.jl
```

## [Running the tests](@id running-tests)

After entering the top-level directory of your local clone of `Breeze.jl` (`cd Breeze.jl`), start Julia with

```sh
julia --project=.
```

to activate the environment of the package.
Then, in an interactive Julia REPL you just started, you can run the tests by typing `]` to enter the Pkg mode and then run

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

Similarly, to run only the doctests, you can use the command

```julia
import Pkg
Pkg.test("Breeze"; test_args=`doctests`)
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

### GPU tests

When running the tests, if a CUDA GPU is detected, they automatically use the [`GPU` Oceananigans architecture](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Architectures.GPU), otherwise they run on [`CPU`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library/#Oceananigans.Architectures.CPU).
To temporarily disable the automatic detection of the GPU and forcibly run the tests on CPU you can set the environment variable `CUDA_VISIBLE_DEVICES=-1`.
For example, from within a Julia session you can do

```julia
ENV["CUDA_VISIBLE_DEVICES"] = "-1"
import Pkg
Pkg.test("Breeze")
```

!!! note "Contributing new tests"

    When contributing new tests to `Breeze.jl`, make sure to pass to the grid the global variable `default_arch`, defined in the [init code of all tests](https://github.com/NumericalEarth/Breeze.jl/blob/155344abedf5a8739202c5faac275c2ca2576680/test/runtests.jl#L10-L19), unless you specifically want to use a different architecture.

## Coding style

### Explicitly imported packages

The `Breeze.jl` community doesn't currently enforce a strict coding style, but it uses the package [`ExplicitImports.jl`](https://github.com/JuliaTesting/ExplicitImports.jl) to ensure consistency of loaded modules and accessed functions and variables.
This is checked during the [tests](@ref running-tests), so you may get test failures if you don't follow the prescribed package importing style, the test error message will contain information to suggest you how to fix the issues, read it carefully.
See [`ExplicitImports.jl` documentation](https://juliatesting.github.io/ExplicitImports.jl/) for the motivation of this style.

## Building the documentation locally

`Breeze.jl` [documentation](https://numericalearth.github.io/BreezeDocumentation/) is generated using [`Documenter.jl`](https://github.com/JuliaDocs/Documenter.jl).
You can preview how the documentation will look like with your changes by building the documentation locally. 
From the top-level directory of your local repository run

```sh
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'
```

to instantiate the documentation environment and then

```sh
julia --project=docs/ docs/make.jl
```

to build the documentation.
If you want to quickly build a draft copy of the documentation (i.e. without running all the examples or running the doctests), modify the [call to the `makedocs`](https://github.com/NumericalEarth/Breeze.jl/blob/cdf8bd25c83f24cbd4f26c8c600d20ef9740e9c7/docs/make.jl#L14-L34) function in `docs/make.jl` to set the keyword argument `draft=true` and run again the `docs/make.jl` script.
When you submit a pull request to `Breeze.jl`, if the documentation building job is successfull a copy of the build will be uploaded as an artifact, which you can retrieve by looking at the summary page of the documentation job.

To view the documentation you can open the generated HTML files in the `docs/build` directory, but you need an HTTP server to be able to move around the website and follow internal links.
The [`LiveServer`](https://github.com/JuliaDocs/LiveServer.jl) package provides a simple HTTP server implementation, which also automatically reloads the pages when they are modified on disk:

```julia
import Pkg
Pkg.activate("live-server"; shared=true)
Pkg.add("LiveServer") # this is necessary only the first time, to install LiveServer
using LiveSever: serve
serve(; dir="docs/build")
```
