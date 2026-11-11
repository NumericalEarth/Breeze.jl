using Literate
using CairoMakie

CairoMakie.activate!(type = "png")
set_theme!(Theme(linewidth = 3))

script_path = ARGS[1]
literated_dir = ARGS[2]

# Have continuous line numbering in literate examples:
# <https://fredrikekre.github.io/DocumenterCodeBlocks.jl/stable/linenumbers/#Continued-numbering>.
example_preamble = """
# ```@codeblocks
# line_counter = :continue
# ```

"""

# We'll append the following postamble to the literate examples, to include
# information about the computing environment used to run them.
example_postamble = """

# ---

# ### Julia version and environment information
#
# This example was executed with the following version of Julia:

using InteractiveUtils: versioninfo
versioninfo()

# These were the top-level packages installed in the environment:

import Pkg
Pkg.status()
"""

# ARM inputs must be staged explicitly; render the LASSO example without executing
# it when they are absent. No substitute case is downloaded for the docs build.
execute_example = basename(script_path) != "ena_differentiability.jl" || haskey(ENV, "ENA_LASSO_BUNDLE")

@time basename(script_path) Literate.markdown(script_path, literated_dir;
                                              flavor = execute_example ? Literate.DocumenterFlavor() : Literate.CommonMarkFlavor(),
                                              preprocess = content -> example_preamble * content * (execute_example ? example_postamble : ""),
                                              execute = execute_example,
                                              )
