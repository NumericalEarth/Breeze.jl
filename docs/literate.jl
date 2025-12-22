using Literate
using CairoMakie

CairoMakie.activate!(type = "png")
set_theme!(Theme(linewidth = 3))

script_path = ARGS[1]
literated_dir = ARGS[2]

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

@time basename(script_path) Literate.markdown(script_path, literated_dir;
                                              flavor = Literate.DocumenterFlavor(),
                                              preprocess = content -> content * example_postamble,
                                              execute = true,
                                              )
