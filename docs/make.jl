using Documenter, PRIMME

makedocs(
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical = "https://RalphAS.github.io/PRIMME.jl/stable/",
        assets=String[],
    ),
    sitename = "PRIMME.jl",
    modules = [PRIMME],
    pages = [
        "Home" => "index.md",
        "Hermitian Eigensystems" => "eigs.md",
        "Partial Singular Value Decompositions" => "svds.md",
        "Distributed Computation" => "mpi.md",
        # "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/RalphAS/PRIMME.jl.git",
    devbranch = "main",
    target = "build",
    deps = nothing,
    make = nothing
)
