using ThesisSimulations
using Documenter

DocMeta.setdocmeta!(ThesisSimulations, :DocTestSetup, :(using ThesisSimulations); recursive=true)

makedocs(;
    modules=[ThesisSimulations],
    authors="Stephan Scholz",
    repo="https://github.com/stephans3/ThesisSimulations.jl/blob/{commit}{path}#{line}",
    sitename="ThesisSimulations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://stephans3.github.io/ThesisSimulations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stephans3/ThesisSimulations.jl",
    devbranch="main",
)
