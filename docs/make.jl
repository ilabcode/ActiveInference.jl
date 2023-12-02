using Actify
using Documenter

DocMeta.setdocmeta!(Actify, :DocTestSetup, :(using Actify); recursive=true)

makedocs(;
    modules=[Actify],
    authors="Jonathan Ehrenreich Laursen, Samuel William Nehrer",
    repo="https://github.com/samuelnehrer02/Actify.jl/blob/{commit}{path}#{line}",
    sitename="Actify.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://samuelnehrer02.github.io/Actify.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/samuelnehrer02/Actify.jl",
    devbranch="master",
)
