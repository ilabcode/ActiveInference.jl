using ActiveInference
using Documenter
using Literate

DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)

# Trying to make the output appear in the markdown_files folder without luck
# markdown_path = raw"src\markdown_files"
# Literate.markdown(raw"docs\src\julia_files\Introduction.jl", outputdir=joinpath(@__DIR__, markdown_path), documenter=true)

makedocs(;
    modules=[ActiveInference, ActiveInference.Environments],
    authors="Jonathan Ehrenreich Laursen, Samuel William Nehrer",
    repo="https://github.com/ilabcode/ActiveInference.jl/blob/{commit}{path}#{line}",
    sitename="ActiveInference.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ilabcode.github.io/ActiveInference.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        
        "Home" => "docs/src/markdown_files/index.md",

        "General Introduction" => [

            "Introduction" => "docs/src/markdown_files/Introduction.md",
            "Creation of the Generative Model" => [],
            "Simulation" => [],
            "Model Fitting" => [],

        ],

        "Usage Examples" => [

            "T-Maze Simulation" => [],
            "T-Maze Model Fitting" => [],

        ],

        "Active Inference Theory" => [

            "Perception" => [],
            "Action" => [],
            "Learning" => [],

        ],

        "Why Active Inference?" => [],
    ],
)
### NOTE Sam: change devbranch to master
deploydocs(;
    repo="github.com/ilabcode/ActiveInference.jl",
    devbranch="dev",
)
