using ActiveInference
using Documenter
using Literate

DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)

# Trying to make the output appear in the markdown_files folder without luck
# Literate.markdown(raw"julia_files\Introduction.jl", outputdir="./src", execute=false, documenter = false)
# Literate.markdown(raw"julia_files\GenerativeModelCreation.jl", outputdir="./src", execute=false, documenter = false)


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
        
        "Home" => "index.md",

        "General Introduction" => [

            "Introduction" => "Introduction.md",
            "Creation of the Generative Model" => "GenerativeModelCreation.md",
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
