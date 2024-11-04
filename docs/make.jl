using ActiveInference
using Documenter

DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)

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

        "General Introduction" => [],

        # "General Introduction" => [

        #     "Introduction" => "",
        #     "Creation of the Generative Model" => "",
        #     "Simulation" => "",
        #     "Model Fitting" => "",

        # ],

        # "Usage Examples" => [

        #     "T-Maze Simulation" => "",
        #     "T-Maze Model Fitting" => "",

        # ],

        # "Active Inference Theory" => [

        #     "Perception" => "",
        #     "Action" => "",
        #     "Learning" => "",

        # ],

        # "Why Active Inference?" => [],
    ],
)
### NOTE Sam: change devbranch to master
deploydocs(;
    repo="github.com/ilabcode/ActiveInference.jl",
    devbranch="dev",
)
