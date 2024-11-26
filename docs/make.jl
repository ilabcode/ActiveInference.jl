using ActiveInference
using Documenter
using Literate

# cd(raw"C:\Users\Jonathan Laursen\Desktop\University\Bachelor\ActiveInference.jl\docs\src")

# if haskey(ENV, "GITHUB_WORKSPACE")
#     project_dir = ENV["GITHUB_WORKSPACE"]
# else
#     project_dir = pwd()
# end

# cd(joinpath(project_dir, "docs", "src"))

DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)

# Automating the creating of the markdown files
# input_folder = raw"..\julia_files"

# julia_files = filter(file -> endswith(file, ".jl"), readdir(input_folder))

# for file in julia_files
#     input_path = joinpath(input_folder, file)
#     Literate.markdown(input_path, outputdir="", execute=true, documenter=true, codefence =  "```julia" => "```")
# end

# Creating the documentation
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

        "General Introduction" => [

            "Introduction" => "Introduction.md",
            "Creation of the Generative Model" => "GenerativeModelCreation.md",
            "Initialising the Agent" => "InitialisingTheAgent.md",
            "Simulation" => [],
            "Model Fitting" => [],

        ],

        "Usage Examples" => [

            "T-Maze Simulation" => [],
            "T-Maze Model Fitting" => [],

        ],

        "Theory" => [

            "Active Inference Theory" => [
                "Perception" => [],
                "Action" => [],
                "Learning" => [],
            ],

            "POMDP Theory" => "GenerativeModelTheory.md",



        ],

        "Why Active Inference?" => [],

        "Index" => "index.md",
    ],
    doctest=true,
)
### NOTE Sam: change devbranch to master
deploydocs(;
    repo="github.com/ilabcode/ActiveInference.jl",
    devbranch="dev",
)
