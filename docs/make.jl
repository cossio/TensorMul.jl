using Documenter, Literate
import TensorMul

function clear_md_files(dir::String)
    for file in readdir(dir; join=true)
        if endswith(file, ".md")
            rm(file)
        end
    end
end

const literate_dir = joinpath(@__DIR__, "src/literate")
clear_md_files(literate_dir)

for file in readdir(literate_dir; join=true)
    if endswith(file, ".jl")
        Literate.markdown(file, literate_dir)
    end
end

makedocs(
    modules = [TensorMul],
    sitename = "TensorMul.jl",
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "tensordot" => "literate/tensordot.md"
        ],
        "Reference" => "reference.md"
    ],
    strict = true
)

clear_md_files(literate_dir)

deploydocs(
    repo = "github.com/cossio/TensorMul.jl.git",
    devbranch = "master"
)
