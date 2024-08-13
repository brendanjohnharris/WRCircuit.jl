using DistributedVisualCortex
using Documenter

DocMeta.setdocmeta!(DistributedVisualCortex, :DocTestSetup, :(using DistributedVisualCortex); recursive=true)

makedocs(;
    modules=[DistributedVisualCortex],
    authors="brendanjohnharris <bhar9988@uni.sydney.edu.au> and contributors",
    sitename="DistributedVisualCortex.jl",
    format=Documenter.HTML(;
        canonical="https://brendanjohnharris.github.io/DistributedVisualCortex.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/brendanjohnharris/DistributedVisualCortex.jl",
    devbranch="main",
)
