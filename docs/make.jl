using Dewdrop
using Documenter

DocMeta.setdocmeta!(Dewdrop, :DocTestSetup, :(using Dewdrop); recursive = true)

makedocs(;
         modules = [Dewdrop],
         authors = "brendanjohnharris <bhar9988@uni.sydney.edu.au> and contributors",
         sitename = "Dewdrop.jl",
         format = Documenter.HTML(;
                                  canonical = "https://brendanjohnharris.github.io/Dewdrop.jl",
                                  edit_link = "main",
                                  assets = String[],),
         pages = ["Home" => "index.md"],)

deploydocs(;
           repo = "github.com/brendanjohnharris/Dewdrop.jl",
           devbranch = "main",)
