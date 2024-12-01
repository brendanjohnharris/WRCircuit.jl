#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using PythonCall
using Dewdrop

FNSPopulations = models.population_model.FNSPopulations

Ne = (5, 5)
Ni = prod(Ne) ÷ 4
FNSnet = FNSPopulations(Ne, Ni)

using CairoMakie
using GraphMakie
using GraphMakie.Graphs

if pyconvert(Bool, prod(FNSnet.E.size) ≤ 100)
    ne = FNSnet.E.num |> convert2(Int)
    ni = FNSnet.I.num |> convert2(Int)
    # Retrieve connection matrices
    E2E = FNSnet.E2E.conn.require("conn_mat") |> convert2(Matrix{Bool})
    I2E = FNSnet.I2E.conn.require("conn_mat") |> convert2(Matrix{Bool})
    E2I = FNSnet.E2I.conn.require("conn_mat") |> convert2(Matrix{Bool})
    I2I = FNSnet.I2I.conn.require("conn_mat") |> convert2(Matrix{Bool})

    fullconn = [E2E E2I; I2E I2I]

    positions_E = FNSnet.E.positions |> convert2(Vector)  # num_E x 2
    positions_I = FNSnet.I.positions |> convert2(Vector) # num_I x 2
    fullpositions = Point2f.(collect.(vcat(positions_E, positions_I)))

    node_colors = vcat(fill("green", ne), fill("red", ni))

    # edges = Graphs.edges(G)
    # edge_colors = map(Pair.(edges)) do (a, b)
    #     if a <= ne && b <= ne
    #         return colorant"cornflowerblue"
    #     elseif a > ne && b > ne
    #         return colorant"crimson"
    #     else
    #         return colorant"blue"
    #     end
    # end

    # edge_colors = fill((colorant"green", 0.1), size(fullconn))
    # edge_colors[(ne + 1):end, (ne + 1):end] .= [(colorant"crimson", 0.1)]
    # edge_colors[(ne + 1):end, 1:ne] .= [(colorant"cornflowerblue", 0.1)]
    # edge_colors[1:ne, (ne + 1):end] .= [(colorant"cornflowerblue", 0.1)]

    G = SimpleDiGraph(fullconn)

    f = Figure()
    ax = Axis(f[1, 1]; aspect = DataAspect())
    p = graphplot!(ax, G; layout = fullpositions, node_color = node_colors)
    hidespines!(ax)
    hidedecorations!(ax)
    f

else
    println("The product of FNSnet.E.size exceeds 100. Graph not plotted.")
end
