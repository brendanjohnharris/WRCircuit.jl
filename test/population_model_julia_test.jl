#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using PythonCall
using Dewdrop

FNSPopulations = models.population_model.FNSPopulations

Ne = (10, 10)
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

    fullconn = [E2E E2I; I2E I2I]' # Python has columns first

    positions_E = FNSnet.E.positions |> convert2(Vector)  # num_E x 2
    positions_I = FNSnet.I.positions |> convert2(Vector) # num_I x 2
    fullpositions = Point2f.(collect.(vcat(positions_E, positions_I)))

    node_colors = vcat(fill("green", ne), fill("red", ni))

    intraconn = [E2E fill(false, size(E2I)); fill(false, size(I2E)) I2I]'
    Gintra = SimpleDiGraph(intraconn)
    edge_colors = fill((colorant"green", 0.1), size(intraconn))
    edge_colors[(ne + 1):end, (ne + 1):end] .= [(colorant"crimson", 0.1)]

    interconn = [fill(false, size(E2E)) E2I; I2E fill(false, size(I2I))]'
    Ginter = SimpleDiGraph(interconn)

    f = Figure()
    ax = Axis(f[1, 1]; aspect = DataAspect())
    pinter = graphplot!(ax, Ginter; layout = fullpositions, node_size = 0,
                        edge_color = (colorant"cornflowerblue", 0.025),
                        edge_plottype = :beziersegments, curve_distance = 1,
                        curve_distance_usage = true)
    pintra = graphplot!(ax, Gintra; layout = fullpositions, node_color = node_colors,
                        edge_color = edge_colors[intraconn],
                        edge_plottype = :beziersegments)
    hidespines!(ax)
    hidedecorations!(ax)
    f

else
    println("The product of FNSnet.E.size exceeds 100. Graph not plotted.")
end
