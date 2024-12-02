#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using PythonCall
using Dewdrop
using TimeseriesTools

FNSPopulations = models.population_model.FNSPopulations

begin # * Model
    N = 100
    epsilon = 0.05
    D = 2.0
    nu = 6
    g = 4
    J = 5 # ! Why is this so HIGH!!!

    FNSnet = FNSPopulations(N; epsilon, D, nu, g, J)

    ne = FNSnet.E.num |> convert2(Int)
    ni = FNSnet.I.num |> convert2(Int)
    ν_thr = FNSnet.E.V_th / (epsilon * ne * J * FNSnet.E.tau) |> convert2(Float64)
end

using CairoMakie
using GraphMakie
using GraphMakie.Graphs

if pyconvert(Bool, prod(FNSnet.E.size) ≤ 100)
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
                        edge_color = (colorant"cornflowerblue", 0.1),
                        edge_plottype = :beziersegments, curve_distance = 1,
                        curve_distance_usage = true)
    pintra = graphplot!(ax, Gintra; layout = fullpositions, node_color = node_colors,
                        edge_color = edge_colors[intraconn],
                        edge_plottype = :beziersegments)
    hidespines!(ax)
    hidedecorations!(ax)
    f |> display

else
    println("The product of FNSnet.E.size exceeds 100. Graph not plotted.")
end

begin # * Simulate
    T = 1000.0
    monitors = monitors = ("E.spike", "I.spike", "E.V", "I.V")
    runner = bprun(FNSnet, T)

    t = runner.mon["ts"].view() |> convert2(Vector)
    Xe = runner.mon["E.spike"].view() |> convert2(Matrix)
    Xi = runner.mon["I.spike"].view() |> convert2(Matrix)
    Ve = runner.mon["E.V"].view() |> convert2(Matrix)
    Vi = runner.mon["I.V"].view() |> convert2(Matrix)

    Evars = Symbol.(["E"] .* string.(1:ne))
    Ivars = Symbol.(["I"] .* string.(1:ni))
    Xe = Timeseries(t, Evars, Xe)
    Xi = Timeseries(t, Ivars, Xi)
    Ve = Timeseries(t, Evars, Ve)
    Vi = Timeseries(t, Ivars, Vi)
end

begin
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time (ms)", ylabel = "Neuron index",
              limits = ((100, 200), nothing))
    spikes = findall([Xe Xi])
    spikes = collect(Iterators.product(t, 1:N))[spikes]
    scatter!(ax, spikes, markersize = 4)
    hlines!(ax, [ne + 0.5], linestyle = :dash, color = :black)
    f |> display
end

begin # * VOltage traces
    lines(Ve[:, 1])
    display(current_figure())
end

# begin # * Distribution of differences
#     dV =
# end

PythonCall.GC.gc()
