using DrWatson
DrWatson.@quickactivate
using WorkingRegime

# Set up a fixed seed
seed = 42
key = WorkingRegime.PRNGKey(seed)

# Simulation parameters (smaller network for speed)
params = (rho = 10000,
          dx = 0.5,
          key = key)

duration = 50.0  # ms, short for quick testing
transient = 0.0  # No transient for this test

println("Creating model 1...")
model1 = WorkingRegime.models.Spatial(; params...)

println("Creating model 2...")
model2 = WorkingRegime.models.Spatial(; params...)

println("\nRunning both models for $(duration) ms...")

# Run model 1
runner1 = WorkingRegime.bprun(model1, duration; monitors = ("E.spike", "I.spike"))

# Run model 2
runner2 = WorkingRegime.bprun(model2, duration; monitors = ("E.spike", "I.spike"))

# Extract spike data
e_spikes_1 = runner1.mon["E.spike"] |> convert2(Array)
i_spikes_1 = runner1.mon["I.spike"] |> convert2(Array)
e_spikes_2 = runner2.mon["E.spike"] |> convert2(Array)
i_spikes_2 = runner2.mon["I.spike"] |> convert2(Array)

# Compare outputs
e_match = e_spikes_1 == e_spikes_2
i_match = i_spikes_1 == i_spikes_2

println("\n" * "="^50)
println("REPRODUCIBILITY TEST RESULTS")
println("="^50)
println("Excitatory spikes match: $(e_match)")
println("Inhibitory spikes match: $(i_match)")
println("\nE spike shape: $(size(e_spikes_1))")
println("I spike shape: $(size(i_spikes_1))")

if e_match && i_match
    println("\n✓ SUCCESS: Simulations are deterministic!")
else
    println("\n✗ FAILURE: Simulations differ!")
    println("  E spikes differ at $(sum(e_spikes_1 .!= e_spikes_2)) positions")
    println("  I spikes differ at $(sum(i_spikes_1 .!= i_spikes_2)) positions")
end
