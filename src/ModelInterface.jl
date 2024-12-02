export bprun

function bprun(net::Py, time; monitors = ("E.spike", "I.spike", "E.V", "I.V"), jit = true,
               kwargs...)
    runner = brainpy.DSRunner(net; monitors, jit, kwargs...)
    runner.run(time)
    return runner
end
