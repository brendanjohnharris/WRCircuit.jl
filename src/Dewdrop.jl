module Dewdrop
using PythonCall
import PythonCall: pycopy!

begin # * Python imports
    const sys = PythonCall.pynew()
    const brainpy = PythonCall.pynew()
    const neurons = PythonCall.pynew()
    const positions = PythonCall.pynew()
    const synapses = PythonCall.pynew()
    const models = PythonCall.pynew()
    const jax = PythonCall.pynew()
    const jax_lib = PythonCall.pynew()
    const xla_bridge = PythonCall.pynew()

    export brainpy, neurons, positions, synapses, models
end

function __init__()
    pycopy!(sys, pyimport("sys"))
    sys.path.append(pwd())

    pycopy!(jax, pyimport("jax"))
    pycopy!(jax_lib, pyimport("jax.lib"))
    pycopy!(xla_bridge, pyimport("jax.lib.xla_bridge"))
    pycopy!(brainpy, pyimport("brainpy"))
    pycopy!(neurons, pyimport("src.neurons"))
    pycopy!(positions, pyimport("src.positions"))
    pycopy!(synapses, pyimport("src.synapses"))
    pycopy!(models, pyimport("src.models"))

    # begin # * CUDA checks
    # _jax_backend = xla_bridge.get_backend().platform
    #     _jax_backend == "gpu" || (@warn "JAX is not using the GPU")
    # end
end
function _cudnn_version()
    jax._src.lib.cuda_versions.cudnn_get_version()
end
function _cudnn_build_version()
    jax._src.lib.cuda_versions.cudnn_build_version()
end

end # module
