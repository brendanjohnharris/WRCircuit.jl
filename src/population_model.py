from neurons import FNSNeuron, LIFNeuron
from synapses import Synapse, maybe_initializer
from positions import ClusteredPositions, Positions
import numpy as np

import brainpy as bp
from abc import ABC, abstractmethod

# * Just two populations, coupled with heavy tailed synaptic weights or degree distributions


class FNSPopulations(bp.Network):
    def __init__(self, num_exc, num_inh, method="exp_auto"):
        super().__init__()

        # geometry
        exc_positions = ClusteredPositions((-1.5, 0), 1)
        inh_positions = ClusteredPositions((1.5, 0), 1)

        # neurons
        self.E = LIFNeuron(
            size=num_exc,
            embedding=exc_positions,
            V_rest=-60.0,
            V_th=-50.0,
            V_reset=-60.0,
            tau=20.0,
            tau_ref=5.0,
            V_initializer=bp.init.Normal(-55.0, 2.0),
        )

        # Create a population of inhibitory neurons
        self.I = LIFNeuron(
            size=num_inh,
            embedding=inh_positions,
            V_rest=-60.0,
            V_th=-50.0,
            V_reset=-60.0,
            tau=20.0,
            tau_ref=5.0,
            V_initializer=bp.init.Normal(-55.0, 2.0),
        )

        # Connectivity topology
        prob = 0.1
        conn_E2E = bp.connect.FixedProb(prob=prob)
        conn_E2I = bp.connect.FixedProb(prob=prob)
        conn_I2E = bp.connect.FixedProb(prob=prob)
        conn_I2I = bp.connect.FixedProb(prob=prob)

        # Synapses
        delay_step = int(2.0 // bp.share["dt"])
        JE = 1 / bp.math.sqrt(prob * np.prod(num_exc))
        JI = -1 / bp.math.sqrt(prob * np.prod(num_inh))
        self.E2E = bp.synapses.Delta(
            self.E, self.E, conn_E2E, delay_step=delay_step, g_max=JE
        )
        self.E2I = bp.synapses.Delta(
            self.E, self.I, conn_E2I, delay_step=delay_step, g_max=JE
        )
        self.I2E = bp.synapses.Delta(
            self.I, self.E, conn_I2E, delay_step=delay_step, g_max=JI
        )
        self.I2I = bp.synapses.Delta(
            self.I, self.I, conn_I2I, delay_step=delay_step, g_max=JI
        )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

    def to_dict(self):
        return {
            self.__class__.__name__: {
                "populations": {
                    "E": self.E.to_dict(),
                    "I": self.I.to_dict(),
                },
                "synapses": {
                    "E2E": self.E2E.to_dict(),
                    "E2I": self.E2I.to_dict(),
                    "I2E": self.I2E.to_dict(),
                    "I2I": self.I2I.to_dict(),
                },
            }
        }
