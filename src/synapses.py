import brainpy as bp
import brainpy.math as bm
from brainpy.connect import TwoEndConnector
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit, vmap
import uuid

import brainpy as bp
import brainpy.math as bm

from brainpy.dyn.neurons import GradNeuDyn
from brainpy.dyn import NeuDyn
from brainpy.initialize import (
    ZeroInit,
    OneInit,
    Uniform,
    variable_,
    noise as init_noise,
)
from brainpy import share
from brainpy.types import Shape, ArrayType
from brainpy.check import is_initializer
from brainpy import odeint, sdeint, JointEq
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial


def maybe_initializer(x, exclude=["rng"]):
    if isinstance(x, bp.init.Initializer):
        name = x.__class__.__name__
        d = {keys: value for keys, value in x.__dict__.items() if keys not in exclude}
        if not d:
            return name
        else:
            return {name: d}
    else:
        return x


class Synapse(bp.Projection):
    def __init__(
        self, pre, post, delay, conn, J=1.0, tau_d=5, tau_r=1, V_rev=0.0, alpha=1.0
    ):
        self.delay = delay
        self.J = J
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.V_rev = V_rev
        self.alpha = alpha

        super().__init__()
        self.proj = bp.dyn.FullProjAlignPreSDMg(
            pre=pre,
            delay=self.delay,
            syn=bp.dyn.AMPA.desc(
                pre.num, alpha=alpha, beta=1 / tau_d, T=1 / tau_r, T_dur=tau_r
            ),
            comm=bp.dnn.CSRLinear(conn(pre_size=pre.size, post_size=post.size), J),
            out=bp.dyn.COBA(E=V_rev),
            post=post,
        )

    def to_dict(self):
        return {
            "delay": maybe_initializer(self.delay),
            "J": maybe_initializer(self.J),
            "tau_d": maybe_initializer(self.tau_d),
            "tau_r": maybe_initializer(self.tau_r),
            "V_rev": maybe_initializer(self.V_rev),
            "alpha": maybe_initializer(self.alpha),
        }
