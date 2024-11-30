import brainpy as bp
import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.synapses import TwoEndConn, SynOut, SynSTP
from brainpy.initialize import Initializer
from brainpy.synouts import CUBA
from typing import Union, Dict, Callable, Optional
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
            "pre": self.pre.name,
            "post": self.post.name,
            "delay": maybe_initializer(self.delay),
            "J": maybe_initializer(self.J),
            "tau_d": maybe_initializer(self.tau_d),
            "tau_r": maybe_initializer(self.tau_r),
            "V_rev": maybe_initializer(self.V_rev),
            "alpha": maybe_initializer(self.alpha),
        }


class DeltaSynapse(TwoEndConn):
    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        output: SynOut = CUBA(target_var="V"),
        stp: Optional[SynSTP] = None,
        comp_method: str = "sparse",
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.0,
        delay_step: Union[float, ArrayType, Initializer, Callable] = None,
        post_ref_key: str = None,
        name: str = None,
        mode: bm.Mode = None,
        stop_spike_gradient: bool = False,
    ):
        super().__init__(
            name=name, pre=pre, post=post, conn=conn, output=output, stp=stp, mode=mode
        )

        # parameters
        self.stop_spike_gradient = stop_spike_gradient
        self.post_ref_key = post_ref_key
        if post_ref_key:
            self.check_post_attrs(post_ref_key)
        self.comp_method = comp_method

        # connections and weights
        self.g_max, self.conn_mask = self._init_weights(
            g_max, comp_method=comp_method, sparse_data="csr"
        )

        # register delay
        self.delay_step = delay_step
        self.pre.register_local_delay("spike", self.name, delay_step=self.delay_step)

    def update(self, pre_spike=None):
        # pre-synaptic spikes
        if pre_spike is None:
            pre_spike = self.pre.get_local_delay("spike", self.name)
        pre_spike = bm.as_jax(pre_spike)
        if self.stop_spike_gradient:
            pre_spike = jax.lax.stop_gradient(pre_spike)

        # update sub-components
        if self.stp is not None:
            self.stp.update(pre_spike)

        # synaptic values onto the post
        if isinstance(self.conn, All2All):
            syn_value = bm.asarray(pre_spike, dtype=bm.float_)
            if self.stp is not None:
                syn_value = self.stp(syn_value)
            post_vs = self._syn2post_with_all2all(syn_value, self.g_max)
        elif isinstance(self.conn, One2One):
            syn_value = bm.asarray(pre_spike, dtype=bm.float_)
            if self.stp is not None:
                syn_value = self.stp(syn_value)
            post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
        else:
            if self.comp_method == "sparse":
                if self.stp is not None:
                    syn_value = self.stp(pre_spike)
                    f = lambda s: bm.sparse.csrmv(
                        self.g_max,
                        self.conn_mask[0],
                        self.conn_mask[1],
                        s,
                        shape=(self.pre.num, self.post.num),
                        transpose=True,
                    )
                else:
                    syn_value = pre_spike
                    f = lambda s: bm.event.csrmv(
                        self.g_max,
                        self.conn_mask[0],
                        self.conn_mask[1],
                        s,
                        shape=(self.pre.num, self.post.num),
                        transpose=True,
                    )
                if isinstance(self.mode, bm.BatchingMode):
                    f = jax.vmap(f)
                post_vs = f(syn_value)
            else:
                syn_value = bm.asarray(pre_spike, dtype=bm.float_)
                if self.stp is not None:
                    syn_value = self.stp(syn_value)
                post_vs = self._syn2post_with_dense(
                    syn_value, self.g_max, self.conn_mask
                )
        if self.post_ref_key:
            post_vs = post_vs * (1.0 - getattr(self.post, self.post_ref_key))

        # update outputs
        return self.output(post_vs)

    def to_dict(self):
        return {
            "pre": self.pre.name,
            "post": self.post.name,
            "delay_step": self.delay_step,
            "g_max": maybe_initializer(self.g_max),
        }
