import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import collections
from functools import partial
from typing import List, Tuple, Callable, Dict, Union

#local imports
from .bdf import bdf_solve


def lax_scan_to_end_time(ode_fn: Callable, jacobian_fn: Callable,
                         initial_state: jnp.ndarray, ts: jnp.ndarray,
                         dt: float):
  """
  Integrate a system to endtime using JAX control flow 
  ode_fn: right hand side of the ODEs 
  jacobian_fn: callable for calculatiing jacobian
  
  """
  def scan_fun(carry, target_t):
    def cond_fn(_state):
      i, _, current_time, success = _state
      return (current_time < target_t) & success

    def step(_state):
      i, current_state, current_time, _ = _state
      results = bdf_solve(ode_fn, current_time, current_state,
                          np.array([current_time, current_time + dt]),
                          jacobian_fn)
      next_state = results.states[-1]
      next_time = results.times[-1]
      success = np.equal(results.diagnostics.status, 0)
      return [i + 1, next_state, next_time, success]

    _, *carry = jax.lax.while_loop(cond_fn, step, [0] + carry)
    y_target, _, _ = carry
    return carry, y_target

  init_carry = [initial_state, 0.0, True]

  _, ys = jax.lax.scan(scan_fun, init_carry, ts[1:])

  return np.concatenate((initial_state[None], ys))
