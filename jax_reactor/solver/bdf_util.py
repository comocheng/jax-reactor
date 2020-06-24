import jax
import jaxlib
import jax.numpy as np
import numpy as onp
from functools import partial
from collections import namedtuple
from typing import List, Tuple, Callable, Dict, Union
from jax.config import config
config.update("jax_enable_x64", True)

#Adapted from tensorflow_probabilty at https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/bdf_util.py
#Accessed 2020-06-24

MAX_ORDER = 5
ORDERS = np.arange(0, MAX_ORDER + 1)
RECIPROCAL_SUMS = np.concatenate([np.array([np.nan]), np.cumsum(1. / ORDERS[1:])])

def error_ratio(backward_difference:np.ndarray, 
                error_coefficient:np.ndarray,
                tol:Union[np.ndarray, float])->np.ndarray:
    """Computes the ratio of the error in the computed state to the tolerance."""
    error_ratio_ = np.linalg.norm(error_coefficient * backward_difference / tol)
    return error_ratio_

def first_step_size(
    atol:Union[np.ndarray, float],
    first_order_error_coefficient:Union[np.ndarray, float],
    initial_state_vec:np.ndarray,
    initial_time:Union[np.ndarray, float],
    ode_fn_vec:Callable,
    rtol:Union[np.ndarray, float],
    safety_factor:Union[np.ndarray, float],
    epsilon:float =1e-12,
    max_step_size:float =1.,
    min_step_size:float =1e-12)->np.ndarray:
    """Selects the first step size to use."""
    next_time = initial_time + epsilon
    first_derivative = ode_fn_vec(initial_time, initial_state_vec)
    next_state_vec = initial_state_vec + first_derivative * epsilon
    second_derivative = (ode_fn_vec(next_time, next_state_vec) -
                           first_derivative) / epsilon
    tol = atol + (rtol * np.abs(initial_state_vec))
    # Local truncation error of an order one step is
    # `err(step_size) = first_order_error_coefficient * second_derivative *
    #                 * step_size**2`.
    # Choose the largest `step_size` such that `norm(err(step_size) / tol) <= 1`.
    norm = np.linalg.norm(first_order_error_coefficient * second_derivative / tol)
    step_size = jax.lax.rsqrt(norm)
    return np.clip(safety_factor * step_size, min_step_size,
                              max_step_size)

def interpolation_matrix(order:Union[np.ndarray, int], 
                        step_size_ratio:Union[np.ndarray, float, int])->np.ndarray:
    """Creates the matrix used to interpolate backward differences."""
    orders = np.arange(1, MAX_ORDER + 1)
    i = orders[:, np.newaxis]
    j = orders[np.newaxis, :]
    # Matrix whose (i, j)-th entry (`1 <= i, j <= order`) is
    # `1/j! (0 - i * step_size_ratio) * ... * ((j-1) - i * step_size_ratio)`.
    full_interpolation_matrix = np.cumprod(
          ((j - 1) - i * step_size_ratio) / j, axis=1)
    zeros_matrix = np.zeros_like(full_interpolation_matrix)
    interpolation_matrix_ = np.where(
          np.arange(1, MAX_ORDER + 1) <= order,
          np.transpose(
              np.where(
                  np.arange(1, MAX_ORDER + 1) <= order,
                  np.transpose(full_interpolation_matrix), zeros_matrix)),
          zeros_matrix)
    return interpolation_matrix_

def interpolate_backward_differences(backward_differences:np.ndarray, 
                                    order:Union[np.ndarray, int],
                                     step_size_ratio:Union[np.ndarray, float, int])->np.ndarray:
    """Updates backward differences when a change in the step size occurs."""
    interpolation_matrix_ = interpolation_matrix(order,
                                                step_size_ratio)
    interpolation_matrix_unit_step_size_ratio = interpolation_matrix(order, 1.)
    interpolated_backward_differences_orders_one_to_five = np.matmul(
          interpolation_matrix_unit_step_size_ratio,
          np.matmul(interpolation_matrix_, backward_differences[1:MAX_ORDER + 1]))
    interpolated_backward_differences = np.concatenate([
          backward_differences[0].reshape(1,np.shape(backward_differences)[1]),
          interpolated_backward_differences_orders_one_to_five,
          np.zeros(
              np.stack(np.array([2, np.shape(backward_differences)[1]],dtype=np.int64)))
      ], axis=0)
    return interpolated_backward_differences

_NewtonIterand = namedtuple('NewtonIterand', [
    'converged',
    'finished',
    'next_backward_difference',
    'next_state_vec',
    'num_iters',
    'prev_delta_norm',
])

def newton_qr(jacobian_mat:np.ndarray, 
              newton_coefficient:Union[np.ndarray, float, int], 
              step_size:Union[np.ndarray, float, int])->Tuple[np.ndarray, np.ndarray]:
    """QR factorizes the matrix used in each iteration of Newton's method."""
    identity = np.eye(np.shape(jacobian_mat)[0], dtype=jacobian_mat.dtype)
    newton_matrix = (
        identity - step_size * newton_coefficient * jacobian_mat)
    q, r = np.linalg.qr(newton_matrix)
    return q, r

def newton(backward_differences:np.ndarray, 
           max_num_iters:Union[np.ndarray, float, int], 
           newton_coefficient:Union[np.ndarray, float, int], 
           ode_fn_vec:Callable,
           order:Union[np.ndarray, float, int], 
           step_size:Union[np.ndarray, float, int], 
           time:Union[np.ndarray, float, int], 
           tol:Union[np.ndarray, float], 
           unitary:np.ndarray, 
           upper:np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs Newton's method to solve the BDF equation."""
    initial_guess = np.sum(
        np.where(
            np.arange(MAX_ORDER + 1).reshape(-1,1) <= order,
            backward_differences[:MAX_ORDER + 1],
            np.zeros_like(backward_differences)[:MAX_ORDER + 1]),
        axis=0)

    rhs_constant_term = newton_coefficient * np.sum(
        np.where(
            np.arange(1, MAX_ORDER + 1).reshape(-1,1) <= order, RECIPROCAL_SUMS[1:, np.newaxis] *
            backward_differences[1:MAX_ORDER + 1],
            np.zeros_like(backward_differences)[1:MAX_ORDER + 1]),
        axis=0)

    next_time = time + step_size
    def newton_body(iterand):
        """Performs one iteration of Newton's method."""
        next_backward_difference = iterand.next_backward_difference
        next_state_vec = iterand.next_state_vec

        rhs = newton_coefficient * step_size * ode_fn_vec(
            next_time,
            next_state_vec) - rhs_constant_term - next_backward_difference
        delta = np.squeeze(
            jax.scipy.linalg.solve_triangular(
                upper,
                np.matmul(np.transpose(unitary), rhs[:, np.newaxis]),
                lower=False))
        num_iters = iterand.num_iters + 1

        next_backward_difference += delta
        next_state_vec += delta

        delta_norm = np.linalg.norm(delta)
        lipschitz_const = delta_norm / iterand.prev_delta_norm

        # Stop if method has converged.
        approx_dist_to_sol = lipschitz_const / (1. - lipschitz_const) * delta_norm
        close_to_sol = approx_dist_to_sol < tol
        delta_norm_is_zero = np.equal(delta_norm, np.array(0., dtype=np.float64))
        converged = close_to_sol | delta_norm_is_zero
        finished = converged

        # Stop if any of the following conditions are met:
        # (A) We have hit the maximum number of iterations.
        # (B) The method is converging too slowly.
        # (C) The method is not expected to converge.
        too_slow = lipschitz_const > 1.
        finished = finished | too_slow

        too_many_iters = np.equal(num_iters, max_num_iters)
        num_iters_left = max_num_iters - num_iters
        wont_converge = (
            approx_dist_to_sol * lipschitz_const**num_iters_left > tol)
        finished = finished | too_many_iters | wont_converge

        return _NewtonIterand(
                    converged=converged,
                    finished=finished,
                    next_backward_difference=next_backward_difference,
                    next_state_vec=next_state_vec,
                    num_iters=num_iters,
                    prev_delta_norm=delta_norm)
            
    iterand= _NewtonIterand(
      converged=False,
      finished=False,
      next_backward_difference=np.zeros_like(initial_guess),
      next_state_vec=np.array(initial_guess.copy()),
      num_iters=0,
      prev_delta_norm=(np.array(-0.)))
    iterand = jax.lax.while_loop(lambda iterand: np.logical_not(iterand.finished),
                            newton_body, iterand)
    ## Krishna: need to double check this
    return (iterand.converged, iterand.next_backward_difference,
          iterand.next_state_vec, iterand.num_iters)