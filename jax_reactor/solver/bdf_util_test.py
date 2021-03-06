import jax.numpy as np
import numpy as onp
from jax.config import config

config.update("jax_enable_x64", True)

# local imports
from jax_reactor.solver import bdf_util

# Adapted from tensorflow_probabilty at https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/bdf_util_test.py
# Accessed 2020-06-24


def test_first_step_size_is_large_when_ode_fn_is_constant():
    initial_state_vec = np.array([1.0], dtype=np.float64)
    atol = np.array(1e-12, dtype=np.float64)
    first_order_bdf_coefficient = -0.1850
    first_order_error_coefficient = first_order_bdf_coefficient + 0.5
    initial_time = np.array(0.0, dtype=np.float64)
    ode_fn_vec = lambda time, state: 1.0
    rtol = np.array(1e-8, dtype=np.float64)
    safety_factor = np.array(0.9, dtype=np.float64)
    max_step_size = 1.0
    step_size = bdf_util.first_step_size(
        atol,
        first_order_error_coefficient,
        initial_state_vec,
        initial_time,
        ode_fn_vec,
        rtol,
        safety_factor,
        max_step_size=max_step_size,
    )
    # Step size should be maximal.
    onp.testing.assert_allclose(
        onp.asarray(max_step_size, dtype=onp.float64),
        onp.asarray(step_size, dtype=onp.float64),
        err_msg="step size is not equal",
    )


def test_interpolation_matrix_unit_step_size_ratio():
    order = np.array(bdf_util.MAX_ORDER, dtype=np.int32)
    step_size_ratio = np.array(1.0, dtype=np.float64)
    interpolation_matrix = bdf_util.interpolation_matrix(order, step_size_ratio)
    onp.testing.assert_allclose(
        interpolation_matrix,
        np.array(
            [
                [-1.0, -0.0, -0.0, -0.0, -0.0],
                [-2.0, 1.0, 0.0, 0.0, 0.0],
                [-3.0, 3.0, -1.0, -0.0, -0.0],
                [-4.0, 6.0, -4.0, 1.0, 0.0],
                [-5.0, 10.0, -10.0, 5.0, -1.0],
            ],
            dtype=np.float64,
        ),
        err_msg="Interpolation matrices are not equal",
    )


def test_interpolate_backward_differences_zeroth_order_is_unchanged():
    backward_differences = np.array(
        onp.random.normal(size=((bdf_util.MAX_ORDER + 3, 3))), dtype=np.float64
    )
    step_size_ratio = np.array(0.5, dtype=np.float64)
    interpolated_backward_differences = bdf_util.interpolate_backward_differences(
        backward_differences, bdf_util.MAX_ORDER, step_size_ratio
    )
    onp.testing.assert_allclose(
        backward_differences[0],
        interpolated_backward_differences[0],
        err_msg="Interpolated backward differences are not equal",
    )


def test_newton_order_one():
    jacobian_mat = np.array([[-1.0]], dtype=np.float64)
    bdf_coefficient = np.array(-0.1850, dtype=np.float64)
    first_order_newton_coefficient = 1.0 / (1.0 - bdf_coefficient)
    step_size = np.array(0.01, dtype=np.float64)
    unitary, upper = bdf_util.newton_qr(
        jacobian_mat, first_order_newton_coefficient, step_size
    )

    backward_differences = np.array(
        [[1.0], [-1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64
    )
    ode_fn_vec = lambda time, state: -state
    order = np.array(1, dtype=np.int32)
    time = np.array(0.0, dtype=np.float64)
    tol = np.array(1e-6, dtype=np.float64)

    # The equation we are trying to solve with Newton's method is linear.
    # Therefore, we should observe exact convergence after one iteration. An
    # additional iteration is required to obtain an accurate error estimate,
    # making the total number of iterations 2.
    max_num_newton_iters = 2

    converged, next_backward_difference, next_state, _ = bdf_util.newton(
        backward_differences,
        max_num_newton_iters,
        first_order_newton_coefficient,
        ode_fn_vec,
        order,
        step_size,
        time,
        tol,
        unitary,
        upper,
    )
    onp.testing.assert_equal(onp.asarray(converged), True)

    state = backward_differences[0, :]
    exact_next_state = ((1.0 - bdf_coefficient) * state + bdf_coefficient) / (
        1.0 + step_size - bdf_coefficient
    )

    onp.testing.assert_allclose(next_backward_difference, exact_next_state)
    onp.testing.assert_allclose(next_state, exact_next_state)


if __name__ == "__main__":
    test_first_step_size_is_large_when_ode_fn_is_constant()
    test_interpolation_matrix_unit_step_size_ratio()
    test_interpolate_backward_differences_zeroth_order_is_unchanged()
    test_newton_order_one()
