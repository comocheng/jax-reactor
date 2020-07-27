import jax
import jax.numpy as np
from functools import partial
from collections import namedtuple, defaultdict
from typing import List, Tuple, Callable, Dict


#local imports
from .dataclass import NASAPolynomials, KineticsCoeffs, KineticsData, GasInfo
from .jax_utils import register_pytree_namedtuple


def get_equilibirum_constants(T: float, P: float, R: float, sdivR: np.ndarray,
                              hdivRT: np.ndarray,
                              gas_info: GasInfo) -> np.ndarray:
  """
    Calculate equilibrium constants
    returns: np.ndarray of equilibrium constants
    """
  vk = -gas_info.reactant_stioch_coeffs + gas_info.product_stioch_coeffs
  delta_entropies = np.matmul(vk.T, sdivR)
  delta_enthalpies = np.matmul(vk.T, hdivRT)
  Ka = np.exp(np.subtract(delta_entropies, delta_enthalpies))
  Kc = np.multiply(Ka, np.power(P / (R * T), vk.sum(0)))
  return Kc


def calculate_rate_constants(T: float, R: float,
                             params: np.ndarray) -> np.ndarray:
  A, b, Ea = params
  k = A * (T**b) * (np.exp(-Ea * 4184.0 / R / T))
  return k


vmap_arrhenius = jax.jit(jax.vmap(calculate_rate_constants,
                                  (None, None, 0)))  #vmap and jit once


def troe_falloff_correction(T: float, lPr: np.ndarray, troe_coeffs: np.ndarray,
                            troe_indices: np.ndarray) -> np.ndarray:
  """
    modify rate constants use TROE falloff parameters
    returns: np.ndarray of F(T,P) 
    """
  troe_coeffs = troe_coeffs[troe_indices]
  F_cent = np.multiply(np.subtract(1,troe_coeffs[:,0]),np.exp(np.divide(-T,troe_coeffs[:,3]))) + \
           np.multiply(troe_coeffs[:,0],np.exp(np.divide(-T,troe_coeffs[:,1]))) + \
           np.exp(np.divide(-troe_coeffs[:,2],T))
  lF_cent = np.log10(F_cent)
  C = np.subtract(-0.4, np.multiply(0.67, lF_cent))
  N = np.subtract(0.75, np.multiply(1.27, lF_cent))
  f1_numerator = lPr + C
  f1_denominator_1 = N
  f1_denominator_2 = np.multiply(0.14, f1_numerator)
  f1 = np.divide(f1_numerator, np.subtract(f1_denominator_1, f1_denominator_2))
  F = np.power(10., np.divide(lF_cent, (1. + np.square(f1))))
  #F = 10**(lF_cent / (1. + f1**2.))
  return F


def get_forward_rate_constants(T: float, R: float, C: np.ndarray,
                               kinetics_coeffs: KineticsCoeffs,
                               kinetics_data: KineticsData) -> np.ndarray:
  """
    Calculate forward rate constants with three body, falloff and troe falloff modifications 
    returns: np.ndarray of forward rate constants
    """
  # vectorize and jit rate constants calculation
  initial_k = vmap_arrhenius(T, R, kinetics_coeffs.arrhenius_coeffs)
  C_M = np.matmul(C, kinetics_coeffs.efficiency_coeffs)  # calculate C_M
  three_body_k = np.multiply(
      initial_k[kinetics_data.three_body_indices],
      C_M[kinetics_data.three_body_indices])  # get kf with three body update
  k = jax.ops.index_update(initial_k,
                           jax.ops.index[kinetics_data.three_body_indices],
                           three_body_k)  #three body update
  total_falloff_indices = np.concatenate(
      [kinetics_data.falloff_indices,
       kinetics_data.troe_falloff_indices]).sort()
  k0 = vmap_arrhenius(T, R, kinetics_coeffs.arrhenius0_coeffs)  # calculate k0
  Pr = np.divide(np.multiply(k0, C_M), k)  # calculate Pr as kinf is k
  log10Pr = np.log10(Pr)
  falloff_k = np.multiply(
      k[total_falloff_indices],
      (Pr[total_falloff_indices] /
       (1. + Pr[total_falloff_indices])))  # update all type of falloff
  kf = jax.ops.index_update(k, jax.ops.index[total_falloff_indices], falloff_k)
  F = troe_falloff_correction(
      T, log10Pr[kinetics_data.troe_falloff_indices],
      kinetics_coeffs.troe_coeffs,
      kinetics_data.troe_falloff_indices)  # calculate F(T, P)
  troe_k = np.multiply(kf[kinetics_data.troe_falloff_indices], F)
  final_k = jax.ops.index_update(
      kf, jax.ops.index[kinetics_data.troe_falloff_indices],
      troe_k)  # final update
  return final_k


def get_reverse_rate_constants(kf: np.ndarray, Kc: np.ndarray,
                               is_reversible: np.ndarray):
  """
    calculate reverse rate constants using Kc and kf
    """
  return np.divide(kf, Kc) * is_reversible
