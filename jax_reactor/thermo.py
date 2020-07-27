import jax
import jax.numpy as np
from collections import namedtuple, defaultdict
from typing import List, Tuple, Callable, Dict, Union


#local imports
from .dataclass import NASAPolynomials, EnthalpyData, EntropyData, CpData


def calculate_cp(T: Union[float, np.ndarray], X: np.ndarray, R: float,
                 nasa_polynomials: NASAPolynomials) -> CpData:
  """
    Calculate molar heat capacity (cp) from NASA Polynomials
    returns: namedtuple containing heat capacity data
    """

  cp_T = np.stack([1., T, T**2, T**3, T**4], axis=0)
  cpdivR = np.matmul(cp_T,nasa_polynomials.low_T_nasa_poly[:,:5].T) * np.where(T<=nasa_polynomials.temp_mid_points,1.,0) +  \
           np.matmul(cp_T,nasa_polynomials.high_T_nasa_poly[:,:5].T) * np.where(T>nasa_polynomials.temp_mid_points,1.,0)
  cp_mole_array = R * cpdivR * X
  cp_mole = cp_mole_array.sum(0)
  return CpData(cpdivR=cpdivR, cp_mole=cp_mole)


def calculate_enthalpy(T: Union[float, np.ndarray], X: np.ndarray, R: float,
                       nasa_polynomials: NASAPolynomials) -> EnthalpyData:
  """
    Calculate molar enthalpy and partial molar enthalpies from NASA Polynomials
    returns: namedtuple containing enthalpy data
    """
  H_T = np.stack([1., T / 2., (T**2) / 3., (T**3) / 4., (T**4) / 5., 1. / T],
                 axis=0)
  hdivRT = np.matmul(H_T, nasa_polynomials.low_T_nasa_poly[:,:6].T) * np.where(T<=nasa_polynomials.temp_mid_points,1.,0)  +  \
          np.matmul(H_T, nasa_polynomials.high_T_nasa_poly[:,:6].T)* np.where(T>nasa_polynomials.temp_mid_points,1.,0)
  h = partial_molar_enthalpies = hdivRT * R * T
  h_mole_array = h * X
  h_mole = h_mole_array.sum(axis=0)
  return EnthalpyData(hdivRT=hdivRT,
                      partial_molar_enthalpies=partial_molar_enthalpies,
                      h_mole=h_mole)


def calculate_entropy(T: Union[float,
                               np.ndarray], P: float, X: np.ndarray, R: float,
                      nasa_polynomials: NASAPolynomials) -> EntropyData:
  """
    Calculate molar entropy and partial molar entropies from NASA Polynomials
    returns: namedtuple containing entropy data
    """
  S_T = np.stack([np.log(T), T, (T**2) / 2., (T**3) / 3., (T**4) / 4., 1.],
                 axis=0)
  sdivR = np.matmul(S_T, nasa_polynomials.low_T_nasa_poly[:,[0,1,2,3,4,6]].T) *np.where(T<=nasa_polynomials.temp_mid_points,1.,0) + \
          np.matmul(S_T, nasa_polynomials.high_T_nasa_poly[:,[0,1,2,3,4,6]].T)*np.where(T>nasa_polynomials.temp_mid_points,1.,0)
  s0 = sdivR * R
  s = partial_molar_entropies = s0 + R * (-np.log(X + 1e-300) -
                                          np.log(P / 101325.))
  s_mole_array = s * X
  s_mole = s_mole_array.sum(axis=0)
  return EntropyData(sdivR=sdivR,
                     partial_molar_entropies=partial_molar_entropies,
                     s_mole=s_mole)
