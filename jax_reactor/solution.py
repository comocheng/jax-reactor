import jax
import jax.numpy as np
import cantera as ct
from collections import namedtuple, defaultdict
from typing import List, Tuple, Callable, Dict, Union
#enable float64 by default 
from jax.config import config
config.update("jax_enable_x64", True)

#local imports 
from .dataclass import NASAPolynomials, KineticsCoeffs, KineticsData, GasInfo, ProductionRates
from .thermo import calculate_cp, calculate_enthalpy, calculate_entropy
from .kinetics import get_forward_rate_constants, get_reverse_rate_constants, get_equilibirum_constants
from .jax_utils import register_pytree_namedtuple, delete_small_numbers

def get_initial_mole_fractions_and_state(path:str,
                                        T:Union[float, np.ndarray],
                                        P:Union[float, np.ndarray],
                                        X:Union[str, dict])->Tuple[np.ndarray, GasInfo]:
    """
    Helper function to get Cantera's Solution object
    """
    gas = ct.Solution(path)
    gas.TPX = T, P, X
    gas_info = GasInfo(reactant_stioch_coeffs=gas.reactant_stoich_coeffs(), 
                      product_stioch_coeffs=gas.product_stoich_coeffs(),
                      molecular_weights=gas.molecular_weights)
    return np.array(gas.Y,dtype=np.float64), gas_info

def Y2X(Y:np.ndarray,
        mol_wts:np.ndarray,
        mean_mol_wt:np.ndarray):
    """
    helper function to calculate mass fractions from mole fractions
    """
    return np.divide(np.multiply(Y,mean_mol_wt),mol_wts)

def Y2C(Y:np.ndarray,
        mol_wts:np.ndarray,
        density_mass:np.ndarray):
    """
    helper function to calculate concentrations from mole fractions
    """
    return np.divide(np.multiply(Y,density_mass),mol_wts)

def get_mean_molecular_weight(Y:np.ndarray,
                              mol_wts:np.ndarray)->np.ndarray:
    """
    helper function to calculate concentrations from mole fractions
    """
    return 1./(np.matmul(Y,1./mol_wts))

def get_production_rates(kf:np.ndarray, 
                         kr:np.ndarray, 
                         C:np.ndarray, 
                         gas_info:GasInfo)->np.ndarray:
    """
    calculate net production rates from rate constants and concentration
    """
    vk = - gas_info.reactant_stioch_coeffs + gas_info.product_stioch_coeffs
    forward_rates_of_progress = kf * np.exp(np.matmul(np.log(C + 1e-300), gas_info.reactant_stioch_coeffs))
    reverse_rates_of_progress = kr * np.exp(np.matmul(np.log(C + 1e-300), gas_info.product_stioch_coeffs))
    qdot = np.subtract(forward_rates_of_progress, reverse_rates_of_progress)
    wdot = np.matmul(qdot, vk.T)
    return ProductionRates(forward_rates_of_progress=forward_rates_of_progress,
                           reverse_rates_of_progress=reverse_rates_of_progress,
                           qdot=qdot,
                           wdot=wdot)

def get_dYdt(P:Union[float, np.ndarray],
             R:float, 
             gas_info:GasInfo,
             nasa_poly:NASAPolynomials,
             kinetics_coeffs:KineticsCoeffs,
             kinetics_data:KineticsData,
             t:Union[float, np.ndarray],
             state_vec:np.ndarray )->np.ndarray:
    """
    get dYdt where dYdt[0] is dTdt and rest is dYdt
    """
    T = state_vec[0]
    Y = state_vec[1:]
    T = np.clip(T, a_min=200., a_max=1e5)
    Y = np.clip(Y, a_min=0., a_max=1.)
    mean_molecular_weight = get_mean_molecular_weight(Y, gas_info.molecular_weights)
    density_mass = P/R/T*mean_molecular_weight
    X = Y2X(Y, gas_info.molecular_weights, mean_molecular_weight)
    C = Y2C(Y, gas_info.molecular_weights, density_mass)
    cp_data = calculate_cp(T, X, R, nasa_poly)
    enthalpy_data = calculate_enthalpy(T, X, R, nasa_poly)
    entropy_data = calculate_entropy(T, P, X, R, nasa_poly)
    cp_mass = cp_data.cp_mole/mean_molecular_weight
    Kc = get_equilibirum_constants(T, P, R, entropy_data.sdivR, enthalpy_data.hdivRT, gas_info)
    kf = get_forward_rate_constants(T, R, C, kinetics_coeffs, kinetics_data)
    kr = get_reverse_rate_constants(kf, Kc, kinetics_data.is_reversible)
    production_rates = get_production_rates(kf, kr, C, gas_info)
    Ydot = (production_rates.wdot*gas_info.molecular_weights)/density_mass
    Tdot = -(np.matmul(enthalpy_data.partial_molar_enthalpies,production_rates.wdot))/(density_mass*cp_mass)
    dYdt = np.hstack((Tdot, Ydot))
    dYdt = delete_small_numbers(dYdt)
    return dYdt