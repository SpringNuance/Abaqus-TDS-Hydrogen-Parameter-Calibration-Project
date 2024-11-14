import numpy as np
from scipy.stats import qmc
from typing import List, Dict
import random
import copy

def clamping_parameters(param_config: Dict[str, Dict[str, float]],
                        current_params_dict: Dict[str, float]) -> Dict[str, float]:
    clamped_params_dict = {}
    for param in param_config:
        if current_params_dict[param] < param_config[param]["lower"] * param_config[param]["exponent"]:
            clamped_params_dict[param] = param_config[param]["lower"] * param_config[param]["exponent"]
        elif current_params_dict[param] > param_config[param]["upper"] * param_config[param]["exponent"]:
            clamped_params_dict[param] = param_config[param]["upper"] * param_config[param]["exponent"]
        else:
            clamped_params_dict[param] = current_params_dict[param]
    return clamped_params_dict

def sampling(param_config: Dict[str, Dict[str, float]], 
             num_samples: int, method: str) -> List[Dict[str, float]]:
    if method == "LHS":
        sampled_points = latin_hypercube_sampling(param_config, num_samples)
        return sampled_points
    elif method == "SobolSequence":
        sampled_points = sobol_sequence_sampling(param_config, num_samples)
        return sampled_points
    else:
        raise ValueError(f"Unsupported sampling method: {method}, please choose 'LHS' or 'SobolSequence'")

def sampling_synthetic_predictions(param_config: Dict[str, Dict[str, float]], 
                                   current_params_dict: Dict[str, float], 
                                   num_synthetic_predictions: int, method: str) -> List[Dict[str, float]]:
    param_config_new_bounds = copy.deepcopy(param_config)

    normalized_current_params_dict = {}
    for param in current_params_dict:
        normalized_current_params_dict[param] = current_params_dict[param] / param_config[param]["exponent"]
    
    for param in param_config_new_bounds:
        param_config_new_bounds[param]["lower"] = normalized_current_params_dict[param] - param_config[param]["iter_max_range"]
        param_config_new_bounds[param]["upper"] = normalized_current_params_dict[param] + param_config[param]["iter_max_range"]
    
    if method == "LHS":
        sampled_points = latin_hypercube_sampling(param_config_new_bounds, num_synthetic_predictions)
        sampled_points = [clamping_parameters(param_config, sample_params) for sample_params in sampled_points]
        return sampled_points
    elif method == "SobolSequence":
        sampled_points = sobol_sequence_sampling(param_config_new_bounds, num_synthetic_predictions)
        sampled_points = [clamping_parameters(param_config, sample_params) for sample_params in sampled_points]
        return sampled_points
    else:
        raise ValueError(f"Unsupported sampling method: {method}, please choose 'LHS' or 'SobolSequence'")
    

def latin_hypercube_sampling(param_config: Dict[str, Dict[str, float]], num_samples: int, 
                             sims_spacing: int = 100000) -> List[Dict[str, float]]:
    """
    Generates a sample of parameter values using Latin Hypercube Sampling (LHS)
    from specified parameter ranges and an exponent for scaling.

    Latin Hypercube Sampling divides each parameter's range into equally probable intervals,
    ensuring that each interval is sampled only once. This function further supports scaling
    of parameter ranges by an exponent before sampling.

    Parameters:
    - param_config (Dict[str, Dict[str, float]]): A dictionary where keys are parameter names,
      and values are dictionaries specifying the 'lower', 'upper', and 'exponent' for each parameter.
      Example:
      {
        'param1': {'lower': 0.1, 'upper': 1.0, 'exponent': 1.0},
        'param2': {'lower': 10, 'upper': 100, 'exponent': 0.5}
      }

    - num_samples (int): The number of samples to generate. This is the number of unique parameter
      combinations that will be generated.

    - sims_spacing (int, optional): The number of divisions for the linspace generation of
      each parameter. Default is 10000, which defines the resolution of the sampling.

    Returns:
    - List[Dict[str, float]]: A list of dictionaries, each representing a unique set of parameter
      values sampled according to the LHS method. Each dictionary's keys are parameter names, and
      values are the sampled data points.

    Raises:
    - ValueError: If `num_samples` is larger than `sims_spacing`, since more samples than the 
      spacing can accommodate would violate the principles of LHS.

    """
    if num_samples > sims_spacing:
        raise ValueError("Number of samples cannot exceed initial simulation spacing.")
    
    # set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    linspace_values = {}
    for param in param_config:
        linspace_values[param] = np.linspace(
            start = param_config[param]["lower"] * param_config[param]["exponent"],
            stop = param_config[param]["upper"] * param_config[param]["exponent"],
            num = sims_spacing)
        linspace_values[param] = linspace_values[param].tolist()
    
    sampled_points = []
    for _ in range(num_samples):
        while True:
            candidate_param = {}
            for param, linspace_values_for_param in linspace_values.items():
                random.shuffle(linspace_values_for_param)
                candidate_param[param] = linspace_values_for_param.pop()
            if candidate_param not in sampled_points:
                break
        sampled_points.append(candidate_param)

    return sampled_points

def sobol_sequence_sampling(param_config: Dict[str, Dict[str, float]], num_samples: int, 
                           scramble = False) -> List[Dict[str, float]]:
    """
    Generate parameter samples using a Sobol sequence.
    
    Args:
    param_config (dict): Configuration for each parameter including lower and upper bounds and exponent.
    num_samples (int): The number of samples to generate. Should be a power of two.
    scramble (bool): Whether to scramble the Sobol sequence.

    Without Scrambling: Sobol sequences are deterministic and reproducible.
    With Scrambling: Adds a layer of non-determinism for better point distribution, but can be made deterministic by setting a seed.
    
    Returns:
    List[Dict[str, float]]: A list of dictionaries with each dictionary representing a sampled point.
    """
    if not is_power_of_two(num_samples):
        print("num_samples is recommended to be a power of two in Sobol Sequence")
        print("Example of num_samples are 16, 32, 64, 128, 256, 512, 1024, etc")
    
    dim = len(param_config)
    sampler = qmc.Sobol(d=dim, scramble=scramble)#, seed = 0)
    bounds = {param: (info["lower"] * info["exponent"], info["upper"] * info["exponent"])
              for param, info in param_config.items()}
    
    raw_samples = sampler.random_base2(m=int(np.log2(num_samples)))
    scaled_samples = [{param: scale_to_bounds(sample[i], *bounds[param])
             for i, param in enumerate(param_config)} for sample in raw_samples]
    
    return scaled_samples 

def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return (n & (n-1) == 0) and n != 0

def scale_to_bounds(value: float, lower: float, upper: float) -> float:
    """Scale a [0, 1] range value to [lower, upper]."""
    return value * (upper - lower) + lower