import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import time

def calculate_loss(sim_measurements, target_measurements, loss_function):
    if loss_function == "MAE":
        return np.mean(abs(sim_measurements - target_measurements))
    elif loss_function == "MSE":
        return np.mean((sim_measurements - target_measurements)**2)
    elif loss_function == "RMSE":
        return np.sqrt(np.mean((sim_measurements - target_measurements)**2))
    else:
        raise ValueError(f"Loss function {loss_function} not supported. Please choose from 'MAE', 'MSE'")

def stop_condition_SOO(target_TDS_measurements, 
                        sim_TDS_measurements,
                        stop_value_deviation_percent, loss_function):
    """
    The function checks if the simulated curve is within the deviation percentage of the target curve
    """
    satisfied_sim_measurements = []
    loss_sim_measurements = []
    stop_condition_is_satisfied = False
    satisfied_sim_index = None
    best_sim_index = None

    for params_tuple, sim_measurements in sim_TDS_measurements.items():
        sim_measurements_C_mol = []
        target_measurements_C_mol = []
        stop_value_deviation_percent_list = []
        for measurement_name, sim_measurement in sim_measurements.items():
            for target_measurement in target_TDS_measurements.values():
                if target_measurement["time"] == sim_measurement["time"]:
                    sim_measurements_C_mol.append(sim_measurement["C_mol"])
                    target_measurements_C_mol.append(target_measurement["C_mol"])
                    stop_value_deviation_percent_list.append(stop_value_deviation_percent[measurement_name])
                    break
        sim_measurements_C_mol = np.array(sim_measurements_C_mol)
        target_measurements_C_mol = np.array(target_measurements_C_mol)
        loss_value = calculate_loss(sim_measurements_C_mol, target_measurements_C_mol, loss_function)

        for i in range(len(sim_measurements_C_mol)):
            current_sim_satisfied = True
            upper_target_measurement = target_measurements_C_mol[i] * (1 + stop_value_deviation_percent_list[i])
            lower_target_measurement = target_measurements_C_mol[i] * (1 - stop_value_deviation_percent_list[i])
            if (sim_measurements_C_mol[i] > upper_target_measurement) or (sim_measurements_C_mol[i] < lower_target_measurement):
                current_sim_satisfied = False
                break
        
        satisfied_sim_measurements.append(current_sim_satisfied)
        loss_sim_measurements.append(loss_value)
    
    stop_condition_is_satisfied = any(satisfied_sim_measurements)
    satisfied_sim_index = None
    for i, satisfied_sim in enumerate(satisfied_sim_measurements):
        if satisfied_sim:
            satisfied_sim_index = i
            break
    best_sim_index = np.argmin(loss_sim_measurements)
    best_sim_loss = loss_sim_measurements[best_sim_index]
    stop_condition_diagnostics = {"stop_condition_is_satisfied": stop_condition_is_satisfied,
                                    "satisfied_sim_index": satisfied_sim_index,
                                    "best_sim_index": best_sim_index,
                                    "best_sim_loss": best_sim_loss}
    return stop_condition_diagnostics
