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

def loss_flow_curve_MSE(true_plastic_strain, interpolated_target_stress, interpolated_sim_stress):
    MSE_loss = np.sqrt(np.mean((interpolated_target_stress - interpolated_sim_stress)**2))
    return MSE_loss

def loss_FD_curve(interpolated_displacement, interpolated_target_force, interpolated_sim_force, loss_function):
    if loss_function == 'MSE':
        return loss_FD_curve_MSE(interpolated_displacement, interpolated_target_force, interpolated_sim_force)
    elif loss_function == 'RMSE':
        return loss_FD_curve_RMSE(interpolated_displacement, interpolated_target_force, interpolated_sim_force)
    elif loss_function == 'area':
        return loss_FD_curve_area(interpolated_displacement, interpolated_target_force, interpolated_sim_force)
    else:
        raise ValueError(f"Loss function {loss_function} not supported. Please choose from 'MSE', 'RMSE', 'area'")
    
def loss_FD_curve_MSE(interpolated_displacement, interpolated_target_force, interpolated_sim_force):
    MSE_loss = np.mean((interpolated_target_force - interpolated_sim_force)**2)
    return MSE_loss

def loss_FD_curve_RMSE(interpolated_displacement, interpolated_target_force, interpolated_sim_force):
    RMSE_loss = np.sqrt(np.mean((interpolated_target_force - interpolated_sim_force)**2))
    return RMSE_loss

def loss_FD_curve_area(interpolated_displacement, interpolated_target_force, interpolated_sim_force):
    # Implementing numerical integration of the area bounded by 
    # the two curves and two vertical x axis
    # Define the x-range boundary

    x_start = min(interpolated_displacement)
    x_end = max(interpolated_displacement)
    # print(interpolated_displacement)

    # Interpolate the simulated force-displacement curve
    sim_FD_func = interp1d(interpolated_displacement, interpolated_sim_force, fill_value="extrapolate")
    target_FD_func = interp1d(interpolated_displacement, interpolated_target_force, fill_value="extrapolate")

    # Evaluate the two curves at various points within the x-range boundary
    x_values = np.linspace(x_start, x_end, num=1000)

    # Create numpy array flag where the sim is higher than the target
    sim_higher_than_target = np.array(sim_FD_func(x_values) > target_FD_func(x_values))

    # Find all index where the boolean turns opposite
    turning_indices = np.where(sim_higher_than_target[:-1] != sim_higher_than_target[1:])

    if len(turning_indices) == 0:
        if sim_higher_than_target[0] == True:
            # Sim curve is higher than target curve
            y_upper_curve = sim_FD_func(x_values)
            y_lower_curve = target_FD_func(x_values)
        else:
            # Target curve is higher than sim curve
            y_upper_curve = target_FD_func(x_values)
            y_lower_curve = sim_FD_func(x_values)
        # Calculate the area under each curve using the trapezoidal rule
        area_upper = simpson(y_upper_curve, x_values)
        area_lower = simpson(y_lower_curve, x_values)
        bounded_area = area_upper - area_lower
    else:
        turning_indices = np.insert(turning_indices, 0, 0)
        turning_indices = np.insert(turning_indices, len(turning_indices), len(x_values) - 1)

        #print(turning_indices)
        bounded_area = 0
        for i in range(len(turning_indices) - 1):
            previous_index, current_index = tuple(turning_indices[i:i+2])
            if sim_higher_than_target[current_index] == True:
                # Sim curve is higher than target curve
                y_upper_curve = sim_FD_func(x_values[previous_index:current_index + 1])
                y_lower_curve = target_FD_func(x_values[previous_index:current_index + 1])
            else:
                # Target curve is higher than sim curve
                y_upper_curve = target_FD_func(x_values[previous_index:current_index + 1])
                y_lower_curve = sim_FD_func(x_values[previous_index:current_index + 1])
            # Calculate the area under each curve using the trapezoidal rule
            area_upper = simpson(y_upper_curve, x_values[previous_index:current_index + 1])
            area_lower = simpson(y_lower_curve, x_values[previous_index:current_index + 1])
            bounded_area += area_upper - area_lower
        return bounded_area

def stop_condition_SOO(interpolated_target_force, 
                        interpolated_target_displacement,
                        interpolated_sim_force, 
                        stop_value_deviation_percent_objective,
                        stop_num_points_percent_objective, 
                        loss_function):
    """
    The function checks if the simulated curve is within the deviation percentage of the target curve
    """
    max_target_force = max(interpolated_target_force)
    deviation_force = max_target_force * stop_value_deviation_percent_objective
    target_force_upper = interpolated_target_force + deviation_force
    target_force_lower = interpolated_target_force - deviation_force
    num_satisfied_points = 0
    for i in range(len(interpolated_target_force)):
        if interpolated_sim_force[i] >= target_force_lower[i] and interpolated_sim_force[i] <= target_force_upper[i]:
            num_satisfied_points += 1

    sim_satisfied = num_satisfied_points / len(interpolated_target_force) >= stop_num_points_percent_objective
    loss_value = loss_FD_curve(interpolated_target_displacement, interpolated_target_force, interpolated_sim_force, loss_function)
    
    # print(f"Max target force: {max_target_force}")
    # print(f"Deviation force: {deviation_force}")
    # print(f"Target force upper: {target_force_upper}")
    # print(f"Target force lower: {target_force_lower}")
    # print(f"Displacement: {interpolated_target_displacement}")
    # print(f"Sim force: {interpolated_sim_force}")
    # print(f"Num satisfied points: {num_satisfied_points}")
    # print(f"Sim satisfied: {sim_satisfied}")
    # print(f"Loss value: {loss_value}")
    # time.sleep(180)
    return sim_satisfied, loss_value, num_satisfied_points

def stop_condition_MOO(target_forces_interpolated_combined: dict[str, np.ndarray], # 1D array
                        target_displacements_interpolated_combined: dict[str, np.ndarray], # 1D array
                        sim_forces_interpolated_combined: dict[str, np.ndarray], # 2D array
                        objectives: list[str], stop_value_deviation_percent: dict[str, float],
                        stop_num_points_percent: dict[str, float], loss_function: str):
    """
    The function checks if the simulated curves are within the deviation percentage of the target curves
    """
    stop_condition_all_objectives_all_sims: list[dict[str, bool]] = []
    loss_values_all_objectives_all_sims: list[dict[str, float]] = []
    num_satisfied_points_all_objectives_all_sims: list[dict[str, int]] = []
    
    stop_condition_is_met_all_objectives_any_sim = False
    stop_condition_is_met_all_objectives_any_sim_index = None

    lowest_loss_MOO_value = np.inf
    lowest_loss_MOO_index = None

    highest_num_satisfied_points_MOO_value = 0
    highest_num_satisfied_points_MOO_index = None

    batch_size = sim_forces_interpolated_combined[objectives[0]].shape[0]
    
    for index in range(batch_size):
        stop_condition_all_objectives_one_sim = {}
        loss_values_all_objectives_one_sim = {}
        satisfied_num_points_all_objectives_one_sim = {}
   
        stop_condition_is_met_all_objectives_one_sim = True
        
        for objective in objectives:
            sim_satisfied_one_objective_one_sim,\
            loss_value_one_objective_one_sim,\
            num_satisfied_points_one_objective_one_sim = stop_condition_SOO(target_forces_interpolated_combined[objective], 
                                                                            target_displacements_interpolated_combined[objective],
                                                                            sim_forces_interpolated_combined[objective][index], 
                                                                            stop_value_deviation_percent[objective],
                                                                            stop_num_points_percent[objective], loss_function)
            stop_condition_all_objectives_one_sim[objective] = sim_satisfied_one_objective_one_sim
            loss_values_all_objectives_one_sim[objective] = loss_value_one_objective_one_sim
            satisfied_num_points_all_objectives_one_sim[objective] = num_satisfied_points_one_objective_one_sim
            
            stop_condition_is_met_all_objectives_one_sim &= sim_satisfied_one_objective_one_sim
        
        stop_condition_all_objectives_all_sims.append(stop_condition_all_objectives_one_sim)
        loss_values_all_objectives_all_sims.append(loss_values_all_objectives_one_sim)
        num_satisfied_points_all_objectives_all_sims.append(satisfied_num_points_all_objectives_one_sim)

        stop_condition_is_met_all_objectives_any_sim |= stop_condition_is_met_all_objectives_one_sim
        
        if stop_condition_is_met_all_objectives_one_sim:
            stop_condition_is_met_all_objectives_any_sim_index = index
        
        loss_MOO_value = loss_MOO(loss_values_all_objectives_one_sim)
        if loss_MOO_value < lowest_loss_MOO_value:
            lowest_loss_MOO_value = loss_MOO_value
            lowest_loss_MOO_index = index

        num_satisfied_points_MOO_value = sum(satisfied_num_points_all_objectives_one_sim.values())
        if num_satisfied_points_MOO_value > highest_num_satisfied_points_MOO_value:
            highest_num_satisfied_points_MOO_value = num_satisfied_points_MOO_value
            highest_num_satisfied_points_MOO_index = index

    stop_diagnostic = {
            
            "stop_condition_is_met_all_objectives_any_sim": stop_condition_is_met_all_objectives_any_sim,
            "stop_condition_is_met_all_objectives_any_sim_index": stop_condition_is_met_all_objectives_any_sim_index,
            "lowest_loss_MOO_value": lowest_loss_MOO_value,
            "lowest_loss_MOO_index": lowest_loss_MOO_index,
            "highest_num_satisfied_points_MOO_value": highest_num_satisfied_points_MOO_value,
            "highest_num_satisfied_points_MOO_index": highest_num_satisfied_points_MOO_index,

            "stop_condition_all_objectives_all_sims": stop_condition_all_objectives_all_sims,
            "loss_values_all_objectives_all_sims": loss_values_all_objectives_all_sims,
            "num_satisfied_points_all_objectives_all_sims": num_satisfied_points_all_objectives_all_sims,
    
            "loss_function": loss_function,
    }

    return stop_diagnostic

def loss_MOO(loss_values_all_objectives_one_sim: dict[str, float]):
    """
    The function calculates the loss for the multi-objective losses
    Optimal loss is 0
    """
    loss_objectives = np.array(list(loss_values_all_objectives_one_sim.values()))
    return np.sqrt(np.sum(loss_objectives**2))