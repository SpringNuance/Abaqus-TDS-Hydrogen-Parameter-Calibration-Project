##############################################
# Util functions for input/output operations #
##############################################

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import time
import os
import logging

def check_create(path):
    """
    Check if the directory exists, if not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def print_log(message, log_path, file_name="output.log", pause=0.02):
    """
    Print the message to the console and write it to the log file.
    We can change the pause to control the speed of the printed output
    """
    message = str(message)
    with open(f"{log_path}/{file_name}", 'a+') as log_file:
        log_file.writelines(message + "\n")
    print(message)
    time.sleep(pause)

def pretty_print_parameters(params_dict, param_config, log_path=None):
    """
    Print the parameters in a pretty table format
    """
    log_table = PrettyTable()
    log_table.field_names = ["parameter", "value"]
    for param, param_value in params_dict.items():
        param_name = param_config[param]['name']
        param_unit = param_config[param]['unit']
        param_value_unit = f"{param_value} {param_unit}" if param_unit != "dimensionless" else param_value
        log_table.add_row([param_name, param_value_unit])

    string_message = "\n"
    string_message += log_table.get_string()
    string_message += "\n"
    if log_path is not None:
        print_log(string_message, log_path)
    else:
        print(string_message)

def pretty_print_stop_diagnostic(stop_diagnostic, objectives, log_path):
    """
    Print the stop_diagnostic in a pretty table format
    """
    
    stop_condition_is_met_all_objectives_any_sim = stop_diagnostic["stop_condition_is_met_all_objectives_any_sim"]
    stop_condition_is_met_all_objectives_any_sim_index = stop_diagnostic["stop_condition_is_met_all_objectives_any_sim_index"]
    lowest_loss_MOO_value = stop_diagnostic["lowest_loss_MOO_value"]
    lowest_loss_MOO_index = stop_diagnostic["lowest_loss_MOO_index"]
    highest_num_satisfied_points_MOO_value = stop_diagnostic["highest_num_satisfied_points_MOO_value"]
    highest_num_satisfied_points_MOO_index = stop_diagnostic["highest_num_satisfied_points_MOO_index"]

    stop_condition_all_objectives_all_sims = stop_diagnostic["stop_condition_all_objectives_all_sims"]
    loss_values_all_objectives_all_sims = stop_diagnostic["loss_values_all_objectives_all_sims"]
    num_satisfied_points_all_objectives_all_sims = stop_diagnostic["num_satisfied_points_all_objectives_all_sims"]

    loss_function = stop_diagnostic["loss_function"]

    print_log(f"There exists a simulation that satisfies all objectives: {stop_condition_is_met_all_objectives_any_sim}", log_path)
    print_log(f"Their index: {stop_condition_is_met_all_objectives_any_sim_index}", log_path)
    
    #############################################
    # PRINT THE SIMULATION WITH LOWEST MOO LOSS #
    #############################################

    print_log(f"The simulation index with the lowest MOO loss: {lowest_loss_MOO_index}", log_path)
    print_log(f"Their MOO loss value: {lowest_loss_MOO_value}", log_path)
    
    log_table = PrettyTable()
    log_table.field_names = ["objective"] + objectives
    
    satisfied_each_objective = ["satisfied status"] + list(stop_condition_all_objectives_all_sims[lowest_loss_MOO_index].values())
    loss_values_each_objective = [f"loss value ({loss_function})"] + [round(loss, 2) for loss in list(loss_values_all_objectives_all_sims[lowest_loss_MOO_index].values())]
    num_satisfied_points_each_objective = ["num satisfied points"] + list(num_satisfied_points_all_objectives_all_sims[lowest_loss_MOO_index].values())
    
    log_table.add_row(satisfied_each_objective)
    log_table.add_row(loss_values_each_objective)
    log_table.add_row(num_satisfied_points_each_objective)

    string_message = "\n"
    string_message += log_table.get_string()
    string_message += "\n"

    print_log(string_message, log_path)

    ################################################################
    # PRINT THE SIMULATION WITH HIGHEST NUMBER OF SATISFIED POINTS #
    ################################################################

    print_log(f"The simulation index with the highest number of satisfied points: {highest_num_satisfied_points_MOO_index}", log_path)
    print_log(f"Their highest number of satisfied points: {highest_num_satisfied_points_MOO_value}", log_path)

    log_table = PrettyTable()
    log_table.field_names = ["objective"] + objectives
    
    satisfied_each_objective = ["satisfied status"] + list(stop_condition_all_objectives_all_sims[highest_num_satisfied_points_MOO_index].values())
    loss_values_each_objective = [f"loss value ({loss_function})"] + [round(loss, 2) for loss in list(loss_values_all_objectives_all_sims[highest_num_satisfied_points_MOO_index].values())]
    num_satisfied_points_each_objective = ["num satisfied points"] + list(num_satisfied_points_all_objectives_all_sims[highest_num_satisfied_points_MOO_index].values())
    
    log_table.add_row(satisfied_each_objective)
    log_table.add_row(loss_values_each_objective)
    log_table.add_row(num_satisfied_points_each_objective)

    string_message = "\n"
    string_message += log_table.get_string()
    string_message += "\n"

    print_log(string_message, log_path)

def read_FD_curve(file_path):
    """
    Read the force-displacement curve from the .txt file from Abaqus output
    """
    output_data = np.loadtxt(file_path, skiprows=2)
    # column 1 is time step
    # column 2 is displacement
    # column 3 is force
    columns = ['X', 'Displacement', 'Force']
    df = pd.DataFrame(data=output_data, columns=columns)
    # Converting to numpy array
    displacement = df.iloc[:, 1].to_numpy()
    force = df.iloc[:, 2].to_numpy()
    return displacement, force

def create_parameter_file(file_path, params_dict, create_excel=True, create_csv=True):
    """
    Create a parameter file in both Excel and CSV format
    """
    columns = ["parameter", "value"]
    df = pd.DataFrame(columns=columns)
    for key, value in params_dict.items():
        df.loc[len(df.index)] = [key, value]
    if create_excel:
        df.to_excel(f"{file_path}/parameters.xlsx", index=False)
    if create_csv:
        df.to_csv(f"{file_path}/parameters.csv", index=False)

def create_flow_curve_file(file_path, true_plastic_strain, true_stress, stress_unit="Pa", create_excel=True, create_csv=True):
    """
    Create a flow curve file in both Excel and CSV format   
    """
    columns = ["strain,-","stress,Pa","stress,MPa"]
    df = pd.DataFrame(columns=columns)
    if stress_unit == "Pa":
        true_stress_Pa = true_stress
        true_stress_MPa = true_stress / 1e6
    elif stress_unit == "MPa":
        true_stress_MPa = true_stress
        true_stress_Pa = true_stress * 1e6
    for i in range(len(true_plastic_strain)):
        df.loc[len(df.index)] = [true_plastic_strain[i], true_stress_Pa[i], true_stress_MPa[i]]
    if create_excel:
        df.to_excel(f"{file_path}/flow_curve.xlsx", index=False)
    if create_csv:
        df.to_csv(f"{file_path}/flow_curve.csv", index=False)

def create_FD_curve_file(file_path, displacement, force, 
                         displacement_unit="m", force_unit="N",
                         create_excel=True, create_csv=True
                         ):
    """
    Create a force-displacement curve file in both Excel and CSV format
    """

    columns = ["displacement,m","displacement,mm","force,kN","force,N"]
    df = pd.DataFrame(columns=columns)
    if force_unit == "kN":
        force_kN = force
        force_N = force * 1e3
    elif force_unit == "N":
        force_N = force
        force_kN = force * 1e-3
    else:
        raise ValueError("force_unit should be either 'N' or 'kN'")
    
    if displacement_unit == "m":
        displacement_m = displacement
        displacement_mm = displacement * 1e3
    elif displacement_unit == "mm":
        displacement_mm = displacement
        displacement_m = displacement * 1e-3
    else:
        raise ValueError("displacement_unit should be either 'm' or 'mm'")
    
    for i in range(len(displacement)):
        df.loc[len(df.index)] = [displacement_m[i], displacement_mm[i], force_kN[i], force_N[i]]
    if create_excel:
        df.to_excel(f"{file_path}/FD_curve.xlsx", index=False)
    if create_csv:
        df.to_csv(f"{file_path}/FD_curve.csv", index=False)
        
def replace_flow_curve(file_path, true_plastic_strain, true_stress):
    """
    Replace the flow curve data in the inp file of Abaqus
    """
    with open(file_path, 'r') as abaqus_inp:
        abaqus_inp_content = abaqus_inp.readlines()
    # Locate the section containing the stress-strain data
    start_line = None
    
    search_last_lines = 1000
    for i, line in enumerate(abaqus_inp_content[-search_last_lines:]):
        if '*Plastic' in line:
            start_line = len(abaqus_inp_content) - search_last_lines + i + 1

    if start_line is None:
        raise ValueError('Could not find the *Plastic data section')
    
    end_line = None
    for i, line in enumerate(abaqus_inp_content[start_line:]):
        if line.startswith('*'):
            end_line = start_line + i
            break

    # Modify the stress-strain data
    flow_curve_data = zip(true_stress, true_plastic_strain)
    # Update the .inp file
    new_lines = []
    new_lines.extend(abaqus_inp_content[:start_line])
    new_lines.extend([f'{stress},{strain}\n' for stress, strain in flow_curve_data])
    new_lines.extend(abaqus_inp_content[end_line:])

    # Write the updated .inp file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)
    #time.sleep(180)
