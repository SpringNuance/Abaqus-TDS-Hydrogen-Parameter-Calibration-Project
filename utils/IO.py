##############################################
# Util functions for input/output operations #
##############################################

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import time
import os
import logging
import copy

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

def pretty_print_parameter(params_dict, param_config, log_path=None):
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

from prettytable import PrettyTable

def pretty_print_parameters(params_dict_list, param_config, log_path=None):
    """
    Print multiple parameter sets in a pretty table format, where each column shows a candidate set of parameters.
    
    :param params_dict_list: List of dictionaries containing parameter sets, e.g., [{'param1': val1, 'param2': val2}, ...]
    :param param_config: Dictionary of parameter configuration, containing name and unit for each parameter.
    :param log_path: Optional path to log the table output.
    """
    # Initialize PrettyTable
    log_table = PrettyTable()

    # Set up field names with the parameter names as rows and candidate labels as columns
    columns = ["Parameter"] + [f"Candidate {i + 1}" for i in range(len(params_dict_list))]
    log_table.field_names = columns

    # Build rows based on parameter names, iterating over each parameter in param_config
    for param, config in param_config.items():
        param_name = config['name']
        param_unit = config['unit']
        
        # Initialize a row with the parameter name
        row = [param_name]
        
        # Add values from each candidate set, formatted with the unit if available
        for params_dict in params_dict_list:
            param_value = params_dict.get(param, "N/A")  # Default to "N/A" if missing
            param_value_unit = f"{param_value} {param_unit}" if param_unit != "dimensionless" else param_value
            row.append(param_value_unit)

        # Add the row to the table
        log_table.add_row(row)

    # Generate the table string
    string_message = "\n" + log_table.get_string() + "\n"
    
    # Print or log the table
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

def read_TDS_measurement(file_path, columns_TDS_measurement):
    """
    Read the force-displacement curve from the .txt file from Abaqus output
    """
    output_data = np.loadtxt(file_path, skiprows=3)
    
    df = pd.DataFrame(data=output_data, columns=columns_TDS_measurement)
    # Converting to numpy array
    # We would like to round the first column (time) to integers
    # round it to 0 decimal places
    df.iloc[:, 0] = df.iloc[:, 0].round(0).astype(int)
    return df

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
    
def return_description_properties(properties_path_excel):
    description_properties_dict = {
        "mechanical_properties": {},
        "hydrogen_diffusion_properties": {},
    }

    # Loading the properties file
    # Cast to string to avoid issues with the mixed types in the excel file
    properties_df = pd.read_excel(properties_path_excel, dtype=str)

    # find number of rows
    # max_nprops = properties_df.shape[0]

    mechanical_descriptions_list = properties_df["mechanical_descriptions"].dropna().tolist()
    mechanical_keys_list = properties_df["mechanical_keys"].dropna().tolist()
    mechanical_values_list = properties_df["mechanical_values"].dropna().tolist()

    hydrogen_diffusion_descriptions_list = properties_df["hydrogen_diffusion_descriptions"].dropna().tolist()
    hydrogen_diffusion_keys_list = properties_df["hydrogen_diffusion_keys"].dropna().tolist()
    hydrogen_diffusion_values_list = properties_df["hydrogen_diffusion_values"].dropna().tolist()

    ### Now we add the values to the dictionary

    for i in range(len(mechanical_keys_list)):
        description_properties_dict["mechanical_properties"][mechanical_keys_list[i]] = {
            "value": mechanical_values_list[i],
            "description": mechanical_descriptions_list[i]
        }
    for i in range(len(hydrogen_diffusion_keys_list)):
        description_properties_dict["hydrogen_diffusion_properties"][hydrogen_diffusion_keys_list[i]] = {
            "value": hydrogen_diffusion_values_list[i],
            "description": hydrogen_diffusion_descriptions_list[i]
        }
    return description_properties_dict

def return_UMAT_property(description_properties_dict): 
    mechanical_descriptions_and_values = list(description_properties_dict["mechanical_properties"].values())
    mechanical_values_list = [mechanical_descriptions_and_values[i]["value"] 
                                for i in range(len(mechanical_descriptions_and_values))]
    mechanical_description_list = [mechanical_descriptions_and_values[i]["description"]
                                for i in range(len(mechanical_descriptions_and_values))]

    # Abaqus needs to define 8 properties each line
    mech_prop_num_lines = int(np.ceil(len(mechanical_values_list)/8))
    mech_prop_num_properties = int(mech_prop_num_lines*8)

    total_UMAT_num_properties = mech_prop_num_properties 

    UMAT_property = []
    
    # The last line would be padded with 0.0 and their corresponding description would be "none"
    # If the number of properties is not a multiple of 8

    # For mechanical properties
    UMAT_property.append("**")
    UMAT_property.append("** =====================")
    UMAT_property.append("**")
    UMAT_property.append("** MECHANICAL PROPERTIES")
    UMAT_property.append("**")
    
    for line_index in range(mech_prop_num_lines):
        if line_index != mech_prop_num_lines - 1:
            subset_properties = mechanical_values_list[line_index*8:(line_index+1)*8]
            subset_description = mechanical_description_list[line_index*8:(line_index+1)*8]
            UMAT_property.append(", ".join(subset_properties))
            UMAT_property.append("** " + ", ".join(subset_description[0:4]))
            UMAT_property.append("** " + ", ".join(subset_description[4:8]))
        else:
            subset_properties = mechanical_values_list[line_index*8:] + ["0.0"]*(8-len(mechanical_values_list[line_index*8:]))
            subset_description = mechanical_description_list[line_index*8:] + ["none"]*(8-len(mechanical_description_list[line_index*8:]))
            UMAT_property.append(", ".join(subset_properties))
            UMAT_property.append("** " + ", ".join(subset_description[0:4]))
            UMAT_property.append("** " + ", ".join(subset_description[4:8]))
    
    UMAT_property.append("**")
    UMAT_property.append("*******************************************************")

    return UMAT_property, total_UMAT_num_properties


def return_UMATHT_property(description_properties_dict): 
    
    hydrogen_diffusion_descriptions_and_values = list(description_properties_dict["hydrogen_diffusion_properties"].values())
    hydrogen_diffusion_values_list = [hydrogen_diffusion_descriptions_and_values[i]["value"]
                                for i in range(len(hydrogen_diffusion_descriptions_and_values))]
    hydrogen_diffusion_description_list = [hydrogen_diffusion_descriptions_and_values[i]["description"]
                                for i in range(len(hydrogen_diffusion_descriptions_and_values))]
    # Abaqus needs to define 8 properties each line
    hydrogen_diffusion_prop_num_lines = int(np.ceil(len(hydrogen_diffusion_values_list)/8))
    hydrogen_diffusion_prop_num_properties = int(hydrogen_diffusion_prop_num_lines*8)

    total_UMATHT_num_properties = hydrogen_diffusion_prop_num_properties 
    
    UMATHT_property = []
    
    # The last line would be padded with 0.0 and their corresponding description would be "none"
    # If the number of properties is not a multiple of 8

    # For hydrogen diffusion properties
    UMATHT_property.append("**")
    UMATHT_property.append("** =============================")
    UMATHT_property.append("**")
    UMATHT_property.append("** HYDROGEN DIFFUSION PROPERTIES")
    UMATHT_property.append("**")

    for line_index in range(hydrogen_diffusion_prop_num_lines):
        if line_index != hydrogen_diffusion_prop_num_lines - 1:
            subset_properties = hydrogen_diffusion_values_list[line_index*8:(line_index+1)*8]
            subset_description = hydrogen_diffusion_description_list[line_index*8:(line_index+1)*8]
            UMATHT_property.append(", ".join(subset_properties))
            UMATHT_property.append("** " + ", ".join(subset_description[0:4]))
            UMATHT_property.append("** " + ", ".join(subset_description[4:8]))
        else:
            subset_properties = hydrogen_diffusion_values_list[line_index*8:] + ["0.0"]*(8-len(hydrogen_diffusion_values_list[line_index*8:]))
            subset_description = hydrogen_diffusion_description_list[line_index*8:] + ["none"]*(8-len(hydrogen_diffusion_description_list[line_index*8:]))
            UMATHT_property.append(", ".join(subset_properties))
            UMATHT_property.append("** " + ", ".join(subset_description[0:4]))
            UMATHT_property.append("** " + ", ".join(subset_description[4:8]))

    UMATHT_property.append("**")
    UMATHT_property.append("*******************************************************")

    return UMATHT_property, total_UMATHT_num_properties


def return_depvar(depvar_excel_path):

    depvar_df = pd.read_excel(depvar_excel_path, dtype=str)
    nstatev = len(depvar_df)
    #print("The number of state variables is: ", nstatev)

    DEPVAR = [
        "*Depvar       ",
        f"  {nstatev},      ",  
    ]

    depvar_index = depvar_df["depvar_index"].tolist()
    depvar_name = depvar_df["depvar_name"].tolist()

    for i in range(1, nstatev + 1):
        index = depvar_index[i-1]
        name = depvar_name[i-1]
        DEPVAR.append(f"{index}, AR{index}_{name}, AR{index}_{name}")


    return DEPVAR, nstatev

def replace_TDS_props(file_path, UMAT_PROPERTY, total_UMAT_num_properties, 
                                UMATHT_PROPERTY, total_UMATHT_num_properties):
                               # DEPVAR, nstatev):
    # Open the input file
    with open(file_path, 'r') as fid:
        flines = fid.readlines()

    # Process the lines
    flines = [line.strip() for line in flines]
    flines_new = copy.deepcopy(flines)

    # Now, we would reconstruct the input file as follows

    # 1. Replace the UMAT properties

    # Replacing UMAT properties
    umat_index = [i for i, line in enumerate(flines_new) if '*USER MATERIAL' in line.upper() and 'MECHANICAL' in line.upper()][0]
    
    # Find where the current UMAT section ends (by finding the next line that starts with '*')
    next_star_line_umat = next(i for i in range(umat_index + 1, len(flines_new)) if flines_new[i].startswith('*'))
    
    # Replace the number of constants in the *User Material line
    flines_new[umat_index] = f"*User Material, constants={total_UMAT_num_properties}, type=MECHANICAL"
    
    # Replace the content under UMAT
    flines_new = flines_new[:umat_index + 1] + UMAT_PROPERTY + flines_new[next_star_line_umat:]

    # 2. Replace the UMATHT properties

    # Replacing UMATHT properties
    umatht_index = [i for i, line in enumerate(flines_new) if '*USER MATERIAL' in line.upper() and 'THERMAL' in line.upper()][0]
    
    # Find where the current UMATHT section ends (by finding the next line that starts with '*')
    next_star_line_umatht = next(i for i in range(umatht_index + 1, len(flines_new)) if flines_new[i].startswith('*'))
    
    # Replace the number of constants in the *User Material line
    flines_new[umatht_index] = f"*User Material, constants={total_UMATHT_num_properties}, type=THERMAL"
    
    # Replace the content under UMATHT
    flines_new = flines_new[:umatht_index + 1] + UMATHT_PROPERTY + flines_new[next_star_line_umatht:]

    # 3. We would also modify the *Depvar section to include the key descriptions
    
    # Replacing Depvar section

    # # find the index of the *Depvar section
    # depvar_index = [i for i, line in enumerate(flines_new) if '*DEPVAR' in line.upper()][0]

    # #print("The initial conditions are: ", INITIAL_CONDITIONS)

    # flines_new = flines_new[:depvar_index] + DEPVAR + flines_new[depvar_index+2:]

    with open(file_path, 'w') as fid:
        for line in flines_new:
            fid.write(line + "\n")


def replace_surface_H(file_path, surface_H):
    # Open the input file
    with open(file_path, 'r') as fid:
        flines = fid.readlines()

    # Now, in the input file, there will be a section that looks like this

    # ** Name: <some name> Type: Temperature
    # *Boundary, amplitude=<some name>
    # <some name>, 11, 11, <surface_H value here>

    # Process the lines
    flines = [line.strip() for line in flines]
    flines_new = copy.deepcopy(flines)

    # Create a deep copy to preserve the original lines
    flines_new = copy.deepcopy(flines)

    # Define a flag to indicate if the target line was found and modified
    modified = False

    # Only process the last 300 lines in the loop
    start_index = max(0, len(flines) - 300)  # Calculate starting index to ensure it doesn't go negative
    for i in range(start_index, len(flines)):
        # Check if the line matches the pattern we're looking for
        if flines[i].startswith("** Name:") and "Type: Temperature" in flines[i]:
            # The next line after "** Name:" should be "*Boundary" and the following one should have the surface_H value
            if i + 2 < len(flines):  # Ensure we don't go out of bounds
                parts = flines[i + 2].split(",")
                if len(parts) >= 4:  # Ensure the line is long enough to contain a surface_H value
                    parts[-1] = str(surface_H)  # Replace the last part with the new surface_H value
                    flines_new[i + 2] = ", ".join(parts)
                    modified = True
                    break  # Stop after finding and modifying the first occurrence

    # Only write changes if modified
    if modified:
        # Write the modified lines back to the file
        with open(file_path, 'w') as fid:
            for line in flines_new:
                fid.write(line + "\n")
        # print("surface_H value updated successfully.")
    else:
        print("surface_H value not found in the last 300 lines of the file.")
