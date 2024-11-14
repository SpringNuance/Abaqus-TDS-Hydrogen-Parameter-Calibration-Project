import os
import time
import json
from prettytable import PrettyTable
from configs.chosen_project import chosen_project_path_default
from src.stage0_initialize_directory import *
from utils.IO import *

# ------------------------------------#
#   Stage 0: Recording configurations #
# ------------------------------------#

def main_global_configs(chosen_project_path=None):


    #########################
    # Global configurations #
    #########################
    
    if chosen_project_path is not None:
        # Load the json file 
        with open(chosen_project_path) as f:
            global_configs = json.load(f)
    else:
        # Load the json file 
        with open(chosen_project_path_default) as f:
            global_configs = json.load(f)

    project = global_configs["project"]
    # objectives = global_configs["objectives"]
    num_measurements = global_configs["num_measurements"]
    
    # Initialize the directories
    all_paths = initialize_directory(project)
    
    ###########################
    # Information declaration #
    ###########################

    global_configs["all_paths"] = all_paths
    
    if chosen_project_path is not None:
        global_configs["chosen_project_path"] = chosen_project_path
    else:
        global_configs["chosen_project_path"] = chosen_project_path_default
        
    ###############################################
    #  Printing the configurations to the console #
    ###############################################
    
    log_path = all_paths['log_path']
    
    # print current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_log(f"\nSTART RUNNING FROM *** {current_time} ***", log_path)

    print_log("\n==========================================", log_path)
    print_log("= Stage 1: Loading configs and all paths =", log_path)
    print_log("==========================================", log_path)

    print_log(f"\nWelcome to Abaqus Seq2Seq flow curve calibration project\n", log_path)
    print_log(f"The configurations you have chosen: \n", log_path)
    
    log_table = PrettyTable()

    log_table.field_names = ["Global Configs", "User choice"]
    #log_table.add_row(["PROJECT", project])
    #objective_string = ", ".join(objectives)
    #log_table.add_row(["OBJECTIVES", objective_string])
    
    for path in all_paths:
        log_table.add_row([path.upper(), all_paths[path]])
    print_log(log_table.get_string() + "\n", log_path)
    
    project_path = all_paths['project_path']
    print_log(f"The root path of your project folder is\n", log_path)
    print_log(f"{project_path}\n", log_path)

    #######################################
    # Returning the global configurations #
    #######################################
    
    return global_configs

if __name__ == "__main__":
    main_global_configs()
