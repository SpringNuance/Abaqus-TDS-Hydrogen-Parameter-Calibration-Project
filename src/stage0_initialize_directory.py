import os
import pandas as pd
import json
from utils.IO import check_create
from configs.chosen_project import chosen_project_path_default
import logging 

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def initialize_directory(project):
    """
    Initialize the necessary directories for the project.
    """

    # For log
    path = f"log/{project}"
    check_create(path)
    
    # Create a txt file called output.log if it does not exist
    if not os.path.exists(f"{path}/output.log"):
        with open(f"{path}/output.log", 'w') as log_file:
            log_file.write("")
    
    # For results_initial_common
    path = f"results_initial_common/{project}"
    check_create(path)

    # For results_initial_data
    path = f"results_initial_data/{project}"
    check_create(path)

    # For results_iteration_common
    path = f"results_iteration_common/{project}"
    check_create(path)
    check_create(f"{path}/iteration_common")
    
    # For results_iteration_data
    path = f"results_iteration_data/{project}"
    check_create(path)
    
    # For scripts
    path = f"scripts/{project}"
    check_create("scripts")

    # For sims_initial
    path = f"sims_initial/{project}"
    check_create(path)

    # For sims_iteration
    path = f"sims_iteration/{project}"
    check_create(path)

    # For targets
    path = f"targets/{project}"
    check_create(path)

    # For templates
    path = f"templates/{project}"
    check_create(path)

    # For models
    path = f"models/{project}"
    check_create(path)

    # For training_data
    path = f"training_data/{project}"
    check_create(path)

    # The project path folder
    project_path = os.getcwd()
    
    # The logging path
    log_path = f"log/{project}"

    # The results path
    results_init_data_path = f"results_initial_data/{project}"
    results_init_common_path = f"results_initial_common/{project}"
    results_iter_data_path = f"results_iteration_data/{project}"
    results_iter_common_path = f"results_iteration_common/{project}"
    
    # The scripts path
    scripts_path = f"scripts/{project}"
    
    # The simulations path
    sims_init_path = f"sims_initial/{project}"
    sims_iter_path = f"sims_iteration/{project}"

    # The target path
    targets_path = f"targets/{project}"

    # The templates path
    templates_path = f"templates/{project}"

    # The models path
    models_path = f"models/{project}"

    # The training data path
    training_data_path = f"training_data/{project}"


    all_paths = {
        "project_path": project_path,
        "log_path": log_path,
        "results_init_data_path": results_init_data_path,
        "results_init_common_path": results_init_common_path,
        "results_iter_data_path": results_iter_data_path,
        "results_iter_common_path": results_iter_common_path,
        "scripts_path": scripts_path,
        "sims_init_path": sims_init_path,
        "sims_iter_path": sims_iter_path,
        "targets_path": targets_path,
        "templates_path": templates_path,
        "models_path": models_path,
        "training_data_path": training_data_path
    }
    return all_paths

if __name__ == "__main__":

    with open(f"configs/{chosen_project_path_default}", encoding='utf-8') as f:
        global_configs = json.load(f)

    project = global_configs["project"]
    # objectives = global_configs["objectives"]

    initialize_directory(project)