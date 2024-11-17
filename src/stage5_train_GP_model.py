import numpy as np
import pandas as pd
import glob

from modules.initial_simulation import *
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *

from optimization.GaussianProcess import *

from src.stage1_global_configs import * 
from src.stage2_prepare_common_data import *
from src.stage3_run_initial_sims import *
from src.stage4_prepare_sim_data import *

def main_train_GP_model(global_configs, stage4_outputs):

    # -----------------------------------------------------#
    #  Stage 5: Training the sequence to sequence ML model #
    # -----------------------------------------------------#

    all_paths = global_configs['all_paths']
    log_path = all_paths['log_path']
    models_path = all_paths['models_path']
    training_data_path = all_paths['training_data_path']
    targets_path = all_paths['targets_path']
    models_path = all_paths['models_path']
    param_config = global_configs['param_config']
    model_name = global_configs['model_config']['model_name']
    
    current_iteration_index = stage4_outputs['current_iteration_index']
    
    print_log("\n===========================================", log_path)
    print_log("= Stage 5: Train Gaussian Process model   =", log_path)
    print_log("===========================================\n", log_path)

    model_config = global_configs['model_config']
    
    if "surface_H" in param_config.keys():
        train_X = stage4_outputs["combined_features_normalized_augmented"]
        train_Y = stage4_outputs["combined_labels_augmented"]
    else:
        train_X = stage4_outputs["combined_features_normalized"]
        train_Y = stage4_outputs["combined_labels"]
    
    GP_model_wrapper = GaussianProcessWrapper(model_config, all_paths, train_X, train_Y)
    GP_model_wrapper.train_model()
    
    if current_iteration_index > 0:
        print_log(f"Previous iterations found. Saving the model to iteration_{index}", log_path)
        if not os.path.exists(f"{models_path}/iteration_{current_iteration_index}"):
            os.makedirs(f"{models_path}/iteration_{current_iteration_index}")
        GP_model_wrapper.save_model(f"{models_path}/iteration_{current_iteration_index}/{model_name}")
    else:
        print_log("No previous iterations found. Saving the model to initial folder", log_path)
        if not os.path.exists(f"{models_path}/initial"):
            os.makedirs(f"{models_path}/initial")
        GP_model_wrapper.save_model(f"{models_path}/initial/{model_name}")
    
    stage5_outputs = {
        "GP_model_wrapper": GP_model_wrapper
    }

    return stage5_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    stage3_outputs = main_run_initial_sims(global_configs, stage2_outputs)
    stage4_outputs = main_prepare_sim_data(global_configs, stage2_outputs)
    stage5_outputs = main_train_GP_model(global_configs, stage4_outputs)
    