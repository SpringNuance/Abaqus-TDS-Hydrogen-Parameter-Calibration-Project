import numpy as np
import pandas as pd
import glob

from modules.initial_simulation import *
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *

from optimization.GaussianProcess import *
from optimization.BayesianOptimization import *

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
    
    print_log("\n===========================================", log_path)
    print_log("= Stage 5: Train Gaussian Process model    =", log_path)
    print_log("===========================================\n", log_path)

    # It is highly recommended to do this stage in the notebook file
    # on either CSC, Google Colab or Kaggle notebook to make use of GPU 
    # and visualize the process. 

    # This stage, therefore, only loads the pretrained models

    model_config = global_configs['model_config']
    
    
    stage5_outputs = {
        "GP_model": GP_model,
        "likelihood": likelihood,
    }
   
    return stage5_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)
    main_prepare_sim_data(global_configs, stage2_outputs)
    main_train_GP_model(global_configs, stage4_outputs)
    