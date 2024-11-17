
##################################################
# IMPORT ALL THE STAGES AND RUN THEM IN SEQUENCE #
##################################################

import src.stage1_global_configs as stage1_global_configs 
import src.stage2_prepare_common_data as stage2_prepare_common_data
import src.stage3_run_initial_sims as stage3_run_initial_sims
import src.stage4_prepare_sim_data as stage4_prepare_sim_data
import src.stage5_train_GP_model as stage5_train_GP_model
import src.stage6_run_iteration_sims as stage6_run_iteration_sims

from configs.chosen_project import chosen_project_path_default
import argparse

############################################################
#                                                          #
#        ABAQUS HARDENING FLOW CURVE CALIBRATION           #
#   Tools required: Abaqus and Finnish Supercomputer CSC   #
#                                                          #
############################################################

def main_pipeline(stage, chosen_project_path):

    # ======================================================= #
    # Stage 1: Initialize directories and load global configs #
    # ======================================================= #
    
    if stage >= 1:
        
        if chosen_project_path is None:
            chosen_project_path = chosen_project_path_default
        else: 
            if not chosen_project_path.endswith(".json"):
                raise ValueError("The configuration file must be a json file")

        global_configs = stage1_global_configs.main_global_configs(chosen_project_path=chosen_project_path)

    # ============================== #
    # Stage 2: Prepare common data   #
    # ============================== #

    if stage >= 2:

        stage2_outputs = stage2_prepare_common_data.main_prepare_common_data(global_configs)

    # ================================ #
    # Stage 3: Run initial simulations #
    # ================================ #

    if stage >= 3:

        stage3_outputs = stage3_run_initial_sims.main_run_initial_sims(global_configs, stage2_outputs)

    # ========================================== #
    # Stage 4: Prepare initial simulation data   #
    # ========================================== #

    if stage >= 4:
        
        stage4_outputs = stage4_prepare_sim_data.main_prepare_sim_data(global_configs, stage2_outputs) 
        
    # ======================================= #
    # Stage 5: Train Gaussian Process models  #
    # ======================================= #

    if stage >= 5:
        
        stage5_outputs = stage5_train_GP_model.main_train_GP_model(global_configs, stage4_outputs)

    # ================================== #
    # Stage 7: Run iterative simulations #
    # ================================== #
    
    if stage >= 6:

        while (True):
            stage6_outputs = stage6_run_iteration_sims.main_run_iteration_sims(global_configs, stage2_outputs, stage3_outputs,
                                                                                stage4_outputs, stage5_outputs)
            stage4_outputs = stage4_prepare_sim_data.main_prepare_sim_data(global_configs, stage2_outputs)
            stage5_outputs = stage5_train_GP_model.main_train_GP_model(global_configs)
     
def parse_args():
    stages_name = ["Stage 1: Initialize directories and load global configs",
                   "Stage 2: Prepare common data",
                   "Stage 3: Run initial simulations",
                   "Stage 4: Prepare simulation data",
                   "Stage 5: Train GP models",
                   "Stage 6: Run iteration simulations"]
    
    parser = argparse.ArgumentParser(description="Abaqus TDS Hydrogen Bayesian Optimization")
    parser.add_argument("--stage", type=int, choices=range(1, 8), default=7, 
                        help=" || ".join(stages_name))
    parser.add_argument("--config-path", type=str, default=None, 
                        help="Path to the configuration file. Default json is taken from configs/chosen_project.py.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main_pipeline(args.stage, args.config_path)
