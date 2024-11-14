import pandas as pd
import time
from modules.initial_simulation import *
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *
from src.stage1_global_configs import * 
from math import *
from utils.sampling import *
from utils.hardening_laws import *

def main_prepare_common_data(global_configs):
    
    # -----------------------------------------#
    #  Stage 2: Preparing all common datas     #
    # -----------------------------------------#
    
    all_paths = global_configs['all_paths']
    num_measurements = global_configs['num_measurements']

    targets_path = all_paths['targets_path']
    log_path = all_paths['log_path']
    results_init_common_path = all_paths['results_init_common_path']
    
    print_log("==================================", log_path)
    print_log("= Stage 2: Preparing common data =", log_path)
    print_log("==================================\n", log_path)

    
    ####################################
    # Generating parameters parameters #
    ####################################
    
    if not os.path.exists(f"{results_init_common_path}/initial_sampled_parameters.npy"):
        print_log(f"\nThe initial sampled parameters do not exist. Generating the initial sampled parameters", log_path)
        param_config = global_configs['param_config']
        initial_sim_config = global_configs['initial_sim_config']
        num_samples = initial_sim_config['num_samples']
        sampling_method = initial_sim_config['sampling_method']
        initial_sampled_parameters = sampling(param_config, num_samples, sampling_method)
        np.save(f"{results_init_common_path}/initial_sampled_parameters.npy", initial_sampled_parameters)
    else:
        print_log(f"\nThe initial sampled parameters exist. Loading the sampled parameters", log_path)
        initial_sampled_parameters = np.load(f"{results_init_common_path}/initial_sampled_parameters.npy", allow_pickle=True)
    print_log(f"The first sampled parameter set is\n{initial_sampled_parameters[0]}", log_path)
    print_log(f"The number of sampled parameters: {len(initial_sampled_parameters)}", log_path)

    #############################
    # Loading the measurements #
    #############################

    target_TDS_measurements = {}
    
    df = pd.read_excel(f'{targets_path}/measurements.xlsx', engine='openpyxl')
    time_TDS = df['time'].to_numpy()
    C_wtppm = df['C_wtppm'].to_numpy()
    C_mol = df['C_mol'].to_numpy()

    for num_measurement in range(num_measurements):
        target_TDS_measurements[f"measurement_{num_measurement + 1}"] = {
            "time": time_TDS[num_measurement],
            "C_wtppm": C_wtppm[num_measurement],
            "C_mol": C_mol[num_measurement]
        }

    for num_measurement in range(num_measurements):
        print_log(f"\nTDS measurement {num_measurement + 1}: \n{target_TDS_measurements[f'measurement_{num_measurement + 1}']}", log_path)
    
    properties_path_excel = f"{targets_path}/properties.xlsx"
    depvar_excel_path = f"{targets_path}/depvar.xlsx"

    description_properties_dict = return_description_properties(properties_path_excel)

    print_log(f"\nSaving the TDS result", log_path)
    np.save(f"{targets_path}/target_TDS_measurements.npy", target_TDS_measurements)
    # time.sleep(180)
    
    stage2_outputs = {
        "initial_sampled_parameters": initial_sampled_parameters,
        "target_TDS_measurements": target_TDS_measurements,
        "description_properties_dict": description_properties_dict,
    }

    return stage2_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    main_prepare_common_data(global_configs)


