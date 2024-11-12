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
    objectives = global_configs['objectives']

    targets_path = all_paths['targets_path']
    log_path = all_paths['log_path']
    results_init_common_path = all_paths['results_init_common_path']
    
    print_log("==================================", log_path)
    print_log("= Stage 2: Preparing common data =", log_path)
    print_log("==================================\n", log_path)

    #################################
    # Plastic strain configurations #
    #################################

    if not os.path.exists(f"{targets_path}/true_plastic_strain.npy"):    
        print_log(f"True plastic strain does not exist. Calculating the true plastic strain", log_path)
        true_plastic_strain_config = global_configs['true_plastic_strain_config']
        strain_start_end = true_plastic_strain_config["strain_start_end"]
        strain_step =  true_plastic_strain_config["strain_step"]
        true_plastic_strain = calculate_true_plastic_strain(strain_start_end, strain_step)
        # assert that true plastic strain must starts with 0 and always increases
        assert true_plastic_strain[0] == 0
        for i in range(1, len(true_plastic_strain)):
            assert true_plastic_strain[i] > true_plastic_strain[i-1]
        print_log(true_plastic_strain, log_path)
        np.save(f"{targets_path}/true_plastic_strain.npy", true_plastic_strain)
    else:
        print_log(f"True plastic strain exists. The true plastic strain is", log_path)
        true_plastic_strain = np.load(f"{targets_path}/true_plastic_strain.npy", allow_pickle=True)
        print_log(true_plastic_strain, log_path)

    ####################################
    # Generating Swift-Voce parameters #
    ####################################
    
    if not os.path.exists(f"{results_init_common_path}/initial_sampled_parameters.npy"):
        print_log(f"\nThe initial sampled parameters do not exist. Generating the initial sampled parameters", log_path)
        param_config = global_configs['param_config']
        initial_sim_config = global_configs['initial_sim_config']
        num_samples = initial_sim_config['num_samples']
        sampling_method = initial_sim_config['sampling_method']
        initial_sampled_parameters = sampling(param_config, num_samples, sampling_method)
    else:
        print_log(f"\nThe initial sampled parameters exist. Loading the sampled parameters", log_path)
        initial_sampled_parameters = np.load(f"{results_init_common_path}/initial_sampled_parameters.npy", allow_pickle=True)
    print_log(f"The first sampled parameter set is\n{initial_sampled_parameters[0]}", log_path)
    print_log(f"The number of sampled parameters: {len(initial_sampled_parameters)}", log_path)
    
    ##########################
    # Generating flow curves #
    ##########################

    if not os.path.exists(f"{results_init_common_path}/initial_sampled_true_stress.npy"):
        print_log(f"\nThe initial sampled true stress does not exist. Generating the initial sampled true stress", log_path)
        
        hardening_law = true_plastic_strain_config["hardening_law"]
        extrapolate_N_first_strain_values = true_plastic_strain_config["extrapolate_N_first_strain_values"]

        initial_sampled_true_stress = []

        for i in range(num_samples):
            parameters = initial_sampled_parameters[i]
            true_stress = calculate_true_stress(parameters, hardening_law, 
                                                                true_plastic_strain,
                                                                extrapolate_N_first_strain_values)
            # assert that the true_stress is monotonically increasing
            for j in range(1, len(true_stress)):
                assert true_stress[j] >= true_stress[j-1]
            initial_sampled_true_stress.append(true_stress)

        np.save(f"{results_init_common_path}/initial_sampled_true_stress.npy", initial_sampled_true_stress)
    else:
        print_log(f"\nThe initial sampled true stress exists. Loading the initial sampled true stress", log_path)
        initial_sampled_true_stress = np.load(f"{results_init_common_path}/initial_sampled_true_stress.npy", allow_pickle=True)

    #####################################
    # Create flow curve dictionary file #
    #####################################
    
    print_log(f"\nCreating the flow curve dictionary file", log_path)
    initial_sampled_flow_curves = {}
    initial_sim_config = global_configs['initial_sim_config']
    num_samples = initial_sim_config['num_samples']
    
    for i in range(num_samples):
        params_dict = initial_sampled_parameters[i]
        params_tuple = tuple(params_dict.items())
        true_stress = initial_sampled_true_stress[i]
        initial_sampled_flow_curves[params_tuple] = {'strain': true_plastic_strain, 'stress': true_stress}
    np.save(f"{results_init_common_path}/initial_sampled_flow_curves.npy", initial_sampled_flow_curves)

    ########################################################
    # Interpolating the referenced flow curve (if we have) #
    ########################################################
    
    from utils.calculation import interpolating_stress

    if os.path.exists(f"{targets_path}/referenced_flow_curve.csv"):
        print_log(f"\nThe referenced flow curve exists. Loading the referenced flow curve", log_path)
        referenced_flow_curve = pd.read_csv(f"{targets_path}/referenced_flow_curve.csv")
        strain = referenced_flow_curve['strain/-']
        stress = referenced_flow_curve['stress/Pa']
        referenced_stress_interpolated = interpolating_stress(strain, stress, true_plastic_strain)
        referenced_flow_curve_interpolated_pd = pd.DataFrame()
        referenced_flow_curve_interpolated_pd['strain/-'] = true_plastic_strain
        referenced_flow_curve_interpolated_pd['stress/Pa'] = referenced_stress_interpolated
        referenced_flow_curve_interpolated_pd['stress/MPa'] = referenced_stress_interpolated / 1e6
        referenced_flow_curve_interpolated_pd.to_csv(f"{targets_path}/referenced_flow_curve_interpolated.csv", index=False)
        
        referenced_flow_curve_interpolated = {
            "strain": true_plastic_strain,
            "stress": referenced_stress_interpolated
        }
        print_log(f"\nThe referenced flow curve is interpolated and saved", log_path)
    else:
        print_log(f"\nThe referenced flow curve does not exist", log_path)
        referenced_flow_curve_interpolated = None

    #############################
    # Loading the target curves #
    #############################

    target_curves_combined = {}
    for objective in objectives:
        df = pd.read_csv(f'{targets_path}/{objective}/FD_curve_final.csv')
        expDisplacement = df['displacement/mm'].to_numpy()
        expForce = df['force/N'].to_numpy()
        target_curve = {}
        target_curve['displacement'] = expDisplacement
        target_curve['force'] = expForce
        target_curves_combined[objective] = target_curve

    print_log(f"\nSaving the target curves", log_path)
    np.save(f"{targets_path}/target_curves_combined.npy", target_curves_combined)
    #time.sleep(180)
    
    ##########################################################
    # Creating interpolating displacement for each objective #
    ##########################################################

    interpolated_displacement_len = global_configs["interpolated_displacement_len"]

    interpolated_displacement_combined = {}

    exp_yielding_index = global_configs["exp_yielding_index"] 
    
    # Since the interpolate displacement are used in common for all data 
    # (initial sim FD curve, iteration sim FD curve and exp FD curve)
    # The most logical folder to store them is in the targets folder

    for i, objective in enumerate(objectives):
        if not os.path.exists(f"{targets_path}/{objective}/interpolated_displacement.xlsx"):
            FD_curve_final = pd.read_excel(f"{targets_path}/{objective}/FD_curve_final.xlsx", engine='openpyxl')
            exp_displacement = FD_curve_final['displacement/m']
            
            yielding_disp = exp_displacement.iloc[exp_yielding_index[objective]]
            fracture_disp = exp_displacement.iloc[-1]
            # print(f"yielding_disp: {yielding_disp}")
            # print(f"fracture_disp: {fracture_disp}")
            interpolated_displacement = np.linspace(yielding_disp, fracture_disp, interpolated_displacement_len, endpoint=True)
            
            interpolated_displacement_combined[objective] = interpolated_displacement
            
            interpolated_displacement_pd = pd.DataFrame()
            interpolated_displacement_pd['displacement/m'] = interpolated_displacement
            interpolated_displacement_pd['displacement/mm'] = interpolated_displacement * 1000
        
            interpolated_displacement_pd.to_excel(f"{targets_path}/{objective}/interpolated_displacement.xlsx", index=False)
        
            print_log(f"Saving the interpolated displacements for {objective}", log_path)
        else:
            print_log(f"Interpolated displacements for {objective} already exist", log_path)
            interpolated_displacement = pd.read_excel(f"{targets_path}/{objective}/interpolated_displacement.xlsx")['displacement/m'].to_numpy()
            interpolated_displacement_combined[objective] = interpolated_displacement
    
    if not os.path.exists(f"{targets_path}/interpolated_displacement_combined.npy"):
        np.save(f"{targets_path}/interpolated_displacement_combined.npy", interpolated_displacement_combined)
        print_log(f"\nInterpolated displacements for all objectives are saved\n", log_path)
    else:
        interpolated_displacement_combined = np.load(f"{targets_path}/interpolated_displacement_combined.npy", allow_pickle=True).tolist()
        print_log(f"\nInterpolated displacements for all objectives already exist\n", log_path)
    

    ##################################
    # Interpolating the exp FD curve #
    ##################################

    target_curves_interpolated_combined = {}

    for objective in objectives:
        if not os.path.exists(f"{targets_path}/{objective}/FD_curve_final_interpolated.xlsx"):
            FD_curve_final = pd.read_excel(f"{targets_path}/{objective}/FD_curve_final.xlsx", engine='openpyxl')
            exp_displacement = FD_curve_final['displacement/m']
            exp_force = FD_curve_final['force/N']
            interpolated_exp_displacement = interpolated_displacement_combined[objective]
            interpolated_exp_force = interpolating_force(exp_displacement, exp_force, interpolated_exp_displacement)
            
            target_curve_interpolated = {
                "displacement": interpolated_exp_displacement,
                "force": interpolated_exp_force
            }

            target_curves_interpolated_combined[objective] = target_curve_interpolated

            exp_FD_curve_interpolated = pd.DataFrame()
            exp_FD_curve_interpolated['displacement/m'] = interpolated_exp_displacement
            exp_FD_curve_interpolated['displacement/mm'] = interpolated_exp_displacement * 1000
            exp_FD_curve_interpolated['force/N'] = interpolated_exp_force
            exp_FD_curve_interpolated['force/kN'] = interpolated_exp_force / 1000
            exp_FD_curve_interpolated.to_excel(f"{targets_path}/{objective}/FD_curve_final_interpolated.xlsx", index=False)
            print_log(f"Saving the interpolated FD curve for {objective}", log_path)    
        else:
            print_log(f"Interpolated FD curve for {objective} already exists", log_path)
            interpolated_FD_curve = pd.read_excel(f"{targets_path}/{objective}/FD_curve_final_interpolated.xlsx", engine='openpyxl')
            interpolated_exp_displacement = interpolated_FD_curve['displacement/m'].to_numpy()
            interpolated_exp_force = interpolated_FD_curve['force/N'].to_numpy()
            target_curve_interpolated = {
                "displacement": interpolated_exp_displacement,
                "force": interpolated_exp_force
            }
            target_curves_interpolated_combined[objective] = target_curve_interpolated   
        # time.sleep(180)
    if not os.path.exists(f"{targets_path}/target_curves_interpolated_combined.npy"):
        np.save(f"{targets_path}/target_curves_interpolated_combined.npy", target_curves_interpolated_combined)
        print_log(f"\nInterpolated FD curves for all objectives are saved", log_path)
    else:
        target_curves_interpolated_combined = np.load(f"{targets_path}/target_curves_interpolated_combined.npy", allow_pickle=True).tolist()
        print_log(f"\nInterpolated FD curves for all objectives already exist", log_path)


    stage2_outputs = {
        "initial_sampled_parameters": initial_sampled_parameters,
        "initial_sampled_true_stress": initial_sampled_true_stress,
        "initial_sampled_flow_curves": initial_sampled_flow_curves,
        "referenced_flow_curve_interpolated": referenced_flow_curve_interpolated,
        "true_plastic_strain": true_plastic_strain,
        "target_curves_combined": target_curves_combined,
        "interpolated_displacement_combined": interpolated_displacement_combined,
        "target_curves_interpolated_combined": target_curves_interpolated_combined
    }

    return stage2_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    main_prepare_common_data(global_configs)


