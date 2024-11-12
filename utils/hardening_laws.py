import numpy as np
from scipy import interpolate
import time
from scipy.optimize import least_squares
from scipy.optimize import minimize

#================ defining the hardening laws =============================

def Swift_hardening_law(K, PEEQ_0, n, true_plastic_strain):
    """
    Calculate the true stress using the Swift hardening law.

    Parameters:
    K (units: Pa): Strength coefficient 
        - Physical meaning: This parameter scales the overall stress level. It represents the material's
          resistance to deformation and is often referred to as the strength coefficient.
    PEEQ_0 (dimensionless): Initial plastic strain offset 
        - Physical meaning: This is the pre-strain or the initial plastic strain offset. It accounts for
          the initial amount of plastic deformation that the material has undergone before the current
          loading.
    n (dimensionless): Strain hardening exponent 
        - Physical meaning: This parameter describes the rate at which the material hardens with increasing
          plastic strain. A higher value of n indicates a greater rate of hardening.
    true_plastic_strain (dimensionless) (float or array-like): True plastic strain 
        - Physical meaning: This is the actual plastic strain experienced by the material. It represents
          the amount of irreversible deformation the material has undergone.

    Returns:
    true_stress (units: Pa) (float or array-like)
        - The calculated true stress corresponding to the provided true plastic strain, based on the
          Swift hardening law.
    """
    true_stress = K * (PEEQ_0 + true_plastic_strain) ** n
    return true_stress

def Voce_hardening_law(sigma_y, sigma_sat, beta, true_plastic_strain):
    """
    Calculate the true stress using the Voce hardening law.

    Parameters:
    sigma_y (units: Pa): Initial yield stress
        - Physical meaning: This is the stress level at which the material begins to plastically deform.
          It represents the initial resistance to plastic deformation.
    sigma_sat (units: Pa): Saturation stress
        - Physical meaning: This parameter represents the maximum additional stress that can be achieved
          through hardening. It defines the asymptotic value that the stress approaches as the plastic
          strain increases indefinitely.
    beta (dimensionless): Rate parameter
        - Physical meaning: This is a material constant that controls how quickly the stress approaches
          the saturation stress. A higher beta value means that the stress reaches the saturation stress
          more rapidly with increasing plastic strain.
    true_plastic_strain (dimensionless) (float or array-like): True plastic strain
        - Physical meaning: This is the actual plastic strain experienced by the material. It represents
          the amount of irreversible deformation the material has undergone.

    Returns:
    true_stress (units: Pa) (float or array-like)
        - The calculated true stress corresponding to the provided true plastic strain, based on the
          Voce hardening law.
    """
    true_stress = sigma_y + sigma_sat * (1- np.exp(-beta * true_plastic_strain))
    return true_stress

def SwiftVoce_hardening_law(W,K,PEEQ_0,n,sigma_y,sigma_sat,beta,true_plastic_strain):
    """
    Calculate the true stress using the Swift-Voce hardening law.

    This is simply a weighted combination of the Swift and Voce hardening laws
    based on the weight parameter W, with value in range [0, 1]
    If W = 1, the Swift hardening law is used exclusively.
    If W = 0, the Voce hardening law is used exclusively. 
    """
    true_stress_Swift = Swift_hardening_law(K, PEEQ_0, n, true_plastic_strain)
    true_stress_Voce = Voce_hardening_law(sigma_y, sigma_sat, beta, true_plastic_strain)
    true_stress = W * true_stress_Swift + (1 - W) * true_stress_Voce
    return true_stress

# Together, the true plastic strain and true plastic stress made up the flow curve

def calculate_true_stress(parameters_dict, hardening_law, true_plastic_strain, extrapolate_N_first_strain_values = None):
    """
    This function calculates the true stress based on the hardening law and the parameters provided.
    """
    # We assume that parameters is a dictionary
    if hardening_law == "Swift":
        c1, c2, c3 = parameters_dict["c1"], parameters_dict["c2"], parameters_dict["c3"]
        true_stress = Swift_hardening_law(c1, c2, c3, true_plastic_strain.copy())
    elif hardening_law == "Voce":
        c1, c2, c3 = parameters_dict["c1"], parameters_dict["c2"], parameters_dict["c3"]
        true_stress = Voce_hardening_law(c1, c2, c3, true_plastic_strain.copy())
    elif hardening_law == "SwiftVoce":
        c1, c2, c3, c4, c5, c6, c7 = parameters_dict["c1"], parameters_dict["c2"], parameters_dict["c3"], parameters_dict["c4"], parameters_dict["c5"], parameters_dict["c6"], parameters_dict["c7"]
        true_stress = SwiftVoce_hardening_law(c1, c2, c3, c4, c5, c6, c7, true_plastic_strain.copy())
    else:
        raise ValueError("Invalid hardening law provided. Please choose 'Swift', 'Voce', or 'SwiftVoce'.")
    
    if extrapolate_N_first_strain_values is not None:
        after_N_point_true_plastic_strain = true_plastic_strain.copy()[extrapolate_N_first_strain_values:]
        after_N_point_true_stress = true_stress.copy()[extrapolate_N_first_strain_values:]
        
        f = interpolate.interp1d(after_N_point_true_plastic_strain, 
                                 after_N_point_true_stress, 
                                 kind="linear",
                                 fill_value='extrapolate')
        
        true_stress = f(true_plastic_strain)
    
    return true_stress
        
def calculate_true_plastic_strain(strain_start_end, strain_step, saving_path = None):
    """
    This function generates the true plastic strain values based on the provided strain ranges and steps.
    Refer strain_start_end and strain_step to this in the configs/global_config_{project}.json file
    """
    assert strain_start_end[0] == 0.0, "The first value of strain_start_end should be 0.0"
    assert len(strain_start_end) == len(strain_step) + 1, "The length of strain_start_end should be one more than the length of strain_step"
    assert all(strain_start_end[i] < strain_start_end[i+1] for i in range(len(strain_start_end) - 1)), "The strain_start_end values should be in increasing order"
    assert all(strain_step[i] > 0 for i in range(len(strain_step))), "The strain_step values should be positive"
    for i in range(len(strain_step)):
        subtracted = round(strain_start_end[i+1] - strain_start_end[i], 12)
        remainder = round(subtracted / strain_step[i], 12)
        assert (remainder.is_integer()), f"The strain_step value {strain_step[i]} (index {i}) should be divisible by the difference between the corresponding strain_start_end values ({strain_start_end[i+1]} - {strain_start_end[i]} = {subtracted})"
    
    strain_start = np.array(strain_start_end[:-1])
    strain_end = np.array(strain_start_end[1:])
    
    true_plastic_strain = np.array([])

    # Iterate through each range and increment
    for i, (start, end, step) in enumerate(zip(strain_start, strain_end, strain_step)):
        
        strain_range = np.arange(start, end + 1e-12, step)
        strain_range = np.around(strain_range, decimals=12)
        # Append strain_range to strain_array
        if i == 0:
            true_plastic_strain = np.concatenate((true_plastic_strain, strain_range))
        else:
            # Remove the repeated value of previous range at the end  
            true_plastic_strain = np.concatenate((true_plastic_strain, strain_range[1:]))
                
    if saving_path is not None:
        np.save(f"{saving_path}/true_plastic_strain.npy", true_plastic_strain)
    
    return true_plastic_strain


##########################################################################
# Inverse engineering of hardening law parameters using gradient descent #
##########################################################################

# Chain rule: dLoss/d(true stress) * d(true stress)/d(parameter) = dLoss/d(parameter)

def partial_stress_partial_param(parameters_dict, hardening_law, true_plastic_strain):

    # Gradient dimension is 3 for Swift, 3 for Voce, and 7 for SwiftVoce x len(true_plastic_strain)

    if hardening_law == 'Swift':
        PEEQ_0 = parameters_dict['c1']
        K = parameters_dict['c2']
        n = parameters_dict['c3']
        
        # stress = K * (PEEQ_0 + true_plastic_strain)**n
        partials = np.zeros((len(parameters_dict), len(true_plastic_strain)))

        partials[0, :] = K * n * (PEEQ_0 + true_plastic_strain)**(n - 1)  # d_sigma/d_PEEQ_0
        partials[1, :] = (PEEQ_0 + true_plastic_strain)**n
        partials[2, :] = K * (PEEQ_0 + true_plastic_strain)**n * np.log(PEEQ_0 + true_plastic_strain)

        return partials

    elif hardening_law == 'Voce':
        sigma_y = parameters_dict['c1']
        sigma_sat = parameters_dict['c2']
        beta = parameters_dict['c3']
        
        # stress = sigma_y + sigma_sat * (1 - np.exp(-beta * true_plastic_strain))
        partials = np.zeros((len(parameters_dict), len(true_plastic_strain)))

        partials[0, :] = 1
        partials[1, :] = 1 - np.exp(-beta * true_plastic_strain)
        partials[2] = sigma_sat * true_plastic_strain * np.exp(-beta * true_plastic_strain)

        return partials

    elif hardening_law == 'SwiftVoce':
        W = parameters_dict['c1']
        K = parameters_dict['c2']
        PEEQ_0 = parameters_dict['c3']
        n = parameters_dict['c4']
        sigma_y = parameters_dict['c5']
        sigma_sat = parameters_dict['c6']
        beta = parameters_dict['c7']

        # W,K,PEEQ_0,n,sigma_y,sigma_sat,beta,true_plastic_strain
        # stress = W * K * (PEEQ_0 + true_plastic_strain)**n + (1 - W) * (sigma_y + sigma_sat * (1 - np.exp(-beta * true_plastic_strain))
        
        sigma_swift = K * (PEEQ_0 + true_plastic_strain)**n
        sigma_voce = sigma_y + sigma_sat * (1 - np.exp(-beta * true_plastic_strain))

        # Partial derivatives
        partials = np.zeros((len(parameters_dict), len(true_plastic_strain)))
        partials[0, :] = sigma_swift - sigma_voce  # d_sigma/d_W
        partials[1, :] = W * (PEEQ_0 + true_plastic_strain)**n  # d_sigma/d_K
        partials[2, :] = W * K * n * (PEEQ_0 + true_plastic_strain)**(n - 1)  # d_sigma/d_PEEQ_0
        partials[3, :] = W * K * (PEEQ_0 + true_plastic_strain)**n * np.log(PEEQ_0 + true_plastic_strain)  # d_sigma/d_n
        partials[4, :] = 1 - W  # d_sigma/d_sigma_y
        partials[5, :] = (1 - W) * (1 - np.exp(-beta * true_plastic_strain))  # d_sigma/d_sigma_sat
        partials[6, :] = (1 - W) * sigma_sat * true_plastic_strain * np.exp(-beta * true_plastic_strain)  # d_sigma/d_beta
        
        return partials

def partial_RMSE_partial_stress(exp_true_stress, predicted_true_stress):

    # Gradient dimensionis number of points on flow curve, such as 100

    m = len(exp_true_stress)  # Number of data points
    differences = exp_true_stress - predicted_true_stress  # Calculate differences
    squared_differences = differences**2  # Square the differences
    RMSE = np.sqrt(np.mean(squared_differences))  # Calculate RMSE
    
    # Calculate the gradient of RMSE w.r.t. each predicted stress
    if RMSE == 0:
        return np.zeros_like(predicted_true_stress)  # To handle division by zero
    gradient = (-2 / (m * RMSE)) * differences
    
    return gradient


def calculate_inverse_hardening_law_parameters(true_plastic_strain, true_stress, 
                                               hardening_law, param_config, 
                                               RMSE_threshold=1, 
                                               extrapolate_N_first_strain_values=None, max_iter=1000):
    """
    This function inverse engineers the hardening law parameters based on the true plastic strain and true stress values.
    """
    def inverse_hardening_law(parameters, true_plastic_strain, true_stress):
        # We need to turn parameters into a dictionary
        parameters_dict = {}
        for i, param in enumerate(list(param_config.keys())):
            parameters_dict[param] = parameters[i] 
        # print(parameters_dict)
        current_true_stress = calculate_true_stress(parameters_dict, hardening_law, true_plastic_strain, extrapolate_N_first_strain_values = extrapolate_N_first_strain_values)
        RMSE = np.sqrt(np.mean((current_true_stress - true_stress) ** 2))

        return RMSE

    def compute_gradient_RMSE_params(parameters, true_plastic_strain, true_stress):

        """
        Computes the gradient of RMSE with respect to the hardening law parameters.

        Parameters:
        - parameters: Current values of the parameters
        - true_plastic_strain: Array of true plastic strain values.
        - true_stress: Array of true stress values observed/experimental.
        - hardening_law: The type of hardening law being used ('Swift', 'Voce', or 'SwiftVoce').
        - param_config: Configuration dictionary for parameter bounds and scaling.
        """
        
        parameters_dict = {}
        for i, param in enumerate(list(param_config.keys())):
            parameters_dict[param] = parameters[i]

        # Calculate predicted stress using current parameter values
        predicted_stress = calculate_true_stress(parameters_dict, hardening_law, true_plastic_strain)

        # Calculate the gradient of RMSE with respect to stress
        rmse_stress_gradient = partial_RMSE_partial_stress(true_stress, predicted_stress)  # This should be the vector for all data points

        # Calculate the gradient of stress with respect to each parameter
        stress_param_jacobian = partial_stress_partial_param(parameters_dict, hardening_law, true_plastic_strain)  # This should be a matrix
        
        # Compute the gradient of RMSE w.r.t. each parameter
        # Matrix multiplication of Jacobian transpose and the RMSE gradient
        param_gradients = np.dot(stress_param_jacobian, rmse_stress_gradient)  # (num_params x num_stress_points) dot (num_stress_points,) -> (num_params,)

        return param_gradients
    
    # set RMSE to infinity
    RMSE = np.inf
    
    lower_bounds = []
    upper_bounds = []

    for param in param_config:
        lower_bounds.append(param_config[param]["lower"] * param_config[param]["exponent"])
        upper_bounds.append(param_config[param]["upper"] * param_config[param]["exponent"])

    # Convert to numpy arrays
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    bounds_least_squares = (lower_bounds, upper_bounds)
    bounds_minimize = list(zip(lower_bounds, upper_bounds))

    # print(zip_normalized_bounds)
    
    lowest_RMSE = np.inf

    best_inverse_parameters_dict = None

    while RMSE > RMSE_threshold and max_iter > 0:
        num_params = len(param_config)

        initial_params = np.ones(num_params)
        
        for i, param in enumerate(list(param_config.keys())):
            # Choose a random value between the lower and upper bounds
            random_value = np.random.uniform(param_config[param]["lower"] * param_config[param]["exponent"], 
                                             param_config[param]["upper"] * param_config[param]["exponent"])
                                      
            initial_params[i] = random_value
        
        result = minimize(fun=inverse_hardening_law, x0=initial_params, jac=compute_gradient_RMSE_params,
                                bounds = bounds_minimize, method='L-BFGS-B', 
                                args=(true_plastic_strain, true_stress), options={'maxiter': 100000, 'ftol': 1e-15, 'gtol': 1e-15}, 
                                # ftol=1e-15, xtol=1e-15, gtol=1e-15,
                                )
        
        # from scipy.optimize import minimize

        # result = minimize(inverse_hardening_law, initial_params, 
        #                 args=(true_plastic_strain, true_stress), 
        #                 method='Nelder-Mead', bounds=zip_normalized_bounds,
        #                 options={'ftol': 1e-15, 'xtol': 1e-15, 'gtol': 1e-15, 'disp': False})

        inverse_parameters_list = np.array(result.x)
        inverse_parameters_dict = {}
        
        params_names = list(param_config.keys())
        for param in params_names:
            inverse_parameters_dict[param] = inverse_parameters_list[params_names.index(param)]
        inverse_true_stress = calculate_true_stress(inverse_parameters_dict, 
                                                    hardening_law, true_plastic_strain, 
                                                    extrapolate_N_first_strain_values = extrapolate_N_first_strain_values)
        #print(inverse_parameters_dict)
        RMSE = np.sqrt(np.mean((inverse_true_stress - true_stress) ** 2))

        max_iter -= 1
        if RMSE < lowest_RMSE:
            lowest_RMSE = RMSE
            best_inverse_parameters_dict = inverse_parameters_dict
        
        #print(f"Current RMSE: {RMSE}")

    return best_inverse_parameters_dict, lowest_RMSE



# def calculate_inverse_hardening_law_parameters(true_plastic_strain, true_stress, 
#                                                hardening_law, param_config, 
#                                                RMSE_threshold=1, 
#                                                extrapolate_N_first_strain_values=None, max_iter=1000):
#     """
#     This function inverse engineers the hardening law parameters based on the true plastic strain and true stress values.
#     """
#     def inverse_hardening_law(parameters, true_plastic_strain, true_stress):
#         # We need to turn parameters into a dictionary
#         parameters_dict = {}
#         for i, param in enumerate(list(param_config.keys())):
#             parameters_dict[param] = parameters[i] 
#         # print(parameters_dict)
#         current_true_stress = calculate_true_stress(parameters_dict, hardening_law, true_plastic_strain, extrapolate_N_first_strain_values = extrapolate_N_first_strain_values)
#         RMSE = np.sqrt(np.mean((current_true_stress - true_stress) ** 2))

#         return RMSE

#     # set RMSE to infinity
#     RMSE = np.inf
    
#     lower_bounds = []
#     upper_bounds = []

#     for param in param_config:
#         lower_bounds.append(param_config[param]["lower"] * param_config[param]["exponent"])
#         upper_bounds.append(param_config[param]["upper"] * param_config[param]["exponent"])

#     # Convert to numpy arrays
#     lower_bounds = np.array(lower_bounds)
#     upper_bounds = np.array(upper_bounds)
    
#     zip_normalized_bounds = list(zip(lower_bounds, upper_bounds))
#     # print(zip_normalized_bounds)
    
#     lowest_RMSE = np.inf

#     best_inverse_parameters_dict = None

#     while RMSE > RMSE_threshold and max_iter > 0:
#         num_params = len(param_config)

#         c0 = np.ones(num_params)
        
#         for i, param in enumerate(list(param_config.keys())):
#             # Choose a random value between the lower and upper bounds
#             random_value = np.random.uniform(param_config[param]["lower"] * param_config[param]["exponent"], 
#                                              param_config[param]["upper"] * param_config[param]["exponent"])
                                      
#             c0[i] = random_value
        
#         result = least_squares(inverse_hardening_law, c0, 
#                                 bounds = (lower_bounds, upper_bounds),
#                                 args=(true_plastic_strain, true_stress), 
#                                     ftol = 1e-15,
#                                     xtol = 1e-15,
#                                     gtol = 1e-15
#                                 )
        
#         # from scipy.optimize import minimize

#         # result = minimize(inverse_hardening_law, c0, 
#         #                 args=(true_plastic_strain, true_stress), 
#         #                 method='Nelder-Mead', bounds=zip_normalized_bounds,
#         #                 options={'ftol': 1e-15, 'xtol': 1e-15, 'gtol': 1e-15, 'disp': False})

#         inverse_parameters_list = np.array(result.x)
#         inverse_parameters_dict = {}
        
#         params_names = list(param_config.keys())
#         for param in params_names:
#             inverse_parameters_dict[param] = inverse_parameters_list[params_names.index(param)]
#         inverse_true_stress = calculate_true_stress(inverse_parameters_dict, 
#                                                     hardening_law, true_plastic_strain, 
#                                                     extrapolate_N_first_strain_values = extrapolate_N_first_strain_values)
#         #print(inverse_parameters_dict)
#         RMSE = np.sqrt(np.mean((inverse_true_stress - true_stress) ** 2))

#         max_iter -= 1
#         if RMSE < lowest_RMSE:
#             lowest_RMSE = RMSE
#             best_inverse_parameters_dict = inverse_parameters_dict
        
#         #print(f"Current RMSE: {RMSE}")
#     return best_inverse_parameters_dict, lowest_RMSE