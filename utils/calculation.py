import numpy as np
import copy
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def smoothing_force(force, startIndex, endIndex, iter=20000):
    smooth_force = copy.deepcopy(force)
    for i in range(iter):
        smooth_force = savgol_filter(smooth_force[startIndex:endIndex],
            window_length=5,
            polyorder=3,
            mode='interp',
            #mode='nearest',
            #mode='mirror',
            #mode='wrap',
            #mode='constant',
            #deriv=0,
            delta=1
            )
        smooth_force = np.concatenate((force[0:startIndex], smooth_force, force[endIndex:]))
    return smooth_force

def interpolating_FD_curves(FD_curves, interpolated_displacement):
    FD_curves_copy = copy.deepcopy(FD_curves)
    for index, dispforce in enumerate(FD_curves_copy):
        simDisp = dispforce["displacement"]
        simForce = dispforce["force"]
        # Interpolate the force
        FD_curves_copy[index]["force"] = interpolating_force(simDisp, simForce, interpolated_displacement)
        FD_curves_copy[index]["displacement"] = interpolated_displacement
    return FD_curves_copy


def interpolating_force(original_displacement, original_force, interpolated_displacement):
    f = interp1d(original_displacement, original_force, fill_value='extrapolate')
    # Interpolate the displacement
    interpolated_force = f(interpolated_displacement)
    return interpolated_force

def interpolating_stress(original_strain, original_stress, interpolated_strain):
    f = interp1d(original_strain, original_stress, fill_value='extrapolate')
    # Interpolate the stress
    interpolated_stress = f(interpolated_strain)
    return interpolated_stress

def rescale_params_dict(params_dict, param_config):
    rescaled_params_dict = {}
    for param, value in params_dict.items():
        rescaled_params_dict[param] = value * param_config[param]['exponent']
    return rescaled_params_dict

def reverse_as_params_to_objective(curves, objective):
    exampleobjective = objective[0]
    reverseCurves = {}
    for params_tuple in curves[exampleobjective]:
        reverseCurves[params_tuple] = {}
        for objective in objective:
            reverseCurves[params_tuple][objective] = curves[objective][params_tuple]
    return reverseCurves

def find_exp_yielding_index(displacement, force, R_squared_threshold = 0.999):
    # We will fit a line to the data from beginning and gradually increase the number of points
    # As soon as the R_squared value drops smaller than the threshold, we will stop and return the yielding index
    from scipy.stats import linregress
    from sklearn.metrics import r2_score
    yielding_index = 0
    for i in range(2, len(displacement)):
        slope, intercept, r_value, p_value, std_err = linregress(displacement[0:i], force[0:i])
        if r_value**2 < R_squared_threshold:
            yielding_index = i
            break
    
    return yielding_index



