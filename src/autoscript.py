import os 
import shutil 
import argparse
import numpy as np
import pandas as pd
import copy

# Going to current directory
os.chdir(os.getcwd())


def return_description_properties(properties_path_excel):
    description_properties_dict = {
        "mechanical_properties": {},
        "hydrogen_diffusion_properties": {},
    }

    # Loading the properties file
    # Cast to string to avoid issues with the mixed types in the excel file
    properties_df = pd.read_excel(properties_path_excel, dtype=str)

    mechanical_descriptions_list = properties_df["mechanical_descriptions"].dropna().tolist()
    mechanical_values_list = properties_df["mechanical_values"].dropna().tolist()

    hydrogen_diffusion_descriptions_list = properties_df["hydrogen_diffusion_descriptions"].dropna().tolist()
    hydrogen_diffusion_values_list = properties_df["hydrogen_diffusion_values"].dropna().tolist()


    ### Now we add the values to the dictionary

    description_properties_dict["mechanical_properties"] = dict(zip(mechanical_descriptions_list, mechanical_values_list))
    description_properties_dict["hydrogen_diffusion_properties"] = dict(zip(hydrogen_diffusion_descriptions_list, hydrogen_diffusion_values_list))

    return description_properties_dict



def return_UMAT_property(description_properties_dict): 
    mechanical_properties_list = list(description_properties_dict["mechanical_properties"].values())
    mechanical_description_list = list(description_properties_dict["mechanical_properties"].keys())

    # Abaqus needs to define 8 properties each line
    mech_prop_num_lines = int(np.ceil(len(mechanical_properties_list)/8))
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
            subset_properties = mechanical_properties_list[line_index*8:(line_index+1)*8]
            subset_description = mechanical_description_list[line_index*8:(line_index+1)*8]
            UMAT_property.append(", ".join(subset_properties))
            UMAT_property.append("** " + ", ".join(subset_description[0:4]))
            UMAT_property.append("** " + ", ".join(subset_description[4:8]))
        else:
            subset_properties = mechanical_properties_list[line_index*8:] + ["0.0"]*(8-len(mechanical_properties_list[line_index*8:]))
            subset_description = mechanical_description_list[line_index*8:] + ["none"]*(8-len(mechanical_description_list[line_index*8:]))
            UMAT_property.append(", ".join(subset_properties))
            UMAT_property.append("** " + ", ".join(subset_description[0:4]))
            UMAT_property.append("** " + ", ".join(subset_description[4:8]))
    
    UMAT_property.append("**")
    UMAT_property.append("*******************************************************")

    return UMAT_property, total_UMAT_num_properties


def return_UMATHT_property(description_properties_dict): 

    hydrogen_diffusion_properties_list = list(description_properties_dict["hydrogen_diffusion_properties"].values())
    hydrogen_diffusion_description_list = list(description_properties_dict["hydrogen_diffusion_properties"].keys())

    # Abaqus needs to define 8 properties each line
    hydrogen_diffusion_prop_num_lines = int(np.ceil(len(hydrogen_diffusion_properties_list)/8))
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
            subset_properties = hydrogen_diffusion_properties_list[line_index*8:(line_index+1)*8]
            subset_description = hydrogen_diffusion_description_list[line_index*8:(line_index+1)*8]
            UMATHT_property.append(", ".join(subset_properties))
            UMATHT_property.append("** " + ", ".join(subset_description[0:4]))
            UMATHT_property.append("** " + ", ".join(subset_description[4:8]))
        else:
            subset_properties = hydrogen_diffusion_properties_list[line_index*8:] + ["0.0"]*(8-len(hydrogen_diffusion_properties_list[line_index*8:]))
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


def constructing_mesh(input_file_path):

    # Open the file
    with open(input_file_path, 'r') as fid:
        flines = fid.readlines()

    # Process the lines
    flines = [line.strip() for line in flines]
    flines_upper = [line.upper() for line in flines]
    start_elements = [i for i, line in enumerate(flines_upper) if '*ELEMENT' in line and '*ELEMENT OUTPUT' not in line]
    start_element_index = start_elements[0]
    element_indices = [] # list of element index
    element_connvectivity = [] # list of of list of nodes that make up the element

    #print("The starting element index is: ", start_element_index)
    #print(start_element_index)

    mesh_index = start_element_index + 1

    while flines_upper[mesh_index][0] != "*" and flines_upper[mesh_index][0] != " ":
    
        # remove all empty spaces and split by comma
        # each line look like this: 1,    35,     2,    36,  2503,  5502,  5503,  5504,  5505
        split_line = flines_upper[mesh_index].replace(" ", "").split(",")
        
        element_indices.append(split_line[0])
        element_connvectivity.append(split_line[1:])
        mesh_index += 1
    
    end_element_index = mesh_index

    # print("The element indices are: ", element_indices)
    # print("The element connectivity are: ", element_connvectivity)

    # Now we would count the number of elements 
    num_elements = len(element_indices)

    original_mesh = []
    for i in range(num_elements):
        reconstructed_line_original = " ".join([str(value) for value in [element_indices[i]] + element_connvectivity[i]])
        original_mesh.append(reconstructed_line_original)
    
    return original_mesh, num_elements, start_element_index, end_element_index
    

def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Process geometry flag.")
    # Add the --geometry flag, expecting a string argument
    parser.add_argument('--input', type=str, required=True, help='input file name')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the geometry argument
    inp_file_name = args.input

    output_simulation_path = os.getcwd()
    combined_original_inp_path = os.path.join(os.getcwd(), f"{inp_file_name}.inp")
    processed_inp_path = os.path.join(os.getcwd(), f"{inp_file_name}_processed.inp")

    ##############################
    # STEP 0: Deleting lck file  #
    ##############################

    # Delete all files ending in .lck in output_simulation_path
    for file in os.listdir(output_simulation_path):
        if file.endswith(".lck"):
            os.remove(os.path.join(output_simulation_path, file))

    #################################################
    # STEP 1: Modifying normal inp to processed inp #
    #################################################

    properties_path_excel = f"processing_input/properties.xlsx"
    flow_curve_excel_path = f"processing_input/flow_curve.xlsx"
    depvar_excel_path = f"processing_input/depvar.xlsx"

    description_properties_dict = return_description_properties(properties_path_excel, flow_curve_excel_path)
    UMAT_PROPERTY, total_UMAT_num_properties = return_UMAT_property(description_properties_dict)
    UMATHT_PROPERTY, total_UMATHT_num_properties = return_UMATHT_property(description_properties_dict)
    DEPVAR, nstatev = return_depvar(depvar_excel_path)

    # print("The UMAT properties are: ", UMAT_PROPERTY)   
    # print("The UMATHT properties are: ", UMATHT_PROPERTY)
    # print("The DEPVAR properties are: ", DEPVAR)
    original_mesh, num_elements, start_element_index, end_element_index =\
        constructing_mesh(combined_original_inp_path)

    # Write the original mesh. We must avoid empty line at the end
    with open("processing_input/original_mesh.txt", 'w') as fid:
        for line_idx in range(len(original_mesh) - 1):
            fid.write(original_mesh[line_idx] + "\n")
        fid.write(original_mesh[-1])

    # Open the input file
    with open(combined_original_inp_path, 'r') as fid:
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

    # find the index of the *Depvar section
    depvar_index = [i for i, line in enumerate(flines_new) if '*DEPVAR' in line.upper()][0]


    #print("The initial conditions are: ", INITIAL_CONDITIONS)

    flines_new = flines_new[:depvar_index] + DEPVAR + flines_new[depvar_index+2:]

    with open(processed_inp_path, 'w') as fid:
        for line in flines_new:
            fid.write(line + "\n")
    

if __name__ == "__main__":
    main()
