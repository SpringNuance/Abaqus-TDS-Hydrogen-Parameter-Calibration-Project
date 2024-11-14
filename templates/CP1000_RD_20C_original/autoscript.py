import os 
import shutil 
# Going to current directory
os.chdir(os.getcwd())

inp_file_name = 'CP1000_diffusion'

# Remove all files that end in .lck in current directory
for file in os.listdir():
    if file.endswith('.lck'):
        os.remove(file)
    # try to remove .odb file. If not possible then pass
    try:
        if file.endswith('.odb'):
            os.remove(file)
    except:
        pass

nstatev = 13
def return_depvar(nstatev):
    DEPVAR = [
        "*Depvar       ",
        f"  {nstatev},      ",       
        "1, AR1_eqplas, AR1_eqplas   ",
        "2, AR2_deqplas, AR2_deqplas   ",
        "3, AR3_C_mol, AR3_C_mol   ",
        "4, AR4_CL_mol, AR4_CL_mol   ",
        "5, AR5_CT_mol, AR5_CT_mol   ",
        "6, AR6_CT_dis_mol, AR6_CT_dis_mol   ",
        "7, AR7_CT_gb_mol, AR7_CT_gb_mol   ",
        "8, AR8_CT_carb_mol, AR8_CT_carb_mol   ",
        "9, AR9_rho_d, AR9_rho_d   ",
        "10, AR10_thetaL, AR10_thetaL   ",
        "11, AR11_thetaT_dis, AR11_thetaT_dis   ",
        "12, AR12_thetaT_gb, AR12_thetaT_gb   ",
        "13, AR13_thetaT_carb, AR13_thetaT_carb   ",
    ]
    return DEPVAR

DEPVAR = return_depvar(nstatev)

nuvarm = 10

# read the input file
with open(inp_file_name + '.inp', 'r') as file:
    lines = file.readlines()

DEPVAR_line = None
# find the line number where the *Depvar is located
for i, line in enumerate(lines):
    if line.startswith('*Depvar'):
        DEPVAR_line = i

# replace the *Depvar line with the new *Depvar line

lines_combined = lines[:DEPVAR_line] + ["\n".join(DEPVAR) + "\n"] + lines[DEPVAR_line+2:]

# write the new input file
with open(inp_file_name + '_post.inp', 'w') as file:
    file.writelines(lines_combined)