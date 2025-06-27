import pySim_lib as pysim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import subprocess

# ----------------------------------------------------- parameters ---------------------------------------------------
n = 25 # number of files
b = 400
set = "L400-N128"
# --------------------------------------------------------------------------------------------------------------------

h = 0.67
box = b * h

Omega_m = 0.31864557807975047
f = Omega_m ** 0.55
lbox_Mpch = h * 800

# Read in dm particles from Gadget4 binary file
def GadgetReader(filename):
    header = pysim.read_header_gadget4(fname=filename)        
    gadget4 = pysim.read_dark_matter_gadget4(fname=filename, load_pos=True, load_vel=True)

    idx = np.argsort(gadget4['ids'])
    gadget4['ids'] = gadget4['ids'][idx]
    gadget4['pos'] = gadget4['pos'][idx].T
    gadget4['vel'] = gadget4['vel'][idx].T
    return gadget4

#create ascii files for all nbody_0000*.bin files
for i in range(0,n):
    filename = f"{set}_set1_run{i:04}"
    filepath = f'/ptmp/ccorrea/Aletheia-ML/analysis/models/He2019Net_Jamieson2023Loss_dispvel_dispvel_L200N128_set1_i2lpt_dic_tmp/emu_field/{filename}.npy'
    data = np.load(filepath)
    fn_ic = f'/ptmp/ccorrea/Aletheia-ML/L800-N512/set1/ic/runs/output_run{i:04}/snapshot_ics_000'
    ic = GadgetReader(fn_ic)

    pos = data[:3] * h + ic['pos'].reshape(3, 512, 512, 512)
    pos = pos % lbox_Mpch
    vel = data[3:] * f * h * 100
    print(pos.shape, vel.shape) 
    N = np.prod(pos.shape[1:]) 
    position_to_save = pos.reshape(3, N).T
    velocity_to_save = vel.reshape(3, N).T
    data_to_save = np.hstack((position_to_save, velocity_to_save))
    #print(data_to_save.shape)
    np.savetxt(f"/u/chahermann/Sparkling/input/emu_input/{filename}.dat", data_to_save, fmt="%.6f", delimiter = "\t", comments="")

for i in range(0,n):
    filename = f"{set}_set1_emu_run{i:04}"
    particles_data = GadgetReader(f'/ptmp/ccorrea/Aletheia-ML/{set}/set1/nbody/runs/output_run{i:04}/snapshot_000')
    #export to .dat file
    positions = particles_data['pos'].T  # Transpose to get the right shape
    velocities = particles_data['vel'].T
    data_to_save = np.hstack((positions, velocities))
    #print(data_to_save.shape)
    np.savetxt(f"/u/chahermann/Sparkling/input/nbody_input/{filename}.dat", data_to_save, fmt="%.6f", delimiter = "\t", comments="")