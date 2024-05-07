import h5py
import numpy as np

from pymdp import utils, maths, learning, control, inference
from pymdp.agent import Agent

# Path to file with generative model
file_path_gm = "test/pymdp_crossval/generative_model_creation/gm_data/A_B_matrices.h5"

#############################################
### Loading Generative Model from h5 file ###
#############################################

# A-matrix
A_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        A_cross[i-1] = file[f"A_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    A_cross[i-1] = np.transpose(A_cross[i-1], (3, 2, 1, 0))

# pA-matrix
pA_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        pA_cross[i-1] = file[f"pA_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    pA_cross[i-1] = np.transpose(pA_cross[i-1], (3, 2, 1, 0))

# B-matrix
B_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        B_cross[i-1] = file[f"B_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    B_cross[i-1] = np.transpose(B_cross[i-1], (2, 1, 0))

# pB-matrix
pB_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        pB_cross[i-1] = file[f"pB_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    pB_cross[i-1] = np.transpose(pB_cross[i-1], (2, 1, 0))

# pD-matrix
pD_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        pD_cross[i-1] = file[f"pD_cross_{i}"][:]

# observation
with h5py.File(file_path_gm, "r") as file:
    obs = file["obs"][:]
obs = obs - 1
obs = list(obs)

# qs_prev
qs_prev = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        qs_prev[i-1] = file[f"qs_prev_{i}"][:]

# action
with h5py.File(file_path_gm, 'r') as file:
        action = file[f"action"][:]
action = action - 1

###################################################
### Running agent functions and storing results ###
###################################################

# Specifying the file path for the results
file_path_res = "test/pymdp_crossval/data/cross_val_results_data_python.h5"

# Creating agent
cross_agent = Agent(A = A_cross, B = B_cross)

#-------------------------- infer_states() --------------------------
qs_res = cross_agent.infer_states(obs)


#-------------------------- infer_policies() --------------------------

q_pi_res, G_res = cross_agent.infer_policies()

#-------------------------- sample_action() --------------------------
# Note:
# for the sample_action() we use the q_pi_res as the Q_pi of the cross_agent
# It is a bit unnecessary as action sampling relies on the posterior policies,
# but for good measure we include it.

# We set the sampling to deterministic to compare it to the ActiveInference.jl
# That it is "deterministic" just means it samples the action (or policy) with the highest posterior probability

# The action selection is by default "deterministic" in pymdp
cross_agent.action_selection

action_res = cross_agent.sample_action()

# To compare it to ActiveInference.jl we add 1 to each entry in this array. This is due to python being 0-indexed
# and julia being 1-indexed
action_res = action_res + 1


#-------------------------- update_A() --------------------------
# Running update_A()
cross_agent = Agent(A = A_cross, B = B_cross, pA = pA_cross)

qA_res = cross_agent.update_A(obs)
A_res = cross_agent.A

#-------------------------- update_B() --------------------------
# Running Update_B()
cross_agent = Agent(A = A_cross, B = B_cross, pB = pB_cross)
cross_agent.action = action

qB_res = cross_agent.update_B(qs_prev)
B_res = cross_agent.B

#-------------------------- update_D!() --------------------------
# Running Update_D()
cross_agent = Agent(A = A_cross, B = B_cross, pD = pD_cross)
qs_t1 = cross_agent.qs

qD_res = cross_agent.update_D(qs_t1)
D_res = cross_agent.D

#-------------------------- Exporting results to h5 file --------------------------

with h5py.File(file_path_res, "w") as hdf:
    
    # qs_res
    hdf.create_dataset("infer_states_1:3_python_res", data=qs_res[0])
    hdf.create_dataset("infer_states_2:3_python_res", data=qs_res[1])
    hdf.create_dataset("infer_states_3:3_python_res", data=qs_res[2])
    
    # q_pi_res and G_res
    hdf.create_dataset("infer_policies_q_pi_python_res", data=q_pi_res)
    hdf.create_dataset("infer_policies_G_python_res", data=G_res)
    
    # action_res
    hdf.create_dataset("sample_action_action_python_res", data=action_res)
    
    # qA_res
    hdf.create_dataset("qA_python_res_1", data=qA_res[0])
    hdf.create_dataset("qA_python_res_2", data=qA_res[1])
    hdf.create_dataset("qA_python_res_3", data=qA_res[2])
    hdf.create_dataset("qA_python_res_4", data=qA_res[3])
    
    # A_res
    hdf.create_dataset("A_python_res_1", data=A_res[0])
    hdf.create_dataset("A_python_res_2", data=A_res[1])
    hdf.create_dataset("A_python_res_3", data=A_res[2])
    hdf.create_dataset("A_python_res_4", data=A_res[3])
    
    # qB_res
    hdf.create_dataset("qB_python_res_1", data=qB_res[0])
    hdf.create_dataset("qB_python_res_2", data=qB_res[1])
    hdf.create_dataset("qB_python_res_3", data=qB_res[2])
    
    # B_res
    hdf.create_dataset("B_python_res_1", data=B_res[0])
    hdf.create_dataset("B_python_res_2", data=B_res[1])
    hdf.create_dataset("B_python_res_3", data=B_res[2])
    
    # qD_res
    hdf.create_dataset("qD_python_res_1", data=qD_res[0])
    hdf.create_dataset("qD_python_res_2", data=qD_res[1])
    hdf.create_dataset("qD_python_res_3", data=qD_res[2])
    
    # D_res
    hdf.create_dataset("D_python_res_1", data=D_res[0])
    hdf.create_dataset("D_python_res_2", data=D_res[1])
    hdf.create_dataset("D_python_res_3", data=D_res[2])