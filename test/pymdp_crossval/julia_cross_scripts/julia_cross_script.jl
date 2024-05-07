using ActiveInference
using HDF5

# Path to file with generative model
file_path_gm = "test/pymdp_crossval/generative_model_creation/gm_data/A_B_matrices.h5"

#############################################
### Loading Generative Model from h5 file ###
#############################################

# A-matrix
A_cross = array_of_any(4)
for i in 1:4
    A_cross[i] = h5read(file_path_gm, "A_cross_$i")
end

# pA-matrix
pA_cross = array_of_any(4)
for i in 1:4
    pA_cross[i] = h5read(file_path_gm, "pA_cross_$i")
end

# B-matrix
B_cross = array_of_any(3)
for i in 1:3
    B_cross[i] = h5read(file_path_gm, "B_cross_$i")
end

# pB-matrix
pB_cross = array_of_any(3)
for i in 1:3
    pB_cross[i] = h5read(file_path_gm, "pB_cross_$i")
end

# pD-matrix
pD_cross = array_of_any(3)
for i in 1:3
    pD_cross[i] = h5read(file_path_gm, "pD_cross_$i")
end

# observation
obs = h5read(file_path_gm, "obs")

# qs_prev
qs_prev = array_of_any(3)
for i in 1:3
    qs_prev[i] = h5read(file_path_gm, "qs_prev_$i")
end

# action
action = h5read(file_path_gm, "action")

###################################################
### Running agent functions and storing results ###
###################################################

# Specifying the file path for the results
file_path_res = "test/pymdp_crossval/data/cross_val_results_data_julia.h5"

# Creating agent
cross_agent = init_aif(A_cross, B_cross);

#-------------------------- infer_states!() --------------------------

qs_res = infer_states!(cross_agent, obs)

h5write(file_path_res, "infer_states_1:3_julia_res", qs_res[1][1])
h5write(file_path_res, "infer_states_2:3_julia_res", qs_res[1][2])
h5write(file_path_res, "infer_states_3:3_julia_res", qs_res[1][3])

#-------------------------- infer_policies!() --------------------------

q_pi_res, G_res = infer_policies!(cross_agent)

h5write(file_path_res, "infer_policies_q_pi_julia_res", q_pi_res)
h5write(file_path_res, "infer_policies_G_julia_res", G_res)

#-------------------------- sample_action!() --------------------------

# Note:
# for the sample_action!() we use the q_pi_res as the Q_pi of the cross_agent
# It is a bit unnecessary as action sampling relies on the posterior policies,
# but for good measure we include it.

# We set the sampling to deterministic to compare it to the pymdp.
# That it is "deterministic" just means it samples the action (or policy) with the highest posterior probability
cross_agent.action_selection = "deterministic"

action_res = sample_action!(cross_agent)

h5write(file_path_res, "sample_action_action_julia_res", action_res)

#-------------------------- update_A!() --------------------------
# Running update_A()
cross_agent = init_aif(A_cross, B_cross; pA = pA_cross);

qA_res = update_A!(cross_agent, obs)
A_res = cross_agent.A

# Exporting the qA to h5 res file
h5write(file_path_res, "qA_julia_res_1", qA_res[1])
h5write(file_path_res, "qA_julia_res_2", qA_res[2])
h5write(file_path_res, "qA_julia_res_3", qA_res[3])
h5write(file_path_res, "qA_julia_res_4", qA_res[4])

# Exporting updated A to h5 res file
h5write(file_path_res, "A_julia_res_1", A_res[1])
h5write(file_path_res, "A_julia_res_2", A_res[2])
h5write(file_path_res, "A_julia_res_3", A_res[3])
h5write(file_path_res, "A_julia_res_4", A_res[4])

#-------------------------- update_B!() --------------------------
# Running Update_B()
cross_agent = init_aif(A_cross, B_cross; pB = pB_cross);
cross_agent.action = action

qB_res = update_B!(cross_agent, qs_prev)
B_res = cross_agent.B

# Exporting the qB to h5 res file
h5write(file_path_res, "qB_julia_res_1", qB_res[1])
h5write(file_path_res, "qB_julia_res_2", qB_res[2])
h5write(file_path_res, "qB_julia_res_3", qB_res[3])

# Exporting updated B to h5 res file
h5write(file_path_res, "B_julia_res_1", B_res[1])
h5write(file_path_res, "B_julia_res_2", B_res[2])
h5write(file_path_res, "B_julia_res_3", B_res[3])

#-------------------------- update_D!() --------------------------
# Running Update_D()
cross_agent = init_aif(A_cross, B_cross; pD = pD_cross);
qs_t1 = cross_agent.qs_current

qD_res = update_D!(cross_agent, qs_t1)
D_res = cross_agent.D

# Exporting the qD to h5 res file
h5write(file_path_res, "qD_julia_res_1", qD_res[1])
h5write(file_path_res, "qD_julia_res_2", qD_res[2])
h5write(file_path_res, "qD_julia_res_3", qD_res[3])

# Exporting updated D to h5 res file
h5write(file_path_res, "D_julia_res_1", D_res[1])
h5write(file_path_res, "D_julia_res_2", D_res[2])
h5write(file_path_res, "D_julia_res_3", D_res[3])





