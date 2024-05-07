using HDF5
using ActiveInference
using DataFrames
using CSV
using ExcelFiles

#-------------------------- ActiveInference.jl & pymdp cross validation --------------------------

file_path_res_julia = "test/pymdp_crossval/data/cross_val_results_data_julia.h5"
file_path_res_python = "test/pymdp_crossval/data/cross_val_results_data_python.h5"

################################
### infer_states() cross_val ###
################################

# ActiveInference.jl
infer_states_julia = array_of_any(3)
for i in 1:3
    infer_states_julia[i] = h5read(file_path_res_julia, "infer_states_$i:3_julia_res")
end

# pymdp
infer_states_python = array_of_any(3)
for i in 1:3
    infer_states_python[i] = h5read(file_path_res_python, "infer_states_$i:3_python_res")
end

# checking to which decimal there is equivalence
# the number 15 is chosen as a starting point based on 
# 64 bits (the conventional bit size) storing 15 decimal places
round_n_infer_states = 15

# function for rounding arrays
function round_arrays(arrays, digits)
    [round.(array, digits=digits) for array in arrays]
end

while round_n_infer_states != 0 && !isequal(infer_states_julia, infer_states_python)

    infer_states_julia = round_arrays(infer_states_julia, round_n_infer_states)
    infer_states_python = round_arrays(infer_states_python, round_n_infer_states)
    round_n_infer_states -= 1
end

infer_states_eq = isequal(infer_states_julia, infer_states_python)
round_n_infer_states

##################################
### infer_policies() cross_val ###
##################################

# ActiveInference.jl
q_pi_julia_res = h5read(file_path_res_julia, "infer_policies_q_pi_julia_res")
G_julia_res = h5read(file_path_res_julia, "infer_policies_G_julia_res")

# pymdp
q_pi_python_res = h5read(file_path_res_python, "infer_policies_q_pi_python_res")
G_python_res = h5read(file_path_res_python, "infer_policies_G_python_res")

# Rounding to check q_pi
round_n_q_pi = 15

while round_n_q_pi != 0 && !isequal(q_pi_julia_res, q_pi_python_res)

    q_pi_julia_res = round_arrays(q_pi_julia_res, round_n_q_pi)
    q_pi_python_res = round_arrays(q_pi_python_res, round_n_q_pi)
    round_n_q_pi -= 1
end

infer_policies_q_pi_eq = isequal(q_pi_julia_res, q_pi_python_res)
round_n_q_pi

# Rounding to check G
round_n_G = 15

while round_n_G != 0 && !isequal(G_julia_res, G_python_res)

    G_julia_res = round_arrays(G_julia_res, round_n_G)
    G_python_res = round_arrays(G_python_res, round_n_G)
    round_n_G -= 1
end

infer_policies_G_eq = isequal(q_pi_julia_res, q_pi_python_res)
round_n_G

#################################
### sample_action() cross_val ###
#################################

# ActiveInference.jl
action_julia_res = h5read(file_path_res_julia, "sample_action_action_julia_res")

# pymdp
action_python_res = h5read(file_path_res_python, "sample_action_action_python_res")

# Note:
# As actions are single integer outputs, we don't want to bother with rounding it,
# as it would technically be equivalent to infinity digits. If one wants a more numeric approach to actions
# look at the previous section for q_pi. Here we will suffice with the isequal() function
sample_action_eq = isequal(action_julia_res, action_python_res)
round_n_sample_action = "is_int"
############################
### update_A() cross_val ###
############################

# ActiveInference.jl
qA_julia_res = array_of_any(4)
for i in 1:4
    qA_julia_res[i] = h5read(file_path_res_julia, "qA_julia_res_$i")
end

A_julia_res = array_of_any(4)
for i in 1:4
    A_julia_res[i] = h5read(file_path_res_julia, "A_julia_res_$i")
end


# pymdp
qA_python_res = array_of_any(4)
for i in 1:4
    qA_python_res[i] = h5read(file_path_res_python, "qA_python_res_$i")
end

A_python_res = array_of_any(4)
for i in 1:4
    A_python_res[i] = h5read(file_path_res_python, "A_python_res_$i")
end

# transposing due to the differences in indexing for python and 3_julia_res
for i in 1:4
    qA_python_res[i] = permutedims(qA_python_res[i], (4, 3, 2, 1))
end

for i in 1:4
    A_python_res[i] = permutedims(A_python_res[i], (4, 3, 2, 1))
end

# Rounding to check qA
round_n_qA = 15

while round_n_qA != 0 && !isequal(qA_julia_res, qA_python_res)

    qA_julia_res = round_arrays(qA_julia_res, round_n_qA)
    qA_python_res = round_arrays(qA_python_res, round_n_qA)
    round_n_qA -= 1
end

update_A_qA_eq = isequal(qA_julia_res, qA_python_res)
round_n_qA

# Rounding to check qA
round_n_A = 15

while round_n_A != 0 && !isequal(A_julia_res, A_python_res)

    A_julia_res = round_arrays(A_julia_res, round_n_A)
    A_python_res = round_arrays(A_python_res, round_n_A)
    round_n_A -= 1
end

update_A_A_eq = isequal(A_julia_res, A_python_res)
round_n_A

############################
### update_B() cross_val ###
############################

# ActiveInference.jl
qB_julia_res = array_of_any(3)
for i in 1:3
    qB_julia_res[i] = h5read(file_path_res_julia, "qB_julia_res_$i")
end

B_julia_res = array_of_any(3)
for i in 1:3
    B_julia_res[i] = h5read(file_path_res_julia, "B_julia_res_$i")
end


# pymdp
qB_python_res = array_of_any(3)
for i in 1:3
    qB_python_res[i] = h5read(file_path_res_python, "qB_python_res_$i")
end

B_python_res = array_of_any(3)
for i in 1:3
    B_python_res[i] = h5read(file_path_res_python, "B_python_res_$i")
end

# transposing due to the differences in indexing for python and 3_julia_res
for i in 1:3
    qB_python_res[i] = permutedims(qB_python_res[i], (3, 2, 1))
end

for i in 1:3
    B_python_res[i] = permutedims(B_python_res[i], (3, 2, 1))
end


# Rounding to check qB
round_n_qB = 15

while round_n_qB != 0 && !isequal(qB_julia_res, qB_python_res)

    qB_julia_res = round_arrays(qB_julia_res, round_n_qB)
    qB_python_res = round_arrays(qB_python_res, round_n_qB)
    round_n_qB -= 1
end

update_B_qB_eq = isequal(qB_julia_res, qB_python_res)
round_n_qB

# Rounding to check qB
round_n_B = 15

while round_n_B != 0 && !isequal(B_julia_res, B_python_res)

    B_julia_res = round_arrays(B_julia_res, round_n_B)
    B_python_res = round_arrays(B_python_res, round_n_B)
    round_n_B -= 1
end

update_B_B_eq = isequal(B_julia_res, B_python_res)
round_n_B

############################
### update_D() cross_val ###
############################

# ActiveInference.jl
qD_julia_res = array_of_any(3)
for i in 1:3
    qD_julia_res[i] = h5read(file_path_res_julia, "qD_julia_res_$i")
end

D_julia_res = array_of_any(3)
for i in 1:3
    D_julia_res[i] = h5read(file_path_res_julia, "D_julia_res_$i")
end


# pymdp
qD_python_res = array_of_any(3)
for i in 1:3
    qD_python_res[i] = h5read(file_path_res_python, "qD_python_res_$i")
end

D_python_res = array_of_any(3)
for i in 1:3
    D_python_res[i] = h5read(file_path_res_python, "D_python_res_$i")
end

# Rounding to check qD
round_n_qD = 15

while round_n_qD != 0 && !isequal(qD_julia_res, qD_python_res)

    qD_julia_res = round_arrays(qD_julia_res, round_n_qD)
    qD_python_res = round_arrays(qD_python_res, round_n_qD)
    round_n_qD -= 1
end

update_D_qD_eq = isequal(qD_julia_res, qD_python_res)
round_n_qD

# Rounding to check qD
round_n_D = 15

while round_n_D != 0 && !isequal(D_julia_res, D_python_res)

    D_julia_res = round_arrays(D_julia_res, round_n_D)
    D_python_res = round_arrays(D_python_res, round_n_D)
    round_n_D -= 1
end

update_D_D_eq = isequal(D_julia_res, D_python_res)
round_n_D

#######################################
### creating dataframe with results ###
#######################################

results_cross_val_df = DataFrame(
    Function = ["infer_states()", "infer_policies()_q_pi", "infer_policies()_G", "sample_action()", "update_A()_qA", "update_A()_A", "update_B()_qB", "update_B()_B", "update_D()_qD", "update_D()_D"],
    is_equal = [infer_states_eq, infer_policies_q_pi_eq, infer_policies_G_eq, sample_action_eq, update_A_qA_eq, update_A_A_eq, update_B_qB_eq, update_B_B_eq, update_D_qD_eq, update_D_D_eq],
    to_decimal_point = [round_n_infer_states, round_n_q_pi, round_n_G, round_n_sample_action, round_n_qA, round_n_A, round_n_qB, round_n_B, round_n_qD, round_n_D]
)

CSV.write("test/pymdp_crossval/cross_validation_&_results/pymdp_cross_val_results.csv", results_cross_val_df)



