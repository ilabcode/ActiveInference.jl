# # Creating the Generative Model

# In this section we will go through the process of creating a generative model and how it should be structured.

# ## The Generative Model Conceptually

# The generative model is the parameters that constitute the agent's beliefs on how the hidden states of the environment generates observations based on states, and how hidden underlying states changes over time.
# In the generative model is also the beliefs of how the agent through actions can influence the states of the environment. Together this holds the buidling blocks that allows for the perception-action loop.

# There are five main buidling blocks of the generative model which are; A, B, C, D, and E.
# Each of these contain parameters that describe the agent's beliefs about the environment.
# We will now go through each of these conecptually one at a time.

# ## A
# A is the observation likelihood model, and describes the agent's beliefs about how the hidden states of the environment generates observations.
# Practically in this package, and other POMDP implemantations as well, this is described through a series of categorical distributions, meaning that for each observation, there is a categorical probability distribution over how likely each hidden state is to generate that observation.
# Let us for example imagine a simple case, where the agent is in a four location state environment, could be a 2x2 gridworld. In this case, there would be one obseration linked to each hidden state, and A then maps the agent's belief of how likely each hidden location state is to generate each observation.
# The agent can then use this belief to infer what state it is in based on the observation it receives. Let's look at an example A, which in this case would be a 4x4 matrix:


# $$
# \begin{array}{c c|c c c c}
#     & & O_1 & O_2 & O_3 & O_4 \\
#     \cline{3-6}
#     \multirow{4}{*}{\left\{\begin{array}{c}
#     \text{Hidden} \\
#     \text{States}
#     \end{array}\right.}
#     & H_1 & 0.85 & 0.05 & 0.05 & 0.05 \\
#     & H_2 & 0.05 & 0.85 & 0.05 & 0.05 \\
#     & H_3 & 0.05 & 0.05 & 0.85 & 0.05 \\
#     & H_4 & 0.05 & 0.05 & 0.05 & 0.85
# \end{array}
# $$
