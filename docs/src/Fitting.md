```@meta
EditURL = "../julia_files/Fitting.jl"
```

# Model Fitting

In many cases, we want to be able to draw conclusions about specific observed phenomena, such as behavioural differences between distinct populations. A conventional approach in this context is model fitting, which involves estimating the parameter values of a model (e.g., prior beliefs) that are most likely given the observed behavior of a participant. This approach is often used in fields such as computational psychiatry or mathematical psychology  to develop more precise models and theories of mental processes, to find mechanistic differences between clinical populations, or to investigate the relationship between computational constructs such as Bayesian beliefs and neuronal dynamics.
## Model Fitting with ActionModels.jl

Model fitting in '**ActiveInference**' is mediated through '**ActionModels**', which is our sister package for implementing and fitting various behavioural models to data. The core of '**ActionModels**' is the action model function, which takes a single observation, runs the inference scheme (updating the agent's beliefs), and calculates the probability distribution over actions from which the agent samples its actions.
*(Check out the [ActionModels documentation](https://ilabcode.github.io/ActionModels.jl/dev/markdowns/Introduction/) for more details)*

To demonstrate this, let's define a very simple generative model with a single state factor and two possible actions, and then initialize our active inference object:
```julia

# Define the number of states, observations, and controls
n_states = [4]
n_observations = [4]
n_controls = [2]

# Define the policy length
policy_length = 1

# Use the create_matrix_templates function to create uniform A and B matrices.
A, B = create_matrix_templates(n_states, n_observations, n_controls, policy_length)

# Initialize an active inference object with the created matrices
aif = init_aif(A, B)
```

We can now use the `action_pomdp!` function (which serves as our active inference "action model") to calculate the probability distribution over actions for a single observation:
```julia

# Define observation
observation = [1]

# Calculate action probabilities
action_distribution = action_pomdp!(aif, observation)
```

#### Agent in ActionModels.jl
Another key component of '**ActionModels**' is an `Agent`, which wraps the action model and active inference object in a more abstract structure. The `Agent` is initialized using a `substruct` to include our active inference object, and the action model is our `action_pomdp!` function.

Let's first install '**ActionModels**' from the official Julia registry and import it:

```julia

Pkg.add("ActionModels")
using ActionModels
```

We can now create an `Agent` with the `action_pomdp!` function and the active inference object:

```julia
# Initialize agent with active inference object as substruct
agent = init_agent(
    action_pomdp!,  # The active inference action model
    substruct = aif # The active inference object
)
```
We use an initialized `Agent` primarily for fitting; however, it can also be used with a set of convenience functions to run simulations, which are described in [Simulation with ActionModels](./SimulationActionModels.md).

#### Fitting a Single Subject Model
We have our `Agent` object defined as above. Next, we need to specify priors for the parameters we want to estimate.

For example, let's estimate the action precision parameter `α` and use a Gamma distribution as its prior.

```julia
# Import the Distributions package
using Distributions

# Define the prior distribution for the alpha parameters inside a dictionary
priors = Dict("alpha" => Gamma(1, 1))
```
We can now use the `create_model` function to instantiate a probabilistic model object with data. This function takes the `Agent` object, the priors, and a set of observations and actions as arguments.

First, let's define some observations and actions as vectors:
```julia
# Define observations and actions
observations = [1, 2, 3, 4]
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

