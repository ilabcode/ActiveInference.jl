# ActiveInference.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://samuelnehrer02.github.io/ActiveInference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://samuelnehrer02.github.io/ActiveInference.jl/dev/)
[![Build Status](https://github.com/samuelnehrer02/ActiveInference.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/samuelnehrer02/ActiveInference.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/samuelnehrer02/ActiveInference.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/samuelnehrer02/ActiveInference.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Julia Package for Active Inference.
ActiveInference.jl is a new Julia package designed to implement active inference within discrete state-spaces using Partially Observable Markov Decision Processes (POMDP's). With ActiveInference.jl you can define a generative model for active inference to run agent-based simulations. Additionally, we provide a functionality for fitting experimental data for parameter recovery. 

![Maze Animation](.github/animation_maze.gif)
* Example visualization of an agent navigating a maze, inspired by the one described in [Bruineberg et al., 2018](https://www.sciencedirect.com/science/article/pii/S0022519318303151?via%3Dihub)
Left: A synthetic agent wants to reach the end of the maze environment while avoiding dark-colored locations.
Right: The agent's noisy prior expectations about the state of the environment parameterized by Dirichlet distributions being updated dynamically as it moves through the maze.*
## Getting Started 

