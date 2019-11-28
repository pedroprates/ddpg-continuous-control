# DDPG Algorithm - Continuous Control

This work is part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), which consists on solving the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. 

![alt-text](images/reacher.gif)

## Set Up

To setup your local Python environment for running Unity environments checkout the instructions on [this Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). On this work we'll use [PyTorch](https://pytorch.org/) to build the networks. On `requirements.txt` you'll also find some other packages required.

## Environment

You'll need to download the Unity environment built for this assigment. There are two options: the first one, with **one** agent, and the second one with multiple agents, **20** agents, actually. You should download it and keep it on an `env` folder, creating a `single` or `multiple` folder inside to store the selected environment. 

### Version 1: Single Agent

* Linux: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* MacOS: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Version 2: Twenty Agents

* Linux: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* MacOS: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Development

You should follow the `report.ipynb` to check how the solution was implemented. On the `models/` folder there are all the model files that was used, and on `utils/` folder there are the support files, such as implementation of the `noise` and `Experience Replay`. On the `agent.py` file the **main agent** is implemented, which is responsible to create and train the `Actor` and `Critic` networks. 