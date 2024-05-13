import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple

# parameter classes
class Param_Agent(NamedTuple):
    start_pos: torch.Tensor
    start_orientation: float
    vel: float
    
class Param_Instance(NamedTuple):
    param_agent: Param_Agent
    n_steps: int

class Parameter(NamedTuple):
    param_inst: Param_Instance
    n_inst: int


class Generation():
    '''run multiple instances with a given set of parameters and keep track of results'''
    def __init__(self, params:Parameter) -> None:
        self.params = params
        self.num_inst = params.n_inst
        self.instances: list[Instance]

    def sim_gen(self):
        for i, inst in enumerate(self.instances):
            inst.run()
            

class Instance():
    def __init__(self, params: Parameter) -> None:
        self.params = params
        self.agents: list[Agent]
        self.targets = list[Target]

    def run(self):
        for i in range(self.params.n_steps):
            for agent in self.agents:
                agent.step()


class Agent():
    def __init__(self) -> None:
        pass

    def step():
        pass

class Target():
    def __init__(self) -> None:
        pass




