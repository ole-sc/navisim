import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple

# parameter classes
class Param_Agent(NamedTuple):
    start_pos: torch.Tensor
    start_orientation: float
    vel: float

    n_neurons: int
    
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
    def __init__(self, params: Param_Agent) -> None:
        self.params = params

    def step(self):
        pass

class NeuralAgent(Agent):
    def __init__(self, params: Param_Agent) -> None:
        super().__init__(params)
        self.targets = list[Target]
        self.neurons = list[Neuron]

    def step(self):
        pass

    def visualizeNet(self):
        '''draw the neural network as well as targets and indicate the neuron activations with colour'''
        angles = torch.zeros_like(self.neurons)
        radii = torch.zeros_like(self.neurons)
        for i, neuron in enumerate(self.neurons):
            angles[i] = neuron.pref_angle
            radii[i] = 1 + neuron.activate(self.targets)

        plt.plot()

class Target():
    def __init__(self, target_pos: torch.Tensor, target_id: int) -> None:
        self.pos = target_pos
        self.id = target_id

class Neuron():
    def __init__(self, pref_angle) -> None:
        self.pref_angle = pref_angle
        
        normal_std = 0.5
        self.activation_func = lambda theta: 1/(normal_std*torch.sqrt(2*torch.pi))*torch.exp(-1/2*(theta/normal_std)**2)

    def activate(self, targets: list[Target]):
        # find the closest angles between neuron preferred angle and targets
        base_activation = torch.sum(self.activation_func(self.smallest_angle_to(targets)))
        return base_activation

    def smallest_angle_to(self, targets):
        angles = self.pref_angle - torch.angle(targets)
        return torch.minimum(torch.absolute(angles), 2*torch.pi - torch.absolute(angles))


if __name__ == "__main__":
    pref_angle = torch.pi/4

    num_targets = 4
    targets = torch.tensor([np.exp(1j*2*torch.pi*i/(num_targets+1))for i in range(num_targets)])
    plt.plot(targets.numpy().real, targets.numpy().imag)

    





