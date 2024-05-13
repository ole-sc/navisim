import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import NamedTuple

def draw_target(spins,spin_targets, target_vectors, agent):
    # track how many of the spins that point to a vector are active
    target_vectors = target_vectors.squeeze()
    for i in range(len(target_vectors)):
        s = np.sum(spins[spin_targets == i])
        plt.plot([agent.pos.real, target_vectors[i].real],[agent.pos.imag, target_vectors[i].imag], linewidth = 10*s/len(spins))
    plt.plot(agent.pos.real, agent.pos.imag)

class Parameters(NamedTuple):
    sim_type: str  # indicates, for example what target selection we chose
    init_target_method: str
    n_steps: int
    n_spins: int
    n_targets: int

    T: float
    nu: float

    v: float

    r_eat: float
    r_detect: float = 0 # not used so far
    max_targets: int = 0 # only used when sim_type = "const_targets"

    init_target_scale: float = 100 # only used for init_target_method = "full_env"

    err_dist: float = 0

class Generation():
    '''run multiple instances with a given set of parameters and keep track of results'''
    def __init__(self, params, num_inst) -> None:
        self.params = params
        self.num_inst = num_inst
        self.instances = [Instance(params) for i in range(num_inst)]
        self.scores = np.zeros(self.num_inst)

    def sim_gen(self):
        for i, inst in enumerate(self.instances):
            self.scores[i] = inst.update()

class Instance():
    ''' The Instance controls the environment and the simulation of the agent. '''
    def __init__(self,  params: Parameters) -> None:
        self.params = params
        self.targets = Instance.init_targets(params.n_targets, init_method=params.init_target_method, params = params)

        # depending on what the sim_type is, we do different initialisations 
        if self.params.sim_type == "const_targets":
            # assign the closest #max_targets as the active targets
            self.agent_targets = self.targets[np.argpartition(np.abs(self.targets), self.params.max_targets)[:self.params.max_targets]]
            self.agent = Agent(0+0j, params.n_spins, params.max_targets, params.T, params.nu, v0=params.v)
        else:
            # normally, just use all the created targets
            self.agent_targets = self.targets
            self.agent = Agent(start_pos=0+0j, n_spins=params.n_spins, n_targets=params.n_targets, T=params.T, nu=params.nu, v0=params.v)
        # container for results
        self.agent_history = np.zeros(self.params.n_steps, dtype="complex")
        self.eating_history = np.zeros(1, dtype=complex)

    @staticmethod
    def init_targets(n_targets, init_method, params:Parameters=None):
        ''' targets are represented as complex numbers '''
        targets = np.zeros(n_targets, dtype="complex")
        match init_method:
            case "unit_circle":
                # place targets equidistributed on a circle of radius 1
                angles = np.linspace(0, 2*np.pi, n_targets+1)
                targets = np.exp(1.j*angles[:-1])
            case "half_unit_circle":
                # place targets equidistributed on a half-circle of radius 1
                angles = np.linspace(0, np.pi, n_targets)
                targets = np.exp(1.j*angles)
            case "random_circle":
                # place them randomly on a circle of radius 1
                angles = 2*np.pi*np.random.uniform(size=n_targets)
                targets = np.exp(1.j*angles)
            case "full_env":
                # init them by drawing from a 2D normal distribution 
                targets_real = np.random.normal(loc=0, scale=params.init_target_scale, size=(n_targets,2))
                targets = targets_real[...,0] +1j * targets_real[...,1]
        return targets


    def some_steps(self, i, num_steps):
        '''function for debugging, only runs a given number of steps'''
        for j in range(i, i+num_steps):
            self.update_targets()
            rel_targets = self.agent_targets - self.agent.pos
            self.agent_history[j] = self.agent.sim_step(rel_targets)

            self.agent.spins_discounted = self.discount()

        self.draw(i+num_steps)
        return i+num_steps

    def update(self):
        for i in range(self.params.n_steps):
            # move agent
            self.update_targets()
            rel_targets = self.agent_targets - self.agent.pos
            self.agent_history[i] = self.agent.sim_step(rel_targets)

            # after moving, we want to discount spins
            self.agent.spins_discounted = self.discount()

            # draw call ?!
        return self.agent.score
    
    def discount(self):
        '''discount spins based on how many spins are already pointing in that direction'''
        ## In the code that I found for the paper, it seems like 
        if self.params.err_dist != 0: # setting err_dist to 0 means no discounting is done
            # pdf of a normal distribution
            norm_pdf = lambda theta: 1/(self.params.err_dist*np.sqrt(2*np.pi))*np.exp(-1/2*(theta/self.params.err_dist)**2)
            norm_sum = 0
            disc_prob = np.zeros(self.agent.n_spins)
            for i in range(len(disc_prob)):
                
                angle = np.absolute(np.angle( self.targets[self.agent.spin_targets_idx[i]] - self.agent.pos))
                disc_prob[i] = norm_pdf(angle)
                norm_sum = norm_sum + disc_prob[i]
            disc_prob = disc_prob/norm_sum
            return disc_prob < np.random.rand(self.agent.n_spins)
        else:
            return np.ones(self.agent.n_spins)

            
    def update_targets(self):
        # check if we eat a target
        # find the closest targets
        if self.params.sim_type == "const_targets":
            distances = np.abs(self.targets - self.agent.pos)

            eaten_targets = distances < self.params.r_eat # distances close enough to be eaten
            if np.sum(eaten_targets) >= 1:
                self.eating_history = np.append(self.eating_history, self.targets[eaten_targets])

            self.agent.score = self.agent.score + np.sum(eaten_targets) # increase score by number of eaten targets
            # agent_targets: [t1, t4, t2, t7], targets: [t1, t2, t3,..., tn]
            
            self.targets = self.targets[~eaten_targets] 
            # after removing eaten items, find the closest targets indices
            closest_targets_idx = np.argpartition(np.abs(self.targets-self.agent.pos), self.params.max_targets)[:self.params.max_targets]
            # closest_targets_idx = [2, 7, 15, 12, 19]

            targets_to_reassign = np.ones(len(self.agent_targets),dtype=bool)
            closest_not_assigned = np.ones(len(self.agent_targets), dtype=bool)
            for idx, target in enumerate(self.agent_targets):
                # subtract target position from closest target positions
                closest_distance_idx = np.argmin(np.abs(target-self.targets[closest_targets_idx]))
                if np.abs(target-self.targets[closest_targets_idx[closest_distance_idx]]) < 0.00001:
                    # if it is further away than our tolerance, we need to replace it by one 
                    targets_to_reassign[idx] = 0 # don't need to reassign this
                    closest_not_assigned[closest_distance_idx] = 0
                
            self.agent_targets[targets_to_reassign] = self.targets[closest_targets_idx[closest_not_assigned]]
            # reset spins to zero 


    def draw_trajectory(self):
        '''draw the trajectory of the agents over time and show targets'''
        plt.plot(self.agent_history.real, self.agent_history.imag)
        plt.scatter(self.targets.real, self.targets.imag)
        hist = np.array(self.eating_history)
        plt.scatter(hist.real, hist.imag)

    def draw(self, i):     
        '''draw the agents current direction and indicate the preference for the different targets'''   
        self.agent.draw_agent(self.agent_history[i] - self.agent_history[i-1])
        draw_target(self.agent.spins, self.agent.spin_targets_idx, self.agent_targets, self.agent)
        plt.scatter(self.agent_targets.real, self.agent_targets.imag)

    def draw_env(self):
        plt.scatter(self.agent.pos.real, self.agent.pos.imag)
        plt.scatter(self.targets.real, self.targets.imag)
        plt.scatter(self.agent_targets.real, self.agent_targets.imag)
        plt.show()

class Agent():
    def __init__(self, start_pos,  n_spins, n_targets, T, nu, v0) -> None:

        self.pos = start_pos
        self.vel = v0

        # initialize spins randomly as 0 or 1 and assign targets
        self.n_spins = n_spins
        self.spins = np.random.randint(0, 2, (n_spins))
        # each spin gets a target id
        self.spin_targets_idx = np.sort(np.array([np.mod(i, n_targets) for i in range(n_spins)]))
        # sometimes do not consider a certain spin
        self.spins_discounted = np.ones_like(self.spins)

        self.n_targets = n_targets # either params.max_targets or params.n_targets depending on params.sim_type

        self.nu = nu # tuning parameter
        self.T = T # neural noise

        self.score = 0

    def sim_step(self, targets):
        # these targets are given by the instance and may change. 
        self.flip(targets)

        # update position
            # move in direction of spins (with non-constant velocity)
            #V = self.vel/self.n_spins * np.dot(targets[self.spin_targets_idx], self.spins)#

        V = np.angle(np.dot(targets[self.spin_targets_idx], self.spins*self.spins_discounted))
        self.pos = self.pos + self.vel*np.exp(1j*V)
        return self.pos
    
    def draw_agent(self, V):
        '''plot the agent position and the preferred direction'''
        plt.scatter(self.pos.real, self.pos.imag)
        # plt.plot([self.pos.real, (self.pos + 10*V).real],[self.pos.imag, (self.pos + 10*V).imag])
            
    def flip(self, targets):
        '''Main step of the Metropolis-Hastings inspired algorithm.'''
        # flip a random spin
        flip_idx = np.random.randint(0, self.n_spins)
        self.spins[flip_idx] = not self.spins[flip_idx]
        turned_on = self.spins[flip_idx]
        spin_target = targets[self.spin_targets_idx[flip_idx]] # get the target of this spin
        # calculate the energy difference
            # find angles between target and other targets
        angles = [ np.angle(spin_target) - np.angle(target) for target in targets]
        abs_angle = np.minimum(np.absolute(angles), 2*np.pi - np.absolute(angles))
        Ji = np.cos(np.pi*(abs_angle/np.pi)**self.nu)
        # print(f"Ji:{Ji}")
        # print(self.spins)
        # print(self.spin_targets_idx[self.spins == 1])
        # print(f"Ji:{Ji[self.spin_targets_idx[self.spins == 1]]}")

        dH = -self.n_targets/self.n_spins * (np.sum( Ji[self.spin_targets_idx[self.spins*self.spins_discounted == 1]]) - self.spins[flip_idx]*self.spins_discounted[flip_idx]) # we don't want to add Ji[i]. It will just add 1 if it is 1, so subtract it again at the end
        
        if not turned_on:
            # we calculated dH as if the flipped spin was turned on. If it was turned off, then it was turned on before, so dH
            # is just the negative of what we calculated
            dH = -dH
            
        # if energy is lower than before, keep the flip, else keep it with probability exp(-DH/T)
        if dH >= 0:
            if np.random.rand() >= np.exp(-dH/self.T):
                # flip back
                self.spins[flip_idx] = not self.spins[flip_idx]
        return self.spins
    

if __name__ == "__main__":

    params_main = Parameters(init_target_method="full_env",n_steps=10000, v=0.2, n_spins=60, n_targets=100, T=0.2, nu=0.3, r_eat=3, err_dist=0,
                        max_targets = 6,  init_target_scale=50, sim_type="const_targets")
    inst = Instance(params_main)
    inst.update()

    inst.draw_trajectory()
    print(f"score = {inst.agent.score}")