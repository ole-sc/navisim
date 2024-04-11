import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
import pickle
import os

def save_data(sim, path):
    with open(path, "wb") as file:
        pickle.dump(sim, file)

        file_size = os.path.getsize(path)
        print(f"Saved {path} ({file_size/1000} kB)")

def load_data(path):
    with open(path, "rb") as file:
        sim = pickle.load(file)
    return sim

class Simulation:
    prng = None # random state for reproducibility
    def __init__(self, params) -> None:
        # create many generations
        self.params = params
        Simulation.prng = RandomState(seed=params["seed"])

        self.num_generations = params["num_generations"]
        self.num_instances = params["num_instances"]
        self.sigmas = self.init_sigmas(self.num_instances)
        
        self.mutation_rate = params["mutation_rate"]

        self.generations = []
        self.all_scores = np.zeros((self.num_generations, self.num_instances))

    def run_sim(self):
        for i in range(self.num_generations):
            print(f"simulating generation {i}...")
            # simulate one generation
            gen = Generation(self.params, self.sigmas)#
            gen.sim_generation()
            self.generations.append(gen)
            self.all_scores[i,:] = gen.scores

            # evolve sigma values
            self.sigmas = self.evolve_sigmas(gen)

    def init_sigmas(self, num_generations):
        # sigmas chosen uniformly in [0,2*pi]
        return Simulation.prng.uniform(0, 2*np.pi, size=num_generations)

    def evolve_sigmas(self, prev_gen):
        # select values for next generations sigma values by selecting higher scoring ones with higher probability 
        # and then adding on a small mutation
        scores = prev_gen.scores
        score_sum = sum(scores)
        probablities = [score/score_sum for score in scores]
        new_sigma_idx = Simulation.prng.choice(a=self.num_instances, size=self.num_instances, p=probablities)
        return np.maximum(np.array(self.sigmas[new_sigma_idx]) + Simulation.prng.uniform(-self.mutation_rate, self.mutation_rate), np.zeros_like(self.sigmas))
        


class Generation:
    def __init__(self, params, sigmas) -> None:
        # create many instances and update the sigma values of the agents
        self.num_instances = params["num_instances"]
        self.sigmas = sigmas
        self.scores = np.zeros(self.num_instances)

        self.instances = [Instance(self.sigmas[i], params) for i in range(self.num_instances)]
        
    def sim_generation(self):
        for i, inst in enumerate(self.instances):
            inst.update()
            self.scores[i] = inst.agent.score

    def plot_scores(self):
        plt.scatter(self.sigmas, self.scores)
        plt.xlabel = "sigmas"
        plt.ylabel = "scores"
        plt.show()

class Instance:
    def __init__(self, sigma_evolved, params) -> None:
        self.sim_steps = params["gen_length"]

        # initialize agent
        self.agent = Agent(sigma_evolved, params)
        self.agent_history = np.zeros((self.sim_steps, 2))

        # place food
        self.num_food = params["num_food"]
        self.normal_scale = params["normal_scale"]
        self.food_items = Simulation.prng.normal(loc=0, scale=self.normal_scale, size=(self.num_food, 2))

    def update(self):
        for i in range(self.sim_steps):
            # move agent
            self.agent.update()
            self.agent_history[i] = self.agent.pos
            # check for food
            distances = np.sqrt(((self.food_items - self.agent.pos)**2).sum(-1)) # calculate distances to food
            self.agent.score = self.agent.score + len(distances[distances < self.agent.detection_radius]) # increase agent score
            self.food_items = self.food_items[distances >= self.agent.detection_radius] # remove food from list

    def show_state(self):
        plt.plot(self.agent_history[:,0], self.agent_history[:,1], color="red")
        plt.scatter(self.food_items[:,0], self.food_items[:,1], s=0.1)
        plt.show()

class Agent:
    def __init__(self, sigma, params) -> None:
        self.sigma = sigma
        self.detection_radius = params["detection_radius"]

        self.pos = np.zeros(2)
        self.rotation = 0
        self.step_length = params["step_length"]

        self.score = 0

    def rotate(self):
        # can be made more advanced
        self.rotation = self.rotation + Simulation.prng.normal(0,self.sigma)

    def update(self):
        self.rotate()
        self.pos = self.pos + self.step_length*np.array([np.cos(self.rotation), np.sin(self.rotation)])

if __name__ == "main":
    num_generations = 5
    num_instances = 50
    gen_lenth = 10000

    mutation_rate = 0.05
    num_food = 1000
    normal_scale = 1000

    params = {"num_generations": num_generations, "num_instances": num_instances, "gen_length": gen_lenth, "mutation_rate": mutation_rate, "num_food": num_food, "normal_scale": normal_scale}
    sim = Simulation(params)
    sim.run_sim()
