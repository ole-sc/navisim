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

def analyze_results(sim, plot = False):
    # return mean, std, maybe mean squared displacement,...
    mean_scores = np.zeros(sim.num_generations) 
    for i, gen in enumerate(sim.generations):
        mean_scores[i] = np.mean(gen.scores)
    if plot:
        plt.boxplot([gen.scores for gen in sim.generations])
        plt.xlabel("generations")
        plt.ylabel("average score")
    return mean_scores  

def plot_sample_trajectories(sim, gen_ids, inst_ids ):
    fig, axs = plt.subplots(len(gen_ids), len(inst_ids), figsize=(10,8))

    for i, gen in enumerate(gen_ids):
        for j, inst in enumerate(inst_ids):
            
            axs[i,j] = sim.generations[gen].instances[inst].show_state(axs[i,j])
            axs[i, j].set_title(f"gen {gen}, inst {inst}")

def plot_sigma_dist(sigmas:np.ndarray):
    '''plot the distribution of sigmas over time.
    in: 2d array of sigma values ~(num_generations, num_instances)'''
    num_fine = 100
    x = np.arange(sigmas.shape[1])
    x_fine = np.linspace(x.min(), x.max(), num_fine)
    y_fine = np.concatenate([np.interp(x_fine, x, y_row) for y_row in sigmas])
    x_fine = np.broadcast_to(x_fine, (sigmas.shape[0], num_fine)).ravel()

    cmap = plt.colormaps["plasma"]
    cmap.with_extremes(bad=cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[25, 50])
    pcm = plt.pcolormesh(xedges, yedges, h.T, cmap=cmap, rasterized=True, vmax=5e1)
    plt.colorbar(pcm, label="# sigmas", pad=0)
    plt.xlabel("generations")
    plt.ylabel("sigmas")
    plt.title("sigma distribution over generations")

    plt.show()

def show_generation_trajectories(gen):
    ''' display a 2d histogram of the positions of the walkers
    in: Generation'''

    # 2d histogram
    traj = np.concatenate([inst.agent_history for inst in gen.instances])
    h, xedges, yedges = np.histogram2d(traj[:,0], traj[:,1], bins=100)
    cmap = plt.colormaps["plasma"]
    cmap.with_extremes(bad=cmap(0))
    pcm = plt.pcolormesh(xedges, yedges, h.T, cmap=cmap, rasterized=True, vmax=1.5e2)
    plt.colorbar(pcm, label="# visits", pad=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trajectories")
    plt.show()


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
        self.all_sigmas = np.zeros((self.num_generations, self.num_instances))

    def run_sim(self):
        for i in range(self.num_generations):
            print(f"simulating generation {i}...")
            # simulate one generation
            gen = Generation(self.params, self.sigmas)#
            gen.sim_generation()
            self.generations.append(gen)
            self.all_scores[i,:] = gen.scores
            self.all_sigmas[i,:] = self.sigmas

            # evolve sigma values
            self.sigmas = self.evolve_sigmas(gen)

    def init_sigmas(self, num_generations):
        # sigmas chosen uniformly in [0,2*pi]
        return Simulation.prng.uniform(0, np.pi, size=num_generations)

    def evolve_sigmas(self, prev_gen):
        # select values for next generations sigma values by selecting higher scoring ones with higher probability 
        # and then adding on a small mutation
        scores = prev_gen.scores
        score_sum = sum(scores)
        probablities = [score/score_sum for score in scores]
        new_sigma_idx = Simulation.prng.choice(a=self.num_instances, size=self.num_instances, p=probablities)
        variations = Simulation.prng.normal(loc=0, scale=self.mutation_rate, size=self.num_instances)
        variations = Simulation.prng.uniform(-self.mutation_rate, self.mutation_rate, size=self.num_instances)
        new_sigma = np.maximum(np.array(self.sigmas[new_sigma_idx]) + variations, np.zeros_like(self.sigmas))
        return new_sigma
        
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

    def show_state(self, ax = None):
        if ax is None:
            ax = plt.subplot()
        ax.plot(self.agent_history[:,0], self.agent_history[:,1], color="red")
        ax.scatter(self.food_items[:,0], self.food_items[:,1], s=0.2)
        return ax

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

if __name__ == "__main__":
    # simulation parameters:
    num_generations = 25 # I think 25 generations should be enough
    num_instances = 100 
    gen_lenth = 1000

    mutation_rate = 0.05 # evolution parameter

    num_food = 100 # fix this at 100 maybe?
    normal_scale = 50

    agent_detection_radius = 10
    agent_step_length = 1 # leave this fixed

    random_seed = np.random.randint(100) # reproducibility

    params = {"num_generations": num_generations, "num_instances": num_instances, "gen_length": gen_lenth, "mutation_rate": mutation_rate, "num_food": num_food, "normal_scale": normal_scale, 
            "detection_radius": agent_detection_radius, "step_length": agent_step_length, "seed": random_seed}
    sim = Simulation(params)
    sim.run_sim()
    plot_sigma_dist(sim.all_sigmas.T)
    show_generation_trajectories(sim.generations[0])
    show_generation_trajectories(sim.generations[-1])   
    plt.plot(analyze_results(sim))
    plt.show()
