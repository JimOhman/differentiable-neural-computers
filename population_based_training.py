from train import Trainer
from tqdm import tqdm
import numpy as np
import torch
import ray
import os


@ray.remote
class Population():

  def __init__(self, size, elitism=False):
    self.fitness = [(-np.inf, 0) for _ in range(size)]
    self.elitism = elitism
    self.size = size

  def get_size(self):
    return self.size

  def get_fitness(self):
    return self.fitness

  def set_fitness(self, worker_id, fitness, step):
    old_fitness, _ = self.fitness[worker_id]
    if self.elitism:
      if fitness >= old_fitness:
        self.fitness[worker_id] = (fitness, step)
    else:
      self.fitness[worker_id] = (fitness, step)


@ray.remote
class PBTTrainer(Trainer):

  def __init__(self, worker_id, population, dataset, args):
    seed = worker_id + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    self.population = population
    population_size = ray.get(population.get_size.remote())
    self.opponent_ids = [i for i in range(population_size) if i != worker_id]

    self.best_fitness = -np.inf

    Trainer.__init__(self, dataset, args, seed, worker_id)

  def train(self):
    self.controller.train()
    with tqdm(total=self.args.training_steps) as pbar:
      while self.training_step < self.args.training_steps:
        loss, accuracy, steps = self.train_one_epoch()
        self.save_state()

        fitness = -loss
        self.population.set_fitness.remote(self.worker_id, fitness, self.training_step)

        force_pbt = False
        if fitness > self.best_fitness:
          self.best_fitness = fitness
        elif (self.best_fitness / fitness) < self.args.max_deviation:
          force_pbt = True

        if self.num_epochs % self.args.pbt_frequency == 0 or force_pbt:
          self._exploit_and_explore(force_pbt)

        pbar.set_description('id: {}, lr: {}, loss: {}, acc: {}%'.format(self.worker_id,
                                                                         round(self.args.lr, 4),
                                                                         round(loss, 4),
                                                                         round(accuracy, 1)))
        pbar.update(steps)

  def _exploit_and_explore(self, force):
    winner_state = self._binary_tournament(force)
    if winner_state is not None:
      perturbed_winner_state = self._explore(winner_state)
      self.load_state(perturbed_winner_state)

      new_step = perturbed_winner_state['training_step']
      self.logger.add_scalar('fitness/lr', self.args.lr, self.num_epochs)
      self.logger.add_scalar('fitness/step', new_step, self.num_epochs)

  def _binary_tournament(self, force=False):
    fitnesses = ray.get(self.population.get_fitness.remote())
    
    worker_fitness, worker_steps = fitnesses[self.worker_id]
    self.logger.add_scalar('fitness', worker_fitness, self.num_epochs)

    opponent_id = np.random.choice(self.opponent_ids)
    opponent_fitness, opponent_steps = fitnesses[opponent_id]

    if (worker_fitness > opponent_fitness) and not force:
      return None

    folder_path = os.path.join(self.dirs['saves'], str(opponent_id))
    state_path = os.path.join(folder_path, str(opponent_steps))
    return torch.load(state_path)

  def _explore(self, state):
    perturb = np.random.choice(self.args.perturbs)
    state['args'].lr *= perturb
    return state

