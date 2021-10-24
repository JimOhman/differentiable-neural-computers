from utils import get_optimizer, get_loss_function, make_copy_repeat_dataset, make_dirs, get_accuracy, AutoClip
from torch.utils.tensorboard import SummaryWriter
from core import Controller
from tqdm import tqdm
import numpy as np
import torch
import os


def unroll_through_time(controller, loss_function, inputs, targets, mask, args):
  outputs = []
  loss, accuracy = 0, 0
  time_steps = inputs.shape[1]
  for t in range(time_steps):
    x = inputs[:, t, :]
    y = targets[:, t, :]
    output = controller(x)
    outputs.append(output)
    _loss = loss_function(output, y).mean(-1)
    _accuracy = get_accuracy(output, y)
    if args.use_mask:
      _loss *= mask[:, t]
      _accuracy *= mask[:, t]
    elif args.mask_weight:
      _loss *= (mask[:, t] + args.mask_weight).clamp(0, 1)
    loss += _loss
    accuracy += _accuracy
  if args.use_mask:
    time_steps = mask.sum(-1)
  accuracy = accuracy / time_steps
  if args.time_average_loss:
    loss = loss / time_steps
  return outputs, loss.mean(), accuracy.mean()


class Trainer(object):

  def __init__(self, dataset, args, seed=None, worker_id=None):
    torch.manual_seed(args.seed if seed is None else seed)

    self.args = args
    self.controller = Controller(args)
    self.device = torch.device('cuda' if args.gpu else 'cpu')
    self.controller.to(self.device)
    self.loss_function = get_loss_function(args)
    self.optimizer = get_optimizer(self.controller, args)
    self.dirs = make_dirs(args, worker_id)
    self.logger = SummaryWriter(self.dirs['tensorboard'])
    self.worker_id = worker_id

    if args.autoclip:
      self.auto_clipper = AutoClip(args.clip_percentile)

    loader_params = {'batch_size': args.batch_size,
                     'shuffle': not args.no_shuffle,
                     'num_workers': args.num_workers,
                     'drop_last': True}
    self.dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    self.training_step = 0
    self.num_epochs = 0

  def train_one_epoch(self):
    steps_per_epoch = 0
    epoch_accuracy = 0
    epoch_loss = 0
    for inputs, targets, mask in self.dataloader:
      inputs = inputs.to(self.device)
      targets = targets.to(self.device)
      mask = mask.to(self.device)

      data = unroll_through_time(self.controller, self.loss_function, inputs, targets, mask, self.args)
      _, loss, accuracy = data

      self.optimizer.zero_grad()
      loss.backward()

      if self.args.clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.args.clip_grad)
      elif self.args.autoclip:
        self.auto_clipper.clip(self.controller)

      self.optimizer.step()
      self.controller.memory.reset(self.device)
      self.training_step += 1

      loss = loss.detach().cpu().item()
      accuracy = accuracy.detach().cpu().item()

      epoch_accuracy += accuracy
      epoch_loss += loss
      steps_per_epoch += 1

      self.logger.add_scalar('loss/batch', loss, self.training_step)
      self.logger.add_scalar('accuracy/batch', accuracy, self.training_step)

      if self.training_step % self.args.save_frequency == 0:
        self.save_state()

      if self.args.debug:
        total_grad_norm = 0
        for name, weights in self.controller.named_parameters():
          total_grad_norm += weights.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        self.logger.add_scalar('total_gradient_norm', total_grad_norm, self.training_step)

      if self.training_step >= self.args.training_steps:
        break

    self.num_epochs += 1
    epoch_loss /= steps_per_epoch
    epoch_accuracy /= steps_per_epoch

    self.logger.add_scalar('loss/epoch', epoch_loss, self.num_epochs)
    self.logger.add_scalar('accuracy/epoch', epoch_accuracy, self.num_epochs)
    return epoch_loss, epoch_accuracy, steps_per_epoch

  def train(self):
    self.controller.train()
    with tqdm(total=self.args.training_steps) as pbar:
      while self.training_step < self.args.training_steps:
        loss, accuracy, steps = self.train_one_epoch()
        pbar.set_description('loss: {}, acc: {}%'.format(round(loss, 4),
                                                         round(accuracy, 1)))
        pbar.update(steps)
  
  def save_state(self):
    state = {'dirs': self.dirs,
             'args': self.args,
             'weights': self.controller.get_weights(),
             'optimizer': self.optimizer.state_dict(),
             'training_step': self.training_step}
    if self.args.autoclip:
      state['grad_history'] = self.auto_clipper.grad_history
    path = os.path.join(self.dirs['worker_saves'], str(self.training_step))
    torch.save(state, path)

  def load_state(self, state):
    self.controller.load_state_dict(state['weights'])
    self.optimizer.load_state_dict(state['optimizer'])
    for g in self.optimizer.param_groups:
      g['lr'] = state['args'].lr
    self.args = state['args']
    if self.args.autoclip:
      self.auto_clipper.grad_history = state['grad_history']


if __name__ == '__main__':
  from config import get_args
  from copy import deepcopy

  args = get_args()
  dataset = make_copy_repeat_dataset(args)

  if args.use_pbt:
    from population_based_training import Population, PBTTrainer
    import ray

    os.environ["OMP_NUM_THREADS"] = "1"
    ray.init()

    all_trainer_args = []
    for lr in args.pbt_lrs:
      trainer_args = deepcopy(args)
      trainer_args.lr = lr
      all_trainer_args.append(trainer_args)

    population_size = len(all_trainer_args)
    assert population_size >= 2

    population = Population.remote(size=population_size, elitism=args.elitism)

    trainers = []
    for worker_id, args in enumerate(all_trainer_args):
      trainers.append(PBTTrainer.remote(worker_id, population, dataset, args))
    ray.get([trainer.train.remote() for trainer in trainers])
    ray.shutdown()
  else:
    trainer = Trainer(dataset, args)
    trainer.train()

