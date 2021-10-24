import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from datetime import datetime
import random
import json
import pytz
import os


class Dataset(torch.utils.data.Dataset):

  def __init__(self, inputs, targets, masks):
    self.inputs = inputs
    self.targets = targets
    self.masks = masks

  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    x = self.inputs[index]
    y = self.targets[index]
    m = self.masks[index]
    return x, y, m

def make_copy_repeat_dataset(args):
  if args.data_seed:
    torch.manual_seed(args.data_seed)
    random.seed(args.data_seed)

  def get_random_pattern(length, width):
    pattern = torch.full((length, width), 0.5)
    pattern = torch.bernoulli(pattern)
    pattern[:, -1] = 0
    has_zero_row = pattern.sum(dim=1) == 0
    if any(has_zero_row):
      pattern = get_random_pattern(length, width)
    return pattern

  zero_pad = torch.zeros((1, args.pattern_width))
  false_pad = torch.zeros(1, dtype=torch.bool)
  true_pad = torch.ones(1, dtype=torch.bool)
  end_pad = torch.zeros((1, args.pattern_width))
  end_pad[0, -1] = 1

  inputs = []
  targets = []
  masks = []

  for _ in range(args.num_sequences):
    target_sequence = []
    sequence = []
    mask = []
    seq_length = 0
    for _ in range(args.num_patterns):
      length = random.randint(args.min_pattern_length, args.max_pattern_length)
      repeats = random.randint(args.min_repeats, args.max_repeats)

      pattern = get_random_pattern(length, args.pattern_width)
      zero_pattern = torch.zeros_like(pattern)
      false_pattern = torch.zeros(length, dtype=torch.bool)
      true_pattern = torch.ones(length, dtype=torch.bool)

      sequence.append(pattern)
      sequence.append(end_pad)
      for _ in range(repeats):
        sequence.append(end_pad)
      sequence.append(zero_pattern)
      for _ in range(repeats):
        sequence.append(zero_pattern)

      target_sequence.append(zero_pattern)
      mask.append(false_pattern)
      target_sequence.append(zero_pad)
      mask.append(false_pad)
      for _ in range(repeats):
        target_sequence.append(zero_pad)
        mask.append(false_pad)
      target_sequence.append(pattern)
      mask.append(true_pattern)
      for _ in range(repeats):
        target_sequence.append(pattern)
        mask.append(true_pattern)

    inputs.append(torch.cat(sequence, dim=0))
    targets.append(torch.cat(target_sequence, dim=0))
    masks.append(torch.cat(mask))

  inputs = pad_sequence(inputs, batch_first=True)
  targets = pad_sequence(targets, batch_first=True)
  masks = pad_sequence(masks, batch_first=True)
  return Dataset(inputs, targets, masks)

def make_dirs(args, worker_id=None):
  dirs = {}
  if not args.run_tag:
    tz = pytz.timezone(args.time_zone)
    args.run_tag = datetime.now(tz=tz).strftime("%d-%b-%Y_%H-%M-%S")
  if args.group_tag:
    base_path = os.path.join('runs', args.group_tag, args.run_tag)
  else:
    base_path = os.path.join('runs', args.run_tag)
  dirs['base'] = base_path
  dirs['saves'] = os.path.join(base_path, 'saves')
  dirs['config'] = os.path.join(base_path, 'config')

  if worker_id is not None:
    dirs['worker_saves'] = os.path.join(dirs['saves'], str(worker_id))
    dirs['tensorboard'] = os.path.join(base_path, 'tensorboard', str(worker_id))
  else:
    dirs['worker_saves'] = dirs['saves']
    dirs['tensorboard'] = os.path.join(base_path, 'tensorboard')

  os.makedirs(dirs['tensorboard'], exist_ok=True)
  os.makedirs(dirs['worker_saves'], exist_ok=True)
  os.makedirs(dirs['config'], exist_ok=True)
  path = os.path.join(dirs['config'], 'config.json')
  if not os.path.isfile(path):
    json.dump(vars(args), open(path, 'w'), indent=2)
  return dirs

def get_accuracy(output, target):
  accuracy = (torch.round(output) == target).sum(-1) / target.shape[-1]
  return 100 * accuracy

def add_weight_decay(model, l2_value, skip_list=()):
  decay, no_decay = [], []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
      no_decay.append(param)
    else:
      decay.append(param)
  return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def get_optimizer(controller, args):
  if args.optimizer == 'Adam':
    optimizer = Adam(controller.parameters(), lr=args.lr)
  elif args.optimizer == 'SGD':
    parameters = add_weight_decay(controller, args.weight_decay)
    optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum)
  elif args.optimizer == 'RMSprop':
    parameters = add_weight_decay(controller, args.weight_decay)
    optimizer = RMSprop(parameters, lr=args.lr, momentum=args.momentum)
  elif args.optimizer == 'AdamW':
    parameters = add_weight_decay(controller, args.weight_decay)
    optimizer = AdamW(parameters, lr=args.lr)
  else:
    raise NotImplementedError
  return optimizer

def get_loss_function(args):
  if args.loss_function == 'MSE':
    loss_function = nn.MSELoss(reduction='none')
  else:
    raise NotImplementedError
  return loss_function


class AutoClip():

  def __init__(self, clip_percentile):
    self.grad_history = []
    self.clip_percentile = clip_percentile

  def _get_grad_norm(self, model):
    total_norm = 0
    for p in model.parameters():
      if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

  def clip(self, model):
    grad_norm = self._get_grad_norm(model)
    self.grad_history.append(grad_norm)
    clip_value = np.percentile(self.grad_history, self.clip_percentile)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) 

  def state_dict(self):
    state_dict = {'grad_history': self.grad_history}
    return state_dict

