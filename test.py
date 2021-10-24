from utils import get_optimizer, get_loss_function, make_copy_repeat_dataset, make_dirs, get_accuracy
from train import unroll_through_time
import matplotlib.pyplot as plt
from core import Controller
from tqdm import tqdm
import numpy as np
import random
import torch
import os
from collections import defaultdict
from matplotlib import animation


def update_memory_state(controller, memory_state):
  memory_state['allocation_gate'].append(controller.memory.gates['allocation_gate'][0])
  memory_state['free_gate'].append(controller.memory.gates['free_gate'][0])
  memory_state['write_weights'].append(controller.memory.write_weights[0].view(-1))
  memory_state['read_weights'].append(controller.memory.read_weights[0].view(-1))
  return memory_state

def init_visualization(inputs, args):
  time_steps = inputs.shape[1]

  init = {}
  init['input and target'] = torch.zeros(args.pattern_width, time_steps)
  init['output'] = torch.zeros(args.pattern_width, time_steps)
  init['free_gate'] = torch.zeros(args.num_reads, time_steps)
  init['allocation_gate'] = torch.zeros(args.num_writes, time_steps)
  init['write_weights'] = torch.zeros(args.num_writes*args.capacity, time_steps)
  init['read_weights'] = torch.zeros(args.num_reads*args.capacity, time_steps)

  images = {}
  def add_subplot(title, gridspec, xmax, vmin=0, vmax=1, cmap='gray', aspect='equal'):
    ax = fig.add_subplot(gridspec)
    ax.set_title(title, fontsize=10, color='white')
    ax.set_xlim(xmin=0, xmax=xmax)
    images[title] = ax.imshow(init[title], cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.grid('off')
    ax.axis('off')

  nwc, nrc = 2*args.num_writes, 2*args.num_reads
  fig = plt.figure(constrained_layout=True, figsize=args.figsize)
  gs = fig.add_gridspec(2 + nwc + nrc, 2, width_ratios=[1, 1])

  xmax = time_steps - 1
  add_subplot('input and target', gs[0, 0], xmax, aspect='auto')
  add_subplot('output', gs[0, 1], xmax, cmap='gist_heat', aspect='auto')
  add_subplot('write_weights', gs[1:1+nwc, 0], xmax, cmap='gist_heat', aspect='auto')
  add_subplot('read_weights', gs[1+nwc:1+nwc+nrc, 0], xmax, cmap='gist_heat', aspect='auto')
  add_subplot('free_gate', gs[1+nwc:1+nwc+nrc, 1], xmax, cmap='gist_heat', aspect='equal')
  add_subplot('allocation_gate', gs[1:1+nwc, 1], xmax, cmap='gist_heat', aspect='equal')

  fig.patch.set_facecolor('black')
  fig.patch.set_alpha(0.8)
  return fig, images

def update_figure(inputs, targets, outputs, mask, images, memory_state, args):
  input_and_target = 0.5*inputs[0].T + targets[0].T
  images['input and target'].set_data(input_and_target)

  outputs = torch.stack(outputs, dim=1)
  if args.use_mask and not args.ignore_mask:
    outputs *= mask[0]
  if args.round:
    outputs = outputs.round()
  images['output'].set_data(outputs)

  allocation_gate = torch.stack(memory_state['allocation_gate'], dim=1)
  free_gate = torch.stack(memory_state['free_gate'], dim=1)
  write_weights = torch.stack(memory_state['write_weights'], dim=1)
  read_weights = torch.stack(memory_state['read_weights'], dim=1)
  images['allocation_gate'].set_data(allocation_gate)
  images['free_gate'].set_data(free_gate)
  images['write_weights'].set_data(write_weights)
  images['read_weights'].set_data(read_weights)

def visualize_training(args):
  states = []
  all_nets = os.listdir(args.saves_dir)
  filtered_nets = [net for net in all_nets if (args.start_net <= int(net) <= args.end_net)]
  for net in sorted(filtered_nets, key=int)[::args.skip+1]:
    path = os.path.join(args.saves_dir, net)
    state = torch.load(path, map_location=torch.device('cpu'))
    state = insert_args(args, state)
    state['args'].batch_size = 1
    states.append(state)

  controller = Controller(state['args'])
  device = torch.device('cpu')

  dataset = make_copy_repeat_dataset(state['args'])
  loader_params = {'batch_size': 1,
                   'shuffle': True,
                   'num_workers': 0,
                   'drop_last': True}
  dataset = torch.utils.data.DataLoader(dataset, **loader_params)

  with torch.inference_mode():
    batch_idx = 0
    for inputs, targets, mask in dataset:
      if args.minimize:
        _, time_steps = torch.nonzero(mask, as_tuple=True)
        time_steps = time_steps.max().item()

        inputs = inputs[:, :time_steps]
        targets = targets[:, :time_steps]
        mask = mask[:, :time_steps]

      fig, images = init_visualization(inputs, state['args'])

      def animate(i):
        state = states[i]
        controller.load_state_dict(state['weights'])

        outputs = []
        memory_state = defaultdict(list)
        for t in range(inputs.shape[1]):
          output = controller(inputs[:, t]).squeeze(0)
          outputs.append(output)
          memory_state = update_memory_state(controller, memory_state)

        update_figure(inputs, targets, outputs, mask, images, memory_state, state['args'])
        controller.memory.reset(device)
        return []

      animation_params = {'frames': len(states),
                          'interval': args.sleep,
                          'blit': True,
                          'repeat': args.repeat}
      anim = animation.FuncAnimation(fig, animate, **animation_params)
      plt.show()

      if args.save_as_gifs:
        gifs_folder = os.path.join(os.path.split(args.saves_dir)[0], 'gifs')
        gif_path = os.path.join(gifs_folder, '{}.gif'.format(batch_idx))
        os.makedirs(gifs_folder, exist_ok=True)
        anim.save(gif_path, writer='imagemagick', fps=60)
      batch_idx += 1

def insert_args(args, state):
  if args.num_patterns is not None:
    state['args'].num_patterns = args.num_patterns
  if args.min_pattern_length is not None:
    state['args'].min_pattern_length = args.min_pattern_length
  if args.max_pattern_length is not None:
    state['args'].max_pattern_length = args.max_pattern_length
  if args.min_repeats is not None:
    state['args'].min_repeats = args.min_repeats
  if args.max_repeats is not None:
    state['args'].max_repeats = args.max_repeats
  if args.num_sequences is not None:
    state['args'].num_sequences = args.num_sequences
  state['args'].batch_size = args.batch_size
  state['args'].ignore_mask = args.ignore_mask
  state['args'].visualize = args.visualize
  state['args'].sleep = args.sleep
  state['args'].figsize = args.figsize
  state['args'].data_seed = args.seed
  state['args'].round = args.round
  assert state['args'].max_pattern_length >= state['args'].min_pattern_length
  assert state['args'].max_repeats >= state['args'].min_repeats
  return state


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  test = parser.add_argument_group('test')
  test.add_argument('--saves_dir', type=str, default='',
      help='The directory where the models are saved.')
  test.add_argument('--net', type=int, default=None,
      help='The filename for the saved model if not --visualize.')
  test.add_argument('--seed', type=int, default=0,
      help='The seed for the data loader.')
  test.add_argument('--batch_size', type=int, default=1,
      help='The amount of sequences per batch.')
  test.add_argument('--gpu', action='store_true',
      help='Run on default GPU if not --visualize.')

  patterns = parser.add_argument_group('patterns')
  patterns.add_argument('--num_sequences', type=int, default=None,
      help='Amount of sequences in the dataset.')
  patterns.add_argument('--min_pattern_length', type=int, default=None,
      help='Amount of sequences in the dataset.')
  patterns.add_argument('--max_pattern_length', type=int, default=None,
      help='Min length of a pattern.')
  patterns.add_argument('--num_patterns', type=int, default=None,
      help='Max length of a pattern.')
  patterns.add_argument('--min_repeats', type=int, default=None,
      help='Min amount of repeats for each pattern.')
  patterns.add_argument('--max_repeats', type=int, default=None,
      help='Max amount of repeats for each pattern.')

  visualize = parser.add_argument_group('visualize')
  visualize.add_argument('--visualize', action='store_true',
      help='Specify to visualize the training process.')
  visualize.add_argument('--ignore_mask', action='store_true',
      help='Ignore mask in visualization if used during training.')
  visualize.add_argument('--save_as_gifs', action='store_true',
      help='Save the visualization as a gif.')
  visualize.add_argument('--start_net', type=int, default=0,
      help='Start visualization from this training step.')
  visualize.add_argument('--end_net', type=int, default=np.inf,
      help='End visualization at this training step.')
  visualize.add_argument('--skip', type=int, default=0,
      help='Skip this amount of saved models during visualization.')
  visualize.add_argument('--round', action='store_true',
      help='Round the output of the network during visualization.')
  visualize.add_argument('--sleep', type=int, default=0,
      help='Sleep this amount in milliseconds with each step.')
  visualize.add_argument('--figsize', nargs=2, type=int, default=None,
      help='Specify the figure size.')
  visualize.add_argument('--minimize', action='store_true',
      help='Remove padding for visualization.')
  visualize.add_argument('--repeat', action='store_true',
      help='Repeat visualization.')
  args = parser.parse_args()

  assert args.start_net <= args.end_net
  if args.net is not None:
    args.start_net = args.net
    args.end_net = args.net

  if not args.visualize:
    path = os.path.join(args.saves_dir, str(args.net))
    state = torch.load(path, map_location=torch.device('cpu'))
    state = insert_args(args, state)
    
    controller = Controller(state['args'])
    controller.load_state_dict(state['weights'])
    device = torch.device('cuda' if args.gpu else 'cpu')
    controller.to(device)

    dataset = make_copy_repeat_dataset(state['args'])
    loader_params = {'batch_size': args.batch_size,
                     'shuffle': True,
                     'num_workers': 0,
                     'drop_last': True}
    dataset = torch.utils.data.DataLoader(dataset, **loader_params)

    loss_function = get_loss_function(state['args'])

    with torch.inference_mode():
      for inputs, targets, mask in dataset:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        info = unroll_through_time(controller, loss_function, inputs, targets, mask, state['args'])
        outputs, loss, accuracy = info

        controller.memory.reset(device)

        loss = loss.cpu().item()
        accuracy = accuracy.cpu().item()
        
        print('loss: {}, accuracy: {}%'.format(round(loss, 4), round(accuracy, 1)))
  else:
    visualize_training(args)

