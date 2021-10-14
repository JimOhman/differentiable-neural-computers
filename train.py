from utils import get_optimizer, get_loss_function, make_copy_repeat_dataset, make_dirs, get_accuracy, AutoClip
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
    loss += _loss
    accuracy += _accuracy
  if args.use_mask:
    time_steps = mask.sum(-1)
  accuracy = accuracy / time_steps
  if args.time_average_loss:
    loss = loss / time_steps
  return outputs, loss.mean(), accuracy.mean()


if __name__ == '__main__':
  from torch.utils.tensorboard import SummaryWriter
  import argparse

  parser = argparse.ArgumentParser()
  patterns = parser.add_argument_group('patterns')
  patterns.add_argument('--num_sequences', type=int, default=2000)
  patterns.add_argument('--pattern_width', type=int, default=6)
  patterns.add_argument('--min_pattern_length', type=int, default=1)
  patterns.add_argument('--max_pattern_length', type=int, default=8)
  patterns.add_argument('--num_patterns', type=int, default=2)
  patterns.add_argument('--min_repeats', type=int, default=0)
  patterns.add_argument('--max_repeats', type=int, default=1)

  memory = parser.add_argument_group('control')
  memory.add_argument('--memory_dim', type=int, default=16)
  memory.add_argument('--capacity', type=int, default=16)
  memory.add_argument('--num_reads', type=int, default=4)
  memory.add_argument('--num_writes', type=int, default=1)

  control = parser.add_argument_group('control')
  control.add_argument('--input_dim', type=int, default=6)
  control.add_argument('--output_dim', type=int, default=6)

  training = parser.add_argument_group('training')
  training.add_argument('--lr', type=float, default=0.01)
  training.add_argument('--batch_size', type=int, default=256)
  training.add_argument('--training_steps', type=int, default=200000)
  training.add_argument('--gpu', action='store_true')
  training.add_argument('--loss_function', choices=['MSE', 'SCEL'], type=str, default='MSE')
  training.add_argument('--optimizer', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'], type=str, default='Adam')
  training.add_argument('--momentum', type=float, default=0.9)
  training.add_argument('--use_mask', action='store_true')
  training.add_argument('--train_seed', type=int, default=0)
  training.add_argument('--data_seed', type=int, default=0)
  training.add_argument('--time_average_loss', action='store_true')
  training.add_argument('--clip_grad', type=float, default=0)
  training.add_argument('--save_frequency', type=int, default=500)
  training.add_argument('--debug', action='store_true')
  training.add_argument('--autoclip', action='store_true')
  training.add_argument('--clip_percentile', type=float, default=10)
  training.add_argument('--weight_decay', type=float, default=1e-5)

  logging = parser.add_argument_group('logging')
  logging.add_argument('--group_tag', type=str, default='')
  logging.add_argument('--run_tag', type=str, default='')
  logging.add_argument('--time_zone', type=str, default='Europe/Stockholm')

  loader = parser.add_argument_group('loader')
  loader.add_argument('--num_workers', type=int, default=0)
  loader.add_argument('--no_shuffle', action='store_true')
  args = parser.parse_args()

  training_dataset = make_copy_repeat_dataset(args)
  torch.manual_seed(args.train_seed)

  loader_params = {'batch_size': args.batch_size,
                   'shuffle': not args.no_shuffle,
                   'num_workers': args.num_workers,
                   'drop_last': True}
  training_generator = torch.utils.data.DataLoader(training_dataset, **loader_params)

  controller = Controller(args)
  device = torch.device('cuda' if args.gpu else 'cpu')
  controller.to(device)

  optimizer = get_optimizer(controller, args)
  loss_function = get_loss_function(args)

  if args.autoclip:
    auto_clipper = AutoClip(args.clip_percentile)

  dirs = make_dirs(args)
  summary_writer = SummaryWriter(dirs['tensorboard'])

  training_step = 0
  with tqdm(total = args.training_steps) as pbar:
    while training_step < args.training_steps:
      for inputs, targets, mask in training_generator:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        data = unroll_through_time(controller, loss_function, inputs, targets, mask, args)
        _, loss, accuracy = data

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad:
          torch.nn.utils.clip_grad_norm_(controller.parameters(), args.clip_grad)
        elif args.autoclip:
          auto_clipper.clip(controller)

        optimizer.step()

        controller.memory.reset(device)

        training_step += 1

        loss = loss.detach().cpu().item()
        accuracy = accuracy.detach().cpu().item()

        summary_writer.add_scalar('loss', loss, training_step)
        summary_writer.add_scalar('accuracy', accuracy, training_step)
        pbar.set_description('loss: {}, acc: {}%'.format(round(loss, 4), round(accuracy, 1)))
        pbar.update(1)

        if training_step % args.save_frequency == 0:
          state = {'dirs': dirs,
                   'args': args,
                   'weights': controller.get_weights(),
                   'optimizer': optimizer.state_dict(),
                   'training_step': training_step}
          path = os.path.join(dirs['saves'], str(training_step))
          torch.save(state, path)

        if args.debug:
          total_grad_norm = 0
          for name, weights in controller.named_parameters():
            total_grad_norm += weights.grad.data.norm(2).item() ** 2
          total_grad_norm = total_grad_norm ** (1. / 2)
          summary_writer.add_scalar('total_gradient_norm', total_grad_norm, training_step)

        if training_step >= args.training_steps:
          break

