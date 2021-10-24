import argparse


def get_args():
  parser = argparse.ArgumentParser()
  patterns = parser.add_argument_group('patterns')
  patterns.add_argument('--num_sequences', type=int, default=2000, 
      help='Amount of sequences in the dataset.')
  patterns.add_argument('--pattern_width', type=int, default=6,
      help='The width of each pattern in a sequence.')
  patterns.add_argument('--min_pattern_length', type=int, default=1,
      help='Amount of sequences in the dataset.')
  patterns.add_argument('--max_pattern_length', type=int, default=8,
      help='Min length of a pattern.')
  patterns.add_argument('--num_patterns', type=int, default=2,
      help='Max length of a pattern.')
  patterns.add_argument('--min_repeats', type=int, default=0,
      help='Min amount of repeats for each pattern.')
  patterns.add_argument('--max_repeats', type=int, default=1,
      help='Max amount of repeats for each pattern.')
  patterns.add_argument('--data_seed', type=int, default=0,
      help='The seed for creating the dataset')

  memory = parser.add_argument_group('memory')
  memory.add_argument('--memory_dim', type=int, default=16,
      help='Amount of columns of the memory matrix.')
  memory.add_argument('--capacity', type=int, default=16,
      help='Amount of rows of the memory matrix.')
  memory.add_argument('--num_reads', type=int, default=4,
      help='Amount of read heads.')
  memory.add_argument('--num_writes', type=int, default=1,
      help='Amount of write heads.')
  memory.add_argument('--free_strengths', action='store_true',
      help='Separate strengths for each memory row.')

  control = parser.add_argument_group('control')
  control.add_argument('--input_dim', type=int, default=6,
      help='The dimension of the inputs to the controller.')
  control.add_argument('--output_dim', type=int, default=6,
      help='The dimension of the outputs from the controller.')

  training = parser.add_argument_group('training')
  training.add_argument('--lr', type=float, default=0.01,
      help='The learning rate.')
  training.add_argument('--batch_size', type=int, default=256,
      help='Amount of sequences per batch.')
  training.add_argument('--training_steps', type=int, default=200000,
      help='Amouint of training steps.')
  training.add_argument('--gpu', action='store_true',
      help='Train on the default GPU.')
  training.add_argument('--loss_function', choices=['MSE'], type=str, default='MSE',
      help='The loss function to use.')
  training.add_argument('--optimizer', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'], type=str, default='Adam',
      help='The optimizer to use.')
  training.add_argument('--momentum', type=float, default=0.9,
      help='The momentum for SGD and RMSprop.')
  training.add_argument('--weight_decay', type=float, default=1e-5,
      help='Amount of weight decay for the optimizer.')
  training.add_argument('--use_mask', action='store_true',
      help='No training targets between patterns.')
  training.add_argument('--seed', type=int, default=0,
      help='The seed for the weight init and data loader.')
  training.add_argument('--time_average_loss', action='store_true',
      help='Average the loss by sequence length.')
  training.add_argument('--clip_grad', type=float, default=None,
      help='Clip the norm of the gradients by this value.')
  training.add_argument('--save_frequency', type=int, default=10,
      help='Save the model at this frequency in training steps.')
  training.add_argument('--debug', action='store_true',
      help='Add the total gradient norm to tensorboard.')
  training.add_argument('--autoclip', action='store_true',
      help='Use adaptive gradient clipping (https://github.com/pseeth/autoclip).')
  training.add_argument('--clip_percentile', type=float, default=10,
      help='The clip percentile for --autoclip.')
  training.add_argument('--mask_weight', type=float, default=0,
      help='Weight to loss for masked targets.')

  pbt_training = parser.add_argument_group('population based training')
  pbt_training.add_argument('--use_pbt', action='store_true',
      help='Enable population based training.')
  pbt_training.add_argument('--pbt_lrs', nargs='+', type=float, default=[],
      help='Learning rates for all individuals.')
  pbt_training.add_argument('--pbt_frequency', type=int, default=1,
      help='Exploit and explore frequency in number of epochs.')
  pbt_training.add_argument('--elitism', action='store_true',
      help='Exploit only the current best individuals.')
  pbt_training.add_argument('--perturbs', nargs=2, type=float, default=[0.9, 1.1],
      help='Perturbation factors for exploration.')

  logging = parser.add_argument_group('logging')
  logging.add_argument('--group_tag', type=str, default='',
      help='A tag for grouping of runs.')
  logging.add_argument('--run_tag', type=str, default='',
      help='A tag to specify the run.')
  logging.add_argument('--time_zone', type=str, default='Europe/Stockholm',
      help='Creates date tags based on this time zone.')

  loader = parser.add_argument_group('loader')
  loader.add_argument('--num_workers', type=int, default=0,
      help='Number of processes for the dataloader.')
  loader.add_argument('--no_shuffle', action='store_true',
      help='Specify to not shuffle the dataset after each epoch.')
  args = parser.parse_args()

  assert args.max_pattern_length >= args.min_pattern_length
  assert args.max_repeats >= args.min_repeats
  return args


