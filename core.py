import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch import jit


class TemporalLinkage(jit.ScriptModule):

  def __init__(self, batch_size, capacity, num_writes):
    super(TemporalLinkage, self).__init__()
    self.batch_size = batch_size
    self.num_writes = num_writes
    self.capacity = capacity

    self.link = torch.zeros((batch_size, num_writes, capacity, capacity))
    self.precedence_weights = torch.zeros((batch_size, num_writes, capacity))
    self.zero_diagonal = 1 - torch.eye(capacity, capacity).view(1, 1, -1, capacity)

    self.strengths_op = nn.Softplus()
    self.epsilon = 1e-6

  def to(self, device):
    super(TemporalLinkage, self).to(device)
    self.link = self.link.to(device)
    self.precedence_weights = self.precedence_weights.to(device)
    self.zero_diagonal = self.zero_diagonal.to(device)

  def reset(self, device):
    self.link = torch.zeros((self.batch_size, self.num_writes, self.capacity, self.capacity), device=device)
    self.precedence_weights = torch.zeros((self.batch_size, self.num_writes, self.capacity), device=device)

  @jit.script_method
  def update_precedence_weights(self, write_weights):
    write_sum = torch.sum(write_weights, 2, keepdim=True)
    self.precedence_weights = (1 - write_sum) * self.precedence_weights + write_weights

  @jit.script_method
  def update_link(self, write_weights):
    write_weights_i = write_weights.unsqueeze(3)
    write_weights_j = write_weights.unsqueeze(2)
    prev_precedence_weights_j = self.precedence_weights.unsqueeze(2)
    new_link = write_weights_i * prev_precedence_weights_j
    prev_link_scale = 1 - write_weights_i - write_weights_j
    self.link = (prev_link_scale * self.link + new_link) * self.zero_diagonal

  @jit.script_method
  def sharpening(self, address, strengths):
    transformed_strengths = self.strengths_op(strengths)
    address = (address + self.epsilon)**(transformed_strengths)
    return address / address.sum(dim=-1, keepdim=True)

  @jit.script_method
  def directional_read_weights(self, prev_read_weights, forward: bool, strengths):
    expanded_read_weights = prev_read_weights.unsqueeze(1).expand(-1, self.num_writes, -1, -1)
    if forward:
      result = torch.matmul(expanded_read_weights, self.link.transpose(-1, -2))
      result = self.sharpening(result, strengths)
    else:
      result = torch.matmul(expanded_read_weights, self.link)
      result = self.sharpening(result, strengths)
    return result.transpose(1, 2)

  @jit.script_method
  def update(self, write_weights):
    self.update_link(write_weights)
    self.update_precedence_weights(write_weights)


class Freeness(jit.ScriptModule):

  def __init__(self, batch_size, capacity):
    super(Freeness, self).__init__()
    self.batch_size = batch_size
    self.capacity = capacity

    self.usage = torch.zeros(batch_size, capacity)

    self.softmax = nn.Softmax(dim=-1)
    self.strengths_op = nn.Softplus()

  def reset(self, device):
    self.usage = torch.zeros(self.batch_size, self.capacity, device=device)

  def to(self, device):
    super(Freeness, self).to(device)
    self.usage = self.usage.to(device)

  @jit.script_method
  def _allocation(self, usage, strengths):
    transformed_strengths = self.strengths_op(strengths)
    sharp_nonusage = (1 - usage) * transformed_strengths
    return self.softmax(sharp_nonusage)
  
  @jit.script_method
  def _usage_after_write(self, write_weights):
    write_weights = 1 - torch.prod(1 - write_weights, 1)
    usage = self.usage + (1 - self.usage) * write_weights
    return usage

  @jit.script_method
  def _usage_after_read(self, free_gate, read_weights):
    free_gate = free_gate.unsqueeze(-1)
    free_read_weights = free_gate * read_weights
    phi = torch.prod(1 - free_read_weights, dim=1)
    usage = self.usage * phi
    return usage, phi
  
  @jit.script_method
  def write_allocation_weights(self, write_gates, num_writes: int, strengths):
    write_gates = write_gates.unsqueeze(-1)
    temp_usage = self.usage
    allocation_weights = []
    for i in range(num_writes):
      allocation_weights.append(self._allocation(temp_usage, strengths))
      temp_usage = temp_usage + ((1 - temp_usage) * write_gates[:, i, :] * allocation_weights[i])
    return torch.stack(allocation_weights, dim=1)

  @jit.script_method
  def update(self, write_weights, free_gate, read_weights):
    self.usage = self._usage_after_write(write_weights)
    self.usage, phi = self._usage_after_read(free_gate, read_weights)
    return phi


class CosineWeights(jit.ScriptModule):

  def __init__(self):
    super(CosineWeights, self).__init__()
    self.softmax = nn.Softmax(dim=-1)
    self.strengths_op = nn.Softplus()
    self.epsilon = 1e-6
  
  @jit.script_method
  def weighted_softmax(self, activations, strengths):
    transformed_strengths = self.strengths_op(strengths)
    sharp_activations = activations * transformed_strengths
    return self.softmax(sharp_activations)

  @jit.script_method
  def forward(self, memory, keys, strengths, mask):
    masked_keys = mask * keys
    masked_memory = torch.einsum('ihk,ijk->ihjk', mask, memory)
    projections = torch.einsum('ihk,ihjk->ihj', masked_keys, masked_memory)
    keys_norm = torch.norm(masked_keys, dim=-1)
    memory_norm = torch.norm(masked_memory, dim=-1)
    norm = torch.einsum('ih,ihj->ihj', keys_norm, memory_norm)
    similarity = projections / (norm + self.epsilon)
    return self.weighted_softmax(similarity, strengths)


class Memory(jit.ScriptModule):

  def __init__(self, batch_size, input_dim, capacity, memory_dim, num_reads=1,
                                                                  num_writes=1,
                                                                  free_strengths=False):
    super(Memory, self).__init__()
    self.batch_size = batch_size
    self.capacity = capacity
    self.memory_dim = memory_dim
    self.num_writes = num_writes
    self.num_reads = num_reads
    self.num_read_modes = 1 + 2 * self.num_writes
    self.strengths_dim = capacity if free_strengths else 1

    self.memories = torch.ones((batch_size, capacity, memory_dim))
    self.write_weights = torch.zeros(batch_size, num_writes, capacity)
    self.read_weights = torch.zeros(batch_size, num_reads, capacity)

    self.linkage = TemporalLinkage(batch_size, capacity, num_writes)
    self.freeness = Freeness(batch_size, capacity)

    self._write_content_weights_mod = CosineWeights()
    self._read_content_weights_mod = CosineWeights()

    self.write_vectors = nn.Linear(input_dim, num_writes * memory_dim)
    self.erase_vectors = nn.Linear(input_dim, num_writes * memory_dim)

    self.free_gate = nn.Linear(input_dim, num_reads)
    self.allocation_gate = nn.Linear(input_dim, num_writes)
    self.write_gate = nn.Linear(input_dim, num_writes)

    self.read_mode = nn.Linear(input_dim, num_reads * self.num_read_modes)
    self.softmax = nn.Softmax(dim=-1)
   
    self.write_keys = nn.Linear(input_dim, num_writes * memory_dim)
    self.write_strengths = nn.Linear(input_dim, num_writes * self.strengths_dim)

    self.read_keys = nn.Linear(input_dim, num_reads * memory_dim)
    self.read_strengths = nn.Linear(input_dim, num_reads * self.strengths_dim)

    self.allocation_strengths = nn.Linear(input_dim, self.strengths_dim)

    self.read_mask = nn.Linear(input_dim, num_reads * memory_dim)
    self.write_mask = nn.Linear(input_dim, num_writes * memory_dim)

    self.mode_strengths = nn.Linear(input_dim, num_writes * num_reads * self.strengths_dim)

    self.gates = {'write_gate': torch.zeros(0),
                  'free_gate': torch.zeros(0),
                  'allocation_gate': torch.zeros(0)}

  def to(self, device):
    super(Memory, self).to(device)
    self.linkage.to(device)
    self.freeness.to(device)
    self.memories = self.memories.to(device)
    self.write_weights = self.write_weights.to(device)
    self.read_weights = self.read_weights.to(device)

  def reset(self, device):
    self.linkage.reset(device)
    self.freeness.reset(device)
    self.memories = torch.ones((self.batch_size, self.capacity, self.memory_dim), device=device)
    self.write_weights = torch.zeros(self.batch_size, self.num_writes, self.capacity, device=device)
    self.read_weights = torch.zeros(self.batch_size, self.num_reads, self.capacity, device=device)

  @jit.script_method
  def _read_inputs(self, inputs):
    write_vectors = self.write_vectors(inputs)
    write_vectors = write_vectors.view(-1, self.batch_size, self.num_writes, self.memory_dim)

    erase_vectors = torch.sigmoid(self.erase_vectors(inputs))
    erase_vectors = erase_vectors.view(-1, self.batch_size, self.num_writes, self.memory_dim)

    free_gate = torch.sigmoid(self.free_gate(inputs))
    free_gate = free_gate.view(-1, self.batch_size, self.num_reads)

    allocation_gate = torch.sigmoid(self.allocation_gate(inputs))
    allocation_gate = allocation_gate.view(-1, self.batch_size, self.num_writes)

    write_gate = torch.sigmoid(self.write_gate(inputs))
    write_gate = write_gate.view(-1, self.batch_size, self.num_writes)

    read_mode = self.read_mode(inputs)
    read_mode = self.softmax(read_mode.view(-1, self.batch_size, self.num_reads, self.num_read_modes))

    write_keys = self.write_keys(inputs)
    write_keys = write_keys.view(-1, self.batch_size, self.num_writes, self.memory_dim)
    write_strengths = self.write_strengths(inputs)
    write_strengths = write_strengths.view(-1, self.batch_size, self.num_writes, self.strengths_dim)

    read_keys = self.read_keys(inputs)
    read_keys = read_keys.view(-1, self.batch_size, self.num_reads, self.memory_dim)
    read_strengths = self.read_strengths(inputs)
    read_strengths = read_strengths.view(-1, self.batch_size, self.num_reads, self.strengths_dim)

    allocation_strengths = self.allocation_strengths(inputs)

    read_mask = torch.sigmoid(self.read_mask(inputs))
    read_mask = read_mask.view(-1, self.batch_size, self.num_reads, self.memory_dim)
    write_mask = torch.sigmoid(self.write_mask(inputs))
    write_mask = write_mask.view(-1, self.batch_size, self.num_writes, self.memory_dim)

    mode_strengths = self.mode_strengths(inputs)
    mode_strengths = mode_strengths.view(-1, self.batch_size, self.num_writes, self.num_reads, self.strengths_dim)

    self.gates = {'write_gate': write_gate,
                  'free_gate': free_gate,
                  'allocation_gate': allocation_gate}

    time_steps = inputs.shape[0]
    result = {'read_content_keys': read_keys,
              'read_content_strengths': read_strengths,
              'write_content_keys': write_keys,
              'write_content_strengths': write_strengths,
              'write_vectors': write_vectors,
              'erase_vectors': erase_vectors,
              'free_gate': free_gate,
              'allocation_gate': allocation_gate,
              'write_gate': write_gate,
              'read_mode': read_mode,
              'allocation_strengths': allocation_strengths,
              'write_mask': write_mask,
              'read_mask': read_mask,
              'mode_strengths': mode_strengths}
    return result, time_steps

  @jit.script_method
  def update_write_weights(self, write_content_keys, 
                                 write_content_strengths, 
                                 write_mask,
                                 allocation_gate,
                                 allocation_strengths,
                                 write_gate):

    write_content_weights = self._write_content_weights_mod(self.memories, 
                                                            write_content_keys,
                                                            write_content_strengths,
                                                            write_mask)

    combined_gate = allocation_gate * write_gate
    write_allocation_weights = self.freeness.write_allocation_weights(combined_gate,
                                                                      self.num_writes,
                                                                      allocation_strengths)

    allocation_gate = allocation_gate.unsqueeze(-1)
    write_gate = write_gate.unsqueeze(-1)

    allocation = (allocation_gate * write_allocation_weights)
    content = (1 - allocation_gate) * write_content_weights

    self.write_weights = write_gate * (allocation + content)

  @jit.script_method
  def update_read_weights(self, read_content_keys,
                                read_content_strengths,
                                read_mask,
                                read_mode,
                                mode_strengths):

    content_weights = self._read_content_weights_mod(self.memories,
                                                     read_content_keys,
                                                     read_content_strengths,
                                                     read_mask)

    backward_weights = self.linkage.directional_read_weights(self.read_weights, 
                                                             forward=False,
                                                             strengths=mode_strengths)

    forward_weights = self.linkage.directional_read_weights(self.read_weights,
                                                            forward=True,
                                                            strengths=mode_strengths)

    backward_mode = read_mode[..., :self.num_writes]
    forward_mode = read_mode[..., self.num_writes:2*self.num_writes]
    content_mode = read_mode[..., 2*self.num_writes]

    backward = torch.sum(backward_mode.unsqueeze(3) * backward_weights, 2)
    forward = torch.sum(forward_mode.unsqueeze(3) * forward_weights, 2)
    content = content_mode.unsqueeze(2) * content_weights

    self.read_weights = content + forward + backward

  @jit.script_method
  def _erase_and_write(self, erase_vectors, write_vectors, phi):
    write_weights = self.write_weights.unsqueeze(3)
    erase_vectors = erase_vectors.unsqueeze(2)

    weighted_resets = write_weights * erase_vectors
    reset_gate = torch.prod(1 - weighted_resets, dim=1)
    full_reset = phi.unsqueeze(-1) * reset_gate

    add_matrix = torch.matmul(self.write_weights.transpose(-1, -2), write_vectors)
    self.memories = full_reset * self.memories + add_matrix

  @jit.script_method
  def forward(self, inputs):
    inputs, time_steps = self._read_inputs(inputs)
    memory_outputs = []
    for t in range(time_steps):
      phi = self.freeness.update(self.write_weights, inputs['free_gate'][t], self.read_weights)

      self.update_write_weights(inputs['write_content_keys'][t],
                                inputs['write_content_strengths'][t],
                                inputs['write_mask'][t],
                                inputs['allocation_gate'][t],
                                inputs['allocation_strengths'][t],
                                inputs['write_gate'][t])

      self._erase_and_write(inputs['erase_vectors'][t], 
                            inputs['write_vectors'][t],
                            phi)

      self.linkage.update(self.write_weights)

      self.update_read_weights(inputs['read_content_keys'][t],
                               inputs['read_content_strengths'][t],
                               inputs['read_mask'][t],
                               inputs['read_mode'][t],
                               inputs['mode_strengths'][t])

      memory_output = torch.matmul(self.read_weights, self.memories)
      memory_outputs.append(memory_output)
    return torch.stack(memory_outputs, dim=0)


class Controller(nn.Module):

  def __init__(self, args):
    super(Controller, self).__init__()
    self.batch_size = args.batch_size

    self.memory = Memory(args.batch_size, 
                         args.input_dim, 
                         args.capacity, 
                         args.memory_dim, 
                         args.num_reads, 
                         args.num_writes,
                         args.free_strengths)

    self.fc_input_dim = args.num_reads * args.memory_dim
    self.fc_output_dim = args.output_dim
    self.fc = nn.Linear(self.fc_input_dim, self.fc_output_dim)

  def to(self, device):
    super(Controller, self).to(device)
    self.memory.to(device)
  
  def forward(self, x):
    memory_outputs = self.memory(x).view(-1, self.fc_input_dim)
    output = torch.sigmoid(self.fc(memory_outputs))
    return output.view(-1, self.batch_size, self.fc_output_dim)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}


