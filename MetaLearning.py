import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch import autograd
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.nn.functional as F
import os


class MetaModel(nn.Module):

	def __init__(self, process, world_size, optimizer=optim.Adam, optim_params=(1e-6, (0.9, 0.995)), meta_lr=0.1):
		super(MetaModel, self).__init__()

		if process == 0:
			optim_params = optim_params.insert(0, self.model_parameters())
			self.optimizer = optimizer(*optim_params)

		self.meta_optimizer = optim.SGD(self.model.parameters(), meta_lr)
		self.process = process
		self.world_size = world_size

	def _update_net(self, new_state_dict):
		self.model.load_state_dict(new_state_dict)
		self.model.train()

	def _hook_grads(self, all_grads):
		hooks = []
		for i, v in enumerate(self.model.parameters()):
			def closure():
				ii = i
				return lambda grad: all_grads[ii]
			hooks.append(v.register_hook(closure()))
		return hooks

	def _write_grads(self, new_state_dict, all_grads):
		self._update_net(new_state_dict)
		dummy_loss, = self.model(temp_data_x, temp_data_y)
		hooks = self._hook_grads(all_grads)
		self.optimizer.zero_grad()
		dummy_loss.backward()
		self.optimizer.step()
		for h in hooks:
			h.remove()

	def forward(self, new_state_dict, data, num_updates=5):
		self._update_net(new_state_dict)

		support_x, support_y, query_x, query_y = map(lambda x: x.to(self.device), data)

		for i in range(num_updates):
			loss, pred = self.model(support_x, suppoert_y)
			self.meta_optimizer.zero_grad()
			loss.backward()
			self.meta_optimizer.step()

		loss, pred = self.model(query_x, query_y)
		all_grads = autograd.grad(loss, self.model.parameters(), create_graph=True)

		for idx in range(len(all_grads)):
			dist.all_reduce(all_grads[idx].data, op=dist.ReduceOp.SUM)

		dist.barrier()
		if self.process == 0:
			self._write_grads(new_state_dict, all_grads)


class MetaTrainer:

	def __init__(self, device, world_size):
		self.device=device
		self.world_size = world_size

		self.backend = "gloo"

		if self.device is not "cpu":
			self.meta_learners = [MetaModel(process=process, gpu=self.device[process % len(self.devices)], 
											world_size=num_processes) for process in range(num_processes)]
			self.backend = "nccl"
		
	def init_process(self, process, data, address='localhost', port='29500'):
		os.environ['MASTER_ADDR'] = address
		os.environ['MASTER_PORT'] = port
		dis.init_process_group(self.backend, rank=process, world_size=self.world_size)
		self.meta_learners[process](self.meta_learners[0].model.state_dict(), data)


	# dataloaders is list of the iterators of the dataloaders for each task
	def train(self, data_loaders, optimizer, tb=None, epochs=20, log_interval=100, checkpoint_interval=10000, num_iters=250000):

		# curr_epoch, model, optimizer, scheduler = from_checkpoint_if_exists(model, optimizer, scheduler)
		for epoch in range(epochs):
			
			for iter_ in range(num_iters):
				sampled_tasks = np.random.randint(0, len(data_loaders), self.world_size)
				for process_id, task in enumerate(sampled_tasks):
					# processes = []
					# for process in range(self.world_size):
					processes.append(Process(target=self.init_process, args=(process_idx, next(data_loaders[task]))))
					processes[-1].start()

				for p in processes:
					p.join()


				# data = next()
				# sample_tasks_from_data_loaders
