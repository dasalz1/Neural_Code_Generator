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


class Learner(nn.Module):

	def __init__(self, process_id, gpu, src_word_emb, trg_word_emb, trg_word_prj, optimizer=optim.Adam, optimizer_sparse=optim.SparseAdam, optim_params=(1e-6, (0.9, 0.995)), lr=1e-3):
		super(Learner, self).__init__()


		self.model = TransformerParallel()

		if process_id == 0:
			self.optimizer = optimizer(self.model.parameters(), lr=lr)
			self.optimizer_sparse = optimizer_sparse(self.model.parameters(), lr=lr)
		self.meta_optimizer = optim.SGD(self.model.parameters(), 0.1)
		self.device='cuda:'+str(gpu)
		self.embed_device = 'cuda:0'
		self.process_id = process_id
		self.src_word_emb  = src_word_emb
		self.trg_word_emb = trg_word_emb
		self.trg_word_prj = trg_word_prj

		# if process == 0:
			# optim_params = optim_params.insert(0, self.model_parameters())
			# self.optimizer = optimizer(*optim_params)

	def _hook_grads(self, all_grads):
		hooks = []
		for i, v in enumerate(self.model.parameters()):
			def closure():
				ii = i
				return lambda grad: all_grads[ii]
			hooks.append(v.register_hook(closure()))
		return hooks

	def _write_grads(self, original_state_dict, all_grads, temp_data):
		# reload original model before taking meta-gradients
		self.model.load_state_dict(original_state_dict)
		self.model.to(self.device)
		self.model.train()


		self.optimizer.zero_grad()
		self.optimizer_sparse.zero_grad()
		dummy_loss, _ = self._evaluate_model(temp_data)
		hooks = self._hook_grads(all_grads)

		dummy_loss.backward()

		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
		torch.nn.utils.clip_grad_norm_(list(self.trg_word_emb.parameters()) + list(self.src_word_emb.parameters()) + list(self.trg_word_prj.parameters()), 0.1)
		
		self.optimizer.step()
		self.optimizer_sparse.step()

		# gpu memory explodes if you dont remove hooks
		for h in hooks:
			h.remove()

	def _evaluate_model(batch_xs, batch_ys):

		src_mask = (batch_xs != PAD_IDX).unsqueeze(-2).to(self.device)
		src_seq = self.src_word_emb(batch_xs).to(self.device)

		enc_output = self.model(src_seq=src_seq, src_mask = src_mask, module="encoder")

		trg_mask = (get_pad_mask(batch_ys[:, :-1], PAD_IDX) & get_subsequent_mask(batch_ys[:, :-1])).to(self.device)
		trg_seq = trg_word_emb(batch_ys[:, :-1]).to(self.device)

		dec_output = model(enc_output=enc_output, trg_seq=trg_seq, src_mask=src_mask, trg_mask=trg_mask, module="decoder").to(self.embed_device)
		pred_logits = (trg_word_prj(dec_output)*x_logit_scale)

		pred_logits = pred_logits.contiguous().view(-1, pred_logits.size(2))
		loss, n_correct = self.compute_mle_loss(pred_logits, batch_ys[:, 1:], smoothing=True)

		return loss, n_correct

	def forward(self, num_updates, data_queue, data_event, model_queue, process_event):
		while(True):
			data_event.wait()
			data = data_queue.get()
			dist.barrier()
			data_event.clear()

			original_state_dict = {}

			# broadcast weights from master process to all others and save them to a detached dictionary for loadinglater
			for k, v in self.model.state_dict().item():
				if self.process_id == 0:
					original_state_dict[k] = v.clone().detach()
				dist.broadcast(v, src=0, async_op=True)

			self.model.to(self.device)
			self.model.train()

			# meta gradients
			support_x, support_y, query_x, query_y = map(lambda x: x.to(self.embed_device), data)
			for i in range(num_updates):
				self.meta_optimizer.zero_grad()
				loss, _ = self._evaluate_model(support_x, support_y)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
				self.meta_optimizer.step()

			loss, pred = self.model(query_x, query_y)
			all_grads = autograd.grad(loss, self.model.parameters())

			for idx in range(len(all_grads)):
				dist.all_reduce(all_grads[idx].data, op=dist.ReduceOp.SUM, async_op=True)

			if self.process_id == 0:
				self._write_grads(original_state_dict, all_grads, (support_x[0], support_y[0]))
				# finished batch so can load data again from master
				process_event.set()


class MetaTrainer:

	def __init__(self, device, world_size, src_word_emb, trg_word_emb, trg_word_prj, local=True):
		self.device=device
		self.world_size = world_size

		# if self.device is not "cpu":
		self.meta_learners = [Learner(process_id=process_id, gpu=process_id+1, 
											world_size=world_size, src_word_emb, trg_word_emb, trg_word_prj) for process_id in range(num_processes)]
		# gpu backend instead of gloo
		self.backend = "nccl"
		
	def init_process(self, process_id, data_queue, data_event, model_queue, process_event, num_updates, address='localhost', port='29500'):
		os.environ['MASTER_ADDR'] = address
		os.environ['MASTER_PORT'] = port
		dis.init_process_group(self.backend, rank=process, world_size=self.world_size)
		self.meta_learners[process_id](num_updates, data_queue, data_event, model_queue, model_event, process_event)


	# dataloaders is list of the iterators of the dataloaders for each task
	def train(self, data_loaders, optimizer, tb=None, num_updates = 5, epochs=20, log_interval=100, checkpoint_interval=10000, num_iters=250000):
		data_queue = Queue()
		model_queue = Queue()

		# for notifying when to recieve data
		data_event = Event()
		# for notifying this method when to send new data
		process_event = Event()
		# so doesn't hang on first iteration
		process_event.set()
		num_tasks = len(data_loaders)
		
		processes = []
		for process_id in range(self.world_size):
			processes.append(Process(target=self.init_process, args=(process_id, data_queue, data_event, model_queue, process_event, num_updates)))
			processes[-1].start()

		for epoch in range(epochs):
			for num_iter in range(num_iters):
				process_event.wait()
				process_event.clear()
				tasks = np.random.randint(0, num_tasks, (self.world_size))
				for task in tasks:
					# place holder for sampling data from dataset
					data_queue.put(data_loaders[task].__getitem__(0))
				data_event.set()

		new_model = self.meta_learners[0].model.state_dict()

		for p in processes:
			p.join()