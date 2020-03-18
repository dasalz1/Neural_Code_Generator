import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformer.Models import Transformer
from torch import nn
from DataClass.MetaTorchData import *
from torch import optim
from torch import autograd
from torch.multiprocessing import Process, Queue
from multiprocessing import Event
from transformer.bart import BartModel
from DataClass.Constants import PAD_IDX, UNKNOWN_WORD
import numpy as np
import pandas as pd
from train_utils import save_checkpoint, from_checkpoint_if_exists, tb_mle_meta_batch
from transformers import AdamW, get_cosine_schedule_with_warmup
import os
from copy import deepcopy
from tqdm import tqdm


D_WORD_VEC = 128

class FTBart(nn.Module):
	def __init__(self, bart_model):
		super(FTBart, self).__init__()
		self.bart_model = bart_model
		for param in self.bart_model.parameters():
			param.requires_grad = False
		
		for param in self.bart_model.decoder.layers.parameters():
			param.requires_grad=True

		# self.meta_proj1 = self.bart_model.decoder_proj.clone().detach()
		# self.final_proj = self.bart_model.decoder_proj.clone().detach()
		# self.meta_proj1 = nn.Linear(VOCAB_SIZE, D_WORD_VEC)
		# self.final_proj = nn.Linear(VOCAB_SIZE, VOCAB_SIZE)

		# self.meta_proj1.weight.data = self.bart_model.decoder_proj.weight.data.T
		# self.final_proj.weight.data = self.bart_model.decoder_proj.weight.data

	def forward(self, input_ids, decoder_input_ids):
		x = self.bart_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
		# x = x.contiguous().view(-1, VOCAB_SIZE)
		# x = self.meta_proj1(x)
		# out = self.final_proj(x)
		return x

	def parameters(self):
		# return list(self.meta_proj1.parameters()) + list(self.meta_proj2.parameters()) + list(self.final_proj.parameters())
		# return list(self.meta_proj1.parameters()) + list(self.final_proj.parameters())
		return self.bart_model.decoder.layers.parameters()




class Learner(nn.Module):

	def __init__(self, process_id, gpu='cpu', world_size=4, optimizer=AdamW, optimizer_sparse=optim.SparseAdam, optim_params=(1e-5,), model_params=None, num_iters=100000, load_model=False, fine_tune=False):
		super(Learner, self).__init__()
		model = BartModel(model_params)
		if load_model:
			params = torch.load('../checkpoint-bigseq2seqnaive.pth')['model']
			k, v = zip(*params.items())
			for k, v in zip(k, v):
				params[k[7:]] = v
				del params[k]
			model.load_state_dict(params)
			self.model = model
			optim_params = (1e-5,)
		elif fine_tune:
			params = torch.load('../checkpoint-bigseq2seqnaive.pth')['model']
			k, v = zip(*params.items())
			for k, v in zip(k, v):
				params[k[7:]] = v
				del params[k]
			model.load_state_dict(params)
			self.model = FTBart(model)
			optim_params = (1e-5,)

		else:
			self.model = model

		if process_id == 0:
			optim_params = (self.model.parameters(),) + optim_params
			self.optimizer = optimizer(*optim_params)
			# os.nice(-19)

		self.meta_optimizer = optim.SGD(self.model.parameters(), 1e-4)
		self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
		self.model.to(self.device)
		self.process_id = process_id
		self.num_iter = 0
		self.world_size = world_size
		self.original_state_dict = {}

	def _write_prediction(self, support_x, pred_logits, query_y):
		support_x_pred = pd.DataFrame(support_x[:, 1:].to('cpu').numpy())
		support_x_pred = np.where(support_x_pred.isin(idx2word.keys()), support_x_pred.replace(idx2word), UNKNOWN_WORD)
		pred_max = pred_logits.reshape(-1, MAX_LINE_LENGTH-1, len(idx2word)).max(2)[1]
		pred = pd.DataFrame(pred_max.to('cpu').numpy())
		pred_words = np.where(pred.isin(idx2word.keys()), pred.replace(idx2word), UNKNOWN_WORD)
		trg_ys = pd.DataFrame(query_y[:, 1:].to('cpu').numpy())
		trg_words = np.where(trg_ys.isin(idx2word.keys()), trg_ys.replace(idx2word), UNKNOWN_WORD)
		with open('meta-predictions.txt', 'a') as f:
			f.write("On iteration %d" % self.num_iter)
			f.write("The support here\n")
			f.write(str(support_x_pred[0]))
			f.write("\n")
			f.write("One of the queries\n")
			f.write(str(trg_words[0]))
			f.write("\n")
			f.write("one of the predictions\n")
			f.write(str(pred_words[0]))
			f.write("\n\n\n\n\n")

	def compute_mle_loss(self, pred, target, smoothing, log=False):
		def compute_loss(pred, target, smoothing):
			target = target.contiguous().view(-1)
			if smoothing:
			  eps = 0.1
			  n_class = pred.size(1)

			  one_hot = torch.zeros_like(pred)
			  one_hot = one_hot.scatter(1, target.view(-1, 1), 1)
			  one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
			  log_prb = F.log_softmax(pred, dim=1)

			  non_pad_mask = target.ne(PAD_IDX)
			  loss = -(one_hot * log_prb).sum(dim=1)
			  loss = loss.masked_select(non_pad_mask).sum()  # average later
			else:
				
			  loss = F.cross_entropy(pred, target, ignore_index=PAD_IDX, reduction='sum')
			return loss
		
		loss = compute_loss(pred, target, smoothing)
		pred_max = pred.max(1)[1]
		target = target.contiguous().view(-1)
		non_pad_mask = target.ne(PAD_IDX)
		n_correct = pred_max.eq(target)
		n_correct = n_correct.masked_select(non_pad_mask).sum().item()
		n_total = non_pad_mask.sum().item()
		return loss, n_correct, n_total

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
		# print(" ")
		self.model.load_state_dict(original_state_dict)
		# print(" ")
		self.model.to(self.device)
		self.model.train()

		self.optimizer.zero_grad()
		dummy_query_x, dummy_query_y = temp_data
		# print(" ")
		pred_logits = self.model(input_ids=dummy_query_x, decoder_input_ids=dummy_query_y[:, :-1])
		pred_logits = pred_logits.contiguous().view(-1, VOCAB_SIZE)
		dummy_loss, _, _ = self.compute_mle_loss(pred_logits, dummy_query_y[:, 1:], smoothing=True)
		# print(" ")
		# dummy_loss, _ = self.model(temp_data)
		hooks = self._hook_grads(all_grads)
		dummy_loss.backward()
		# print(" ")
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		
		self.optimizer.step()
		# print("did optimzier step")
		# gpu memory explodes if you dont remove hooks
		for h in hooks:
			h.remove()

	def forward(self, num_updates, data_queue, data_event, process_event, tb=None, log_interval=250, checkpoint_interval=1000):
		n_word_total = 0.0
		n_word_correct = 0.0
		total_loss = 0.0
		data_event.wait()
		while(True):
			data = data_queue.get()

			if data is None: break
			dist.barrier(async_op=True)

			if self.process_id == 0: 
				original_state_dict = {}
				data_event.clear()

			if self.process_id == 0 and self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
				save_checkpoint(0, self.model, self.optimizer, suffix=str(self.num_iter))
			
			# broadcast weights from master process to all others and save them to a detached dictionary for loadinglater
			for k, v in self.model.state_dict().items():
				if self.process_id == 0:# and self.forward_passes == 0:
					original_state_dict[k] = v.clone().detach()
				# v.to(self.device)
				dist.broadcast(v, src=0, async_op=True)

			self.model.to(self.device)
			self.model.train()

			# meta gradients
			support_x, support_y, query_x, query_y = map(lambda x: torch.LongTensor(x).to(self.device), data)
			for i in range(num_updates):
				self.meta_optimizer.zero_grad()
				pred_logits = self.model(input_ids=support_x, decoder_input_ids=support_y[:, :-1])
				pred_logits = pred_logits.contiguous().view(-1, VOCAB_SIZE)
				loss, n_correct, _ = self.compute_mle_loss(pred_logits, support_y[:, 1:], smoothing=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				self.meta_optimizer.step()

			pred_logits = self.model(input_ids=query_x, decoder_input_ids=query_y[:, :-1])
			pred_logits = pred_logits.contiguous().view(-1, VOCAB_SIZE)
			loss, n_correct, n_total = self.compute_mle_loss(pred_logits, query_y[:, 1:], smoothing=True)

			n_word_total += n_total
			n_word_correct += n_correct
			total_loss += loss.item()

			# loss, pred = self.model(query_x, query_y)
			all_grads = autograd.grad(loss, self.model.parameters())
				
			for idx in range(len(all_grads)):
				dist.reduce(all_grads[idx].data, 0, op=dist.ReduceOp.SUM, async_op=True)
				all_grads[idx].data = all_grads[idx].data / self.world_size

			if self.process_id == 0 and tb is not None and self.num_iter % log_interval == 0 and self.num_iter != 0:
				tb_mle_meta_batch(tb, total_loss/n_word_total, n_word_correct/n_word_total, self.num_iter)
				# n_word_total = 0.0; n_word_correct = 0.0; total_loss = 0.0
				self._write_prediction(support_x, pred_logits, query_y)
				

			if self.process_id == 0:
				self.num_iter += 1
				self._write_grads(original_state_dict, all_grads, (query_x, query_y))
				process_event.set()

			data_event.wait()



class MetaTrainer:

	def __init__(self, world_size, device='cpu', model_params=None, num_iters=25000, load_model=False, fine_tune=False):
		self.world_size = world_size

		self.meta_learners = [Learner(process_id=process_id, gpu=process_id if device is not 'cpu' else 'cpu', world_size=world_size, model_params=model_params, num_iters=num_iters, load_model=load_model, fine_tune=fine_tune) for process_id in range(world_size)]
		# gpu backend instead of gloo
		self.backend = "gloo"
		self.num_iters = num_iters
		
	def init_process(self, process_id, data_queue, data_event, process_event, num_updates, tb, address='localhost', port='29500'):
		os.environ['MASTER_ADDR'] = address
		os.environ['MASTER_PORT'] = port
		dist.init_process_group(self.backend, rank=process_id, world_size=self.world_size)
		self.meta_learners[process_id](num_updates, data_queue, data_event, process_event, tb)

	def load_next(self, task, data_loaders):
		try:
			task_data = next(data_loaders[task])
			return task_data
		except:
			del data_loaders[task]
			self.num_tasks -= 1
			return self.load_next(np.random.randint(0, self.num_tasks), data_loaders)

	# dataloaders is list of the iterators of the dataloaders for each task
	def train(self, data_loaders, tb=None, num_updates = 5):
		data_queue = Queue()
		# for notifying when to recieve data
		data_event = Event()
		# for notifying this method when to send new data
		process_event = Event()
		# so doesn't hang on first iteration
		process_event.set()
		self.num_tasks = len(data_loaders)
		m = self.num_tasks
		# print(num_tasks)
		processes = []
		for process_id in range(self.world_size):
			processes.append(Process(target=self.init_process, 
												args=(process_id, data_queue, data_event, 
													process_event, num_updates, 
													tb if process_id == 0 else None)))


			processes[-1].start()

		for num_iter in tqdm(range(self.num_iters), mininterval=2, leave=False):
			process_event.wait()
			process_event.clear()
			tasks = np.random.randint(0, self.num_tasks, (self.world_size))
			for task in sorted(tasks)[::-1]:
				# place holder for sampling data from dataset
				task_data = self.load_next(task, data_loaders)
				try:
				# print(hey[0].shape)	
					data_queue.put((task_data[0].numpy()[0], task_data[1].numpy()[0], 
									task_data[2].numpy()[0], task_data[3].numpy()[0]))
				except:
					task_data = self.load_next(task, data_loaders)
					data_queue.put((task_data[0].numpy()[0], task_data[1].numpy()[0], 
									task_data[2].numpy()[0], task_data[3].numpy()[0]))
				
			data_event.set()

		for i in range(self.world_size):
			data_queue.put(None)

		for p in processes:
			p.terminate()
			p.join()