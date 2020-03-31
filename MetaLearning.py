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

class Learner(nn.Module):

	def __init__(self, gpu='cpu', meta_lr=1e-3, model_params=None, checkpoint_path=None):
		super(Learner, self).__init__()
		self.model = BartModel(model_params)
		self.model_pi = BartModel(model_params)
		if checkpoint_path:
			model_dict = torch.load(checkpoint_path)['model']
			for k, v in list(model_dict.items()):
				kn = k.replace('module.', '')
				if kn != k:
					model_dict[kn] = v
					del model_dict[k]

			self.model.load_state_dict(model_dict)
			self.model_pi.load_state_dict(model_dict)

		self.meta_optimizer = optim.SGD(self.model_pi.parameters(), meta_lr)
		self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
		self.model.to(self.device)
		self.model_pi.to(self.device)
		self.model.train()
		self.model_pi.train()
		self.num_iter = 0

	def forward_temp(self, temp_data):
		dummy_query_x, dummy_query_y = temp_data
		pred_logits = self.model(input_ids = dummy_query_x, decoder_input_ids=dummy_query_y[:, :-1])
		pred_logits = pred_logits.contiguous().view(-1, VOCAB_SIZE)
		dummy_loss, _, _ = self.compute_mle_loss(pred_logits, dummy_query_y[:, 1:], smoothing=False)
		return dummy_loss

	def forward(self, num_updates, data, log_interval=250, checkpoint_interval=1000):
		if self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
			torch.save({'model': model.state_dict(), 'num_iter': self.num_iter}, "checkpoint-{}.pth".format(self.num_iter))

		for copy_param, param in zip(self.model.parameters(), self.model_pi.parameters()):
			param.data.copy_(copy_param.data)

		support_x, support_y, query_x, query_y = map(lambda x: x.squeeze(0).to(self.device), data)
		for i in range(num_updates):
			self.meta_optimizer.zero_grad()
			pred_logits = self.model_pi(input_ids=support_x, decoder_input_ids=support_y[:, :-1])
			pred_logits = pred_logits.contiguous().view(-1, VOCAB_SIZE)
			loss, n_correct, _ = self.compute_mle_loss(pred_logits, support_y[:, 1:], smoothing=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model_pi.parameters(), 1.0)
			self.meta_optimizer.step()

		pred_logits = self.model_pi(input_ids=query_x, decoder_input_ids=query_y[:, :-1])
		pred_logits = pred_logits.contiguous().view(-1, VOCAB_SIZE)
		loss, n_correct, n_total = self.compute_mle_loss(pred_logits, query_y[:, 1:], smoothing=True)

		if self.num_iter != 0 and self.num_iter % log_interval == 0:
			self._write_prediction(support_x, pred_logits, query_y)

		all_grads = autograd.grad(loss, self.model_pi.parameters(), create_graph=True)
		self.num_iter += 1

		return all_grads, (query_x, query_y), (n_total, n_correct, loss.item())

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

	def parameters(self):
		return self.model.parameters()

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


class MetaTrainer:

	def __init__(self, device='cpu', lr=1e-5, meta_lr=1e-4, model_params=None, tb=None, checkpoint_path=None):

		self.meta_learner = Learner(gpu='cpu' if str(device) == 'cpu' else 0, meta_lr=meta_lr, model_params=model_params, checkpoint_path=checkpoint_path)
		self.device=device
		self.optimizer = AdamW(self.meta_learner.parameters(), lr)
		self.tb = tb

	def load_next(self, task, data_loaders):
		try:
			task_data = next(data_loaders[task])
			return task_data
		except:
			del data_loaders[task]
			self.num_tasks -= 1
			return self.load_next(np.random.randint(0, self.num_tasks), data_loaders)

	# dataloaders is list of the iterators of the dataloaders for each task
	def train(self, data_loaders, num_updates=5, num_iterations=250000, meta_batch_size=4, tb_interval=20):
		total_loss = 0.0; n_word_correct = 0.0; n_word_total = 0.0
		self.num_tasks = len(data_loaders)
		m = self.num_tasks
		for num_iter in tqdm(range(int(num_iterations/meta_batch_size)), mininterval=2, leave=False):
			sum_grads = None
			tasks = np.random.randint(0, self.num_tasks, (meta_batch_size))
			for idx, task in enumerate(sorted(tasks)[::-1]):
				# place holder for sampling data from dataset
				task_data = self.load_next(task, data_loaders)
				try:
					curr_grads, temp_data, tb_data = self.meta_learner(num_updates, task_data)
				except:
					task_data = self.load_next(task, data_loaders)
					curr_grads, temp_data, tb_data = self.meta_learner(num_updates, task_data)

				scalar = 1.0
				if idx+1 == meta_batch_size: scalar /= meta_batch_size
				sum_grads = [torch.add(i, j)*scalar for i, j in zip(sum_grads, curr_grads)] if sum_grads else curr_grads
			
			dummy_loss = self.meta_learner.forward_temp(temp_data)
			self._write_grads(sum_grads, dummy_loss)
			n_word_total += tb_data[0]; n_word_correct += tb_data[1]; total_loss += tb_data[2]
			if num_iter != 0 and num_iter % tb_interval == 0:
				tb_mle_meta_batch(self.tb, total_loss/n_word_correct, n_word_correct/n_word_total, num_iter)
				total_loss = 0.0; n_word_correct = 0.0; n_word_total = 0.0

	def _write_grads(self, all_grads, dummy_loss):
		self.optimizer.zero_grad()
		hooks = self._hook_grads(all_grads)
		dummy_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), 1.0)
		self.optimizer.step()

		for h in hooks:
			h.remove()

	def _hook_grads(self, all_grads):
		hooks = []
		for i, v in enumerate(self.meta_learner.parameters()):
			def closure():
				ii = i
				return lambda grad: all_grads[ii]
			hooks.append(v.register_hook(closure()))
		return hooks


