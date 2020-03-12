import torch
import torch.nn as nn
import torch.nn.functional as F
from DataClass.Constants import PAD_IDX, UNKNOWN_WORD, END_IDX
from train_utils import save_checkpoint, from_checkpoint_if_exists, tb_vae
import numpy as np
import pandas as pd
from DataClass.torchData import idx2word
from DataClass.torchData import MAX_LINE_LENGTH
from copy import deepcopy
from tqdm import tqdm

NLL = torch.nn.NLLLoss(size_average=False, ignore_index=PAD_IDX)

class VAETrainer:

	def __init__(self, device):
		self.device = device


	def kl_anneal_function(self, anneal_function, step, k, x0):
		if anneal_function == 'logistic':
			return float(1/(1+np.exp(-k*(step-x0))))
		elif anneal_function == 'linear':
			return min(1, step/x0)

	def loss_fn(self, logp, target, length, mean, logv, anneal_function, step, k, x0):
		# cut-off unnecessary padding from target, and flatten
		target = target[:, :int(torch.max(length))].contiguous().view(-1)
		logp = logp.view(-1, logp.size(2))

		# Negative Log Likelihood
		NLL_loss = NLL(logp, target)

		# KL Divergence
		KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
		KL_weight = self.kl_anneal_function(anneal_function, step, k, x0)

		return NLL_loss, KL_loss, KL_weight

	def train(self, model, data_loader, validation_loader, tb=None, epochs=20, log_interval=100, checkpoint_interval=500):
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		total_loss = 0.0
		for epoch in range(epochs):
			model.train()

			for batch_idx, batch in enumerate(tqdm(data_loader, mininterval=2, leave=False)):
				batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)

				lengths = []
				for i in range(batch_xs.shape[0]):
					lengths.append(list(batch_xs[i, :]).index(END_IDX))

				lengths = torch.LongTensor(lengths)

				lengths = torch.Tensor([MAX_LINE_LENGTH]*batch_xs.shape[0])
				logp, mean, logv, z = model(batch_xs, lengths)

				NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch_ys, lengths, mean, logv, 'logistic', batch_idx, 0.0025, 2500)

				loss = (NLL_loss + KL_loss * KL_weight)/batch_xs.shape[0]

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				total_loss += loss.item()

				if tb is not None and batch_idx % log_interval == 0:
					tb_vae(tb, loss/log_interval, num_iter)
					total_loss = 0.0

				if batch_idx != 0 and batch_idx % checkpoint_interval == 0:
					save_checkpoint(epoch, model, optimizer, suffix=str(batch_idx))
