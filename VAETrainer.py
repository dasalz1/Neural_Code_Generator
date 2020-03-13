import torch
import torch.nn as nn
import torch.nn.functional as F
from DataClass.Constants import PAD_IDX, UNKNOWN_WORD, END_IDX
from train_utils import save_checkpoint, from_checkpoint_if_exists, tb_vae
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from DataClass.torchData import idx2word
from DataClass.torchData import MAX_LINE_LENGTH
from copy import deepcopy
from tqdm import tqdm
from threading import Thread

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

	def train(self, model, data_loader, validation_loader, tb=None, epochs=20, log_interval=100, checkpoint_interval=10000):
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		total_loss = 0.0
		for epoch in range(epochs):
			model.train()

			for batch_idx, batch in enumerate(tqdm(data_loader, mininterval=2, leave=False)):
				batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)

				# lengths = []
				# for i in range(batch_xs.shape[0]):
					# lengths.append(list(batch_xs[i, :]).index(END_IDX))

				# lengths = torch.LongTensor(lengths)
				lengths = (batch_xs==END_IDX).nonzero()[:, 1]
				logp, mean, logv, z = model(batch_xs, lengths)

				NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch_ys, lengths, mean, logv, 'logistic', batch_idx, 0.0025, 2500)

				loss = (NLL_loss + KL_loss * KL_weight)/batch_xs.shape[0]

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				total_loss += loss.item()

				if tb is not None and batch_idx % log_interval == 0:
					tb_vae(tb, loss/log_interval, batch_idx)
					total_loss = 0.0

				if batch_idx != 0 and batch_idx % checkpoint_interval == 0:
					save_checkpoint(epoch, model, optimizer, suffix=str(batch_idx))


	def _create_file(self, model, dataset, path, file, total_threads):
		curr_repo = None
		for batch_idx, batch in enumerate(dataset):

			batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)

			lengths = (batch_xs==END_IDX).nonzero()[:, 1]
			# encs = model(batch_xs, lengths, only_enc=True).to('cpu').numpy()
			encs = np.random.rand(batch_xs.shape[0], 16)
			curr_repo = np.vstack([curr_repo, encs]) if curr_repo is not None else encs


		all_samples = pd.read_csv(path+'/'+file)
		new_repo = None
		ne = NearestNeighbors(10)
		ne.fit(curr_repo)
		neighbors = ne.kneighbors()[1]
		for idx, row in enumerate(neighbors):
			if idx == 0:
				new_repo = pd.DataFrame(np.array([all_samples.shape[0]] + [None]*11).reshape(1, -1))
			retrieved_lines = all_samples.iloc[np.where((row < (idx-2)) | (row > (idx+2)))[0]].values.flatten()[:5*2]
			full_row = pd.DataFrame(np.hstack([all_samples.iloc[idx].values, retrieved_lines]).reshape(1, -1))
			new_repo = pd.concat([new_repo, full_row], axis=0)

		new_repo.to_csv('../vae_files/' + file.replace('_line_pairs.csv', '').replace('.', '_')+'_vae_retrieve.csv', header=None, index=None)
		total_threads[0] = total_threads[0] - 1

	def create_files(self, model, files, threading=False):
		# model.eval()
		MAX_THREADS = 10
		if threading:
			threads = []; total_threads = [0]
			for dataset, path, file in tqdm(files, mininterval=2, leave=False):
				while(total_threads[0] >= MAX_THREADS): continue
				total_threads[0] = total_threads[0] + 1
				threads.append(Thread(target=self._create_file, args=(model, dataset, path, file, total_threads,)))
				threads[-1].start()

			for t in threads:
				t.join()
		else:
			for dataset, path, file in tqdm(files, mininterval=2, leave=False):
				self._create_file(model, dataset, path, file, [0])











