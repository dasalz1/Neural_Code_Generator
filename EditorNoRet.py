import torch
import torch.nn as nn
import torch.nn.functional as F
from DataClass.Constants import PAD_IDX, UNKNOWN_WORD
from train_utils import save_checkpoint, from_checkpoint_if_exists, tb_mle_epoch, tb_mle_batch, tb_bleu_validation_epoch
from tqdm import tqdm
import numpy as np
import pandas as pd
from DataClass.torchData import idx2word
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate import bleu
from transformers import AdamW, get_cosine_schedule_with_warmup


from copy import deepcopy

class EditorNoRetrievalTrainer:

	def __init__(self, device):
		self.device = device

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
	
	def validate_BLEU(self, model, validation_loader, epoch, tb=None):
		model.eval()

		bleu_scores = []
		accuracies = []
		with torch.no_grad():
			for idx, batch in enumerate(tqdm(validation_loader, mininterval=2, leave=False)):
				batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)
				trg_ys = pd.DataFrame(batch_ys[:, 1:].to('cpu').numpy())

				pred = model(input_ids=batch_xs, decoder_input_ids=batch_ys[:, :-1])

				# pred_max = pred.max(1)[1]
				pred_max = pred.max(2)[1]
				pred = pd.DataFrame(pred_max.to('cpu').numpy())

				target = batch_ys[:, 1:].contiguous().view(-1)
				non_pad_mask = target.ne(PAD_IDX)
				n_correct = pred_max.contiguous().view(-1).eq(target)
				n_correct = n_correct.masked_select(non_pad_mask).sum().item()
				n_word = non_pad_mask.sum().item()
				accuracies.append(n_correct/n_word)

				pred_words = np.where(pred.isin(idx2word.keys()), pred.replace(idx2word), UNKNOWN_WORD)
				
				trg_words = np.where(trg_ys.isin(idx2word.keys()), trg_ys.replace(idx2word), UNKNOWN_WORD)
				trg_words = np.expand_dims(trg_words, axis=1)
				bleu_scores.append(corpus_bleu(trg_words.tolist(), pred_words.tolist(), smoothing_function=SmoothingFunction().method1))

			avg_bleu = np.mean(bleu_scores)
			avg_accuracy = np.mean(accuracies)
			print("Validation BLEU score: %.4f, Accuracy: %.4f" % (avg_bleu, avg_accuracy))
			if tb is not None:
				tb_bleu_validation_epoch(tb, avg_bleu, avg_accuracy, epoch)


	def train(self, model, data_loader, validation_loader, tb=None, epochs=20, log_interval=100, checkpoint_interval=5000):
		for epoch in range(epochs):
			model.train()
			total_mle_loss = 0.0
			n_word_total = 0.0
			n_word_correct = 0.0
			optimizer = AdamW(model.parameters(), lr=6e-4)
			scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=32000, num_training_steps=len(data_loader))
			for batch_idx, batch in enumerate(tqdm(data_loader, mininterval=2, leave=False)):
				batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)
				trg_ys = batch_ys[:, 1:]
		
				pred_logits = model(input_ids=batch_xs, decoder_input_ids=batch_ys[:, :-1])
				# pred_logits = pred_logits.contiguous().view(-1, pred_logits.size(2))
				pred_logits = pred_logits.reshape(-1, pred_logits.size(2))
				loss, n_correct, n_total = self.compute_mle_loss(pred_logits, trg_ys, smoothing=True)
				loss.backward()
				
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

				optimizer.step()
				scheduler.step()
				total_mle_loss += loss.item()
				optimizer.zero_grad()

				# non_pad_mask = trg_ys.ne(PAD_IDX)
				# n_word = non_pad_mask.sum().item()
				n_word_total += n_total
				n_word_correct += n_correct
				if tb is not None and batch_idx % log_interval == 0:
					tb_mle_batch(tb, total_mle_loss, n_word_total, n_word_correct, epoch, batch_idx, len(data_loader))

				if batch_idx != 0 and batch_idx % checkpoint_interval == 0:
					pred_max = pred_logits.reshape(-1, 127, len(idx2word)).max(2)[1]
					pred = pd.DataFrame(pred_max.to('cpu').numpy())
					pred_words = np.where(pred.isin(idx2word.keys()), pred.replace(idx2word), UNKNOWN_WORD)
					trg_ys = pd.DataFrame(batch_ys[:, 1:].to('cpu').numpy())
					trg_words = np.where(trg_ys.isin(idx2word.keys()), trg_ys.replace(idx2word), UNKNOWN_WORD)
					with open('output_tests.txt', 'a') as f:
						f.write("On the iteration %d" % batch_idx)
						f.write("The actual line:\n")
						f.write(trg_words[0])
						f.write("The prediciton of the line:\n")
						f.write(pred_words[0])
						f.write('\n\n\n\n\n')
						
					save_checkpoint(epoch, model, optimizer, scheduler, suffix=str(batch_idx))

			loss_per_word = total_mle_loss / n_word_total
			accuracy = n_word_correct / n_word_total

			if tb is not None:
				tb_mle_epoch(tb, loss_per_word, accuracy, epoch)

			self.validate_BLEU(model, deepcopy(validation_loader), epoch, tb)


# for batch_idx, batch in enumerate(tqdm(validation_loader, mininterval=2, leave=False)):
# 	with torch.no_grad():
# 		batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)
# 		trg_ys = pd.DataFrame(batch_ys[:, 1:].to('cpu').numpy())
# 		pred = model(batch_xs, batch_ys[:, :-1])
# 		pred_max = pred.to('cpu').max(2)[1]
# 		pred = pd.DataFrame(pred_max.numpy())
# 		pred_words = np.where(pred.isin(idx2word.keys()), pred.replace(idx2word), UNKNOWN_WORD)
# 		trg_words = np.where(trg_ys.isin(idx2word.keys()), trg_ys.replace(idx2word), UNKNOWN_WORD)
# 		print(pred_words[0])
# 		print(trg_words[0])
# 		break