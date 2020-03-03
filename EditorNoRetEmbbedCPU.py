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


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class EditorNoRetrievalTrainerEmbbedCPU:

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
		return loss, n_correct
	
	def validate_BLEU(self, model, validation_loader, epoch, tb=None):
		model.eval()

		bleu_scores = []
		accuracies = []
		with torch.no_grad():
			for batch in tqdm(validation_loader):
				batch_xs, batch_ys = map(lambda x: x, batch)#.to(self.device), batch)
				# batch_ys = batch_ys.to(self.device)
				# trg_ys = batch_ys[:, 1:].to(self.device)
				trg_ys = pd.DataFrame(batch_ys[:, 1:].numpy())


				src_mask = (batch_xs != PAD_IDX).unsqueeze(-2).to(self.device)
				src_seq = src_word_emb(batch_xs).to(self.device)

				enc_output = model.forward_enc(src_seq, src_mask)

				trg_mask = (get_pad_mask(batch_ys[:, :-1], PAD_IDX) & get_subsequent_mask(batch_ys[:, :-1])).to(self.device)
				trg_seq = trg_word_emb(batch_ys[:, :-1]).to(self.device)

				dec_output = model.forward_dec(enc_output, trg_seq, src_mask, trg_mask).cpu()
				pred = trg_word_prj(dec_output)*x_logit_scale

				# pred_max = pred.max(1)[1]
				pred_max = pred.max(2)[1]
				pred = pd.DataFrame(pred_max.numpy())

				target = batch_ys[:, 1:].contiguous().view(-1)
				non_pad_mask = target.ne(PAD_IDX)
				n_correct = pred_max.contiguous().view(-1).eq(target)
				n_correct = n_correct.masked_select(non_pad_mask).sum().item()
				n_word = non_pad_mask.sum().item()
				accuracies.append(n_correct/n_word)

				pred_words = np.where(pred.isin(idx2word.keys()), pred.replace(idx2word), UNKNOWN_WORD)
				# print(pred.shape)
				# print(trg_ys.shape)
				# print(trg_ys)
				trg_words = np.where(trg_ys.isin(idx2word.keys()), trg_ys.replace(idx2word), UNKNOWN_WORD)
				trg_words = np.expand_dims(trg_words, axis=1)
				bleu_scores.append(corpus_bleu(trg_words.tolist(), pred_words.tolist(), smoothing_function=SmoothingFunction().method1))

			avg_bleu = np.mean(bleu_scores)
			avg_accuracy = np.mean(accuracies)
			print("Validation BLEU score: %.4f, Accuracy: %.4f" % (avg_bleu, avg_accuracy))
			if tb is not None:
				tb_bleu_validation_epoch(tb, avg_bleu, avg_accuracy, epoch)


	def train(self, model, src_word_emb, trg_word_emb, trg_word_prj, x_logit_scale, optimizer, data_loader, validation_loader, scheduler=None, tb=None, epochs=20, log_interval=100, checkpoint_interval=10000):
		
		curr_epoch, model, optimizer, scheduler = from_checkpoint_if_exists(model, optimizer, scheduler)
		

		for epoch in range(epochs):
			model.train()
			total_mle_loss = 0.0
			n_word_total = 0.0
			n_word_correct = 0.0
			for batch_idx, batch in enumerate(tqdm(data_loader, mininterval=2, leave=False)): 
				batch_xs, batch_ys = map(lambda x: x, batch)#.to(self.device), batch)
				# batch_ys = batch_ys.to(self.device)
				trg_ys = batch_ys[:, 1:].to(self.device)

				optimizer.zero_grad()

				src_mask = (batch_xs != PAD_IDX).unsqueeze(-2).to(self.device)
				src_seq = src_word_emb(batch_xs).to(self.device)

				enc_output = model.forward(src_seq=src_seq, src_mask=src_mask, module="encoder")

				trg_mask = (get_pad_mask(batch_ys[:, :-1], PAD_IDX) & get_subsequent_mask(batch_ys[:, :-1])).to(self.device)
				trg_seq = trg_word_emb(batch_ys[:, :-1]).to(self.device)

				dec_output = model.forward(enc_output=enc_output, trg_seq=trg_seq, src_mask=src_mask, trg_mask=trg_mask, module="decoder").cpu()
				pred_logits = trg_word_prj(dec_output)*x_logit_scale
				pred_logits.to(self.device)
    			# pred_logits


				# pred_logits = model(batch_xs, batch_ys[:, :-1])
				pred_logits = pred_logits.contiguous().view(-1, pred_logits.size(2))
				loss, n_correct = self.compute_mle_loss(pred_logits, trg_ys, smoothing=True)

				loss.backward()
				
				torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
				optimizer.step()

				if scheduler:
					scheduler.step()

				total_mle_loss += loss.item()

				non_pad_mask = trg_ys.ne(PAD_IDX)
				n_word = non_pad_mask.sum().item()
				n_word_total += n_word
				n_word_correct += n_correct

				if tb is not None and batch_idx % log_interval == 0:
					tb_mle_batch(tb, total_mle_loss, n_word_total, n_word_correct, epoch, batch_idx, len(data_loader))

				if batch_idx != 0 and batch_idx % checkpoint_interval == 0:
					save_checkpoint(epoch, model, optimizer, scheduler, suffix=str(batch_idx))
			
			loss_per_word = total_mle_loss / n_word_total
			accuracy = n_word_correct / n_word_total

			if tb is not None:
				tb_mle_epoch(tb, loss_per_word, accuracy, epoch)

			self.validate_BLEU(model, validation_loader, epoch, tb)