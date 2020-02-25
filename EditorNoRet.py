import torch
import torch.nn as nn
import torch.nn.functional as F
from DataClass.torchData import tokens_dict, createDataLoaderAllFiles, PAD_IDX
from transformer.Models import Transformer
from tqdm import tqdm


# exp_name='editor_no_retriever_seq2seq', unique_id='2020-02-24'
# tb = Tensorboard(exp_name, unique_name=unique_id)
MAX_LINE_LENGTH = 256
VOCAB_SIZE = len(tokens_dict)

class EditorNoRetrieval:

	def __init__(self, filepath='.', num_layers=6, num_heads=8, key_dimension=64, 
                 value_dimension=64, dropout=0.1, n_position = MAX_LINE_LENGTH, 
                 d_word_vec=512, inner_dimension=2048, device='cpu',
                 n_trg_position = MAX_LINE_LENGTH, n_src_position = MAX_LINE_LENGTH, padding = 1,
                 batch_size=4):


		self.model = Transformer(n_src_vocab=VOCAB_SIZE+3, n_trg_vocab=VOCAB_SIZE+3, src_pad_idx=0, trg_pad_idx=0, 
					d_word_vec=d_word_vec, d_model=d_word_vec, d_inner=inner_dimension, n_layers=num_layers,
					n_head=num_heads, d_k=key_dimension, d_v=value_dimension, dropout=dropout,
					n_trg_position=n_trg_position, n_src_position=n_src_position, 
					trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True)


		self.data_loader = createDataLoaderAllFiles(dataset_dir=filepath, batch_size=batch_size)
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


	def train(self, optimizer, tb=None, epochs=20, log_interval=100):
		self.model.train()

		for epoch in range(epochs):
			total_mle_loss = 0.0
			n_word_total = 0.0
			n_word_correct = 0.0
			for batch_idx, batch in enumerate(tqdm(self.data_loader, mininterval=2, leave=False)):
				batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)
				trg_ys = batch_ys[:, 1:]
				optimizer.zero_grad()

				pred_logits = self.model(batch_xs, batch_ys[:, :-1])
				pred_logits = pred_logits.view(-1, pred_logits.size(2))
				loss, n_correct = self.compute_mle_loss(pred_logits, trg_ys, smoothing=True)

				loss.backward()
				optimizer.step()

				total_mle_loss += loss.item()

				non_pad_mask = trg_ys.ne(PAD_IDX)
				n_word = non_pad_mask.sum().item()
				n_word_total += n_word
				n_word_correct += n_correct

				if tb is not None and batch_idx % log_interval == 0:
					self.tb_mle_batch(tb, total_mle_loss, n_word_total, n_word_correct, epoch, batch_idx, len(training_data))
			
			loss_per_word = total_loss / n_word_total
			accuracy = n_word_correct / n_word_total

			if tb is not None:
				self.tb_mle_epoch(tb, loss_per_word, accuracy, epoch)


	def tb_mle_epoch(self, tb, loss_per_word, accuracy, epoch):
		tb.add_scalars(
			{
				"loss_per_word" : loss_per_word,
				"accuracy" : accuracy,
			},
			group="train",
			sub_group="epoch",
			global_step=epoch
		)

	def tb_mle_batch(self, tb, total_loss, n_word_total, n_word_correct, epoch, batch_idx, data_len):
		tb.add_scalars(
			{
				"loss_per_word" : total_loss / n_word_total,
				"accuracy": n_word_correct / n_word_total,
			},
		group="mle_train",
		sub_group="batch",
		global_step = epoch*data_len+batch_idx)

