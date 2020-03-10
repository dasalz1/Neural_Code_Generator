from tensorboard_utils import Tensorboard
from DataClass.torchData import *
from DataClass.Constants import PAD_IDX, END_IDX
from DataClass.torchData import MAX_LINE_LENGTH
from EditorNoRet import EditorNoRetrievalTrainer
from torch.utils.data import ConcatDataset, DataLoader
from transformer.bart import BartModel
from transformer.configuration_bart import BartConfig
import torch
from random import shuffle
from datetime import date
import argparse, random
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument("--filepath", default='../repo_files', type=str)
parser.add_argument("--exp_name", default='EditorPairTrain', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--key_dimension", default=64, type=int)
parser.add_argument("--value_dimension", default=64, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--d_word_vec", default=512, type=int)
parser.add_argument("--inner_dimension", default=2048, type=int)
parser.add_argument("--batch_size", default=24, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--use_retriever", default=False, action='store_true')
parser.add_argument("--n_retrieved", default=2, type=int)
parser.add_argument("--n_src_length", default=MAX_LINE_LENGTH, type=int)
parser.add_argument("--n_trg_length", default=MAX_LINE_LENGTH, type=int)
args = parser.parse_args()

def main(args):
	random.seed(68492)
	np.random.seed(68492)
	torch.manual_seed(68492)

	VOCAB_SIZE = len(word2idx)
	num_validation_repos = 50

	if args.use_retriever:
		args.n_src_length = (2*args.n_retrieved-1)+(args.n_src_length)*(args.n_retrieved*2)+1+args.n_retrieved

	if args.use_retriever: args.exp_name += ("retrieved_%d" % args.n_retrieved)

	tb = Tensorboard(args.exp_name, unique_name=args.unique_id)

	repo_files = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(args.filepath))[2]))
	# shuffle(repo_files)

	if args.use_retriever:
		datasets = ConcatDataset([RetrieveDataset(args.filepath +'/'+dataset, args.n_retrieved) for dataset in repo_files[num_validation_repos:]])
		validsets = ConcatDataset([RetrieveDataset(args.filepath +'/'+dataset, args.n_retrieved) for dataset in repo_files[:num_validation_repos]])
	else:
		datasets = ConcatDataset([PairDataset(args.filepath +'/'+dataset) for dataset in repo_files[num_validation_repos:]])
		validsets = ConcatDataset([PairDataset(args.filepath +'/'+dataset) for dataset in repo_files[:num_validation_repos]])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data_loader = DataLoader(datasets,
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=batch_collate_fn)#,
							# num_workers=92)

	print("Finished creating data loader")

	validation_loader = DataLoader(validsets,
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=batch_collate_fn)#,
							# num_workers=92)

	print("Finished creating validation data loader")
	num_iterations = len(data_loader)

	model_config = BartConfig(
		vocab_size=VOCAB_SIZE, pad_token_id=PAD_IDX,
		eos_token_id=END_IDX, d_model=args.d_word_vec,
		encoder_ffn_dim=args.inner_dimension,
		encoder_layers=args.num_layers,
		encoder_attention_heads=args.num_heads,
		decoder_ffn_dim=args.inner_dimension,
		decoder_layers=args.num_layers,
		decoder_attention_heads=args.num_heads,
		dropout=args.dropout,
		max_encoder_position_embeddings=args.n_src_length,
		max_decoder_position_embeddings=args.n_trg_length)

	model = BartModel(model_config)#Transformer(n_src_vocab=VOCAB_SIZE, n_trg_vocab=VOCAB_SIZE, src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, 
						# d_word_vec=args.d_word_vec, d_model=args.d_word_vec, d_inner=args.inner_dimension, n_layers=args.num_layers,
						# n_head=args.num_heads, d_k=args.key_dimension, d_v=args.value_dimension, dropout=args.dropout,
						# n_trg_position=MAX_LINE_LENGTH, n_src_position=MAX_LINE_LENGTH, 
						# trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True)

	

	if torch.cuda.is_available:
		torch.backends.cudnn.deterministic=True
		torch.backends.cudnn.benchmark = False
		
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs...")
		model = torch.nn.DataParallel(model)
	
	model.to(device)

	trainer = EditorNoRetrievalTrainer(device)

	trainer.train(model, data_loader, validation_loader, tb=tb, epochs=args.epochs)

if __name__=='__main__':
	main(args)