import cPickle as pickle
import codecs
import random
from collections import defaultdict
from itertools import izip
from os.path import dirname, realpath, join
import os
import numpy as np
import torch.optim as optim
import unicodedata
import sys
from tqdm import tqdm
from editor_code.copy_editor import data
from editor_code.copy_editor.attention_decoder import AttentionDecoderCell
from editor_code.copy_editor.editor import EditExample, Editor
from editor_code.copy_editor.encoder import Encoder
from editor_code.copy_editor.vocab import load_embeddings
from editor_code.copy_editor.utils import edit_dist
from gtd.chrono import verboserate
from gtd.io import num_lines, makedirs
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.training_run import TorchTrainingRun
from gtd.ml.torch.utils import similar_size_batches, try_gpu, random_seed, random_state
from gtd.ml.training_run import TrainingRuns
from gtd.ml.training_run_viewer import TrainingRunViewer, Commit, JSONSelector, NumSteps, run_name, Owner
from gtd.utils import sample_if_large, bleu, Config, chunks


class EditTrainingRuns(TrainingRuns):
    def __init__(self, check_commit=True):
        data_dir = data.workspace.edit_runs
        src_dir = dirname(dirname(realpath(__file__)))  # root of the Git repo
        super(EditTrainingRuns, self).__init__(data_dir, src_dir, EditTrainingRun, check_commit=check_commit)


class EditTrainingRunsViewer(TrainingRunViewer):
    def __init__(self):
        runs = EditTrainingRuns(check_commit=False)
        super(EditTrainingRunsViewer, self).__init__(runs)

        meta = lambda paths: JSONSelector('metadata.txt', paths)
        two_digits = lambda x: round(x, 2)

        # list different formats in which stats were logged
        legacy = lambda name, high_or_low: [
                ('{}_valid'.format(name),),
                ('stats', 'big', name, 'valid'),
                ('stats', 'big', name, 'valid', high_or_low),
                ('stats', 'big', name, high_or_low, 'valid'),
            ]

        self.add('#', run_name)
        self.add('owner', Owner({}))
        self.add('dset', meta(('config', 'dataset', 'path')))
        self.add('enc dropout', meta(('config', 'model', 'encoder_dropout_prob')))
        self.add('dec dropout', meta(('config', 'model', 'decoder_dropout_prob')))
        self.add('steps', NumSteps())

        # best scores on the big validation set
        self.add('bleu', meta(legacy('bleu', 'high') + legacy('avg_bleu', 'high')), two_digits)
        self.add('edit dist', meta(legacy('edit_dist', 'low')), two_digits)
        self.add('exact match', meta(legacy('exact_match', 'high') + legacy('exact_match_prob', 'high')), two_digits)
        self.add('loss', meta(legacy('loss', 'low')), two_digits)

        self.add('last seen', meta(('last_seen',)))
        self.add('commit', Commit())

import pathlib2
import pandas as pd
from DataClass.Constants import NO_CONTEXT_WORD, UNKNOWN_IDX
from DataClass.data_utils import tokenize_fine_grained, create_vocab_dictionary, preprocess_tokens, preprocess_context, fix_quote_strings
from itertools import ifilterfalse
from multiprocessing import Pool

class EditDataSplits(object):
    """
    Attributes:
        train (list[EditExample])
        valid (list[EditExample])
        test (list[EditExample])
    """

    def __init__(self, data_dir, config):
        """Load examples for training, validation and testing.

        Args:
            data_dir (str): absolute path to dataset
            config (Config): dataset config

        Returns:
            EditDataSplits
        """
        max_seq_length = lambda ex: max(max(len(seq) for seq in ex.input_words), len(ex.target_words))
        ##########################################################################################
        # def examples_from_file(path, seq_length_limit):
        #     """Return list[EditExample] from file path."""
        #     examples = []
        #
        #     # count total lines before loading
        #     total_lines = num_lines(path)
        #
        #     with codecs.open(path, 'r', encoding='utf-8') as f:
        #         lnum = 0
        #         for line in verboserate(f, desc='Reading data file.', total=total_lines):
        #             split = line.strip().split('\t')
        #             lnum += 1
        #             input_words = []
        #             try:
        #                 for c in config.source_cols:
        #                     input_words.append(split[c].split(' '))
        #                 trg_words = split[config.target_col].split(' ')  # gold answer
        #                 assert len(trg_words) > 0
        #                 ex = EditExample(input_words, trg_words)
        #                 # skip sequences that are too long, because they use up memory
        #                 if max_seq_length(ex) > seq_length_limit:
        #                     continue
        #                 examples.append(ex)
        #             except:
        #                 print 'bad formatting in line ' + str(lnum)
        #                 print line
        #
        #     return examples
        ##########################################################################################

        def examples_from_file(data_paths, seq_length_limit, fname):
            examples = {}
            MAX_LINE_LENGTH = 128
            name = '{}.pickle'.format(fname)
            file = pathlib2.Path.cwd() / 'github_data' / 'processed_repo_pkl' / name
            # if os.path.exists(str(file)):
            #     with open(str(file), 'rb') as f:
            #         examples = pickle.load(f)
            #     f.close()
            #     return list(examples.values())
            # count total lines before loading
            num_direct = len(data_paths)
            for line in verboserate(data_paths, desc='Reading data file.', total=num_direct):
                df = pd.read_csv(line, skiprows= 2, header = None, names = [0, 1], dtype=str).fillna(NO_CONTEXT_WORD)
                df[0] = df[0].apply(lambda x: tokenize_fine_grained(x))
                # df[0] = df[0].apply(lambda x: preprocess_tokens(x, MAX_LINE_LENGTH))
                df[1] = df[1].apply(lambda x: tokenize_fine_grained(x))
                try:
                    ex = []
                    for i, row in df.iterrows():
                        try:
                            ex.append(EditExample(row[0], row[1]))
                        except:
                            # print 'bad formatting in file ' + str(line).split('/')[-1]
                            # print line
                            count = 1
                    # skip sequences that are too long, because they use up memory
                    # if max_seq_length(ex) > seq_length_limit:
                    #     continue
                    ex = list(ifilterfalse(lambda x: max_seq_length(x) > seq_length_limit, ex))
                    # examples[(str(line).split('/')[-1], len(ex))] = ex
                    file = pathlib2.Path.cwd() / 'github_data' / 'processed_repo_pkl' / fname
                    result = {(str(line).split('/')[-1], len(ex)) : ex}
                    k = str(line).split('/')[-1].split('.')[0]
                    pick_obj = {(str(line).split('/')[-1], len(ex)): ex}
                    obj_name = str(file / k) + '.pickle'
                    with open(obj_name, 'wb') as f:
                        pickle.dump(pick_obj, f)
                    f.close()
                except Exception as e:
                    print e
                    print 'bad formatting in file ' + str(line).split('/')[-1]
                    print line
            # name = '{}.pickle'.format(fname)
            # file = pathlib2.Path.cwd() / 'github_data' / 'processed_repo_pkl' / name
            # if fname == 'train':
            # file = pathlib2.Path.cwd() / 'github_data' / 'processed_repo_pkl' / fname
            # for k, v in tqdm(examples.items()):
            #     obj_name = file / k[0].split('.')[0]
            #     pick_obj = {k : v}
            #     with open(str(obj_name), 'wb') as f:
            #         pickle.dump(pick_obj, f)
            #     f.close()
            # else:
            #     if not os.path.exists(str(file)):
            #         with open(str(file), 'wb') as f:
            #             pickle.dump(examples, f)
            #         f.close()
            return list(examples.values())

        # data_path = pathlib2.Path.cwd() / 'github_data' / 'repo_files'
        # existing_files = [str(x) for x in data_path.iterdir() if '.csv' in str(x)]
        # flatten = lambda l: [item for sublist in l for item in sublist]
        # self.valid = examples_from_file(existing_files[:50], config.seq_max, 'valid')
        # self.valid = flatten(self.valid)
        # self.train = examples_from_file(existing_files[50:-50], config.seq_max, 'train')
        # self.train = flatten(self.train)
        # self.test = examples_from_file(existing_files[-50:], float('inf'), 'test')
        # self.test = flatten(self.test)
        # load pickled directory
        data_dir = pathlib2.Path.cwd() / 'github_data' / 'processed_repo_pkl'
        val_paths = list((data_dir / 'valid').glob('*.pickle'))
        self.valid = self.flatten(self.load_pickled_dir(val_paths))
        tr_paths = list((data_dir / 'train').glob('*.pickle'))[:10]
        self.train = self.flatten(self.load_pickled_dir(tr_paths))
        te_paths = list((data_dir / 'test').glob('*.pickle'))
        self.test = self.flatten(self.load_pickled_dir(te_paths))
        # self.train = examples_from_file(join(data_dir, 'train.tsv'), config.seq_max)
        # self.valid = examples_from_file(join(data_dir, 'valid.tsv'), config.seq_max)
        # self.test = examples_from_file(join(data_dir, 'test.tsv'), float('inf'))  # evaluation should not skip examples

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def load_pickled_dir(self, pickle_dir):
        list_of_ex = []
        for pickle_path in tqdm(pickle_dir):
            with open(str(pickle_path), 'rb') as f:
                result = pickle.load(f)
            f.close()
            list_of_ex.append(list(result.values()))
        list_of_ex = self.flatten(list_of_ex)
        return list_of_ex


class EditTrainingRun(TorchTrainingRun):
    def __init__(self, config, save_dir):
        super(EditTrainingRun, self).__init__(config, save_dir)

        # extra dir for storing TrainStates where NaN was encountered
        self.workspace.add_dir('nan_checkpoints')
        self.workspace.add_dir('traces')

        # build model
        with random_seed(config.optim.seed):
            print 'seed:'+str(config.optim.seed)
            model, optimizer = self._build_model(config.model, config.optim, config.dataset)
            self.train_state = self.checkpoints.load_latest(model, optimizer)

        # load data
        data_dir = join(data.workspace.datasets, config.dataset.path)
        self._examples = EditDataSplits(data_dir, config.dataset)

    @property
    def editor(self):
        """Return the Editor."""
        return self.train_state.model

    @property
    def examples(self):
        """Return data splits (EditDataSplits)."""
        return self._examples

    @classmethod
    def _build_model(cls, model_config, optim_config, data_config):
        """Build Editor.

        Args:
            model_config (Config): Editor config
            optim_config (Config): optimization config
            data_config (Config): dataset config

        Returns:
            Editor
        """
        file_path = join(data.workspace.word_vectors, model_config.wvec_path)
        word_embeddings = load_embeddings(file_path, model_config.word_dim,
                                          model_config.vocab_size,
                                          model_config.num_copy_tokens)
        word_dim = word_embeddings.embed_dim

        source_token_embedder = TokenEmbedder(word_embeddings, model_config.train_source_embeds)
        target_token_embedder = TokenEmbedder(word_embeddings, model_config.train_target_embeds)

        # number of input channels
        num_inputs = len(data_config.source_cols)

        decoder_cell = AttentionDecoderCell(target_token_embedder, 2 * word_dim,  # 2 * word_dim because we concat base and copy vectors
                                            model_config.agenda_dim, model_config.hidden_dim, model_config.hidden_dim,
                                            model_config.attention_dim,
                                            num_layers=model_config.decoder_layers,
                                            num_inputs=num_inputs, dropout_prob=model_config.decoder_dropout_prob,
                                            disable_attention=False)

        encoder = Encoder(word_dim, model_config.agenda_dim, model_config.hidden_dim,
                          model_config.encoder_layers, num_inputs, model_config.encoder_dropout_prob,
                          False)

        copy_len = [5,5,40]
        model = Editor(source_token_embedder, encoder, decoder_cell, copy_lens=copy_len)
        model = try_gpu(model)

        optimizer = optim.Adam(model.parameters(), lr=optim_config.learning_rate)

        return model, optimizer

    def train(self):
        """Train a model.

        NOTE: modifies TrainState in place.
        - parameters of the Editor and Optimizer are updated
        - train_steps is updated
        - random number generator states are updated at every checkpoint
        """
        # TODO(kelvin): do something to preserve random state upon reload?
        train_state = self.train_state
        examples = self._examples
        config = self.config
        workspace = self.workspace
        with random_state(self.train_state.random_state):
            editor = train_state.model
            train_batches = similar_size_batches(examples.train, config.optim.batch_size)
            editor.test_batch(train_batches[0])
            best_exact_match_score = 0.0
            while True:
                random.shuffle(train_batches)
                loss = 0
                for batch in verboserate(train_batches, desc='Streaming training examples'):
                    loss, _, _ = editor.loss(batch)

                    finite_grads, grad_norm = self._take_grad_step(train_state, loss)
                    if not finite_grads:
                        train_state.save(workspace.nan_checkpoints)

                        examples_path = join(workspace.nan_checkpoints, '{}.examples'.format(train_state.train_steps))
                        with open(examples_path, 'w') as f:
                            pickle.dump(batch, f)

                        print 'Gradient was NaN/inf on step {}.'.format(train_state.train_steps)

                    step = train_state.train_steps

                    # run periodic evaluation and saving
                    if step != 0:
                        if step % 10 == 0:
                            self._update_metadata(train_state)
                        if step % config.timing.eval_small == 0:
                            self.evaluate(step, big_eval=False)
                            self.tb_logger.log_value('grad_norm', grad_norm, step)
                        if step % config.timing.eval_big == 0:
                            train_stats, valid_stats = self.evaluate(step, big_eval=True)
                            # train_stats, valid_stats = self.evaluate(step, big_eval=False)
                            exact_match_score = valid_stats[('big', 'exact_match', 'valid')]
                            self.checkpoints.save(train_state)
                        if step >= config.optim.max_iters:
                            return

    def evaluate(self, train_steps, big_eval, log=True):
        config = self.config
        examples = self.examples
        editor = self.editor

        def evaluate_on_examples(split_name, examples):
            # use more samples for big evaluation
            num_eval = config.eval.big_num_examples if big_eval else config.eval.num_examples
            big_str = 'big' if big_eval else 'small'

            # compute metrics
            stats, edit_traces, loss_traces = self._compute_metrics(editor, examples, num_eval, self.config.optim.batch_size)

            # prefix the stats
            stats = {(big_str, stat, split_name): val for stat, val in stats.items()}

            if log:
                self._log_stats(stats, train_steps)

                # write traces for the small evaluation
                if not big_eval:
                    self._write_traces(split_name, train_steps, edit_traces, loss_traces)

            return stats

        train_stats = evaluate_on_examples('train', examples.train)
        valid_stats = evaluate_on_examples('valid', examples.valid)

        return train_stats, valid_stats

    @classmethod
    def _compute_metrics(cls, editor, examples, num_evaluate_examples, batch_size):
        """
        
        Args:
            editor (Editor)
            examples (list[EditExample])
            num_evaluate_examples (int)
            batch_size (int)

        Returns:
            stats (dict[str, float])
            edit_traces (list[EditTrace]): of length num_evaluate_examples
            loss_traces (list[LossTrace]): of length num_evaluate_examples

        """
        sample = sample_if_large(examples, num_evaluate_examples, replace=False)

        # compute loss
        # need to break the sample into batches, in case the sample is too large to fit in GPU memory
        losses, loss_traces, weights, enc_losses = [], [], [], []

        for batch in verboserate(chunks(sample, batch_size), desc='Computing loss on examples'):
            weights.append(len(batch))
            loss_var, loss_trace_batch, enc_loss = editor.loss(batch)

            # convert loss Variable into float
            loss_val = loss_var.item()
            assert isinstance(loss_val, float)
            losses.append(loss_val)
            enc_losses.append(enc_loss)

            loss_traces.extend(loss_trace_batch)

        losses, weights = np.array(losses), np.array(weights)
        loss = np.sum(losses * weights) / np.sum(weights)  # weighted average
        enc_loss = np.sum(np.array(enc_losses) * weights) / np.sum(weights)

        punct_table = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P'))

        def remove_punct(s):
            new_s = []
            for t in s:
                t = unicode(t).translate(punct_table)
                if t != '':
                    new_s.append(t)
            return new_s

        metrics = {
            'bleu': (bleu, max),
            'edit_dist': (lambda s, t: edit_dist(s, t)[0] / len(s) if len(s) > 0 else len(t), min),
            'exact_match': (lambda s, t: 1.0 if remove_punct(s) == remove_punct(t) else 0.0, max)
        }

        top_results = defaultdict(list)
        top5_results = defaultdict(list)

        # compute predictions
        beams, edit_traces = editor.edit(sample, batch_size=batch_size, max_seq_length=150, verbose=True)
        for ex, beam in izip(sample, beams):
            top = beam[0]
            top5 = beam[:5]
            target = ex.target_words
            for name, (fxn, best) in metrics.items():
                top_results[name].append(fxn(top, target))
                top5_results[name].append(best(fxn(predict, target) for predict in top5))

        # compute averages
        stats_top = {name: np.mean(vals) for name, vals in top_results.items()}
        stats_top5 = {'{}_top5'.format(name): np.mean(vals) for name, vals in top5_results.items()}

        # combine into a single stats object
        stats = {'loss': loss, 'enc_loss': enc_loss}
        stats.update(stats_top)
        stats.update(stats_top5)

        return stats, edit_traces, loss_traces

    def _write_traces(self, split_name, train_steps, edit_traces, loss_traces):
        trace_dir = join(self.workspace.traces, split_name)
        trace_path = join(trace_dir, '{}.txt'.format(train_steps))
        makedirs(trace_dir)

        with codecs.open(trace_path, 'w', encoding='utf-8') as f:
            for edit_trace, loss_trace in zip(edit_traces, loss_traces):
                f.write(unicode(edit_trace))
                f.write('\n')
                f.write(unicode(loss_trace))
                f.write('\n\n')
