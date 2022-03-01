# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from itertools import cycle
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling_gated import BertConfig, BertForSequenceClassification, BertForMultiTask
from optimization import BERTAdam

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"]="3"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and test examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()


    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
            "test_matched")


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples
 

class STSProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")



    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[7])
            text_b = tokenization.convert_to_unicode(line[8])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


class QQPProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            
            if i == 0 or len(line) != 3:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        
        return examples
       
class QNLIProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples

class RTEProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""



    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


class SSTProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""



    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""



    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def _create_examples(self, lines, set_type):
        """Creates examples for the test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, task='none'):
    """Loads a data file into a list of `InputBatch`s."""

    

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        '''if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #if task != 'sts':
            logger.info("label: %s (id = %d)" % (example.label, label_id))'''

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def do_eval(model, device, processor, tokenizer, eval_dataloader, task_id, i, label_list):
    model.eval()
    output = np.empty(0)
    labels = label_list[i]
    output = np.empty(0)
    for input_ids, input_mask, segment_ids in eval_dataloader:
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, i, task_id)
        if task_id == 'cola':
            logits = logits.detach()
            _, logits = logits.max(dim=1)
            logits = logits.detach().cpu().numpy()   
            output = np.concatenate((output, logits), axis = 0) 
        elif task_id == 'sts':
            logits = logits.detach()
            logits = logits.squeeze(-1).data.cpu().tolist()
            logits = [min(max(0., pred * 1.), 1.) for pred in logits]
            output = np.concatenate((output, np.array(logits)), axis = 0)
        else:
            logits = logits.detach().cpu().numpy()  
            logits = np.argmax(logits, axis = 1)
            logits = np.array([labels[int(id)] for id in logits])
            output = np.concatenate((output, logits), axis = 0)
    
    if task_id == 'cola':
        output = pd.DataFrame(output)
    elif task_id == 'sts':
        output = pd.DataFrame(output)
    else:
        output = pd.DataFrame(output)
    output.columns =["prediction"]
    
    pd.DataFrame(output).to_csv(task_id+"_output.csv")


parser = argparse.ArgumentParser()

    ## Required parameters
parser.add_argument("--model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The test model path.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()    
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "rte": RTEProcessor,
    "sts": STSProcessor,
    "sst": SSTProcessor,
    "qqp": QQPProcessor,
    "qnli": QNLIProcessor,
}

task_names =['cola', 'mrpc', 'mnli', 'rte', 'sts', 'sst', 'qqp', 'qnli']
data_dirs = ['CoLA', 'MRPC', 'MNLI', 'RTE', 'STS-B', 'SST-2', 'QQP', 'QNLI']

if task_names[0] not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor_list = [processors[task_name]() for task_name in task_names]


tokenizer = tokenization.FullTokenizer(
    vocab_file="init_bert/uncased_L-12_H-768_A-12/vocab.txt", do_lower_case=True)

train_examples = None
num_train_steps = None
num_tasks = len(task_names)

eval_loaders = []
for i, task in enumerate(task_names):
    print("Constructing...")
    print(i, task)
    eval_examples = processor_list[i].get_test_examples(os.path.join("data/glue", data_dirs[i]))
    eval_features = convert_examples_to_features(eval_examples, 128, tokenizer, task)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_loaders.append(DataLoader(eval_data, sampler=eval_sampler, batch_size=32))

global_step = 0
bert_config = BertConfig.from_json_file("configs/embert_config.json")
label_list = [['0', '1'], ['0', '1'], ['contradiction', 'entailment', 'neutral'], ['not_entailment', 'entailment'], ['None'], ['0', '1'], ['0', '1'], ['not_entailment', 'entailment']]
bert_config.num_tasks = 8
model = BertForMultiTask(bert_config, [len(labels) for labels in label_list])
model.load_state_dict(torch.load(args.model_dir))

model.to(device)
for i, task in enumerate(task_names):
    print(i, task)
    do_eval(model, device, processor_list[i], tokenizer, eval_loaders[i], task, i, label_list)
