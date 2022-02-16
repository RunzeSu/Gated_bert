import csv
from itertools import cycle
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling_gated import BertConfig, BertForSequenceClassification, BertForMultiTask
from optimization_gated import BERTAdam
from torch.utils.tensorboard import SummaryWriter
from run_multi_task_gated import ColaProcessor, MnliProcessor, MrpcProcessor, RTEProcessor, STSProcessor, SSTProcessor, QQPProcessor, QNLIProcessor
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

bert_config_file = "configs/embert_config.json"
bert_config = BertConfig.from_json_file(bert_config_file)

task_names =['cola', 'mrpc', 'mnli', 'rte', 'sts', 'sst', 'qqp', 'qnli']
data_dirs = ['CoLA', 'MRPC', 'MNLI', 'RTE', 'STS-B', 'SST-2', 'QQP', 'QNLI']
processor_list = [processors[task_name]() for task_name in task_names]
label_list = [processor.get_labels() for processor in processor_list]

bert_config.num_tasks = 8


"""
model = BertForMultiTask(bert_config, [len(labels) for labels in label_list])
partial = torch.load("init_bert/uncased_L-12_H-768_A-12/pytorch_model.bin", map_location='cpu')
model_dict = model.bert.state_dict()
#print(model_dict.keys())
update = {}
for n, p in model_dict.items():
    if 'aug' in n or 'mult' in n:
        update[n] = p
        if 'pooler.mult' in n and 'bias' in n:
            update[n] = partial['pooler.dense.bias']
        if 'pooler.mult' in n and 'weight' in n:
            update[n] = partial['pooler.dense.weight']
    else:
        if ('_list' in n) or ('task_emb' in n):
            update[n] = model_dict[n]
        else:
            if ('weight_layer' in n):
                update[n] = torch.zeros(model_dict[n].shape)
            else:
                update[n] = partial[n]

model.bert.load_state_dict(update)
        
"""
model = torch.load("/mnt/ufs18/home-052/surunze/Gated_bert/saved/gated_bert/gatedbert/1", map_location='cpu')
