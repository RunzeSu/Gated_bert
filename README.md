# PyTorch implementation of EmBERT

This is the original implementation of EmBERT model from the paper ["Multitask Learning Using BERT with Task-Embedded Attention"](https://ieeexplore.ieee.org/abstract/document/9533990).
Our code is strongly based on the [BERT and PALs implementation](https://github.com/AsaCooperStickland/Bert-n-Pals) of Asa Cooper Stickland and Iain Murray.


## Models weights

Below one can find weights for the following models:
- [BERT pretrained weights](https://drive.google.com/drive/folders/1fVbmGvq1qfpxMrPK0GQGn-K8-PNkWkyV?usp=sharing) - we used the `uncased_L-12_H-768_A-12` model's weights [shared by Google](https://github.com/google-research/bert#pre-trained-models).
- [EmBERT weights](https://drive.google.com/drive/folders/1fp6S3EwsZWt3eUaIlNYm-pF16m3hWB0o?usp=sharing).

Moreover we share a file with [EmBERT GLUE submission](https://drive.google.com/file/d/1_WogoIgbgnyD9NvJgaR95OAQW6caB6nV/view?usp=sharing). 

## EmBERT training

In `configs/embert_config.json` one can find the config needed to train the EmBERT model.

- `run_multi_task.py` is a script that runs a multitask model training.
- `run_test_multi_task` is a script that returns the model predictions on [GLUE](gluebenchmark.com) benchmark.

Below one can see, how to run EmBERT training
```shell

export BERT_DIR=init_bert/uncased_L-12_H-768_A-12
export CONFIG_DIR=init_bert/uncased_L-12_H-768_A-12
export GLUE_DIR=data/glue
export SAVE_DIR=saved/em_bert


python run_multi_task.py \
  --seed 1 \
  --output_dir $SAVE_DIR/embert \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file configs/embert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 25.0 \
  --gradient_accumulation_steps 1

```


Gated_BERT is proposed and modified based on the original codes of EMBERT. To train the model, under the same environment, use

```shell

export BERT_DIR=init_bert/uncased_L-12_H-768_A-12
export CONFIG_DIR=init_bert/uncased_L-12_H-768_A-12
export GLUE_DIR=data/glue
export SAVE_DIR=saved/gated_bert


python run_multi_task_gated.py \
  --seed 1 \
  --output_dir $SAVE_DIR/gatedbert \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file configs/embert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 25.0 \
  --gradient_accumulation_steps 1
