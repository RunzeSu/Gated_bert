#!/bin/bash --login

#SBATCH --qos=normal
#SBATCH --time=30:00:00
#SBATCH --mem=40G  
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm-%A_%a.out

cd /mnt/ufs18/home-052/surunze/eg_multitask/
source eg_env/bin/activate
module purge
module load GCCcore/11.1.0
module load Python/3.9.6
cd /mnt/ufs18/home-052/surunze/Gated_bert/

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