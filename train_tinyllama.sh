#!/bin/bash

# GPU configuration
GPUS_PER_NODE=1
  # Adjust based on your GPUs
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# Paths (relative to Megatron-LM directory)
CHECKPOINT_PATH=./checkpoints/tinyllama-code
mkdir -p $CHECKPOINT_PATH
VOCAB_FILE=./vocab.json
MERGE_FILE=./merges.txt
DATA_PATH=./code_dataset_text_document

# TinyLlama config
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 22 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --micro-batch-size 2 \
       --global-batch-size 32 \
       --lr 4.0e-4 \
       --train-iters 10000 \
       --lr-decay-iters 10000 \
       --lr-decay-style cosine \
       --lr-warmup-iters 500 \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --fp16 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --split 98,2,0 \
       --log-interval 10 \
       --save-interval 1000 \
       --eval-interval 500 \
       --eval-iters 10 \
       --tensorboard-dir ./tensorboard
