#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


PYT=/home/ubuntu/anaconda3/envs/AQ/bin/python

# BZ PSG GD BZD
# 64 100  8  12
# 64  50  4  12
# 64  20  2  16
# 64  10  1  16

CKPT_ALL=/home/ubuntu/data/MyFusionInDecoderOut/08300303-MFID-NQ-b64psg100gd8-lr2e5/output/out/best-model.pt

PSG=$1
BZP=$2
GD=$3

$PYT cli.py \
--task=qa_noamb_aq \
--train_file=ambigqa/train.json \
--predict_file=ambigqa/dev.json \
--output_dir=aq-fid \
--do_train=True \
--ambigqa=True \
--wiki_2020=True \
--bert_name=bart-large \
--checkpoint=$CKPT_ALL \
--max_answer_length=20 \
--n_jobs=96 \
--psg_sel_dir=08230210-RRK-NQ-DPRmultiset-M32-b16-lr1e5 \
--discard_not_found_answers=True \
--top_k_passages=${PSG} \
--use_reranker=True \
--decoder_start_token_id=2 \
--train_batch_size=64 \
--predict_batch_size=${BZP} \
--learning_rate=1e-5 \
--warmup_proportion=0.1 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--gradient_accumulation_steps=${GD} \
--num_train_epochs=10 \
--wait_step=1000000 \
--verbose=True \
--eval_period=10 \
--use_gpu_ids=0,1,2,3,4,5,6,7
