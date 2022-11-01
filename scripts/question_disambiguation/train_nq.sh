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

PSG=100
GD=8
BZD=4

PSG_DIR=$1

$PYT cli.py \
--task=qg_mask \
--train_file=nqopen/train.json \
--predict_file=nqopen/dev.json \
--output_dir=question_disambiguation_nq \
--do_train=True \
--bert_name=bart-large \
--max_question_length=32 \
--n_jobs=96 \
--psg_sel_dir=${PSG_DIR} \
--top_k_passages=${PSG} \
--use_reranker=True \
--decoder_start_token_id=2 \
--train_batch_size=64 \
--predict_batch_size=${BZD} \
--learning_rate=1e-5 \
--warmup_proportion=0.1 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--gradient_accumulation_steps=${GD} \
--num_train_epochs=10 \
--wait_step=100000000 \
--verbose=True \
--eval_period=600 \
--use_gpu_ids=0,1,2,3,4,5,6,7 \
--skip_inference=True



