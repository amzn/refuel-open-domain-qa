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

# 10 16
# 20 8
# 25 6
# 50 3
# 100 1

PSG=100
BZ=8
GPU=$1
OUT=$2
PSG_DIR=$3
CKPT=$4

$PYT cli.py \
--task=qg_mask \
--predict_file=nqopen/dev.json \
--output_dir=${OUT} \
--do_predict=True \
--bert_name=bart-large \
--max_question_length=32 \
--n_jobs=96 \
--psg_sel_dir=${PSG_DIR} \
--top_k_passages=${PSG} \
--use_reranker=True \
--decoder_start_token_id=2 \
--predict_batch_size=${BZ} \
--verbose=True \
--use_gpu_ids=${GPU} \
--checkpoint=${CKPT}