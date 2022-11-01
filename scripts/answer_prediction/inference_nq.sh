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
# 64  25  2  12
# 64  20  2  16
# 64  10  1  16

GPUID=$1

TK=100
BZD=12
split=$2
OUT=$3
CKPT=$OUT/output/out/best-model.pt
RRK_DIR=$4

$PYT cli.py \
--do_predict=True \
--task=qa \
--output_dir=$OUT \
--predict_file=nqopen/${split}.json \
--bert_name=bart-large \
--max_answer_length=16 \
--psg_sel_dir=${RRK_DIR} \
--top_k_passages=$TK \
--predict_batch_size=$BZD \
--verbose=True \
--n_jobs=96 \
--use_gpu_ids=$GPUID \
--decoder_start_token_id=2 \
--checkpoint=$CKPT \
--use_reranker=True


