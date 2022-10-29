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

GPUID=$1
PSG=$2
BZ=$3
OUT=$3

$PYT cli.py \
--task=qa_noamb_aq \
--predict_file=ambigqa/dev.json \
--do_predict=True \
--output_dir=${OUT} \
--ambigqa=True \
--wiki_2020=True \
--bert_name=bart-large \
--checkpoint=/home/ubuntu/data/MyFusionInDecoderOut/${OUT}/output/out/best-model.pt \
--max_answer_length=20 \
--n_jobs=96 \
--psg_sel_dir=08230210-RRK-NQ-DPRmultiset-M32-b16-lr1e5 \
--top_k_passages=${PSG} \
--use_reranker=True \
--decoder_start_token_id=2 \
--predict_batch_size=${BZ} \
--verbose=True \
--use_gpu_ids=${GPUID}

