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

#09010230-MFID-AQG-b64psg10gd1-disa0-lr5e5-nockpt


QA_CKPT=$1
QG_CKPT=$2
OG_Pass=$3
DATA_TYPE=$4
IS_REPLACE=0
TOP_K_PASSAGES=100
BZ=64
GPUID=$5

$PYT cli.py \
--task=over_generate \
--predict_file=ambigqa/${DATA_TYPE}.json \
--do_over_generate_predict=True \
--map_ckpt=${QA_CKPT} \
--qd_ckpt=${QG_CKPT} \
--over_generate_pass=${OG_Pass} \
--replace_prompt_question=${IS_REPLACE} \
--ambigqa=True \
--wiki_2020=True \
--bert_name=bart-large \
--max_question_length=32 \
--max_answer_length=64 \
--n_jobs=96 \
--psg_sel_dir=08230210-RRK-NQ-DPRmultiset-M32-b16-lr1e5 \
--top_k_passages=${TOP_K_PASSAGES} \
--use_reranker=True \
--decoder_start_token_id=2 \
--predict_batch_size=${BZ} \
--verbose=True \
--use_gpu_ids=${GPUID}
