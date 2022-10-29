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

# PSG BZD
# 100  12
#  50  12
#  20  16
#  10  32

STEP=210
CKPT='09081203-MFID-AQG-b64psg25gd2-disa0-e20-lr5e5'
OUT_HOME='/home/ubuntu/data/MyFusionInDecoderOut'

OUT_DIR="${OUT_HOME}/${CKPT}/result-step${STEP}"
mkdir -p ${OUT_DIR}
PSG=10
BZ=32
GPUID=7

$PYT cli.py \
--task=qg \
--predict_file=ambigqa/dev.json \
--output_dir=${OUT_DIR} \
--do_predict=True \
--ambigqa=True \
--wiki_2020=True \
--bert_name=bart-large \
--max_question_length=32 \
--n_jobs=96 \
--psg_sel_dir=08230210-RRK-NQ-DPRmultiset-M32-b16-lr1e5 \
--top_k_passages=${PSG} \
--use_reranker=True \
--decoder_start_token_id=2 \
--predict_batch_size=${BZ} \
--verbose=True \
--use_gpu_ids=${GPUID} \
--checkpoint="${OUT_HOME}/${CKPT}/output/out/model-step${STEP}.pt"



