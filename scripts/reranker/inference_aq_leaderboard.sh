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

outdir=$1
dpr_ckpt=multiset
split=test

$PYT cli.py \
--task=rrk \
--predict_file=ambigqa/${split}.json \
--output_dir=${outdir} \
--do_predict=True \
--bert_name=bert-large-uncased \
--checkpoint=/home/ubuntu/data/MyFusionInDecoderOut/${outdir}/output/out/best-model.pt \
--dpr_checkpoint=${dpr_ckpt} \
--test_M=1000 \
--n_jobs=96 \
--predict_batch_size=16 \
--verbose=True \
--use_gpu_ids=0,1,2,3,4,5,6,7 \
--ambigqa=True \
--wiki_2020=True \
--leaderboard=True