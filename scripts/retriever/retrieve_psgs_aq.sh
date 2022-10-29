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
HOME=/home/ubuntu/data

split=$1
GPU=$2

$PYT cli.py \
--bert_name=bert-base-uncased \
--do_predict=True \
--task=dpr \
--predict_batch_size=512 \
--predict_file=ambigqa/${split}.json \
--verbose=True \
--use_gpu_ids=${GPU} \
--dpr_retrieval_topk_psgs=1000 \
--dpr_checkpoint=multiset \
--wiki_2020=True \
--ambigqa=True