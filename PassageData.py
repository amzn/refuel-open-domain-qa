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


import os
import pickle as pkl
import gzip
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from ambigqa_evaluate_script import normalize_answer
from DataLoader import MyDataLoader

class PassageData(object):
    def __init__(self, logger, args, tokenizer):
        self.logger = logger
        self.args = args
        self.data_path = os.path.join(args.dpr_data_dir, "wikipedia_split/psgs_w100{}.tsv.gz".format("_20200201" if args.wiki_2020 else ""))

        self.passages = None
        self.titles = None
        self.tokenizer = tokenizer
        self.tokenized_data = None

    def load_db(self):
        if not self.args.skip_db_load:
            data = []
            with gzip.open(self.data_path, "rb") as f:
                for line in f:
                    data.append(line.decode().strip().split("\t"))
                    if self.args.debug and len(data)==100:
                        break
            assert all([len(d)==3 for d in data])
            assert data[0]==["id", "text", "title"]
            self.passages = {int(d[0])-1:d[1].lower() for d in data[1:]}
            self.titles = {int(d[0])-1:d[2].lower() for d in data[1:]}
            self.logger.info("Loaded {} passages".format(len(self.passages)))

    def load_tokenized_data(self, model_name, all=False, do_return=False, index=None):
        if all:
            tokenized_data = {"input_ids": [], "attention_mask": []}
            for index in range(10):
                curr_tokenized_data = self.load_tokenized_data(model_name, all=False, do_return=True, index=index)
                tokenized_data["input_ids"] += curr_tokenized_data["input_ids"]
                tokenized_data["attention_mask"] += curr_tokenized_data["attention_mask"]
        else:
            index=self.args.db_index if index is None else index
            assert 0<=index<10
            if model_name=="bert":
                cache_path = self.data_path.replace(".tsv.gz", "_{}_BertTokenized.pkl".format(index))
            elif model_name=="albert":
                cache_path = self.data_path.replace(".tsv.gz", "_{}_AlbertTokenized.pkl".format(index))
            elif model_name=="bart":
                cache_path = self.data_path.replace(".tsv.gz", "_{}_BartTokenized.pkl".format(index))
            elif model_name=="t5":
                cache_path = self.data_path.replace(".tsv.gz", "_{}{}_T5Tokenized.pkl".format("reos_" if self.args.t5_no_intermediate_eos else "", index))
            else:
                raise NotImplementedError(model_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    tokenized_data = pkl.load(f)
            else:
                assert not self.args.skip_db_load
                if self.titles is None or self.passages is None:
                    self.load_db()
                # tokenize 2.2M for each thread
                psgs_per_thread = 2500000 if self.args.wiki_2020 else 2200000
                min_idx = index*psgs_per_thread
                max_idx = min(len(self.titles), (index+1)*psgs_per_thread)
                if self.args.pycharm_debug:
                    psgs_per_thread = 250000 if self.args.wiki_2020 else 220000
                    min_idx = index * psgs_per_thread  # Yifan: for debug
                    max_idx = min(len(self.titles), (index + 1) * psgs_per_thread)
                self.logger.info("Start tokenizing from {} to {}".format(min_idx, max_idx))
                if self.args.bert_name.startswith("t5"):
                    if self.args.t5_no_intermediate_eos:
                        input_data = ["title: " + self.titles[_id] + " context: " + self.passages[_id] + " </s>" for _id in range(min_idx, max_idx)]
                    else:
                        input_data = ["title: " + self.titles[_id] + " </s>" + " context: " + self.passages[_id] + " </s>" for _id in range(min_idx, max_idx)]
                else:
                    input_data = [self.titles[_id] + " " + self.tokenizer.sep_token + " " + self.passages[_id]
                                for _id in range(min_idx, max_idx)]
                tokenized_data = self.tokenizer.batch_encode_plus(input_data,
                        max_length=128,
                        pad_to_max_length=True)
                with open(cache_path, "wb") as f:
                    pkl.dump({"input_ids": tokenized_data["input_ids"],
                              "attention_mask": tokenized_data["attention_mask"]}, f)

        self.tokenized_data = tokenized_data
        self.logger.info("Finish loading {} {} tokenized data".format(len(tokenized_data["input_ids"]), model_name))
        if do_return:
            return tokenized_data

    def load_dataset(self, model_name, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data("bert", index=self.args.db_index)
        tokenized_data = self.tokenized_data
        assert tokenized_data is not None
        input_ids = torch.LongTensor(tokenized_data["input_ids"])
        attention_mask = torch.LongTensor(tokenized_data["attention_mask"])
        print (model_name, input_ids.size(), attention_mask.size())
        self.dataset = TensorDataset(input_ids, attention_mask)
        if do_return:
            return self.dataset

    def load_dataloader(self, batch_size, is_training=None, do_return=False):
        self.dataloader = MyDataLoader(self.args,
                                       self.dataset,
                                       batch_size=batch_size,
                                       is_training=self.is_training if is_training is None else is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, answers):
        if self.args.skip_db_load:
            return [0]
        if self.passages is None:
            self.load_db()
        assert len(predictions)==len(answers)
        assert not self.args.skip_db_load
        recall = defaultdict(list)
        k_list = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for (pids, answer) in tqdm(zip(predictions, answers), total=len(predictions)):
            passages = [normalize_answer(self.passages[pid]) for pid in pids]
            if self.args.ambigqa and not self.args.leaderboard:
                normalized_answer = []
                for ann in answer:
                    for ans_cluster in ann:
                        normalized_answer.extend([normalize_answer(a) for a in ans_cluster])
                answer = normalized_answer
            else:
                answer = [normalize_answer(a) for a in answer]
            curr_recall = [any([a in p for a in answer]) for p in passages]
            for k in k_list:
                recall[k].append(any(curr_recall[:k]))
        for k in k_list:
            self.logger.info("Recall @ %d\t%.3f" % (k, np.mean(recall[k])))
        return recall[100]


