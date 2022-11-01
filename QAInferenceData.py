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
import json
import gzip
import re
import pickle as pkl
import string
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import itertools

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from DataLoader import MySimpleQADataset, MyQADataset, MyDataLoader
from util import decode_span_batch

# for evaluation
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
from pycocoevalcap.bleu.bleu import Bleu
from QAData import QAData

class AmbigQAInferenceData():
    def __init__(self, logger, args, data_path, is_training, passages=None):
        self.data_path = data_path
        self.passages = passages

        if "dev_test" in self.data_path:
            self.data_type = "dev_test"
        elif "dev_dev" in self.data_path:
            self.data_type = "dev_dev"
        elif "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        else:
            raise NotImplementedError()

        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        # convert list of qa pairs into tuples
        for d in self.data:
            for pass_idx in range(args.over_generate_pass):
                d['over_generate_{}_prompt_answer'.format(pass_idx)] = [tuple(x) for x in d['over_generate_{}_prompt_answer'.format(pass_idx)]]
                d['over_generate_{}_noambq_answer'.format(pass_idx)] = [tuple(x) for x in d['over_generate_{}_noambq_answer'.format(pass_idx)]]

        assert type(self.data) == list
        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.tokenizer = None
        self.tokenized_data = None
        self.dpr_tokenized_data = None
        self.dataset = None
        self.dataloader = None
        self.metric = "F1"
        self.SEP = "<SEP>"

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        if type(tokens[0])==list:
            return [self.decode(_tokens) for _tokens in tokens]
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip().replace(" - ", "-").replace(" : ", ":")

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)

        input_ids, attention_mask, metadata = self.tokenized_data
        self.dataset = MySimpleQADataset(input_ids,
                                         attention_mask,
                                         is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False, **kwargs):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training, **kwargs)
        if do_return:
            return self.dataloader

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix or 'T5' in postfix

        print ("Start tokenizing...")
        if self.args.over_generate_pass == 0:
            questions = [[d["question"]] for d in self.data]
        else:
            questions = [[v[0] for v in d["over_generate_{}_noambq_answer".format(self.args.over_generate_pass-1)]] for d in self.data]
        questions, question_metadata = self.flatten_question(questions)
        self.map_input_questions = questions
        if self.args.bert_name.startswith("t5"):
            if self.args.t5_no_intermediate_eos:
                questions = ["question: " + question for question in questions]
            else:
                questions = ["question: " + question + " </s>" for question in questions]
        if self.args.do_lowercase:
            questions = [question.lower() for question in questions]
        if self.args.append_another_bos:
            questions = ["<s> "+question for question in questions]
        question_input = tokenizer.batch_encode_plus(questions,
                                                     pad_to_max_length=True,
                                                     max_length=32)
        input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
        tokenized_data = [input_ids, attention_mask, question_metadata]

        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    # override
    def flatten_question(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type + "_20200201" if self.args.wiki_2020 else self.data_type,
            "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, "ambigqa", "{}{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "", ))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}_dprpassages.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
        if "Bart" in postfix:
            self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info(dpr_tokenized_path)

        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR tokenized data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                dpr_predictions_tokenized = json.load(f)
        else:
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)

            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)

            dpr_predictions_tokenized = {"input_ids": [], "attention_mask": []}
            for dpr_ids in dpr_passages:
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                dpr_predictions_tokenized["input_ids"].append(dpr_input_ids)
                dpr_predictions_tokenized["attention_mask"].append(dpr_attention_mask)

            with open(dpr_tokenized_path, "w") as f:
                json.dump(dpr_predictions_tokenized, f)
            self.logger.info("Saving DPR tokenized data Done {}".format(dpr_tokenized_path))
            # exit()
        dpr_predictions_tokenized_input_ids, dpr_predictions_tokenized_attention_mask = dpr_predictions_tokenized["input_ids"], dpr_predictions_tokenized["attention_mask"],

        if self.args.use_reranker:
            assert self.args.psg_sel_dir is not None
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}{}_psg_sel.json".format(self.data_type.replace("train", "train_for_inference"),
                                                                   "_20200201" if self.args.wiki_2020 else "",
                                                                   "_aq" if self.args.ambigqa else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
            assert len(fg_passages)==len(dpr_predictions_tokenized_input_ids)
            dpr_predictions_tokenized_input_ids = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_predictions_tokenized_input_ids, fg_passages)]
            dpr_predictions_tokenized_attention_mask = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_predictions_tokenized_attention_mask, fg_passages)]
        else:
            raise NotImplementedError
            # dpr_predictions_tokenized_input_ids = [psgs[:100] for psgs in dpr_predictions_tokenized_input_ids]
            # dpr_predictions_tokenized_attention_mask = [psgs[:100] for psgs in dpr_predictions_tokenized_attention_mask]

        self.dpr_reranked_tokenized_data = (dpr_predictions_tokenized_input_ids, dpr_predictions_tokenized_attention_mask)

        input_ids, attention_mask, input_metadata = self.tokenized_data
        assert len(input_ids)==len(attention_mask)==input_metadata[-1][-1]
        assert len(dpr_predictions_tokenized_input_ids) == len(dpr_predictions_tokenized_attention_mask) == len(input_metadata)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

        # question - passage (with title)
        qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

        for idx, (dpr_input_ids, dpr_attention_mask, curr_input_metadata) in enumerate(
                zip(tqdm(dpr_predictions_tokenized_input_ids), dpr_predictions_tokenized_attention_mask, input_metadata)):
            for question_jdx in range(*curr_input_metadata):
                curr_input_ids, curr_attention_mask, = input_ids[question_jdx], attention_mask[question_jdx]
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[question_jdx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[question_jdx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[question_jdx][jdx]) == len(qp_attention_mask[question_jdx][jdx]) == 160  # here we use 32+128
                assert len(qp_input_ids[question_jdx]) == len(qp_attention_mask[question_jdx])

        assert len(qp_input_ids) == len(qp_attention_mask) == len(input_ids) == len(attention_mask)
        qp_input_ids = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        qp_attention_mask = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]
        self.tokenized_data = [qp_input_ids, qp_attention_mask, input_metadata]



