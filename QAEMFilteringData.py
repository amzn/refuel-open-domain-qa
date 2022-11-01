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

from DataLoader import MySimpleQADataset, MyQADataset, MyDataLoader, MySimpleQALMFilteringDataset
from util import decode_span_batch

# for evaluation
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics, QAPairEvaluation
from pycocoevalcap.bleu.bleu import Bleu
from QAData import QAData
from copy import deepcopy
import numpy as np

class AmbigQAEMFilteringData():
    def __init__(self, logger, args, data_path, is_training, passages=None):
        self.data_path = data_path
        self.passages = passages

        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        else:
            raise NotImplementedError()

        with open(self.data_path, "r") as f:
            self.data = json.load(f)

        # convert list of qa pairs into tuples
        for d in self.data:
            for pass_idx in range(args.over_generate_pass+1):
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
        questions = [[v[0] for v in d["over_generate_{}_noambq_answer".format(self.args.over_generate_pass)]] for d in self.data]
        answers = [[v[1] for v in d["over_generate_{}_noambq_answer".format(self.args.over_generate_pass)]] for d in self.data]
        questions, answers, metadata = self.flatten(questions, answers)
        self.input_questions = questions
        self.input_answers = answers

        if self.args.do_lowercase:
            questions = [question.lower() for question in questions]
            answers = [answer.lower() for answer in answers]

        question_input = tokenizer.batch_encode_plus(questions,
                                                     pad_to_max_length=True,
                                                     max_length=32)
        # answer_input = tokenizer.batch_encode_plus(answers,
        #                                            pad_to_max_length=True,
        #                                            max_length=20)

        input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
        # decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
        tokenized_data = [input_ids, attention_mask, metadata]

        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    # override
    def flatten(self, questions, answers):
        new_questions, new_answers, metadata = [], [], []
        for question, answer in zip(questions, answers):
            assert type(answer)==list
            assert len(question) == len(answer)
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
            new_questions += question
        return new_questions, new_answers, metadata

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
            assert len(fg_passages) == len(dpr_predictions_tokenized_input_ids)
            dpr_predictions_tokenized_input_ids = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_predictions_tokenized_input_ids, fg_passages)]
            dpr_predictions_tokenized_attention_mask = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_predictions_tokenized_attention_mask, fg_passages)]
        else:
            raise NotImplementedError
            # dpr_predictions_tokenized_input_ids = [psgs[:100] for psgs in dpr_predictions_tokenized_input_ids]
            # dpr_predictions_tokenized_attention_mask = [psgs[:100] for psgs in dpr_predictions_tokenized_attention_mask]

        input_ids, attention_mask, metadata = self.tokenized_data
        assert len(input_ids)==len(attention_mask)==metadata[-1][-1]
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

        # question - passage (with title)
        qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

        for idx, (dpr_input_ids, dpr_attention_mask, curr_metadata) in enumerate(
                zip(tqdm(dpr_predictions_tokenized_input_ids), dpr_predictions_tokenized_attention_mask, metadata)):
            for question_jdx in range(*curr_metadata):
                curr_input_ids, curr_attention_mask, = input_ids[question_jdx], attention_mask[question_jdx]
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [pad_token_id for _ in range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[question_jdx].append(qp_inputs_ids_idx_jdx[:160])
                    qp_attention_mask[question_jdx].append(qp_attention_mask_idx_jdx[:160])
                    assert len(qp_input_ids[question_jdx][jdx]) == len(qp_attention_mask[question_jdx][jdx]) == 160  # here we use 32+128
                assert len(qp_input_ids[question_jdx]) == len(qp_attention_mask[question_jdx])

        assert len(qp_input_ids) == len(qp_attention_mask) == len(input_ids) == len(attention_mask)
        qp_input_ids = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        qp_attention_mask = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]
        self.tokenized_data[0] = qp_input_ids
        self.tokenized_data[1] = qp_attention_mask
        self.tokenized_data[2] = metadata

    # override
    def evaluate(self, predicted_answers):
        reference = deepcopy(self.data)

        metadata = self.tokenized_data[-1]
        input_questions = self.input_questions
        input_answers = self.input_answers
        is_same = [normalize_answer(_ia) == normalize_answer(_pa) for _ia, _pa in zip(input_answers, predicted_answers)]
        print('{:.2f} answers are matched!'.format(np.mean(is_same)*100))

        predictions = {}
        num_answers_per_sample = []
        for idx, (m, d) in enumerate(zip(metadata, self.data)):
            curr_qa_pairs = {}
            if any(is_same[m[0]:m[1]]):
                for qa_jdx in range(*m):
                    curr_question = input_questions[qa_jdx]
                    curr_answer = normalize_answer(input_answers[qa_jdx])
                    curr_is_same = is_same[qa_jdx]
                    if curr_is_same and (curr_answer not in curr_qa_pairs.keys() or len(curr_question) > len(curr_qa_pairs[curr_answer])):
                        curr_qa_pairs[curr_answer] = curr_question
            else:
                for qa_jdx in range(*m):
                    curr_question = input_questions[qa_jdx]
                    curr_answer = normalize_answer(input_answers[qa_jdx])
                    if curr_answer not in curr_qa_pairs.keys() or len(curr_question) > len(curr_qa_pairs[curr_answer]):
                        curr_qa_pairs[curr_answer] = curr_question
            predictions[d['id']] = [{'question': x[1], 'answer': x[0]} for x in curr_qa_pairs.items()]
            num_answers_per_sample.append(len(predictions[d['id']]))
        print('On average {:.2f} answers per sample'.format(np.mean(num_answers_per_sample)))
        evaluation = QAPairEvaluation(deepcopy(reference), deepcopy(predictions))
        results = evaluation.print_all_metrics(verbose=False)
        print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(results['F1 answer'],
                                                                                      results['F1 answer (multi)'],
                                                                                      results["F1 bleu4"],
                                                                                      results["F1 edit-f1"]))



