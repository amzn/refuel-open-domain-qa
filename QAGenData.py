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
import itertools
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from DataLoader import MySimpleQADataset, MyQADataset, MyDataLoader, MyQAGenDataset, MyQAGenDataLoader
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics, QAPairEvaluation

class QAGenPassageData(object):
    def __init__(self, logger, args, tokenizer):
        self.logger = logger
        self.args = args
        self.tokenizer = tokenizer

        self.wiki_nq_path = os.path.join(args.dpr_data_dir, "wikipedia_split/psgs_w100.tsv.gz")
        self.wiki_aq_path = os.path.join(args.dpr_data_dir, "wikipedia_split/psgs_w100_20200201.tsv.gz")

        self.nq_passages = None
        self.nq_titles = None
        self.nq_tokenized_data = None

        self.aq_passages = None
        self.aq_titles = None
        self.aq_tokenized_data = None

    def load_db(self, mode=None):
        assert mode in ['nq', 'aq']
        data_path = self.wiki_nq_path if mode == 'nq' else self.wiki_aq_path
        data = []
        with gzip.open(data_path, "rb") as f:
            for line in f:
                data.append(line.decode().strip().split("\t"))
        assert all([len(d)==3 for d in data])
        assert data[0]==["id", "text", "title"]
        if mode == 'nq':
            self.nq_passages = {int(d[0])-1:d[1].lower() for d in data[1:]}
            self.nq_titles = {int(d[0])-1:d[2].lower() for d in data[1:]}
            self.logger.info("Loaded {} passages".format(len(self.nq_passages)))
            return self.nq_titles, self.nq_passages
        else:
            self.aq_passages = {int(d[0])-1:d[1].lower() for d in data[1:]}
            self.aq_titles = {int(d[0])-1:d[2].lower() for d in data[1:]}
            self.logger.info("Loaded {} passages".format(len(self.aq_passages)))
            return self.aq_titles, self.aq_passages

    def load_tokenized_data(self, model_name, all=False, do_return=False, index=None, mode=None):
        assert mode in ['nq', 'aq']
        data_path = self.wiki_nq_path if mode == 'nq' else self.wiki_aq_path
        if all:
            tokenized_data = {"input_ids": [], "attention_mask": []}
            for index in range(10):
                curr_tokenized_data = self.load_tokenized_data(model_name, all=False, do_return=True, index=index, mode=mode)
                tokenized_data["input_ids"] += curr_tokenized_data["input_ids"]
                tokenized_data["attention_mask"] += curr_tokenized_data["attention_mask"]
        else:
            index=self.args.db_index if index is None else index
            assert 0<=index<10
            if model_name=="bert":
                cache_path = data_path.replace(".tsv.gz", "_{}_BertTokenized.pkl".format(index))
            elif model_name=="albert":
                cache_path = data_path.replace(".tsv.gz", "_{}_AlbertTokenized.pkl".format(index))
            elif model_name=="bart":
                cache_path = data_path.replace(".tsv.gz", "_{}_BartTokenized.pkl".format(index))
            elif model_name=="t5":
                cache_path = data_path.replace(".tsv.gz", "_{}{}_T5Tokenized.pkl".format("reos_" if self.args.t5_no_intermediate_eos else "", index))
            else:
                raise NotImplementedError(model_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    tokenized_data = pkl.load(f)
            else:
                assert not self.args.skip_db_load
                if mode == 'nq':
                    if self.nq_titles is None or self.nq_passages is None:
                        titles, passages = self.load_db(mode=mode)
                else:
                    if self.aq_titles is None or self.aq_passages is None:
                        titles, passages = self.load_db(mode=mode)

                # tokenize 2.2M for each thread
                psgs_per_thread = 2500000 if mode == 'aq' else 2200000
                min_idx = index * psgs_per_thread
                max_idx = min(len(titles), (index+1)*psgs_per_thread)
                if self.args.pycharm_debug:
                    min_idx = index * 2200  # Yifan: for debug
                    max_idx = min(len(titles), (index + 1) * 2200)
                self.logger.info("Start tokenizing from {} to {}".format(min_idx, max_idx))
                if self.args.bert_name.startswith("t5"):
                    if self.args.t5_no_intermediate_eos:
                        input_data = ["title: " + titles[_id] + " context: " + passages[_id] + " </s>" for _id in range(min_idx, max_idx)]
                    else:
                        input_data = ["title: " + titles[_id] + " </s>" + " context: " + passages[_id] + " </s>" for _id in range(min_idx, max_idx)]
                else:
                    input_data = [titles[_id] + " " + self.tokenizer.sep_token + " " + passages[_id]
                                for _id in range(min_idx, max_idx)]
                tokenized_data = self.tokenizer.batch_encode_plus(input_data,
                        max_length=128,
                        pad_to_max_length=True)
                with open(cache_path, "wb") as f:
                    pkl.dump({"input_ids": tokenized_data["input_ids"],
                              "attention_mask": tokenized_data["attention_mask"]}, f)

        if mode == 'nq':
            self.nq_tokenized_data = tokenized_data
        else:
            self.aq_tokenized_data = tokenized_data
        self.logger.info("Finish loading {} {} {} tokenized data".format(mode, len(tokenized_data["input_ids"]), model_name))
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

    def load_dataloader(self, batch_size, is_training=None, do_return=False, **kwargs):
        self.dataloader = MyDataLoader(self.args,
                                       self.dataset,
                                       batch_size=batch_size,
                                       is_training=self.is_training if is_training is None else is_training, **kwargs)
        if do_return:
            return self.dataloader


class QAGenData(object):

    def __init__(self, logger, args, data_path, is_training, passages=None):
        self.nq_data_path = data_path
        self.aq_data_path = data_path.replace('nqopen', 'ambigqa')
        self.passages = passages

        if "test" in self.nq_data_path:
            self.data_type = "test"
        elif "dev" in self.nq_data_path:
            self.data_type = "dev"
        elif "train" in self.nq_data_path:
            self.data_type = "train" if is_training or args.dpr else "train_for_inference"
        else:
            raise NotImplementedError()

        with open(self.nq_data_path, "r") as f:
            self.nq_data = json.load(f)
        with open(self.aq_data_path, "r") as f:
            self.aq_data = json.load(f)
        assert type(self.nq_data)==type(self.aq_data)==list
        id2answer_path = os.path.join("/".join(self.nq_data_path.split("/")[:-1]), "{}_id2answers.json".format(self.data_type.replace("train_for_inference", "train")))
        with open(id2answer_path, "r") as f:
            id2answers = json.load(f)
        for i, d in enumerate(self.nq_data):
            if is_training:
                for ans in id2answers[d["id"]]:
                    if ans not in self.nq_data[i]["answer"]:
                        self.nq_data[i]["answer"].append(ans)
            else:
                self.nq_data[i]["answer"] = id2answers[d["id"]]

        for i, d in enumerate(self.aq_data):
            answers = []
            disambiguated_questions = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    answers.append([list(set(annotation["answer"]))])
                    disambiguated_questions.append([])
                else:
                    answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
                    disambiguated_questions.append([pair["question"] for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers for _answer in answer for _a in _answer])
            assert len(answers) == len(disambiguated_questions)
            self.aq_data[i]["answer"] = answers
            self.aq_data[i]["disambiguated_question"] = disambiguated_questions

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.metric_map = "MAP-F1"
        self.metric_qd = "QD-EDIT-F1"
        self.SEP = "<SEP>"
        self.QBOS = "<QAGEN-Q>"
        self.ABOS = "<QAGEN-A>"
        self.tokenizer = None
        self.nq_tokenized_data = None
        self.aq_tokenized_data = None
        self.dpr_tokenized_data = None
        self.dataset = None
        self.dataloader = None

    def get_nq_answers(self):
        return [d["answer"] for d in self.nq_data]

    def decode(self, tokens):
        if type(tokens[0])==list:
            return [self.decode(_tokens) for _tokens in tokens]
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip().replace(" - ", "-").replace(" : ", ":")

    def flatten_nq(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def flatten_aq_answer(self, answers):
        new_answers, metadata = [], []
        # per annotator
        for _answers in answers:
            assert type(_answers)==list
            metadata.append([])
            # per answer cluster
            for answer in _answers:
                metadata[-1].append([])
                # per answer
                for _answer in answer:
                    assert len(_answer)>0, _answers
                    assert type(_answer)==list and type(_answer[0])==str, _answers
                    metadata[-1][-1].append((len(new_answers), len(new_answers)+len(_answer)))
                    new_answers += _answer
        return new_answers, metadata

    def flatten_aq_question(self, questions):
        new_questions, metadata = [], []
        for _questions in questions:
            assert type(_questions)==list
            metadata.append([])
            # per annotator
            for _question in _questions:
                metadata[-1].append((len(new_questions), len(new_questions)+len(_question)))
                new_questions += _question
        return new_questions, metadata

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix
        nq_preprocessed_path = os.path.join(
            "/".join(self.nq_data_path.split("/")[:-1]),
            self.nq_data_path.split("/")[-1].replace(".json", "{}-{}.json".format("-uncased" if self.args.do_lowercase else "", postfix)))
        if self.load and os.path.exists(nq_preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(nq_preprocessed_path))
            with open(nq_preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print ("Start tokenizing NQ data...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.nq_data]
            answers = [d["answer"] for d in self.nq_data]
            answers, metadata = self.flatten_nq(answers)
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix or "T5" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = {
                'question_ids': input_ids,
                'question_attention_mask': attention_mask,
                'answer_ids': decoder_input_ids,
                'answer_mask': decoder_attention_mask,
                'answer_metadata': metadata,
            }
            with open(nq_preprocessed_path, "w") as f:
                json.dump(tokenized_data, f)
        self.nq_tokenized_data = tokenized_data

        aq_preprocessed_path = os.path.join(
            "/".join(self.aq_data_path.split("/")[:-1]),
            self.aq_data_path.split("/")[-1].replace(".json", "{}-{}.json".format("-uncased" if self.args.do_lowercase else "", postfix)))
        if self.load and os.path.exists(aq_preprocessed_path):
            self.logger.info("Loading AQ pre-tokenized data from {}".format(aq_preprocessed_path))
            with open(aq_preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print("Start tokenizing AQ data...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"] + "?"
                         for d in self.aq_data]
            disambiguated_questions = [d["disambiguated_question"] for d in self.aq_data]
            disambiguated_questions, disambiguated_question_metadata = self.flatten_aq_question(disambiguated_questions)
            answers = [d["answer"] for d in self.aq_data]
            answers, answer_metadata = self.flatten_aq_answer(answers)
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                disambiguated_questions = [question.lower()  for question in disambiguated_questions]
                answers = [answer.lower() for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            disambiguated_question_input = tokenizer.batch_encode_plus(disambiguated_questions,
                                                                       pad_to_max_length=True,
                                                                       max_length=32)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix or "T5" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            answer_input_ids, answer_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            disambiguated_question_input_ids, disambiguated_question_attention_mask = disambiguated_question_input["input_ids"], disambiguated_question_input["attention_mask"]
            tokenized_data = {
                'question_ids': input_ids,
                'question_attention_mask': attention_mask,
                'answer_ids': answer_input_ids,
                'answer_mask': answer_attention_mask,
                'answer_metadata': answer_metadata,
                'disambiguated_question_ids': disambiguated_question_input_ids,
                'disambiguated_question_mask': disambiguated_question_attention_mask,
                'disambiguated_question_metadata': disambiguated_question_metadata,
            }
            with open(aq_preprocessed_path, "w") as f:
                json.dump(tokenized_data, f)
        self.aq_tokenized_data = tokenized_data

        if not self.args.dpr:
            self.load_dpr_data()

    def load_dpr_data(self):
        nq_dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}_predictions.json".format(self.data_type)).replace('train_for_inference', 'train')
        aq_dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}_predictions.json".format(self.data_type + "_20200201_aq")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, "{}_predictions_rrk{}.json".format(self.data_type, int(self.args.use_reranker)))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_{}.json".format(postfix))
        if "Bart" in postfix:
            return self.load_dpr_data_bart(nq_dpr_retrieval_path, aq_dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError()

    def load_task_3_1_inference_dataset(self, questions, question_metadata):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
        abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
        if self.args.do_lowercase:
            questions = [question.lower() for question in questions]
        question_tokenized = self.tokenizer.batch_encode_plus(questions, pad_to_max_length=False, max_length=32)
        question_input_ids, question_attention_mask = question_tokenized['input_ids'], question_tokenized['attention_mask']
        assert len(questions) == question_metadata[-1][-1]
        assert len(question_metadata) == len(self.dpr_tokenized_data['aq_p_input_ids'])
        input_ids, attention_mask = [], []
        for idx, (curr_question_metadata, curr_aq_p_input_id, curr_aq_p_attention_mask) in enumerate(tqdm(zip(
                question_metadata, self.dpr_tokenized_data['aq_p_input_ids'], self.dpr_tokenized_data['aq_p_attention_mask']
                ), total=len(question_metadata))):
            for question_idx in range(*curr_question_metadata):
                curr_q_input_ids = question_input_ids[question_idx]
                curr_q_attention_mask = question_attention_mask[question_idx]
                curr_input_ids, curr_attention_mask = [], []
                for _p_input_id, _p_attention_mask in zip(curr_aq_p_input_id, curr_aq_p_attention_mask):
                    _input_ids = [qbos_token_id] + curr_q_input_ids[1:] + _p_input_id[1:]
                    _attention_mask = curr_q_attention_mask + _p_attention_mask[1:]
                    if len(_input_ids) > 160:
                        _input_ids = _input_ids[:160]
                        _attention_mask = _attention_mask[:160]
                    else:
                        _input_ids += [pad_token_id for _ in range(32 + 128 - len(_input_ids))]
                        _attention_mask += [0 for _ in range(32 + 128 - len(_attention_mask))]
                    curr_input_ids.append(_input_ids)
                    curr_attention_mask.append(_attention_mask)
                input_ids.append(curr_input_ids)
                attention_mask.append(curr_attention_mask)

        dataset = MyQAGenDataset(input_ids=input_ids, attention_mask=attention_mask, is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(dataset), self.data_type))
        return dataset

    def load_dpr_data_bart(self, nq_dpr_retrieval_path, aq_dpr_retrieval_path, dpr_tokenized_path):
        assert self.args.use_reranker == True, 'currently only support using reranker'
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                dpr_tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.nq_tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True, mode='nq')
            if self.passages.aq_tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True, mode='aq')
            with open(nq_dpr_retrieval_path, "r") as f:
                nq_dpr_passages = json.load(f)
            assert len(nq_dpr_passages)==len(self.nq_data)
            with open(aq_dpr_retrieval_path, "r") as f:
                aq_dpr_passages = json.load(f)
            assert len(aq_dpr_passages)==len(self.aq_data)

            if self.args.use_reranker:
                assert self.args.nq_psg_sel_dir is not None
                nq_psg_sel_fn = os.path.join(self.args.nq_psg_sel_dir, "{}_psg_sel.json".format(
                                              self.data_type.replace("train", "train_for_inference")))
                self.logger.info("Loading NQ passage selection from DPR reader: {}".format(nq_psg_sel_fn))
                with open(nq_psg_sel_fn, "r") as f:
                    nq_fg_passages = json.load(f)
                assert len(nq_fg_passages) == len(nq_dpr_passages)
                nq_dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(nq_dpr_passages, nq_fg_passages)]

                assert self.args.aq_psg_sel_dir is not None
                aq_psg_sel_fn = os.path.join(self.args.aq_psg_sel_dir, "{}_20200201_aq_psg_sel.json".format(
                    self.data_type.replace("train", "train_for_inference")))
                self.logger.info("Loading AQ passage selection from DPR reader: {}".format(aq_psg_sel_fn))
                with open(aq_psg_sel_fn, "r") as f:
                    aq_fg_passages = json.load(f)
                assert len(aq_fg_passages) == len(aq_dpr_passages)
                aq_dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(aq_dpr_passages, aq_fg_passages)]
            else:
                raise NotImplementedError

            if self.is_training:
                # Task 1
                # 1)  AQ_amb_q + AQ_passage -> <q> AQ_no_amb_q1 <SEP> ... AQ_no_amb_q3 <end>  |  aq_q_p -> aq_nq
                self.logger.info("Processing Training Task 1: aq_q_p -> aq_nq")
                task_1 = self.load_dpr_data_bart_training_task_1(aq_dpr_passages)

                # Task 2
                # 2)  no_amb_q + passage -> <q> <end>
                #     2.1) AQ_no_amb_q + AQ_passage -> <q> <end>  |  aq_nq_p -> aq_end
                #     2.2) NQ_q* + NQ_passage -> <q> <end>  |  nq_nq_p -> nq_end
                self.logger.info("Processing Training Task 2.1: aq_nq_p -> aq_end")
                task_2_1 = self.load_dpr_data_bart_training_task_2_1(aq_dpr_passages)
                self.logger.info("Processing Training Task 2.2: nq_nq_p -> nq_end")
                task_2_2 = self.load_dpr_data_bart_training_task_2_2(nq_dpr_passages)

                # Task 3
                # 3)  no_amb_q + passage -> <a> answer <end>
                #     3.1) AQ_no_amb_q + AQ_passage -> <a> AQ_answer <end>  |  aq_nq_p -> aq_a
                #     3.2) NQ_q* + NQ_passage -> <a> NQ_answer <end>  |  nq_nq_p -> nq_a
                self.logger.info("Processing Training Task 3.1: aq_nq_p -> aq_a")
                task_3_1 = self.load_dpr_data_bart_training_task_3_1(aq_dpr_passages)
                self.logger.info("Processing Training Task 3.2: nq_nq_p -> nq_a")
                task_3_2 = self.load_dpr_data_bart_training_task_3_2(nq_dpr_passages)

                dpr_tokenized_data = {
                    'task_1': task_1,
                    'task_2_1': task_2_1,
                    'task_2_2': task_2_2,
                    'task_3_1': task_3_1,
                    'task_3_2': task_3_2,
                }

            else:
                bos_token_id = self.tokenizer.bos_token_id
                eos_token_id = self.tokenizer.eos_token_id
                pad_token_id = self.tokenizer.pad_token_id
                # sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
                # qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
                # abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
                # nq_tokenized_qa_data = self.nq_tokenized_data
                aq_tokenized_qa_data = self.aq_tokenized_data
                # nq_tokenized_passage_data = self.passages.nq_tokenized_data
                aq_tokenized_passage_data = self.passages.aq_tokenized_data
                # 1)  AQ_amb_q + AQ_passage -> <q> AQ_no_amb_q1 <SEP> ... AQ_no_amb_q3 <end>
                # 2)  AQ_no_amb_q + AQ_passage -> <a> AQ_answer <end>  (need to prepare during evaluation time)
                qp_input_ids, qp_attention_mask, p_input_ids, p_attention_mask = [], [], [], []
                for idx, (curr_aq_promptQ_ids, curr_aq_promptQ_attention_mask, aq_dpr_ids) in enumerate(zip(
                        aq_tokenized_qa_data['question_ids'], aq_tokenized_qa_data['question_attention_mask'], aq_dpr_passages)):
                    end_of_question = curr_aq_promptQ_ids.index(eos_token_id) + 1
                    dpr_input_ids = [aq_tokenized_passage_data["input_ids"][_id] for _id in aq_dpr_ids]
                    dpr_attention_mask = [aq_tokenized_passage_data["attention_mask"][_id] for _id in aq_dpr_ids]
                    p_input_ids.append(dpr_input_ids)
                    p_attention_mask.append(dpr_attention_mask)
                    qp_input_ids_idx, qp_attention_mask_idx = [], []
                    for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                        assert _dpr_input_ids[0] == bos_token_id
                        qp_inputs_ids_idx_jdx = curr_aq_promptQ_ids[:end_of_question] + _dpr_input_ids[1:]
                        qp_attention_mask_idx_jdx = curr_aq_promptQ_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                        assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                        qp_inputs_ids_idx_jdx += [pad_token_id for _ in range(32+128 - len(qp_inputs_ids_idx_jdx))]
                        qp_attention_mask_idx_jdx += [0 for _ in range(32+128 - len(qp_attention_mask_idx_jdx))]
                        qp_input_ids_idx.append(qp_inputs_ids_idx_jdx)
                        qp_attention_mask_idx.append(qp_attention_mask_idx_jdx)
                        assert len(qp_input_ids_idx[jdx]) == len(qp_attention_mask_idx[jdx]) == 160  # here we use 32+128
                    qp_input_ids.append(qp_input_ids_idx)
                    qp_attention_mask.append(qp_attention_mask_idx)
                assert len(qp_input_ids) == len(qp_attention_mask)

                dpr_tokenized_data = {
                    'aq_qp_input_ids': qp_input_ids,
                    'aq_qp_attention_mask': qp_attention_mask,
                    'aq_p_input_ids': p_input_ids,
                    'aq_p_attention_mask': p_attention_mask,
                }

            with open(dpr_tokenized_path, "w") as f:
                json.dump(dpr_tokenized_data, f)
            self.logger.info("Finish saving {} tokenized DPR data to {}".format(self.data_type, dpr_tokenized_path))

        if self.is_training:
            self.dpr_tokenized_data = {
                'task_1': self.crop_by_top_k_passages(dpr_tokenized_data['task_1']),
                'task_2_1': self.crop_by_top_k_passages(dpr_tokenized_data['task_2_1']),
                'task_2_2': self.crop_by_top_k_passages(dpr_tokenized_data['task_2_2']),
                'task_3_1': self.crop_by_top_k_passages(dpr_tokenized_data['task_3_1']),
                'task_3_2': self.crop_by_top_k_passages(dpr_tokenized_data['task_3_2']),
            }
        else:
            self.dpr_tokenized_data = self.crop_by_top_k_passages(dpr_tokenized_data)

    def crop_by_top_k_passages(self, task):
        new_task = {}
        for k, v in task.items():
            if not self.is_training or k in ['input_ids', 'input_attention_mask']:
                new_task[k] = [_v[:self.args.top_k_passages] for _v in v]
            else:
                new_task[k] = v
        return new_task

    def load_dpr_data_bart_training_task_1(self, aq_dpr_passages):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
        # abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
        # nq_tokenized_qa_data = self.nq_tokenized_data
        aq_tokenized_qa_data = self.aq_tokenized_data
        # nq_tokenized_passage_data = self.passages.nq_tokenized_data
        aq_tokenized_passage_data = self.passages.aq_tokenized_data

        aq_q_p_input_ids, aq_q_p_attention_mask, aq_nq_input_ids, aq_nq_attention_mask, aq_nq_metadata = [], [], [], [], []

        for idx, (curr_aq_q_ids, curr_aq_q_attention_mask, curr_aq_nq_metadata, aq_dpr_ids) in enumerate(tqdm(zip(
                aq_tokenized_qa_data['question_ids'], aq_tokenized_qa_data['question_attention_mask'],
                aq_tokenized_qa_data['disambiguated_question_metadata'], aq_dpr_passages), total=len(aq_dpr_passages))):
            end_of_aq_q = curr_aq_q_ids.index(eos_token_id) + 1
            curr_aq_q_ids = curr_aq_q_ids[:end_of_aq_q]
            curr_aq_q_attention_mask = curr_aq_q_attention_mask[:end_of_aq_q]
            aq_dpr_input_ids = [aq_tokenized_passage_data["input_ids"][_id] for _id in aq_dpr_ids]
            aq_dpr_attention_mask = [aq_tokenized_passage_data["attention_mask"][_id] for _id in aq_dpr_ids]
            # ensure the current sample exists disambiguated questions
            multiple_qa_anns = []
            for ann_idx, ann in enumerate(self.aq_data[idx]['annotations']):
                if ann['type'] == 'multipleQAs':
                    multiple_qa_anns.append(ann_idx)
            if len(multiple_qa_anns) == 0:
                continue
            # Task 1: encoder side: <s> question </s> passage <s>
            aq_q_p_input_ids_idx, aq_q_p_attention_mask_idx = [], []
            for aq_dpr_jdx, (_aq_dpr_input_ids, _aq_dpr_attention_mask) in enumerate(
                    zip(aq_dpr_input_ids, aq_dpr_attention_mask)):
                assert _aq_dpr_input_ids[0] == bos_token_id
                aq_q_p_input_ids_idx_jdx = curr_aq_q_ids + _aq_dpr_input_ids[1:]
                aq_q_p_attention_mask_idx_jdx = curr_aq_q_attention_mask + _aq_dpr_attention_mask[1:]
                assert len(aq_q_p_input_ids_idx_jdx) == len(aq_q_p_attention_mask_idx_jdx)
                aq_q_p_input_ids_idx_jdx += [pad_token_id for _ in range(32 + 128 - len(aq_q_p_input_ids_idx_jdx))]
                aq_q_p_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(aq_q_p_attention_mask_idx_jdx))]
                aq_q_p_input_ids_idx.append(aq_q_p_input_ids_idx_jdx)
                aq_q_p_attention_mask_idx.append(aq_q_p_attention_mask_idx_jdx)
                assert len(aq_q_p_input_ids_idx[aq_dpr_jdx]) == len(
                    aq_q_p_attention_mask_idx[aq_dpr_jdx]) == 160  # here we use 32+128
            aq_q_p_input_ids.append(aq_q_p_input_ids_idx)
            aq_q_p_attention_mask.append(aq_q_p_attention_mask_idx)
            # Task 1: decoder side: <QBOS> q1 SEP q2 SEP q3 </s>
            aq_nq_input_ids_idx, aq_nq_attention_mask_idx = [], []
            for selected_ann_idx in multiple_qa_anns:
                aq_nq_idx_start = curr_aq_nq_metadata[selected_ann_idx][0]
                aq_nq_idx_end = curr_aq_nq_metadata[selected_ann_idx][-1]
                assert aq_nq_idx_start < aq_nq_idx_end
                aq_nq_permutations = list(itertools.permutations(list(range(aq_nq_idx_start, aq_nq_idx_end))))
                if len(aq_nq_permutations) > 5:
                    aq_nq_permutations = [aq_nq_permutations[_selected_aq_nq_idx] for _selected_aq_nq_idx in
                                          np.random.permutation(range(len(aq_nq_permutations)))[:5]]
                for aq_nq_idxs in aq_nq_permutations:
                    aq_nq_input_ids_idx_jdx, aq_nq_attention_mask_idx_jdx = [qbos_token_id], [1]
                    for aq_nq_idx in aq_nq_idxs:
                        curr_aq_nq_ids = aq_tokenized_qa_data['disambiguated_question_ids'][aq_nq_idx]
                        curr_aq_nq_attention_mask = aq_tokenized_qa_data['disambiguated_question_mask'][aq_nq_idx]
                        end_of_aq_nq = curr_aq_nq_ids.index(eos_token_id)
                        curr_aq_nq_ids = curr_aq_nq_ids[1:end_of_aq_nq]
                        curr_aq_nq_attention_mask = curr_aq_nq_attention_mask[1:end_of_aq_nq]
                        aq_nq_input_ids_idx_jdx += curr_aq_nq_ids + [sep_token_id]
                        aq_nq_attention_mask_idx_jdx += curr_aq_nq_attention_mask + [1]
                    # remove the last SEP, add eos
                    aq_nq_input_ids_idx_jdx = aq_nq_input_ids_idx_jdx[:-1] + [eos_token_id]
                    if len(aq_nq_input_ids_idx_jdx) > self.args.max_qagen_catq_length:
                        aq_nq_input_ids_idx_jdx = aq_nq_input_ids_idx_jdx[:self.args.max_qagen_catq_length]
                        aq_nq_attention_mask_idx_jdx = aq_nq_attention_mask_idx_jdx[:self.args.max_qagen_catq_length]
                    else:
                        aq_nq_input_ids_idx_jdx += [pad_token_id for _ in
                                                    range(self.args.max_qagen_catq_length - len(aq_nq_input_ids_idx_jdx))]
                        aq_nq_attention_mask_idx_jdx += [0 for _ in range(
                            self.args.max_qagen_catq_length - len(aq_nq_attention_mask_idx_jdx))]
                    assert len(aq_nq_input_ids_idx_jdx) == len(
                        aq_nq_attention_mask_idx_jdx) == self.args.max_qagen_catq_length
                    aq_nq_input_ids_idx.append(aq_nq_input_ids_idx_jdx)
                    aq_nq_attention_mask_idx.append(aq_nq_attention_mask_idx_jdx)
            assert len(aq_nq_input_ids_idx) == len(aq_nq_attention_mask_idx)
            aq_nq_metadata.append((len(aq_nq_input_ids), len(aq_nq_input_ids) + len(aq_nq_input_ids_idx)))
            aq_nq_input_ids.extend(aq_nq_input_ids_idx)
            aq_nq_attention_mask.extend(aq_nq_attention_mask_idx)
            assert len(aq_nq_metadata) == len(aq_q_p_input_ids) == len(aq_q_p_attention_mask)
        self.logger.info("Processing Training Task 1 Done! {} aq_q_p, {} aq_nq".format(len(aq_q_p_input_ids), len(aq_nq_input_ids)))
        task_1 = {
            'input_ids': aq_q_p_input_ids,
            'input_attention_mask': aq_q_p_attention_mask,
            'decoder_input_ids': aq_nq_input_ids,
            'decoder_attention_mask': aq_nq_attention_mask,
            'decoder_metadata': aq_nq_metadata,
        }
        return task_1

    def load_dpr_data_bart_training_task_2_1(self, aq_dpr_passages):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        # sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
        # abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
        # nq_tokenized_qa_data = self.nq_tokenized_data
        aq_tokenized_qa_data = self.aq_tokenized_data
        # nq_tokenized_passage_data = self.passages.nq_tokenized_data
        aq_tokenized_passage_data = self.passages.aq_tokenized_data

        aq_nq_p_input_ids, aq_nq_p_attention_mask, aq_end_input_ids, aq_end_attention_mask = [], [], [], []

        for idx, (curr_aq_nq_metadata, aq_dpr_ids) in enumerate(tqdm(zip(
                aq_tokenized_qa_data['disambiguated_question_metadata'], aq_dpr_passages), total=len(aq_dpr_passages))):
            aq_dpr_input_ids = [aq_tokenized_passage_data["input_ids"][_id] for _id in aq_dpr_ids]
            aq_dpr_attention_mask = [aq_tokenized_passage_data["attention_mask"][_id] for _id in aq_dpr_ids]

            # ensure the current sample exists disambiguated questions, Or the question itself is not ambiguous
            multiple_qa_anns = []
            all_single_anns = True
            for ann_idx, ann in enumerate(self.aq_data[idx]['annotations']):
                if ann['type'] == 'multipleQAs':
                    multiple_qa_anns.append(ann_idx)
                    all_single_anns = False

            # Task 2.1: encoder side: <s> no_amb_question </s> passage <s>; decoder side: <QBOS> </s>
            # 2.1 multi -> noambQ -> eos
            for selected_ann_idx in multiple_qa_anns:
                for aq_nq_idx in range(*curr_aq_nq_metadata[selected_ann_idx]):
                    curr_aq_nq_ids = aq_tokenized_qa_data['disambiguated_question_ids'][aq_nq_idx]
                    curr_aq_nq_attention_mask = aq_tokenized_qa_data['disambiguated_question_mask'][aq_nq_idx]
                    end_of_aq_nq = curr_aq_nq_ids.index(eos_token_id) + 1
                    curr_aq_nq_ids = curr_aq_nq_ids[:end_of_aq_nq]
                    curr_aq_nq_attention_mask = curr_aq_nq_attention_mask[:end_of_aq_nq]
                    aq_nq_p_input_ids_idx, aq_nq_p_attention_mask_idx = [], []
                    for aq_dpr_jdx, (_aq_dpr_input_ids, _aq_dpr_attention_mask) in enumerate(zip(aq_dpr_input_ids, aq_dpr_attention_mask)):
                        assert _aq_dpr_input_ids[0] == bos_token_id
                        aq_nq_p_input_ids_idx_jdx = curr_aq_nq_ids + _aq_dpr_input_ids[1:]
                        aq_nq_p_attention_mask_idx_jdx = curr_aq_nq_attention_mask + _aq_dpr_attention_mask[1:]
                        assert len(aq_nq_p_input_ids_idx_jdx) == len(aq_nq_p_attention_mask_idx_jdx)
                        aq_nq_p_input_ids_idx_jdx += [pad_token_id for _ in range(32 + 128 - len(aq_nq_p_input_ids_idx_jdx))]
                        aq_nq_p_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(aq_nq_p_attention_mask_idx_jdx))]
                        aq_nq_p_input_ids_idx.append(aq_nq_p_input_ids_idx_jdx)
                        aq_nq_p_attention_mask_idx.append(aq_nq_p_attention_mask_idx_jdx)
                        assert len(aq_nq_p_input_ids_idx[aq_dpr_jdx]) == len(aq_nq_p_attention_mask_idx[aq_dpr_jdx]) == 160  # here we use 32+128
                    aq_nq_p_input_ids.append(aq_nq_p_input_ids_idx)
                    aq_nq_p_attention_mask.append(aq_nq_p_attention_mask_idx)
                    # target
                    aq_end_input_ids_idx = [qbos_token_id, eos_token_id]
                    aq_end_attention_mask_idx = [1, 1]
                    aq_end_input_ids.append(aq_end_input_ids_idx)
                    aq_end_attention_mask.append(aq_end_attention_mask_idx)

            # 2.1 singleAns -> ambQ -> eos
            if all_single_anns:
                curr_aq_nq_ids = aq_tokenized_qa_data['question_ids'][idx]
                curr_aq_nq_attention_mask = aq_tokenized_qa_data['question_attention_mask'][idx]
                end_of_aq_nq = curr_aq_nq_ids.index(eos_token_id) + 1
                curr_aq_nq_ids = curr_aq_nq_ids[:end_of_aq_nq]
                curr_aq_nq_attention_mask = curr_aq_nq_attention_mask[:end_of_aq_nq]
                aq_nq_p_input_ids_idx, aq_nq_p_attention_mask_idx = [], []
                for aq_dpr_jdx, (_aq_dpr_input_ids, _aq_dpr_attention_mask) in enumerate(
                        zip(aq_dpr_input_ids, aq_dpr_attention_mask)):
                    assert _aq_dpr_input_ids[0] == bos_token_id
                    aq_nq_p_input_ids_idx_jdx = curr_aq_nq_ids + _aq_dpr_input_ids[1:]
                    aq_nq_p_attention_mask_idx_jdx = curr_aq_nq_attention_mask + _aq_dpr_attention_mask[1:]
                    assert len(aq_nq_p_input_ids_idx_jdx) == len(aq_nq_p_attention_mask_idx_jdx)
                    aq_nq_p_input_ids_idx_jdx += [pad_token_id for _ in
                                                  range(32 + 128 - len(aq_nq_p_input_ids_idx_jdx))]
                    aq_nq_p_attention_mask_idx_jdx += [0 for _ in range(
                        32 + 128 - len(aq_nq_p_attention_mask_idx_jdx))]
                    aq_nq_p_input_ids_idx.append(aq_nq_p_input_ids_idx_jdx)
                    aq_nq_p_attention_mask_idx.append(aq_nq_p_attention_mask_idx_jdx)
                    assert len(aq_nq_p_input_ids_idx[aq_dpr_jdx]) == len(
                        aq_nq_p_attention_mask_idx[aq_dpr_jdx]) == 160  # here we use 32+128
                aq_nq_p_input_ids.append(aq_nq_p_input_ids_idx)
                aq_nq_p_attention_mask.append(aq_nq_p_attention_mask_idx)
                # target
                aq_end_input_ids_idx = [qbos_token_id, eos_token_id]
                aq_end_attention_mask_idx = [1, 1]
                aq_end_input_ids.append(aq_end_input_ids_idx)
                aq_end_attention_mask.append(aq_end_attention_mask_idx)
        assert len(aq_end_input_ids) == len(aq_end_attention_mask) == len(aq_nq_p_input_ids) == len(aq_nq_p_attention_mask)
        self.logger.info("Processing Training Task 2.1 Done! {} aq_nq_p".format(len(aq_nq_p_input_ids)))
        task_2_1 = {
            'input_ids': aq_nq_p_input_ids,
            'input_attention_mask': aq_nq_p_attention_mask,
            'decoder_input_ids': aq_end_input_ids,
            'decoder_attention_mask': aq_end_attention_mask,
        }
        return task_2_1

    def load_dpr_data_bart_training_task_2_2(self, nq_dpr_passages):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        # sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
        # abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
        nq_tokenized_qa_data = self.nq_tokenized_data
        # aq_tokenized_qa_data = self.aq_tokenized_data
        nq_tokenized_passage_data = self.passages.nq_tokenized_data
        # aq_tokenized_passage_data = self.passages.aq_tokenized_data

        nq_nq_p_input_ids, nq_nq_p_attention_mask, nq_end_input_ids, nq_end_attention_mask = [], [], [], []
        aq_ids = [ex['id'] for ex in self.aq_data]
        for idx, (curr_nq_nq_ids, curr_nq_nq_attention_mask, nq_dpr_ids) in enumerate(tqdm(zip(
                nq_tokenized_qa_data['question_ids'], nq_tokenized_qa_data['question_attention_mask'], nq_dpr_passages),
                total=len(nq_dpr_passages))):
            # not using ambiguous questions in this case
            curr_id = self.nq_data[idx]['id']
            if curr_id in aq_ids:
                continue
            nq_dpr_input_ids = [nq_tokenized_passage_data["input_ids"][_id] for _id in nq_dpr_ids]
            nq_dpr_attention_mask = [nq_tokenized_passage_data["attention_mask"][_id] for _id in nq_dpr_ids]
            # Task 2.2: encoder side: <s> nq_question </s> passage <s>; decoder side: <QBOS> </s>
            end_of_nq_nq = curr_nq_nq_ids.index(eos_token_id) + 1
            curr_nq_nq_ids = curr_nq_nq_ids[:end_of_nq_nq]
            curr_nq_nq_attention_mask = curr_nq_nq_attention_mask[:end_of_nq_nq]
            nq_nq_p_input_ids_idx, nq_nq_p_attention_mask_idx = [], []
            for nq_dpr_jdx, (_nq_dpr_input_ids, _nq_dpr_attention_mask) in enumerate(
                    zip(nq_dpr_input_ids, nq_dpr_attention_mask)):
                assert _nq_dpr_input_ids[0] == bos_token_id
                nq_nq_p_input_ids_idx_jdx = curr_nq_nq_ids + _nq_dpr_input_ids[1:]
                nq_nq_p_attention_mask_idx_jdx = curr_nq_nq_attention_mask + _nq_dpr_attention_mask[1:]
                assert len(nq_nq_p_input_ids_idx_jdx) == len(nq_nq_p_attention_mask_idx_jdx)
                nq_nq_p_input_ids_idx_jdx += [pad_token_id for _ in range(32 + 128 - len(nq_nq_p_input_ids_idx_jdx))]
                nq_nq_p_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(nq_nq_p_attention_mask_idx_jdx))]
                nq_nq_p_input_ids_idx.append(nq_nq_p_input_ids_idx_jdx)
                nq_nq_p_attention_mask_idx.append(nq_nq_p_attention_mask_idx_jdx)
                assert len(nq_nq_p_input_ids_idx[nq_dpr_jdx]) == len(
                    nq_nq_p_attention_mask_idx[nq_dpr_jdx]) == 160  # here we use 32+128
            nq_nq_p_input_ids.append(nq_nq_p_input_ids_idx)
            nq_nq_p_attention_mask.append(nq_nq_p_attention_mask_idx)
            # target
            nq_end_input_ids_idx = [qbos_token_id, eos_token_id]
            nq_end_attention_mask_idx = [1, 1]
            nq_end_input_ids.append(nq_end_input_ids_idx)
            nq_end_attention_mask.append(nq_end_attention_mask_idx)
        assert len(nq_end_input_ids) == len(nq_end_attention_mask) == len(nq_nq_p_input_ids) == len(nq_nq_p_attention_mask)
        self.logger.info("Processing Training Task 2.1 Done! {} nq_nq_p".format(len(nq_nq_p_input_ids)))
        task_2_2 = {
            'input_ids': nq_nq_p_input_ids,
            'input_attention_mask': nq_nq_p_attention_mask,
            'decoder_input_ids': nq_end_input_ids,
            'decoder_attention_mask': nq_end_attention_mask,
        }
        return task_2_2

    def load_dpr_data_bart_training_task_3_1(self, aq_dpr_passages):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        # sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        # qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
        abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
        # nq_tokenized_qa_data = self.nq_tokenized_data
        aq_tokenized_qa_data = self.aq_tokenized_data
        # nq_tokenized_passage_data = self.passages.nq_tokenized_data
        aq_tokenized_passage_data = self.passages.aq_tokenized_data

        aq_nq_p_input_ids, aq_nq_p_attention_mask, aq_a_input_ids, aq_a_attention_mask, aq_a_metadata = [], [], [], [], []
        for idx, (curr_aq_nq_metadata, curr_aq_a_metadata, aq_dpr_ids) in enumerate(tqdm(zip(
                aq_tokenized_qa_data['disambiguated_question_metadata'],
                aq_tokenized_qa_data['answer_metadata'], aq_dpr_passages),
                total=len(aq_dpr_passages))):
            aq_dpr_input_ids = [aq_tokenized_passage_data["input_ids"][_id] for _id in aq_dpr_ids]
            aq_dpr_attention_mask = [aq_tokenized_passage_data["attention_mask"][_id] for _id in aq_dpr_ids]

            # <s> question </s> passage </s> -> <ABOS> answer </s>
            for ann_idx, ann in enumerate(self.aq_data[idx]['annotations']):
                if ann['type'] == 'singleAnswer':
                    # promptQ -> answer
                    curr_aq_nq_ids, curr_aq_nq_attention_mask = aq_tokenized_qa_data['question_ids'][idx], \
                                                                aq_tokenized_qa_data['question_attention_mask'][idx]
                    end_of_aq_nq = curr_aq_nq_ids.index(eos_token_id) + 1
                    curr_aq_nq_ids = curr_aq_nq_ids[:end_of_aq_nq]
                    curr_aq_nq_attention_mask = curr_aq_nq_attention_mask[:end_of_aq_nq]
                    aq_nq_p_input_ids_idx, aq_nq_p_attention_mask_idx = [], []
                    for aq_dpr_jdx, (_aq_dpr_input_ids, _aq_dpr_attention_mask) in enumerate(
                            zip(aq_dpr_input_ids, aq_dpr_attention_mask)):
                        assert _aq_dpr_input_ids[0] == bos_token_id
                        aq_nq_p_input_ids_idx_jdx = curr_aq_nq_ids + _aq_dpr_input_ids[1:]
                        aq_nq_p_attention_mask_idx_jdx = curr_aq_nq_attention_mask + _aq_dpr_attention_mask[1:]
                        assert len(aq_nq_p_input_ids_idx_jdx) == len(aq_nq_p_attention_mask_idx_jdx)
                        aq_nq_p_input_ids_idx_jdx += [pad_token_id for _ in
                                                      range(32 + 128 - len(aq_nq_p_input_ids_idx_jdx))]
                        aq_nq_p_attention_mask_idx_jdx += [0 for _ in
                                                           range(32 + 128 - len(aq_nq_p_attention_mask_idx_jdx))]
                        aq_nq_p_input_ids_idx.append(aq_nq_p_input_ids_idx_jdx)
                        aq_nq_p_attention_mask_idx.append(aq_nq_p_attention_mask_idx_jdx)
                        assert len(aq_nq_p_input_ids_idx[aq_dpr_jdx]) == len(
                            aq_nq_p_attention_mask_idx[aq_dpr_jdx]) == 160  # here we use 32+128
                    aq_nq_p_input_ids.append(aq_nq_p_input_ids_idx)
                    aq_nq_p_attention_mask.append(aq_nq_p_attention_mask_idx)
                    # answer
                    offset = len(aq_a_input_ids)
                    for curr_aq_a_idx in range(*curr_aq_a_metadata[ann_idx][0]):
                        aq_a_input_ids_idx, aq_a_attention_mask_idx = aq_tokenized_qa_data['answer_ids'][curr_aq_a_idx], \
                                                                      aq_tokenized_qa_data['answer_mask'][curr_aq_a_idx]
                        assert aq_a_input_ids_idx[0] == bos_token_id
                        end_of_aq_a = aq_a_input_ids_idx.index(eos_token_id) + 1
                        new_aq_a_input_ids_idx = [abos_token_id] + aq_a_input_ids_idx[1:end_of_aq_a]
                        aq_a_attention_mask_idx = aq_a_attention_mask_idx[:end_of_aq_a]
                        if len(new_aq_a_input_ids_idx) > self.args.max_qagen_answer_length:
                            new_aq_a_input_ids_idx = new_aq_a_input_ids_idx[:self.args.max_qagen_answer_length]
                            aq_a_attention_mask_idx = aq_a_attention_mask_idx[:self.args.max_qagen_answer_length]
                        else:
                            new_aq_a_input_ids_idx += [pad_token_id for _ in range(
                                self.args.max_qagen_answer_length - len(new_aq_a_input_ids_idx))]
                            aq_a_attention_mask_idx += [0 for _ in range(
                                self.args.max_qagen_answer_length - len(aq_a_attention_mask_idx))]
                        assert len(new_aq_a_input_ids_idx) == len(
                            aq_a_attention_mask_idx) == self.args.max_qagen_answer_length
                        aq_a_input_ids.append(new_aq_a_input_ids_idx)
                        aq_a_attention_mask.append(aq_a_attention_mask_idx)
                    aq_a_metadata.append((offset, len(aq_a_input_ids)))
                else:
                    # disQ -> answer
                    for qapair_idx, aq_nq_idx in enumerate(range(*curr_aq_nq_metadata[ann_idx])):
                        curr_aq_nq_ids = aq_tokenized_qa_data['disambiguated_question_ids'][aq_nq_idx]
                        curr_aq_nq_attention_mask = aq_tokenized_qa_data['disambiguated_question_mask'][aq_nq_idx]
                        end_of_aq_nq = curr_aq_nq_ids.index(eos_token_id) + 1
                        curr_aq_nq_ids = curr_aq_nq_ids[:end_of_aq_nq]
                        curr_aq_nq_attention_mask = curr_aq_nq_attention_mask[:end_of_aq_nq]
                        aq_nq_p_input_ids_idx, aq_nq_p_attention_mask_idx = [], []
                        for aq_dpr_jdx, (_aq_dpr_input_ids, _aq_dpr_attention_mask) in enumerate(
                                zip(aq_dpr_input_ids, aq_dpr_attention_mask)):
                            assert _aq_dpr_input_ids[0] == bos_token_id
                            aq_nq_p_input_ids_idx_jdx = curr_aq_nq_ids + _aq_dpr_input_ids[1:]
                            aq_nq_p_attention_mask_idx_jdx = curr_aq_nq_attention_mask + _aq_dpr_attention_mask[1:]
                            assert len(aq_nq_p_input_ids_idx_jdx) == len(aq_nq_p_attention_mask_idx_jdx)
                            aq_nq_p_input_ids_idx_jdx += [pad_token_id for _ in
                                                          range(32 + 128 - len(aq_nq_p_input_ids_idx_jdx))]
                            aq_nq_p_attention_mask_idx_jdx += [0 for _ in
                                                               range(
                                                                   32 + 128 - len(aq_nq_p_attention_mask_idx_jdx))]
                            aq_nq_p_input_ids_idx.append(aq_nq_p_input_ids_idx_jdx)
                            aq_nq_p_attention_mask_idx.append(aq_nq_p_attention_mask_idx_jdx)
                            assert len(aq_nq_p_input_ids_idx[aq_dpr_jdx]) == len(
                                aq_nq_p_attention_mask_idx[aq_dpr_jdx]) == 160  # here we use 32+128
                        aq_nq_p_input_ids.append(aq_nq_p_input_ids_idx)
                        aq_nq_p_attention_mask.append(aq_nq_p_attention_mask_idx)
                        # answer
                        offset = len(aq_a_input_ids)
                        for curr_aq_a_idx in range(*curr_aq_a_metadata[ann_idx][qapair_idx]):
                            aq_a_input_ids_idx, aq_a_attention_mask_idx = aq_tokenized_qa_data['answer_ids'][
                                                                              curr_aq_a_idx], \
                                                                          aq_tokenized_qa_data['answer_mask'][
                                                                              curr_aq_a_idx]
                            assert len(aq_a_input_ids_idx) == len(aq_a_attention_mask_idx)
                            end_of_aq_a = aq_a_input_ids_idx.index(eos_token_id) + 1
                            new_aq_a_input_ids_idx = [abos_token_id] + aq_a_input_ids_idx[1:end_of_aq_a]
                            aq_a_attention_mask_idx = aq_a_attention_mask_idx[:end_of_aq_a]
                            if len(new_aq_a_input_ids_idx) > self.args.max_qagen_answer_length:
                                new_aq_a_input_ids_idx = new_aq_a_input_ids_idx[:self.args.max_qagen_answer_length]
                                aq_a_attention_mask_idx = aq_a_attention_mask_idx[:self.args.max_qagen_answer_length]
                            else:
                                new_aq_a_input_ids_idx += [pad_token_id for _ in range(
                                    self.args.max_qagen_answer_length - len(new_aq_a_input_ids_idx))]
                                aq_a_attention_mask_idx += [0 for _ in range(
                                    self.args.max_qagen_answer_length - len(aq_a_attention_mask_idx))]
                            assert len(new_aq_a_input_ids_idx) == len(aq_a_attention_mask_idx) == self.args.max_qagen_answer_length, (len(new_aq_a_input_ids_idx) , len(aq_a_attention_mask_idx) , self.args.max_qagen_answer_length)
                            aq_a_input_ids.append(new_aq_a_input_ids_idx)
                            aq_a_attention_mask.append(aq_a_attention_mask_idx)
                        aq_a_metadata.append((offset, len(aq_a_input_ids)))
        assert len(aq_a_input_ids) == len(aq_a_attention_mask) == aq_a_metadata[-1][-1]
        assert len(aq_nq_p_input_ids) == len(aq_nq_p_attention_mask) == len(aq_a_metadata)
        self.logger.info("Processing Training Task 3.1 Done! {} aq_nq_p, {} aq_a".format(len(aq_nq_p_input_ids),
                                                                                         len(aq_a_input_ids)))
        task_3_1 = {
            'input_ids': aq_nq_p_input_ids,
            'input_attention_mask': aq_nq_p_attention_mask,
            'decoder_input_ids': aq_a_input_ids,
            'decoder_attention_mask': aq_a_attention_mask,
            'decoder_metadata': aq_a_metadata,
        }
        return task_3_1

    def load_dpr_data_bart_training_task_3_2(self, nq_dpr_passages):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        # sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        # qbos_token_id = self.tokenizer.convert_tokens_to_ids(self.QBOS)
        abos_token_id = self.tokenizer.convert_tokens_to_ids(self.ABOS)
        nq_tokenized_qa_data = self.nq_tokenized_data
        # aq_tokenized_qa_data = self.aq_tokenized_data
        nq_tokenized_passage_data = self.passages.nq_tokenized_data
        # aq_tokenized_passage_data = self.passages.aq_tokenized_data
        aq_ids = [ex['id'] for ex in self.aq_data]
        nq_nq_p_input_ids, nq_nq_p_attention_mask, nq_a_input_ids, nq_a_attention_mask, nq_a_metadata = [], [], [], [], []
        for idx, (curr_nq_a_metadata, nq_dpr_ids) in enumerate(tqdm(zip(
                nq_tokenized_qa_data['answer_metadata'], nq_dpr_passages),
                total=len(nq_dpr_passages))):
            # not using ambiguous questions in this case
            curr_id = self.nq_data[idx]['id']
            if curr_id in aq_ids:
                continue
            # if there exist more than 1 answer, filter this sample, to reduce the possibility of ambiguity
            if curr_nq_a_metadata[1] - curr_nq_a_metadata[0] > 1:
                continue
            nq_dpr_input_ids = [nq_tokenized_passage_data["input_ids"][_id] for _id in nq_dpr_ids]
            nq_dpr_attention_mask = [nq_tokenized_passage_data["attention_mask"][_id] for _id in nq_dpr_ids]

            # <s> question </s> passage </s> -> <ABOS> answer </s>
            curr_nq_nq_ids, curr_nq_nq_attention_mask = nq_tokenized_qa_data['question_ids'][idx], \
                                                        nq_tokenized_qa_data['question_attention_mask'][idx]
            end_of_nq_nq = curr_nq_nq_ids.index(eos_token_id) + 1
            curr_nq_nq_ids = curr_nq_nq_ids[:end_of_nq_nq]
            curr_nq_nq_attention_mask = curr_nq_nq_attention_mask[:end_of_nq_nq]
            nq_nq_p_input_ids_idx, nq_nq_p_attention_mask_idx = [], []
            for nq_dpr_jdx, (_nq_dpr_input_ids, _nq_dpr_attention_mask) in enumerate(
                    zip(nq_dpr_input_ids, nq_dpr_attention_mask)):
                assert _nq_dpr_input_ids[0] == bos_token_id
                nq_nq_p_input_ids_idx_jdx = curr_nq_nq_ids + _nq_dpr_input_ids[1:]
                nq_nq_p_attention_mask_idx_jdx = curr_nq_nq_attention_mask + _nq_dpr_attention_mask[1:]
                assert len(nq_nq_p_input_ids_idx_jdx) == len(nq_nq_p_attention_mask_idx_jdx)
                nq_nq_p_input_ids_idx_jdx += [pad_token_id for _ in range(32 + 128 - len(nq_nq_p_input_ids_idx_jdx))]
                nq_nq_p_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(nq_nq_p_attention_mask_idx_jdx))]
                nq_nq_p_input_ids_idx.append(nq_nq_p_input_ids_idx_jdx)
                nq_nq_p_attention_mask_idx.append(nq_nq_p_attention_mask_idx_jdx)
                assert len(nq_nq_p_input_ids_idx[nq_dpr_jdx]) == len(
                    nq_nq_p_attention_mask_idx[nq_dpr_jdx]) == 160  # here we use 32+128
            nq_nq_p_input_ids.append(nq_nq_p_input_ids_idx)
            nq_nq_p_attention_mask.append(nq_nq_p_attention_mask_idx)
            # answer
            offset = len(nq_a_input_ids)
            for curr_nq_a_idx in range(*curr_nq_a_metadata):
                nq_a_input_ids_idx, nq_a_attention_mask_idx = nq_tokenized_qa_data['answer_ids'][curr_nq_a_idx], nq_tokenized_qa_data['answer_mask'][curr_nq_a_idx]
                assert nq_a_input_ids_idx[0] == bos_token_id
                end_of_nq_a = nq_a_input_ids_idx.index(eos_token_id) + 1
                new_nq_a_input_ids_idx = [abos_token_id] + nq_a_input_ids_idx[1:end_of_nq_a]
                nq_a_attention_mask_idx = nq_a_attention_mask_idx[:end_of_nq_a]
                if len(new_nq_a_input_ids_idx) > self.args.max_qagen_answer_length:
                    new_nq_a_input_ids_idx = new_nq_a_input_ids_idx[:self.args.max_qagen_answer_length]
                    nq_a_attention_mask_idx = nq_a_attention_mask_idx[:self.args.max_qagen_answer_length]
                else:
                    new_nq_a_input_ids_idx += [pad_token_id for _ in range(self.args.max_qagen_answer_length - len(new_nq_a_input_ids_idx))]
                    nq_a_attention_mask_idx += [0 for _ in range(self.args.max_qagen_answer_length - len(nq_a_attention_mask_idx))]
                assert len(new_nq_a_input_ids_idx) == len(nq_a_attention_mask_idx) == self.args.max_qagen_answer_length
                nq_a_input_ids.append(new_nq_a_input_ids_idx)
                nq_a_attention_mask.append(nq_a_attention_mask_idx)
            nq_a_metadata.append((offset, len(nq_a_input_ids)))
        assert len(nq_a_input_ids) == len(nq_a_attention_mask) == nq_a_metadata[-1][-1]
        assert len(nq_nq_p_input_ids) == len(nq_nq_p_attention_mask) == len(nq_a_metadata)
        self.logger.info("Processing Training Task 3.2 Done! {} nq_nq_p, {} nq_a".format(len(nq_nq_p_input_ids),
                                                                                         len(nq_a_input_ids)))
        task_3_2 = {
            'input_ids': nq_nq_p_input_ids,
            'input_attention_mask': nq_nq_p_attention_mask,
            'decoder_input_ids': nq_a_input_ids,
            'decoder_attention_mask': nq_a_attention_mask,
            'decoder_metadata': nq_a_metadata,
        }
        return task_3_2

    def load_dataset(self, tokenizer, do_return=False):
        if self.aq_tokenized_data is None or self.nq_tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if self.is_training:
            self.dataset_task_1 = MyQAGenDataset(input_ids=self.dpr_tokenized_data['task_1']['input_ids'],
                                                 attention_mask=self.dpr_tokenized_data['task_1']['input_attention_mask'],
                                                 decoder_input_ids=self.dpr_tokenized_data['task_1']['decoder_input_ids'],
                                                 decoder_attention_mask=self.dpr_tokenized_data['task_1']['decoder_attention_mask'],
                                                 out_metadata=self.dpr_tokenized_data['task_1']['decoder_metadata'],
                                                 is_training=self.is_training)
            self.logger.info("Loaded {} examples from {} data task 1".format(len(self.dataset_task_1), self.data_type))
            self.dataset_task_2_1 = MyQAGenDataset(input_ids=self.dpr_tokenized_data['task_2_1']['input_ids'],
                                                 attention_mask=self.dpr_tokenized_data['task_2_1']['input_attention_mask'],
                                                 decoder_input_ids=self.dpr_tokenized_data['task_2_1']['decoder_input_ids'],
                                                 decoder_attention_mask=self.dpr_tokenized_data['task_2_1']['decoder_attention_mask'],
                                                 is_training=self.is_training)
            self.logger.info("Loaded {} examples from {} data task 2_1".format(len(self.dataset_task_2_1), self.data_type))
            self.dataset_task_2_2 = MyQAGenDataset(input_ids=self.dpr_tokenized_data['task_2_2']['input_ids'],
                                                 attention_mask=self.dpr_tokenized_data['task_2_2']['input_attention_mask'],
                                                 decoder_input_ids=self.dpr_tokenized_data['task_2_2']['decoder_input_ids'],
                                                 decoder_attention_mask=self.dpr_tokenized_data['task_2_2']['decoder_attention_mask'],
                                                 is_training=self.is_training)
            self.logger.info("Loaded {} examples from {} data task 2_2".format(len(self.dataset_task_2_2), self.data_type))
            self.dataset_task_3_1 = MyQAGenDataset(input_ids=self.dpr_tokenized_data['task_3_1']['input_ids'],
                                                 attention_mask=self.dpr_tokenized_data['task_3_1']['input_attention_mask'],
                                                 decoder_input_ids=self.dpr_tokenized_data['task_3_1']['decoder_input_ids'],
                                                 decoder_attention_mask=self.dpr_tokenized_data['task_3_1']['decoder_attention_mask'],
                                                 out_metadata=self.dpr_tokenized_data['task_3_1']['decoder_metadata'],
                                                 is_training=self.is_training)
            self.logger.info("Loaded {} examples from {} data task 3_1".format(len(self.dataset_task_3_1), self.data_type))
            self.dataset_task_3_2 = MyQAGenDataset(input_ids=self.dpr_tokenized_data['task_3_2']['input_ids'],
                                                 attention_mask=self.dpr_tokenized_data['task_3_2']['input_attention_mask'],
                                                 decoder_input_ids=self.dpr_tokenized_data['task_3_2']['decoder_input_ids'],
                                                 decoder_attention_mask=self.dpr_tokenized_data['task_3_2']['decoder_attention_mask'],
                                                 out_metadata=self.dpr_tokenized_data['task_3_2']['decoder_metadata'],
                                                 is_training=self.is_training)
            self.logger.info("Loaded {} examples from {} data task 3_2".format(len(self.dataset_task_3_2), self.data_type))
        else:
            self.dataset = MyQAGenDataset(input_ids=self.dpr_tokenized_data['aq_qp_input_ids'],
                                          attention_mask=self.dpr_tokenized_data['aq_qp_attention_mask'],
                                          is_training=self.is_training)
            self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False, **kwargs):
        if self.is_training:
            self.dataloader_task_1 = MyQAGenDataLoader(self.args, self.dataset_task_1, is_training=self.is_training,
                                                       batch_size=self.args.train_batch_size_task_1, **kwargs)
            self.dataloader_task_2_1 = MyQAGenDataLoader(self.args, self.dataset_task_2_1, is_training=self.is_training,
                                                       batch_size=self.args.train_batch_size_task_2_1, **kwargs)
            self.dataloader_task_2_2 = MyQAGenDataLoader(self.args, self.dataset_task_2_2, is_training=self.is_training,
                                                       batch_size=self.args.train_batch_size_task_2_2, **kwargs)
            self.dataloader_task_3_1 = MyQAGenDataLoader(self.args, self.dataset_task_3_1, is_training=self.is_training,
                                                       batch_size=self.args.train_batch_size_task_3_1, **kwargs)
            self.dataloader_task_3_2 = MyQAGenDataLoader(self.args, self.dataset_task_3_2, is_training=self.is_training,
                                                       batch_size=self.args.train_batch_size_task_3_2, **kwargs)
            if do_return:
                return self.dataloader_task_1, self.dataloader_task_2_1, self.dataloader_task_2_2, \
                       self.dataloader_task_3_1, self.dataloader_task_3_2
        else:
            self.dataloader = MyQAGenDataLoader(self.args, self.dataset, is_training=self.is_training)
            if do_return:
                return self.dataloader

    def load_task_3_1_inference_dataloader(self, dataset):
        assert not self.is_training
        dataloader = MyQAGenDataLoader(self.args, dataset, is_training=self.is_training)
        return dataloader

    def evaluate(self, predictions_question, predictions_answer, predictions_question_metadata):
        reference = deepcopy(self.aq_data)
        if not (type(reference) == list and \
                all([type(ref) == dict and "id" in ref and "question" in ref and "annotations" in ref and \
                     type(ref["question"]) == str and type(ref["annotations"]) == list and \
                     all([type(ann) == dict and ann["type"] in ["singleAnswer", "multipleQAs"] for ann in
                          ref["annotations"]]) \
                     for ref in reference])):
            raise Exception("Reference file is wrong")
        # construct prediction samples
        prediction = {}
        for idx, (d, m) in enumerate(zip(reference, predictions_question_metadata)):
            prediction[d['id']] = []
            for jdx_ans, jdx_ques in enumerate(range(*m)):
                answer = predictions_answer[jdx_ans]
                question = predictions_question[jdx_ques]
                prediction[d['id']].append({'question': question, 'answer': answer, })
        evaluation = QAPairEvaluation(deepcopy(reference), deepcopy(prediction))
        results = evaluation.print_all_metrics()
        return results['F1 answer'], results['F1 answer (multi)'], results["F1 bleu4"], results["F1 edit-f1"], deepcopy(prediction)

    def save_predictions(self, predictions, mode=''):
        save_path = os.path.join(self.args.output_dir, "{}{}{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 else "",
            "_aq" if self.args.ambigqa else "",
            mode,
        ))
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))


