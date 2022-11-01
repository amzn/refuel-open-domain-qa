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
import random
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

class QAData(object):

    def __init__(self, logger, args, data_path, is_training, passages=None):
        self.data_path = data_path
        self.passages = passages
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            if args.task == "cotraining_label":
                idx = int(self.data_path[self.data_path.find("train_")+6:self.data_path.find("train_")+7])
                self.data_type = "train_{}".format(idx)
            else:
                self.data_type = "train" if is_training or args.dpr else "train_for_inference"
        else:
            raise NotImplementedError()

        with open(self.data_path, "r") as f:
            self.data = json.load(f)

        if "data" in self.data:
            self.data = self.data["data"]

        if "answers" in self.data[0]:
            self.data = [{"id": d["id"], "question": d["question"], "answer": d["answers"]} for d in self.data]

        if args.debug:
            self.data = self.data[:40]
        assert type(self.data)==list

        if not (args.ambigqa or args.leaderboard or args.task == 'cotraining_label'):
            id2answer_path = os.path.join("/".join(self.data_path.split("/")[:-1]),
                                          "{}_id2answers.json".format(self.data_type.replace("train_for_inference", "train")))
            with open(id2answer_path, "r") as f:
                id2answers = json.load(f)
            for i, d in enumerate(self.data):
                if is_training:
                    for ans in id2answers[d["id"]]:
                        if ans not in self.data[i]["answer"]:
                            self.data[i]["answer"].append(ans)
                else:
                    self.data[i]["answer"] = id2answers[d["id"]]

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.metric = "EM"
        self.tokenizer = None
        self.tokenized_data = None
        self.dpr_tokenized_data = None
        self.dataset = None
        self.dataloader = None

    def __len__(self):
        return len(self.data)

    def get_answers(self):
        return [d["answer"] for d in self.data]

    def decode(self, tokens):
        if type(tokens[0])==list:
            return [self.decode(_tokens) for _tokens in tokens]
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip().replace(" - ", "-").replace(" : ", ":")

    def decode_span(self, outputs, n_paragraphs):
        assert len(self.data)==len(self.tokenized_data["positive_input_ids"])==\
            len(self.tokenized_data["positive_input_mask"])==\
            len(outputs)
        return decode_span_batch(list(zip(self.tokenized_data["positive_input_ids"],
                                          self.tokenized_data["positive_input_mask"])),
                                 outputs,
                                 tokenizer=self.tokenizer,
                                 max_answer_length=self.args.max_answer_length,
                                 n_paragraphs=n_paragraphs,
                                 topk_answer=self.args.topk_answer,
                                 verbose=self.args.verbose,
                                 n_jobs=self.args.n_jobs,
                                 save_psg_sel_only=self.args.save_psg_sel_only)

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix or 'T5' in postfix
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}{}{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if self.args.append_another_bos else "",
                    "-reos" if self.args.t5_no_intermediate_eos else "",
                    postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print ("Start tokenizing...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
            answers, metadata = self.flatten(answers)
            if self.args.bert_name.startswith("t5"):
                if self.args.t5_no_intermediate_eos:
                    questions = ["question: " + question for question in questions]
                else:
                    questions = ["question: " + question + " </s>" for question in questions]
                answers = [answer + " </s>" for answer in answers]
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix or "T5" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask, metadata]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f)
        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type)).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2], "{}{}_predictions.json".format(self.data_type,
                                                                                                                                           "-reos" if self.args.t5_no_intermediate_eos else "",))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_{}.json".format(postfix))
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        elif "T5" in postfix:
            return self.load_dpr_data_t5(dpr_retrieval_path, dpr_tokenized_path)
        elif "Bert" in postfix or "Albert" in postfix:
            return self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError()

    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            assert self.args.use_reranker == True
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            assert len(dpr_passages)==len(self)

            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}_psg_sel.json".format(
                                              self.data_type.replace("train", "train_for_inference"),
                                              "_20200201" if self.args.wiki_2020 else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    tqdm(input_ids), attention_mask, metadata, dpr_passages)):
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32+128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32+128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[idx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[idx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[idx][jdx]) == len(qp_attention_mask[idx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data")

        self.tokenized_data[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        if self.is_training and self.args.discard_not_found_answers:
            self.discard_not_found_answers()

    def discard_not_found_answers(self):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data[:5]
        new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], [], [], []

        skipped_idxs = []

        self.logger.info("Discarding training examples where retrieval fails...")

        def _get_tokenized_answer(idx, append_another_bos):
            tokens = self.tokenized_data[2][idx]
            # remove padded token
            if 0 in self.tokenized_data[3][idx]:
                tokens = tokens[:self.tokenized_data[3][idx].index(0)]
            if append_another_bos:
                assert tokens[0] == tokens[1] == self.tokenizer.bos_token_id and tokens[-1] == self.tokenizer.eos_token_id
                return tokens[2:-1]
            else:
                assert tokens[0] == self.tokenizer.bos_token_id and tokens[-1] == self.tokenizer.eos_token_id
                return tokens[1:-1]

        for idx, (curr_input_ids, curr_attention_mask, curr_metadata) in enumerate(zip(
                input_ids, attention_mask, metadata)):
            end_of_question = curr_input_ids[0].index(self.tokenizer.eos_token_id)+1
            def _included(tokens):
                for _curr_input_ids in curr_input_ids:
                    for jdx in range(end_of_question, len(_curr_input_ids)-len(tokens)+1):
                        if _curr_input_ids[jdx:jdx+len(tokens)]==tokens:
                            return True
                return False

            valid_answer_idxs = [answer_idx for answer_idx in range(curr_metadata[0], curr_metadata[1])
                                    if _included(_get_tokenized_answer(answer_idx, self.args.append_another_bos))]
            if len(valid_answer_idxs)==0:
                skipped_idxs.append(idx)
                continue
            new_input_ids.append(curr_input_ids)
            new_attention_mask.append(curr_attention_mask)
            new_decoder_input_ids += [decoder_input_ids[i] for i in valid_answer_idxs]
            new_decoder_attention_mask += [decoder_attention_mask[i] for i in valid_answer_idxs]
            new_metadata.append([len(new_decoder_input_ids)-len(valid_answer_idxs), len(new_decoder_input_ids)])

        self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata]

        self.logger.info("original samples: {}, new training samples {}, {} filtered because of no answer found".format(len(input_ids), len(new_input_ids), len(skipped_idxs)))
        self.logger.info("Equivalent Recall {:.2f}".format(len(new_input_ids)/len(input_ids)*100))

    def load_dpr_data_t5(self, dpr_retrieval_path, dpr_tokenized_path):
        raise NotImplementedError
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("t5", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            assert len(dpr_passages)==len(self)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            # bos_token_id = self.tokenizer.bos_token_id

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                try:
                    if self.args.t5_no_intermediate_eos:
                        # in some cases, the question may exceed the length of 32, so eos/pad is not appended at the end
                        if self.tokenizer.pad_token_id not in curr_input_ids:
                            end_of_question = len(curr_input_ids)
                        else:
                            end_of_question = curr_input_ids.index(self.tokenizer.pad_token_id)
                    else:
                        # in this case, eos must appear in the sequence
                        if self.tokenizer.eos_token_id not in curr_input_ids:
                            curr_input_ids[-1] = self.tokenizer.eos_token_id
                        end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
                except:
                    from IPython import embed; embed(); exit()
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    # see if eos is trimed or not:
                    if self.tokenizer.eos_token_id not in _dpr_input_ids:
                        _dpr_input_ids[-1] = self.tokenizer.eos_token_id
                    # assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    # TODO need to change 32+128 (queslen + psglen) if we use different seqlen
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32+128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32+128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[idx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[idx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[idx][jdx]) == len(qp_attention_mask[idx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data")

        if self.args.use_reranker:
            assert self.args.psg_sel_dir is not None
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference"),
                                          "_20200201" if self.args.wiki_2020 else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
            assert len(fg_passages) == len(qp_input_ids)
            qp_input_ids = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(qp_input_ids, fg_passages)]
            qp_attention_mask = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(qp_attention_mask, fg_passages)]

        self.tokenized_data[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        if self.is_training and self.args.discard_not_found_answers:
            self.logger.info('it seems that discard not found answers will degenerate the results')
            raise NotImplementedError
            # self.discard_not_found_answers()

    def load_dpr_data_bert(self, dpr_retrieval_path, dpr_tokenized_path):
        raise NotImplementedError
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
                return
        self.logger.info("Start processing DPR data")
        with open(dpr_retrieval_path, "r") as f:
            dpr_passages = json.load(f)

        if self.args.ambigqa:
            # added to convert original DPR data to AmbigQA DPR data
            dpr_passages = [dpr_passages[d["orig_idx"]] for d in self.data]
        elif self.is_training:
            with open(os.path.join(self.args.reader_data_dir, "gold_passages_info/nq_train.json"), "r") as f:
                gold_titles = [d["title"] for d in json.load(f)["data"]]
                assert len(gold_titles)==len(self)

        input_ids, attention_mask, answer_input_ids, _, metadata = self.tokenized_data
        assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        if self.passages.tokenized_data is None:
            self.passages.load_tokenized_data("albert" if "Albert" in dpr_tokenized_path else "bert", all=True)
        features = defaultdict(list)
        max_n_answers = self.args.max_n_answers
        oracle_exact_matches = []
        flatten_exact_matches = []
        positive_contains_gold_title = []
        for i, (q_input_ids, q_attention_mask, retrieved) in \
                tqdm(enumerate(zip(input_ids, attention_mask, dpr_passages))):
            assert len(q_input_ids)==len(q_attention_mask)==32
            q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
            assert 3<=len(q_input_ids)<=32
            # TODO Yifan: here we remove the leading [CLS] of retrieved passage, otherwise it will be [CLS] ques [SEP] [CLS] title [SEP] passage [SEP]
            # p_input_ids = [self.passages.tokenized_data["input_ids"][p_idx][1:] + [self.tokenizer.pad_token_id] for p_idx in retrieved]
            # p_attention_mask = [self.passages.tokenized_data["attention_mask"][p_idx][1:] + [0] for p_idx in retrieved]
            p_input_ids = [self.passages.tokenized_data["input_ids"][p_idx] for p_idx in retrieved]
            p_attention_mask = [self.passages.tokenized_data["attention_mask"][p_idx] for p_idx in retrieved]
            a_input_ids = []  # Yifan: in case some answers are ''
            for idx in range(metadata[i][0], metadata[i][1]):
                if len(answer_input_ids[idx]) > 2:
                    a_input_ids.append(answer_input_ids[idx][1:-1])
                else:
                    print(idx)
            detected_spans = []
            for _p_input_ids in p_input_ids:
                detected_spans.append([])
                for _a_input_ids in a_input_ids:
                    decoded_a_input_ids = self.decode(_a_input_ids)
                    for j in range(len(_p_input_ids)-len(_a_input_ids)+1):
                        if _p_input_ids[j:j+len(_a_input_ids)]==_a_input_ids:
                            detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+len(_a_input_ids)-1))
                        elif "Albert" in dpr_tokenized_path and \
                                _p_input_ids[j]==_a_input_ids[0] and \
                                13 in _p_input_ids[j:j+len(_a_input_ids)]:
                            k = j + len(_a_input_ids)+1
                            while k<len(_p_input_ids) and np.sum([_p_input_ids[z]!=13 for z in range(j, k)])<len(_a_input_ids):
                                k += 1
                            if decoded_a_input_ids==self.decode(_p_input_ids[j:k]):
                                detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+k-1))
            if self.args.ambigqa and self.is_training:
                positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0][:20]
                negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
                if len(positives)==0:
                    continue
            elif self.is_training:
                gold_title = normalize_answer(gold_titles[i])
                _positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0]
                if len(_positives)==0:
                    continue
                positives = [j for j in _positives if normalize_answer(self.decode(p_input_ids[j][:p_input_ids[j].index(self.tokenizer.sep_token_id)]))==gold_title]
                positive_contains_gold_title.append(len(positives)>0)
                if len(positives)==0:
                    positives = _positives[:20]
                negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
            else:
                positives = [j for j in range(len(detected_spans))]
                negatives = []
            for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
                        "positive_start_positions", "positive_end_positions", "positive_answer_mask",
                        "negative_input_ids", "negative_input_mask", "negative_token_type_ids"]:
                features[key].append([])

            def _form_input(p_input_ids, p_attention_mask):
                assert len(p_input_ids)==len(p_attention_mask)
                assert len(p_input_ids)==128 or (len(p_input_ids)<=128 and np.sum(p_attention_mask)==len(p_attention_mask))
                if len(p_input_ids)<128:
                    p_input_ids += [self.tokenizer.pad_token_id for _ in range(128-len(p_input_ids))]
                    p_attention_mask += [0 for _ in range(128-len(p_attention_mask))]
                input_ids = q_input_ids + p_input_ids + [self.tokenizer.pad_token_id for _ in range(32-len(q_input_ids))]
                attention_mask = [1 for _ in range(len(q_input_ids))]  + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                token_type_ids = [0 for _ in range(len(q_input_ids))] + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                return input_ids, attention_mask, token_type_ids

            for idx in positives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["positive_input_ids"][-1].append(input_ids)
                features["positive_input_mask"][-1].append(attention_mask)
                features["positive_token_type_ids"][-1].append(token_type_ids)
                detected_span = detected_spans[idx]
                features["positive_start_positions"][-1].append(
                    [s[0] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_end_positions"][-1].append(
                    [s[1] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_answer_mask"][-1].append(
                    [1 for _ in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
            for idx in negatives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["negative_input_ids"][-1].append(input_ids)
                features["negative_input_mask"][-1].append(attention_mask)
                features["negative_token_type_ids"][-1].append(token_type_ids)
            # for debugging
            for p_input_ids, starts, ends, masks in zip(features["positive_input_ids"][-1],
                                                    features["positive_start_positions"][-1],
                                                    features["positive_end_positions"][-1],
                                                    features["positive_answer_mask"][-1]):
                if np.sum(masks)==0: continue
                assert len(starts)==len(ends)==len(masks)==max_n_answers
                decoded_answers = [self.tokenizer.decode(p_input_ids[start:end+1]) for start, end, mask in zip(starts, ends, masks) if mask]
                ems = [get_exact_match(decoded_answer, self.data[i]["answer"]) for decoded_answer in decoded_answers]
                oracle_exact_matches.append(np.max(ems))
                flatten_exact_matches += ems
        print ("oracle exact matches", np.mean(oracle_exact_matches))
        print ("flatten exact matches", np.mean(flatten_exact_matches))
        if self.is_training:
            print ("positive contains gold title", np.mean(positive_contains_gold_title))
        self.tokenized_data = features

        print('Saving', dpr_tokenized_path)
        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)
        print('Done!')

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if isinstance(self.tokenized_data, dict):
            self.dataset = MyQADataset(self.tokenized_data,
                                       is_training=self.is_training,
                                       train_M=self.args.train_M,
                                       test_M=self.args.test_M)
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data[:5]
            self.dataset = MySimpleQADataset(input_ids,
                                             attention_mask,
                                             decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                             decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                             in_metadata=None,
                                             out_metadata=metadata,
                                             is_training=self.is_training,
                                             answer_as_prefix=self.args.nq_answer_as_prefix)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False, **kwargs):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training, **kwargs)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, n_paragraphs=None, predictions_id=None):
        def _included(tokens, curr_input_ids):
            end_of_question = curr_input_ids[0].index(2) + 1
            for _curr_input_ids in curr_input_ids:
                for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                    if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                        return True
            return False
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        ems_ext, ems_abs = [], []
        num_ans_abs, num_ans_ext = 0, 0
        for idx, (pred, pred_id, dp) in enumerate(zip(predictions, predictions_id, self.data)):
            if type(pred)==list:
                pred = pred[0]
            if type(pred)==dict:
                pred = pred["text"]
            ems.append(get_exact_match(pred, dp["answer"]))
            # get prediction ids
            pred_id = pred_id[1:]
            if self.tokenizer.eos_token_id in pred_id:
                eos_idx = pred_id.index(self.tokenizer.eos_token_id)
                pred_id = pred_id[:eos_idx]
            if _included(pred_id, self.tokenized_data[0][idx]):
                num_ans_ext += 1
                ems_ext.append(get_exact_match(pred, dp["answer"]))
            else:
                num_ans_abs += 1
                ems_abs.append(get_exact_match(pred, dp["answer"]))
        self.logger.info("Extractive-Answers={:.2f}, Extractive-Answers-Results={:.2f}; "
                         "Abstractive-Answers={:.2f}, Abstractive-Answers-Results={:.2f};".format(
            100*num_ans_ext/(num_ans_ext+num_ans_abs), 100*np.mean(ems_ext),
            100*num_ans_abs/(num_ans_ext+num_ans_abs), 100*np.mean(ems_abs)))
        result = {
            'EM': 100*np.mean(ems),
            'EM_ext': 100*np.mean(ems_ext),
            'EM_abs': 100*np.mean(ems_abs),
            'ext_percent': 100*num_ans_ext/(num_ans_ext+num_ans_abs),
            'abs_percent': 100*num_ans_abs/(num_ans_ext+num_ans_abs)
        }
        return result['EM'], result

    def save_predictions(self, predictions, mode=''):
        # assert len(predictions)==len(self), (len(predictions), len(self))
        save_path = os.path.join(self.args.output_dir, "{}{}{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 else "",
            "_aq" if self.args.ambigqa else "",
            mode,
        ))
        if self.args.save_psg_sel_only:
            save_path = save_path.replace("predictions.json", "psg_sel.json")
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))


class AmbigQAData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQAData, self).__init__(logger, args, data_path, is_training, passages)

        for i, d in enumerate(self.data):
            answers = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    answers.append([list(set(annotation["answer"]))])
                else:
                    answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers for _answer in answer for _a in _answer])
            self.data[i]["answer"] = answers

        self.metric = "F1"
        self.SEP = "<SEP>"

    # override
    def flatten(self, answers):
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

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type, "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2], "{}{}_predictions.json".format(self.data_type,
                                                                                                                                           "-reos" if self.args.t5_no_intermediate_eos else "",))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        elif "T5" in postfix:
            return self.load_dpr_data_t5(dpr_retrieval_path, dpr_tokenized_path)
        else:
            metadata, new_metadata = self.tokenized_data[-1], []
            for curr_metadata in metadata:
                new_metadata.append((curr_metadata[0][0][0], curr_metadata[-1][-1][-1]))
            self.tokenized_data[-1] = new_metadata
            return self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)

    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            assert self.args.use_reranker, 'currently DPR 1000 passages, so reranker is needed'
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            assert len(dpr_passages)==len(self)

            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(
                                              self.data_type.replace("train", "train_for_inference"),
                                              "_20200201" if self.args.wiki_2020 else "",
                                              "_aq" if self.args.ambigqa else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
                assert len(fg_passages) == len(dpr_passages)
            else:
                raise NotImplementedError

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[idx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[idx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[idx][jdx]) == len(qp_attention_mask[idx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))

        self.tokenized_data[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        if self.is_training:
            _, _, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], []

            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

            # record which input_ids are effective (not filtered by discard_not_found_answers)
            effective_input_idxs = []

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata) in enumerate(zip(
                    self.tokenized_data[0], self.tokenized_data[1], metadata)):
                # now, re-creating decoder_input_ids and metadata
                def _included(tokens, end_of_question):
                    for _curr_input_ids in curr_input_ids:
                        for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                            if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                                return True
                    return False

                def _get_tokenized_answer(idx, append_another_bos):
                    tokens = decoder_input_ids[idx]
                    # remove padded token
                    if 0 in decoder_attention_mask[idx]:
                        tokens = tokens[:decoder_attention_mask[idx].index(0)]
                    if append_another_bos:
                        assert tokens[0] == tokens[1] == bos_token_id and tokens[
                            -1] == self.tokenizer.eos_token_id
                        return tokens[2:-1]
                    else:
                        assert tokens[0] == bos_token_id and tokens[-1] == eos_token_id
                        return tokens[1:-1]

                end_of_question = curr_input_ids[0].index(eos_token_id) + 1
                decoder_offset = len(new_decoder_input_ids)
                for _curr_metadata in curr_metadata:
                    # Yifan: handle answers from each annotator
                    found_answers = []
                    for start, end in _curr_metadata:
                        # Yifan: handle answers from each clusters
                        _answers = []
                        for j in range(start, end):
                            answer = _get_tokenized_answer(j, self.args.append_another_bos)
                            if self.args.discard_not_found_answers:
                                if not _included(answer, end_of_question):
                                    continue
                            if answer in _answers:
                                continue
                            _answers.append(answer)
                        if len(_answers) > 0:
                            found_answers.append(_answers)

                    if len(found_answers) == 0:
                        continue

                    decoder_offset_curr = len(new_decoder_input_ids)
                    cnt = 0
                    cat_answers = []
                    # Yifan: get a combination of answers from all clusters (sample 1 answer from each cluster)
                    for _cat_answers in itertools.product(*found_answers):
                        _cat_answers = list(_cat_answers)
                        cnt_perm = 0
                        for _cat_answers_perm in itertools.permutations(_cat_answers):
                            _cat_answers_perm = list(_cat_answers_perm)
                            answer_input_ids = [bos_token_id]
                            for j, curr_answer in enumerate(_cat_answers_perm):
                                if j > 0: answer_input_ids.append(sep_token_id)
                                answer_input_ids += curr_answer
                            answer_input_ids.append(eos_token_id)
                            if len(answer_input_ids) > self.args.max_cat_answer_length:
                                answer_input_ids = answer_input_ids[:self.args.max_cat_answer_length]
                            cat_answers.append(answer_input_ids)
                            cnt += 1
                            cnt_perm += 1
                            if cnt_perm == 500:
                                break

                    # sample 5 answers per ann
                    if cnt > 100:
                        cnt = 100
                        sel_idx = random.sample(range(len(cat_answers)), cnt)
                    elif 0 < cnt <= 100:
                        sel_idx = list(range(len(cat_answers)))
                    else:
                        continue
                    for jdx in sel_idx:
                        answers = cat_answers[jdx]
                        new_decoder_input_ids.append(
                            answers + [pad_token_id for _ in range(self.args.max_cat_answer_length - len(answers))])
                        new_decoder_attention_mask.append(
                            [1 for _ in answers] + [0 for _ in range(self.args.max_cat_answer_length - len(answers))])
                    assert decoder_offset_curr + cnt == len(new_decoder_input_ids)
                if decoder_offset == len(new_decoder_input_ids):
                    continue
                new_metadata.append([decoder_offset, len(new_decoder_input_ids)])
                effective_input_idxs.append(idx)
            assert len(effective_input_idxs) == len(new_metadata)
            print('Discard Not Found Answers: {}, Training Data {} -> {}'.format(self.args.discard_not_found_answers,
                                                                                 len(self.tokenized_data[0]), len(effective_input_idxs)))
            self.tokenized_data[0] = [self.tokenized_data[0][effective_input_idx] for effective_input_idx in effective_input_idxs]
            self.tokenized_data[1] = [self.tokenized_data[1][effective_input_idx] for effective_input_idx in effective_input_idxs]
            assert len(self.tokenized_data[0]) == len(self.tokenized_data[1]) == len(new_metadata) and \
                   len(new_decoder_input_ids) == len(new_decoder_attention_mask) == new_metadata[-1][-1]
        else:
            new_decoder_input_ids, new_decoder_attention_mask, new_metadata = None, None, None

        self.tokenized_data[2] = new_decoder_input_ids
        self.tokenized_data[3] = new_decoder_attention_mask
        self.tokenized_data[4] = new_metadata

    # override
    def evaluate(self, predictions, n_paragraphs=None, predictions_id=None):
        def _included(tokens, curr_input_ids):
            end_of_question = curr_input_ids[0].index(2) + 1
            for _curr_input_ids in curr_input_ids:
                for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                    if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                        return True
            return False
        assert len(predictions)==len(self), (len(predictions), len(self))
        prfs, prfs_wo_dupli = [], []
        num_ans_pred, num_ans_wo_dupli_pred = [], []
        num_ans_abs, num_ans_ext = 0, 0
        assert self.args.is_seq2seq
        assert len(self.tokenized_data[0]) == len(predictions)
        for idx, (pred, pred_id, dp) in enumerate(zip(predictions, predictions_id, self.data)):
            pred_1 = [text.strip() for text in pred.split(self.SEP)]
            pred_2 = list(set(pred_1))
            num_ans_pred.append(len(pred_1))
            num_ans_wo_dupli_pred.append(len(pred_2))
            curr_prfs, curr_prfs_wo_dupli = [], []
            for answer in dp["answer"]:
                curr_prfs.append(get_f1(answer, pred_1, return_p_and_r=True))
                curr_prfs_wo_dupli.append(get_f1(answer, pred_2, return_p_and_r=True))
            best_curr_prfs = sorted(curr_prfs, key=lambda x:x[0], reverse=True)
            best_curr_prfs_wo_dupli = sorted(curr_prfs_wo_dupli, key=lambda x: x[0], reverse=True)
            prfs.append(best_curr_prfs[0])
            prfs_wo_dupli.append(best_curr_prfs_wo_dupli[0])

            # get prediction ids
            pred_1_id = pred_id[1:]
            if self.tokenizer.eos_token_id in pred_1_id:
                eos_idx = pred_1_id.index(self.tokenizer.eos_token_id)
                pred_1_id = pred_1_id[:eos_idx]
            pred_2_id = [[]]
            for id in pred_1_id:
                if id != self.tokenizer.convert_tokens_to_ids(self.SEP):
                    pred_2_id[-1].append(id)
                else:
                    pred_2_id.append([])
            for idx in range(len(pred_2_id)):
                pred_2_id[idx] = tuple(pred_2_id[idx])
            pred_2_id = list(set(pred_2_id))
            pred_2_id = [list(x) for x in pred_2_id]
            for pred_tkd in pred_2_id:
                if _included(pred_tkd, self.tokenized_data[0][idx]):
                    num_ans_ext += 1
                else:
                    num_ans_abs += 1

        self.logger.info("Num-Ans={}, Num-Ans-No-Dupli={}".format(
            sum(num_ans_pred), sum(num_ans_wo_dupli_pred)))
        self.logger.info("Dupli-Answers-F1={:.2f}".format(np.mean([x[0] for x in prfs])*100))
        self.logger.info("PAND-P={:.2f}, PAND-R={:.2f}, PAND-F1={:.2f}".format(
            np.mean([x[1] for x in prfs_wo_dupli])*100,
            np.mean([x[2] for x in prfs_wo_dupli])*100,
            np.mean([x[0] for x in prfs_wo_dupli])*100,))
        self.logger.info("Ext={:.2f}, Abs={:.2f}".format(
            num_ans_ext / (num_ans_ext + num_ans_abs) * 100,
            num_ans_abs / (num_ans_ext + num_ans_abs) * 100))
        results = {
            'Ans-F1': np.mean([x[0] for x in prfs_wo_dupli])*100,
            'Ans-P': np.mean([x[1] for x in prfs_wo_dupli])*100,
            'Ans-R': np.mean([x[2] for x in prfs_wo_dupli]) * 100,
            'Ext_percent': num_ans_ext / (num_ans_ext + num_ans_abs) * 100,
            'Abs_percent': num_ans_abs / (num_ans_ext + num_ans_abs) * 100,
            'Num_Ans': sum(num_ans_pred),
            'Num-Ans-No-Dupli': sum(num_ans_wo_dupli_pred),
            'Ans-F1-Dupli': np.mean([x[0] for x in prfs])*100,
        }
        return results['Ans-F1'], results


class AmbigQADataLeaderboard(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQADataLeaderboard, self).__init__(logger, args, data_path, is_training, passages)
        self.metric = "F1"
        self.SEP = "<SEP>"

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type, "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2], "{}{}_predictions.json".format(self.data_type,
                                                                                                                                           "-reos" if self.args.t5_no_intermediate_eos else "",))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        elif "T5" in postfix:
            return self.load_dpr_data_t5(dpr_retrieval_path, dpr_tokenized_path)
        else:
            metadata, new_metadata = self.tokenized_data[-1], []
            for curr_metadata in metadata:
                new_metadata.append((curr_metadata[0][0][0], curr_metadata[-1][-1][-1]))
            self.tokenized_data[-1] = new_metadata
            return self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)


class DisAmbigQAData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(DisAmbigQAData, self).__init__(logger, args, data_path, is_training, passages)

        for i, d in enumerate(self.data):
            qapairs = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    qapairs.append({
                        'question': d['question'],
                        'answers': list(set(annotation["answer"])),
                    })
                else:
                    for pair in annotation["qaPairs"]:
                        qapairs.append({
                            'question': pair["question"],
                            'answers': list(set(pair["answer"])),
                        })
            self.data[i]["qapair"] = qapairs

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix or 'T5' in postfix
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "-noamb{}{}{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if self.args.append_another_bos else "",
                    "-reos" if self.args.t5_no_intermediate_eos else "",
                    postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print ("Start tokenizing...")
            # reformat questions and answers
            qapair_metadata, questions, answers = [], [], []
            for d in self.data:
                curr_questions = [qapair['question'] for qapair in d['qapair']]
                qapair_metadata.append((len(questions), len(questions)+len(curr_questions)))
                curr_answers = [qapair['answers'] for qapair in d['qapair']]
                questions.extend(curr_questions)
                answers.extend(curr_answers)
            answers, metadata = self.flatten(answers)
            if self.args.bert_name.startswith("t5"):
                if self.args.t5_no_intermediate_eos:
                    questions = ["question: " + question for question in questions]
                else:
                    questions = ["question: " + question + " </s>" for question in questions]
                answers = [answer + " </s>" for answer in answers]
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix or "T5" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask, metadata, qapair_metadata]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f)
        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type, "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}_noamb{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "",))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        elif "T5" in postfix:
            return self.load_dpr_data_t5(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            assert len(dpr_passages) == len(self)

            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(
                                              self.data_type.replace("train", "train_for_inference"),
                                              "_20200201" if self.args.wiki_2020 else "",
                                              "_aq" if self.args.ambigqa else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                assert len(fg_passages) == len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata, qapair_metadata = self.tokenized_data
            assert len(dpr_passages) == len(qapair_metadata)
            assert len(input_ids) == len(attention_mask) == len(metadata)
            assert metadata[-1][-1] == len(decoder_input_ids) == len(decoder_attention_mask)
            assert len(metadata) == qapair_metadata[-1][-1]
            bos_token_id = self.tokenizer.bos_token_id

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]
            for idx, (curr_qapair_metadata, dpr_ids) in enumerate(zip(qapair_metadata, dpr_passages)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for qdx in range(*curr_qapair_metadata):
                    curr_input_ids, curr_attention_mask = input_ids[qdx], attention_mask[qdx]
                    end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                        assert _dpr_input_ids[0] == bos_token_id
                        qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                        qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                        assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                        qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in
                                                  range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                        qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                        qp_input_ids[qdx].append(qp_inputs_ids_idx_jdx)
                        qp_attention_mask[qdx].append(qp_attention_mask_idx_jdx)
                        assert len(qp_input_ids[qdx][jdx]) == len(qp_attention_mask[qdx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data")

        self.tokenized_data[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        if self.is_training and self.args.discard_not_found_answers:
            self.discard_not_found_answers()

    def evaluate(self, predictions, n_paragraphs=None, predictions_id=None):
        def _included(tokens, curr_input_ids):
            end_of_question = curr_input_ids[0].index(2) + 1
            for _curr_input_ids in curr_input_ids:
                for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                    if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                        return True
            return False
        assert len(predictions) == self.tokenized_data[-1][-1][-1]
        ems = []
        ems_ext, ems_abs = [], []
        num_ans_abs, num_ans_ext = 0, 0
        for idx, (pred_metadata, dp) in enumerate(zip(self.tokenized_data[-1], self.data)):
            for dp_jdx, pred_jdx in enumerate(range(*pred_metadata)):
                pred = predictions[pred_jdx]
                pred_id = predictions_id[pred_jdx]
                dp_answer = dp['qapair'][dp_jdx]['answers']
                if type(pred) == list:
                    pred = pred[0]
                if type(pred) == dict:
                    pred = pred["text"]
                ems.append(get_exact_match(pred, dp_answer))
                # get prediction ids
                pred_id = pred_id[1:]
                if self.tokenizer.eos_token_id in pred_id:
                    eos_idx = pred_id.index(self.tokenizer.eos_token_id)
                    pred_id = pred_id[:eos_idx]
                if _included(pred_id, self.tokenized_data[0][idx]):
                    num_ans_ext += 1
                    ems_ext.append(get_exact_match(pred, dp_answer))
                else:
                    num_ans_abs += 1
                    ems_abs.append(get_exact_match(pred, dp_answer))
        self.logger.info("Ext-Ans={:.2f}%, Ext-Ans-Results={:.2f}; "
                         "Abs-Ans={:.2f}%, Abs-Ans-Results={:.2f};".format(
            100 * num_ans_ext / (num_ans_ext + num_ans_abs), 100 * np.mean(ems_ext),
            100 * num_ans_abs / (num_ans_ext + num_ans_abs), 100 * np.mean(ems_abs)))
        result = {
            'EM': 100 * np.mean(ems),
            'EM_ext': 100 * np.mean(ems_ext),
            'EM_abs': 100 * np.mean(ems_abs),
            'ext_percent': 100 * num_ans_ext / (num_ans_ext + num_ans_abs),
            'abs_percent': 100 * num_ans_abs / (num_ans_ext + num_ans_abs)
        }
        return result['EM'], result


class AmbigQACoTrainingLabelData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQACoTrainingLabelData, self).__init__(logger, args, data_path, is_training, passages)

        self.metric = "F1"
        self.SEP = "<SEP>"

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}_predictions.json".format(self.data_type)).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2], "{}_predictions_cotraining.json".format(self.data_type))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            assert self.args.use_reranker, 'currently DPR 1000 passages, so reranker is needed'
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            assert len(dpr_passages)==len(self)

            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(
                                              self.data_type.replace("train", "train_for_inference"),
                                              "_20200201" if self.args.wiki_2020 else "",
                                              "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
                assert len(fg_passages) == len(dpr_passages)
            else:
                raise NotImplementedError

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[idx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[idx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[idx][jdx]) == len(qp_attention_mask[idx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))

        self.tokenized_data[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        new_decoder_input_ids, new_decoder_attention_mask, new_metadata = None, None, None

        self.tokenized_data[2] = new_decoder_input_ids
        self.tokenized_data[3] = new_decoder_attention_mask
        self.tokenized_data[4] = new_metadata


class AmbigQACoTrainingTrainData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQACoTrainingTrainData, self).__init__(logger, args, data_path, is_training, passages)

        for i, d in enumerate(self.data):
            answers = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    answers.append([list(set(annotation["answer"]))])
                else:
                    answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers for _answer in answer for _a in _answer])
            self.data[i]["answer"] = answers

        self.metric = "F1"
        self.SEP = "<SEP>"

    # override
    def flatten(self, answers):
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

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix or 'T5' in postfix
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}{}{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if self.args.append_another_bos else "",
                    "-reos" if self.args.t5_no_intermediate_eos else "",
                    postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print ("Start tokenizing...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
            answers, metadata = self.flatten(answers)
            if self.args.bert_name.startswith("t5"):
                if self.args.t5_no_intermediate_eos:
                    questions = ["question: " + question for question in questions]
                else:
                    questions = ["question: " + question + " </s>" for question in questions]
                answers = [answer + " </s>" for answer in answers]
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix or "T5" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask, metadata]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f)
        if self.is_training:
            self.tokenized_data_silver = tokenized_data
            preprocessed_path = os.path.join("/".join(self.data_path.split("/")[:-1]), "{}{}-{}.json".format(
                self.data_type, "-uncased" if self.args.do_lowercase else "", postfix))
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type, "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")

        if "Bart" in postfix:
            if self.is_training:
                dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                                  "{}{}_predictions.json".format(self.data_type,
                                                                                 "-reos" if self.args.t5_no_intermediate_eos else "", ))
                dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
                self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
                gold_tokenized_data = self.tokenized_data

                silver_version = self.args.train_file.split("/")[-1].replace(".json", "").replace("train_", "")
                dpr_tokenized_path_silver = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                                  "{}{}_predictions.json".format(self.args.train_file.split("/")[-1].replace(".json", ""),
                                                                                 "-reos" if self.args.t5_no_intermediate_eos else "", ))
                dpr_tokenized_path_silver = dpr_tokenized_path_silver.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
                dpr_retrieval_path_silver = os.path.join(self.args.dpr_data_dir, "{}{}_predictions_{}.json".format(self.data_type, "", silver_version)).replace('train_for_inference', 'train')
                reranking_path_silver = os.path.join(self.args.psg_sel_dir.replace('ambigqa', 'nqopen'), "{}_psg_sel_{}.json".format(
                    self.data_type.replace("train", "train_for_inference"), silver_version))
                silver_tokenized_data = self.load_dpr_data_bart_silver(dpr_retrieval_path_silver, dpr_tokenized_path_silver, reranking_path_silver)
                # update merged metadata
                silver_metadata_offset = len(gold_tokenized_data[2])
                for idx in range(len(silver_tokenized_data[-1])):
                    for jdx in range(len(silver_tokenized_data[-1][idx])):
                        silver_tokenized_data[-1][idx][jdx] += silver_metadata_offset
                merged_tokenized_data = [x + y for x, y in zip(gold_tokenized_data, silver_tokenized_data)]
                self.tokenized_data = merged_tokenized_data
            else:
                dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                                  "{}{}_predictions.json".format(self.args.predict_file.split("/")[-1].replace(".json", ""),
                                                                                 "-reos" if self.args.t5_no_intermediate_eos else "", ))
                dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
                return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            assert self.args.use_reranker, 'currently DPR 1000 passages, so reranker is needed'
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            assert len(dpr_passages)==len(self)

            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(
                                              self.data_type.replace("train", "train_for_inference"),
                                              "_20200201" if self.args.wiki_2020 else "",
                                              "_aq" if self.args.ambigqa else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
                assert len(fg_passages) == len(dpr_passages)
            else:
                raise NotImplementedError

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[idx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[idx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[idx][jdx]) == len(qp_attention_mask[idx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))

        self.tokenized_data[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        if self.is_training:
            _, _, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], []

            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

            # record which input_ids are effective (not filtered by discard_not_found_answers)
            effective_input_idxs = []

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata) in enumerate(zip(
                    self.tokenized_data[0], self.tokenized_data[1], metadata)):
                # now, re-creating decoder_input_ids and metadata
                def _included(tokens, end_of_question):
                    for _curr_input_ids in curr_input_ids:
                        for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                            if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                                return True
                    return False

                def _get_tokenized_answer(idx, append_another_bos):
                    tokens = decoder_input_ids[idx]
                    # remove padded token
                    if 0 in decoder_attention_mask[idx]:
                        tokens = tokens[:decoder_attention_mask[idx].index(0)]
                    if append_another_bos:
                        assert tokens[0] == tokens[1] == bos_token_id and tokens[
                            -1] == self.tokenizer.eos_token_id
                        return tokens[2:-1]
                    else:
                        assert tokens[0] == bos_token_id and tokens[-1] == eos_token_id
                        return tokens[1:-1]

                end_of_question = curr_input_ids[0].index(eos_token_id) + 1
                decoder_offset = len(new_decoder_input_ids)
                for _curr_metadata in curr_metadata:
                    # Yifan: handle answers from each annotator
                    found_answers = []
                    for start, end in _curr_metadata:
                        # Yifan: handle answers from each clusters
                        _answers = []
                        for j in range(start, end):
                            answer = _get_tokenized_answer(j, self.args.append_another_bos)
                            if self.args.discard_not_found_answers:
                                if not _included(answer, end_of_question):
                                    continue
                            if answer in _answers:
                                continue
                            _answers.append(answer)
                        if len(_answers) > 0:
                            found_answers.append(_answers)

                    if len(found_answers) == 0:
                        continue

                    decoder_offset_curr = len(new_decoder_input_ids)
                    cnt = 0
                    cat_answers = []
                    # Yifan: get a combination of answers from all clusters (sample 1 answer from each cluster)
                    for _cat_answers in itertools.product(*found_answers):
                        _cat_answers = list(_cat_answers)
                        cnt_perm = 0
                        for _cat_answers_perm in itertools.permutations(_cat_answers):
                            _cat_answers_perm = list(_cat_answers_perm)
                            answer_input_ids = [bos_token_id]
                            for j, curr_answer in enumerate(_cat_answers_perm):
                                if j > 0: answer_input_ids.append(sep_token_id)
                                answer_input_ids += curr_answer
                            answer_input_ids.append(eos_token_id)
                            if len(answer_input_ids) > self.args.max_cat_answer_length:
                                answer_input_ids = answer_input_ids[:self.args.max_cat_answer_length]
                            cat_answers.append(answer_input_ids)
                            cnt += 1
                            cnt_perm += 1
                            if cnt_perm == 500:
                                break

                    # sample 5 answers per ann
                    if cnt > 100:
                        cnt = 100
                        sel_idx = random.sample(range(len(cat_answers)), cnt)
                    elif 0 < cnt <= 100:
                        sel_idx = list(range(len(cat_answers)))
                    else:
                        continue
                    for jdx in sel_idx:
                        answers = cat_answers[jdx]
                        new_decoder_input_ids.append(
                            answers + [pad_token_id for _ in range(self.args.max_cat_answer_length - len(answers))])
                        new_decoder_attention_mask.append(
                            [1 for _ in answers] + [0 for _ in range(self.args.max_cat_answer_length - len(answers))])
                    assert decoder_offset_curr + cnt == len(new_decoder_input_ids)
                if decoder_offset == len(new_decoder_input_ids):
                    continue
                new_metadata.append([decoder_offset, len(new_decoder_input_ids)])
                effective_input_idxs.append(idx)
            assert len(effective_input_idxs) == len(new_metadata)
            print('Discard Not Found Answers: {}, Training Data {} -> {}'.format(self.args.discard_not_found_answers,
                                                                                 len(self.tokenized_data[0]), len(effective_input_idxs)))
            self.tokenized_data[0] = [self.tokenized_data[0][effective_input_idx] for effective_input_idx in effective_input_idxs]
            self.tokenized_data[1] = [self.tokenized_data[1][effective_input_idx] for effective_input_idx in effective_input_idxs]
            assert len(self.tokenized_data[0]) == len(self.tokenized_data[1]) == len(new_metadata) and \
                   len(new_decoder_input_ids) == len(new_decoder_attention_mask) == new_metadata[-1][-1]
        else:
            new_decoder_input_ids, new_decoder_attention_mask, new_metadata = None, None, None

        self.tokenized_data[2] = new_decoder_input_ids
        self.tokenized_data[3] = new_decoder_attention_mask
        self.tokenized_data[4] = new_metadata


    def load_dpr_data_bart_silver(self, dpr_retrieval_path, dpr_tokenized_path, reranking_path):
        # process silver data
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                qp_input_ids, qp_attention_mask = json.load(f)
        else:
            assert self.args.use_reranker, 'currently DPR 1000 passages, so reranker is needed'
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.data_path = os.path.join(self.args.dpr_data_dir, "wikipedia_split/psgs_w100.tsv.gz")
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)

            if self.args.use_reranker:
                self.logger.info("Loading passage selection from DPR reader: {}".format(reranking_path))
                with open(reranking_path, "r") as f:
                    fg_passages = json.load(f)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
                assert len(fg_passages) == len(dpr_passages)
            else:
                raise NotImplementedError

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data_silver
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            # question - passage (with title)
            qp_input_ids, qp_attention_mask = [[] for _ in input_ids], [[] for _ in attention_mask]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                    assert _dpr_input_ids[0] == bos_token_id
                    qp_inputs_ids_idx_jdx = curr_input_ids[:end_of_question] + _dpr_input_ids[1:]
                    qp_attention_mask_idx_jdx = curr_attention_mask[:end_of_question] + _dpr_attention_mask[1:]
                    assert len(qp_inputs_ids_idx_jdx) == len(qp_attention_mask_idx_jdx)
                    qp_inputs_ids_idx_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(qp_inputs_ids_idx_jdx))]
                    qp_attention_mask_idx_jdx += [0 for _ in range(32 + 128 - len(qp_attention_mask_idx_jdx))]
                    qp_input_ids[idx].append(qp_inputs_ids_idx_jdx)
                    qp_attention_mask[idx].append(qp_attention_mask_idx_jdx)
                    assert len(qp_input_ids[idx][jdx]) == len(qp_attention_mask[idx][jdx]) == 160  # here we use 32+128

            with open(dpr_tokenized_path, "w") as f:
                json.dump([qp_input_ids, qp_attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))

        self.tokenized_data_silver[0] = [_qp_input_ids[:self.args.top_k_passages] for _qp_input_ids in qp_input_ids]
        self.tokenized_data_silver[1] = [_qp_attention_mask[:self.args.top_k_passages] for _qp_attention_mask in qp_attention_mask]

        _, _, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data_silver
        new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], []

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

        # record which input_ids are effective (not filtered by discard_not_found_answers)
        effective_input_idxs = []

        for idx, (curr_input_ids, curr_attention_mask, curr_metadata) in enumerate(zip(
                self.tokenized_data_silver[0], self.tokenized_data_silver[1], metadata)):
            # now, re-creating decoder_input_ids and metadata
            def _included(tokens, end_of_question):
                for _curr_input_ids in curr_input_ids:
                    for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                        if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                            return True
                return False

            def _get_tokenized_answer(idx, append_another_bos):
                tokens = decoder_input_ids[idx]
                # remove padded token
                if 0 in decoder_attention_mask[idx]:
                    tokens = tokens[:decoder_attention_mask[idx].index(0)]
                if append_another_bos:
                    assert tokens[0] == tokens[1] == bos_token_id and tokens[
                        -1] == self.tokenizer.eos_token_id
                    return tokens[2:-1]
                else:
                    assert tokens[0] == bos_token_id and tokens[-1] == eos_token_id
                    return tokens[1:-1]

            end_of_question = curr_input_ids[0].index(eos_token_id) + 1
            decoder_offset = len(new_decoder_input_ids)
            for _curr_metadata in curr_metadata:
                # Yifan: handle answers from each annotator
                found_answers = []
                for start, end in _curr_metadata:
                    # Yifan: handle answers from each clusters
                    _answers = []
                    for j in range(start, end):
                        answer = _get_tokenized_answer(j, self.args.append_another_bos)
                        if self.args.discard_not_found_answers:
                            if not _included(answer, end_of_question):
                                continue
                        if answer in _answers:
                            continue
                        _answers.append(answer)
                    if len(_answers) > 0:
                        found_answers.append(_answers)

                if len(found_answers) == 0:
                    continue

                decoder_offset_curr = len(new_decoder_input_ids)
                cnt = 0
                cat_answers = []
                # Yifan: get a combination of answers from all clusters (sample 1 answer from each cluster)
                for _cat_answers in itertools.product(*found_answers):
                    _cat_answers = list(_cat_answers)
                    cnt_perm = 0
                    for _cat_answers_perm in itertools.permutations(_cat_answers):
                        _cat_answers_perm = list(_cat_answers_perm)
                        answer_input_ids = [bos_token_id]
                        for j, curr_answer in enumerate(_cat_answers_perm):
                            if j > 0: answer_input_ids.append(sep_token_id)
                            answer_input_ids += curr_answer
                        answer_input_ids.append(eos_token_id)
                        if len(answer_input_ids) > self.args.max_cat_answer_length:
                            answer_input_ids = answer_input_ids[:self.args.max_cat_answer_length]
                        cat_answers.append(answer_input_ids)
                        cnt += 1
                        cnt_perm += 1
                        if cnt_perm == 500:
                            break

                # sample 5 answers per ann
                if cnt > 100:
                    cnt = 100
                    sel_idx = random.sample(range(len(cat_answers)), cnt)
                elif 0 < cnt <= 100:
                    sel_idx = list(range(len(cat_answers)))
                else:
                    continue
                for jdx in sel_idx:
                    answers = cat_answers[jdx]
                    new_decoder_input_ids.append(
                        answers + [pad_token_id for _ in range(self.args.max_cat_answer_length - len(answers))])
                    new_decoder_attention_mask.append(
                        [1 for _ in answers] + [0 for _ in range(self.args.max_cat_answer_length - len(answers))])
                assert decoder_offset_curr + cnt == len(new_decoder_input_ids)
            if decoder_offset == len(new_decoder_input_ids):
                continue
            new_metadata.append([decoder_offset, len(new_decoder_input_ids)])
            effective_input_idxs.append(idx)
        assert len(effective_input_idxs) == len(new_metadata)
        print('Discard Not Found Answers: {}, Training Data {} -> {}'.format(self.args.discard_not_found_answers,
                                                                             len(self.tokenized_data_silver[0]), len(effective_input_idxs)))
        self.tokenized_data_silver[0] = [self.tokenized_data_silver[0][effective_input_idx] for effective_input_idx in effective_input_idxs]
        self.tokenized_data_silver[1] = [self.tokenized_data_silver[1][effective_input_idx] for effective_input_idx in effective_input_idxs]
        assert len(self.tokenized_data_silver[0]) == len(self.tokenized_data_silver[1]) == len(new_metadata) and \
               len(new_decoder_input_ids) == len(new_decoder_attention_mask) == new_metadata[-1][-1]

        self.tokenized_data_silver[2] = new_decoder_input_ids
        self.tokenized_data_silver[3] = new_decoder_attention_mask
        self.tokenized_data_silver[4] = new_metadata
        return self.tokenized_data_silver

    # override
    def evaluate(self, predictions, n_paragraphs=None, predictions_id=None):
        def _included(tokens, curr_input_ids):
            end_of_question = curr_input_ids[0].index(2) + 1
            for _curr_input_ids in curr_input_ids:
                for jdx in range(end_of_question, len(_curr_input_ids) - len(tokens) + 1):
                    if _curr_input_ids[jdx:jdx + len(tokens)] == tokens:
                        return True
            return False
        assert len(predictions)==len(self), (len(predictions), len(self))
        prfs, prfs_wo_dupli = [], []
        num_ans_pred, num_ans_wo_dupli_pred = [], []
        num_ans_abs, num_ans_ext = 0, 0
        assert self.args.is_seq2seq
        assert len(self.tokenized_data[0]) == len(predictions)
        for idx, (pred, pred_id, dp) in enumerate(zip(predictions, predictions_id, self.data)):
            pred_1 = [text.strip() for text in pred.split(self.SEP)]
            pred_2 = list(set(pred_1))
            num_ans_pred.append(len(pred_1))
            num_ans_wo_dupli_pred.append(len(pred_2))
            curr_prfs, curr_prfs_wo_dupli = [], []
            for answer in dp["answer"]:
                curr_prfs.append(get_f1(answer, pred_1, return_p_and_r=True))
                curr_prfs_wo_dupli.append(get_f1(answer, pred_2, return_p_and_r=True))
            best_curr_prfs = sorted(curr_prfs, key=lambda x:x[0], reverse=True)
            best_curr_prfs_wo_dupli = sorted(curr_prfs_wo_dupli, key=lambda x: x[0], reverse=True)
            prfs.append(best_curr_prfs[0])
            prfs_wo_dupli.append(best_curr_prfs_wo_dupli[0])

            # get prediction ids
            pred_1_id = pred_id[1:]
            if self.tokenizer.eos_token_id in pred_1_id:
                eos_idx = pred_1_id.index(self.tokenizer.eos_token_id)
                pred_1_id = pred_1_id[:eos_idx]
            pred_2_id = [[]]
            for id in pred_1_id:
                if id != self.tokenizer.convert_tokens_to_ids(self.SEP):
                    pred_2_id[-1].append(id)
                else:
                    pred_2_id.append([])
            for idx in range(len(pred_2_id)):
                pred_2_id[idx] = tuple(pred_2_id[idx])
            pred_2_id = list(set(pred_2_id))
            pred_2_id = [list(x) for x in pred_2_id]
            for pred_tkd in pred_2_id:
                if _included(pred_tkd, self.tokenized_data[0][idx]):
                    num_ans_ext += 1
                else:
                    num_ans_abs += 1

        self.logger.info("Num-Ans={}, Num-Ans-No-Dupli={}".format(
            sum(num_ans_pred), sum(num_ans_wo_dupli_pred)))
        self.logger.info("Dupli-Answers-F1={:.2f}".format(np.mean([x[0] for x in prfs])*100))
        self.logger.info("PAND-P={:.2f}, PAND-R={:.2f}, PAND-F1={:.2f}".format(
            np.mean([x[1] for x in prfs_wo_dupli])*100,
            np.mean([x[2] for x in prfs_wo_dupli])*100,
            np.mean([x[0] for x in prfs_wo_dupli])*100,))
        self.logger.info("Ext={:.2f}, Abs={:.2f}".format(
            num_ans_ext / (num_ans_ext + num_ans_abs) * 100,
            num_ans_abs / (num_ans_ext + num_ans_abs) * 100))
        results = {
            'Ans-F1': np.mean([x[0] for x in prfs_wo_dupli])*100,
            'Ans-P': np.mean([x[1] for x in prfs_wo_dupli])*100,
            'Ans-R': np.mean([x[2] for x in prfs_wo_dupli]) * 100,
            'Ext_percent': num_ans_ext / (num_ans_ext + num_ans_abs) * 100,
            'Abs_percent': num_ans_abs / (num_ans_ext + num_ans_abs) * 100,
            'Num_Ans': sum(num_ans_pred),
            'Num-Ans-No-Dupli': sum(num_ans_wo_dupli_pred),
            'Ans-F1-Dupli': np.mean([x[0] for x in prfs])*100,
        }
        return results['Ans-F1'], results