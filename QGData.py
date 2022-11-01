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

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from QAData import QAData, AmbigQAData
from DataLoader import MySimpleQADataset, MySimpleQGDataset, MyDataLoader, MySimpleQGDynamicDataset, MySimpleQGDynamicWeightedLossDataset, MySimpleQGWeightedLossDataset
from util import decode_span_batch

# for evaluation
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
from pycocoevalcap.bleu.bleu import Bleu

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]
PUNCT_WORDS = set(string.punctuation)
IGNORE_WORDS = {'in', 'the', 'is', 'at', 'which', 'on', 'what', 'who', 'where', 'when', 'how', 'with', 'a', 'about', 'an', 'are', 'as', 'at', 'be',
                'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'will', 'with', 'do', 'does', 'did',
                'i', 'me', 'we', 'our', 'ours', 'he', 'his', 'her', 'she', 'they', 'their', 'mine', 'theirs', 'ours', 'how', 'is', 'are', 'were', 'was',
                'will', 'would', 'by', 'in', 'on', 'under', 'above', }
VALID_POS = ['ADJ', 'NOUN', 'NUM', 'PROPN', 'SYM', 'VERB']

class QGData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(QGData, self).__init__(logger, args, data_path, is_training, passages)
        self.metric = "Bleu"
        if args.do_train or args.task == "qg_mask":
            import spacy
            self.qg_tokenizer = spacy.load("en_core_web_sm")
        else:
            self.qg_tokenizer = PTBTokenizer()

    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}_predictions.json".format(
            self.data_type + "_20200201" if self.args.wiki_2020 else self.data_type)).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "", ))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_{}_qg.json".format(postfix))
        if "Bart" in postfix:
            if not self.args.filter_not_found_answer_passages:
                return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
            else:
                raise NotImplementedError
                return self.load_dpr_data_bart_filter_nfa_psgs(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    def load_dpr_data_bart_old(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            # if "train_for_inference" not in dpr_retrieval_path:
            #     dpr_retrieval_path = dpr_retrieval_path.replace("train", "train_for_inference")
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
                assert len(dpr_passages)==len(self)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages) == len(input_ids) == len(attention_mask) == len(metadata)
            bos_token_id = self.tokenizer.bos_token_id

            def _included(tokens, curr_input_ids, end_of_answer):
                is_answer_exist = []
                for _curr_input_ids in curr_input_ids:
                    is_exist = False
                    for jdx in range(end_of_answer, len(_curr_input_ids)-len(tokens)+1):
                        if _curr_input_ids[jdx:jdx+len(tokens)]==tokens:
                            is_exist = True
                    is_answer_exist.append(is_exist)
                return is_answer_exist

            new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_is_valid_list = [], [], [], [], []
            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                # create multiple inputs
                answer_input_ids_list, answer_attention_mask_list, is_valid_list = [], [], []
                for answer_idx in range(*curr_metadata):
                    end_of_answer = decoder_input_ids[answer_idx].index(self.tokenizer.eos_token_id) + 1
                    answer_input_ids = decoder_input_ids[answer_idx][:end_of_answer]
                    answer_attention_mask = decoder_attention_mask[answer_idx][:end_of_answer]
                    ap_input_ids, ap_attention_mask = [], []
                    for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                        assert _dpr_input_ids[0] == bos_token_id
                        answer_input_ids_jdx = answer_input_ids + _dpr_input_ids[1:]
                        answer_attention_mask_jdx = answer_attention_mask + _dpr_attention_mask[1:]
                        assert len(answer_input_ids_jdx) == len(answer_attention_mask_jdx)
                        answer_input_ids_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(answer_input_ids_jdx))]
                        answer_attention_mask_jdx += [0 for _ in range(32 + 128 - len(answer_attention_mask_jdx))]
                        ap_input_ids.append(answer_input_ids_jdx)
                        ap_attention_mask.append(answer_attention_mask_jdx)
                        assert len(ap_input_ids[jdx]) == len(ap_attention_mask[jdx]) == 160  # here we use 32+128
                    assert len(ap_input_ids) == len(ap_attention_mask) == 100
                    answer_input_ids_list.append(ap_input_ids)
                    answer_attention_mask_list.append(ap_attention_mask)
                    is_valid_list.append(_included(decoder_input_ids[answer_idx][1:end_of_answer - 1], ap_input_ids, end_of_answer))

                assert len(answer_input_ids_list) == len(answer_attention_mask_list) == len(is_valid_list) == curr_metadata[1] - curr_metadata[0]

                new_decoder_input_ids.append(curr_input_ids)
                new_decoder_attention_mask.append(curr_attention_mask)
                new_input_ids.append(answer_input_ids_list)
                new_attention_mask.append(answer_attention_mask_list)
                new_is_valid_list.append(is_valid_list)
            assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(new_input_ids) == len(new_attention_mask) == len(new_is_valid_list) == len(self)

            self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_is_valid_list]
            with open(dpr_tokenized_path, "w") as f:
                json.dump(self.tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))

        raw_input_ids, raw_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, raw_is_valid_list = self.tokenized_data
        if self.args.use_reranker:
            assert self.args.psg_sel_dir is not None
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference"),
                                          "_20200201" if self.args.wiki_2020 else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
            assert len(fg_passages) == len(raw_input_ids)

            ranked_raw_input_ids, ranked_raw_attention_mask, ranked_raw_is_valid_list = [], [], []
            for idx, fg_psgs in enumerate(fg_passages):
                ranked_raw_input_ids.append([[x[i] for i in fg_psgs][:self.args.top_k_passages] for x in raw_input_ids[idx]])
                ranked_raw_attention_mask.append([[x[i] for i in fg_psgs][:self.args.top_k_passages] for x in raw_attention_mask[idx]])
                ranked_raw_is_valid_list.append([[x[i] for i in fg_psgs][:self.args.top_k_passages] for x in raw_is_valid_list[idx]])
        else:
            ranked_raw_input_ids, ranked_raw_attention_mask, ranked_raw_is_valid_list = [], [], []
            for idx in range(len(raw_input_ids)):
                ranked_raw_input_ids.append([x[:self.args.top_k_passages] for x in raw_input_ids[idx]])
                ranked_raw_attention_mask.append([x[:self.args.top_k_passages] for x in raw_attention_mask[idx]])
                ranked_raw_is_valid_list.append([x[:self.args.top_k_passages] for x in raw_is_valid_list[idx]])

        # prepare for customized training and inference dataset
        new_input_ids, new_attention_mask, new_metadata = [], [], []
        has_valid = []
        if not self.is_training:
            old_decoder_input_ids, old_decoder_attention_mask, = new_decoder_input_ids, new_decoder_attention_mask,
            new_decoder_input_ids, new_decoder_attention_mask, = [], []
        for idx, (rr_input_ids, rr_attention_mask, rr_is_valid_list) in enumerate(zip(ranked_raw_input_ids, ranked_raw_attention_mask, ranked_raw_is_valid_list)):
            topk_is_valid_list = [any(rr_is_valid) for rr_is_valid in rr_is_valid_list]
            has_valid.append(any(topk_is_valid_list))
            if self.is_training:
                if self.args.discard_not_found_answers:
                    if not any(topk_is_valid_list):
                        topk_is_valid_list = [True for _ in topk_is_valid_list]
                    new_metadata.append((len(new_input_ids), len(new_input_ids) + sum(topk_is_valid_list)))
                    new_input_ids += [answer_input_ids for answer_input_ids, is_valid in zip(rr_input_ids, topk_is_valid_list) if is_valid]
                    new_attention_mask += [answer_attention_mask for answer_attention_mask, is_valid in zip(rr_attention_mask, topk_is_valid_list) if is_valid]
                else:
                    offset = len(new_input_ids)
                    new_input_ids.extend(rr_input_ids)
                    new_attention_mask.extend(rr_attention_mask)
                    new_metadata.append((offset, len(new_input_ids)))
            else:
                # Yifan: original code only evaluates when gold sample is found, but here we evaluate all qa pairs
                # we generate all questions conditioned on all answers, and get the best if the question have multiple answer candidates
                offset = len(new_input_ids)
                new_input_ids.extend(rr_input_ids)
                new_attention_mask.extend(rr_attention_mask)
                for i in range(offset, len(new_input_ids)):
                    new_metadata.append((i,i+1))
                    new_decoder_input_ids.append(old_decoder_input_ids[idx])
                    new_decoder_attention_mask.append(old_decoder_attention_mask[idx])
        assert len(new_input_ids) == len(new_attention_mask) == new_metadata[-1][-1]
        if not self.is_training:
            assert len(new_input_ids) == len(new_attention_mask) == len(new_metadata) == len(new_decoder_input_ids) == len(new_decoder_attention_mask)
        self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
        self.logger.info("%.2f%% questions have at least one answer mentioned in passages" % (100*np.mean(has_valid)))


    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
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
                assert len(fg_passages) == len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata, = self.tokenized_data
            assert len(dpr_passages) == len(input_ids) == len(attention_mask) == len(metadata)
            assert len(decoder_input_ids) == len(decoder_attention_mask) == metadata[-1][-1]

            def _included(tokens, curr_input_ids, end_of_answer_prompt):
                is_answer_exist = []
                for _curr_input_ids in curr_input_ids:
                    is_exist = False
                    for jdx in range(end_of_answer_prompt, len(_curr_input_ids)-len(tokens)+1):
                        if _curr_input_ids[jdx:jdx+len(tokens)]==tokens:
                            is_exist = True
                    is_answer_exist.append(is_exist)
                return is_answer_exist

            new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_is_valid_list, = [], [], [], [], []
            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids, ) \
                    in enumerate(zip(tqdm(input_ids), attention_mask, metadata, dpr_passages,)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                # create multiple inputs
                for answer_idx in range(*curr_metadata):
                    curr_answer_ids = decoder_input_ids[answer_idx]
                    end_of_answer = curr_answer_ids.index(eos_token_id) if eos_token_id in curr_answer_ids else len(curr_answer_ids)
                    answer_input_ids = curr_answer_ids[:end_of_answer] + [eos_token_id]
                    answer_attention_mask = [1] * len(answer_input_ids)
                    end_of_answer_prompt = len(answer_input_ids)
                    ap_input_ids, ap_attention_mask = [], []
                    for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                        end_of_dpr_input = _dpr_attention_mask.index(0) if 0 in _dpr_attention_mask else len(_dpr_attention_mask)
                        _dpr_input_ids = _dpr_input_ids[:end_of_dpr_input]
                        _dpr_attention_mask = _dpr_attention_mask[:end_of_dpr_input]
                        assert _dpr_input_ids[0] == bos_token_id
                        answer_input_ids_jdx = answer_input_ids + _dpr_input_ids[1:]
                        answer_attention_mask_jdx = answer_attention_mask + _dpr_attention_mask[1:]
                        assert len(answer_input_ids_jdx) == len(answer_attention_mask_jdx)
                        if len(answer_input_ids_jdx) > 160:
                            answer_input_ids_jdx = answer_input_ids_jdx[:159] + [eos_token_id]
                            answer_attention_mask_jdx = answer_attention_mask_jdx[:160]
                        else:
                            answer_input_ids_jdx += [pad_token_id for _ in range(32 + 128 - len(answer_input_ids_jdx))]
                            answer_attention_mask_jdx += [0 for _ in range(32 + 128 - len(answer_attention_mask_jdx))]
                        ap_input_ids.append(answer_input_ids_jdx)
                        ap_attention_mask.append(answer_attention_mask_jdx)
                        assert len(ap_input_ids[jdx]) == len(ap_attention_mask[jdx]) == 160  # here we use 32+128
                    assert len(ap_input_ids) == len(ap_attention_mask) == 100
                    curr_is_valid_list = _included(curr_answer_ids[1:end_of_answer], ap_input_ids, end_of_answer_prompt)
                    new_input_ids.append(ap_input_ids)
                    new_attention_mask.append(ap_attention_mask)
                    new_is_valid_list.append(curr_is_valid_list)

                new_decoder_input_ids.append(curr_input_ids)
                new_decoder_attention_mask.append(curr_attention_mask)

            assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(metadata)
            assert metadata[-1][-1] == len(new_input_ids) == len(new_attention_mask) == len(new_is_valid_list)

            self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, metadata, new_is_valid_list]
            with open(dpr_tokenized_path, "w") as f:
                json.dump(self.tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))
            # if self.is_training:
            #     exit()

        aq_psgs_input_ids, aq_psgs_attention_mask, _, _, _, aq_psgs_discard = self.tokenized_data
        self.tokenized_data[0] = [_aq_psgs_input_ids[:self.args.top_k_passages] for _aq_psgs_input_ids in aq_psgs_input_ids]
        self.tokenized_data[1] = [_aq_psgs_attention_mask[:self.args.top_k_passages] for _aq_psgs_attention_mask in aq_psgs_attention_mask]
        self.tokenized_data[5] = [_aq_psgs_discard[:self.args.top_k_passages] for _aq_psgs_discard in aq_psgs_discard]

        if self.is_training and self.args.discard_not_found_answers:
            raise NotImplementedError

        self.tokenized_data = self.tokenized_data[:5]

    def load_dpr_data_bart_filter_nfa_psgs(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            # if "train_for_inference" not in dpr_retrieval_path:
            #     dpr_retrieval_path = dpr_retrieval_path.replace("train", "train_for_inference")
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
                assert len(dpr_passages)==len(self)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages) == len(input_ids) == len(attention_mask) == len(metadata)
            bos_token_id = self.tokenizer.bos_token_id

            def _included(tokens, curr_input_ids, end_of_answer):
                is_answer_exist = []
                for _curr_input_ids in curr_input_ids:
                    is_exist = False
                    for jdx in range(end_of_answer, len(_curr_input_ids)-len(tokens)+1):
                        if _curr_input_ids[jdx:jdx+len(tokens)]==tokens:
                            is_exist = True
                    is_answer_exist.append(is_exist)
                return is_answer_exist

            new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_is_valid_list = [], [], [], [], []
            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                # create multiple inputs
                answer_input_ids_list, answer_attention_mask_list, is_valid_list = [], [], []
                for answer_idx in range(*curr_metadata):
                    end_of_answer = decoder_input_ids[answer_idx].index(self.tokenizer.eos_token_id) + 1
                    answer_input_ids = decoder_input_ids[answer_idx][:end_of_answer]
                    answer_attention_mask = decoder_attention_mask[answer_idx][:end_of_answer]
                    ap_input_ids, ap_attention_mask = [], []
                    for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                        assert _dpr_input_ids[0] == bos_token_id
                        answer_input_ids_jdx = answer_input_ids + _dpr_input_ids[1:]
                        answer_attention_mask_jdx = answer_attention_mask + _dpr_attention_mask[1:]
                        assert len(answer_input_ids_jdx) == len(answer_attention_mask_jdx)
                        answer_input_ids_jdx += [self.tokenizer.pad_token_id for _ in range(32 + 128 - len(answer_input_ids_jdx))]
                        answer_attention_mask_jdx += [0 for _ in range(32 + 128 - len(answer_attention_mask_jdx))]
                        ap_input_ids.append(answer_input_ids_jdx)
                        ap_attention_mask.append(answer_attention_mask_jdx)
                        assert len(ap_input_ids[jdx]) == len(ap_attention_mask[jdx]) == 160  # here we use 32+128
                    assert len(ap_input_ids) == len(ap_attention_mask) == 100
                    answer_input_ids_list.append(ap_input_ids)
                    answer_attention_mask_list.append(ap_attention_mask)
                    is_valid_list.append(_included(decoder_input_ids[answer_idx][1:end_of_answer - 1], ap_input_ids, end_of_answer))

                assert len(answer_input_ids_list) == len(answer_attention_mask_list) == len(is_valid_list) == curr_metadata[1] - curr_metadata[0]

                new_decoder_input_ids.append(curr_input_ids)
                new_decoder_attention_mask.append(curr_attention_mask)
                new_input_ids.append(answer_input_ids_list)
                new_attention_mask.append(answer_attention_mask_list)
                new_is_valid_list.append(is_valid_list)
            assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(new_input_ids) == len(new_attention_mask) == len(new_is_valid_list) == len(self)

            self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_is_valid_list]
            with open(dpr_tokenized_path, "w") as f:
                json.dump(self.tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))

        raw_input_ids, raw_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, raw_is_valid_list = self.tokenized_data
        if self.args.use_reranker:
            assert self.args.psg_sel_dir is not None
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference"),
                                          "_20200201" if self.args.wiki_2020 else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
            assert len(fg_passages) == len(raw_input_ids)

            ranked_raw_input_ids, ranked_raw_attention_mask, ranked_raw_is_valid_list = [], [], []
            for idx, fg_psgs in enumerate(fg_passages):
                # re-order fg_psgs according to containing answers or not (raw_is_valid)
                ranked_raw_input_ids_i, ranked_raw_attention_mask_i, ranked_raw_is_valid_list_i = [], [], []
                for rid, ratt, rvalid in zip(raw_input_ids[idx], raw_attention_mask[idx], raw_is_valid_list[idx]):
                    new_fg_psgs_ext, new_fg_psgs_abs = [], []
                    for i in fg_psgs:
                        if rvalid[i]:
                            new_fg_psgs_ext.append(i)
                        else:
                            new_fg_psgs_abs.append(i)
                    new_fg_psgs = new_fg_psgs_ext + new_fg_psgs_abs
                    ranked_raw_input_ids_i.append([rid[i] for i in new_fg_psgs][:self.args.top_k_passages])
                    ranked_raw_attention_mask_i.append([ratt[i] for i in new_fg_psgs][:self.args.top_k_passages])
                    ranked_raw_is_valid_list_i.append([rvalid[i] for i in new_fg_psgs][:self.args.top_k_passages])
                ranked_raw_input_ids.append(ranked_raw_input_ids_i)
                ranked_raw_attention_mask.append(ranked_raw_attention_mask_i)
                ranked_raw_is_valid_list.append(ranked_raw_is_valid_list_i)
        else:
            ranked_raw_input_ids, ranked_raw_attention_mask, ranked_raw_is_valid_list = [], [], []
            for idx in range(len(raw_input_ids)):
                ranked_raw_input_ids_i, ranked_raw_attention_mask_i, ranked_raw_is_valid_list_i = [], [], []
                fg_psgs = list(range(len(raw_input_ids[0][0])))
                for rid, ratt, rvalid in zip(raw_input_ids[idx], raw_attention_mask[idx], raw_is_valid_list[idx]):
                    new_fg_psgs_ext, new_fg_psgs_abs = [], []
                    for i in fg_psgs:
                        if rvalid[i]:
                            new_fg_psgs_ext.append(i)
                        else:
                            new_fg_psgs_abs.append(i)
                    new_fg_psgs = new_fg_psgs_ext + new_fg_psgs_abs
                    ranked_raw_input_ids_i.append([rid[i] for i in new_fg_psgs][:self.args.top_k_passages])
                    ranked_raw_attention_mask_i.append([ratt[i] for i in new_fg_psgs][:self.args.top_k_passages])
                    ranked_raw_is_valid_list_i.append([rvalid[i] for i in new_fg_psgs][:self.args.top_k_passages])
                ranked_raw_input_ids.append(ranked_raw_input_ids_i)
                ranked_raw_attention_mask.append(ranked_raw_attention_mask_i)
                ranked_raw_is_valid_list.append(ranked_raw_is_valid_list_i)

        # prepare for customized training and inference dataset
        new_input_ids, new_attention_mask, new_metadata = [], [], []
        has_valid = []
        if not self.is_training:
            old_decoder_input_ids, old_decoder_attention_mask, = new_decoder_input_ids, new_decoder_attention_mask,
            new_decoder_input_ids, new_decoder_attention_mask, = [], []
        for idx, (rr_input_ids, rr_attention_mask, rr_is_valid_list) in enumerate(zip(ranked_raw_input_ids, ranked_raw_attention_mask, ranked_raw_is_valid_list)):
            topk_is_valid_list = [any(rr_is_valid) for rr_is_valid in rr_is_valid_list]
            has_valid.append(any(topk_is_valid_list))
            if self.is_training:
                if self.args.discard_not_found_answers:
                    if not any(topk_is_valid_list):
                        topk_is_valid_list = [True for _ in topk_is_valid_list]
                    new_metadata.append((len(new_input_ids), len(new_input_ids) + sum(topk_is_valid_list)))
                    new_input_ids += [answer_input_ids for answer_input_ids, is_valid in zip(rr_input_ids, topk_is_valid_list) if is_valid]
                    new_attention_mask += [answer_attention_mask for answer_attention_mask, is_valid in zip(rr_attention_mask, topk_is_valid_list) if is_valid]
                else:
                    offset = len(new_input_ids)
                    new_input_ids.extend(rr_input_ids)
                    new_attention_mask.extend(rr_attention_mask)
                    new_metadata.append((offset, len(new_input_ids)))
            else:
                # Yifan: original code only evaluates when gold sample is found, but here we evaluate all qa pairs
                # we generate all questions conditioned on all answers, and get the best if the question have multiple answer candidates
                offset = len(new_input_ids)
                new_input_ids.extend(rr_input_ids)
                new_attention_mask.extend(rr_attention_mask)
                for i in range(offset, len(new_input_ids)):
                    new_metadata.append((i,i+1))
                    new_decoder_input_ids.append(old_decoder_input_ids[idx])
                    new_decoder_attention_mask.append(old_decoder_attention_mask[idx])
        assert len(new_input_ids) == len(new_attention_mask) == new_metadata[-1][-1]
        if not self.is_training:
            assert len(new_input_ids) == len(new_attention_mask) == len(new_metadata) == len(new_decoder_input_ids) == len(new_decoder_attention_mask)
        self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
        self.logger.info("%.2f%% questions have at least one answer mentioned in passages" % (100*np.mean(has_valid)))

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        self.dataset = MySimpleQADataset(input_ids,
                                            attention_mask,
                                            decoder_input_ids if self.is_training else None,
                                            decoder_attention_mask if self.is_training else None,
                                            in_metadata=metadata if self.is_training else None,
                                            out_metadata=None,
                                            is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False, **kwargs):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training, **kwargs)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, n_paragraphs=None):
        # create a reference set
        references = []
        for i in range(len(self.data)):
            ans = self.data[i]["answer"]
            ques = self.data[i]["question"]
            for _ in ans:
                references.append(ques)
        assert len(predictions) == len(references), (len(predictions), len(references))

        # first, tokenize
        data_to_tokenize = {}
        for i, (ref, pred, ) in enumerate(zip(references, predictions,)):
            data_to_tokenize["ref.{}".format(i)] = [{"caption": ref}]
            data_to_tokenize["gen.{}".format(i)] = [{"caption": pred if type(pred) == str else pred[0]}]
        if self.args.do_train or self.args.task == 'qg_mask':
            all_tokens = {}
            for k, v in data_to_tokenize.items():
                doc = self.qg_tokenizer(v[0]['caption'])
                tkied = [tk.text for tk in doc if tk.text not in PUNCTUATIONS]
                all_tokens[k] = [' '.join(tkied)]
        else:
            all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu_flatten = []
        for i in range(len(references)):
            reference = {"sent": [normalize_answer(text) for text in all_tokens["ref.{}".format(i)]]}
            generated = {"sent": [normalize_answer(text) for text in all_tokens["gen.{}".format(i)]]}
            bleu_flatten.append(Bleu(4).compute_score(reference, generated)[0][-1])

        bleu, = [],
        metadata = self.tokenized_data[-1]
        assert metadata[-1][-1] == len(bleu_flatten)
        for idx, m in enumerate(metadata):
            start, end = m
            bleu.append(np.mean(bleu_flatten[start:end]))
        assert len(bleu) == len(self.data)
        self.logger.info("BLEU=%.2f" % (100 * np.mean(bleu)))

        results = {
            'BLEU': np.mean(bleu) * 100,
        }
        return results['BLEU'], results

    def save_predictions(self, predictions, mode=''):
        # assert len(predictions)==len(self), (len(predictions), len(self))
        save_path = os.path.join(self.args.output_dir, "{}{}{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 else "",
            "_aq" if self.args.ambigqa else "",
            mode,
        ))
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))


class QGMaskedData(QGData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(QGMaskedData, self).__init__(logger, args, data_path, is_training, passages)
        self.SEP = "<SEP>"
        self.metric = "EDIT-F1"
        self.masked_span_length = [1,2,3,4,5]

        self.masked_data_path = data_path.replace('.json', '_mask.json')
        if os.path.exists(self.masked_data_path):
            with open(self.masked_data_path, "r") as f:
                self.data = json.load(f)
        else:
            for d in self.data:
                question = d["question"] if d["question"].endswith("?") else d["question"] + "?"
                # generate masked questions
                span_length = np.random.choice(self.masked_span_length)
                question_spacy = self.qg_tokenizer(question)
                question_t = [tk.text_with_ws for tk in question_spacy]
                valid_token_idx = [1 if tk.pos_ in VALID_POS else 0 for tk in question_spacy]
                valid_start_positions = []
                for idx in range(len(valid_token_idx)):
                    if idx <= len(valid_token_idx) - span_length:
                        is_valid = False
                        for jdx in range(idx, idx + span_length):
                            if valid_token_idx[jdx] == 1:
                                is_valid = True
                                break
                        if is_valid:
                            valid_start_positions.append(idx)
                if len(valid_start_positions) == 0:
                    span_start = np.random.choice(len(question_t) - span_length - 1)  # -1 for not masking the last question mark ?
                else:
                    span_start = np.random.choice(valid_start_positions)  # -1 for not masking the last question mark ?
                question_t_masked = question_t[:span_start] + question_t[span_start + span_length:]
                assert len(question_t_masked) + span_length == len(question_t)
                d["question"] = "".join(question_t)
                d["prompt"] = "".join(question_t_masked)
            with open(self.masked_data_path, "w") as f:
                json.dump(self.data, f)

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix or 'T5' in postfix
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}{}{}-masked-{}.json".format(
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
            questions = [d["question"] for d in self.data]
            prompts = [d["prompt"] for d in self.data]
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
                prompts = [prompt.lower() for prompt in prompts]
                answers = [answer.lower() for answer in answers]

            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            question_masked_input = tokenizer.batch_encode_plus(prompts,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix or "T5" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            input_masked_ids, attention_masked_mask = question_masked_input["input_ids"], question_masked_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask, metadata,
                              input_masked_ids, attention_masked_mask]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f)
        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}_predictions.json".format(
            self.data_type + "_20200201" if self.args.wiki_2020 else self.data_type)).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "", ))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_{}_qg_masked.json".format(postfix))
        if "Bart" in postfix:
            if not self.args.filter_not_found_answer_passages:
                return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
            else:
                return self.load_dpr_data_bart_filter_nfa_psgs(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
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
                assert len(fg_passages) == len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata, input_masked_ids, attention_masked_mask = self.tokenized_data
            assert len(dpr_passages) == len(input_ids) == len(attention_mask) == len(metadata) == len(input_masked_ids) == len(attention_masked_mask)
            assert len(decoder_input_ids) == len(decoder_attention_mask) == metadata[-1][-1]

            def _included(tokens, curr_input_ids, end_of_answer_prompt):
                is_answer_exist = []
                for _curr_input_ids in curr_input_ids:
                    is_exist = False
                    for jdx in range(end_of_answer_prompt, len(_curr_input_ids)-len(tokens)+1):
                        if _curr_input_ids[jdx:jdx+len(tokens)]==tokens:
                            is_exist = True
                    is_answer_exist.append(is_exist)
                return is_answer_exist

            new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_is_valid_list, = [], [], [], [], []
            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids, curr_input_masked_ids, curr_attention_masked_mask) \
                    in enumerate(zip(tqdm(input_ids), attention_mask, metadata, dpr_passages, input_masked_ids, attention_masked_mask)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                end_of_masked_question = curr_input_masked_ids.index(eos_token_id) + 1 if eos_token_id in curr_input_masked_ids else len(curr_input_masked_ids)
                curr_input_masked_ids = curr_input_masked_ids[:end_of_masked_question]

                # create multiple inputs
                for answer_idx in range(*curr_metadata):
                    curr_answer_ids = decoder_input_ids[answer_idx]
                    end_of_answer = curr_answer_ids.index(eos_token_id) if eos_token_id in curr_answer_ids else len(curr_answer_ids)
                    answer_input_ids = curr_answer_ids[:end_of_answer] + [sep_token_id] + curr_input_masked_ids[1:]
                    answer_attention_mask = [1] * len(answer_input_ids)
                    end_of_answer_prompt = len(answer_input_ids)
                    ap_input_ids, ap_attention_mask = [], []
                    for jdx, (_dpr_input_ids, _dpr_attention_mask) in enumerate(zip(dpr_input_ids, dpr_attention_mask)):
                        end_of_dpr_input = _dpr_attention_mask.index(0) if 0 in _dpr_attention_mask else len(_dpr_attention_mask)
                        _dpr_input_ids = _dpr_input_ids[:end_of_dpr_input]
                        _dpr_attention_mask = _dpr_attention_mask[:end_of_dpr_input]
                        assert _dpr_input_ids[0] == bos_token_id
                        answer_input_ids_jdx = answer_input_ids + _dpr_input_ids[1:]
                        answer_attention_mask_jdx = answer_attention_mask + _dpr_attention_mask[1:]
                        assert len(answer_input_ids_jdx) == len(answer_attention_mask_jdx)
                        if len(answer_input_ids_jdx) > 160:
                            answer_input_ids_jdx = answer_input_ids_jdx[:159] + [eos_token_id]
                            answer_attention_mask_jdx = answer_attention_mask_jdx[:160]
                        else:
                            answer_input_ids_jdx += [pad_token_id for _ in range(32 + 128 - len(answer_input_ids_jdx))]
                            answer_attention_mask_jdx += [0 for _ in range(32 + 128 - len(answer_attention_mask_jdx))]
                        ap_input_ids.append(answer_input_ids_jdx)
                        ap_attention_mask.append(answer_attention_mask_jdx)
                        assert len(ap_input_ids[jdx]) == len(ap_attention_mask[jdx]) == 160  # here we use 32+128
                    assert len(ap_input_ids) == len(ap_attention_mask) == 100
                    curr_is_valid_list = _included(curr_answer_ids[1:end_of_answer], ap_input_ids, end_of_answer_prompt)
                    new_input_ids.append(ap_input_ids)
                    new_attention_mask.append(ap_attention_mask)
                    new_is_valid_list.append(curr_is_valid_list)

                new_decoder_input_ids.append(curr_input_ids)
                new_decoder_attention_mask.append(curr_attention_mask)

            assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(metadata)
            assert metadata[-1][-1] == len(new_input_ids) == len(new_attention_mask) == len(new_is_valid_list)

            self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, metadata, new_is_valid_list]
            with open(dpr_tokenized_path, "w") as f:
                json.dump(self.tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))
            # if self.is_training:
            #     exit()

        aq_psgs_input_ids, aq_psgs_attention_mask, _, _, _, aq_psgs_discard = self.tokenized_data
        self.tokenized_data[0] = [_aq_psgs_input_ids[:self.args.top_k_passages] for _aq_psgs_input_ids in aq_psgs_input_ids]
        self.tokenized_data[1] = [_aq_psgs_attention_mask[:self.args.top_k_passages] for _aq_psgs_attention_mask in aq_psgs_attention_mask]
        self.tokenized_data[5] = [_aq_psgs_discard[:self.args.top_k_passages] for _aq_psgs_discard in aq_psgs_discard]

        if self.is_training and self.args.discard_not_found_answers:
            raise NotImplementedError

        self.tokenized_data = self.tokenized_data[:5]

    def evaluate(self, predictions, n_paragraphs=None):
        # create a reference set
        references = []
        prompts = []
        for i in range(len(self.data)):
            ans = self.data[i]["answer"]
            ques = self.data[i]["question"]
            pmt = self.data[i]["prompt"]
            for _ in ans:
                references.append(ques)
                prompts.append(pmt)
        assert len(predictions) == len(references) == len(prompts), (len(predictions), len(references))

        # first, tokenize
        data_to_tokenize = {}
        for i, (ref, pred, pmt) in enumerate(zip(references, predictions, prompts)):
            data_to_tokenize["ref.{}".format(i)] = [{"caption": ref}]
            data_to_tokenize["pmt.{}".format(i)] = [{"caption": pmt}]
            data_to_tokenize["gen.{}".format(i)] = [{"caption": pred if type(pred)==str else pred[0]}]
        if self.args.do_train or self.args.task == 'qg_mask':
            all_tokens = {}
            for k, v in data_to_tokenize.items():
                doc = self.qg_tokenizer(v[0]['caption'])
                tkied = [tk.text for tk in doc if tk.text not in PUNCTUATIONS]
                all_tokens[k] = [' '.join(tkied)]
        else:
            all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu_flatten, f1s_flatten = [], []
        for i in range(len(references)):
            e = get_qg_metrics(_get("gen.{}".format(i)),
                               _get("ref.{}".format(i)),
                               _get("pmt.{}".format(i)),
                               metrics=["bleu4", "edit-f1"])
            bleu_flatten.append(e["bleu4"])
            f1s_flatten.append(e["edit-f1"])

        bleu, f1s = [], []
        metadata = self.tokenized_data[-1]
        assert metadata[-1][-1] == len(bleu_flatten)
        for idx, m in enumerate(metadata):
            start, end = m
            bleu.append(np.mean(bleu_flatten[start:end]))
            f1s.append(np.mean(f1s_flatten[start:end]))
        assert len(bleu) == len(self.data) == len(f1s)
        self.logger.info("BLEU=%.2f, EDIT-F1=%.2f" % (100 * np.mean(bleu), 100 * np.mean(f1s)))

        results = {
            'BLEU': np.mean(bleu) * 100,
            'EDIT-F1': np.mean(f1s) * 100,
        }
        return results['EDIT-F1'], results


class AmbigQGData(AmbigQAData, QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGData, self).__init__(logger, args, data_path, is_training, passages)

        self.ref_questions = []
        self.ref_answers = []

        if args.ambigqa_editqg:
            import spacy
            nlp = spacy.load("en_core_web_sm")

        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                curr_questions = []
                for pair in annotation["qaPairs"]:
                    curr_q = [q.strip() for q in pair["question"].split("|")]
                    if is_training:
                        curr_questions.append([curr_q[0]])
                    else:
                        curr_questions.append(curr_q)
                if args.ambigqa_editqg:
                    curr_questions_inserted = []
                    prompt = nlp(d['question'].lower().strip())
                    prompt_tkd = [token.text for token in prompt]
                    for curr_q in curr_questions:
                        curr_q_inserted = []
                        for curr_q_i in curr_q:
                            noamb_q = nlp(curr_q_i.lower().strip())
                            noamb_q_tkd = [token.text for token in noamb_q]
                            deletion, insertion = self._get_edits(prompt_tkd, noamb_q_tkd)
                            if len(insertion) == 0:
                                if len(deletion) != 0:
                                    curr_q_inserted.append(" ".join(deletion))
                                else:
                                    curr_q_inserted.append(curr_q_i.lower().strip())
                                logger.info("DQ == Prompt!")
                            else:
                                curr_q_inserted.append(" ".join(insertion))
                        assert len(curr_q_inserted) == len(curr_q)
                        curr_questions_inserted.append(curr_q_inserted)
                    assert len(curr_questions) == len(curr_questions_inserted)
                    questions.append(curr_questions_inserted)
                else:
                    questions.append(curr_questions)
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])
            self.ref_questions.append(questions)
            self.ref_answers.append(answers)

        self.SEP = "<SEP>"
        self.metric = "EDIT-F1"
        if args.do_train or args.is_sagemaker:
            import spacy
            self.qg_tokenizer = spacy.load("en_core_web_sm")
        else:
            self.qg_tokenizer = PTBTokenizer()

    def _get_edits(self, tokens1, tokens2):
        allCommon = []
        while True:
            commons = list(set(tokens1) & set(tokens2))
            if len(commons) == 0:
                break
            allCommon += commons
            for c in commons:
                ind1, ind2 = tokens1.index(c), tokens2.index(c)
                tokens1 = tokens1[:ind1] + tokens1[ind1 + 1:]
                tokens2 = tokens2[:ind2] + tokens2[ind2 + 1:]
        original_tokens2 = tokens2
        while len(tokens2) > 0 and (tokens2[0] in PUNCT_WORDS or tokens2[0] in IGNORE_WORDS):
            tokens2 = tokens2[1:]
        while len(tokens2) > 0 and (tokens2[-1] in PUNCT_WORDS or tokens2[-1] in IGNORE_WORDS):
            tokens2 = tokens2[:-1]
        if len(tokens2) > 0:
            return tokens1, tokens2
        else:
            return tokens1, original_tokens2

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type + "_20200201" if self.args.wiki_2020 else self.data_type,
            "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "", ))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}_qg{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix, "_edit" if self.args.ambigqa_editqg else ""))
        if "Bart" in postfix:
            self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    # v0: answer [SEP] promptQ </s> passage
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(self.data_type.replace("train", "train_for_inference"),
                                                                       "_20200201" if self.args.wiki_2020 else "",
                                                                       "_aq" if self.args.ambigqa else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                assert len(fg_passages)==len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            assert len(dpr_passages)==len(self)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            def _get_tokenized_answer(idx, append_another_bos=False):
                tokens = decoder_input_ids[idx]
                # remove padded token
                if 0 in decoder_attention_mask[idx]:
                    tokens = tokens[:decoder_attention_mask[idx].index(0)]
                if append_another_bos:
                    assert tokens[0] == tokens[1] == bos_token_id and tokens[-1] == self.tokenizer.eos_token_id
                    return tokens[2:-1]
                else:
                    assert tokens[0] == bos_token_id and tokens[-1] == eos_token_id
                    return tokens[1:-1]

            def _included(tokens, psg_input_ids):
                is_token_included = []
                for _psg_input_ids in psg_input_ids:
                    is_token_icl = False
                    for jdx in range(len(_psg_input_ids) - len(tokens) + 1):
                        if _psg_input_ids[jdx:jdx + len(tokens)] == tokens:
                            is_token_icl = True
                            break
                    is_token_included.append(is_token_icl)
                return is_token_included

            new_input_ids, new_attention_mask, new_output, new_metadata = [], [], [], []

            if self.is_training:
                discard_not_found_answers = []
                for idx, (curr_input_ids, curr_attention_mask, curr_ref_questions, curr_ref_answers, curr_metadata, dpr_ids) \
                        in enumerate(zip(tqdm(input_ids), attention_mask, self.ref_questions, self.ref_answers, metadata, dpr_passages)):
                    if curr_ref_questions is None:
                        continue
                    end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    q_input_ids = curr_input_ids[:end_of_question]
                    dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                    dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                    for annotation_idx in range(len(curr_metadata)):
                        curr_ann_ref_questions = curr_ref_questions[annotation_idx]
                        curr_ann_ref_answers = curr_ref_answers[annotation_idx]
                        curr_ann_metadata = curr_metadata[annotation_idx]
                        assert type(curr_ann_metadata[0][0]) == int
                        assert [len(ast_ref_answer) == ast_end - ast_start for ast_ref_answer, (ast_start, ast_end) in zip(curr_ann_ref_answers, curr_ann_metadata)]
                        curr_ann_ref_answers_tokenized = [[_get_tokenized_answer(i, self.args.append_another_bos) for i in range(*m)] for m in curr_ann_metadata]
                        curr_ann_ref_answers_is_appear = [[_included(tokens, dpr_input_ids) for tokens in _tokens] for _tokens in curr_ann_ref_answers_tokenized]
                        assert len(curr_ann_ref_questions) == len(curr_ann_ref_answers) == len(curr_ann_ref_answers_tokenized) == len(curr_ann_ref_answers_is_appear)

                        for jdx, (curr_ann_ref_questions_i, curr_ann_ref_answers_i, curr_ann_metadata_i, curr_ann_ref_answers_tokenized_i, curr_ann_ref_answers_is_appear_i) in \
                                enumerate(zip(curr_ann_ref_questions, curr_ann_ref_answers, curr_ann_metadata, curr_ann_ref_answers_tokenized, curr_ann_ref_answers_is_appear)):
                            # enumerate multiple answers for the disambiguated question
                            new_input_ids_offset = len(new_input_ids)
                            for (curr_ann_ref_answers_tokenized_i_j, curr_ann_ref_answers_is_appear_i_j) in zip(curr_ann_ref_answers_tokenized_i, curr_ann_ref_answers_is_appear_i):
                                aq_input_ids = [bos_token_id] + curr_ann_ref_answers_tokenized_i_j + [sep_token_id] + q_input_ids[1:]
                                aq_attention_mask = [1 for _ in aq_input_ids]
                                aq_psgs_input_ids, aq_psgs_attention_mask = [], []
                                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                                    assert psg_attention_mask[-1] == 1
                                    aq_psgs_input_ids_i = aq_input_ids + psg_input_id[1:]
                                    aq_psgs_attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                                    assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    max_input_len = 32 + 128
                                    if len(aq_psgs_input_ids_i) > max_input_len:
                                        aq_psgs_input_ids_i = aq_psgs_input_ids_i[:max_input_len]
                                        aq_psgs_attention_mask_i = aq_psgs_attention_mask_i[:max_input_len]
                                        assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    else:
                                        aq_psgs_input_ids_i += [pad_token_id for _ in range(max_input_len - len(aq_psgs_input_ids_i))]
                                        aq_psgs_attention_mask_i += [0 for _ in range(max_input_len - len(aq_psgs_attention_mask_i))]
                                    aq_psgs_input_ids.append(aq_psgs_input_ids_i)
                                    aq_psgs_attention_mask.append(aq_psgs_attention_mask_i)
                                assert len(aq_psgs_input_ids) == len(aq_psgs_attention_mask)
                                new_input_ids.append(aq_psgs_input_ids)
                                new_attention_mask.append(aq_psgs_attention_mask)
                                discard_not_found_answers.append(curr_ann_ref_answers_is_appear_i_j)
                            assert len(curr_ann_ref_questions_i) == 1
                            new_output.append(curr_ann_ref_questions_i[0])
                            new_metadata.append((new_input_ids_offset, len(new_input_ids)))

                new_output = self.tokenizer.batch_encode_plus(new_output, max_length=32, pad_to_max_length=True)
                new_decoder_input_ids, new_decoder_attention_mask = new_output["input_ids"], new_output["attention_mask"]
                assert len(new_input_ids) == len(new_attention_mask) == len(discard_not_found_answers) == new_metadata[-1][-1]
                assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(new_metadata)
                tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata, discard_not_found_answers]
            else:
                metadata_perann_perqapair = []
                # record valid question answer pairs (some datapoint prompt question is not ambiguous,
                # so we need to skip them)
                metadata_perann_perqapair_offset = 0
                discard_not_found_answers = []
                for idx, (curr_input_ids, curr_attention_mask, curr_ref_questions, curr_ref_answers, curr_metadata, dpr_ids) \
                        in enumerate(zip(tqdm(input_ids), attention_mask, self.ref_questions, self.ref_answers, metadata, dpr_passages)):
                    if curr_ref_questions is None:
                        continue
                    end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    q_input_ids = curr_input_ids[:end_of_question]
                    dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                    dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                    # here we consider all annotations because we dont want this dependent on the retrieval results
                    metadata_perann_perqapair.append([])  # per data point
                    for annotation_idx in range(len(curr_metadata)):
                        metadata_perann_perqapair[metadata_perann_perqapair_offset].append([])  # per annotator
                        ref_questions = curr_ref_questions[annotation_idx]
                        ref_answers = curr_ref_answers[annotation_idx]
                        ref_metadata = curr_metadata[annotation_idx]
                        tokenized_ref_answers = [[_get_tokenized_answer(i, self.args.append_another_bos) for i in range(*m)] for m in ref_metadata]
                        ref_answers_is_appear = [[_included(tokens, dpr_input_ids) for tokens in _tokens] for _tokens in tokenized_ref_answers]
                        assert len(ref_questions) == len(ref_answers) == len(tokenized_ref_answers)

                        # per answer cluster
                        for qapair_idx, (_tkd_ref_answers, ref_question, _ref_answers_is_appear_per_qapair) in enumerate(zip(tokenized_ref_answers, ref_questions, ref_answers_is_appear)):
                            predictions_offset = len(new_output)
                            # per answer in each cluster
                            for tkd_ref_answer, _ref_answers_is_appear_per_qapair_per_ans in zip(_tkd_ref_answers, _ref_answers_is_appear_per_qapair):
                                aq_input_ids = [bos_token_id] + tkd_ref_answer + [sep_token_id] + q_input_ids[1:]
                                aq_attention_mask = [1 for _ in aq_input_ids]
                                aq_psgs_input_ids, aq_psgs_attention_mask = [], []
                                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                                    assert psg_attention_mask[-1] == 1
                                    aq_psgs_input_ids_i = aq_input_ids + psg_input_id[1:]
                                    aq_psgs_attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                                    assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    max_input_len = 32 + 128
                                    if len(aq_psgs_input_ids_i) > max_input_len:
                                        aq_psgs_input_ids_i = aq_psgs_input_ids_i[:max_input_len]
                                        aq_psgs_attention_mask_i = aq_psgs_attention_mask_i[:max_input_len]
                                        assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    else:
                                        aq_psgs_input_ids_i += [pad_token_id for _ in range(max_input_len - len(aq_psgs_input_ids_i))]
                                        aq_psgs_attention_mask_i += [0 for _ in range(max_input_len - len(aq_psgs_attention_mask_i))]
                                    aq_psgs_input_ids.append(aq_psgs_input_ids_i)
                                    aq_psgs_attention_mask.append(aq_psgs_attention_mask_i)
                                discard_not_found_answers.append(_ref_answers_is_appear_per_qapair_per_ans)
                                new_input_ids.append(aq_psgs_input_ids)
                                new_attention_mask.append(aq_psgs_attention_mask)
                                new_output.append(ref_question)
                                new_metadata.append((len(new_output)-1, len(new_output)))
                            # start and end for per answer cluster
                            metadata_perann_perqapair[metadata_perann_perqapair_offset][annotation_idx].append((predictions_offset, len(new_output)))
                    metadata_perann_perqapair_offset += 1

                new_decoder_input_ids, new_decoder_attention_mask = None, None
                assert metadata_perann_perqapair[-1][-1][-1][-1] == len(new_input_ids)

                tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask,
                                  new_metadata, discard_not_found_answers, metadata_perann_perqapair,]

            with open(dpr_tokenized_path, "w") as f:
                json.dump(tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))
            self.tokenized_data = tokenized_data

        if self.args.filter_not_found_answer_passages:
            old_input_ids, old_attention_mask, _, _, _, old_discard_not_found_answers = self.tokenized_data[:6]
            filtered_input_ids, filtered_attention_mask, filtered_discard_not_found_answers = [], [], [],
            truly_filtered_discard_not_found_answers = []
            assert len(old_input_ids) == len(old_attention_mask) == len(old_discard_not_found_answers)
            for old_input_ids_i, old_attention_mask_i, old_discard_not_found_answers_i in zip(old_input_ids, old_attention_mask, old_discard_not_found_answers):
                assert len(old_input_ids_i) == len(old_attention_mask_i) == len(old_discard_not_found_answers_i) == 100
                filtered_input_ids_i, filtered_attention_mask_i, filtered_discard_not_found_answers_i = [], [], [],
                truly_filtered_discard_not_found_answers_i = []
                for old_input_ids_i_j, old_attention_mask_i_j, old_discard_not_found_answers_i_j in zip(old_input_ids_i, old_attention_mask_i, old_discard_not_found_answers_i):
                    if old_discard_not_found_answers_i_j:
                        filtered_input_ids_i.append(old_input_ids_i_j)
                        filtered_attention_mask_i.append(old_attention_mask_i_j)
                        filtered_discard_not_found_answers_i.append(old_discard_not_found_answers_i_j)
                        truly_filtered_discard_not_found_answers_i.append(old_discard_not_found_answers_i_j)
                if len(filtered_input_ids_i) == 0:
                    filtered_input_ids_i, filtered_attention_mask_i = old_input_ids_i, old_attention_mask_i
                    filtered_discard_not_found_answers_i = [True] * len(filtered_input_ids_i)
                    truly_filtered_discard_not_found_answers_i = [False] * len(filtered_input_ids_i)
                else:
                    # pad some tokens
                    while len(filtered_input_ids_i) < 100:
                        # hello -> 20760
                        filtered_input_ids_i.append([20760] * 160)
                        filtered_attention_mask_i.append([1] * 160)
                        filtered_discard_not_found_answers_i.append(False)
                        truly_filtered_discard_not_found_answers_i.append(False)
                filtered_input_ids.append(filtered_input_ids_i)
                filtered_attention_mask.append(filtered_attention_mask_i)
                filtered_discard_not_found_answers.append(filtered_discard_not_found_answers_i)
                truly_filtered_discard_not_found_answers.append(truly_filtered_discard_not_found_answers_i)
            self.tokenized_data[0] = filtered_input_ids
            self.tokenized_data[1] = filtered_attention_mask
            self.tokenized_data[5] = filtered_discard_not_found_answers
            truly_filtered_discard_not_found_answers = [_aq_psgs_truly_discard[:self.args.top_k_passages] for _aq_psgs_truly_discard in truly_filtered_discard_not_found_answers]

        aq_psgs_input_ids, aq_psgs_attention_mask, _, _, _, aq_psgs_discard = self.tokenized_data[:6]
        self.tokenized_data[0] = [_aq_psgs_input_ids[:self.args.top_k_passages] for _aq_psgs_input_ids in aq_psgs_input_ids]
        self.tokenized_data[1] = [_aq_psgs_attention_mask[:self.args.top_k_passages] for _aq_psgs_attention_mask in aq_psgs_attention_mask]
        self.tokenized_data[5] = [_aq_psgs_discard[:self.args.top_k_passages] for _aq_psgs_discard in aq_psgs_discard]

        if self.is_training and self.args.discard_not_found_answers:
            old_input_ids, old_attention_mask, old_decoder_input_ids, old_decoder_attention_mask, old_metadata, discard_not_found_answers = self.tokenized_data
            filtered_input_ids, filtered_attention_mask, filtered_decoder_input_ids, filtered_decoder_attention_mask, filtered_metadata, filtered_discard_not_found_answers = [], [], [], [], [], []
            for old_curr_m, old_curr_decoder_input_ids, old_curr_decoder_attention_mask in zip(old_metadata, old_decoder_input_ids, old_decoder_attention_mask):
                new_start = len(filtered_input_ids)
                is_keep_sample = False
                for old_idx in range(*old_curr_m):
                    if not self.args.filter_not_found_answer_passages:
                        curr_discard_not_found_answers = discard_not_found_answers[old_idx][:self.args.top_k_passages]
                    else:
                        curr_discard_not_found_answers = truly_filtered_discard_not_found_answers[old_idx][:self.args.top_k_passages]
                    if any(curr_discard_not_found_answers):
                        is_keep_sample = True
                        filtered_input_ids.append(old_input_ids[old_idx])
                        filtered_attention_mask.append(old_attention_mask[old_idx])
                        filtered_discard_not_found_answers.append(discard_not_found_answers[old_idx])
                if is_keep_sample:
                    filtered_decoder_input_ids.append(old_curr_decoder_input_ids)
                    filtered_decoder_attention_mask.append(old_curr_decoder_attention_mask)
                    filtered_metadata.append((new_start, len(filtered_input_ids)))
            self.tokenized_data = [filtered_input_ids, filtered_attention_mask, filtered_decoder_input_ids, filtered_decoder_attention_mask, filtered_metadata, filtered_discard_not_found_answers]

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if not self.args.filter_not_found_answer_passages:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, in_metadata = self.tokenized_data[:5]
            self.dataset = MySimpleQGDataset(input_ids,
                                                    attention_mask,
                                                    decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                                    decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                                    in_metadata=in_metadata,
                                                    is_training=self.is_training)
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, in_metadata, discard_not_found_answers = self.tokenized_data[:6]
            self.dataset = MySimpleQGDynamicDataset(input_ids,
                                             attention_mask,
                                             decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                             decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                             in_metadata=in_metadata,
                                             is_training=self.is_training,
                                             discard_not_found_answers=discard_not_found_answers)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def evaluate(self, predictions, n_paragraphs=None):
        metadata_perann_perqapair = self.tokenized_data[-1]
        assert metadata_perann_perqapair[-1][-1][-1][-1] == len(predictions)
        data_to_tokenize = {}
        offset = 0
        for i, (d, ref_questions, ref_answers) in enumerate(zip(self.data, self.ref_questions, self.ref_answers)):
            if ref_questions is None: continue
            data_to_tokenize["prompt.{}".format(offset)] = [{"caption": d["question"]}]
            for ann_idx in range(len(d['annotations'])):
                assert len(ref_questions[ann_idx]) == len(ref_answers[ann_idx]) == len(metadata_perann_perqapair[offset][ann_idx])
                for qapair_idx in range(len(d['annotations'][ann_idx]['qaPairs'])):
                    start, end = metadata_perann_perqapair[offset][ann_idx][qapair_idx]
                    for answer_idx in range(start, end):
                        if not self.args.ambigqa_editqg:
                            pred = predictions[answer_idx]
                            # ref_question = ref_questions[ann_idx][qapair_idx]
                            ref_question = [q.strip() for q in d['annotations'][ann_idx]['qaPairs'][qapair_idx]['question'].split("|")]
                            # per sample - per annotator - per qapair - per answer
                            data_to_tokenize["gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": pred if type(pred) == str else pred[0]}]
                            data_to_tokenize["ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": ref} for ref in ref_question]
                        else:
                            pred = predictions[answer_idx] if type(predictions[answer_idx]) == str else predictions[answer_idx][0]
                            ref_question = [q.strip() for q in d['annotations'][ann_idx]['qaPairs'][qapair_idx]['question'].split("|")]
                            # per sample - per annotator - per qapair - per answer
                            data_to_tokenize["gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": "{} {}?".format(d["question"][:-1], pred)}]
                            data_to_tokenize["ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": ref} for ref in ref_question]
            offset += 1
        assert offset == len(metadata_perann_perqapair)

        if self.args.do_train or self.args.is_sagemaker:
            all_tokens = {}
            for k, v in data_to_tokenize.items():
                doc = self.qg_tokenizer(v[0]['caption'])
                tkied = [tk.text for tk in doc if tk.text not in PUNCTUATIONS]
                all_tokens[k] = [' '.join(tkied)]
        else:
            all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu, f1s = [], []
        # per sample
        for offset in range(len(metadata_perann_perqapair)):
            ann_all = []
            for ann_idx in range(len(metadata_perann_perqapair[offset])):
                qapair_bf = []
                for qapair_idx in range(len(metadata_perann_perqapair[offset][ann_idx])):
                    ans_all = []
                    for answer_idx in range(*metadata_perann_perqapair[offset][ann_idx][qapair_idx]):
                        e = get_qg_metrics(_get("gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)),
                                           _get("ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)),
                                           _get("prompt.{}".format(offset)),
                                           metrics=["bleu4", "edit-f1"])
                        ans_all.append((e["bleu4"], e["edit-f1"]))
                    ans_avg = (np.mean([x[0] for x in ans_all]), np.mean([x[1] for x in ans_all]))
                    qapair_bf.append(ans_avg)
                # get average result on qapair_bf
                qapair_avg_bleu = np.mean([b[0] for b in qapair_bf])
                qapair_avg_editf1 = np.mean([b[1] for b in qapair_bf])
                ann_all.append((qapair_avg_bleu, qapair_avg_editf1))
            ann_avg = (np.mean([x[0] for x in ann_all]), np.mean([x[1] for x in ann_all]))
            bleu.append(ann_avg[0])
            f1s.append(ann_avg[1])
        self.logger.info("BLEU=%.2f; EDIT-F1=%.2f" % (100 * np.mean(bleu), 100 * np.mean(f1s)))
        results = {
            'BLEU': np.mean(bleu) * 100,
            'EDIT-F1': np.mean(f1s) * 100,
        }
        return results['EDIT-F1'], results


class AmbigQGWeightedData(AmbigQAData, QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGWeightedData, self).__init__(logger, args, data_path, is_training, passages)

        self.ref_questions = []
        self.ref_answers = []

        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                curr_questions = []
                for pair in annotation["qaPairs"]:
                    curr_q = [q.strip() for q in pair["question"].split("|")]
                    if is_training:
                        curr_questions.append([curr_q[0]])
                    else:
                        curr_questions.append(curr_q)
                questions.append(curr_questions)
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])
            self.ref_questions.append(questions)
            self.ref_answers.append(answers)

        self.SEP = "<SEP>"
        self.metric = "EDIT-F1"
        if args.do_train:
            import spacy
            self.qg_tokenizer = spacy.load("en_core_web_sm")
        else:
            self.qg_tokenizer = PTBTokenizer()

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type + "_20200201" if self.args.wiki_2020 else self.data_type,
            "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "", ))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}_qg_weighted_loss.json".format("_20200201" if self.args.wiki_2020 else "", postfix))
        if "Bart" in postfix:
            self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    # v0: answer [SEP] promptQ </s> passage
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(self.data_type.replace("train", "train_for_inference"),
                                                                       "_20200201" if self.args.wiki_2020 else "",
                                                                       "_aq" if self.args.ambigqa else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                assert len(fg_passages)==len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            assert len(dpr_passages)==len(self)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            def _get_tokenized_answer(idx, append_another_bos=False):
                tokens = decoder_input_ids[idx]
                # remove padded token
                if 0 in decoder_attention_mask[idx]:
                    tokens = tokens[:decoder_attention_mask[idx].index(0)]
                if append_another_bos:
                    assert tokens[0] == tokens[1] == bos_token_id and tokens[-1] == self.tokenizer.eos_token_id
                    return tokens[2:-1]
                else:
                    assert tokens[0] == bos_token_id and tokens[-1] == eos_token_id
                    return tokens[1:-1]

            def _included(tokens, psg_input_ids):
                is_token_included = []
                for _psg_input_ids in psg_input_ids:
                    is_token_icl = False
                    for jdx in range(len(_psg_input_ids) - len(tokens) + 1):
                        if _psg_input_ids[jdx:jdx + len(tokens)] == tokens:
                            is_token_icl = True
                            break
                    is_token_included.append(is_token_icl)
                return is_token_included

            new_input_ids, new_attention_mask, new_output, new_metadata, prompt_question_ids = [], [], [], [], []

            if self.is_training:
                discard_not_found_answers = []
                for idx, (curr_input_ids, curr_attention_mask, curr_ref_questions, curr_ref_answers, curr_metadata, dpr_ids) \
                        in enumerate(zip(tqdm(input_ids), attention_mask, self.ref_questions, self.ref_answers, metadata, dpr_passages)):
                    if curr_ref_questions is None:
                        continue
                    end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    q_input_ids = curr_input_ids[:end_of_question]
                    dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                    dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                    for annotation_idx in range(len(curr_metadata)):
                        curr_ann_ref_questions = curr_ref_questions[annotation_idx]
                        curr_ann_ref_answers = curr_ref_answers[annotation_idx]
                        curr_ann_metadata = curr_metadata[annotation_idx]
                        assert type(curr_ann_metadata[0][0]) == int
                        assert [len(ast_ref_answer) == ast_end - ast_start for ast_ref_answer, (ast_start, ast_end) in zip(curr_ann_ref_answers, curr_ann_metadata)]
                        curr_ann_ref_answers_tokenized = [[_get_tokenized_answer(i, self.args.append_another_bos) for i in range(*m)] for m in curr_ann_metadata]
                        curr_ann_ref_answers_is_appear = [[_included(tokens, dpr_input_ids) for tokens in _tokens] for _tokens in curr_ann_ref_answers_tokenized]
                        assert len(curr_ann_ref_questions) == len(curr_ann_ref_answers) == len(curr_ann_ref_answers_tokenized) == len(curr_ann_ref_answers_is_appear)

                        for jdx, (curr_ann_ref_questions_i, curr_ann_ref_answers_i, curr_ann_metadata_i, curr_ann_ref_answers_tokenized_i, curr_ann_ref_answers_is_appear_i) in \
                                enumerate(zip(curr_ann_ref_questions, curr_ann_ref_answers, curr_ann_metadata, curr_ann_ref_answers_tokenized, curr_ann_ref_answers_is_appear)):
                            # enumerate multiple answers for the disambiguated question
                            new_input_ids_offset = len(new_input_ids)
                            for (curr_ann_ref_answers_tokenized_i_j, curr_ann_ref_answers_is_appear_i_j) in zip(curr_ann_ref_answers_tokenized_i, curr_ann_ref_answers_is_appear_i):
                                aq_input_ids = [bos_token_id] + curr_ann_ref_answers_tokenized_i_j + [sep_token_id] + q_input_ids[1:]
                                aq_attention_mask = [1 for _ in aq_input_ids]
                                aq_psgs_input_ids, aq_psgs_attention_mask = [], []
                                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                                    assert psg_attention_mask[-1] == 1
                                    aq_psgs_input_ids_i = aq_input_ids + psg_input_id[1:]
                                    aq_psgs_attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                                    assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    max_input_len = 32 + 128
                                    if len(aq_psgs_input_ids_i) > max_input_len:
                                        aq_psgs_input_ids_i = aq_psgs_input_ids_i[:max_input_len]
                                        aq_psgs_attention_mask_i = aq_psgs_attention_mask_i[:max_input_len]
                                        assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    else:
                                        aq_psgs_input_ids_i += [pad_token_id for _ in range(max_input_len - len(aq_psgs_input_ids_i))]
                                        aq_psgs_attention_mask_i += [0 for _ in range(max_input_len - len(aq_psgs_attention_mask_i))]
                                    aq_psgs_input_ids.append(aq_psgs_input_ids_i)
                                    aq_psgs_attention_mask.append(aq_psgs_attention_mask_i)
                                assert len(aq_psgs_input_ids) == len(aq_psgs_attention_mask)
                                new_input_ids.append(aq_psgs_input_ids)
                                new_attention_mask.append(aq_psgs_attention_mask)
                                discard_not_found_answers.append(curr_ann_ref_answers_is_appear_i_j)
                            assert len(curr_ann_ref_questions_i) == 1
                            new_output.append(curr_ann_ref_questions_i[0])
                            new_metadata.append((new_input_ids_offset, len(new_input_ids)))
                            prompt_question_ids.append(q_input_ids)

                new_output = self.tokenizer.batch_encode_plus(new_output, max_length=32, pad_to_max_length=True)
                new_decoder_input_ids, new_decoder_attention_mask = new_output["input_ids"], new_output["attention_mask"]
                assert len(new_input_ids) == len(new_attention_mask) == len(discard_not_found_answers) == new_metadata[-1][-1]
                assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(new_metadata) == len(prompt_question_ids)

                # process weighted loss data
                weighted_positions = []
                for curr_prompt_question_ids, curr_noamb_question_ids, curr_noamb_question_mask in zip(prompt_question_ids, new_decoder_input_ids, new_decoder_attention_mask):
                    curr_weighted_positions = [int(tk_mask == 1 and tk_id not in curr_prompt_question_ids and tk_id not in [bos_token_id, eos_token_id, pad_token_id, sep_token_id]) for tk_id, tk_mask in zip(curr_noamb_question_ids, curr_noamb_question_mask)]
                    weighted_positions.append(curr_weighted_positions)
                tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata, discard_not_found_answers, weighted_positions]
            else:
                metadata_perann_perqapair = []
                # record valid question answer pairs (some datapoint prompt question is not ambiguous,
                # so we need to skip them)
                metadata_perann_perqapair_offset = 0
                discard_not_found_answers = []
                for idx, (curr_input_ids, curr_attention_mask, curr_ref_questions, curr_ref_answers, curr_metadata, dpr_ids) \
                        in enumerate(zip(tqdm(input_ids), attention_mask, self.ref_questions, self.ref_answers, metadata, dpr_passages)):
                    if curr_ref_questions is None:
                        continue
                    end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    q_input_ids = curr_input_ids[:end_of_question]
                    dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                    dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                    # here we consider all annotations because we dont want this dependent on the retrieval results
                    metadata_perann_perqapair.append([])  # per data point
                    for annotation_idx in range(len(curr_metadata)):
                        metadata_perann_perqapair[metadata_perann_perqapair_offset].append([])  # per annotator
                        ref_questions = curr_ref_questions[annotation_idx]
                        ref_answers = curr_ref_answers[annotation_idx]
                        ref_metadata = curr_metadata[annotation_idx]
                        tokenized_ref_answers = [[_get_tokenized_answer(i, self.args.append_another_bos) for i in range(*m)] for m in ref_metadata]
                        ref_answers_is_appear = [[_included(tokens, dpr_input_ids) for tokens in _tokens] for _tokens in tokenized_ref_answers]
                        assert len(ref_questions) == len(ref_answers) == len(tokenized_ref_answers)

                        # per answer cluster
                        for qapair_idx, (_tkd_ref_answers, ref_question, _ref_answers_is_appear_per_qapair) in enumerate(zip(tokenized_ref_answers, ref_questions, ref_answers_is_appear)):
                            predictions_offset = len(new_output)
                            # per answer in each cluster
                            for tkd_ref_answer, _ref_answers_is_appear_per_qapair_per_ans in zip(_tkd_ref_answers, _ref_answers_is_appear_per_qapair):
                                aq_input_ids = [bos_token_id] + tkd_ref_answer + [sep_token_id] + q_input_ids[1:]
                                aq_attention_mask = [1 for _ in aq_input_ids]
                                aq_psgs_input_ids, aq_psgs_attention_mask = [], []
                                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                                    assert psg_attention_mask[-1] == 1
                                    aq_psgs_input_ids_i = aq_input_ids + psg_input_id[1:]
                                    aq_psgs_attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                                    assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    max_input_len = 32 + 128
                                    if len(aq_psgs_input_ids_i) > max_input_len:
                                        aq_psgs_input_ids_i = aq_psgs_input_ids_i[:max_input_len]
                                        aq_psgs_attention_mask_i = aq_psgs_attention_mask_i[:max_input_len]
                                        assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    else:
                                        aq_psgs_input_ids_i += [pad_token_id for _ in range(max_input_len - len(aq_psgs_input_ids_i))]
                                        aq_psgs_attention_mask_i += [0 for _ in range(max_input_len - len(aq_psgs_attention_mask_i))]
                                    aq_psgs_input_ids.append(aq_psgs_input_ids_i)
                                    aq_psgs_attention_mask.append(aq_psgs_attention_mask_i)
                                discard_not_found_answers.append(_ref_answers_is_appear_per_qapair_per_ans)
                                new_input_ids.append(aq_psgs_input_ids)
                                new_attention_mask.append(aq_psgs_attention_mask)
                                new_output.append(ref_question)
                                new_metadata.append((len(new_output)-1, len(new_output)))
                            # start and end for per answer cluster
                            metadata_perann_perqapair[metadata_perann_perqapair_offset][annotation_idx].append((predictions_offset, len(new_output)))
                    metadata_perann_perqapair_offset += 1

                new_decoder_input_ids, new_decoder_attention_mask = None, None
                assert metadata_perann_perqapair[-1][-1][-1][-1] == len(new_input_ids)

                tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask,
                                  new_metadata, discard_not_found_answers, metadata_perann_perqapair,]

            with open(dpr_tokenized_path, "w") as f:
                json.dump(tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))
            self.tokenized_data = tokenized_data

        if self.args.filter_not_found_answer_passages:
            old_input_ids, old_attention_mask, _, _, _, old_discard_not_found_answers = self.tokenized_data[:6]
            filtered_input_ids, filtered_attention_mask, filtered_discard_not_found_answers = [], [], [],
            truly_filtered_discard_not_found_answers = []
            assert len(old_input_ids) == len(old_attention_mask) == len(old_discard_not_found_answers)
            for old_input_ids_i, old_attention_mask_i, old_discard_not_found_answers_i in zip(old_input_ids, old_attention_mask, old_discard_not_found_answers):
                assert len(old_input_ids_i) == len(old_attention_mask_i) == len(old_discard_not_found_answers_i) == 100
                filtered_input_ids_i, filtered_attention_mask_i, filtered_discard_not_found_answers_i = [], [], [],
                truly_filtered_discard_not_found_answers_i = []
                for old_input_ids_i_j, old_attention_mask_i_j, old_discard_not_found_answers_i_j in zip(old_input_ids_i, old_attention_mask_i, old_discard_not_found_answers_i):
                    if old_discard_not_found_answers_i_j:
                        filtered_input_ids_i.append(old_input_ids_i_j)
                        filtered_attention_mask_i.append(old_attention_mask_i_j)
                        filtered_discard_not_found_answers_i.append(old_discard_not_found_answers_i_j)
                        truly_filtered_discard_not_found_answers_i.append(old_discard_not_found_answers_i_j)
                if len(filtered_input_ids_i) == 0:
                    filtered_input_ids_i, filtered_attention_mask_i = old_input_ids_i, old_attention_mask_i
                    filtered_discard_not_found_answers_i = [True] * len(filtered_input_ids_i)
                    truly_filtered_discard_not_found_answers_i = [False] * len(filtered_input_ids_i)
                else:
                    # pad some tokens
                    while len(filtered_input_ids_i) < 100:
                        # hello -> 20760
                        filtered_input_ids_i.append([20760] * 160)
                        filtered_attention_mask_i.append([1] * 160)
                        filtered_discard_not_found_answers_i.append(False)
                        truly_filtered_discard_not_found_answers_i.append(False)
                filtered_input_ids.append(filtered_input_ids_i)
                filtered_attention_mask.append(filtered_attention_mask_i)
                filtered_discard_not_found_answers.append(filtered_discard_not_found_answers_i)
                truly_filtered_discard_not_found_answers.append(truly_filtered_discard_not_found_answers_i)
            self.tokenized_data[0] = filtered_input_ids
            self.tokenized_data[1] = filtered_attention_mask
            self.tokenized_data[5] = filtered_discard_not_found_answers
            truly_filtered_discard_not_found_answers = [_aq_psgs_truly_discard[:self.args.top_k_passages] for _aq_psgs_truly_discard in truly_filtered_discard_not_found_answers]

        aq_psgs_input_ids, aq_psgs_attention_mask, _, _, _, aq_psgs_discard = self.tokenized_data[:6]
        self.tokenized_data[0] = [_aq_psgs_input_ids[:self.args.top_k_passages] for _aq_psgs_input_ids in aq_psgs_input_ids]
        self.tokenized_data[1] = [_aq_psgs_attention_mask[:self.args.top_k_passages] for _aq_psgs_attention_mask in aq_psgs_attention_mask]
        self.tokenized_data[5] = [_aq_psgs_discard[:self.args.top_k_passages] for _aq_psgs_discard in aq_psgs_discard]

        if self.is_training and self.args.discard_not_found_answers:
            old_input_ids, old_attention_mask, old_decoder_input_ids, old_decoder_attention_mask, old_metadata, discard_not_found_answers, old_weighted_positions = self.tokenized_data
            filtered_input_ids, filtered_attention_mask, filtered_decoder_input_ids, filtered_decoder_attention_mask, filtered_metadata, filtered_discard_not_found_answers, filtered_weighted_positions = [], [], [], [], [], [], []
            for old_curr_m, old_curr_decoder_input_ids, old_curr_decoder_attention_mask, old_curr_weighted_positions in zip(old_metadata, old_decoder_input_ids, old_decoder_attention_mask, old_weighted_positions):
                new_start = len(filtered_input_ids)
                is_keep_sample = False
                for old_idx in range(*old_curr_m):
                    if not self.args.filter_not_found_answer_passages:
                        curr_discard_not_found_answers = discard_not_found_answers[old_idx][:self.args.top_k_passages]
                    else:
                        curr_discard_not_found_answers = truly_filtered_discard_not_found_answers[old_idx][:self.args.top_k_passages]
                    if any(curr_discard_not_found_answers):
                        is_keep_sample = True
                        filtered_input_ids.append(old_input_ids[old_idx])
                        filtered_attention_mask.append(old_attention_mask[old_idx])
                        filtered_discard_not_found_answers.append(discard_not_found_answers[old_idx])
                if is_keep_sample:
                    filtered_decoder_input_ids.append(old_curr_decoder_input_ids)
                    filtered_decoder_attention_mask.append(old_curr_decoder_attention_mask)
                    filtered_metadata.append((new_start, len(filtered_input_ids)))
                    filtered_weighted_positions.append(old_curr_weighted_positions)
            self.tokenized_data = [filtered_input_ids, filtered_attention_mask, filtered_decoder_input_ids, filtered_decoder_attention_mask, filtered_metadata, filtered_discard_not_found_answers, filtered_weighted_positions]

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if not self.args.filter_not_found_answer_passages:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, in_metadata = self.tokenized_data[:5]
            self.dataset = MySimpleQGWeightedLossDataset(input_ids,
                                                         attention_mask,
                                                         decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                                         decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                                         in_metadata=in_metadata,
                                                         is_training=self.is_training,
                                                         weighted_position=None if not self.is_training else self.tokenized_data[-1])
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, in_metadata, discard_not_found_answers = self.tokenized_data[:6]
            self.dataset = MySimpleQGDynamicWeightedLossDataset(input_ids,
                                                                attention_mask,
                                                                decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                                                decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                                                in_metadata=in_metadata,
                                                                is_training=self.is_training,
                                                                discard_not_found_answers=discard_not_found_answers,
                                                                weighted_position=None if not self.is_training else self.tokenized_data[6])
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def evaluate(self, predictions, n_paragraphs=None):
        metadata_perann_perqapair = self.tokenized_data[-1]
        assert metadata_perann_perqapair[-1][-1][-1][-1] == len(predictions)
        data_to_tokenize = {}
        offset = 0
        for i, (d, ref_questions, ref_answers) in enumerate(zip(self.data, self.ref_questions, self.ref_answers)):
            if ref_questions is None: continue
            data_to_tokenize["prompt.{}".format(offset)] = [{"caption": d["question"]}]
            for ann_idx in range(len(d['annotations'])):
                assert len(ref_questions[ann_idx]) == len(ref_answers[ann_idx]) == len(metadata_perann_perqapair[offset][ann_idx])
                for qapair_idx in range(len(d['annotations'][ann_idx]['qaPairs'])):
                    start, end = metadata_perann_perqapair[offset][ann_idx][qapair_idx]
                    for answer_idx in range(start, end):
                        if not self.args.ambigqa_editqg:
                            pred = predictions[answer_idx]
                            # ref_question = ref_questions[ann_idx][qapair_idx]
                            ref_question = [q.strip() for q in d['annotations'][ann_idx]['qaPairs'][qapair_idx]['question'].split("|")]
                            # per sample - per annotator - per qapair - per answer
                            data_to_tokenize["gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": pred if type(pred) == str else pred[0]}]
                            data_to_tokenize["ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": ref} for ref in ref_question]
                        else:
                            pred = predictions[answer_idx] if type(predictions[answer_idx]) == str else predictions[answer_idx][0]
                            ref_question = [q.strip() for q in d['annotations'][ann_idx]['qaPairs'][qapair_idx]['question'].split("|")]
                            # per sample - per annotator - per qapair - per answer
                            data_to_tokenize["gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": "{} {}?".format(d["question"][:-1], pred)}]
                            data_to_tokenize["ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": ref} for ref in ref_question]
            offset += 1
        assert offset == len(metadata_perann_perqapair)

        if self.args.do_train:
            all_tokens = {}
            for k, v in data_to_tokenize.items():
                doc = self.qg_tokenizer(v[0]['caption'])
                tkied = [tk.text for tk in doc if tk.text not in PUNCTUATIONS]
                all_tokens[k] = [' '.join(tkied)]
        else:
            all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu, f1s = [], []
        # per sample
        for offset in range(len(metadata_perann_perqapair)):
            ann_all = []
            for ann_idx in range(len(metadata_perann_perqapair[offset])):
                qapair_bf = []
                for qapair_idx in range(len(metadata_perann_perqapair[offset][ann_idx])):
                    ans_all = []
                    for answer_idx in range(*metadata_perann_perqapair[offset][ann_idx][qapair_idx]):
                        e = get_qg_metrics(_get("gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)),
                                           _get("ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)),
                                           _get("prompt.{}".format(offset)),
                                           metrics=["bleu4", "edit-f1"])
                        ans_all.append((e["bleu4"], e["edit-f1"]))
                    ans_avg = (np.mean([x[0] for x in ans_all]), np.mean([x[1] for x in ans_all]))
                    qapair_bf.append(ans_avg)
                # get average result on qapair_bf
                qapair_avg_bleu = np.mean([b[0] for b in qapair_bf])
                qapair_avg_editf1 = np.mean([b[1] for b in qapair_bf])
                ann_all.append((qapair_avg_bleu, qapair_avg_editf1))
            ann_avg = (np.mean([x[0] for x in ann_all]), np.mean([x[1] for x in ann_all]))
            bleu.append(ann_avg[0])
            f1s.append(ann_avg[1])
        self.logger.info("BLEU=%.2f; EDIT-F1=%.2f" % (100 * np.mean(bleu), 100 * np.mean(f1s)))
        results = {
            'BLEU': np.mean(bleu) * 100,
            'EDIT-F1': np.mean(f1s) * 100,
        }
        return results['EDIT-F1'], results


class AmbigQGRewriteData(AmbigQAData, QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGRewriteData, self).__init__(logger, args, data_path, is_training, passages)

        import spacy
        nlp = spacy.load("en_core_web_sm")
        self.questions = []
        self.keyphrases = []
        self.prompts = []

        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                continue
            prompt = nlp(d['question'].lower().strip())
            prompt_tkd = [token.text for token in prompt]
            for annotation in d["annotations"]:
                for pair in annotation["qaPairs"]:
                    curr_q = [q.strip() for q in pair["question"].split("|")]
                    for curr_q_i in curr_q:
                        noamb_q = nlp(curr_q_i.lower().strip())
                        noamb_q_tkd = [token.text for token in noamb_q]
                        deletion, insertion = self._get_edits(prompt_tkd, noamb_q_tkd)
                        if len(insertion) == 0:
                            continue
                        else:
                            self.keyphrases.append(" ".join(insertion))
                            self.questions.append(curr_q_i)
                            self.prompts.append(d['question'])
        assert len(self.questions) == len(self.keyphrases) == len(self.prompts)
        self.SEP = "<SEP>"
        self.metric = "BLEU"
        if args.do_train:
            import spacy
            self.qg_tokenizer = spacy.load("en_core_web_sm")
        else:
            self.qg_tokenizer = PTBTokenizer()

    def _get_edits(self, tokens1, tokens2):
        allCommon = []
        while True:
            commons = list(set(tokens1) & set(tokens2))
            if len(commons) == 0:
                break
            allCommon += commons
            for c in commons:
                ind1, ind2 = tokens1.index(c), tokens2.index(c)
                tokens1 = tokens1[:ind1] + tokens1[ind1 + 1:]
                tokens2 = tokens2[:ind2] + tokens2[ind2 + 1:]
        original_tokens2 = tokens2
        while len(tokens2) > 0 and (tokens2[0] in PUNCT_WORDS or tokens2[0] in IGNORE_WORDS):
            tokens2 = tokens2[1:]
        while len(tokens2) > 0 and (tokens2[-1] in PUNCT_WORDS or tokens2[-1] in IGNORE_WORDS):
            tokens2 = tokens2[:-1]
        if len(tokens2) > 0:
            return tokens1, tokens2
        else:
            return tokens1, original_tokens2

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self.tokenized_data
        self.dataset = MySimpleQADataset(input_ids,
                                         attention_mask,
                                         decoder_input_ids if self.is_training else None,
                                         decoder_attention_mask if self.is_training else None,
                                         is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

    def load_dataloader(self, do_return=False, **kwargs):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training, **kwargs)
        if do_return:
            return self.dataloader

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix or 'T5' in postfix
        print ("Start tokenizing...")
        questions, prompts, keyphrases = self.questions, self.prompts, self.keyphrases
        if self.args.do_lowercase:
            questions = [question.lower() for question in questions]
            prompts = [prompt.lower() for prompt in prompts]
            keyphrases = [keyphrase.lower() for keyphrase in keyphrases]
        question_input = tokenizer.batch_encode_plus(questions, pad_to_max_length=True, max_length=32)
        prompt_input = tokenizer.batch_encode_plus(prompts, pad_to_max_length=False, max_length=32)
        keyphrase_input = tokenizer.batch_encode_plus(keyphrases, pad_to_max_length=False, max_length=32)

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

        input_ids, attention_mask = [], []
        for _prompt, _keyphrase in zip(prompt_input["input_ids"], keyphrase_input["input_ids"]):
            assert _prompt[0] == _keyphrase[0] == bos_token_id
            assert _prompt[-1] == _keyphrase[-1] == eos_token_id
            # <bos> prompt [SEP] keyphrase <eos>
            input_ids_i = _prompt[:-1] + [sep_token_id] + _keyphrase[1:]
            attention_mask_i = [1] * len(input_ids_i)
            max_input_len = 32
            if len(input_ids_i) > max_input_len:
                input_ids_i = input_ids_i[:max_input_len]
                attention_mask_i = attention_mask_i[:max_input_len]
            else:
                input_ids_i += [pad_token_id for _ in range(max_input_len - len(input_ids_i))]
                attention_mask_i += [0 for _ in range(max_input_len - len(attention_mask_i))]
            input_ids.append(input_ids_i)
            attention_mask.append(attention_mask_i)
        decoder_input_ids, decoder_attention_mask = question_input["input_ids"], question_input["attention_mask"]
        tokenized_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask]

        self.tokenized_data = tokenized_data

    def evaluate(self, predictions):
        # create a reference set
        references = self.questions
        assert len(predictions) == len(references), (len(predictions), len(references))

        bleu_flatten = []
        data_to_tokenize = {}
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            data_to_tokenize["ref.{}".format(i)] = [{"caption": ref}]
            data_to_tokenize["gen.{}".format(i)] = [{"caption": pred if type(pred)==str else pred[0]}]
        if self.args.do_train:
            all_tokens = {}
            for k, v in data_to_tokenize.items():
                doc = self.qg_tokenizer(v[0]['caption'])
                tkied = [tk.text for tk in doc if tk.text not in PUNCTUATIONS]
                all_tokens[k] = [' '.join(tkied)]
        else:
            all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        for i in range(len(references)):
            reference = {"sent": [normalize_answer(text) for text in all_tokens["ref.{}".format(i)]]}
            generated = {"sent": [normalize_answer(text) for text in all_tokens["gen.{}".format(i)]]}
            bleu_flatten.append(Bleu(4).compute_score(reference, generated)[0][-1])

        results = {
            'BLEU': np.mean(bleu_flatten) * 100,
        }

        return results['BLEU'], results


class AmbigQGNoPromptData(AmbigQAData, QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGNoPromptData, self).__init__(logger, args, data_path, is_training, passages)

        self.ref_questions = []
        self.ref_answers = []

        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                curr_questions = []
                for pair in annotation["qaPairs"]:
                    curr_q = [q.strip() for q in pair["question"].split("|")]
                    if is_training:
                        curr_questions.append([curr_q[0]])
                    else:
                        curr_questions.append(curr_q)
                questions.append(curr_questions)
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])
            self.ref_questions.append(questions)
            self.ref_answers.append(answers)

        self.SEP = "<SEP>"
        self.metric = "EDIT-F1"
        if args.do_train or args.is_sagemaker:
            import spacy
            self.qg_tokenizer = spacy.load("en_core_web_sm")
        else:
            self.qg_tokenizer = PTBTokenizer()

    # override
    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_predictions.json".format(
            self.data_type + "_20200201" if self.args.wiki_2020 else self.data_type,
            "_aq" if self.args.ambigqa else "")).replace('train_for_inference', 'train')
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}{}_predictions.json".format(self.data_type, "-reos" if self.args.t5_no_intermediate_eos else "", ))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "{}_{}_qg_noprompt{}.json".format("_20200201" if self.args.wiki_2020 else "", postfix, "_edit" if self.args.ambigqa_editqg else ""))
        if "Bart" in postfix:
            self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError

    # v0: answer [SEP] promptQ </s> passage
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            if self.args.use_reranker:
                assert self.args.psg_sel_dir is not None
                psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                          "{}{}{}_psg_sel.json".format(self.data_type.replace("train", "train_for_inference"),
                                                                       "_20200201" if self.args.wiki_2020 else "",
                                                                       "_aq" if self.args.ambigqa else ""))
                self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
                with open(psg_sel_fn, "r") as f:
                    fg_passages = json.load(f)
                assert len(fg_passages)==len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs][:100] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            else:
                raise NotImplementedError

            assert len(dpr_passages)==len(self)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
            assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

            def _get_tokenized_answer(idx, append_another_bos=False):
                tokens = decoder_input_ids[idx]
                # remove padded token
                if 0 in decoder_attention_mask[idx]:
                    tokens = tokens[:decoder_attention_mask[idx].index(0)]
                if append_another_bos:
                    assert tokens[0] == tokens[1] == bos_token_id and tokens[-1] == self.tokenizer.eos_token_id
                    return tokens[2:-1]
                else:
                    assert tokens[0] == bos_token_id and tokens[-1] == eos_token_id
                    return tokens[1:-1]

            def _included(tokens, psg_input_ids):
                is_token_included = []
                for _psg_input_ids in psg_input_ids:
                    is_token_icl = False
                    for jdx in range(len(_psg_input_ids) - len(tokens) + 1):
                        if _psg_input_ids[jdx:jdx + len(tokens)] == tokens:
                            is_token_icl = True
                            break
                    is_token_included.append(is_token_icl)
                return is_token_included

            new_input_ids, new_attention_mask, new_output, new_metadata = [], [], [], []

            if self.is_training:
                discard_not_found_answers = []
                for idx, (curr_input_ids, curr_attention_mask, curr_ref_questions, curr_ref_answers, curr_metadata, dpr_ids) \
                        in enumerate(zip(tqdm(input_ids), attention_mask, self.ref_questions, self.ref_answers, metadata, dpr_passages)):
                    if curr_ref_questions is None:
                        continue
                    # end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    # q_input_ids = curr_input_ids[:end_of_question]
                    dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                    dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                    for annotation_idx in range(len(curr_metadata)):
                        curr_ann_ref_questions = curr_ref_questions[annotation_idx]
                        curr_ann_ref_answers = curr_ref_answers[annotation_idx]
                        curr_ann_metadata = curr_metadata[annotation_idx]
                        assert type(curr_ann_metadata[0][0]) == int
                        assert [len(ast_ref_answer) == ast_end - ast_start for ast_ref_answer, (ast_start, ast_end) in zip(curr_ann_ref_answers, curr_ann_metadata)]
                        curr_ann_ref_answers_tokenized = [[_get_tokenized_answer(i, self.args.append_another_bos) for i in range(*m)] for m in curr_ann_metadata]
                        curr_ann_ref_answers_is_appear = [[_included(tokens, dpr_input_ids) for tokens in _tokens] for _tokens in curr_ann_ref_answers_tokenized]
                        assert len(curr_ann_ref_questions) == len(curr_ann_ref_answers) == len(curr_ann_ref_answers_tokenized) == len(curr_ann_ref_answers_is_appear)

                        for jdx, (curr_ann_ref_questions_i, curr_ann_ref_answers_i, curr_ann_metadata_i, curr_ann_ref_answers_tokenized_i, curr_ann_ref_answers_is_appear_i) in \
                                enumerate(zip(curr_ann_ref_questions, curr_ann_ref_answers, curr_ann_metadata, curr_ann_ref_answers_tokenized, curr_ann_ref_answers_is_appear)):
                            # enumerate multiple answers for the disambiguated question
                            new_input_ids_offset = len(new_input_ids)
                            for (curr_ann_ref_answers_tokenized_i_j, curr_ann_ref_answers_is_appear_i_j) in zip(curr_ann_ref_answers_tokenized_i, curr_ann_ref_answers_is_appear_i):
                                aq_input_ids = [bos_token_id] + curr_ann_ref_answers_tokenized_i_j + [eos_token_id]
                                aq_attention_mask = [1 for _ in aq_input_ids]
                                aq_psgs_input_ids, aq_psgs_attention_mask = [], []
                                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                                    assert psg_attention_mask[-1] == 1
                                    aq_psgs_input_ids_i = aq_input_ids + psg_input_id[1:]
                                    aq_psgs_attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                                    assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    max_input_len = 32 + 128
                                    if len(aq_psgs_input_ids_i) > max_input_len:
                                        aq_psgs_input_ids_i = aq_psgs_input_ids_i[:max_input_len]
                                        aq_psgs_attention_mask_i = aq_psgs_attention_mask_i[:max_input_len]
                                        assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    else:
                                        aq_psgs_input_ids_i += [pad_token_id for _ in range(max_input_len - len(aq_psgs_input_ids_i))]
                                        aq_psgs_attention_mask_i += [0 for _ in range(max_input_len - len(aq_psgs_attention_mask_i))]
                                    aq_psgs_input_ids.append(aq_psgs_input_ids_i)
                                    aq_psgs_attention_mask.append(aq_psgs_attention_mask_i)
                                assert len(aq_psgs_input_ids) == len(aq_psgs_attention_mask)
                                new_input_ids.append(aq_psgs_input_ids)
                                new_attention_mask.append(aq_psgs_attention_mask)
                                discard_not_found_answers.append(curr_ann_ref_answers_is_appear_i_j)
                            assert len(curr_ann_ref_questions_i) == 1
                            new_output.append(curr_ann_ref_questions_i[0])
                            new_metadata.append((new_input_ids_offset, len(new_input_ids)))

                new_output = self.tokenizer.batch_encode_plus(new_output, max_length=32, pad_to_max_length=True)
                new_decoder_input_ids, new_decoder_attention_mask = new_output["input_ids"], new_output["attention_mask"]
                assert len(new_input_ids) == len(new_attention_mask) == len(discard_not_found_answers) == new_metadata[-1][-1]
                assert len(new_decoder_input_ids) == len(new_decoder_attention_mask) == len(new_metadata)
                tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata, discard_not_found_answers]
            else:
                metadata_perann_perqapair = []
                # record valid question answer pairs (some datapoint prompt question is not ambiguous,
                # so we need to skip them)
                metadata_perann_perqapair_offset = 0
                discard_not_found_answers = []
                for idx, (curr_input_ids, curr_attention_mask, curr_ref_questions, curr_ref_answers, curr_metadata, dpr_ids) \
                        in enumerate(zip(tqdm(input_ids), attention_mask, self.ref_questions, self.ref_answers, metadata, dpr_passages)):
                    if curr_ref_questions is None:
                        continue
                    # end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id) + 1
                    # q_input_ids = curr_input_ids[:end_of_question]
                    dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                    dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                    # here we consider all annotations because we dont want this dependent on the retrieval results
                    metadata_perann_perqapair.append([])  # per data point
                    for annotation_idx in range(len(curr_metadata)):
                        metadata_perann_perqapair[metadata_perann_perqapair_offset].append([])  # per annotator
                        ref_questions = curr_ref_questions[annotation_idx]
                        ref_answers = curr_ref_answers[annotation_idx]
                        ref_metadata = curr_metadata[annotation_idx]
                        tokenized_ref_answers = [[_get_tokenized_answer(i, self.args.append_another_bos) for i in range(*m)] for m in ref_metadata]
                        ref_answers_is_appear = [[_included(tokens, dpr_input_ids) for tokens in _tokens] for _tokens in tokenized_ref_answers]
                        assert len(ref_questions) == len(ref_answers) == len(tokenized_ref_answers)

                        # per answer cluster
                        for qapair_idx, (_tkd_ref_answers, ref_question, _ref_answers_is_appear_per_qapair) in enumerate(zip(tokenized_ref_answers, ref_questions, ref_answers_is_appear)):
                            predictions_offset = len(new_output)
                            # per answer in each cluster
                            for tkd_ref_answer, _ref_answers_is_appear_per_qapair_per_ans in zip(_tkd_ref_answers, _ref_answers_is_appear_per_qapair):
                                aq_input_ids = [bos_token_id] + tkd_ref_answer + [eos_token_id]
                                aq_attention_mask = [1 for _ in aq_input_ids]
                                aq_psgs_input_ids, aq_psgs_attention_mask = [], []
                                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                                    assert psg_attention_mask[-1] == 1
                                    aq_psgs_input_ids_i = aq_input_ids + psg_input_id[1:]
                                    aq_psgs_attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                                    assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    max_input_len = 32 + 128
                                    if len(aq_psgs_input_ids_i) > max_input_len:
                                        aq_psgs_input_ids_i = aq_psgs_input_ids_i[:max_input_len]
                                        aq_psgs_attention_mask_i = aq_psgs_attention_mask_i[:max_input_len]
                                        assert len(aq_psgs_input_ids_i) == len(aq_psgs_attention_mask_i)
                                    else:
                                        aq_psgs_input_ids_i += [pad_token_id for _ in range(max_input_len - len(aq_psgs_input_ids_i))]
                                        aq_psgs_attention_mask_i += [0 for _ in range(max_input_len - len(aq_psgs_attention_mask_i))]
                                    aq_psgs_input_ids.append(aq_psgs_input_ids_i)
                                    aq_psgs_attention_mask.append(aq_psgs_attention_mask_i)
                                discard_not_found_answers.append(_ref_answers_is_appear_per_qapair_per_ans)
                                new_input_ids.append(aq_psgs_input_ids)
                                new_attention_mask.append(aq_psgs_attention_mask)
                                new_output.append(ref_question)
                                new_metadata.append((len(new_output)-1, len(new_output)))
                            # start and end for per answer cluster
                            metadata_perann_perqapair[metadata_perann_perqapair_offset][annotation_idx].append((predictions_offset, len(new_output)))
                    metadata_perann_perqapair_offset += 1

                new_decoder_input_ids, new_decoder_attention_mask = None, None
                assert metadata_perann_perqapair[-1][-1][-1][-1] == len(new_input_ids)

                tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask,
                                  new_metadata, discard_not_found_answers, metadata_perann_perqapair,]

            with open(dpr_tokenized_path, "w") as f:
                json.dump(tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data {}".format(dpr_tokenized_path))
            self.tokenized_data = tokenized_data

        if self.args.filter_not_found_answer_passages:
            old_input_ids, old_attention_mask, _, _, _, old_discard_not_found_answers = self.tokenized_data[:6]
            filtered_input_ids, filtered_attention_mask, filtered_discard_not_found_answers = [], [], [],
            truly_filtered_discard_not_found_answers = []
            assert len(old_input_ids) == len(old_attention_mask) == len(old_discard_not_found_answers)
            for old_input_ids_i, old_attention_mask_i, old_discard_not_found_answers_i in zip(old_input_ids, old_attention_mask, old_discard_not_found_answers):
                assert len(old_input_ids_i) == len(old_attention_mask_i) == len(old_discard_not_found_answers_i) == 100
                filtered_input_ids_i, filtered_attention_mask_i, filtered_discard_not_found_answers_i = [], [], [],
                truly_filtered_discard_not_found_answers_i = []
                for old_input_ids_i_j, old_attention_mask_i_j, old_discard_not_found_answers_i_j in zip(old_input_ids_i, old_attention_mask_i, old_discard_not_found_answers_i):
                    if old_discard_not_found_answers_i_j:
                        filtered_input_ids_i.append(old_input_ids_i_j)
                        filtered_attention_mask_i.append(old_attention_mask_i_j)
                        filtered_discard_not_found_answers_i.append(old_discard_not_found_answers_i_j)
                        truly_filtered_discard_not_found_answers_i.append(old_discard_not_found_answers_i_j)
                if len(filtered_input_ids_i) == 0:
                    filtered_input_ids_i, filtered_attention_mask_i = old_input_ids_i, old_attention_mask_i
                    filtered_discard_not_found_answers_i = [True] * len(filtered_input_ids_i)
                    truly_filtered_discard_not_found_answers_i = [False] * len(filtered_input_ids_i)
                else:
                    # pad some tokens
                    while len(filtered_input_ids_i) < 100:
                        # hello -> 20760
                        filtered_input_ids_i.append([20760] * 160)
                        filtered_attention_mask_i.append([1] * 160)
                        filtered_discard_not_found_answers_i.append(False)
                        truly_filtered_discard_not_found_answers_i.append(False)
                filtered_input_ids.append(filtered_input_ids_i)
                filtered_attention_mask.append(filtered_attention_mask_i)
                filtered_discard_not_found_answers.append(filtered_discard_not_found_answers_i)
                truly_filtered_discard_not_found_answers.append(truly_filtered_discard_not_found_answers_i)
            self.tokenized_data[0] = filtered_input_ids
            self.tokenized_data[1] = filtered_attention_mask
            self.tokenized_data[5] = filtered_discard_not_found_answers
            truly_filtered_discard_not_found_answers = [_aq_psgs_truly_discard[:self.args.top_k_passages] for _aq_psgs_truly_discard in truly_filtered_discard_not_found_answers]

        aq_psgs_input_ids, aq_psgs_attention_mask, _, _, _, aq_psgs_discard = self.tokenized_data[:6]
        self.tokenized_data[0] = [_aq_psgs_input_ids[:self.args.top_k_passages] for _aq_psgs_input_ids in aq_psgs_input_ids]
        self.tokenized_data[1] = [_aq_psgs_attention_mask[:self.args.top_k_passages] for _aq_psgs_attention_mask in aq_psgs_attention_mask]
        self.tokenized_data[5] = [_aq_psgs_discard[:self.args.top_k_passages] for _aq_psgs_discard in aq_psgs_discard]

        if self.is_training and self.args.discard_not_found_answers:
            old_input_ids, old_attention_mask, old_decoder_input_ids, old_decoder_attention_mask, old_metadata, discard_not_found_answers = self.tokenized_data
            filtered_input_ids, filtered_attention_mask, filtered_decoder_input_ids, filtered_decoder_attention_mask, filtered_metadata, filtered_discard_not_found_answers = [], [], [], [], [], []
            for old_curr_m, old_curr_decoder_input_ids, old_curr_decoder_attention_mask in zip(old_metadata, old_decoder_input_ids, old_decoder_attention_mask):
                new_start = len(filtered_input_ids)
                is_keep_sample = False
                for old_idx in range(*old_curr_m):
                    if not self.args.filter_not_found_answer_passages:
                        curr_discard_not_found_answers = discard_not_found_answers[old_idx][:self.args.top_k_passages]
                    else:
                        curr_discard_not_found_answers = truly_filtered_discard_not_found_answers[old_idx][:self.args.top_k_passages]
                    if any(curr_discard_not_found_answers):
                        is_keep_sample = True
                        filtered_input_ids.append(old_input_ids[old_idx])
                        filtered_attention_mask.append(old_attention_mask[old_idx])
                        filtered_discard_not_found_answers.append(discard_not_found_answers[old_idx])
                if is_keep_sample:
                    filtered_decoder_input_ids.append(old_curr_decoder_input_ids)
                    filtered_decoder_attention_mask.append(old_curr_decoder_attention_mask)
                    filtered_metadata.append((new_start, len(filtered_input_ids)))
            self.tokenized_data = [filtered_input_ids, filtered_attention_mask, filtered_decoder_input_ids, filtered_decoder_attention_mask, filtered_metadata, filtered_discard_not_found_answers]

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if not self.args.filter_not_found_answer_passages:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, in_metadata = self.tokenized_data[:5]
            self.dataset = MySimpleQGDataset(input_ids,
                                                    attention_mask,
                                                    decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                                    decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                                    in_metadata=in_metadata,
                                                    is_training=self.is_training)
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, in_metadata, discard_not_found_answers = self.tokenized_data[:6]
            self.dataset = MySimpleQGDynamicDataset(input_ids,
                                             attention_mask,
                                             decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                             decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                             in_metadata=in_metadata,
                                             is_training=self.is_training,
                                             discard_not_found_answers=discard_not_found_answers)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def evaluate(self, predictions, n_paragraphs=None):
        metadata_perann_perqapair = self.tokenized_data[-1]
        assert metadata_perann_perqapair[-1][-1][-1][-1] == len(predictions)
        data_to_tokenize = {}
        offset = 0
        for i, (d, ref_questions, ref_answers) in enumerate(zip(self.data, self.ref_questions, self.ref_answers)):
            if ref_questions is None: continue
            data_to_tokenize["prompt.{}".format(offset)] = [{"caption": d["question"]}]
            for ann_idx in range(len(d['annotations'])):
                assert len(ref_questions[ann_idx]) == len(ref_answers[ann_idx]) == len(metadata_perann_perqapair[offset][ann_idx])
                for qapair_idx in range(len(d['annotations'][ann_idx]['qaPairs'])):
                    start, end = metadata_perann_perqapair[offset][ann_idx][qapair_idx]
                    for answer_idx in range(start, end):
                        if not self.args.ambigqa_editqg:
                            pred = predictions[answer_idx]
                            # ref_question = ref_questions[ann_idx][qapair_idx]
                            ref_question = [q.strip() for q in d['annotations'][ann_idx]['qaPairs'][qapair_idx]['question'].split("|")]
                            # per sample - per annotator - per qapair - per answer
                            data_to_tokenize["gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": pred if type(pred) == str else pred[0]}]
                            data_to_tokenize["ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": ref} for ref in ref_question]
                        else:
                            pred = predictions[answer_idx] if type(predictions[answer_idx]) == str else predictions[answer_idx][0]
                            ref_question = [q.strip() for q in d['annotations'][ann_idx]['qaPairs'][qapair_idx]['question'].split("|")]
                            # per sample - per annotator - per qapair - per answer
                            data_to_tokenize["gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": "{} {}?".format(d["question"][:-1], pred)}]
                            data_to_tokenize["ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)] = [{"caption": ref} for ref in ref_question]
            offset += 1
        assert offset == len(metadata_perann_perqapair)

        if self.args.do_train or self.args.is_sagemaker:
            all_tokens = {}
            for k, v in data_to_tokenize.items():
                doc = self.qg_tokenizer(v[0]['caption'])
                tkied = [tk.text for tk in doc if tk.text not in PUNCTUATIONS]
                all_tokens[k] = [' '.join(tkied)]
        else:
            all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu, f1s = [], []
        # per sample
        for offset in range(len(metadata_perann_perqapair)):
            ann_all = []
            for ann_idx in range(len(metadata_perann_perqapair[offset])):
                qapair_bf = []
                for qapair_idx in range(len(metadata_perann_perqapair[offset][ann_idx])):
                    ans_all = []
                    for answer_idx in range(*metadata_perann_perqapair[offset][ann_idx][qapair_idx]):
                        e = get_qg_metrics(_get("gen.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)),
                                           _get("ref.{}.{}.{}.{}".format(offset, ann_idx, qapair_idx, answer_idx)),
                                           _get("prompt.{}".format(offset)),
                                           metrics=["bleu4", "edit-f1"])
                        ans_all.append((e["bleu4"], e["edit-f1"]))
                    ans_avg = (np.mean([x[0] for x in ans_all]), np.mean([x[1] for x in ans_all]))
                    qapair_bf.append(ans_avg)
                # get average result on qapair_bf
                qapair_avg_bleu = np.mean([b[0] for b in qapair_bf])
                qapair_avg_editf1 = np.mean([b[1] for b in qapair_bf])
                ann_all.append((qapair_avg_bleu, qapair_avg_editf1))
            ann_avg = (np.mean([x[0] for x in ann_all]), np.mean([x[1] for x in ann_all]))
            bleu.append(ann_avg[0])
            f1s.append(ann_avg[1])
        self.logger.info("BLEU=%.2f; EDIT-F1=%.2f" % (100 * np.mean(bleu), 100 * np.mean(f1s)))
        results = {
            'BLEU': np.mean(bleu) * 100,
            'EDIT-F1': np.mean(f1s) * 100,
        }
        return results['EDIT-F1'], results