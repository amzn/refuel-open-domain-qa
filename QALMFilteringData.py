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

class AmbigQALMFilteringData():
    def __init__(self, logger, args, data_path, is_training, passages=None, over_generate_data=None):
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

        if over_generate_data is not None:
            self.data = over_generate_data
        else:
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

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        self.dataset = MySimpleQALMFilteringDataset(input_ids,
                                                    attention_mask,
                                                    decoder_input_ids=decoder_input_ids,
                                                    decoder_attention_mask=decoder_attention_mask,
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
        answer_input = tokenizer.batch_encode_plus(answers,
                                                   pad_to_max_length=True,
                                                   max_length=20)

        input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
        decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
        tokenized_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata]

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

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
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
        self.tokenized_data[2] = decoder_input_ids
        self.tokenized_data[3] = decoder_attention_mask
        self.tokenized_data[4] = metadata

    # override
    def evaluate(self, lm_scores):
        from scipy.stats.stats import pearsonr
        reference = deepcopy(self.data)
        num_qapairs_annotated = []
        for idx, d in enumerate(reference):
            # get gold number of qapairs per question
            num_qapairs_per_ann = []
            for ann in d['annotations']:
                if ann['type'] == 'singleAnswer':
                    num_qapairs_per_ann.append(1)
                else:
                    num_qapairs_per_ann.append(len(ann['qaPairs']))
            num_qapairs_annotated.append(np.mean(num_qapairs_per_ann))

        all_filtering_methods_results = {}
        all_filtering_methods_predictions = {}


        metadata = self.tokenized_data[-1]

        # first, select only 1 question for each predicted answers
        num_qapairs_unfiltered = []
        num_qapairs_no_threshold = []
        prediction_no_threshold = {}
        # compute lm_scores when > 1 answers predicted, for single answer question predictions, we need to use them in any case
        lm_scores_no_threshold = []
        for idx, d in enumerate(self.data):
            curr_metadata = metadata[idx]
            curr_answer_question_pairs = {}
            prediction_no_threshold[d['id']] = []
            for jdx in range(*curr_metadata):
                curr_lm_score = lm_scores[jdx]
                curr_question = self.input_questions[jdx]
                curr_answer = normalize_answer(self.input_answers[jdx])
                if curr_answer not in curr_answer_question_pairs.keys():
                    curr_answer_question_pairs[curr_answer] = [(curr_question, curr_lm_score)]
                else:
                    curr_answer_question_pairs[curr_answer].append((curr_question, curr_lm_score))
            num_qapairs_no_threshold.append(len(curr_answer_question_pairs))
            num_qapairs_unfiltered.append(curr_metadata[1] - curr_metadata[0])
            for answer, question_score in curr_answer_question_pairs.items():
                best_question_score = sorted(question_score, key=lambda x: x[1],)[0]
                prediction_no_threshold[d['id']].append({'question': best_question_score[0], 'answer': answer, 'lm_score': best_question_score[1]})
                if len(curr_answer_question_pairs) > 1:
                    lm_scores_no_threshold.append(best_question_score[1])
        # get a sense of current lm scores
        print('LM scores (no threshold>1) avg {:.2f}, min {:.2f}, max {:.2f}, median {:.2f}, std {:.2f}'.format(
            np.mean(lm_scores_no_threshold), min(lm_scores_no_threshold), max(lm_scores_no_threshold), np.median(lm_scores_no_threshold), np.std(lm_scores_no_threshold)))

        all_filtering_methods_results['lm_score'] = {
            'avg': np.mean(lm_scores_no_threshold),
            'std': np.std(lm_scores_no_threshold),
            'min': min(lm_scores_no_threshold),
            'max': max(lm_scores_no_threshold),
            'median': np.median(lm_scores_no_threshold),
            }

        print("Select the best question of each answer: {:.2f} -> {:.2f}".format(np.mean(num_qapairs_unfiltered), np.mean(num_qapairs_no_threshold)))
        # evaluate this method
        evaluation_no_threshold = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_no_threshold))
        results_no_threshold = evaluation_no_threshold.print_all_metrics(verbose=False)
        print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
            results_no_threshold['F1 answer'], results_no_threshold['F1 answer (multi)'], results_no_threshold["F1 bleu4"], results_no_threshold["F1 edit-f1"]))

        all_filtering_methods_results['best_question'] = {
            'threshold': 1000000000,
            'Avg QAPair': np.mean(num_qapairs_no_threshold),
            'QAPair Corr': pearsonr(num_qapairs_no_threshold, num_qapairs_annotated)[0],
            'F1 answer': results_no_threshold['F1 answer'],
            'F1 answer (multi)': results_no_threshold['F1 answer (multi)'],
            "F1 bleu4": results_no_threshold["F1 bleu4"],
            "F1 edit-f1": results_no_threshold["F1 edit-f1"],
        }
        all_filtering_methods_predictions['best_question'] = deepcopy(prediction_no_threshold)

        lm_avg = np.mean(lm_scores_no_threshold)
        lm_std = np.std(lm_scores_no_threshold)

        print('=*=*' * 10)
        print('=*=*' * 10)
        print('=*=*' * 10)

        threshold_avg_std = lm_avg + lm_std
        prediction_threshold_avg_std = {}
        num_qapairs_threshold_avg_std = []
        for idx, d in enumerate(self.data):
            curr_prediction_no_threshold = prediction_no_threshold[d['id']]
            curr_prediction_threshold = [x for x in curr_prediction_no_threshold if x['lm_score'] < threshold_avg_std]
            if len(curr_prediction_threshold) == 0:
                curr_prediction_threshold = [sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])[0]]
            prediction_threshold_avg_std[d['id']] = curr_prediction_threshold
            num_qapairs_threshold_avg_std.append(len(prediction_threshold_avg_std[d['id']]))
        evaluation_threshold_avg_std = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_avg_std))
        results_threshold_avg_std = evaluation_threshold_avg_std.print_all_metrics(verbose=False)
        print("Threshold lm_avg+lm_std {:.2f}: {:.2f} -> {:.2f}".format(
            threshold_avg_std, np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_avg_std)))
        print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
            results_threshold_avg_std['F1 answer'], results_threshold_avg_std['F1 answer (multi)'], results_threshold_avg_std["F1 bleu4"], results_threshold_avg_std["F1 edit-f1"]))

        all_filtering_methods_results['th_lm_avg_std'] = {
            'threshold': threshold_avg_std,
            'Avg QAPair': np.mean(num_qapairs_threshold_avg_std),
            'QAPair Corr': pearsonr(num_qapairs_threshold_avg_std, num_qapairs_annotated)[0],
            'F1 answer': results_threshold_avg_std['F1 answer'],
            'F1 answer (multi)': results_threshold_avg_std['F1 answer (multi)'],
            "F1 bleu4": results_threshold_avg_std["F1 bleu4"],
            "F1 edit-f1": results_threshold_avg_std["F1 edit-f1"],
        }
        all_filtering_methods_predictions['th_lm_avg_std'] = deepcopy(prediction_threshold_avg_std)


        print('=*=*' * 10)
        print('=*=*' * 10)
        print('=*=*' * 10)

        threshold_avg_std_2 = lm_avg + lm_std * 2
        prediction_threshold_avg_std_2 = {}
        num_qapairs_threshold_avg_std_2 = []
        for idx, d in enumerate(self.data):
            curr_prediction_no_threshold = prediction_no_threshold[d['id']]
            curr_prediction_threshold = [x for x in curr_prediction_no_threshold if x['lm_score'] < threshold_avg_std_2]
            if len(curr_prediction_threshold) == 0:
                curr_prediction_threshold = [sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])[0]]
            prediction_threshold_avg_std_2[d['id']] = curr_prediction_threshold
            num_qapairs_threshold_avg_std_2.append(len(prediction_threshold_avg_std_2[d['id']]))
        evaluation_threshold_avg_std_2 = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_avg_std_2))
        results_threshold_avg_std_2 = evaluation_threshold_avg_std_2.print_all_metrics(verbose=False)
        print("Threshold lm_avg+lm_std*2 {:.2f}: {:.2f} -> {:.2f}".format(
            threshold_avg_std_2, np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_avg_std_2)))
        print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
            results_threshold_avg_std_2['F1 answer'], results_threshold_avg_std_2['F1 answer (multi)'], results_threshold_avg_std_2["F1 bleu4"], results_threshold_avg_std_2["F1 edit-f1"]))

        all_filtering_methods_results['th_lm_avg_std_2'] = {
            'threshold': threshold_avg_std_2,
            'Avg QAPair': np.mean(num_qapairs_threshold_avg_std_2),
            'QAPair Corr': pearsonr(num_qapairs_threshold_avg_std_2, num_qapairs_annotated)[0],
            'F1 answer': results_threshold_avg_std_2['F1 answer'],
            'F1 answer (multi)': results_threshold_avg_std_2['F1 answer (multi)'],
            "F1 bleu4": results_threshold_avg_std_2["F1 bleu4"],
            "F1 edit-f1": results_threshold_avg_std_2["F1 edit-f1"],
        }
        all_filtering_methods_predictions['th_lm_avg_std_2'] = deepcopy(prediction_threshold_avg_std_2)

        # print('=*=*' * 10)
        # print('=*=*' * 10)
        # print('=*=*' * 10)
        #
        # prediction_threshold_adaptive_std_avg = {}  # <avg-std
        # num_qapairs_threshold_adaptive_std_avg = []
        # threshold_adaptive_std_avg = []
        # for idx, d in enumerate(self.data):
        #     curr_prediction_no_threshold = prediction_no_threshold[d['id']]
        #     if len(curr_prediction_no_threshold) == 1:
        #         prediction_threshold_adaptive_std_avg[d['id']] = curr_prediction_no_threshold
        #         num_qapairs_threshold_adaptive_std_avg.append(1)
        #     else:
        #         curr_prediction_lm_score_avg = np.mean([x['lm_score'] for x in curr_prediction_no_threshold])
        #         curr_prediction_lm_score_std = np.std([x['lm_score'] for x in curr_prediction_no_threshold])
        #         # threshold = avg - std
        #         curr_threshold = curr_prediction_lm_score_avg - curr_prediction_lm_score_std
        #         curr_prediction_no_threshold_sorted = sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])
        #         curr_prediction_threshold = [x for x in curr_prediction_no_threshold_sorted if x['lm_score'] < curr_threshold]
        #         if len(curr_prediction_threshold) == 0:
        #             curr_prediction_threshold = [curr_prediction_no_threshold_sorted[0]]
        #         else:
        #             threshold_adaptive_std_avg.append(curr_threshold)
        #         prediction_threshold_adaptive_std_avg[d['id']] = curr_prediction_threshold
        #         num_qapairs_threshold_adaptive_std_avg.append(len(curr_prediction_threshold))
        # evaluation_threshold_adaptive_std_avg = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_adaptive_std_avg))
        # results_threshold_adaptive_std_avg = evaluation_threshold_adaptive_std_avg.print_all_metrics(verbose=False)
        # print("Threshold adaptive avg-std: mean {:.2f} std {:.2f}: {:.2f} -> {:.2f}".format(
        #     np.mean(threshold_adaptive_std_avg), np.std(threshold_adaptive_std_avg), np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_adaptive_std_avg)))
        # print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
        #     results_threshold_adaptive_std_avg['F1 answer'], results_threshold_adaptive_std_avg['F1 answer (multi)'], results_threshold_adaptive_std_avg["F1 bleu4"], results_threshold_adaptive_std_avg["F1 edit-f1"]))
        #
        # all_filtering_methods_results['th_lm_adaptive_std_avg'] = {
        #     'threshold_avg': np.mean(threshold_adaptive_std_avg),
        #     'threshold_std': np.std(threshold_adaptive_std_avg),
        #     'Avg QAPair': np.mean(num_qapairs_threshold_adaptive_std_avg),
        #     'QAPair Corr': pearsonr(num_qapairs_threshold_adaptive_std_avg, num_qapairs_annotated)[0],
        #     'F1 answer': results_threshold_adaptive_std_avg['F1 answer'],
        #     'F1 answer (multi)': results_threshold_adaptive_std_avg['F1 answer (multi)'],
        #     "F1 bleu4": results_threshold_adaptive_std_avg["F1 bleu4"],
        #     "F1 edit-f1": results_threshold_adaptive_std_avg["F1 edit-f1"],
        # }
        # all_filtering_methods_predictions['th_lm_adaptive_std_avg'] = deepcopy(prediction_threshold_adaptive_std_avg)
        #
        # print('=*=*' * 10)
        # print('=*=*' * 10)
        #
        # prediction_threshold_adaptive_avg = {}  # <avg
        # num_qapairs_threshold_adaptive_avg = []
        # threshold_adaptive_avg = []
        # for idx, d in enumerate(self.data):
        #     curr_prediction_no_threshold = prediction_no_threshold[d['id']]
        #     if len(curr_prediction_no_threshold) == 1:
        #         prediction_threshold_adaptive_avg[d['id']] = curr_prediction_no_threshold
        #         num_qapairs_threshold_adaptive_avg.append(1)
        #     else:
        #         curr_prediction_lm_score_avg = np.mean([x['lm_score'] for x in curr_prediction_no_threshold])
        #         curr_prediction_lm_score_std = np.std([x['lm_score'] for x in curr_prediction_no_threshold])
        #         # threshold = avg
        #         curr_threshold = curr_prediction_lm_score_avg
        #         curr_prediction_no_threshold_sorted = sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])
        #         curr_prediction_threshold = [x for x in curr_prediction_no_threshold_sorted if x['lm_score'] < curr_threshold]
        #         if len(curr_prediction_threshold) == 0:
        #             curr_prediction_threshold = [curr_prediction_no_threshold_sorted[0]]
        #         else:
        #             threshold_adaptive_avg.append(curr_threshold)
        #         prediction_threshold_adaptive_avg[d['id']] = curr_prediction_threshold
        #         num_qapairs_threshold_adaptive_avg.append(len(curr_prediction_threshold))
        # evaluation_threshold_adaptive_avg = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_adaptive_avg))
        # results_threshold_adaptive_avg = evaluation_threshold_adaptive_avg.print_all_metrics(verbose=False)
        # print("Threshold adaptive avg: mean {:.2f} std {:.2f}: {:.2f} -> {:.2f}".format(
        #     np.mean(threshold_adaptive_avg), np.std(threshold_adaptive_avg), np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_adaptive_avg)))
        # print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
        #     results_threshold_adaptive_avg['F1 answer'], results_threshold_adaptive_avg['F1 answer (multi)'], results_threshold_adaptive_avg["F1 bleu4"],
        #     results_threshold_adaptive_avg["F1 edit-f1"]))
        #
        # all_filtering_methods_results['th_lm_adaptive_avg'] = {
        #     'threshold_avg': np.mean(threshold_adaptive_avg),
        #     'threshold_std': np.std(threshold_adaptive_avg),
        #     'Avg QAPair': np.mean(num_qapairs_threshold_adaptive_avg),
        #     'QAPair Corr': pearsonr(num_qapairs_threshold_adaptive_avg, num_qapairs_annotated)[0],
        #     'F1 answer': results_threshold_adaptive_avg['F1 answer'],
        #     'F1 answer (multi)': results_threshold_adaptive_avg['F1 answer (multi)'],
        #     "F1 bleu4": results_threshold_adaptive_avg["F1 bleu4"],
        #     "F1 edit-f1": results_threshold_adaptive_avg["F1 edit-f1"],
        # }
        # all_filtering_methods_predictions['th_lm_adaptive_avg'] = deepcopy(prediction_threshold_adaptive_avg)
        #
        # print('=*=*' * 10)
        # print('=*=*' * 10)
        #
        # prediction_threshold_adaptive_avg_std = {}  # <avg+std
        # num_qapairs_threshold_adaptive_avg_std = []
        # threshold_adaptive_avg_std = []
        # for idx, d in enumerate(self.data):
        #     curr_prediction_no_threshold = prediction_no_threshold[d['id']]
        #     if len(curr_prediction_no_threshold) == 1:
        #         prediction_threshold_adaptive_avg_std[d['id']] = curr_prediction_no_threshold
        #         num_qapairs_threshold_adaptive_avg_std.append(1)
        #     else:
        #         curr_prediction_lm_score_avg = np.mean([x['lm_score'] for x in curr_prediction_no_threshold])
        #         curr_prediction_lm_score_std = np.std([x['lm_score'] for x in curr_prediction_no_threshold])
        #         # threshold = avg + std
        #         curr_threshold = curr_prediction_lm_score_avg + curr_prediction_lm_score_std
        #         curr_prediction_no_threshold_sorted = sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])
        #         curr_prediction_threshold = [x for x in curr_prediction_no_threshold_sorted if x['lm_score'] < curr_threshold]
        #         if len(curr_prediction_threshold) == 0:
        #             curr_prediction_threshold = [curr_prediction_no_threshold_sorted[0]]
        #         else:
        #             threshold_adaptive_avg_std.append(curr_threshold)
        #         prediction_threshold_adaptive_avg_std[d['id']] = curr_prediction_threshold
        #         num_qapairs_threshold_adaptive_avg_std.append(len(curr_prediction_threshold))
        # evaluation_threshold_adaptive_avg_std = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_adaptive_avg_std))
        # results_threshold_adaptive_avg_std = evaluation_threshold_adaptive_avg_std.print_all_metrics(verbose=False)
        # print("Threshold adaptive avg+std: mean {:.2f} std {:.2f}: {:.2f} -> {:.2f}".format(
        #     np.mean(threshold_adaptive_avg_std), np.std(threshold_adaptive_avg_std), np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_adaptive_avg_std)))
        # print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
        #     results_threshold_adaptive_avg_std['F1 answer'], results_threshold_adaptive_avg_std['F1 answer (multi)'], results_threshold_adaptive_avg_std["F1 bleu4"],
        #     results_threshold_adaptive_avg_std["F1 edit-f1"]))
        #
        # all_filtering_methods_results['th_lm_adaptive_avg_std'] = {
        #     'threshold_avg': np.mean(threshold_adaptive_avg_std),
        #     'threshold_std': np.std(threshold_adaptive_avg_std),
        #     'Avg QAPair': np.mean(num_qapairs_threshold_adaptive_avg_std),
        #     'QAPair Corr': pearsonr(num_qapairs_threshold_adaptive_avg_std, num_qapairs_annotated)[0],
        #     'F1 answer': results_threshold_adaptive_avg_std['F1 answer'],
        #     'F1 answer (multi)': results_threshold_adaptive_avg_std['F1 answer (multi)'],
        #     "F1 bleu4": results_threshold_adaptive_avg_std["F1 bleu4"],
        #     "F1 edit-f1": results_threshold_adaptive_avg_std["F1 edit-f1"],
        # }
        # all_filtering_methods_predictions['th_lm_adaptive_avg_std'] = deepcopy(prediction_threshold_adaptive_avg_std)
        #
        # print('=*=*' * 10)
        # print('=*=*' * 10)
        #
        # prediction_threshold_adaptive_avg_std_2 = {}  # <avg+2*std
        # num_qapairs_threshold_adaptive_avg_std_2 = []
        # threshold_adaptive_avg_std_2 = []
        # for idx, d in enumerate(self.data):
        #     curr_prediction_no_threshold = prediction_no_threshold[d['id']]
        #     if len(curr_prediction_no_threshold) == 1:
        #         prediction_threshold_adaptive_avg_std_2[d['id']] = curr_prediction_no_threshold
        #         num_qapairs_threshold_adaptive_avg_std_2.append(1)
        #     else:
        #         curr_prediction_lm_score_avg = np.mean([x['lm_score'] for x in curr_prediction_no_threshold])
        #         curr_prediction_lm_score_std = np.std([x['lm_score'] for x in curr_prediction_no_threshold])
        #         # threshold = avg + std * 2
        #         curr_threshold = curr_prediction_lm_score_avg + curr_prediction_lm_score_std * 2
        #         curr_prediction_no_threshold_sorted = sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])
        #         curr_prediction_threshold = [x for x in curr_prediction_no_threshold_sorted if x['lm_score'] < curr_threshold]
        #         if len(curr_prediction_threshold) == 0:
        #             curr_prediction_threshold = [curr_prediction_no_threshold_sorted[0]]
        #         else:
        #             threshold_adaptive_avg_std_2.append(curr_threshold)
        #         prediction_threshold_adaptive_avg_std_2[d['id']] = curr_prediction_threshold
        #         num_qapairs_threshold_adaptive_avg_std_2.append(len(curr_prediction_threshold))
        # evaluation_threshold_adaptive_avg_std_2 = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_adaptive_avg_std_2))
        # results_threshold_adaptive_avg_std_2 = evaluation_threshold_adaptive_avg_std_2.print_all_metrics(verbose=False)
        # print("Threshold adaptive avg+std*2: mean {:.2f} std {:.2f}: {:.2f} -> {:.2f}".format(
        #     np.mean(threshold_adaptive_avg_std_2), np.std(threshold_adaptive_avg_std_2), np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_adaptive_avg_std_2)))
        # print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
        #     results_threshold_adaptive_avg_std_2['F1 answer'], results_threshold_adaptive_avg_std_2['F1 answer (multi)'], results_threshold_adaptive_avg_std_2["F1 bleu4"],
        #     results_threshold_adaptive_avg_std_2["F1 edit-f1"]))
        #
        # all_filtering_methods_results['th_lm_adaptive_avg_std_2'] = {
        #     'threshold_avg': np.mean(threshold_adaptive_avg_std_2),
        #     'threshold_std': np.std(threshold_adaptive_avg_std_2),
        #     'Avg QAPair': np.mean(num_qapairs_threshold_adaptive_avg_std_2),
        #     'QAPair Corr': pearsonr(num_qapairs_threshold_adaptive_avg_std_2, num_qapairs_annotated)[0],
        #     'F1 answer': results_threshold_adaptive_avg_std_2['F1 answer'],
        #     'F1 answer (multi)': results_threshold_adaptive_avg_std_2['F1 answer (multi)'],
        #     "F1 bleu4": results_threshold_adaptive_avg_std_2["F1 bleu4"],
        #     "F1 edit-f1": results_threshold_adaptive_avg_std_2["F1 edit-f1"],
        # }
        # all_filtering_methods_predictions['th_lm_adaptive_avg_std_2'] = deepcopy(prediction_threshold_adaptive_avg_std_2)
        #
        # print('=*=*' * 10)
        # print('=*=*' * 10)
        #
        # # local best threshold for each prompt question
        # threshold_adaptive = []
        # from collections import defaultdict
        # threshold_adaptive_interval = defaultdict(int)
        # prediction_threshold_adaptive = {}
        # num_qapairs_threshold_adaptive = []
        # for idx, d in enumerate(self.data):
        #     curr_prediction_no_threshold = prediction_no_threshold[d['id']]
        #     if len(curr_prediction_no_threshold) == 1:
        #         prediction_threshold_adaptive[d['id']] = curr_prediction_no_threshold
        #         num_qapairs_threshold_adaptive.append(1)
        #     else:
        #         curr_prediction_lm_score_avg = np.mean([x['lm_score'] for x in curr_prediction_no_threshold])
        #         curr_prediction_lm_score_std = np.std([x['lm_score'] for x in curr_prediction_no_threshold])
        #         curr_prediction_no_threshold_sorted = sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])
        #         curr_best_top_k, curr_best_score = -1, -1
        #         for curr_top_idx in range(1, len(curr_prediction_no_threshold_sorted)+1):
        #             if any([x['type'] == 'multipleQAs' for x in d['annotations']]):
        #                 curr_prediction_top_idx = {d['id']: curr_prediction_no_threshold_sorted[:curr_top_idx]}
        #                 curr_evaluation_top_idx = QAPairEvaluation(deepcopy([d]), deepcopy(curr_prediction_top_idx))
        #                 curr_evaluation_result_top_idx = curr_evaluation_top_idx.print_all_metrics(verbose=False)
        #                 if curr_evaluation_result_top_idx['F1 answer'] + curr_evaluation_result_top_idx["F1 edit-f1"] > curr_best_score:
        #                     curr_best_top_k = curr_top_idx
        #                     curr_best_score = curr_evaluation_result_top_idx['F1 answer'] + curr_evaluation_result_top_idx["F1 edit-f1"]
        #             else:
        #                 curr_prediction_top_idx = {d['id']: [x['answer'] for x in curr_prediction_no_threshold_sorted[:curr_top_idx]]}
        #                 curr_evaluation_top_idx = QAPairEvaluation(deepcopy([d]), deepcopy(curr_prediction_top_idx))
        #                 curr_evaluation_result_top_idx = curr_evaluation_top_idx.print_all_metrics(verbose=False)
        #                 if curr_evaluation_result_top_idx['F1 answer'] > curr_best_score:
        #                     curr_best_top_k = curr_top_idx
        #                     curr_best_score = curr_evaluation_result_top_idx['F1 answer']
        #         if curr_best_top_k == -1:
        #             curr_best_top_k = 1
        #         curr_threshold = curr_prediction_no_threshold_sorted[curr_best_top_k-1]['lm_score']
        #         num_qapairs_threshold_adaptive.append(curr_best_top_k)
        #         threshold_adaptive.append(curr_threshold)
        #         prediction_threshold_adaptive[d['id']] = curr_prediction_no_threshold_sorted[:curr_best_top_k]
        #         if curr_threshold < curr_prediction_lm_score_avg+0.005:
        #             threshold_adaptive_interval['<avg'] += 1
        #         elif curr_prediction_lm_score_avg+0.005 <= curr_threshold < curr_prediction_lm_score_avg + curr_prediction_lm_score_std + 0.005:
        #             threshold_adaptive_interval['avg~avg+std'] += 1
        #         elif curr_prediction_lm_score_avg+curr_prediction_lm_score_std+0.005 <= curr_threshold < curr_prediction_lm_score_avg + 2*curr_prediction_lm_score_std + 0.005:
        #             threshold_adaptive_interval['avg~avg+2*std'] += 1
        #         else:
        #             threshold_adaptive_interval['>avg+2*std'] += 1
        #
        # evaluation_threshold_adaptive = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold_adaptive))
        # results_threshold_adaptive = evaluation_threshold_adaptive.print_all_metrics(verbose=False)
        # print("Threshold adaptive mean {:.2f} std {:.2f}: {:.2f} -> {:.2f}".format(
        #     np.mean(threshold_adaptive), np.std(threshold_adaptive), np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_adaptive)))
        # print('Threshold adaptive intervals: ')
        # print(threshold_adaptive_interval)
        # print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
        #     results_threshold_adaptive['F1 answer'], results_threshold_adaptive['F1 answer (multi)'], results_threshold_adaptive["F1 bleu4"], results_threshold_adaptive["F1 edit-f1"]))
        #
        # all_filtering_methods_results['th_lm_adaptive'] = {
        #     'threshold_avg': np.mean(threshold_adaptive),
        #     'threshold_std': np.std(threshold_adaptive),
        #     'Avg QAPair': np.mean(num_qapairs_threshold_adaptive),
        #     'QAPair Corr': pearsonr(num_qapairs_threshold_adaptive, num_qapairs_annotated)[0],
        #     'F1 answer': results_threshold_adaptive['F1 answer'],
        #     'F1 answer (multi)': results_threshold_adaptive['F1 answer (multi)'],
        #     "F1 bleu4": results_threshold_adaptive["F1 bleu4"],
        #     "F1 edit-f1": results_threshold_adaptive["F1 edit-f1"],
        #     "interval": threshold_adaptive_interval,
        # }
        # all_filtering_methods_predictions['th_lm_adaptive'] = deepcopy(prediction_threshold_adaptive)

        print('=*=*' * 10)
        print('=*=*' * 10)
        print('=*=*' * 10)

        # select top answers according to a threshold
        best_threshold, best_predictions, best_evaluation_results, best_evaluation_threshold, best_num_answer_question_pairs_threshold = -1, None, -1, None, None
        # for threshold in np.linspace(min(lm_scores), max(lm_scores), num=30):
        for reverse_threshold in np.linspace(0.0, 20, num=201):
            threshold = 20.5 - reverse_threshold
            prediction_threshold = {}
            lm_scores_threshold = []
            num_answer_question_pairs_threshold = []
            for idx, d in enumerate(self.data):
                curr_prediction_no_threshold = prediction_no_threshold[d['id']]
                curr_prediction_threshold = [x for x in curr_prediction_no_threshold if x['lm_score'] < threshold]
                if len(curr_prediction_threshold) == 0:
                    curr_prediction_threshold = [sorted(curr_prediction_no_threshold, key=lambda x:x['lm_score'])[0]]
                prediction_threshold[d['id']] = curr_prediction_threshold
                if len(prediction_threshold[d['id']]) > 1:
                    lm_scores_threshold.extend([x['lm_score'] for x in prediction_threshold[d['id']]])
                num_answer_question_pairs_threshold.append(len(prediction_threshold[d['id']]))

            # evaluate this method
            evaluation_threshold = QAPairEvaluation(deepcopy(reference), deepcopy(prediction_threshold))
            results_threshold = evaluation_threshold.print_all_metrics(verbose=False)
            print("Threshold {:.2f} filtering: {:.2f} -> {:.2f}".format(
                threshold, np.mean(num_qapairs_no_threshold), np.mean(num_answer_question_pairs_threshold)))
            print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
                results_threshold['F1 answer'], results_threshold['F1 answer (multi)'], results_threshold["F1 bleu4"], results_threshold["F1 edit-f1"]))
            if results_threshold['F1 answer'] + results_threshold["F1 edit-f1"] > best_evaluation_results:
                best_threshold = threshold
                best_predictions = deepcopy(prediction_threshold)
                best_evaluation_results = results_threshold['F1 answer'] + results_threshold["F1 edit-f1"]
                best_evaluation_threshold = evaluation_threshold
                best_num_answer_question_pairs_threshold = num_answer_question_pairs_threshold
            print('=*=*' * 10)
        print("Best Threshold {:.2f}".format(best_threshold))
        best_results = best_evaluation_threshold.print_all_metrics(verbose=False)
        print("filtering: {:.2f} -> {:.2f}".format( np.mean(num_qapairs_no_threshold), np.mean(best_num_answer_question_pairs_threshold)))
        print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
            best_results['F1 answer'], best_results['F1 answer (multi)'], best_results["F1 bleu4"], best_results["F1 edit-f1"]))

        all_filtering_methods_results['th_best'] = {
            'threshold': best_threshold,
            'Avg QAPair': np.mean(best_num_answer_question_pairs_threshold),
            'QAPair Corr': pearsonr(best_num_answer_question_pairs_threshold, num_qapairs_annotated)[0],
            'F1 answer': best_results['F1 answer'],
            'F1 answer (multi)': best_results['F1 answer (multi)'],
            "F1 bleu4": best_results["F1 bleu4"],
            "F1 edit-f1": best_results["F1 edit-f1"],
        }
        all_filtering_methods_predictions['th_best'] = deepcopy(best_predictions)

        return all_filtering_methods_predictions, all_filtering_methods_results


    def predict(self, lm_scores, predefined_threshold, mode):
        metadata = self.tokenized_data[-1]

        # first, select only 1 question for each predicted answers
        num_qapairs_unfiltered = []
        num_qapairs_no_threshold = []
        prediction_no_threshold = {}
        # compute lm_scores when > 1 answers predicted, for single answer question predictions, we need to use them in any case
        lm_scores_no_threshold = []
        for idx, d in enumerate(self.data):
            curr_metadata = metadata[idx]
            curr_answer_question_pairs = {}
            prediction_no_threshold[d['id']] = []
            for jdx in range(*curr_metadata):
                curr_lm_score = lm_scores[jdx]
                curr_question = self.input_questions[jdx]
                curr_answer = normalize_answer(self.input_answers[jdx])
                if curr_answer not in curr_answer_question_pairs.keys():
                    curr_answer_question_pairs[curr_answer] = [(curr_question, curr_lm_score)]
                else:
                    curr_answer_question_pairs[curr_answer].append((curr_question, curr_lm_score))
            num_qapairs_no_threshold.append(len(curr_answer_question_pairs))
            num_qapairs_unfiltered.append(curr_metadata[1] - curr_metadata[0])
            for answer, question_score in curr_answer_question_pairs.items():
                best_question_score = sorted(question_score, key=lambda x: x[1],)[0]
                prediction_no_threshold[d['id']].append({'question': best_question_score[0], 'answer': answer, 'lm_score': best_question_score[1]})
                if len(curr_answer_question_pairs) > 1:
                    lm_scores_no_threshold.append(best_question_score[1])
        # get a sense of current lm scores
        print('LM scores (no threshold>1) avg {:.2f}, min {:.2f}, max {:.2f}, median {:.2f}, std {:.2f}'.format(
            np.mean(lm_scores_no_threshold), min(lm_scores_no_threshold), max(lm_scores_no_threshold), np.median(lm_scores_no_threshold), np.std(lm_scores_no_threshold)))

        print("Select the best question of each answer: {:.2f} -> {:.2f}".format(np.mean(num_qapairs_unfiltered), np.mean(num_qapairs_no_threshold)))

        if mode == "global_avg_std":
            lm_avg = np.mean(lm_scores_no_threshold)
            lm_std = np.std(lm_scores_no_threshold)

            print('=*=*' * 10)
            print('=*=*' * 10)
            print('=*=*' * 10)

            threshold_avg_std = lm_avg + lm_std
            prediction_threshold_avg_std = {}
            num_qapairs_threshold_avg_std = []
            for idx, d in enumerate(self.data):
                curr_prediction_no_threshold = prediction_no_threshold[d['id']]
                curr_prediction_threshold = [x for x in curr_prediction_no_threshold if x['lm_score'] < threshold_avg_std]
                if len(curr_prediction_threshold) == 0:
                    curr_prediction_threshold = [sorted(curr_prediction_no_threshold, key=lambda x: x['lm_score'])[0]]
                prediction_threshold_avg_std[d['id']] = curr_prediction_threshold
                num_qapairs_threshold_avg_std.append(len(prediction_threshold_avg_std[d['id']]))

            print("Threshold lm_avg+lm_std {:.2f}: {:.2f} -> {:.2f}".format(
                threshold_avg_std, np.mean(num_qapairs_no_threshold), np.mean(num_qapairs_threshold_avg_std)))
            return prediction_threshold_avg_std

        elif mode == "fixed":
            print('=*=*' * 10)
            print('=*=*' * 10)
            print('=*=*' * 10)

            # select top answers according to a threshold
            prediction_threshold = {}
            num_answer_question_pairs_threshold = []
            for idx, d in enumerate(self.data):
                curr_prediction_no_threshold = prediction_no_threshold[d['id']]
                curr_prediction_threshold = [x for x in curr_prediction_no_threshold if x['lm_score'] < predefined_threshold]
                if len(curr_prediction_threshold) == 0:
                    curr_prediction_threshold = [sorted(curr_prediction_no_threshold, key=lambda x:x['lm_score'])[0]]
                prediction_threshold[d['id']] = curr_prediction_threshold
                num_answer_question_pairs_threshold.append(len(prediction_threshold[d['id']]))
            print("Threshold {:.2f} filtering: {:.2f} -> {:.2f}".format(predefined_threshold, np.mean(num_qapairs_no_threshold), np.mean(num_answer_question_pairs_threshold)))
            return prediction_threshold
        else:
            raise NotImplementedError
