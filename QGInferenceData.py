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
from copy import deepcopy

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from QAData import QAData, AmbigQAData
from QGData import QGData, AmbigQGData
from DataLoader import MySimpleQADataset, MySimpleQADatasetForPair, MyDataLoader, MySimpleQGDynamicDataset, MySimpleQGDataset
from util import decode_span_batch

# for evaluation
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics, QAPairEvaluation
from pycocoevalcap.bleu.bleu import Bleu

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class AmbigQGInferenceData():
    def __init__(self, logger, args, data_path, is_training, passages=None, qa_data=None, dpr_data=None):
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

        self.data = qa_data
        self.dpr_reranked_tokenized_data = dpr_data

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
        self.SEP = "<SEP>"
        self.qg_tokenizer = PTBTokenizer()

    def __len__(self):
        return len(self.data)

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def decode(self, tokens):
        if type(tokens[0])==list:
            return [self.decode(_tokens) for _tokens in tokens]
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip().replace(" - ", "-").replace(" : ", ":")

    def save_predictions(self, predictions, mode=''):
        assert len(predictions)==len(self), (len(predictions), len(self))
        final_predictions = {
            'answer_prediction_file': self.args.answer_prediction_file,
            'qg_ckpt': self.args.checkpoint,
            'predictions': predictions,
        }
        save_path = os.path.join(self.args.output_dir, "{}{}{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 else "",
            "_aq" if self.args.ambigqa else "",
            mode,
        ))
        with open(save_path, "w") as f:
            json.dump(final_predictions, f, indent=2)
        self.logger.info("Saved prediction in {}".format(save_path))

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        input_ids, attention_mask, _,  = self.tokenized_data
        self.dataset = MySimpleQGDataset(input_ids,
                                         attention_mask,
                                         is_training=self.is_training,)
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
        questions, answers, metadata = [], [], []
        for idx, d in enumerate(self.data):
            curr_prompt_answer_pairs = d["over_generate_{}_prompt_answer".format(self.args.over_generate_pass)]
            curr_questions = [x[0] for x in curr_prompt_answer_pairs]
            curr_answers = [x[1] for x in curr_prompt_answer_pairs]
            metadata.append((len(questions), len(questions)+len(curr_questions)))
            questions.extend(curr_questions)
            answers.extend(curr_answers)

        if self.args.do_lowercase:
            questions = [question.lower() for question in questions]
            answers = [answer.lower() for answer in answers]
        if self.args.append_another_bos:
            questions = ["<s> "+question for question in questions]
            answers = ["<s> " +answer for answer in answers]
        question_input = tokenizer.batch_encode_plus(questions,
                                                     add_special_tokens=False,
                                                     pad_to_max_length=False,
                                                     max_length=32)
        answer_input = tokenizer.batch_encode_plus(answers,
                                                   add_special_tokens=False,
                                                   pad_to_max_length=False,
                                                   max_length=20)
        input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
        decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
        tokenized_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata]

        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    def load_dpr_data(self):
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        if "Bart" in postfix:
            self.load_dpr_data_bart()
        else:
            raise NotImplementedError

    def load_dpr_data_bart(self):
        dpr_predictions_tokenized_input_ids, dpr_predictions_tokenized_attention_mask = self.dpr_reranked_tokenized_data

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        assert len(dpr_predictions_tokenized_input_ids)==len(metadata)
        assert metadata[-1][-1] ==len(input_ids)==len(attention_mask) == len(decoder_attention_mask) == len(decoder_input_ids)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

        new_input_ids, new_attention_mask = [], []

        for idx_m, m in enumerate(tqdm(metadata)):
            dpr_input_ids = dpr_predictions_tokenized_input_ids[idx_m]
            dpr_attention_mask = dpr_predictions_tokenized_attention_mask[idx_m]
            for jdx_qa in range(*m):
                curr_prompt_question_ids = input_ids[jdx_qa]
                curr_answer_ids = decoder_input_ids[jdx_qa]

                if self.args.do_over_generate_qg_noprompt_predict:
                    aq_input_ids = [bos_token_id] + curr_answer_ids + [eos_token_id]
                else:
                    aq_input_ids = [bos_token_id] + curr_answer_ids + [sep_token_id] + curr_prompt_question_ids + [eos_token_id]
                aq_attention_mask = [1 for _ in aq_input_ids]
                new_input_ids_i, new_attention_mask_i = [], []
                for psg_input_id, psg_attention_mask in zip(dpr_input_ids, dpr_attention_mask):
                    end_of_passage = psg_attention_mask.index(0) if 0 in psg_attention_mask else len(psg_attention_mask)
                    psg_input_id, psg_attention_mask = psg_input_id[:end_of_passage], psg_attention_mask[:end_of_passage]
                    assert psg_attention_mask[-1] == 1
                    input_ids_i = aq_input_ids + psg_input_id[1:]
                    attention_mask_i = aq_attention_mask + psg_attention_mask[1:]
                    assert len(input_ids_i) == len(attention_mask_i)
                    max_input_len = 32 + 128
                    if len(input_ids_i) > max_input_len:
                        input_ids_i = input_ids_i[:max_input_len]
                        attention_mask_i = attention_mask_i[:max_input_len]
                        assert len(input_ids_i) == len(attention_mask_i)
                    else:
                        input_ids_i += [pad_token_id for _ in range(max_input_len - len(input_ids_i))]
                        attention_mask_i += [0 for _ in range(max_input_len - len(attention_mask_i))]
                    new_input_ids_i.append(input_ids_i)
                    new_attention_mask_i.append(attention_mask_i)
                new_input_ids.append(new_input_ids_i)
                new_attention_mask.append(new_attention_mask_i)

        self.tokenized_data = [new_input_ids, new_attention_mask, metadata,]

        aq_psgs_input_ids, aq_psgs_attention_mask, _, = self.tokenized_data
        self.tokenized_data[0] = [_aq_psgs_input_ids[:self.args.top_k_passages] for _aq_psgs_input_ids in aq_psgs_input_ids]
        self.tokenized_data[1] = [_aq_psgs_attention_mask[:self.args.top_k_passages] for _aq_psgs_attention_mask in aq_psgs_attention_mask]

        self.logger.info("Finish processing tokenized DPR data")

    def evaluate(self):
        from scipy.stats.stats import pearsonr
        # construct reference and prediction list
        current_overgenerate_pass = self.args.over_generate_pass
        reference = deepcopy(self.data)
        num_qapairs_predicted = []
        num_qapairs_annotated = []
        prediction = {}
        for idx, d in enumerate(self.data):
            curr_noambq_answer_pairs = d["over_generate_{}_noambq_answer".format(current_overgenerate_pass)]
            prediction[d['id']] = [{'question': qa[0], 'answer': qa[1]} for qa in curr_noambq_answer_pairs]
            num_qapairs_predicted.append(len(prediction[d['id']]))
            # get gold number of qapairs per question
            num_qapairs_per_ann = []
            for ann in d['annotations']:
                if ann['type'] == 'singleAnswer':
                    num_qapairs_per_ann.append(1)
                else:
                    num_qapairs_per_ann.append(len(ann['qaPairs']))
            num_qapairs_annotated.append(np.mean(num_qapairs_per_ann))
        print('Predicted {:.2f} qapairs per question, Annotated {:.2f}, correlation {:.3f}'.format(np.mean(num_qapairs_predicted), np.mean(num_qapairs_annotated), pearsonr(num_qapairs_predicted,num_qapairs_annotated)[0]))
        evaluation = QAPairEvaluation(reference, deepcopy(prediction))
        results = evaluation.print_all_metrics(verbose=False)
        results['Avg QAPair'] = np.mean(num_qapairs_predicted)
        results['QAPair Corr'] = pearsonr(num_qapairs_predicted,num_qapairs_annotated)[0]
        print('Ans (all) {:.2f}, Ans (multi) {:.2f}, BLEU {:.2f}, EDIT {:.2f}'.format(
            results['F1 answer'], results['F1 answer (multi)'], results["F1 bleu4"], results["F1 edit-f1"]))
        return results
