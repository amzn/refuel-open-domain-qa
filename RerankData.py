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
from ambigqa_evaluate_script import normalize_answer

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from DataLoader import MySimpleQADataset, MyRerankerDataset, MyDataLoader
from util import decode_span_batch

# for evaluation
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
#from pycocoevalcap.bleu.bleu import Bleu

class NQRerankerData(object):

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
            self.data_type = "train" if is_training or args.dpr else "train_for_inference"
        else:
            logger.info(self.data_path)
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

        if not args.ambigqa or args.leaderboard:
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
        self.labels = None

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
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if self.args.append_another_bos else "",
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
                                                       pad_to_max_length="Bart" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask, metadata]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f, indent=4)
        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    def load_labels(self, dpr_retrieval_path, dpr_retrieval_label_path):
        if os.path.exists(dpr_retrieval_label_path):
            self.logger.info("Loading DPR data labels from {}".format(dpr_retrieval_label_path))
            with open(dpr_retrieval_label_path, "r") as f:
                self.labels = json.load(f)['label']
                return
        else:
            self.logger.info("Process DPR data labels")
            if self.passages.titles is None or self.passages.passages is None:
                self.passages.load_db()
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
            labels = []
            for (pids, qa) in zip(tqdm(dpr_passages), self.data):
                passages = [normalize_answer(self.passages.passages[pid]) for pid in pids]
                answer = [normalize_answer(a) for a in qa['answer']]
                label = [any([a in p for a in answer]) for p in passages]
                labels.append(label)
            self.labels = labels
            with open(dpr_retrieval_label_path, 'wt') as f:
                json.dump({'label': labels}, f)

    def load_dpr_data(self):
        dpr_retrieval_path = os.path.join(self.args.dpr_data_dir, "{}{}_{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type,
            "_aq" if self.args.ambigqa else "",
            self.args.dpr_checkpoint)).replace('train_for_inference', 'train')
        dpr_retrieval_label_path = os.path.join(self.args.reader_data_dir, 'nqopen' if not self.args.wiki_2020 else 'ambigqa', "{}_{}_reranking_labels.json".format(
            self.data_type, self.args.dpr_checkpoint)).replace('train_for_inference', 'train')
        if self.labels is None:
            self.load_labels(dpr_retrieval_path, dpr_retrieval_label_path)
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = os.path.join(self.args.reader_data_dir, self.args.predict_file.split("/")[-2],
                                          "{}_{}_reranking_predictions.json".format(self.data_type, self.args.dpr_checkpoint))
        dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_{}.json".format(postfix))
        if "Bert" in postfix:
            return self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError()

    def load_dpr_data_bert(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
                return
        self.logger.info("Start processing DPR data")
        with open(dpr_retrieval_path, "r") as f:
            dpr_passages = json.load(f)

        if self.is_training:
            if self.args.ambigqa:
                gold_titles = [d["gold_passage_title"] for d in self.data]
            else:
                with open(os.path.join(self.args.reader_data_dir, "nqopen/gold_passages_info/nq-train_gold_info.json"), "r") as f:
                    gold_titles = [d["title"] for d in json.load(f)["data"]]
                    assert len(gold_titles)==len(self)

        input_ids, attention_mask, answer_input_ids, _, metadata = self.tokenized_data
        assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        if self.passages.tokenized_data is None:
            self.passages.load_tokenized_data("albert" if "Albert" in dpr_tokenized_path else "bert", all=True)
        features = defaultdict(list)
        for i, (q_input_ids, q_attention_mask, retrieved) in \
                enumerate(zip(tqdm(input_ids), attention_mask, dpr_passages)):
            assert len(q_input_ids)==len(q_attention_mask)==32
            q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
            assert 3<=len(q_input_ids)<=32
            p_input_ids = [self.passages.tokenized_data["input_ids"][p_idx][1:] + [self.tokenizer.pad_token_id] for p_idx in retrieved]
            p_attention_mask = [self.passages.tokenized_data["attention_mask"][p_idx][1:] + [0] for p_idx in retrieved]
            if self.is_training:
                if self.args.ambigqa:
                    all_gold_title = [normalize_answer(gt) for gt in gold_titles[i]]
                    _positives = [j for j, hasAns in enumerate(self.labels[i]) if hasAns]
                    if len(_positives) == 0:
                        continue
                    positives = [j for j in _positives if normalize_answer(self.decode(p_input_ids[j][:p_input_ids[j].index(self.tokenizer.sep_token_id)])) in all_gold_title]
                else:
                    gold_title = normalize_answer(gold_titles[i])
                    _positives = [j for j, hasAns in enumerate(self.labels[i]) if hasAns]
                    if len(_positives)==0:
                        continue
                    positives = [j for j in _positives if normalize_answer(self.decode(p_input_ids[j][:p_input_ids[j].index(self.tokenizer.sep_token_id)]))==gold_title]
                if len(positives)==0:
                    positives = _positives[:20]
                negatives = [j for j in range(len(self.labels[i])) if j not in positives]
                positives = positives[:20]
                negatives = negatives[:200]
            else:
                positives = [j for j in range(len(self.labels[i]))]
                negatives = []
            for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
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
                assert len(input_ids) == len(attention_mask) == len(token_type_ids) == 160
                return input_ids, attention_mask, token_type_ids

            for idx in positives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["positive_input_ids"][-1].append(input_ids)
                features["positive_input_mask"][-1].append(attention_mask)
                features["positive_token_type_ids"][-1].append(token_type_ids)
            for idx in negatives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["negative_input_ids"][-1].append(input_ids)
                features["negative_input_mask"][-1].append(attention_mask)
                features["negative_token_type_ids"][-1].append(token_type_ids)

        self.tokenized_data = features

        print('Saving', dpr_tokenized_path)
        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f, indent=4)
        print('Done!')

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        self.dataset = MyRerankerDataset(self.tokenized_data,
                                         is_training=self.is_training,
                                         train_MP=self.args.train_MP,
                                         train_MN=self.args.train_MN,
                                         test_M=self.args.test_M)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False, **kwargs):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training, **kwargs)
        if do_return:
            return self.dataloader

    def evaluate(self, outputs):
        assert len(outputs) == len(self.labels)
        recall = defaultdict(list)
        k_list = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for (sel_logits, hasAnswer) in tqdm(zip(outputs, self.labels)):
            reranked_psgs_ids = np.argsort(-np.array(sel_logits)).tolist()
            curr_recall = [hasAnswer[pid] for pid in reranked_psgs_ids]
            for k in k_list:
                recall[k].append(any(curr_recall[:k]))
        scores = []
        for k in k_list:
            if k <= len(outputs[0]):
                self.logger.info("~~~~~~~~~~~~~~~~~~~~~~~Recall @ %d=%.3f" % (k, np.mean(recall[k])))
            if k <= 100:
                scores.append(np.mean(recall[k]))
        return np.mean(scores)

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        save_path = os.path.join(self.args.output_dir, "{}{}{}_dpr-{}_reranking_psg_sel.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 else "",
            "_aq" if self.args.ambigqa else "",
            self.args.dpr_checkpoint,
        ))
        with open(save_path, "w") as f:
            json.dump(predictions, f, indent=4)
        self.logger.info("Saved prediction in {}".format(save_path))

class AQRerankerData(NQRerankerData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AQRerankerData, self).__init__(logger, args, data_path, is_training, passages)

        for i, d in enumerate(self.data):
            answers = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    answers.extend(list(set(annotation["answer"])))
                else:
                    for pair in annotation["qaPairs"]:
                        answers.extend(list(set(pair["answer"])))
            self.data[i]["answer"] = list(set(answers))
            d['nq_doc_title'] = [d['nq_doc_title']] if type(d['nq_doc_title']) == str else d['nq_doc_title']
            self.data[i]["gold_passage_title"] = list(set(d['viewed_doc_titles'] + d['nq_doc_title']))

