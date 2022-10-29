# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import argparse
import logging

import random
import numpy as np
import torch

from run import run, run_over_generate, run_lm_filtering, run_em_filtering, run_over_generate_lm_filtering

import requests, json

############## SM ################
WEBHOOK_URI = 'https://hooks.chime.aws/incomingwebhooks/233ea0a7-b376-4a7c-9159-a2518fc73f8b?token=TU9nbVc4ZTV8MXxqVnBvanQwb293U0JyN3prSzZtZy1DRmlrR09kRVE1SXNIUkRoT2pFM3pr'

def post_message(msg):
  response = None
  try:
    response = requests.post(url=WEBHOOK_URI, json={"Content": msg})
    return json.loads(response.text)
  except:
    return response.text

def notify_success():
  train_config = os.environ['SM_TRAINING_ENV']
  print(train_config)
  train_config = json.loads(train_config)
  message = 'Training ' + train_config['job_name'] + ' is Done'
  print(message)
  req_res = post_message(message)

def notify_failure():
  train_config = os.environ['SM_TRAINING_ENV']
  print(train_config)
  train_config = json.loads(train_config)
  message = 'Training ' + train_config['job_name'] + ' Failed!!!!!!!!!!!!'
  print(message)
  req_res = post_message(message)
############## SM ################

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--task", default="qa", choices=["dpr", "qa", "qg", "rrk", "qg_mask", "qa_gen", "qa_noamb_aq",
                                                         "qg_rewrite", "over_generate", "qg_weighted_loss", "lm_filtering", "em_filtering",
                                                         "qg_noprompt", "over_generate_lm_filtering", "cotraining_label", "cotraining_train"],
                        type=str,
                        help="1) dpr: dense passage retrieval (inference only);"
                             "2) qa: NQ/AQ; "
                             "3) qg: NQG/AQG, NQG: answer + passage -> question, AQG: answer + prompt + passage -> question;"
                             "4) rrk: rerank retrieved passages;"
                             "5) qg_mask: masked pretraining NQG: answer + masked_q + passage -> question;"
                             "6) [inactive] qa_gen: jointly train qa & qg, cicero's idea;"
                             "7) qa_noamb_aq: finetune NQ ckpt on no_amb AQ data;"
                             "8) [inactive] qg_rewrite: rephrase promptQ and keyphrase into disambiguated Q;"
                             "9) over_generate: over generate qa pairs!"
                             "10) qg_weighted_loss: weighted loss for ambigqa question generation"
                             "11): lm_filtering: use lm model to filter / select qa pairs"
                             "12): em_filtering: use exact match as the filtering/selection metric"
                             "13): qg_noprompt: do AQG as: answer + passage -> question"
                             "14): over_generate_lm_filtering, over_generate + lm_filtering"
                             "15): cotraining_label: cotraining to label nqopen training set"
                             "16): cotraining_train: cotraining to train ambigqa on gold and silver data"
                        )
    parser.add_argument("--train_file", default="",
                        type=str)
    parser.add_argument("--predict_file", default="",
                        type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--dpr_data_dir", default="", type=str,
                        help="path where you downloaded DPR related files"
                        "(Wikipedia DB, checkpoints, etc)")
    parser.add_argument("--do_train", type=bool, default=False)
    parser.add_argument("--do_predict", type=bool, default=False)
    # over-generate related
    parser.add_argument("--do_over_generate_predict", type=bool, default=False,
                        help="Combine QA and QG prediction into one pass")
    parser.add_argument("--map_ckpt", type=str, default=None,
                        help="do_over_generate_predict: checkpoint for multiple answer prediction")
    parser.add_argument("--qd_ckpt", type=str, default=None,
                        help="do_over_generate_predict: checkpoint for question disambiguation")
    parser.add_argument("--qd_ckpt_step", type=int, default=None,
                        help="do_over_generate_predict: ckpt step")
    parser.add_argument("--over_generate_pass", default=-1, type=int,
                        help="which time the current generation is? relevant to suffix of saved qa-predictions, "
                             "used when do_over_generate_predict=True")
    parser.add_argument("--replace_prompt_question", default=0, type=int,
                        help="which time the current generation is? relevant to suffix of saved qa-predictions, "
                             "used when do_over_generate_predict=True")
    # verify related
    parser.add_argument("--do_lm_filtering", type=bool, default=False,
                        help="For overgenerate and verify, use lm to reranking and filtering data")
    parser.add_argument("--do_em_filtering", type=bool, default=False,
                        help="For overgenerate and verify, use another model to filtering by exact match answers")

    # over_generate + lm_filtering
    parser.add_argument("--do_over_generate_lm_filtering_predict", type=bool, default=False,
                        help="Combine QA and QG prediction, and LM Filtering into one pass")

    # ablation: qg_noprompt
    parser.add_argument("--do_over_generate_qg_noprompt_predict", type=bool, default=False,
                        help="Combine QA and QG prediction into one pass, and use qg_noprompt for QG")

    parser.add_argument("--skip_db_load", type=bool, default=False)
    parser.add_argument("--skip_inference", type=bool, default=False, help="save all checkpoints. skip inference")
    parser.add_argument("--leaderboard", type=bool, default=False, help="do inference only (for leaderboard submission)")
    parser.add_argument("--leaderboard_threshold", type=float, default=-1, help="LM threshold for filtering QAs (selected from dev set)")
    parser.add_argument("--leaderboard_threshold_mode", default="fixed", choices=["fixed", "global_avg_std"],
                        type=str, help="fixed: predefined threshold, global_avg_std: LM_avg + LM_std")
    parser.add_argument("--db_index", default=-1, type=int)
    # ambigqa related
    parser.add_argument("--ambigqa", type=bool, default=False,
                        help="[For AmbigQA] specify if you are experimenting with AmbigQA")
    parser.add_argument("--wiki_2020", type=bool, default=False,
                        help="[For AmbigQA] Use Wikipedia dump from 02/01/2020"
                        "instead of 12/20/2018")

    ## Model parameters
    parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--checkpoint", type=str,
                        help="Initial checkpoint; when not specified, it will use pretrained BERT/BART models", \
                        default=None)
    parser.add_argument("--do_lowercase", type=bool, default=True)

    # DPR parameters & reranker params
    parser.add_argument('--dpr_retrieval_topk_psgs', type=int, default=1000,
                        help="Use DPR to retrieve K passages")
    parser.add_argument('--dpr_checkpoint', type=str, default='',
                        help="single/multiset")

    # Preprocessing-related parameters
    parser.add_argument('--max_passage_length', type=int, default=200)
    parser.add_argument('--max_question_length', type=int, default=32)
    parser.add_argument('--min_question_length', type=int, default=1)
    parser.add_argument('--max_cat_answer_length', type=int, default=64,
                        help="max concatenated answer length in training for ambigQA")
    parser.add_argument("--max_answer_length", default=10, type=int,
                        help="evaluation: max generated answer length, for both nq and ambigqa")
    parser.add_argument("--min_answer_length", default=1, type=int,
                        help="evaluation: min generated answer length, for both nq and ambigqa")
    parser.add_argument('--train_MP', type=int, default=1,
                        help="# of positive passages / question in BERT reranker")
    parser.add_argument('--train_MN', type=int, default=31,
                        help="# of negative passages / question in BERT reranker")
    parser.add_argument('--test_M', type=int, default=1000,
                        help="# of passages / question in BERT reranker")

    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument("--append_another_bos", type=bool, default=False,
                        help="For SpanSeqGen, append extra BOS token in the"
                        "beginning of the sequence (by default, automatically"
                        "set to `True` when using BART)")
    parser.add_argument("--psg_sel_dir", type=str, default=None,
                        help="For SpanSeqGen, DPR reader path which contains"
                        "passage selection predictions")
    parser.add_argument("--discard_not_found_answers", type=bool, default=False,
                        help="For SpanSeqGen, do not learn to generate answers"
                        "if they are not found in DPR passages"
                        "for FID: discard data point if the answer is not found in"
                         "any retrieved passages")
    parser.add_argument("--filter_not_found_answer_passages", type=bool, default=False,
                        help="For FID Question Generation, filter out irrelevant passages"
                             "which do not contain the answer to ask")
    parser.add_argument("--nq_answer_as_prefix", type=bool, default=False,
                        help="[For AmbigQA] For co-training, use known answer as prefix"
                        "to generate extra answers")
    parser.add_argument('--top_k_passages', type=int, default=50,
                        help="For FusionInDecoder, select topk reranked passages")
    parser.add_argument('--use_reranker', type=bool, default=False,
                        help="For FusionInDecoder, use the reranked passages for training and inference or not")
    parser.add_argument('--decoder_start_token_id', type=int, default=0,
                        help="For FusionInDecoder, not sure use 0 / 2 for bart decoder input id, config file uses 2 but i think it should be 0?")

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)

    ## Evaluation-related parameters
    parser.add_argument("--verbose", type=bool, default=False,
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=400,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default=None,
                        help="Prefix for saving predictions; split name (e.g. `dev` or `test`) if not specified")
    parser.add_argument('--n_paragraphs', type=str, default='10,20,50,100',
                        help="A list of numbers separated by comma, for ablations on number of passages per question (e.g. `20,50,100`)")
    parser.add_argument("--save_psg_sel_only", type=bool, default=False,
                        help="For DPR reader, only save the passage selection predictions without span predictions (mainly for preprocessing for SpanSeqGen)")

    ## Other parameters
    parser.add_argument('--pycharm_debug', type=bool, default=False,
                        help="Use a subset of data for debugging")
    parser.add_argument('--old_data', type=bool, default=False,
                        help="use official ranker processed dataset")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # distributed related, not used
    parser.add_argument('--is_distributed', type=int, default='0', help="0: false (dataparallel), 1: distributed dataparallel")
    parser.add_argument('--use_gpu_ids', type=str, default='0,1,2,3,4,5,6,7,', help="if dataparallel, use this to set gpu ids")
    parser.add_argument('--num_gpus', type=int, default=8, help="if distributed dataparallel, use this to set number of gpus per node")
    parser.add_argument('--hosts', type=list, default=['localhost'])
    parser.add_argument('--current_host', type=str, default='localhost')
    parser.add_argument('--average_gradient', type=int, default='0', help='check do we need to average gradients or not using ddp')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    # unused
    parser.add_argument("--do_e2e_predict", type=bool, default=False,
                        help="After we have QA and QG model, first do answer prediction using do_predict, then use "
                             "this to predict questions taking predicted answers.")
    parser.add_argument("--do_overgenvfy_qa_predict", type=bool, default=False,
                        help="when we have generated the first QA pairs from the ambigqa, we use this command to "
                             "generate answers conditioned on previously generated questions")
    parser.add_argument("--do_overgenvfy_qg_predict", type=bool, default=False,
                        help="we use this option to "
                             "generate questions conditioned on previously generated answers")
    parser.add_argument("--do_qg_edit_rewrite_predict", type=bool, default=False,
                        help="First generate keyphrase, then rewrite it to the question")
    parser.add_argument("--answer_prediction_file", default="", type=str,
                        help="multiple answer predictions from the QA model, used when do_e2e_predict=True")
    parser.add_argument("--qapair_prediction_file", default="", type=str,
                        help="generate more answers from generated questions, used when do_overgenvfy_qa_predict=True")
    parser.add_argument("--overgenvfy_time", default=1,
                        help="which time the current generation is? relevant to suffix of saved qa-predictions, "
                             "used when do_overgenvfy_qa_predict=True")
    parser.add_argument("--ambigqa_editqg", type=bool, default=False,
                        help="[For AmbigQA] replace question target with minimum span")
    parser.add_argument("--lambda_qg_loss_weight", default=1.0, type=float,
                        help="loss weight for qg weighted idea, put more emphasis on inserted words")
    parser.add_argument("--max_n_answers", default=10, type=int,
                        help="used for DPR reader like extraction models ")
    parser.add_argument('--t5_no_intermediate_eos', type=bool, default=False,
                        help="For FusionInDecoder T5 model, remove eos inside the input seqence"
                             "question <eos> title <eos> passage <eos> -> question title passage <eos>")
    parser.add_argument('--topk_answer', type=int, default=1,
                        help="# of top answers per question to save")
    parser.add_argument('--debug', type=bool, default=False,
                        help="Use a subset of data for debugging")
    parser.add_argument('--fp16', type=bool, default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    # QAGen parameters, not used
    parser.add_argument('--max_qagen_catq_length', type=int, default=64,
                        help='task1: max cat question generation length for QAGen model')
    parser.add_argument('--max_qagen_answer_length', type=int, default=12,
                        help='task3: max answer generation length for QAGen model')
    parser.add_argument('--train_batch_size_task_1', type=int, default=16,
                        help='max decoder generation length for QAGen model')
    parser.add_argument('--train_batch_size_task_2_1', type=int, default=64,
                        help='max decoder generation length for QAGen model')
    parser.add_argument('--train_batch_size_task_2_2', type=int, default=64,
                        help='max decoder generation length for QAGen model')
    parser.add_argument('--train_batch_size_task_3_1', type=int, default=64,
                        help='max decoder generation length for QAGen model')
    parser.add_argument('--train_batch_size_task_3_2', type=int, default=64,
                        help='max decoder generation length for QAGen model')
    parser.add_argument("--task_1_gradient_accumulation_steps", default=4, type=int,
                        help="gradient acc for task 1")


    args = parser.parse_args()

    is_sagemaker = 'SM_MODEL_DIR' in os.environ
    args.is_sagemaker = is_sagemaker

    args.dpr = args.task == "dpr"

    if args.is_distributed == 1:
        if is_sagemaker:
            args.hosts = json.loads(os.environ['SM_HOSTS'])
            args.current_host = os.environ['SM_CURRENT_HOST']
            args.num_gpus = os.environ['SM_NUM_GPUS']
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_ids

    if is_sagemaker:
        FusionInDecoderDataReader = os.environ['SM_CHANNEL_DATA']
        FusionInDecoderDataRetriever = ''
        FusionInDecoderOut = os.environ['SM_MODEL_DIR']
        if args.ambigqa or args.task == 'qg_mask' and args.do_predict:
            NQCheckpoint = os.environ.get('SM_CHANNEL_CKPT')
            args.checkpoint = os.path.join(NQCheckpoint, 'best-model.pt') if NQCheckpoint != '' and NQCheckpoint else None
    else:
        if args.pycharm_debug:
            FusionInDecoderDataReader = '/home/ubuntu/data/MyFusionInDecoderDataReaderDebug'
            FusionInDecoderDataRetriever = '/home/ubuntu/data/MyFusionInDecoderDataRetrieverDebug'
            FusionInDecoderOut = '/home/ubuntu/data/MyFusionInDecoderOutDebug'
        else:
            FusionInDecoderDataReader = '/home/ubuntu/data/MyFusionInDecoderDataReader'
            FusionInDecoderDataRetriever = '/home/ubuntu/data/MyFusionInDecoderDataRetriever'
            FusionInDecoderOut = '/home/ubuntu/data/MyFusionInDecoderOut'
        if args.old_data:
            FusionInDecoderDataReader = os.path.join(FusionInDecoderDataReader, "old")
            FusionInDecoderDataRetriever = os.path.join(FusionInDecoderDataRetriever, "old")
        if args.task == 'qa_gen':
            FusionInDecoderDataReader = os.path.join(FusionInDecoderDataReader, "qagen")
        if args.task == 'rrk':
            FusionInDecoderDataReader = os.path.join(FusionInDecoderDataReader, "Reranker_{}".format(args.dpr_checkpoint))

    if args.task not in ["over_generate", "lm_filtering", "em_filtering", "over_generate_lm_filtering"]:
        args.train_file = os.path.join(FusionInDecoderDataReader, args.train_file) if args.train_file != '' and args.train_file else ''
        args.predict_file = os.path.join(FusionInDecoderDataReader, args.predict_file) if args.predict_file != '' and args.predict_file else ''

    args.dpr_data_dir = FusionInDecoderDataRetriever
    args.reader_data_dir = FusionInDecoderDataReader
    print("Reader    Dir:\t{}".format(FusionInDecoderDataReader))
    print("Retriever Dir:\t{}".format(FusionInDecoderDataRetriever))

    if args.task in ['qa', 'qg', 'rrk', 'qg_mask', 'qa_noamb_aq', 'qg_rewrite', "qg_weighted_loss", "qg_noprompt", "cotraining_label", "cotraining_train"]:
        args.output_dir = os.path.join(FusionInDecoderOut, args.output_dir) if args.output_dir != '' and args.output_dir else ''
        args.psg_sel_dir = os.path.join(FusionInDecoderDataReader, 'ambigqa' if args.ambigqa else 'nqopen', 'psg_sel', args.psg_sel_dir) if args.psg_sel_dir != '' and args.psg_sel_dir else ''
        print("PsgSel    Dir:\t{}".format(args.psg_sel_dir))
    elif args.task == 'qa_gen':
        args.nq_psg_sel_dir = os.path.join(FusionInDecoderDataReader, 'nqopen', 'psg_sel', args.psg_sel_dir) if args.psg_sel_dir != '' and args.psg_sel_dir else ''
        args.aq_psg_sel_dir = os.path.join(FusionInDecoderDataReader, 'ambigqa', 'psg_sel', args.psg_sel_dir) if args.psg_sel_dir != '' and args.psg_sel_dir else ''
        args.output_dir = os.path.join(FusionInDecoderOut, args.output_dir) if args.output_dir != '' and args.output_dir else ''
        print("PsgSel NQ Dir:\t{}".format(args.nq_psg_sel_dir))
        print("PsgSel AQ Dir:\t{}".format(args.aq_psg_sel_dir))
    elif args.task in ['over_generate']:
        # args.output_dir = os.path.join(FusionInDecoderOut, "MAP_{}_QD_{}".format(args.map_ckpt, args.qd_ckpt), "replace_prompt_qg" if bool(args.replace_prompt_question) else "original_prompt_qg")
        map_ckpt, qd_ckpt = args.map_ckpt, args.qd_ckpt
        args.map_ckpt = os.path.join(FusionInDecoderOut, map_ckpt, "output", "out", "best-model.pt")
        if args.qd_ckpt_step is not None:
            args.qd_ckpt = os.path.join(FusionInDecoderOut, qd_ckpt, "output", "out", "model-step{}.pt".format(args.qd_ckpt_step))
            args.output_dir = os.path.join(FusionInDecoderOut, qd_ckpt, "result-step{}".format(args.qd_ckpt_step))
        else:
            args.output_dir = os.path.join(FusionInDecoderOut, "MAP_{}_QD_{}".format(map_ckpt, qd_ckpt), )
            args.qd_ckpt = os.path.join(FusionInDecoderOut, qd_ckpt, "output", "out", "best-model.pt")
        args.psg_sel_dir = os.path.join(FusionInDecoderDataReader, 'ambigqa' if args.ambigqa else 'nqopen', 'psg_sel', args.psg_sel_dir) if args.psg_sel_dir != '' and args.psg_sel_dir else ''
        if args.over_generate_pass == 0:
            args.predict_file = os.path.join(FusionInDecoderDataReader, args.predict_file)
        else:
            args.predict_file = os.path.join(args.output_dir, args.predict_file.split('/')[-1])
        print("PsgSel    Dir:\t{}".format(args.psg_sel_dir))
        print("MAP      CKPT:\t{}".format(args.map_ckpt))
        print("QD       CKPT:\t{}".format(args.qd_ckpt))
        print("Infer    FILE:\t{}".format(args.predict_file))
    elif args.task in ['lm_filtering', 'em_filtering']:
        # args.output_dir = os.path.join(FusionInDecoderOut, "MAP_{}_QD_{}".format(args.map_ckpt, args.qd_ckpt), "replace_prompt_qg" if bool(args.replace_prompt_question) else "original_prompt_qg")
        args.output_dir = os.path.join(FusionInDecoderOut, "MAP_{}_QD_{}".format(args.map_ckpt, args.qd_ckpt), "Verifier_{}".format(args.checkpoint))
        args.verifier_ckpt = os.path.join(FusionInDecoderOut, args.checkpoint, "output", "out", "best-model.pt")
        args.psg_sel_dir = os.path.join(FusionInDecoderDataReader, 'ambigqa' if args.ambigqa else 'nqopen', 'psg_sel', args.psg_sel_dir) if args.psg_sel_dir != '' and args.psg_sel_dir else ''
        assert args.over_generate_pass >= 0
        args.predict_file = os.path.join(FusionInDecoderOut, "MAP_{}_QD_{}".format(args.map_ckpt, args.qd_ckpt), args.predict_file.split('/')[-1])
        print("PsgSel    Dir:\t{}".format(args.psg_sel_dir))
        print("Infer    FILE:\t{}".format(args.predict_file))
        print("Verifier CKPT:\t{}".format(args.verifier_ckpt))
    elif args.task in ["over_generate_lm_filtering"]:
        map_ckpt, qd_ckpt = args.map_ckpt, args.qd_ckpt
        args.output_dir = os.path.join(FusionInDecoderOut, "MAP_{}_QD_{}_Verifier_{}".format(map_ckpt, qd_ckpt, args.checkpoint), )
        args.map_ckpt = os.path.join(FusionInDecoderOut, map_ckpt, "output", "out", "best-model.pt")
        args.qd_ckpt = os.path.join(FusionInDecoderOut, qd_ckpt, "output", "out", "best-model.pt")
        args.verifier_ckpt = os.path.join(FusionInDecoderOut, args.checkpoint, "output", "out", "best-model.pt")
        args.psg_sel_dir = os.path.join(FusionInDecoderDataReader, 'ambigqa' if args.ambigqa else 'nqopen', 'psg_sel', args.psg_sel_dir) if args.psg_sel_dir != '' and args.psg_sel_dir else ''
        if args.over_generate_pass == 0:
            args.predict_file = os.path.join(FusionInDecoderDataReader, args.predict_file)
        else:
            args.predict_file = os.path.join(args.output_dir, args.predict_file.split('/')[-1])
        print("PsgSel    Dir:\t{}".format(args.psg_sel_dir))
        print("MAP      CKPT:\t{}".format(args.map_ckpt))
        print("QD       CKPT:\t{}".format(args.qd_ckpt))
        print("Verifier CKPT:\t{}".format(args.verifier_ckpt))
        print("Infer    FILE:\t{}".format(args.predict_file))
    else:
        args.output_dir = FusionInDecoderDataRetriever
    print("Output    Dir:\t{}".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir) and args.output_dir != "":
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs
    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    if args.use_reranker:
        assert args.psg_sel_dir is not None

    if args.do_over_generate_predict:
        assert not bool(args.replace_prompt_question), 'In QG task, replacing prompt with generated q has worse performance!'
        assert args.map_ckpt is not None
        assert args.qd_ckpt is not None
        assert args.over_generate_pass >= 0
        assert os.path.exists(args.map_ckpt)
        assert os.path.exists(args.qd_ckpt)

    if args.do_lm_filtering or args.do_em_filtering:
        assert not bool(args.replace_prompt_question), 'In QG task, replacing prompt with generated q has worse performance!'
        assert args.map_ckpt is not None
        assert args.qd_ckpt is not None
        assert args.over_generate_pass >= 0
        assert os.path.exists(args.output_dir)
        assert os.path.exists(args.predict_file)
        assert args.checkpoint is not None
        assert os.path.exists(args.verifier_ckpt)

    if args.do_over_generate_lm_filtering_predict:
        assert not bool(args.replace_prompt_question), 'In QG task, replacing prompt with generated q has worse performance!'
        assert args.map_ckpt is not None
        assert args.qd_ckpt is not None
        assert args.over_generate_pass >= 0
        assert os.path.exists(args.map_ckpt)
        assert os.path.exists(args.qd_ckpt)
        assert os.path.exists(args.output_dir)
        assert os.path.exists(args.predict_file)
        assert args.checkpoint is not None
        assert os.path.exists(args.verifier_ckpt)

    logger.info("Using {} gpus".format(args.n_gpu))

    if args.bert_name.startswith("bart") or args.bert_name.startswith("t5"):
        args.is_seq2seq = True
    elif args.bert_name.startswith("bert") or args.bert_name.startswith("roberta") or args.bert_name.startswith("albert"):
        args.is_seq2seq = False
    else:
        raise NotImplementedError("Pretrained model not recognized: {}".format(args.bert_name))

    if args.task == 'over_generate':
        run_over_generate(args, logger)
    elif args.task == 'lm_filtering':
        run_lm_filtering(args, logger)
    elif args.task == 'em_filtering':
        run_em_filtering(args, logger)
    elif args.task == "over_generate_lm_filtering":
        run_over_generate_lm_filtering(args, logger)
    else:
        run(args, logger)


if __name__=='__main__':
    if 'SM_MODEL_DIR' in os.environ:
        import traceback

        try:
            main()
            notify_success()
        except Exception as e:
            track = traceback.format_exc()
            print(track)
            notify_failure()
    else:
        main()
