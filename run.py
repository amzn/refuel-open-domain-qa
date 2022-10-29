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
import numpy as np
import json
import torch
import torch.distributed as dist

from tqdm import tqdm
from collections import defaultdict
from transformers import BartTokenizer, AlbertTokenizer, BertTokenizer, T5Tokenizer
from transformers import BartConfig, AlbertConfig, BertConfig, T5Config
from transformers import AdamW, get_linear_schedule_with_warmup

from QAData import QAData, AmbigQAData, DisAmbigQAData, AmbigQADataLeaderboard, AmbigQACoTrainingLabelData, AmbigQACoTrainingTrainData
from QGData import QGData, AmbigQGData, QGMaskedData, AmbigQGRewriteData, AmbigQGWeightedData, AmbigQGNoPromptData
from QGInferenceData import AmbigQGInferenceData
from QAInferenceData import AmbigQAInferenceData
from QALMFilteringData import AmbigQALMFilteringData
from QAEMFilteringData import AmbigQAEMFilteringData
from PassageData import PassageData
from RerankData import NQRerankerData, AQRerankerData
from models.span_predictor import SpanPredictor, AlbertSpanPredictor
from models.reranker import BertReranker
from models.seq2seq import MyBart, MyT5, MyBartS2S, MyBartDynamic, MyBartDynamicWeightedLoss, MyBartWeightedLoss
from models.seq2seq_with_prefix import MyBartWithPrefix
from models.lm_filtering import MyBartLMFiltering
from models.biencoder import MyBiEncoder
from models.qagen import MyBart_QAGen
from QAGenData import QAGenPassageData, QAGenData
from ambigqa_evaluate_script import get_exact_match
# from IPython import embed

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def run(args, logger):
    if args.is_distributed == 1:
        logger.debug("Distributed training - {}".format(bool(args.is_distributed)))
        logger.debug("Number of gpus available - {}".format(args.num_gpus))
        kwargs = {'num_workers': 1, 'pin_memory': True}
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        # dist.init_process_group(backend='nccl')
        dist.init_process_group(backend='NCCL', rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            'nccl', dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))
    else:
        kwargs = {}

    # args.dpr = args.task=="dpr"
    # args.is_seq2seq = 'bart' in args.bert_name
    if 'bart' in args.bert_name:
        tokenizer = BartTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        if args.do_predict and args.nq_answer_as_prefix:
            Model = MyBartWithPrefix
        elif args.task in ['qg'] and args.filter_not_found_answer_passages:
            Model = MyBartDynamic
        elif args.task in ["qg_weighted_loss"]:
            if args.filter_not_found_answer_passages:
                Model = MyBartDynamicWeightedLoss
            else:
                Model = MyBartWeightedLoss
        else:
            Model = MyBart
        Config = BartConfig
        if args.task == 'qa_gen':
            tokenizer.add_tokens(["<QAGEN-Q>"])
            tokenizer.add_tokens(["<QAGEN-A>"])
        # args.append_another_bos = True
    elif 'albert' in args.bert_name:
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_name)
        Model = AlbertSpanPredictor
        Config = AlbertConfig
    elif 'bert' in args.bert_name:
        tokenizer = BertTokenizer.from_pretrained(args.bert_name)
        Model = MyBiEncoder if args.dpr else BertReranker
        Config = BertConfig
    elif 't5' in args.bert_name:
        logger.info('Usage: https://github.com/huggingface/transformers/issues/4092')
        # https://huggingface.co/transformers/model_doc/t5.html#training
        #
        # input_ids = tokenizer.encode('translate English to German: The house is wonderful. </s>', return_tensors='pt')
        # labels = tokenizer.encode('Das Haus ist wunderbar. </s>', return_tensors='pt')
        # # the forward function automatically creates the correct decoder_input_ids
        # model(input_ids=input_ids, labels=labels)
        #
        # Gotcha for me was that the decoder_input_ids at inference should be prepended by the padding token as stated in the docs for T5ForConditionalGeneration.
        # During training, there is no need to prepend the padding token since this is done automatically in T5 when lm_labels is provided.
        # During evaluation, one has to prepend the PAD token as you stated in your example.
        # After training, the mode can be used with the generate() method (which actually powers the summarization, translation and text-generation pipeline).
        # In the generate() method, the padding token is automatically prepended.
        # from transformers import T5Tokenizer, T5Model
        # tokenizer = T5Tokenizer.from_pretrained('t5-small')
        # model = T5Model.from_pretrained('t5-small')
        # input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")
        # decoder_input_ids = tokenizer.encode("<pad>", return_tensors="pt")
        # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # outputs[0]
        # Do note that T5ForConditionalGeneration already prepends the padding by default. Above is only necessary if you're doing a forward pass straight from T5Model.
        #
        # you should add the </s> token to the end of a sentence.
        tokenizer = T5Tokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        Model = MyT5
        Config = T5Config
        # args.append_another_bos = True
    else:
        raise NotImplementedError()

    if args.dpr:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        Model = MyBiEncoder
        args.checkpoint = os.path.join(args.dpr_data_dir, "checkpoint/retriever/{}/bert-base-encoder.cp".format(args.dpr_checkpoint))
        assert not args.do_train, "Training DPR is not supported yet"

    if args.task == 'qa_gen':
        passages = QAGenPassageData(logger, args, tokenizer)
    else:
        passages = PassageData(logger, args, tokenizer)

    def _getData():
        if args.task == 'rrk':
            return AQRerankerData if args.ambigqa and not args.leaderboard else NQRerankerData
        elif args.task == 'qa_noamb_aq':
            return DisAmbigQAData
        elif args.task == "qg_weighted_loss":
            assert args.ambigqa == True, 'Not support NQG pretraining!'
            return AmbigQGWeightedData
        elif args.task == 'qg_noprompt':
            return AmbigQGNoPromptData
        elif args.task == "qg":
            return AmbigQGData if args.ambigqa else QGData
        elif args.task == 'qg_mask':
            return QGMaskedData
        elif args.task == 'cotraining_label':
            return AmbigQACoTrainingLabelData
        elif args.task == 'cotraining_train':
            return AmbigQACoTrainingTrainData
        else:
            if args.leaderboard:
                # for the use of dpr
                return AmbigQADataLeaderboard
            else:
                return AmbigQAData if args.ambigqa else QAData

    def _load_from_checkpoint(checkpoint):
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(Config.from_pretrained(args.bert_name))
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("Loading from {}".format(checkpoint))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)

    dev_data = _getData()(logger, args, args.predict_file, False, passages)
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_train:
        train_data = _getData()(logger, args, args.train_file, True, passages)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader(**kwargs)

        if args.checkpoint is not None:
            model = _load_from_checkpoint(args.checkpoint)
        else:
            model = Model.from_pretrained(args.bert_name)
        if "bart" in args.bert_name:
            # see https://github.com/pytorch/fairseq/issues/1389
            model.decoder_start_token_id = args.decoder_start_token_id
            model.resize_token_embeddings(len(tokenizer))

        if args.is_distributed == 1:
            logger.info('Use DDP!')
            model.to(torch.device("cuda"))
            model = torch.nn.parallel.DistributedDataParallel(model)
            # model.to(torch.device("cuda", args.local_rank))
            # model = torch.nn.parallel.DistributedDataParallel(model,
            #                                                   device_ids=[args.local_rank],
            #                                                   output_device=args.local_rank)
        else:
            if args.n_gpu>1:
                logger.info('Use DP!')
                model = torch.nn.DataParallel(model)
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.task == 'qa_gen':
            num_training_steps = args.num_train_epochs * min(
                [int(len(train_data.dataset_task_1) / args.train_batch_size_task_1),
                 int(len(train_data.dataset_task_2_1) / args.train_batch_size_task_2_1),
                 int(len(train_data.dataset_task_2_2) / args.train_batch_size_task_2_2),
                 int(len(train_data.dataset_task_3_1) / args.train_batch_size_task_3_1),
                 int(len(train_data.dataset_task_3_2) / args.train_batch_size_task_3_2), ])
            args.warmup_steps = int(num_training_steps * args.warmup_proportion)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=num_training_steps)
            train_qa_gen(args, logger, model, train_data, dev_data, optimizer, scheduler)
        else:
            num_training_steps = args.num_train_epochs * int(len(train_data) / args.train_batch_size)
            args.warmup_steps = int(num_training_steps * args.warmup_proportion)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=num_training_steps)
            if args.task == 'rrk':
                train_reranker(args, logger, model, train_data, dev_data, optimizer, scheduler)
            else:
                train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        model = _load_from_checkpoint(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if "bart" in args.bert_name:
            model.decoder_start_token_id = args.decoder_start_token_id
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        if args.n_gpu>1 and 'bert' in args.bert_name:
            model = torch.nn.DataParallel(model)
        model.to(torch.device("cuda"))
        model.eval()
        ems, result = inference(model, dev_data, save_predictions=False, logger=logger)
        logger.info("%s on test data = %.2f" % (dev_data.metric, ems))
        if dev_data.args.task not in ['dpr', 'rrk', 'cotraining_label']:
            with open(os.path.join(args.output_dir, "{}{}-best-result-{}.json".format(args.task, "-aq" if args.ambigqa else "", dev_data.data_type)), 'w') as f:
                json.dump(result, f, indent=4)


def run_over_generate(args, logger):
    def _load_from_checkpoint(checkpoint, Model):
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(Config.from_pretrained(args.bert_name))
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)

    if 'bart' in args.bert_name:
        tokenizer = BartTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        MAP_Model = MyBart
        QD_Model = MyBart
        Config = BartConfig
    else:
        raise NotImplementedError()

    passages = PassageData(logger, args, tokenizer)

    map_dev_data = AmbigQAInferenceData(logger, args, args.predict_file, False, passages)
    map_dev_data.load_dataset(tokenizer)
    map_dev_data.load_dataloader()

    map_prediction_file = '/' + os.path.join(*args.map_ckpt.split('/')[:-3], '{}.json'.format(map_dev_data.data_type))
    if os.path.exists(map_prediction_file) and args.over_generate_pass == 0 and not args.leaderboard:
        with open(map_prediction_file) as f:
            map_dev_data.data = json.load(f)
        logger.info('Loading MAP Prediction File from {}!'.format(map_prediction_file))
    else:
        map_model = _load_from_checkpoint(args.map_ckpt, MAP_Model)
        logger.info("Loading map checkpoint from {}".format(args.map_ckpt))
        if "bart" in args.bert_name:
            map_model.decoder_start_token_id = args.decoder_start_token_id
            map_model.resize_token_embeddings(len(tokenizer))
        map_model.to(torch.device("cuda"))
        map_model.eval()
        map_predictions_metadata = map_dev_data.tokenized_data[-1]
        inference_overgenerate('qa', map_model, map_dev_data, map_predictions_metadata)
        del map_model; torch.cuda.empty_cache()
        if args.over_generate_pass == 0 and not args.leaderboard:
            # save map data
            with open(map_prediction_file, 'w') as f:
                json.dump(map_dev_data.data, f)
            logger.info('Saving MAP Prediction File to {}!'.format(map_prediction_file))

    # 2. Question Disambiguation
    # reduce the batch size
    args.predict_batch_size = int(args.predict_batch_size/4)
    qd_dev_data = AmbigQGInferenceData(logger, args, args.predict_file, False, passages, map_dev_data.data, map_dev_data.dpr_reranked_tokenized_data)
    qd_dev_data.load_dataset(tokenizer)
    qd_dev_data.load_dataloader()

    qd_model = _load_from_checkpoint(args.qd_ckpt, QD_Model)
    logger.info("Loading qd checkpoint from {}".format(args.qd_ckpt))
    if "bart" in args.bert_name:
        qd_model.decoder_start_token_id = args.decoder_start_token_id
        qd_model.resize_token_embeddings(len(tokenizer))
    qd_model.to(torch.device("cuda"))
    qd_model.eval()
    qd_predictions_metadata = qd_dev_data.tokenized_data[2]
    inference_overgenerate('qg', qd_model, qd_dev_data, qd_predictions_metadata)

    # evaluate un-filtered data
    if not args.leaderboard:
        results = qd_dev_data.evaluate()
        with open(os.path.join(args.output_dir, "{}.json".format(qd_dev_data.data_type)), 'w') as f:
            json.dump(qd_dev_data.data, f)
        logger.info('Saving predictions to \n{}'.format(os.path.join(args.output_dir, "{}.json".format(qd_dev_data.data_type))))
        with open(os.path.join(args.output_dir, "{}_over_generate_pass_{}_result.json".format(qd_dev_data.data_type, args.over_generate_pass)), 'w') as f:
            json.dump(results, f, indent=4)
        logger.info('Saving results to \n{}'.format(os.path.join(args.output_dir, "{}_over_generate_pass_{}_result.json".format(qd_dev_data.data_type, args.over_generate_pass))))
    else:
        if args.over_generate_pass == 0:
            # save pass 0 predictions
            e2e_predictions = {}
            for d in qd_dev_data.data:
                e2e_predictions[d['id']] = [{"question": x[0], "answer": x[1]} for x in d['over_generate_0_noambq_answer']]
            with open(os.path.join(args.output_dir, "{}_e2e_leaderboard.json".format(qd_dev_data.data_type)), 'w') as f:
                json.dump(e2e_predictions, f, indent=2)
            logger.info('Saving e2e predictions (pass=0) to \n{}'.format(os.path.join(args.output_dir, "{}_e2e_leaderboard.json".format(qd_dev_data.data_type))))
        with open(os.path.join(args.output_dir, "{}.json".format(qd_dev_data.data_type)), 'w') as f:
            json.dump(qd_dev_data.data, f)
        logger.info('Saving all predictions to \n{}'.format(os.path.join(args.output_dir, "{}.json".format(qd_dev_data.data_type))))


def run_lm_filtering(args, logger):
    def _load_from_checkpoint(checkpoint, Model):
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(BartConfig.from_pretrained(args.bert_name))
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)

    assert 'bart' in args.bert_name, 'only support bart!'
    tokenizer = BartTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_tokens(["<SEP>"])
    passages = PassageData(logger, args, tokenizer)
    dev_data = AmbigQALMFilteringData(logger, args, args.predict_file, False, passages)
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    verifier = _load_from_checkpoint(args.verifier_ckpt, MyBartLMFiltering)
    logger.info("Loading checkpoint from {}".format(args.verifier_ckpt))
    verifier.decoder_start_token_id = args.decoder_start_token_id
    verifier.resize_token_embeddings(len(tokenizer))
    verifier.to(torch.device("cuda"))
    verifier.eval()
    lm_scores = inference_lm_filtering(verifier, dev_data, logger=logger)
    del verifier; torch.cuda.empty_cache()
    if not args.leaderboard:
        # evaluate un-filtered data
        predictions, results = dev_data.evaluate(lm_scores)
        with open(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction.json".format(dev_data.data_type, args.over_generate_pass)), 'w') as f:
            json.dump(predictions, f)
        logger.info('Saving predictions to \n{}'.format(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction.json".format(dev_data.data_type, args.over_generate_pass))))
        with open(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_result.json".format(dev_data.data_type, args.over_generate_pass)), 'w') as f:
            json.dump(results, f, indent=4)
        logger.info('Saving results to \n{}'.format(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_result.json".format(dev_data.data_type, args.over_generate_pass))))
    else:
        predictions = dev_data.predict(lm_scores, dev_data.args.leaderboard_threshold, dev_data.args.leaderboard_threshold_mode)
        path_predictions = os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction_leaderboard_{}.json".format(
            dev_data.data_type, args.over_generate_pass, "{}{}".format(
                dev_data.args.leaderboard_threshold_mode, dev_data.args.leaderboard_threshold if dev_data.args.leaderboard_threshold_mode == 'fixed' else "")))
        with open(path_predictions, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info('Saving predictions to \n{}'.format(path_predictions))
        #
        # path_global_threshold_avg_std = os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction_leaderboard_global_avg_std.json".format(dev_data.data_type, args.over_generate_pass))
        # with open(path_global_threshold_avg_std, 'w') as f:
        #     json.dump(predictions_avg_std, f, indent=2)
        # logger.info('Saving global avg std predictions to \n{}'.format(path_global_threshold_avg_std))
        #
        #
        # path_global_threshold_predefined = os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction_leaderboard_global_threshold_{:.2f}.json".format(dev_data.data_type, args.over_generate_pass, dev_data.args.leaderboard_threshold))
        # with open(path_global_threshold_predefined, 'w') as f:
        #     json.dump(predictions_threshold, f, indent=2)
        # logger.info('Saving global threshold predictions to \n{}'.format(path_global_threshold_predefined))


def run_em_filtering(args, logger):
    def _load_from_checkpoint(checkpoint, Model):
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]

            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key

            return {_convert(key): value for key, value in state_dict.items()}

        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(BartConfig.from_pretrained(args.bert_name))
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)

    assert 'bart' in args.bert_name, 'only support bart!'
    tokenizer = BartTokenizer.from_pretrained(args.bert_name)
    tokenizer.add_tokens(["<SEP>"])
    passages = PassageData(logger, args, tokenizer)
    dev_data = AmbigQAEMFilteringData(logger, args, args.predict_file, False, passages)
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    verifier = _load_from_checkpoint(args.verifier_ckpt, MyBart)
    logger.info("Loading checkpoint from {}".format(args.verifier_ckpt))
    verifier.decoder_start_token_id = args.decoder_start_token_id
    verifier.resize_token_embeddings(len(tokenizer))
    verifier.to(torch.device("cuda"))
    verifier.eval()
    em_predictions = inference_em_filtering(verifier, dev_data, logger=logger)

    # evaluate un-filtered data
    dev_data.evaluate(em_predictions)


def run_over_generate_lm_filtering(args, logger):
    def _load_from_checkpoint(checkpoint, Model):
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(Config.from_pretrained(args.bert_name))
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)

    if 'bart' in args.bert_name:
        tokenizer = BartTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        MAP_Model = MyBart
        QD_Model = MyBart
        Config = BartConfig
    else:
        raise NotImplementedError()

    passages = PassageData(logger, args, tokenizer)

    map_dev_data = AmbigQAInferenceData(logger, args, args.predict_file, False, passages)
    map_dev_data.load_dataset(tokenizer)
    map_dev_data.load_dataloader()

    map_prediction_file = '/' + os.path.join(*args.map_ckpt.split('/')[:-3], '{}.json'.format(map_dev_data.data_type))
    if os.path.exists(map_prediction_file) and args.over_generate_pass == 0 and not args.leaderboard:
        with open(map_prediction_file) as f:
            map_dev_data.data = json.load(f)
        logger.info('Loading MAP Prediction File from {}!'.format(map_prediction_file))
    else:
        map_model = _load_from_checkpoint(args.map_ckpt, MAP_Model)
        logger.info("Loading map checkpoint from {}".format(args.map_ckpt))
        if "bart" in args.bert_name:
            map_model.decoder_start_token_id = args.decoder_start_token_id
            map_model.resize_token_embeddings(len(tokenizer))
        map_model.to(torch.device("cuda"))
        map_model.eval()
        map_predictions_metadata = map_dev_data.tokenized_data[-1]
        inference_overgenerate('qa', map_model, map_dev_data, map_predictions_metadata)
        del map_model; torch.cuda.empty_cache()
        if args.over_generate_pass == 0 and not args.leaderboard:
            # save map data
            with open(map_prediction_file, 'w') as f:
                json.dump(map_dev_data.data, f)
            logger.info('Saving MAP Prediction File to {}!'.format(map_prediction_file))

    # 2. Question Disambiguation
    # reduce the batch size
    args.predict_batch_size = int(args.predict_batch_size/4)
    qd_dev_data = AmbigQGInferenceData(logger, args, args.predict_file, False, passages, map_dev_data.data, map_dev_data.dpr_reranked_tokenized_data)
    qd_dev_data.load_dataset(tokenizer)
    qd_dev_data.load_dataloader()

    qd_model = _load_from_checkpoint(args.qd_ckpt, QD_Model)
    logger.info("Loading qd checkpoint from {}".format(args.qd_ckpt))
    if "bart" in args.bert_name:
        qd_model.decoder_start_token_id = args.decoder_start_token_id
        qd_model.resize_token_embeddings(len(tokenizer))
    qd_model.to(torch.device("cuda"))
    qd_model.eval()
    qd_predictions_metadata = qd_dev_data.tokenized_data[2]
    inference_overgenerate('qg', qd_model, qd_dev_data, qd_predictions_metadata)
    del qd_model; torch.cuda.empty_cache()

    # 3. LM Filtering
    args.predict_batch_size = args.predict_batch_size * 4
    verifier_data = AmbigQALMFilteringData(logger, args, args.predict_file, False, passages, over_generate_data=qd_dev_data.data)
    verifier_data.load_dataset(tokenizer)
    verifier_data.load_dataloader()

    verifier = _load_from_checkpoint(args.verifier_ckpt, MyBartLMFiltering)
    logger.info("Loading checkpoint from {}".format(args.verifier_ckpt))
    verifier.decoder_start_token_id = args.decoder_start_token_id
    verifier.resize_token_embeddings(len(tokenizer))
    verifier.to(torch.device("cuda"))
    verifier.eval()
    lm_scores = inference_lm_filtering(verifier, verifier_data, logger=logger)
    del verifier; torch.cuda.empty_cache()

    # evaluate un-filtered data
    predictions, results = verifier_data.evaluate(lm_scores)
    # replace the original predictions, and save
    for ex in verifier_data.data:
        filtered_qapairs = predictions['th_best'][ex['id']]
        ex['over_generate_{}_noambq_answer'.format(args.over_generate_pass)] = [(qapair['question'], qapair['answer']) for qapair in filtered_qapairs]

    with open(os.path.join(args.output_dir, "{}.json".format(verifier_data.data_type)), 'w') as f:
        json.dump(verifier_data.data, f)
    logger.info('Saving data file to \n{}'.format(os.path.join(args.output_dir, "{}.json".format(verifier_data.data_type))))
    with open(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction.json".format(verifier_data.data_type, args.over_generate_pass)), 'w') as f:
        json.dump(predictions, f)
    logger.info('Saving predictions to \n{}'.format(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_prediction.json".format(verifier_data.data_type, args.over_generate_pass))))
    with open(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_result.json".format(verifier_data.data_type, args.over_generate_pass)), 'w') as f:
        json.dump(results, f, indent=4)
    logger.info('Saving results to \n{}'.format(os.path.join(args.output_dir, "{}_over_generate_pass_{}_lm_filtered_result.json".format(verifier_data.data_type, args.over_generate_pass))))


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    train_preds = []
    best_accuracy = -1
    stop_training=False
    wait_step = 0

    logger.info("Start training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            # debug code
            if global_step == 0:
                logger.info('Batch size {}'.format(len(batch[0])))

            global_step += 1

            # total number of predictions
            all_decoder_attention_mask = batch[3]
            if 't5' in args.bert_name:
                num_preds = torch.sum(torch.ne(all_decoder_attention_mask, 0))
            elif 'bart' in args.bert_name:
                num_preds = torch.sum(torch.ne(all_decoder_attention_mask[..., 1:], 0))
            else:
                raise NotImplementedError

            # actual batchsize for grad step
            batch_losses = []
            actual_train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

            for grad_step in range(0, len(batch[0]), actual_train_batch_size):
                actual_batch = [b[grad_step: grad_step + actual_train_batch_size].to(torch.device("cuda")) for b in batch]

                assert args.is_seq2seq is True
                if 't5' in args.bert_name:
                    lm_labels = actual_batch[2]
                    lm_labels[lm_labels == 0] = -100
                    loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                 lm_labels=lm_labels, decoder_attention_mask=actual_batch[3],
                                 is_training=True)
                elif 'bart' in args.bert_name:
                    if args.filter_not_found_answer_passages:
                        if args.task == "qg_weighted_loss":
                            loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                         decoder_input_ids=actual_batch[2], decoder_attention_mask=actual_batch[3],
                                         is_training=True, is_discard=actual_batch[4], weighted_positions=actual_batch[5],
                                         insert_loss_weight=args.lambda_qg_loss_weight)
                        else:
                            loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                         decoder_input_ids=actual_batch[2], decoder_attention_mask=actual_batch[3],
                                         is_training=True, is_discard=actual_batch[4])
                    else:
                        if args.task == "qg_weighted_loss":
                            loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                         decoder_input_ids=actual_batch[2], decoder_attention_mask=actual_batch[3],
                                         is_training=True, weighted_positions=actual_batch[4],
                                         insert_loss_weight=args.lambda_qg_loss_weight)
                        else:
                            loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                         decoder_input_ids=actual_batch[2], decoder_attention_mask=actual_batch[3],
                                         is_training=True)
                else:
                    raise NotImplementedError

                # if we average over all gpus, then the model will be inclined to generate shorter answers
                loss = torch.sum(loss) / num_preds
                if torch.isnan(loss).data:
                    logger.info("Stop training because loss=%s" % (loss.data))
                    stop_training = True
                    break
                batch_losses.append(loss.detach().cpu().item())
                loss.backward()
                # TODO, check if needed
                if args.average_gradient == 1:
                    _average_gradients(model)

            train_losses.append(sum(batch_losses) * num_preds.item())
            train_preds.append(num_preds.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()  # We have accumulated enough gradients
            scheduler.step()
            model.zero_grad()

            if global_step % args.eval_period == 0:
                if args.skip_inference:
                    avg_train_loss = sum(train_losses) / sum(train_preds)
                    logger.info("Epoch=%d, Global-step=%d, Train-loss=%.2f" % (
                        epoch,
                        global_step,
                        avg_train_loss,
                    ))
                    model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "model-step{}.pt".format(global_step)))
                    train_losses = []
                    train_preds = []
                    wait_step = 0
                    stop_training = False
                    logger.info("=" * 20 + '\n')
                    model.train()
                else:
                    model.eval()
                    curr_em = inference(model, dev_data, save_predictions=False, logger=logger)
                    if type(curr_em) == tuple:
                        curr_em, curr_results = curr_em
                    else:
                        curr_results = None
                    avg_train_loss = sum(train_losses) / sum(train_preds)
                    logger.info("Epoch=%d, Global-step=%d, Train-loss=%.2f, %s=%.2f%%" % (
                        epoch,
                        global_step,
                        avg_train_loss,
                        dev_data.metric,
                        curr_em,
                    ))
                    train_losses = []
                    train_preds = []
                    if best_accuracy < curr_em:
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        if curr_results:
                            curr_results['epoch'] = epoch
                            curr_results['global_step'] = global_step
                            curr_results['train_loss'] = avg_train_loss
                            with open(os.path.join(args.output_dir, "best-result.json"), 'w') as f:
                                json.dump(curr_results, f, indent=4)
                        logger.info("New best %s: %.2f%% -> %.2f%%" % (dev_data.metric, best_accuracy, curr_em,))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                    logger.info("=" * 20 + '\n')
                    model.train()
        if stop_training:
            break


def train_qa_gen(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = [[], [], []]  # task 1,2,3
    train_preds = [[], [], []]  # task 1,2,3
    best_accuracy = -1
    stop_training=False
    wait_step = 0

    logger.info("Start training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch_task_1, batch_task_2_1, batch_task_2_2, batch_task_3_1, batch_task_3_2 in \
                zip(train_data.dataloader_task_1, train_data.dataloader_task_2_1, train_data.dataloader_task_2_2,
                    train_data.dataloader_task_3_1, train_data.dataloader_task_3_2):
            global_step += 1

            # total number of predictions
            all_decoder_attention_mask_task_1 = batch_task_1[3]
            all_decoder_attention_mask_task_2_1 = batch_task_2_1[3]
            all_decoder_attention_mask_task_3_1 = batch_task_3_1[3]
            all_decoder_attention_mask_task_3_2 = batch_task_3_2[3]
            if 'bart' in args.bert_name:
                num_preds_task_1 = torch.sum(torch.ne(all_decoder_attention_mask_task_1[..., 1:], 0))
                num_preds_task_2_1 = torch.sum(torch.ne(all_decoder_attention_mask_task_2_1[..., 1:], 0))
                num_preds_task_3_1 = torch.sum(torch.ne(all_decoder_attention_mask_task_3_1[..., 1:], 0))
                num_preds_task_3_2 = torch.sum(torch.ne(all_decoder_attention_mask_task_3_2[..., 1:], 0))
            else:
                raise NotImplementedError

            # actual batchsize for grad step
            batch_losses = [[], [], [],]  # batch 1,2,3
            actual_train_batch_size_task_1 = int(args.train_batch_size_task_1 / args.task_1_gradient_accumulation_steps)
            actual_train_batch_size_task_2_1 = int(args.train_batch_size_task_2_1 / args.gradient_accumulation_steps)
            actual_train_batch_size_task_3_1 = int(args.train_batch_size_task_3_1 / args.gradient_accumulation_steps)
            actual_train_batch_size_task_3_2 = int(args.train_batch_size_task_3_2 / args.gradient_accumulation_steps)

            for task_name in ['task_1', 'task_2', 'task_3']:
                if task_name == 'task_1':
                    for grad_step in range(args.task_1_gradient_accumulation_steps):
                        decoder_start_token_id = train_data.tokenizer.convert_tokens_to_ids(train_data.QBOS)
                        actual_batch = [b[grad_step * actual_train_batch_size_task_1: (grad_step+1) * actual_train_batch_size_task_1].to(torch.device("cuda")) for b in batch_task_1]

                        loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                     decoder_input_ids=actual_batch[2], decoder_attention_mask=actual_batch[3],
                                     is_training=True, decoder_start_token_id=decoder_start_token_id)

                        loss = torch.sum(loss) / num_preds_task_1
                        batch_losses[0].append(loss.detach().cpu().item())

                        if torch.isnan(loss).data:
                            logger.info("Stop training because loss=%s" % (loss.data))
                            stop_training = True
                            break
                        loss.backward()
                else:
                    for grad_step in range(args.gradient_accumulation_steps):
                        if task_name == 'task_2':
                            decoder_start_token_id = train_data.tokenizer.convert_tokens_to_ids(train_data.QBOS)
                            actual_batch = [b[grad_step * actual_train_batch_size_task_2_1: (grad_step + 1) * actual_train_batch_size_task_2_1].to(torch.device("cuda")) for b in batch_task_2_1]
                        elif task_name == 'task_3':
                            decoder_start_token_id = train_data.tokenizer.convert_tokens_to_ids(train_data.ABOS)
                            actual_batch_task_3_1 = [b[grad_step * actual_train_batch_size_task_3_1: (grad_step + 1) * actual_train_batch_size_task_3_1].to(torch.device("cuda")) for b in batch_task_3_1]
                            actual_batch_task_3_2 = [b[grad_step * actual_train_batch_size_task_3_2: (grad_step + 1) * actual_train_batch_size_task_3_2].to(torch.device("cuda")) for b in batch_task_3_2]
                            actual_batch = [torch.cat([b1,b2], dim=0) for b1, b2 in zip(actual_batch_task_3_1, actual_batch_task_3_2)]
                        else:
                            raise NotImplementedError

                        loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1],
                                     decoder_input_ids=actual_batch[2], decoder_attention_mask=actual_batch[3],
                                     is_training=True, decoder_start_token_id=decoder_start_token_id)

                        # if we average over all gpus, then the model will be inclined to generate shorter answers
                        if task_name == 'task_2':
                            loss = torch.sum(loss) / num_preds_task_2_1
                            batch_losses[1].append(loss.detach().cpu().item())
                        elif task_name == 'task_3':
                            loss = torch.sum(loss) / (num_preds_task_3_1+num_preds_task_3_2)
                            batch_losses[2].append(loss.detach().cpu().item())
                        else:
                            raise NotImplementedError

                        if torch.isnan(loss).data:
                            logger.info("Stop training because loss=%s" % (loss.data))
                            stop_training = True
                            break
                        loss.backward()

            train_losses[0].append(sum(batch_losses[0]) * num_preds_task_1.item())
            train_preds[0].append(num_preds_task_1.item())
            train_losses[1].append(sum(batch_losses[1]) * num_preds_task_2_1.item())
            train_preds[1].append(num_preds_task_2_1.item())
            train_losses[2].append(sum(batch_losses[2]) * (num_preds_task_3_1+num_preds_task_3_2).item())
            train_preds[2].append((num_preds_task_3_1+num_preds_task_3_2).item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()  # We have accumulated enought gradients
            scheduler.step()
            model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_f1_ans, curr_f1_ans_multi, curr_ques_f1_bleu, curr_ques_f1_edit_f1, dev_predictions = inference_qa_gen(model.module, dev_data, logger=logger)
                logger.info(
                    "Epoch={}, Global-step={}, Loss-T1={:.2f}, Loss-T2={:.2f}, Loss-T3={:.2f}, A-All={:.2f}, A-Multi={:.2f}, Q-Bleu={:.2f}, Q-Edit={:.2f}".format(
                        epoch,
                        global_step,
                        sum(train_losses[0]) / sum(train_preds[0]),
                        sum(train_losses[1]) / sum(train_preds[1]),
                        sum(train_losses[2]) / sum(train_preds[2]),
                        curr_f1_ans * 100.0,
                        curr_f1_ans_multi * 100.0,
                        curr_ques_f1_bleu * 100.0,
                        curr_ques_f1_edit_f1 * 100.0,
                    ))
                train_losses = [[],[],[]]
                train_preds = [[],[],[]]
                if best_accuracy < curr_f1_ans+curr_ques_f1_edit_f1:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("New best %s: %.2f%% -> %.2f%%" % ('Ans-All + Ques-Edit', best_accuracy*100.0, (curr_f1_ans+curr_ques_f1_edit_f1)*100.0,))
                    best_accuracy = curr_f1_ans+curr_ques_f1_edit_f1
                    dev_data.save_predictions(dev_predictions, mode='_{}'.format(dev_data.args.task))
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def train_reranker(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False
    wait_step = 0

    logger.info("Start training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1

            # actual batchsize for grad step
            batch_losses = []
            actual_train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
            for grad_step in range(0, len(batch[0]), actual_train_batch_size):
                actual_batch = [b[grad_step: grad_step + actual_train_batch_size].to(torch.device("cuda")) for b in batch]
                loss = model(input_ids=actual_batch[0], attention_mask=actual_batch[1], token_type_ids=actual_batch[2],
                             labels=actual_batch[3], is_training=True)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if torch.isnan(loss).data:
                    logger.info("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                batch_losses.append(loss.detach().cpu().item())
                loss.backward()

            train_losses.append(np.mean(batch_losses))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()  # We have accumulated enought gradients
            scheduler.step()
            model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(model, dev_data, save_predictions=False, logger=logger)
                logger.info("Epoch=%d, Global-step=%d, Train-loss=%.2f, %s=%.2f%%" % (
                    epoch,
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_em * 100.0,
                ))
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("New best %s: %.2f%% -> %.2f%%" % (dev_data.metric, best_accuracy * 100.0, curr_em * 100.0,))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def inference(model, dev_data, save_predictions=False, logger=None):
    if dev_data.args.task == 'dpr':
        return inference_dpr(model, dev_data, save_predictions=True)
    elif dev_data.args.task == 'rrk':
        return inference_reranker(model, dev_data, save_predictions=True, logger=logger)
    elif dev_data.args.task in ["qa", "qg", "qg_mask", "qa_noamb_aq", "qg_rewrite", "qg_weighted_loss", "qg_noprompt", "cotraining_label", "cotraining_train"]:
        if "bart" in dev_data.args.bert_name:
            if dev_data.args.ambigqa_editqg:
                return inference_seq2seq_editqg(model if dev_data.args.n_gpu == 1 or dev_data.args.do_predict or dev_data.args.do_e2e_predict else model.module, dev_data, save_predictions=save_predictions, logger=logger)
            if dev_data.args.filter_not_found_answer_passages:
                return inference_seq2seq_dynamic(model if dev_data.args.n_gpu == 1 or dev_data.args.do_predict or dev_data.args.do_e2e_predict else model.module, dev_data, save_predictions=save_predictions, logger=logger)
            return inference_seq2seq(model if dev_data.args.n_gpu==1 or dev_data.args.do_predict or dev_data.args.do_e2e_predict else model.module, dev_data, save_predictions=save_predictions, logger=logger)
        if "t5" in dev_data.args.bert_name:
            return inference_seq2seq_t5(model if dev_data.args.n_gpu == 1 or dev_data.args.do_predict or dev_data.args.do_e2e_predict else model.module, dev_data, save_predictions=save_predictions, logger=logger)
    else:
        raise NotImplementedError


def inference_dpr(model, dev_data, save_predictions):

    def _inference(dataloader, is_passages):
        if dev_data.args.n_gpu>1:
            curr_model = model.module.ctx_model if is_passages else model.module.question_model
            curr_model = torch.nn.DataParallel(curr_model)
        else:
            curr_model = model.ctx_model if is_passages else model.question_model
        vectors = []
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                batch = [b.to(torch.device("cuda")) for b in batch]
                outputs = curr_model(input_ids=batch[0], attention_mask=batch[1])[0][:,0,:]
                vectors.append(outputs.detach().cpu().numpy())
        return np.concatenate(vectors, axis=0)

    checkpoint = dev_data.args.checkpoint
    assert checkpoint is not None
    import faiss
    postfix = "_20200201" if dev_data.args.wiki_2020 else ""
    index_path = checkpoint[:checkpoint.index(".")] + "{}.IndexFlatIP".format(postfix)
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        checkpoint = dev_data.args.checkpoint
        # load passage vectors
        index = dev_data.args.db_index
        if index==-1:
            for index in range(10):
                pvec_path = checkpoint[:checkpoint.index(".")] + ".psgs_w100{}_{}.npy".format(postfix, index)
                assert os.path.exists(pvec_path)
                if index==0:
                    pvec = np.load(pvec_path)
                else:
                    pvec = np.concatenate([pvec, np.load(pvec_path)], axis=0)
        else:
            pvec_path = checkpoint[:checkpoint.index(".")] + ".psgs_w100{}_{}.npy".format(postfix, index)
            print (pvec_path)
            if os.path.exists(pvec_path):
                pvec = np.load(pvec_path)
            else:
                dev_data.passages.load_tokenized_data("bert")
                dev_data.passages.load_dataset("bert")
                dataloader = dev_data.passages.load_dataloader(
                    dev_data.args.predict_batch_size,
                    is_training=False,
                    do_return=True)
                if dev_data.args.verbose:
                    dataloader = tqdm(dataloader)
                pvec = _inference(dataloader, is_passages=True)
                np.save(pvec_path, pvec)
            exit()
        print (pvec.shape)
        index = faiss.IndexFlatIP(pvec.shape[1])
        index.add(pvec)
        faiss.write_index(index, index_path)
    # load question vectors
    qvec = _inference(dev_data.dataloader, is_passages=False) #model.inference(dev_data.dataloader, is_passages=False)
    print (qvec.shape)
    D, I = index.search(qvec, dev_data.args.dpr_retrieval_topk_psgs)
    assert D.shape == I.shape == (qvec.shape[0], dev_data.args.dpr_retrieval_topk_psgs)
    predictions = I.tolist()
    accuracy = dev_data.passages.evaluate(predictions, dev_data.get_answers())
    if save_predictions:
        dev_data.save_predictions(predictions, mode="_{}".format(dev_data.args.dpr_checkpoint))
    return np.mean(accuracy), None


def inference_reranker(model, dev_data, save_predictions=False, logger=None):
    outputs = []
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch]
            batch_sel_logits = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])
            batch_sel_logits = batch_sel_logits.detach().cpu().tolist()
            for sel_logit in batch_sel_logits:
                outputs.append((sel_logit))
    score = dev_data.evaluate(outputs)
    if save_predictions:
        output_ids = []
        for sel_logits in outputs:
            reranked_psgs_ids = np.argsort(-np.array(sel_logits)).tolist()
            output_ids.append(reranked_psgs_ids)
        dev_data.save_predictions(output_ids)
    return score, outputs


def inference_seq2seq(model, dev_data, save_predictions=False, logger=None):
    predictions = []
    predictions_id = []
    if dev_data.args.task in ["qa", "qa_noamb_aq", "cotraining_label", "cotraining_train"]:
        max_generation_length = dev_data.args.max_answer_length
        assert max_generation_length>=25 or not dev_data.args.ambigqa or dev_data.args.task == 'qa_noamb_aq'
        num_beams = 1
        min_generation_length = dev_data.args.min_answer_length
    else:
        max_generation_length = dev_data.args.max_question_length
        num_beams = 4
        min_generation_length = dev_data.args.min_question_length

    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=num_beams,
                                     min_length=min_generation_length,
                                     max_length=max_generation_length,
                                     early_stopping=True,
                                     decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))
                predictions_id.append(output.detach().cpu().tolist())
        if not dev_data.args.verbose and i % 100 == 0:
            logger.info("{} samples completed!".format(i * dev_data.args.predict_batch_size))
    if dev_data.args.task in ["qa", "qa_noamb_aq", "cotraining_train"]:
        score, results = dev_data.evaluate(predictions, predictions_id=predictions_id)
    elif dev_data.args.task == "cotraining_label":
        id2predictions = {}
        for idx, (pred, dp) in enumerate(zip(predictions, dev_data.data)):
            pred_1 = [text.strip() for text in pred.split(dev_data.SEP)]
            pred_2 = list(set(pred_1))
            id2predictions[dp['id']] = pred_2
        logger.info("Avg {:.2f} QAs per question".format(np.mean([len(x) for x in id2predictions.values()])))
        dev_data.save_predictions(id2predictions, mode='_{}'.format(dev_data.args.task))
        return -1, -1
    else:
        score, results = dev_data.evaluate(predictions)
    if save_predictions:
        dev_data.save_predictions(predictions, mode='_{}_b{}_maxlen{}_minlen{}'.format(dev_data.args.task, num_beams, max_generation_length, min_generation_length))
    return score, results


def inference_overgenerate(task, model, dev_data, metadata):
    assert task in ['qa', 'qg']
    if task == 'qa':
        max_generation_length = dev_data.args.max_answer_length
        assert max_generation_length >= 8 or not dev_data.args.ambigqa or dev_data.args.task == 'qa_noamb_aq'
        num_beams = 1
        min_generation_length = dev_data.args.min_answer_length
    elif task == 'qg':
        max_generation_length = dev_data.args.max_question_length
        num_beams = 4
        min_generation_length = dev_data.args.min_question_length

    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)

    predictions = []
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=num_beams,
                                     min_length=min_generation_length,
                                     max_length=max_generation_length,
                                     early_stopping=True,
                                     decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))

    current_overgenerate_pass = dev_data.args.over_generate_pass
    if task == 'qa':
        for idx, m in enumerate(metadata):
            curr_prompt_answer_pairs = []
            for jdx in range(*m):
                pred = [text.strip() for text in predictions[jdx].split(dev_data.SEP) if text.strip() != ""]
                pred = list(set(pred))
                for pred_i in pred:
                    # (prompt, answer)
                    prompt_answer = tuple([dev_data.map_input_questions[jdx], pred_i])
                    curr_prompt_answer_pairs.append(prompt_answer)
            curr_prompt_answer_pairs = list(set(curr_prompt_answer_pairs))
            # if not use generated q as QG input, replace the generated q as prompt q
            if not bool(dev_data.args.replace_prompt_question):
                curr_answers = list(set([x[1] for x in curr_prompt_answer_pairs]))
                curr_prompt_answer_pairs = [tuple([dev_data.data[idx]['question'].lower(), a]) for a in curr_answers]
            dev_data.data[idx]["over_generate_{}_prompt_answer".format(current_overgenerate_pass)] = curr_prompt_answer_pairs
    else:
        for idx, m in enumerate(metadata):
            curr_prompt_answer_pairs = dev_data.data[idx]["over_generate_{}_prompt_answer".format(current_overgenerate_pass)]
            curr_noambq_answer_pairs = []
            prev_noambq_answer_pairs = dev_data.data[idx]["over_generate_{}_noambq_answer".format(current_overgenerate_pass-1)] if current_overgenerate_pass > 0 else []
            assert len(curr_prompt_answer_pairs) == m[1] - m[0]
            for jdx_ans, jdx_pred in enumerate(range(*m)):
                pred_answer = curr_prompt_answer_pairs[jdx_ans][1]
                pred_question = predictions[jdx_pred].strip()
                curr_noambq_answer_pairs.append(tuple([pred_question, pred_answer]))
            # merge current pass generated qa pairs and previously generated qa pairs (will be used for filtering)
            curr_noambq_answer_pairs = list(set(curr_noambq_answer_pairs + prev_noambq_answer_pairs))
            dev_data.data[idx]["over_generate_{}_noambq_answer".format(current_overgenerate_pass)] = curr_noambq_answer_pairs


def inference_lm_filtering(model, dev_data, logger=None):
    lm_scores = []
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch]
            batch_score = model(input_ids=batch[0], attention_mask=batch[1],
                                decoder_input_ids=batch[2], decoder_attention_mask=batch[3])
            lm_scores.extend(batch_score.tolist())
    return lm_scores


def inference_em_filtering(model, dev_data, logger=None):
    predictions = []
    max_generation_length = dev_data.args.max_answer_length
    num_beams = 1
    min_generation_length = dev_data.args.min_answer_length

    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=num_beams,
                                     min_length=min_generation_length,
                                     max_length=max_generation_length,
                                     early_stopping=True,
                                     decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))
    return predictions


def inference_seq2seq_dynamic(model, dev_data, save_predictions=False, logger=None):
    predictions = []
    predictions_id = []
    if dev_data.args.task in ["qa", "qa_noamb_aq"]:
        max_generation_length = dev_data.args.max_answer_length
        assert max_generation_length>=25 or not dev_data.args.ambigqa or dev_data.args.task == 'qa_noamb_aq'
        num_beams = 1
        min_generation_length = dev_data.args.min_answer_length
    else:
        max_generation_length = dev_data.args.max_question_length
        num_beams = 4
        min_generation_length = dev_data.args.min_question_length

    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:3]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=num_beams,
                                     min_length=min_generation_length,
                                     max_length=max_generation_length,
                                     early_stopping=True,
                                     decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     is_discard=batch[2],
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))
                predictions_id.append(output.detach().cpu().tolist())
    if dev_data.args.task in ["qa", "qa_noamb_aq"]:
        score, results = dev_data.evaluate(predictions, predictions_id=predictions_id)
    else:
        score, results = dev_data.evaluate(predictions)
    if save_predictions:
        dev_data.save_predictions(predictions, mode='_{}_b{}_maxlen{}_minlen{}'.format(dev_data.args.task, num_beams, max_generation_length, min_generation_length))
    return score, results


def inference_seq2seq_editqg(editing_model, dev_data, save_predictions=False, logger=None, rewriting_model=None):
    best_score = -1
    all_results = {}
    for num_beams in [4]:
        predictions = []
        max_generation_length = dev_data.args.max_question_length
        min_generation_length = dev_data.args.min_question_length
        if dev_data.args.verbose:
            dev_data.dataloader = tqdm(dev_data.dataloader)
        for i, batch in enumerate(dev_data.dataloader):
            with torch.no_grad():
                batch = [b.to(torch.device("cuda")) for b in batch[:2]]
                keyphrase_outputs = editing_model.generate(input_ids=batch[0],
                                         attention_mask=batch[1],
                                         num_beams=num_beams,
                                         min_length=min_generation_length,
                                         max_length=max_generation_length,
                                         early_stopping=True,
                                         decoder_start_token_id=editing_model.decoder_start_token_id,
                                         num_return_sequences=1,
                                         )
                if rewriting_model is None:
                    for input_, output in zip(batch[0], keyphrase_outputs):
                        predictions.append(dev_data.decode(output))
                else:
                    bos_token_id = dev_data.tokenizer.bos_token_id
                    eos_token_id = dev_data.tokenizer.eos_token_id
                    pad_token_id = dev_data.tokenizer.pad_token_id
                    sep_token_id = dev_data.tokenizer.convert_tokens_to_ids(dev_data.SEP)
                    new_input_ids, new_attention_mask = [], []
                    for input_id_i, input_attn_mask_i, keyphrase_output in zip(batch[0], batch[1], keyphrase_outputs):
                        # question
                        input_id_i_0 = input_id_i[0].tolist()
                        question_start_id = input_id_i_0.index(sep_token_id)
                        question_end_id = input_id_i_0[question_start_id:].index(eos_token_id)
                        question_id = input_id_i_0[question_start_id+1:question_start_id+question_end_id]
                        # keyphrase
                        keyphrase_output = keyphrase_output.tolist()[1:]
                        if eos_token_id in keyphrase_output:
                            keyphrase_end_id = keyphrase_output.index(eos_token_id)
                            keyphrase_output = keyphrase_output[:keyphrase_end_id]
                        else:
                            print(keyphrase_output)
                        new_input_ids_i = [bos_token_id] + question_id + [sep_token_id] + keyphrase_output + [eos_token_id]
                        new_attention_mask_i = [1] * len(new_input_ids_i)
                        max_input_len = dev_data.args.max_question_length * 2
                        if len(new_input_ids_i) > max_input_len:
                            new_input_ids_i = new_input_ids_i[:max_input_len]
                            new_attention_mask_i = new_attention_mask_i[:max_input_len]
                        else:
                            new_input_ids_i += [pad_token_id for _ in range(max_input_len - len(new_input_ids_i))]
                            new_attention_mask_i += [0 for _ in range(max_input_len - len(new_attention_mask_i))]
                        new_input_ids.append(new_input_ids_i)
                        new_attention_mask.append(new_attention_mask_i)

                    outputs = rewriting_model.generate(input_ids=torch.LongTensor(new_input_ids).to(torch.device("cuda")),
                                         attention_mask=torch.LongTensor(new_attention_mask).to(torch.device("cuda")),
                                         num_beams=4,
                                         min_length=min_generation_length,
                                         max_length=max_generation_length,
                                         early_stopping=True,
                                         decoder_start_token_id=rewriting_model.decoder_start_token_id,
                                         num_return_sequences=1,
                                         )
                    for input_, output in zip(batch[0], outputs):
                        predictions.append(dev_data.decode(output))
                    dev_data.args.ambigqa_editqg = False
        score, results = dev_data.evaluate(predictions)
        best_score = max(score, best_score)
        all_results.update({'B{}_'.format(num_beams) + k: v for k, v in results.items()})
        if save_predictions:
            dev_data.save_predictions(predictions, mode='_{}_edit_b{}_maxlen{}_minlen{}'.format(dev_data.args.task, num_beams, max_generation_length, min_generation_length))
    return best_score, all_results


def inference_qa_gen(model, dev_data, logger=None):
    # 1. first question generation
    decoder_start_token_id = dev_data.tokenizer.convert_tokens_to_ids(dev_data.QBOS)
    predictions_question = []
    predictions_question_metadata = []
    max_catq_length = 80
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=4,
                                     min_length=1,
                                     max_length=max_catq_length,
                                     early_stopping=True,
                                     decoder_start_token_id=decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                output_cat_question = dev_data.decode(output).replace(dev_data.QBOS, "").strip()
                curr_predicted_questions = list(set(output_cat_question.split(dev_data.SEP)))
                predictions_question_metadata.append((len(predictions_question), len(predictions_question)+len(curr_predicted_questions)))
                predictions_question.extend(curr_predicted_questions)
    # replace <end> with the original question (in this case, model thinks the prompt Q is not ambiguous)
    assert len(predictions_question_metadata) == len(dev_data.aq_data)
    noamb_count = 0
    for idx, curr_q_metadata in enumerate(predictions_question_metadata):
        for idx_q in range(*curr_q_metadata):
            if predictions_question[idx_q] == "":
                predictions_question[idx_q] = dev_data.aq_data[idx]["question"]
                noamb_count += 1
    logger.info("{} ambiguous, {} not ambiguous".format(len(predictions_question_metadata)-noamb_count, noamb_count))
    # 2. then generate answer
    decoder_start_token_id = dev_data.tokenizer.convert_tokens_to_ids(dev_data.ABOS)
    predictions_answer = []
    max_answer_length = 20
    # prepare input data
    tokenized_qa_dataset = dev_data.load_task_3_1_inference_dataset(predictions_question, predictions_question_metadata)
    qa_dataloader = dev_data.load_task_3_1_inference_dataloader(tokenized_qa_dataset)
    if dev_data.args.verbose:
        qa_dataloader = tqdm(qa_dataloader)
    for i, batch in enumerate(qa_dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=1,
                                     min_length=1,
                                     max_length=max_answer_length,
                                     early_stopping=True,
                                     decoder_start_token_id=decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                output_answer = dev_data.decode(output).replace(dev_data.ABOS, "").strip()
                predictions_answer.append(output_answer)
    assert len(predictions_answer) == len(predictions_question) == predictions_question_metadata[-1][-1]
    ans_all, ans_multi, question_bleu, question_edit, qa_pair_predictions = dev_data.evaluate(predictions_question, predictions_answer, predictions_question_metadata)
    return ans_all, ans_multi, question_bleu, question_edit, qa_pair_predictions


def inference_overgenvfy_qg_seq2seq(model, dev_data, save_predictions=False, logger=None):
    # this function is only used for qg
    predictions = []
    assert dev_data.args.task != "qa"
    max_generation_length = dev_data.args.max_question_length
    min_generation_length = dev_data.args.min_question_length
    beam=4
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=beam,
                                     min_length=min_generation_length,
                                     max_length=max_generation_length,
                                     early_stopping=True,
                                     decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))
    qapair_predictions = dev_data.evaluate(predictions)

    if save_predictions:
        # dev_20200201_aq_qa_b1_maxlen80_minlen1_predictions.json
        answer_prediction_file = dev_data.args.answer_prediction_file.split('/')
        answer_prediction_prefix = "MODEL-{}_DCD-{}".format(answer_prediction_file[-2], answer_prediction_file[-1].replace("dev_20200201_aq_qa_", "").replace("_predictions.json", ""))
        if dev_data.args.do_e2e_predict:
            dev_data.save_predictions(qapair_predictions, mode='AF-{}_{}_b{}_maxlen{}_minlen{}'.format(
                answer_prediction_prefix, dev_data.args.task, beam, max_generation_length, min_generation_length))
        elif dev_data.args.do_overgenvfy_qg_predict:
            dev_data.save_predictions(qapair_predictions, mode='AF-{}_{}_b{}_maxlen{}_minlen{}_ogtime{}'.format(
                answer_prediction_prefix, dev_data.args.task, beam, max_generation_length, min_generation_length, dev_data.args.overgenvfy_time))


def inference_overgenvfy_qa_seq2seq(model, dev_data, save_predictions=False, logger=None):
    predictions = []
    max_generation_length = dev_data.args.max_answer_length
    assert max_generation_length >= 24
    min_generation_length = dev_data.args.min_answer_length
    beam = 1

    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=beam,
                                     min_length=min_generation_length,
                                     max_length=max_generation_length,
                                     early_stopping=True,
                                     decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))
    predictions_current, predictions_merged = dev_data.evaluate(predictions)
    if save_predictions:
        dev_data.save_predictions((predictions_current, predictions_merged), mode='_{}_b{}_maxlen{}_minlen{}_ogtime{}'.format(
            dev_data.args.task, beam, max_generation_length, min_generation_length, dev_data.args.overgenvfy_time))


def inference_seq2seq_t5(model, dev_data, save_predictions=False):
    predictions = []
    # bos_token_id = dev_data.tokenizer.bos_token_id
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        with torch.no_grad():
            batch = [b.to(torch.device("cuda")) for b in batch[:2]]
            # decoder_input_ids = model.tokenizer.encode("<pad>", return_tensors="pt")
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     num_beams=1,
                                     min_length=1,
                                     max_length=25,
                                     early_stopping=True,
                                     # decoder_start_token_id=model.decoder_start_token_id,
                                     num_return_sequences=1,
                                     )
            for input_, output in zip(batch[0], outputs):
                predictions.append(dev_data.decode(output))
    greedy = np.mean(dev_data.evaluate(predictions))
    if save_predictions:
        dev_data.save_predictions(predictions)
    return greedy





