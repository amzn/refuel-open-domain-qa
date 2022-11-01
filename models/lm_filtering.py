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


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

class MyBartLMFiltering(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            lm_labels=None, use_cache=False, is_training=False):
        # generate LM labels0
        pad_token_id = self.config.pad_token_id
        y_ids = decoder_input_ids[:, :-1].contiguous()
        # modify the decoder_start_token_id, 0 or 2?
        y_ids[..., 0] = self.decoder_start_token_id
        y_mask = decoder_attention_mask[:, :-1].contiguous()
        lm_labels = decoder_input_ids[:, 1:].clone()
        lm_labels[decoder_input_ids[:, 1:] == pad_token_id] = -100

        N, M, L = input_ids.shape
        H = 1024  # for bart-large

        multi_enc_out = self.model.encoder(input_ids=input_ids.view(N*M,L), attention_mask=attention_mask.view(N*M,L))[0]
        multi_enc_out = (multi_enc_out.view(N,M,L,H).reshape(N,M*L,H), [], [])
        multi_enc_attn_mask = attention_mask.view(N,-1)

        outputs = self.model(
            None,
            attention_mask=multi_enc_attn_mask,
            encoder_outputs=multi_enc_out,
            decoder_input_ids=y_ids,
            decoder_attention_mask=y_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
        return torch.sum(loss.view(lm_labels.shape[0], -1), dim=-1)

    def prepare_scores_for_generation(self, scores, cur_len, max_length):
        if cur_len == 1:
            return scores
            # self._force_token_ids_generation(scores, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(scores, self.config.eos_token_id)
        return scores


