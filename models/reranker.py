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
import  numpy
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForQuestionAnswering

class BertReranker(BertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None,
                labels=None, is_training=False):
        N, M, L = input_ids.size()
        output = self.bert(input_ids.view(N*M, L),
                           attention_mask=attention_mask.view(N*M, L),
                           token_type_ids=token_type_ids.view(N*M, L),
                           inputs_embeds=None if inputs_embeds is None else inputs_embeds.view(N*M, L, -1))[0]
        sel_logits = self.qa_classifier(output[:,0,:])

        if is_training:
            return get_loss(sel_logits, labels, N, M)
        else:
            return sel_logits.view(N, M)


def get_loss(sel_logits, sel_labels, N, M):
    sel_logits = sel_logits.view(N, M)
    # assert sel_labels.shape[1] == 1, "multi positive is not supported!"
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-1)
    sel_loss = []
    for _sel_labels in torch.unbind(sel_labels, dim=1):
        _sel_loss = loss_fct(sel_logits, _sel_labels)
        sel_loss.append(_sel_loss)
    return torch.sum(torch.cat(sel_loss))/torch.sum(sel_labels[:,0] != -1)

def _take_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
            - loss_tensor - 1e10 * (loss_tensor==0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood + \
                                torch.ones(loss_tensor.size(0)).cuda()*(marginal_likelihood==0).float()))

