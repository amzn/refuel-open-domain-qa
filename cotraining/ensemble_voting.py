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


# a hard voting method for ensemble
import json
import numpy as np
from collections import Counter
from copy import deepcopy
from ambigqa_evaluate_script import QAPairEvaluation

# predictions
aq_ckpts = [
'09122344-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5-seed45',
'09122351-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5-seed66',
'09122308-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5-seed78',
'09131232-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5-seed85',
'09051132-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5',
# '09131232-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5-seed19',
# '09122309-MFID-AQ-b64psg100gd8-disa1-e30-lr3e5-seed59',
]

qd_ckpt = '09111601-MFID-AQG-b64psg100gd8-e30-lossw3d5-lr5e5'

with open('/home/ubuntu/data/MyFusionInDecoderDataReader/ambigqa/dev.json') as f:
    dev = json.load(f)

split = 'dev'

model2predictions = []
for aq_ckpt in aq_ckpts:
    with open('/home/ubuntu/data/MyFusionInDecoderOut/MAP_{}_QD_{}/{}_e2e_leaderboard.json'.format(aq_ckpt, qd_ckpt, split)) as f:
        model2predictions.append(json.load(f))

ids = list(model2predictions[-1].keys())

# merge by hard voting
id2predictions = {}
for id in ids:
    id2predictions[id] = []
    curr_answers = Counter()
    answer2question = {}
    for model2prediction in model2predictions:
        for pred_qapair in model2prediction[id]:
            curr_answers[pred_qapair['answer']] += 1
            if pred_qapair['answer'] not in answer2question:
                answer2question[pred_qapair['answer']] = pred_qapair['question']
            else:
                assert answer2question[pred_qapair['answer']] == pred_qapair['question']
    current_max_vote = max(curr_answers.values())
    if len(aq_ckpts) % 2 == 0:
        min_voter = min(len(aq_ckpts) // 2, current_max_vote)
    else:
        min_voter = min(len(aq_ckpts) // 2+1, current_max_vote)
    for k, v in curr_answers.items():
        if v >= min_voter:
            id2predictions[id].append({'answer': k, 'question': answer2question[k]})

# with open('/home/ubuntu/data/MyFusionInDecoderOut/20201008_ensemble_{}.json'.format(split), 'w') as f:
#     json.dump(id2predictions, f)
print("Avg {:.2f} qa pairs per prompt".format(np.mean([len(x) for x in id2predictions.values()])))
evaluation = QAPairEvaluation(deepcopy(dev), deepcopy(id2predictions))
results = evaluation.print_all_metrics()


