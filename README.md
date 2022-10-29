# [ACL'21] Round-trip Evidence FUsion via gEneration with retrievaL (REFULE)

This repository is the implementation of our ACL 2021 Paper Round-trip Evidence FUsion via gEneration with retrievaL.

REFUEL achieves new state-of-the-art results on AmbigQA benchmark (Sep 2021).

If you have any question, please open an issue or contact yifangao95@gmail.com

## Reference
If you find our code useful, please cite our papers as follows:

```
@inproceedings{gao-etal-2021-answering,
    title = "Answering Ambiguous Questions through Generative Evidence Fusion and Round-Trip Prediction",
    author = "Gao, Yifan  and
      Zhu, Henghui  and
      Ng, Patrick  and
      Nogueira dos Santos, Cicero  and
      Wang, Zhiguo  and
      Nan, Feng  and
      Zhang, Dejiao  and
      Nallapati, Ramesh  and
      Arnold, Andrew O.  and
      Xiang, Bing",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.253",
    doi = "10.18653/v1/2021.acl-long.253",
    pages = "3263--3276",
    abstract = "In open-domain question answering, questions are highly likely to be ambiguous because users may not know the scope of relevant topics when formulating them. Therefore, a system needs to find possible interpretations of the question, and predict one or multiple plausible answers. When multiple plausible answers are found, the system should rewrite the question for each answer to resolve the ambiguity. In this paper, we present a model that aggregates and combines evidence from multiple passages to adaptively predict a single answer or a set of question-answer pairs for ambiguous questions. In addition, we propose a novel round-trip prediction approach to iteratively generate additional interpretations that our model fails to find in the first pass, and then verify and filter out the incorrect question-answer pairs to arrive at the final disambiguated output. Our model, named Refuel, achieves a new state-of-the-art performance on the AmbigQA dataset, and shows competitive performance on NQ-Open and TriviaQA. The proposed round-trip prediction is a model-agnostic general approach for answering ambiguous open-domain questions, which improves our Refuel as well as several baseline models. We release source code for our models and experiments at https://github.com/amzn/refuel-open-domain-qa.",
}
```

## Requirements

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

8 V100 (32G) GPUs are required to run the following experiments.

## Content
- [Overview](#overview)
- [Retrieval \& Reranking](#retrieval-and-reranking)
- [Answer Prediction](#answer-prediction)
- [Question Disambiguation](#question-disambiguation)
- [End-to-End Evaluation & Round-Trip Prediction](#end-to-end-evaluation--round-trip-prediction)
- [References](#references)


### Overview
Our solution is a pipelined approach which contains the following steps:

1) Given an ambiguous prompt question, the retrieval module firstly retrieves 1000 question-relevant passages;

2) The reranking module reranks 1000 passages by interacting the prompt question with each passage;

3) The answer prediction module takes top 100 reranked passages, and generate one or multiple answers.

4) The question disambiguation module generate a disambiguated question for each predicted answer. 


### Resources \& Dataset
Download the following datasets and checkpoints

-  Wikipedia corpus for NQ-open and AmbigQA (The questions in these two datasets are annotated by difference Wikipedia dumps):

```
# Wikipedia corpus for NQ-open
python download_data.py \
    --resource data.wikipedia_split.psgs_w100 \
    --output_dir retriever_data/wikipedia_split/

# Wikipedia corpus for AmbigQA
python download_data.py \
    --resource data.wikipedia_split.psgs_w100_20200201 \
    --output_dir retriever_data/wikipedia_split/
```

- NQ-open dataset, AmbigQA dataset, and official Natural Question answers ([why?](https://github.com/shmsw25/AmbigQA/blob/master/codes/README.md): see "Note" there)

```
# AmbigQA dataset
wget https://nlp.cs.washington.edu/ambigqa/data/ambignq.zip -O reader_data/ambigqa/

# NQ-open dataset
python download_data.py \
    --resource data.nqopen.{train|dev|test} \
    --output_dir reader_data/nqopen/
python download_data.py \
    --resource data.nqopen.{train|dev|test}_id2answers \
    --output_dir reader_data/nqopen/
```

### Retrieval and Reranking

#### Retrieval 

We use the Dense Passage Retriever [1] as our retriever. There two versions of DPR checkpoints, one is trained only on the NQ-open dataset (single.pt), another is jointly trained on five QA datasets (multiset.pt). According to our experiments, `multiset.pt` performs slightly better than the `single.pt`. So here we only use the multiset version.

- Download DPR checkpoints

```
python download_data.py \
    --resource checkpoint.retriever.multiset.bert-base-encoder \
    --output_dir retriever_data/checkpoint/retriever/multiset/
```

- Encode all passages in Wikipedia into d-dimensional dense representations (it may take several hours to finish this step)

The wikipedia is splitted into 10 shards for encoding. Here we use a for loop to encode the dense representations, it would be better to split these cmds into different GPUs to save time.

```
# For NQ-open Wikipedia Dump

for i in 0 1 2 3 4 5 6 7 8 9
do
    ./scripts/retriever/generate_dense_representations_nq.sh $i $GPU_ID
done

# For AmbigQA Wikipedia Dump
for i in 0 1 2 3 4 5 6 7 8 9
do
    ./scripts/retriever/generate_dense_representations_aq.sh $i $GPU_ID
done
```

- Retrieve 1000 relevant passages for questions in NQ-open / AmbigQA: (still, it may take several hours to finish this step)

```
# NQ-open
./scripts/retriever/retrieve_psgs_nq.sh reader_data/nqopen/{train|dev|test}.json $GPU_ID

# AmbigQA
./scripts/retriever/retrieve_psgs_aq.sh reader_data/ambigqa/{train|dev}.json $GPU_ID

# Leaderboard Submission
./scripts/retriever/retrieve_psgs_aq_leaderboard.sh $GPU_ID
```

#### Reranking
We train a `bert-large-uncased`-based reranker with listwise ranking loss. It takes ~1 day to finish training. 

Firstly, we train the reranker on the NQ-open dataset:

```
./script/reranker/train_nq.sh
```

Then, use the trained reranker to rerank passages for train, dev, test set of NQ-Open.

```
./script/reranker/inference_nq.sh path/to/saved/model.pt {train|dev|test}
```

Finetune the reranker on AmbigQA dataset is optional, we tried to finetune it on AmbigQA but the results are comparable。

Here we directly use this reranker rerank passages for train, dev, test (leaderboard) of AmbigQA.

```
# train/dev set of AmbigQA
./script/reranker/inference_aq.sh path/to/saved/model.pt {train|dev}

# Leaderboard Submission
./script/reranker/inference_aq_leaderboard.sh path/to/saved/model.pt
```

### Answer Prediction
We firstly pre-train the answer prediction model on NQ-open, and fine-tune it on AmbigQA.

It takes ~2 days to for pre-training and fine-tuning each.

- Pre-train on NQ-open

```
./script/answer_prediction/train_nq.sh path/to/reranker/outputs
```

- Evaluate for NQ-open

```
./script/answer_prediction/inference_nq.sh $GPU_ID {dev|test} path/to/model path/to/reranker/outputs
```

- Fine-tune on AmbigQA

```
./script/answer_prediction/train_aq.sh path/to/pretrained/ckpt path/to/reranker/outputs
```

- Evaluate for Fine-tuned model

```
./script/answer_prediction/inference_aq.sh $GPU_ID path/to/model/ckpt path/to/reranker/outputs
```


### Question Disambiguation
- Pre-train on NQ-open (Token-Deletion Pretraining)

```
./script/question_disambiguation/train_nq.sh path/to/reranker/outputs
```

- Evaluate on NQ-open for gold answers: `gold-answer + partial-question + passages -> complete-question`. 

```
./script/question_disambiguation/inference_nq.sh gpu_id path/to/save/outputs path/to/reranker/outputs path/to/trained/ckpt
```

- Fine-tune on AmbigQA (with insertion-based weighted loss)

```
./script/question_disambiguation/train_aq.sh path/to/save/outputs path/to/reranker/outputs path/to/pretrained/ckpt
```

- Evaluate on AmbigQA for gold answers: `gold-answer + ambiguous-question + passages -> disambiguated question`.

```
./script/question_disambiguation/inference_aq.sh gpu_id path/to/save/outputs path/to/reranker/outputs path/to/trained/ckpt
```

### End-to-End Evaluation & Round-Trip Prediction

#### End-to-End Evaluation
Up to now we have trained answer prediction model and question disambiguation model, we can conduct end-to-end evaluateion, i.e., do question disambiguation towards predicted answers

```
Pass=0
./script/round_trip/round_trip_generation.sh path/to/answer/prediction/ckpt path/to/answer/prediction/ckpt ${Pass} {dev|test} gpu_id 
```

Here `Pass` means we are doing zero pass of round-trip generation, which is the end-to-end prediction.

#### Round-Trip Prediction
If we want to iteratively do the generation process, we can set the `Pass` from 1 to X, X is the number of iteration which we find our model cannot predict more answers after this pass (Usually we can set X = 10)

```
for Pass in 1 2 3 4 5 6 7 8 9
do
./script/round_trip/round_trip_generation.sh path/to/answer/prediction/ckpt path/to/answer/prediction/ckpt ${Pass} {dev|test} gpu_id 
done
```

After we over-generate a bunch of QA pairs, we can conduct Language Model (LM) Verification to filter out incorrect QA pairs

- Firstly, we need to train a Verification model on disambiguated QA pairs

```
./script/round_trip/train_verification.sh path/to/pretrained/ckpt path/to/reranker/outputs
```

- Then, we can get a sense of the performance of this verification model

```
./script/round_trip/inference_verification.sh $GPU_ID path/to/model/ckpt path/to/reranker/outputs
```

- Finally, we can use this verification model to filter out incorrect QA pairs, the threshold is tuned towards the dev set performance

```
Pass=X
./script/round_trip/lm_verification.sh path/to/answer/prediction/ckpt path/to/answer/prediction/ckpt path/to/verification/model {dev|test} $Pass gpu_id 
```

Here `Pass` is the pass you choose to do LM verification, we found it is also a hyperparameter needed to tune.

    
## References
1. [AmbigQA: Answering Ambiguous Open-domain Questions](https://arxiv.org/abs/2004.10645), Min et al., EMNLP 2020.
2. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Karpukhin et al., EMNLP 2020.
3. [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282), Gautier Izacard, Edouard Grave, ArXiv, July 2020.


