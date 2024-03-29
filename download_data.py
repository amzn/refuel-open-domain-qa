#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to download various preprocessed data sources & checkpoints for DPR
"""

import gzip
import os
import pathlib

import argparse
import wget

NQ_LICENSE_FILES = [
    'https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE',
    'https://dl.fbaipublicfiles.com/dpr/nq_license/README',
]

RESOURCES_MAP = {
    # Wikipedia DB 2018/12/20 (provided by DPR)
    'data.wikipedia_split.psgs_w100': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz',
        'original_ext': '.tsv.gz',
        'compressed': False,
        'desc': 'Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)'
    },

    # Wikipedia DB 2020/02/01 (provided by DPR)
    'data.wikipedia_split.psgs_w100_20200201': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/psgs_w100_20200201.tsv.gz',
        'original_ext': '.tsv.gz',
        'compressed': False,
        'desc': 'Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)'
    },

    # QA data / checkpoints provided by DPR
    'data.retriever.nq-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'NQ dev subset with passages pools for the Retriever train time validation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.nq-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'NQ train subset with passages pools for the Retriever training',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ dev subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ test subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ train subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our train subset) gold positive passages and alternative question tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our dev subset) gold positive passages and alternative question tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our test, original dev subset) gold positive passages and alternative question '
                'tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever_results.nq.single.wikipedia_passages': {
        's3_url': ['https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq/wiki_passages_{}'.format(i) for i in
                   range(50)],
        'original_ext': '.pkl',
        'compressed': False,
        'desc': 'Encoded wikipedia files using a biencoder checkpoint('
                'checkpoint.retriever.single.nq.bert-base-encoder) trained on NQ dataset '
    },

    'data.retriever_results.nq.single.test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ test dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    },
    'data.retriever_results.nq.single.dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ dev dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    },
    'data.retriever_results.nq.single.train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ train dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    },

    'checkpoint.retriever.single.nq.bert-base-encoder': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on NQ data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.multiset.bert-base-encoder': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on multi set data and HF bert-base-uncased model'
    },

    'data.reader.nq.single.train': {
        's3_url': ['https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/train.{}.pkl'.format(i) for i in range(8)],
        'original_ext': '.pkl',
        'compressed': False,
        'desc': 'Reader model NQ train dataset input data preprocessed from retriever results (also trained on NQ)',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.reader.nq.single.dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/dev.0.pkl',
        'original_ext': '.pkl',
        'compressed': False,
        'desc': 'Reader model NQ dev dataset input data preprocessed from retriever results (also trained on NQ)',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.reader.nq.single.test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/test.0.pkl',
        'original_ext': '.pkl',
        'compressed': False,
        'desc': 'Reader model NQ test dataset input data preprocessed from retriever results (also trained on NQ)',
        'license_files': NQ_LICENSE_FILES,
    },

    'checkpoint.reader.nq-single.hf-bert-base': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single/hf_bert_base.cp',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Reader weights trained on NQ-single retriever results and HF bert-base-uncased model'
    },

    'checkpoint.reader.nq-trivia-hybrid.hf-bert-base': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-trivia-hybrid/hf_bert_base.cp',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Reader weights trained on Trivia multi hybrid retriever results and HF bert-base-uncased model'
    },

    # resources provided by AmbigQA
    'data.ambigqa.train_light': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/train_light.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Train file for AmbigQA'
    },

    'data.ambigqa.dev_light': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/dev_light.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Dev file for AmbigQA'
    },

    'data.nqopen.train': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/nqopen-train.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Train file for NQ-open'
    },

    'data.nqopen.dev': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/nqopen-dev.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Dev file for NQ-open'
    },

    'data.nqopen.test': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/nqopen-test.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Test file for NQ-open'
    },

    'data.nqopen.train_id2answers': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/train_id2answers.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Dev id to official answers provided by Google'
    },

    'data.nqopen.dev_id2answers': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/dev_id2answers.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Dev id to official answers provided by Google'
    },

    'data.nqopen.test_id2answers': {
        's3_url': 'https://nlp.cs.washington.edu/ambigqa/data/test_id2answers.json',
        'original_ext': '.json',
        'compressed': False,
        'desc': 'Test id to official answers provided by Google'
    },


}


def unpack(gzip_file: str, out_file: str):
    print('Uncompressing ', gzip_file)
    input = gzip.GzipFile(gzip_file, 'rb')
    s = input.read()
    input.close()
    output = open(out_file, 'wb')
    output.write(s)
    output.close()
    print('Saved to ', out_file)


def download_resource(s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str) -> str:
    print('Loading from ', s3_url)

    # create local dir
    path_names = resource_key.split('.')

    root_dir = out_dir if out_dir else './'
    save_root = os.path.join(root_dir, *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file = os.path.join(save_root, path_names[-1] + ('.tmp' if compressed else original_ext))

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return save_root

    wget.download(s3_url, out=local_file)

    print('Saved to ', local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file)
        os.remove(local_file)
    return save_root


def download_file(s3_url: str, out_dir: str, file_name: str):
    print('Loading from ', s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return

    wget.download(s3_url, out=local_file)
    print('Saved to ', local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print('no resources found for specified key')
        return
    download_info = RESOURCES_MAP[resource_key]

    s3_url = download_info['s3_url']

    save_root_dir = None
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir = download_resource(url,
                                              download_info['original_ext'],
                                              download_info['compressed'],
                                              '{}_{}'.format(resource_key, i),
                                              out_dir)
    else:
        save_root_dir = download_resource(s3_url,
                                          download_info['original_ext'],
                                          download_info['compressed'],
                                          resource_key,
                                          out_dir)

    license_files = download_info.get('license_files', None)
    if not license_files:
        return

    download_file(license_files[0], save_root_dir, 'LICENSE')
    download_file(license_files[1], save_root_dir, 'README')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory to download file")
    parser.add_argument("--resource", type=str,
                        help="Resource name. See RESOURCES_MAP for all possible values")
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print('Please specify resource value. Possible options are:')
        for k, v in RESOURCES_MAP.items():
            print('Resource key={}  description: {}'.format(k, v['desc']))


if __name__ == '__main__':
    main()
