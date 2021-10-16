from ast import parse
import logging
import os
import sys

from typing import List, Callable, NoReturn, NewType, Any
from argparse import ArgumentParser
from datasets.load import load_dataset
from omegaconf import OmegaConf
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, training_args

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

import preprocessing
from trainer_qa import QuestionAnsweringTrainer


def main(cfg, do_train):
    set_seed(42)  # set random seed

    model_name = cfg.model_name
    dataset_dir = cfg.dataset_dir

    # load model config
    config = AutoConfig.from_pretrained(model_name)

    # set training arguments
    training_args = TrainingArguments(
        overwrite_output_dir='./models/training_dataset',
        output_dir='./output',
        logging_dir='./logs',
        do_train=True if do_train else False,
        do_eval=True if not do_train else True,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.strategy,
        evaluation_strategy=cfg.strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=eval(cfg.load_best_model_at_end)
    )

    # load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # load dataset
    datasets = load_from_disk(dataset_dir)
    print(datasets)

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    column_names = [question_column_name,
                    context_column_name, answer_column_name]

    if training_args.do_train:
        train_dataset = datasets["train"]

        train_dataset = train_dataset.map(
            preprocessing.prepare_train_features,
            batched=True,
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=column_names,
            fn_kwargs=dict(tokenizer=tokenizer, config=cfg,
                           pad_on_right=True, column_names=column_names)
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--do_train', type=bool, default=True)

    args = parser.parse_args()
    print(args)

    cfg = OmegaConf.load('config.json')

    main(cfg, args.do_train)
