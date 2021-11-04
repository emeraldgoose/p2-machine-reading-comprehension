import gc
import logging
import os
from re import L
import sys

import torch
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, adamw

from transformers import (
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

import tokenizerAndModel
import make_dataset
from postprocessing import postprocessor

import wandb


logger = logging.getLogger(__name__)


def train(
    training_args,
    model_args,
    data_args,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    datasets,
    last_checkpoint,
    data_collator,
    compute_metrics,
):
    # Garbage Collector와 gpu 캐쉬 비우기
    gc.collect()
    torch.cuda.empty_cache()

    # optimizer, lr_scheduler
    # adamW = AdamW(model.parameters(),lr=1e-5,weight_decay=0.009)
    # reduceLROnPlateau = ReduceLROnPlateau(adamW, patience=3)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocessor(training_args, data_args, datasets),
        compute_metrics=compute_metrics,
        # optimizers=(adamW, reduceLROnPlateau)
    )

    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args_filename="do_train_arguments.txt"
    )

    # set training arguments
    training_args = TrainingArguments(
        output_dir="./models/train_dataset/",
        # report_to="wandb",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
    )

    data_args = DataTrainingArguments(
        dataset_name="../data/train_dataset",
        overwrite_cache=False,
        preprocessing_num_workers=8,
        max_seq_length=config.max_seq_length,
        pad_to_max_length=config.pad_to_max_length,
        max_answer_length=config.max_answer_length,
    )

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(42)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    tokenizer, model = tokenizerAndModel.init(model_args=model_args)
    wandb.watch(model)
    model.to(device)

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    train_dataset, eval_dataset = make_dataset.make_dataset(
        data_args, datasets, tokenizer, max_seq_length
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # train
    train(
        training_args,
        model_args,
        data_args,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        datasets,
        last_checkpoint,
        data_collator,
        compute_metrics,
    )


if __name__ == "__main__":
    defaults = dict(
        learning_rate=1e-5,
        epochs=2,
        weight_decay=0.009,
        pad_to_max_length=True,
        max_answer_length=30,
        max_seq_length=200,
    )
    wandb.init(config=defaults)
    config = wandb.config
    main(config)
