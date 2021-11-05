import logging
import os
import sys
import gc

import torch
from datasets import load_metric, load_from_disk

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import check_no_error
from trainer_qa import QuestionAnsweringTrainer

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

import modelList
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

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocessor(data_args, datasets),
        compute_metrics=compute_metrics,
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
    # set cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set training arguments
    # eval_step마다 validation set을 이용해 측정된 EM을 기준으로 가장 좋은 모델이 저장됩니다
    training_args = TrainingArguments(
        output_dir=f"./models/train_dataset/{model_args.model_name_or_path}",
        report_to="wandb",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=5,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
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

    # load dataset
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # load tokenizer, model
    tokenizer, model = modelList.init(model_args=model_args)

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
    # training arguments의 hyperparameters
    # wandb sweep을 활성화하면 sweep.yaml 설정에 따라 Fine tuning을 진행합니다
    defaults = dict(learning_rate=1e-5, epochs=2, weight_decay=0.009)

    wandb.init(
        config=defaults, project="", entity="", name="",
    )

    wandb.agent(entity="", project="", sweep_id="")

    config = wandb.config

    main(config)
