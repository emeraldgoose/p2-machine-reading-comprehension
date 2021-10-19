from preprocessing import eval_preprocessor, train_preprocessor


def make_dataset(
    training_args, data_args, datasets, tokenizer, max_seq_length, column_names
):
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_trai장n requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            train_preprocessor(
                tokenizer,
                data_args,
                max_seq_length,
                pad_on_right,
                question_column_name,
                context_column_name,
                answer_column_name,
            ),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        return train_dataset

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            eval_preprocessor(
                tokenizer,
                data_args,
                max_seq_length,
                pad_on_right,
                question_column_name,
                context_column_name,
                answer_column_name,
            ),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        return eval_dataset
