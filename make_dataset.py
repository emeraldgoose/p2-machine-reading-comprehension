from preprocessing import eval_preprocessor, train_preprocessor


def make_dataset(data_args, datasets, tokenizer, max_seq_length):
    """
        dataset을 받아 train dataset과 eval dataset을 만들어 리턴하는 함수입니다
    """

    # train dataset
    train_column_names = datasets["train"].column_names

    train_question_column_name = (
        "question" if "question" in train_column_names else train_column_names[0]
    )
    train_context_column_name = (
        "context" if "context" in train_column_names else train_column_names[1]
    )
    train_answer_column_name = (
        "answers" if "answers" in train_column_names else train_column_names[2]
    )

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    train_dataset = datasets["train"]

    # Train feature 생성
    train_dataset = train_dataset.map(
        train_preprocessor(
            tokenizer,
            data_args,
            max_seq_length,
            pad_on_right,
            train_question_column_name,
            train_context_column_name,
            train_answer_column_name,
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # eval dataset
    eval_column_names = datasets["validation"].column_names

    eval_question_column_name = (
        "question" if "question" in eval_column_names else eval_column_names[0]
    )
    eval_context_column_name = (
        "context" if "context" in eval_column_names else eval_column_names[1]
    )
    eval_answer_column_name = (
        "answers" if "answers" in eval_column_names else eval_column_names[2]
    )

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        eval_preprocessor(
            tokenizer,
            data_args,
            max_seq_length,
            pad_on_right,
            eval_question_column_name,
            eval_context_column_name,
            eval_answer_column_name,
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    return train_dataset, eval_dataset
