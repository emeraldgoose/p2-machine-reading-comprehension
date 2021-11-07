"""
    함수 post_processing_function을 리턴하는 함수입니다.
    train.py의 코드를 분리하기 위해 작성되었습니다.
"""

from utils_qa import postprocess_qa_predictions
from transformers import EvalPrediction


def postprocessor(data_args, datasets):
    """
        model에서 예측된 start_logits와 end_logits를 받아 context에서 답변을 찾는 함수입니다.
    """
    column_names = datasets["validation"].column_names
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        else:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    return post_processing_function
