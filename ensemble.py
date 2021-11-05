import json
import heapq


def main():
    """
        model.py에 정의된 RobertaQA, BertQA, ElectraQA에 대해 inference를 진행한 결과를 가지고 앙상블을 시도합니다.
    """
    with open("../outputs/test_dataset/nbest_predictions_roberta.json") as f:
        roberta = json.load(f)

    with open("../outputs/test_dataset/nbest_predictions_bert.json") as f:
        bert = json.load(f)

    with open("../outputs/test_dataset/nbest_predictions_electra.json") as f:
        electra = json.load(f)

    query_id = list(roberta.keys())

    dic = {}
    for id in query_id:
        # 각 결과에 대해 같은 id를 가지는 답변을 모두 가져옵니다
        roberta_ = roberta.get(id)
        bert_ = bert.get(id)
        electra_ = electra.get(id)

        roberta_text = roberta_[0]["text"]
        roberta_prob = roberta_[0]["probability"]
        bert_text = bert_[0]["text"]
        bert_prob = bert_[0]["probability"]
        electra_text = electra_[0]["text"]
        electra_prob = electra_[0]["probability"]

        # 동일 표를 받게 될 경우 총 각 문서마다 가장 높은 확률을 답변 중 가장 확률이 높은 답변을 채택합니다.
        #
        pq = [
            (-roberta_prob, roberta_text),
            (-bert_prob, bert_text),
            (-electra_prob, electra_text),
        ]
        heapq.heapify(pq)
        _, text = heapq.heappop(pq)

        m = {}

        if roberta_text not in m:
            m[roberta_text] = 1
        else:
            m[roberta_text] += 1 * 2

        if bert_text not in m:
            m[bert_text] = 1
        else:
            m[bert_text] += 1

        if electra_text not in m:
            m[electra_text] = 1
        else:
            m[electra_text] += 1 * 1.5

        if len(m) == 3:
            dic[id] = text
        else:
            sorted(m.items())
            dic[id] = list(m)[0]

    # 결과를 prediction.json으로 저장합니다
    with open(
        "../outputs/test_dataset/predictions.json", "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(dic, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
