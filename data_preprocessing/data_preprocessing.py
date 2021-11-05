from datasets import load_from_disk
import re

def preprocess(text):
    text = re.sub(r'\n', " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r'#', " ", text)

    text = re.sub(r"[“”‘’]", "\'", text)
    text = re.sub(r"[〈<＜「≪《『]", "<", text)
    text = re.sub(r"[〉>＞」≫》』]", ">", text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧｜]", "", text)

    return text


def answer_text_process(answer):
    text_list = answer['text']
    answer['text'] = list(map(preprocess, text_list))

    return answer


def run_preprocess(example):
    context = example["context"]
    start_ids = example["answers"]["answer_start"][0]

    before = context[:start_ids]
    after = context[start_ids:]

    process_before = preprocess(before)
    process_after = preprocess(after)
    process_data = process_before + process_after

    ids_move = len(before) - len(process_before)

    example["answers"] = answer_text_process(example["answers"])
    example["answers"]['text'][0] = process_data(example['answers']['text'][0])
    example["context"] = process_data
    example["answers"]["answer_start"][0] = start_ids - ids_move

    return example


def check(data_list):
    for data in data_list:
        start_ids = data["answers"]["answer_start"][0]
        end_ids = start_ids + len(data["answers"]["text"][0])
        # print(data["answers"]["text"][0], data["context"][start_ids : end_ids])
        if data["answers"]["text"][0] != data["context"][start_ids: end_ids]:
            print("wrong")
            return
    print("good")

dataset = load_from_disk("../../data/train_dataset/")
dataset['train'] = dataset['train'].map(run_preprocess)
dataset['validation'] = dataset['validation'].map(run_preprocess)
dataset.save_to_disk('./data/train_dataset/')
