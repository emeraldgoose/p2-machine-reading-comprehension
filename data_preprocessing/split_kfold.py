from datasets import load_from_disk
from datasets import concatenate_datasets
from sklearn.model_selection import KFold

dataset = load_from_disk("../../data/train_dataset/")
dataset = concatenate_datasets([dataset['train'], dataset['validation']])

n=5
kf = KFold(n_splits=n, random_state=10, shuffle=True)

i=-1
for train_index, val_index in kf.split(dataset):
    i+=1
    dataset.select(train_index).save_to_disk(f'data/train_dataset_{i}')
    dataset.select(val_index).save_to_disk(f'data/validation_dataset_{i}')