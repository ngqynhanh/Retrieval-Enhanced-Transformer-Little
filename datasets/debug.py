import json

train_dataset_filepath = "datasets/retroli_train.jsonl"
val_dataset_filepath = "datasets/retroli_val-custom_phukhoa.jsonl"

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl(train_dataset_filepath)
val_data = load_jsonl(val_dataset_filepath)

print("Train dataset length:", len(train_data))
print("Val dataset length:", len(val_data))