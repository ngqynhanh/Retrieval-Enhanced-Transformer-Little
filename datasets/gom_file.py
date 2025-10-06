import pickle, json

# Đọc file pkl
with open("datasets/retroli_train_merged.pkl", "rb") as f:
    train_data = pickle.load(f)

# Nếu là list of dict thì có thể ghi trực tiếp ra JSON
with open("datasets/retroli_train_merged.jsonl", "w", encoding="utf-8") as f:
    for ex in train_data:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("✅ Converted train.pkl → train.json")