# python retro-li/dataset.py ^
# --train_index_filepath datasets/retroli_phukhoa.index ^
# --val_index_filepath datasets/retroli_phukhoa.index ^
# --train_dataset_filepath datasets/retroli_train.json ^
# --val_dataset_filepath datasets/retroli_val.json ^
# --config_filepath retro-li/configs/retro_small_model_wikitext103-gpt2-10.json ^
# --flattened_text_tokenized_path_train None ^
# --flattened_text_tokenized_path_val None ^
# --huggingface_dataset custom_phukhoa ^
# --embedding_type multi-qa-mpnet-base-dot-v1 ^
# --truncation True
import json
from pathlib import Path
import argparse
import numpy as np
import pickle
import re
import os
import glob
import random
import torch
from torch.utils.data import Dataset as PyTorchDataset
from labml import lab, monit
from retro_index import RetroIndex
from datasets import load_dataset, concatenate_datasets
from tokenizers import normalizers
from transformers import GPT2Tokenizer
from datasets import Dataset as HFDataset

torch.random.manual_seed(42)
np.random.seed(42)
random.seed(42)
def build_dataset(train_index_filepath, val_index_filepath, 
                train_dataset_filepath, val_dataset_filepath, device, config_json, 
                flattened_text_tokenized_path_train, flattened_text_tokenized_path_val,
                huggingface_dataset, truncation, embedding_type,trainseq_split=0):
    chunk_len = config_json["chunk_len"]
    chunks_per_sample = config_json["chunks_per_sample"]
    skip_range = config_json["skip_range"]
    n_neighbors = config_json["n_neighbors"]
    n_extra = config_json["n_extra"]
    """
    ## Build the dataset

    * `chunk_len` is the chunk length in tokens
    * `chunks_per_sample` is the number of chunks per training sample
    * `skip_range` is the maximum number of tokens to skip between two samples.
        We skip a few tokens between samples to make sure the samples
        aren't aligned perfectly with the chunks in the [database](database.html).
        Thus, if skip_range = 0, we must make sure that the retrieval database
        for training and for validation are not the same.
    """

    if config_json["gpt2"] == "False":
        raise NotImplementedError("This is deprecated.")
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    normalizer = normalizers.BertNormalizer()
    text_train_tokenized = None
    train_retdb_split = 100 - trainseq_split
    if "cnn_dailymail" in huggingface_dataset:
        text_type="article"
    else:
        text_type="text"

    def tokenize_function(sentence):
        sentence[text_type] = [re.sub(' +', ' ', x) for x in sentence[text_type]]
        sentence[text_type] = [re.sub(' +', ' ', normalizer.normalize_str(x)) for x in sentence[text_type]]
        for i in range(2):
            sentence[text_type] = [re.sub(' +', ' ', re.sub("\.\.\.+","...",x)) for x in sentence[text_type]] #replace repeated periods
            sentence[text_type] = [re.sub(' +', ' ', re.sub("\-\-+","--",x)) for x in sentence[text_type]] #this happens a lot in code
            sentence[text_type] = [re.sub(' +', ' ', re.sub("\_+|\•+|\=+|\*+|\+\++|\\\+|\//+|\##+","",x)) for x in sentence[text_type]]
            sentence[text_type] = [re.sub(' +', ' ', re.sub("\!!+","!",x)) for x in sentence[text_type]]
        return gpt2_tokenizer(sentence[text_type], truncation=truncation, add_special_tokens=False)
    
    # We can't keep it as general here as we did for database_train because we want specific versions
    if "wikitext" in huggingface_dataset:
        if skip_range == 0: # we don't skip tokens, we keep the sets seperate for training.
           raise Exception("Please specify a complementary set as explained in the readme.")
        else: # if we skip tokens
            if val_dataset_filepath != "":
                text_val = load_dataset('wikitext', huggingface_dataset, cache_dir="test",ignore_verifications=True)["validation"]
                if train_dataset_filepath == "": #if no training dataset is needed, we still load train set as retrieve database
                    text_train = load_dataset('wikitext', huggingface_dataset, cache_dir="test",ignore_verifications=True)["train"]
            if train_dataset_filepath != "":
                text_train = load_dataset('wikitext', huggingface_dataset, cache_dir="test",ignore_verifications=True)["train"]
                # for wikitext training, both sequence and retrieval database are the same but offset is different
                text_train_seq = text_train
                text_train_retdb = text_train
        
    elif "slimpajama" in huggingface_dataset:
        if skip_range == 0:
           raise Exception("Please specify a complementary set as explained in the readme.")
        else:
            #text_train = load_dataset("DKYoon/SlimPajama-6B", split="train")
            with open(flattened_text_tokenized_path_train, "rb") as fb:
                text_train_tokenized = pickle.load(fb)
                text_train_tokenized = [y for x in text_train_tokenized for y in x]
                print("this is the length of text_train_tokenized")
                print(len(text_train_tokenized))
        data_files = {"validation": "test/.cache/huggingface/datasets/DKYoon___parquet/DKYoon--SlimPajama-6B-1735e67c45f712b7/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec/parquet-validation.arrow"}
        text_val = load_dataset("arrow", data_files = data_files, split="validation")
    elif "atticus_contracts" in huggingface_dataset:
        if val_dataset_filepath != "":
            text_train = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "atticus_contracts", split="train", cache_dir="test")
            text_val =  load_dataset("huggingface_download_pyfiles/pile-of-law.py", "atticus_contracts", split="validation[:5%]", cache_dir="test")
        if train_dataset_filepath != "":
            text_train_seq = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "atticus_contracts", split=f'train[-{trainseq_split}%:]', cache_dir="test")
            text_train_retdb = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "atticus_contracts", split=f'train[:{train_retdb_split}%]', cache_dir="test")
        
    elif "founding_docs" in huggingface_dataset:
        if val_dataset_filepath != "":
            text = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "founding_docs", cache_dir="test")
            text_train = text["train"]
            text_val = text["validation"]
        if train_dataset_filepath != "":
            text_train_seq = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "founding_docs", split=f'train[-{trainseq_split}%:]', cache_dir="test")
            text_train_retdb = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "founding_docs", split=f'train[:{train_retdb_split}%]', cache_dir="test")

    elif "reuters" in huggingface_dataset:
        if val_dataset_filepath != "":
            text = load_dataset("reuters21578", "ModHayes")
            text_train = text["train"]
            text_val = text["test"]
        if train_dataset_filepath != "":
            text_train_seq = load_dataset("reuters21578", "ModHayes", split=f'train[-{trainseq_split}%:]')
            text_train_retdb = load_dataset("reuters21578", "ModHayes", split=f'train[:{train_retdb_split}%]')
    elif "cnn_dailymail" in huggingface_dataset:
        if val_dataset_filepath != "":
            text = load_dataset("cnn_dailymail", "1.0.0", cache_dir="test")
            text_train = text["train"]
        text_val = text["test"]
        if train_dataset_filepath != "":
            text_train_seq = load_dataset("cnn_dailymail", "1.0.0", split=f'train[-{trainseq_split}%:]', cache_dir="test")
            text_train_retdb = load_dataset("cnn_dailymail", "1.0.0", split=f'train[:{train_retdb_split}%]', cache_dir="test")
    elif "openwebtext" in huggingface_dataset:
        if val_dataset_filepath != "":
            text = load_dataset("Skylion007/openwebtext", cache_dir="test")
            text = text["train"]
            text = text.train_test_split(test_size=0.01)
            text_val = text["test"]
            text_train = text["train"]
        if train_dataset_filepath != "":
            text_train_seq = load_dataset("Skylion007/openwebtext", split=f'train[-{trainseq_split}%:]', cache_dir="test")
            text_train_retdb = load_dataset("Skylion007/openwebtext", split=f'train[:{train_retdb_split}%]', cache_dir="test")
    elif "bbc" in huggingface_dataset:
        if val_dataset_filepath != "":
            text = load_dataset("SetFit/bbc-news")
            text_val = text["test"]
            text_train = text["train"]
        if train_dataset_filepath != "":
            text_train_seq = load_dataset("SetFit/bbc-news", split=f'train[-{trainseq_split}%:]')
            text_train_retdb = load_dataset("SetFit/bbc-news", split=f'train[:{train_retdb_split}%]')
    elif "custom_phukhoa" in huggingface_dataset:
        # ✅ Load dữ liệu từ local JSONL thay vì HuggingFace
        print("Loading local dataset: datasets/passages.jsonl")
        local_path = Path("datasets/passages.jsonl")
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} not found!")

        # đọc từng dòng JSONL (mỗi dòng là {"text": "..."} hoặc plain text line)
        lines = []
        with open(local_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    # nếu là JSON object, lấy key "text" nếu có
                    obj = json.loads(ln)
                    if isinstance(obj, dict) and "text" in obj:
                        lines.append(obj["text"])
                    elif isinstance(obj, str):
                        lines.append(obj)
                    else:
                        lines.append(str(obj))
                except json.JSONDecodeError:
                    # nếu không parse được JSON, coi cả dòng là text
                    lines.append(ln)

        print(f"✅ Loaded {len(lines)} passages from {local_path}")

        # đưa vào dataset dạng HuggingFace-compatible
        text_train_seq = HFDataset.from_dict({"text": lines})
        text_train = text_train_seq
        text_train_retdb = HFDataset.from_dict({"text": lines})
        text_val = HFDataset.from_dict({"text": lines})
        text_type = "text"

    else:
        try:
            text = load_dataset(huggingface_dataset, split='train')
        except Exception:
            raise Exception("Please specify a valid huggingface dataset.")


    # --- chuẩn hoá flattened paths nếu người dùng truyền "None" hoặc ""
    if flattened_text_tokenized_path_train in [None, "None", ""]:
        flattened_text_tokenized_path_train = "cache_train"
    if flattened_text_tokenized_path_val in [None, "None", ""]:
        flattened_text_tokenized_path_val = "cache_val"

    # ensure folders exist
    os.makedirs(flattened_text_tokenized_path_train, exist_ok=True)
    os.makedirs(flattened_text_tokenized_path_val, exist_ok=True)

    if train_dataset_filepath != "": 
        # tokenize train retrieval database (flattened)
        train_retrieval_database_tokens_path = os.path.join(flattened_text_tokenized_path_train, "train_retrieval_database_tokens.pkl")
        traintok_file = Path(train_retrieval_database_tokens_path)
        if traintok_file.is_file():
            with open(train_retrieval_database_tokens_path, "rb") as fb:
                train_retrieval_database_tokens = pickle.load(fb)
            print(f"✅ Loaded cached train retrieval tokens ({len(train_retrieval_database_tokens)} tokens)")
        else:
            print("🔹 Tokenizing retrieval DB (this may take a while)...")
            # .map với num_proc có thể gây vấn đề trên windows; mặc định 1 nếu lỗi
            try:
                text_train_tokenized = text_train_retdb.map(tokenize_function, batched=True, num_proc=1, remove_columns=[text_type])
                text_train_tokenized = [sample["input_ids"] for sample in text_train_tokenized if sample.get("input_ids")]
                train_retrieval_database_tokens = [y for x in text_train_tokenized for y in x]
            except Exception as e:
                # fallback: token hoá theo từng dòng để dễ debug
                print("⚠️ .map failed, fallback to per-sample tokenization:", e)
                train_retrieval_database_tokens = []
                for s in text_train_retdb:
                    toks = tokenize_function({"text":[s["text"]]})
                    if toks and "input_ids" in toks:
                        train_retrieval_database_tokens.extend(toks["input_ids"])
            with open(train_retrieval_database_tokens_path, "wb") as fp:
                pickle.dump(train_retrieval_database_tokens, fp)
            print(f"✅ Tokenized retrieval DB -> {len(train_retrieval_database_tokens)} tokens saved to {train_retrieval_database_tokens_path}")

        # tokenize train sequence
        train_seq_tokens_path = os.path.join(flattened_text_tokenized_path_train, "train_seq_tokens.pkl")
        traintok_file = Path(train_seq_tokens_path)
        if traintok_file.is_file():
            with open(train_seq_tokens_path, "rb") as fb:
                text_train_tokenized = pickle.load(fb)
            print(f"✅ Loaded cached train sequence tokens ({len(text_train_tokenized)} tokens)")
        else:
            print("🔹 Tokenizing train sequences...")
            try:
                seq_toks = text_train_seq.map(tokenize_function, batched=True, num_proc=1, remove_columns=[text_type])
                text_train_tokenized = [sample["input_ids"] for sample in seq_toks if sample.get("input_ids")]
                text_train_tokenized = [y for x in text_train_tokenized for y in x]  # flatten
            except Exception as e:
                print("⚠️ .map failed for train_seq, fallback to per-sample tokenization:", e)
                text_train_tokenized = []
                for s in text_train_seq:
                    toks = tokenize_function({"text":[s["text"]]})
                    if toks and "input_ids" in toks:
                        text_train_tokenized.extend(toks["input_ids"])
            with open(train_seq_tokens_path, "wb") as fp:
                pickle.dump(text_train_tokenized, fp)
            print(f"✅ Tokenized train sequences -> {len(text_train_tokenized)} tokens saved to {train_seq_tokens_path}")

        train_retrieval_database_tokens = train_retrieval_database_tokens if 'train_retrieval_database_tokens' in locals() else []
        train_index = RetroIndex(device, train_index_filepath, config_json, embedding_type)

    # --- BUILD SAMPLES ---
    if train_dataset_filepath != "":
        sample_offsets_train = []
        i = 0
        L = len(text_train_tokenized)
        print(f"🔢 Building samples: total tokens = {L}, chunk_len={chunk_len}, chunks_per_sample={chunks_per_sample}")
        # --- allow final partial segment instead of dropping it ---
        while i < L:
            if skip_range != 0:
                skip = np.random.randint(skip_range)
                i += skip
                if i >= L:
                    break
            # compute end (allow partial final)
            end = min(i + chunks_per_sample * chunk_len, L)
            # require at least 1 token to form a sample
            if end - i >= 1:
                sample_offsets_train.append(i)
            else:
                break
            i = end  # move to next block

        print(f"✅ Created {len(sample_offsets_train)} sample offsets for training")

        sequences_train = []
        is_dump_segments = False
        dump_sequence_segment = []
        cnt = 0
        totalcnt = len(sample_offsets_train)
        if totalcnt == 0 and L > 0:
            # fallback: create one sample from start if nothing created
            sample_offsets_train = [0]
            totalcnt = 1

        for i in monit.iterate('Gather Neighbors', sample_offsets_train):
            # Get the sample including an extra token (for prediction)
            sample = text_train_tokenized[i: i + chunks_per_sample * chunk_len + 1]
            if len(sample) < 2:
                # skip ridiculously small ones
                continue
            src = sample[:-1]
            chunks = [src[j:j + chunk_len] for j in range(0, len(src), chunk_len)]
            chunk_offsets = None if skip_range == 0 else [k + i for k in range(0, len(src), chunk_len)]
            D, neighbor_offsets = train_index(chunks, None)
            D = D.tolist()
            neighbors = [[train_retrieval_database_tokens[j: j + chunk_len * 2] for j in n_off] for n_off in neighbor_offsets]
            sequences_train.append((sample[:-1], sample[1:], neighbors, D))
            cnt += 1

        print(f"🔹 Gathered neighbors for {len(sequences_train)} sequences (requested {totalcnt})")

        # write out the json (as single JSON array)
        with open(train_dataset_filepath, 'w', encoding='utf-8') as f:
            json.dump(sequences_train, f, ensure_ascii=False)
        print(f"💾 Train dataset written to {train_dataset_filepath} with {len(sequences_train)} samples")
    
    # Validation json
    if val_dataset_filepath != "": # so we want to save a val dataset
        os.makedirs(flattened_text_tokenized_path_val, exist_ok=True)
        val_retrieval_database_tokens_path = flattened_text_tokenized_path_val+"/val_retrieval_database_tokens"
        valtok_file = Path(val_retrieval_database_tokens_path)
        if valtok_file.is_file():
            with open(val_retrieval_database_tokens_path, "rb") as fb:
                val_retrieval_database_tokens = pickle.load(fb)
        else:
            text_train_tokenized = text_train.map(tokenize_function, batched=True, num_proc=8, remove_columns=[text_type])
            text_train_tokenized = [sample["input_ids"] for sample in text_train_tokenized if sample["input_ids"]!=[]]
            text_train_tokenized = [y for x in text_train_tokenized for y in x] #flatten it.
            val_retrieval_database_tokens = text_train_tokenized
            with open(val_retrieval_database_tokens_path, "wb") as fp:
                pickle.dump(val_retrieval_database_tokens, fp)
        
        val_seq_tokens_path = flattened_text_tokenized_path_val+"/val_seq_tokens"
        valtok_file = Path(val_seq_tokens_path)
        if valtok_file.is_file():
            with open(val_seq_tokens_path, "rb") as fb:
                text_val_tokenized = pickle.load(fb)
        else:
            text_val_tokenized = text_val.map(tokenize_function, batched=True, num_proc=8, remove_columns=[text_type])
            text_val_tokenized = [sample["input_ids"] for sample in text_val_tokenized if sample["input_ids"]!=[]]
            text_val_tokenized = [y for x in text_val_tokenized for y in x]
            text_val_tokenized = {huggingface_dataset: {"text_val_tokenized": text_val_tokenized}} #this is so you can create several validation datasets from the same index at once if you want.
            #the idea was that in each "elif xx in huggingface dataset" you would tokenize and flatten the text, then add a key to this dict. since we don't do that and it would lead to more lines of code, I reverted these changes.
            with open(val_seq_tokens_path, "wb") as fp:
                    pickle.dump(text_val_tokenized, fp)

    if val_dataset_filepath != "": # so we want to save a val dataset
        val_index = RetroIndex(device, val_index_filepath, config_json,embedding_type)
        for key in text_val_tokenized.keys(): #keep the validation datasets separate
            sample_offsets_val = []
            j = 0
            while j < len(text_val_tokenized[key]["text_val_tokenized"]):
                # no need to to skip tokens for validation since the retrieval db is train
                # Stop if we've reached the end of the text
                if j + chunks_per_sample * chunk_len > len(text_val_tokenized[key]["text_val_tokenized"]):
                    break
                # Collect the offset
                sample_offsets_val.append(j)
                # Increment the cursor
                j += chunks_per_sample * chunk_len
            text_val_tokenized[key].setdefault("sample_offsets_val", sample_offsets_val)

        for key in text_val_tokenized.keys(): #keep the validation datasets separate
            ## val ##
            sequences_val = []

            for j in monit.iterate('Gather Val Neighbors', text_val_tokenized[key]["sample_offsets_val"]):
                # Get the sample including an extra token (for prediction)
                sample = text_val_tokenized[key]["text_val_tokenized"][j: j + chunks_per_sample * chunk_len + 1]
                # The input
                src = sample[:-1]
                # Break it into chunks
                chunks = [src[k:k + chunk_len] for k in range(0, len(src), chunk_len)]
                
                # The chunk offsets are None for validation since the sets are disjoint.
                chunk_offsets = None
                # Retrieve nearest neighbors
                D, neighbor_offsets = val_index(chunks, chunk_offsets) #D is the distance matrix
                D = D.tolist()
                neighbors = [[val_retrieval_database_tokens[j: j + chunk_len * 2] for j in n_off] for n_off in neighbor_offsets]
                
                sequences_val.append((sample[:-1], sample[1:], neighbors, D))
                
            path = val_dataset_filepath.replace(".json", "-"+str(key)+".json")
            with open(path, 'w') as f:
                f.write(json.dumps(sequences_val))
    
class Dataset(PyTorchDataset):
    """
    ## Dataset

    This is the PyTorch dataset that loads the dataset created
    by `build_dataset`.
    """
    def __init__(self, file_path: Path):
        """
        * `file_path` is the path of the saved JSON file
        """
        file_list=file_path.replace(" ", "").split(",")
        self.samples = []
        for fname in file_list:
            if fname.endswith("json"):
                with open(str(fname), 'r') as f:
                    self.samples.extend(json.loads(f.read()))
            elif fname.endswith("pkl"):
                with open(fname, "rb") as fb:
                    self.samples.extend(pickle.load(fb))

    def __len__(self):
        """
        Number of samples
        """
        return len(self.samples)

    PAD_TOKEN = 0  # hoặc token bạn dùng để padding

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        src = torch.tensor(s[0], dtype=torch.long)
        tgt = torch.tensor(s[1], dtype=torch.long)

        # padding src và tgt
        max_len = 1024  # ví dụ
        if len(src) < max_len:
            src = torch.cat([src, torch.full((max_len - len(src),), PAD_TOKEN, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.full((max_len - len(tgt),), PAD_TOKEN, dtype=torch.long)])

        ns = [torch.stack([torch.tensor(n, dtype=torch.long) for n in chunks]) for chunks in s[2]]
        neighbors = torch.stack(ns)

        if len(s) == 4:
            ds = [torch.stack([torch.tensor(n) for n in chunks]) for chunks in s[3]]
            D = torch.stack(ds)
            return src, tgt, neighbors, D
        else:
            return src, tgt, neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_index_filepath", help="Path and filename of the train index.",default="")
    parser.add_argument("--val_index_filepath", help="Path and filename of the val index.",default="")
    parser.add_argument("--train_dataset_filepath", help="Path where to save the train dataset.",default="")
    parser.add_argument("--val_dataset_filepath", help="Path where to save the val dataset.",default="")
    parser.add_argument("--config_filepath", help="Path to the config json file.")
    parser.add_argument("--flattened_text_tokenized_path_train", help="Path to the flattened tokenized text used for train retrieval database.", default="cache")
    parser.add_argument("--flattened_text_tokenized_path_val", help="Path to the flattened tokenized text used for validation retrieval database.", default="cache")
    parser.add_argument("--huggingface_dataset", help="Specify which huggingface dataset you would like to load.", default=None)
    parser.add_argument("--truncation", help="Whether or not to truncate the tokenizer input to 1024 tokens.", default=True)
    parser.add_argument("--embedding_type", help="Which embedding to use: bert or SentenceTransformer variants.", default="multi-qa-mpnet-base-dot-v1")
    
    args = parser.parse_args()
    config_json = json.load(open(args.config_filepath))
    truncation = False if args.truncation=="False" else True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if "minifit_retdb_split" in config_json:
        trainseq_split = 100-config_json["minifit_retdb_split"]
    else:
        trainseq_split = 0
    print("device: "+str(device))
    build_dataset(train_index_filepath=args.train_index_filepath, 
                    val_index_filepath=args.val_index_filepath, train_dataset_filepath=args.train_dataset_filepath, 
                    val_dataset_filepath=args.val_dataset_filepath, device=device, config_json=config_json, 
                    flattened_text_tokenized_path_train=args.flattened_text_tokenized_path_train, 
                    flattened_text_tokenized_path_val=args.flattened_text_tokenized_path_val, 
                    huggingface_dataset=args.huggingface_dataset, truncation=truncation, embedding_type = args.embedding_type,trainseq_split=trainseq_split)