from datasets import load_dataset
import json
from tqdm import tqdm

# Download Python subset of The Stack
dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)

# Convert to JSONL
with open("code_data.jsonl", "w") as f:
    for i, record in enumerate(tqdm(dataset, desc="Downloading")):
        if i >= 100000:  # Limit for testing
            break
        json.dump({"text": record["content"]}, f)
        f.write("\n")

