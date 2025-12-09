"""Prepare and tokenize datasets for training."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm
from transformers import AutoTokenizer


def prepare_cpt_data(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    max_length: int = 2048,
    text_column: str = "content"
) -> None:
    """Prepare continuous pretraining data.
    
    Args:
        input_path: Input JSONL file
        output_path: Output directory
        tokenizer_name: Tokenizer name or path
        max_length: Maximum sequence length
        text_column: Name of text column in data
    """
    print(f"Preparing CPT data: {input_path}")
    print(f"Output: {output_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Read input data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples")
    
    # Process and save
    output_file = os.path.join(output_path, "data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(data, desc="Tokenizing"):
            # Get text
            text = sample.get("text", sample.get(text_column, ""))
            
            if not text:
                continue
            
            # Keep original text for now (tokenize on-the-fly during training)
            # For very large datasets, you might want to pre-tokenize
            output_sample = {"text": text}
            
            f.write(json.dumps(output_sample) + '\n')
    
    print(f"✓ Saved {len(data)} samples to {output_file}")
    
    # Save metadata
    metadata = {
        "num_samples": len(data),
        "max_length": max_length,
        "tokenizer": tokenizer_name,
        "text_column": text_column
    }
    
    with open(os.path.join(output_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def prepare_sft_data(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    max_length: int = 2048
) -> None:
    """Prepare supervised finetuning data.
    
    Args:
        input_path: Input JSONL file
        output_path: Output directory
        tokenizer_name: Tokenizer name or path
        max_length: Maximum sequence length
    """
    print(f"Preparing SFT data: {input_path}")
    print(f"Output: {output_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Read input data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples")
    
    # Process and save
    output_file = os.path.join(output_path, "data.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(data, desc="Processing"):
            # Extract instruction and response
            instruction = sample.get("instruction", sample.get("prompt", ""))
            response = sample.get("response", sample.get("output", sample.get("completion", "")))
            
            if not instruction or not response:
                continue
            
            output_sample = {
                "instruction": instruction,
                "response": response
            }
            
            f.write(json.dumps(output_sample) + '\n')
    
    print(f"✓ Saved {len(data)} samples to {output_file}")
    
    # Save metadata
    metadata = {
        "num_samples": len(data),
        "max_length": max_length,
        "tokenizer": tokenizer_name
    }
    
    with open(os.path.join(output_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cpt", "sft"],
        required=True,
        help="Dataset type to prepare"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="content",
        help="Name of text column (for CPT)"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "cpt":
        prepare_cpt_data(
            args.input,
            args.output,
            args.tokenizer,
            args.max_length,
            args.text_column
        )
    elif args.dataset == "sft":
        prepare_sft_data(
            args.input,
            args.output,
            args.tokenizer,
            args.max_length
        )
    
    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
