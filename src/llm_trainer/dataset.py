"""Dataset classes for CPT and SFT training."""

import os
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset

from .config import DataConfig
from .utils import get_logger


class CPTDataset(Dataset):
    """Continuous Pre-Training dataset for code.
    
    Expects pre-tokenized data or raw text files.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None
    ):
        """Initialize CPT dataset.
        
        Args:
            data_path: Path to tokenized data directory or text files
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        
        self.logger = get_logger()
        self.logger.info(f"Loading CPT dataset from {data_path}")
        
        # Load data
        self.data = self._load_data()
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        self.logger.info(f"Loaded {len(self.data)} samples")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from files.
        
        Returns:
            List of data samples
        """
        data_path = Path(self.data_path)
        
        # If directory, load all .jsonl or .txt files
        if data_path.is_dir():
            data = []
            
            # Try loading .jsonl files first
            jsonl_files = list(data_path.glob("*.jsonl"))
            if jsonl_files:
                for file in jsonl_files:
                    data.extend(self._load_jsonl(str(file)))
            
            # Try .json files
            json_files = list(data_path.glob("*.json"))
            if json_files:
                for file in json_files:
                    data.extend(self._load_json(str(file)))
            
            # Try .txt files
            txt_files = list(data_path.glob("*.txt"))
            if txt_files:
                for file in txt_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data.append({"text": line.strip()})
            
            if not data:
                raise ValueError(f"No data files found in {data_path}")
            
            return data
        
        # Single file
        elif data_path.suffix == ".jsonl":
            return self._load_jsonl(str(data_path))
        elif data_path.suffix == ".json":
            return self._load_json(str(data_path))
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_json(self, filepath: str) -> List[Dict]:
        """Load JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and single dict
        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        sample = self.data[idx]
        
        # Get text content
        text = sample.get("text", sample.get("content", ""))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM, labels = input_ids
        labels = encoding["input_ids"].clone()
        
        # Mask padding tokens in labels (-100 is ignore index)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset for instruction following.
    
    Expects data in instruction-response format.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        prompt_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ):
        """Initialize SFT dataset.
        
        Args:
            data_path: Path to instruction data
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            max_samples: Maximum number of samples
            prompt_template: Template for formatting instruction-response pairs
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        
        self.logger = get_logger()
        self.logger.info(f"Loading SFT dataset from {data_path}")
        
        # Load data
        self.data = self._load_data()
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        self.logger.info(f"Loaded {len(self.data)} instruction samples")
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load instruction-response data."""
        data_path = Path(self.data_path)
        
        # Directory of files
        if data_path.is_dir():
            data = []
            for file in data_path.glob("*.jsonl"):
                data.extend(self._load_jsonl(str(file)))
            for file in data_path.glob("*.json"):
                data.extend(self._load_json(str(file)))
            return data
        
        # Single file
        elif data_path.suffix == ".jsonl":
            return self._load_jsonl(str(data_path))
        elif data_path.suffix == ".json":
            return self._load_json(str(data_path))
        else:
            raise ValueError(f"Unsupported format: {data_path.suffix}")
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_json(self, filepath: str) -> List[Dict]:
        """Load JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single instruction-response sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        sample = self.data[idx]
        
        # Extract instruction and response
        instruction = sample.get("instruction", sample.get("prompt", ""))
        response = sample.get("response", sample.get("output", sample.get("completion", "")))
        
        # Format with template
        full_text = self.prompt_template.format(
            instruction=instruction,
            response=response
        )
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (only compute loss on response part)
        labels = encoding["input_ids"].clone()
        
        # Tokenize just the instruction part to find where response starts
        instruction_text = self.prompt_template.split("{response}")[0].format(instruction=instruction)
        instruction_encoding = self.tokenizer(
            instruction_text,
            truncation=True,
            add_special_tokens=False
        )
        
        # Mask instruction tokens in labels
        instruction_length = len(instruction_encoding["input_ids"])
        labels[0, :instruction_length] = -100
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 2048,
        text_column: str = "text",
        max_samples: Optional[int] = None
    ):
        """Initialize HuggingFace dataset wrapper.
        
        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: Tokenizer instance
            split: Dataset split to load
            max_length: Maximum sequence length
            text_column: Name of text column
            max_samples: Maximum samples to load
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        logger = get_logger()
        logger.info(f"Loading HuggingFace dataset: {dataset_name}, split: {split}")
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized sample."""
        sample = self.dataset[idx]
        text = sample[self.text_column]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def create_dataset(
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_length: int = 2048
) -> Dataset:
    """Factory function to create appropriate dataset.
    
    Args:
        data_config: Data configuration
        tokenizer: Tokenizer instance
        split: "train" or "val"
        max_length: Maximum sequence length
        
    Returns:
        Dataset instance
    """
    logger = get_logger()
    
    # Determine data path
    data_path = data_config.train_path if split == "train" else data_config.val_path
    
    if data_path is None:
        raise ValueError(f"No data path specified for split: {split}")
    
    # Create dataset based on type
    if data_config.dataset_type == "cpt":
        dataset = CPTDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=data_config.max_samples
        )
    elif data_config.dataset_type == "sft":
        dataset = SFTDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=data_config.max_samples
        )
    else:
        raise ValueError(f"Unknown dataset type: {data_config.dataset_type}")
    
    logger.info(f"Created {data_config.dataset_type.upper()} dataset: {len(dataset)} samples")
    
    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """Create DataLoader with proper settings.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for distributed training
    )
