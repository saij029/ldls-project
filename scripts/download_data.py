"""Download models and datasets from HuggingFace."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset

# Model mappings
MODELS = {
    "smollm_135m": "HuggingFaceTB/SmolLM-135M",
    "smollm_360m": "HuggingFaceTB/SmolLM-360M",
    "smollm_1.7b": "HuggingFaceTB/SmolLM-1.7B",
}

# Dataset mappings
DATASETS = {
    "codeparrot": "codeparrot/codeparrot-clean-train",
    "code_alpaca": "sahil2801/CodeAlpaca-20k",
    "humaneval": "openai_humaneval",
    "mbpp": "mbpp",
}


def download_model(model_name: str, output_dir: str = "models/base") -> str:
    """Download model from HuggingFace.
    
    Args:
        model_name: Model name key
        output_dir: Output directory
        
    Returns:
        Path to downloaded model
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_id = MODELS[model_name]
    model_path = os.path.join(output_dir, model_name)
    
    print(f"Downloading model: {model_id}")
    print(f"Destination: {model_path}")
    
    # Download
    snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        local_dir_use_symlinks=False
    )
    
    print(f"✓ Model downloaded: {model_path}")
    return model_path


def download_dataset(
    dataset_name: str,
    output_dir: str = "data/raw",
    subset: str = None,
    split: str = "train",
    max_samples: int = None
) -> str:
    """Download dataset from HuggingFace.
    
    Args:
        dataset_name: Dataset name key
        output_dir: Output directory
        subset: Dataset subset/configuration
        split: Dataset split to download
        max_samples: Maximum samples to download (for testing)
        
    Returns:
        Path to downloaded dataset
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    dataset_id = DATASETS[dataset_name]
    dataset_path = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Destination: {dataset_path}")
    
    # Special handling for CodeParrot (large dataset)
    if dataset_name == "codeparrot":
        print("Note: CodeParrot is very large. Consider using streaming or limiting samples.")
        if max_samples is None:
            print("Setting max_samples=100000 for CodeParrot (1B token subset)")
            max_samples = 100000
    
    # Load dataset
    try:
        dataset = load_dataset(
            dataset_id,
            subset,
            split=split,
            streaming=False
        )
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")
        
        # Save as JSONL
        output_file = os.path.join(dataset_path, f"{split}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dataset:
                import json
                f.write(json.dumps(sample) + '\n')
        
        print(f"✓ Dataset downloaded: {len(dataset)} samples")
        print(f"✓ Saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        raise
    
    return dataset_path


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets")
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        help="Models to download"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/base",
        help="Directory to save models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for testing)"
    )
    
    args = parser.parse_args()
    
    # Download models
    if args.models:
        models_to_download = list(MODELS.keys()) if "all" in args.models else args.models
        
        print("=" * 80)
        print("DOWNLOADING MODELS")
        print("=" * 80)
        
        for model_name in models_to_download:
            try:
                download_model(model_name, args.model_dir)
                print()
            except Exception as e:
                print(f"Failed to download {model_name}: {e}\n")
    
    # Download datasets
    if args.datasets:
        datasets_to_download = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
        
        print("=" * 80)
        print("DOWNLOADING DATASETS")
        print("=" * 80)
        
        for dataset_name in datasets_to_download:
            try:
                download_dataset(
                    dataset_name,
                    args.data_dir,
                    max_samples=args.max_samples
                )
                print()
            except Exception as e:
                print(f"Failed to download {dataset_name}: {e}\n")
    
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
