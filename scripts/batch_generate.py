"""Batch generation script."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm

from src.llm_trainer.inference import load_inference_engine
from src.llm_trainer.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Batch code generation")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with prompts (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file for generations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logging(log_level="INFO", rank=0)
    
    logger.info("=" * 80)
    logger.info("BATCH GENERATION")
    logger.info("=" * 80)
    
    # Load prompts
    logger.info(f"Loading prompts from: {args.input}")
    with open(args.input, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Load engine
    logger.info("Loading model...")
    engine = load_inference_engine(
        model_path=args.model,
        engine_type="transformers",
        device=args.device
    )
    
    # Generate
    logger.info("Generating...")
    generations = engine.batch_generate(
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Save
    logger.info(f"Saving to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w') as f:
        for generation in generations:
            f.write(generation + '\n\n' + '='*80 + '\n\n')
    
    logger.info(f"✓ Generated {len(generations)} completions")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
