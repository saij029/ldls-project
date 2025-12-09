"""Evaluation script for code generation benchmarks."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_trainer.evaluator import HumanEvalEvaluator, MBPPEvaluator
from src.llm_trainer.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on code benchmarks")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["humaneval", "mbpp", "all"],
        default=["humaneval"],
        help="Benchmarks to run"
    )
    parser.add_argument(
        "--humaneval-path",
        type=str,
        default="data/raw/humaneval/test.jsonl",
        help="Path to HumanEval data"
    )
    parser.add_argument(
        "--mbpp-path",
        type=str,
        default="data/raw/mbpp/test.jsonl",
        help="Path to MBPP data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per problem (for pass@k)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Limit number of problems (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_level="INFO", rank=0)
    
    logger.info("=" * 80)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Benchmarks: {args.benchmarks}")
    logger.info(f"Device: {args.device}")
    
    # Load model and tokenizer
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine benchmarks to run
    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["humaneval", "mbpp"]
    
    results = {}
    
    # Run HumanEval
    if "humaneval" in benchmarks:
        logger.info("\n" + "=" * 80)
        logger.info("HUMANEVAL EVALUATION")
        logger.info("=" * 80)
        
        evaluator = HumanEvalEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        output_path = os.path.join(args.output_dir, "humaneval_results.json")
        
        humaneval_metrics = evaluator.evaluate(
            data_path=args.humaneval_path,
            output_path=output_path,
            num_problems=args.num_problems
        )
        
        results["humaneval"] = humaneval_metrics
    
    # Run MBPP
    if "mbpp" in benchmarks:
        logger.info("\n" + "=" * 80)
        logger.info("MBPP EVALUATION")
        logger.info("=" * 80)
        
        evaluator = MBPPEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        output_path = os.path.join(args.output_dir, "mbpp_results.json")
        
        mbpp_metrics = evaluator.evaluate(
            data_path=args.mbpp_path,
            output_path=output_path,
            num_problems=args.num_problems
        )
        
        results["mbpp"] = mbpp_metrics
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    for benchmark, metrics in results.items():
        logger.info(f"\n{benchmark.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, "combined_results.json")
    from src.llm_trainer.utils import save_json
    save_json(results, combined_path)
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
