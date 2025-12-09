"""Serve model with inference API."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_trainer.inference import load_inference_engine
from src.llm_trainer.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Serve model for inference")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference engine to use"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logging(log_level="INFO", rank=0)
    
    logger.info("=" * 80)
    logger.info("STARTING INFERENCE SERVER")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Engine: {args.engine}")
    logger.info(f"Address: http://{args.host}:{args.port}")
    
    # Load engine
    logger.info("Loading model...")
    engine = load_inference_engine(
        model_path=args.model,
        engine_type=args.engine,
        device=args.device,
        load_in_8bit=args.load_in_8bit
    )
    
    logger.info("Model loaded successfully!")
    
    # Interactive mode
    logger.info("\n" + "=" * 80)
    logger.info("INTERACTIVE MODE")
    logger.info("=" * 80)
    logger.info("Enter prompts (or 'quit' to exit):\n")
    
    while True:
        try:
            prompt = input(">>> ")
            
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            
            if not prompt.strip():
                continue
            
            # Generate
            logger.info("Generating...")
            output = engine.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True
            )
            
            print(f"\n{output}\n")
            print("-" * 80)
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("\nShutting down...")


if __name__ == "__main__":
    main()
