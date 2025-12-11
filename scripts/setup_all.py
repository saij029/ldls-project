#!/usr/bin/env python3
"""
Complete setup script for LDLS project.
Downloads models and datasets, then prepares them for training.
Run on LOGIN NODE with internet access.
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    """Handle project setup including model and data downloads."""
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize project setup.
        
        Args:
            base_dir: Base directory for the project. Defaults to current directory.
        """
        self.base_dir = base_dir or Path.cwd()
        self.scripts_dir = self.base_dir / "scripts"
        self.models_dir = self.base_dir / "models" / "base"
        self.data_raw_dir = self.base_dir / "data" / "raw"
        self.data_processed_dir = self.base_dir / "data" / "processed"
        
    def validate_environment(self) -> bool:
        """
        Validate that required scripts exist.
        
        Returns:
            True if environment is valid, False otherwise.
        """
        required_scripts = [
            self.scripts_dir / "download_data.py",
            self.scripts_dir / "prepare_data.py"
        ]
        
        for script in required_scripts:
            if not script.exists():
                logger.error(f"Required script not found: {script}")
                return False
        
        logger.info("Environment validation passed")
        return True
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        Execute shell command and log results.
        
        Args:
            cmd: Command and arguments as list.
            description: Human-readable description of the command.
            
        Returns:
            True if command succeeded, False otherwise.
        """
        logger.info("=" * 70)
        logger.info(description)
        logger.info("=" * 70)
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                cwd=self.base_dir
            )
            logger.info(f"SUCCESS: {description}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FAILED: {description}")
            logger.error(f"Return code: {e.returncode}")
            return False
    
    def download_models(self) -> Tuple[int, int]:
        """
        Download all required models.
        
        Returns:
            Tuple of (successful_downloads, total_downloads).
        """
        models = ["smollm_135m", "smollm_360m", "smollm_1.7b"]
        success_count = 0
        
        logger.info("\nSTEP 1: DOWNLOADING MODELS")
        
        for model in models:
            cmd = [
                "python", 
                str(self.scripts_dir / "download_data.py"),
                "--models", model,
                "--model-dir", str(self.models_dir)
            ]
            
            if self.run_command(cmd, f"Downloading {model}"):
                success_count += 1
        
        return success_count, len(models)
    
    def download_datasets(self) -> Tuple[int, int]:
        """
        Download all required datasets.
        
        Returns:
            Tuple of (successful_downloads, total_downloads).
        """
        datasets = [
            {
                "name": "code_alpaca",
                "args": ["--datasets", "code_alpaca", "--data-dir", str(self.data_raw_dir)]
            },
            {
                "name": "codeparrot",
                "args": [
                    "--datasets", "codeparrot",
                    "--data-dir", str(self.data_raw_dir),
                    "--max-samples", "100000"
                ]
            }
        ]
        
        success_count = 0
        logger.info("\nSTEP 2: DOWNLOADING DATASETS")
        
        for dataset in datasets:
            cmd = ["python", str(self.scripts_dir / "download_data.py")] + dataset["args"]
            
            if self.run_command(cmd, f"Downloading {dataset['name']} dataset"):
                success_count += 1
        
        return success_count, len(datasets)
    
    def prepare_datasets(self) -> Tuple[int, int]:
        """
        Prepare and tokenize all datasets.
        
        Returns:
            Tuple of (successful_preparations, total_preparations).
        """
        preparations = [
            {
                "desc": "SFT training data",
                "args": [
                    "--dataset", "sft",
                    "--input", str(self.data_raw_dir / "code_alpaca" / "train.jsonl"),
                    "--output", str(self.data_processed_dir / "sft" / "train"),
                    "--tokenizer", str(self.models_dir / "smollm_360m")
                ]
            },
            {
                "desc": "SFT validation data",
                "args": [
                    "--dataset", "sft",
                    "--input", str(self.data_raw_dir / "code_alpaca" / "val.jsonl"),
                    "--output", str(self.data_processed_dir / "sft" / "val"),
                    "--tokenizer", str(self.models_dir / "smollm_360m")
                ]
            },
            {
                "desc": "CPT training data",
                "args": [
                    "--dataset", "cpt",
                    "--input", str(self.data_raw_dir / "codeparrot" / "train.jsonl"),
                    "--output", str(self.data_processed_dir / "cpt" / "train"),
                    "--tokenizer", str(self.models_dir / "smollm_360m")
                ]
            },
            {
                "desc": "CPT validation data",
                "args": [
                    "--dataset", "cpt",
                    "--input", str(self.data_raw_dir / "codeparrot" / "val.jsonl"),
                    "--output", str(self.data_processed_dir / "cpt" / "val"),
                    "--tokenizer", str(self.models_dir / "smollm_360m")
                ]
            }
        ]
        
        success_count = 0
        logger.info("\nSTEP 3: PREPARING AND TOKENIZING DATASETS")
        
        for prep in preparations:
            cmd = ["python", str(self.scripts_dir / "prepare_data.py")] + prep["args"]
            
            if self.run_command(cmd, f"Preparing {prep['desc']}"):
                success_count += 1
        
        return success_count, len(preparations)
    
    def print_summary(self, results: dict):
        """
        Print setup summary.
        
        Args:
            results: Dictionary containing setup results.
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"SETUP COMPLETE: {results['total_success']}/{results['total_steps']} steps successful")
        logger.info("=" * 70)
        
        if results['total_success'] == results['total_steps']:
            logger.info("\nAll steps completed successfully")
            
            logger.info("\nModels downloaded:")
            for model in ["smollm_135m", "smollm_360m", "smollm_1.7b"]:
                model_path = self.models_dir / model
                if model_path.exists():
                    logger.info(f"  - {model}")
            
            logger.info("\nDatasets prepared:")
            logger.info("  - SFT (CodeAlpaca):")
            logger.info("    - data/processed/sft/train")
            logger.info("    - data/processed/sft/val")
            logger.info("  - CPT (CodeParrot):")
            logger.info("    - data/processed/cpt/train")
            logger.info("    - data/processed/cpt/val")
            
            logger.info("\nNext steps:")
            logger.info("  1. SSH to compute node: ssh gpu02")
            logger.info("  2. cd ~/sai/ldls-project")
            logger.info("  3. source .venv/bin/activate")
            logger.info("  4. export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
            logger.info("  5. Start training:")
            logger.info("     python scripts/train.py \\")
            logger.info("       --config configs/training/sft.yaml \\")
            logger.info("       --model-config configs/models/smollm_360m.yaml \\")
            logger.info("       --output-dir models/checkpoints/sft/smollm_360m")
        else:
            logger.warning(f"\n{results['total_steps'] - results['total_success']} steps failed")
            logger.info("Check the logs above for errors")


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("LDLS PROJECT SETUP")
    logger.info("=" * 70)
    
    setup = ProjectSetup()
    
    # Validate environment
    if not setup.validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Track results
    results = {
        'models': (0, 0),
        'datasets': (0, 0),
        'preparations': (0, 0)
    }
    
    # Execute setup steps
    results['models'] = setup.download_models()
    results['datasets'] = setup.download_datasets()
    results['preparations'] = setup.prepare_datasets()
    
    # Calculate totals
    total_success = sum(r[0] for r in results.values())
    total_steps = sum(r[1] for r in results.values())
    
    results['total_success'] = total_success
    results['total_steps'] = total_steps
    
    # Print summary
    setup.print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if total_success == total_steps else 1)


if __name__ == "__main__":
    main()
