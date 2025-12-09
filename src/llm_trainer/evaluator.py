"""Evaluation module for code generation benchmarks."""

import os
import json
import time
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import multiprocessing as mp

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .utils import get_logger, save_json


class CodeExecutor:
    """Safe code execution with timeout."""
    
    def __init__(self, timeout: int = 5):
        """Initialize code executor.
        
        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout
        self.logger = get_logger()
    
    def execute_code(self, code: str, test_code: str) -> Tuple[bool, str]:
        """Execute code with test cases.
        
        Args:
            code: Generated code
            test_code: Test code to run
            
        Returns:
            Tuple of (success, error_message)
        """
        # Combine code and test
        full_code = f"{code}\n\n{test_code}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Check if successful
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
        
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        
        except Exception as e:
            return False, str(e)
        
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass


class HumanEvalEvaluator:
    """Evaluator for HumanEval benchmark."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        num_samples: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512
    ):
        """Initialize HumanEval evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            device: Device to run on
            num_samples: Number of samples per problem (for pass@k)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.logger = get_logger()
        
        self.executor = CodeExecutor(timeout=5)
    
    def load_humaneval(self, data_path: str) -> List[Dict]:
        """Load HumanEval dataset.
        
        Args:
            data_path: Path to HumanEval JSONL file
            
        Returns:
            List of problems
        """
        problems = []
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems
    
    def generate_completion(self, prompt: str) -> str:
        """Generate code completion.
        
        Args:
            prompt: Code prompt
            
        Returns:
            Generated completion
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            if self.temperature == 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode (only new tokens)
        completion = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return completion
    
    def evaluate_problem(self, problem: Dict) -> Dict[str, Any]:
        """Evaluate single problem.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Evaluation result
        """
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]
        
        results = []
        
        # Generate multiple samples
        for i in range(self.num_samples):
            # Generate completion
            completion = self.generate_completion(prompt)
            
            # Combine prompt and completion
            full_code = prompt + completion
            
            # Execute test
            success, error = self.executor.execute_code(full_code, test)
            
            results.append({
                "completion": completion,
                "passed": success,
                "error": error
            })
        
        # Calculate pass rate
        num_passed = sum(1 for r in results if r["passed"])
        
        return {
            "task_id": task_id,
            "prompt": prompt,
            "results": results,
            "num_passed": num_passed,
            "num_total": self.num_samples,
            "pass_rate": num_passed / self.num_samples
        }
    
    def evaluate(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        num_problems: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate on HumanEval.
        
        Args:
            data_path: Path to HumanEval data
            output_path: Optional path to save detailed results
            num_problems: Optional limit on number of problems
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Starting HumanEval evaluation")
        
        # Load problems
        problems = self.load_humaneval(data_path)
        
        if num_problems:
            problems = problems[:num_problems]
        
        # Evaluate each problem
        all_results = []
        
        for problem in tqdm(problems, desc="Evaluating HumanEval"):
            result = self.evaluate_problem(problem)
            all_results.append(result)
        
        # Calculate metrics
        total_passed = sum(r["num_passed"] for r in all_results)
        total_samples = sum(r["num_total"] for r in all_results)
        
        pass_at_1 = total_passed / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "pass@1": pass_at_1,
            "num_problems": len(problems),
            "num_samples": self.num_samples,
            "temperature": self.temperature
        }
        
        self.logger.info(f"HumanEval Results: pass@1 = {pass_at_1:.2%}")
        
        # Save detailed results
        if output_path:
            save_json({
                "metrics": metrics,
                "results": all_results
            }, output_path)
            self.logger.info(f"Saved results to {output_path}")
        
        return metrics


class MBPPEvaluator:
    """Evaluator for MBPP benchmark."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        num_samples: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512
    ):
        """Initialize MBPP evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            device: Device to run on
            num_samples: Number of samples per problem
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.logger = get_logger()
        
        self.executor = CodeExecutor(timeout=5)
    
    def load_mbpp(self, data_path: str) -> List[Dict]:
        """Load MBPP dataset.
        
        Args:
            data_path: Path to MBPP JSONL file
            
        Returns:
            List of problems
        """
        problems = []
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(problems)} MBPP problems")
        return problems
    
    def generate_solution(self, prompt: str) -> str:
        """Generate solution code.
        
        Args:
            prompt: Problem description
            
        Returns:
            Generated solution
        """
        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            if self.temperature == 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode
        solution = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return solution
    
    def evaluate_problem(self, problem: Dict) -> Dict[str, Any]:
        """Evaluate single MBPP problem.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Evaluation result
        """
        task_id = problem.get("task_id", problem.get("id"))
        text = problem["text"]
        test_list = problem["test_list"]
        
        results = []
        
        # Generate multiple samples
        for i in range(self.num_samples):
            # Generate solution
            solution = self.generate_solution(text)
            
            # Create test code
            test_code = "\n".join(test_list)
            
            # Execute
            success, error = self.executor.execute_code(solution, test_code)
            
            results.append({
                "solution": solution,
                "passed": success,
                "error": error
            })
        
        # Calculate pass rate
        num_passed = sum(1 for r in results if r["passed"])
        
        return {
            "task_id": task_id,
            "text": text,
            "results": results,
            "num_passed": num_passed,
            "num_total": self.num_samples,
            "pass_rate": num_passed / self.num_samples
        }
    
    def evaluate(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        num_problems: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate on MBPP.
        
        Args:
            data_path: Path to MBPP data
            output_path: Optional path to save results
            num_problems: Optional limit on problems
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Starting MBPP evaluation")
        
        # Load problems
        problems = self.load_mbpp(data_path)
        
        if num_problems:
            problems = problems[:num_problems]
        
        # Evaluate
        all_results = []
        
        for problem in tqdm(problems, desc="Evaluating MBPP"):
            result = self.evaluate_problem(problem)
            all_results.append(result)
        
        # Calculate metrics
        total_passed = sum(r["num_passed"] for r in all_results)
        total_samples = sum(r["num_total"] for r in all_results)
        
        pass_at_1 = total_passed / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "pass@1": pass_at_1,
            "num_problems": len(problems),
            "num_samples": self.num_samples,
            "temperature": self.temperature
        }
        
        self.logger.info(f"MBPP Results: pass@1 = {pass_at_1:.2%}")
        
        # Save results
        if output_path:
            save_json({
                "metrics": metrics,
                "results": all_results
            }, output_path)
            self.logger.info(f"Saved results to {output_path}")
        
        return metrics


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k for pass@k
        
    Returns:
        pass@k score
    """
    if n - c < k:
        return 1.0
    
    return 1.0 - (
        np.prod([1.0 - k / (n - i) for i in range(c)])
    )
