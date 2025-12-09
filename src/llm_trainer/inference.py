"""Inference utilities for code generation."""

import os
from typing import List, Optional, Dict, Any, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .utils import get_logger


class InferenceEngine:
    """Simple inference engine for code generation."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run on
            torch_dtype: Model dtype
            load_in_8bit: Load model in 8-bit
            load_in_4bit: Load model in 4-bit
        """
        self.model_path = model_path
        self.device = device
        self.logger = get_logger()
        
        self.logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "fp16": torch.float16
        }
        torch_dtype_obj = dtype_map.get(torch_dtype.lower(), torch.bfloat16)
        
        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch_dtype_obj,
            "device_map": device if device != "cpu" else None,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
            self.logger.info("Loading in 8-bit mode")
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
            self.logger.info("Loading in 4-bit mode")
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None
    ) -> Union[str, List[str]]:
        """Generate code completion.
        
        Args:
            prompt: Input prompt(s)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling
            stop_sequences: Optional stop sequences
            
        Returns:
            Generated text(s)
        """
        # Handle single or batch input
        is_single = isinstance(prompt, str)
        if is_single:
            prompt = [prompt]
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            if temperature == 0.0 or not do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode (only new tokens)
        generated_texts = []
        for output in outputs:
            # Get only newly generated tokens
            generated_ids = output[inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Apply stop sequences if provided
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
            
            generated_texts.append(generated_text)
        
        # Return format
        if is_single and num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """Generate completions for batch of prompts.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size for generation
            **generation_kwargs: Arguments for generate()
            
        Returns:
            List of generated texts
        """
        all_generations = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            generations = self.generate(batch, **generation_kwargs)
            
            if isinstance(generations, str):
                generations = [generations]
            
            all_generations.extend(generations)
        
        return all_generations


class vLLMEngine:
    """vLLM-based inference engine (if vLLM is installed)."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048
    ):
        """Initialize vLLM engine.
        
        Args:
            model_path: Path to model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization
            max_model_len: Maximum model length
        """
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        self.logger = get_logger()
        self.logger.info(f"Initializing vLLM engine: {model_path}")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        
        self.logger.info("vLLM engine initialized")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 50,
        n: int = 1,
        stop: Optional[List[str]] = None
    ) -> Union[str, List[str]]:
        """Generate with vLLM.
        
        Args:
            prompts: Input prompt(s)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            n: Number of sequences per prompt
            stop: Stop sequences
            
        Returns:
            Generated text(s)
        """
        from vllm import SamplingParams
        
        # Handle single input
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            stop=stop
        )
        
        # Generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract texts
        generated_texts = []
        for output in outputs:
            for completion in output.outputs:
                generated_texts.append(completion.text)
        
        # Return format
        if is_single and n == 1:
            return generated_texts[0]
        else:
            return generated_texts


def load_inference_engine(
    model_path: str,
    engine_type: str = "transformers",
    **kwargs
) -> Union[InferenceEngine, vLLMEngine]:
    """Load appropriate inference engine.
    
    Args:
        model_path: Path to model
        engine_type: "transformers" or "vllm"
        **kwargs: Engine-specific arguments
        
    Returns:
        Inference engine instance
    """
    if engine_type == "transformers":
        return InferenceEngine(model_path, **kwargs)
    elif engine_type == "vllm":
        return vLLMEngine(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
