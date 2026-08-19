"""Built-in model catalog for RTX 5090 (32 GB VRAM).

A curated list of models known to work well with vLLM on 32 GB consumer GPUs.
All entries use NVFP4 quantization from unsloth for optimal VRAM efficiency.

To extend: add entries to CATALOG below. The dashboard shows these as
one-click "Add to Registry" cards.
"""
from __future__ import annotations

from typing import List, Optional

CATALOG: List[dict] = [
    {
        "id": "qwen3.8-27b-nvfp4",
        "name": "Qwen3.8 27B",
        "repo": "unsloth/Qwen3.8-27B-NVFP4",
        "description": "Hybrid MoE reasoning model (27B total, 3.8B active). "
                       "Excellent quality with low VRAM thanks to sparse activation.",
        "vram_gb": 16,
        "params": "27B (3.8B active)",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["reasoning", "tool-calling", "multilingual", "moe"],
        "suggested_alias": "qwen3.8-27b-nvfp4",
        "suggested_served_name": "qwen3.8-27b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "65536",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
            "enable-auto-tool-choice": "true",
            "tool-call-parser": "hermes",
        },
    },
    {
        "id": "qwen3-32b-nvfp4",
        "name": "Qwen3 32B",
        "repo": "unsloth/Qwen3-32B-NVFP4",
        "description": "Largest dense Qwen3 that fits in 32 GB. "
                       "Strong reasoning and instruction following.",
        "vram_gb": 18,
        "params": "32B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["reasoning", "tool-calling", "multilingual"],
        "suggested_alias": "qwen3-32b-nvfp4",
        "suggested_served_name": "qwen3-32b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "32768",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
            "enable-auto-tool-choice": "true",
            "tool-call-parser": "hermes",
        },
    },
    {
        "id": "qwen3-14b-nvfp4",
        "name": "Qwen3 14B",
        "repo": "unsloth/Qwen3-14B-NVFP4",
        "description": "Fast mid-size model. Good balance of speed and quality "
                       "for everyday tasks.",
        "vram_gb": 8,
        "params": "14B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["reasoning", "tool-calling", "multilingual"],
        "suggested_alias": "qwen3-14b-nvfp4",
        "suggested_served_name": "qwen3-14b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "65536",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
            "enable-auto-tool-choice": "true",
            "tool-call-parser": "hermes",
        },
    },
    {
        "id": "qwen3-8b-nvfp4",
        "name": "Qwen3 8B",
        "repo": "unsloth/Qwen3-8B-NVFP4",
        "description": "Compact and fast. Surprisingly capable for its size.",
        "vram_gb": 5,
        "params": "8B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["reasoning", "tool-calling", "multilingual"],
        "suggested_alias": "qwen3-8b-nvfp4",
        "suggested_served_name": "qwen3-8b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "131072",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
            "enable-auto-tool-choice": "true",
            "tool-call-parser": "hermes",
        },
    },
    {
        "id": "qwen3-4b-nvfp4",
        "name": "Qwen3 4B",
        "repo": "unsloth/Qwen3-4B-NVFP4",
        "description": "Tiny and very fast. Good for simple tasks and testing.",
        "vram_gb": 3,
        "params": "4B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["fast", "multilingual"],
        "suggested_alias": "qwen3-4b-nvfp4",
        "suggested_served_name": "qwen3-4b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "131072",
            "gpu-memory-utilization": "0.90",
            "enforce-eager": "true",
        },
    },
    {
        "id": "qwen2.5-coder-32b-nvfp4",
        "name": "Qwen2.5 Coder 32B",
        "repo": "unsloth/Qwen2.5-Coder-32B-Instruct-NVFP4",
        "description": "Specialized coding model. Strong at code generation, "
                       "review, and debugging.",
        "vram_gb": 18,
        "params": "32B",
        "quantization": "NVFP4",
        "category": "coding",
        "tags": ["coding", "code-generation", "debugging"],
        "suggested_alias": "qwen2.5-coder-32b-nvfp4",
        "suggested_served_name": "qwen2.5-coder-32b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "32768",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
        },
    },
    {
        "id": "gemma-3-27b-it-nvfp4",
        "name": "Gemma 3 27B IT",
        "repo": "unsloth/gemma-3-27b-it-NVFP4",
        "description": "Google's 27B instruction-tuned model. "
                       "Competitive with larger models on many benchmarks.",
        "vram_gb": 16,
        "params": "27B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["instruction-following", "multilingual"],
        "suggested_alias": "gemma-3-27b-it-nvfp4",
        "suggested_served_name": "gemma-3-27b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "65536",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
        },
    },
    {
        "id": "phi-4-nvfp4",
        "name": "Phi-4",
        "repo": "unsloth/Phi-4-NVFP4",
        "description": "Microsoft's 14B model. Strong reasoning and math.",
        "vram_gb": 8,
        "params": "14B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["reasoning", "math", "science"],
        "suggested_alias": "phi-4-nvfp4",
        "suggested_served_name": "phi-4",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "65536",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
        },
    },
    {
        "id": "llama-3.1-8b-instruct-nvfp4",
        "name": "Llama 3.1 8B Instruct",
        "repo": "unsloth/Llama-3.1-8B-Instruct-NVFP4",
        "description": "Meta's 8B instruction model. Widely used, well-tested.",
        "vram_gb": 5,
        "params": "8B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["instruction-following", "tool-calling"],
        "suggested_alias": "llama-3.1-8b-instruct-nvfp4",
        "suggested_served_name": "llama-3.1-8b-instruct",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "131072",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
            "enable-auto-tool-choice": "true",
            "tool-call-parser": "llama3_json",
        },
    },
    {
        "id": "mistral-small-3.1-24b-nvfp4",
        "name": "Mistral Small 3.1 24B",
        "repo": "unsloth/Mistral-Small-3.1-24B-Instruct-2503-NVFP4",
        "description": "Mistral's 24B instruction model. Excellent function calling.",
        "vram_gb": 14,
        "params": "24B",
        "quantization": "NVFP4",
        "category": "general",
        "tags": ["instruction-following", "function-calling", "multilingual"],
        "suggested_alias": "mistral-small-3.1-24b-nvfp4",
        "suggested_served_name": "mistral-small-3.1-24b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "65536",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
            "enable-auto-tool-choice": "true",
            "tool-call-parser": "mistral_nemo",
        },
    },
    {
        "id": "devstral-small-2505-nvfp4",
        "name": "Devstral Small 2505",
        "repo": "unsloth/Devstral-Small-2505-NVFP4",
        "description": "Mistral's coding-focused model. Optimized for "
                       "code completion and agentic coding tasks.",
        "vram_gb": 14,
        "params": "24B",
        "quantization": "NVFP4",
        "category": "coding",
        "tags": ["coding", "agentic", "code-completion"],
        "suggested_alias": "devstral-small-2505-nvfp4",
        "suggested_served_name": "devstral-small",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "65536",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
        },
    },
    {
        "id": "deepseek-r1-0528-qwen3-8b-nvfp4",
        "name": "DeepSeek R1 0528 Qwen3 8B",
        "repo": "unsloth/DeepSeek-R1-0528-Qwen3-8B-NVFP4",
        "description": "DeepSeek R1 reasoning distilled into Qwen3 8B. "
                       "Strong chain-of-thought in a compact model.",
        "vram_gb": 5,
        "params": "8B",
        "quantization": "NVFP4",
        "category": "reasoning",
        "tags": ["reasoning", "chain-of-thought", "math"],
        "suggested_alias": "deepseek-r1-qwen3-8b-nvfp4",
        "suggested_served_name": "deepseek-r1-qwen3-8b",
        "engine_args": {
            "kv-cache-dtype": "fp8_e4m3",
            "max-model-len": "131072",
            "gpu-memory-utilization": "0.93",
            "enforce-eager": "true",
        },
    },
]


def get_catalog() -> List[dict]:
    """Return the full catalog list."""
    return list(CATALOG)


def get_catalog_entry(catalog_id: str) -> Optional[dict]:
    """Look up a single catalog entry by id."""
    for entry in CATALOG:
        if entry["id"] == catalog_id:
            return dict(entry)
    return None
