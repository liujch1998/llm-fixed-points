# uv run analyze.py

import os
import glob
import json
import numpy as np
import torch

MODELS = [
    'olmo2-1b-base',
    'olmo2-1b-inst',
    'olmo2-7b-base',
    'olmo2-7b-inst',
    'olmo2-13b-base',
    'olmo2-13b-inst',
    'qwen3-0.6b-base',
    'qwen3-0.6b-inst',
    'qwen3-1.7b-base',
    'qwen3-1.7b-inst',
    'qwen3-4b-base',
    'qwen3-4b-inst',
    'qwen3-8b-base',
    'qwen3-8b-inst',
    'qwen3-14b-base',
    'qwen3-14b-inst',
    'gemma3-270m-base',
    'gemma3-270m-inst',
    'gemma3-1b-base',
    'gemma3-1b-inst',
]

# Collect num_fps data for all models
model_data = {}

for model in MODELS:
    with open(f'output_discrete/{model}.json', 'r') as f:
        js = json.load(f)
    num_fps = js['num_fps']

    # Extract base model name and type
    if model.endswith('-base'):
        base_name = model[:-5]  # Remove '-base'
        model_type = 'base'
    elif model.endswith('-inst'):
        base_name = model[:-5]  # Remove '-inst'
        model_type = 'inst'
    else:
        continue  # Skip if neither inst nor base

    if base_name not in model_data:
        model_data[base_name] = {}
    model_data[base_name][model_type] = num_fps

# Generate markdown table
print("\n## Number of FPs by Model\n")
print("| Model | Base | Inst |")
print("|-------|-----:|-----:|")

for base_name in model_data.keys():
    base_fps = model_data[base_name].get('base', 'N/A')
    inst_fps = model_data[base_name].get('inst', 'N/A')
    print(f"| {base_name} | {base_fps} | {inst_fps} |")
