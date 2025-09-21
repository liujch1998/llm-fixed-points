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
    print(f'{model}')
    fps = []
    for seed in range(10):
        with open(f'output_iteration/{model}_seed{seed}.json', 'r') as f:
            js = json.load(f)
            fps += js['fps']
    xs = torch.stack([torch.tensor(fp['x'], dtype=torch.float32) for fp in fps]) # (B, D)
    dist = torch.norm(xs.unsqueeze(0) - xs.unsqueeze(1), dim=-1) # (B, B)
    # print(dist)
    group_by_ix = {i: i for i in range(len(fps))}
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if dist[i, j] < 1e-4:
                for k in range(len(fps)):
                    if group_by_ix[k] == group_by_ix[j]:
                        group_by_ix[k] = group_by_ix[i]
    true_ixs = []
    for i in set(group_by_ix.values()):
        if fps[i]['rollout_dist_by_l']['1'] < 1e-4:
            true_ixs.append(i)
    print(f'Found {len(true_ixs)} true FPs')
    for i in true_ixs:
        print(f'FP #{i}: rollout_dist_first = {fps[i]["rollout_dist_by_l"]["1"]:.8f}, rollout_dist_max = {fps[i]["rollout_dist_max"]:.8f}, output_probs_argmax = {repr(fps[i]["output_probs"][0]["token"])} ({fps[i]["output_probs"][0]["prob"]:.4f})')
    print()

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
    model_data[base_name][model_type] = len(true_ixs)

# Generate markdown table
print("\n## Number of FPs by Model\n")
print("| Model | Base | Inst |")
print("|-------|-----:|-----:|")

for base_name in model_data.keys():
    base_fps = model_data[base_name].get('base', 'N/A')
    inst_fps = model_data[base_name].get('inst', 'N/A')
    print(f"| {base_name} | {base_fps} | {inst_fps} |")
