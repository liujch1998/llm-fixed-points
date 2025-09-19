# uv run fp_iteration.py

import argparse
from collections import defaultdict
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--exp_name', type=str, required=True)
parser.add_argument('--model_name', type=str, default='allenai/OLMo-2-0425-1B-Instruct')
parser.add_argument('--n_steps', type=int, default=10000)
parser.add_argument('--eval_len', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=19260817)
args = parser.parse_args()

def set_seed(seed=args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

device = torch.device('cuda')

print('='*100)
print(args.exp_name)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)
model.eval() # turn off dropout
try:
    D = model.config.hidden_size
except:
    D = model.config.text_config.hidden_size
x = torch.nn.Parameter(torch.randn(args.batch_size, D, dtype=torch.float32, device=device) * model.config.initializer_range)
loss = 1.0
for step in range(args.n_steps + 1):
    if step % args.eval_every == 0 or loss < 1e-6:
        with torch.no_grad():
            print(f'evaluating step = {step} ...')
            fps = [{'index': b, 'x': x[b].tolist()} for b in range(args.batch_size)]

            print(f'pairwise L2 distance between FPs:')
            x1 = x.unsqueeze(0) # (1, B, D)
            x2 = x.unsqueeze(1) # (B, 1, D)
            dist = torch.norm(x1 - x2, dim=-1) # (B, B)
            print(dist)

            print(f'output probs:')
            inputs_embeds = x.clone().detach().unsqueeze(1) # (B, L=1, D)
            outputs = model(
                inputs_embeds=inputs_embeds,
                position_ids=torch.zeros((args.batch_size, 1), dtype=torch.long, device=device),
            )
            y = outputs.logits[:, -1, :] # (B, V)
            y = F.softmax(y, dim=-1) # (B, V)
            ranks = torch.argsort(y, dim=-1, descending=True)
            for b in range(args.batch_size):
                print(f'\tFP #{b}:')
                fps[b]['output_probs'] = []
                for r, token_id in enumerate(ranks[b, :10]):
                    prob = y[b, token_id].item()
                    token = tokenizer.decode(token_id, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    print(f'\t\trank = {r}, token_id = {token_id}, token = {repr(token)}, prob = {prob:.8f}')
                    fps[b]['output_probs'].append({
                        'rank': r,
                        'token_id': token_id.item(),
                        'token': token,
                        'prob': prob,
                    })

            print(f'rollout with normal positional embeddings:')
            inputs_embeds = x.clone().detach().unsqueeze(1) # (B, L=1, D)
            dist_by_l = defaultdict(list)
            past_key_values = transformers.DynamicCache()
            for l in range(1, args.eval_len + 1):
                outputs = model(
                    inputs_embeds=inputs_embeds[:, -1:, :],
                    position_ids=torch.full((args.batch_size, 1), l-1, dtype=torch.long, device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                y = outputs.logits[:, -1, :] # (B, V)
                y = F.softmax(y, dim=-1) # (B, V)
                y = y @ model.model.embed_tokens.weight.data # (B, D)
                dist = torch.norm(y - x, dim=-1) # (B)
                dist_by_l[l] = dist.tolist()
                inputs_embeds = torch.cat([inputs_embeds, y.clone().detach().unsqueeze(1)], dim=1) # (B, L=l+1, D)
            for b in range(args.batch_size):
                print(f'\tFP #{b}:')
                fps[b]['rollout_dist_max'] = 0.0
                fps[b]['rollout_dist_by_l'] = {}
                for l in range(1, args.eval_len + 1):
                    if l <= 10:
                        print(f'\t\tl = {l}, dist = {dist_by_l[l][b]:.8f}')
                    fps[b]['rollout_dist_by_l'][l] = dist_by_l[l][b]
                    fps[b]['rollout_dist_max'] = max(fps[b]['rollout_dist_max'], dist_by_l[l][b])

            output = {
                'step': step,
                'fps': fps,
            }
            with open(f'output_iteration/{args.exp_name}.json', 'w') as f:
                json.dump(output, f, indent=4)

    if loss < 1e-6:
        break

    inputs_embeds = x.unsqueeze(1) # (B, L=1, D)
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            position_ids=torch.zeros((args.batch_size, 1), dtype=torch.long, device=device),
        )
        y = outputs.logits[:, -1, :] # (B, V)
        y = F.softmax(y, dim=-1) # (B, V)
        y = y @ model.model.embed_tokens.weight.data # (B, D)
        loss = torch.norm(y - x, dim=-1).mean().item()
        norm = torch.norm(x, dim=-1).mean().item()
        x = y
    if step % args.print_every == 0:
        print(f'step = {step}, loss = {loss:.8f}, norm = {norm:.8f}')
