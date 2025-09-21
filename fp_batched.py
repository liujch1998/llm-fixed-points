# uv run fp_batched.py

import argparse
from collections import defaultdict
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--exp_name', type=str, required=True)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--model_name', type=str, default='allenai/OLMo-2-0425-1B-Instruct')
parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
parser.add_argument('--n_steps', type=int, default=100000)
parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--scheduler_patience', type=int, default=10)
parser.add_argument('--detach', action='store_true')
parser.add_argument('--eval_len', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=19260817)
args = parser.parse_args()
if args.dtype == 'bf16':
    dtype = torch.bfloat16
elif args.dtype == 'fp32':
    dtype = torch.float32
else:
    raise ValueError(f'Invalid dtype: {args.dtype}')

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

if args.wandb:
    wandb.init(project='f-eigen', name=args.exp_name)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
model.eval() # turn off dropout
try:
    D = model.config.hidden_size
except:
    D = model.config.text_config.hidden_size
x = torch.nn.Parameter(torch.randn(args.batch_size, D, dtype=dtype, device=device) * model.config.initializer_range)
if args.optim == 'adamw':
    optimizer = torch.optim.AdamW([x], lr=args.lr, weight_decay=0.0)
elif args.optim == 'sgd':
    optimizer = torch.optim.SGD([x], lr=args.lr)
else:
    raise ValueError(f'Invalid optimizer: {args.optim}')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, eps=1e-9)

for step in range(args.n_steps + 1):
    if step % args.eval_every == 0 or optimizer.param_groups[0]['lr'] < 3e-9:
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
            with open(f'output_batched/{args.exp_name}.json', 'w') as f:
                json.dump(output, f, indent=4)

    if optimizer.param_groups[0]['lr'] < 3e-9:
        break

    inputs_embeds = x.unsqueeze(1) # (B, L=1, D)
    if args.detach:
        torch.set_grad_enabled(False)
    outputs = model(
        inputs_embeds=inputs_embeds,
        position_ids=torch.zeros((args.batch_size, 1), dtype=torch.long, device=device),
    )
    y = outputs.logits[:, -1, :] # (B, V)
    y = F.softmax(y, dim=-1) # (B, V)
    y = y @ model.model.embed_tokens.weight.data # (B, D)
    if args.detach:
        y = y.detach()
        torch.set_grad_enabled(True)
    # loss = torch.nn.MSELoss()(100 * y, 100 * x)
    loss = torch.norm(y - x, dim=-1).mean()
    loss.backward()
    norm = torch.norm(x, dim=-1).mean().item()
    grad_norm = torch.norm(x.grad, dim=-1).mean().item()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss)
    if args.wandb:
        wandb.log({
            'loss': loss.item(),
            'norm': norm,
            'grad_norm': grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
        }, step=step)
    if step % args.print_every == 0:
        print(f'step = {step}, loss = {loss.item():.8f}, norm = {norm:.8f}, lr = {optimizer.param_groups[0]["lr"]:.8f}')
