# uv run olmo2.py

import argparse
import os
import torch
import transformers
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--exp_name', type=str, default=None)
parser.add_argument('--model_name', type=str, default='allenai/OLMo-2-0425-1B-Instruct')
parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
parser.add_argument('--n_steps', type=int, default=1000000)
parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-2)
# parser.add_argument('--scheduler_step_size', type=int, default=5000)
parser.add_argument('--scheduler_patience', type=int, default=10)
parser.add_argument('--detach', action='store_true')
parser.add_argument('--eval_len', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--print_every', type=int, default=100)
args = parser.parse_args()
if args.dtype == 'bf16':
    dtype = torch.bfloat16
elif args.dtype == 'fp32':
    dtype = torch.float32
else:
    raise ValueError(f'Invalid dtype: {args.dtype}')

def set_seed(seed=19260817):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

device = torch.device('cuda')

if args.exp_name is not None:
    wandb.init(project='f-eigen', name=args.exp_name)

model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
model.eval() # turn off dropout
D = model.config.hidden_size
x = torch.nn.Parameter(torch.randn(D, dtype=dtype, device=device))
if args.optim == 'adamw':
    optimizer = torch.optim.AdamW([x], lr=args.lr, weight_decay=0.0)
elif args.optim == 'sgd':
    optimizer = torch.optim.SGD([x], lr=args.lr)
else:
    raise ValueError(f'Invalid optimizer: {args.optim}')
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience)
for step in range(args.n_steps + 1):
    if step % args.eval_every == 0:
        with torch.no_grad():
            print(f'step = {step}, evaluating with zero positional embeddings ...')
            inputs_embeds = x.clone().detach().unsqueeze(0).unsqueeze(0) # (1, L=1, D)
            for l in range(1, args.eval_len + 1):
                outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True,
                    position_ids=torch.zeros((1, l), dtype=torch.long, device=device),
                )
                y = outputs.hidden_states[-1][0, -1, :] # (D)
                loss = torch.norm(y - x)
                print(f'\tl = {l}, loss = {loss.item():.8f}')
                inputs_embeds = torch.cat([inputs_embeds, y.clone().detach().unsqueeze(0).unsqueeze(0)], dim=1) # (1, L=l+1, D)

            print(f'step = {step}, evaluating with normal positional embeddings ...')
            inputs_embeds = x.clone().detach().unsqueeze(0).unsqueeze(0) # (1, L=1, D)
            for l in range(1, args.eval_len + 1):
                outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True,
                    position_ids=torch.arange(l, dtype=torch.long, device=device).unsqueeze(0),
                )
                y = outputs.hidden_states[-1][0, -1, :] # (D)
                loss = torch.norm(y - x)
                print(f'\tl = {l}, loss = {loss.item():.8f}')
                inputs_embeds = torch.cat([inputs_embeds, y.clone().detach().unsqueeze(0).unsqueeze(0)], dim=1) # (1, L=l+1, D)

    inputs_embeds = x.unsqueeze(0).unsqueeze(0) # (1, L=1, D)
    if args.detach:
        torch.set_grad_enabled(False)
    outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True,
        position_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
    )
    y = outputs.hidden_states[-1][0, -1, :] # (D)
    if args.detach:
        y = y.detach()
        torch.set_grad_enabled(True)
    loss = torch.norm(y - x) # TODO: revisit this, huber loss?
    loss.backward()
    grad_norm = torch.norm(x.grad).item()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss)
    if args.exp_name is not None:
        wandb.log({
            'loss': loss.item(),
            'norm': torch.norm(x).item(),
            'grad_norm': grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
        }, step=step)
    if step % args.print_every == 0:
        print(f'step = {step}, loss = {loss.item():.8f}, norm = {torch.norm(x).item():.8f}, lr = {optimizer.param_groups[0]["lr"]:.8f}')
