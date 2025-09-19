# uv run fp_discrete.py

import argparse
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
parser.add_argument('--model_name', type=str, default='allenai/OLMo-2-0425-1B-Instruct')
parser.add_argument('--batch_size', type=int, default=256)
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

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
model.eval() # turn off dropout
try:
    D = model.config.hidden_size
except:
    D = model.config.text_config.hidden_size

with torch.no_grad():
    fps = []
    for i in range(0, tokenizer.vocab_size, args.batch_size):
        input_ids = torch.arange(i, i+args.batch_size, dtype=torch.long, device=device).unsqueeze(1) # (B, 1)
        outputs = model(
            input_ids=input_ids,
        )
        logits = outputs.logits[:, -1, :] # (B, V)
        probs = F.softmax(logits, dim=-1) # (B, V)
        logits_argmax = logits.argmax(dim=-1) # (B, 1)
        for b in range(args.batch_size):
            if logits_argmax[b] == input_ids[b]:
                fps.append({
                    'token_id': input_ids[b].item(),
                    'token': tokenizer.decode(input_ids[b].item(), skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    'prob': probs[b, input_ids[b]].item(),
                })

fps = sorted(fps, key=lambda x: x['prob'], reverse=True)

with open(f'output_discrete/{args.exp_name}.json', 'w') as f:
    output = {
        'num_fps': len(fps),
        'fps': fps,
    }
    json.dump(output, f, indent=4)
