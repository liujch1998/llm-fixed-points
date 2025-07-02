model_type = 'gpt2'
L = 1
D = 768

import numpy as np
import torch
import pytorch_lightning as pl
import transformers
device = torch.device('cuda')
pl.seed_everything(19260817)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)

'''
from torch.utils.data import IterableDataset, Dataset, DataLoader
class IterDs(IterableDataset):
    def __iter__(self):
        while True:
            yield 0
train_ds = IterDs()
train_dl = DataLoader(train_ds, batch_size=1)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_type)
        self.v = torch.nn.Parameter(torch.randn(D))

    def training_step(self, batch, batch_idx):
        inputs_embeds = self.v.unsqueeze(0).expand(L, -1).unsqueeze(0)
        outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        if batch_idx % 100 == 0:
            print(torch.norm(self.v))
        output = outputs.hidden_states[-1][0, -1, :]
        loss = torch.norm(output - self.v)
        self.log('train/loss', loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([self.v], lr=1e-2)

model = Model()

trainer = pl.Trainer(
    deterministic=True,
    gpus=1,
    max_steps=10000,
)

trainer.fit(model, train_dl)
'''

model = transformers.GPT2LMHeadModel.from_pretrained(model_type).to(device)
x = torch.nn.Parameter(torch.randn(D, device=device))
optimizer = torch.optim.Adam([x], lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000)
for it in range(10000):
    inputs_embeds = x.unsqueeze(0).expand(L, -1).unsqueeze(0)
    outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True,
        position_ids=torch.zeros((1, L), dtype=torch.long, device=device),
    )
    y = outputs.hidden_states[-1][0, -1, :]
    loss = torch.norm(y - x)
    if it % 100 == 0:
        print('%d %.2f %.2f' % (it, loss.item(), torch.norm(x).item()))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
for L in range(1, 2):
    inputs_embeds = x.unsqueeze(0).expand(L, -1).unsqueeze(0)
    outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True,
        position_ids=torch.zeros((1, L), dtype=torch.long, device=device),
    )
    y = outputs.hidden_states[-1][0, -1, :]
    loss = torch.norm(y - x)
    print('%d %.2f %.2f' % (L, loss.item(), torch.norm(x).item()))
    dists = torch.norm(y.unsqueeze(0) - model.transformer.wpe.weight.data, p=2, dim=1)
    ranks = torch.argsort(dists)
    for i in ranks[:20]:
        dist = dists[i]
        output = tokenizer.decode(i)
        print(i.item(), output, dist.item())

'''
model = transformers.GPT2LMHeadModel.from_pretrained(model_type).to(device)
v = torch.randn(D, requires_grad=True, device=device)

for it in range(30):
    inputs_embeds = v.unsqueeze(0).expand(L, -1).unsqueeze(0)
    outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    output = outputs.hidden_states[-1][0, -1, :]
    loss = torch.norm(output - v)
    print(torch.norm(v), loss.item())
    v.data = output.detach()
'''

