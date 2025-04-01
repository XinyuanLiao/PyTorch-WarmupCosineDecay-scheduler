# PyTorch-WarmupCosineDecay-scheduler
This repo is a PyTorch implementation of the ```optax.schedules.warmup_cosine_decay_schedule``` in Jax.

## Parameters
**init_value** – Initial value for the scalar to be annealed.

**peak_value** – Peak value for scalar to be annealed at end of warmup.

**warmup_steps** – Positive integer, the length of the linear warmup.

**decay_steps** – Positive integer, the length of the cosine decay. Note that the total optimization steps are the sum of ```warmup_steps``` and ```decay_steps```, which is different with ```optax.schedules.warmup_cosine_decay_schedule```.

**end_value** – End value of the scalar to be annealed. Defaults to 1e-7.

**exponent** – The default decay is 0.5 * (1 + cos(pi t/T)), where t is the current timestep and T is decay_steps. The exponent modifies this to be (0.5 * (1 + cos(pi * t/T))) ** exponent. Defaults to 1.0.

## Quick Start
A training template demo:
```python
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from WarmupCosineDecay import WarmupCosineDecay

epochs = 15
batchs = len(Train_loader)
total_steps = epochs * batchs
warmup_rate = 0.2

optimizer = optim.AdamW(model.parameters())
scheduler = WarmupCosineDecay(
                        optimizer,
                        init_value=1e-6,
                        peak_value=1e-3,
                        warmup_steps=warmup_rate*total_steps,
                        decay_steps=(1-warmup_rate)*total_steps)

criterion = nn.MSELoss() 
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(Train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    for x, y in progress_bar:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.2e}", 
            'lr': scheduler.get_last_lr()[0]
        })
        total_loss += loss.item()
        scheduler.step()
    print(f'Epoch {epoch+1} Average Loss: {total_loss/len(Train_loader):.2e}\n')
```
