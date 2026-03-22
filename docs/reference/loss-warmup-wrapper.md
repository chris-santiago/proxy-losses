# LossWarmupWrapper

A training utility that wraps a warmup loss and a main ranking loss. Manages three phases: warmup (standard loss only), optional linear blend, and main phase with geometric temperature decay. Exposes PyTorch Lightning hooks.

::: imbalanced_losses.warmup_wrapper.LossWarmupWrapper

## Quick example

**Epoch-based warmup** (default):

```python
from imbalanced_losses import SmoothAPLoss, LossWarmupWrapper
import torch.nn as nn

loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_epochs=5,
    blend_epochs=2,
    temp_start=0.5,
    temp_end=0.01,
    temp_decay_steps=50_000,
)
```

**Step-based warmup** (use when you prefer step counts over epochs):

```python
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.CrossEntropyLoss(),
    main_loss=SmoothAPLoss(num_classes=10, queue_size=1024),
    warmup_steps=5_000,
    blend_steps=2_000,
    temp_start=0.5,
    temp_end=0.01,
    temp_decay_steps=50_000,
)
```

`warmup_epochs`/`blend_epochs` and `warmup_steps`/`blend_steps` are mutually exclusive pairs.

## PyTorch Lightning integration

**Epoch mode** — call both hooks:

```python
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = LossWarmupWrapper(...)

    def on_train_epoch_start(self):
        self.loss_fn.on_train_epoch_start(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        self.loss_fn.on_train_batch_start(self.global_step)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss)
        self.log("train/main_weight", self.loss_fn.main_weight)
        if (t := self.loss_fn.current_temperature) is not None:
            self.log("train/temperature", t)
        return loss
```

**Step mode** — only the batch hook is required:

```python
class MyModel(pl.LightningModule):
    def on_train_batch_start(self, batch, batch_idx):
        self.loss_fn.on_train_batch_start(self.global_step)

    def training_step(self, batch, batch_idx):
        logits, targets = batch
        return self.loss_fn(logits, targets)
```

## Phase schedule

**Epoch mode** — with `warmup_epochs=5, blend_epochs=2`:

| Epoch range | Phase | `in_warmup` | `in_blend` | `main_weight` |
|---|---|---|---|---|
| 0–4 | warmup | `True` | `False` | `0.0` |
| 5 | blend | `False` | `True` | `0.333` |
| 6 | blend | `False` | `True` | `0.667` |
| 7+ | main | `False` | `False` | `1.0` |

**Step mode** — with `warmup_steps=500, blend_steps=3`:

| Step range | Phase | `in_warmup` | `in_blend` | `main_weight` |
|---|---|---|---|---|
| 0–499 | warmup | `True` | `False` | `0.0` |
| 500 | blend | `False` | `True` | `0.25` |
| 501 | blend | `False` | `True` | `0.50` |
| 502 | blend | `False` | `True` | `0.75` |
| 503+ | main | `False` | `False` | `1.0` |

## Temperature schedule

Temperature decays geometrically from `temp_start` to `temp_end` over `temp_decay_steps` steps, measured from the moment of phase switch:

```
temp(t) = temp_start * (temp_end / temp_start) ^ (elapsed / temp_decay_steps)
```

The clock starts at the first main-phase batch, not at training epoch 0.

## Parameter reference

| Parameter | Default | Description |
|---|---|---|
| `warmup_loss` | required | Loss used during warmup (e.g. `CrossEntropyLoss`) |
| `main_loss` | required | Loss used after warmup (e.g. `SmoothAPLoss`) |
| `warmup_epochs` | `0` | Epochs before switching; `0` skips warmup. Mutually exclusive with `warmup_steps`. |
| `temp_start` | `0.05` | Temperature at phase switch |
| `temp_end` | `0.005` | Temperature after `temp_decay_steps` steps |
| `temp_decay_steps` | `10_000` | Steps over which to decay temperature |
| `blend_epochs` | `0` | Linear blend epochs; `0` = hard switch. Mutually exclusive with `blend_steps`. |
| `warmup_steps` | `None` | Steps before switching. Mutually exclusive with `warmup_epochs > 0`. |
| `blend_steps` | `None` | Linear blend steps. Mutually exclusive with `blend_epochs > 0`. |
| `reset_queue_each_epoch` | `False` | Reset `main_loss` queue each main-phase epoch |
| `gather_distributed` | `None` | Forwarded to `main_loss.gather_distributed`; `None` auto-detects DDP |

## Properties

| Property | Type | Description |
|---|---|---|
| `in_warmup` | `bool` | `True` while in the warmup phase |
| `in_blend` | `bool` | `True` during the blend transition |
| `main_weight` | `float` | Current main loss weight: `0.0` → ramp → `1.0` |
| `current_temperature` | `float or None` | Current `main_loss.temperature`; `None` if unavailable |
