from pathlib import Path
import torch

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE     = dict(total_updates=5_00_000, n_steps=20,  lr=1e-4)
FEUDAL   = dict(total_updates=5_00_000, n_steps=40,  lr=1e-4)
MAML     = dict(meta_iters=5_00_000,   inner_lr=1e-3, meta_lr=1e-4)
FINETUNE = dict(total_updates=20_000,  n_steps=40,  lr=1e-4)

VALUE_COEF   = 0.5
ENTROPY_COEF = 0.01
CLIP_GRAD    = 0.5
GAMMA        = 0.99
