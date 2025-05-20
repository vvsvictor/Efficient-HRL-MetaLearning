from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from src.utils import set_seed
import math
set_seed()

class ACModel(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        c,h,w = 3,*obs_space['image'].shape[:2]
        self.backbone = nn.Sequential(
            nn.Conv2d(c,16,2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,2), nn.ReLU(),
            nn.Conv2d(32,64,2), nn.ReLU())
        flat = ((h-1)//2-2)*((w-1)//2-2)*64
        self.dir_emb = nn.Embedding(4,32)
        inp = flat+32
        self.actor  = nn.Sequential(nn.Linear(inp,64), nn.Tanh(),
                                    nn.Linear(64, act_space.n))
        self.critic = nn.Sequential(nn.Linear(inp,64), nn.Tanh(),
                                    nn.Linear(64,1))
        self.apply(self._ortho)

    @staticmethod
    def _ortho(m):                           # orthogonal init
        if isinstance(m,nn.Linear):
            nn.init.orthogonal_(m.weight, math.sqrt(2))
            nn.init.constant_(m.bias,0)

    def forward(self, obs):
        x = obs['image'].float().permute(0,3,1,2)
        z = self.backbone(x).flatten(1)
        z = torch.cat([z, self.dir_emb(obs['direction'])], 1)
        logits = self.actor(z)
        value  = self.critic(z).squeeze(-1)
        return Categorical(logits=F.log_softmax(logits,1)), value
