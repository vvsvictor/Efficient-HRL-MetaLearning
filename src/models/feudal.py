import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class DilatedLSTM(nn.Module):
    def __init__(self, inp, hid, r):
        super().__init__()
        self.r    = r
        self.cell = nn.LSTMCell(inp, hid)

    def forward(self, x, state, t):
        h_prev, c_prev = state
        slot = t % self.r
        h_s, c_s = self.cell(x, (h_prev[slot], c_prev[slot]))
        h, c = h_prev.clone(), c_prev.clone()
        h[slot], c[slot] = h_s, c_s
        return h_s, (h, c)

    @staticmethod
    def init_state(B, H, r, device):
        return (torch.zeros(r, B, H, device=device),
                torch.zeros(r, B, H, device=device))

class FuNModel(nn.Module):
    def __init__(self, obs_space, act_space, k=8, g_dim=64, c=30):
        super().__init__()
        self.A, self.k, self.c = act_space.n, k, c
        n, m, _ = obs_space['image'].shape
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,2), nn.ReLU(),
            nn.Conv2d(32,64,2), nn.ReLU()
        )
        flat = ((n-1)//2 - 2) * ((m-1)//2 - 2) * 64
        self.fc        = nn.Linear(flat, 64)
        self.dir_emb   = nn.Embedding(4, 8)

        enc_dim        = 64 + 8
        self.manager_sp= nn.Sequential(nn.Linear(enc_dim, g_dim), nn.ELU())
        self.m_rnn     = DilatedLSTM(g_dim, g_dim, c)
        self.manager_v = nn.Linear(g_dim, 1)
        self.phi       = nn.Linear(g_dim, k, bias=False)

        self.worker_rnn= nn.LSTM(enc_dim, self.A*k, batch_first=True)
        self.worker_v  = nn.Linear(self.A*k, 1)
        self.apply(self._init_)

    @staticmethod
    def _init_(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, obs, mgr_s, wrk_s, goals, t):
        B = obs['image'].shape[0]
        x = obs['image'].float().permute(0,3,1,2)
        h = self.encoder(x).view(B, -1)
        z = F.elu(self.fc(h))
        d = self.dir_emb(obs['direction'].long())
        enc = torch.cat([z, d], 1)

        # Manager
        s       = self.manager_sp(enc)
        g_h, m_s= self.m_rnn(s, mgr_s, t)
        g       = F.normalize(g_h, dim=1)
        m_val   = self.manager_v(g_h).squeeze(-1)

        # Worker
        h_w, c_w  = wrk_s
        out, (h_n, c_n) = self.worker_rnn(enc.unsqueeze(1), (h_w, c_w))
        flat      = out.squeeze(1)
        w_val     = self.worker_v(flat).squeeze(-1)
        U         = flat.view(B, self.A, self.k)
        w         = self.phi(goals.sum(1) + g)
        logits    = (U @ w.unsqueeze(-1)).squeeze(-1)
        dist      = Categorical(logits=logits)

        return dist, m_val, w_val, g, m_s, (h_n, c_n)

    def init_states(self, B, device):
        mgr   = DilatedLSTM.init_state(B, 64, self.c, device)
        h     = torch.zeros(1, B, self.A*self.k, device=device)
        c     = torch.zeros(1, B, self.A*self.k, device=device)
        goals = torch.zeros(B, self.c, 64, device=device)
        return mgr, (h, c), goals