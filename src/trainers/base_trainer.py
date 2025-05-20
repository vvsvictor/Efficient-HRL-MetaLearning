from dataclasses import dataclass
import gymnasium as gym, torch, math
from collections import namedtuple
from src.config import DEVICE, VALUE_COEF, ENTROPY_COEF, CLIP_GRAD, GAMMA
from src.utils import tracker, csv_append

Step = namedtuple("Step", "logp ent val rew")

@dataclass
class Trainer:
    env_id: str
    model: torch.nn.Module
    optim: torch.optim.Optimizer
    total_updates: int
    n_steps: int
    csv_path: str

    def run(self):
        env = gym.make(self.env_id)
        obs,_ = env.reset(seed=0)
        tracker("emissions.csv").start()
        for upd in range(self.total_updates):
            traj = []; R=0
            for _ in range(self.n_steps):
                dist,val = self.model(obs_to_t(obs))
                a = dist.sample(); nxt,r,term,trunc,_ = env.step(a.item())
                traj.append(Step(dist.log_prob(a),dist.entropy(),val,r))
                R += r; obs = nxt
                if term or trunc: obs,_=env.reset(); break
            self._update(traj)
            if (upd+1) % 500 == 0:
                csv_append(self.csv_path,
                           Step(update=upd+1, loss=self.loss, return_=R, ent=self.ent))
        env.close()

    def _update(self, traj):
        lp,ent,v,r = zip(*traj)
        returns, G = [], 0
        for rew in reversed(r):
            G = rew + GAMMA*G; returns.insert(0,G)
        returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32)
        adv = returns - torch.stack(v).squeeze()
        policy = -(torch.stack(lp)*adv.detach()).mean()
        value  = adv.pow(2).mean()*VALUE_COEF
        entropy= torch.stack(ent).mean()*-ENTROPY_COEF
        loss = policy+value+entropy
        self.optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD)
        self.optim.step(); self.loss,self.ent=loss.item(),entropy.item()

def obs_to_t(obs):
    return {'image':torch.tensor(obs['image']).unsqueeze(0).to(DEVICE),
            'direction':torch.tensor([obs['direction']]).to(DEVICE)}
