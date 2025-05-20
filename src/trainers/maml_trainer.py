import copy, random
import torch
import gymnasium as gym

from src.models.feudal import FuNModel
from src.models.feudal import compute_losses
from src.config import MAML, DEVICE, DATA
from src.utils import set_seed
import csv
from pathlib import Path
from codecarbon import EmissionsTracker

class MetaTrainer:
    def __init__(
        self,
        tasks: list,
        vocab_size: int = 64,
        meta_batch: int = 3,
        inner_steps: int = 1
    ):
        self.tasks       = tasks
        self.meta_batch  = meta_batch
        self.inner_steps = inner_steps
        self.vocab_size  = vocab_size

        # Carbon tracker
        self.tracker = EmissionsTracker(
            output_dir=str(DATA),
            output_file="emissions_feudal_maml.csv",
            allow_multiple_runs=True
        )

        # “Master” model
        env0 = gym.make(self.tasks[0])
        self.model = FuNModel(env0.observation_space, env0.action_space).to(DEVICE)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=MAML['meta_lr'])

    def _adapt(self, fast: FuNModel, env):
        """Inner-loop: few SGD steps on one task."""
        opt = torch.optim.SGD(fast.parameters(), lr=MAML['inner_lr'])
        mgr_s, wrk_s, goals = fast.init_states(1, DEVICE)
        obs, _ = env.reset(seed=random.randint(0, 9999))
        t = 0
        # one inner update
        traj = []
        for _ in range(self.inner_steps * MAML.get('n_steps', 40)):
            obs_t = {
                'image': torch.tensor(obs['image'], device=DEVICE).unsqueeze(0),
                'direction': torch.tensor([obs['direction']], device=DEVICE)
            }
            dist, m_val, w_val, g, mgr_s, wrk_s = fast(obs_t, mgr_s, wrk_s, goals, t)
            lp    = dist.log_prob(dist.sample())
            ent   = dist.entropy()
            nxt, r, term, trunc, _ = env.step(dist.sample().item())
            done  = term or trunc
            traj.append({
                'logprob': lp, 'entropy': ent,
                'm_value': m_val, 'w_value': w_val,
                'ext_r': torch.tensor([r], device=DEVICE),
                'intr_r': torch.tensor([0.], device=DEVICE),
                'g': g, 's': None
            })
            obs = nxt; t += 1
            if done:
                obs, _ = env.reset(seed=random.randint(0, 9999))
                mgr_s, wrk_s, goals = fast.init_states(1, DEVICE)
                break
        with torch.no_grad():
            _, last_m, last_w, _, _, _ = fast(obs_t, mgr_s, wrk_s, goals, t)
        loss, _ = compute_losses(traj, last_m, last_w)
        opt.zero_grad(); loss.backward(); opt.step()

    def _meta_loss_on_task(self, fast: FuNModel, env) -> torch.Tensor:
        """Query rollouts (no updates) to compute meta-loss."""
        mgr_s, wrk_s, goals = fast.init_states(1, DEVICE)
        obs, _ = env.reset(seed=0)
        t = 0
        traj = []
        for _ in range(MAML.get('n_steps', 40)):
            obs_t = {
                'image': torch.tensor(obs['image'], device=DEVICE).unsqueeze(0),
                'direction': torch.tensor([obs['direction']], device=DEVICE)
            }
            dist, m_val, w_val, g, mgr_s, wrk_s = fast(obs_t, mgr_s, wrk_s, goals, t)
            lp    = dist.log_prob(dist.sample())
            ent   = dist.entropy()
            nxt, r, term, trunc, _ = env.step(dist.sample().item())
            done  = term or trunc
            traj.append({
                'logprob': lp, 'entropy': ent,
                'm_value': m_val, 'w_value': w_val,
                'ext_r': torch.tensor([r], device=DEVICE),
                'intr_r': torch.tensor([0.], device=DEVICE),
                'g': g, 's': None
            })
            obs = nxt; t += 1
            if done: break
        with torch.no_grad():
            _, last_m, last_w, _, _, _ = fast(obs_t, mgr_s, wrk_s, goals, t)
        loss, _ = compute_losses(traj, last_m, last_w)
        return loss

    def train(self, iterations: int = None):
        set_seed(0)
        self.tracker.start()
        log_path = DATA / "checkpoints" / "meta_feudal_stats.csv"
        for it in range(iterations or MAML['meta_iters']):
            meta_losses = []
            sample = random.sample(self.tasks, self.meta_batch)
            for task in sample:
                env = gym.make(task)
                fast = copy.deepcopy(self.model)
                self._adapt(fast, env)
                meta_losses.append(self._meta_loss_on_task(fast, env))
            meta_loss = torch.stack(meta_losses).mean()
            self.opt.zero_grad(); meta_loss.backward(); self.opt.step()
            if (it+1) % MAML.get('save_every', 500) == 0:
                torch.save(self.model.state_dict(),
                           DATA / "checkpoints" / f"meta_feudal_{it+1}.pth")
        self.tracker.stop()