import re, csv
import torch
import gymnasium as gym
from pathlib import Path
from codecarbon import EmissionsTracker

from src.models.feudal import FuNModel, compute_losses
from src.config import FEUDAL, DEVICE, DATA
from src.utils import set_seed

def train_feudal(env_id: str = "MiniGrid-Unlock-v0"):
    set_seed(0)
    tracker = EmissionsTracker(
        output_dir=str(DATA),
        output_file="emissions_feudal.csv",
        allow_multiple_runs=True
    )
    tracker.start()

    # Checkpoints & logs
    ckpt_dir = DATA / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    env_name = env_id.replace('/', '_')
    log_file = ckpt_dir / f"feudal_{env_name}_stats.csv"

    # Resume point
    pattern = re.compile(fr'feudal_(\d+)_{env_name}\.pth')
    existing = list(ckpt_dir.glob(f'feudal_*_{env_name}.pth'))
    last_upd = max((int(pattern.match(f.name).group(1))
                    for f in existing if pattern.match(f.name)), default=0)

    # Init log CSV
    if not log_file.exists():
        with log_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['update', 'loss', 'ext_r', 'intr_r'])

    # Environment and model
    env = gym.make(env_id)
    model = FuNModel(env.observation_space, env.action_space).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=FEUDAL['lr'])

    # Load checkpoint if exists
    if last_upd > 0:
        ckpt_path = ckpt_dir / f"feudal_{last_upd}_{env_name}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    mgr_s, wrk_s, goals = model.init_states(1, DEVICE)
    obs, _ = env.reset(seed=0)
    t = 0

    for upd in range(last_upd, FEUDAL['total_updates']):
        traj = []
        for _ in range(FEUDAL['n_steps']):
            obs_t = {
                'image': torch.tensor(obs['image'], device=DEVICE).unsqueeze(0),
                'direction': torch.tensor([obs['direction']], device=DEVICE)
            }
            dist, m_val, w_val, g, mgr_s, wrk_s = model(obs_t, mgr_s, wrk_s, goals, t)
            lp = dist.log_prob(dist.sample())
            ent = dist.entropy()

            nxt, r, term, trunc, _ = env.step(dist.sample().item())
            done = term or trunc

            traj.append({
                'logprob': lp,
                'entropy': ent,
                'm_value': m_val,
                'w_value': w_val,
                'ext_r': torch.tensor([r], device=DEVICE),
                'intr_r': torch.tensor([0.], device=DEVICE),
                'g': g,
                's': None
            })

            obs = nxt
            t += 1
            if done:
                obs, _ = env.reset()
                mgr_s, wrk_s, goals = model.init_states(1, DEVICE)
                break

        with torch.no_grad():
            obs_t = {
                'image': torch.tensor(obs['image'], device=DEVICE).unsqueeze(0),
                'direction': torch.tensor([obs['direction']], device=DEVICE)
            }
            _, last_m, last_w, _, _, _ = model(obs_t, mgr_s, wrk_s, goals, t)

        loss, stats = compute_losses(traj, last_m, last_w)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), FEUDAL.get('clip_grad_norm', 0.5))
        optimizer.step()

        if (upd + 1) % FEUDAL.get('save_every', 500) == 0:
            ckpt_path = ckpt_dir / f"feudal_{upd+1}_{env_name}.pth"
            torch.save(model.state_dict(), ckpt_path)
            with log_file.open('a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([upd+1, stats['loss'],
                                 stats.get('ext_r', 0.0), stats.get('intr_r', 0.0)])

    tracker.stop()