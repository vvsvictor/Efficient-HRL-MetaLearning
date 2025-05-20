import os, csv
import torch
import gymnasium as gym
from pathlib import Path
from codecarbon import EmissionsTracker

from src.models.feudal import FuNModel
from src.config import FINETUNE, DEVICE, DATA
from src.utils import set_seed

def finetune(
    env_id: str,
    meta_ckpt: str,
    vocab_size: int = 64
):
    set_seed(0)
    tracker = EmissionsTracker(
        output_dir=str(DATA),
        output_file=f"emissions_finetune_{env_id.replace('/','_')}.csv",
        allow_multiple_runs=True
    )
    tracker.start()

    # Load environment & model
    base_env = gym.make(env_id)
    env = base_env
    model = FuNModel(env.observation_space, env.action_space).to(DEVICE)
    model.load_state_dict(torch.load(meta_ckpt, map_location=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE['lr'])

    # Logging
    ckpt_dir = DATA / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    log_file = ckpt_dir / f"finetune_{env_id.replace('/','_')}_stats.csv"
    if not log_file.exists():
        with log_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['update', 'loss', 'ext_r', 'intr_r'])

    mgr_s, wrk_s, goals = model.init_states(1, DEVICE)
    obs, _ = env.reset(seed=0)
    t = 0

    for upd in range(FINETUNE['total_updates']):
        traj = []
        for _ in range(FINETUNE['n_steps']):
            obs_t = {
                'image': torch.tensor(obs['image'], device=DEVICE).unsqueeze(0),
                'direction': torch.tensor([obs['direction']], device=DEVICE)
            }
            dist, m_val, w_val, g, mgr_s, wrk_s = model(obs_t, mgr_s, wrk_s, goals, t)
            lp  = dist.log_prob(dist.sample())
            ent = dist.entropy()

            nxt, r, term, trunc, _ = env.step(dist.sample().item())
            done = term or trunc

            traj.append({
                'logprob': lp, 'entropy': ent,
                'm_value': m_val, 'w_value': w_val,
                'ext_r': torch.tensor([r], device=DEVICE),
                'intr_r': torch.tensor([0.], device=DEVICE),
                'g': g, 's': None
            })

            obs = nxt; t += 1
            if done:
                obs, _ = env.reset(); mgr_s, wrk_s, goals = model.init_states(1, DEVICE)
                break

        with torch.no_grad():
            _, last_m, last_w, _, _, _ = model(obs_t, mgr_s, wrk_s, goals, t)

        # Compute loss like in FeudalTrainer
        from src.models.feudal import compute_losses
        loss, stats = compute_losses(traj, last_m, last_w)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), FINETUNE.get('clip_grad_norm', 0.5))
        optimizer.step()

        if (upd+1) % FINETUNE.get('save_every', 500) == 0:
            ckpt_path = ckpt_dir / f"finetune_{upd+1}_{env_id.replace('/','_')}.pth"
            torch.save(model.state_dict(), ckpt_path)
            with log_file.open('a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([upd+1, stats['loss'],
                                 stats.get('ext_r', 0.0), stats.get('intr_r', 0.0)])

    tracker.stop()